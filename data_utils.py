"""
Lightweight loaders to align pheno/covar with FAM order and drop missing values.

Assumptions:
- IID is unique in FAM/pheno/covar and is the sample identifier.
- FAM order defines the target IID sequence.
- pheno file has columns: FID IID PHENO (whitespace or tab separated).
- covar file (optional) has columns: FID IID C1 C2 ...
- Missing values marked as -9, NA, NaN, or empty are dropped.
"""
from __future__ import annotations
import dataclasses
import logging
from typing import Optional
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_MISSING = {"-9", "NA", "NaN", "nan", ""}
_MISSING_LIST = sorted(_MISSING)
_RAW_SEP = "\x1f"
_TAB_SEP = "\t"


@dataclasses.dataclass(frozen=True)
class CovariateTransform:
    """Training-set covariate standardization applied to future data."""

    raw_dim: int
    keep_mask: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    add_intercept: bool = True

def _read_raw_lines(path: str) -> pl.DataFrame:
    """
    Read each physical line as one raw string column named 'raw'.
    Parsing is strict (no ignore_errors) to avoid silently dropping malformed rows.
    """
    return pl.read_csv(
        path,
        has_header=False,
        separator=_RAW_SEP,
        quote_char=None,
        new_columns=["raw"],
    )


def _split_parts_expr() -> pl.Expr:
    """
    Normalize whitespace to tabs, then split into fields.
    """
    return (
        pl.col("raw")
        .str.strip_chars()
        .str.replace_all(r"\s+", _TAB_SEP)
        .str.split(_TAB_SEP)
    )


def _check_duplicate_iid(df: pl.DataFrame, name: str) -> None:
    """
    Raise if duplicated IID exists.
    """
    if df.is_empty():
        return

    n_rows = df.height
    n_unique = df.select(pl.col("iid").n_unique()).item()
    if n_rows == n_unique:
        return

    dup_df = (
        df.group_by("iid")
        .len()
        .filter(pl.col("len") > 1)
        .sort("iid")
        .head(10)
    )
    dup_vals = dup_df["iid"].to_list()
    raise ValueError(
        f"{name} contains duplicated IID values. "
        f"Examples: {dup_vals}"
    )


def _read_fam_df(fam_path: str) -> pl.DataFrame:
    """
    Parse FAM and keep target sample order via fam_idx.
    Expected at least 2 columns: FID IID.
    """
    parts = _split_parts_expr()
    df = _read_raw_lines(fam_path).with_row_index("fam_idx")

    fam = (
        df.select(
            pl.col("fam_idx"),
            parts.list.len().alias("n_fields"),
            parts.list.get(1).cast(pl.Utf8, strict=False).alias("iid"),
        )
        .filter(pl.col("n_fields") >= 2)
        .filter(pl.col("iid").is_not_null() & (pl.col("iid") != ""))
        .select("fam_idx", "iid")
    )

    _check_duplicate_iid(fam.select("iid"), "FAM")
    return fam


def _parse_pheno_df(pheno_path: str) -> pl.DataFrame:
    """
    Parse pheno file into columns: iid, pheno.
    Expected columns: FID IID PHENO.
    """
    parts = _split_parts_expr()
    df = _read_raw_lines(pheno_path)

    pheno = (
        df.select(
            parts.list.len().alias("n_fields"),
            parts.list.get(1).cast(pl.Utf8, strict=False).alias("iid"),
            parts.list.get(2).cast(pl.Utf8, strict=False).alias("pheno_raw"),
        )
        .filter(pl.col("n_fields") >= 3)
        .filter(pl.col("iid").is_not_null() & (pl.col("iid") != ""))
        .filter(pl.col("pheno_raw").is_not_null())
        .filter(~pl.col("pheno_raw").is_in(_MISSING_LIST))
        .with_columns(pl.col("pheno_raw").cast(pl.Float32, strict=False).alias("pheno"))
        .filter(pl.col("pheno").is_not_null())
        .filter(pl.col("pheno") != -9.0)
        .select("iid", "pheno")
    )

    _check_duplicate_iid(pheno.select("iid"), "pheno file")
    return pheno


def _parse_covar_df(covar_path: str) -> tuple[pl.DataFrame, int]:
    """
    Parse covar file into columns: iid, cov_num(list[f32]).
    Expected columns: FID IID C1 C2 ...

    Returns:
        covar_df: columns ['iid', 'cov_num']
        cov_dim: number of covariates
    """
    parts = _split_parts_expr()
    df = _read_raw_lines(covar_path)

    covar = (
        df.select(
            parts.list.len().alias("n_fields"),
            parts.list.get(1).cast(pl.Utf8, strict=False).alias("iid"),
            parts.list.slice(2).alias("cov_raw"),
        )
        .filter(pl.col("n_fields") > 2)
        .filter(pl.col("iid").is_not_null() & (pl.col("iid") != ""))
        .filter(pl.col("cov_raw").list.len() > 0)
        .filter(~pl.col("cov_raw").list.eval(pl.element().is_in(_MISSING_LIST)).list.any())
        .with_columns(pl.col("cov_raw").list.eval(pl.element().cast(pl.Float32, strict=False)).alias("cov_num"))
        .filter(~pl.col("cov_num").list.eval(pl.element().is_null()).list.any())
        .filter(~pl.col("cov_num").list.eval(pl.element() == -9.0).list.any())
        .with_columns(pl.col("cov_num").list.len().alias("cov_dim"))
        .select("iid", "cov_num", "cov_dim")
    )

    _check_duplicate_iid(covar.select("iid"), "covar file")

    if covar.is_empty():
        return covar.select("iid", "cov_num"), 0

    dim_stats = (
        covar.group_by("cov_dim")
        .len()
        .sort(["len", "cov_dim"], descending=[True, False])
    )
    cov_dim_mode = int(dim_stats["cov_dim"][0])
    if dim_stats.height > 1:
        bad = (
            covar.filter(pl.col("cov_dim") != cov_dim_mode)
            .select("iid", "cov_dim")
            .head(10)
        )
        raise ValueError(
            "Covariate file has inconsistent number of numeric columns across rows. "
            f"Expected (mode) cov_dim={cov_dim_mode}. "
            f"Examples:\n{bad}"
        )

    return covar.select("iid", "cov_num"), cov_dim_mode


def _fit_covar_transform_from_matrix(
    X_raw: np.ndarray,
    *,
    raw_dim: int,
    add_intercept: bool,
) -> tuple[np.ndarray, CovariateTransform]:
    if X_raw.ndim != 2:
        raise ValueError("Covariate matrix is not 2D after alignment; check covariate file formatting.")
    if X_raw.shape[1] != raw_dim:
        raise ValueError("Covariate matrix width mismatch after alignment; check covariate file formatting.")
    if not np.isfinite(X_raw).all():
        raise ValueError("Covariate matrix contains non-finite values.")

    means_full = X_raw.mean(axis=0, dtype=np.float64)
    stds_full = X_raw.std(axis=0, dtype=np.float64)
    keep_mask = np.asarray(stds_full > 0, dtype=bool)
    dropped_cols = int(np.sum(~keep_mask))
    if dropped_cols:
        logger.warning("Dropped %d covariate columns with zero std.", dropped_cols)

    X_keep = X_raw[:, keep_mask]
    means = means_full[keep_mask].astype(np.float32, copy=False)
    stds = stds_full[keep_mask].astype(np.float32, copy=False)
    if X_keep.shape[1] > 0:
        X_std = ((X_keep - means) / stds).astype(np.float32, copy=False)
    else:
        X_std = X_keep.astype(np.float32, copy=False)

    if add_intercept:
        X_std = np.concatenate([np.ones((X_std.shape[0], 1), dtype=np.float32), X_std], axis=1)

    transform = CovariateTransform(
        raw_dim=int(raw_dim),
        keep_mask=keep_mask.astype(np.bool_, copy=False),
        means=np.asarray(means, dtype=np.float32),
        stds=np.asarray(stds, dtype=np.float32),
        add_intercept=bool(add_intercept),
    )
    return X_std, transform


def _apply_covar_transform_to_matrix(
    X_raw: np.ndarray,
    *,
    transform: CovariateTransform,
) -> np.ndarray:
    if X_raw.ndim != 2:
        raise ValueError("Covariate matrix is not 2D after alignment; check covariate file formatting.")
    if X_raw.shape[1] != int(transform.raw_dim):
        raise ValueError(
            f"Prediction covariate width mismatch: expected {int(transform.raw_dim)}, got {X_raw.shape[1]}."
        )
    if not np.isfinite(X_raw).all():
        raise ValueError("Covariate matrix contains non-finite values.")

    X_keep = X_raw[:, np.asarray(transform.keep_mask, dtype=bool)]
    if X_keep.shape[1] != int(transform.means.size) or X_keep.shape[1] != int(transform.stds.size):
        raise ValueError("Stored covariate transform is inconsistent with kept covariate columns.")
    if X_keep.shape[1] > 0:
        X_std = ((X_keep - transform.means) / transform.stds).astype(np.float32, copy=False)
    else:
        X_std = X_keep.astype(np.float32, copy=False)

    if transform.add_intercept:
        X_std = np.concatenate([np.ones((X_std.shape[0], 1), dtype=np.float32), X_std], axis=1)
    return X_std


def load_pheno_covar_aligned(
    fam_path: str,
    pheno_path: str,
    covar_path: Optional[str] = None,
    add_intercept: bool = True,
    keep_ids: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], list[str], list[str]]:
    """
    Align pheno/covar to FAM IID order and drop rows with missing or invalid pheno/covar after parsing/filtering.

    Returns:
        y: (n_kept,) float32
        X: (n_kept, p) float32 or None
        keep_ids: list of kept IIDs in FAM order
        dropped: list of IIDs dropped because phenotype/covariate data were missing
            or invalid after parsing/filtering, plus valid IIDs excluded by
            keep_ids when provided.
    """
    fam_df = _read_fam_df(fam_path)
    pheno_df = _parse_pheno_df(pheno_path)

    joined = fam_df.join(pheno_df, on="iid", how="left")

    cov_dim = 0
    if covar_path:
        covar_df, cov_dim = _parse_covar_df(covar_path)
        joined = joined.join(covar_df, on="iid", how="left")
        valid_mask = pl.col("pheno").is_not_null() & pl.col("cov_num").is_not_null()
    else:
        valid_mask = pl.col("pheno").is_not_null()

    dropped_df = joined.filter(~valid_mask).select("iid")
    kept_df = joined.filter(valid_mask)
    if keep_ids is not None:
        keep_set = set(keep_ids)
        keep_expr = pl.col("iid").is_in(list(keep_set))
        extra_dropped_df = kept_df.filter(~keep_expr).select("iid")
        if extra_dropped_df.height > 0:
            dropped_df = pl.concat([dropped_df, extra_dropped_df], how="vertical")
        kept_df = kept_df.filter(keep_expr)

    if kept_df.height == 0:
        raise ValueError(
            "No valid samples remained after aligning FAM with phenotype"
            + (" and covariate files." if covar_path else " file.")
        )

    dropped = dropped_df["iid"].to_list()
    keep_ids = kept_df["iid"].to_list()

    y = kept_df["pheno"].to_numpy().astype(np.float32, copy=False)

    X: Optional[np.ndarray]
    if covar_path:
        cov_exprs = [pl.col("cov_num").list.get(i).alias(f"c{i}") for i in range(cov_dim)]
        X = kept_df.select(cov_exprs).to_numpy().astype(np.float32, copy=False)
        if X.ndim != 2:
            raise ValueError("Covariate matrix is not 2D after alignment; check covariate file formatting.")
        if X.shape[1] != cov_dim:
            raise ValueError("Covariate matrix width mismatch after alignment; check covariate file formatting.")
        if not np.isfinite(X).all():
            raise ValueError("Covariate matrix contains non-finite values.")

        means = X.mean(axis=0, dtype=np.float64)
        stds = X.std(axis=0, dtype=np.float64)
        keep = stds > 0
        dropped_cols = int(np.sum(~keep))
        if dropped_cols:
            logger.warning("Dropped %d covariate columns with zero std.", dropped_cols)
        X = X[:, keep]
        means = means[keep]
        stds = stds[keep]
        if X.shape[1] > 0:
            X = ((X - means) / stds).astype(np.float32, copy=False)
        else:
            X = X.astype(np.float32, copy=False)

        if add_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    else:
        if add_intercept:
            X = np.ones((y.shape[0], 1), dtype=np.float32)
        else:
            X = None

    if not np.isfinite(y).all():
        raise ValueError("Phenotype vector contains non-finite values.")

    return y, X, keep_ids, dropped


def load_pheno_covar_aligned_with_transform(
    fam_path: str,
    pheno_path: str,
    covar_path: Optional[str] = None,
    add_intercept: bool = True,
    keep_ids: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], list[str], list[str], CovariateTransform]:
    fam_df = _read_fam_df(fam_path)
    pheno_df = _parse_pheno_df(pheno_path)

    joined = fam_df.join(pheno_df, on="iid", how="left")

    cov_dim = 0
    if covar_path:
        covar_df, cov_dim = _parse_covar_df(covar_path)
        joined = joined.join(covar_df, on="iid", how="left")
        valid_mask = pl.col("pheno").is_not_null() & pl.col("cov_num").is_not_null()
    else:
        valid_mask = pl.col("pheno").is_not_null()

    dropped_df = joined.filter(~valid_mask).select("iid")
    kept_df = joined.filter(valid_mask)
    if keep_ids is not None:
        keep_set = set(keep_ids)
        keep_expr = pl.col("iid").is_in(list(keep_set))
        extra_dropped_df = kept_df.filter(~keep_expr).select("iid")
        if extra_dropped_df.height > 0:
            dropped_df = pl.concat([dropped_df, extra_dropped_df], how="vertical")
        kept_df = kept_df.filter(keep_expr)

    if kept_df.height == 0:
        raise ValueError(
            "No valid samples remained after aligning FAM with phenotype"
            + (" and covariate files." if covar_path else " file.")
        )

    dropped = dropped_df["iid"].to_list()
    keep_ids = kept_df["iid"].to_list()
    y = kept_df["pheno"].to_numpy().astype(np.float32, copy=False)

    if covar_path:
        cov_exprs = [pl.col("cov_num").list.get(i).alias(f"c{i}") for i in range(cov_dim)]
        X_raw = kept_df.select(cov_exprs).to_numpy().astype(np.float32, copy=False)
        X, transform = _fit_covar_transform_from_matrix(
            X_raw, raw_dim=cov_dim, add_intercept=add_intercept,
        )
    else:
        transform = CovariateTransform(
            raw_dim=0,
            keep_mask=np.zeros((0,), dtype=np.bool_),
            means=np.zeros((0,), dtype=np.float32),
            stds=np.zeros((0,), dtype=np.float32),
            add_intercept=bool(add_intercept),
        )
        X = np.ones((y.shape[0], 1), dtype=np.float32) if add_intercept else None

    if not np.isfinite(y).all():
        raise ValueError("Phenotype vector contains non-finite values.")

    return y, X, keep_ids, dropped, transform


def load_covar_aligned(
    fam_path: str,
    covar_path: Optional[str],
    *,
    transform: CovariateTransform,
) -> tuple[Optional[np.ndarray], list[str], list[str]]:
    fam_df = _read_fam_df(fam_path)
    joined = fam_df

    if covar_path:
        covar_df, cov_dim = _parse_covar_df(covar_path)
        if cov_dim != int(transform.raw_dim):
            raise ValueError(
                f"Prediction covariate width mismatch: expected {int(transform.raw_dim)}, got {cov_dim}."
            )
        joined = joined.join(covar_df, on="iid", how="left")
        valid_mask = pl.col("cov_num").is_not_null()
    else:
        if int(transform.raw_dim) > 0:
            raise ValueError(
                "Prediction covariates are required because training used non-empty covariates."
            )
        valid_mask = pl.lit(True)

    dropped_df = joined.filter(~valid_mask).select("iid")
    kept_df = joined.filter(valid_mask)
    keep_ids = kept_df["iid"].to_list()
    dropped = dropped_df["iid"].to_list()

    if kept_df.height == 0:
        raise ValueError("No valid samples remained after aligning prediction covariates to FAM order.")

    if int(transform.raw_dim) > 0:
        cov_exprs = [pl.col("cov_num").list.get(i).alias(f"c{i}") for i in range(int(transform.raw_dim))]
        X_raw = kept_df.select(cov_exprs).to_numpy().astype(np.float32, copy=False)
        X = _apply_covar_transform_to_matrix(X_raw, transform=transform)
    else:
        X = np.ones((kept_df.height, 1), dtype=np.float32) if transform.add_intercept else None

    return X, keep_ids, dropped
