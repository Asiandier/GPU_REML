from __future__ import annotations

import dataclasses
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.stats

from .gwas_io import open_gwas_tsv, write_gwas_metadata
from .variant_io import iter_variant_records_for_prefix

logger = logging.getLogger(__name__)
_GWAS_WRITE_CHUNK_ROWS = 8192


@dataclasses.dataclass(frozen=True)
class GWASSummary:
    out_path: str
    metadata_path: str
    n_samples: int
    n_variants: int
    dof: int
    n_covar_cols: int


def _orthonormal_covariate_basis(X: np.ndarray) -> tuple[np.ndarray, int]:
    if X.ndim != 2:
        raise ValueError("Covariate matrix must be 2D.")
    if X.shape[1] == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32), 0
    Q, R, _ = scipy.linalg.qr(X.astype(np.float64, copy=False), mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32), 0
    tol = np.finfo(np.float64).eps * max(X.shape) * max(1.0, float(diag.max()))
    rank = int(np.count_nonzero(diag > tol))
    if rank == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32), 0
    return np.asarray(Q[:, :rank], dtype=np.float32), rank


def _resolve_streamer_variant_prefix_and_format(streamer) -> tuple[str, str]:
    prefix = getattr(streamer, "_variant_prefix", None)
    fmt = getattr(streamer, "_variant_format", None)
    if prefix and fmt:
        return str(prefix), str(fmt)
    src = getattr(streamer, "_source", None)
    if src is None:
        raise ValueError("Dense GWAS requires streamer source metadata.")
    bed_prefix = getattr(src, "_bed_prefix", None)
    if bed_prefix:
        return str(bed_prefix), "bed"
    pgen_prefix = getattr(src, "_pgen_prefix", None)
    if pgen_prefix:
        return str(pgen_prefix), "pgen"
    if hasattr(src, "_bed") and hasattr(src._bed, "location"):
        path = str(src._bed.location)
        if path.endswith(".bed"):
            return path[:-4], "bed"
    raise ValueError(
        "Unable to resolve variant metadata sidecar path for GWAS input. "
        "Supported dense inputs require a BED .bim or PGEN .pvar/.pvar.zst sidecar."
    )


def run_continuous_gwas(
    fitter,
    y: np.ndarray,
    covar: np.ndarray | None = None,
    *,
    out_prefix: str,
) -> GWASSummary:
    if getattr(fitter, "_has_sparse", False):
        raise ValueError("GWAS currently supports dense genotype inputs only.")
    if not fitter.streamers:
        raise ValueError("GWAS requires at least one dense genotype streamer.")

    y_np = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(y_np.shape[0])
    dense_streamers = tuple(fitter.streamers[: fitter._n_dense_streamers])
    if not dense_streamers:
        raise ValueError("GWAS requires at least one dense genotype streamer.")
    for st in dense_streamers:
        if int(st.n) != n:
            raise ValueError(
                f"Sample-size mismatch between phenotype (n={n}) and streamer (n={int(st.n)})."
            )
        if getattr(st, "_means_host", None) is None or getattr(st, "_inv_sds_host", None) is None:
            raise ValueError("GWAS requires keep_host_stats=True on dense streamers.")
        if getattr(st, "_count_host", None) is None:
            raise ValueError(
                "GWAS requires per-SNP non-missing counts; rebuild streamers without "
                "standardization_override or with host stats retained."
            )

    if covar is None or np.size(covar) == 0:
        X = np.ones((n, 1), dtype=np.float32)
    else:
        X = np.asarray(covar, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != n:
            raise ValueError(
                f"Covariate row mismatch: expected {n}, got {int(X.shape[0])}."
            )

    Q, rank = _orthonormal_covariate_basis(X)
    if rank == 0:
        y_perp = y_np.astype(np.float32, copy=False)
    else:
        proj_y = Q @ (Q.T @ y_np.astype(np.float32, copy=False))
        y_perp = (y_np - proj_y).astype(np.float32, copy=False)
    y_perp_ss = float(np.dot(y_perp.astype(np.float64), y_perp.astype(np.float64)))
    dof = int(n - rank - 1)
    if dof <= 0:
        raise ValueError(
            f"GWAS degrees of freedom must be positive; got n={n}, rank(X)={rank}, dof={dof}."
        )

    V_parts = [y_perp[:, None]]
    if rank > 0:
        V_parts.append(Q.astype(np.float32, copy=False))
    V = np.concatenate(V_parts, axis=1).astype(np.float32, copy=False)

    out_path, out_fh = open_gwas_tsv(out_prefix)
    total_variants = 0
    global_idx = 0
    component_names: list[str] = []
    component_variant_counts: list[int] = []
    V_dev = jnp.asarray(V, dtype=jnp.float32)

    try:
        for component_idx, st in enumerate(dense_streamers):
            prefix, fmt = _resolve_streamer_variant_prefix_and_format(st)
            component_name = os.path.basename(prefix)
            component_names.append(component_name)
            component_variant_counts.append(int(st.m))
            logger.info(
                "[GWAS] component %d/%d start name=%s m=%d",
                component_idx + 1,
                len(dense_streamers),
                component_name,
                int(st.m),
            )
            cross = np.asarray(
                jax.device_get(st.xtv(V_dev, normalize=False)),
                dtype=np.float64,
            )
            x_ty = cross[:, 0]
            if rank > 0:
                x_t_q = cross[:, 1:]
                proj_ss = np.einsum("ij,ij->i", x_t_q, x_t_q, optimize=True)
            else:
                proj_ss = np.zeros((int(st.m),), dtype=np.float64)

            counts = np.asarray(st._count_host, dtype=np.float64).reshape(-1)
            means = np.asarray(st._means_host, dtype=np.float64).reshape(-1)
            inv_sd = np.asarray(st._inv_sds_host, dtype=np.float64).reshape(-1)
            if counts.size != int(st.m):
                raise RuntimeError("Internal GWAS count vector length mismatch.")

            x_var = counts - proj_ss
            valid = (inv_sd > 0.0) & (x_var > 1e-8)
            beta_std = np.full(int(st.m), np.nan, dtype=np.float64)
            se_std = np.full(int(st.m), np.nan, dtype=np.float64)
            beta = np.full(int(st.m), np.nan, dtype=np.float64)
            se = np.full(int(st.m), np.nan, dtype=np.float64)
            t_stat = np.full(int(st.m), np.nan, dtype=np.float64)
            p_val = np.full(int(st.m), np.nan, dtype=np.float64)

            beta_std[valid] = x_ty[valid] / x_var[valid]
            rss = np.full(int(st.m), np.nan, dtype=np.float64)
            rss[valid] = y_perp_ss - beta_std[valid] * x_ty[valid]
            rss[valid] = np.maximum(rss[valid], 0.0)
            sigma2 = np.full(int(st.m), np.nan, dtype=np.float64)
            sigma2[valid] = rss[valid] / float(dof)
            se_std[valid] = np.sqrt(np.maximum(sigma2[valid], 0.0) / x_var[valid])
            nonzero_se = valid & (se_std > 0.0)
            beta[nonzero_se] = beta_std[nonzero_se] * inv_sd[nonzero_se]
            se[nonzero_se] = se_std[nonzero_se] * inv_sd[nonzero_se]
            t_stat[nonzero_se] = beta_std[nonzero_se] / se_std[nonzero_se]
            p_val[nonzero_se] = 2.0 * scipy.stats.t.sf(np.abs(t_stat[nonzero_se]), dof)
            af = means / 2.0

            variant_iter = iter_variant_records_for_prefix(prefix, fmt)
            local_idx = -1
            chunk: list[str] = []
            for local_idx, record in enumerate(variant_iter):
                chunk.append(
                    f"{component_idx}\t{component_name}\t{global_idx}\t{local_idx}\t"
                    f"{record.chrom}\t{record.variant_id}\t{record.cm}\t{record.bp}\t"
                    f"{record.a1}\t{record.a2}\t{af[local_idx]:.8g}\t{int(counts[local_idx])}\t"
                    f"{beta[local_idx]:.8g}\t{se[local_idx]:.8g}\t"
                    f"{beta_std[local_idx]:.8g}\t{se_std[local_idx]:.8g}\t"
                    f"{t_stat[local_idx]:.8g}\t{p_val[local_idx]:.8g}\n"
                )
                global_idx += 1
                if len(chunk) >= _GWAS_WRITE_CHUNK_ROWS:
                    out_fh.write("".join(chunk))
                    chunk.clear()
            if chunk:
                out_fh.write("".join(chunk))
            if local_idx + 1 != int(st.m):
                raise RuntimeError(
                    f"Variant metadata count mismatch for {component_name}: "
                    f"expected {int(st.m)}, got {local_idx + 1}."
                )
            total_variants += int(st.m)
            logger.info(
                "[GWAS] component %d/%d done name=%s valid=%d/%d",
                component_idx + 1,
                len(dense_streamers),
                component_name,
                int(np.count_nonzero(nonzero_se)),
                int(st.m),
            )
    finally:
        out_fh.close()

    meta_path = write_gwas_metadata(
        out_prefix,
        {
            "trait_type": "continuous",
            "model": "marginal_ols_fwl",
            "n_samples": n,
            "n_variants": total_variants,
            "n_covar_cols": int(X.shape[1]),
            "rank_covar": rank,
            "dof": dof,
            "files": {"gwas": out_path},
            "components": [
                {
                    "component_index": idx,
                    "component_name": name,
                    "n_variants": count,
                }
                for idx, (name, count) in enumerate(zip(component_names, component_variant_counts))
            ],
        },
    )
    return GWASSummary(
        out_path=out_path,
        metadata_path=meta_path,
        n_samples=n,
        n_variants=total_variants,
        dof=dof,
        n_covar_cols=int(X.shape[1]),
    )


__all__ = ["GWASSummary", "run_continuous_gwas"]
