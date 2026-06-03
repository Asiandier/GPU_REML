"""
SMILE-style block-weighted GRM operators.

This module is intentionally separate from the production infinitesimal GRM
path.  It implements one genetic kernel of the form

    K V = sum_i Z_i W_i Z_i.T V / c

for contiguous block-diagonal weight matrices W = diag(W_1, ..., W_G).  The
blocks are a computational decomposition of one variance component, not
separate variance components.
The genotype standardisation and packed streaming cache are reused from
``GenoBlockStreamer``; only the block weight logic lives here.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .kv_impl import _device_put_block, _unpack_impute_center

Normalization = Literal["kernel_trace", "effective_rank"]
DiagMode = Literal["full", "mean"]
_IDENTITY_TRACE_SCAN_TARGET_BYTES = 256 * 1024 * 1024
_VALID_NORMALIZATIONS = ("kernel_trace", "effective_rank")
_EFFECTIVE_RANK_KEYS = ("effective_rank", "retained_rank", "weight_rank", "rank")
_W_DEVICE_CACHE_CAP_BYTES = 8 * 1024**3
_W_DEVICE_CACHE_FRACTION = 0.35
_W_BUCKET_WIDTH_MULTIPLE = 512
_W_BUCKET_LOCAL_TARGET_BYTES = 4 * 1024**3
_W_BUCKET_ASSUMED_RHS = 1024


@dataclasses.dataclass(frozen=True)
class SmileBlockWeight:
    """A single contiguous block W_i in source/cache order."""

    matrix: np.ndarray | None
    start: int
    size_value: int
    trace_per_sample: float
    source: str | None = None
    is_identity: bool = False
    device_matrix: object | None = None

    @property
    def size(self) -> int:
        return int(self.size_value)

    @property
    def stop(self) -> int:
        return int(self.start + self.size)


@dataclasses.dataclass(frozen=True)
class SmileBlockBucket:
    """Padded device cache chunk for batched dense block multiplications."""

    matrices: object
    starts: object
    sizes: object
    width: int


def default_w_device_cache_bytes(gpu_budget_bytes: float | None) -> float:
    """Return the default SMILE dense-W device cache limit."""

    if gpu_budget_bytes is None:
        return 0.0
    budget = float(gpu_budget_bytes)
    if not np.isfinite(budget) or budget <= 0.0:
        return 0.0
    return float(min(_W_DEVICE_CACHE_CAP_BYTES, _W_DEVICE_CACHE_FRACTION * budget))


def _bucket_width(width: int) -> int:
    width = int(width)
    multiple = int(_W_BUCKET_WIDTH_MULTIPLE)
    return int(((width + multiple - 1) // multiple) * multiple)


def estimate_bucketed_w_device_cache_bytes(
    widths: Sequence[int],
    cache_limit_bytes: float | None,
) -> float:
    """Return padded bucket-cache bytes if the W blocks fit the cache limit."""

    limit = float(cache_limit_bytes or 0.0)
    if limit <= 0.0:
        return 0.0
    padded = float(
        sum(_bucket_width(int(width)) ** 2 * np.dtype(np.float32).itemsize for width in widths)
    )
    return padded if 0.0 < padded <= limit else 0.0


def estimate_bucketed_w_local_workspace_bytes(
    widths: Sequence[int],
    *,
    rhs_cols: int = _W_BUCKET_ASSUMED_RHS,
    cache_enabled: bool,
) -> float:
    """Return a conservative local+output workspace estimate for W buckets."""

    if not cache_enabled:
        return 0.0
    groups: dict[int, int] = {}
    for width in widths:
        bucket_width = _bucket_width(int(width))
        groups[bucket_width] = groups.get(bucket_width, 0) + 1
    max_local = 0.0
    itemsize = np.dtype(np.float32).itemsize
    for width, count in groups.items():
        max_blocks = max(
            1,
            int(_W_BUCKET_LOCAL_TARGET_BYTES // (int(width) * int(rhs_cols) * itemsize)),
        )
        chunk_blocks = min(int(count), int(max_blocks))
        local = float(chunk_blocks * int(width) * int(rhs_cols) * itemsize)
        max_local = max(max_local, local)
    return 2.0 * max_local


def load_rds_matrix(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a numeric R matrix from ``.rds`` using Rscript without Python deps."""

    path = os.fspath(path)
    script = r"""
path <- commandArgs(trailingOnly=TRUE)[1]
out_path <- commandArgs(trailingOnly=TRUE)[2]
x <- readRDS(path)
if (!is.matrix(x) || !is.numeric(x)) {
  stop("RDS object must be a numeric matrix")
}
con <- file(out_path, "wb")
on.exit(close(con))
writeBin(as.integer(nrow(x)), con, size=4, endian="little")
writeBin(as.integer(ncol(x)), con, size=4, endian="little")
writeBin(as.double(x), con, size=8, endian="little")
"""
    with tempfile.TemporaryDirectory(prefix="gpu_reml_rds_") as tmpdir:
        out_path = os.path.join(tmpdir, "matrix.bin")
        try:
            subprocess.run(
                ["Rscript", "--vanilla", "--slave", "-e", script, path, out_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Rscript is required to load .rds weight matrices.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to load RDS matrix {path!r}: {stderr}") from exc

        payload = Path(out_path).read_bytes()

    if len(payload) < 8:
        raise RuntimeError(f"RDS matrix payload from {path!r} is truncated.")
    rows = int(np.frombuffer(payload[:4], dtype="<i4")[0])
    cols = int(np.frombuffer(payload[4:8], dtype="<i4")[0])
    values = np.frombuffer(payload[8:], dtype="<f8")
    if rows <= 0 or cols <= 0 or values.size != rows * cols:
        raise RuntimeError(
            f"RDS matrix payload shape mismatch for {path!r}: "
            f"rows={rows}, cols={cols}, values={values.size}."
        )
    return np.asarray(values.reshape((rows, cols), order="F"), dtype=np.float64)


def load_weight_matrix(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a dense weight matrix from RDS, NPY/NPZ, or text/CSV."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".rds":
        return load_rds_matrix(path)
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        data = np.load(path)
        try:
            keys = list(data.files)
            if len(keys) != 1:
                raise ValueError(f"NPZ weight file must contain exactly one array, got {keys}.")
            return data[keys[0]]
        finally:
            data.close()
    delimiter = "," if suffix == ".csv" else None
    return np.loadtxt(path, delimiter=delimiter)


def load_weight_matrix_shape(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Return a W matrix shape without reading the full dense payload when possible."""

    path = Path(path)
    meta = load_weight_matrix_metadata(path, required=False)
    if meta:
        for key in ("shape", "matrix_shape"):
            if key in meta:
                shape = tuple(int(x) for x in meta[key])
                if len(shape) == 2:
                    return shape
        for key in ("width", "size", "p", "n_snps"):
            if key in meta:
                width = int(meta[key])
                return (width, width)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return tuple(int(x) for x in arr.shape)
    if suffix == ".npz":
        data = np.load(path)
        try:
            keys = list(data.files)
            if len(keys) != 1:
                raise ValueError(f"NPZ weight file must contain exactly one array, got {keys}.")
            return tuple(int(x) for x in data[keys[0]].shape)
        finally:
            data.close()
    matrix = load_weight_matrix(path)
    return tuple(int(x) for x in np.asarray(matrix).shape)


def load_weight_matrix_metadata(
    path: str | os.PathLike[str],
    *,
    required: bool = False,
) -> dict:
    """Load optional sidecar metadata next to a W file."""

    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        if required:
            raise ValueError(f"Missing sidecar JSON metadata for W file {os.fspath(path)!r}.")
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if required:
            raise ValueError(f"Failed to read sidecar JSON metadata {os.fspath(meta_path)!r}.") from exc
        return {}
    if not isinstance(meta, dict):
        raise ValueError(f"Sidecar JSON metadata must be an object: {os.fspath(meta_path)!r}.")
    return meta


def _max_abs_symmetric_difference(W: np.ndarray, *, block_rows: int = 2048) -> float:
    """Return max |W-W.T| without materializing a full dense difference."""

    n = int(W.shape[0])
    max_diff = 0.0
    block = max(1, int(block_rows))
    for start in range(0, n, block):
        stop = min(start + block, n)
        diff = np.max(np.abs(W[start:stop, :] - W[:, start:stop].T))
        max_diff = max(max_diff, float(diff))
    return max_diff


def validate_weight_matrix(
    matrix: np.ndarray,
    *,
    name: str = "W",
    symmetrize: bool = True,
    symmetry_tol: float = 1e-6,
    check_symmetry: bool = True,
    check_psd: bool = True,
    psd_tol: float = 1e-7,
) -> np.ndarray:
    """Return a finite, square float32 matrix suitable for SMILE GPU use."""

    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype={arr.dtype}.")

    W = np.asarray(arr, dtype=np.float32)
    if not W.flags.c_contiguous:
        W = np.asarray(W, dtype=np.float32, order="C")
    if not np.all(np.isfinite(W)):
        raise ValueError(f"{name} contains non-finite values.")

    if check_symmetry:
        max_abs = float(np.max(np.abs(W))) if W.size else 0.0
        asym = _max_abs_symmetric_difference(W)
        allowed_asym = float(symmetry_tol * max(1.0, max_abs))
        if asym > allowed_asym:
            raise ValueError(f"{name} is not symmetric: max |W-W.T|={asym:g}.")
        if symmetrize and asym > 0.0:
            W = np.asarray(0.5 * (W + W.T), dtype=np.float32, order="C")

    if check_psd:
        eigvals = np.linalg.eigvalsh(np.asarray(W, dtype=np.float64))
        eig_min = float(eigvals[0])
        eig_max = float(eigvals[-1])
        if eig_min < -float(psd_tol) * max(1.0, abs(eig_max)):
            raise ValueError(f"{name} is not positive semidefinite: min eigenvalue={eig_min:g}.")

    return W


@jax.jit
def _accumulate_weighted_scores_bucket_jit(
    scores: jnp.ndarray,
    XtV: jnp.ndarray,
    W_bucket: jnp.ndarray,
    starts: jnp.ndarray,
    sizes: jnp.ndarray,
    scale: jnp.ndarray,
    normalizer: jnp.ndarray,
) -> jnp.ndarray:
    """Accumulate one padded bucket of block-diagonal W multiplications."""

    width = W_bucket.shape[1]
    rows = jnp.arange(width, dtype=starts.dtype)
    mask = rows[None, :] < sizes[:, None]
    idx = starts[:, None] + rows[None, :]
    safe_idx = jnp.where(mask, idx, jnp.zeros((), dtype=idx.dtype))
    local = XtV[safe_idx, :] * mask[:, :, None].astype(XtV.dtype)
    weighted = jnp.matmul(
        W_bucket.astype(XtV.dtype),
        local,
        precision=jax.lax.Precision.HIGHEST,
    )
    weighted = weighted * (scale / normalizer).astype(XtV.dtype)
    weighted = weighted * mask[:, :, None].astype(XtV.dtype)
    return scores.at[safe_idx.reshape(-1), :].add(
        weighted.reshape((-1, XtV.shape[1]))
    )


@jax.jit
def _zxm_one_call_jit(
    g_dev_packed,
    true_width,
    means_call,
    inv_call,
    b_call,
    miss_u8,
    acc,
):
    """Accumulate one packed call-block contribution for Z @ B."""

    diff, inv_f = _unpack_impute_center(
        g_dev_packed, true_width, means_call, inv_call, b_call, miss_u8
    )
    width = diff.shape[1]
    fp = b_call.dtype
    cmask = (jnp.arange(width, dtype=true_width.dtype) < true_width).astype(fp)
    b_f = b_call[:width, :].astype(fp) * cmask[:, None]
    return acc + diff @ (inv_f[:, None] * b_f)


def _zxm_impl_streamed_from_scores(
    streamer,
    scores: jnp.ndarray,
    *,
    missing_val: int,
) -> jnp.ndarray:
    """Streaming Z @ scores without materializing a 3D call-layout array."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape (m, rhs).")
    if scores.shape[0] != int(streamer.m):
        raise ValueError(f"scores row mismatch: expected m={int(streamer.m)}, got {scores.shape[0]}.")

    streamer._prepare_kv_pass()
    fp = scores.dtype
    dev = next(iter(scores.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    rhs = int(scores.shape[1])
    acc = jnp.zeros((int(streamer.n), rhs), dtype=fp)
    if int(streamer._n_calls) == 0:
        return acc

    pad = jnp.zeros((int(streamer._max_unpack_width), rhs), dtype=fp)
    scores_pad = jnp.concatenate([scores, pad], axis=0)
    g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
    for c in range(int(streamer._n_calls)):
        g_dev_cur = g_dev_next
        if c + 1 < int(streamer._n_calls):
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        start = int(streamer._call_snp_starts[c])
        b_call = jax.lax.dynamic_slice(
            scores_pad,
            (start, 0),
            (int(streamer._max_unpack_width), rhs),
        )
        acc = _zxm_one_call_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            b_call,
            miss_u8,
            acc,
        )
        del g_dev_cur
    return acc


class SmileBlockWeightedOperator:
    """Matrix-free SMILE block-diagonal W operator over a GenoBlockStreamer."""

    def __init__(
        self,
        streamer,
        weight_matrices: Sequence[np.ndarray],
        *,
        normalization: Normalization = "kernel_trace",
        strict_coverage: bool = True,
        symmetrize: bool = True,
        check_psd: bool = True,
        sources: Sequence[str | None] | None = None,
        start_offsets: Sequence[int] | None = None,
        effective_ranks: Sequence[float] | None = None,
        diag_mode: DiagMode = "full",
        device_cache_max_bytes: float | None = None,
    ):
        if normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_VALID_NORMALIZATIONS}.")
        if diag_mode not in ("full", "mean"):
            raise ValueError("diag_mode must be 'full' or 'mean'.")
        if not weight_matrices:
            raise ValueError("At least one weight matrix is required.")
        if sources is not None and len(sources) != len(weight_matrices):
            raise ValueError("sources length must match weight_matrices length.")
        if start_offsets is not None and len(start_offsets) != len(weight_matrices):
            raise ValueError("start_offsets length must match weight_matrices length.")
        if effective_ranks is not None and len(effective_ranks) != len(weight_matrices):
            raise ValueError("effective_ranks length must match weight_matrices length.")

        blocks: list[SmileBlockWeight] = []
        raw_diag = np.zeros((int(streamer.n),), dtype=np.float64) if diag_mode == "full" else None
        next_start = 0
        cache_limit = float(device_cache_max_bytes or 0.0)
        raw_widths: list[int] = []
        for raw_W in weight_matrices:
            raw_shape = np.asarray(raw_W).shape
            if len(raw_shape) == 2:
                raw_widths.append(int(raw_shape[0]))
        bucket_cache_bytes = estimate_bucketed_w_device_cache_bytes(raw_widths, cache_limit)
        cache_w_on_device = bucket_cache_bytes > 0.0
        for idx, raw_W in enumerate(weight_matrices):
            source = None if sources is None else sources[idx]
            W = validate_weight_matrix(
                raw_W,
                name=f"W[{idx}]",
                symmetrize=symmetrize,
                check_symmetry=check_psd,
                check_psd=check_psd,
            )
            start = (
                int(start_offsets[idx])
                if start_offsets is not None
                else int(next_start)
            )
            if start < 0:
                raise ValueError(f"W[{idx}] start offset must be non-negative, got {start}.")
            stop = start + int(W.shape[0])
            if stop > int(streamer.m):
                raise ValueError(
                    f"W[{idx}] with size {W.shape[0]} exceeds streamer.m={int(streamer.m)}."
                )
            if normalization == "effective_rank":
                trace_per_sample = self._effective_rank_contribution(
                    source=source,
                    explicit_rank=None if effective_ranks is None else effective_ranks[idx],
                    width=int(W.shape[0]),
                    block_name=f"W[{idx}]",
                )

            if diag_mode == "full":
                diag_contrib = self._compute_diag_contribution(streamer, W, start=start)
                if normalization == "kernel_trace":
                    trace_per_sample = self._compute_trace_per_sample(
                        diag_contrib,
                        normalization=normalization,
                    )
            else:
                diag_contrib = None
                if normalization == "kernel_trace":
                    trace_per_sample = self._compute_trace_per_sample_from_block(
                        streamer,
                        W,
                        start=start,
                    )
            if raw_diag is not None and diag_contrib is not None:
                raw_diag += diag_contrib
            blocks.append(
                SmileBlockWeight(
                    matrix=W,
                    start=start,
                    size_value=int(W.shape[0]),
                    trace_per_sample=trace_per_sample,
                    source=source,
                )
            )
            next_start = stop

        if strict_coverage and start_offsets is None and next_start != int(streamer.m):
            raise ValueError(
                f"Block W matrices cover {next_start} SNPs but streamer.m={int(streamer.m)}. "
                "Pass strict_coverage=False to intentionally ignore trailing SNPs."
            )

        self.streamer = streamer
        self.blocks = tuple(blocks)
        self._w_device_buckets = (
            self._build_w_device_buckets(self.blocks, streamer=streamer)
            if cache_w_on_device
            else ()
        )
        self.normalization = normalization
        self.normalizer = self._compute_global_normalizer(
            [block.trace_per_sample for block in blocks],
            normalization=normalization,
        )
        if raw_diag is None:
            self._diag_host = np.asarray(1.0, dtype=np.float32)
        else:
            self._diag_host = np.asarray(raw_diag / float(self.normalizer), dtype=np.float32)
        self._diag_dev = jax.device_put(jnp.asarray(self._diag_host), streamer.dev)
        self.strict_coverage = bool(strict_coverage)
        self.diag_mode = diag_mode
        self.w_device_cache_bytes = bucket_cache_bytes if cache_w_on_device else 0.0

    @classmethod
    def identity(
        cls,
        streamer,
        *,
        block_size: int | None = None,
        normalization: Normalization = "kernel_trace",
        strict_coverage: bool = True,
        diag_mode: DiagMode = "full",
    ) -> "SmileBlockWeightedOperator":
        """Create the SMILE operator for W=I without materializing I."""

        if normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_VALID_NORMALIZATIONS}.")
        if diag_mode not in ("full", "mean"):
            raise ValueError("diag_mode must be 'full' or 'mean'.")
        if block_size is None:
            block_size = cls._auto_identity_block_size(streamer, normalization=normalization)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if int(streamer.m) <= 0:
            raise ValueError("streamer must contain at least one SNP.")

        blocks: list[SmileBlockWeight] = []
        raw_diag = np.zeros((int(streamer.n),), dtype=np.float64) if diag_mode == "full" else None
        start = 0
        while start < int(streamer.m):
            size = min(block_size, int(streamer.m) - start)
            if normalization == "effective_rank":
                trace_per_sample = float(size)

            if diag_mode == "full":
                diag_contrib = cls._compute_diag_contribution(
                    streamer,
                    None,
                    start=start,
                    size=size,
                )
                raw_diag += diag_contrib
                if normalization == "kernel_trace":
                    trace_per_sample = cls._compute_trace_per_sample(
                        diag_contrib,
                        normalization=normalization,
                    )
            else:
                if normalization == "kernel_trace":
                    trace_per_sample = cls._compute_trace_per_sample_from_block(
                        streamer,
                        None,
                        start=start,
                        size=size,
                    )
            blocks.append(
                SmileBlockWeight(
                    matrix=None,
                    start=start,
                    size_value=size,
                    trace_per_sample=trace_per_sample,
                    source="identity",
                    is_identity=True,
                )
            )
            start += size

        self = cls.__new__(cls)
        self.streamer = streamer
        self.blocks = tuple(blocks)
        self._w_device_buckets = ()
        self.normalization = normalization
        self.normalizer = cls._compute_global_normalizer(
            [block.trace_per_sample for block in blocks],
            normalization=normalization,
        )
        if raw_diag is None:
            self._diag_host = np.asarray(1.0, dtype=np.float32)
        else:
            self._diag_host = np.asarray(raw_diag / float(self.normalizer), dtype=np.float32)
        self._diag_dev = jax.device_put(jnp.asarray(self._diag_host), streamer.dev)
        self.strict_coverage = bool(strict_coverage)
        self.diag_mode = diag_mode
        self.w_device_cache_bytes = 0.0
        return self

    @staticmethod
    def _auto_identity_block_size(
        streamer,
        *,
        normalization: Normalization,
    ) -> int:
        bytes_per_value = np.dtype(np.float32).itemsize
        n = max(1, int(streamer.n))
        width = int(_IDENTITY_TRACE_SCAN_TARGET_BYTES // (n * bytes_per_value))
        width = max(1, width)
        return int(min(int(streamer.m), width))

    @classmethod
    def from_weight_files(
        cls,
        streamer,
        paths: Sequence[str | os.PathLike[str]],
        **kwargs,
    ) -> "SmileBlockWeightedOperator":
        matrices = [load_weight_matrix(path) for path in paths]
        return cls(
            streamer,
            matrices,
            sources=[os.fspath(path) for path in paths],
            **kwargs,
        )

    @staticmethod
    def _effective_rank_contribution(
        *,
        source: str | None,
        explicit_rank: float | None,
        width: int,
        block_name: str,
    ) -> float:
        if explicit_rank is None:
            if source is None:
                raise ValueError(
                    f"{block_name} uses effective_rank normalization but has no sidecar "
                    "metadata or explicit effective rank."
                )
            meta = load_weight_matrix_metadata(source, required=True)
            for key in _EFFECTIVE_RANK_KEYS:
                if key in meta:
                    explicit_rank = meta[key]
                    break
            else:
                raise ValueError(
                    f"{block_name} sidecar metadata for effective_rank normalization must "
                    f"contain one of {_EFFECTIVE_RANK_KEYS}."
                )

        value = float(explicit_rank)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid effective rank for {block_name}: {explicit_rank!r}.")
        if value > float(width) * (1.0 + 1e-6):
            raise ValueError(
                f"Invalid effective rank for {block_name}: {value:g} exceeds block width {width}."
            )
        return value

    @staticmethod
    def _compute_diag_contribution(
        streamer,
        W: np.ndarray | None,
        *,
        start: int,
        size: int | None = None,
    ) -> np.ndarray:
        width = int(size if W is None else W.shape[0])
        idx = np.arange(int(start), int(start + width), dtype=np.int64)
        Z = np.asarray(streamer.extract_standardized_columns(idx), dtype=np.float32, order="C")
        if W is None:
            return np.sum(Z * Z, axis=1, dtype=np.float64)

        dev = streamer.dev
        Z_dev = jax.device_put(jnp.asarray(Z, dtype=jnp.float32), dev)
        W_dev = jax.device_put(jnp.asarray(W, dtype=jnp.float32), dev)
        weighted_dev = jnp.matmul(Z_dev, W_dev, precision=jax.lax.Precision.HIGHEST)
        diag_dev = jnp.sum(weighted_dev * Z_dev, axis=1)
        diag = np.asarray(jax.device_get(diag_dev), dtype=np.float64)
        del Z_dev, W_dev, weighted_dev, diag_dev
        return diag

    @staticmethod
    def _compute_trace_per_sample(
        diag_contrib: np.ndarray,
        *,
        normalization: Normalization,
    ) -> float:
        diag_arr = np.asarray(diag_contrib, dtype=np.float64)
        value = float(np.sum(diag_arr) / float(diag_arr.size))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid SMILE block trace contribution: {value!r}.")
        return value

    @staticmethod
    def _compute_trace_per_sample_from_block(
        streamer,
        W: np.ndarray | None,
        *,
        start: int,
        size: int | None = None,
    ) -> float:
        width = int(size if W is None else W.shape[0])
        idx = np.arange(int(start), int(start + width), dtype=np.int64)
        Z = np.asarray(streamer.extract_standardized_columns(idx), dtype=np.float32, order="C")
        if W is None:
            value = float(np.sum(Z * Z, dtype=np.float64) / float(streamer.n))
        else:
            dev = streamer.dev
            Z_dev = jax.device_put(jnp.asarray(Z, dtype=jnp.float32), dev)
            W_dev = jax.device_put(jnp.asarray(W, dtype=jnp.float32), dev)
            weighted_dev = jnp.matmul(Z_dev, W_dev, precision=jax.lax.Precision.HIGHEST)
            diag_dev = jnp.sum(weighted_dev * Z_dev, axis=1)
            diag = np.asarray(jax.device_get(diag_dev), dtype=np.float64)
            value = float(np.sum(diag) / float(streamer.n))
            del Z_dev, W_dev, weighted_dev, diag_dev
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid SMILE block trace contribution: {value!r}.")
        return value

    @staticmethod
    def _compute_global_normalizer(
        trace_values: Sequence[float],
        *,
        normalization: Normalization,
    ) -> float:
        value = float(np.sum(np.asarray(trace_values, dtype=np.float64)))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid SMILE global normalizer: {value!r}.")
        return value

    @staticmethod
    def _build_w_device_buckets(
        blocks: Sequence[SmileBlockWeight],
        *,
        streamer,
    ) -> tuple[SmileBlockBucket, ...]:
        groups: dict[int, list[SmileBlockWeight]] = {}
        for block in blocks:
            if block.is_identity or block.matrix is None:
                continue
            groups.setdefault(_bucket_width(block.size), []).append(block)

        buckets: list[SmileBlockBucket] = []
        itemsize = np.dtype(np.float32).itemsize
        for width in sorted(groups):
            group = groups[width]
            max_blocks = max(
                1,
                int(
                    _W_BUCKET_LOCAL_TARGET_BYTES
                    // (int(width) * int(_W_BUCKET_ASSUMED_RHS) * itemsize)
                ),
            )
            for chunk_start in range(0, len(group), max_blocks):
                chunk = group[chunk_start : chunk_start + max_blocks]
                W_host = np.zeros((len(chunk), width, width), dtype=np.float32)
                starts = np.empty((len(chunk),), dtype=np.int32)
                sizes = np.empty((len(chunk),), dtype=np.int32)
                for idx, block in enumerate(chunk):
                    size = int(block.size)
                    W_host[idx, :size, :size] = np.asarray(block.matrix, dtype=np.float32)
                    starts[idx] = int(block.start)
                    sizes[idx] = size
                buckets.append(
                    SmileBlockBucket(
                        matrices=jax.device_put(jnp.asarray(W_host), streamer.dev),
                        starts=jax.device_put(jnp.asarray(starts), streamer.dev),
                        sizes=jax.device_put(jnp.asarray(sizes), streamer.dev),
                        width=int(width),
                    )
                )
        return tuple(buckets)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def diag(self) -> jnp.ndarray:
        """Return kernel diagonal information without forming the dense kernel.

        ``diag_mode="full"`` returns the per-sample diagonal.  ``diag_mode="mean"``
        returns the scalar average diagonal, which is one under kernel-trace
        normalization and is sufficient for the scalar-identity preconditioner.
        """

        return self._diag_dev

    def _accumulate_weighted_scores_from_xtv(
        self,
        scores: jnp.ndarray,
        XtV: jnp.ndarray,
        *,
        fp,
        scale=1.0,
        block_idx: int | None = None,
    ) -> jnp.ndarray:
        if block_idx is None and self._w_device_buckets:
            scale_dev = jnp.asarray(scale, dtype=fp)
            normalizer_dev = jnp.asarray(self.normalizer, dtype=fp)
            for bucket in self._w_device_buckets:
                scores = _accumulate_weighted_scores_bucket_jit(
                    scores,
                    XtV,
                    bucket.matrices,
                    bucket.starts,
                    bucket.sizes,
                    scale_dev,
                    normalizer_dev,
                )
            return scores

        blocks = enumerate(self.blocks)
        scale_dev = jnp.asarray(scale, dtype=fp)
        normalizer_dev = jnp.asarray(self.normalizer, dtype=fp)
        for idx, block in blocks:
            if block_idx is not None and idx != int(block_idx):
                continue
            if block.is_identity:
                local = XtV[block.start : block.stop, :]
            else:
                W_dev = (
                    block.device_matrix
                    if block.device_matrix is not None
                    else jax.device_put(jnp.asarray(block.matrix, dtype=fp), self.streamer.dev)
                )
                if block.device_matrix is not None and W_dev.dtype != fp:
                    W_dev = W_dev.astype(fp)
                local = W_dev @ XtV[block.start : block.stop, :]
            local = scale_dev * local / normalizer_dev
            scores = scores.at[block.start : block.stop, :].add(local)
        return scores

    def _weighted_scores(
        self,
        V: jnp.ndarray,
        block_idx: int | None = None,
    ) -> jnp.ndarray:
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        V = jax.device_put(jnp.asarray(V), self.streamer.dev)
        XtV = self.streamer.xtv(V, normalize=False)
        if XtV.ndim == 1:
            XtV = XtV[:, None]

        fp = V.dtype
        scores = jnp.zeros((int(self.streamer.m), int(V.shape[1])), dtype=fp)
        return self._accumulate_weighted_scores_from_xtv(
            scores,
            XtV,
            fp=fp,
            block_idx=block_idx,
        )

    def kv(self, V: jnp.ndarray) -> jnp.ndarray:
        """Return ``sum_i Z_i W_i Z_i.T V / c`` for one genetic kernel."""

        squeeze = V.ndim == 1
        scores = self._weighted_scores(V)
        out = _zxm_impl_streamed_from_scores(
            self.streamer,
            scores,
            missing_val=int(self.streamer._missing_val),
        )
        return out[:, 0] if squeeze else out

    def block_kv(self, V: jnp.ndarray, block_idx: int) -> jnp.ndarray:
        """Return one computational block contribution ``Z_i W_i Z_i.T V / c``."""

        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(f"block_idx={block_idx} out of range for {len(self.blocks)} blocks.")
        squeeze = V.ndim == 1
        scores = self._weighted_scores(V, block_idx=block_idx)
        out = _zxm_impl_streamed_from_scores(
            self.streamer,
            scores,
            missing_val=int(self.streamer._missing_val),
        )
        return out[:, 0] if squeeze else out

    def stacked_block_kv(self, V: jnp.ndarray) -> jnp.ndarray:
        """Return normalized computational block contributions stacked on axis 0."""

        return jnp.stack([self.block_kv(V, idx) for idx in range(len(self.blocks))], axis=0)

    def weighted_hv(
        self,
        theta_g: jnp.ndarray,
        theta_e: jnp.ndarray | None,
        V: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return ``theta_e V + theta_g K V`` with one genetic variance component."""

        theta_g_arr = jnp.asarray(theta_g)
        if theta_g_arr.ndim == 0:
            theta_g_scalar = theta_g_arr
        elif theta_g_arr.ndim == 1 and int(theta_g_arr.shape[0]) == 1:
            theta_g_scalar = theta_g_arr[0]
        else:
            raise ValueError("theta_g must be a scalar or a length-one array for SMILE block-W.")
        squeeze = V.ndim == 1
        if squeeze:
            V_work = V[:, None]
        else:
            V_work = V
        V_dev = jax.device_put(jnp.asarray(V_work), self.streamer.dev)
        scores = self._weighted_scores(V_dev)
        out = _zxm_impl_streamed_from_scores(
            self.streamer,
            scores,
            missing_val=int(self.streamer._missing_val),
        )
        out = jnp.asarray(theta_g_scalar, dtype=V_dev.dtype) * out
        if theta_e is not None:
            out = out + jnp.asarray(theta_e, dtype=V_dev.dtype) * V_dev
        return out[:, 0] if squeeze else out

    def snp_effects(self, alpha: jnp.ndarray, theta_g: jnp.ndarray) -> jnp.ndarray:
        """Return beta such that ``Z @ beta = theta_g * K @ alpha``."""

        theta_g_arr = jnp.asarray(theta_g)
        if theta_g_arr.ndim == 0:
            theta_g_scalar = theta_g_arr
        elif theta_g_arr.ndim == 1 and int(theta_g_arr.shape[0]) == 1:
            theta_g_scalar = theta_g_arr[0]
        else:
            raise ValueError("theta_g must be a scalar or a length-one array for SMILE block-W.")
        alpha_dev = jax.device_put(jnp.asarray(alpha), self.streamer.dev)
        Xtalpha = self.streamer.xtv(alpha_dev, normalize=False)
        if Xtalpha.ndim != 1:
            if int(Xtalpha.shape[1]) != 1:
                raise ValueError("snp_effects expects a single alpha vector.")
            Xtalpha = Xtalpha[:, 0]

        fp = Xtalpha.dtype
        beta = jnp.zeros((int(self.streamer.m),), dtype=fp)
        scale = jnp.asarray(theta_g_scalar, dtype=fp) / jnp.asarray(self.normalizer, dtype=fp)
        for block in self.blocks:
            if block.is_identity:
                local = scale * Xtalpha[block.start : block.stop]
            else:
                W_dev = (
                    block.device_matrix
                    if block.device_matrix is not None
                    else jax.device_put(jnp.asarray(block.matrix, dtype=fp), self.streamer.dev)
                )
                if block.device_matrix is not None and W_dev.dtype != fp:
                    W_dev = W_dev.astype(fp)
                local = scale * (W_dev @ Xtalpha[block.start : block.stop])
            beta = beta.at[block.start : block.stop].set(local)
        return beta


class SmileMultiBlockWeightedOperator:
    """A collection of SMILE GRMs sharing one genotype streamer.

    Each contained ``SmileBlockWeightedOperator`` is one REML genetic variance
    component.  Its internal blocks are computational terms summed inside that
    GRM, not separate variance components.
    """

    def __init__(self, operators: Sequence[SmileBlockWeightedOperator]):
        if not operators:
            raise ValueError("At least one SMILE operator is required.")
        streamer = operators[0].streamer
        for idx, op in enumerate(operators):
            if op.streamer is not streamer:
                raise ValueError(f"SMILE operator {idx} uses a different streamer.")
        self.streamer = streamer
        self.operators = tuple(operators)

    @classmethod
    def from_weight_matrix_groups(
        cls,
        streamer,
        weight_matrix_groups: Sequence[Sequence[np.ndarray]],
        *,
        normalization: Normalization = "kernel_trace",
        strict_coverage: bool = True,
        symmetrize: bool = True,
        check_psd: bool = True,
        source_groups: Sequence[Sequence[str | None]] | None = None,
        effective_rank_groups: Sequence[Sequence[float]] | None = None,
        diag_mode: DiagMode = "full",
        device_cache_max_bytes: float | None = None,
    ) -> "SmileMultiBlockWeightedOperator":
        groups = [list(group) for group in weight_matrix_groups]
        if not groups or any(len(group) == 0 for group in groups):
            raise ValueError("SMILE GRM groups must be non-empty.")
        if source_groups is not None and len(source_groups) != len(groups):
            raise ValueError("source_groups length must match weight_matrix_groups length.")
        if effective_rank_groups is not None and len(effective_rank_groups) != len(groups):
            raise ValueError("effective_rank_groups length must match weight_matrix_groups length.")

        operators: list[SmileBlockWeightedOperator] = []
        next_start = 0
        remaining_cache_bytes = float(device_cache_max_bytes or 0.0)
        for group_idx, group in enumerate(groups):
            starts: list[int] = []
            for matrix in group:
                W_arr = np.asarray(matrix)
                if W_arr.ndim != 2 or W_arr.shape[0] != W_arr.shape[1]:
                    raise ValueError(
                        f"W group {group_idx} contains a non-square matrix with shape {W_arr.shape}."
                    )
                starts.append(next_start)
                next_start += int(W_arr.shape[0])

            sources = None if source_groups is None else list(source_groups[group_idx])
            effective_ranks = (
                None if effective_rank_groups is None else list(effective_rank_groups[group_idx])
            )
            if effective_ranks is not None and len(effective_ranks) != len(group):
                raise ValueError(
                    f"effective_rank_groups[{group_idx}] length must match W group length."
                )
            operators.append(
                SmileBlockWeightedOperator(
                    streamer,
                    group,
                    normalization=normalization,
                    strict_coverage=False,
                    symmetrize=symmetrize,
                    check_psd=check_psd,
                    sources=sources,
                    start_offsets=starts,
                    effective_ranks=effective_ranks,
                    diag_mode=diag_mode,
                    device_cache_max_bytes=remaining_cache_bytes,
                )
            )
            remaining_cache_bytes = max(
                0.0,
                remaining_cache_bytes - float(operators[-1].w_device_cache_bytes),
            )

        if strict_coverage and next_start != int(streamer.m):
            raise ValueError(
                f"SMILE GRM groups cover {next_start} SNPs but streamer.m={int(streamer.m)}. "
                "Pass strict_coverage=False to intentionally ignore trailing SNPs."
            )
        return cls(operators)

    @classmethod
    def from_weight_file_groups(
        cls,
        streamer,
        path_groups: Sequence[Sequence[str | os.PathLike[str]]],
        **kwargs,
    ) -> "SmileMultiBlockWeightedOperator":
        matrix_groups = [[load_weight_matrix(path) for path in group] for group in path_groups]
        source_groups = [[os.fspath(path) for path in group] for group in path_groups]
        return cls.from_weight_matrix_groups(
            streamer,
            matrix_groups,
            source_groups=source_groups,
            **kwargs,
        )

    @property
    def n_grm(self) -> int:
        return len(self.operators)

    def diag_list(self) -> tuple[jnp.ndarray, ...]:
        return tuple(op.diag() for op in self.operators)

    def kv(self, V: jnp.ndarray, grm_idx: int) -> jnp.ndarray:
        return self.operators[int(grm_idx)].kv(V)

    def stacked_kv(self, V: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([op.kv(V) for op in self.operators], axis=0)

    def weighted_hv(
        self,
        theta_g: jnp.ndarray,
        theta_e: jnp.ndarray | None,
        V: jnp.ndarray,
    ) -> jnp.ndarray:
        theta_g_arr = jnp.asarray(theta_g)
        if theta_g_arr.ndim != 1 or int(theta_g_arr.shape[0]) != len(self.operators):
            raise ValueError(
                f"theta_g must be a length-{len(self.operators)} array for SMILE multi-GRM."
            )
        squeeze = V.ndim == 1
        if squeeze:
            V_work = V[:, None]
        else:
            V_work = V
        V_dev = jax.device_put(jnp.asarray(V_work), self.streamer.dev)
        XtV = self.streamer.xtv(V_dev, normalize=False)
        if XtV.ndim == 1:
            XtV = XtV[:, None]

        fp = V_dev.dtype
        scores = jnp.zeros((int(self.streamer.m), int(V_dev.shape[1])), dtype=fp)
        for idx, op in enumerate(self.operators):
            scores = op._accumulate_weighted_scores_from_xtv(
                scores,
                XtV,
                fp=fp,
                scale=theta_g_arr[idx],
            )
        out = _zxm_impl_streamed_from_scores(
            self.streamer,
            scores,
            missing_val=int(self.streamer._missing_val),
        )
        if theta_e is not None:
            out = out + jnp.asarray(theta_e, dtype=fp) * V_dev
        return out[:, 0] if squeeze else out

    def snp_effects(self, alpha: jnp.ndarray, theta_g: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        theta_g_arr = jnp.asarray(theta_g)
        if theta_g_arr.ndim != 1 or int(theta_g_arr.shape[0]) != len(self.operators):
            raise ValueError(
                f"theta_g must be a length-{len(self.operators)} array for SMILE multi-GRM."
            )
        return tuple(
            op.snp_effects(alpha, theta_g_arr[idx])
            for idx, op in enumerate(self.operators)
        )
