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

Normalization = Literal["kernel_trace", "weight_trace", "none"]
_IDENTITY_TRACE_SCAN_TARGET_BYTES = 256 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class SmileBlockWeight:
    """A single contiguous block W_i in source/cache order."""

    matrix: np.ndarray | None
    start: int
    size_value: int
    trace_per_sample: float
    source: str | None = None
    is_identity: bool = False

    @property
    def size(self) -> int:
        return int(self.size_value)

    @property
    def stop(self) -> int:
        return int(self.start + self.size)


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
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
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


def load_weight_kernel_trace_per_sample(path: str | os.PathLike[str]) -> float | None:
    """Load an optional precomputed kernel trace normalizer from sidecar JSON."""

    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("kernel_trace_per_sample", "trace_per_sample"):
        if key in meta:
            value = float(meta[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Invalid {key}={value!r} in SMILE W metadata {meta_path}."
                )
            return value
    return None


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
    check_psd: bool = True,
    psd_tol: float = 1e-7,
) -> np.ndarray:
    """Return a finite, square, symmetric float32 copy suitable for GPU use."""

    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype={arr.dtype}.")

    W = np.asarray(arr, dtype=np.float32, order="C")
    if not np.all(np.isfinite(W)):
        raise ValueError(f"{name} contains non-finite values.")

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


def _zxm_impl_streamed(
    streamer,
    b_by_call: jnp.ndarray,
    *,
    missing_val: int,
) -> jnp.ndarray:
    """Streaming matrix version of Z @ B using the streamer's packed cache."""

    if b_by_call.ndim != 3:
        raise ValueError("b_by_call must have shape (n_calls, max_unpack_width, rhs).")
    if b_by_call.shape[0] != int(streamer._n_calls):
        raise ValueError(
            f"b_by_call n_calls mismatch: expected {int(streamer._n_calls)}, "
            f"got {int(b_by_call.shape[0])}."
        )
    if b_by_call.shape[1] != int(streamer._max_unpack_width):
        raise ValueError(
            f"b_by_call width mismatch: expected {int(streamer._max_unpack_width)}, "
            f"got {int(b_by_call.shape[1])}."
        )

    streamer._prepare_kv_pass()
    fp = b_by_call.dtype
    dev = next(iter(b_by_call.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    acc = jnp.zeros((int(streamer.n), int(b_by_call.shape[2])), dtype=fp)
    if int(streamer._n_calls) == 0:
        return acc

    g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
    for c in range(int(streamer._n_calls)):
        g_dev_cur = g_dev_next
        if c + 1 < int(streamer._n_calls):
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        acc = _zxm_one_call_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            b_by_call[c],
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
        trace_per_sample_values: Sequence[float | None] | None = None,
    ):
        if normalization not in ("kernel_trace", "weight_trace", "none"):
            raise ValueError(
                "normalization must be one of 'kernel_trace', 'weight_trace', or 'none'."
            )
        if not weight_matrices:
            raise ValueError("At least one weight matrix is required.")
        if sources is not None and len(sources) != len(weight_matrices):
            raise ValueError("sources length must match weight_matrices length.")
        if start_offsets is not None and len(start_offsets) != len(weight_matrices):
            raise ValueError("start_offsets length must match weight_matrices length.")
        if trace_per_sample_values is not None and len(trace_per_sample_values) != len(weight_matrices):
            raise ValueError("trace_per_sample_values length must match weight_matrices length.")

        blocks: list[SmileBlockWeight] = []
        raw_diag = np.zeros((int(streamer.n),), dtype=np.float64)
        next_start = 0
        for idx, raw_W in enumerate(weight_matrices):
            source = None if sources is None else sources[idx]
            W = validate_weight_matrix(
                raw_W,
                name=f"W[{idx}]",
                symmetrize=symmetrize,
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
            precomputed_trace = (
                None
                if trace_per_sample_values is None
                else trace_per_sample_values[idx]
            )
            diag_contrib = self._compute_diag_contribution(
                streamer,
                W,
                start=start,
            )
            if normalization == "kernel_trace" and precomputed_trace is not None:
                trace_per_sample = float(precomputed_trace)
                if not np.isfinite(trace_per_sample) or trace_per_sample <= 0.0:
                    raise ValueError(
                        f"Invalid precomputed trace_per_sample for W[{idx}]: {trace_per_sample!r}."
                    )
            else:
                trace_per_sample = self._compute_trace_per_sample(
                    diag_contrib,
                    W,
                    normalization=normalization,
                )
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
        self.normalization = normalization
        self.normalizer = self._compute_global_normalizer(
            [block.trace_per_sample for block in blocks],
            normalization=normalization,
        )
        self._diag_host = np.asarray(raw_diag / float(self.normalizer), dtype=np.float32)
        self._diag_dev = jax.device_put(jnp.asarray(self._diag_host), streamer.dev)
        self.strict_coverage = bool(strict_coverage)

    @classmethod
    def identity(
        cls,
        streamer,
        *,
        block_size: int | None = None,
        normalization: Normalization = "kernel_trace",
        strict_coverage: bool = True,
    ) -> "SmileBlockWeightedOperator":
        """Create the SMILE operator for W=I without materializing I."""

        if normalization not in ("kernel_trace", "weight_trace", "none"):
            raise ValueError(
                "normalization must be one of 'kernel_trace', 'weight_trace', or 'none'."
            )
        if block_size is None:
            block_size = cls._auto_identity_block_size(streamer, normalization=normalization)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if int(streamer.m) <= 0:
            raise ValueError("streamer must contain at least one SNP.")

        blocks: list[SmileBlockWeight] = []
        use_unit_diag = normalization == "weight_trace"
        raw_diag = (
            np.full((int(streamer.n),), float(streamer.m), dtype=np.float64)
            if use_unit_diag
            else np.zeros((int(streamer.n),), dtype=np.float64)
        )
        start = 0
        while start < int(streamer.m):
            size = min(block_size, int(streamer.m) - start)
            if use_unit_diag:
                diag_contrib = np.zeros((0,), dtype=np.float64)
            else:
                diag_contrib = cls._compute_diag_contribution(
                    streamer,
                    None,
                    start=start,
                    size=size,
                )
                raw_diag += diag_contrib
            trace_per_sample = cls._compute_trace_per_sample(
                diag_contrib,
                None,
                normalization=normalization,
                identity_size=size,
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
        self.normalization = normalization
        self.normalizer = cls._compute_global_normalizer(
            [block.trace_per_sample for block in blocks],
            normalization=normalization,
        )
        self._diag_host = np.asarray(raw_diag / float(self.normalizer), dtype=np.float32)
        self._diag_dev = jax.device_put(jnp.asarray(self._diag_host), streamer.dev)
        self.strict_coverage = bool(strict_coverage)
        return self

    @staticmethod
    def _auto_identity_block_size(
        streamer,
        *,
        normalization: Normalization,
    ) -> int:
        if normalization == "weight_trace":
            return int(streamer.m)
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
        trace_values = [load_weight_kernel_trace_per_sample(path) for path in paths]
        return cls(
            streamer,
            matrices,
            sources=[os.fspath(path) for path in paths],
            trace_per_sample_values=trace_values,
            **kwargs,
        )

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
        W: np.ndarray | None,
        *,
        normalization: Normalization,
        identity_size: int | None = None,
    ) -> float:
        if normalization == "none":
            return 0.0
        if normalization == "weight_trace":
            if W is None:
                if identity_size is None:
                    raise ValueError("identity_size is required for identity weight_trace.")
                value = float(identity_size)
            else:
                value = float(np.trace(np.asarray(W, dtype=np.float64)))
        else:
            diag_arr = np.asarray(diag_contrib, dtype=np.float64)
            value = float(np.sum(diag_arr) / float(diag_arr.size))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid SMILE block trace contribution: {value!r}.")
        return value

    @staticmethod
    def _compute_global_normalizer(
        trace_values: Sequence[float],
        *,
        normalization: Normalization,
    ) -> float:
        if normalization == "none":
            return 1.0
        value = float(np.sum(np.asarray(trace_values, dtype=np.float64)))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid SMILE global normalizer: {value!r}.")
        return value

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def diag(self) -> jnp.ndarray:
        """Return ``diag(Z W Z.T / c)`` without forming the dense kernel."""

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
        blocks = enumerate(self.blocks)
        scale_dev = jnp.asarray(scale, dtype=fp)
        normalizer_dev = jnp.asarray(self.normalizer, dtype=fp)
        for idx, block in blocks:
            if block_idx is not None and idx != int(block_idx):
                continue
            if block.is_identity:
                local = XtV[block.start : block.stop, :]
            else:
                W_dev = jax.device_put(jnp.asarray(block.matrix, dtype=fp), self.streamer.dev)
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

    def _scores_by_call(self, scores: jnp.ndarray) -> jnp.ndarray:
        fp = scores.dtype
        rhs = int(scores.shape[1])
        b_by_call = jnp.zeros(
            (int(self.streamer._n_calls), int(self.streamer._max_unpack_width), rhs),
            dtype=fp,
        )
        for c in range(int(self.streamer._n_calls)):
            start = int(self.streamer._call_snp_starts[c])
            width = int(self.streamer._call_true_widths[c])
            if width > 0:
                b_by_call = b_by_call.at[c, :width, :].set(scores[start : start + width, :])
        return b_by_call

    def kv(self, V: jnp.ndarray) -> jnp.ndarray:
        """Return ``sum_i Z_i W_i Z_i.T V / c`` for one genetic kernel."""

        squeeze = V.ndim == 1
        scores = self._weighted_scores(V)
        out = _zxm_impl_streamed(
            self.streamer,
            self._scores_by_call(scores),
            missing_val=int(self.streamer._missing_val),
        )
        return out[:, 0] if squeeze else out

    def block_kv(self, V: jnp.ndarray, block_idx: int) -> jnp.ndarray:
        """Return one computational block contribution ``Z_i W_i Z_i.T V / c``."""

        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(f"block_idx={block_idx} out of range for {len(self.blocks)} blocks.")
        squeeze = V.ndim == 1
        scores = self._weighted_scores(V, block_idx=block_idx)
        out = _zxm_impl_streamed(
            self.streamer,
            self._scores_by_call(scores),
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
        out = _zxm_impl_streamed(
            self.streamer,
            self._scores_by_call(scores),
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
                W_dev = jax.device_put(jnp.asarray(block.matrix, dtype=fp), self.streamer.dev)
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
        trace_per_sample_groups: Sequence[Sequence[float | None]] | None = None,
    ) -> "SmileMultiBlockWeightedOperator":
        groups = [list(group) for group in weight_matrix_groups]
        if not groups or any(len(group) == 0 for group in groups):
            raise ValueError("SMILE GRM groups must be non-empty.")
        if source_groups is not None and len(source_groups) != len(groups):
            raise ValueError("source_groups length must match weight_matrix_groups length.")
        if trace_per_sample_groups is not None and len(trace_per_sample_groups) != len(groups):
            raise ValueError("trace_per_sample_groups length must match weight_matrix_groups length.")

        operators: list[SmileBlockWeightedOperator] = []
        next_start = 0
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
            trace_values = (
                None
                if trace_per_sample_groups is None
                else list(trace_per_sample_groups[group_idx])
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
                    trace_per_sample_values=trace_values,
                )
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
        trace_groups = [
            [load_weight_kernel_trace_per_sample(path) for path in group]
            for group in path_groups
        ]
        return cls.from_weight_matrix_groups(
            streamer,
            matrix_groups,
            source_groups=source_groups,
            trace_per_sample_groups=trace_groups,
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
        out = _zxm_impl_streamed(
            self.streamer,
            self.operators[0]._scores_by_call(scores),
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
