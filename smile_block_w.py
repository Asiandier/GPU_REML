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


@dataclasses.dataclass(frozen=True)
class SmileBlockWeight:
    """A single contiguous block W_i in source/cache order."""

    matrix: np.ndarray
    start: int
    trace_per_sample: float
    source: str | None = None

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

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
        return np.load(path)
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

    W64 = np.asarray(matrix, dtype=np.float64)
    if W64.ndim != 2 or W64.shape[0] != W64.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {W64.shape}.")
    if W64.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(W64)):
        raise ValueError(f"{name} contains non-finite values.")

    max_abs = float(np.max(np.abs(W64))) if W64.size else 0.0
    asym = float(np.max(np.abs(W64 - W64.T)))
    allowed_asym = float(symmetry_tol * max(1.0, max_abs))
    if asym > allowed_asym:
        raise ValueError(f"{name} is not symmetric: max |W-W.T|={asym:g}.")
    if symmetrize:
        W64 = 0.5 * (W64 + W64.T)

    if check_psd:
        eigvals = np.linalg.eigvalsh(W64)
        eig_min = float(eigvals[0])
        eig_max = float(eigvals[-1])
        if eig_min < -float(psd_tol) * max(1.0, abs(eig_max)):
            raise ValueError(f"{name} is not positive semidefinite: min eigenvalue={eig_min:g}.")

    return np.asarray(W64, dtype=np.float32, order="C")


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
    ):
        if normalization not in ("kernel_trace", "weight_trace", "none"):
            raise ValueError(
                "normalization must be one of 'kernel_trace', 'weight_trace', or 'none'."
            )
        if not weight_matrices:
            raise ValueError("At least one weight matrix is required.")
        if sources is not None and len(sources) != len(weight_matrices):
            raise ValueError("sources length must match weight_matrices length.")

        blocks: list[SmileBlockWeight] = []
        start = 0
        for idx, raw_W in enumerate(weight_matrices):
            source = None if sources is None else sources[idx]
            W = validate_weight_matrix(
                raw_W,
                name=f"W[{idx}]",
                symmetrize=symmetrize,
                check_psd=check_psd,
            )
            stop = start + int(W.shape[0])
            if stop > int(streamer.m):
                raise ValueError(
                    f"W[{idx}] with size {W.shape[0]} exceeds streamer.m={int(streamer.m)}."
                )
            trace_per_sample = self._compute_trace_per_sample(
                streamer,
                W,
                start=start,
                normalization=normalization,
            )
            blocks.append(
                SmileBlockWeight(
                    matrix=W,
                    start=start,
                    trace_per_sample=trace_per_sample,
                    source=source,
                )
            )
            start = stop

        if strict_coverage and start != int(streamer.m):
            raise ValueError(
                f"Block W matrices cover {start} SNPs but streamer.m={int(streamer.m)}. "
                "Pass strict_coverage=False to intentionally ignore trailing SNPs."
            )

        self.streamer = streamer
        self.blocks = tuple(blocks)
        self.normalization = normalization
        self.normalizer = self._compute_global_normalizer(
            [block.trace_per_sample for block in blocks],
            normalization=normalization,
        )
        self.strict_coverage = bool(strict_coverage)

    @classmethod
    def from_weight_files(
        cls,
        streamer,
        paths: Sequence[str | os.PathLike[str]],
        **kwargs,
    ) -> "SmileBlockWeightedOperator":
        matrices = [load_weight_matrix(path) for path in paths]
        return cls(streamer, matrices, sources=[os.fspath(path) for path in paths], **kwargs)

    @staticmethod
    def _compute_trace_per_sample(
        streamer,
        W: np.ndarray,
        *,
        start: int,
        normalization: Normalization,
    ) -> float:
        if normalization == "none":
            return 0.0
        if normalization == "weight_trace":
            value = float(np.trace(np.asarray(W, dtype=np.float64)))
        else:
            idx = np.arange(int(start), int(start + W.shape[0]), dtype=np.int64)
            Z = np.asarray(streamer.extract_standardized_columns(idx), dtype=np.float64)
            value = float(np.sum((Z @ np.asarray(W, dtype=np.float64)) * Z) / float(streamer.n))
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
        blocks = enumerate(self.blocks)
        for idx, block in blocks:
            if block_idx is not None and idx != int(block_idx):
                continue
            W_dev = jax.device_put(jnp.asarray(block.matrix, dtype=fp), self.streamer.dev)
            local = W_dev @ XtV[block.start : block.stop, :]
            local = local / jnp.asarray(self.normalizer, dtype=fp)
            scores = scores.at[block.start : block.stop, :].set(local)
        return scores

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
