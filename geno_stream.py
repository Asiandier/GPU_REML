"""
geno_stream.py — Streaming genotype reader with low-memory fast build paths.

Architecture
------------
1.  **Build**: reads genotype blocks from a ``GenoBlockSource`` (BED or
    PGEN), computes per-SNP stats, and packs to a 2-bit tmpfile-backed cache.
    Host peak is dominated by O(nw) staging buffers instead of a persistent
    O(np/4) heap allocation.

2.  **Streaming**: the packed cache is served from a read-only mmap over the
    tmpfile. The OS page cache decides what stays resident in RAM.

3.  **call_width** (= w) controls runtime GPU block geometry. For
    arbitrary single-source regrouping, build-time source scans can use a
    separate ``source_build_chunk_width`` to avoid tying CPU preprocessing
    throughput to runtime kernel width.

The ``GenoBlockSource`` abstraction (defined in ``geno_source.py``) decouples
file-format I/O from the GPU streaming pipeline. BED uses a raw-byte fast
path; PGEN uses a variant-major fast path.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import mmap
import os
import threading
import time
import numpy as np
import jax
import jax.numpy as jnp
from numba import get_num_threads, njit, prange, set_num_threads

logger = logging.getLogger(__name__)

from .block_backend import DensePackedBlockDescriptor


def _resolve_tmpdir() -> str | None:
    for key in ("GPU_REML_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Tmpfile page-evicting ring — RSS bounded at DEPTH × block_size
# ---------------------------------------------------------------------------

_libcudart_handle = None


def _get_libcudart():
    global _libcudart_handle
    if _libcudart_handle is None:
        import ctypes, ctypes.util
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
            lib.cudaSetDevice.argtypes = [ctypes.c_int]
            lib.cudaSetDevice.restype = ctypes.c_int
            lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            lib.cudaHostUnregister.restype = ctypes.c_int
            _libcudart_handle = lib
        except (OSError, AttributeError, TypeError):
            logger.debug("CUDA runtime library initialization failed; pinned host buffers disabled.", exc_info=True)
    return _libcudart_handle


def _cuda_device_ordinal(dev) -> int | None:
    if getattr(dev, "platform", None) != "gpu":
        return None
    for attr in ("local_hardware_id", "id"):
        try:
            value = int(getattr(dev, attr))
            if value >= 0:
                return value
        except (AttributeError, TypeError, ValueError):
            logger.debug("Could not read CUDA device ordinal attribute %s.", attr, exc_info=True)
    return None


def _pin_buffer(arr: np.ndarray, dev) -> bool:
    """Register a numpy buffer as CUDA pinned memory for direct DMA.

    Use the CUDA runtime API rather than the driver API so worker threads do
    not fail with "invalid device context" when building streamers in parallel.
    """
    import ctypes
    if not arr.data.contiguous:
        return False
    ordinal = _cuda_device_ordinal(dev)
    lib = _get_libcudart()
    if lib is None or ordinal is None:
        return False
    try:
        if lib.cudaSetDevice(ctypes.c_int(int(ordinal))) != 0:
            return False
        rc = lib.cudaHostRegister(
            ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes),
            ctypes.c_uint(0),
        )
        return rc == 0
    except (OSError, AttributeError, TypeError, ValueError):
        logger.debug("CUDA host buffer registration failed.", exc_info=True)
        return False


def _unpin_buffer(arr: np.ndarray, dev) -> None:
    import ctypes
    ordinal = _cuda_device_ordinal(dev)
    lib = _get_libcudart()
    if lib is None:
        return
    try:
        if ordinal is not None:
            lib.cudaSetDevice(ctypes.c_int(int(ordinal)))
        lib.cudaHostUnregister(ctypes.c_void_p(arr.ctypes.data))
    except (OSError, AttributeError, TypeError, ValueError):
        logger.debug("CUDA host buffer unregistration failed.", exc_info=True)


class _PinnedHostBuffer(np.ndarray):
    """Numpy ndarray view backed by host memory registered for direct GPU DMA."""


class _EvictRing:
    """Pinned staging buffers + trail-behind page eviction.

    Staging buffers are registered as CUDA pinned memory so that
    jax.device_put can DMA directly without an internal copy.
    Up to 4 workers read blocks in a strided pattern (worker k handles
    blocks k, k+step, k+2*step, ...); each worker independently evicts
    pages TRAIL_BEHIND blocks behind its own position to preserve
    kernel readahead.
    """

    DEPTH = 32

    def __init__(self, packed_mmap, packed_np, packed_offsets,
                 n_calls, n, max_packed_width, depth=None, dev=None):
        self._mm = packed_mmap
        self._np = packed_np
        self._offsets = packed_offsets
        self._n_calls = int(n_calls)
        self._n = int(n)
        self._mpw = int(max_packed_width)
        self._dev = dev

        if depth is not None:
            self.DEPTH = int(depth)
        self.DEPTH = min(self.DEPTH, max(1, self._n_calls))
        buf_bytes = self._n * self._mpw
        self._bufs = []
        self._pinned = 0
        for _ in range(self.DEPTH):
            buf = np.zeros(buf_bytes, dtype=np.uint8)
            if _pin_buffer(buf, self._dev):
                buf = buf.view(_PinnedHostBuffer)
                self._pinned += 1
            self._bufs.append(buf)
        self._ready = [threading.Event() for _ in range(self.DEPTH)]
        self._writable = [threading.Event() for _ in range(self.DEPTH)]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        for ev in self._writable:
            ev.set()

    def start_pass(self, prefill: int = 2):
        """Start worker and optionally pre-fill first N slots before returning."""
        self.finish_pass()
        self._stop.clear()
        for ev in self._ready:
            ev.clear()
        for ev in self._writable:
            ev.set()
        self._threads = []
        n_workers = min(4, self._n_calls)
        for tid in range(n_workers):
            th = threading.Thread(target=self._worker, args=(tid, n_workers), daemon=True)
            th.start()
            self._threads.append(th)

        # Pre-fill: wait for first N slots so get(0) never cold-stalls
        for i in range(min(prefill, self._n_calls)):
            self._ready[i % self.DEPTH].wait()

    _TRAIL_BEHIND = 8  # evict this far behind worker — preserves readahead

    def _worker(self, start_c, step):
        """Copy blocks in strided pattern. Evict TRAIL_BEHIND blocks behind."""
        c = start_c
        while c < self._n_calls:
            if self._stop.is_set():
                return
            slot = c % self.DEPTH
            while not self._stop.is_set():
                if self._writable[slot].wait(timeout=0.1):
                    break
            if self._stop.is_set():
                return
            self._writable[slot].clear()

            off0 = int(self._offsets[c])
            off1 = int(self._offsets[c + 1])
            self._bufs[slot][:off1 - off0] = self._np[off0:off1]

            # Evict pages far behind — readahead window (~2MB) is untouched
            if c >= self._TRAIL_BEHIND:
                self._evict_block(c - self._TRAIL_BEHIND)

            self._ready[slot].set()
            c += step

    def _evict_block(self, c):
        """DONTNEED one block's tmpfile pages (called from worker thread)."""
        if c < 0 or c >= self._n_calls:
            return
        off0 = int(self._offsets[c])
        off1 = int(self._offsets[c + 1])
        ps = mmap.PAGESIZE
        ev_start = ((off0 + ps - 1) // ps) * ps
        ev_end = (off1 // ps) * ps
        if ev_end > ev_start and self._mm is not None:
            try:
                self._mm.madvise(mmap.MADV_DONTNEED, ev_start, ev_end - ev_start)
            except (AttributeError, OSError, ValueError):
                logger.debug("mmap DONTNEED advisory failed for ring block %s.", c, exc_info=True)

    def get(self, c):
        c = int(c)
        # Release old ring slot (eviction handled by worker)
        if c >= self.DEPTH:
            old_slot = (c - self.DEPTH) % self.DEPTH
            self._ready[old_slot].clear()
            self._writable[old_slot].set()

        slot = c % self.DEPTH
        self._ready[slot].wait()
        off0 = int(self._offsets[c])
        off1 = int(self._offsets[c + 1])
        return self._bufs[slot][:off1 - off0]

    def finish_pass(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads = []
        # Don't blanket-DONTNEED — trail-behind already evicted most pages.
        # Keeping residual pages warm avoids cold-start stalls on next pass.

    def close(self):
        self.finish_pass()
        if self._bufs is not None:
            for buf in self._bufs:
                _unpin_buffer(buf, self._dev)
        self._bufs = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_full(fd: int, data) -> None:
    """Write all bytes to fd, handling partial writes."""
    mv = memoryview(data).cast("B")
    total = len(mv)
    written = 0
    while written < total:
        n = os.write(fd, mv[written:])
        if n <= 0:
            raise OSError("os.write returned 0")
        written += n

def _to_device(dev) -> jax.Device:
    if dev is None or dev == "auto":
        return jax.devices()[0]
    if isinstance(dev, str):
        return jax.devices(dev)[0]
    return dev


def _ensure_on_device(x, dev):
    try:
        if isinstance(x, jax.Array):
            devs = x.devices()
            if len(devs) == 1 and next(iter(devs)) == dev:
                return x
    except (AttributeError, RuntimeError, TypeError):
        logger.debug("Could not determine existing JAX array device; copying to requested device.", exc_info=True)
    return jax.device_put(x, dev)


@dataclasses.dataclass(frozen=True)
class _ComponentPartitionPlan:
    component_sizes: tuple[int, ...]
    cache_to_source_variant_indices: np.ndarray
    has_arbitrary_groups: bool


def _normalize_component_block_sizes(
    m: int,
    component_block_sizes,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if component_block_sizes is None:
        return (int(m),)
    sizes = np.asarray(component_block_sizes, dtype=np.int64).reshape(-1)
    if sizes.size == 0:
        raise ValueError("component_block_sizes must contain at least one block.")
    if allow_empty:
        if np.any(sizes < 0):
            raise ValueError("component_block_sizes must be non-negative.")
    elif np.any(sizes <= 0):
        raise ValueError("component_block_sizes must be strictly positive.")
    total = int(np.sum(sizes))
    if total != int(m):
        raise ValueError(
            f"component_block_sizes must sum to m={int(m)}; got {total}."
        )
    return tuple(int(x) for x in sizes.tolist())


def _normalize_component_variant_indices(
    m: int,
    component_variant_indices,
) -> tuple[tuple[int, ...], np.ndarray]:
    groups = list(component_variant_indices or [])
    if not groups:
        raise ValueError("component_variant_indices must contain at least one component.")

    normalized: list[np.ndarray] = []
    for comp_idx, group in enumerate(groups):
        arr = np.asarray(group, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            normalized.append(arr)
            continue
        if np.any((arr < 0) | (arr >= int(m))):
            raise IndexError(
                f"component_variant_indices[{comp_idx}] contains out-of-range SNP indices for m={int(m)}."
            )
        uniq = np.unique(arr)
        if uniq.size != arr.size:
            raise ValueError(
                f"component_variant_indices[{comp_idx}] contains duplicate SNP indices."
            )
        normalized.append(uniq)

    if normalized:
        cache_to_source = np.concatenate(normalized, axis=0)
    else:
        cache_to_source = np.empty((0,), dtype=np.int64)
    if cache_to_source.size == 0:
        raise ValueError("component_variant_indices must cover at least one SNP.")
    if cache_to_source.size > int(m):
        raise ValueError(
            f"component_variant_indices assign {cache_to_source.size} SNPs, "
            f"which exceeds source m={int(m)}."
        )
    if np.unique(cache_to_source).size != cache_to_source.size:
        raise ValueError("component_variant_indices must be disjoint across components.")
    if cache_to_source.size == int(m):
        src_sorted = np.sort(cache_to_source)
        if not np.array_equal(src_sorted, np.arange(int(m), dtype=np.int64)):
            raise ValueError("component_variant_indices must cover every SNP exactly once.")
    component_sizes = tuple(int(arr.size) for arr in normalized)
    return component_sizes, cache_to_source


def _build_component_partition_plan(
    m: int,
    *,
    component_block_sizes=None,
    component_variant_indices=None,
) -> _ComponentPartitionPlan:
    if component_block_sizes is not None and component_variant_indices is not None:
        raise ValueError(
            "Provide either component_block_sizes or component_variant_indices, not both."
        )
    if component_variant_indices is not None:
        component_sizes, cache_to_source = _normalize_component_variant_indices(
            m,
            component_variant_indices,
        )
    else:
        component_sizes = _normalize_component_block_sizes(m, component_block_sizes)
        cache_to_source = np.arange(int(m), dtype=np.int64)
    return _ComponentPartitionPlan(
        component_sizes=component_sizes,
        cache_to_source_variant_indices=cache_to_source,
        has_arbitrary_groups=not np.array_equal(
            cache_to_source,
            np.arange(int(m), dtype=np.int64),
        ),
    )


def _build_call_geometry(
    m: int,
    call_width: int,
    component_block_sizes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sizes = _normalize_component_block_sizes(m, component_block_sizes, allow_empty=True)
    if m == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.zeros((len(sizes) + 1,), dtype=np.int32),
        )

    starts: list[int] = []
    widths: list[int] = []
    component_ids: list[int] = []
    component_call_offsets = [0]
    snp_start = 0

    for comp_id, comp_size in enumerate(sizes):
        remaining = int(comp_size)
        while remaining > 0:
            width = min(int(call_width), remaining)
            starts.append(snp_start)
            widths.append(width)
            component_ids.append(comp_id)
            snp_start += width
            remaining -= width
        component_call_offsets.append(len(starts))

    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(widths, dtype=np.int32),
        np.asarray(component_ids, dtype=np.int32),
        np.asarray(component_call_offsets, dtype=np.int32),
    )


def _build_call_source_segments(
    cache_to_source_variant_indices: np.ndarray,
    call_snp_starts: np.ndarray,
    call_true_widths: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for call_start, true_width in zip(call_snp_starts.tolist(), call_true_widths.tolist()):
        tw = int(true_width)
        if tw <= 0:
            segments.append(
                (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int32),
                )
            )
            continue
        src_idx = np.asarray(
            cache_to_source_variant_indices[int(call_start) : int(call_start) + tw],
            dtype=np.int64,
        )
        if src_idx.size == 0:
            segments.append(
                (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int32),
                )
            )
            continue
        starts = [int(src_idx[0])]
        widths = [1]
        prev = int(src_idx[0])
        for value in src_idx[1:]:
            cur = int(value)
            if cur == prev + 1:
                widths[-1] += 1
            else:
                starts.append(cur)
                widths.append(1)
            prev = cur
        segments.append(
            (
                np.asarray(starts, dtype=np.int64),
                np.asarray(widths, dtype=np.int32),
            )
        )
    return tuple(segments)


def _build_source_scatter_plan(
    m: int,
    cache_to_source_variant_indices: np.ndarray,
    call_snp_starts: np.ndarray,
    call_true_widths: np.ndarray,
    packed_call_widths: np.ndarray,
    packed_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_to_source = np.asarray(cache_to_source_variant_indices, dtype=np.int64)
    source_to_cache = np.full((int(m),), -1, dtype=np.int64)
    source_to_cache[cache_to_source] = np.arange(cache_to_source.size, dtype=np.int64)
    valid = source_to_cache >= 0

    source_call_idx = np.zeros((int(m),), dtype=np.int32)
    source_local_col = np.zeros((int(m),), dtype=np.int32)
    source_byte_idx = np.zeros((int(m),), dtype=np.int32)
    source_bit_shift = np.zeros((int(m),), dtype=np.uint8)
    source_call_offsets = np.zeros((int(m),), dtype=np.int64)
    source_packed_widths = np.zeros((int(m),), dtype=np.int32)
    if np.any(valid):
        call_idx_valid = np.searchsorted(
            np.asarray(call_snp_starts, dtype=np.int64),
            source_to_cache[valid],
            side="right",
        ) - 1
        call_idx_valid = np.clip(call_idx_valid, 0, len(call_snp_starts) - 1).astype(np.int32)
        local_col_valid = (
            source_to_cache[valid]
            - np.asarray(call_snp_starts, dtype=np.int64)[call_idx_valid]
        ).astype(np.int32)
        source_call_idx[valid] = call_idx_valid
        source_local_col[valid] = local_col_valid
        source_byte_idx[valid] = (local_col_valid >> 2).astype(np.int32)
        source_bit_shift[valid] = ((local_col_valid & 3) << 1).astype(np.uint8)
        source_call_offsets[valid] = np.asarray(packed_offsets[:-1], dtype=np.int64)[call_idx_valid]
        source_packed_widths[valid] = np.asarray(packed_call_widths, dtype=np.int32)[call_idx_valid]
    return (
        source_to_cache,
        source_call_offsets,
        source_packed_widths,
        source_byte_idx,
        source_bit_shift,
    )


def _iter_source_order_build_spans(
    source_m: int,
    cache_to_source_variant_indices: np.ndarray,
    chunk_width: int,
):
    """Yield source-order spans that contain only SNPs covered by the GRM cache."""
    cache_to_source = np.asarray(cache_to_source_variant_indices, dtype=np.int64)
    if cache_to_source.size == int(source_m) and np.array_equal(
        cache_to_source,
        np.arange(int(source_m), dtype=np.int64),
    ):
        for source_start in range(0, int(source_m), max(1, int(chunk_width))):
            width = min(max(1, int(chunk_width)), int(source_m) - int(source_start))
            yield int(source_start), int(width)
        return

    selected = np.sort(cache_to_source)
    if selected.size == 0:
        return
    width = max(1, int(chunk_width))
    pos = 0
    while pos < selected.size:
        source_start = int(selected[pos])
        source_stop = min(int(source_m), source_start + width)
        yield source_start, source_stop - source_start
        pos = int(np.searchsorted(selected, source_stop, side="left"))


@njit(cache=True, nogil=True, parallel=True)
def _compute_stats_numba_forder(block_sub: np.ndarray, missing_val: int):
    n_rows, n_cols = block_sub.shape
    cnt = np.zeros(n_cols, dtype=np.int64)
    s1  = np.zeros(n_cols, dtype=np.int64)
    s2  = np.zeros(n_cols, dtype=np.int64)
    for j in prange(n_cols):
        for i in range(n_rows):
            v = int(block_sub[i, j])
            if v == missing_val:
                continue
            cnt[j] += 1
            if v == 1:
                s1[j] += 1;  s2[j] += 1
            elif v == 2:
                s1[j] += 2;  s2[j] += 4
            elif v != 0:
                s1[j] += v;  s2[j] += v * v
    return cnt, s1, s2


@njit(cache=True, nogil=True, parallel=True)
def _compute_stats_numba_varmaj(block_vm: np.ndarray, missing_val: int):
    n_cols, n_rows = block_vm.shape
    cnt = np.zeros(n_cols, dtype=np.int64)
    s1  = np.zeros(n_cols, dtype=np.int64)
    s2  = np.zeros(n_cols, dtype=np.int64)
    for j in prange(n_cols):
        for i in range(n_rows):
            v = int(block_vm[j, i])
            if v == missing_val:
                continue
            cnt[j] += 1
            if v == 1:
                s1[j] += 1;  s2[j] += 1
            elif v == 2:
                s1[j] += 2;  s2[j] += 4
            elif v != 0:
                s1[j] += v;  s2[j] += v * v
    return cnt, s1, s2


@njit(cache=True, nogil=True, parallel=True)
def _stats_from_raw_bed_numba(
    bed_raw,
    snp_start,
    snp_count,
    bytes_per_snp,
    sample_byte_offsets,
    sample_bit_shifts,
    cnt_out,
    s1_out,
    s2_out,
):
    """Compute per-variant stats directly from raw SNP-major BED bytes."""
    n_keep = sample_byte_offsets.shape[0]
    hdr = np.int64(3)
    for j in prange(snp_count):
        row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
        c = np.int64(0)
        a1 = np.int64(0)
        a2 = np.int64(0)
        for i in range(n_keep):
            raw = bed_raw[row_base + np.int64(sample_byte_offsets[i])]
            g = (raw >> np.uint8(sample_bit_shifts[i])) & np.uint8(3)
            if g == np.uint8(1):
                continue
            c += 1
            if g == np.uint8(0):
                a1 += 2
                a2 += 4
            elif g == np.uint8(2):
                a1 += 1
                a2 += 1
        cnt_out[j] = c
        s1_out[j] = a1
        s2_out[j] = a2


@njit(cache=True, nogil=True, parallel=True)
def _transcode_raw_bed_numba(
    bed_raw,
    snp_start,
    snp_count,
    bytes_per_snp,
    sample_byte_offsets,
    sample_bit_shifts,
    packed_out,
):
    """Transcode BED 2-bit encoding directly into our packed representation."""
    n_keep = sample_byte_offsets.shape[0]
    packed_cols = packed_out.shape[1]
    lut = np.array([np.uint8(2), np.uint8(3), np.uint8(1), np.uint8(0)])
    hdr = np.int64(3)
    for i in prange(n_keep):
        s_boff = np.int64(sample_byte_offsets[i])
        s_bshift = np.uint8(sample_bit_shifts[i])
        for pc in range(packed_cols):
            snp_base = pc << 2
            val = np.uint8(0)
            if snp_base < snp_count:
                r0 = hdr + np.int64(snp_start + snp_base) * np.int64(bytes_per_snp)
                g = (bed_raw[r0 + s_boff] >> s_bshift) & np.uint8(3)
                val = lut[g]
            if snp_base + 1 < snp_count:
                r1 = hdr + np.int64(snp_start + snp_base + 1) * np.int64(bytes_per_snp)
                g = (bed_raw[r1 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g] << np.uint8(2)
            if snp_base + 2 < snp_count:
                r2 = hdr + np.int64(snp_start + snp_base + 2) * np.int64(bytes_per_snp)
                g = (bed_raw[r2 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g] << np.uint8(4)
            if snp_base + 3 < snp_count:
                r3 = hdr + np.int64(snp_start + snp_base + 3) * np.int64(bytes_per_snp)
                g = (bed_raw[r3 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g] << np.uint8(6)
            packed_out[i, pc] = val


@njit(cache=True, nogil=True, parallel=True)
def _stats_and_transcode_raw_bed_numba(
    bed_raw,
    snp_start,
    snp_count,
    bytes_per_snp,
    sample_byte_offsets,
    sample_bit_shifts,
    cnt_out,
    s1_out,
    s2_out,
    packed_out,
):
    """Fuse raw-BED stats and packed transcode in one pass over SNP groups."""
    n_keep = sample_byte_offsets.shape[0]
    packed_cols = packed_out.shape[1]
    lut = np.array([np.uint8(2), np.uint8(3), np.uint8(1), np.uint8(0)])
    hdr = np.int64(3)
    for pc in prange(packed_cols):
        snp_base = pc << 2
        c0 = np.int64(0)
        c1 = np.int64(0)
        c2 = np.int64(0)
        c3 = np.int64(0)
        a10 = np.int64(0)
        a11 = np.int64(0)
        a12 = np.int64(0)
        a13 = np.int64(0)
        a20 = np.int64(0)
        a21 = np.int64(0)
        a22 = np.int64(0)
        a23 = np.int64(0)
        for i in range(n_keep):
            s_boff = np.int64(sample_byte_offsets[i])
            s_bshift = np.uint8(sample_bit_shifts[i])
            val = np.uint8(0)
            if snp_base < snp_count:
                r0 = hdr + np.int64(snp_start + snp_base) * np.int64(bytes_per_snp)
                g0 = (bed_raw[r0 + s_boff] >> s_bshift) & np.uint8(3)
                val = lut[g0]
                if g0 != np.uint8(1):
                    c0 += 1
                    if g0 == np.uint8(0):
                        a10 += 2
                        a20 += 4
                    elif g0 == np.uint8(2):
                        a10 += 1
                        a20 += 1
            if snp_base + 1 < snp_count:
                r1 = hdr + np.int64(snp_start + snp_base + 1) * np.int64(bytes_per_snp)
                g1 = (bed_raw[r1 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g1] << np.uint8(2)
                if g1 != np.uint8(1):
                    c1 += 1
                    if g1 == np.uint8(0):
                        a11 += 2
                        a21 += 4
                    elif g1 == np.uint8(2):
                        a11 += 1
                        a21 += 1
            if snp_base + 2 < snp_count:
                r2 = hdr + np.int64(snp_start + snp_base + 2) * np.int64(bytes_per_snp)
                g2 = (bed_raw[r2 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g2] << np.uint8(4)
                if g2 != np.uint8(1):
                    c2 += 1
                    if g2 == np.uint8(0):
                        a12 += 2
                        a22 += 4
                    elif g2 == np.uint8(2):
                        a12 += 1
                        a22 += 1
            if snp_base + 3 < snp_count:
                r3 = hdr + np.int64(snp_start + snp_base + 3) * np.int64(bytes_per_snp)
                g3 = (bed_raw[r3 + s_boff] >> s_bshift) & np.uint8(3)
                val |= lut[g3] << np.uint8(6)
                if g3 != np.uint8(1):
                    c3 += 1
                    if g3 == np.uint8(0):
                        a13 += 2
                        a23 += 4
                    elif g3 == np.uint8(2):
                        a13 += 1
                        a23 += 1
            packed_out[i, pc] = val
        if snp_base < snp_count:
            cnt_out[snp_base] = c0
            s1_out[snp_base] = a10
            s2_out[snp_base] = a20
        if snp_base + 1 < snp_count:
            cnt_out[snp_base + 1] = c1
            s1_out[snp_base + 1] = a11
            s2_out[snp_base + 1] = a21
        if snp_base + 2 < snp_count:
            cnt_out[snp_base + 2] = c2
            s1_out[snp_base + 2] = a12
            s2_out[snp_base + 2] = a22
        if snp_base + 3 < snp_count:
            cnt_out[snp_base + 3] = c3
            s1_out[snp_base + 3] = a13
            s2_out[snp_base + 3] = a23


@njit(cache=True, nogil=True, parallel=True)
def _scatter_pack_raw_bed_to_cache_numba(
    bed_raw,
    snp_start,
    snp_count,
    bytes_per_snp,
    sample_byte_offsets,
    sample_bit_shifts,
    packed_out_flat,
    call_offsets,
    packed_widths,
    byte_indices,
    bit_shifts,
) -> None:
    """Scatter source-order raw BED variants directly into cache-order packed storage."""
    n_keep = sample_byte_offsets.shape[0]
    lut = np.array([np.uint8(2), np.uint8(3), np.uint8(1), np.uint8(0)])
    hdr = np.int64(3)
    for i in prange(n_keep):
        s_boff = np.int64(sample_byte_offsets[i])
        s_bshift = np.uint8(sample_bit_shifts[i])
        for j in range(snp_count):
            if packed_widths[j] <= 0:
                continue
            row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
            g = (bed_raw[row_base + s_boff] >> s_bshift) & np.uint8(3)
            out_idx = (
                np.int64(call_offsets[j])
                + np.int64(i) * np.int64(packed_widths[j])
                + np.int64(byte_indices[j])
            )
            packed_out_flat[out_idx] |= np.uint8(lut[g] << bit_shifts[j])


@njit(cache=True, nogil=True, parallel=True)
def _pack_block_numba(
    block_i8: np.ndarray,
    packed_out: np.ndarray,
    lut_u8: np.ndarray,
) -> None:
    n_rows = block_i8.shape[0]
    n_cols = block_i8.shape[1]
    packed_cols = packed_out.shape[1]
    for i in prange(n_rows):
        for j in range(packed_cols):
            col0 = j << 2
            val = lut_u8[np.uint8(block_i8[i, col0])]
            if col0 + 1 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_i8[i, col0 + 1])] << np.uint8(2))
            if col0 + 2 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_i8[i, col0 + 2])] << np.uint8(4))
            if col0 + 3 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_i8[i, col0 + 3])] << np.uint8(6))
            packed_out[i, j] = val


@njit(cache=True, nogil=True, parallel=True)
def _pack_block_varmaj_numba(
    block_vm: np.ndarray,
    packed_out: np.ndarray,
    lut_u8: np.ndarray,
) -> None:
    n_cols = block_vm.shape[0]
    n_rows = block_vm.shape[1]
    packed_cols = packed_out.shape[1]
    for i in prange(n_rows):
        for j in range(packed_cols):
            col0 = j << 2
            val = lut_u8[np.uint8(block_vm[col0, i])]
            if col0 + 1 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 1, i])] << np.uint8(2))
            if col0 + 2 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 2, i])] << np.uint8(4))
            if col0 + 3 < n_cols:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 3, i])] << np.uint8(6))
            packed_out[i, j] = val


@njit(cache=True, nogil=True, parallel=True)
def _stats_and_pack_varmaj_numba(
    block_vm: np.ndarray,
    packed_out: np.ndarray,
    lut_u8: np.ndarray,
    missing_val: int,
):
    """Fused stats+pack pass for variant-major int8 blocks."""
    n_snps = block_vm.shape[0]
    n_samples = block_vm.shape[1]
    packed_cols = packed_out.shape[1]

    cnt = np.zeros(n_snps, dtype=np.int64)
    s1 = np.zeros(n_snps, dtype=np.int64)
    s2 = np.zeros(n_snps, dtype=np.int64)

    for j in prange(n_snps):
        c = np.int64(0)
        a1 = np.int64(0)
        a2 = np.int64(0)
        for i in range(n_samples):
            v = int(block_vm[j, i])
            if v != missing_val:
                c += 1
                if v == 1:
                    a1 += 1
                    a2 += 1
                elif v == 2:
                    a1 += 2
                    a2 += 4
                elif v != 0:
                    a1 += v
                    a2 += v * v
        cnt[j] = c
        s1[j] = a1
        s2[j] = a2

    for i in prange(n_samples):
        for pc in range(packed_cols):
            col0 = pc << 2
            val = lut_u8[np.uint8(block_vm[col0, i])]
            if col0 + 1 < n_snps:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 1, i])] << np.uint8(2))
            if col0 + 2 < n_snps:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 2, i])] << np.uint8(4))
            if col0 + 3 < n_snps:
                val |= np.uint8(lut_u8[np.uint8(block_vm[col0 + 3, i])] << np.uint8(6))
            packed_out[i, pc] = val

    return cnt, s1, s2


@njit(cache=True, nogil=True, parallel=True)
def _stats_from_varmaj_numba(
    block_vm: np.ndarray,
    missing_val: int,
):
    """Stats-only pass for variant-major int8 blocks."""
    n_snps = block_vm.shape[0]
    n_samples = block_vm.shape[1]

    cnt = np.zeros(n_snps, dtype=np.int64)
    s1 = np.zeros(n_snps, dtype=np.int64)
    s2 = np.zeros(n_snps, dtype=np.int64)

    for j in prange(n_snps):
        c = np.int64(0)
        a1 = np.int64(0)
        a2 = np.int64(0)
        for i in range(n_samples):
            v = int(block_vm[j, i])
            if v != missing_val:
                c += 1
                if v == 1:
                    a1 += 1
                    a2 += 1
                elif v == 2:
                    a1 += 2
                    a2 += 4
                elif v != 0:
                    a1 += v
                    a2 += v * v
        cnt[j] = c
        s1[j] = a1
        s2[j] = a2

    return cnt, s1, s2


@njit(cache=True, nogil=True, parallel=True)
def _scatter_pack_varmaj_to_cache_numba(
    block_vm: np.ndarray,
    packed_out_flat: np.ndarray,
    lut_u8: np.ndarray,
    call_offsets: np.ndarray,
    packed_widths: np.ndarray,
    byte_indices: np.ndarray,
    bit_shifts: np.ndarray,
) -> None:
    """Scatter source-order variant-major block into packed cache-order tmp storage."""
    n_snps = block_vm.shape[0]
    n_samples = block_vm.shape[1]
    for i in prange(n_samples):
        for j in range(n_snps):
            if packed_widths[j] <= 0:
                continue
            out_idx = (
                np.int64(call_offsets[j])
                + np.int64(i) * np.int64(packed_widths[j])
                + np.int64(byte_indices[j])
            )
            packed_out_flat[out_idx] |= np.uint8(
                lut_u8[np.uint8(block_vm[j, i])] << bit_shifts[j]
            )


@njit(cache=True, nogil=True)
def _unpack_selected_u2_columns_numba(
    packed_block: np.ndarray,
    local_cols: np.ndarray,
    out_u8: np.ndarray,
):
    n_rows = packed_block.shape[0]
    n_cols = local_cols.shape[0]
    for j in range(n_cols):
        col = int(local_cols[j])
        byte_idx = col >> 2
        shift = (col & 3) << 1
        for i in range(n_rows):
            out_u8[i, j] = np.uint8((int(packed_block[i, byte_idx]) >> shift) & 0x3)


@contextlib.contextmanager
def _numba_thread_mask(n_threads: int):
    prev = get_num_threads()
    n_use = max(1, int(n_threads))
    if n_use != prev:
        set_num_threads(n_use)
    try:
        yield
    finally:
        if n_use != prev:
            set_num_threads(prev)


_NUMBA_WARMED = False
_AUTO_NUMBA_WARMUP = os.environ.get("GPU_REML_NUMBA_WARMUP", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _warmup_numba() -> None:
    dummy_i8 = np.zeros((2, 4), dtype=np.int8, order="F")
    dummy_vm = np.zeros((4, 2), dtype=np.int8)
    dummy_u8 = np.zeros((2, 1), dtype=np.uint8)
    dummy_lut = np.zeros(256, dtype=np.uint8)
    _pack_block_numba(dummy_i8, dummy_u8, dummy_lut)
    _pack_block_varmaj_numba(dummy_vm, dummy_u8, dummy_lut)
    _compute_stats_numba_forder(dummy_i8[:, :2], -127)
    _compute_stats_numba_varmaj(dummy_vm[:2, :], -127)
    _bed = np.zeros(7, dtype=np.uint8)
    _boff = np.zeros(2, dtype=np.int32)
    _bsh = np.zeros(2, dtype=np.uint8)
    _c = np.zeros(1, dtype=np.int64)
    _s1 = np.zeros(1, dtype=np.int64)
    _s2 = np.zeros(1, dtype=np.int64)
    _stats_from_raw_bed_numba(_bed, 0, 1, 1, _boff, _bsh, _c, _s1, _s2)
    _stats_and_pack_varmaj_numba(dummy_vm[:2, :], dummy_u8, dummy_lut, -127)
    _stats_from_varmaj_numba(dummy_vm[:2, :], -127)
    _po = np.zeros((2, 1), dtype=np.uint8)
    _transcode_raw_bed_numba(_bed, 0, 1, 1, _boff, _bsh, _po)
    _stats_and_transcode_raw_bed_numba(_bed, 0, 1, 1, _boff, _bsh, _c, _s1, _s2, _po)


def _ensure_numba_warmup() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    _warmup_numba()
    _NUMBA_WARMED = True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GenoBlockStreamer:
    """Format-agnostic streaming genotype reader.

    Accepts any ``GenoBlockSource`` (BED or PGEN) and builds a
    2-bit mmap cache that feeds the GPU streaming pipeline.

    Parameters
    ----------
    source : GenoBlockSource, optional
        Pre-constructed genotype source.  Mutually exclusive with
        *bed_prefix*.
    bed_prefix : str, optional
        PLINK1 BED file prefix (convenience shortcut that creates a
        ``BedGenoSource`` internally).
    sample_mask : ndarray of bool, optional
        Boolean mask over the source's full sample set.  When supplied
        only the ``True`` rows are retained in the cache and all
        downstream GPU operations use the subset.
    """

    def __init__(
        self,
        source=None,
        *,
        bed_prefix: str | None = None,
        call_width: int = 131072,
        component_block_sizes = None,
        component_variant_indices = None,
        standardization_override: tuple[np.ndarray, np.ndarray] | None = None,
        device=None,
        keep_host_stats: bool = True,
        build_threads: int | None = None,
        sample_mask: np.ndarray | None = None,
        ring_depth: int | None = None,
        source_build_chunk_width: int | None = None,
    ):
        if _AUTO_NUMBA_WARMUP:
            _ensure_numba_warmup()
        self._ring_depth_cfg = ring_depth
        # ---- Resolve source --------------------------------------------------
        if source is not None and bed_prefix is not None:
            raise ValueError("Provide either source or bed_prefix, not both.")
        if source is None:
            if bed_prefix is None:
                raise ValueError("Either source or bed_prefix must be provided.")
            from .geno_source import BedGenoSource
            source = BedGenoSource(
                bed_prefix, threads=build_threads, sample_mask=sample_mask,
            )
            # sample_mask is now handled inside the source — clear it here
            sample_mask = None

        self._source = source
        self._variant_prefix = getattr(source, "_bed_prefix", None) or getattr(source, "_pgen_prefix", None)
        self._variant_format = (
            "bed" if getattr(source, "_bed_prefix", None) is not None else
            "pgen" if getattr(source, "_pgen_prefix", None) is not None else
            None
        )
        self._sample_mask = (
            np.asarray(sample_mask, dtype=bool) if sample_mask is not None else None
        )

        self.bed_prefix = bed_prefix  # kept for debug messages / compat
        source_n = source.n
        self.source_m = int(source.m)
        self.n = int(np.sum(self._sample_mask)) if self._sample_mask is not None else source_n
        self.m = int(source.m)

        self._bed_int_missing = source.missing_val
        self._build_threads = max(1, int(build_threads or (os.cpu_count() or 1)))
        self._source_build_chunk_width_cfg = (
            max(1, int(source_build_chunk_width))
            if source_build_chunk_width is not None and int(source_build_chunk_width) > 0
            else None
        )

        self._missing_val = np.uint8(3)
        missing_raw_u8 = np.asarray(
            [self._bed_int_missing], dtype=np.int8,
        ).view(np.uint8)[0]
        self._pack_lut_u8 = np.zeros(256, dtype=np.uint8)
        self._pack_lut_u8[1] = np.uint8(1)
        self._pack_lut_u8[2] = np.uint8(2)
        self._pack_lut_u8[int(missing_raw_u8)] = self._missing_val

        # ---- Geometry: call_width → internal block_size=w, blocks_per_call=1
        self.call_width = max(1, int(call_width))
        self.block_size = self.call_width
        self.blocks_per_call = 1
        self.dev = _to_device(device)
        self.keep_host_stats = bool(keep_host_stats)
        self._standardization_override = None
        self._has_component_partition = (
            component_block_sizes is not None or component_variant_indices is not None
        )
        self._component_partition_plan = _build_component_partition_plan(
            self.source_m,
            component_block_sizes=component_block_sizes,
            component_variant_indices=component_variant_indices,
        )
        self._component_block_sizes = self._component_partition_plan.component_sizes
        if self._has_component_partition:
            self.m = int(sum(self._component_block_sizes))
        self._cache_to_source_variant_indices = (
            np.asarray(
                self._component_partition_plan.cache_to_source_variant_indices,
                dtype=np.int64,
            )
            if self._has_component_partition
            else None
        )
        self._has_arbitrary_component_partition = bool(
            self._has_component_partition
            and self._component_partition_plan.has_arbitrary_groups
        )
        if standardization_override is not None:
            means_override, inv_override = standardization_override
            means_override = np.asarray(means_override, dtype=np.float32).reshape(-1)
            inv_override = np.asarray(inv_override, dtype=np.float32).reshape(-1)
            if means_override.size != int(self.m) or inv_override.size != int(self.m):
                raise ValueError(
                    f"standardization_override length mismatch: expected m={int(self.m)}, "
                    f"got means={means_override.size}, inv={inv_override.size}."
                )
            self._standardization_override = (means_override, inv_override)

        (
            self._block_starts,
            self._block_sizes,
            self._call_component_ids,
            self._component_call_offsets,
        ) = _build_call_geometry(self.m, self.call_width, self._component_block_sizes)
        self._n_blocks = int(self._block_starts.shape[0])
        self._n_calls = self._n_blocks  # bpc=1 → one block per call
        self._n_components = len(self._component_block_sizes)

        self._call_block_starts = np.arange(self._n_calls, dtype=np.int32)

        self._call_true_widths = self._block_sizes.copy()
        self._packed_call_widths = ((self._call_true_widths + 3) // 4).astype(np.int32)

        self._call_snp_starts = self._block_starts.astype(np.int32)
        packed_sizes = self.n * self._packed_call_widths.astype(np.int64)
        self._packed_offsets_host = np.zeros(self._n_calls + 1, dtype=np.int64)
        np.cumsum(packed_sizes, out=self._packed_offsets_host[1:])
        self._call_source_segments = (
            _build_call_source_segments(
                self._cache_to_source_variant_indices,
                self._call_snp_starts,
                self._call_true_widths,
            )
            if self._has_component_partition
            else None
        )
        if self._has_arbitrary_component_partition:
            (
                self._source_to_cache_variant_indices,
                self._source_call_offsets,
                self._source_packed_widths,
                self._source_packed_byte_indices,
                self._source_packed_bit_shifts,
            ) = _build_source_scatter_plan(
                self.source_m,
                self._cache_to_source_variant_indices,
                self._call_snp_starts,
                self._call_true_widths,
                self._packed_call_widths,
                self._packed_offsets_host,
            )
        else:
            self._source_to_cache_variant_indices = None
            self._source_call_offsets = None
            self._source_packed_widths = None
            self._source_packed_byte_indices = None
            self._source_packed_bit_shifts = None

        self._max_call_width = int(self._call_true_widths.max()) if self._n_calls > 0 else 1
        self._max_packed_width = int(self._packed_call_widths.max()) if self._n_calls > 0 else 1
        self._max_unpack_width = max(1, 4 * self._max_packed_width)

        # ---- Build (low-memory: packed data lives in tmpfile, not heap)
        self._mode = None
        self._packed_offsets = None
        self._packed_mmap = None
        self._packed_buf = None
        self._tmpfile_fd = None

        from .geno_source import BedGenoSource
        if isinstance(self._source, BedGenoSource):
            bed_path = self._source._bed.location
            means_flat, inv_sds_flat, eff, counts_flat = (
                self._build_tmpfile_bed(
                    bed_path,
                    self._source._n_full,
                    self._source._sample_idx,
                )
            )
        elif hasattr(self._source, "read_block_variant_major"):
            means_flat, inv_sds_flat, eff, counts_flat = (
                self._build_tmpfile_varmaj()
            )
        else:
            raise NotImplementedError(
                "GenoBlockStreamer now requires either BedGenoSource or a "
                "source implementing read_block_variant_major()."
            )

        self._means_host = means_flat if self.keep_host_stats else None
        self._inv_sds_host = inv_sds_flat if self.keep_host_stats else None
        self._count_host = counts_flat if self.keep_host_stats else None

        # ---- Device arrays
        from .kv_impl import build_packed_stats
        means_padded, inv_padded = build_packed_stats(
            jnp.asarray(means_flat, dtype=jnp.float32),
            jnp.asarray(inv_sds_flat, dtype=jnp.float32),
            self._max_unpack_width,
        )
        self._means_padded = jax.device_put(means_padded, self.dev)
        self._inv_padded   = jax.device_put(inv_padded, self.dev)
        means_by_call = np.zeros((self._n_calls, self._max_unpack_width), dtype=np.float32)
        inv_by_call = np.zeros((self._n_calls, self._max_unpack_width), dtype=np.float32)
        for c in range(self._n_calls):
            s0 = int(self._call_snp_starts[c])
            tw = int(self._call_true_widths[c])
            means_by_call[c, :tw] = means_flat[s0 : s0 + tw]
            inv_by_call[c, :tw] = inv_sds_flat[s0 : s0 + tw]
        self._means_by_call = jax.device_put(jnp.asarray(means_by_call), self.dev)
        self._inv_by_call = jax.device_put(jnp.asarray(inv_by_call), self.dev)
        self._eff_m_const  = jax.device_put(jnp.asarray(eff, dtype=jnp.float32), self.dev)
        valid_mask = inv_sds_flat > 0.0
        component_eff = np.zeros((self._n_components,), dtype=np.float32)
        component_snp_offsets = np.zeros((self._n_components + 1,), dtype=np.int32)
        np.cumsum(
            np.asarray(self._component_block_sizes, dtype=np.int32),
            out=component_snp_offsets[1:],
        )
        for comp_idx in range(self._n_components):
            s0 = int(component_snp_offsets[comp_idx])
            s1 = int(component_snp_offsets[comp_idx + 1])
            component_eff[comp_idx] = float(np.count_nonzero(valid_mask[s0:s1]))
        self._component_snp_offsets = component_snp_offsets
        self._component_eff_m_host = component_eff
        self._component_eff_m_const = jax.device_put(
            jnp.asarray(component_eff, dtype=jnp.float32), self.dev,
        )
        self._snp_starts_dev = jax.device_put(
            jnp.asarray(self._call_snp_starts, dtype=jnp.int32), self.dev)
        self._true_widths_dev = jax.device_put(
            jnp.asarray(self._call_true_widths, dtype=jnp.int32), self.dev)
        self._block_backend_plan = self._build_block_backend_plan()
        self._has_sparse_backend = any(
            getattr(desc, "kind", "dense_packed") != "dense_packed"
            for desc in self._block_backend_plan
        )

        if getattr(self, "_source", None) is not None:
            close_source = getattr(self._source, "close", None)
            if callable(close_source):
                try:
                    close_source()
                except (OSError, RuntimeError, ValueError):
                    logger.debug("Failed to close genotype source after streamer build.", exc_info=True)
            self._source = None

    def _init_build_stat_buffers(self):
        if self._standardization_override is not None:
            means_flat = np.asarray(self._standardization_override[0], dtype=np.float32)
            inv_sds_flat = np.asarray(self._standardization_override[1], dtype=np.float32)
            counts_flat = None
            eff = float(np.count_nonzero(inv_sds_flat > 0.0))
            use_precomputed_stats = True
        else:
            means_flat = np.zeros(self.m, dtype=np.float32)
            inv_sds_flat = np.zeros(self.m, dtype=np.float32)
            counts_flat = np.zeros(self.m, dtype=np.int32)
            eff = 0.0
            use_precomputed_stats = False
        return means_flat, inv_sds_flat, counts_flat, eff, use_precomputed_stats

    def _source_build_chunk_width(self) -> int:
        if self._source_build_chunk_width_cfg is not None:
            return max(1, min(int(self.source_m), int(self._source_build_chunk_width_cfg)))
        target_bytes = 256 * 2**20
        width = max(1, target_bytes // max(1, int(self.n)))
        return max(1, min(int(self.call_width), int(self.source_m), int(width)))

    def _store_call_stats(
        self,
        means_flat: np.ndarray,
        inv_sds_flat: np.ndarray,
        counts_flat: np.ndarray | None,
        snp_off: int,
        mean: np.ndarray,
        inv_sd: np.ndarray,
        cnt: np.ndarray | None,
    ) -> float:
        tw = int(mean.shape[0])
        means_flat[snp_off : snp_off + tw] = mean
        inv_sds_flat[snp_off : snp_off + tw] = inv_sd
        if counts_flat is not None:
            if cnt is None:
                counts_flat[snp_off : snp_off + tw] = 0
            else:
                counts_flat[snp_off : snp_off + tw] = cnt.astype(np.int32, copy=False)
        return float(np.count_nonzero(inv_sd > 0.0))

    def _log_build_progress(
        self,
        *,
        call_idx: int,
        snps_done: int,
        t0_wall: float,
        t_last: float,
    ) -> float:
        t_now = time.perf_counter()
        if t_now - t_last < 10.0 and call_idx + 1 != self._n_calls:
            return t_last
        elapsed = t_now - t0_wall
        rate = snps_done * self.n / max(elapsed, 1e-9) / 1e6
        logger.debug(
            "  build [%d/%d] %.0f%% (%d/%d SNPs) %.0f M geno/s  elapsed=%.1fs",
            call_idx + 1,
            self._n_calls,
            100.0 * snps_done / max(self.m, 1),
            snps_done,
            self.m,
            rate,
            elapsed,
        )
        return t_now

    def _finalize_tmpfile_build(
        self,
        *,
        format_name: str,
        t_wall: float,
        write_packed_cache: bool,
        total_packed_bytes: int,
        tmp_fd,
        tmp_path,
        packed_offsets: np.ndarray,
    ):
        cache_gib = total_packed_bytes / (1024**3)
        if write_packed_cache:
            logger.debug(
                "[GenoBlockStreamer] %s tmpfile build: wall=%.1fs n_calls=%d m=%d "
                "call_width=%d packed_MB=%.0f build_threads=%d",
                format_name,
                t_wall,
                self._n_calls,
                self.m,
                self.call_width,
                total_packed_bytes / 2**20,
                self._build_threads,
            )
            self._setup_tmpfile_mmap(tmp_fd, tmp_path, total_packed_bytes, packed_offsets, cache_gib)
            return None, None
        self._packed_mmap = None
        self._packed_buf = None
        self._tmpfile_fd = None
        self._packed_offsets = None
        self._mode = "sparse_no_tmpfile"
        self._ring = None
        logger.debug(
            "[GenoBlockStreamer] %s sparse build: wall=%.1fs n_calls=%d m=%d "
            "call_width=%d packed_cache=skipped build_threads=%d",
            format_name,
            t_wall,
            self._n_calls,
            self.m,
            self.call_width,
            self._build_threads,
        )
        return tmp_fd, tmp_path

    def _build_tmpfile_bed(
        self,
        bed_path: str,
        n_full: int,
        sample_idx: "np.ndarray | None",
    ):
        """Stats + transcode/pack from BED into a tmpfile, then mmap read-only."""
        import tempfile

        if getattr(self, "_has_arbitrary_component_partition", False) and self._should_write_packed_cache():
            return self._build_tmpfile_bed_source_order(
                bed_path,
                n_full,
                sample_idx,
            )

        bytes_per_snp = (n_full + 3) // 4

        bed_fd = os.open(bed_path, os.O_RDONLY)
        bed_mmap = None
        bed_raw = None
        tmp_fd = None
        tmp_path = None
        pack_buf = None
        try:
            bed_size = os.fstat(bed_fd).st_size
            source_m = int(getattr(self, "source_m", self.m))
            expected = 3 + source_m * bytes_per_snp
            if bed_size < expected:
                raise ValueError(f"BED file too small: {bed_size} < {expected}")
            bed_mmap = mmap.mmap(bed_fd, 0, access=mmap.ACCESS_READ)
            from .sliding_window import madvise_sequential
            madvise_sequential(bed_mmap)
            bed_raw = np.frombuffer(bed_mmap, dtype=np.uint8)
            if bed_raw[0] != 0x6C or bed_raw[1] != 0x1B or bed_raw[2] != 0x01:
                raise ValueError("Not a valid SNP-major BED file")

            if sample_idx is not None:
                kept = np.asarray(sample_idx, dtype=np.int32)
            else:
                kept = np.arange(n_full, dtype=np.int32)
            sample_byte_offsets = (kept >> 2).astype(np.int32)
            sample_bit_shifts = ((kept & 3) << 1).astype(np.uint8)

            packed_sizes = self.n * self._packed_call_widths.astype(np.int64)
            packed_offsets = np.zeros(self._n_calls + 1, dtype=np.int64)
            np.cumsum(packed_sizes, out=packed_offsets[1:])
            total_packed_bytes = int(packed_offsets[-1])

            write_packed_cache = self._should_write_packed_cache()
            if write_packed_cache:
                tmp_dir = _resolve_tmpdir()
                tmp_fd, tmp_path = tempfile.mkstemp(
                    prefix="geno_packed_",
                    suffix=".tmp",
                    dir=tmp_dir,
                )

            use_raw_bed_hook = (not write_packed_cache) and self._can_post_build_from_raw_bed()
            max_block_bytes = self.n * self._max_packed_width
            pack_buf = None if use_raw_bed_hook else np.zeros(max_block_bytes, dtype=np.uint8)
            means_flat, inv_sds_flat, counts_flat, eff, use_precomputed_stats = (
                self._init_build_stat_buffers()
            )
            t0_wall = time.perf_counter()
            t_last_progress = t0_wall
            snps_done = 0

            with _numba_thread_mask(self._build_threads):
                for c in range(self._n_calls):
                    tw = int(self._call_true_widths[c])
                    if tw <= 0:
                        continue
                    snp_off = int(self._block_starts[c])
                    pw = int(self._packed_call_widths[c])

                    handled_raw = False
                    if use_raw_bed_hook:
                        built = self._build_bed_raw_block(
                            c,
                            bed_raw,
                            snp_off,
                            tw,
                            bytes_per_snp,
                            sample_byte_offsets,
                            sample_bit_shifts,
                        )
                        if built is not None:
                            mean, inv_sd, eff_inc = built
                            self._store_call_stats(
                                means_flat, inv_sds_flat, counts_flat, snp_off, mean, inv_sd, None,
                            )
                            eff += float(eff_inc)
                            handled_raw = True

                    if not handled_raw:
                        nbytes = self.n * pw
                        packed_view = None
                        if not use_raw_bed_hook:
                            packed_view = pack_buf[:nbytes].reshape(self.n, pw)
                            packed_view[:] = 0

                        if use_precomputed_stats:
                            mean = means_flat[snp_off : snp_off + tw]
                            inv_sd = inv_sds_flat[snp_off : snp_off + tw]
                            if packed_view is not None:
                                _transcode_raw_bed_numba(
                                    bed_raw,
                                    snp_off,
                                    tw,
                                    bytes_per_snp,
                                    sample_byte_offsets,
                                    sample_bit_shifts,
                                    packed_view,
                                )
                        else:
                            cnt = np.zeros(tw, dtype=np.int64)
                            s1 = np.zeros(tw, dtype=np.int64)
                            s2 = np.zeros(tw, dtype=np.int64)
                            if packed_view is not None:
                                _stats_and_transcode_raw_bed_numba(
                                    bed_raw,
                                    snp_off,
                                    tw,
                                    bytes_per_snp,
                                    sample_byte_offsets,
                                    sample_bit_shifts,
                                    cnt,
                                    s1,
                                    s2,
                                    packed_view,
                                )
                            else:
                                _stats_from_raw_bed_numba(
                                    bed_raw,
                                    snp_off,
                                    tw,
                                    bytes_per_snp,
                                    sample_byte_offsets,
                                    sample_bit_shifts,
                                    cnt,
                                    s1,
                                    s2,
                                )
                            cnt_f = cnt.astype(np.float32)
                            s1_f = s1.astype(np.float32)
                            s2_f = s2.astype(np.float32)
                            denom = np.maximum(cnt_f, 1.0)
                            mean = (s1_f / denom).astype(np.float32)
                            var = np.maximum(s2_f / denom - mean * mean, 0.0)
                            valid = (cnt_f > 0.0) & (var > 0.0)
                            inv_sd = np.where(
                                valid, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0,
                            ).astype(np.float32)
                            eff += self._store_call_stats(
                                means_flat, inv_sds_flat, counts_flat, snp_off, mean, inv_sd, cnt,
                            )

                        if packed_view is not None:
                            self._post_build_block(c, packed_view, tw, mean, inv_sd)
                            if write_packed_cache:
                                _write_full(tmp_fd, pack_buf[:nbytes])

                    bed_start = 3 + snp_off * bytes_per_snp
                    page_start = (bed_start // mmap.PAGESIZE) * mmap.PAGESIZE
                    adv_len = (bed_start + tw * bytes_per_snp) - page_start
                    bed_mmap.madvise(mmap.MADV_DONTNEED, page_start, adv_len)

                    snps_done += tw
                    t_last_progress = self._log_build_progress(
                        call_idx=c,
                        snps_done=snps_done,
                        t0_wall=t0_wall,
                        t_last=t_last_progress,
                    )

            t_wall = time.perf_counter() - t0_wall
            tmp_fd, tmp_path = self._finalize_tmpfile_build(
                format_name="BED",
                t_wall=t_wall,
                write_packed_cache=write_packed_cache,
                total_packed_bytes=total_packed_bytes,
                tmp_fd=tmp_fd,
                tmp_path=tmp_path,
                packed_offsets=packed_offsets,
            )
            return means_flat, inv_sds_flat, eff, counts_flat
        finally:
            if bed_raw is not None:
                del bed_raw
            if bed_mmap is not None:
                try:
                    bed_mmap.close()
                except (BufferError, OSError, ValueError):
                    logger.debug("Failed to close BED mmap during cleanup.", exc_info=True)
            try:
                os.close(bed_fd)
            except OSError:
                logger.debug("Failed to close BED file descriptor during cleanup.", exc_info=True)
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    logger.debug("Failed to close temporary packed file descriptor during cleanup.", exc_info=True)
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if pack_buf is not None:
                del pack_buf

    def _build_tmpfile_bed_source_order(
        self,
        bed_path: str,
        n_full: int,
        sample_idx: "np.ndarray | None",
    ):
        """Build packed cache for arbitrary grouping by scanning raw BED in source order."""
        import tempfile

        bytes_per_snp = (n_full + 3) // 4
        packed_offsets = np.asarray(self._packed_offsets_host, dtype=np.int64)
        total_packed_bytes = int(packed_offsets[-1])

        bed_fd = os.open(bed_path, os.O_RDONLY)
        bed_mmap = None
        bed_raw = None
        tmp_fd = None
        tmp_path = None
        wr_mmap = None
        packed_write = None
        try:
            bed_size = os.fstat(bed_fd).st_size
            source_m = int(getattr(self, "source_m", self.m))
            expected = 3 + source_m * bytes_per_snp
            if bed_size < expected:
                raise ValueError(f"BED file too small: {bed_size} < {expected}")
            bed_mmap = mmap.mmap(bed_fd, 0, access=mmap.ACCESS_READ)
            from .sliding_window import madvise_sequential

            madvise_sequential(bed_mmap)
            bed_raw = np.frombuffer(bed_mmap, dtype=np.uint8)
            if bed_raw[0] != 0x6C or bed_raw[1] != 0x1B or bed_raw[2] != 0x01:
                raise ValueError("Not a valid SNP-major BED file")

            if sample_idx is not None:
                kept = np.asarray(sample_idx, dtype=np.int32)
            else:
                kept = np.arange(n_full, dtype=np.int32)
            sample_byte_offsets = (kept >> 2).astype(np.int32)
            sample_bit_shifts = ((kept & 3) << 1).astype(np.uint8)

            tmp_dir = _resolve_tmpdir()
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix="geno_packed_",
                suffix=".tmp",
                dir=tmp_dir,
            )
            if total_packed_bytes > 0:
                os.ftruncate(tmp_fd, total_packed_bytes)
                wr_mmap = mmap.mmap(tmp_fd, total_packed_bytes, access=mmap.ACCESS_WRITE)
                packed_write = np.frombuffer(wr_mmap, dtype=np.uint8)
                packed_write[:] = 0
            else:
                packed_write = np.empty((0,), dtype=np.uint8)

            means_flat, inv_sds_flat, counts_flat, eff, use_precomputed_stats = (
                self._init_build_stat_buffers()
            )
            chunk_width = self._source_build_chunk_width()
            spans = list(
                _iter_source_order_build_spans(
                    self.source_m,
                    self._cache_to_source_variant_indices,
                    chunk_width,
                )
            )
            n_chunks = max(1, len(spans))

            t0_wall = time.perf_counter()
            t_last = t0_wall
            snps_done = 0

            with _numba_thread_mask(self._build_threads):
                for chunk_idx, (source_start, tw) in enumerate(spans):
                    src_slice = slice(int(source_start), int(source_start) + int(tw))
                    cache_idx = np.asarray(
                        self._source_to_cache_variant_indices[src_slice],
                        dtype=np.int64,
                    )
                    selected_mask = cache_idx >= 0
                    cache_idx_selected = cache_idx[selected_mask]
                    if cache_idx_selected.size == 0:
                        continue

                    if use_precomputed_stats:
                        mean = np.asarray(means_flat[cache_idx_selected], dtype=np.float32)
                        inv_sd = np.asarray(inv_sds_flat[cache_idx_selected], dtype=np.float32)
                    else:
                        cnt = np.zeros(tw, dtype=np.int64)
                        s1 = np.zeros(tw, dtype=np.int64)
                        s2 = np.zeros(tw, dtype=np.int64)
                        _stats_from_raw_bed_numba(
                            bed_raw,
                            int(source_start),
                            int(tw),
                            int(bytes_per_snp),
                            sample_byte_offsets,
                            sample_bit_shifts,
                            cnt,
                            s1,
                            s2,
                        )
                        cnt_f = cnt.astype(np.float32)
                        s1_f = s1.astype(np.float32)
                        s2_f = s2.astype(np.float32)
                        denom = np.maximum(cnt_f, 1.0)
                        mean = (s1_f / denom).astype(np.float32)
                        var = np.maximum(s2_f / denom - mean * mean, 0.0)
                        valid = (cnt_f > 0.0) & (var > 0.0)
                        inv_sd = np.where(
                            valid, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0,
                        ).astype(np.float32)
                        means_flat[cache_idx_selected] = mean[selected_mask]
                        inv_sds_flat[cache_idx_selected] = inv_sd[selected_mask]
                        if counts_flat is not None:
                            counts_flat[cache_idx_selected] = cnt[selected_mask].astype(
                                np.int32,
                                copy=False,
                            )
                        eff += float(np.count_nonzero(inv_sd[selected_mask] > 0.0))

                    _scatter_pack_raw_bed_to_cache_numba(
                        bed_raw,
                        int(source_start),
                        int(tw),
                        int(bytes_per_snp),
                        sample_byte_offsets,
                        sample_bit_shifts,
                        packed_write,
                        np.asarray(self._source_call_offsets[src_slice], dtype=np.int64),
                        np.asarray(self._source_packed_widths[src_slice], dtype=np.int32),
                        np.asarray(self._source_packed_byte_indices[src_slice], dtype=np.int32),
                        np.asarray(self._source_packed_bit_shifts[src_slice], dtype=np.uint8),
                    )

                    bed_start = 3 + int(source_start) * bytes_per_snp
                    page_start = (bed_start // mmap.PAGESIZE) * mmap.PAGESIZE
                    adv_len = (bed_start + int(tw) * bytes_per_snp) - page_start
                    bed_mmap.madvise(mmap.MADV_DONTNEED, page_start, adv_len)

                    snps_done += int(tw)
                    t_now = time.perf_counter()
                    if t_now - t_last >= 10.0 or chunk_idx + 1 == n_chunks:
                        elapsed = t_now - t0_wall
                        rate = snps_done * self.n / max(elapsed, 1e-9) / 1e6
                        logger.debug(
                            "  build-src [%d/%d] %.0f%% (%d/%d SNPs) %.0f M geno/s  elapsed=%.1fs",
                            chunk_idx + 1,
                            n_chunks,
                            100.0 * snps_done / max(self.source_m, 1),
                            snps_done,
                            self.source_m,
                            rate,
                            elapsed,
                        )
                        t_last = t_now

            t_wall = time.perf_counter() - t0_wall
            if wr_mmap is not None:
                packed_write = None
                wr_mmap.flush()
                wr_mmap.close()
                wr_mmap = None
            tmp_fd, tmp_path = self._finalize_tmpfile_build(
                format_name="BED-reordered",
                t_wall=t_wall,
                write_packed_cache=True,
                total_packed_bytes=total_packed_bytes,
                tmp_fd=tmp_fd,
                tmp_path=tmp_path,
                packed_offsets=packed_offsets,
            )
            return means_flat, inv_sds_flat, eff, counts_flat
        finally:
            if packed_write is not None:
                del packed_write
            if bed_raw is not None:
                del bed_raw
            if wr_mmap is not None:
                try:
                    wr_mmap.close()
                except (BufferError, OSError, ValueError):
                    logger.debug("Failed to close packed write mmap during cleanup.", exc_info=True)
            if bed_mmap is not None:
                try:
                    bed_mmap.close()
                except (BufferError, OSError, ValueError):
                    logger.debug("Failed to close BED mmap during cleanup.", exc_info=True)
            try:
                os.close(bed_fd)
            except OSError:
                logger.debug("Failed to close BED file descriptor during cleanup.", exc_info=True)
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    logger.debug("Failed to close temporary packed file descriptor during cleanup.", exc_info=True)
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _build_tmpfile_varmaj(self):
        """Build packed data into a tmpfile, then mmap it read-only."""
        import tempfile

        if self._has_arbitrary_component_partition and self._should_write_packed_cache():
            return self._build_tmpfile_varmaj_source_order()

        packed_offsets = np.asarray(self._packed_offsets_host, dtype=np.int64)
        total_packed_bytes = int(packed_offsets[-1])

        write_packed_cache = self._should_write_packed_cache()
        tmp_fd = None
        tmp_path = None
        if write_packed_cache:
            tmp_dir = _resolve_tmpdir()
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix="geno_packed_",
                suffix=".tmp",
                dir=tmp_dir,
            )

        pack_buf = None
        try:
            max_block_bytes = self.n * self._max_packed_width
            pack_buf = np.zeros(max_block_bytes, dtype=np.uint8)
            means_flat, inv_sds_flat, counts_flat, eff, use_precomputed_stats = (
                self._init_build_stat_buffers()
            )
            miss_val = int(self._bed_int_missing)

            t0_wall = time.perf_counter()
            t_last = t0_wall
            snps_done = 0

            with _numba_thread_mask(self._build_threads):
                for c in range(self._n_calls):
                    tw = int(self._call_true_widths[c])
                    if tw <= 0:
                        continue
                    snp_off = int(self._block_starts[c])
                    pw = int(self._packed_call_widths[c])

                    block_vm = self._read_call_block_variant_major(c)
                    packed_view = None
                    if use_precomputed_stats:
                        mean = means_flat[snp_off : snp_off + tw]
                        inv_sd = inv_sds_flat[snp_off : snp_off + tw]
                    else:
                        if write_packed_cache:
                            nbytes = self.n * pw
                            packed_view = pack_buf[:nbytes].reshape(self.n, pw)
                            packed_view[:] = 0
                            cnt, s1_arr, s2_arr = _stats_and_pack_varmaj_numba(
                                block_vm[:tw, :], packed_view, self._pack_lut_u8, miss_val,
                            )
                        else:
                            cnt, s1_arr, s2_arr = _stats_from_varmaj_numba(
                                block_vm[:tw, :], miss_val,
                            )

                        cnt_f = cnt.astype(np.float32)
                        s1_f = s1_arr.astype(np.float32)
                        s2_f = s2_arr.astype(np.float32)
                        denom = np.maximum(cnt_f, 1.0)
                        mean = (s1_f / denom).astype(np.float32)
                        var = np.maximum(s2_f / denom - mean * mean, 0.0)
                        valid = (cnt_f > 0.0) & (var > 0.0)
                        inv_sd = np.where(
                            valid, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0,
                        ).astype(np.float32)
                        eff += self._store_call_stats(
                            means_flat, inv_sds_flat, counts_flat, snp_off, mean, inv_sd, cnt,
                        )
                    handled = self._build_varmaj_block(
                        c,
                        block_vm[:tw, :],
                        snp_off,
                        tw,
                        mean,
                        inv_sd,
                        miss_val,
                    )
                    if not handled:
                        nbytes = self.n * pw
                        packed_view = pack_buf[:nbytes].reshape(self.n, pw)
                        packed_view[:] = 0
                        _pack_block_varmaj_numba(
                            block_vm[:tw, :], packed_view, self._pack_lut_u8,
                        )
                        self._post_build_block(c, packed_view, tw, mean, inv_sd)
                    if write_packed_cache:
                        _write_full(tmp_fd, pack_buf[:nbytes])

                    snps_done += tw
                    t_last = self._log_build_progress(
                        call_idx=c,
                        snps_done=snps_done,
                        t0_wall=t0_wall,
                        t_last=t_last,
                    )

            t_wall = time.perf_counter() - t0_wall
            tmp_fd, tmp_path = self._finalize_tmpfile_build(
                format_name="PGEN",
                t_wall=t_wall,
                write_packed_cache=write_packed_cache,
                total_packed_bytes=total_packed_bytes,
                tmp_fd=tmp_fd,
                tmp_path=tmp_path,
                packed_offsets=packed_offsets,
            )
            return means_flat, inv_sds_flat, eff, counts_flat
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if pack_buf is not None:
                del pack_buf

    def _build_tmpfile_varmaj_source_order(self):
        """Build packed cache for arbitrary grouping by scanning source variants in source order."""
        import tempfile

        packed_offsets = np.asarray(self._packed_offsets_host, dtype=np.int64)
        total_packed_bytes = int(packed_offsets[-1])
        if not self._should_write_packed_cache():
            return self._build_tmpfile_varmaj()

        tmp_dir = _resolve_tmpdir()
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="geno_packed_",
            suffix=".tmp",
            dir=tmp_dir,
        )
        wr_mmap = None
        packed_write = None
        try:
            if total_packed_bytes > 0:
                os.ftruncate(tmp_fd, total_packed_bytes)
                wr_mmap = mmap.mmap(tmp_fd, total_packed_bytes, access=mmap.ACCESS_WRITE)
                packed_write = np.frombuffer(wr_mmap, dtype=np.uint8)
                packed_write[:] = 0
            else:
                packed_write = np.empty((0,), dtype=np.uint8)

            means_flat, inv_sds_flat, counts_flat, eff, use_precomputed_stats = (
                self._init_build_stat_buffers()
            )
            miss_val = int(self._bed_int_missing)
            chunk_width = self._source_build_chunk_width()
            spans = list(
                _iter_source_order_build_spans(
                    self.source_m,
                    self._cache_to_source_variant_indices,
                    chunk_width,
                )
            )
            n_chunks = max(1, len(spans))

            t0_wall = time.perf_counter()
            t_last = t0_wall
            snps_done = 0

            with _numba_thread_mask(self._build_threads):
                for chunk_idx, (source_start, tw) in enumerate(spans):
                    block_vm = self._read_variant_major_source_span(int(source_start), int(tw))
                    src_slice = slice(int(source_start), int(source_start) + int(tw))
                    cache_idx = np.asarray(
                        self._source_to_cache_variant_indices[src_slice],
                        dtype=np.int64,
                    )
                    selected_mask = cache_idx >= 0
                    cache_idx_selected = cache_idx[selected_mask]
                    if cache_idx_selected.size == 0:
                        continue

                    if use_precomputed_stats:
                        mean = np.asarray(means_flat[cache_idx_selected], dtype=np.float32)
                        inv_sd = np.asarray(inv_sds_flat[cache_idx_selected], dtype=np.float32)
                    else:
                        cnt, s1_arr, s2_arr = _stats_from_varmaj_numba(
                            block_vm[:tw, :], miss_val,
                        )
                        cnt_f = cnt.astype(np.float32)
                        s1_f = s1_arr.astype(np.float32)
                        s2_f = s2_arr.astype(np.float32)
                        denom = np.maximum(cnt_f, 1.0)
                        mean = (s1_f / denom).astype(np.float32)
                        var = np.maximum(s2_f / denom - mean * mean, 0.0)
                        valid = (cnt_f > 0.0) & (var > 0.0)
                        inv_sd = np.where(
                            valid, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0,
                        ).astype(np.float32)
                        means_flat[cache_idx_selected] = mean[selected_mask]
                        inv_sds_flat[cache_idx_selected] = inv_sd[selected_mask]
                        if counts_flat is not None:
                            counts_flat[cache_idx_selected] = cnt[selected_mask].astype(
                                np.int32,
                                copy=False,
                            )
                        eff += float(np.count_nonzero(inv_sd[selected_mask] > 0.0))

                    _scatter_pack_varmaj_to_cache_numba(
                        np.asarray(block_vm[:tw, :], dtype=np.int8),
                        packed_write,
                        self._pack_lut_u8,
                        np.asarray(self._source_call_offsets[src_slice], dtype=np.int64),
                        np.asarray(self._source_packed_widths[src_slice], dtype=np.int32),
                        np.asarray(self._source_packed_byte_indices[src_slice], dtype=np.int32),
                        np.asarray(self._source_packed_bit_shifts[src_slice], dtype=np.uint8),
                    )

                    snps_done += int(tw)
                    t_now = time.perf_counter()
                    if t_now - t_last >= 10.0 or chunk_idx + 1 == n_chunks:
                        elapsed = t_now - t0_wall
                        rate = snps_done * self.n / max(elapsed, 1e-9) / 1e6
                        logger.debug(
                            "  build-src [%d/%d] %.0f%% (%d/%d SNPs) %.0f M geno/s  elapsed=%.1fs",
                            chunk_idx + 1,
                            n_chunks,
                            100.0 * snps_done / max(self.source_m, 1),
                            snps_done,
                            self.source_m,
                            rate,
                            elapsed,
                        )
                        t_last = t_now

            t_wall = time.perf_counter() - t0_wall
            if wr_mmap is not None:
                packed_write = None
                wr_mmap.flush()
                wr_mmap.close()
                wr_mmap = None
            tmp_fd, tmp_path = self._finalize_tmpfile_build(
                format_name="VARMAJ-reordered",
                t_wall=t_wall,
                write_packed_cache=True,
                total_packed_bytes=total_packed_bytes,
                tmp_fd=tmp_fd,
                tmp_path=tmp_path,
                packed_offsets=packed_offsets,
            )
            return means_flat, inv_sds_flat, eff, counts_flat
        finally:
            if packed_write is not None:
                del packed_write
            if wr_mmap is not None:
                try:
                    wr_mmap.close()
                except (BufferError, OSError, ValueError):
                    logger.debug("Failed to close packed write mmap during cleanup.", exc_info=True)
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _setup_tmpfile_mmap(self, tmp_fd, tmp_path, total_packed_bytes, packed_offsets, cache_gib):
        rd_mmap = None
        try:
            if total_packed_bytes > 0:
                rd_mmap = mmap.mmap(tmp_fd, total_packed_bytes, access=mmap.ACCESS_READ)
                from .sliding_window import madvise_sequential
                madvise_sequential(rd_mmap)
                self._packed_mmap = rd_mmap
                self._packed_buf = np.frombuffer(rd_mmap, dtype=np.uint8)
            else:
                self._packed_mmap = None
                self._packed_buf = np.empty(0, dtype=np.uint8)
        except (BufferError, OSError, ValueError):
            if rd_mmap is not None:
                try:
                    rd_mmap.close()
                except (BufferError, OSError, ValueError):
                    logger.debug("Failed to close read mmap after mmap setup failure.", exc_info=True)
            self._packed_mmap = None
            self._packed_buf = None
            raise
        os.unlink(tmp_path)
        self._tmpfile_fd = tmp_fd
        self._packed_offsets = packed_offsets
        self._mode = "tmpfile"
        logger.debug("[GenoBlockStreamer] tmpfile mmap: %.2f GB disk", cache_gib)

        # Create evict ring for bounded RSS
        self._ring = None
        if self._packed_mmap is not None and self._n_calls > 0:
            self._ring = _EvictRing(
                self._packed_mmap, self._packed_buf, packed_offsets,
                self._n_calls, self.n, self._max_packed_width,
                depth=self._ring_depth_cfg,
                dev=self.dev,
            )
            ring_mb = self._ring.DEPTH * self.n * self._max_packed_width / 2**20
            logger.debug("[GenoBlockStreamer] evict ring: depth=%d buf=%.0f MB pinned=%d/%d",
                         self._ring.DEPTH, ring_mb, self._ring._pinned, self._ring.DEPTH)

    def _build_block_backend_plan(self):
        return tuple(
            DensePackedBlockDescriptor(
                call_idx=c,
                snp_start=int(self._call_snp_starts[c]),
                true_width=int(self._call_true_widths[c]),
                packed_width=int(self._packed_call_widths[c]),
            )
            for c in range(self._n_calls)
        )

    def _post_build_block(
        self,
        call_idx: int,
        packed_view: np.ndarray,
        true_width: int,
        mean: np.ndarray,
        inv_sd: np.ndarray,
    ) -> None:
        return None

    def _can_post_build_from_raw_bed(self) -> bool:
        return False

    def _build_bed_raw_block(
        self,
        call_idx: int,
        bed_raw: np.ndarray,
        snp_off: int,
        true_width: int,
        bytes_per_snp: int,
        sample_byte_offsets: np.ndarray,
        sample_bit_shifts: np.ndarray,
    ):
        return None

    def _build_varmaj_block(
        self,
        call_idx: int,
        block_vm: np.ndarray,
        snp_off: int,
        true_width: int,
        mean: np.ndarray,
        inv_sd: np.ndarray,
        missing_val: int,
    ):
        return False

    def _should_write_packed_cache(self) -> bool:
        return True

    def _prepare_standardized_column_requests(
        self,
        snp_indices: np.ndarray,
        *,
        closed_message: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._mode is None:
            raise RuntimeError(closed_message)
        if self._means_host is None or self._inv_sds_host is None:
            raise RuntimeError("Host-side SNP statistics were released.")
        idx = np.asarray(snp_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("snp_indices must be 1D.")
        if idx.size == 0:
            return idx, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        if np.any((idx < 0) | (idx >= self.m)):
            raise IndexError("snp_indices out of range.")
        uniq_idx, inverse = np.unique(idx, return_inverse=True)
        call_ids = np.searchsorted(self._call_snp_starts, uniq_idx, side="right") - 1
        call_ids = np.clip(call_ids, 0, self._n_calls - 1)
        starts = np.r_[np.nonzero(np.diff(call_ids))[0] + 1, uniq_idx.size]
        return uniq_idx, inverse, call_ids, starts

    def _extract_standardized_columns_via_reader(
        self,
        *,
        uniq_idx: np.ndarray,
        inverse: np.ndarray,
        call_ids: np.ndarray,
        starts: np.ndarray,
        missing_val: int,
        read_block_columns,
    ) -> np.ndarray:
        if uniq_idx.size == 0:
            return np.empty((self.n, 0), dtype=np.float32)
        out_uniq = np.empty((self.n, uniq_idx.size), dtype=np.float32)
        begin = 0
        for end in starts:
            c = int(call_ids[begin])
            cols = uniq_idx[begin:end]
            local = cols - int(self._call_snp_starts[c])
            width = int(self._call_true_widths[c])
            if np.any((local < 0) | (local >= width)):
                raise RuntimeError("Internal call-index mapping failed.")
            block = np.asarray(
                read_block_columns(c, np.asarray(local, dtype=np.int64), width),
                dtype=np.float32,
                order="C",
            )
            mean = self._means_host[cols][None, :]
            inv = self._inv_sds_host[cols][None, :]
            g_imp = np.where(block == missing_val, mean, block)
            out_uniq[:, begin:end] = (g_imp - mean) * inv
            begin = end
        return out_uniq[:, inverse]

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        for attr in (
            "_means_padded", "_inv_padded", "_eff_m_const",
            "_means_by_call", "_inv_by_call",
            "_snp_starts_dev", "_true_widths_dev",
            "_means_host", "_inv_sds_host", "_count_host",
            "_component_eff_m_const",
            "_cache_to_source_variant_indices",
            "_call_source_segments",
            "_packed_offsets_host",
            "_source_to_cache_variant_indices",
            "_source_call_offsets",
            "_source_packed_widths",
            "_source_packed_byte_indices",
            "_source_packed_bit_shifts",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if getattr(self, "_source", None) is not None:
            close_source = getattr(self._source, "close", None)
            if callable(close_source):
                try:
                    close_source()
                except (OSError, RuntimeError, ValueError):
                    logger.debug("Failed to close genotype source.", exc_info=True)
        self._source = None

        if getattr(self, "_ring", None) is not None:
            try:
                self._ring.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close packed block ring.", exc_info=True)
            self._ring = None

        self._packed_buf = None
        if getattr(self, "_packed_mmap", None) is not None:
            try:
                self._packed_mmap.close()
            except (BufferError, OSError, ValueError):
                logger.debug("Failed to close packed mmap.", exc_info=True)
            self._packed_mmap = None
        if getattr(self, "_tmpfile_fd", None) is not None:
            try:
                os.close(self._tmpfile_fd)
            except OSError:
                logger.debug("Failed to close packed tmpfile descriptor.", exc_info=True)
            self._tmpfile_fd = None
        self._packed_offsets = None
        self._mode = None

    def __del__(self):
        with contextlib.suppress(OSError, RuntimeError, ValueError, BufferError, AttributeError):
            self.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_call_block_variant_major(self, call_idx: int) -> np.ndarray | None:
        if self._source is None:
            return None
        tw = int(self._call_true_widths[call_idx])
        if tw <= 0:
            return np.empty((0, self.n), dtype=np.int8)
        if self._call_source_segments is None:
            j0 = int(self._call_snp_starts[call_idx])
            return self._read_variant_major_source_span(j0, tw)

        starts, widths = self._call_source_segments[int(call_idx)]
        if starts.size == 1 and int(widths[0]) == tw:
            return self._read_variant_major_source_span(int(starts[0]), tw)

        out = np.empty((tw, self.n), dtype=np.int8)
        dst = 0
        for src_start, seg_width in zip(starts.tolist(), widths.tolist()):
            seg_width_i = int(seg_width)
            block = self._read_variant_major_source_span(int(src_start), seg_width_i)
            out[dst : dst + seg_width_i, :] = block
            dst += seg_width_i
        return out

    def _read_variant_major_source_span(self, snp_start: int, snp_count: int) -> np.ndarray:
        if hasattr(self._source, "read_block_variant_major"):
            block = self._source.read_block_variant_major(snp_start, snp_count)
        else:
            block = np.ascontiguousarray(self._source.read_block(snp_start, snp_count).T)
        if self._sample_mask is not None:
            block = np.ascontiguousarray(block[:, self._sample_mask])
        return block

    # ------------------------------------------------------------------
    # Block accessor (staging)
    # ------------------------------------------------------------------

    def _pop_cached(self, call_idx) -> np.ndarray:
        c = int(call_idx)
        pw = int(self._packed_call_widths[c])
        if self._ring is not None:
            return self._ring.get(c).reshape(self.n, pw)
        if self._packed_offsets is None or self._packed_buf is None:
            raise RuntimeError(
                "Packed block cache is unavailable for this streamer; "
                "sparse paths must override direct packed-block access."
            )
        off0 = int(self._packed_offsets[c])
        off1 = int(self._packed_offsets[c + 1])
        return self._packed_buf[off0:off1].reshape(self.n, pw)

    def _packed_block_host(self, call_idx: int) -> np.ndarray:
        c = int(call_idx)
        pw = int(self._packed_call_widths[c])
        if self._packed_offsets is None or self._packed_buf is None:
            raise RuntimeError(
                "Packed block cache is unavailable for this streamer; "
                "sparse paths must override direct packed-block access."
            )
        off0 = int(self._packed_offsets[c])
        off1 = int(self._packed_offsets[c + 1])
        return self._packed_buf[off0:off1].reshape(self.n, pw)

    def _prepare_kv_pass(self) -> None:
        if self._ring is not None:
            self._ring.start_pass()

    def block_backend_plan(self):
        return self._block_backend_plan

    @property
    def has_sparse_backend(self) -> bool:
        return self._has_sparse_backend

    @property
    def has_component_partition(self) -> bool:
        return self._has_component_partition

    @property
    def has_arbitrary_component_partition(self) -> bool:
        return self._has_arbitrary_component_partition

    @property
    def n_components(self) -> int:
        return self._n_components

    def component_source_variant_indices(self, component_idx: int) -> np.ndarray:
        if not self._has_component_partition or self._cache_to_source_variant_indices is None:
            raise ValueError("Component source SNP indices require component-partitioned streamer geometry.")
        if component_idx < 0 or component_idx >= self._n_components:
            raise IndexError(
                f"component_idx={component_idx} out of range for {self._n_components} components."
            )
        s0 = int(self._component_snp_offsets[component_idx])
        s1 = int(self._component_snp_offsets[component_idx + 1])
        return np.asarray(self._cache_to_source_variant_indices[s0:s1], dtype=np.int64)

    # ------------------------------------------------------------------
    # K·V and X^T·V products
    # ------------------------------------------------------------------

    def kv(self, V: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        from .kv_impl import kv_impl_streamed
        return kv_impl_streamed(
            V, self._true_widths_dev,
            self._means_by_call, self._inv_by_call, self._eff_m_const,
            n=self.n, n_calls=self._n_calls,
            pop_block=self._pop_cached,
            missing_val=int(self._missing_val), normalize=normalize,
        )

    def component_kv(
        self,
        V: jnp.ndarray,
        component_idx: int,
        normalize: bool = True,
    ) -> jnp.ndarray:
        if component_idx < 0 or component_idx >= self._n_components:
            raise IndexError(
                f"component_idx={component_idx} out of range for {self._n_components} components."
            )
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        from .kv_impl import kv_impl_partitioned_component
        return kv_impl_partitioned_component(
            V,
            self,
            component_idx=component_idx,
            missing_val=int(self._missing_val),
            normalize=normalize,
        )

    def stacked_component_kv(self, V: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        from .kv_impl import kv_impl_partitioned_stacked
        return kv_impl_partitioned_stacked(
            V,
            self,
            missing_val=int(self._missing_val),
            normalize=normalize,
        )

    def weighted_component_hv(
        self,
        theta_g: jnp.ndarray,
        theta_e: jnp.ndarray | None,
        V: jnp.ndarray,
    ) -> jnp.ndarray:
        if int(theta_g.shape[0]) != self._n_components:
            raise ValueError(
                f"theta_g length mismatch: expected {self._n_components}, got {int(theta_g.shape[0])}."
            )
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        from .kv_impl import kv_impl_partitioned_weighted
        return kv_impl_partitioned_weighted(
            V,
            self,
            theta_g,
            theta_e=theta_e,
            missing_val=int(self._missing_val),
        )

    def xtv(self, V: jnp.ndarray, normalize: bool = False) -> jnp.ndarray:
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        from .kv_impl import xtv_impl_streamed
        return xtv_impl_streamed(
            V, m=self.m,
            snp_starts=self._snp_starts_dev,
            true_widths=self._true_widths_dev,
            means_by_call=self._means_by_call, inv_by_call=self._inv_by_call,
            n_calls=self._n_calls, pop_block=self._pop_cached,
            missing_val=int(self._missing_val),
            normalize=normalize, eff_m=self._eff_m_const,
        )

    def extract_standardized_columns(self, snp_indices: np.ndarray) -> np.ndarray:
        uniq_idx, inverse, call_ids, starts = self._prepare_standardized_column_requests(
            snp_indices,
            closed_message="GenoBlockStreamer is closed.",
        )
        return self._extract_standardized_columns_via_reader(
            uniq_idx=uniq_idx,
            inverse=inverse,
            call_ids=call_ids,
            starts=starts,
            missing_val=int(self._missing_val),
            read_block_columns=self._read_standardized_dense_block_columns,
        )

    def _read_standardized_dense_block_columns(
        self,
        call_idx: int,
        local_cols: np.ndarray,
        width: int,
    ) -> np.ndarray:
        packed_block = self._packed_block_host(call_idx)
        g_u8 = np.empty((self.n, local_cols.size), dtype=np.uint8)
        _unpack_selected_u2_columns_numba(packed_block, local_cols, g_u8)
        return g_u8.astype(np.float32, copy=False)

    def diag(self) -> jnp.ndarray:
        return jax.device_put(jnp.ones((self.n,), dtype=jnp.float32), self.dev)

    def component_diag_list(self) -> list[jnp.ndarray]:
        diag_one = self.diag()
        diag_zero = jnp.zeros_like(diag_one)
        return [
            diag_one if float(eff) > 0.0 else diag_zero
            for eff in self._component_eff_m_host
        ]

    def build_projected_core_atom(
        self,
        U: jnp.ndarray,
        *,
        subtract_identity: bool = True,
    ) -> jnp.ndarray:
        self._prepare_kv_pass()
        U = _ensure_on_device(U, self.dev)
        from .kv_impl import build_projected_core_atom_streamed
        return build_projected_core_atom_streamed(
            U,
            self,
            missing_val=int(self._missing_val),
            subtract_identity=subtract_identity,
        )

    def build_projected_core_atoms(
        self,
        U: jnp.ndarray,
        *,
        subtract_identity: bool = True,
    ) -> jnp.ndarray:
        if not self._has_component_partition:
            raise ValueError("Projected-core atoms require component-partitioned streamer geometry.")
        self._prepare_kv_pass()
        U = _ensure_on_device(U, self.dev)
        from .kv_impl import build_projected_core_atoms_partitioned
        return build_projected_core_atoms_partitioned(
            U,
            self,
            missing_val=int(self._missing_val),
            subtract_identity=subtract_identity,
        )


class BedBlockStreamer(GenoBlockStreamer):
    """Backward-compatible alias: ``BedBlockStreamer(bed_prefix, ...)``."""

    def __init__(
        self,
        bed_prefix: str,
        call_width: int = 131072,
        component_block_sizes = None,
        component_variant_indices = None,
        standardization_override: tuple[np.ndarray, np.ndarray] | None = None,
        device=None,
        keep_host_stats: bool = True,
        build_threads: int | None = None,
        sample_mask: np.ndarray | None = None,
        ring_depth: int | None = None,
        source_build_chunk_width: int | None = None,
    ):
        super().__init__(
            bed_prefix=bed_prefix,
            call_width=call_width,
            component_block_sizes=component_block_sizes,
            component_variant_indices=component_variant_indices,
            standardization_override=standardization_override,
            device=device,
            keep_host_stats=keep_host_stats,
            build_threads=build_threads,
            sample_mask=sample_mask,
            ring_depth=ring_depth,
            source_build_chunk_width=source_build_chunk_width,
        )
