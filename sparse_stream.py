from __future__ import annotations

import logging
import time

import numpy as np
import jax
import jax.numpy as jnp
from numba import njit, prange

from .block_backend import Sparse12BlockDescriptor
from .geno_stream import (
    GenoBlockStreamer,
    _ensure_on_device,
    _stats_and_transcode_raw_bed_numba,
    _stats_from_raw_bed_numba,
)

logger = logging.getLogger(__name__)

_SPARSE_SHARD_REF_BUDGET_GIB = 48.0
_SPARSE_SHARD_REF_CALLS = 5.0
_SPARSE_SHARD_MIN_CALLS = 2
_SPARSE_SHARD_MAX_CALLS = 8
_SPARSE_INDEX_KEYS = (
    "csc_all_rows",
    "csc_all_cols",
)
_SPARSE_EXEC_PAYLOAD_KEYS = {
    "kv": (
        "mean",
        "b",
        "inv_sq",
        "row_b",
        "sum_a0_sq",
        "csc_all_rows",
        "csc_all_cols",
        "csc_all_vals",
    ),
    "xtv": (
        "mean",
        "inv",
        "csc_all_rows",
        "csc_all_cols",
        "csc_all_vals",
    ),
}
_SPARSE_EXEC_PAYLOAD_KEYS["core"] = _SPARSE_EXEC_PAYLOAD_KEYS["xtv"]


def _choose_sparse_shard_cols(
    *,
    call_width: int,
    gpu_budget_bytes: float | None,
    mixed_dense_sparse: bool,
    n_samples: int,
    m_total: int,
) -> int:
    """Choose shard width from planner-derived call_width plus GPU budget context."""
    call_width = max(1, int(call_width))
    m_total = max(1, int(m_total))
    n_samples = max(1, int(n_samples))

    budget_gib = (
        float(gpu_budget_bytes) / (1024**3)
        if gpu_budget_bytes is not None and float(gpu_budget_bytes) > 0.0
        else _SPARSE_SHARD_REF_BUDGET_GIB
    )
    budget_scale = budget_gib / _SPARSE_SHARD_REF_BUDGET_GIB
    sample_scale = 50_000.0 / float(n_samples)
    mixed_scale = 0.75 if mixed_dense_sparse else 1.0
    calls = int(round(_SPARSE_SHARD_REF_CALLS * budget_scale * sample_scale * mixed_scale))
    calls = max(_SPARSE_SHARD_MIN_CALLS, min(_SPARSE_SHARD_MAX_CALLS, calls))
    target = calls * call_width
    return max(call_width, min(m_total, int(target)))


def _assert_sparse_exec_index_dtypes(exec_host: dict[str, object]) -> None:
    for key in _SPARSE_INDEX_KEYS:
        arr = exec_host.get(key)
        if arr is None:
            continue
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{key} must be a numpy array, got {type(arr)!r}")
        if arr.dtype != np.int32:
            raise TypeError(f"{key} must have dtype int32, got {arr.dtype}")


def _sparse_exec_device_payload(
    host: dict[str, object],
    kind: str = "kv",
) -> dict[str, object]:
    try:
        keys = _SPARSE_EXEC_PAYLOAD_KEYS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown sparse exec payload kind: {kind!r}") from exc
    return {k: host[k] for k in keys}


def _sparse_exec_nbytes(host: dict[str, object], kind: str = "kv") -> int:
    total = 0
    for key, value in _sparse_exec_device_payload(host, kind).items():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
        elif np.isscalar(value):
            total += int(np.asarray(value).nbytes)
        else:
            logger.debug("Skipping sparse device-byte estimate for %s=%r", key, type(value))
    return total


@njit(cache=True)
def _csr_row_positions(row_count: np.ndarray) -> tuple[np.ndarray, int]:
    pos = np.zeros(row_count.shape[0], dtype=np.int32)
    running = 0
    for i in range(row_count.shape[0]):
        pos[i] = running
        running += int(row_count[i])
    return pos, running


@njit(cache=True)
def _alloc_sparse_csr_outputs(
    row_count_het: np.ndarray,
    row_count_hom: np.ndarray,
    row_count_miss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    csr_het_pos, nnz_het = _csr_row_positions(row_count_het)
    csr_hom_pos, nnz_hom = _csr_row_positions(row_count_hom)
    csr_miss_pos, nnz_miss = _csr_row_positions(row_count_miss)
    csr_het_rows = np.empty(nnz_het, dtype=np.int32)
    csr_het_cols = np.empty(nnz_het, dtype=np.int32)
    csr_hom_rows = np.empty(nnz_hom, dtype=np.int32)
    csr_hom_cols = np.empty(nnz_hom, dtype=np.int32)
    csr_miss_rows = np.empty(nnz_miss, dtype=np.int32)
    csr_miss_cols = np.empty(nnz_miss, dtype=np.int32)
    return (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        csr_het_pos,
        csr_hom_pos,
        csr_miss_pos,
        nnz_het,
        nnz_hom,
        nnz_miss,
    )

@njit(cache=True, parallel=True)
def _extract_sparse_csr_from_packed_numba(
    packed_block: np.ndarray,
    width: int,
):
    """Extract row-major CSR het/hom coordinates directly from packed 2-bit storage."""
    n, packed_width = packed_block.shape
    row_count_het = np.zeros(n, dtype=np.int32)
    row_count_hom = np.zeros(n, dtype=np.int32)
    row_count_miss = np.zeros(n, dtype=np.int32)
    has_missing_u8 = np.zeros(n, dtype=np.uint8)

    for i in prange(n):
        base_col = 0
        cnt_het = 0
        cnt_hom = 0
        cnt_miss = 0
        has_missing = np.uint8(0)
        for pj in range(packed_width):
            byte = int(packed_block[i, pj])
            for shift in range(4):
                col = base_col + shift
                if col >= width:
                    break
                v = (byte >> (2 * shift)) & 3
                if v == 1:
                    cnt_het += 1
                elif v == 2:
                    cnt_hom += 1
                elif v == 3:
                    cnt_miss += 1
                    has_missing = np.uint8(1)
            base_col += 4
        row_count_het[i] = cnt_het
        row_count_hom[i] = cnt_hom
        row_count_miss[i] = cnt_miss
        has_missing_u8[i] = has_missing

    (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        csr_het_pos,
        csr_hom_pos,
        csr_miss_pos,
        nnz_het,
        nnz_hom,
        nnz_miss,
    ) = _alloc_sparse_csr_outputs(row_count_het, row_count_hom, row_count_miss)

    for i in prange(n):
        base_col = 0
        p_het = csr_het_pos[i]
        p_hom = csr_hom_pos[i]
        p_miss = csr_miss_pos[i]
        for pj in range(packed_width):
            byte = int(packed_block[i, pj])
            for shift in range(4):
                col = base_col + shift
                if col >= width:
                    break
                v = (byte >> (2 * shift)) & 3
                if v == 1:
                    csr_het_rows[p_het] = i
                    csr_het_cols[p_het] = col
                    p_het += 1
                elif v == 2:
                    csr_hom_rows[p_hom] = i
                    csr_hom_cols[p_hom] = col
                    p_hom += 1
                elif v == 3:
                    csr_miss_rows[p_miss] = i
                    csr_miss_cols[p_miss] = col
                    p_miss += 1
            base_col += 4

    return (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        nnz_het,
        nnz_hom,
        nnz_miss,
        has_missing_u8,
    )


@njit(cache=True, parallel=True)
def _extract_sparse_csr_from_raw_bed_numba(
    bed_raw: np.ndarray,
    snp_start: int,
    snp_count: int,
    bytes_per_snp: int,
    sample_byte_offsets: np.ndarray,
    sample_bit_shifts: np.ndarray,
):
    n_keep = sample_byte_offsets.shape[0]
    hdr = np.int64(3)
    row_count_het = np.zeros(n_keep, dtype=np.int32)
    row_count_hom = np.zeros(n_keep, dtype=np.int32)
    row_count_miss = np.zeros(n_keep, dtype=np.int32)
    has_missing_u8 = np.zeros(n_keep, dtype=np.uint8)

    for i in prange(n_keep):
        s_boff = np.int64(sample_byte_offsets[i])
        s_bshift = np.uint8(sample_bit_shifts[i])
        cnt_het = 0
        cnt_hom = 0
        cnt_miss = 0
        has_missing = np.uint8(0)
        for j in range(snp_count):
            row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
            g = (bed_raw[row_base + s_boff] >> s_bshift) & np.uint8(3)
            if g == np.uint8(2):
                cnt_het += 1
            elif g == np.uint8(0):
                cnt_hom += 1
            elif g == np.uint8(1):
                cnt_miss += 1
                has_missing = np.uint8(1)
        row_count_het[i] = cnt_het
        row_count_hom[i] = cnt_hom
        row_count_miss[i] = cnt_miss
        has_missing_u8[i] = has_missing

    (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        csr_het_pos,
        csr_hom_pos,
        csr_miss_pos,
        nnz_het,
        nnz_hom,
        nnz_miss,
    ) = _alloc_sparse_csr_outputs(row_count_het, row_count_hom, row_count_miss)

    for i in prange(n_keep):
        s_boff = np.int64(sample_byte_offsets[i])
        s_bshift = np.uint8(sample_bit_shifts[i])
        p_het = csr_het_pos[i]
        p_hom = csr_hom_pos[i]
        p_miss = csr_miss_pos[i]
        for j in range(snp_count):
            row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
            g = (bed_raw[row_base + s_boff] >> s_bshift) & np.uint8(3)
            if g == np.uint8(2):
                csr_het_rows[p_het] = i
                csr_het_cols[p_het] = j
                p_het += 1
            elif g == np.uint8(0):
                csr_hom_rows[p_hom] = i
                csr_hom_cols[p_hom] = j
                p_hom += 1
            elif g == np.uint8(1):
                csr_miss_rows[p_miss] = i
                csr_miss_cols[p_miss] = j
                p_miss += 1

    return (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        nnz_het,
        nnz_hom,
        nnz_miss,
        has_missing_u8,
    )


@njit(cache=True, parallel=True)
def _extract_sparse_csr_from_varmaj_numba(
    block_vm: np.ndarray,
    missing_val: int,
):
    n_snps = block_vm.shape[0]
    n_keep = block_vm.shape[1]
    row_count_het = np.zeros(n_keep, dtype=np.int32)
    row_count_hom = np.zeros(n_keep, dtype=np.int32)
    row_count_miss = np.zeros(n_keep, dtype=np.int32)
    has_missing_u8 = np.zeros(n_keep, dtype=np.uint8)

    for i in prange(n_keep):
        cnt_het = 0
        cnt_hom = 0
        cnt_miss = 0
        has_missing = np.uint8(0)
        for j in range(n_snps):
            v = int(block_vm[j, i])
            if v == 1:
                cnt_het += 1
            elif v == 2:
                cnt_hom += 1
            elif v == missing_val:
                cnt_miss += 1
                has_missing = np.uint8(1)
        row_count_het[i] = cnt_het
        row_count_hom[i] = cnt_hom
        row_count_miss[i] = cnt_miss
        has_missing_u8[i] = has_missing

    (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        csr_het_pos,
        csr_hom_pos,
        csr_miss_pos,
        nnz_het,
        nnz_hom,
        nnz_miss,
    ) = _alloc_sparse_csr_outputs(row_count_het, row_count_hom, row_count_miss)

    for i in prange(n_keep):
        p_het = csr_het_pos[i]
        p_hom = csr_hom_pos[i]
        p_miss = csr_miss_pos[i]
        for j in range(n_snps):
            v = int(block_vm[j, i])
            if v == 1:
                csr_het_rows[p_het] = i
                csr_het_cols[p_het] = j
                p_het += 1
            elif v == 2:
                csr_hom_rows[p_hom] = i
                csr_hom_cols[p_hom] = j
                p_hom += 1
            elif v == missing_val:
                csr_miss_rows[p_miss] = i
                csr_miss_cols[p_miss] = j
                p_miss += 1

    return (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        nnz_het,
        nnz_hom,
        nnz_miss,
        has_missing_u8,
    )


def _extract_sparse_index_orders(
    extractor,
    width: int,
    *extract_args,
):
    (
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        nnz_het,
        nnz_hom,
        nnz_miss,
        has_missing_u8,
    ) = extractor(*extract_args)
    return _pack_sparse_index_orders(
        csr_het_rows,
        csr_het_cols,
        csr_hom_rows,
        csr_hom_cols,
        csr_miss_rows,
        csr_miss_cols,
        int(width),
        int(nnz_het),
        int(nnz_hom),
        int(nnz_miss),
        has_missing_u8,
    )


@njit(cache=True)
def _stable_csr_to_csc(rows: np.ndarray, cols: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    if rows.size == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    col_count = np.zeros(width, dtype=np.int32)
    for k in range(rows.shape[0]):
        col_count[cols[k]] += 1
    pos = np.empty(width, dtype=np.int32)
    running = 0
    for col in range(width):
        pos[col] = running
        running += int(col_count[col])
    csc_rows = np.empty(rows.shape[0], dtype=np.int32)
    csc_cols = np.empty(rows.shape[0], dtype=np.int32)
    next_pos = pos.copy()
    for k in range(rows.shape[0]):
        col = int(cols[k])
        p = next_pos[col]
        csc_rows[p] = rows[k]
        csc_cols[p] = col
        next_pos[col] = p + 1
    return csc_rows, csc_cols


@njit(cache=True)
def _col_positions(col_count: np.ndarray) -> tuple[np.ndarray, int]:
    pos = np.zeros(col_count.shape[0], dtype=np.int32)
    running = 0
    for j in range(col_count.shape[0]):
        pos[j] = running
        running += int(col_count[j])
    return pos, running


@njit(cache=True, nogil=True, parallel=True)
def _raw_bed_sparse_counts_stats_numba(
    bed_raw: np.ndarray,
    snp_start: int,
    snp_count: int,
    bytes_per_snp: int,
    sample_byte_offsets: np.ndarray,
    sample_bit_shifts: np.ndarray,
    cnt_out: np.ndarray,
    s1_out: np.ndarray,
    s2_out: np.ndarray,
    count_het: np.ndarray,
    count_hom: np.ndarray,
    count_miss: np.ndarray,
    count_all: np.ndarray,
) -> None:
    n_keep = sample_byte_offsets.shape[0]
    hdr = np.int64(3)
    for j in prange(snp_count):
        row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
        c = np.int64(0)
        a1 = np.int64(0)
        a2 = np.int64(0)
        nh = 0
        nd = 0
        nm = 0
        for i in range(n_keep):
            raw = bed_raw[row_base + np.int64(sample_byte_offsets[i])]
            g = (raw >> np.uint8(sample_bit_shifts[i])) & np.uint8(3)
            if g == np.uint8(2):
                c += 1
                a1 += 1
                a2 += 1
                nh += 1
            elif g == np.uint8(0):
                c += 1
                a1 += 2
                a2 += 4
                nd += 1
            elif g == np.uint8(1):
                nm += 1
            else:
                c += 1
        cnt_out[j] = c
        s1_out[j] = a1
        s2_out[j] = a2
        count_het[j] = nh
        count_hom[j] = nd
        count_miss[j] = nm
        count_all[j] = nh + nd + nm


@njit(cache=True, nogil=True, parallel=True)
def _raw_bed_fill_csc_all_numba(
    bed_raw: np.ndarray,
    snp_start: int,
    snp_count: int,
    bytes_per_snp: int,
    sample_byte_offsets: np.ndarray,
    sample_bit_shifts: np.ndarray,
    pos_all: np.ndarray,
    mean: np.ndarray,
    csc_all_rows: np.ndarray,
    csc_all_cols: np.ndarray,
    csc_all_vals: np.ndarray,
) -> None:
    n_keep = sample_byte_offsets.shape[0]
    hdr = np.int64(3)
    for j in prange(snp_count):
        row_base = hdr + np.int64(snp_start + j) * np.int64(bytes_per_snp)
        p_all = int(pos_all[j])
        for i in range(n_keep):
            raw = bed_raw[row_base + np.int64(sample_byte_offsets[i])]
            g = (raw >> np.uint8(sample_bit_shifts[i])) & np.uint8(3)
            if g == np.uint8(2):
                csc_all_rows[p_all] = i
                csc_all_cols[p_all] = j
                csc_all_vals[p_all] = np.int8(1)
                p_all += 1
            elif g == np.uint8(0):
                csc_all_rows[p_all] = i
                csc_all_cols[p_all] = j
                csc_all_vals[p_all] = np.int8(2)
                p_all += 1
            elif g == np.uint8(1):
                csc_all_rows[p_all] = i
                csc_all_cols[p_all] = j
                csc_all_vals[p_all] = np.int8(3)
                p_all += 1


@njit(cache=True)
def _merge_three_sorted_csc_with_vals(
    het_rows: np.ndarray,
    het_cols: np.ndarray,
    hom_rows: np.ndarray,
    hom_cols: np.ndarray,
    miss_rows: np.ndarray,
    miss_cols: np.ndarray,
    mean: np.ndarray,
):
    nh = het_rows.shape[0]
    nd = hom_rows.shape[0]
    nm = miss_rows.shape[0]
    total = nh + nd + nm
    out_rows = np.empty(total, dtype=np.int32)
    out_cols = np.empty(total, dtype=np.int32)
    out_vals = np.empty(total, dtype=np.float32)
    i = j = k = p = 0
    inf = np.int32(2**31 - 1)
    while p < total:
        ch = het_cols[i] if i < nh else inf
        cd = hom_cols[j] if j < nd else inf
        cm = miss_cols[k] if k < nm else inf
        if ch <= cd and ch <= cm:
            out_rows[p] = het_rows[i]
            out_cols[p] = ch
            out_vals[p] = np.float32(1.0)
            i += 1
        elif cd <= ch and cd <= cm:
            out_rows[p] = hom_rows[j]
            out_cols[p] = cd
            out_vals[p] = np.float32(2.0)
            j += 1
        else:
            out_rows[p] = miss_rows[k]
            out_cols[p] = cm
            out_vals[p] = np.float32(mean[cm])
            k += 1
        p += 1
    return out_rows, out_cols, out_vals


def _extract_sparse_index_orders_from_packed(packed_block: np.ndarray, width: int):
    return _extract_sparse_index_orders(
        _extract_sparse_csr_from_packed_numba,
        int(width),
        packed_block,
        int(width),
    )


def _extract_sparse_index_orders_from_varmaj(
    block_vm: np.ndarray,
    missing_val: int,
):
    return _extract_sparse_index_orders(
        _extract_sparse_csr_from_varmaj_numba,
        int(block_vm.shape[0]),
        block_vm,
        int(missing_val),
    )


def _pack_sparse_index_orders(
    csr_het_rows: np.ndarray,
    csr_het_cols: np.ndarray,
    csr_hom_rows: np.ndarray,
    csr_hom_cols: np.ndarray,
    csr_miss_rows: np.ndarray,
    csr_miss_cols: np.ndarray,
    width: int,
    nnz_het: int,
    nnz_hom: int,
    nnz_miss: int,
    has_missing_u8: np.ndarray,
):
    csc_het_rows, csc_het_cols = _stable_csr_to_csc(csr_het_rows, csr_het_cols, width)
    csc_hom_rows, csc_hom_cols = _stable_csr_to_csc(csr_hom_rows, csr_hom_cols, width)
    csc_miss_rows, csc_miss_cols = _stable_csr_to_csc(csr_miss_rows, csr_miss_cols, width)
    return {
        "csr": {
            "het_rows": csr_het_rows,
            "het_cols": csr_het_cols,
            "hom_rows": csr_hom_rows,
            "hom_cols": csr_hom_cols,
            "miss_rows": csr_miss_rows,
            "miss_cols": csr_miss_cols,
        },
        "csc": {
            "het_rows": csc_het_rows,
            "het_cols": csc_het_cols,
            "hom_rows": csc_hom_rows,
            "hom_cols": csc_hom_cols,
            "miss_rows": csc_miss_rows,
            "miss_cols": csc_miss_cols,
        },
        "nnz_het": nnz_het,
        "nnz_hom": nnz_hom,
        "nnz_miss": nnz_miss,
        "has_missing": bool(np.any(has_missing_u8)),
    }


def _build_sparse_meta_from_raw_bed_single_pass(
    bed_raw: np.ndarray,
    snp_start: int,
    snp_count: int,
    bytes_per_snp: int,
    sample_byte_offsets: np.ndarray,
    sample_bit_shifts: np.ndarray,
):
    width = int(snp_count)
    cnt = np.zeros(width, dtype=np.int64)
    s1 = np.zeros(width, dtype=np.int64)
    s2 = np.zeros(width, dtype=np.int64)
    count_het = np.zeros(width, dtype=np.int32)
    count_hom = np.zeros(width, dtype=np.int32)
    count_miss = np.zeros(width, dtype=np.int32)
    count_all = np.zeros(width, dtype=np.int32)
    _raw_bed_sparse_counts_stats_numba(
        bed_raw,
        int(snp_start),
        width,
        int(bytes_per_snp),
        sample_byte_offsets,
        sample_bit_shifts,
        cnt,
        s1,
        s2,
        count_het,
        count_hom,
        count_miss,
        count_all,
    )
    cnt_f = cnt.astype(np.float32)
    s1_f = s1.astype(np.float32)
    s2_f = s2.astype(np.float32)
    denom = np.maximum(cnt_f, 1.0)
    mean = (s1_f / denom).astype(np.float32)
    var = np.maximum(s2_f / denom - mean * mean, 0.0)
    valid = (cnt_f > 0.0) & (var > 0.0)
    inv_sd = np.where(valid, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0).astype(np.float32)
    eff_inc = int(np.count_nonzero(valid))

    nnz_het = int(np.sum(count_het, dtype=np.int64))
    nnz_hom = int(np.sum(count_hom, dtype=np.int64))
    nnz_miss = int(np.sum(count_miss, dtype=np.int64))
    pos_all, nnz_all = _col_positions(count_all)

    empty = np.empty((0,), dtype=np.int32)
    csc_all_rows = np.empty(int(nnz_all), dtype=np.int32)
    csc_all_cols = np.empty(int(nnz_all), dtype=np.int32)
    csc_all_vals = np.empty(int(nnz_all), dtype=np.int8)
    _raw_bed_fill_csc_all_numba(
        bed_raw,
        int(snp_start),
        width,
        int(bytes_per_snp),
        sample_byte_offsets,
        sample_bit_shifts,
        pos_all,
        mean,
        csc_all_rows,
        csc_all_cols,
        csc_all_vals,
    )
    meta = {
        "csr": {
            "het_rows": empty,
            "het_cols": empty,
            "hom_rows": empty,
            "hom_cols": empty,
            "miss_rows": empty,
            "miss_cols": empty,
        },
        "csc": {
            "het_rows": empty,
            "het_cols": empty,
            "hom_rows": empty,
            "hom_cols": empty,
            "miss_rows": empty,
            "miss_cols": empty,
        },
        "csc_all": {
            "rows": csc_all_rows,
            "cols": csc_all_cols,
            "vals": csc_all_vals,
        },
        "nnz_het": int(nnz_het),
        "nnz_hom": int(nnz_hom),
        "nnz_miss": int(nnz_miss),
        "has_missing": bool(int(nnz_miss) > 0),
    }
    return meta, mean, inv_sd, eff_inc


class SparseGenoBlockStreamer(GenoBlockStreamer):
    """Sparse-path streamer with real host-side sparse metadata build.

    Current status:
    - reuses the mature v6 dense tmpfile + mmap build path
    - collects CSC/CSR + het/hom metadata during the first build pass
    - keeps each block in a compact per-block sparse representation
    - runs `kv()` through compact per-block sparse kernels without padded scans
    """

    def __init__(self, *args, **kwargs):
        self._gpu_budget_bytes = kwargs.pop("gpu_budget_bytes", None)
        self._mixed_dense_sparse = bool(kwargs.pop("mixed_dense_sparse", False))
        self._sparse_metadata_ready = False
        self._sparse_block_meta: list[dict[str, object]] = []
        self._sparse_block_summary: list[dict[str, object]] = []
        self._sparse_kv_shard_hosts: list[dict[str, object]] = []
        self._sparse_extract_elapsed = 0.0
        self._sparse_total_nnz_het = 0
        self._sparse_total_nnz_hom = 0
        self._sparse_max_nnz_het = 0
        self._sparse_max_nnz_hom = 0
        self._sparse_extract_t0 = time.perf_counter()
        self._sparse_shard_target_cols = 0
        self._sparse_device_mode = "stream_prefetch"
        self._sparse_exec_total_nbytes = 0
        self._sparse_exec_max_shard_nbytes = 0
        super().__init__(*args, **kwargs)
        if getattr(self, "_ring", None) is not None:
            try:
                self._ring.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close dense packed ring after sparse streamer initialization.", exc_info=True)
            self._ring = None
        self._finalize_sparse_metadata()

    def _should_write_packed_cache(self) -> bool:
        return False

    def _can_post_build_from_raw_bed(self) -> bool:
        return True

    def _build_block_backend_plan(self):
        plan = []
        for c in range(self._n_calls):
            nnz_het = 0
            nnz_hom = 0
            has_csc = False
            has_csr = False
            idx_dtype = "int32"
            if self._sparse_metadata_ready and c < len(self._sparse_block_summary):
                meta = self._sparse_block_summary[c]
                nnz_het = int(meta["nnz_het"])
                nnz_hom = int(meta["nnz_hom"])
                has_csc = True
                has_csr = True
            elif self._sparse_metadata_ready and c < len(self._sparse_block_meta):
                meta = self._sparse_block_meta[c]
                nnz_het = int(meta["nnz_het"])
                nnz_hom = int(meta["nnz_hom"])
                has_csc = True
                has_csr = True
            plan.append(
                Sparse12BlockDescriptor(
                    call_idx=c,
                    snp_start=int(self._call_snp_starts[c]),
                    true_width=int(self._call_true_widths[c]),
                    nnz_het=nnz_het,
                    nnz_hom=nnz_hom,
                    has_csc=has_csc,
                    has_csr=has_csr,
                    idx_dtype=idx_dtype,
                    storage=(
                        "host_sparse_metadata_compact"
                        if self._sparse_metadata_ready
                        else "build_pending"
                    ),
                )
            )
        return tuple(plan)

    def _finalize_sparse_metadata(self) -> None:
        self._build_sparse_kv_shards()
        self._sparse_block_summary = [
            {
                "nnz_het": int(meta["nnz_het"]),
                "nnz_hom": int(meta["nnz_hom"]),
                "nnz_miss": int(meta.get("nnz_miss", 0)),
                "has_missing": bool(meta["has_missing"]),
                "skip_kv": bool(meta.get("skip_kv", False)),
            }
            for meta in self._sparse_block_meta
        ]
        self._sparse_block_meta = []
        self._sparse_extract_elapsed = time.perf_counter() - self._sparse_extract_t0
        self._sparse_metadata_ready = True
        self._block_backend_plan = self._build_block_backend_plan()
        self._has_sparse_backend = True

    def extract_standardized_columns(self, snp_indices: np.ndarray) -> np.ndarray:
        uniq_idx, inverse, call_ids, starts = self._prepare_standardized_column_requests(
            snp_indices,
            closed_message="SparseGenoBlockStreamer is closed.",
        )
        source = getattr(self, "_source", None)
        close_source = False
        if source is None:
            prefix = getattr(self, "_variant_prefix", None)
            fmt = getattr(self, "_variant_format", None)
            if not prefix or not fmt:
                raise RuntimeError(
                    "SparseGenoBlockStreamer cannot reopen genotype source for column extraction."
                )
            from .geno_source import BedGenoSource, PgenGenoSource

            if fmt == "bed":
                source = BedGenoSource(
                    prefix,
                    threads=getattr(self, "_build_threads", None),
                    sample_mask=getattr(self, "_sample_mask", None),
                )
            elif fmt == "pgen":
                source = PgenGenoSource(
                    prefix,
                    sample_mask=getattr(self, "_sample_mask", None),
                )
            else:
                raise RuntimeError(
                    f"Unsupported genotype source format for sparse column extraction: {fmt!r}"
                )
            close_source = True
        try:
            return self._extract_standardized_columns_via_reader(
                uniq_idx=uniq_idx,
                inverse=inverse,
                call_ids=call_ids,
                starts=starts,
                missing_val=int(self._bed_int_missing),
                read_block_columns=lambda c, local, width: self._read_standardized_sparse_block_columns(
                    source,
                    c,
                    local,
                    width,
                ),
            )
        finally:
            if close_source:
                close_sparse_source = getattr(source, "close", None)
                if callable(close_sparse_source):
                    try:
                        close_sparse_source()
                    except (OSError, RuntimeError, ValueError):
                        logger.debug("Failed to close sparse genotype source.", exc_info=True)

    def _read_standardized_sparse_block_columns(
        self,
        source,
        call_idx: int,
        local_cols: np.ndarray,
        width: int,
    ) -> np.ndarray:
        j0 = int(self._call_snp_starts[call_idx])
        if hasattr(source, "read_block_variant_major"):
            block = np.asarray(source.read_block_variant_major(j0, width))
            if block.shape == (width, self.n):
                block = np.ascontiguousarray(block.T)
            elif block.shape != (self.n, width):
                raise RuntimeError(
                    "Unexpected variant-major block shape for sparse column extraction: "
                    f"got {block.shape}, expected {(width, self.n)} or {(self.n, width)}."
                )
        elif hasattr(source, "read_block"):
            block = np.ascontiguousarray(source.read_block(j0, width).T)
        else:
            raise RuntimeError(
                "SparseGenoBlockStreamer requires a source with read_block_variant_major() "
                "or read_block() for column extraction."
            )
        return np.asarray(block[:, local_cols], dtype=np.float32, order="C")

    def sparse_block_metadata(self):
        return self._sparse_block_summary

    def _prepare_kv_pass(self) -> None:
        return None

    def _build_sparse_exec_from_call_range(
        self,
        call_lo: int,
        call_hi: int,
    ) -> dict[str, object]:
        n = int(self.n)
        if call_hi <= call_lo:
            empty_i32 = np.empty((0,), dtype=np.int32)
            empty_f32 = np.empty((0,), dtype=np.float32)
            return {
                "mean": empty_f32,
                "inv": empty_f32,
                "b": empty_f32,
                "inv_sq": empty_f32,
                "row_b": np.zeros((n,), dtype=np.float32),
                "sum_a0_sq": np.asarray(0.0, dtype=np.float32),
                "csc_all_rows": empty_i32,
                "csc_all_cols": empty_i32,
                "csc_all_vals": empty_f32,
                "_host_m": 0,
                "_host_snp_off": 0,
            }

        snp_lo = int(self._call_snp_starts[call_lo])
        snp_hi = int(self._call_snp_starts[call_hi - 1]) + int(self._call_true_widths[call_hi - 1])
        m = max(0, snp_hi - snp_lo)
        mean_global = np.zeros((m,), dtype=np.float32)
        inv_global = np.zeros((m,), dtype=np.float32)
        b_global = np.zeros((m,), dtype=np.float32)
        inv_sq_global = np.zeros((m,), dtype=np.float32)
        row_b = np.zeros((n,), dtype=np.float32)
        sum_a0_sq = np.float32(0.0)
        csc_all_rows_l: list[np.ndarray] = []
        csc_all_cols_l: list[np.ndarray] = []
        csc_all_vals_l: list[np.ndarray] = []

        for call_idx in range(call_lo, call_hi):
            meta = self._sparse_block_meta[call_idx]
            snp_off = int(self._call_snp_starts[call_idx])
            true_width = int(self._call_true_widths[call_idx])
            if true_width <= 0:
                continue
            local_off = snp_off - snp_lo
            a0 = np.asarray(meta["a0"], dtype=np.float32)
            inv = np.asarray(meta["inv"], dtype=np.float32)
            mean = np.asarray(meta["mean"], dtype=np.float32)
            b_blk = np.asarray(a0 * inv, dtype=np.float32)
            inv_sq_blk = np.asarray(inv * inv, dtype=np.float32)
            mean_global[local_off : local_off + true_width] = mean
            inv_global[local_off : local_off + true_width] = inv
            b_global[local_off : local_off + true_width] = b_blk
            inv_sq_global[local_off : local_off + true_width] = inv_sq_blk
            sum_a0_sq = np.float32(sum_a0_sq + np.dot(a0, a0))

            csc_het_rows = np.asarray(meta["csc"]["het_rows"], dtype=np.int32)
            csc_het_cols = np.asarray(meta["csc"]["het_cols"], dtype=np.int32)
            csc_hom_rows = np.asarray(meta["csc"]["hom_rows"], dtype=np.int32)
            csc_hom_cols = np.asarray(meta["csc"]["hom_cols"], dtype=np.int32)
            csc_miss_rows = np.asarray(meta["csc"]["miss_rows"], dtype=np.int32)
            csc_miss_cols = np.asarray(meta["csc"]["miss_cols"], dtype=np.int32)
            csr_het_rows = np.asarray(meta["csr"]["het_rows"], dtype=np.int32)
            csr_het_cols = np.asarray(meta["csr"]["het_cols"], dtype=np.int32)
            csr_hom_rows = np.asarray(meta["csr"]["hom_rows"], dtype=np.int32)
            csr_hom_cols = np.asarray(meta["csr"]["hom_cols"], dtype=np.int32)
            csr_miss_rows = np.asarray(meta["csr"]["miss_rows"], dtype=np.int32)
            csr_miss_cols = np.asarray(meta["csr"]["miss_cols"], dtype=np.int32)
            if bool(meta.get("skip_kv", False)):
                continue

            csc_all = meta.get("csc_all")
            if isinstance(csc_all, dict):
                csc_all_rows = np.asarray(csc_all["rows"], dtype=np.int32)
                csc_all_cols = np.asarray(csc_all["cols"], dtype=np.int32)
                csc_all_vals = np.asarray(csc_all["vals"])
            elif csc_het_rows.size or csc_hom_rows.size or csc_miss_rows.size:
                csc_all_rows, csc_all_cols, csc_all_vals = _merge_three_sorted_csc_with_vals(
                    csc_het_rows,
                    csc_het_cols,
                    csc_hom_rows,
                    csc_hom_cols,
                    csc_miss_rows,
                    csc_miss_cols,
                    mean,
                )
            else:
                csc_all_rows = np.empty((0,), dtype=np.int32)
                csc_all_cols = np.empty((0,), dtype=np.int32)
                csc_all_vals = np.empty((0,), dtype=np.float32)
            if csc_all_rows.size:
                csc_all_rows_l.append(csc_all_rows)
                csc_all_cols_l.append(csc_all_cols + np.int32(local_off))
                csc_all_vals_l.append(np.asarray(csc_all_vals))
            has_csr_entries = bool(csr_het_rows.size or csr_hom_rows.size or csr_miss_rows.size)
            if csr_het_rows.size:
                row_b += np.bincount(
                    csr_het_rows.astype(np.intp, copy=False),
                    weights=b_blk[csr_het_cols.astype(np.intp, copy=False)],
                    minlength=n,
                ).astype(np.float32, copy=False)
            if csr_hom_rows.size:
                row_b += np.bincount(
                    csr_hom_rows.astype(np.intp, copy=False),
                    weights=2.0 * b_blk[csr_hom_cols.astype(np.intp, copy=False)],
                    minlength=n,
                ).astype(np.float32, copy=False)
            if csr_miss_rows.size:
                row_b += np.bincount(
                    csr_miss_rows.astype(np.intp, copy=False),
                    weights=mean[csr_miss_cols.astype(np.intp, copy=False)] * b_blk[csr_miss_cols.astype(np.intp, copy=False)],
                    minlength=n,
                ).astype(np.float32, copy=False)
            if not has_csr_entries and csc_all_rows.size:
                csc_all_vals_f = np.asarray(csc_all_vals, dtype=np.float32)
                if np.asarray(csc_all_vals).dtype == np.int8:
                    miss = np.asarray(csc_all_vals) == np.int8(3)
                    if np.any(miss):
                        csc_all_vals_f = csc_all_vals_f.copy()
                        csc_all_vals_f[miss] = mean[csc_all_cols.astype(np.intp, copy=False)[miss]]
                row_b += np.bincount(
                    csc_all_rows.astype(np.intp, copy=False),
                    weights=(
                        csc_all_vals_f * b_blk[csc_all_cols.astype(np.intp, copy=False)]
                    ),
                    minlength=n,
                ).astype(np.float32, copy=False)

        def _cat(xs: list[np.ndarray]) -> np.ndarray:
            if not xs:
                return np.empty((0,), dtype=np.int32)
            return np.concatenate(xs, axis=0).astype(np.int32, copy=False)

        def _cat_vals(xs: list[np.ndarray]) -> np.ndarray:
            if not xs:
                return np.empty((0,), dtype=np.float32)
            return np.concatenate(xs, axis=0)

        return {
            "mean": mean_global,
            "inv": inv_global,
            "b": b_global,
            "inv_sq": inv_sq_global,
            "row_b": row_b,
            "sum_a0_sq": np.asarray(sum_a0_sq, dtype=np.float32),
            "csc_all_rows": _cat(csc_all_rows_l),
            "csc_all_cols": _cat(csc_all_cols_l),
            "csc_all_vals": _cat_vals(csc_all_vals_l),
            "_host_m": m,
            "_host_snp_off": snp_lo,
        }

    def _build_sparse_kv_shards(self, target_cols: int = 0) -> None:
        if target_cols <= 0:
            target_cols = _choose_sparse_shard_cols(
                call_width=self.call_width,
                gpu_budget_bytes=self._gpu_budget_bytes,
                mixed_dense_sparse=self._mixed_dense_sparse,
                n_samples=self.n,
                m_total=self.m,
            )
        self._sparse_shard_target_cols = int(target_cols)
        ranges: list[tuple[int, int]] = []
        start = 0
        width_acc = 0
        for call_idx in range(int(self._n_calls)):
            true_width = int(self._call_true_widths[call_idx])
            if true_width <= 0:
                continue
            if start < call_idx and width_acc > 0 and width_acc + true_width > target_cols:
                ranges.append((start, call_idx))
                start = call_idx
                width_acc = 0
            width_acc += true_width
        if width_acc > 0 or not ranges:
            ranges.append((start, int(self._n_calls)))

        host_build_t0 = time.perf_counter()
        self._sparse_kv_shard_hosts = [
            self._build_sparse_exec_from_call_range(lo, hi) for lo, hi in ranges
        ]
        host_build_elapsed = time.perf_counter() - host_build_t0
        for host in self._sparse_kv_shard_hosts:
            _assert_sparse_exec_index_dtypes(host)
        shard_nbytes = [_sparse_exec_nbytes(host) for host in self._sparse_kv_shard_hosts]
        total_nbytes = int(sum(shard_nbytes))
        max_shard_nbytes = int(max(shard_nbytes, default=0))
        self._sparse_exec_total_nbytes = total_nbytes
        self._sparse_exec_max_shard_nbytes = max_shard_nbytes
        self._sparse_device_mode = "stream_prefetch"
        logger.info(
            "[SparseStreamer] shard plan: target_cols=%d shard_count=%d mixed=%s budget_gib=%s "
            "device_mode=%s sparse_exec=%.2fGiB max_shard=%.2fGiB host_shard_build=%.1fs",
            self._sparse_shard_target_cols,
            len(self._sparse_kv_shard_hosts),
            self._mixed_dense_sparse,
            (
                f"{float(self._gpu_budget_bytes) / (1024**3):.1f}"
                if self._gpu_budget_bytes is not None and float(self._gpu_budget_bytes) > 0.0
                else "auto"
            ),
            self._sparse_device_mode,
            total_nbytes / float(1024**3),
            max_shard_nbytes / float(1024**3),
            host_build_elapsed,
        )

    def _put_sparse_exec_on_device(
        self,
        host: dict[str, object],
        *,
        kind: str = "kv",
    ) -> dict[str, object]:
        return jax.device_put(_sparse_exec_device_payload(host, kind), self.dev)

    def _iter_sparse_exec_pairs(self, kind: str = "kv"):
        hosts = iter(self._sparse_kv_shard_hosts)
        try:
            cur_host = next(hosts)
        except StopIteration:
            return
        cur_ex = self._put_sparse_exec_on_device(cur_host, kind=kind)
        for next_host in hosts:
            next_ex = self._put_sparse_exec_on_device(next_host, kind=kind)
            yield cur_host, cur_ex
            cur_host, cur_ex = next_host, next_ex
        yield cur_host, cur_ex

    def _empty_sparse_block_meta(self) -> dict[str, object]:
        empty = np.empty((0,), dtype=np.int32)
        return {
            "csr": {
                "het_rows": empty,
                "het_cols": empty,
                "hom_rows": empty,
                "hom_cols": empty,
                "miss_rows": empty,
                "miss_cols": empty,
            },
            "csc": {
                "het_rows": empty,
                "het_cols": empty,
                "hom_rows": empty,
                "hom_cols": empty,
                "miss_rows": empty,
                "miss_cols": empty,
            },
            "csc_all": {
                "rows": empty,
                "cols": empty,
                "vals": np.empty((0,), dtype=np.float32),
            },
            "nnz_het": 0,
            "nnz_hom": 0,
            "nnz_miss": 0,
            "has_missing": False,
            "skip_kv": True,
            "mean": np.empty((0,), dtype=np.float32),
            "inv": np.empty((0,), dtype=np.float32),
            "a0": np.empty((0,), dtype=np.float32),
        }

    def _attach_sparse_block_meta(
        self,
        meta: dict[str, object],
        mean: np.ndarray,
        inv_sd: np.ndarray,
    ) -> None:
        meta["mean"] = np.asarray(mean, dtype=np.float32).copy()
        meta["inv"] = np.asarray(inv_sd, dtype=np.float32).copy()
        meta["a0"] = (-meta["mean"] * meta["inv"]).astype(np.float32, copy=False)
        meta["skip_kv"] = bool(
            meta["nnz_het"] == 0
            and meta["nnz_hom"] == 0
            and not np.any(meta["inv"])
        )
        self._sparse_block_meta.append(meta)

        nh = int(meta["nnz_het"])
        nd = int(meta["nnz_hom"])
        self._sparse_total_nnz_het += nh
        self._sparse_total_nnz_hom += nd
        self._sparse_max_nnz_het = max(self._sparse_max_nnz_het, nh)
        self._sparse_max_nnz_hom = max(self._sparse_max_nnz_hom, nd)

    def _post_build_block(
        self,
        call_idx: int,
        packed_view: np.ndarray,
        true_width: int,
        mean: np.ndarray,
        inv_sd: np.ndarray,
    ) -> None:
        if int(true_width) <= 0:
            self._sparse_block_meta.append(self._empty_sparse_block_meta())
            return

        meta = _extract_sparse_index_orders_from_packed(packed_view, int(true_width))
        self._attach_sparse_block_meta(meta, mean, inv_sd)

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
        if int(true_width) <= 0:
            return False
        meta = _extract_sparse_index_orders_from_varmaj(
            np.asarray(block_vm, dtype=np.int8),
            int(missing_val),
        )
        self._attach_sparse_block_meta(meta, mean, inv_sd)
        return True

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
        if int(true_width) <= 0:
            return None
        meta, mean, inv_sd, eff_inc = _build_sparse_meta_from_raw_bed_single_pass(
            bed_raw,
            int(snp_off),
            int(true_width),
            int(bytes_per_snp),
            sample_byte_offsets,
            sample_bit_shifts,
        )
        self._attach_sparse_block_meta(meta, mean, inv_sd)
        return mean, inv_sd, eff_inc

    def kv(
        self,
        V: jnp.ndarray,
        normalize: bool = True,
        *,
        sum_v: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]

        from .kv_impl import sparse_kv_shard

        fp = V.dtype
        if sum_v is None:
            sum_v = jnp.sum(V, axis=0)
        else:
            sum_v = _ensure_on_device(sum_v, self.dev)
            if sum_v.ndim == 0:
                sum_v = sum_v[None]
        acc = jnp.zeros((self.n, V.shape[1]), dtype=fp)
        for host, ex in self._iter_sparse_exec_pairs("kv"):
            acc = acc + sparse_kv_shard(
                V,
                n=self.n,
                m=int(host["_host_m"]),
                sum_v=sum_v,
                mean=ex["mean"],
                b=ex["b"],
                inv_sq=ex["inv_sq"],
                row_b=ex["row_b"],
                sum_a0_sq=ex["sum_a0_sq"],
                csc_all_rows=ex["csc_all_rows"],
                csc_all_cols=ex["csc_all_cols"],
                csc_all_vals=ex["csc_all_vals"],
            )
            acc.block_until_ready()

        if normalize:
            eff = self._eff_m_const.astype(fp)
            acc = jnp.where(eff > 0, acc / eff, acc)

        return acc[:, 0] if squeeze else acc

    def xtv(self, V: jnp.ndarray, normalize: bool = False) -> jnp.ndarray:
        self._prepare_kv_pass()
        V = _ensure_on_device(V, self.dev)

        from .kv_impl import sparse_xtv_shard

        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        out = jnp.zeros((self.m, V.shape[1]), dtype=V.dtype)
        for host, ex in self._iter_sparse_exec_pairs("xtv"):
            block = sparse_xtv_shard(
                V,
                m=int(host["_host_m"]),
                mean=ex["mean"],
                inv=ex["inv"],
                csc_all_rows=ex["csc_all_rows"],
                csc_all_cols=ex["csc_all_cols"],
                csc_all_vals=ex["csc_all_vals"],
            )
            out = jax.lax.dynamic_update_slice(out, block, (int(host["_host_snp_off"]), 0))
            out.block_until_ready()
        if normalize:
            eff = self._eff_m_const.astype(out.dtype)
            out = jnp.where(eff > 0, out / eff, out)
        return out[:, 0] if squeeze else out

    def build_projected_core_atom(
        self,
        U: jnp.ndarray,
        *,
        subtract_identity: bool = True,
    ) -> jnp.ndarray:
        self._prepare_kv_pass()
        U = _ensure_on_device(U, self.dev)

        from .kv_impl import sparse_projected_core_atom_shard

        if U.ndim != 2:
            raise ValueError("build_projected_core_atom expects U with shape (n, k).")

        fp = U.dtype
        k = int(U.shape[1])
        core = jnp.zeros((k, k), dtype=fp)
        for host, ex in self._iter_sparse_exec_pairs("core"):
            core = core + sparse_projected_core_atom_shard(
                U,
                m=int(host["_host_m"]),
                mean=ex["mean"],
                inv=ex["inv"],
                csc_all_rows=ex["csc_all_rows"],
                csc_all_cols=ex["csc_all_cols"],
                csc_all_vals=ex["csc_all_vals"],
            )
            core.block_until_ready()

        eff = self._eff_m_const.astype(fp)
        core = jax.lax.cond(
            eff > 0,
            lambda m: m / eff,
            lambda m: jnp.zeros_like(m),
            core,
        )
        if subtract_identity:
            eye = jnp.eye(k, dtype=fp)
            core = jax.lax.cond(
                eff > 0,
                lambda m: m - eye,
                lambda m: m,
                core,
            )
        return core

    def close(self) -> None:
        self._sparse_kv_shard_hosts = []
        self._sparse_block_meta = []
        self._sparse_block_summary = []
        super().close()

__all__ = ["SparseGenoBlockStreamer"]
