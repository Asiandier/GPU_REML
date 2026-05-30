from __future__ import annotations

import os
import sys
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

jax.config.update("jax_platform_name", "cpu")

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
KV_IMPL = importlib.import_module(f"{PKG.__name__}.kv_impl")
SPARSE_STREAM = importlib.import_module(f"{PKG.__name__}.sparse_stream")
GENO_STREAM = importlib.import_module(f"{PKG.__name__}.geno_stream")

GenoBlockStreamer = GENO_STREAM.GenoBlockStreamer
from_import = SPARSE_STREAM
SparseGenoBlockStreamer = from_import.SparseGenoBlockStreamer
_extract_sparse_index_orders_from_packed = from_import._extract_sparse_index_orders_from_packed
_extract_sparse_index_orders_from_raw_bed = from_import._extract_sparse_index_orders_from_raw_bed
_extract_sparse_index_orders_from_varmaj = from_import._extract_sparse_index_orders_from_varmaj
_choose_sparse_shard_cols = from_import._choose_sparse_shard_cols


def test_extract_sparse_index_orders_from_packed_preserves_ascending_layout():
    block = np.array(
        [
            [0, 1, 0, 2, 1],
            [1, 0, 2, 0, 0],
            [0, 2, 0, 0, 2],
            [2, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    packed = np.zeros((block.shape[0], (block.shape[1] + 3) // 4), dtype=np.uint8)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            packed[i, j // 4] |= np.uint8(block[i, j] & 3) << np.uint8(2 * (j % 4))

    meta = _extract_sparse_index_orders_from_packed(packed, block.shape[1])

    for rows, cols in (
        (meta["csr"]["het_rows"], meta["csr"]["het_cols"]),
        (meta["csr"]["hom_rows"], meta["csr"]["hom_cols"]),
    ):
        prev_r = -1
        prev_c = -1
        for r, c in zip(rows.tolist(), cols.tolist()):
            assert r >= prev_r
            if r == prev_r:
                assert c > prev_c
            prev_r, prev_c = r, c
    for rows, cols in (
        (meta["csc"]["het_rows"], meta["csc"]["het_cols"]),
        (meta["csc"]["hom_rows"], meta["csc"]["hom_cols"]),
    ):
        prev_r = -1
        prev_c = -1
        for r, c in zip(rows.tolist(), cols.tolist()):
            assert c >= prev_c
            if c == prev_c:
                assert r >= prev_r
            prev_r, prev_c = r, c


def test_choose_sparse_shard_cols_scales_with_budget():
    low = _choose_sparse_shard_cols(
        call_width=200_000,
        gpu_budget_bytes=24 * 1024**3,
        mixed_dense_sparse=False,
        n_samples=50_000,
        m_total=2_000_000,
    )
    high = _choose_sparse_shard_cols(
        call_width=200_000,
        gpu_budget_bytes=48 * 1024**3,
        mixed_dense_sparse=False,
        n_samples=50_000,
        m_total=2_000_000,
    )
    assert low < high
    assert low >= 200_000
    assert high <= 2_000_000


def test_choose_sparse_shard_cols_tightens_for_mixed_dense_sparse():
    sparse_only = _choose_sparse_shard_cols(
        call_width=200_000,
        gpu_budget_bytes=48 * 1024**3,
        mixed_dense_sparse=False,
        n_samples=50_000,
        m_total=2_000_000,
    )
    mixed = _choose_sparse_shard_cols(
        call_width=200_000,
        gpu_budget_bytes=48 * 1024**3,
        mixed_dense_sparse=True,
        n_samples=50_000,
        m_total=2_000_000,
    )
    assert mixed < sparse_only


def test_extract_sparse_index_orders_from_raw_bed_matches_packed():
    block = np.array(
        [
            [0, 1, 0, 2, 1],
            [1, 0, 2, 0, 0],
            [0, 2, 0, 0, 2],
            [2, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    # internal -> BED raw encoding: 0->3, 1->2, 2->0, missing(3)->1
    inv_lut = np.array([3, 2, 0, 1], dtype=np.uint8)
    bytes_per_snp = (block.shape[0] + 3) // 4
    bed_raw = np.zeros(3 + block.shape[1] * bytes_per_snp, dtype=np.uint8)
    bed_raw[0] = 0x6C
    bed_raw[1] = 0x1B
    bed_raw[2] = 0x01
    for j in range(block.shape[1]):
        for i in range(block.shape[0]):
            raw = inv_lut[block[i, j]]
            bed_raw[3 + j * bytes_per_snp + (i // 4)] |= np.uint8(raw << (2 * (i % 4)))

    packed = np.zeros((block.shape[0], (block.shape[1] + 3) // 4), dtype=np.uint8)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            packed[i, j // 4] |= np.uint8(block[i, j] & 3) << np.uint8(2 * (j % 4))

    got_raw = _extract_sparse_index_orders_from_raw_bed(
        bed_raw,
        0,
        block.shape[1],
        bytes_per_snp,
        np.asarray([0, 0, 0, 0], dtype=np.int32),
        np.asarray([0, 2, 4, 6], dtype=np.uint8),
    )
    got_packed = _extract_sparse_index_orders_from_packed(packed, block.shape[1])
    for space in ("csr",):
        for kind in ("het_rows", "het_cols", "hom_rows", "hom_cols", "miss_rows", "miss_cols"):
            assert np.array_equal(got_raw[space][kind], got_packed[space][kind])
    assert got_raw["nnz_het"] == got_packed["nnz_het"]
    assert got_raw["nnz_hom"] == got_packed["nnz_hom"]
    assert got_raw["nnz_miss"] == got_packed["nnz_miss"]


def test_extract_sparse_index_orders_from_varmaj_matches_packed():
    miss = -9
    block = np.array(
        [
            [0, 1, 0, 2, miss],
            [1, 0, 2, 0, 0],
            [0, 2, 0, miss, 2],
            [2, 0, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    packed = np.zeros((block.shape[0], (block.shape[1] + 3) // 4), dtype=np.uint8)
    lut = {0: 0, 1: 1, 2: 2, miss: 3}
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            packed[i, j // 4] |= np.uint8(lut[int(block[i, j])] & 3) << np.uint8(2 * (j % 4))

    block_vm = np.asfortranarray(block.T)
    got_vm = _extract_sparse_index_orders_from_varmaj(block_vm, miss)
    got_packed = _extract_sparse_index_orders_from_packed(packed, block.shape[1])
    for space in ("csr",):
        for kind in ("het_rows", "het_cols", "hom_rows", "hom_cols", "miss_rows", "miss_cols"):
            assert np.array_equal(got_vm[space][kind], got_packed[space][kind])
    assert got_vm["nnz_het"] == got_packed["nnz_het"]
    assert got_vm["nnz_hom"] == got_packed["nnz_hom"]
    assert got_vm["nnz_miss"] == got_packed["nnz_miss"]


def test_zxb_impl_streamed_rejects_component_id_length_mismatch():
    with pytest.raises(ValueError, match="component_ids"):
        KV_IMPL.zxb_impl_streamed(
            jnp.zeros((2, 4), dtype=jnp.float32),
            jnp.asarray([1, 1], dtype=jnp.int32),
            jnp.zeros((2, 4), dtype=jnp.float32),
            jnp.ones((2, 4), dtype=jnp.float32),
            n=3,
            n_calls=2,
            pop_block=lambda _c: np.zeros((3, 1), dtype=np.uint8),
            component_ids=np.asarray([0], dtype=np.int32),
            n_components=1,
        )


class _ArraySource:
    def __init__(self, block: np.ndarray, missing_val: int = -9):
        self._block = np.asarray(block, dtype=np.int8)
        self.n, self.m = self._block.shape
        self.missing_val = int(missing_val)

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        return np.asfortranarray(self._block[:, snp_start : snp_start + snp_count].T)

    def close(self):
        return None


def test_sparse_kv_matches_dense_kv_without_missing():
    X = np.array(
        [
            [0, 1, 0, 2, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 2, 0, 1, 0, 2],
            [2, 0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0, 1],
            [1, 0, 2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X.shape[0] * 2, dtype=jnp.float32).reshape(X.shape[0], 2) / 10.0

    dense = GenoBlockStreamer(_ArraySource(X), call_width=3, keep_host_stats=True)
    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=3, keep_host_stats=True)
    try:
        assert all(not meta["has_missing"] for meta in sparse.sparse_block_metadata())
        got_dense = np.asarray(dense.kv(V))
        got_sparse = np.asarray(sparse.kv(V))
        assert np.allclose(got_sparse, got_dense, atol=1e-5)
    finally:
        dense.close()
        sparse.close()


def test_sparse_kv_matches_dense_semantics_with_missing():
    miss = -9
    X = np.array(
        [
            [0, 1, 0, 2, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 2, miss, 1, 0, 2],
            [2, 0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0, 1],
            [1, 0, 2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    V = jnp.linspace(0.0, 1.0, X.shape[0] * 3, dtype=jnp.float32).reshape(X.shape[0], 3)

    sparse = SparseGenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=3, keep_host_stats=True)
    try:
        assert any(meta["has_missing"] for meta in sparse.sparse_block_metadata())
        dense = GenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=3, keep_host_stats=True)
        got_sparse = np.asarray(sparse.kv(V))
        got_dense = np.asarray(dense.kv(V))
        dense.close()
        assert np.allclose(got_sparse, got_dense, atol=1e-5)
    finally:
        sparse.close()


def test_sparse_kv_accepts_precomputed_sum_v():
    X = np.array(
        [
            [0, 1, 0, 2],
            [1, 0, 2, 0],
            [0, 2, 0, 1],
            [2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X.shape[0] * 2, dtype=jnp.float32).reshape(X.shape[0], 2) / 7.0

    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        got_default = np.asarray(sparse.kv(V))
        got_shared = np.asarray(sparse.kv(V, sum_v=jnp.sum(V, axis=0)))
        assert np.allclose(got_default, got_shared, atol=1e-6)
    finally:
        sparse.close()


def test_sparse_xtv_matches_dense_semantics():
    miss = -9
    X = np.array(
        [
            [0, 1, 0, 2, miss],
            [1, 0, 2, 0, 0],
            [0, 2, miss, 1, 2],
            [2, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X.shape[0] * 3, dtype=jnp.float32).reshape(X.shape[0], 3) / 5.0

    dense = GenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=2, keep_host_stats=True)
    sparse = SparseGenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=2, keep_host_stats=True)
    try:
        got_dense = np.asarray(dense.xtv(V))
        got_sparse = np.asarray(sparse.xtv(V))
        assert np.allclose(got_sparse, got_dense, atol=1e-5)
    finally:
        dense.close()
        sparse.close()


def test_sparse_xtv_and_kv_handle_hom_only_blocks():
    X = np.array(
        [
            [0],
            [2],
        ],
        dtype=np.int8,
    )
    V = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=jnp.float32,
    )

    dense = GenoBlockStreamer(_ArraySource(X), call_width=1, keep_host_stats=True)
    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=1, keep_host_stats=True)
    try:
        got_dense_kv = np.asarray(dense.kv(V))
        got_sparse_kv = np.asarray(sparse.kv(V))
        got_dense_xtv = np.asarray(dense.xtv(V))
        got_sparse_xtv = np.asarray(sparse.xtv(V))
        assert np.allclose(got_sparse_kv, got_dense_kv, atol=1e-6)
        assert np.allclose(got_sparse_xtv, got_dense_xtv, atol=1e-6)
    finally:
        dense.close()
        sparse.close()


def test_sparse_projected_core_atom_matches_dense_semantics():
    miss = -9
    X = np.array(
        [
            [0, 1, 0, 2, miss, 1],
            [1, 0, 2, 0, 0, 1],
            [0, 2, miss, 1, 2, 0],
            [2, 0, 1, 0, 1, 2],
            [0, 1, 0, 2, 0, miss],
            [1, 0, 2, miss, 1, 0],
        ],
        dtype=np.int8,
    )
    U_raw = np.random.RandomState(23).standard_normal((X.shape[0], 3)).astype(np.float32)
    U_np, _ = np.linalg.qr(U_raw)
    U = jnp.asarray(U_np[:, :3], dtype=jnp.float32)

    dense = GenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=3, keep_host_stats=True)
    sparse = SparseGenoBlockStreamer(_ArraySource(X, missing_val=miss), call_width=3, keep_host_stats=True)
    try:
        got_dense = np.asarray(dense.build_projected_core_atom(U))
        got_sparse = np.asarray(sparse.build_projected_core_atom(U))
        assert got_sparse.shape == got_dense.shape == (3, 3)
        assert np.allclose(got_sparse, got_dense, atol=2e-5, rtol=2e-5)
    finally:
        dense.close()
        sparse.close()


def test_sparse_streamer_uses_shards_without_global_exec():
    X1 = np.array(
        [
            [0, 1, 0, 2],
            [1, 0, 2, 0],
            [0, 2, 0, 1],
            [2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    X2 = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [2, 0, 1, 0],
        ],
        dtype=np.int8,
    )

    s1 = SparseGenoBlockStreamer(_ArraySource(X1), call_width=2, keep_host_stats=True)
    s2 = SparseGenoBlockStreamer(_ArraySource(X2), call_width=2, keep_host_stats=True)
    try:
        assert len(s1._sparse_kv_shard_hosts) == 1
        assert len(s2._sparse_kv_shard_hosts) == 1
    finally:
        s1.close()
        s2.close()


def test_sparse_stacked_outputs_match_individual_streamers_without_global_merge():
    X1 = np.array(
        [
            [0, 1, 0, 2],
            [1, 0, 2, 0],
            [0, 2, 0, 1],
            [2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    X2 = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [2, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X1.shape[0] * 4, dtype=jnp.float32).reshape(X1.shape[0], 4) / 11.0

    s1 = SparseGenoBlockStreamer(_ArraySource(X1), call_width=2, keep_host_stats=True)
    s2 = SparseGenoBlockStreamer(_ArraySource(X2), call_width=2, keep_host_stats=True)
    try:
        sum_v = jnp.sum(V, axis=0)
        got = jnp.stack(
            [
                s1.kv(V, normalize=True, sum_v=sum_v),
                s2.kv(V, normalize=True, sum_v=sum_v),
            ],
            axis=0,
        )
        expected = jnp.stack(
            [
                s1.kv(V, normalize=True),
                s2.kv(V, normalize=True),
            ],
            axis=0,
        )
        assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-5)
    finally:
        s1.close()
        s2.close()


def test_sparse_shards_sum_to_same_result_as_single_streamer_kv():
    X = np.array(
        [
            [0, 1, 0, 2, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 2, 0, 1, 0, 2],
            [2, 0, 1, 0, 2, 0],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X.shape[0] * 3, dtype=jnp.float32).reshape(X.shape[0], 3) / 13.0
    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        assert len(sparse._sparse_kv_shard_hosts) >= 1
        sum_v = jnp.sum(V, axis=0)
        got = sparse.kv(V, normalize=False, sum_v=sum_v)
        accum = jnp.zeros_like(got)
        for host, ex in sparse._iter_sparse_exec_pairs():
            piece = KV_IMPL.sparse_kv_shard(
                V,
                n=sparse.n,
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
            accum = accum + piece
        assert np.allclose(np.asarray(got), np.asarray(accum), atol=1e-5)
    finally:
        sparse.close()


def test_sparse_stream_prefetch_mode_matches_dense_reference():
    X = np.array(
        [
            [0, 1, 0, 2, 3, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 2, 3, 1, 0, 2],
            [2, 0, 1, 0, 2, 3],
        ],
        dtype=np.int8,
    )
    V = jnp.arange(X.shape[0] * 3, dtype=jnp.float32).reshape(X.shape[0], 3) / 17.0
    U = jnp.arange(X.shape[0] * 2, dtype=jnp.float32).reshape(X.shape[0], 2) / 19.0

    dense = GenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        assert sparse._sparse_device_mode == "stream_prefetch"
        assert len(sparse._sparse_kv_shard_hosts) >= 1
        assert np.allclose(np.asarray(sparse.kv(V)), np.asarray(dense.kv(V)), atol=1e-5)
        assert np.allclose(np.asarray(sparse.xtv(V)), np.asarray(dense.xtv(V)), atol=1e-5)
        assert np.allclose(
            np.asarray(sparse.build_projected_core_atom(U)),
            np.asarray(dense.build_projected_core_atom(U)),
            atol=1e-5,
        )
    finally:
        dense.close()
        sparse.close()


def test_sparse_stream_prefetch_keeps_no_persistent_device_execs():
    X = np.array(
        [
            [0, 1, 0, 2, 3, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 2, 3, 1, 0, 2],
            [2, 0, 1, 0, 2, 3],
        ],
        dtype=np.int8,
    )
    sparse = SparseGenoBlockStreamer(
        _ArraySource(X),
        call_width=2,
        keep_host_stats=True,
        gpu_budget_bytes=4096,
        mixed_dense_sparse=True,
    )
    try:
        assert sparse._sparse_device_mode == "stream_prefetch"
    finally:
        sparse.close()


def test_empty_sparse_exec_has_merged_csc_keys():
    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int8,
    )
    sparse = SparseGenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        ex = sparse._build_sparse_exec_from_call_range(1, 1)
        assert ex["csc_all_rows"].dtype == np.int32
        assert ex["csc_all_cols"].dtype == np.int32
        assert ex["csc_all_vals"].dtype == np.float32
        assert ex["csc_all_rows"].size == 0
        assert ex["csc_all_cols"].size == 0
        assert ex["csc_all_vals"].size == 0
    finally:
        sparse.close()
