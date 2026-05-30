"""Tests for geno_source.py — genotype block providers.

Verifies that BED and PGEN sources produce valid genotype blocks, and that
the GPU streaming pipeline (GenoBlockStreamer) works identically regardless
of the underlying format.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
BED_PREFIX = os.path.join(DATA_DIR, "all_auto_1k")
PGEN_PREFIX = os.path.join(DATA_DIR, "ukb22828_c1_b0_v3.n1000_p5000_simple")
pytestmark = pytest.mark.skipif(
    not os.path.exists(BED_PREFIX + ".bed") or not os.path.exists(PGEN_PREFIX + ".pgen"),
    reason="local genotype test data are not included in the public repository",
)

_PKG = os.path.basename(REPO_ROOT)
import importlib
_geno_source = importlib.import_module(f"{_PKG}.geno_source")
_geno_stream = importlib.import_module(f"{_PKG}.geno_stream")
_sparse_stream = importlib.import_module(f"{_PKG}.sparse_stream")
_kv_impl = importlib.import_module(f"{_PKG}.kv_impl")
BedGenoSource, PgenGenoSource = _geno_source.BedGenoSource, _geno_source.PgenGenoSource
GenoBlockStreamer, BedBlockStreamer = _geno_stream.GenoBlockStreamer, _geno_stream.BedBlockStreamer
SparseGenoBlockStreamer = _sparse_stream.SparseGenoBlockStreamer
_PinnedHostBuffer = _geno_stream._PinnedHostBuffer
_device_put_block = _kv_impl._device_put_block


# ===================================================================
# Source unit tests
# ===================================================================

class TestBedGenoSource:

    def test_shape_and_dtype(self):
        src = BedGenoSource(BED_PREFIX)
        assert src.n == 1000
        assert src.m == 639577
        block = src.read_block(0, 100)
        assert block.shape == (1000, 100)
        assert block.dtype == np.int8
        src.close()

    def test_values_in_valid_range(self):
        src = BedGenoSource(BED_PREFIX)
        block = src.read_block(1000, 200)
        valid = np.isin(block, [0, 1, 2, src.missing_val])
        assert valid.all(), f"unexpected values: {np.unique(block[~valid])}"
        src.close()

    def test_empty_block(self):
        src = BedGenoSource(BED_PREFIX)
        block = src.read_block(0, 0)
        assert block.shape == (1000, 0)
        src.close()


class TestPgenGenoSource:

    def test_shape_and_dtype(self):
        src = PgenGenoSource(PGEN_PREFIX)
        assert src.n == 1000
        assert src.m == 5000
        block = src.read_block(0, 100)
        assert block.shape == (1000, 100)
        assert block.dtype == np.int8
        src.close()

    def test_values_in_valid_range(self):
        src = PgenGenoSource(PGEN_PREFIX)
        block = src.read_block(0, 500)
        valid = np.isin(block, [0, 1, 2, src.missing_val])
        assert valid.all(), f"unexpected values: {np.unique(block[~valid])}"
        src.close()

    def test_sample_mask_matches_full_read_subset(self):
        full = PgenGenoSource(PGEN_PREFIX)
        mask = np.zeros(full._n_full, dtype=bool)
        mask[[0, 2, 5, 10, 11, 20]] = True
        sub = PgenGenoSource(PGEN_PREFIX, sample_mask=mask)
        block_full = full.read_block(7, 53)
        block_sub = sub.read_block(7, 53)
        assert block_sub.shape == (int(mask.sum()), 53)
        assert np.array_equal(block_sub, block_full[mask, :])
        full.close()
        sub.close()

    def test_reused_chunk_buffer_with_varying_widths(self):
        src = PgenGenoSource(PGEN_PREFIX)
        block_large = src.read_block(0, 503)
        block_small = src.read_block(17, 53)
        block_small_2 = src.read_block(17, 53)
        assert block_large.shape == (1000, 503)
        assert block_small.shape == (1000, 53)
        assert np.array_equal(block_small, block_small_2)
        valid = np.isin(block_small, [0, 1, 2, src.missing_val])
        assert valid.all(), f"unexpected values: {np.unique(block_small[~valid])}"
        src.close()

    def test_variant_major_block_transposes_to_read_block(self):
        src = PgenGenoSource(PGEN_PREFIX)
        block = src.read_block(11, 97)
        block_vm = src.read_block_variant_major(11, 97)
        assert block_vm.shape == (97, 1000)
        assert np.array_equal(block, np.asfortranarray(block_vm.T))
        src.close()


# ===================================================================
# GenoBlockStreamer with non-BED sources
# ===================================================================

class TestGenoBlockStreamer:

    def test_pgen_streamer_builds(self):
        src = PgenGenoSource(PGEN_PREFIX)
        st = GenoBlockStreamer(src, call_width=5000)
        assert st.n == 1000
        assert st.m == 5000
        st.close()

    def test_kv_output_shape(self):
        src = PgenGenoSource(PGEN_PREFIX)
        st = GenoBlockStreamer(src, call_width=5000)
        V = jnp.ones((1000, 2), dtype=jnp.float32)
        out = st.kv(V)
        assert out.shape == (1000, 2)
        assert np.all(np.isfinite(np.asarray(out)))
        st.close()

    def test_xtv_output_shape(self):
        src = PgenGenoSource(PGEN_PREFIX)
        st = GenoBlockStreamer(src, call_width=5000, keep_host_stats=True)
        V = jnp.ones((1000,), dtype=jnp.float32)
        out = st.xtv(V)
        assert out.shape == (5000,)
        assert np.all(np.isfinite(np.asarray(out)))
        st.close()

    def test_fast_bed_build_uses_tmpfile_cache(self):
        fast = GenoBlockStreamer(
            BedGenoSource(BED_PREFIX),
            call_width=4096,
            keep_host_stats=True,
        )
        assert fast._mode == "tmpfile"
        assert fast._packed_mmap is not None
        assert fast._packed_buf is not None
        assert getattr(fast, "_blocks_host_packed", None) is None
        assert fast._packed_offsets is not None
        assert fast._means_host is not None
        assert fast._inv_sds_host is not None
        fast.close()
        assert fast._packed_mmap is None
        assert fast._packed_offsets is None

    def test_fast_pgen_build_uses_tmpfile_cache(self):
        fast = GenoBlockStreamer(
            PgenGenoSource(PGEN_PREFIX),
            call_width=1024,
            keep_host_stats=True,
        )
        assert fast._mode == "tmpfile"
        assert fast._packed_mmap is not None
        assert fast._packed_buf is not None
        assert getattr(fast, "_blocks_host_packed", None) is None
        assert fast._packed_offsets is not None
        assert fast._means_host is not None
        assert fast._inv_sds_host is not None
        fast.close()
        assert fast._packed_mmap is None
        assert fast._packed_offsets is None


# ===================================================================
# Sample mask (subsetting)
# ===================================================================

class TestSampleMask:

    def test_mask_reduces_n(self):
        src = PgenGenoSource(PGEN_PREFIX)
        rng = np.random.RandomState(123)
        mask = rng.rand(1000) > 0.3  # keep ~70%
        n_keep = int(mask.sum())
        st = GenoBlockStreamer(src, call_width=5000, sample_mask=mask)
        assert st.n == n_keep
        V = jnp.ones((n_keep, 2), dtype=jnp.float32)
        out = st.kv(V)
        assert out.shape == (n_keep, 2)
        st.close()

    def test_fast_paths_build_with_sample_mask(self):
        mask = np.zeros(1000, dtype=bool)
        mask[::3] = True

        fast_bed = GenoBlockStreamer(
            BedGenoSource(BED_PREFIX, sample_mask=mask),
            call_width=4096,
            keep_host_stats=True,
        )
        assert fast_bed.n == int(mask.sum())
        assert fast_bed._mode == "tmpfile"
        assert fast_bed._packed_mmap is not None
        fast_bed.close()

        fast_pgen = GenoBlockStreamer(
            PgenGenoSource(PGEN_PREFIX, sample_mask=mask),
            call_width=1024,
            keep_host_stats=True,
        )
        assert fast_pgen.n == int(mask.sum())
        assert fast_pgen._mode == "tmpfile"
        assert fast_pgen._packed_mmap is not None
        fast_pgen.close()

    @pytest.mark.parametrize(
        "src_ctor,n_rows,cols",
        [
            (lambda: BedGenoSource(BED_PREFIX), 1000, np.array([0, 1, 2, 33, 257, 4095], dtype=np.int64)),
            (lambda: PgenGenoSource(PGEN_PREFIX), 1000, np.array([0, 1, 2, 33, 257, 4095], dtype=np.int64)),
        ],
    )
    def test_extract_standardized_columns_matches_manual(self, src_ctor, n_rows, cols):
        src = src_ctor()
        ref_src = src_ctor()
        st = GenoBlockStreamer(src, call_width=4096, keep_host_stats=True)
        got = st.extract_standardized_columns(cols)

        block = ref_src.read_block(int(cols.min()), int(cols.max() - cols.min() + 1))
        raw = block[:, cols - int(cols.min())].astype(np.float32, copy=False)
        miss = ref_src.missing_val
        ref = np.empty_like(got)
        for j, col in enumerate(cols):
            g = raw[:, j]
            valid = g != miss
            mean = float(np.mean(g[valid])) if np.any(valid) else 0.0
            var = float(np.var(g[valid])) if np.any(valid) else 0.0
            inv = 1.0 / np.sqrt(max(var, 1e-6)) if valid.any() and var > 0.0 else 0.0
            g_imp = np.where(valid, g, mean)
            ref[:, j] = (g_imp - mean) * inv
        assert np.allclose(got, ref, atol=1e-6)
        st.close()
        ref_src.close()


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompat:

    def test_bed_block_streamer_still_works(self):
        """BedBlockStreamer(bed_prefix=...) should behave as before."""
        st = BedBlockStreamer(
            bed_prefix=BED_PREFIX, call_width=131072,
        )
        assert st.n == 1000
        assert st.m == 639577
        V = jnp.ones((1000, 1), dtype=jnp.float32)
        out = st.kv(V)
        assert out.shape == (1000, 1)
        assert np.all(np.isfinite(np.asarray(out)))
        st.close()

    def test_geno_block_streamer_with_bed_prefix(self):
        """GenoBlockStreamer(bed_prefix=...) should also work."""
        st = GenoBlockStreamer(bed_prefix=BED_PREFIX, call_width=131072)
        assert st.n == 1000
        V = jnp.ones((1000, 1), dtype=jnp.float32)
        out = st.kv(V)
        assert out.shape == (1000, 1)
        st.close()

    def test_bed_block_streamer_with_sample_mask(self):
        """BED path should support logical sample subsetting without crop."""
        mask = np.zeros(1000, dtype=bool)
        mask[:400] = True
        st = BedBlockStreamer(
            bed_prefix=BED_PREFIX, call_width=131072, sample_mask=mask,
        )
        assert st.n == 400
        V = jnp.ones((400, 1), dtype=jnp.float32)
        out = st.kv(V)
        assert out.shape == (400, 1)
        st.close()


class _FakeGpuDevice:
    platform = "gpu"


class TestPinnedDevicePut:

    def test_device_put_block_skips_copy_for_pinned_ring_blocks(self, monkeypatch):
        base = np.arange(32, dtype=np.uint8).view(_PinnedHostBuffer)
        block = base[:16].reshape(4, 4)
        seen = {}

        def _fake_device_put(x, dev):
            seen["x"] = x
            seen["dev"] = dev
            return x

        monkeypatch.setattr(_kv_impl.jax, "device_put", _fake_device_put)
        out = _device_put_block(block, _FakeGpuDevice())

        assert seen["dev"].platform == "gpu"
        assert np.shares_memory(seen["x"], block)
        assert np.shares_memory(out, block)

    def test_device_put_block_keeps_copy_for_unpinned_blocks(self, monkeypatch):
        block = np.arange(16, dtype=np.uint8).reshape(4, 4)
        seen = {}

        def _fake_device_put(x, dev):
            seen["x"] = x
            seen["dev"] = dev
            return x

        monkeypatch.setattr(_kv_impl.jax, "device_put", _fake_device_put)
        out = _device_put_block(block, _FakeGpuDevice())

        assert seen["dev"].platform == "gpu"
        assert not np.shares_memory(seen["x"], block)
        assert not np.shares_memory(out, block)
        assert np.array_equal(out, block)


def test_sparse_extract_standardized_columns_uses_raw_variant_blocks():
    class _FakeSource:
        missing_val = 3

        def read_block_variant_major(self, j0, tw):
            full = np.array(
                [
                    [0, 1, 2, 3],
                    [1, 2, 3, 0],
                    [2, 3, 0, 1],
                ],
                dtype=np.int8,
            )
            return full[:, j0 : j0 + tw]

    st = object.__new__(SparseGenoBlockStreamer)
    st._mode = "tmpfile"
    st._source = _FakeSource()
    st._sample_mask = None
    st._means_host = np.array([1.0, 1.5, 1.0, 0.5], dtype=np.float32)
    st._inv_sds_host = np.array([1.0, 2.0, 0.5, 4.0], dtype=np.float32)
    st._call_snp_starts = np.array([0, 2], dtype=np.int64)
    st._call_true_widths = np.array([2, 2], dtype=np.int64)
    st._n_calls = 2
    st._bed_int_missing = 3
    st.n = 3
    st.m = 4

    got = st.extract_standardized_columns(np.array([3, 1, 3], dtype=np.int64))
    ref_full = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    ref = ref_full[:, [3, 1, 3]]
    assert np.allclose(got, ref, atol=1e-6)


def test_sparse_extract_standardized_columns_can_reopen_pgen_source():
    src = PgenGenoSource(PGEN_PREFIX)
    st = SparseGenoBlockStreamer(src, call_width=1024, keep_host_stats=True)
    cols = np.array([0, 7, 25], dtype=np.int64)

    assert st._source is None
    got = st.extract_standardized_columns(cols)

    ref_src = PgenGenoSource(PGEN_PREFIX)
    block = ref_src.read_block(int(cols.min()), int(cols.max() - cols.min() + 1))
    raw = block[:, cols - int(cols.min())].astype(np.float32, copy=False)
    miss = ref_src.missing_val
    ref = np.empty_like(got)
    for j, _col in enumerate(cols):
        g = raw[:, j]
        valid = g != miss
        mean = float(np.mean(g[valid])) if np.any(valid) else 0.0
        var = float(np.var(g[valid])) if np.any(valid) else 0.0
        inv = 1.0 / np.sqrt(max(var, 1e-6)) if valid.any() and var > 0.0 else 0.0
        g_imp = np.where(valid, g, mean)
        ref[:, j] = (g_imp - mean) * inv
    assert np.allclose(got, ref, atol=1e-6)
    ref_src.close()
    st.close()


def test_geno_streamer_packed_access_raises_clean_error_without_cache():
    st = object.__new__(GenoBlockStreamer)
    st._packed_offsets = None
    st._packed_buf = None
    st._ring = None
    st._packed_call_widths = np.array([1], dtype=np.int32)
    st.n = 1

    with pytest.raises(RuntimeError, match="Packed block cache is unavailable"):
        st._pop_cached(0)


def test_build_tmpfile_bed_uses_fused_stats_transcode_when_writing_cache(monkeypatch, tmp_path):
    bed_path = tmp_path / "tiny.bed"
    bed_raw = np.array(
        [
            0x6C,
            0x1B,
            0x01,
            0b00001111,
            0b00110011,
            0b00001010,
            0b00000000,
        ],
        dtype=np.uint8,
    )
    bed_path.write_bytes(bed_raw.tobytes())

    st = object.__new__(GenoBlockStreamer)
    st.n = 2
    st.m = 4
    st.call_width = 4
    st._n_calls = 1
    st._call_true_widths = np.array([4], dtype=np.int32)
    st._block_starts = np.array([0], dtype=np.int64)
    st._packed_call_widths = np.array([1], dtype=np.int32)
    st._max_packed_width = 1
    st._bed_int_missing = 3
    st._build_threads = 1
    st._standardization_override = None
    st._should_write_packed_cache = lambda: True
    st._can_post_build_from_raw_bed = lambda: False
    st._build_bed_raw_block = lambda *args, **kwargs: None
    st._post_build_block = lambda *args, **kwargs: None
    st._finalize_tmpfile_build = lambda **kwargs: (kwargs["tmp_fd"], kwargs["tmp_path"])

    seen = {"stats": 0, "transcode": 0, "fused": 0}
    orig_stats = _geno_stream._stats_from_raw_bed_numba
    orig_transcode = _geno_stream._transcode_raw_bed_numba
    orig_fused = _geno_stream._stats_and_transcode_raw_bed_numba

    def _wrap_stats(*args, **kwargs):
        seen["stats"] += 1
        return orig_stats(*args, **kwargs)

    def _wrap_transcode(*args, **kwargs):
        seen["transcode"] += 1
        return orig_transcode(*args, **kwargs)

    def _wrap_fused(*args, **kwargs):
        seen["fused"] += 1
        return orig_fused(*args, **kwargs)

    monkeypatch.setattr(_geno_stream, "_stats_from_raw_bed_numba", _wrap_stats)
    monkeypatch.setattr(_geno_stream, "_transcode_raw_bed_numba", _wrap_transcode)
    monkeypatch.setattr(_geno_stream, "_stats_and_transcode_raw_bed_numba", _wrap_fused)

    means, inv_sds, eff, counts = st._build_tmpfile_bed(str(bed_path), n_full=2, sample_idx=None)

    assert means.shape == (4,)
    assert inv_sds.shape == (4,)
    assert counts.shape == (4,)
    assert eff >= 0.0
    assert seen["fused"] == 1
    assert seen["stats"] == 0
    assert seen["transcode"] == 0


def test_cuda_device_ordinal_returns_none_without_device_ids():
    dev = type("Dev", (), {"platform": "gpu"})()
    assert _geno_stream._cuda_device_ordinal(dev) is None
