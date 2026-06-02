from __future__ import annotations

import importlib
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
GENO_STREAM = importlib.import_module(f"{PKG.__name__}.geno_stream")
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")
PRECOND = importlib.import_module(f"{PKG.__name__}.precond")
KV_IMPL = importlib.import_module(f"{PKG.__name__}.kv_impl")
GENO_SOURCE = importlib.import_module(f"{PKG.__name__}.geno_source")

GenoBlockStreamer = GENO_STREAM.GenoBlockStreamer
BedBlockStreamer = GENO_STREAM.BedBlockStreamer
FitConfig = REML_MODEL.FitConfig
InfinitesimalREMLFitter = REML_MODEL.InfinitesimalREMLFitter
ProjectedCorePrecondConf = PRECOND.ProjectedCorePrecondConf
build_projected_core_atoms_multi_streamed = KV_IMPL.build_projected_core_atoms_multi_streamed
BedGenoSource = GENO_SOURCE.BedGenoSource


class _ArraySource:
    def __init__(self, block: np.ndarray, missing_val: int = -9):
        self._block = np.asarray(block, dtype=np.int8)
        self.n, self.m = self._block.shape
        self.missing_val = int(missing_val)

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        return np.asfortranarray(self._block[:, snp_start : snp_start + snp_count].T)

    def close(self):
        return None


class _CountingVariantMajorSource(_ArraySource):
    def __init__(self, block: np.ndarray, missing_val: int = -9):
        super().__init__(block, missing_val=missing_val)
        self.read_calls: list[tuple[int, int]] = []

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        self.read_calls.append((int(snp_start), int(snp_count)))
        return super().read_block_variant_major(snp_start, snp_count)


def _make_non_degenerate_genotypes(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    for _ in range(1024):
        X = rng.randint(0, 3, size=(n, m), dtype=np.int8)
        if np.all(np.var(X.astype(np.float32), axis=0) > 0.0):
            return X
    raise RuntimeError("failed to build non-degenerate genotype matrix")


def test_partitioned_streamer_matches_split_streamers():
    X = _make_non_degenerate_genotypes(n=10, m=12, seed=0)
    block_sizes = [5, 4, 3]
    V = jnp.asarray(
        np.random.RandomState(1).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    theta_g = jnp.asarray([0.2, 0.35, 0.15], dtype=jnp.float32)
    theta_e = jnp.asarray(0.4, dtype=jnp.float32)

    part = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_block_sizes=block_sizes,
        keep_host_stats=True,
    )
    split = [
        GenoBlockStreamer(_ArraySource(X[:, :5]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 5:9]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 9:]), call_width=4, keep_host_stats=True),
    ]
    try:
        got_stack = np.asarray(part.stacked_component_kv(V))
        ref_stack = np.stack([np.asarray(st.kv(V)) for st in split], axis=0)
        assert got_stack.shape == ref_stack.shape == (3, X.shape[0], 2)
        assert np.allclose(got_stack, ref_stack, atol=1e-5)

        got_hv = np.asarray(part.weighted_component_hv(theta_g, theta_e, V))
        ref_hv = theta_e * np.asarray(V)
        for g_idx, st in enumerate(split):
            ref_hv = ref_hv + float(theta_g[g_idx]) * np.asarray(st.kv(V))
        assert np.allclose(got_hv, ref_hv, atol=1e-5)

        for g_idx, st in enumerate(split):
            got_one = np.asarray(part.component_kv(V, g_idx))
            ref_one = np.asarray(st.kv(V))
            assert np.allclose(got_one, ref_one, atol=1e-5)
    finally:
        part.close()
        for st in split:
            st.close()


def test_arbitrary_grouped_streamer_matches_split_streamers():
    X = _make_non_degenerate_genotypes(n=10, m=12, seed=101)
    component_variant_indices = [
        [0, 3, 4, 10],
        [1, 2, 7],
        [5, 6, 8, 9, 11],
    ]
    V = jnp.asarray(
        np.random.RandomState(102).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    theta_g = jnp.asarray([0.2, 0.35, 0.15], dtype=jnp.float32)
    theta_e = jnp.asarray(0.4, dtype=jnp.float32)

    part = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_variant_indices=component_variant_indices,
        keep_host_stats=True,
    )
    split = [
        GenoBlockStreamer(
            _ArraySource(X[:, np.asarray(group, dtype=np.int64)]),
            call_width=4,
            keep_host_stats=True,
        )
        for group in component_variant_indices
    ]
    try:
        got_stack = np.asarray(part.stacked_component_kv(V))
        ref_stack = np.stack([np.asarray(st.kv(V)) for st in split], axis=0)
        assert got_stack.shape == ref_stack.shape == (3, X.shape[0], 2)
        assert np.allclose(got_stack, ref_stack, atol=1e-5)

        got_hv = np.asarray(part.weighted_component_hv(theta_g, theta_e, V))
        ref_hv = theta_e * np.asarray(V)
        for g_idx, st in enumerate(split):
            ref_hv = ref_hv + float(theta_g[g_idx]) * np.asarray(st.kv(V))
        assert np.allclose(got_hv, ref_hv, atol=1e-5)

        for g_idx, st in enumerate(split):
            got_one = np.asarray(part.component_kv(V, g_idx))
            ref_one = np.asarray(st.kv(V))
            assert np.allclose(got_one, ref_one, atol=1e-5)
            assert np.array_equal(
                part.component_source_variant_indices(g_idx),
                np.asarray(component_variant_indices[g_idx], dtype=np.int64),
            )
    finally:
        part.close()
        for st in split:
            st.close()


def test_arbitrary_grouped_streamer_build_reads_source_in_source_order_chunks():
    X = _make_non_degenerate_genotypes(n=10, m=12, seed=103)
    src = _CountingVariantMajorSource(X)
    st = GenoBlockStreamer(
        src,
        call_width=4,
        component_variant_indices=[
            [0, 4, 8, 9],
            [1, 5, 10, 11],
            [2, 3, 6, 7],
        ],
        keep_host_stats=True,
    )
    try:
        assert src.read_calls == [(0, 4), (4, 4), (8, 4)]
    finally:
        st.close()


def test_partial_arbitrary_grouped_streamer_matches_subset_reference():
    X = _make_non_degenerate_genotypes(n=18, m=14, seed=1041)
    component_variant_indices = [
        [0, 5, 9],
        [2, 6, 13],
    ]
    selected = np.concatenate(
        [np.asarray(group, dtype=np.int64) for group in component_variant_indices]
    )
    local_groups = [
        np.arange(0, 3, dtype=np.int64),
        np.arange(3, 6, dtype=np.int64),
    ]
    V = jnp.asarray(
        np.random.RandomState(1042).standard_normal((X.shape[0], 3)).astype(np.float32)
    )

    part = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_variant_indices=component_variant_indices,
        keep_host_stats=True,
    )
    ref = GenoBlockStreamer(
        _ArraySource(X[:, selected]),
        call_width=3,
        component_variant_indices=local_groups,
        keep_host_stats=True,
    )
    try:
        assert part.m == selected.size
        assert part.source_m == X.shape[1]
        assert np.array_equal(
            np.asarray(part._cache_to_source_variant_indices, dtype=np.int64),
            selected,
        )
        assert np.allclose(
            np.asarray(part._means_host, dtype=np.float32),
            np.asarray(ref._means_host, dtype=np.float32),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(part._inv_sds_host, dtype=np.float32),
            np.asarray(ref._inv_sds_host, dtype=np.float32),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(part.stacked_component_kv(V)),
            np.asarray(ref.stacked_component_kv(V)),
            atol=1e-5,
        )
    finally:
        part.close()
        ref.close()


def test_partial_arbitrary_grouped_streamer_skips_unselected_source_gaps():
    X = _make_non_degenerate_genotypes(n=10, m=20, seed=1046)
    src = _CountingVariantMajorSource(X)
    st = GenoBlockStreamer(
        src,
        call_width=4,
        component_variant_indices=[
            [0],
            [9],
            [19],
        ],
        keep_host_stats=True,
        source_build_chunk_width=4,
    )
    try:
        assert src.read_calls == [(0, 4), (9, 4), (19, 1)]
    finally:
        st.close()


def test_arbitrary_grouped_streamer_honors_source_build_chunk_width_override():
    X = _make_non_degenerate_genotypes(n=10, m=12, seed=1043)
    src = _CountingVariantMajorSource(X)
    st = GenoBlockStreamer(
        src,
        call_width=4,
        source_build_chunk_width=6,
        component_variant_indices=[
            [0, 4, 8, 9],
            [1, 5, 10, 11],
            [2, 3, 6, 7],
        ],
        keep_host_stats=True,
    )
    try:
        assert src.read_calls == [(0, 6), (6, 6)]
    finally:
        st.close()


def test_arbitrary_grouped_bed_streamer_uses_raw_source_order_build(tmp_path, monkeypatch):
    bed_reader = pytest.importorskip("bed_reader")

    X = _make_non_degenerate_genotypes(n=10, m=12, seed=104)
    sample_mask = np.array([True, False, True, True, False, True, True, False, True, True])
    component_variant_indices = [
        [0, 4, 8, 9],
        [1, 5, 10, 11],
        [2, 3, 6, 7],
    ]
    V = jnp.asarray(
        np.random.RandomState(105).standard_normal((int(sample_mask.sum()), 2)).astype(np.float32)
    )

    prefix = tmp_path / "toy"
    bed_reader.to_bed(str(prefix) + ".bed", X.astype(np.float32))

    ref_st = GenoBlockStreamer(
        _ArraySource(X[sample_mask, :]),
        call_width=4,
        component_variant_indices=component_variant_indices,
        keep_host_stats=True,
    )

    def _unexpected_varmaj_source_order(*_args, **_kwargs):
        raise AssertionError("BED arbitrary grouping should not fall back to variant-major source-order build.")

    def _unexpected_bed_varmaj_read(*_args, **_kwargs):
        raise AssertionError("BED arbitrary grouping should not read variant-major blocks.")

    monkeypatch.setattr(
        GenoBlockStreamer,
        "_build_tmpfile_varmaj_source_order",
        _unexpected_varmaj_source_order,
    )
    monkeypatch.setattr(
        BedGenoSource,
        "read_block_variant_major",
        _unexpected_bed_varmaj_read,
    )

    bed_st = BedBlockStreamer(
        str(prefix),
        call_width=4,
        component_variant_indices=component_variant_indices,
        sample_mask=sample_mask,
        keep_host_stats=True,
    )
    try:
        assert bed_st._has_arbitrary_component_partition
        assert np.array_equal(
            np.asarray(bed_st._cache_to_source_variant_indices, dtype=np.int64),
            np.asarray(ref_st._cache_to_source_variant_indices, dtype=np.int64),
        )
        assert np.allclose(
            np.asarray(bed_st._means_host, dtype=np.float32),
            np.asarray(ref_st._means_host, dtype=np.float32),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(bed_st._inv_sds_host, dtype=np.float32),
            np.asarray(ref_st._inv_sds_host, dtype=np.float32),
            atol=1e-6,
        )
        got = np.asarray(bed_st.stacked_component_kv(V))
        ref = np.asarray(ref_st.stacked_component_kv(V))
        assert np.allclose(got, ref, atol=1e-5)
    finally:
        bed_st.close()
        ref_st.close()


def test_partial_arbitrary_grouped_bed_streamer_matches_subset_reference(tmp_path):
    bed_reader = pytest.importorskip("bed_reader")

    X = _make_non_degenerate_genotypes(n=12, m=16, seed=1044)
    sample_mask = np.array(
        [True, True, False, True, False, True, True, False, True, True, False, True]
    )
    component_variant_indices = [
        [0, 5, 8],
        [2, 11, 15],
    ]
    selected = np.concatenate(
        [np.asarray(group, dtype=np.int64) for group in component_variant_indices]
    )
    local_groups = [
        np.arange(0, 3, dtype=np.int64),
        np.arange(3, 6, dtype=np.int64),
    ]
    V = jnp.asarray(
        np.random.RandomState(1045)
        .standard_normal((int(sample_mask.sum()), 2))
        .astype(np.float32)
    )

    prefix = tmp_path / "toy_partial"
    bed_reader.to_bed(str(prefix) + ".bed", X.astype(np.float32))

    ref_st = GenoBlockStreamer(
        _ArraySource(X[sample_mask, :][:, selected]),
        call_width=3,
        component_variant_indices=local_groups,
        keep_host_stats=True,
    )
    bed_st = BedBlockStreamer(
        str(prefix),
        call_width=3,
        component_variant_indices=component_variant_indices,
        sample_mask=sample_mask,
        keep_host_stats=True,
    )
    try:
        assert bed_st.m == selected.size
        assert bed_st.source_m == X.shape[1]
        assert np.array_equal(
            np.asarray(bed_st._cache_to_source_variant_indices, dtype=np.int64),
            selected,
        )
        assert np.allclose(
            np.asarray(bed_st._means_host, dtype=np.float32),
            np.asarray(ref_st._means_host, dtype=np.float32),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(bed_st._inv_sds_host, dtype=np.float32),
            np.asarray(ref_st._inv_sds_host, dtype=np.float32),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(bed_st.stacked_component_kv(V)),
            np.asarray(ref_st.stacked_component_kv(V)),
            atol=1e-5,
        )
    finally:
        bed_st.close()
        ref_st.close()


def test_streamed_kv_matches_manual_reference_on_active_device():
    X = _make_non_degenerate_genotypes(n=24, m=12, seed=5)
    V = np.random.RandomState(2).standard_normal((X.shape[0], 2)).astype(np.float32)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        keep_host_stats=True,
    )
    try:
        got = np.asarray(st.kv(jnp.asarray(V)))
        Z = st.extract_standardized_columns(np.arange(st.m))
        eff = float(np.count_nonzero(st._inv_sds_host > 0.0))
        ref = (Z @ (Z.T @ V)) / eff
        assert np.allclose(got, ref, atol=2e-3, rtol=2e-4)
    finally:
        st.close()


def test_partitioned_fit_path_matches_multi_stream_operators(monkeypatch):
    X = _make_non_degenerate_genotypes(n=24, m=12, seed=7)
    V = jnp.asarray(
        np.random.RandomState(3).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    theta_g = jnp.asarray([0.25, 0.15, 0.4], dtype=jnp.float32)
    theta_e = jnp.asarray(0.35, dtype=jnp.float32)

    ref_streamers = [
        GenoBlockStreamer(_ArraySource(X[:, :5]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 5:9]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 9:]), call_width=4, keep_host_stats=True),
    ]

    def _fake_fit_reml(*, K_mvs, diag_list, weighted_hv=None, stacked_kv=None, **_kwargs):
        ref_stack = np.stack([np.asarray(st.kv(V)) for st in ref_streamers], axis=0)
        got_stack = np.asarray(stacked_kv(V))
        assert len(K_mvs) == 3
        assert len(diag_list) == 3
        assert got_stack.shape == ref_stack.shape
        assert np.allclose(got_stack, ref_stack, atol=1e-5)

        got_one = np.asarray(K_mvs[1](V))
        ref_one = np.asarray(ref_streamers[1].kv(V))
        assert np.allclose(got_one, ref_one, atol=1e-5)

        got_hv = np.asarray(weighted_hv(theta_g, theta_e, V))
        ref_hv = theta_e * np.asarray(V)
        for g_idx, st in enumerate(ref_streamers):
            ref_hv = ref_hv + float(theta_g[g_idx]) * np.asarray(st.kv(V))
        assert np.allclose(got_hv, ref_hv, atol=1e-5)
        return jnp.asarray([0.2, 0.3, 0.1, 0.4], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fit_part = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=[5, 4, 3],
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res_part = fit_part.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
    finally:
        for st in fit_part.streamers:
            st.close()
        for st in ref_streamers:
            st.close()

    assert np.allclose(np.asarray(res_part.var_components), [0.2, 0.3, 0.1, 0.4])


def test_arbitrary_grouped_fit_path_matches_multi_stream_operators(monkeypatch):
    X = _make_non_degenerate_genotypes(n=24, m=12, seed=107)
    component_variant_indices = [
        [0, 3, 4, 10],
        [1, 2, 7],
        [5, 6, 8, 9, 11],
    ]
    V = jnp.asarray(
        np.random.RandomState(108).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    theta_g = jnp.asarray([0.25, 0.15, 0.4], dtype=jnp.float32)
    theta_e = jnp.asarray(0.35, dtype=jnp.float32)

    ref_streamers = [
        GenoBlockStreamer(
            _ArraySource(X[:, np.asarray(group, dtype=np.int64)]),
            call_width=4,
            keep_host_stats=True,
        )
        for group in component_variant_indices
    ]

    def _fake_fit_reml(*, K_mvs, diag_list, weighted_hv=None, stacked_kv=None, **_kwargs):
        ref_stack = np.stack([np.asarray(st.kv(V)) for st in ref_streamers], axis=0)
        got_stack = np.asarray(stacked_kv(V))
        assert len(K_mvs) == 3
        assert len(diag_list) == 3
        assert got_stack.shape == ref_stack.shape
        assert np.allclose(got_stack, ref_stack, atol=1e-5)

        got_one = np.asarray(K_mvs[1](V))
        ref_one = np.asarray(ref_streamers[1].kv(V))
        assert np.allclose(got_one, ref_one, atol=1e-5)

        got_hv = np.asarray(weighted_hv(theta_g, theta_e, V))
        ref_hv = theta_e * np.asarray(V)
        for g_idx, st in enumerate(ref_streamers):
            ref_hv = ref_hv + float(theta_g[g_idx]) * np.asarray(st.kv(V))
        assert np.allclose(got_hv, ref_hv, atol=1e-5)
        return jnp.asarray([0.2, 0.3, 0.1, 0.4], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fit_part = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            component_variant_indices=component_variant_indices,
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res_part = fit_part.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
    finally:
        for st in fit_part.streamers:
            st.close()
        for st in ref_streamers:
            st.close()

    assert np.allclose(np.asarray(res_part.var_components), [0.2, 0.3, 0.1, 0.4])


def test_partitioned_fit_smoke_runs_with_real_reml():
    X = _make_non_degenerate_genotypes(n=20, m=12, seed=13)
    y = (
        0.5 * X[:, 1].astype(np.float32)
        - 0.3 * X[:, 7].astype(np.float32)
        + np.random.RandomState(21).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=[5, 4, 3],
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            n_rand_vec=4,
            max_pcg_iters=40,
            minq_iter=2,
            slq_samples=4,
            slq_m=4,
            seed=17,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.asarray(y))
    finally:
        for st in fitter.streamers:
            st.close()

    assert res.var_components.shape == (4,)
    assert np.all(np.isfinite(np.asarray(res.var_components)))
    assert res.history


def test_partitioned_projected_core_atoms_match_manual_reference():
    X = _make_non_degenerate_genotypes(n=18, m=12, seed=23)
    block_sizes = [5, 4, 3]
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_block_sizes=block_sizes,
        keep_host_stats=True,
    )
    try:
        U_raw = np.random.RandomState(9).standard_normal((X.shape[0], 3)).astype(np.float32)
        U_np, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U_np[:, :3], dtype=jnp.float32)

        got = np.asarray(st.build_projected_core_atoms(U, subtract_identity=True))
        offsets = np.r_[0, np.cumsum(block_sizes)]
        eye = np.eye(U.shape[1], dtype=np.float32)
        ref = []
        for g_idx in range(len(block_sizes)):
            cols = np.arange(offsets[g_idx], offsets[g_idx + 1])
            Z = st.extract_standardized_columns(cols)
            eff = float(np.count_nonzero(st._inv_sds_host[cols] > 0.0))
            if eff > 0.0:
                KgU = (Z @ (Z.T @ U_np)) / eff
                ref.append(U_np.T @ KgU - eye)
            else:
                ref.append(np.zeros_like(eye))
        ref = np.stack(ref, axis=0)
        assert got.shape == ref.shape == (3, 3, 3)
        assert np.allclose(got, ref, atol=2e-3, rtol=2e-4)
    finally:
        st.close()


def test_arbitrary_grouped_projected_core_atoms_match_manual_reference():
    X = _make_non_degenerate_genotypes(n=18, m=12, seed=123)
    component_variant_indices = [
        [0, 3, 4, 10],
        [1, 2, 7],
        [5, 6, 8, 9, 11],
    ]
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_variant_indices=component_variant_indices,
        keep_host_stats=True,
    )
    split = [
        GenoBlockStreamer(
            _ArraySource(X[:, np.asarray(group, dtype=np.int64)]),
            call_width=4,
            keep_host_stats=True,
        )
        for group in component_variant_indices
    ]
    try:
        U_raw = np.random.RandomState(124).standard_normal((X.shape[0], 3)).astype(np.float32)
        U_np, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U_np[:, :3], dtype=jnp.float32)

        got = np.asarray(st.build_projected_core_atoms(U, subtract_identity=True))
        eye = np.eye(U.shape[1], dtype=np.float32)
        ref = []
        for split_st in split:
            Z = split_st.extract_standardized_columns(np.arange(split_st.m))
            eff = float(np.count_nonzero(split_st._inv_sds_host > 0.0))
            if eff > 0.0:
                KgU = (Z @ (Z.T @ U_np)) / eff
                ref.append(U_np.T @ KgU - eye)
            else:
                ref.append(np.zeros_like(eye))
        ref = np.stack(ref, axis=0)
        assert got.shape == ref.shape == (3, 3, 3)
        assert np.allclose(got, ref, atol=2e-3, rtol=2e-4)
    finally:
        st.close()
        for split_st in split:
            split_st.close()


def test_multi_stream_projected_core_atoms_match_manual_reference():
    X = _make_non_degenerate_genotypes(n=18, m=12, seed=29)
    streamers = [
        GenoBlockStreamer(_ArraySource(X[:, :5]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 5:9]), call_width=4, keep_host_stats=True),
        GenoBlockStreamer(_ArraySource(X[:, 9:]), call_width=4, keep_host_stats=True),
    ]
    try:
        U_raw = np.random.RandomState(11).standard_normal((X.shape[0], 3)).astype(np.float32)
        U_np, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U_np[:, :3], dtype=jnp.float32)

        for st in streamers:
            st._prepare_kv_pass()
        call_plan = tuple(
            (g_idx, c_idx)
            for g_idx, st in enumerate(streamers)
            for c_idx in range(int(st._n_calls))
        )
        got = np.asarray(
            build_projected_core_atoms_multi_streamed(
                U,
                streamers,
                call_plan,
                subtract_identity=True,
            )
        )

        splits = [slice(0, 5), slice(5, 9), slice(9, 12)]
        eye = np.eye(U.shape[1], dtype=np.float32)
        ref = []
        for g_idx, cols in enumerate(splits):
            Z = streamers[g_idx].extract_standardized_columns(np.arange(cols.start, cols.stop) - cols.start)
            eff = float(np.count_nonzero(streamers[g_idx]._inv_sds_host > 0.0))
            if eff > 0.0:
                KgU = (Z @ (Z.T @ U_np)) / eff
                ref.append(U_np.T @ KgU - eye)
            else:
                ref.append(np.zeros_like(eye))
        ref = np.stack(ref, axis=0)
        assert got.shape == ref.shape == (3, 3, 3)
        assert np.allclose(got, ref, atol=2e-3, rtol=2e-4)
    finally:
        for st in streamers:
            st.close()


def test_partitioned_projected_core_precond_reaches_fit_reml(monkeypatch):
    X = _make_non_degenerate_genotypes(n=20, m=12, seed=31)

    def _fake_fit_reml(*, K_mvs, diag_list, precond_conf=None, **_kwargs):
        assert len(K_mvs) == 3
        assert len(diag_list) == 3
        assert isinstance(precond_conf, ProjectedCorePrecondConf)
        assert precond_conf.total_rank == 3
        assert precond_conf.n_grm == 3
        assert precond_conf.U.shape == (X.shape[0], 3)
        assert precond_conf.core_atoms.shape == (3, 3, 3)
        return jnp.asarray([0.2, 0.3, 0.1, 0.4], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=[5, 4, 3],
            call_width=4,
            keep_host_stats=True,
            precond_type="projected_core",
            precond_rank=3,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
    finally:
        for st in fitter.streamers:
            st.close()

    assert np.allclose(np.asarray(res.var_components), [0.2, 0.3, 0.1, 0.4])


def test_partitioned_projected_core_build_uses_streamed_atoms(monkeypatch):
    X = _make_non_degenerate_genotypes(n=20, m=12, seed=41)
    calls = {"atoms": 0}

    def _fake_basis(K_mv, n, max_rank, key, oversample=8, dtype=jnp.float32):
        del K_mv, key, oversample
        U = jnp.eye(n, dtype=dtype)[:, :max_rank]
        evals = jnp.ones((max_rank,), dtype=dtype)
        return U, evals

    def _fake_fit_reml(*, precond_conf=None, **_kwargs):
        assert isinstance(precond_conf, ProjectedCorePrecondConf)
        return jnp.asarray([0.2, 0.3, 0.1, 0.4], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.build_lowrank_basis", _fake_basis)
    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=[5, 4, 3],
            call_width=4,
            keep_host_stats=True,
            precond_type="projected_core",
            precond_rank=3,
            verbose=False,
        )
    )
    try:
        st = fitter._partitioned_streamer

        def _fake_atoms(U, *, subtract_identity=True):
            calls["atoms"] += 1
            assert subtract_identity is True
            return jnp.zeros((st.n_components, U.shape[1], U.shape[1]), dtype=U.dtype)

        def _fail_stacked(_V, normalize=True):
            del normalize
            raise AssertionError("partitioned projected-core build should not call stacked_component_kv")

        monkeypatch.setattr(st, "build_projected_core_atoms", _fake_atoms)
        monkeypatch.setattr(st, "stacked_component_kv", _fail_stacked)

        fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
    finally:
        for st_i in fitter.streamers:
            st_i.close()

    assert calls["atoms"] == 1


def test_multi_stream_projected_core_precond_avoids_stacked_kv(monkeypatch):
    X = _make_non_degenerate_genotypes(n=20, m=12, seed=41)
    calls = {"atoms": 0}

    def _fake_basis(*, n, max_rank, **_kwargs):
        U_raw = np.random.RandomState(13).standard_normal((n, max_rank)).astype(np.float32)
        U_np, _ = np.linalg.qr(U_raw)
        return jnp.asarray(U_np[:, :max_rank], dtype=jnp.float32), None

    def _fake_fit_reml(*, precond_conf=None, **_kwargs):
        assert isinstance(precond_conf, ProjectedCorePrecondConf)
        return jnp.asarray([0.2, 0.3, 0.1, 0.4], dtype=jnp.float32), [{"iter": 1}]

    def _fake_atoms(U, streamers, call_plan, *, subtract_identity=True, **_kwargs):
        calls["atoms"] += 1
        assert subtract_identity is True
        assert len(streamers) == 3
        assert len(call_plan) > 0
        return jnp.zeros((len(streamers), U.shape[1], U.shape[1]), dtype=U.dtype)

    def _fail_stacked(*_args, **_kwargs):
        raise AssertionError("multi-stream projected-core build should not call kv_impl_multi_streamed_stacked")

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.build_lowrank_basis", _fake_basis)
    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)
    monkeypatch.setattr(f"{PKG.__name__}.kv_impl.build_projected_core_atoms_multi_streamed", _fake_atoms)
    monkeypatch.setattr(f"{PKG.__name__}.kv_impl.kv_impl_multi_streamed_stacked", _fail_stacked)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[
                _ArraySource(X[:, :5]),
                _ArraySource(X[:, 5:9]),
                _ArraySource(X[:, 9:]),
            ],
            call_width=4,
            keep_host_stats=True,
            precond_type="projected_core",
            precond_rank=3,
            verbose=False,
        )
    )
    try:
        fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
    finally:
        for st in fitter.streamers:
            st.close()

    assert calls["atoms"] == 1


def test_partitioned_projected_core_smoke_runs_with_real_reml():
    X = _make_non_degenerate_genotypes(n=18, m=12, seed=37)
    y = (
        0.4 * X[:, 1].astype(np.float32)
        - 0.2 * X[:, 8].astype(np.float32)
        + np.random.RandomState(17).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=[5, 4, 3],
            call_width=4,
            keep_host_stats=True,
            precond_type="projected_core",
            precond_rank=3,
            n_rand_vec=4,
            max_pcg_iters=40,
            minq_iter=2,
            slq_samples=4,
            slq_m=4,
            seed=19,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.asarray(y))
    finally:
        for st in fitter.streamers:
            st.close()

    assert res.var_components.shape == (4,)
    assert np.all(np.isfinite(np.asarray(res.var_components)))
    assert res.history


def test_partitioned_unsupported_preconditioner_type_rejected():
    with pytest.raises(ValueError, match="Only 'projected_core'"):
        InfinitesimalREMLFitter(
            FitConfig(
                sources=[object()],
                vc_block_sizes=[2, 2],
                precond_type="unsupported",
                precond_rank=8,
                verbose=False,
            )
        )


def test_partitioned_multi_grm_requires_single_dense_input():
    with pytest.raises(ValueError, match="exactly one dense source"):
        InfinitesimalREMLFitter(
            FitConfig(
                sources=[object(), object()],
                vc_block_sizes=[2, 2],
                precond_rank=0,
                verbose=False,
            )
        )
