from __future__ import annotations

import importlib
import json
import os
import shutil
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
SMILE = importlib.import_module(f"{PKG.__name__}.smile_block_w")
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")

GenoBlockStreamer = GENO_STREAM.GenoBlockStreamer
SmileBlockWeightedOperator = SMILE.SmileBlockWeightedOperator
SmileMultiBlockWeightedOperator = SMILE.SmileMultiBlockWeightedOperator
FitConfig = REML_MODEL.FitConfig
InfinitesimalREMLFitter = REML_MODEL.InfinitesimalREMLFitter


class _ArraySource:
    def __init__(self, block: np.ndarray, missing_val: int = -9):
        self._block = np.asarray(block, dtype=np.int8)
        self.n, self.m = self._block.shape
        self.missing_val = int(missing_val)

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        return np.asfortranarray(self._block[:, snp_start : snp_start + snp_count].T)

    def close(self):
        return None


def _make_non_degenerate_genotypes(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    for _ in range(1024):
        X = rng.randint(0, 3, size=(n, m), dtype=np.int8)
        if np.all(np.var(X.astype(np.float32), axis=0) > 0.0):
            return X
    raise RuntimeError("failed to build non-degenerate genotype matrix")


def test_identity_block_weights_match_existing_global_kv():
    X = _make_non_degenerate_genotypes(n=24, m=10, seed=201)
    block_sizes = [4, 6]
    V = jnp.asarray(
        np.random.RandomState(202).standard_normal((X.shape[0], 3)).astype(np.float32)
    )
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=block_sizes,
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [np.eye(4), np.eye(6)],
            normalization="kernel_trace",
            check_psd=True,
        )
        got = np.asarray(op.kv(V))
        ref = np.asarray(st.kv(V))
        assert np.allclose(got, ref, atol=2e-3, rtol=3e-4)
        assert np.allclose(
            np.asarray(jnp.sum(op.stacked_block_kv(V), axis=0)),
            got,
            atol=2e-3,
            rtol=3e-4,
        )
    finally:
        st.close()


def test_implicit_identity_operator_matches_existing_global_kv():
    X = _make_non_degenerate_genotypes(n=24, m=10, seed=216)
    V = jnp.asarray(
        np.random.RandomState(217).standard_normal((X.shape[0], 3)).astype(np.float32)
    )
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator.identity(
            st,
            block_size=4,
            normalization="kernel_trace",
        )
        assert op.n_blocks == 3
        assert all(block.is_identity for block in op.blocks)
        assert all(block.matrix is None for block in op.blocks)
        got = np.asarray(op.kv(V))
        ref = np.asarray(st.kv(V))
        assert np.allclose(got, ref, atol=2e-3, rtol=3e-4)
    finally:
        st.close()


def test_weighted_blocks_match_explicit_matrix_reference():
    X = _make_non_degenerate_genotypes(n=28, m=9, seed=203)
    rng = np.random.RandomState(204)
    A0 = rng.standard_normal((4, 4))
    A1 = rng.standard_normal((5, 5))
    W0 = A0 @ A0.T + 0.05 * np.eye(4)
    W1 = A1 @ A1.T + 0.05 * np.eye(5)
    V = rng.standard_normal((X.shape[0], 2)).astype(np.float32)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_block_sizes=[4, 5],
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
        )
        got = np.asarray(op.kv(jnp.asarray(V)))

        ref = jnp.zeros_like(jnp.asarray(V))
        raw_trace_per_sample = 0.0
        for block in op.blocks:
            start = block.start
            W = jnp.asarray(block.matrix, dtype=jnp.float32)
            idx = np.arange(start, start + W.shape[0], dtype=np.int64)
            Z = jnp.asarray(st.extract_standardized_columns(idx), dtype=jnp.float32)
            ref = ref + Z @ (W @ (Z.T @ jnp.asarray(V)))
            raw_trace_per_sample += float(np.sum((np.asarray(Z) @ np.asarray(W)) * np.asarray(Z)) / X.shape[0])
        ref = ref / jnp.asarray(op.normalizer, dtype=jnp.float32)
        assert op.normalizer == pytest.approx(raw_trace_per_sample, rel=1e-6)
        assert np.allclose(got, np.asarray(ref), atol=3e-3, rtol=5e-4)
    finally:
        st.close()


def test_global_trace_normalization_sets_average_diagonal_to_one():
    X = _make_non_degenerate_genotypes(n=18, m=7, seed=208)
    rng = np.random.RandomState(209)
    A0 = rng.standard_normal((3, 3))
    A1 = rng.standard_normal((4, 4))
    W0 = A0 @ A0.T + 0.1 * np.eye(3)
    W1 = A1 @ A1.T + 0.1 * np.eye(4)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=[3, 4],
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
        )
        trace_value = 0.0
        for block in op.blocks:
            idx = np.arange(block.start, block.stop, dtype=np.int64)
            Z = np.asarray(st.extract_standardized_columns(idx), dtype=np.float64)
            W = np.asarray(block.matrix, dtype=np.float64)
            trace_value += float(np.sum((Z @ W) * Z))
        assert trace_value / op.normalizer == pytest.approx(
            float(X.shape[0]), rel=2e-6, abs=2e-6
        )
        diag = np.asarray(op.diag())
        assert diag.shape == (X.shape[0],)
        assert float(np.sum(diag)) == pytest.approx(float(X.shape[0]), rel=2e-6, abs=2e-6)
    finally:
        st.close()


def test_mean_diag_mode_keeps_exact_normalizer_but_returns_scalar_diag():
    X = _make_non_degenerate_genotypes(n=18, m=7, seed=239)
    rng = np.random.RandomState(240)
    A0 = rng.standard_normal((3, 3))
    A1 = rng.standard_normal((4, 4))
    W0 = A0 @ A0.T + 0.1 * np.eye(3)
    W1 = A1 @ A1.T + 0.1 * np.eye(4)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=[3, 4],
        keep_host_stats=True,
    )
    try:
        full = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
            diag_mode="full",
        )
        mean = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
            diag_mode="mean",
        )
        assert mean.normalizer == pytest.approx(full.normalizer, rel=1e-6)
        assert np.asarray(mean.diag()).shape == ()
        assert float(np.asarray(mean.diag())) == pytest.approx(
            float(np.mean(np.asarray(full.diag()))), rel=2e-6, abs=2e-6
        )
        V = jnp.asarray(rng.standard_normal((X.shape[0], 2)).astype(np.float32))
        assert np.allclose(np.asarray(mean.kv(V)), np.asarray(full.kv(V)), atol=3e-3, rtol=5e-4)
    finally:
        st.close()


def test_weighted_hv_uses_single_genetic_variance_component():
    X = _make_non_degenerate_genotypes(n=20, m=7, seed=205)
    V = jnp.asarray(
        np.random.RandomState(206).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=[3, 4],
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [np.eye(3), np.eye(4)],
            normalization="kernel_trace",
            check_psd=True,
        )
        theta_g = jnp.asarray(0.25, dtype=jnp.float32)
        theta_e = jnp.asarray(0.35, dtype=jnp.float32)
        got = np.asarray(op.weighted_hv(theta_g, theta_e, V))
        ref = theta_e * np.asarray(V) + 0.25 * np.asarray(op.kv(V))
        assert np.allclose(got, ref, atol=2e-3, rtol=3e-4)
        with pytest.raises(ValueError, match="scalar or a length-one array"):
            op.weighted_hv(jnp.asarray([0.25, 0.4], dtype=jnp.float32), theta_e, V)
    finally:
        st.close()


def test_multi_grm_groups_sum_blocks_inside_each_grm():
    X = _make_non_degenerate_genotypes(n=22, m=8, seed=222)
    rng = np.random.RandomState(223)
    mats = []
    for size in (2, 3, 3):
        A = rng.standard_normal((size, size))
        mats.append(A @ A.T + 0.1 * np.eye(size))
    V = jnp.asarray(rng.standard_normal((X.shape[0], 2)).astype(np.float32))
    theta_g = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    theta_e = jnp.asarray(0.4, dtype=jnp.float32)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        keep_host_stats=True,
    )
    try:
        multi = SmileMultiBlockWeightedOperator.from_weight_matrix_groups(
            st,
            [[mats[0], mats[1]], [mats[2]]],
            normalization="kernel_trace",
            check_psd=True,
        )
        assert multi.n_grm == 2
        assert [block.start for block in multi.operators[0].blocks] == [0, 2]
        assert [block.start for block in multi.operators[1].blocks] == [5]

        refs = []
        for op in multi.operators:
            ref = jnp.zeros_like(V)
            raw_trace_per_sample = 0.0
            for block in op.blocks:
                idx = np.arange(block.start, block.stop, dtype=np.int64)
                Z = jnp.asarray(st.extract_standardized_columns(idx), dtype=jnp.float32)
                W = jnp.asarray(block.matrix, dtype=jnp.float32)
                ref = ref + Z @ (W @ (Z.T @ V))
                raw_trace_per_sample += float(
                    np.sum((np.asarray(Z) @ np.asarray(W)) * np.asarray(Z)) / X.shape[0]
                )
            assert op.normalizer == pytest.approx(raw_trace_per_sample, rel=1e-6)
            refs.append(ref / jnp.asarray(op.normalizer, dtype=jnp.float32))

        stack = np.asarray(multi.stacked_kv(V))
        assert stack.shape == (2, X.shape[0], 2)
        assert np.allclose(stack[0], np.asarray(refs[0]), atol=3e-3, rtol=5e-4)
        assert np.allclose(stack[1], np.asarray(refs[1]), atol=3e-3, rtol=5e-4)

        got_hv = np.asarray(multi.weighted_hv(theta_g, theta_e, V))
        ref_hv = theta_e * np.asarray(V) + theta_g[0] * np.asarray(refs[0]) + theta_g[1] * np.asarray(refs[1])
        assert np.allclose(got_hv, np.asarray(ref_hv), atol=4e-3, rtol=6e-4)
    finally:
        st.close()


def test_snp_effects_include_block_weight_matrix():
    X = _make_non_degenerate_genotypes(n=16, m=5, seed=212)
    rng = np.random.RandomState(213)
    A0 = rng.standard_normal((2, 2))
    A1 = rng.standard_normal((3, 3))
    W0 = A0 @ A0.T + 0.1 * np.eye(2)
    W1 = A1 @ A1.T + 0.1 * np.eye(3)
    alpha = jnp.asarray(rng.standard_normal((X.shape[0],)).astype(np.float32))
    theta_g = jnp.asarray([0.3], dtype=jnp.float32)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
        )
        got = np.asarray(op.snp_effects(alpha, theta_g))
        ref_parts = []
        for block in op.blocks:
            idx = np.arange(block.start, block.stop, dtype=np.int64)
            Z = jnp.asarray(st.extract_standardized_columns(idx), dtype=jnp.float32)
            W = jnp.asarray(block.matrix, dtype=jnp.float32)
            ref_parts.append(theta_g[0] * (W @ (Z.T @ alpha)) / op.normalizer)
        ref = np.asarray(jnp.concatenate(ref_parts, axis=0))
        assert np.allclose(got, ref, atol=2e-3, rtol=5e-4)
        assert np.allclose(
            np.asarray(st.extract_standardized_columns(np.arange(st.m)) @ got),
            np.asarray(theta_g[0] * op.kv(alpha)),
            atol=3e-3,
            rtol=5e-4,
        )
    finally:
        st.close()


def test_fitter_assembles_smile_single_kernel_reml_operator(monkeypatch):
    X = _make_non_degenerate_genotypes(n=18, m=7, seed=210)
    V = jnp.asarray(
        np.random.RandomState(211).standard_normal((X.shape[0], 2)).astype(np.float32)
    )

    def _fake_fit_reml(*, y, K_mvs, diag_list, weighted_hv=None, stacked_kv=None, **_kwargs):
        del y, _kwargs
        assert len(K_mvs) == 1
        assert len(diag_list) == 1
        assert weighted_hv is not None
        assert stacked_kv is not None

        K_V = K_mvs[0](V)
        stack = stacked_kv(V)
        assert stack.shape == (1, X.shape[0], 2)
        assert np.allclose(np.asarray(stack[0]), np.asarray(K_V), atol=2e-3, rtol=3e-4)

        theta_g = jnp.asarray([0.25], dtype=jnp.float32)
        theta_e = jnp.asarray(0.35, dtype=jnp.float32)
        got_hv = weighted_hv(theta_g, theta_e, V)
        ref_hv = theta_e * V + theta_g[0] * K_V
        assert np.allclose(np.asarray(got_hv), np.asarray(ref_hv), atol=2e-3, rtol=3e-4)

        diag = np.asarray(diag_list[0])
        assert diag.shape == ()
        assert float(diag) == pytest.approx(1.0, rel=2e-6, abs=2e-6)
        return jnp.asarray([0.2, 0.8], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_weight_matrices=[np.eye(3), np.eye(4)],
        call_width=3,
        keep_host_stats=False,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
        assert np.allclose(np.asarray(res.var_components), [0.2, 0.8])
    finally:
        fitter.close()


def test_fitter_assembles_smile_multi_grm_groups(monkeypatch):
    X = _make_non_degenerate_genotypes(n=18, m=7, seed=224)
    V = jnp.asarray(
        np.random.RandomState(225).standard_normal((X.shape[0], 2)).astype(np.float32)
    )

    def _fake_fit_reml(*, y, K_mvs, diag_list, weighted_hv=None, stacked_kv=None, **_kwargs):
        del y, _kwargs
        assert len(K_mvs) == 2
        assert len(diag_list) == 2
        assert weighted_hv is not None
        assert stacked_kv is not None

        K0 = K_mvs[0](V)
        K1 = K_mvs[1](V)
        stack = stacked_kv(V)
        assert stack.shape == (2, X.shape[0], 2)
        assert np.allclose(np.asarray(stack[0]), np.asarray(K0), atol=2e-3, rtol=3e-4)
        assert np.allclose(np.asarray(stack[1]), np.asarray(K1), atol=2e-3, rtol=3e-4)

        theta_g = jnp.asarray([0.25, 0.15], dtype=jnp.float32)
        theta_e = jnp.asarray(0.35, dtype=jnp.float32)
        got_hv = weighted_hv(theta_g, theta_e, V)
        ref_hv = theta_e * V + theta_g[0] * K0 + theta_g[1] * K1
        assert np.allclose(np.asarray(got_hv), np.asarray(ref_hv), atol=2e-3, rtol=3e-4)
        return jnp.asarray([0.2, 0.1, 0.7], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_weight_matrix_groups=[[np.eye(2), np.eye(2)], [np.eye(3)]],
        smile_normalization="kernel_trace",
        call_width=3,
        keep_host_stats=False,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
        assert np.allclose(np.asarray(res.var_components), [0.2, 0.1, 0.7])
    finally:
        fitter.close()


def test_fitter_matrix_groups_compute_exact_trace_normalizer(monkeypatch):
    X = _make_non_degenerate_genotypes(n=18, m=7, seed=236)
    V = jnp.asarray(
        np.random.RandomState(237).standard_normal((X.shape[0], 2)).astype(np.float32)
    )

    def _fake_fit_reml(*, y, K_mvs, diag_list, weighted_hv=None, **_kwargs):
        del y, weighted_hv, _kwargs
        assert len(K_mvs) == 2
        assert len(diag_list) == 2
        assert np.all(np.isfinite(np.asarray(diag_list[0])))
        assert np.all(np.isfinite(np.asarray(diag_list[1])))
        for mv in K_mvs:
            got = np.asarray(mv(V))
            assert got.shape == (X.shape[0], 2)
            assert np.all(np.isfinite(got))
        return jnp.asarray([0.2, 0.1, 0.7], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_weight_matrix_groups=[[np.eye(2), np.eye(2)], [np.eye(3)]],
        smile_normalization="kernel_trace",
        call_width=3,
        keep_host_stats=False,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        expected = []
        st = fitter.streamers[0]
        for op in fitter._smile_operators:
            total = 0.0
            for block in op.blocks:
                idx = np.arange(block.start, block.stop, dtype=np.int64)
                Z = np.asarray(st.extract_standardized_columns(idx), dtype=np.float32)
                W = np.asarray(block.matrix, dtype=np.float32)
                total += float(np.sum((Z @ W) * Z, dtype=np.float64) / X.shape[0])
            expected.append(total)
        assert [op.normalizer for op in fitter._smile_operators] == pytest.approx(expected, rel=1e-6)
        res = fitter.fit_infinitesimal(jnp.asarray(X[:, 0], dtype=jnp.float32))
        assert np.allclose(np.asarray(res.var_components), [0.2, 0.1, 0.7])
    finally:
        fitter.close()


def test_npy_weight_loader_uses_memmap_and_shape_metadata(tmp_path):
    path = tmp_path / "W.npy"
    W = np.eye(4, dtype=np.float32)
    np.save(path, W)
    loaded = SMILE.load_weight_matrix(path)
    try:
        assert isinstance(loaded, np.memmap)
        assert SMILE.load_weight_matrix_shape(path) == (4, 4)
    finally:
        del loaded

    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps({"width": 4}) + "\n")
    assert SMILE.load_weight_matrix_shape(path) == (4, 4)


def test_file_groups_compute_exact_trace_normalizer(tmp_path):
    X = _make_non_degenerate_genotypes(n=18, m=5, seed=238)
    paths = []
    for idx, width in enumerate((2, 3)):
        path = tmp_path / f"W{idx}.npy"
        np.save(path, np.eye(width, dtype=np.float32))
        path.with_suffix(".json").write_text(json.dumps({"width": width}) + "\n")
        paths.append(path)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        keep_host_stats=True,
    )
    try:
        multi = SmileMultiBlockWeightedOperator.from_weight_file_groups(
            st,
            [[paths[0]], [paths[1]]],
            normalization="kernel_trace",
            check_psd=False,
        )
        assert multi.n_grm == 2
        expected = []
        for op in multi.operators:
            ref_diag = np.zeros((X.shape[0],), dtype=np.float64)
            for block in op.blocks:
                idx = np.arange(block.start, block.stop, dtype=np.int64)
                Z = np.asarray(st.extract_standardized_columns(idx), dtype=np.float32)
                W = np.asarray(block.matrix, dtype=np.float32)
                ref_diag += np.sum((Z @ W) * Z, axis=1, dtype=np.float64)
            expected.append(float(np.sum(ref_diag) / X.shape[0]))
            ref_diag = ref_diag / float(op.normalizer)
            assert np.allclose(np.asarray(op.diag()), ref_diag, atol=2e-5, rtol=2e-5)
        assert [op.normalizer for op in multi.operators] == pytest.approx(expected, rel=1e-6)
    finally:
        st.close()


def test_fitter_runs_smile_end_to_end_variance_estimation():
    X = _make_non_degenerate_genotypes(n=14, m=6, seed=214)
    y = jnp.asarray(
        np.random.RandomState(215).standard_normal((X.shape[0],)).astype(np.float32)
    )
    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_weight_matrices=[np.eye(2), np.eye(4)],
        call_width=3,
        precond_rank=0,
        n_rand_vec=2,
        slq_samples=2,
        slq_m=2,
        minq_iter=2,
        max_pcg_iters=30,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(y, h2_init=0.4)
        vc = np.asarray(res.var_components)
        assert vc.shape == (2,)
        assert np.all(np.isfinite(vc))
        assert np.all(vc >= 0.0)
    finally:
        fitter.close()


def test_fitter_runs_smile_multi_grm_end_to_end_variance_estimation():
    X = _make_non_degenerate_genotypes(n=14, m=7, seed=226)
    y = jnp.asarray(
        np.random.RandomState(227).standard_normal((X.shape[0],)).astype(np.float32)
    )
    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_weight_matrix_groups=[[np.eye(2), np.eye(2)], [np.eye(3)]],
        smile_normalization="kernel_trace",
        call_width=3,
        precond_rank=0,
        n_rand_vec=2,
        slq_samples=2,
        slq_m=2,
        minq_iter=2,
        max_pcg_iters=30,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(y, h2_init=0.4)
        vc = np.asarray(res.var_components)
        assert vc.shape == (3,)
        assert np.all(np.isfinite(vc))
        assert np.all(vc >= 0.0)
    finally:
        fitter.close()


def test_fitter_runs_implicit_smile_identity_end_to_end():
    X = _make_non_degenerate_genotypes(n=14, m=6, seed=218)
    y = jnp.asarray(
        np.random.RandomState(219).standard_normal((X.shape[0],)).astype(np.float32)
    )
    cfg = FitConfig(
        sources=[_ArraySource(X)],
        smile_identity=True,
        smile_identity_block_size=3,
        call_width=3,
        precond_rank=0,
        n_rand_vec=2,
        slq_samples=2,
        slq_m=2,
        minq_iter=2,
        max_pcg_iters=30,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(y, h2_init=0.4)
        vc = np.asarray(res.var_components)
        assert vc.shape == (2,)
        assert np.all(np.isfinite(vc))
        assert np.all(vc >= 0.0)
    finally:
        fitter.close()


def test_rejects_non_psd_weight_matrix():
    X = _make_non_degenerate_genotypes(n=12, m=2, seed=207)
    st = GenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        with pytest.raises(ValueError, match="positive semidefinite"):
            SmileBlockWeightedOperator(
                st,
                [np.asarray([[1.0, 0.0], [0.0, -0.1]])],
                check_psd=True,
            )
    finally:
        st.close()


def test_load_real_rds_ld_weight_matrix_if_available():
    path = os.environ.get("GPU_REML_SMILE_RDS_FIXTURE")
    if not path:
        pytest.skip("set GPU_REML_SMILE_RDS_FIXTURE to validate a local LD RDS matrix")
    if not os.path.exists(path):
        pytest.skip("configured LD RDS fixture is not available")
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available")

    W = SMILE.load_rds_matrix(path)
    assert W.shape == (1158, 1158)
    assert np.all(np.isfinite(W))
    assert np.max(np.abs(W - W.T)) == pytest.approx(0.0, abs=1e-12)
    eig_min = float(np.linalg.eigvalsh(W)[0])
    assert eig_min > 0.0
