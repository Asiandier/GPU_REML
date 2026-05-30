from __future__ import annotations

import importlib
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")
PRED_IO = importlib.import_module(f"{PKG.__name__}.prediction_io")
DATA_UTILS = importlib.import_module(f"{PKG.__name__}.data_utils")

FitConfig = REML_MODEL.FitConfig
InfinitesimalREMLFitter = REML_MODEL.InfinitesimalREMLFitter
write_prediction_outputs = PRED_IO.write_prediction_outputs
load_pheno_covar_aligned_with_transform = DATA_UTILS.load_pheno_covar_aligned_with_transform
load_covar_aligned = DATA_UTILS.load_covar_aligned


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


def _standardize_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y32 = np.asarray(y, dtype=np.float32).reshape(-1)
    y_mean = float(y32.mean())
    y_scale = float(y32.std() + np.float32(1e-6))
    y_std = ((y32 - y_mean) / y_scale).astype(np.float64)
    return y_std, y_mean, y_scale


def _standardize_genotypes_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_f = np.asarray(X_train, dtype=np.float64)
    test_f = np.asarray(X_test, dtype=np.float64)
    mean = train_f.mean(axis=0)
    var = np.maximum(train_f.var(axis=0), 0.0)
    inv = np.where(var > 0.0, 1.0 / np.sqrt(np.maximum(var, 1e-6)), 0.0)
    Z_train = (train_f - mean) * inv
    Z_test = (test_f - mean) * inv
    return Z_train, Z_test


def _reference_prediction(
    Z_train_list: list[np.ndarray],
    Z_test_list: list[np.ndarray],
    theta_g: np.ndarray,
    theta_e: float,
    y_train: np.ndarray,
    covar_train: np.ndarray | None,
    covar_test: np.ndarray | None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    y_std, y_mean, y_scale = _standardize_y(y_train)
    n_train = y_std.size
    V = theta_e * np.eye(n_train, dtype=np.float64)
    for theta, Z in zip(theta_g.tolist(), Z_train_list):
        eff = float(Z.shape[1])
        if eff > 0.0:
            V = V + float(theta) * (Z @ Z.T) / eff

    alpha_rhs = np.linalg.solve(V, y_std)
    if covar_train is not None and covar_train.size > 0:
        X = np.asarray(covar_train, dtype=np.float64)
        VinvX = np.linalg.solve(V, X)
        beta = np.linalg.solve(
            X.T @ VinvX + 1e-10 * np.eye(X.shape[1], dtype=np.float64),
            X.T @ alpha_rhs,
        )
        alpha = alpha_rhs - VinvX @ beta
    else:
        beta = np.empty((0,), dtype=np.float64)
        alpha = alpha_rhs

    b_list: list[np.ndarray] = []
    g_test_list: list[np.ndarray] = []
    for theta, Z_train, Z_test in zip(theta_g.tolist(), Z_train_list, Z_test_list):
        eff = float(Z_train.shape[1])
        if eff > 0.0:
            b = float(theta) * (Z_train.T @ alpha) / eff
            g_test = Z_test @ b
        else:
            b = np.zeros((Z_train.shape[1],), dtype=np.float64)
            g_test = np.zeros((Z_test.shape[0],), dtype=np.float64)
        b_list.append(b)
        g_test_list.append(g_test)

    fixed_test = (
        np.asarray(covar_test, dtype=np.float64) @ beta
        if covar_test is not None and beta.size > 0
        else np.zeros((Z_test_list[0].shape[0],), dtype=np.float64)
    )
    g_total = np.sum(np.stack(g_test_list, axis=0), axis=0)
    y_pred_std = fixed_test + g_total
    y_pred = y_mean + y_scale * y_pred_std
    return beta, g_test_list, fixed_test, y_pred_std, y_pred


def test_predict_matches_dense_reference_single_grm():
    X_train = _make_non_degenerate_genotypes(n=16, m=7, seed=11)
    X_test = _make_non_degenerate_genotypes(n=9, m=7, seed=12)
    y_train = (
        0.35 * X_train[:, 1].astype(np.float32)
        - 0.2 * X_train[:, 5].astype(np.float32)
        + np.random.RandomState(13).standard_normal(X_train.shape[0]).astype(np.float32) * 0.1
    )
    covar_train = np.c_[
        np.ones((X_train.shape[0],), dtype=np.float32),
        np.linspace(-1.0, 1.0, X_train.shape[0], dtype=np.float32),
    ]
    covar_test = np.c_[
        np.ones((X_test.shape[0],), dtype=np.float32),
        np.linspace(0.2, 1.4, X_test.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.28, 0.72], dtype=np.float32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_train)],
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            effect_pcg_tol=1e-6,
            verbose=False,
        )
    )
    test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = train_fitter.estimate_effects(
            jnp.asarray(y_train),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar_train),
        )
        preds = train_fitter.predict(
            effects,
            test_fitter=test_fitter,
            test_covar=jnp.asarray(covar_test),
        )
    finally:
        for st in train_fitter.streamers:
            st.close()
        for st in test_fitter.streamers:
            st.close()

    Z_train, Z_test = _standardize_genotypes_train_test(X_train, X_test)
    beta_ref, g_ref, fixed_ref, y_std_ref, y_ref = _reference_prediction(
        [Z_train], [Z_test], theta[:-1], float(theta[-1]), y_train, covar_train, covar_test
    )

    np.testing.assert_allclose(np.asarray(preds.fixed_effect), fixed_ref, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(
        np.asarray(preds.random_effect_components[0]), g_ref[0], atol=2e-3, rtol=3e-3
    )
    np.testing.assert_allclose(np.asarray(preds.random_effect), g_ref[0], atol=2e-3, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred_std), y_std_ref, atol=2e-3, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred), y_ref, atol=2e-3, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(effects.fixed_effects), beta_ref, atol=2e-3, rtol=2e-3)


def test_predict_matches_dense_reference_partitioned_multi_grm():
    X_train = _make_non_degenerate_genotypes(n=18, m=9, seed=21)
    X_test = _make_non_degenerate_genotypes(n=10, m=9, seed=22)
    block_sizes = [4, 3, 2]
    y_train = (
        0.4 * X_train[:, 0].astype(np.float32)
        - 0.3 * X_train[:, 7].astype(np.float32)
        + np.random.RandomState(23).standard_normal(X_train.shape[0]).astype(np.float32) * 0.1
    )
    covar_train = np.c_[
        np.ones((X_train.shape[0],), dtype=np.float32),
        np.linspace(-0.5, 0.8, X_train.shape[0], dtype=np.float32),
    ]
    covar_test = np.c_[
        np.ones((X_test.shape[0],), dtype=np.float32),
        np.linspace(0.1, 1.1, X_test.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.16, 0.07, 0.11, 0.66], dtype=np.float32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_train)],
            vc_block_sizes=block_sizes,
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            effect_pcg_tol=1e-6,
            verbose=False,
        )
    )
    test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            vc_block_sizes=block_sizes,
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = train_fitter.estimate_effects(
            jnp.asarray(y_train),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar_train),
        )
        preds = train_fitter.predict(
            effects,
            test_fitter=test_fitter,
            test_covar=jnp.asarray(covar_test),
        )
    finally:
        for st in train_fitter.streamers:
            st.close()
        for st in test_fitter.streamers:
            st.close()

    offsets = np.r_[0, np.cumsum(block_sizes)]
    Z_train_full, Z_test_full = _standardize_genotypes_train_test(X_train, X_test)
    Z_train_list = [Z_train_full[:, offsets[g] : offsets[g + 1]] for g in range(len(block_sizes))]
    Z_test_list = [Z_test_full[:, offsets[g] : offsets[g + 1]] for g in range(len(block_sizes))]
    _, g_ref, fixed_ref, y_std_ref, y_ref = _reference_prediction(
        Z_train_list, Z_test_list, theta[:-1], float(theta[-1]), y_train, covar_train, covar_test
    )

    np.testing.assert_allclose(np.asarray(preds.fixed_effect), fixed_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(preds.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred_std), y_std_ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred), y_ref, atol=3e-3, rtol=4e-3)


def test_predict_matches_dense_reference_arbitrary_grouped_multi_grm():
    X_train = _make_non_degenerate_genotypes(n=18, m=9, seed=321)
    X_test = _make_non_degenerate_genotypes(n=10, m=9, seed=322)
    component_variant_indices = [
        [0, 3, 4],
        [1, 2],
        [5, 6, 7, 8],
    ]
    y_train = (
        0.4 * X_train[:, 0].astype(np.float32)
        - 0.3 * X_train[:, 7].astype(np.float32)
        + np.random.RandomState(323).standard_normal(X_train.shape[0]).astype(np.float32) * 0.1
    )
    covar_train = np.c_[
        np.ones((X_train.shape[0],), dtype=np.float32),
        np.linspace(-0.5, 0.8, X_train.shape[0], dtype=np.float32),
    ]
    covar_test = np.c_[
        np.ones((X_test.shape[0],), dtype=np.float32),
        np.linspace(0.1, 1.1, X_test.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.16, 0.07, 0.11, 0.66], dtype=np.float32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_train)],
            component_variant_indices=component_variant_indices,
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            effect_pcg_tol=1e-6,
            verbose=False,
        )
    )
    test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            component_variant_indices=component_variant_indices,
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = train_fitter.estimate_effects(
            jnp.asarray(y_train),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar_train),
        )
        preds = train_fitter.predict(
            effects,
            test_fitter=test_fitter,
            test_covar=jnp.asarray(covar_test),
        )
    finally:
        for st in train_fitter.streamers:
            st.close()
        for st in test_fitter.streamers:
            st.close()

    Z_train_full, Z_test_full = _standardize_genotypes_train_test(X_train, X_test)
    Z_train_list = [
        Z_train_full[:, np.asarray(group, dtype=np.int64)]
        for group in component_variant_indices
    ]
    Z_test_list = [
        Z_test_full[:, np.asarray(group, dtype=np.int64)]
        for group in component_variant_indices
    ]
    _, g_ref, fixed_ref, y_std_ref, y_ref = _reference_prediction(
        Z_train_list, Z_test_list, theta[:-1], float(theta[-1]), y_train, covar_train, covar_test
    )

    np.testing.assert_allclose(np.asarray(preds.fixed_effect), fixed_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(preds.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(
        np.asarray(preds.random_effect),
        np.sum(np.stack(g_ref, axis=0), axis=0),
        atol=3e-3,
        rtol=4e-3,
    )
    np.testing.assert_allclose(np.asarray(preds.y_pred_std), y_std_ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred), y_ref, atol=3e-3, rtol=4e-3)


def test_predict_matches_dense_reference_separate_multi_grm():
    X_train = _make_non_degenerate_genotypes(n=18, m=9, seed=51)
    X_test = _make_non_degenerate_genotypes(n=10, m=9, seed=52)
    block_sizes = [4, 3, 2]
    offsets = np.r_[0, np.cumsum(block_sizes)]
    y_train = (
        0.38 * X_train[:, 0].astype(np.float32)
        - 0.27 * X_train[:, 7].astype(np.float32)
        + np.random.RandomState(53).standard_normal(X_train.shape[0]).astype(np.float32) * 0.1
    )
    covar_train = np.c_[
        np.ones((X_train.shape[0],), dtype=np.float32),
        np.linspace(-0.7, 0.9, X_train.shape[0], dtype=np.float32),
    ]
    covar_test = np.c_[
        np.ones((X_test.shape[0],), dtype=np.float32),
        np.linspace(0.0, 1.2, X_test.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.14, 0.09, 0.12, 0.65], dtype=np.float32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[
                _ArraySource(X_train[:, offsets[g] : offsets[g + 1]])
                for g in range(len(block_sizes))
            ],
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            effect_pcg_tol=1e-6,
            verbose=False,
        )
    )
    test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[
                _ArraySource(X_test[:, offsets[g] : offsets[g + 1]])
                for g in range(len(block_sizes))
            ],
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = train_fitter.estimate_effects(
            jnp.asarray(y_train),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar_train),
        )
        preds = train_fitter.predict(
            effects,
            test_fitter=test_fitter,
            test_covar=jnp.asarray(covar_test),
        )
    finally:
        for st in train_fitter.streamers:
            st.close()
        for st in test_fitter.streamers:
            st.close()

    Z_train_full, Z_test_full = _standardize_genotypes_train_test(X_train, X_test)
    Z_train_list = [Z_train_full[:, offsets[g] : offsets[g + 1]] for g in range(len(block_sizes))]
    Z_test_list = [Z_test_full[:, offsets[g] : offsets[g + 1]] for g in range(len(block_sizes))]
    _, g_ref, fixed_ref, y_std_ref, y_ref = _reference_prediction(
        Z_train_list, Z_test_list, theta[:-1], float(theta[-1]), y_train, covar_train, covar_test
    )

    np.testing.assert_allclose(np.asarray(preds.fixed_effect), fixed_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(preds.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.random_effect), np.sum(np.stack(g_ref, axis=0), axis=0), atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred_std), y_std_ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred), y_ref, atol=3e-3, rtol=4e-3)


def test_prediction_streamer_build_can_reuse_training_standardization():
    X_train = _make_non_degenerate_genotypes(n=12, m=8, seed=31)
    X_test = _make_non_degenerate_genotypes(n=7, m=8, seed=32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_train)],
            call_width=4,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    plain_test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            call_width=4,
            keep_host_stats=False,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    override_test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            standardization_overrides=[
                (
                    np.asarray(train_fitter.streamers[0]._means_host, dtype=np.float32),
                    np.asarray(train_fitter.streamers[0]._inv_sds_host, dtype=np.float32),
                )
            ],
            call_width=4,
            keep_host_stats=False,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        train_means = np.asarray(jax.device_get(train_fitter.streamers[0]._means_by_call))
        plain_means = np.asarray(jax.device_get(plain_test_fitter.streamers[0]._means_by_call))
        override_means = np.asarray(jax.device_get(override_test_fitter.streamers[0]._means_by_call))
        train_inv = np.asarray(jax.device_get(train_fitter.streamers[0]._inv_by_call))
        override_inv = np.asarray(jax.device_get(override_test_fitter.streamers[0]._inv_by_call))

        assert not np.allclose(plain_means, train_means)
        np.testing.assert_allclose(override_means, train_means, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(override_inv, train_inv, atol=0.0, rtol=0.0)
        assert override_test_fitter.streamers[0]._means_host is None
        assert override_test_fitter.streamers[0]._inv_sds_host is None
    finally:
        for fitter in (train_fitter, plain_test_fitter, override_test_fitter):
            for st in fitter.streamers:
                st.close()


def test_predict_handles_tail_call_narrower_than_max_unpack_width():
    X_train = _make_non_degenerate_genotypes(n=14, m=17, seed=41)
    X_test = _make_non_degenerate_genotypes(n=8, m=17, seed=42)
    y_train = (
        0.25 * X_train[:, 2].astype(np.float32)
        - 0.18 * X_train[:, 12].astype(np.float32)
        + np.random.RandomState(43).standard_normal(X_train.shape[0]).astype(np.float32) * 0.08
    )
    covar_train = np.c_[
        np.ones((X_train.shape[0],), dtype=np.float32),
        np.linspace(-1.2, 0.7, X_train.shape[0], dtype=np.float32),
    ]
    covar_test = np.c_[
        np.ones((X_test.shape[0],), dtype=np.float32),
        np.linspace(0.0, 1.0, X_test.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.22, 0.78], dtype=np.float32)

    train_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_train)],
            call_width=9,
            keep_host_stats=True,
            cpu_threads=4,
            precond_rank=0,
            effect_pcg_tol=1e-6,
            verbose=False,
        )
    )
    test_fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X_test)],
            standardization_overrides=[
                (
                    np.asarray(train_fitter.streamers[0]._means_host, dtype=np.float32),
                    np.asarray(train_fitter.streamers[0]._inv_sds_host, dtype=np.float32),
                )
            ],
            call_width=9,
            keep_host_stats=False,
            cpu_threads=4,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = train_fitter.estimate_effects(
            jnp.asarray(y_train),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar_train),
        )
        preds = train_fitter.predict(
            effects,
            test_fitter=test_fitter,
            test_covar=jnp.asarray(covar_test),
        )
    finally:
        for fitter in (train_fitter, test_fitter):
            for st in fitter.streamers:
                st.close()

    Z_train, Z_test = _standardize_genotypes_train_test(X_train, X_test)
    _, g_ref, fixed_ref, y_std_ref, y_ref = _reference_prediction(
        [Z_train], [Z_test], theta[:-1], float(theta[-1]), y_train, covar_train, covar_test
    )
    np.testing.assert_allclose(np.asarray(preds.fixed_effect), fixed_ref, atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(preds.random_effect), g_ref[0], atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred_std), y_std_ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(preds.y_pred), y_ref, atol=3e-3, rtol=4e-3)


def test_write_prediction_outputs_writes_expected_files(tmp_path):
    preds = REML_MODEL.PredictionEstimates(
        fixed_effect=jnp.asarray([0.1, -0.2], dtype=jnp.float32),
        random_effect=jnp.asarray([0.3, -0.4], dtype=jnp.float32),
        random_effect_components=(
            jnp.asarray([0.05, -0.1], dtype=jnp.float32),
            jnp.asarray([0.25, -0.3], dtype=jnp.float32),
        ),
        y_pred_std=jnp.asarray([0.4, -0.6], dtype=jnp.float32),
        y_pred=jnp.asarray([1.4, 0.4], dtype=jnp.float32),
    )

    out_prefix = str(tmp_path / "pred" / "demo")
    paths = write_prediction_outputs(
        out_prefix=out_prefix,
        predictions=preds,
        sample_ids=["iid1", "iid2"],
    )

    txt = (tmp_path / "pred" / "demo.prediction.tsv").read_text()
    assert "y_pred_std" in txt
    assert "random_component_001" in txt
    meta = json.loads((tmp_path / "pred" / "demo.prediction_metadata.json").read_text())
    assert meta["n_samples"] == 2
    assert meta["n_components"] == 2
    assert os.path.exists(paths["prediction"])


def test_prediction_covar_uses_training_standardization(tmp_path):
    train_fam = tmp_path / "train.fam"
    train_pheno = tmp_path / "train.pheno"
    train_covar = tmp_path / "train.qcovar"
    test_fam = tmp_path / "test.fam"
    test_covar = tmp_path / "test.qcovar"

    train_fam.write_text(
        "f1 i1 0 0 0 -9\n"
        "f2 i2 0 0 0 -9\n"
        "f3 i3 0 0 0 -9\n"
    )
    train_pheno.write_text(
        "f1 i1 1.0\n"
        "f2 i2 2.0\n"
        "f3 i3 3.0\n"
    )
    train_covar.write_text(
        "f1 i1 10 1\n"
        "f2 i2 20 1\n"
        "f3 i3 40 1\n"
    )
    test_fam.write_text(
        "f4 j1 0 0 0 -9\n"
        "f5 j2 0 0 0 -9\n"
    )
    test_covar.write_text(
        "f4 j1 25 1\n"
        "f5 j2 40 1\n"
    )

    _, X_train, _, _, transform = load_pheno_covar_aligned_with_transform(
        str(train_fam), str(train_pheno), str(train_covar), add_intercept=True
    )
    X_test, keep_ids, dropped = load_covar_aligned(
        str(test_fam), str(test_covar), transform=transform
    )

    assert keep_ids == ["j1", "j2"]
    assert dropped == []
    assert X_train.shape == (3, 2)
    assert X_test.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(X_train[:, 0]), np.ones((3,), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(X_test[:, 0]), np.ones((2,), dtype=np.float32))

    train_mean = np.mean(np.asarray([10.0, 20.0, 40.0], dtype=np.float32), dtype=np.float64)
    train_std = np.std(np.asarray([10.0, 20.0, 40.0], dtype=np.float32), dtype=np.float64)
    ref = ((np.asarray([25.0, 40.0], dtype=np.float32) - train_mean) / train_std).astype(np.float32)
    np.testing.assert_allclose(np.asarray(X_test[:, 1]), ref, atol=1e-6)


def test_training_covar_transform_respects_keep_subset(tmp_path):
    train_fam = tmp_path / "train_keep.fam"
    train_pheno = tmp_path / "train_keep.pheno"
    train_covar = tmp_path / "train_keep.qcovar"

    train_fam.write_text(
        "f1 i1 0 0 0 -9\n"
        "f2 i2 0 0 0 -9\n"
        "f3 i3 0 0 0 -9\n"
    )
    train_pheno.write_text(
        "f1 i1 1.0\n"
        "f2 i2 2.0\n"
        "f3 i3 3.0\n"
    )
    train_covar.write_text(
        "f1 i1 10 1\n"
        "f2 i2 20 1\n"
        "f3 i3 100 1\n"
    )

    _, X_keep, keep_ids, dropped, transform = load_pheno_covar_aligned_with_transform(
        str(train_fam),
        str(train_pheno),
        str(train_covar),
        add_intercept=True,
        keep_ids=["i1", "i2"],
    )

    assert keep_ids == ["i1", "i2"]
    assert "i3" in dropped
    assert X_keep.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(transform.means), np.asarray([15.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(transform.stds), np.asarray([5.0], dtype=np.float32))
