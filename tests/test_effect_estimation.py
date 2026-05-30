from __future__ import annotations

import importlib
import os
import sys

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
GENO_STREAM = importlib.import_module(f"{PKG.__name__}.geno_stream")
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")

GenoBlockStreamer = GENO_STREAM.GenoBlockStreamer
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


def _standardize_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y32 = np.asarray(y, dtype=np.float32).reshape(-1)
    y_mean = float(y32.mean())
    y_scale = float(y32.std() + np.float32(1e-6))
    y_std = ((y32 - y_mean) / y_scale).astype(np.float64)
    return y_std, y_mean, y_scale


def _reference_effects(
    Z_list: list[np.ndarray],
    theta_g: np.ndarray,
    theta_e: float,
    y: np.ndarray,
    covar: np.ndarray | None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, float, float]:
    y_std, y_mean, y_scale = _standardize_y(y)
    n = y_std.size
    V = theta_e * np.eye(n, dtype=np.float64)
    for theta, Z in zip(theta_g.tolist(), Z_list):
        eff = float(Z.shape[1])
        if eff > 0.0:
            V = V + float(theta) * (Z @ Z.T) / eff

    alpha_rhs = np.linalg.solve(V, y_std)
    if covar is not None and covar.size > 0:
        X = np.asarray(covar, dtype=np.float64)
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
    g_list: list[np.ndarray] = []
    for theta, Z in zip(theta_g.tolist(), Z_list):
        eff = float(Z.shape[1])
        if eff > 0.0:
            b = float(theta) * (Z.T @ alpha) / eff
            g = Z @ b
        else:
            b = np.zeros((Z.shape[1],), dtype=np.float64)
            g = np.zeros((n,), dtype=np.float64)
        b_list.append(b)
        g_list.append(g)
    g_total = np.sum(np.stack(g_list, axis=0), axis=0)
    return beta, b_list, g_list, g_total, y_mean, y_scale


def test_estimate_effects_matches_dense_reference_single_grm():
    X = _make_non_degenerate_genotypes(n=16, m=7, seed=0)
    y = (
        0.4 * X[:, 1].astype(np.float32)
        - 0.25 * X[:, 5].astype(np.float32)
        + np.random.RandomState(1).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )
    covar = np.c_[
        np.ones((X.shape[0],), dtype=np.float32),
        np.linspace(-1.0, 1.0, X.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.35, 0.65], dtype=np.float32)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = fitter.estimate_effects(
            jnp.asarray(y),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar),
        )
        Z = np.asarray(
            fitter.streamers[0].extract_standardized_columns(np.arange(X.shape[1])),
            dtype=np.float64,
        )
    finally:
        for st in fitter.streamers:
            st.close()

    beta_ref, b_ref, g_ref, g_total_ref, y_mean_ref, y_scale_ref = _reference_effects(
        [Z], theta[:-1], float(theta[-1]), y, covar
    )

    np.testing.assert_allclose(np.asarray(effects.fixed_effects), beta_ref, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(effects.snp_effects[0]), b_ref[0], atol=2e-3, rtol=3e-3)
    np.testing.assert_allclose(
        np.asarray(effects.random_effect_components[0]), g_ref[0], atol=2e-3, rtol=3e-3
    )
    np.testing.assert_allclose(np.asarray(effects.random_effect), g_total_ref, atol=2e-3, rtol=3e-3)
    assert abs(effects.y_mean - y_mean_ref) < 1e-6
    assert abs(effects.y_scale - y_scale_ref) < 1e-6


def test_estimate_effects_matches_dense_reference_partitioned_multi_grm():
    X = _make_non_degenerate_genotypes(n=18, m=9, seed=3)
    block_sizes = [4, 3, 2]
    y = (
        0.35 * X[:, 0].astype(np.float32)
        - 0.2 * X[:, 6].astype(np.float32)
        + np.random.RandomState(4).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )
    covar = np.c_[
        np.ones((X.shape[0],), dtype=np.float32),
        np.linspace(0.0, 1.0, X.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.18, 0.09, 0.14, 0.59], dtype=np.float32)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            vc_block_sizes=block_sizes,
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = fitter.estimate_effects(
            jnp.asarray(y),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar),
        )
        st = fitter._partitioned_streamer
        offsets = np.r_[0, np.cumsum(block_sizes)]
        Z_list = [
            np.asarray(
                st.extract_standardized_columns(np.arange(offsets[g], offsets[g + 1])),
                dtype=np.float64,
            )
            for g in range(len(block_sizes))
        ]
    finally:
        for st_i in fitter.streamers:
            st_i.close()

    beta_ref, b_ref, g_ref, g_total_ref, _, _ = _reference_effects(
        Z_list, theta[:-1], float(theta[-1]), y, covar
    )

    np.testing.assert_allclose(np.asarray(effects.fixed_effects), beta_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(effects.snp_effects, b_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    for got, ref in zip(effects.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(effects.random_effect), g_total_ref, atol=3e-3, rtol=4e-3)


def test_estimate_effects_matches_dense_reference_arbitrary_grouped_multi_grm():
    X = _make_non_degenerate_genotypes(n=18, m=9, seed=303)
    component_variant_indices = [
        [0, 3, 4],
        [1, 2],
        [5, 6, 7, 8],
    ]
    y = (
        0.35 * X[:, 0].astype(np.float32)
        - 0.2 * X[:, 6].astype(np.float32)
        + np.random.RandomState(304).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )
    covar = np.c_[
        np.ones((X.shape[0],), dtype=np.float32),
        np.linspace(0.0, 1.0, X.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.18, 0.09, 0.14, 0.59], dtype=np.float32)

    fitter = InfinitesimalREMLFitter(
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
        effects = fitter.estimate_effects(
            jnp.asarray(y),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar),
        )
        Z_list = [
            np.asarray(
                fitter.streamers[0].extract_standardized_columns(
                    np.arange(
                        int(fitter.streamers[0]._component_snp_offsets[g]),
                        int(fitter.streamers[0]._component_snp_offsets[g + 1]),
                    )
                ),
                dtype=np.float64,
            )
            for g in range(len(component_variant_indices))
        ]
    finally:
        for st in fitter.streamers:
            st.close()

    beta_ref, b_ref, g_ref, g_total_ref, _, _ = _reference_effects(
        Z_list, theta[:-1], float(theta[-1]), y, covar
    )

    np.testing.assert_allclose(np.asarray(effects.fixed_effects), beta_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(effects.snp_effects, b_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    for got, ref in zip(effects.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(effects.random_effect), g_total_ref, atol=3e-3, rtol=4e-3)


def test_estimate_effects_matches_dense_reference_separate_multi_grm():
    X = _make_non_degenerate_genotypes(n=18, m=9, seed=33)
    block_sizes = [4, 3, 2]
    offsets = np.r_[0, np.cumsum(block_sizes)]
    y = (
        0.28 * X[:, 1].astype(np.float32)
        - 0.24 * X[:, 7].astype(np.float32)
        + np.random.RandomState(34).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )
    covar = np.c_[
        np.ones((X.shape[0],), dtype=np.float32),
        np.linspace(-0.2, 1.1, X.shape[0], dtype=np.float32),
    ]
    theta = np.asarray([0.15, 0.08, 0.11, 0.66], dtype=np.float32)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[
                _ArraySource(X[:, offsets[g] : offsets[g + 1]])
                for g in range(len(block_sizes))
            ],
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        effects = fitter.estimate_effects(
            jnp.asarray(y),
            var_components=jnp.asarray(theta),
            covar=jnp.asarray(covar),
        )
        Z_list = [
            np.asarray(
                st.extract_standardized_columns(np.arange(st.m)),
                dtype=np.float64,
            )
            for st in fitter.streamers
        ]
    finally:
        for st in fitter.streamers:
            st.close()

    beta_ref, b_ref, g_ref, g_total_ref, _, _ = _reference_effects(
        Z_list, theta[:-1], float(theta[-1]), y, covar
    )

    np.testing.assert_allclose(np.asarray(effects.fixed_effects), beta_ref, atol=3e-3, rtol=3e-3)
    for got, ref in zip(effects.snp_effects, b_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    for got, ref in zip(effects.random_effect_components, g_ref):
        np.testing.assert_allclose(np.asarray(got), ref, atol=3e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(effects.random_effect), g_total_ref, atol=3e-3, rtol=4e-3)


def test_fit_infinitesimal_can_attach_effect_estimates(monkeypatch):
    X = _make_non_degenerate_genotypes(n=14, m=6, seed=7)
    y = (
        0.3 * X[:, 2].astype(np.float32)
        + np.random.RandomState(9).standard_normal(X.shape[0]).astype(np.float32) * 0.1
    )
    covar = np.c_[np.ones((X.shape[0],), dtype=np.float32)]
    theta = jnp.asarray([0.25, 0.75], dtype=jnp.float32)

    def _fake_fit_reml(**_kwargs):
        return theta, [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(X)],
            call_width=4,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(
            jnp.asarray(y),
            covar=jnp.asarray(covar),
            estimate_effects=True,
        )
        direct = fitter.estimate_effects(
            jnp.asarray(y),
            var_components=theta,
            covar=jnp.asarray(covar),
        )
    finally:
        for st in fitter.streamers:
            st.close()

    assert res.effects is not None
    np.testing.assert_allclose(np.asarray(res.effects.fixed_effects), np.asarray(direct.fixed_effects))
    np.testing.assert_allclose(np.asarray(res.effects.random_effect), np.asarray(direct.random_effect))
    np.testing.assert_allclose(np.asarray(res.effects.snp_effects[0]), np.asarray(direct.snp_effects[0]))
