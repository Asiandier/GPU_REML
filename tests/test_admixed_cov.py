from __future__ import annotations

import importlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
ADMIX = importlib.import_module(f"{PKG.__name__}.admixed_cov")
MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")


def test_load_admixture_q_aligns_by_iid(tmp_path):
    fam = tmp_path / "admix.fam"
    fam.write_text("F1 id1 0 0 1 -9\nF2 id2 0 0 1 -9\nF3 id3 0 0 1 -9\n")
    q = tmp_path / "admix.2.Q"
    np.savetxt(q, np.asarray([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]], dtype=np.float32))

    got = ADMIX.load_admixture_q_aligned(
        q_path=str(q),
        q_fam_path=str(fam),
        sample_ids=["id3", "id1"],
        component_names=["A", "B"],
    )

    assert got.component_names == ("A", "B")
    assert np.allclose(got.weights, [[0.4, 0.6], [0.8, 0.2]])


def test_load_admixture_q_rejects_missing_iid(tmp_path):
    fam = tmp_path / "admix.fam"
    fam.write_text("F1 id1 0 0 1 -9\n")
    q = tmp_path / "admix.1.Q"
    np.savetxt(q, np.asarray([[1.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="missing from ADMIXTURE FAM"):
        ADMIX.load_admixture_q_aligned(
            q_path=str(q),
            q_fam_path=str(fam),
            sample_ids=["id2"],
        )


def test_admixed_weighted_hv_matches_dkd_formula(monkeypatch):
    dev = jax.devices()[0]

    class _FakeStreamer:
        n = 3
        m = 5
        _n_calls = 1
        _missing_val = 3

        def kv(self, V, normalize=True):
            del normalize
            return V

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

    _FakeStreamer.dev = dev

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())

    weights = np.asarray(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def _fake_fit_reml(*, weighted_hv=None, stacked_kv=None, K_mvs=None, **_kwargs):
        V = jnp.asarray([[2.0], [3.0], [5.0]], dtype=jnp.float32)
        theta_g = jnp.asarray([0.4, 0.7], dtype=jnp.float32)
        theta_e = jnp.asarray(0.2, dtype=jnp.float32)
        out = weighted_hv(theta_g, theta_e, V)
        expected_scale = theta_e + theta_g[0] * weights[:, 0] + theta_g[1] * weights[:, 1]
        expected = jnp.asarray(expected_scale[:, None], dtype=jnp.float32) * V
        assert np.allclose(np.asarray(out), np.asarray(expected), rtol=1e-6, atol=1e-6)

        stack = stacked_kv(V)
        assert stack.shape == (2, 3, 1)
        assert np.allclose(np.asarray(K_mvs[0](V)), np.asarray(weights[:, 0:1] * V))
        return jnp.asarray([0.4, 0.7, 0.2], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = MODEL.InfinitesimalREMLFitter(
        MODEL.FitConfig(
            sources=[object()],
            admix_weights=weights,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.ones((3,), dtype=jnp.float32))
    finally:
        fitter.close()

    assert np.allclose(np.asarray(res.var_components), [0.4, 0.7, 0.2])


def test_admixed_weighted_hv_matches_dkd_formula_for_nondiagonal_k(monkeypatch):
    dev = jax.devices()[0]
    k_mat = jnp.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.3],
            [-0.1, 0.3, 1.0],
        ],
        dtype=jnp.float32,
    )

    class _FakeStreamer:
        n = 3
        m = 5
        _n_calls = 1
        _missing_val = 3

        def kv(self, V, normalize=True):
            del normalize
            return k_mat @ V

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

    _FakeStreamer.dev = dev

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())

    weights = np.asarray(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    def _fake_fit_reml(*, weighted_hv=None, stacked_kv=None, K_mvs=None, **_kwargs):
        V = jnp.asarray([[2.0, -1.0], [3.0, 4.0], [5.0, 0.5]], dtype=jnp.float32)
        theta_g = jnp.asarray([0.4, 0.7], dtype=jnp.float32)
        theta_e = jnp.asarray(0.2, dtype=jnp.float32)
        out = weighted_hv(theta_g, theta_e, V)

        expected = theta_e * V
        for component_idx in range(weights.shape[1]):
            w = jnp.sqrt(jnp.asarray(weights[:, component_idx], dtype=jnp.float32))[:, None]
            expected = expected + theta_g[component_idx] * (w * (k_mat @ (w * V)))
        assert np.allclose(np.asarray(out), np.asarray(expected), rtol=1e-6, atol=1e-6)

        stack = stacked_kv(V)
        assert stack.shape == (2, 3, 2)
        for component_idx in range(weights.shape[1]):
            w = jnp.sqrt(jnp.asarray(weights[:, component_idx], dtype=jnp.float32))[:, None]
            expected_component = w * (k_mat @ (w * V))
            assert np.allclose(
                np.asarray(K_mvs[component_idx](V)),
                np.asarray(expected_component),
                rtol=1e-6,
                atol=1e-6,
            )
        return jnp.asarray([0.4, 0.7, 0.2], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = MODEL.InfinitesimalREMLFitter(
        MODEL.FitConfig(
            sources=[object()],
            admix_weights=weights,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.ones((3,), dtype=jnp.float32))
    finally:
        fitter.close()

    assert np.allclose(np.asarray(res.var_components), [0.4, 0.7, 0.2])


def test_admixed_per_ancestry_residual_hv_matches_weighted_diagonal(monkeypatch):
    dev = jax.devices()[0]

    class _FakeStreamer:
        n = 3
        m = 5
        _n_calls = 1
        _missing_val = 3

        def kv(self, V, normalize=True):
            del normalize
            return V

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

    _FakeStreamer.dev = dev

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())

    weights = np.asarray(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def _fake_fit_reml(*, weighted_hv=None, residual_diag_list=None, **_kwargs):
        V = jnp.asarray([[2.0], [3.0], [5.0]], dtype=jnp.float32)
        theta_g = jnp.asarray([0.4, 0.7], dtype=jnp.float32)
        theta_e = jnp.asarray([0.2, 0.9], dtype=jnp.float32)
        out = weighted_hv(theta_g, theta_e, V)
        expected_scale = (
            theta_g[0] * weights[:, 0]
            + theta_g[1] * weights[:, 1]
            + theta_e[0] * weights[:, 0]
            + theta_e[1] * weights[:, 1]
        )
        expected = jnp.asarray(expected_scale[:, None], dtype=jnp.float32) * V
        assert np.allclose(np.asarray(out), np.asarray(expected), rtol=1e-6, atol=1e-6)
        assert residual_diag_list is not None
        assert len(residual_diag_list) == 2
        assert np.allclose(np.asarray(residual_diag_list[0]), weights[:, 0])
        assert np.allclose(np.asarray(residual_diag_list[1]), weights[:, 1])
        return jnp.asarray([0.4, 0.7, 0.2, 0.9], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{PKG.__name__}.reml_model.fit_reml", _fake_fit_reml)

    fitter = MODEL.InfinitesimalREMLFitter(
        MODEL.FitConfig(
            sources=[object()],
            admix_weights=weights,
            admix_residual_mode="per-ancestry",
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        res = fitter.fit_infinitesimal(jnp.ones((3,), dtype=jnp.float32))
    finally:
        fitter.close()

    assert np.allclose(np.asarray(res.var_components), [0.4, 0.7, 0.2, 0.9])
