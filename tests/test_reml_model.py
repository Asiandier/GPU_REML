from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

_PKG = os.path.basename(REPO_ROOT)
import importlib
_common = importlib.import_module(f"{_PKG}.pipeline_common")
_kv_impl = importlib.import_module(f"{_PKG}.kv_impl")
_model = importlib.import_module(f"{_PKG}.reml_model")
_precond = importlib.import_module(f"{_PKG}.precond")
resolve_cpu_threads = _common.resolve_cpu_threads
FitConfig, InfinitesimalREMLFitter = _model.FitConfig, _model.InfinitesimalREMLFitter
ProjectedCorePrecondConf = _precond.ProjectedCorePrecondConf


def test_resolve_cpu_threads_prefers_omp_env(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "56")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    threads, source = resolve_cpu_threads()
    assert threads == 56
    assert source == "OMP_NUM_THREADS"


def test_fitter_passes_explicit_cpu_threads_to_streamer(monkeypatch):
    captured: list[int] = []

    @dataclass
    class _FakeStreamer:
        n: int = 8
        m: int = 16
        _n_calls: int = 1

    def _fake_streamer(*, build_threads=None, **_kwargs):
        captured.append(build_threads)
        return _FakeStreamer()

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", _fake_streamer)

    cfg = FitConfig(
        sources=[object()],
        cpu_threads=56,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    assert captured == [56]
    assert len(fitter.streamers) == 1


def test_fitter_passes_source_build_chunk_width_to_dense_streamer(monkeypatch):
    captured: list[int | None] = []

    @dataclass
    class _FakeStreamer:
        n: int = 8
        m: int = 16
        _n_calls: int = 1

    def _fake_streamer(*, source_build_chunk_width=None, **_kwargs):
        captured.append(source_build_chunk_width)
        return _FakeStreamer()

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", _fake_streamer)

    cfg = FitConfig(
        sources=[object()],
        component_variant_indices=[[0, 2], [1, 3]],
        source_build_chunk_width=8192,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    assert captured == [8192]
    assert len(fitter.streamers) == 1


def test_non_smile_path_forces_strict_optimizer(monkeypatch):
    captured: list[str] = []

    @dataclass
    class _FakeStreamer:
        n: int = 8
        m: int = 16
        _n_calls: int = 1

        def kv(self, V, normalize=True):
            del normalize
            return V

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

    def _fake_streamer(**_kwargs):
        return _FakeStreamer()

    def _fake_fit_reml(*, optimizer="strict", **_kwargs):
        captured.append(optimizer)
        return jnp.asarray([0.2, 0.8], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", _fake_streamer)
    monkeypatch.setattr(f"{_PKG}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[object()],
        smile_optimizer="smile_scoring",
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    try:
        res = fitter.fit_infinitesimal(jnp.ones((8,), dtype=jnp.float32))
        assert np.allclose(np.asarray(res.var_components), [0.2, 0.8])
        assert captured == ["strict"]
    finally:
        fitter.close()


def test_fitter_builds_sparse_streamer_for_rare_sources(monkeypatch):
    captured_dense: list[int] = []
    captured_sparse: list[int] = []

    @dataclass
    class _FakeStreamer:
        n: int = 8
        m: int = 16
        _n_calls: int = 1
        _missing_val: int = 3

        def kv(self, V, normalize=True):
            return V

        def diag(self):
            import jax.numpy as jnp
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

    def _fake_dense(*, build_threads=None, **_kwargs):
        captured_dense.append(build_threads)
        return _FakeStreamer()

    def _fake_sparse(*, build_threads=None, **_kwargs):
        captured_sparse.append(build_threads)
        return _FakeStreamer()

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", _fake_dense)
    monkeypatch.setattr(f"{_PKG}.sparse_stream.SparseGenoBlockStreamer", _fake_sparse)

    cfg = FitConfig(
        sources=[object()],
        rare_sources=[object()],
        cpu_threads=32,
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    assert len(fitter.streamers) == 2
    assert fitter._has_sparse is True
    assert fitter._n_dense_streamers == 1
    assert captured_dense == [32]
    assert captured_sparse == [32]
    fitter.close()
    assert fitter._has_sparse is False


def test_multi_grm_path_avoids_direct_device_put_in_weighted_hv(monkeypatch):
    dev = jax.devices()[0]

    class _FakeStreamer:
        n = 8
        m = 16
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

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())
    monkeypatch.setattr(
        _kv_impl,
        "kv_impl_multi_streamed_weighted",
        lambda V, streamers, call_plan, theta_g, theta_e=None, missing_val=3: V,
    )
    monkeypatch.setattr(
        _kv_impl,
        "kv_impl_multi_streamed_stacked",
        lambda V, streamers, call_plan, missing_val=3, normalize=True: jnp.stack(
            [V, 2.0 * V], axis=0
        ),
    )
    monkeypatch.setattr(f"{_PKG}.reml_model._ensure_on_device", lambda x, dev: x)

    def _fake_fit_reml(*, y, K_mvs, weighted_hv=None, stacked_kv=None, **_kwargs):
        del y, _kwargs
        V = jnp.ones((8, 2), dtype=jnp.float32)
        theta_g = jnp.ones((len(K_mvs),), dtype=jnp.float32)
        theta_e = jnp.asarray(0.5, dtype=jnp.float32)
        original = _model.jax.device_put

        def _boom(*args, **kwargs):
            raise AssertionError("unexpected direct device_put in multi-GRM closure")

        _model.jax.device_put = _boom
        try:
            out_hv = weighted_hv(theta_g, theta_e, V)
            out_stack = stacked_kv(V)
        finally:
            _model.jax.device_put = original

        assert out_hv.shape == V.shape
        assert out_stack.shape == (2, 8, 2)
        return jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{_PKG}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[object(), object()],
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    result = fitter.fit_infinitesimal(jnp.ones((8,), dtype=jnp.float32))
    assert result.var_components.shape == (3,)


def test_hybrid_multi_grm_path_avoids_direct_device_put_in_weighted_hv(monkeypatch):
    dev = jax.devices()[0]

    class _FakeDenseStreamer:
        n = 8
        m = 16
        _n_calls = 1
        _missing_val = 3

        def kv(self, V, normalize=True):
            del normalize
            return V

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

    class _FakeSparseStreamer(_FakeDenseStreamer):
        _eff_m_const = jnp.asarray(1.0, dtype=jnp.float32)

        def kv(self, V, normalize=True, *, sum_v=None):
            del normalize, sum_v
            return V

    _FakeDenseStreamer.dev = dev
    _FakeSparseStreamer.dev = dev

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeDenseStreamer())
    monkeypatch.setattr(f"{_PKG}.sparse_stream.SparseGenoBlockStreamer", lambda **_kwargs: _FakeSparseStreamer())
    monkeypatch.setattr(f"{_PKG}.reml_model._ensure_on_device", lambda x, dev: x)

    def _fake_fit_reml(*, y, K_mvs, weighted_hv=None, stacked_kv=None, **_kwargs):
        del y, _kwargs
        V = jnp.ones((8, 2), dtype=jnp.float32)
        theta_g = jnp.ones((len(K_mvs),), dtype=jnp.float32)
        theta_e = jnp.asarray(0.5, dtype=jnp.float32)
        original = _model.jax.device_put

        def _boom(*args, **kwargs):
            raise AssertionError("unexpected direct device_put in hybrid closure")

        _model.jax.device_put = _boom
        try:
            out_hv = weighted_hv(theta_g, theta_e, V)
            out_stack = stacked_kv(V)
        finally:
            _model.jax.device_put = original

        assert out_hv.shape == V.shape
        assert out_stack.shape == (2, 8, 2)
        return jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{_PKG}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[object()],
        rare_sources=[object()],
        precond_rank=0,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    result = fitter.fit_infinitesimal(jnp.ones((8,), dtype=jnp.float32))
    assert result.var_components.shape == (3,)


def test_multi_streamed_weighted_zero_eff_component_is_ignored(monkeypatch):
    class _FakeStreamer:
        n = 3
        _eff_m_const = jnp.asarray(0.0, dtype=jnp.float32)
        _missing_val = 3

        def __init__(self):
            self._true_widths_dev = [jnp.asarray(1, dtype=jnp.int32)]
            self._means_by_call = [jnp.asarray([0.0], dtype=jnp.float32)]
            self._inv_by_call = [jnp.asarray([1.0], dtype=jnp.float32)]

        def _pop_cached(self, _call_idx):
            return jnp.zeros((self.n, 1), dtype=jnp.uint8)

    st = _FakeStreamer()
    V = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    theta_g = jnp.asarray([2.0], dtype=jnp.float32)
    theta_e = jnp.asarray(0.5, dtype=jnp.float32)

    monkeypatch.setattr(_kv_impl, "_device_put_block", lambda block, dev: jnp.asarray(block))

    got = _kv_impl.kv_impl_multi_streamed_weighted(
        V,
        [st],
        [(0, 0)],
        theta_g,
        theta_e=theta_e,
    )

    assert np.all(np.isfinite(np.asarray(got)))
    assert np.allclose(np.asarray(got), np.asarray(theta_e * V), atol=1e-6)


def test_multi_grm_projected_core_precond_reaches_fit_reml(monkeypatch):
    dev = jax.devices()[0]

    class _FakeStreamer:
        n = 8
        m = 16
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

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())
    monkeypatch.setattr(
        _kv_impl,
        "kv_impl_multi_streamed_weighted",
        lambda V, streamers, call_plan, theta_g, theta_e=None, missing_val=3: V,
    )
    monkeypatch.setattr(
        _kv_impl,
        "kv_impl_multi_streamed_stacked",
        lambda V, streamers, call_plan, missing_val=3, normalize=True: jnp.stack(
            [V, 2.0 * V], axis=0
        ),
    )
    monkeypatch.setattr(
        _kv_impl,
        "build_projected_core_atoms_multi_streamed",
        lambda U, streamers, call_plan, missing_val=3, subtract_identity=True: jnp.stack(
            [
                jnp.zeros((U.shape[1], U.shape[1]), dtype=U.dtype),
                jnp.eye(U.shape[1], dtype=U.dtype),
            ],
            axis=0,
        ),
    )
    monkeypatch.setattr(
        f"{_PKG}.reml_model.build_lowrank_basis",
        lambda K_mv, n, max_rank, key: (
            jnp.eye(n, max_rank, dtype=jnp.float32),
            jnp.ones((max_rank,), dtype=jnp.float32),
        ),
    )

    def _fake_fit_reml(*, precond_conf=None, slq_mode="raw", precond_refresh_fn=None, precond_refresh_reldp=0.0, **_kwargs):
        assert isinstance(precond_conf, ProjectedCorePrecondConf)
        assert slq_mode == "projected_core_residual"
        assert precond_refresh_fn is not None
        assert precond_refresh_reldp > 0.0
        assert precond_conf.total_rank == 2
        assert precond_conf.n_grm == 2
        assert precond_conf.core_atoms.shape == (2, 2, 2)
        assert jnp.allclose(precond_conf.diag_atoms, jnp.ones((2,), dtype=jnp.float32))
        expected = jnp.stack(
            [
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.eye(2, dtype=jnp.float32),
            ],
            axis=0,
        )
        assert jnp.allclose(precond_conf.core_atoms, expected)
        return jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{_PKG}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[object(), object()],
        slq_mode="projected_core_residual",
        precond_type="projected_core",
        precond_rank=2,
        precond_refresh_reldp=0.1,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    result = fitter.fit_infinitesimal(jnp.ones((8,), dtype=jnp.float32))
    assert result.var_components.shape == (3,)


def test_projected_core_precond_stacked_kv_fallback_builds_core_atoms(monkeypatch):
    dev = jax.devices()[0]

    class _FakeStreamer:
        n = 4
        m = 8
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

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeStreamer())
    monkeypatch.setattr(
        f"{_PKG}.reml_model.build_lowrank_basis",
        lambda K_mv, n, max_rank, key: (
            jnp.eye(n, max_rank, dtype=jnp.float32),
            jnp.ones((max_rank,), dtype=jnp.float32),
        ),
    )

    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[object(), object()],
            precond_type="projected_core",
            precond_rank=2,
            verbose=False,
        )
    )
    conf = fitter._build_projected_core_precond(
        K_mvs=(lambda V: V, lambda V: 2.0 * V),
        diag_list=(
            jnp.ones((4,), dtype=jnp.float32),
            jnp.ones((4,), dtype=jnp.float32),
        ),
        weighted_hv=None,
        stacked_kv=lambda U: jnp.stack([U, 2.0 * U], axis=0),
        projected_core_atoms=None,
        var_components_init=None,
    )
    expected = jnp.stack(
        [
            jnp.zeros((2, 2), dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float32),
        ],
        axis=0,
    )
    assert isinstance(conf, ProjectedCorePrecondConf)
    assert conf.core_atoms.shape == (2, 2, 2)
    assert jnp.allclose(conf.core_atoms, expected)


def test_hybrid_projected_core_precond_reaches_fit_reml(monkeypatch):
    dev = jax.devices()[0]
    calls = {"dense_atoms": 0, "sparse_atoms": 0}

    class _FakeDenseStreamer:
        n = 8
        m = 16
        _n_calls = 1
        _missing_val = 3

        def kv(self, V, normalize=True):
            raise AssertionError("hybrid projected-core setup should not fall back to kv()")

        def diag(self):
            return jnp.ones((self.n,), dtype=jnp.float32)

        def _prepare_kv_pass(self):
            return None

        def build_projected_core_atom(self, U, *, subtract_identity=True):
            calls["dense_atoms"] += 1
            assert subtract_identity is True
            return jnp.zeros((U.shape[1], U.shape[1]), dtype=U.dtype)

    class _FakeSparseStreamer(_FakeDenseStreamer):
        _eff_m_const = jnp.asarray(1.0, dtype=jnp.float32)

        def build_projected_core_atom(self, U, *, subtract_identity=True):
            calls["sparse_atoms"] += 1
            assert subtract_identity is True
            return jnp.eye(U.shape[1], dtype=U.dtype)

    _FakeDenseStreamer.dev = dev
    _FakeSparseStreamer.dev = dev

    monkeypatch.setattr(f"{_PKG}.reml_model.GenoBlockStreamer", lambda **_kwargs: _FakeDenseStreamer())
    monkeypatch.setattr(f"{_PKG}.sparse_stream.SparseGenoBlockStreamer", lambda **_kwargs: _FakeSparseStreamer())
    monkeypatch.setattr(
        f"{_PKG}.reml_model.build_lowrank_basis",
        lambda K_mv, n, max_rank, key: (
            jnp.eye(n, max_rank, dtype=jnp.float32),
            jnp.ones((max_rank,), dtype=jnp.float32),
        ),
    )

    def _fake_fit_reml(*, precond_conf=None, **_kwargs):
        assert isinstance(precond_conf, ProjectedCorePrecondConf)
        assert precond_conf.total_rank == 2
        assert precond_conf.n_grm == 2
        assert precond_conf.core_atoms.shape == (2, 2, 2)
        expected = jnp.stack(
            [
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.eye(2, dtype=jnp.float32),
            ],
            axis=0,
        )
        assert jnp.allclose(precond_conf.core_atoms, expected)
        return jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32), [{"iter": 1}]

    monkeypatch.setattr(f"{_PKG}.reml_model.fit_reml", _fake_fit_reml)

    cfg = FitConfig(
        sources=[object()],
        rare_sources=[object()],
        precond_type="projected_core",
        precond_rank=2,
        verbose=False,
    )
    fitter = InfinitesimalREMLFitter(cfg)
    result = fitter.fit_infinitesimal(jnp.ones((8,), dtype=jnp.float32))
    assert result.var_components.shape == (3,)
    assert calls == {"dense_atoms": 1, "sparse_atoms": 1}
