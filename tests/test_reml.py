from __future__ import annotations

import importlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

jax.config.update("jax_platform_name", "cpu")

PKG = os.path.basename(REPO_ROOT)
REML = importlib.import_module(f"{PKG}.reml")
PRE = importlib.import_module(f"{PKG}.precond")

REMLContext = REML.REMLContext
_compute_traces_from_pcg = REML._compute_traces_from_pcg
_scalar_diag_from_diag_list = REML._scalar_diag_from_diag_list
fit_reml = REML.fit_reml
ProjectedCorePrecondConf = PRE.ProjectedCorePrecondConf


def test_projected_fisher_step_freezes_boundary_variable_with_nonpositive_gradient():
    param = jnp.array([0.0, 0.4], dtype=jnp.float32)
    grad = jnp.array([-0.2, 0.1], dtype=jnp.float32)
    fi = jnp.eye(2, dtype=jnp.float32)

    param_updated, delta_param, step_dir, alpha_max, freeze_mask = REML._projected_fisher_step(
        param,
        grad,
        fi,
        genetic_zero_tol=1e-8,
        residual_floor=1e-6,
    )

    assert bool(freeze_mask[0])
    assert np.isclose(float(step_dir[0]), 0.0, atol=1e-7)
    assert np.isclose(float(delta_param[0]), 0.0)
    assert np.isclose(float(param_updated[0]), 0.0)
    assert np.isclose(alpha_max, 1.0)


def test_projected_fisher_step_can_hit_exact_zero_without_genetic_floor():
    param = jnp.array([0.5, 0.5], dtype=jnp.float32)
    grad = jnp.array([-1.0, 0.0], dtype=jnp.float32)
    fi = jnp.eye(2, dtype=jnp.float32)

    param_updated, delta_param, step_dir, alpha_max, freeze_mask = REML._projected_fisher_step(
        param,
        grad,
        fi,
        genetic_zero_tol=1e-8,
        residual_floor=1e-6,
    )

    assert bool(freeze_mask[0])
    assert np.isclose(float(step_dir[0]), -0.5, atol=2e-4)
    assert np.isclose(alpha_max, 1.0)
    assert np.isclose(float(param_updated[0]), 0.0)
    assert np.isclose(float(delta_param[0]), -0.5)


def test_projected_fisher_step_reduces_free_rhs_after_fixing_boundary_hits():
    param = jnp.array([0.5, 0.4, 0.3], dtype=jnp.float32)
    grad = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32)
    fi = jnp.array(
        [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    param_updated, delta_param, step_dir, alpha_max, freeze_mask = REML._projected_fisher_step(
        param,
        grad,
        fi,
        genetic_zero_tol=1e-8,
        residual_floor=1e-6,
    )

    assert bool(freeze_mask[0])
    assert not bool(freeze_mask[1])
    assert np.isclose(alpha_max, 1.0)
    assert np.isclose(float(step_dir[0]), -0.5, atol=2e-4)
    assert np.isclose(float(step_dir[1]), 0.25, atol=2e-4)
    assert np.isclose(float(delta_param[0]), -0.5)
    assert np.isclose(float(delta_param[1]), 0.25, atol=2e-4)
    assert np.isclose(float(param_updated[0]), 0.0)
    assert np.isclose(float(param_updated[1]), 0.65, atol=2e-4)


def test_projected_gradient_inf_norm_uses_kkt_boundary_rule():
    param = jnp.array([0.0, 0.2, 0.4], dtype=jnp.float32)
    grad = jnp.array([-0.3, 0.05, -0.1], dtype=jnp.float32)

    proj_inf = REML._projected_gradient_inf_norm(param, grad, 1e-8)

    assert np.isclose(proj_inf, 0.1)


def test_compute_traces_from_pcg_uses_vrand_for_hinv_k_trace():
    vrand = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    hinv_vrand = jnp.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    hinv_k_vrand = jnp.array(
        [
            [4.0, 7.0],
            [5.0, 6.0],
        ],
        dtype=jnp.float32,
    )

    rhs_const = jnp.concatenate(
        [
            jnp.zeros((2, 1), dtype=jnp.float32),
            vrand,
            jnp.zeros((2, 2), dtype=jnp.float32),
        ],
        axis=1,
    )
    sol_all = jnp.concatenate(
        [
            jnp.zeros((2, 1), dtype=jnp.float32),
            hinv_vrand,
            hinv_k_vrand,
        ],
        axis=1,
    )
    ctx = REMLContext(
        n=2,
        G=1,
        K_mvs=(),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.zeros((1, 2), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((2,), dtype=jnp.float32),
        rhs_const=rhs_const,
        y_col=0,
        rand_stop=3,
        n_XyZ_cols=3,
        n_GZrand_components=1,
        R_rand=2,
        precond_conf=None,
    )

    tr_hinv, tr_hinv_k = _compute_traces_from_pcg(sol_all, ctx)

    assert np.isclose(float(tr_hinv), (2.0 + 3.0) / 2.0)
    assert np.isclose(float(tr_hinv_k[0]), (4.0 + 6.0) / 2.0)


def test_compute_traces_from_pcg_uses_cached_kvrand_stack():
    vrand = jnp.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ],
        dtype=jnp.float32,
    )
    hinv_vrand = jnp.array(
        [
            [2.0, 1.0],
            [3.0, -4.0],
        ],
        dtype=jnp.float32,
    )
    kvrand_stack = jnp.array(
        [
            [
                [4.0, 0.0],
                [5.0, 6.0],
            ],
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        ],
        dtype=jnp.float32,
    )

    rhs_const = jnp.concatenate(
        [
            jnp.zeros((2, 1), dtype=jnp.float32),
            vrand,
        ],
        axis=1,
    )
    sol_all = jnp.concatenate(
        [
            jnp.zeros((2, 1), dtype=jnp.float32),
            hinv_vrand,
        ],
        axis=1,
    )
    ctx = REMLContext(
        n=2,
        G=2,
        K_mvs=(),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.zeros((2, 2), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((2,), dtype=jnp.float32),
        rhs_const=rhs_const,
        y_col=0,
        rand_stop=3,
        n_XyZ_cols=3,
        n_GZrand_components=2,
        R_rand=2,
        precond_conf=None,
        kvrand_stack=kvrand_stack,
    )

    tr_hinv, tr_hinv_k = _compute_traces_from_pcg(sol_all, ctx)

    assert np.isclose(float(tr_hinv), np.sum(np.asarray(vrand) * np.asarray(hinv_vrand)) / 2.0)
    expected = np.einsum("nr,inr->i", np.asarray(hinv_vrand), np.asarray(kvrand_stack)) / 2.0
    assert np.allclose(np.asarray(tr_hinv_k), expected)


def test_scalar_diag_from_diag_list_detects_constant_vectors():
    got = _scalar_diag_from_diag_list(
        [
            jnp.ones((4,), dtype=jnp.float32),
            jnp.full((4,), 2.5, dtype=jnp.float32),
        ]
    )
    assert got is not None
    assert np.allclose(np.asarray(got), [1.0, 2.5])


def test_scalar_diag_from_diag_list_rejects_nonconstant_vectors():
    got = _scalar_diag_from_diag_list(
        [
            jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1.0, 2.0, 1.0], dtype=jnp.float32),
        ]
    )
    assert got is None


def test_fit_reml_allows_none_preconditioner():
    def mv(v):
        return v

    y = jnp.array([0.5, -0.1, 1.2, 0.3], dtype=jnp.float32)
    param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        verbose=False,
    )

    assert param.shape == (2,)
    assert len(history) == 1
    assert np.all(np.isfinite(np.asarray(param)))
    assert "step_sec" in history[0]
    assert "ai_pcg_iters" in history[0]
    assert "ai_elapsed_sec" in history[0]
    assert "ws_resolve_count" in history[0]
    assert "ws_fixed_total" in history[0]


def test_newton_step_solves_dense_average_information_system():
    stats = REML.FisherSolveStats()
    fi = REML.AverageInfoMatrix(
        mat=jnp.eye(2, dtype=jnp.float32),
        stats=stats,
    )

    REML._reset_fisher_solve_stats(stats, free_dim=2, frozen_genetic=0)
    step = REML._newton_step(jnp.array([1.0, -2.0], dtype=jnp.float32), fi)

    expected = np.array([1.0, -2.0], dtype=np.float32) / (1.0 + REML.FI_SYSTEM_RIDGE)
    assert np.allclose(np.asarray(step), expected, atol=3e-4)
    assert stats.free_dim == 2
    assert stats.frozen_genetic == 0


def test_eval_once_ai_pcg_uses_current_rhs_precond_start(monkeypatch):
    pcg_x0s = []

    def fake_pcg_solve(Hv, B, M=None, tol=1e-2, maxiter=200, X0=None, check_every=2):
        del Hv, M, tol, maxiter, check_every
        pcg_x0s.append(np.asarray(X0))
        sol = jnp.full(B.shape, float(len(pcg_x0s)), dtype=jnp.float32)
        return sol, jnp.asarray(0.0, dtype=jnp.float32), 1

    monkeypatch.setattr(REML, "pcg_solve", fake_pcg_solve)

    ctx = REMLContext(
        n=3,
        G=1,
        K_mvs=(lambda v: v,),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.ones((1, 3), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((3,), dtype=jnp.float32),
        rhs_const=jnp.concatenate(
            [
                jnp.zeros((3, 1), dtype=jnp.float32),
                jnp.array([[1.0], [-1.0], [1.0]], dtype=jnp.float32),
            ],
            axis=1,
        ),
        y_col=0,
        rand_stop=2,
        n_XyZ_cols=2,
        n_GZrand_components=1,
        R_rand=1,
        precond_conf=None,
        kvrand_stack=jnp.ones((1, 3, 1), dtype=jnp.float32),
    )

    ll, grad, fi, _k_pcg, _warm_next, _warm_ai_next, _tr_hinv, _tr_hinv_k, _logdet = REML._eval_once(
        ctx,
        jnp.array([0.3, 0.7], dtype=jnp.float32),
        warm_all=jnp.zeros((3, 2), dtype=jnp.float32),
        key_slq=jax.random.PRNGKey(0),
        minq_tol=1e-2,
        maxiter=4,
        precond_eps=1e-6,
        slq_samples=2,
        slq_m=2,
        warm_ready=False,
        taylor_logdet=jnp.asarray(0.0, dtype=jnp.float32),
        compute_traces=False,
    )

    assert np.all(np.isfinite(np.asarray(ll)))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert isinstance(fi, REML.AverageInfoMatrix)
    assert fi.stats is not None
    assert fi.stats.ai_pcg_iters == 1
    assert len(pcg_x0s) == 2
    np.testing.assert_allclose(pcg_x0s[1], np.zeros((3, 2), dtype=np.float32))
    assert np.all(np.isfinite(np.asarray(fi.mat)))
    assert fi.stats is not None


def test_eval_once_main_pcg_uses_defect_corrected_warm_start(monkeypatch):
    pcg_x0s = []
    pcg_bs = []

    def fake_pcg_solve(Hv, B, M=None, tol=1e-2, maxiter=200, X0=None, check_every=2):
        del Hv, tol, maxiter, check_every
        pcg_x0s.append(np.asarray(X0))
        pcg_bs.append(np.asarray(B))
        sol = jnp.full(B.shape, float(len(pcg_x0s)), dtype=jnp.float32)
        return sol, jnp.asarray(0.0, dtype=jnp.float32), 1

    monkeypatch.setattr(REML, "pcg_solve", fake_pcg_solve)
    monkeypatch.setattr(REML, "make_precond", lambda *_args, **_kwargs: (lambda B: B))

    ctx = REMLContext(
        n=3,
        G=1,
        K_mvs=(lambda v: jnp.zeros_like(v),),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.zeros((1, 3), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((3,), dtype=jnp.float32),
        rhs_const=jnp.concatenate(
            [
                jnp.zeros((3, 1), dtype=jnp.float32),
                jnp.array([[1.0], [-1.0], [1.0]], dtype=jnp.float32),
            ],
            axis=1,
        ),
        y_col=0,
        rand_stop=2,
        n_XyZ_cols=2,
        n_GZrand_components=1,
        R_rand=1,
        precond_conf=None,
        kvrand_stack=jnp.zeros((1, 3, 1), dtype=jnp.float32),
    )

    REML._eval_once(
        ctx,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        warm_all=jnp.full((3, 2), 7.0, dtype=jnp.float32),
        key_slq=jax.random.PRNGKey(0),
        minq_tol=1e-2,
        maxiter=4,
        precond_eps=1e-6,
        slq_samples=2,
        slq_m=2,
        warm_ready=True,
        warm_ai_ready=False,
        taylor_logdet=jnp.asarray(0.0, dtype=jnp.float32),
        compute_traces=False,
    )

    assert len(pcg_x0s) == 2
    np.testing.assert_allclose(pcg_x0s[0], pcg_bs[0])


def test_eval_once_ai_pcg_uses_defect_corrected_warm_start(monkeypatch):
    pcg_x0s = []
    pcg_bs = []

    def fake_pcg_solve(Hv, B, M=None, tol=1e-2, maxiter=200, X0=None, check_every=2):
        del Hv, tol, maxiter, check_every
        pcg_x0s.append(np.asarray(X0))
        pcg_bs.append(np.asarray(B))
        sol = jnp.full(B.shape, float(len(pcg_x0s)), dtype=jnp.float32)
        return sol, jnp.asarray(0.0, dtype=jnp.float32), 1

    monkeypatch.setattr(REML, "pcg_solve", fake_pcg_solve)
    monkeypatch.setattr(REML, "make_precond", lambda *_args, **_kwargs: (lambda B: B))

    ctx = REMLContext(
        n=3,
        G=1,
        K_mvs=(lambda v: v,),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.ones((1, 3), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((3,), dtype=jnp.float32),
        rhs_const=jnp.concatenate(
            [
                jnp.zeros((3, 1), dtype=jnp.float32),
                jnp.array([[1.0], [-1.0], [1.0]], dtype=jnp.float32),
            ],
            axis=1,
        ),
        y_col=0,
        rand_stop=2,
        n_XyZ_cols=2,
        n_GZrand_components=1,
        R_rand=1,
        precond_conf=None,
        kvrand_stack=jnp.ones((1, 3, 1), dtype=jnp.float32),
    )

    _ll, _grad, _fi, _k_pcg, warm_next, warm_ai_next, _tr_hinv, _tr_hinv_k, _logdet = REML._eval_once(
        ctx,
        jnp.array([0.3, 0.7], dtype=jnp.float32),
        warm_all=jnp.zeros((3, 2), dtype=jnp.float32),
        key_slq=jax.random.PRNGKey(0),
        minq_tol=1e-2,
        maxiter=4,
        precond_eps=1e-6,
        slq_samples=2,
        slq_m=2,
        warm_ready=True,
        warm_ai_ready=True,
        taylor_logdet=jnp.asarray(0.0, dtype=jnp.float32),
        compute_traces=False,
    )

    assert len(pcg_x0s) == 2
    np.testing.assert_allclose(pcg_x0s[1], pcg_bs[1])


def test_eval_once_keeps_full_width_ai_pcg(monkeypatch):
    pcg_bs = []

    def fake_pcg_solve(Hv, B, M=None, tol=1e-2, maxiter=200, X0=None, check_every=2):
        del Hv, M, tol, maxiter, X0, check_every
        pcg_bs.append(np.asarray(B))
        sol = jnp.full(B.shape, float(len(pcg_bs)), dtype=jnp.float32)
        return sol, jnp.asarray(0.0, dtype=jnp.float32), 1

    monkeypatch.setattr(REML, "pcg_solve", fake_pcg_solve)

    ctx = REMLContext(
        n=3,
        G=3,
        K_mvs=(
            lambda v: v,
            lambda v: 2.0 * v,
            lambda v: 3.0 * v,
        ),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.ones((3, 3), dtype=jnp.float32),
        xmat=None,
        y=jnp.zeros((3,), dtype=jnp.float32),
        rhs_const=jnp.concatenate(
            [
                jnp.zeros((3, 1), dtype=jnp.float32),
                jnp.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
            ],
            axis=1,
        ),
        y_col=0,
        rand_stop=3,
        n_XyZ_cols=3,
        n_GZrand_components=3,
        R_rand=2,
        precond_conf=None,
        kvrand_stack=jnp.ones((3, 3, 2), dtype=jnp.float32),
    )

    _ll, grad, fi, _k_pcg, _warm_next, warm_ai_next, _tr_hinv, _tr_hinv_k, _logdet = REML._eval_once(
        ctx,
        jnp.array([0.1, 0.2, 0.3, 0.7], dtype=jnp.float32),
        warm_all=jnp.zeros((3, 3), dtype=jnp.float32),
        warm_ai=jnp.zeros((3, 4), dtype=jnp.float32),
        key_slq=jax.random.PRNGKey(0),
        minq_tol=1e-2,
        maxiter=4,
        precond_eps=1e-6,
        slq_samples=2,
        slq_m=2,
        warm_ready=False,
        warm_ai_ready=False,
        taylor_logdet=jnp.asarray(0.0, dtype=jnp.float32),
        compute_traces=False,
    )

    assert len(pcg_bs) == 2
    assert pcg_bs[0].shape[1] == 3
    assert pcg_bs[1].shape[1] == 4
    assert warm_ai_next.shape == (3, 4)
    assert grad.shape == (4,)
    assert fi.mat.shape == (4, 4)


def test_fit_reml_warm_width_depends_only_on_xy_and_probes(monkeypatch):
    seen_warm_shapes = []

    def fake_eval_once(ctx, pvec, warm_all, **_kwargs):
        del ctx, pvec, _kwargs
        seen_warm_shapes.append(warm_all.shape)
        ll = jnp.asarray(0.0, dtype=jnp.float32)
        grad = jnp.zeros((3,), dtype=jnp.float32)
        fi = jnp.eye(3, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((2,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)

    def mv(v):
        return v

    y = jnp.array([0.5, -0.1, 1.2, 0.3], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv, mv],
        diag_list=[
            jnp.ones((4,), dtype=jnp.float32),
            jnp.ones((4,), dtype=jnp.float32),
        ],
        covar=None,
        n_rand_vec=3,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        verbose=False,
    )

    assert seen_warm_shapes
    assert seen_warm_shapes[0] == (4, 4)


def test_fit_reml_skips_exact_slq_once_taylor_is_used(monkeypatch):
    calls = {"slq": 0}

    def fake_slq(*_args, **_kwargs):
        calls["slq"] += 1
        return jnp.asarray(0.0, dtype=jnp.float32)

    def fake_newton_step(grad, fi):
        del grad, fi
        return jnp.array([1e-4, -1e-4], dtype=jnp.float32)

    monkeypatch.setattr(REML, "_slq_logdet", fake_slq)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        verbose=False,
    )

    assert calls["slq"] == 1


def test_fit_reml_skips_diag_stack_for_scalar_identity_precond(monkeypatch):
    seen_diag_stacks = []

    def fake_eval_once(ctx, pvec, warm_all, **_kwargs):
        seen_diag_stacks.append(ctx.diag_stack)
        ll = jnp.asarray(0.0, dtype=jnp.float32)
        grad = jnp.zeros_like(pvec)
        fi = jnp.eye(pvec.shape[0], dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((ctx.G,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)

    def mv(v):
        return v

    precond_conf = ProjectedCorePrecondConf(
        U=jnp.empty((4, 0), dtype=jnp.float32),
        core_atoms=jnp.empty((1, 0, 0), dtype=jnp.float32),
        total_rank=0,
        n_grm=1,
        diag_mode="scalar_identity",
        diag_atoms=jnp.ones((1,), dtype=jnp.float32),
    )
    y = jnp.array([0.3, -0.2, 0.8, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=precond_conf,
        verbose=False,
    )

    assert seen_diag_stacks
    assert all(diag_stack is None for diag_stack in seen_diag_stacks)


def test_fit_reml_uses_projected_core_residual_slq_when_requested(monkeypatch):
    calls = {"raw": 0, "residual": 0}

    def fake_raw(*_args, **_kwargs):
        calls["raw"] += 1
        return jnp.asarray(0.0, dtype=jnp.float32)

    def fake_residual(*_args, **_kwargs):
        calls["residual"] += 1
        return jnp.asarray(0.0, dtype=jnp.float32)

    monkeypatch.setattr(REML, "_slq_logdet", fake_raw)
    monkeypatch.setattr(REML, "_slq_logdet_projected_core_residual", fake_residual)

    def mv(v):
        return 2.0 * v

    precond_conf = ProjectedCorePrecondConf(
        U=jnp.eye(4, dtype=jnp.float32),
        core_atoms=jnp.stack([jnp.eye(4, dtype=jnp.float32)], axis=0),
        total_rank=4,
        n_grm=1,
        diag_mode="scalar_identity",
        diag_atoms=jnp.zeros((1,), dtype=jnp.float32),
    )

    y = jnp.array([0.5, -0.1, 1.2, 0.3], dtype=jnp.float32)
    param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.zeros((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=0,
        slq_samples=2,
        slq_m=3,
        slq_mode="projected_core_residual",
        precond_conf=precond_conf,
        verbose=False,
    )

    assert param.shape == (2,)
    assert history == []
    assert calls["residual"] == 1
    assert calls["raw"] == 0


def test_fit_reml_refreshes_preconditioner_for_each_candidate_eval(monkeypatch):
    refresh_calls = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, pvec, kwargs
        call_idx = fake_eval_once.calls
        fake_eval_once.calls += 1
        ll = jnp.asarray(float(call_idx), dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        fi = jnp.eye(2, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    fake_eval_once.calls = 0

    def fake_newton_step(grad, fi):
        del grad, fi
        return jnp.array([0.2, -0.2], dtype=jnp.float32)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)
    monkeypatch.setattr(
        REML,
        "_compute_traces_from_pcg",
        lambda warm_all, ctx: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((ctx.G,), dtype=jnp.float32),
        ),
    )

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)

    def refresh_fn(param):
        refresh_calls.append(np.asarray(param))
        return None

    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=2,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        precond_refresh_fn=refresh_fn,
        precond_refresh_reldp=0.1,
        verbose=False,
    )

    assert len(refresh_calls) == 2


def test_fit_reml_refreshes_candidate_even_if_rel_dll_stop_triggers(monkeypatch):
    refresh_calls = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, pvec, kwargs
        call_idx = fake_eval_once.calls
        fake_eval_once.calls += 1
        ll = jnp.asarray(1.0 if call_idx == 0 else 1.0005, dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        fi = jnp.eye(2, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    fake_eval_once.calls = 0

    def fake_newton_step(grad, fi):
        del grad, fi
        return jnp.array([0.3, -0.3], dtype=jnp.float32)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)
    monkeypatch.setattr(
        REML,
        "_compute_traces_from_pcg",
        lambda warm_all, ctx: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((ctx.G,), dtype=jnp.float32),
        ),
    )

    def mv(v):
        return v

    def refresh_fn(param):
        refresh_calls.append(np.asarray(param))
        return None

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    _param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=2,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        precond_refresh_fn=refresh_fn,
        precond_refresh_reldp=0.1,
        verbose=False,
    )

    assert len(refresh_calls) == 1
    assert len(history) == 1
    assert history[0]["stop_reason"] == "rel_dll"
    assert history[0]["precond_refreshed"] is True


def test_fit_reml_refreshes_using_theta_try_during_line_search(monkeypatch):
    refresh_calls = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, kwargs
        theta_g = float(pvec[0])
        ll = jnp.asarray(-((theta_g - 0.5) ** 2), dtype=jnp.float32)
        grad = jnp.zeros_like(pvec)
        fi = jnp.eye(pvec.shape[0], dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del fi
        out = np.zeros((grad.shape[0],), dtype=np.float32)
        out[0] = -1.0
        return jnp.asarray(out)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)
    monkeypatch.setattr(
        REML,
        "_compute_traces_from_pcg",
        lambda warm_all, ctx: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((ctx.G,), dtype=jnp.float32),
        ),
    )

    def mv(v):
        return v

    def refresh_fn(param):
        refresh_calls.append(np.asarray(param))
        return None

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        param_init=jnp.array([0.8, 0.2], dtype=jnp.float32),
        precond_conf=None,
        precond_refresh_fn=refresh_fn,
        precond_refresh_reldp=0.1,
        verbose=False,
    )

    assert len(refresh_calls) == 2
    assert np.isclose(refresh_calls[0][0], 0.0)
    assert np.isclose(refresh_calls[1][0], 0.3)


def test_fit_reml_can_return_exact_zero_genetic_component(monkeypatch):
    seen_params = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, kwargs
        seen_params.append(np.asarray(pvec))
        ll = jnp.asarray(float(len(seen_params) - 1), dtype=jnp.float32)
        grad = jnp.zeros_like(pvec)
        fi = jnp.eye(pvec.shape[0], dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        if grad.shape[0] == 2:
            return jnp.array([-1.0, 0.0], dtype=jnp.float32)
        del fi
        return jnp.array([0.0], dtype=jnp.float32)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        verbose=False,
    )

    assert len(seen_params) >= 2
    assert np.isclose(seen_params[1][0], 0.0)
    assert np.isclose(float(param[0]), 0.0)
    assert history[0]["n_freeze"] == 1


def test_fit_reml_backtracks_projected_step_when_full_step_lowers_ll(monkeypatch):
    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, kwargs
        theta_g = float(pvec[0])
        ll = jnp.asarray(-((theta_g - 0.5) ** 2), dtype=jnp.float32)
        grad = jnp.zeros_like(pvec)
        fi = jnp.eye(pvec.shape[0], dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del fi
        out = np.zeros((grad.shape[0],), dtype=np.float32)
        out[0] = -1.0
        return jnp.asarray(out)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        param_init=jnp.array([0.8, 0.2], dtype=jnp.float32),
        precond_conf=None,
        verbose=False,
    )

    assert np.isclose(float(param[0]), 0.3)
    assert history[0]["accepted"] is True
    assert history[0]["line_search_trials"] == 2
    assert np.isclose(history[0]["alpha_max"], 1.0)
    assert np.isclose(history[0]["step_alpha"], 0.5)


def test_fit_reml_backtracks_scaled_full_step_without_freezing(monkeypatch):
    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, kwargs
        theta_g = float(pvec[0])
        if np.isclose(theta_g, 0.8, atol=1e-6):
            ll_val = 0.0
        elif np.isclose(theta_g, 0.0, atol=1e-6):
            ll_val = -1.0
        elif np.isclose(theta_g, 0.3, atol=1e-6):
            ll_val = 0.5
        elif np.isclose(theta_g, 0.4, atol=1e-6):
            ll_val = -0.5
        else:
            raise AssertionError(f"unexpected theta_g={theta_g}")
        ll = jnp.asarray(ll_val, dtype=jnp.float32)
        grad = jnp.zeros_like(pvec)
        fi = jnp.eye(pvec.shape[0], dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del fi
        if grad.shape[0] == 2:
            return jnp.asarray([-1.0, 0.0], dtype=jnp.float32)
        return jnp.asarray([0.0], dtype=jnp.float32)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    param, history = fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        param_init=jnp.array([0.8, 0.2], dtype=jnp.float32),
        precond_conf=None,
        verbose=False,
    )

    assert np.isclose(float(param[0]), 0.3)
    assert history[0]["accepted"] is True
    assert history[0]["line_search_trials"] == 2
    assert np.isclose(history[0]["step_alpha"], 0.5)
    assert history[0]["n_freeze"] == 0


def test_fit_reml_reuses_trial_warm_start_during_line_search(monkeypatch):
    seen_warms = []
    seen_ready = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, pvec
        seen_ready.append(bool(kwargs["warm_ready"]))
        seen_warms.append(np.asarray(warm_all))
        call_idx = len(seen_warms)
        ll_values = [0.0, -1.0, 0.5]
        ll = jnp.asarray(ll_values[min(call_idx - 1, len(ll_values) - 1)], dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        fi = jnp.eye(2, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        warm_next = jnp.full_like(warm_all, float(call_idx))
        return ll, grad, fi, 0, warm_next, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del fi
        out = np.zeros((grad.shape[0],), dtype=np.float32)
        out[0] = -1.0
        return jnp.asarray(out)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        param_init=jnp.array([0.8, 0.2], dtype=jnp.float32),
        precond_conf=None,
        verbose=False,
    )

    assert seen_ready[:3] == [False, True, True]
    assert np.allclose(seen_warms[2], np.full_like(seen_warms[2], 2.0))


def test_fit_reml_reuses_ai_warm_start_during_line_search(monkeypatch):
    seen_ai_warms = []
    seen_ai_ready = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, pvec, warm_all
        seen_ai_ready.append(bool(kwargs["warm_ai_ready"]))
        seen_ai_warms.append(np.asarray(kwargs["warm_ai"]))
        call_idx = len(seen_ai_warms)
        ll_values = [0.0, -1.0, 0.5]
        ll = jnp.asarray(ll_values[min(call_idx - 1, len(ll_values) - 1)], dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        fi = jnp.eye(2, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        warm_next = jnp.zeros((4, 3), dtype=jnp.float32)
        warm_ai_next = jnp.full_like(kwargs["warm_ai"], float(call_idx))
        return ll, grad, fi, 0, warm_next, warm_ai_next, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del fi
        out = np.zeros((grad.shape[0],), dtype=np.float32)
        out[0] = -1.0
        return jnp.asarray(out)

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        param_init=jnp.array([0.8, 0.2], dtype=jnp.float32),
        precond_conf=None,
        verbose=False,
    )

    assert seen_ai_ready[:3] == [False, True, True]
    assert np.allclose(seen_ai_warms[2], np.full_like(seen_ai_warms[2], 2.0))


def test_fit_reml_keeps_warm_start_after_preconditioner_refresh(monkeypatch):
    warm_ready_flags = []

    def fake_eval_once(ctx, pvec, warm_all, **kwargs):
        del ctx, pvec
        warm_ready_flags.append(bool(kwargs["warm_ready"]))
        call_idx = len(warm_ready_flags)
        ll = jnp.asarray(float(call_idx - 1), dtype=jnp.float32)
        grad = jnp.zeros((2,), dtype=jnp.float32)
        fi = jnp.eye(2, dtype=jnp.float32)
        tr_hinv = jnp.asarray(0.0, dtype=jnp.float32)
        tr_hinv_k = jnp.zeros((1,), dtype=jnp.float32)
        logdet = jnp.asarray(0.0, dtype=jnp.float32)
        return ll, grad, fi, 0, warm_all, tr_hinv, tr_hinv_k, logdet

    def fake_newton_step(grad, fi):
        del grad, fi
        return jnp.array([0.2, -0.2], dtype=jnp.float32)

    precond_conf = ProjectedCorePrecondConf(
        U=jnp.empty((4, 0), dtype=jnp.float32),
        core_atoms=jnp.empty((1, 0, 0), dtype=jnp.float32),
        total_rank=0,
        n_grm=1,
        diag_mode="scalar_identity",
        diag_atoms=jnp.ones((1,), dtype=jnp.float32),
    )

    refreshes = []

    def refresh_fn(param):
        refreshes.append(np.asarray(param))
        return precond_conf if len(refreshes) == 1 else None

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(REML, "_newton_step", fake_newton_step)

    def mv(v):
        return v

    y = jnp.array([0.2, -0.4, 1.0, 0.1], dtype=jnp.float32)
    fit_reml(
        y=y,
        K_mvs=[mv],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=2,
        slq_samples=2,
        slq_m=3,
        precond_conf=precond_conf,
        precond_refresh_fn=refresh_fn,
        precond_refresh_reldp=0.1,
        verbose=False,
    )

    assert len(refreshes) >= 1
    assert warm_ready_flags[:3] == [False, True, True]
