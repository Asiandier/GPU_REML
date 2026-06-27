"""
reml.py — REML fitting via AI-REML / Fisher scoring.

Performance architecture (ARCH-FIX-H + improvements)
-----------------------------------------------------
All kv() calls happen in Python scope — never inside traced JAX primitives.
Sub-computations that don't involve kv are JIT-compiled.

Key optimisations over previous version:
  • SLQ Lanczos: recurrence arithmetic fused into `_lanczos_step_jit`
    (one JIT dispatch per step instead of 5–7 individual dispatches).
  • SLQ eigendecomposition: fused into `_slq_tridiag_logdet_jit`.
  • _hv_apply: lightweight, called in Python scope (no JIT needed since
    the individual kv() and scalar ops are already dispatched efficiently).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

logger = logging.getLogger(__name__)

Array = jnp.ndarray
FI_SYSTEM_RIDGE = 1e-4

from .pcg import pcg_solve
from .precond import (
    ProjectedCorePrecondConf,
    build_projected_core_runtime,
    make_precond,
    make_projected_core_precond_from_runtime,
    projected_core_apply_invsqrt,
    projected_core_logdet,
    scalar_diag_from_precond_conf,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class REMLContext:
    n: int
    G: int
    K_mvs: tuple
    weighted_hv: Optional[Callable[[Array, Array, Array], Array]]
    stacked_kv: Optional[Callable[[Array], Array]]
    diag_stack: Optional[Array]
    xmat: Optional[Array]
    y: Array
    rhs_const: Array
    y_col: int
    rand_stop: int
    n_XyZ_cols: int
    n_GZrand_components: int
    R_rand: int
    precond_conf: Optional[ProjectedCorePrecondConf]
    kvrand_stack: Optional[Array] = None
    diag_atoms: Optional[Array] = None


@dataclass
class AverageInfoMatrix:
    mat: Array
    ridge: float = FI_SYSTEM_RIDGE
    stats: Optional["FisherSolveStats"] = None


@dataclass
class FisherSolveStats:
    free_dim: int = 0
    frozen_genetic: int = 0
    ai_pcg_iters: int = 0
    ai_elapsed_sec: float = 0.0
    ws_resolve_count: int = 0
    ws_fixed_total: int = 0
    ws_trace: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "free_dim": int(self.free_dim),
            "frozen_genetic": int(self.frozen_genetic),
            "ai_pcg_iters": int(self.ai_pcg_iters),
            "ai_elapsed_sec": float(self.ai_elapsed_sec),
            "ws_resolve_count": int(self.ws_resolve_count),
            "ws_fixed_total": int(self.ws_fixed_total),
            "ws_trace": str(self.ws_trace),
        }


def _reset_fisher_solve_stats(
    stats: FisherSolveStats,
    *,
    free_dim: int,
    frozen_genetic: int,
) -> None:
    stats.free_dim = int(free_dim)
    stats.frozen_genetic = int(frozen_genetic)


def _reset_fisher_workingset_stats(stats: FisherSolveStats) -> None:
    stats.ws_resolve_count = 0
    stats.ws_fixed_total = 0
    stats.ws_trace = ""


def standardize_response(y: Array) -> tuple[Array, Array, Array]:
    """Standardize a phenotype vector using the same convention as fit_reml."""
    y = jnp.asarray(y, dtype=jnp.float32).reshape(-1)
    y_mean = jnp.mean(y)
    y_scale = jnp.std(y) + jnp.asarray(1e-6, dtype=y.dtype)
    return (y - y_mean) / y_scale, y_mean, y_scale


def _scalar_diag_from_diag_list(diag_list: Sequence[Array]) -> Optional[Array]:
    """Return per-component scalar diagonal atoms when every diag entry is constant."""
    atoms: list[float] = []
    for d in diag_list:
        darr = np.asarray(jax.device_get(jnp.asarray(d, dtype=jnp.float32)))
        if darr.size == 0:
            atoms.append(0.0)
            continue
        if darr.ndim == 0:
            atoms.append(float(darr))
            continue
        flat = darr.reshape(-1)
        first = float(flat[0])
        if not np.all(flat == first):
            return None
        atoms.append(first)
    if not atoms:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.asarray(atoms, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Pure-XLA helpers (JIT-compiled, no kv inside)
# ---------------------------------------------------------------------------

def _stable_cho_factor_spd(
    A: Array,
    *,
    base_jitter: float = 1e-6,
    max_tries: int = 8,
) -> tuple:
    """
    Robust Cholesky factorisation for near-SPD matrices.

    Symmetrises A and escalates diagonal jitter geometrically until
    Cholesky diagonal entries are finite and positive.
    """
    A_sym = 0.5 * (A + A.T)
    p = A_sym.shape[0]
    if p == 0:
        raise ValueError("stable_cho_factor_spd requires a non-empty square matrix.")

    scale = float(jnp.mean(jnp.abs(jnp.diag(A_sym))))
    if (not math.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    eye = jnp.eye(p, dtype=A_sym.dtype)

    for k in range(max_tries):
        jitter = jnp.asarray(base_jitter * scale * (10.0 ** k), dtype=A_sym.dtype)
        A_try = A_sym + jitter * eye
        chol = jsp.linalg.cho_factor(A_try, lower=True, check_finite=False)
        d = jnp.diag(chol[0])
        if bool(jnp.all(jnp.isfinite(d)) and jnp.all(d > 0)):
            return chol

    raise FloatingPointError(
        "Failed to stabilize XtHinvX Cholesky with jitter escalation."
    )


# ---------------------------------------------------------------------------
# SLQ logdet — Lanczos in Python loop, arithmetic fused via JIT
# ---------------------------------------------------------------------------

@jax.jit
def _lanczos_step_jit(
    v: Array,          # (n, S) current Lanczos vector
    w_raw: Array,      # (n, S) = H @ v  (already computed in Python scope)
    prev: Array,       # (n, S) previous Lanczos vector
    beta_prev: Array,  # (S,)   previous beta
) -> tuple[Array, Array, Array]:
    """
    Fused Lanczos recurrence step.

    Combines β-correction, α-computation, re-orthogonalisation against v,
    norm, and safe normalisation into a single JIT-compiled kernel.
    Returns (v_next, alpha, norm_w).
    """
    w     = w_raw - prev * beta_prev[None, :]
    alpha = jnp.sum(v * w, axis=0)
    w     = w - v * alpha[None, :]
    norm_w = jnp.linalg.norm(w, axis=0)
    safe   = jnp.where(norm_w > 0, norm_w, jnp.ones_like(norm_w))
    v_next = w / safe[None, :]
    return v_next, alpha, norm_w


@jax.jit
def _slq_tridiag_logdet_jit(
    alphas: Array,      # (m, S)
    betas: Array,       # (m-1, S)
    z_norm_sq: Array,   # () = float(n)
    nsamples_f: Array,  # () = float(nsamples)
) -> Array:
    """
    Diagonalise the batched tridiagonal Lanczos matrices and return the
    SLQ logdet estimate.
    """
    m = alphas.shape[0]
    T_batch = jnp.zeros((alphas.shape[1], m, m), dtype=alphas.dtype)
    idx = jnp.arange(m)
    T_batch = T_batch.at[:, idx, idx].set(alphas.T)
    off = betas.T
    off_idx = jnp.arange(m - 1)
    T_batch = T_batch.at[:, off_idx + 1, off_idx].set(off)
    T_batch = T_batch.at[:, off_idx, off_idx + 1].set(off)
    evals, evecs = jnp.linalg.eigh(T_batch)
    eval_floor = jnp.asarray(
        max(float(jnp.finfo(alphas.dtype).eps) * float(m), 1e-6),
        dtype=alphas.dtype,
    )
    e1   = evecs[:, 0, :]                                   # (S, m)
    ests = z_norm_sq * jnp.sum(
        e1 * e1 * jnp.log(jnp.clip(evals, eval_floor)), axis=1
    )
    return jnp.sum(ests) / nsamples_f


def _slq_logdet(
    Hv_fn: Callable,
    n_dim: int,
    key: "jax.random.PRNGKey",
    nsamples: int = 30,
    m: int = 50,
) -> Array:
    """
    Stochastic Lanczos Quadrature for logdet(H).

    Hv_fn is called in Python scope — concrete arrays, no tracing.
    The Lanczos recurrence arithmetic is fused into `_lanczos_step_jit`
    (one JIT dispatch per Lanczos step instead of 5–7 separate dispatches).
    """
    keys_b = jax.random.split(key, nsamples)
    z = jax.vmap(
        lambda k: jax.random.rademacher(k, (n_dim,), dtype=jnp.int32)
    )(keys_b).astype(jnp.float32)
    z_norm_sq = jnp.array(n_dim, dtype=jnp.float32)
    z_norm    = jnp.sqrt(z_norm_sq)
    v = (z / z_norm).T                                      # (n, S)

    # Collect coefficients in Python lists, stack once at end.
    # Avoids m functional array updates (each allocating a new (m, S) array).
    alpha_list = []
    beta_list  = []
    prev      = jnp.zeros_like(v)
    beta_prev = jnp.zeros((nsamples,), dtype=jnp.float32)

    for i in range(m):
        w_raw = Hv_fn(v)                                    # kv in Python scope
        v_next, alpha, norm_w = _lanczos_step_jit(v, w_raw, prev, beta_prev)
        alpha_list.append(alpha)
        if i < m - 1:
            beta_list.append(norm_w)
        prev      = v
        beta_prev = norm_w
        v         = v_next

    alphas = jnp.stack(alpha_list, axis=0)                  # (m, S)
    betas  = jnp.stack(beta_list,  axis=0)                  # (m-1, S)
    return _slq_tridiag_logdet_jit(
        alphas, betas, z_norm_sq, jnp.float32(nsamples)
    )


def _slq_logdet_projected_core_residual(
    Hv_fn: Callable,
    precond_runtime,
    n_dim: int,
    key: "jax.random.PRNGKey",
    nsamples: int = 30,
    m: int = 50,
) -> Array:
    """
    Residual SLQ for logdet(H):

        logdet(H) = logdet(M) + logdet(M^{-1/2} H M^{-1/2})

    with M given by the current projected-core preconditioner.
    """
    def Bv(v: Array) -> Array:
        v_left = projected_core_apply_invsqrt(precond_runtime, v)
        hv = Hv_fn(v_left)
        return projected_core_apply_invsqrt(precond_runtime, hv)

    return projected_core_logdet(precond_runtime, n_dim) + _slq_logdet(
        Bv, n_dim, key, nsamples=nsamples, m=m
    )


# ---------------------------------------------------------------------------
# H·V — called in Python scope
# ---------------------------------------------------------------------------

def _hv_apply(
    K_mvs: tuple,
    theta_g: Array,
    theta_e: Array,
    V: Array,
) -> Array:
    """H·V = Σ_i theta_g[i] K_i(V) + theta_e * V."""
    acc = theta_e * V
    for i, mv in enumerate(K_mvs):
        acc = acc + theta_g[i] * mv(V)
    return acc


def _apply_genetic_stack(
    ctx: REMLContext,
    V: Array,
    component_idx_np: Optional[np.ndarray] = None,
) -> Array:
    """Return stacked genetic-component matvecs with shape (G_sel, n, rhs)."""
    if component_idx_np is None:
        component_idx_np = np.arange(ctx.G, dtype=np.int64)
    else:
        component_idx_np = np.asarray(component_idx_np, dtype=np.int64)

    if component_idx_np.size == 0:
        return jnp.zeros((0, V.shape[0], V.shape[1]), dtype=V.dtype)
    if component_idx_np.size == ctx.G and ctx.stacked_kv is not None:
        return ctx.stacked_kv(V)
    return jnp.stack([ctx.K_mvs[int(i)](V) for i in component_idx_np], axis=0)


# ---------------------------------------------------------------------------
# eval_once: Python-level orchestration (NOT jax.jit wrapped as a whole)
# ---------------------------------------------------------------------------

def _compute_traces_from_pcg(
    sol_all: Array,
    ctx: REMLContext,
) -> tuple[Array, Array]:
    """Estimate tr(H⁻¹) and tr(H⁻¹ K_i) from the PCG solution.

    Uses the random vectors already embedded in rhs_const and their
    PCG solutions. When cached K_i @ Vrand blocks are available, the
    component traces need no extra PCG columns.

    Returns:
        tr_Hinv:     () scalar  ≈ tr(H⁻¹)
        tr_Hinv_K:   (G,) array ≈ [tr(H⁻¹ K_0), ..., tr(H⁻¹ K_{G-1})]
    """
    vrand_start = ctx.y_col + 1          # column right after y in RHS
    vrand_stop  = ctx.rand_stop           # column after Vrand
    R = ctx.R_rand

    # H⁻¹ Vrand  (n, R)
    HinvVrand = sol_all[:, vrand_start:vrand_stop]
    # Vrand is stored in rhs_const at the same column range
    Vrand_cols = ctx.rhs_const[:, vrand_start:vrand_stop]

    # tr(H⁻¹) ≈ (1/R) Σ_r v_r^T (H⁻¹ v_r)
    tr_Hinv = jnp.sum(Vrand_cols * HinvVrand) / float(R)

    if ctx.kvrand_stack is not None:
        tr_Hinv_K = jnp.einsum(
            "nr,inr->i",
            HinvVrand,
            ctx.kvrand_stack,
            precision=jax.lax.Precision.HIGH,
        ) / float(R)
    else:
        # Backward-compatible fallback for callers that still append H⁻¹K_iZ
        # blocks to the PCG solution.
        HinvGZrand = sol_all[:, ctx.n_XyZ_cols:]  # (n, G*R)
        G = ctx.n_GZrand_components
        HinvGZrand = HinvGZrand.reshape(HinvGZrand.shape[0], G, R)
        tr_Hinv_K = jnp.sum(
            Vrand_cols[:, None, :] * HinvGZrand, axis=(0, 2)
        ) / float(R)

    return tr_Hinv, tr_Hinv_K


def _eval_once(
    ctx: REMLContext,
    pvec: Array,
    warm_all: Array,
    warm_ai: Optional[Array] = None,
    *,
    key_slq: "jax.random.PRNGKey",
    minq_tol: float,
    maxiter: int,
    precond_eps: float,
    slq_samples: int,
    slq_m: int,
    slq_mode: str = "raw",
    warm_ready: bool = False,
    warm_ai_ready: bool = False,
    taylor_logdet: Optional[Array] = None,
    compute_traces: bool = True,
) -> tuple:
    """One REML objective evaluation.  kv() only in Python scope.

    If *taylor_logdet* is provided (not None), the SLQ logdet computation
    is skipped and this pre-computed value is used instead.
    """
    theta_g = pvec[:-1]
    theta_e = pvec[-1]

    def Hv(V: Array) -> Array:
        if ctx.weighted_hv is not None:
            return ctx.weighted_hv(theta_g, theta_e, V)
        return _hv_apply(ctx.K_mvs, theta_g, theta_e, V)

    diag_H_scalar = scalar_diag_from_precond_conf(ctx.precond_conf, theta_g, theta_e)
    if diag_H_scalar is not None:
        diag_H = diag_H_scalar
    else:
        if ctx.diag_atoms is not None:
            diag_H = theta_e + jnp.dot(theta_g, ctx.diag_atoms)
        else:
            if ctx.diag_stack is None:
                raise ValueError(
                    "diag_stack is required when the preconditioner does not provide a scalar diagonal."
                )
            diag_H = theta_e + jnp.tensordot(theta_g, ctx.diag_stack, axes=1)
    use_residual_slq = (
        slq_mode == "projected_core_residual"
        and taylor_logdet is None
        and ctx.precond_conf is not None
        and getattr(ctx.precond_conf, "diag_mode", None) == "scalar_identity"
        and getattr(ctx.precond_conf, "total_rank", 0) > 0
    )
    precond_runtime = build_projected_core_runtime(
        ctx.precond_conf,
        theta_g,
        diag_H,
        eps=precond_eps,
        need_invsqrt=use_residual_slq,
    )
    if precond_runtime is not None:
        M_cur = make_projected_core_precond_from_runtime(precond_runtime)
    else:
        M_cur = make_precond(ctx.precond_conf, theta_g, diag_H, eps=precond_eps)

    rhs_all = ctx.rhs_const

    # Use Python flag — avoids GPU→CPU sync that jnp.isfinite() would cause.
    if warm_ready:
        if M_cur is not None:
            main_resid = rhs_all - Hv(warm_all)
            X0_all = warm_all + M_cur(main_resid)
        else:
            X0_all = warm_all
    elif M_cur is not None:
        X0_all = M_cur(rhs_all)
    else:
        X0_all = jnp.zeros_like(rhs_all)

    # ---- PCG: Python while loop, Hv per iteration in Python scope ----------
    sol_all, _, k_pcg = pcg_solve(
        Hv, rhs_all, M=M_cur, tol=minq_tol, maxiter=maxiter, X0=X0_all,
    )
    warm_all_next = sol_all

    # ---- Projection (pure XLA) ---------------------------------------------
    HinvXyZ = sol_all[:, :ctx.n_XyZ_cols]
    if ctx.xmat is not None and ctx.xmat.shape[1] > 0:
        x_cols = ctx.xmat.shape[1]
        HinvX   = HinvXyZ[:, :x_cols]
        XtHinvX = ctx.xmat.T @ HinvX
        chol    = _stable_cho_factor_spd(XtHinvX)

        def proj(mat):
            mid = jsp.linalg.cho_solve(chol, ctx.xmat.T @ mat, check_finite=False)
            return mat - HinvX @ mid
    else:
        proj = lambda mat: mat

    PYstar = proj(HinvXyZ[:, ctx.y_col : ctx.rand_stop])

    # K_i(P[y | Z]) — kv in Python scope. The final stacked row is the
    # residual component I(P[y | Z]) = P[y | Z].
    GPYstar_genetic = _apply_genetic_stack(ctx, PYstar)
    GPYstar_stack = jnp.concatenate([GPYstar_genetic, PYstar[None, :, :]], axis=0)
    GPZrand_stack = GPYstar_stack[:, :, 1:]  # (G+1, n, R)
    Vrand_cols = ctx.rhs_const[:, ctx.y_col + 1 : ctx.rand_stop]

    # ---- REML statistics (pure XLA) ----------------------------------------
    q_sel = jnp.einsum(
        "n,in->i",
        PYstar[:, 0],
        GPYstar_stack[:, :, 0],
        precision=jax.lax.Precision.HIGH,
    )
    trace_pg_sel = jnp.einsum(
        "nr,inr->i",
        Vrand_cols,
        GPZrand_stack,
        precision=jax.lax.Precision.HIGH,
    ) / float(ctx.R_rand)

    yPy = jnp.dot(ctx.y, PYstar[:, 0])

    if ctx.xmat is not None and ctx.xmat.shape[1] > 0:
        logdet_x = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(chol[0]), 1e-30)))
    else:
        logdet_x = jnp.array(0.0, dtype=pvec.dtype)

    # ---- SLQ logdet — Python loop, Hv in Python scope ----------------------
    if taylor_logdet is not None:
        logdet = taylor_logdet
    elif use_residual_slq and precond_runtime is not None:
        logdet = _slq_logdet_projected_core_residual(
            Hv,
            precond_runtime,
            ctx.n,
            key_slq,
            nsamples=slq_samples,
            m=slq_m,
        )
    else:
        logdet = _slq_logdet(Hv, ctx.n, key_slq, nsamples=slq_samples, m=slq_m)

    fisher_stats = FisherSolveStats()
    fisher_stats.free_dim = int(ctx.G + 1)
    fisher_stats.frozen_genetic = 0
    GPy_cols = jnp.swapaxes(GPYstar_stack[:, :, 0], 0, 1)  # (n, G+1)
    if warm_ai_ready and warm_ai is not None and warm_ai.shape[1] == ctx.G + 1:
        if M_cur is not None:
            ai_resid = GPy_cols - Hv(warm_ai)
            X0_ai = warm_ai + M_cur(ai_resid)
        else:
            X0_ai = warm_ai
    elif M_cur is not None:
        X0_ai = M_cur(GPy_cols)
    else:
        X0_ai = jnp.zeros_like(GPy_cols)
    ai_t0 = time.perf_counter()
    HinvGPy, _, k_ai = pcg_solve(
        Hv, GPy_cols, M=M_cur, tol=minq_tol, maxiter=maxiter, X0=X0_ai,
    )
    warm_ai_next = HinvGPy
    fisher_stats.ai_pcg_iters = int(k_ai)
    fisher_stats.ai_elapsed_sec = time.perf_counter() - ai_t0
    PGPy_cols = proj(HinvGPy)

    # ---- Trace estimates for Taylor warm-start ------------------------------
    if compute_traces:
        tr_Hinv, tr_Hinv_K = _compute_traces_from_pcg(sol_all, ctx)
    else:
        tr_Hinv = jnp.asarray(jnp.nan, dtype=sol_all.dtype)
        tr_Hinv_K = jnp.full((ctx.G,), jnp.nan, dtype=sol_all.dtype)

    # ---- Assemble ll, grad, FI ----------------------------------------------
    scale = jnp.asarray(ctx.n, dtype=pvec.dtype)
    AI_sel = (
        0.5
        * jnp.einsum(
            "ni,nj->ij",
            GPy_cols,
            PGPy_cols,
            precision=jax.lax.Precision.HIGH,
        )
        / scale
    )
    AI = 0.5 * (AI_sel + AI_sel.T)
    FI = AverageInfoMatrix(
        mat=AI,
        stats=fisher_stats,
    )
    grad = 0.5 * (q_sel - trace_pg_sel) / scale
    ll    = -0.5 * (yPy + logdet + logdet_x) / scale

    return (
        ll,
        grad,
        FI,
        jnp.asarray(k_pcg, dtype=jnp.int32),
        warm_all_next,
        warm_ai_next,
        tr_Hinv,
        tr_Hinv_K,
        logdet,
    )


# ---------------------------------------------------------------------------
# Newton step (dense AI solve)
# ---------------------------------------------------------------------------


def _newton_step(grad: Array, FI) -> Array:
    if grad.shape[0] == 0:
        return grad
    if isinstance(FI, AverageInfoMatrix):
        FI_mat = FI.mat
        ridge = jnp.asarray(FI.ridge, dtype=FI_mat.dtype)
    else:
        FI_mat = FI
        ridge = jnp.asarray(FI_SYSTEM_RIDGE, dtype=FI_mat.dtype)
    FI_sym = 0.5 * (FI_mat + FI_mat.T)
    reg = ridge * jnp.eye(FI_sym.shape[0], dtype=FI_sym.dtype)
    chol = _stable_cho_factor_spd(FI_sym + reg)
    return jsp.linalg.cho_solve(chol, grad, check_finite=False)


def _freeze_mask(theta_g: Array, grad_g: Array, zero_tol: float) -> Array:
    zero_tol_arr = jnp.asarray(zero_tol, dtype=theta_g.dtype)
    return jnp.logical_and(theta_g <= zero_tol_arr, grad_g <= 0.0)


def _apply_fisher_system(FI, vec: Array) -> Array:
    if isinstance(FI, AverageInfoMatrix):
        FI_mat = FI.mat
        ridge = jnp.asarray(FI.ridge, dtype=FI_mat.dtype)
    else:
        FI_mat = FI
        ridge = jnp.asarray(FI_SYSTEM_RIDGE, dtype=FI_mat.dtype)
    FI_sym = 0.5 * (FI_mat + FI_mat.T)
    return FI_sym @ vec + ridge * vec


def _solve_reduced_fisher_step(
    param: Array,
    grad: Array,
    FI,
    *,
    freeze_mask_np: np.ndarray,
    fixed_step: Array,
) -> tuple[Array, np.ndarray]:
    G = param.shape[0] - 1
    free_g_idx_np = np.flatnonzero(~freeze_mask_np)
    free_idx_np = np.concatenate([free_g_idx_np, np.array([G], dtype=np.int64)])
    free_idx = jnp.asarray(free_idx_np, dtype=jnp.int32)

    rhs_f = grad[free_idx] - _apply_fisher_system(FI, fixed_step)[free_idx]

    if isinstance(FI, AverageInfoMatrix):
        if FI.stats is not None:
            _reset_fisher_solve_stats(
                FI.stats,
                free_dim=free_idx_np.shape[0],
                frozen_genetic=int(np.sum(freeze_mask_np)),
            )
        FI_ff = AverageInfoMatrix(
            mat=FI.mat[free_idx[:, None], free_idx[None, :]],
            ridge=FI.ridge,
            stats=FI.stats,
        )
    else:
        FI_ff = FI[free_idx[:, None], free_idx[None, :]]

    step_f = _newton_step(rhs_f, FI_ff)
    step_dir = fixed_step.at[free_idx].add(step_f)
    return step_dir, free_idx_np


def _projected_gradient_inf_norm(param: Array, grad: Array, zero_tol: float) -> Array:
    theta_g = param[:-1]
    grad_g = grad[:-1]
    zero_tol_arr = jnp.asarray(zero_tol, dtype=param.dtype)
    proj_g = jnp.where(theta_g > zero_tol_arr, grad_g, jnp.maximum(grad_g, 0.0))
    proj_all = jnp.concatenate([proj_g, grad[-1:]])
    return jnp.max(jnp.abs(proj_all))


def _projected_fisher_direction(
    param: Array,
    grad: Array,
    FI,
    *,
    genetic_zero_tol: float,
    trial_alpha: float = 1.0,
    workset_log_fn: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[Array, float, Array]:
    """Projected Fisher-scoring direction via reduced freeze-set resolves.

    The freeze set is recomputed from the current iterate using the KKT rule
    ``theta_i <= zero_tol and grad_i <= 0``. For a given trial step size
    ``trial_alpha``, any free genetic component that would cross below zero is
    added to the freeze set, fixed to hit the boundary exactly at that trial
    step, and the reduced Newton system is resolved on the remaining free set.
    """
    G = param.shape[0] - 1
    if isinstance(FI, AverageInfoMatrix) and FI.stats is not None:
        initial_freeze = np.asarray(
            _freeze_mask(param[:-1], grad[:-1], genetic_zero_tol),
            dtype=bool,
        )
        _reset_fisher_solve_stats(
            FI.stats,
            free_dim=int(param.shape[0] - np.sum(initial_freeze)),
            frozen_genetic=int(np.sum(initial_freeze)),
        )
        _reset_fisher_workingset_stats(FI.stats)
    freeze_mask_np = np.asarray(
        _freeze_mask(param[:-1], grad[:-1], genetic_zero_tol),
        dtype=bool,
    )
    if trial_alpha <= 0.0:
        raise ValueError("trial_alpha must be > 0.")

    fixed_step = jnp.zeros_like(param)
    resolve_idx = 0

    while True:
        resolve_idx += 1
        step_dir, free_idx_np = _solve_reduced_fisher_step(
            param,
            grad,
            FI,
            freeze_mask_np=freeze_mask_np,
            fixed_step=fixed_step,
        )
        theta_trial_g = param[:-1] + jnp.asarray(trial_alpha, dtype=param.dtype) * step_dir[:-1]
        violating_np = np.asarray(theta_trial_g < -genetic_zero_tol, dtype=bool) & (~freeze_mask_np)
        fixed_this_resolve = int(np.sum(violating_np))

        if isinstance(FI, AverageInfoMatrix) and FI.stats is not None:
            FI.stats.ws_resolve_count += 1
            FI.stats.ws_fixed_total += fixed_this_resolve
            resolve_str = (
                f"free={free_idx_np.shape[0]} freeze={int(np.sum(freeze_mask_np))} "
                f"add={fixed_this_resolve} drop=0"
            )
            FI.stats.ws_trace = (
                resolve_str if not FI.stats.ws_trace else f"{FI.stats.ws_trace} -> {resolve_str}"
            )
            _reset_fisher_solve_stats(
                FI.stats,
                free_dim=int(free_idx_np.shape[0]),
                frozen_genetic=int(np.sum(freeze_mask_np)),
            )
        if workset_log_fn is not None:
            workset_log_fn(
                {
                    "resolve_idx": resolve_idx,
                    "free_dim": int(free_idx_np.shape[0]),
                    "frozen_genetic": int(np.sum(freeze_mask_np)),
                    "fixed_this_resolve": fixed_this_resolve,
                }
            )

        if fixed_this_resolve == 0:
            return step_dir, 1.0, jnp.asarray(freeze_mask_np, dtype=bool)

        freeze_mask_np = np.logical_or(freeze_mask_np, violating_np)
        violating_idx = jnp.asarray(np.flatnonzero(violating_np), dtype=jnp.int32)
        boundary_step = -param[violating_idx] / jnp.asarray(trial_alpha, dtype=param.dtype)
        fixed_step = fixed_step.at[violating_idx].set(boundary_step)


def _apply_projected_step(
    param: Array,
    step_dir: Array,
    alpha: float,
    *,
    genetic_zero_tol: float,
    residual_floor: float,
) -> tuple[Array, Array]:
    alpha_arr = jnp.asarray(alpha, dtype=param.dtype)
    theta_g = param[:-1]
    step_g = step_dir[:-1]

    theta_g_updated = theta_g + alpha_arr * step_g
    theta_g_updated = jnp.maximum(theta_g_updated, 0.0)
    theta_g_updated = jnp.where(theta_g_updated <= genetic_zero_tol, 0.0, theta_g_updated)
    theta_e_updated = jnp.maximum(
        param[-1] + alpha_arr * step_dir[-1],
        jnp.asarray(residual_floor, dtype=param.dtype),
    )
    param_updated = jnp.concatenate([theta_g_updated, theta_e_updated[None]], axis=0)
    delta_param = param_updated - param
    return param_updated, delta_param


# ---------------------------------------------------------------------------
# fit_reml
# ---------------------------------------------------------------------------

def fit_reml(
    y: Array,
    K_mvs: Sequence[Callable[[Array], Array]],
    diag_list: Sequence[Array],
    covar: Optional[Array],
    n_rand_vec: int,
    maxiter: int,
    seed: int = 0,
    h2_init: float = 0.5,
    param_init: Optional[Array] = None,
    minq_iter: int = 50,
    slq_samples: int = 30,
    slq_m: int = 30,
    slq_mode: str = "raw",
    precond_conf: Optional[ProjectedCorePrecondConf] = None,
    precond_refresh_fn: Optional[Callable[[Array], Optional[ProjectedCorePrecondConf]]] = None,
    precond_refresh_reldp: float = 0.0,
    precond_eps: float = 1e-6,
    weighted_hv: Optional[Callable[[Array, Array, Array], Array]] = None,
    stacked_kv: Optional[Callable[[Array], Array]] = None,
    residual_floor: float = 1e-2,
    genetic_zero_tol: float = 1e-8,
    max_linesearch_trials: int = 3,
    optimizer: str = "strict",
    scoring_step_tol: float = 1e-4,
    rel_dll_tol: float = 1e-3,
    taylor_threshold: float = 0.01,
    warmup_pcg_tol: float = 1e-2,
    early_pcg_tol: float = 5e-3,
    default_pcg_tol: float = 1e-3,
    verbose: bool = True,
    log_detail: str = "full",
    return_diagnostics: bool = False,
):
    """Fit single-trait Gaussian REML with AI/Fisher updates.

    Notes
    -----
    The phenotype is standardized internally before optimization:
    ``y_std = (y - mean(y)) / std(y)``.
    The returned variance components and history therefore live on the
    standardized phenotype scale. Higher-level wrappers are responsible for
    carrying ``y_mean``/``y_scale`` when outputs need to be interpreted back on
    the original phenotype scale.
    """
    n = y.shape[0]
    G = len(K_mvs)
    if G != len(diag_list):
        raise ValueError("Length of K_mvs and diag_list must match.")
    if slq_samples <= 0:
        raise ValueError("fit_reml requires slq_samples > 0.")
    if slq_mode not in {"raw", "projected_core_residual"}:
        raise ValueError(
            f"Unsupported slq_mode={slq_mode!r}. Expected 'raw' or 'projected_core_residual'."
        )
    if precond_refresh_reldp < 0.0:
        raise ValueError("precond_refresh_reldp must be >= 0.")
    if residual_floor <= 0.0:
        raise ValueError("residual_floor must be > 0.")
    if genetic_zero_tol < 0.0:
        raise ValueError("genetic_zero_tol must be >= 0.")
    if max_linesearch_trials < 1:
        raise ValueError("max_linesearch_trials must be >= 1.")
    if log_detail not in {"compact", "full"}:
        raise ValueError("log_detail must be 'compact' or 'full'.")
    if optimizer not in {"strict", "smile_scoring"}:
        raise ValueError("optimizer must be 'strict' or 'smile_scoring'.")
    if scoring_step_tol < 0.0:
        raise ValueError("scoring_step_tol must be >= 0.")
    if rel_dll_tol < 0.0:
        raise ValueError("rel_dll_tol must be >= 0.")
    if taylor_threshold < 0.0:
        raise ValueError("taylor_threshold must be >= 0.")
    if warmup_pcg_tol <= 0.0 or early_pcg_tol <= 0.0 or default_pcg_tol <= 0.0:
        raise ValueError("PCG tolerances must be > 0.")
    _t0 = time.time()
    full_log = bool(verbose) and log_detail == "full"
    compact_log = bool(verbose) and log_detail == "compact"
    if verbose:
        logger.info(
            "[REML] start @ %s n=%d G=%d n_rand_vec=%d slq_samples=%d slq_mode=%s",
            datetime.now().isoformat(timespec='seconds'), n, G, n_rand_vec, slq_samples, slq_mode,
        )

    key_master = jax.random.PRNGKey(seed + 2026)
    key_master, key_vrand, key_slq_fixed = jax.random.split(key_master, 3)
    Vrand_fixed = (
        jax.random.rademacher(key_vrand, (n, n_rand_vec), dtype=jnp.int32)
        .astype(jnp.float32)
    )

    y, y_mean, y_scale = standardize_response(y)
    y_mean_host, y_scale_host = jax.device_get((y_mean, y_scale))

    xmat = (
        jnp.asarray(covar, dtype=jnp.float32)
        if covar is not None and covar.size > 0
        else None
    )

    if param_init is not None:
        p0 = jnp.asarray(param_init, dtype=jnp.float32).reshape(-1)
        if p0.shape[0] != G + 1:
            raise ValueError(
                f"param_init length mismatch: expected {G + 1}, got {p0.shape[0]}"
            )
        if not bool(jnp.all(jnp.isfinite(p0))):
            raise ValueError("param_init contains non-finite values.")
        theta_g0 = jnp.maximum(p0[:-1], 0.0)
        theta_g0 = jnp.where(theta_g0 <= genetic_zero_tol, 0.0, theta_g0)
        theta_e0 = jnp.maximum(p0[-1], jnp.asarray(residual_floor, dtype=p0.dtype))
        param = jnp.concatenate([theta_g0, theta_e0[None]], axis=0)
    else:
        theta_g = jnp.full((G,), h2_init / max(G, 1), dtype=jnp.float32)
        theta_e = jnp.array(1.0 - h2_init, dtype=jnp.float32)
        param = jnp.concatenate(
            [jnp.maximum(theta_g, 0.0), jnp.maximum(theta_e, jnp.asarray(residual_floor, dtype=theta_e.dtype))[None]],
            axis=0,
        )

    diag_atoms = None
    need_diag_stack = False
    if precond_conf is None or getattr(precond_conf, "diag_mode", None) != "scalar_identity":
        diag_atoms = _scalar_diag_from_diag_list(diag_list)
        need_diag_stack = diag_atoms is None
    diag_stack = jnp.stack(diag_list, axis=0) if need_diag_stack else None
    del diag_list

    # ---- Precompute K_i @ Vrand — constant across REML iterations ----------
    # These cached probe responses are reused for:
    #   1) Taylor logdet trace estimates tr(H^{-1} K_i)
    #   2) direct Hutchinson score traces tr(P K_i)
    if full_log:
        _t_kv_cache = time.time()
        logger.info("[REML] precompute K_i @ Vrand (%d kv passes) ...", G)
    if stacked_kv is not None:
        KVrand_stack = stacked_kv(Vrand_fixed)
    else:
        KVrand_stack = jnp.stack([mv(Vrand_fixed) for mv in K_mvs], axis=0)
    if full_log:
        logger.info("[REML] K_i @ Vrand done elapsed=%.1fs", time.time() - _t_kv_cache)

    # ---- Cache constant RHS [X | y | Vrand] once ---------------------------
    rhs_parts = []
    x_cols = 0
    if xmat is not None and xmat.shape[1] > 0:
        x_cols = xmat.shape[1]
        rhs_parts.append(xmat)
    y_col = x_cols
    rhs_parts.append(y[:, None])
    rand_stop = x_cols + 1 + n_rand_vec
    rhs_parts.append(Vrand_fixed)
    n_XyZ_cols = rand_stop
    rhs_const = jnp.concatenate(rhs_parts, axis=1)

    ctx = REMLContext(
        n=n, G=G,
        K_mvs=tuple(K_mvs),
        weighted_hv=weighted_hv,
        stacked_kv=stacked_kv,
        diag_stack=diag_stack,
        xmat=xmat,
        y=y,
        rhs_const=rhs_const,
        y_col=y_col,
        rand_stop=rand_stop,
        n_XyZ_cols=n_XyZ_cols,
        n_GZrand_components=G,
        R_rand=n_rand_vec,
        precond_conf=precond_conf,
        kvrand_stack=KVrand_stack,
        diag_atoms=diag_atoms,
    )
    del rhs_parts
    del KVrand_stack
    del Vrand_fixed

    def _pcg_tol(it: int) -> float:
        if it == 0:
            return warmup_pcg_tol
        if it == 1:
            return early_pcg_tol
        return default_pcg_tol

    n_warm_cols   = rhs_const.shape[1]
    warm_all      = jnp.full((n, n_warm_cols), jnp.nan, dtype=jnp.float32)
    warm_ai       = jnp.full((n, G + 1), jnp.nan, dtype=jnp.float32)
    warm_ready    = False       # Python flag — avoids GPU→CPU sync on NaN check
    warm_ai_ready = False

    if full_log:
        logger.info(
            "[REML] warm shapes: n_XyZ=%d n_warm=%d n_rand_vec=%d n_covar=%d",
            n_XyZ_cols, n_warm_cols, n_rand_vec,
            xmat.shape[1] if xmat is not None else 0,
        )

    def _run_eval(
        pvec,
        warm,
        warm_ai_cur,
        tol,
        warm_is_ready,
        warm_ai_is_ready,
        taylor_logdet_val=None,
        *,
        compute_traces=True,
    ):
        result = _eval_once(
            ctx, pvec, warm,
            warm_ai=warm_ai_cur,
            key_slq=key_slq_fixed,
            minq_tol=tol,
            maxiter=maxiter,
            precond_eps=precond_eps,
            slq_samples=slq_samples,
            slq_m=slq_m,
            slq_mode=slq_mode,
            warm_ready=warm_is_ready,
            warm_ai_ready=warm_ai_is_ready,
            taylor_logdet=taylor_logdet_val,
            compute_traces=compute_traces,
        )
        if len(result) == 8:
            ll, grad, FI, k_pcg, warm_next, tr_Hinv, tr_Hinv_K, logdet = result
            return ll, grad, FI, k_pcg, warm_next, warm_ai_cur, tr_Hinv, tr_Hinv_K, logdet
        return result

    # Warmup
    if full_log:
        _t_eval = time.time()
        logger.info("[REML] warmup eval @ %s", datetime.now().isoformat(timespec='seconds'))
    ll, grad, FI, k_pcg0, warm_all, warm_ai, tr_Hinv_cached, tr_Hinv_K_cached, logdet_cached = _run_eval(
        param, warm_all, warm_ai, warmup_pcg_tol, warm_ready, warm_ai_ready, compute_traces=True
    )
    warm_ready = True
    warm_ai_ready = True
    fi_finite = bool(
        jnp.all(jnp.isfinite(FI.mat))
        if isinstance(FI, AverageInfoMatrix)
        else jnp.all(jnp.isfinite(FI))
    )
    if not bool(jnp.isfinite(ll) and jnp.all(jnp.isfinite(grad)) and fi_finite):
        raise FloatingPointError("Non-finite warmup state (ll/grad/FI).")
    if full_log:
        ai_pcg0 = int(FI.stats.ai_pcg_iters) if isinstance(FI, AverageInfoMatrix) and FI.stats is not None else 0
        logger.info("[REML] warmup done elapsed=%.1fs pcg=%d ai_pcg=%d", time.time() - _t_eval, int(k_pcg0), ai_pcg0)
    elif compact_log:
        ll0_host = float(jax.device_get(ll))
        ai_pcg0 = int(FI.stats.ai_pcg_iters) if isinstance(FI, AverageInfoMatrix) and FI.stats is not None else 0
        logger.info("[REML] warmup: ll=%.6e pcg=%d ai_pcg=%d", ll0_host, int(k_pcg0), ai_pcg0)

    stop_reason = "max_iter"
    history: list[dict] = []
    for it in range(minq_iter):
        tol_cur = _pcg_tol(it)
        iter_t0 = time.time()
        iter_precond_refreshed = False
        trial_count = 0

        def _log_workset_resolve(info: dict[str, object]) -> None:
            if not full_log:
                return
            if trial_count <= 1:
                logger.info(
                    "[REML] iter %d workset-resolve %d free=%d freeze=%d fix=%d",
                    it + 1,
                    int(info["resolve_idx"]),
                    int(info["free_dim"]),
                    int(info["frozen_genetic"]),
                    int(info["fixed_this_resolve"]),
                )
            else:
                logger.info(
                    "[REML] iter %d trial %d workset-resolve %d free=%d freeze=%d fix=%d",
                    it + 1,
                    trial_count,
                    int(info["resolve_idx"]),
                    int(info["free_dim"]),
                    int(info["frozen_genetic"]),
                    int(info["fixed_this_resolve"]),
                )

        accepted = False
        alpha_max = 1.0
        alpha_try = alpha_max
        alpha_used = alpha_try
        ls_trace: list[tuple[float, float, bool, int, int, bool]] = []
        ll_new = ll
        grad_new = grad
        FI_new = FI
        warm_next = warm_all
        warm_ai_next = warm_ai
        tr_Hinv_new = tr_Hinv_cached
        tr_Hinv_K_new = tr_Hinv_K_cached
        logdet_new = logdet_cached
        eval_elapsed = 0.0
        k_pcg = 0
        eval_ai_pcg = 0
        use_taylor = False
        max_rel_dp = 0.0
        delta_param = jnp.zeros_like(param)
        param_updated = param
        step_dir = jnp.zeros_like(param)
        freeze_mask = jnp.zeros((G,), dtype=bool)
        step_elapsed = 0.0
        step_fisher_stats_dict = FisherSolveStats(
            free_dim=int(param.shape[0]),
            frozen_genetic=0,
        ).to_dict()
        trial_warm = warm_all
        trial_warm_ai = warm_ai
        trial_warm_ready = warm_ready
        trial_warm_ai_ready = warm_ai_ready

        trial_limit = 1 if optimizer == "smile_scoring" else max_linesearch_trials
        while trial_count < trial_limit:
            trial_count += 1
            alpha_used = alpha_try
            step_t0 = time.perf_counter()
            step_dir, alpha_max, freeze_mask = _projected_fisher_direction(
                param,
                grad,
                FI,
                genetic_zero_tol=genetic_zero_tol,
                trial_alpha=alpha_used,
                workset_log_fn=_log_workset_resolve,
            )
            step_elapsed += time.perf_counter() - step_t0
            frozen_genetic_count = int(np.sum(np.asarray(freeze_mask)))
            if isinstance(FI, AverageInfoMatrix) and FI.stats is not None:
                step_fisher_stats_dict = FI.stats.to_dict()
            else:
                step_fisher_stats_dict = FisherSolveStats(
                    free_dim=int(param.shape[0] - frozen_genetic_count),
                    frozen_genetic=frozen_genetic_count,
                ).to_dict()
            param_updated, delta_param = _apply_projected_step(
                param,
                step_dir,
                alpha_used,
                genetic_zero_tol=genetic_zero_tol,
                residual_floor=residual_floor,
            )
            trial_use_taylor = False
            taylor_ld = None
            param_scale = jnp.maximum(jnp.abs(param), 1e-8)
            step_norm_arr = jnp.linalg.norm(delta_param)
            max_rel_dp_arr = jnp.max(jnp.abs(delta_param) / param_scale)
            step_norm, max_rel_dp = (
                float(v) for v in jax.device_get((step_norm_arr, max_rel_dp_arr))
            )
            if max_rel_dp < taylor_threshold and logdet_cached is not None:
                d_theta_g = delta_param[:-1]
                d_theta_e = delta_param[-1]
                # This is the cheap first-order update for log|H| only.
                # The restricted term log|X^T H^{-1} X| is intentionally not
                # expanded here; for n >> p this is a small approximation.
                taylor_ld = logdet_cached + jnp.dot(d_theta_g, tr_Hinv_K_cached) + d_theta_e * tr_Hinv_cached
                trial_use_taylor = True

            if precond_refresh_fn is not None and precond_refresh_reldp > 0.0:
                if full_log:
                    logger.info(
                        "[REML] iter %d refresh projected_core preconditioner "
                        "(trial=%d alpha=%.3e)",
                        it + 1,
                        trial_count,
                        alpha_used,
                    )
                refreshed = precond_refresh_fn(param_updated)
                if refreshed is not None:
                    ctx.precond_conf = refreshed
                iter_precond_refreshed = True

            eval_t0 = time.time()
            ll_try, grad_try, FI_try, k_pcg_try, warm_try, warm_ai_try, _tr_Hinv_try, _tr_Hinv_K_try, logdet_try = _run_eval(
                param_updated,
                trial_warm,
                trial_warm_ai,
                tol_cur,
                trial_warm_ready,
                trial_warm_ai_ready,
                taylor_logdet_val=taylor_ld,
                compute_traces=False,
            )
            eval_elapsed += time.time() - eval_t0
            dll_arr, k_pcg_arr = jax.device_get((ll_try - ll, k_pcg_try))
            dll = float(dll_arr)
            k_pcg_trial = int(k_pcg_arr)
            ai_pcg_trial = (
                int(FI_try.stats.ai_pcg_iters)
                if isinstance(FI_try, AverageInfoMatrix) and FI_try.stats is not None
                else 0
            )
            eval_ai_pcg = ai_pcg_trial
            trial_accepted = (dll >= 0.0) or optimizer == "smile_scoring"
            ls_trace.append((alpha_used, dll, trial_accepted, k_pcg_trial, ai_pcg_trial, trial_use_taylor))
            if trial_accepted:
                accepted = True
                ll_new = ll_try
                grad_new = grad_try
                FI_new = FI_try
                warm_next = warm_try
                warm_ai_next = warm_ai_try
                tr_Hinv_new, tr_Hinv_K_new = _compute_traces_from_pcg(warm_try, ctx)
                logdet_new = logdet_try
                k_pcg = k_pcg_trial
                eval_ai_pcg = ai_pcg_trial
                use_taylor = trial_use_taylor
                break
            trial_warm = warm_try
            trial_warm_ai = warm_ai_try
            trial_warm_ready = True
            trial_warm_ai_ready = True
            alpha_try *= 0.5

        (
            ll_new_host,
            ll_host,
            grad_norm_host,
            proj_grad_host,
            delta_param_host,
            param_updated_host,
            n_frozen_host,
        ) = jax.device_get(
            (
                ll_new,
                ll,
                jnp.linalg.norm(grad_new),
                _projected_gradient_inf_norm(param_updated, grad_new, genetic_zero_tol),
                delta_param,
                param_updated,
                jnp.sum(freeze_mask),
            )
        )
        dll = float(ll_new_host - ll_host)
        rel_improve = dll / max(abs(float(ll_host)), 1e-12)
        if accepted:
            status = "accept" if dll >= 0.0 else "accept_downhill"
        else:
            status = "ll_down"
        dparam_str  = "[" + ", ".join(f"{float(v):.3e}" for v in np.asarray(delta_param_host).reshape(-1)) + "]"

        history.append({
            "iter": it + 1,
            "status": status,
            "accepted": accepted,
            "grad_norm": float(grad_norm_host),
            "proj_grad_inf": float(proj_grad_host),
            "loglik": float(ll_new_host),
            "loglik_prev": float(ll_host),
            "dll_true": dll,
            "rel_dll": rel_improve,
            "pcg_iters": k_pcg,
            "pcg_tol": tol_cur,
            "step_norm": step_norm,
            "step_sec": step_elapsed,
            "step_alpha": alpha_used,
            "alpha_max": alpha_max,
            "line_search_trials": trial_count,
            "n_freeze": int(n_frozen_host),
            "eval_ai_pcg_iters": int(eval_ai_pcg),
            "eval_sec": eval_elapsed,
            "iter_sec": time.time() - iter_t0,
            "params": [float(v) for v in np.asarray(param_updated_host).reshape(-1)],
            "slq_taylor": use_taylor,
            "max_rel_dp": max_rel_dp,
            "precond_refreshed": iter_precond_refreshed,
            "y_mean": float(y_mean_host),
            "y_scale": float(y_scale_host),
            **step_fisher_stats_dict,
        })

        if full_log:
            slq_tag = "taylor" if use_taylor else "slq"
            step_ai_pcg = int(step_fisher_stats_dict.get("ai_pcg_iters", 0))
            ws_resolves = int(step_fisher_stats_dict.get("ws_resolve_count", 0))
            ws_fixed_total = int(step_fisher_stats_dict.get("ws_fixed_total", 0))
            ws_trace = str(step_fisher_stats_dict.get("ws_trace", ""))
            step_free_dim = int(step_fisher_stats_dict.get("free_dim", 0))
            step_frozen = int(step_fisher_stats_dict.get("frozen_genetic", 0))
            if trial_count > 1:
                trace_str = " -> ".join(
                    f"a={alpha:.3e} dll={dll_i:.3e} pcg={pcg_i} ai_pcg={ai_pcg_i} slq={'taylor' if taylor_i else 'slq'}{'*' if ok else ''}"
                    for alpha, dll_i, ok, pcg_i, ai_pcg_i, taylor_i in ls_trace
                )
                logger.info("[REML] iter %d line-search %s", it + 1, trace_str)
            logger.info(
                "[REML] iter %d/%d status=%s pcg_tol=%.1e\n"
                "  ll: %.6e -> %.6e  dll=%.3e rel_dll=%.3e\n"
                "  step: alpha=%.3e/%.3e ls=%d |dparam|=%.3e max_reldp=%.3e step=%.1fs\n"
                "  workset: free=%d freeze=%d resolves=%d fix_total=%d trace=%s\n"
                "  solves: step_ai_pcg=%d eval_pcg=%d eval_ai_pcg=%d slq=%s eval=%.1fs iter=%.1fs\n"
                "  dparam: %s",
                it + 1, minq_iter, status, tol_cur,
                float(ll_host), float(ll_new_host), dll, rel_improve,
                alpha_used, alpha_max, trial_count, step_norm, max_rel_dp, step_elapsed,
                step_free_dim, step_frozen, ws_resolves, ws_fixed_total, ws_trace if ws_trace else "<none>",
                step_ai_pcg, k_pcg, eval_ai_pcg, slq_tag, eval_elapsed, time.time() - iter_t0,
                dparam_str,
            )
        elif compact_log:
            step_ai_pcg = int(step_fisher_stats_dict.get("ai_pcg_iters", 0))
            ls_text = ""
            if trial_count > 1:
                ls_text = " ls_trace=" + " -> ".join(
                    f"a={alpha:.3g},dll={dll_i:+.2e}{'*' if ok else ''}"
                    for alpha, dll_i, ok, _pcg_i, _ai_pcg_i, _taylor_i in ls_trace
                )
            logger.info(
                "[REML] iter %d/%d %s ll=%.6e dll=%+.3e alpha=%.3e ls=%d pcg=%d ai_pcg=%d step=%.3e dp=%s%s",
                it + 1,
                minq_iter,
                status,
                float(ll_new_host),
                dll,
                alpha_used,
                trial_count,
                k_pcg,
                eval_ai_pcg,
                step_norm,
                dparam_str,
                ls_text,
            )

        if accepted:
            param    = param_updated
            grad     = grad_new
            FI       = FI_new
            ll       = ll_new
            warm_all = warm_next
            warm_ai  = warm_ai_next
            # Update cached traces for next Taylor warm-start.
            tr_Hinv_cached   = tr_Hinv_new
            tr_Hinv_K_cached = tr_Hinv_K_new
            # Update logdet anchor: on full SLQ iterations, replace with the
            # freshly computed value; on Taylor iterations, the Taylor estimate
            # becomes the new anchor (it is the best available estimate).
            logdet_cached = logdet_new
            if optimizer == "smile_scoring":
                should_stop = (not math.isfinite(rel_improve)) or (max_rel_dp < scoring_step_tol)
            else:
                should_stop = (not math.isfinite(rel_improve)) or (rel_improve < rel_dll_tol)
            if should_stop:
                stop_reason = "scoring_step" if optimizer == "smile_scoring" else "rel_dll"
                break
        else:
            stop_reason = "ll_down"
            break

    if history:
        history[-1]["stop_reason"] = stop_reason

    if verbose:
        _t1 = time.time()
        logger.info("[REML] done @ %s elapsed=%.1fs stop=%s",
                    datetime.now().isoformat(timespec='seconds'), _t1 - _t0, stop_reason)
    if return_diagnostics:
        fi_mat = FI.mat if isinstance(FI, AverageInfoMatrix) else FI
        diagnostics = {
            "theta": param,
            "grad": grad,
            "ai": fi_mat,
            "loglik": ll,
            "stop_reason": stop_reason,
            "y_mean": y_mean,
            "y_scale": y_scale,
        }
        return param, history, diagnostics
    return param, history


__all__ = ["fit_reml", "standardize_response"]
