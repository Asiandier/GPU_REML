"""
Sparse weighted LASSO utilities for large-scale REML pipelines.

This module provides:
- Coordinate-descent LASSO solver on a precomputed Gram system.
- EBIC-based lambda-path selection.
- Weighted sparse-effect fitting with unpenalized covariates and
  penalized SNP effects.

Objective (given H^{-1}):
    min_{b_c, b_s} 0.5 * (y - C b_c - Z b_s)^T H^{-1} (y - C b_c - Z b_s)
                  + lambda * ||b_s||_1
where C is unpenalized covariates and Z is penalized SNP matrix.

Performance notes:
- The CD inner loop is Numba-JIT compiled (>50× faster than pure Python).
- SPD solves are delegated to pipeline_common.solve_spd.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from numba import njit

logger = logging.getLogger(__name__)

from .pipeline_common import solve_spd


@dataclass
class LassoPathConfig:
    lam_min_ratio: float = 0.05
    n_lambda: int = 60
    ebic_gamma: float = 0.5
    ebic_eps: float = 1e-12
    max_cd_iter: int = 2000
    cd_tol: float = 1e-6
    ebic_early_stop: bool = True
    ebic_early_stop_patience: int = 10
    ebic_early_stop_min_delta: float = 0.0
    active_set_period: int = 5
    verbose: bool = False


@dataclass(frozen=True)
class _FactorizedLinearSystem:
    factor: np.ndarray
    lower: bool
    use_cholesky: bool
    piv: np.ndarray | None = None


def _factor_linear_system(mat: np.ndarray) -> _FactorizedLinearSystem:
    A = np.asarray(mat, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Linear solve factorization expects a square matrix.")
    if A.shape[0] == 0:
        return _FactorizedLinearSystem(
            factor=np.empty((0, 0), dtype=np.float64),
            lower=True,
            use_cholesky=True,
            piv=None,
        )
    try:
        factor, lower = sla.cho_factor(A, lower=True, check_finite=False)
        return _FactorizedLinearSystem(
            factor=factor,
            lower=bool(lower),
            use_cholesky=True,
            piv=None,
        )
    except np.linalg.LinAlgError:
        factor, piv = sla.lu_factor(A, check_finite=False)
        return _FactorizedLinearSystem(
            factor=factor,
            lower=False,
            use_cholesky=False,
            piv=piv,
        )


def _solve_factorized_system(system: _FactorizedLinearSystem, rhs: np.ndarray) -> np.ndarray:
    B = np.asarray(rhs, dtype=np.float64)
    if system.factor.shape[0] == 0:
        return np.zeros_like(B, dtype=np.float64)
    if system.use_cholesky:
        return sla.cho_solve((system.factor, system.lower), B, check_finite=False)
    if system.piv is None:
        raise RuntimeError("LU factorization is missing pivot metadata.")
    return sla.lu_solve((system.factor, system.piv), B, check_finite=False)


def _log_choose(p: int, k: int) -> float:
    if k < 0 or k > p:
        return float("-inf")
    if k == 0 or k == p:
        return 0.0
    kk = min(k, p - k)
    return math.lgamma(p + 1.0) - math.lgamma(kk + 1.0) - math.lgamma(p - kk + 1.0)


def ebic_from_rss(
    n: int,
    p: int,
    k: int,
    rss: float,
    gamma: float,
    eps: float,
) -> float:
    rss_eff = float(rss)
    if (not math.isfinite(rss_eff)) or rss_eff <= 0.0:
        rss_eff = float(eps)
    else:
        rss_eff = max(rss_eff, float(eps))

    n_f = float(n)
    term1 = n_f * math.log(rss_eff / n_f)
    term2 = float(k) * math.log(n_f)
    term3 = 2.0 * float(gamma) * _log_choose(int(p), int(k))
    return float(term1 + term2 + term3)


def make_lambda_sequence(lam_max: float, lam_min_ratio: float, n_lambda: int) -> np.ndarray:
    lam_max = max(float(lam_max), 0.0)
    n_lambda = max(int(n_lambda), 1)
    if lam_max <= 0.0:
        return np.array([0.0], dtype=np.float64)
    lam_min = max(lam_max * float(lam_min_ratio), lam_max * 1e-6)
    return np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_lambda)).astype(np.float64)


# ---------------------------------------------------------------------------
# Numba-JIT coordinate descent kernel
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _cd_epoch(Q: np.ndarray, q: np.ndarray, diag: np.ndarray,
              beta: np.ndarray, Qb: np.ndarray, lam: float) -> float:
    """
    One full CD sweep over all coordinates. Returns max |delta|.

    Fused inner loop: soft-threshold + Qb update in compiled code.
    ~50-100× faster than the equivalent Python loop.
    """
    k = beta.shape[0]
    max_delta = 0.0
    for j in range(k):
        q_j_tilde = q[j] - (Qb[j] - diag[j] * beta[j])
        # inline soft-threshold
        if q_j_tilde > lam:
            beta_new = (q_j_tilde - lam) / diag[j]
        elif q_j_tilde < -lam:
            beta_new = (q_j_tilde + lam) / diag[j]
        else:
            beta_new = 0.0
        delta = beta_new - beta[j]
        if delta != 0.0:
            beta[j] = beta_new
            # Qb += Q[:, j] * delta — column update
            for i in range(k):
                Qb[i] += Q[i, j] * delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
    return max_delta


@njit(cache=True, nogil=True)
def _cd_active_epoch(Q: np.ndarray, q: np.ndarray, diag: np.ndarray,
                     beta: np.ndarray, Qb: np.ndarray, lam: float,
                     active: np.ndarray) -> float:
    """
    CD sweep over active set only — much faster when most coordinates are zero.
    """
    n_active = active.shape[0]
    max_delta = 0.0
    for idx in range(n_active):
        j = active[idx]
        q_j_tilde = q[j] - (Qb[j] - diag[j] * beta[j])
        if q_j_tilde > lam:
            beta_new = (q_j_tilde - lam) / diag[j]
        elif q_j_tilde < -lam:
            beta_new = (q_j_tilde + lam) / diag[j]
        else:
            beta_new = 0.0
        delta = beta_new - beta[j]
        if delta != 0.0:
            beta[j] = beta_new
            for idx_i in range(n_active):
                i = active[idx_i]
                Qb[i] += Q[i, j] * delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
    return max_delta


def solve_lasso_cd_gram(
    Q: np.ndarray,
    q: np.ndarray,
    lam: float,
    *,
    beta0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    active_set_period: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Coordinate descent for
        min_b 0.5 b^T Q b - q^T b + lam ||b||_1,
    where Q is symmetric PSD and diag(Q) > 0.

    Uses Numba-JIT inner loop with active-set acceleration:
    alternates between full sweeps and active-set-only sweeps.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q.reshape(-1), dtype=np.float64)
    k = q.size

    if Q.shape != (k, k):
        raise ValueError("Q shape mismatch.")

    if beta0 is None:
        beta = np.zeros(k, dtype=np.float64)
    else:
        beta = np.ascontiguousarray(beta0.reshape(-1), dtype=np.float64).copy()
        if beta.size != k:
            raise ValueError("beta0 shape mismatch.")

    diag = np.diag(Q).copy()
    min_diag = float(np.min(diag)) if diag.size > 0 else 1.0
    if min_diag <= 0.0:
        raise ValueError("Q diagonal must be strictly positive for coordinate descent.")

    Qb = Q @ beta
    converged = False
    qb_full_stale = False

    max_iter = max(int(max_iter), 1)
    # Active-set strategy: after first full sweep, do active-set sweeps
    # until convergence, then verify with a full sweep.
    active_set_period = max(int(active_set_period), 1)
    it = 0

    for it_idx in range(1, max_iter + 1):
        it = it_idx
        if it_idx == 1 or it_idx % active_set_period == 0:
            # Full sweep
            if qb_full_stale:
                Qb = Q @ beta
                qb_full_stale = False
            max_delta = _cd_epoch(Q, q, diag, beta, Qb, lam)
        else:
            # Active-set sweep
            active = np.flatnonzero(beta != 0.0).astype(np.int64)
            if active.size == 0:
                # All zero — check full sweep to see if any should activate
                if qb_full_stale:
                    Qb = Q @ beta
                    qb_full_stale = False
                max_delta = _cd_epoch(Q, q, diag, beta, Qb, lam)
            else:
                max_delta = _cd_active_epoch(Q, q, diag, beta, Qb, lam, active)
                if max_delta > 0.0:
                    qb_full_stale = True

        if max_delta <= tol:
            # Verify convergence with a full sweep
            if qb_full_stale:
                Qb = Q @ beta
                qb_full_stale = False
            max_delta_full = _cd_epoch(Q, q, diag, beta, Qb, lam)
            it += 1
            if max_delta_full <= tol:
                converged = True
                break

    if qb_full_stale:
        Qb = Q @ beta
    return beta, Qb, it, converged


def solve_lasso_path_and_select_ebic(
    Q: np.ndarray,
    q: np.ndarray,
    yHy: float,
    n_samples: int,
    p_total: int,
    cfg: LassoPathConfig,
) -> dict:
    """
    Solve a lambda path and select lambda by EBIC.

    Args:
        Q, q: weighted quadratic system.
        yHy: y^T H^{-1} y.
        n_samples: sample size used in EBIC.
        p_total: total number of SNPs in full genome (for EBIC combinatorics).
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape != (q.size, q.size):
        raise ValueError("Q/q shape mismatch.")

    lam_max = float(np.max(np.abs(q))) if q.size > 0 else 0.0
    lam_seq = make_lambda_sequence(lam_max, cfg.lam_min_ratio, cfg.n_lambda)

    # Numerical floor scales with yHy and n.
    ebic_eps = max(float(cfg.ebic_eps), 1e-8 * max(float(yHy), 1.0))

    best_idx = 0
    best_ebic = float("inf")
    beta_warm = np.zeros_like(q)
    path = []
    beta_best = np.zeros_like(q)
    lam_best = float(lam_seq[0]) if lam_seq.size > 0 else 0.0
    es_counter = 0  # early-stop patience counter
    es_patience = max(int(cfg.ebic_early_stop_patience), 1)

    for i, lam in enumerate(lam_seq):
        beta, Qb, n_iter, converged = solve_lasso_cd_gram(
            Q,
            q,
            float(lam),
            beta0=beta_warm,
            max_iter=cfg.max_cd_iter,
            tol=cfg.cd_tol,
            active_set_period=cfg.active_set_period,
        )
        beta_warm = beta

        rss = max(float(yHy - 2.0 * (beta @ q) + (beta @ Qb)), 0.0)
        k = int(np.count_nonzero(beta))
        ebic = ebic_from_rss(
            n=n_samples,
            p=p_total,
            k=k,
            rss=rss,
            gamma=cfg.ebic_gamma,
            eps=ebic_eps,
        )

        path.append(
            {
                "lam": float(lam),
                "k": k,
                "rss": rss,
                "ebic": float(ebic),
                "cd_iter": int(n_iter),
                "converged": bool(converged),
            }
        )

        if (ebic < best_ebic - cfg.ebic_early_stop_min_delta) or (
            math.isclose(ebic, best_ebic) and k < path[best_idx]["k"]
        ):
            best_ebic = float(ebic)
            best_idx = i
            beta_best = beta.copy()
            lam_best = float(lam)
            es_counter = 0
        else:
            es_counter += 1

        if cfg.verbose and (i == 0 or i == len(lam_seq) - 1 or (i + 1) % 10 == 0):
            logger.info(
                "[lasso_path] %03d/%03d lam=%.3e k=%4d rss=%.4e ebic=%.4e "
                "cd_iter=%d conv=%s",
                i + 1, len(lam_seq), lam, k, rss, ebic, n_iter, converged,
            )

        # Early stop: if EBIC hasn't improved for `patience` consecutive lambdas
        if cfg.ebic_early_stop and es_counter >= es_patience and i >= 2:
            if cfg.verbose:
                logger.info(
                    "[lasso_path] EBIC early stop at %d/%d: no improvement for %d steps",
                    i + 1, len(lam_seq), es_counter,
                )
            break

    beta_best = np.asarray(beta_best, dtype=np.float64)
    active_idx = np.flatnonzero(beta_best != 0.0).astype(np.int64)

    return {
        "lam": lam_best,
        "beta": beta_best,
        "active_idx": active_idx,
        "best_ebic": float(best_ebic),
        "path": path,
    }


def compute_projected_hinv_vector(
    covar: np.ndarray | None,
    Hinv_covar: np.ndarray | None,
    Hinv_target: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Compute P_C * target in H^{-1} metric:
        P_C target = H^{-1} target - H^{-1}C (C^T H^{-1}C)^{-1} C^T H^{-1} target.
    """
    t = np.asarray(Hinv_target, dtype=np.float64).reshape(-1)

    if covar is None or Hinv_covar is None or Hinv_covar.size == 0:
        return t

    C = np.asarray(covar, dtype=np.float64)
    HC = np.asarray(Hinv_covar, dtype=np.float64)
    if C.ndim != 2 or HC.ndim != 2:
        raise ValueError("covar/Hinv_covar must be 2D.")

    if C.shape != HC.shape:
        raise ValueError("covar and Hinv_covar shape mismatch.")
    if C.shape[0] != t.size:
        raise ValueError("Hinv_target length mismatch with covariates.")

    A = C.T @ HC
    if A.size > 0:
        A = 0.5 * (A + A.T)
        A = A + float(ridge) * np.eye(A.shape[0], dtype=np.float64)
    rhs = C.T @ t
    coef = solve_spd(A, rhs)
    return t - HC @ coef


def fit_weighted_lasso_with_covariates(
    y: np.ndarray,
    covar: np.ndarray | None,
    geno: np.ndarray,
    Hinv_y: np.ndarray,
    Hinv_covar: np.ndarray | None,
    Hinv_geno: np.ndarray,
    p_total: int,
    cfg: LassoPathConfig,
    ridge: float = 1e-6,
) -> dict:
    """
    Weighted sparse fitting with unpenalized covariates and penalized SNP effects.

    Args:
        y, covar, geno: design matrices in sample order.
        Hinv_*: PCG solves under current variance components.
        p_total: full genome SNP count used in EBIC combinatorial term.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Z = np.asarray(geno, dtype=np.float64)
    Hy = np.asarray(Hinv_y, dtype=np.float64).reshape(-1)
    HZ = np.asarray(Hinv_geno, dtype=np.float64)

    if Z.ndim != 2:
        raise ValueError("geno must be a 2D matrix.")
    if HZ.shape != Z.shape:
        raise ValueError("geno and Hinv_geno shape mismatch.")
    if y.size != Z.shape[0] or Hy.size != y.size:
        raise ValueError("y/Hinv_y size mismatch with geno rows.")

    n = y.size
    k = Z.shape[1]

    # Compute Gram products in float64 to avoid float32 accumulation error.
    # For n=50k, k=256, float32 matmul can lose ~3 digits of precision.
    yHy = float(y @ Hy)
    gZy = Z.T @ Hy
    GZZ = Z.T @ HZ
    GCC = None
    GCZ = None
    gCy = None

    if covar is None or (covar.size == 0):
        Q = 0.5 * (GZZ + GZZ.T)
        q = gZy
        beta_cov = np.empty((0,), dtype=np.float64)
    else:
        C = np.asarray(covar, dtype=np.float64)
        HC = np.asarray(Hinv_covar, dtype=np.float64)
        if C.ndim != 2 or HC.ndim != 2:
            raise ValueError("covar/Hinv_covar must be 2D.")
        if C.shape != HC.shape:
            raise ValueError("covar and Hinv_covar shape mismatch.")
        if C.shape[0] != n:
            raise ValueError("covar row count mismatch with y.")

        GCC = C.T @ HC
        GCC = 0.5 * (GCC + GCC.T)
        GCC = GCC + float(ridge) * np.eye(GCC.shape[0], dtype=np.float64)

        GCZ = C.T @ HZ
        gCy = C.T @ Hy

        gcc_factor = _factor_linear_system(GCC)
        Ainv_gCy = _solve_factorized_system(gcc_factor, gCy)
        Ainv_GCZ = _solve_factorized_system(gcc_factor, GCZ)

        Q = GZZ - GCZ.T @ Ainv_GCZ
        Q = 0.5 * (Q + Q.T)
        q = gZy - GCZ.T @ Ainv_gCy

        beta_cov = np.zeros(C.shape[1], dtype=np.float64)

    if k > 0:
        d = np.diag(Q).copy()
        need = d < 1e-8
        if np.any(need):
            Q[np.diag_indices(k)] += (1e-8 - d) * need

    lasso = solve_lasso_path_and_select_ebic(
        Q=Q,
        q=q,
        yHy=yHy,
        n_samples=n,
        p_total=int(p_total),
        cfg=cfg,
    )

    beta_snp = np.asarray(lasso["beta"], dtype=np.float64)

    if covar is not None and covar.size > 0:
        if GCC is None or GCZ is None or gCy is None:
            raise RuntimeError("Internal error: covariate normal equations were not built.")
        rhs = gCy - GCZ @ beta_snp
        beta_cov = _solve_factorized_system(gcc_factor, rhs)

    active_idx = np.flatnonzero(beta_snp != 0.0).astype(np.int64)

    return {
        "beta_cov": beta_cov,
        "beta_snp": beta_snp,
        "active_idx": active_idx,
        "lam": float(lasso["lam"]),
        "best_ebic": float(lasso["best_ebic"]),
        "path": lasso["path"],
    }


__all__ = [
    "LassoPathConfig",
    "ebic_from_rss",
    "make_lambda_sequence",
    "solve_lasso_cd_gram",
    "solve_lasso_path_and_select_ebic",
    "compute_projected_hinv_vector",
    "fit_weighted_lasso_with_covariates",
]
