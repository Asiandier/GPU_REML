"""
Block preconditioned conjugate gradient (multi-RHS).

Performance notes
-----------------
*  `Hv(P)` calls `kv()` which is the dominant cost per iteration.
*  Convergence check (`bool(jnp.any(...))`) forces a GPU→CPU synchronisation,
   draining the entire async dispatch pipeline.  This is done every
   `check_every` iterations instead of every iteration to keep the GPU busy.
*  The state-update and direction-update steps are individually JIT-compiled
   so that XLA fuses the elementwise arithmetic into efficient kernels.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _identity(v: Array) -> Array:
    return v


@partial(jax.jit, donate_argnums=(4,))
def _pcg_state_update(
    X: Array,
    R: Array,
    P: Array,
    rs: Array,
    AP: Array,
) -> Tuple[Array, Array]:
    """Compute (X_new, R_new) given AP = H @ P."""
    denom = jnp.sum(P * AP, axis=0) + 1e-12
    alpha = rs / denom
    X_new = X + P * alpha
    R_new = R - AP * alpha
    return X_new, R_new


@jax.jit
def _pcg_direction_update(
    P: Array,
    R_new: Array,
    Z_new: Array,
    rs_old: Array,
) -> Tuple[Array, Array, Array]:
    """Update search direction and compute new residual norm."""
    rs_new    = jnp.sum(R_new * Z_new, axis=0)
    rnorm_new = jnp.linalg.norm(R_new, axis=0)
    beta      = rs_new / (rs_old + 1e-12)
    P_new     = Z_new + P * beta
    return P_new, rs_new, rnorm_new


def pcg_solve(
    Hv: Callable[[Array], Array],
    B: Array,
    M: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-2,
    maxiter: int = 200,
    X0: Optional[Array] = None,
    check_every: int = 2,
) -> Tuple[Array, Array, int]:
    """
    Solve H X = B with block PCG.

    Parameters
    ----------
    check_every : int
        Check convergence every this many iterations.  Reducing sync
        frequency from every-iteration to every-N keeps the GPU dispatch
        pipeline full.  The solver may overshoot convergence by at most
        ``check_every - 1`` iterations (cheap compared to sync cost).

    Returns ``(X, max_rel_res, iters)``.
    """
    if M is None:
        M = _identity
    if X0 is None:
        X0 = jnp.zeros_like(B)

    check_every = max(1, int(check_every))
    b_norm = jnp.linalg.norm(B, axis=0) + 1e-12

    X     = X0
    R     = B - Hv(X)
    Z     = M(R)
    P     = Z
    rs    = jnp.sum(R * Z, axis=0)
    del Z
    rnorm = jnp.linalg.norm(R, axis=0)

    # Check once up front — warm-started X0 may already be converged.
    if not bool(jnp.any(rnorm > tol * b_norm)):
        return X, jnp.max(rnorm / b_norm), 0

    k = 0
    while k < maxiter:
        AP   = Hv(P)                                      # ← dominant cost
        X, R = _pcg_state_update(X, R, P, rs, AP)
        del AP
        Z    = M(R)
        P, rs, rnorm = _pcg_direction_update(P, R, Z, rs)
        del Z
        k += 1

        # Periodic convergence check (GPU→CPU sync).
        if k % check_every == 0 or k == maxiter:
            if not bool(jnp.any(rnorm > tol * b_norm)):
                break

    res_final = jnp.max(rnorm / b_norm)
    return X, res_final, k


__all__ = ["pcg_solve"]
