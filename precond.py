"""Projected-core preconditioners tuned for GPU-friendly PCG runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jnp.ndarray


@dataclass
class ProjectedCorePrecondConf:
    """Projected-core preconditioner configuration for unified many-GRM fits."""

    U: Array                               # (n, k) shared orthonormal basis
    core_atoms: Array                      # (G, k, k) projected component cores
    total_rank: int
    n_grm: int
    diag_mode: Optional[str] = None        # MVP: "scalar_identity"
    diag_atoms: Optional[Array] = None
    residual_diag_atoms: Optional[Array] = None
    identity: Optional[Array] = None


@dataclass
class ProjectedCoreRuntime:
    """Theta-specific projected-core factorization reused within one evaluation."""

    U: Array
    total_rank: int
    d: Array
    d_inv: Array
    chol: Array
    eigvals: Optional[Array] = None
    eigvecs: Optional[Array] = None


@jax.jit
def _projected_core_apply_scalar_diag_jit(
    v: Array,
    d_inv: Array,
    chol: Array,
    U: Array,
) -> Array:
    """
    Apply (d I + U C U^T)^{-1} v for orthonormal U and scalar diagonal base.

    If U^T U = I, then the inverse decomposes as:
        M^{-1} v = d^{-1} v + U ((d I + C)^{-1} - d^{-1} I) U^T v
    """
    rhs = U.T @ v
    y = jsp.linalg.solve_triangular(chol, rhs, lower=True, check_finite=False)
    z = jsp.linalg.solve_triangular(chol.T, y, lower=False, check_finite=False)
    return d_inv * v + U @ (z - d_inv * rhs)


@jax.jit
def _projected_core_apply_scalar_diag_invsqrt_jit(
    v: Array,
    d_inv_sqrt: Array,
    eigvals: Array,
    eigvecs: Array,
    U: Array,
) -> Array:
    """
    Apply (d I + U C U^T)^(-1/2) v for orthonormal U and scalar diagonal base.

    If A = d I + C on the projected subspace, then:
        M^{-1/2} v = d^{-1/2} v + U (A^{-1/2} - d^{-1/2} I) U^T v
    """
    v_is_vector = v.ndim == 1
    if v_is_vector:
        v = v[:, None]
    rhs = U.T @ v
    coeff = eigvecs.T @ rhs
    eig_floor = jnp.asarray(1e-8, dtype=eigvals.dtype)
    scaled = coeff / jnp.sqrt(jnp.clip(eigvals, eig_floor))[:, None]
    z = eigvecs @ scaled
    out = d_inv_sqrt * v + U @ (z - d_inv_sqrt * rhs)
    return out[:, 0] if v_is_vector else out


def _scalar_diag_value(diag_H: Array, *, dtype) -> Array:
    diag_arr = jnp.asarray(diag_H, dtype=dtype)
    if diag_arr.ndim == 0:
        return diag_arr
    if diag_arr.size != 1:
        raise ValueError("Projected-core MVP currently requires a scalar diagonal base.")
    return diag_arr.reshape(())


def scalar_diag_from_precond_conf(
    precond_conf: Optional[ProjectedCorePrecondConf],
    theta_g: Array,
    theta_e: Array,
) -> Optional[Array]:
    """Return scalar diagonal base implied by scalar-identity preconditioners."""
    if precond_conf is None or precond_conf.diag_mode != "scalar_identity":
        return None
    if precond_conf.diag_atoms is None:
        diag_atoms = jnp.ones((precond_conf.n_grm,), dtype=theta_g.dtype)
    else:
        diag_atoms = jnp.asarray(precond_conf.diag_atoms, dtype=theta_g.dtype)
    theta_e_arr = jnp.asarray(theta_e, dtype=theta_g.dtype).reshape(-1)
    if precond_conf.residual_diag_atoms is None:
        residual_atoms = jnp.ones((theta_e_arr.shape[0],), dtype=theta_g.dtype)
    else:
        residual_atoms = jnp.asarray(precond_conf.residual_diag_atoms, dtype=theta_g.dtype).reshape(-1)
        if int(residual_atoms.shape[0]) != int(theta_e_arr.shape[0]):
            raise ValueError(
                "Projected-core residual_diag_atoms length mismatch: "
                f"expected {int(theta_e_arr.shape[0])}, got {int(residual_atoms.shape[0])}."
            )
    return jnp.dot(theta_e_arr, residual_atoms) + jnp.dot(theta_g, diag_atoms)


def build_projected_core_runtime(
    precond_conf: Optional[ProjectedCorePrecondConf],
    theta_g: Array,
    diag_H: Array,
    eps: float = 0.0,
    *,
    need_invsqrt: bool = False,
) -> Optional[ProjectedCoreRuntime]:
    """Assemble the theta-specific projected-core factorization."""
    if precond_conf is None:
        return None
    if precond_conf.U.size == 0 or precond_conf.total_rank <= 0:
        return None
    if precond_conf.diag_mode != "scalar_identity":
        raise NotImplementedError(
            "Projected-core preconditioner currently supports only diag_mode='scalar_identity'."
        )

    U = precond_conf.U
    core = jnp.tensordot(theta_g, precond_conf.core_atoms, axes=1)
    core = 0.5 * (core + core.T)
    d = _scalar_diag_value(diag_H, dtype=U.dtype) + jnp.asarray(eps, dtype=U.dtype)
    d = jnp.maximum(d, jnp.asarray(1e-8, dtype=U.dtype))
    if precond_conf.identity is None:
        identity = jnp.eye(precond_conf.total_rank, dtype=U.dtype)
    else:
        identity = jnp.asarray(precond_conf.identity, dtype=U.dtype)
        expected_shape = (precond_conf.total_rank, precond_conf.total_rank)
        if identity.shape != expected_shape:
            raise ValueError(
                f"Projected-core identity cache shape mismatch: expected {expected_shape}, "
                f"got {identity.shape}."
            )
    middle = core + d * identity
    chol = jnp.linalg.cholesky(middle)

    eigvals = None
    eigvecs = None
    if need_invsqrt:
        eigvals_raw, eigvecs = jnp.linalg.eigh(middle)
        eigvals = jnp.clip(eigvals_raw, 1e-8)

    return ProjectedCoreRuntime(
        U=U,
        total_rank=precond_conf.total_rank,
        d=d,
        d_inv=1.0 / d,
        chol=chol,
        eigvals=eigvals,
        eigvecs=eigvecs,
    )


def make_projected_core_precond_from_runtime(
    runtime: Optional[ProjectedCoreRuntime],
) -> Optional[Callable[[Array], Array]]:
    """Create an inverse-apply closure from a projected-core runtime."""
    if runtime is None:
        return None

    def apply(v: Array) -> Array:
        return _projected_core_apply_scalar_diag_jit(
            v, runtime.d_inv, runtime.chol, runtime.U
        )

    return apply


def projected_core_apply_invsqrt(
    runtime: ProjectedCoreRuntime,
    v: Array,
) -> Array:
    """Apply the inverse square root of a projected-core factorization."""
    if runtime.eigvals is None or runtime.eigvecs is None:
        raise ValueError(
            "Projected-core runtime missing eigendecomposition required for inverse-square-root apply."
        )
    return _projected_core_apply_scalar_diag_invsqrt_jit(
        v,
        jnp.sqrt(runtime.d_inv),
        runtime.eigvals,
        runtime.eigvecs,
        runtime.U,
    )


def projected_core_logdet(
    runtime: ProjectedCoreRuntime,
    n_dim: int,
) -> Array:
    """Compute logdet(d I + U C U^T) from the projected-core factorization."""
    if n_dim < runtime.total_rank:
        raise ValueError(
            f"n_dim={n_dim} is smaller than projected-core rank {runtime.total_rank}."
        )
    chol_diag = jnp.diag(runtime.chol)
    diag_floor = jnp.asarray(jnp.finfo(chol_diag.dtype).tiny, dtype=chol_diag.dtype)
    return (
        jnp.asarray(n_dim - runtime.total_rank, dtype=runtime.d.dtype) * jnp.log(runtime.d)
        + 2.0 * jnp.sum(jnp.log(jnp.clip(chol_diag, diag_floor)))
    )


def make_projected_core_precond(
    precond_conf: Optional[ProjectedCorePrecondConf],
    theta_g: Array,
    diag_H: Array,
    eps: float = 0.0,
) -> Optional[Callable[[Array], Array]]:
    """Assemble projected-core preconditioner from shared basis + component cores."""
    runtime = build_projected_core_runtime(
        precond_conf, theta_g, diag_H, eps=eps, need_invsqrt=False
    )
    return make_projected_core_precond_from_runtime(runtime)


def make_precond(
    precond_conf: Optional[ProjectedCorePrecondConf],
    theta_g: Array,
    diag_H: Array,
    eps: float = 0.0,
) -> Optional[Callable[[Array], Array]]:
    """Assemble the configured projected-core preconditioner."""
    return make_projected_core_precond(precond_conf, theta_g, diag_H, eps=eps)


# ---------------------------------------------------------------------------
# Low-rank basis construction (Nyström sketch)
# ---------------------------------------------------------------------------

def build_lowrank_basis(
    K_mv: Callable[[Array], Array],
    n: int,
    max_rank: int,
    key: "jax.random.PRNGKey",
    oversample: int = 8,
    dtype=jnp.float32,
) -> tuple[Array, Array]:
    """
    Build a low-rank basis U, evals that captures leading spectrum of K.
    Nyström-style sketch: 2 passes over K.
    """
    r = max_rank + oversample
    key, sub = jax.random.split(key)
    Omega = jax.random.normal(sub, (n, r), dtype=dtype)
    Y = K_mv(Omega)
    Q, _ = jnp.linalg.qr(Y, mode="reduced")
    Z = K_mv(Q)
    B = Q.T @ Z
    evals, evecs = jnp.linalg.eigh(B)
    idx   = jnp.argsort(evals)[::-1]
    evals = jnp.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]
    U_full = Q @ evecs
    k = min(max_rank, U_full.shape[1])
    return U_full[:, :k], evals[:k]


__all__ = [
    "ProjectedCorePrecondConf",
    "ProjectedCoreRuntime",
    "build_projected_core_runtime",
    "make_projected_core_precond",
    "make_projected_core_precond_from_runtime",
    "make_precond",
    "projected_core_apply_invsqrt",
    "projected_core_logdet",
    "scalar_diag_from_precond_conf",
    "build_lowrank_basis",
]
