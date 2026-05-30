"""
Numerical correctness tests for PCG and preconditioner.

These tests construct small SPD systems and verify convergence / correctness.
Requires JAX (CPU backend is fine).

Run with: pytest tests/test_pcg_precond.py -v
"""

import os
import sys
import importlib

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_REPO_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PKG = os.path.basename(_REPO_ROOT)

import jax
import jax.numpy as jnp

# Force CPU to avoid GPU dependency in CI
jax.config.update("jax_platform_name", "cpu")

_PCG = importlib.import_module(f"{_PKG}.pcg")
_PRE = importlib.import_module(f"{_PKG}.precond")

pcg_solve = _PCG.pcg_solve
build_lowrank_basis = _PRE.build_lowrank_basis
ProjectedCorePrecondConf = _PRE.ProjectedCorePrecondConf
build_projected_core_runtime = _PRE.build_projected_core_runtime
make_projected_core_precond = _PRE.make_projected_core_precond
projected_core_apply_invsqrt = _PRE.projected_core_apply_invsqrt
projected_core_logdet = _PRE.projected_core_logdet


# ======================================================================
# Helpers
# ======================================================================

def _random_spd(n: int, cond: float = 10.0, seed: int = 42) -> jnp.ndarray:
    """Generate a random SPD matrix with bounded condition number."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n).astype(np.float32)
    # A^T A has eigenvalues >= 0
    M = A.T @ A
    # Shift to control condition number
    evals = np.linalg.eigvalsh(M)
    shift = max(0, evals.max() / cond - evals.min())
    M = M + shift * np.eye(n, dtype=np.float32)
    return jnp.asarray(M)


# ======================================================================
# PCG tests
# ======================================================================

class TestPCG:
    def test_identity_preconditioner(self):
        """PCG with M=I should converge to the exact solution."""
        n = 100
        H = _random_spd(n, cond=10.0)
        b = jnp.ones((n, 1), dtype=jnp.float32)

        Hv = lambda v: H @ v
        x, res, iters = pcg_solve(Hv, b, M=None, tol=1e-4, maxiter=200)

        x_exact = jnp.linalg.solve(H, b)
        rel_err = float(jnp.linalg.norm(x - x_exact) / jnp.linalg.norm(x_exact))
        assert rel_err < 1e-3, f"Relative error {rel_err:.2e} too large"
        assert iters > 0

    def test_multi_rhs(self):
        """PCG should handle multiple right-hand sides simultaneously."""
        n, k = 80, 5
        H = _random_spd(n, cond=5.0)
        B = jax.random.normal(jax.random.PRNGKey(0), (n, k))

        Hv = lambda v: H @ v
        X, res, iters = pcg_solve(Hv, B, tol=1e-4, maxiter=200)

        X_exact = jnp.linalg.solve(H, B)
        rel_err = float(jnp.linalg.norm(X - X_exact) / jnp.linalg.norm(X_exact))
        assert rel_err < 1e-2, f"Multi-RHS relative error {rel_err:.2e}"

    def test_warm_start_fewer_iters(self):
        """Warm-starting from the solution should converge in 0 iterations."""
        n = 50
        H = _random_spd(n, cond=5.0)
        b = jnp.ones((n, 1), dtype=jnp.float32)
        x_exact = jnp.linalg.solve(H, b)

        Hv = lambda v: H @ v
        _, _, iters = pcg_solve(Hv, b, X0=x_exact, tol=1e-4, maxiter=100)
        assert iters == 0, f"Warm start should converge immediately, got {iters}"

    def test_well_conditioned_fast(self):
        """Well-conditioned system should converge in few iterations."""
        n = 100
        # Nearly diagonal → condition ~2
        H = jnp.eye(n) + 0.1 * _random_spd(n, cond=2.0, seed=99)
        b = jnp.ones((n, 1), dtype=jnp.float32)

        Hv = lambda v: H @ v
        _, _, iters = pcg_solve(Hv, b, tol=1e-4, maxiter=200)
        assert iters < 30, f"Well-conditioned system took {iters} iters"

    def test_check_every_does_not_change_result(self):
        """Different check_every values should give same answer."""
        n = 60
        H = _random_spd(n, cond=10.0)
        b = jnp.ones((n, 1), dtype=jnp.float32)
        Hv = lambda v: H @ v

        x1, _, _ = pcg_solve(Hv, b, tol=1e-4, maxiter=200, check_every=1)
        x5, _, _ = pcg_solve(Hv, b, tol=1e-4, maxiter=200, check_every=5)

        # Solutions should be very close (check_every only affects when we stop)
        rel_diff = float(jnp.linalg.norm(x1 - x5) / (jnp.linalg.norm(x1) + 1e-12))
        assert rel_diff < 1e-2


# ======================================================================
# Preconditioner tests
# ======================================================================

class TestPreconditioner:
    def test_projected_core_scalar_diag_matches_exact_inverse(self):
        """Projected-core scalar-diagonal apply should match direct dense solve."""
        n, rank = 50, 8
        rng = np.random.RandomState(123)
        U_raw = rng.randn(n, rank).astype(np.float32)
        U, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U[:, :rank])

        A = rng.randn(rank, rank).astype(np.float32)
        core = (A + A.T) / 2.0
        # Keep d I + core safely SPD.
        min_eval = float(np.linalg.eigvalsh(core).min())
        d = np.float32(max(1.5, -min_eval + 0.5))

        conf = ProjectedCorePrecondConf(
            U=U,
            core_atoms=jnp.asarray(core[None, :, :]),
            total_rank=rank,
            n_grm=1,
            diag_mode="scalar_identity",
        )
        M_apply = make_projected_core_precond(
            conf,
            theta_g=jnp.asarray([1.0], dtype=jnp.float32),
            diag_H=jnp.asarray(d, dtype=jnp.float32),
            eps=1e-6,
        )

        key = jax.random.PRNGKey(7)
        V = jax.random.normal(key, (n, 3))
        M_dense = d * jnp.eye(n, dtype=jnp.float32) + U @ jnp.asarray(core) @ U.T
        ref = jnp.linalg.solve(M_dense, V)
        got = M_apply(V)

        rel_err = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
        assert rel_err < 1e-4, f"Projected-core apply mismatch: {rel_err:.2e}"

    def test_projected_core_invsqrt_and_logdet_match_dense(self):
        """Projected-core inverse-square-root and logdet should match dense algebra."""
        n, rank = 40, 6
        rng = np.random.RandomState(321)
        U_raw = rng.randn(n, rank).astype(np.float32)
        U, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U[:, :rank])

        A = rng.randn(rank, rank).astype(np.float32)
        core = (A + A.T) / 2.0
        min_eval = float(np.linalg.eigvalsh(core).min())
        d = np.float32(max(2.0, -min_eval + 0.75))

        conf = ProjectedCorePrecondConf(
            U=U,
            core_atoms=jnp.asarray(core[None, :, :]),
            total_rank=rank,
            n_grm=1,
            diag_mode="scalar_identity",
        )
        runtime = build_projected_core_runtime(
            conf,
            theta_g=jnp.asarray([1.0], dtype=jnp.float32),
            diag_H=jnp.asarray(d, dtype=jnp.float32),
            need_invsqrt=True,
        )

        M_dense = d * jnp.eye(n, dtype=jnp.float32) + U @ jnp.asarray(core) @ U.T
        sign, ref_logdet = np.linalg.slogdet(np.asarray(M_dense))
        assert sign > 0
        got_logdet = float(projected_core_logdet(runtime, n))
        assert abs(got_logdet - float(ref_logdet)) < 1e-4

        V = jax.random.normal(jax.random.PRNGKey(11), (n, 3))
        whitened = projected_core_apply_invsqrt(
            runtime,
            M_dense @ projected_core_apply_invsqrt(runtime, V),
        )
        rel_err = float(jnp.linalg.norm(whitened - V) / jnp.linalg.norm(V))
        assert rel_err < 2e-4, f"Projected-core invsqrt mismatch: {rel_err:.2e}"

    def test_projected_core_invsqrt_accepts_vector_input(self):
        n, rank = 32, 5
        rng = np.random.RandomState(77)
        U_raw = rng.randn(n, rank).astype(np.float32)
        U, _ = np.linalg.qr(U_raw)
        U = jnp.asarray(U[:, :rank])

        A = rng.randn(rank, rank).astype(np.float32)
        core = (A + A.T) / 2.0
        min_eval = float(np.linalg.eigvalsh(core).min())
        d = np.float32(max(2.0, -min_eval + 0.75))

        conf = ProjectedCorePrecondConf(
            U=U,
            core_atoms=jnp.asarray(core[None, :, :]),
            total_rank=rank,
            n_grm=1,
            diag_mode="scalar_identity",
        )
        runtime = build_projected_core_runtime(
            conf,
            theta_g=jnp.asarray([1.0], dtype=jnp.float32),
            diag_H=jnp.asarray(d, dtype=jnp.float32),
            need_invsqrt=True,
        )

        v = jax.random.normal(jax.random.PRNGKey(23), (n,), dtype=jnp.float32)
        got = projected_core_apply_invsqrt(runtime, v)

        M_dense = d * jnp.eye(n, dtype=jnp.float32) + U @ jnp.asarray(core) @ U.T
        evals, evecs = np.linalg.eigh(np.asarray(M_dense))
        ref = evecs @ ((evecs.T @ np.asarray(v)) / np.sqrt(np.clip(evals, 1e-8, None)))

        assert got.ndim == 1
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-4, atol=1e-4)


# ======================================================================
# build_lowrank_basis tests
# ======================================================================

class TestBuildLowrankBasis:
    def test_captures_leading_eigenspace(self):
        """Nyström sketch should capture the leading eigenvalues."""
        n, true_rank = 200, 10
        rng = np.random.RandomState(42)
        # K with a clear spectral gap: top-10 eigenvalues >> rest
        Q_true, _ = np.linalg.qr(rng.randn(n, true_rank).astype(np.float32))
        lam_true = np.array([100.0 - 8 * i for i in range(true_rank)], dtype=np.float32)
        K = Q_true @ np.diag(lam_true) @ Q_true.T + 0.01 * np.eye(n, dtype=np.float32)
        K_jnp = jnp.asarray(K)

        K_mv = lambda V: K_jnp @ V
        key = jax.random.PRNGKey(0)
        U, evals = build_lowrank_basis(K_mv, n, max_rank=true_rank, key=key)

        assert U.shape == (n, true_rank)
        assert evals.shape == (true_rank,)
        # Leading eigenvalues should be close
        for i in range(min(5, true_rank)):
            rel_err = abs(float(evals[i]) - lam_true[i]) / lam_true[i]
            assert rel_err < 0.1, f"Eigenvalue {i}: {float(evals[i]):.2f} vs {lam_true[i]:.2f}"

    def test_orthogonal_basis(self):
        """Returned U should have approximately orthonormal columns."""
        n = 100
        K = _random_spd(n, cond=5.0)
        K_mv = lambda V: K @ V
        key = jax.random.PRNGKey(1)
        U, _ = build_lowrank_basis(K_mv, n, max_rank=20, key=key)

        gram = U.T @ U
        I = jnp.eye(gram.shape[0])
        off_diag = float(jnp.max(jnp.abs(gram - I)))
        assert off_diag < 0.05, f"U^T U off-diagonal max = {off_diag:.4f}"
