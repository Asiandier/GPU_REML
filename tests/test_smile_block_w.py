from __future__ import annotations

import importlib
import os
import shutil
import sys

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
GENO_STREAM = importlib.import_module(f"{PKG.__name__}.geno_stream")
SMILE = importlib.import_module(f"{PKG.__name__}.smile_block_w")

GenoBlockStreamer = GENO_STREAM.GenoBlockStreamer
SmileBlockWeightedOperator = SMILE.SmileBlockWeightedOperator


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


def test_identity_block_weights_match_existing_component_kv():
    X = _make_non_degenerate_genotypes(n=24, m=10, seed=201)
    block_sizes = [4, 6]
    V = jnp.asarray(
        np.random.RandomState(202).standard_normal((X.shape[0], 3)).astype(np.float32)
    )
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=block_sizes,
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [np.eye(4), np.eye(6)],
            normalization="kernel_trace",
            check_psd=True,
        )
        got = np.asarray(op.stacked_block_kv(V))
        ref = np.asarray(st.stacked_component_kv(V))
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=2e-3, rtol=3e-4)
    finally:
        st.close()


def test_weighted_blocks_match_explicit_matrix_reference():
    X = _make_non_degenerate_genotypes(n=28, m=9, seed=203)
    rng = np.random.RandomState(204)
    A0 = rng.standard_normal((4, 4))
    A1 = rng.standard_normal((5, 5))
    W0 = A0 @ A0.T + 0.05 * np.eye(4)
    W1 = A1 @ A1.T + 0.05 * np.eye(5)
    V = rng.standard_normal((X.shape[0], 2)).astype(np.float32)

    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=4,
        component_block_sizes=[4, 5],
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [W0, W1],
            normalization="kernel_trace",
            check_psd=True,
        )
        got = np.asarray(op.kv(jnp.asarray(V)))

        ref = jnp.zeros_like(jnp.asarray(V))
        for block in op.blocks:
            start = block.start
            W = jnp.asarray(block.matrix, dtype=jnp.float32)
            idx = np.arange(start, start + W.shape[0], dtype=np.int64)
            Z = jnp.asarray(st.extract_standardized_columns(idx), dtype=jnp.float32)
            ref = ref + (Z @ (W @ (Z.T @ jnp.asarray(V)))) / jnp.asarray(
                block.normalizer, dtype=jnp.float32
            )
        assert np.allclose(got, np.asarray(ref), atol=3e-3, rtol=5e-4)
    finally:
        st.close()


def test_weighted_hv_matches_manual_block_sum():
    X = _make_non_degenerate_genotypes(n=20, m=7, seed=205)
    V = jnp.asarray(
        np.random.RandomState(206).standard_normal((X.shape[0], 2)).astype(np.float32)
    )
    st = GenoBlockStreamer(
        _ArraySource(X),
        call_width=3,
        component_block_sizes=[3, 4],
        keep_host_stats=True,
    )
    try:
        op = SmileBlockWeightedOperator(
            st,
            [np.eye(3), np.eye(4)],
            normalization="kernel_trace",
            check_psd=True,
        )
        theta_g = jnp.asarray([0.25, 0.4], dtype=jnp.float32)
        theta_e = jnp.asarray(0.35, dtype=jnp.float32)
        got = np.asarray(op.weighted_hv(theta_g, theta_e, V))
        ref = theta_e * np.asarray(V)
        ref = ref + 0.25 * np.asarray(op.block_kv(V, 0))
        ref = ref + 0.4 * np.asarray(op.block_kv(V, 1))
        assert np.allclose(got, ref, atol=2e-3, rtol=3e-4)
    finally:
        st.close()


def test_rejects_non_psd_weight_matrix():
    X = _make_non_degenerate_genotypes(n=12, m=2, seed=207)
    st = GenoBlockStreamer(_ArraySource(X), call_width=2, keep_host_stats=True)
    try:
        with pytest.raises(ValueError, match="positive semidefinite"):
            SmileBlockWeightedOperator(
                st,
                [np.asarray([[1.0, 0.0], [0.0, -0.1]])],
                check_psd=True,
            )
    finally:
        st.close()


def test_load_real_rds_ld_weight_matrix_if_available():
    path = os.environ.get("GPU_REML_SMILE_RDS_FIXTURE")
    if not path:
        pytest.skip("set GPU_REML_SMILE_RDS_FIXTURE to validate a local LD RDS matrix")
    if not os.path.exists(path):
        pytest.skip("configured LD RDS fixture is not available")
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available")

    W = SMILE.load_rds_matrix(path)
    assert W.shape == (1158, 1158)
    assert np.all(np.isfinite(W))
    assert np.max(np.abs(W - W.T)) == pytest.approx(0.0, abs=1e-12)
    eig_min = float(np.linalg.eigvalsh(W)[0])
    assert eig_min > 0.0
