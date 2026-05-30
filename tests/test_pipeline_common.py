"""Integration-style tests for shared pipeline helpers and sparse fixed-point logic."""

import importlib
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_REPO_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PKG = os.path.basename(_REPO_ROOT)

_COMMON = importlib.import_module(f"{_PKG}.pipeline_common")
_GENO_STREAM = importlib.import_module(f"{_PKG}.geno_stream")
_SPARSE = importlib.import_module(f"{_PKG}.run_sparse_reml_pipeline")

run_planner = _COMMON.run_planner
cleanup_path = _COMMON.cleanup_path
make_nonbed_input_fam = _COMMON.make_nonbed_input_fam
GenoBlockStreamer = _GENO_STREAM.GenoBlockStreamer
MultiGRMIndex = _SPARSE.MultiGRMIndex
_fixed_point_skip_reml = _SPARSE._fixed_point_skip_reml
_outside_kkt_violators = _SPARSE._outside_kkt_violators


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


class TestRunPlanner:
    def test_typical_call_width_plan(self):
        plan = run_planner(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free=float(24 * 2**30),
            n_covar=10,
            n_rand_vec=100,
            gpu_name="test-gpu",
        )
        assert plan.feasible
        assert plan.call_width > 0
        assert plan.precond_rank > 0

    def test_infeasible_when_gpu_too_small(self):
        try:
            run_planner(
                n_samples=50_000,
                p_list=[500_000],
                gpu_free=float(0.5 * 2**30),
                n_covar=10,
                n_rand_vec=300,
                gpu_name="tiny-gpu",
            )
        except SystemExit as e:
            assert "Planner found no feasible configuration" in str(e)
        else:
            raise AssertionError("Expected SystemExit for infeasible planner run")

    def test_call_width_capped_by_max_p(self):
        plan = run_planner(
            n_samples=50_000,
            p_list=[20_000],
            gpu_free=float(80 * 2**30),
            n_covar=10,
            n_rand_vec=50,
            gpu_name="test-gpu",
        )
        assert plan.feasible
        assert plan.call_width <= 20_000 or plan.call_width == 4096


class TestSparseFixedPointSkip:
    def test_first_repeat_can_continue_when_more_stable_rounds_required(self):
        stop_now, rounds = _fixed_point_skip_reml(
            support_same=True,
            theta_stable_prev=True,
            has_reml_refit=True,
            stable_rounds=0,
            support_stable_rounds=2,
        )
        assert not stop_now
        assert rounds == 1

    def test_second_repeat_stops_when_threshold_reached(self):
        stop_now, rounds = _fixed_point_skip_reml(
            support_same=True,
            theta_stable_prev=True,
            has_reml_refit=True,
            stable_rounds=1,
            support_stable_rounds=2,
        )
        assert stop_now
        assert rounds == 2

    def test_no_repeat_does_not_change_rounds(self):
        stop_now, rounds = _fixed_point_skip_reml(
            support_same=False,
            theta_stable_prev=True,
            has_reml_refit=True,
            stable_rounds=3,
            support_stable_rounds=2,
        )
        assert not stop_now
        assert rounds == 3


class TestSparseLassoKKT:
    def test_outside_kkt_violators_ignore_candidate_and_apply_tolerance(self):
        scores = np.array([0.05, 1.01, 1.20, 0.99, 1.15], dtype=np.float64)
        candidate = np.array([2], dtype=np.int64)

        violators, max_outside, threshold = _outside_kkt_violators(
            score_abs=scores,
            candidate=candidate,
            lam=1.0,
            abs_tol=0.02,
            rel_tol=0.0,
        )

        np.testing.assert_array_equal(violators, np.array([4], dtype=np.int64))
        assert np.isclose(max_outside, 1.15)
        assert np.isclose(threshold, 1.02)

    def test_outside_kkt_violators_can_certify_empty_set(self):
        scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        violators, max_outside, threshold = _outside_kkt_violators(
            score_abs=scores,
            candidate=np.array([], dtype=np.int64),
            lam=0.5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )

        assert violators.size == 0
        assert np.isclose(max_outside, 0.3)
        assert threshold > 0.5


class TestMultiGRMIndex:
    def test_xtv_all_matches_concatenated_split_streamers(self):
        X = _make_non_degenerate_genotypes(n=14, m=9, seed=17)
        splits = [X[:, :4], X[:, 4:7], X[:, 7:]]
        streamers = [
            GenoBlockStreamer(_ArraySource(block), call_width=4, keep_host_stats=True)
            for block in splits
        ]
        try:
            call_plan = tuple(
                (g_idx, c_idx)
                for g_idx, st in enumerate(streamers)
                for c_idx in range(int(st._n_calls))
            )
            grm_index = MultiGRMIndex(streamers, call_plan=call_plan)
            u = jnp.asarray(
                np.random.RandomState(18).standard_normal((X.shape[0],)).astype(np.float32)
            )

            got = grm_index.xtv_all(u, normalize=False)
            ref = np.concatenate(
                [
                    np.asarray(st.xtv(u, normalize=False), dtype=np.float64)
                    for st in streamers
                ],
                axis=0,
            )

            assert got.shape == ref.shape == (X.shape[1],)
            assert np.allclose(got, ref, atol=1e-5)
        finally:
            for st in streamers:
                st.close()

    def test_component_partition_uses_cache_indices_and_tracks_source_indices(self):
        X = _make_non_degenerate_genotypes(n=12, m=6, seed=21)
        groups = [
            np.array([4, 1], dtype=np.int64),
            np.array([5, 0, 2, 3], dtype=np.int64),
        ]
        st = GenoBlockStreamer(
            _ArraySource(X),
            call_width=3,
            keep_host_stats=True,
            component_variant_indices=groups,
        )
        try:
            grm_index = MultiGRMIndex([st], component_variant_indices=groups)
            np.testing.assert_array_equal(grm_index.m_per_grm, np.array([2, 4]))
            assert grm_index.m_total == 6

            cache_idx = np.array([0, 1, 2, 5], dtype=np.int64)
            np.testing.assert_array_equal(
                grm_index.source_variant_indices(cache_idx),
                np.array([4, 1, 5, 3], dtype=np.int64),
            )

            got = grm_index.extract_standardized_columns(cache_idx)
            ref = st.extract_standardized_columns(cache_idx)
            np.testing.assert_allclose(got, ref)
        finally:
            st.close()

class TestPgenFam:
    def test_make_nonbed_input_fam_from_pgen_psam(self, tmp_path):
        psam = tmp_path / "data.psam"
        psam.write_text("#FID IID SEX\nf1 i1 1\nf2 i2 2\n")
        fam_path = make_nonbed_input_fam(pgen_prefix=str(tmp_path / "data"))
        try:
            lines = Path(fam_path).read_text().splitlines()
            assert lines == ["i1\ti1\t0\t0\t0\t-9", "i2\ti2\t0\t0\t0\t-9"]
        finally:
            cleanup_path(fam_path)
