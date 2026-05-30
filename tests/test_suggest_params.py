"""Unit tests for the closed-form GPU-budget planner."""

import os
import sys
import importlib

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_REPO_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PKG = os.path.basename(_REPO_ROOT)
_PLAN = importlib.import_module(f"{_PKG}.suggest_params_v3")

suggest_call_width = _PLAN.suggest_call_width
PlanResult = _PLAN.PlanResult


class TestFeasibility:
    def test_typical_case_feasible(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
        )
        assert plan.feasible
        assert plan.call_width > 0

    def test_empty_p_list(self):
        plan = suggest_call_width(n_samples=50_000, p_list=[], gpu_free_bytes=float(24 * 2**30))
        assert not plan.feasible
        assert "No GRMs" in plan.note

    def test_tiny_gpu_infeasible(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(0.25 * 2**30),
            n_rand_vec=100,
        )
        assert not plan.feasible

    def test_budget_exhausted_by_fixed_state(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(4 * 2**30),
            n_rand_vec=300,
        )
        assert plan.feasible
        assert plan.precond_rank == 1000
        assert plan.call_width > 0

    def test_precond_rank_is_capped_by_total_snp_count(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[128],
            gpu_free_bytes=float(24 * 2**30),
        )
        assert plan.precond_rank <= 128


class TestOutputConstraints:
    def test_call_width_aligned(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000], gpu_free_bytes=float(24 * 2**30)
        )
        assert plan.call_width % 256 == 0

    def test_gpu_peak_within_budget(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000], gpu_free_bytes=float(24 * 2**30)
        )
        assert plan.gpu_peak_gib <= plan.gpu_budget_gib * 1.01

    def test_precond_build_peak_within_budget(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000], gpu_free_bytes=float(24 * 2**30)
        )
        assert plan.gpu_precond_build_peak_gib <= plan.gpu_budget_gib * 1.01

    def test_zero_gpu_free_is_not_replaced_by_default(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=0.0,
        )
        assert plan.gpu_budget_gib == 0.0
        assert not plan.feasible

    def test_zero_gpu_budget_is_not_replaced_by_headroom_budget(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            gpu_budget_bytes=0.0,
        )
        assert plan.gpu_budget_gib == 0.0
        assert not plan.feasible


class TestRingDepth:
    def test_default_ring_depth(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000], gpu_free_bytes=float(24 * 2**30)
        )
        assert 4 <= plan.ring_depth <= 64

    def test_explicit_ring_depth(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            ring_depth=16,
        )
        assert plan.ring_depth == 16

    def test_ring_depth_clamped_min(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            ring_depth=1,
        )
        assert plan.ring_depth == 4  # clamped to _RING_DEPTH_MIN

    def test_ring_depth_clamped_max(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            ring_depth=200,
        )
        assert plan.ring_depth == 64  # clamped to _RING_DEPTH_MAX

    def test_ring_depth_capped_by_n_calls(self):
        """When there are few calls, default depth should not exceed n_calls."""
        plan = suggest_call_width(
            n_samples=50_000, p_list=[100_000],
            gpu_free_bytes=float(40 * 2**30),  # big GPU → big call_width → few calls
        )
        # With 40G and 100K SNPs, call_width could cover all SNPs in 1-2 calls
        assert plan.ring_depth <= max(plan.n_calls, 4)

    def test_host_anon_estimate_reported(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000], gpu_free_bytes=float(24 * 2**30)
        )
        assert plan.host_anon_est_gib > 0.0
        assert plan.host_ring_gib > 0.0
        assert plan.block_bytes > 0

    def test_no_cpu_reject(self):
        """Even with small ring_depth, planner should never set feasible=False
        due to host memory — only GPU budget can make it infeasible."""
        plan = suggest_call_width(
            n_samples=50_000, p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            ring_depth=4,
        )
        assert plan.feasible


class TestMultiGRM:
    def test_two_grm_feasible(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[300_000, 200_000],
            gpu_free_bytes=float(40 * 2**30),
        )
        assert plan.feasible

    def test_multi_grm_larger_gpu_peak(self):
        plan_1 = suggest_call_width(
            n_samples=50_000, p_list=[500_000],
            gpu_free_bytes=float(40 * 2**30), n_rand_vec=100,
        )
        plan_2 = suggest_call_width(
            n_samples=50_000, p_list=[500_000, 500_000],
            gpu_free_bytes=float(40 * 2**30), n_rand_vec=100,
        )
        assert plan_2.gpu_peak_gib >= plan_1.gpu_peak_gib * 0.95

    def test_projected_core_many_components_still_feasible(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[1_000_000],
            n_grm=100,
            precond_type="projected_core",
            gpu_free_bytes=float(48 * 2**30),
            gpu_budget_bytes=float(48 * 2**30),
            n_rand_vec=100,
        )
        assert plan.feasible
        assert plan.precond_rank > 0

    def test_projected_core_build_peak_matches_many_grm_closed_form(self):
        n = 50_000
        G = 60
        n_covar = 10
        n_rand_vec = 100
        plan = suggest_call_width(
            n_samples=n,
            p_list=[1_000_000],
            n_grm=G,
            gpu_free_bytes=float(80 * 2**30),
            gpu_budget_bytes=float(80 * 2**30),
            n_covar=n_covar,
            n_rand_vec=n_rand_vec,
        )
        assert plan.precond_rank > 0
        k = int(plan.precond_rank)
        segment_sizes = (1_000_000,)
        streamer_state, geom = _PLAN._dense_streamer_state_bytes(
            n,
            segment_sizes,
            call_width=plan.call_width,
            n_grm=G,
        )
        expected_gib = max(
            streamer_state + _PLAN._basis_build_live_bytes(n, geom, k),
            streamer_state + _PLAN._generic_atoms_live_bytes(
                n,
                geom,
                n_grm=G,
                rank=k,
            ),
        ) / (1024**3)
        assert plan.gpu_precond_build_peak_gib == pytest.approx(expected_gib, rel=1e-7)

    def test_component_block_sizes_reduce_many_grm_build_peak(self):
        plain = suggest_call_width(
            n_samples=50_000,
            p_list=[1_000_000],
            n_grm=60,
            gpu_free_bytes=float(80 * 2**30),
            gpu_budget_bytes=float(80 * 2**30),
            n_covar=10,
            n_rand_vec=100,
        )
        partitioned = suggest_call_width(
            n_samples=50_000,
            p_list=[1_000_000],
            n_grm=60,
            component_block_sizes=[10_000] * 100,
            gpu_free_bytes=float(80 * 2**30),
            gpu_budget_bytes=float(80 * 2**30),
            n_covar=10,
            n_rand_vec=100,
        )
        assert partitioned.gpu_precond_build_peak_gib < plain.gpu_precond_build_peak_gib
        assert partitioned.call_width <= 10_000

    def test_component_block_sizes_still_searches_smaller_call_widths(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[639_577],
            n_grm=1,
            component_block_sizes=[639_577],
            gpu_free_bytes=float(48 * 2**30),
            gpu_budget_bytes=float(48 * 2**30),
            n_covar=0,
            n_rand_vec=100,
        )
        assert plan.feasible
        assert plan.call_width < 639_577

    def test_arbitrary_bed_partition_gets_separate_source_build_chunk_plan(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[1_000_000],
            n_grm=100,
            component_block_sizes=[10_000] * 100,
            gpu_free_bytes=float(48 * 2**30),
            gpu_budget_bytes=float(48 * 2**30),
            n_rand_vec=100,
            source_format="bed",
            arbitrary_component_partition=True,
        )
        assert plan.feasible
        assert plan.source_build_chunk_width >= plan.call_width
        assert plan.source_build_chunk_width > 0
        assert plan.source_build_chunks > 0
        assert plan.source_build_est_gib > 0.0
        assert "source_build_chunk=" in plan.note

    def test_non_arbitrary_plan_does_not_emit_source_build_chunk_plan(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
        )
        assert plan.source_build_chunk_width == 0
        assert plan.source_build_chunks == 0
        assert plan.source_build_est_gib == 0.0


class TestRegression:
    def test_solve_live_bytes_accounts_for_ai_rhs_width(self):
        n = 1_000
        geom = _PLAN._CallGeometry(
            n_calls=4,
            max_true_width=256,
            max_packed_width=256,
            max_unpack_width=256,
            inflight_packed_row_bytes=8.0,
        )
        got = _PLAN._solve_live_bytes(
            n,
            geom,
            n_grm=80,
            rank=16,
            n_covar=2,
            n_rand_vec=5,
        )
        solve_cols = 81  # max(n_covar + 1 + n_rand_vec, n_grm + 1)
        expected = (
            _PLAN._projected_core_state_bytes(n, 80, 16)
            + geom.inflight_packed_row_bytes * n
            + _PLAN._mat_bytes(n, geom.max_unpack_width)
            + 2.0 * _PLAN._mat_bytes(geom.max_unpack_width, solve_cols)
            + _PLAN._PCG_WORK_MATS * _PLAN._mat_bytes(n, solve_cols)
        )
        assert got == pytest.approx(expected, rel=1e-7)

    def test_width_matches_closed_form(self):
        plan = suggest_call_width(
            n_samples=50_000,
            p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
            n_covar=10,
            n_rand_vec=100,
        )
        assert plan.feasible
        assert plan.gpu_budget_gib > 0.0


class TestEdgeCases:
    def test_unsupported_precond_type_rejected(self):
        with pytest.raises(ValueError, match="Only 'projected_core'"):
            suggest_call_width(
                n_samples=50_000,
                p_list=[1_000_000],
                precond_type="unsupported",
                gpu_free_bytes=float(24 * 2**30),
            )

    def test_single_snp(self):
        plan = suggest_call_width(
            n_samples=50_000, p_list=[1], gpu_free_bytes=float(24 * 2**30)
        )
        assert plan.feasible
        assert plan.call_width >= 256

    def test_very_large_n(self):
        plan = suggest_call_width(
            n_samples=500_000,
            p_list=[100_000],
            gpu_free_bytes=float(80 * 2**30),
            n_rand_vec=50,
        )
        assert isinstance(plan, PlanResult)

    def test_precond_rank_fixed_to_1000_for_large_n(self):
        p1 = suggest_call_width(n_samples=50_000, p_list=[1_000_000], gpu_free_bytes=float(24 * 2**30))
        p2 = suggest_call_width(n_samples=200_000, p_list=[1_000_000], gpu_free_bytes=float(24 * 2**30))
        p3 = suggest_call_width(n_samples=400_000, p_list=[1_000_000], gpu_free_bytes=float(24 * 2**30))
        assert p1.precond_rank == 1000
        assert p2.precond_rank == 1000
        assert p3.precond_rank == 1000

    def test_budget_pressure_shrinks_call_width_not_rank(self):
        rich = suggest_call_width(n_samples=200_000, p_list=[3_000_000], gpu_free_bytes=float(80 * 2**30))
        tight = suggest_call_width(n_samples=200_000, p_list=[3_000_000], gpu_free_bytes=float(24 * 2**30))
        assert rich.feasible
        assert tight.feasible
        assert rich.precond_rank == tight.precond_rank == 1000
        assert rich.call_width >= tight.call_width

    def test_precond_rank_is_capped_by_n(self):
        plan = suggest_call_width(
            n_samples=500,
            p_list=[500_000],
            gpu_free_bytes=float(24 * 2**30),
        )
        assert plan.precond_rank == 500

    def test_streamed_atoms_can_keep_large_n_fixed_rank_case_feasible(self):
        tight = suggest_call_width(
            n_samples=400_000,
            p_list=[1_000_000],
            n_grm=60,
            gpu_free_bytes=float(80 * 2**30),
            gpu_budget_bytes=float(48 * 2**30),
            n_covar=14,
            n_rand_vec=100,
        )
        assert tight.feasible
        assert tight.precond_rank == 1000
        assert tight.call_width > 0
        assert "gpu_live_peak=" in tight.note
