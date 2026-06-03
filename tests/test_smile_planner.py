from __future__ import annotations

from GPU_REML.smile_planner import (
    estimate_smile_extra_live_bytes,
    run_smile_planner,
)
from GPU_REML.suggest_params_v3 import suggest_call_width


def test_smile_extra_workspace_estimate_includes_w_staging():
    without_w = estimate_smile_extra_live_bytes(
        total_p=100_000,
        n_samples=20_000,
        n_grm=2,
        n_covar=4,
        n_rand_vec=32,
        slq_samples=4,
        smile_w_block_sizes=[],
    )
    with_w = estimate_smile_extra_live_bytes(
        total_p=100_000,
        n_samples=20_000,
        n_grm=2,
        n_covar=4,
        n_rand_vec=32,
        slq_samples=4,
        smile_w_block_sizes=[400, 600, 500],
    )
    assert without_w > 0.0
    assert with_w > without_w


def test_smile_planner_isolated_from_base_planner():
    base = suggest_call_width(
        n_samples=20_000,
        p_list=[100_000],
        n_grm=2,
        gpu_free_bytes=float(24 * 2**30),
        gpu_budget_bytes=float(24 * 2**30),
        n_rand_vec=32,
    )
    smile = run_smile_planner(
        n_samples=20_000,
        p_list=[100_000],
        n_grm=2,
        gpu_free=float(24 * 2**30),
        gpu_budget=float(24 * 2**30),
        n_covar=0,
        n_rand_vec=32,
        smile_w_block_sizes=[400, 600, 500],
    )
    assert smile.feasible
    assert smile.call_width <= base.call_width
    assert "smile_extra_reserved=" in smile.note
    assert not hasattr(base, "gpu_smile_extra_gib")
