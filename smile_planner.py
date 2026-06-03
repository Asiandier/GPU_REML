"""SMILE-only memory planner wrapper.

The production GPU planner in ``suggest_params_v3.py`` is intentionally kept
agnostic to SMILE.  This module reserves SMILE-specific score/W workspaces and
then delegates to the base planner with the remaining GPU budget.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .pipeline_common import run_planner
from .smile_block_w import (
    default_w_device_cache_bytes,
    estimate_bucketed_w_device_cache_bytes,
    estimate_bucketed_w_local_workspace_bytes,
)

_GIB = 1024**3
_F32 = 4
_AUTO_PRECOND_FLOOR = 1000
_PRECOND_BUILD_OVERSAMPLE = 8


def _mat_bytes(rows: int, cols: int, itemsize: int = _F32) -> float:
    return float(max(0, int(rows)) * max(0, int(cols)) * int(itemsize))


def estimate_smile_extra_live_bytes(
    *,
    total_p: int,
    n_samples: int,
    n_grm: int,
    n_covar: int,
    n_rand_vec: int,
    slq_samples: int,
    smile_w_block_sizes: Optional[Sequence[int]] = None,
    w_device_cache_limit: float = 0.0,
) -> float:
    """Return a conservative SMILE workspace reserve in bytes."""

    total_p = max(0, int(total_p))
    if total_p <= 0:
        return 0.0
    max_w_block = max(
        (max(0, int(x)) for x in (smile_w_block_sizes or ())),
        default=0,
    )
    precond_rank = min(_AUTO_PRECOND_FLOOR, max(0, int(n_samples)), total_p)
    rhs_cols = max(
        precond_rank + _PRECOND_BUILD_OVERSAMPLE,
        precond_rank,
        int(n_rand_vec),
        int(n_rand_vec) + 1,
        max(int(n_covar) + 1 + int(n_rand_vec), int(n_grm) + 1),
        max(1, int(slq_samples)),
    )
    xtv_scores_and_call_layout = 3.0 * _mat_bytes(total_p, rhs_cols)
    local_scores = _mat_bytes(max_w_block, rhs_cols)
    w_staging = float(_F32 * max_w_block * max_w_block)
    w_device_cache = estimate_bucketed_w_device_cache_bytes(
        tuple(int(width) for width in (smile_w_block_sizes or ())),
        w_device_cache_limit,
    )
    bucket_workspace = estimate_bucketed_w_local_workspace_bytes(
        tuple(int(width) for width in (smile_w_block_sizes or ())),
        cache_enabled=w_device_cache > 0.0,
    )
    return (
        xtv_scores_and_call_layout
        + local_scores
        + w_staging
        + w_device_cache
        + bucket_workspace
    )


def run_smile_planner(
    *,
    n_samples: int,
    p_list: Sequence[int],
    n_grm: Optional[int] = None,
    component_block_sizes: Optional[Sequence[int]] = None,
    precond_type: str = "projected_core",
    gpu_free: Optional[float],
    gpu_budget: Optional[float] = None,
    n_covar: int,
    n_rand_vec: int,
    slq_samples: int = 30,
    gpu_name: Optional[str] = None,
    ring_depth: Optional[int] = None,
    source_format: Optional[str] = None,
    arbitrary_component_partition: bool = False,
    smile_w_block_sizes: Optional[Sequence[int]] = None,
):
    """Run the base planner after reserving SMILE-specific GPU workspace."""

    total_p = sum(max(0, int(x)) for x in p_list)
    G = int(n_grm) if n_grm is not None else max(1, len(p_list))
    base_budget = (
        float(gpu_budget)
        if gpu_budget is not None
        else float(gpu_free) * 0.85
        if gpu_free is not None
        else None
    )
    w_cache_limit = default_w_device_cache_bytes(base_budget)
    reserve = estimate_smile_extra_live_bytes(
        total_p=total_p,
        n_samples=n_samples,
        n_grm=G,
        n_covar=n_covar,
        n_rand_vec=n_rand_vec,
        slq_samples=slq_samples,
        smile_w_block_sizes=smile_w_block_sizes,
        w_device_cache_limit=w_cache_limit,
    )
    if base_budget is not None and reserve > 0.90 * float(base_budget):
        w_cache_limit = 0.0
        reserve = estimate_smile_extra_live_bytes(
            total_p=total_p,
            n_samples=n_samples,
            n_grm=G,
            n_covar=n_covar,
            n_rand_vec=n_rand_vec,
            slq_samples=slq_samples,
            smile_w_block_sizes=smile_w_block_sizes,
            w_device_cache_limit=w_cache_limit,
        )
    adjusted_budget = (
        max(0.0, float(base_budget) - reserve)
        if base_budget is not None
        else None
    )
    plan = run_planner(
        n_samples=n_samples,
        p_list=p_list,
        n_grm=n_grm,
        component_block_sizes=component_block_sizes,
        precond_type=precond_type,
        gpu_free=gpu_free,
        gpu_budget=adjusted_budget,
        n_covar=n_covar,
        n_rand_vec=n_rand_vec,
        slq_samples=slq_samples,
        gpu_name=gpu_name,
        ring_depth=ring_depth,
        source_format=source_format,
        arbitrary_component_partition=arbitrary_component_partition,
    )
    plan.note += (
        f" smile_extra_reserved={reserve / _GIB:.2f}GiB"
        f" smile_w_device_cache_limit={w_cache_limit / _GIB:.2f}GiB"
        f" base_gpu_budget={base_budget / _GIB:.1f}GiB"
        if base_budget is not None
        else f" smile_extra_reserved={reserve / _GIB:.2f}GiB"
    )
    plan.smile_w_device_cache_bytes = w_cache_limit
    return plan
