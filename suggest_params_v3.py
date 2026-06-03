"""
suggest_params_v3.py — GPU-only planner with direct ring_depth control.

GPU budget → call_width w (how much data per GPU kernel launch).
ring_depth is set directly (default 32, user-overridable) — no CPU budget model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

_GIB = 1024**3
_W_ALIGN = 256
_GPU_HEADROOM = 0.85
_DEFAULT_GPU_FREE = 16 * _GIB
_AUTO_PRECOND_FLOOR = 1000
_PRECOND_BUILD_OVERSAMPLE = 8
_F32 = 4
_I32 = 4
_NYSTROM_WORK_MATS = 6
_PCG_WORK_MATS = 7
_SLQ_WORK_VECS = 4

_RING_DEPTH_MIN = 4
_RING_DEPTH_MAX = 64
_RING_DEPTH_DEFAULT = 32
_HOST_ANON_BASE = 3 * _GIB  # JAX + Python baseline anon memory
_SOURCE_BUILD_TARGET_BYTES_VARMAJ = 256 * 2**20
_SOURCE_BUILD_TARGET_BYTES_RAW_BED = 512 * 2**20
_SOURCE_BUILD_WORK_BYTES_RAW_BED = 256 * 2**20
_SOURCE_BUILD_RAW_BED_BYTES_PER_VARIANT = 56.0
_SOURCE_BUILD_VARMAJ_BYTES_PER_VARIANT = 16.0


@dataclass
class PlanResult:
    feasible: bool
    call_width: int
    precond_rank: int
    gpu_budget_gib: float
    gpu_fixed_gib: float
    gpu_scratch_gib: float
    target_width: int
    ring_depth: int = _RING_DEPTH_DEFAULT
    note: str = ""
    block_bytes: int = 0
    n_calls: int = 0
    host_ring_gib: float = 0.0
    host_anon_est_gib: float = 0.0
    gpu_peak_gib: float = 0.0
    gpu_steady_peak_gib: float = 0.0
    gpu_precond_build_peak_gib: float = 0.0
    gpu_precompute_peak_gib: float = 0.0
    gpu_projection_peak_gib: float = 0.0
    gpu_slq_peak_gib: float = 0.0
    gpu_streamer_state_gib: float = 0.0
    gpu_precond_state_gib: float = 0.0
    gpu_allocator_pool_limit_gib: float = 0.0
    source_build_chunk_width: int = 0
    source_build_chunks: int = 0
    source_build_est_gib: float = 0.0
    gpu_smile_extra_gib: float = 0.0


@dataclass(frozen=True)
class _CallGeometry:
    n_calls: int
    max_true_width: int
    max_packed_width: int
    max_unpack_width: int
    inflight_packed_row_bytes: float


def _align_down_256(width: float) -> int:
    width_i = int(width)
    if width_i <= 0:
        return _W_ALIGN
    return max(_W_ALIGN, (width_i // _W_ALIGN) * _W_ALIGN)


def _normalize_segments(
    p_list: Sequence[int],
    component_block_sizes: Optional[Sequence[int]],
    *,
    allow_partial_component_blocks: bool = False,
) -> tuple[int, ...]:
    if component_block_sizes is not None:
        segs = tuple(int(x) for x in component_block_sizes)
        if not segs:
            raise ValueError("component_block_sizes must contain at least one block.")
        if any(x <= 0 for x in segs):
            raise ValueError("component_block_sizes must be strictly positive.")
        total = sum(segs)
        source_total = sum(int(p) for p in p_list)
        if p_list and total != source_total:
            if allow_partial_component_blocks and total < source_total:
                return segs
            raise ValueError(
                f"component_block_sizes must sum to total m={source_total}; got {total}."
            )
        return segs
    return tuple(int(p) for p in p_list)


def _build_call_geometry(segment_sizes: Sequence[int], call_width: int) -> _CallGeometry:
    if not segment_sizes:
        return _CallGeometry(
            n_calls=0,
            max_true_width=0,
            max_packed_width=0,
            max_unpack_width=0,
            inflight_packed_row_bytes=0.0,
        )
    widths: list[int] = []
    for seg in segment_sizes:
        remaining = int(seg)
        while remaining > 0:
            width = min(int(call_width), remaining)
            widths.append(width)
            remaining -= width
    max_true_width = max(widths) if widths else 0
    max_packed_width = (max_true_width + 3) // 4 if max_true_width > 0 else 0
    max_unpack_width = 4 * max_packed_width if max_packed_width > 0 else 0
    inflight_blocks = 2 if len(widths) > 1 else int(bool(widths))
    inflight_packed_row_bytes = float(inflight_blocks * max_packed_width)
    return _CallGeometry(
        n_calls=len(widths),
        max_true_width=max_true_width,
        max_packed_width=max_packed_width,
        max_unpack_width=max_unpack_width,
        inflight_packed_row_bytes=inflight_packed_row_bytes,
    )


def _mat_bytes(rows: int, cols: int) -> float:
    return float(_F32 * rows * cols)


def _dense_streamer_state_bytes(
    n: int,
    segment_sizes: Sequence[int],
    *,
    call_width: int,
    n_grm: int,
) -> tuple[float, _CallGeometry]:
    geom = _build_call_geometry(segment_sizes, call_width)
    m_total = sum(int(p) for p in segment_sizes)
    means_by_call = 2.0 * _mat_bytes(geom.n_calls, geom.max_unpack_width)
    means_padded = 2.0 * float(_F32 * (m_total + geom.max_unpack_width))
    geom_arrays = float(2 * geom.n_calls * _I32)
    eff_consts = float((n_grm + 1) * _F32)
    streamer_bytes = means_by_call + means_padded + geom_arrays + eff_consts
    return streamer_bytes, geom


def _projected_core_state_bytes(n: int, n_grm: int, rank: int) -> float:
    if rank <= 0:
        return 0.0
    u_bytes = _mat_bytes(n, rank)
    core_bytes = float(_F32 * n_grm * rank * rank)
    diag_bytes = float(2 * n_grm * _F32)
    return u_bytes + core_bytes + diag_bytes


def _basis_build_live_bytes(
    n: int,
    geom: _CallGeometry,
    rank: int,
) -> float:
    if rank <= 0 or geom.max_unpack_width <= 0:
        return 0.0
    build_rank = rank + _PRECOND_BUILD_OVERSAMPLE
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    inner = _mat_bytes(geom.max_unpack_width, build_rank)
    work_vec = _mat_bytes(n, build_rank)
    return (
        geom.inflight_packed_row_bytes * n
        + wide_block
        + 2.0 * inner
        + _NYSTROM_WORK_MATS * work_vec
    )


def _partitioned_atoms_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
) -> float:
    if rank <= 0 or geom.max_unpack_width <= 0:
        return 0.0
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    block_out = _mat_bytes(geom.max_unpack_width, rank)
    return (
        _projected_core_state_bytes(n, n_grm, rank)
        + geom.inflight_packed_row_bytes * n
        + wide_block
        + block_out
    )


def _generic_atoms_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
) -> float:
    # Dense multi-stream projected-core atoms are now built in a streamed
    # reduction, matching the partitioned path and avoiding K_g U stacks.
    return _partitioned_atoms_live_bytes(
        n,
        geom,
        n_grm=n_grm,
        rank=rank,
    )


def _solve_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
    n_covar: int,
    n_rand_vec: int,
) -> float:
    warm_cols = n_covar + 1 + n_rand_vec
    ai_cols = n_grm + 1
    solve_cols = max(warm_cols, ai_cols)
    if solve_cols <= 0:
        return _projected_core_state_bytes(n, n_grm, rank)
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    inner = _mat_bytes(geom.max_unpack_width, solve_cols)
    warm_vec = _mat_bytes(n, solve_cols)
    return (
        _projected_core_state_bytes(n, n_grm, rank)
        + geom.inflight_packed_row_bytes * n
        + wide_block
        + 2.0 * inner
        + _PCG_WORK_MATS * warm_vec
    )


def _precompute_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
    n_covar: int,
    n_rand_vec: int,
) -> float:
    warm_cols = n_covar + 1 + n_rand_vec
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    inner = _mat_bytes(geom.max_unpack_width, n_rand_vec)
    vrand = _mat_bytes(n, n_rand_vec)
    kvrand = float(n_grm) * vrand
    rhs_const = _mat_bytes(n, warm_cols)
    return (
        _projected_core_state_bytes(n, n_grm, rank)
        + geom.inflight_packed_row_bytes * n
        + wide_block
        + 2.0 * inner
        + vrand
        + kvrand
        + rhs_const
    )


def _projection_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
    n_covar: int,
    n_rand_vec: int,
) -> float:
    warm_cols = n_covar + 1 + n_rand_vec
    proj_cols = n_rand_vec + 1
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    inner = _mat_bytes(geom.max_unpack_width, proj_cols)
    py = _mat_bytes(n, proj_cols)
    gpy = float(n_grm + 1) * py
    sol = _mat_bytes(n, warm_cols)
    return (
        _projected_core_state_bytes(n, n_grm, rank)
        + geom.inflight_packed_row_bytes * n
        + wide_block
        + 2.0 * inner
        + sol
        + py
        + gpy
    )


def _slq_live_bytes(
    n: int,
    geom: _CallGeometry,
    *,
    n_grm: int,
    rank: int,
    slq_samples: int,
) -> float:
    wide_block = _mat_bytes(n, geom.max_unpack_width)
    inner = _mat_bytes(geom.max_unpack_width, slq_samples)
    slq_vec = _mat_bytes(n, slq_samples)
    return (
        _projected_core_state_bytes(n, n_grm, rank)
        + geom.inflight_packed_row_bytes * n
        + wide_block
        + 2.0 * inner
        + _SLQ_WORK_VECS * slq_vec
    )


def _smile_extra_live_bytes(
    total_p: int,
    *,
    rhs_cols: int,
    max_w_block_size: int,
) -> float:
    if total_p <= 0 or rhs_cols <= 0:
        return 0.0
    m = int(total_p)
    rhs = int(rhs_cols)
    xtv_scores_and_call_layout = 3.0 * _mat_bytes(m, rhs)
    local_scores = _mat_bytes(max(0, int(max_w_block_size)), rhs)
    w_staging = float(_F32 * max(0, int(max_w_block_size)) ** 2)
    return xtv_scores_and_call_layout + local_scores + w_staging


def _allocator_pool_limit_bytes() -> Optional[float]:
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").strip().lower() == "platform":
        return None
    try:
        import jax

        dev = jax.devices("gpu")[0]
        stats = dev.memory_stats()
        limit = stats.get("bytes_limit")
        if limit is None:
            return None
        limit_f = float(limit)
        return limit_f if limit_f > 0.0 else None
    except (ImportError, RuntimeError, IndexError, AttributeError, TypeError, ValueError):
        return None


def _estimate_source_build_plan(
    n: int,
    total_p: int,
    *,
    call_width: int,
    source_format: Optional[str],
    arbitrary_component_partition: bool,
) -> tuple[int, int, float]:
    if total_p <= 0 or not arbitrary_component_partition:
        return 0, 0, 0.0

    width_floor = max(1, int(call_width))
    fmt = (source_format or "").strip().lower()
    if fmt == "bed":
        bytes_per_source_variant = max(1, (max(1, int(n)) + 3) // 4)
        width_by_source = max(
            1,
            int(_SOURCE_BUILD_TARGET_BYTES_RAW_BED // bytes_per_source_variant),
        )
        width_by_workspace = max(
            1,
            int(_SOURCE_BUILD_WORK_BYTES_RAW_BED // _SOURCE_BUILD_RAW_BED_BYTES_PER_VARIANT),
        )
        width = min(int(total_p), max(width_floor, min(width_by_source, width_by_workspace)))
        live_bytes = min(
            float(_SOURCE_BUILD_TARGET_BYTES_RAW_BED),
            float(bytes_per_source_variant * width),
        ) + _SOURCE_BUILD_RAW_BED_BYTES_PER_VARIANT * float(width)
    else:
        width_by_block = max(
            1,
            int(_SOURCE_BUILD_TARGET_BYTES_VARMAJ // max(1, int(n))),
        )
        width = min(int(total_p), max(width_floor, width_by_block))
        live_bytes = (
            float(max(1, int(n)) * width)
            + _SOURCE_BUILD_VARMAJ_BYTES_PER_VARIANT * float(width)
        )

    n_chunks = max(1, math.ceil(int(total_p) / max(1, int(width))))
    return int(width), int(n_chunks), float(live_bytes)


def suggest_call_width(
    n_samples: int,
    p_list: Sequence[int],
    *,
    n_grm: Optional[int] = None,
    component_block_sizes: Optional[Sequence[int]] = None,
    precond_type: str = "projected_core",
    gpu_free_bytes: Optional[float] = None,
    gpu_budget_bytes: Optional[float] = None,
    gpu_headroom: float = _GPU_HEADROOM,
    gpu_name: Optional[str] = None,
    n_covar: int = 0,
    n_rand_vec: int = 100,
    slq_samples: int = 30,
    ring_depth: Optional[int] = None,
    source_format: Optional[str] = None,
    arbitrary_component_partition: bool = False,
    smile_mode: bool = False,
    smile_w_block_sizes: Optional[Sequence[int]] = None,
) -> PlanResult:
    """
    GPU-only planner.

    Reports estimated host anon memory for informational purposes.
    Never rejects based on host memory.
    """
    segment_sizes = _normalize_segments(
        p_list,
        component_block_sizes,
        allow_partial_component_blocks=bool(arbitrary_component_partition),
    )
    G_geom = len(segment_sizes)
    G = int(n_grm) if n_grm is not None else G_geom
    if precond_type != "projected_core":
        raise ValueError(
            f"Unsupported precond_type={precond_type!r}. "
            "Only 'projected_core' is available."
        )
    if G == 0:
        return PlanResult(
            feasible=False,
            call_width=_W_ALIGN,
            precond_rank=0,
            gpu_budget_gib=0.0,
            gpu_fixed_gib=0.0,
            gpu_scratch_gib=0.0,
            target_width=0,
            note="No GRMs provided.",
        )

    n = n_samples
    gpu_free = (
        float(gpu_free_bytes)
        if gpu_free_bytes is not None
        else float(_DEFAULT_GPU_FREE)
    )
    gpu_budget = (
        float(gpu_budget_bytes)
        if gpu_budget_bytes is not None
        else gpu_free * float(gpu_headroom)
    )
    gpu_budget_gib = gpu_budget / _GIB

    total_p = sum(max(0, int(sz)) for sz in segment_sizes)
    precond_rank = min(_AUTO_PRECOND_FLOOR, max(0, n), max(0, total_p))
    p_cap = max(segment_sizes) if segment_sizes else _W_ALIGN
    max_smile_w_block_size = (
        max((max(0, int(x)) for x in (smile_w_block_sizes or ())), default=0)
        if smile_mode
        else 0
    )

    def _smile_extra(rhs_cols: int) -> float:
        if not smile_mode:
            return 0.0
        return _smile_extra_live_bytes(
            total_p,
            rhs_cols=rhs_cols,
            max_w_block_size=max_smile_w_block_size,
        )

    def _estimate_live_peaks(width: int) -> tuple[float, float, float, float, float, float, float, float]:
        streamer_state, geom = _dense_streamer_state_bytes(
            n,
            segment_sizes,
            call_width=width,
            n_grm=G,
        )
        precond_state = _projected_core_state_bytes(n, G, precond_rank)
        build_basis = (
            streamer_state
            + _basis_build_live_bytes(n, geom, precond_rank)
            + _smile_extra(precond_rank + _PRECOND_BUILD_OVERSAMPLE)
        )
        if component_block_sizes is not None:
            build_atoms = (
                streamer_state
                + _partitioned_atoms_live_bytes(
                    n,
                    geom,
                    n_grm=G,
                    rank=precond_rank,
                )
                + _smile_extra(precond_rank)
            )
        else:
            build_atoms = (
                streamer_state
                + _generic_atoms_live_bytes(
                    n,
                    geom,
                    n_grm=G,
                    rank=precond_rank,
                )
                + _smile_extra(precond_rank)
            )
        build_peak = max(build_basis, build_atoms)
        precompute_peak = streamer_state + _precompute_live_bytes(
            n,
            geom,
            n_grm=G,
            rank=precond_rank,
            n_covar=n_covar,
            n_rand_vec=n_rand_vec,
        ) + _smile_extra(n_rand_vec)
        solve_cols = max(n_covar + 1 + n_rand_vec, G + 1)
        solve_peak = streamer_state + _solve_live_bytes(
            n,
            geom,
            n_grm=G,
            rank=precond_rank,
            n_covar=n_covar,
            n_rand_vec=n_rand_vec,
        ) + _smile_extra(solve_cols)
        proj_cols = n_rand_vec + 1
        projection_peak = streamer_state + _projection_live_bytes(
            n,
            geom,
            n_grm=G,
            rank=precond_rank,
            n_covar=n_covar,
            n_rand_vec=n_rand_vec,
        ) + _smile_extra(proj_cols)
        slq_peak = streamer_state + _slq_live_bytes(
            n,
            geom,
            n_grm=G,
            rank=precond_rank,
            slq_samples=max(1, int(slq_samples)),
        ) + _smile_extra(max(1, int(slq_samples)))
        live_peak = max(build_peak, precompute_peak, solve_peak, projection_peak, slq_peak)
        return (
            streamer_state,
            precond_state,
            build_peak,
            precompute_peak,
            solve_peak,
            projection_peak,
            slq_peak,
            live_peak,
        )

    lo = _W_ALIGN
    hi = _align_down_256(p_cap)
    best_w = _W_ALIGN
    best_state = _estimate_live_peaks(best_w)
    if best_state[-1] > gpu_budget:
        gpu_feasible = False
        w = best_w
        streamer_state, precond_state, gpu_precond_build_peak, gpu_precompute_peak, gpu_steady_peak, gpu_projection_peak, gpu_slq_peak, gpu_peak = best_state
    else:
        gpu_feasible = True
        while lo <= hi:
            mid = _align_down_256((lo + hi) // 2)
            state = _estimate_live_peaks(mid)
            if state[-1] <= gpu_budget:
                best_w = mid
                best_state = state
                lo = mid + _W_ALIGN
            else:
                hi = mid - _W_ALIGN
        w = best_w
        streamer_state, precond_state, gpu_precond_build_peak, gpu_precompute_peak, gpu_steady_peak, gpu_projection_peak, gpu_slq_peak, gpu_peak = best_state

    scratch = max(0.0, gpu_steady_peak - streamer_state - precond_state)
    target_width = int(p_cap) if component_block_sizes is not None else max(_W_ALIGN, _align_down_256(p_cap))
    allocator_pool_limit = _allocator_pool_limit_bytes()
    gpu_fixed_gib = (streamer_state + precond_state) / _GIB
    w = int(w)
    source_build_chunk_width, source_build_chunks, source_build_live_bytes = _estimate_source_build_plan(
        n,
        total_p,
        call_width=w,
        source_format=source_format,
        arbitrary_component_partition=bool(arbitrary_component_partition),
    )

    if not gpu_feasible:
        note = (
            f"Planner found no feasible fixed-rank projected-core setup under "
            f"{gpu_budget_gib:.1f}GiB "
            f"(live_peak={gpu_peak/_GIB:.1f}GiB, "
            f"build={gpu_precond_build_peak/_GIB:.1f}GiB, "
            f"solve={gpu_steady_peak/_GIB:.1f}GiB)."
        )
        if allocator_pool_limit is not None:
            note += f" allocator_pool_cap={allocator_pool_limit/_GIB:.1f}GiB."
        return PlanResult(
            feasible=False, call_width=w, precond_rank=precond_rank,
            gpu_budget_gib=gpu_budget_gib,
            gpu_fixed_gib=gpu_fixed_gib, gpu_scratch_gib=scratch / _GIB,
            target_width=target_width,
            note=note,
            gpu_peak_gib=gpu_peak / _GIB if math.isfinite(gpu_peak) else float("inf"),
            gpu_steady_peak_gib=gpu_steady_peak / _GIB,
            gpu_precond_build_peak_gib=(
                gpu_precond_build_peak / _GIB
                if math.isfinite(gpu_precond_build_peak) else float("inf")
            ),
            gpu_precompute_peak_gib=gpu_precompute_peak / _GIB,
            gpu_projection_peak_gib=gpu_projection_peak / _GIB,
            gpu_slq_peak_gib=gpu_slq_peak / _GIB,
            gpu_streamer_state_gib=streamer_state / _GIB,
            gpu_precond_state_gib=precond_state / _GIB,
            gpu_allocator_pool_limit_gib=(
                allocator_pool_limit / _GIB if allocator_pool_limit is not None else 0.0
            ),
            source_build_chunk_width=source_build_chunk_width,
            source_build_chunks=source_build_chunks,
            source_build_est_gib=source_build_live_bytes / _GIB,
            gpu_smile_extra_gib=(
                _smile_extra(max(n_covar + 1 + n_rand_vec, G + 1)) / _GIB
                if smile_mode else 0.0
            ),
        )

    # ---- ring_depth: direct control, no CPU budget model ------------------
    _, geom = _dense_streamer_state_bytes(n, segment_sizes, call_width=w, n_grm=G)
    n_calls = max(1, geom.n_calls)
    if ring_depth is not None:
        depth = max(_RING_DEPTH_MIN, min(_RING_DEPTH_MAX, int(ring_depth)))
    else:
        depth = min(n_calls, _RING_DEPTH_DEFAULT)
        depth = max(_RING_DEPTH_MIN, min(_RING_DEPTH_MAX, depth))

    # ---- Host anon estimate (informational only) --------------------------
    block_bytes = int(geom.max_packed_width) * int(n)
    ring_bytes = depth * block_bytes
    anon_est = ring_bytes + _HOST_ANON_BASE

    note = (
        f"precond_type={precond_type} "
        f"precond_rank={precond_rank} "
        f"effective_w={geom.max_true_width} "
        f"build_live={gpu_precond_build_peak/_GIB:.1f}GiB "
        f"precompute_live={gpu_precompute_peak/_GIB:.1f}GiB "
        f"solve_live={gpu_steady_peak/_GIB:.1f}GiB "
        f"projection_live={gpu_projection_peak/_GIB:.1f}GiB "
        f"slq_live={gpu_slq_peak/_GIB:.1f}GiB "
        f"streamer_state={streamer_state/_GIB:.2f}GiB "
        f"precond_state={precond_state/_GIB:.2f}GiB "
    )
    if smile_mode:
        note += (
            f"smile_extra~{_smile_extra(max(n_covar + 1 + n_rand_vec, G + 1))/_GIB:.2f}GiB "
            f"max_w_block={max_smile_w_block_size} "
        )
    note += f"gpu_live_peak={gpu_peak/_GIB:.1f}GiB/{gpu_budget_gib:.1f}GiB "
    if allocator_pool_limit is not None:
        note += (
            f"allocator_pool_cap={allocator_pool_limit/_GIB:.1f}GiB "
            "(default BFC pool; nvidia-smi can exceed live_peak) "
        )
    if source_build_chunk_width > 0:
        note += (
            f"source_build_chunk={source_build_chunk_width} "
            f"source_build_chunks={source_build_chunks} "
            f"source_build_live~{source_build_live_bytes/_GIB:.2f}GiB "
        )
    note += (
        f"ring_depth={depth} "
        f"host_anon~{anon_est/_GIB:.1f}GiB "
        f"(ring={ring_bytes/_GIB:.1f}GiB + base~{_HOST_ANON_BASE/_GIB:.0f}GiB) "
        f"device={gpu_name or 'unknown'}"
    )

    return PlanResult(
        feasible=gpu_feasible, call_width=w, precond_rank=precond_rank,
        gpu_budget_gib=gpu_budget_gib,
        gpu_fixed_gib=gpu_fixed_gib,
        gpu_scratch_gib=scratch / _GIB,
        target_width=target_width,
        ring_depth=depth,
        note=note,
        block_bytes=block_bytes,
        n_calls=n_calls,
        host_ring_gib=ring_bytes / _GIB,
        host_anon_est_gib=anon_est / _GIB,
        gpu_peak_gib=gpu_peak / _GIB,
        gpu_steady_peak_gib=gpu_steady_peak / _GIB,
        gpu_precond_build_peak_gib=gpu_precond_build_peak / _GIB,
        gpu_precompute_peak_gib=gpu_precompute_peak / _GIB,
        gpu_projection_peak_gib=gpu_projection_peak / _GIB,
        gpu_slq_peak_gib=gpu_slq_peak / _GIB,
        gpu_streamer_state_gib=streamer_state / _GIB,
        gpu_precond_state_gib=precond_state / _GIB,
        gpu_allocator_pool_limit_gib=(
            allocator_pool_limit / _GIB if allocator_pool_limit is not None else 0.0
        ),
        source_build_chunk_width=source_build_chunk_width,
        source_build_chunks=source_build_chunks,
        source_build_est_gib=source_build_live_bytes / _GIB,
        gpu_smile_extra_gib=(
            _smile_extra(max(n_covar + 1 + n_rand_vec, G + 1)) / _GIB
            if smile_mode else 0.0
        ),
    )
