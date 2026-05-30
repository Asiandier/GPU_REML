"""
kv_impl.py — GPU K·V implementations.

Fast path (`kv_impl_streamed`):
Host-resident genotype blocks (2-bit packed, C-contiguous per call) are transferred
one-at-a-time via `jax.device_put`.  A **double-buffer pipeline** pre-fetches
block c+1 while the GPU computes on block c, overlapping PCIe transfer
with matmul.

Mathematical contract:
    out = (1/eff_m) * Σ_c  Z_c @ (Z_c^T @ V)
where Z_c is the standardised genotype slice for call c.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from .geno_stream import _PinnedHostBuffer

Array = jnp.ndarray


def _device_put_block(block: np.ndarray, dev) -> Array:
    """Copy host blocks before GPU H2D to avoid view-backed transfer corruption."""
    if getattr(dev, "platform", None) == "cpu":
        return jax.device_put(block, dev)
    if isinstance(block, _PinnedHostBuffer):
        return jax.device_put(np.asarray(block), dev)
    return jax.device_put(np.array(block, copy=True, order="C"), dev)


def _stack_leading_axis(arrays: Sequence[Array]) -> Array:
    """Assemble equal-shaped arrays without building one large concatenate op."""
    if not arrays:
        raise ValueError("_stack_leading_axis requires at least one array.")
    if len(arrays) == 1:
        return arrays[0][None, ...]
    first = arrays[0]
    out = jnp.zeros((len(arrays),) + first.shape, dtype=first.dtype)
    zero_tail = (0,) * first.ndim
    for idx, arr in enumerate(arrays):
        out = jax.lax.dynamic_update_slice(out, arr[None, ...], (idx,) + zero_tail)
    return out


# ======================================================================
# Streaming path — double-buffered H2D + per-block JIT kernel
# ======================================================================

def _unpack_u2_block(g_dev_packed: Array) -> Array:
    """Expand packed 2-bit genotypes to one uint8 value per SNP."""
    g0 = g_dev_packed & jnp.uint8(0x3)
    g1 = (g_dev_packed >> jnp.uint8(2)) & jnp.uint8(0x3)
    g2 = (g_dev_packed >> jnp.uint8(4)) & jnp.uint8(0x3)
    g3 = (g_dev_packed >> jnp.uint8(6)) & jnp.uint8(0x3)
    return jnp.stack((g0, g1, g2, g3), axis=2).reshape(
        g_dev_packed.shape[0], g_dev_packed.shape[1] * 4
    )


def _unpack_impute_center(g_dev_packed, true_width, means_call,
                          inv_call, V, miss_u8):
    """Shared unpack → impute → center logic for all per-block JIT kernels.

    NOT @jax.jit — called inside JIT-compiled callers so XLA traces through it.
    Returns (diff, inv_f) where diff = (g_imputed - mean) and inv_f = 1/sd
    (masked to zero for padding columns).
    """
    g_dev = _unpack_u2_block(g_dev_packed)
    width = g_dev.shape[1]
    fp = V.dtype

    mean_s = means_call[:width]
    inv_s = inv_call[:width]
    cmask = (jnp.arange(width, dtype=true_width.dtype) < true_width).astype(inv_s.dtype)

    mean_f = mean_s.astype(fp)
    inv_f = (inv_s * cmask).astype(fp)

    g_f = g_dev.astype(fp)
    g_imp = jnp.where(g_dev == miss_u8, mean_f[None, :], g_f)
    diff = g_imp - mean_f[None, :]
    return diff, inv_f


@jax.jit
def _kv_one_call_jit(
    g_dev_packed: Array,   # (n, ceil(W / 4)) uint8 packed 2-bit genotypes
    true_width: Array,     # () int32, number of valid SNP columns
    means_call: Array,     # (max_unpack_width,) f32
    inv_call: Array,       # (max_unpack_width,) f32
    V: Array,              # (n, rhs)
    miss_u8: Array,        # () uint8
    acc: Array,            # (n, rhs)
) -> Array:
    """Accumulate one call-block contribution from packed 2-bit genotypes."""
    diff, inv_f = _unpack_impute_center(
        g_dev_packed, true_width, means_call, inv_call, V, miss_u8)
    inner = diff.T @ V
    inv_sq = inv_f * inv_f
    return acc + diff @ (inv_sq[:, None] * inner)


@jax.jit
def _kv_one_call_scaled_jit(
    g_dev_packed: Array,   # (n, ceil(W / 4)) uint8 packed 2-bit genotypes
    true_width: Array,     # () int32, number of valid SNP columns
    means_call: Array,     # (max_unpack_width,) f32
    inv_call: Array,       # (max_unpack_width,) f32
    V: Array,              # (n, rhs)
    miss_u8: Array,        # () uint8
    scale: Array,          # () scalar multiplier (already includes eff_m scaling)
    acc: Array,            # (n, rhs)
) -> Array:
    """Accumulate one scaled call-block contribution from packed genotypes."""
    diff, inv_f = _unpack_impute_center(
        g_dev_packed, true_width, means_call, inv_call, V, miss_u8)
    inner = diff.T @ V
    inv_sq = inv_f * inv_f
    return acc + scale * (diff @ (inv_sq[:, None] * inner))


@jax.jit
def _xtv_one_call_jit(
    g_dev_packed: Array,   # (n, ceil(W / 4)) uint8 packed 2-bit genotypes
    true_width: Array,     # () int32
    means_call: Array,     # (max_unpack_width,) f32
    inv_call: Array,       # (max_unpack_width,) f32
    V: Array,              # (n, rhs)
    miss_u8: Array,        # () uint8
) -> Array:
    """Compute one call-block contribution for X^T @ V from packed genotypes."""
    diff, inv_f = _unpack_impute_center(
        g_dev_packed, true_width, means_call, inv_call, V, miss_u8)
    return inv_f[:, None] * (diff.T @ V)


@jax.jit
def _xtv_scatter_jit(out: Array, block_out: Array, snp_off: Array) -> Array:
    """Write one call-block X^T @ V result into the full output matrix."""
    return jax.lax.dynamic_update_slice(out, block_out, (snp_off, 0))


@jax.jit
def _xtx_from_block_jit(block_out: Array) -> Array:
    """Compute one projected Gram contribution from X_c^T U."""
    return block_out.T @ block_out


def _finalize_projected_core_atom(
    core: Array,
    eff: Array,
    eye: Array,
    *,
    subtract_identity: bool,
) -> Array:
    core = jax.lax.cond(
        eff > 0,
        lambda m: m / eff,
        lambda m: jnp.zeros_like(m),
        core,
    )
    if subtract_identity:
        core = jax.lax.cond(
            eff > 0,
            lambda m: m - eye,
            lambda m: m,
            core,
        )
    return core


@jax.jit
def _zxb_one_call_jit(
    g_dev_packed: Array,
    true_width: Array,
    means_call: Array,
    inv_call: Array,
    b_call: Array,
    miss_u8: Array,
) -> Array:
    """Compute one call-block contribution for Z @ b on test samples."""
    fp = b_call.dtype
    diff, inv_f = _unpack_impute_center(
        g_dev_packed, true_width, means_call, inv_call, b_call, miss_u8)
    width = diff.shape[1]
    cmask = (jnp.arange(width, dtype=true_width.dtype) < true_width).astype(fp)
    b_f = b_call[:width].astype(fp) * cmask
    return diff @ (inv_f * b_f)


def kv_impl_streamed(
    V: Array,
    true_widths: Array,      # device, (n_calls,) int32
    means_by_call: Array,
    inv_by_call: Array,
    eff_m: Array,
    *,
    n: int,
    n_calls: int,
    pop_block: Callable[[int], np.ndarray],
    missing_val: int = 3,
    normalize: bool = True,
) -> Array:
    """
    Double-buffered streaming K·V.

    While the GPU computes on block *c*, the CPU fetches block *c+1*
    from the host cache and queues its H2D transfer asynchronously.
    This overlaps PCIe bandwidth with GPU matmul throughput.
    """
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    fp  = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)

    acc = jnp.zeros((n, V.shape[1]), dtype=fp)
    if n_calls == 0:
        return acc[:, 0] if squeeze else acc

    # ---- double-buffer: pre-fetch block 0 --------------------------------
    g_dev_next = _device_put_block(pop_block(0), dev)

    for c in range(n_calls):
        g_dev_cur = g_dev_next
        # Initiate H2D for block c+1 *before* launching compute on block c.
        # jax.device_put is asynchronous: it returns immediately while the
        # DMA engine copies data over PCIe in the background.
        if c + 1 < n_calls:
            g_dev_next = _device_put_block(pop_block(c + 1), dev)

        acc = _kv_one_call_jit(
            g_dev_cur,
            true_widths[c],
            means_by_call[c],
            inv_by_call[c],
            V,
            miss_u8,
            acc,
        )
        del g_dev_cur   # release device buffer for reuse

    if normalize:
        eff = eff_m.astype(fp)
        acc = jax.lax.cond(eff > 0, lambda o: o / eff, lambda o: o, acc)

    return acc[:, 0] if squeeze else acc


def xtv_impl_streamed(
    V: Array,
    m: int,
    snp_starts: Array,       # device, (n_calls,) int32
    true_widths: Array,      # device, (n_calls,) int32
    means_by_call: Array,
    inv_by_call: Array,
    *,
    n_calls: int,
    pop_block: Callable[[int], np.ndarray],
    missing_val: int = 3,
    normalize: bool = False,
    eff_m: Array | None = None,
) -> Array:
    """
    Double-buffered streaming X^T·V.

    Returns shape (m, rhs) or (m,) when V is 1D.
    """
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)

    out = jnp.zeros((m + means_by_call.shape[1], V.shape[1]), dtype=fp)
    if n_calls == 0:
        out = out[:m]
        return out[:, 0] if squeeze else out

    g_dev_next = _device_put_block(pop_block(0), dev)

    for c in range(n_calls):
        g_dev_cur = g_dev_next
        if c + 1 < n_calls:
            g_dev_next = _device_put_block(pop_block(c + 1), dev)

        block_out = _xtv_one_call_jit(
            g_dev_cur,
            true_widths[c],
            means_by_call[c],
            inv_by_call[c],
            V,
            miss_u8,
        )
        del g_dev_cur   # release device buffer for reuse
        out = _xtv_scatter_jit(out, block_out, snp_starts[c])

    if normalize:
        if eff_m is None:
            raise ValueError("xtv_impl_streamed(normalize=True) requires eff_m.")
        eff = eff_m.astype(fp)
        out = jax.lax.cond(eff > 0, lambda o: o / eff, lambda o: o, out)

    out = out[:m]
    return out[:, 0] if squeeze else out


def xtv_impl_multi_streamed(
    V: Array,
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    *,
    missing_val: int = 3,
    normalize: bool = False,
) -> tuple[Array, ...]:
    """Multi-GRM X^T·V with cross-streamer double buffering."""
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    if not streamers:
        raise ValueError("xtv_impl_multi_streamed requires at least one streamer.")

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    outs = [
        jnp.zeros((int(st.m) + int(st._max_unpack_width), V.shape[1]), dtype=fp)
        for st in streamers
    ]

    if call_plan:
        def _prefetch(plan_idx: int):
            g_idx, c_idx = call_plan[plan_idx]
            st = streamers[g_idx]
            return _device_put_block(st._pop_cached(c_idx), dev)

        g_dev_next = _prefetch(0)
        for p_idx, (g_idx, c_idx) in enumerate(call_plan):
            st = streamers[g_idx]
            g_dev_cur = g_dev_next
            if p_idx + 1 < len(call_plan):
                g_dev_next = _prefetch(p_idx + 1)
            block_out = _xtv_one_call_jit(
                g_dev_cur,
                st._true_widths_dev[c_idx],
                st._means_by_call[c_idx],
                st._inv_by_call[c_idx],
                V,
                miss_u8,
            )
            del g_dev_cur
            outs[g_idx] = _xtv_scatter_jit(outs[g_idx], block_out, st._snp_starts_dev[c_idx])

    results = []
    for g_idx, st in enumerate(streamers):
        out_g = outs[g_idx][: int(st.m)]
        if normalize:
            eff = st._eff_m_const.astype(fp)
            out_g = jax.lax.cond(eff > 0, lambda o: o / eff, lambda o: o, out_g)
        results.append(out_g[:, 0] if squeeze else out_g)
    return tuple(results)


def xtv_impl_multi_streamed_concat(
    V: Array,
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    *,
    missing_val: int = 3,
    normalize: bool = False,
) -> Array:
    """Concatenate multi-stream X^T·V outputs in streamer order."""
    parts = xtv_impl_multi_streamed(
        V,
        streamers,
        call_plan,
        missing_val=missing_val,
        normalize=normalize,
    )
    if not parts:
        squeeze = V.ndim == 1
        empty = jnp.zeros((0,), dtype=V.dtype) if squeeze else jnp.zeros((0, V.shape[1]), dtype=V.dtype)
        return empty
    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate(parts, axis=0)


def zxb_impl_streamed(
    b_by_call: Array,
    true_widths: Array,
    means_by_call: Array,
    inv_by_call: Array,
    *,
    n: int,
    n_calls: int,
    pop_block: Callable[[int], np.ndarray],
    missing_val: int = 3,
    component_ids: np.ndarray | None = None,
    n_components: int = 0,
) -> tuple[Array, tuple[Array, ...]]:
    """Double-buffered streaming test prediction g = Z_test @ b."""
    if b_by_call.ndim != 2:
        raise ValueError("zxb_impl_streamed expects b_by_call with shape (n_calls, max_unpack_width).")
    if b_by_call.shape[0] != int(n_calls):
        raise ValueError(
            f"zxb_impl_streamed b_by_call row mismatch: expected n_calls={int(n_calls)}, "
            f"got {int(b_by_call.shape[0])}."
        )
    if true_widths.ndim != 1 or true_widths.shape[0] != int(n_calls):
        raise ValueError(
            f"zxb_impl_streamed true_widths must have shape ({int(n_calls)},), "
            f"got {true_widths.shape}."
        )
    if means_by_call.ndim != 2 or means_by_call.shape[0] != int(n_calls):
        raise ValueError(
            f"zxb_impl_streamed means_by_call must have shape (n_calls, max_unpack_width), "
            f"got {means_by_call.shape}."
        )
    if inv_by_call.ndim != 2 or inv_by_call.shape != means_by_call.shape:
        raise ValueError(
            f"zxb_impl_streamed inv_by_call shape mismatch: expected {means_by_call.shape}, "
            f"got {inv_by_call.shape}."
        )
    if component_ids is not None:
        component_ids = np.asarray(component_ids, dtype=np.int32).reshape(-1)
        if component_ids.shape[0] != int(n_calls):
            raise ValueError(
                f"zxb_impl_streamed component_ids must have shape ({int(n_calls)},), "
                f"got {component_ids.shape}."
            )
        if int(n_components) <= 0:
            raise ValueError("zxb_impl_streamed requires n_components > 0 when component_ids is provided.")
        if np.any(component_ids < 0) or np.any(component_ids >= int(n_components)):
            raise ValueError(
                f"zxb_impl_streamed component_ids entries must lie in [0, {int(n_components)})."
            )

    fp = b_by_call.dtype
    dev = next(iter(b_by_call.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)

    total = jnp.zeros((n,), dtype=fp)
    comp_outs = [jnp.zeros((n,), dtype=fp) for _ in range(int(n_components))]
    if n_calls == 0:
        return total, tuple(comp_outs)

    g_dev_next = _device_put_block(pop_block(0), dev)
    for c in range(n_calls):
        g_dev_cur = g_dev_next
        if c + 1 < n_calls:
            g_dev_next = _device_put_block(pop_block(c + 1), dev)
        block_out = _zxb_one_call_jit(
            g_dev_cur,
            true_widths[c],
            means_by_call[c],
            inv_by_call[c],
            b_by_call[c],
            miss_u8,
        )
        del g_dev_cur
        total = total + block_out
        if component_ids is not None:
            comp_idx = int(component_ids[c])
            comp_outs[comp_idx] = comp_outs[comp_idx] + block_out

    return total, tuple(comp_outs)


def zxb_impl_multi_streamed(
    b_by_call_list: Sequence[Array],
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    *,
    missing_val: int = 3,
) -> tuple[Array, tuple[Array, ...]]:
    """Multi-GRM test prediction g = Z_test @ b with cross-streamer double buffering."""
    if len(b_by_call_list) != len(streamers):
        raise ValueError(
            f"zxb_impl_multi_streamed b_by_call_list length mismatch: "
            f"expected {len(streamers)}, got {len(b_by_call_list)}."
        )
    if not streamers:
        raise ValueError("zxb_impl_multi_streamed requires at least one streamer.")

    fp = b_by_call_list[0].dtype
    dev = next(iter(b_by_call_list[0].devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamers[0].n)
    total = jnp.zeros((n,), dtype=fp)
    comp_outs = [jnp.zeros((n,), dtype=fp) for _ in streamers]

    for g_idx, (b_by_call, st) in enumerate(zip(b_by_call_list, streamers)):
        if b_by_call.ndim != 2:
            raise ValueError(
                f"zxb_impl_multi_streamed expects b_by_call_list[{g_idx}] to be 2D, got ndim={b_by_call.ndim}."
            )
        if b_by_call.shape[0] != int(st._n_calls):
            raise ValueError(
                f"zxb_impl_multi_streamed row mismatch for component {g_idx}: "
                f"expected n_calls={int(st._n_calls)}, got {int(b_by_call.shape[0])}."
            )

    if not call_plan:
        return total, tuple(comp_outs)

    def _prefetch(plan_idx: int):
        g_idx, c_idx = call_plan[plan_idx]
        st = streamers[g_idx]
        return _device_put_block(st._pop_cached(c_idx), dev)

    g_dev_next = _prefetch(0)
    for p_idx, (g_idx, c_idx) in enumerate(call_plan):
        st = streamers[g_idx]
        g_dev_cur = g_dev_next
        if p_idx + 1 < len(call_plan):
            g_dev_next = _prefetch(p_idx + 1)
        block_out = _zxb_one_call_jit(
            g_dev_cur,
            st._true_widths_dev[c_idx],
            st._means_by_call[c_idx],
            st._inv_by_call[c_idx],
            b_by_call_list[g_idx][c_idx],
            miss_u8,
        )
        del g_dev_cur
        total = total + block_out
        comp_outs[g_idx] = comp_outs[g_idx] + block_out

    return total, tuple(comp_outs)


def _segment_sum_or_zeros(
    values: Array,
    segment_ids: Array,
    *,
    num_segments: int,
    dtype,
    indices_are_sorted: bool = False,
) -> Array:
    """Return segment sums or a correctly-shaped zero matrix for empty input."""
    if values.shape[0] == 0:
        return jnp.zeros((num_segments, values.shape[1]), dtype=dtype)
    return jax.ops.segment_sum(
        values,
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )


def _sparse_values_as_fp(vals: Array, cols: Array, mean: Array, fp) -> Array:
    if vals.dtype == jnp.int8:
        return jnp.where(vals == jnp.int8(3), mean[cols], vals.astype(fp))
    return vals.astype(fp)


def _sparse_xtv_global_mono(
    V: Array,
    *,
    m: int,
    mean: Array,
    inv: Array,
    csc_all_rows: Array,
    csc_all_cols: Array,
    csc_all_vals: Array,
) -> Array:
    fp = V.dtype
    mean = mean.astype(fp)
    inv = inv.astype(fp)
    sum_v = jnp.sum(V, axis=0)
    sparse_vals = _sparse_values_as_fp(csc_all_vals, csc_all_cols, mean, fp)
    xtv = _segment_sum_or_zeros(
        sparse_vals[:, None] * V[csc_all_rows],
        csc_all_cols,
        num_segments=m,
        dtype=fp,
        indices_are_sorted=True,
    )
    return inv[:, None] * (xtv - mean[:, None] * sum_v[None, :])


def _sparse_kv_global_mono(
    V: Array,
    *,
    n: int,
    m: int,
    sum_v: Array,
    mean: Array,
    b: Array,
    inv_sq: Array,
    row_b: Array,
    sum_a0_sq: Array,
    csc_all_rows: Array,
    csc_all_cols: Array,
    csc_all_vals: Array,
) -> Array:
    fp = V.dtype
    mean = mean.astype(fp)
    b = b.astype(fp)
    inv_sq = inv_sq.astype(fp)
    row_b = row_b.astype(fp)
    sum_a0_sq = sum_a0_sq.astype(fp)
    sum_v = sum_v.astype(fp)

    sparse_vals = _sparse_values_as_fp(csc_all_vals, csc_all_cols, mean, fp)
    xtv = _segment_sum_or_zeros(
        sparse_vals[:, None] * V[csc_all_rows],
        csc_all_cols,
        num_segments=m,
        dtype=fp,
        indices_are_sorted=True,
    )
    weighted = inv_sq[:, None] * xtv

    row_sparse = _segment_sum_or_zeros(
        sparse_vals[:, None] * weighted[csc_all_cols],
        csc_all_rows,
        num_segments=n,
        dtype=fp,
    )
    base = sum_a0_sq * sum_v + (b @ xtv)
    return row_sparse + row_b[:, None] * sum_v[None, :] + base[None, :]


_sparse_xtv_global_mono_jit = jax.jit(_sparse_xtv_global_mono, static_argnames=("m",))


def sparse_xtv_shard(
    V: Array,
    *,
    m: int,
    mean: Array,
    inv: Array,
    csc_all_rows: Array,
    csc_all_cols: Array,
    csc_all_vals: Array,
) -> Array:
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]
    out = _sparse_xtv_global_mono_jit(
        V,
        m=m,
        mean=mean,
        inv=inv,
        csc_all_rows=csc_all_rows,
        csc_all_cols=csc_all_cols,
        csc_all_vals=csc_all_vals,
    )
    return out[:, 0] if squeeze else out


def sparse_projected_core_atom_shard(
    U: Array,
    *,
    m: int,
    mean: Array,
    inv: Array,
    csc_all_rows: Array,
    csc_all_cols: Array,
    csc_all_vals: Array,
) -> Array:
    """Return one sparse-shard contribution to U^T K U before eff scaling."""
    if U.ndim != 2:
        raise ValueError("sparse_projected_core_atom_shard expects U with shape (n, k).")
    block = _sparse_xtv_global_mono_jit(
        U,
        m=m,
        mean=mean,
        inv=inv,
        csc_all_rows=csc_all_rows,
        csc_all_cols=csc_all_cols,
        csc_all_vals=csc_all_vals,
    )
    return _xtx_from_block_jit(block)


def sparse_kv_shard(
    V: Array,
    *,
    n: int,
    m: int,
    sum_v: Array,
    mean: Array,
    b: Array,
    inv_sq: Array,
    row_b: Array,
    sum_a0_sq: Array,
    csc_all_rows: Array,
    csc_all_cols: Array,
    csc_all_vals: Array,
) -> Array:
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]
    out = _sparse_kv_global_mono_jit(
        V,
        n=n,
        m=m,
        sum_v=sum_v,
        mean=mean,
        b=b,
        inv_sq=inv_sq,
        row_b=row_b,
        sum_a0_sq=sum_a0_sq,
        csc_all_rows=csc_all_rows,
        csc_all_cols=csc_all_cols,
        csc_all_vals=csc_all_vals,
    )
    return out[:, 0] if squeeze else out


_sparse_kv_global_mono_jit = jax.jit(_sparse_kv_global_mono, static_argnames=("n", "m"))


def kv_impl_multi_streamed_weighted(
    V: Array,
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    theta_g: Array,
    *,
    theta_e: Array | None = None,
    missing_val: int = 3,
) -> Array:
    """
    Multi-GRM weighted K·V with cross-streamer double buffering.

    This is primarily for REML's H·V path. It preserves the streamed host-cache
    design, but pipelines H2D across the flattened global call plan so that
    multi-GRM chromosome partitions (often `n_calls == 1` each) still get
    transfer/compute overlap.
    """
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    if not streamers:
        raise ValueError("kv_impl_multi_streamed_weighted requires at least one streamer.")

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamers[0].n)

    acc = theta_e * V if theta_e is not None else jnp.zeros((n, V.shape[1]), dtype=fp)
    if not call_plan:
        return acc[:, 0] if squeeze else acc

    def _prefetch(plan_idx: int):
        g_idx, c_idx = call_plan[plan_idx]
        st = streamers[g_idx]
        return _device_put_block(st._pop_cached(c_idx), dev)

    g_dev_next = _prefetch(0)
    for p_idx, (g_idx, c_idx) in enumerate(call_plan):
        st = streamers[g_idx]
        g_dev_cur = g_dev_next
        if p_idx + 1 < len(call_plan):
            g_dev_next = _prefetch(p_idx + 1)
        eff = st._eff_m_const.astype(fp)
        scale = jnp.where(
            eff > 0,
            theta_g[g_idx] / eff,
            jnp.asarray(0.0, dtype=fp),
        )
        acc = _kv_one_call_scaled_jit(
            g_dev_cur,
            st._true_widths_dev[c_idx],
            st._means_by_call[c_idx],
            st._inv_by_call[c_idx],
            V,
            miss_u8,
            scale,
            acc,
        )
        del g_dev_cur

    return acc[:, 0] if squeeze else acc


def kv_impl_multi_streamed_stacked(
    V: Array,
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    *,
    missing_val: int = 3,
    normalize: bool = True,
) -> Array:
    """
    Multi-GRM stacked K·V with cross-streamer double buffering.

    Returns one K_g(V) per GRM stacked on axis 0, while flattening all calls
    from all streamers into a single prefetch pipeline.
    """
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    if not streamers:
        raise ValueError("kv_impl_multi_streamed_stacked requires at least one streamer.")

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamers[0].n)
    G = len(streamers)
    outs = [jnp.zeros((n, V.shape[1]), dtype=fp) for _ in range(G)]
    if not call_plan:
        out = _stack_leading_axis(outs)
        if squeeze:
            return out[:, :, 0]
        return out

    def _prefetch(plan_idx: int):
        g_idx, c_idx = call_plan[plan_idx]
        st = streamers[g_idx]
        return _device_put_block(st._pop_cached(c_idx), dev)

    g_dev_next = _prefetch(0)
    for p_idx, (g_idx, c_idx) in enumerate(call_plan):
        st = streamers[g_idx]
        g_dev_cur = g_dev_next
        if p_idx + 1 < len(call_plan):
            g_dev_next = _prefetch(p_idx + 1)
        outs[g_idx] = _kv_one_call_jit(
            g_dev_cur,
            st._true_widths_dev[c_idx],
            st._means_by_call[c_idx],
            st._inv_by_call[c_idx],
            V,
            miss_u8,
            outs[g_idx],
        )
        del g_dev_cur

    if normalize:
        for g_idx, st in enumerate(streamers):
            eff = st._eff_m_const.astype(fp)
            outs[g_idx] = jnp.where(eff > 0, outs[g_idx] / eff, outs[g_idx])

    out = _stack_leading_axis(outs)
    if squeeze:
        return out[:, :, 0]
    return out


def kv_impl_partitioned_component(
    V: Array,
    streamer: object,
    *,
    component_idx: int,
    missing_val: int = 3,
    normalize: bool = True,
) -> Array:
    """Single-stream, single-component K·V for contiguous block partitions."""
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamer.n)
    acc = jnp.zeros((n, V.shape[1]), dtype=fp)

    call_start = int(streamer._component_call_offsets[component_idx])
    call_stop = int(streamer._component_call_offsets[component_idx + 1])
    if call_start >= call_stop:
        return acc[:, 0] if squeeze else acc

    g_dev_next = _device_put_block(streamer._pop_cached(call_start), dev)
    for c in range(call_start, call_stop):
        g_dev_cur = g_dev_next
        if c + 1 < call_stop:
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        acc = _kv_one_call_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            V,
            miss_u8,
            acc,
        )
        del g_dev_cur

    if normalize:
        eff = streamer._component_eff_m_const[component_idx].astype(fp)
        acc = jax.lax.cond(eff > 0, lambda o: o / eff, lambda o: o, acc)

    return acc[:, 0] if squeeze else acc


def kv_impl_partitioned_weighted(
    V: Array,
    streamer: object,
    theta_g: Array,
    *,
    theta_e: Array | None = None,
    missing_val: int = 3,
) -> Array:
    """Single-stream weighted H·V for contiguous block-partitioned GRMs."""
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamer.n)
    acc = theta_e * V if theta_e is not None else jnp.zeros((n, V.shape[1]), dtype=fp)

    if streamer._n_calls == 0:
        return acc[:, 0] if squeeze else acc

    g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
    for c in range(streamer._n_calls):
        g_dev_cur = g_dev_next
        if c + 1 < streamer._n_calls:
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        comp_idx = int(streamer._call_component_ids[c])
        eff = streamer._component_eff_m_const[comp_idx].astype(fp)
        scale = jnp.where(
            eff > 0,
            theta_g[comp_idx] / eff,
            jnp.asarray(0.0, dtype=fp),
        )
        acc = _kv_one_call_scaled_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            V,
            miss_u8,
            scale,
            acc,
        )
        del g_dev_cur

    return acc[:, 0] if squeeze else acc


def kv_impl_partitioned_stacked(
    V: Array,
    streamer: object,
    *,
    missing_val: int = 3,
    normalize: bool = True,
) -> Array:
    """Return one normalized K_g(V) per contiguous block component."""
    squeeze = V.ndim == 1
    if squeeze:
        V = V[:, None]

    fp = V.dtype
    dev = next(iter(V.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    n = int(streamer.n)
    G = int(streamer.n_components)
    outs = [jnp.zeros((n, V.shape[1]), dtype=fp) for _ in range(G)]
    if streamer._n_calls == 0:
        out = _stack_leading_axis(outs)
        return out[:, :, 0] if squeeze else out

    g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
    for c in range(streamer._n_calls):
        g_dev_cur = g_dev_next
        if c + 1 < streamer._n_calls:
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        comp_idx = int(streamer._call_component_ids[c])
        outs[comp_idx] = _kv_one_call_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            V,
            miss_u8,
            outs[comp_idx],
        )
        del g_dev_cur

    if normalize:
        for comp_idx in range(G):
            eff = streamer._component_eff_m_const[comp_idx].astype(fp)
            outs[comp_idx] = jax.lax.cond(
                eff > 0, lambda o: o / eff, lambda o: o, outs[comp_idx]
            )

    out = _stack_leading_axis(outs)
    if squeeze:
        return out[:, :, 0]
    return out


def build_projected_core_atom_streamed(
    U: Array,
    streamer: object,
    *,
    missing_val: int = 3,
    subtract_identity: bool = True,
) -> Array:
    """Build one projected-core atom for a single dense streamer."""
    if U.ndim != 2:
        raise ValueError("build_projected_core_atom_streamed expects U with shape (n, k).")

    fp = U.dtype
    dev = next(iter(U.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    k = int(U.shape[1])
    core = jnp.zeros((k, k), dtype=fp)

    if streamer._n_calls > 0:
        g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
        for c in range(streamer._n_calls):
            g_dev_cur = g_dev_next
            if c + 1 < streamer._n_calls:
                g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
            block_out = _xtv_one_call_jit(
                g_dev_cur,
                streamer._true_widths_dev[c],
                streamer._means_by_call[c],
                streamer._inv_by_call[c],
                U,
                miss_u8,
            )
            del g_dev_cur
            core = core + _xtx_from_block_jit(block_out)

    return _finalize_projected_core_atom(
        core,
        streamer._eff_m_const.astype(fp),
        jnp.eye(k, dtype=fp),
        subtract_identity=subtract_identity,
    )


def build_projected_core_atoms_multi_streamed(
    U: Array,
    streamers: Sequence[object],
    call_plan: Sequence[tuple[int, int]],
    *,
    missing_val: int = 3,
    subtract_identity: bool = True,
) -> Array:
    """Build per-streamer projected-core atoms without materializing K_g U stacks."""
    if U.ndim != 2:
        raise ValueError("build_projected_core_atoms_multi_streamed expects U with shape (n, k).")
    if not streamers:
        raise ValueError("build_projected_core_atoms_multi_streamed requires at least one streamer.")

    fp = U.dtype
    dev = next(iter(U.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    G = len(streamers)
    k = int(U.shape[1])
    cores = [jnp.zeros((k, k), dtype=fp) for _ in range(G)]

    if call_plan:
        def _prefetch(plan_idx: int):
            g_idx, c_idx = call_plan[plan_idx]
            st = streamers[g_idx]
            return _device_put_block(st._pop_cached(c_idx), dev)

        g_dev_next = _prefetch(0)
        for p_idx, (g_idx, c_idx) in enumerate(call_plan):
            st = streamers[g_idx]
            g_dev_cur = g_dev_next
            if p_idx + 1 < len(call_plan):
                g_dev_next = _prefetch(p_idx + 1)
            block_out = _xtv_one_call_jit(
                g_dev_cur,
                st._true_widths_dev[c_idx],
                st._means_by_call[c_idx],
                st._inv_by_call[c_idx],
                U,
                miss_u8,
            )
            del g_dev_cur
            cores[g_idx] = cores[g_idx] + _xtx_from_block_jit(block_out)

    eye = jnp.eye(k, dtype=fp)
    for g_idx, st in enumerate(streamers):
        cores[g_idx] = _finalize_projected_core_atom(
            cores[g_idx],
            st._eff_m_const.astype(fp),
            eye,
            subtract_identity=subtract_identity,
        )

    return _stack_leading_axis(cores)


def build_projected_core_atoms_partitioned(
    U: Array,
    streamer: object,
    *,
    missing_val: int = 3,
    subtract_identity: bool = True,
) -> Array:
    """
    Build per-component projected cores T_g = U^T K_g U (optionally minus I).

    This is the streamed reduction needed by the projected-core preconditioner
    for contiguous single-stream multi-GRM mode.
    """
    if U.ndim != 2:
        raise ValueError("build_projected_core_atoms_partitioned expects U with shape (n, k).")

    fp = U.dtype
    dev = next(iter(U.devices()))
    miss_u8 = jnp.asarray(np.uint8(missing_val), dtype=jnp.uint8)
    G = int(streamer.n_components)
    k = int(U.shape[1])
    cores = [jnp.zeros((k, k), dtype=fp) for _ in range(G)]

    if streamer._n_calls == 0:
        return _stack_leading_axis(cores)

    g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
    for c in range(streamer._n_calls):
        g_dev_cur = g_dev_next
        if c + 1 < streamer._n_calls:
            g_dev_next = _device_put_block(streamer._pop_cached(c + 1), dev)
        block_out = _xtv_one_call_jit(
            g_dev_cur,
            streamer._true_widths_dev[c],
            streamer._means_by_call[c],
            streamer._inv_by_call[c],
            U,
            miss_u8,
        )
        del g_dev_cur
        comp_idx = int(streamer._call_component_ids[c])
        cores[comp_idx] = cores[comp_idx] + _xtx_from_block_jit(block_out)

    eye = jnp.eye(k, dtype=fp)
    for comp_idx in range(G):
        cores[comp_idx] = _finalize_projected_core_atom(
            cores[comp_idx],
            streamer._component_eff_m_const[comp_idx].astype(fp),
            eye,
            subtract_identity=subtract_identity,
        )

    return _stack_leading_axis(cores)


# ======================================================================
# Helpers (unchanged interface)
# ======================================================================

def build_packed_stats(
    means_flat: Array,
    inv_flat: Array,
    pad_width: int,
) -> tuple[Array, Array]:
    """Right-pad stats so dynamic_slice is always in-bounds."""
    pad = jnp.zeros(pad_width, dtype=means_flat.dtype)
    return jnp.concatenate([means_flat, pad]), jnp.concatenate([inv_flat, pad])
