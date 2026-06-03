from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from datetime import datetime
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

logger = logging.getLogger(__name__)

from .geno_stream import BedBlockStreamer, GenoBlockStreamer, _ensure_on_device
from .pipeline_common import fam_order_mismatch as _fam_order_mismatch
from .pipeline_common import resolve_cpu_threads as _resolve_cpu_threads
from .pcg import pcg_solve
from .precond import (
    ProjectedCorePrecondConf,
    build_lowrank_basis,
    make_precond,
    scalar_diag_from_precond_conf,
)
from .reml import fit_reml, standardize_response

@dataclasses.dataclass
class FitConfig:
    bed_prefix: str | Sequence[str] = ""
    sources: Sequence | None = None        # list of GenoBlockSource
    rare_bed_prefix: str | Sequence[str] = ""
    rare_sources: Sequence | None = None
    vc_block_sizes: Sequence[int] | None = None
    component_variant_indices: Sequence[Sequence[int]] | None = None
    smile_w_files: Sequence[str] | None = None
    smile_w_file_groups: Sequence[Sequence[str]] | None = None
    smile_weight_matrices: Sequence[np.ndarray] | None = None
    smile_weight_matrix_groups: Sequence[Sequence[np.ndarray]] | None = None
    smile_identity: bool = False
    smile_identity_block_size: int | None = None
    smile_normalization: str = "kernel_trace"
    smile_diag_mode: str = "mean"
    smile_check_psd: bool = True
    smile_strict_coverage: bool = True
    standardization_overrides: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
    sample_mask: np.ndarray | None = None  # bool mask for sample subsetting
    call_width: int = 131072
    device: str | None = "auto"
    keep_host_stats: bool = True
    cpu_threads: int | None = None
    n_rand_vec: int = 100
    max_pcg_iters: int = 400
    minq_iter: int = 50
    seed: int = 0
    slq_samples: int = 4
    slq_m: int = 8
    slq_mode: str = "projected_core_residual"
    precond_type: str = "projected_core"
    precond_rank: int = 500
    precond_refresh_reldp: float = 0.20
    pcg_ridge: float = 1e-6
    effect_pcg_tol: float = 1e-3
    n_reml_reps: int = 1
    ring_depth: int | None = None
    source_build_chunk_width: int | None = None
    verbose: bool = True
    gpu_budget_bytes: float | None = None


@dataclasses.dataclass
class EffectEstimates:
    """Post-REML effect estimates on the internally standardized phenotype scale."""

    fixed_effects: jnp.ndarray
    random_effect: jnp.ndarray
    random_effect_components: tuple[jnp.ndarray, ...]
    snp_effects: tuple[jnp.ndarray, ...]
    pcg_rel_res: float
    pcg_iters: int
    y_mean: float
    y_scale: float


@dataclasses.dataclass
class PredictionEstimates:
    """Test-set predictions derived from standardized-scale effect estimates."""

    fixed_effect: jnp.ndarray
    random_effect: jnp.ndarray
    random_effect_components: tuple[jnp.ndarray, ...]
    y_pred_std: jnp.ndarray
    y_pred: jnp.ndarray


@dataclasses.dataclass
class FitResult:
    var_components: jnp.ndarray
    history: list[dict[str, object]]
    rep_var_components: Optional[jnp.ndarray] = None
    jackknife_se_var: Optional[jnp.ndarray] = None
    jackknife_se_h2: Optional[float] = None
    effects: Optional[EffectEstimates] = None


@dataclasses.dataclass(frozen=True)
class _OperatorBundle:
    K_mvs: tuple
    diag_list: tuple[jnp.ndarray, ...]
    weighted_hv: object
    stacked_kv: object
    projected_core_atoms: object | None = None


def _validate_dense_prediction_streamers(
    train_streamers: Sequence[object],
    test_streamers: Sequence[object],
) -> None:
    if len(train_streamers) != len(test_streamers):
        raise ValueError(
            f"Prediction component count mismatch: expected {len(train_streamers)}, got {len(test_streamers)}."
        )
    for idx, (st_train, st_test) in enumerate(zip(train_streamers, test_streamers)):
        if int(st_train.m) != int(st_test.m):
            raise ValueError(
                f"Prediction SNP count mismatch for component {idx}: "
                f"expected {int(st_train.m)}, got {int(st_test.m)}."
            )
        if int(st_train._n_calls) != int(st_test._n_calls):
            raise ValueError(
                f"Prediction call geometry mismatch for component {idx}: "
                f"expected {int(st_train._n_calls)} calls, got {int(st_test._n_calls)}."
            )
        if not np.array_equal(
            np.asarray(st_train._call_snp_starts, dtype=np.int64),
            np.asarray(st_test._call_snp_starts, dtype=np.int64),
        ):
            raise ValueError(f"Prediction SNP call starts mismatch for component {idx}.")
        if not np.array_equal(
            np.asarray(st_train._call_true_widths, dtype=np.int32),
            np.asarray(st_test._call_true_widths, dtype=np.int32),
        ):
            raise ValueError(f"Prediction call widths mismatch for component {idx}.")
        if bool(getattr(st_train, "has_component_partition", False)) != bool(
            getattr(st_test, "has_component_partition", False)
        ):
            raise ValueError(f"Prediction partitioning mode mismatch for component {idx}.")
        train_cache_to_source = getattr(st_train, "_cache_to_source_variant_indices", None)
        test_cache_to_source = getattr(st_test, "_cache_to_source_variant_indices", None)
        if train_cache_to_source is not None or test_cache_to_source is not None:
            if train_cache_to_source is None or test_cache_to_source is None:
                raise ValueError(f"Prediction variant-order mapping mismatch for component {idx}.")
            if not np.array_equal(
                np.asarray(train_cache_to_source, dtype=np.int64),
                np.asarray(test_cache_to_source, dtype=np.int64),
            ):
                raise ValueError(f"Prediction source-variant order mismatch for component {idx}.")
        if bool(getattr(st_train, "has_component_partition", False)):
            if not np.array_equal(
                np.asarray(st_train._component_snp_offsets, dtype=np.int32),
                np.asarray(st_test._component_snp_offsets, dtype=np.int32),
            ):
                raise ValueError("Prediction component block layout mismatch.")


def _copy_training_standardization_to_test_streamer(train_st, test_st) -> None:
    test_st._means_by_call = _ensure_on_device(train_st._means_by_call, test_st.dev)
    test_st._inv_by_call = _ensure_on_device(train_st._inv_by_call, test_st.dev)
    test_st._means_padded = _ensure_on_device(train_st._means_padded, test_st.dev)
    test_st._inv_padded = _ensure_on_device(train_st._inv_padded, test_st.dev)
    test_st._means_host = train_st._means_host
    test_st._inv_sds_host = train_st._inv_sds_host


def _build_b_by_call(streamer, b_full: jnp.ndarray) -> jnp.ndarray:
    b_np = np.asarray(jax.device_get(b_full), dtype=np.float32).reshape(-1)
    if b_np.size != int(streamer.m):
        raise ValueError(
            f"SNP-effect length mismatch: expected {int(streamer.m)}, got {b_np.size}."
        )
    out = np.zeros((int(streamer._n_calls), int(streamer._max_unpack_width)), dtype=np.float32)
    for c in range(int(streamer._n_calls)):
        s0 = int(streamer._call_snp_starts[c])
        tw = int(streamer._call_true_widths[c])
        out[c, :tw] = b_np[s0 : s0 + tw]
    return jax.device_put(jnp.asarray(out), streamer.dev)


def _solve_small_spd(
    A: jnp.ndarray,
    b: jnp.ndarray,
    *,
    base_jitter: float = 1e-6,
    max_tries: int = 8,
) -> jnp.ndarray:
    """Robust host-side SPD solve for small GLS systems."""
    A_sym = 0.5 * (A + A.T)
    A_np = np.asarray(jax.device_get(A_sym), dtype=np.float64)
    b_np = np.asarray(jax.device_get(b), dtype=np.float64).reshape(-1)
    p = int(A_np.shape[0])
    if p == 0:
        raise ValueError("solve_small_spd requires a non-empty square matrix.")

    scale = float(np.mean(np.abs(np.diag(A_np))))
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    eye = np.eye(p, dtype=np.float64)

    for k in range(max_tries):
        jitter = base_jitter * scale * (10.0 ** k)
        A_try = A_np + jitter * eye
        try:
            L = np.linalg.cholesky(A_try)
        except np.linalg.LinAlgError:
            continue
        y = np.linalg.solve(L, b_np)
        x = np.linalg.solve(L.T, y).astype(np.float32, copy=False)
        return jax.device_put(jnp.asarray(x, dtype=A.dtype), next(iter(A.devices())))

    raise FloatingPointError("Failed to stabilize GLS normal equations with jitter escalation.")


def _read_fam_iids(fam_path: str) -> list[str]:
    iids: list[str] = []
    with open(fam_path) as f:
        for line_no, line in enumerate(f, start=1):
            cols = line.split()
            if len(cols) < 2:
                raise ValueError(f"{fam_path}: line {line_no} has fewer than 2 columns.")
            iids.append(cols[1])
    return iids


def _validate_multi_grm_sample_order(prefixes: list[str]) -> None:
    if len(prefixes) <= 1:
        return
    ref_iids = _read_fam_iids(prefixes[0] + ".fam")
    for i, pref in enumerate(prefixes[1:], start=2):
        mismatch = _fam_order_mismatch(pref + ".fam", ref_iids)
        if mismatch is not None:
            raise ValueError(
                f"Multi-GRM sample-order mismatch. GRM {i} ({pref}) vs {prefixes[0]}. {mismatch}")


def _build_parallelism(n_grm: int, cpu_threads: int) -> tuple[int, int]:
    if n_grm <= 1:
        return 1, max(1, cpu_threads)
    n_parallel = min(n_grm, max(1, cpu_threads // 16), 4)
    threads_per = max(1, cpu_threads // n_parallel)
    return n_parallel, threads_per


def _available_cpu_threads(explicit: int | None = None) -> int:
    """Resolve build thread budget from explicit config or thread env vars."""
    value, _ = _resolve_cpu_threads(explicit)
    return value


class InfinitesimalREMLFitter:
    """Streaming REML fitter for single-trait infinitesimal LMMs."""

    def __init__(self, cfg: FitConfig):
        self.cfg = cfg
        t0 = time.time() if cfg.verbose else None
        if cfg.verbose:
            logger.info("build streamers start @ %s", datetime.now().isoformat(timespec='seconds'))

        self.streamers = []
        self._multi_call_plan: tuple[tuple[int, int], ...] = ()
        self._dense_call_plan: tuple[tuple[int, int], ...] = ()
        self._n_dense_streamers = 0
        self._has_sparse = False
        self._partitioned_streamer = None
        self._smile_operator = None
        self._smile_operators = ()
        cpu_threads = _available_cpu_threads(cfg.cpu_threads)
        use_component_partition = (
            cfg.vc_block_sizes is not None
            or cfg.component_variant_indices is not None
        )
        smile_w_files_present = cfg.smile_w_files is not None and len(cfg.smile_w_files) > 0
        smile_w_file_groups_present = (
            cfg.smile_w_file_groups is not None and len(cfg.smile_w_file_groups) > 0
        )
        smile_matrices_present = (
            cfg.smile_weight_matrices is not None and len(cfg.smile_weight_matrices) > 0
        )
        smile_matrix_groups_present = (
            cfg.smile_weight_matrix_groups is not None
            and len(cfg.smile_weight_matrix_groups) > 0
        )
        use_smile = bool(
            cfg.smile_identity
            or smile_w_files_present
            or smile_w_file_groups_present
            or smile_matrices_present
            or smile_matrix_groups_present
        )
        dense_standardization_overrides = (
            list(cfg.standardization_overrides)
            if cfg.standardization_overrides is not None
            else None
        )

        if cfg.precond_type != "projected_core":
            raise ValueError(
                f"Unsupported precond_type={cfg.precond_type!r}. "
                "Only 'projected_core' is available."
            )
        if cfg.vc_block_sizes is not None and cfg.component_variant_indices is not None:
            raise ValueError(
                "Use either vc_block_sizes or component_variant_indices, not both."
            )
        if smile_w_files_present and smile_matrices_present:
            raise ValueError("Use either smile_w_files or smile_weight_matrices, not both.")
        smile_modes = sum(
            bool(x)
            for x in (
                cfg.smile_identity,
                smile_w_files_present,
                smile_w_file_groups_present,
                smile_matrices_present,
                smile_matrix_groups_present,
            )
        )
        if smile_modes > 1:
            raise ValueError(
                "Use only one SMILE W mode: identity, single W list, W file groups, "
                "single matrix list, or matrix groups."
            )
        if use_smile and cfg.smile_normalization != "kernel_trace":
            raise ValueError("SMILE normalization must be 'kernel_trace'.")
        if use_smile and cfg.smile_diag_mode not in ("full", "mean"):
            raise ValueError("SMILE diag mode must be 'full' or 'mean'.")
        if use_smile and use_component_partition:
            raise ValueError("SMILE block-W mode cannot be combined with component partitioning.")
        if use_smile and (cfg.rare_sources is not None or cfg.rare_bed_prefix):
            raise ValueError("SMILE block-W mode is currently supported only for dense-only fits.")

        if use_component_partition:
            if cfg.rare_sources is not None or cfg.rare_bed_prefix:
                raise ValueError(
                    "single-source component partitioning is currently supported only for dense-only fits."
                )

        if cfg.sources is not None:
            # ---- Source-based path (PGEN / any GenoBlockSource) ----
            sources = list(cfg.sources)
            if use_component_partition and len(sources) != 1:
                raise ValueError("single-source component partitioning requires exactly one dense source.")
            if dense_standardization_overrides is not None and len(dense_standardization_overrides) != len(sources):
                raise ValueError(
                    "standardization_overrides must match the number of dense sources."
                )
            _, build_threads = _build_parallelism(len(sources), cpu_threads)

            def _build_from_source(item) -> GenoBlockStreamer:
                src, stats_override = item
                return GenoBlockStreamer(
                    source=src,
                    call_width=cfg.call_width,
                    component_block_sizes=cfg.vc_block_sizes,
                    component_variant_indices=cfg.component_variant_indices,
                    standardization_override=stats_override,
                    device=cfg.device,
                    keep_host_stats=(cfg.keep_host_stats or use_smile),
                    build_threads=build_threads,
                    sample_mask=cfg.sample_mask,
                    ring_depth=cfg.ring_depth,
                    source_build_chunk_width=cfg.source_build_chunk_width,
                )

            build_items = list(zip(sources, dense_standardization_overrides or [None] * len(sources)))
            n_parallel, _ = _build_parallelism(len(sources), cpu_threads)
            if n_parallel > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as ex:
                    self.streamers = list(ex.map(_build_from_source, build_items))
            else:
                self.streamers = [_build_from_source(item) for item in build_items]
        else:
            # ---- BED-prefix path (backward compatible) --------------------
            prefixes = cfg.bed_prefix if isinstance(cfg.bed_prefix, (list, tuple)) else [cfg.bed_prefix]
            if use_component_partition and len(prefixes) != 1:
                raise ValueError("single-source component partitioning requires exactly one dense BED prefix.")
            if dense_standardization_overrides is not None and len(dense_standardization_overrides) != len(prefixes):
                raise ValueError(
                    "standardization_overrides must match the number of dense BED prefixes."
                )
            _validate_multi_grm_sample_order(list(prefixes))
            n_parallel, build_threads = _build_parallelism(len(prefixes), cpu_threads)

            def _build_one(item) -> BedBlockStreamer:
                pref, stats_override = item
                return BedBlockStreamer(
                    bed_prefix=pref,
                    call_width=cfg.call_width,
                    component_block_sizes=cfg.vc_block_sizes,
                    component_variant_indices=cfg.component_variant_indices,
                    standardization_override=stats_override,
                    device=cfg.device,
                    keep_host_stats=(cfg.keep_host_stats or use_smile),
                    build_threads=build_threads,
                    sample_mask=cfg.sample_mask,
                    ring_depth=cfg.ring_depth,
                    source_build_chunk_width=cfg.source_build_chunk_width,
                )

            build_items = list(zip(prefixes, dense_standardization_overrides or [None] * len(prefixes)))
            if n_parallel > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as ex:
                    self.streamers = list(ex.map(_build_one, build_items))
            else:
                self.streamers = [_build_one(item) for item in build_items]

        n_dense = len(self.streamers)
        self._n_dense_streamers = n_dense
        if use_component_partition and self.streamers:
            self._partitioned_streamer = self.streamers[0]
        if cfg.rare_sources is not None:
            from .sparse_stream import SparseGenoBlockStreamer
            rare_srcs = list(cfg.rare_sources)
            if rare_srcs:
                n_parallel_r, build_threads_r = _build_parallelism(len(rare_srcs), cpu_threads)

                def _build_sparse_from_source(src):
                    return SparseGenoBlockStreamer(
                        source=src,
                        call_width=cfg.call_width,
                        device=cfg.device,
                        keep_host_stats=cfg.keep_host_stats,
                        build_threads=build_threads_r,
                        sample_mask=cfg.sample_mask,
                        ring_depth=cfg.ring_depth,
                        gpu_budget_bytes=cfg.gpu_budget_bytes,
                        mixed_dense_sparse=bool(self._n_dense_streamers),
                    )

                if n_parallel_r > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel_r) as ex:
                        self.streamers.extend(list(ex.map(_build_sparse_from_source, rare_srcs)))
                else:
                    self.streamers.extend(_build_sparse_from_source(src) for src in rare_srcs)
                self._has_sparse = True
        elif cfg.rare_bed_prefix:
            from .sparse_stream import SparseGenoBlockStreamer
            prefixes_r = (
                cfg.rare_bed_prefix
                if isinstance(cfg.rare_bed_prefix, (list, tuple))
                else [cfg.rare_bed_prefix]
            )
            prefixes_r = [p for p in prefixes_r if p]
            if prefixes_r:
                n_parallel_r, build_threads_r = _build_parallelism(len(prefixes_r), cpu_threads)

                def _build_sparse_one(pref: str):
                    return SparseGenoBlockStreamer(
                        bed_prefix=pref,
                        call_width=cfg.call_width,
                        device=cfg.device,
                        keep_host_stats=cfg.keep_host_stats,
                        build_threads=build_threads_r,
                        sample_mask=cfg.sample_mask,
                        ring_depth=cfg.ring_depth,
                        gpu_budget_bytes=cfg.gpu_budget_bytes,
                        mixed_dense_sparse=bool(self._n_dense_streamers),
                    )

                if n_parallel_r > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel_r) as ex:
                        self.streamers.extend(list(ex.map(_build_sparse_one, prefixes_r)))
                else:
                    self.streamers.extend(_build_sparse_one(pref) for pref in prefixes_r)
                self._has_sparse = True

        self._sparse_streamers = tuple(self.streamers[self._n_dense_streamers:]) if self._has_sparse else ()
        self._sparse_merged_global = None

        if use_smile:
            if self._n_dense_streamers != 1 or len(self.streamers) != 1:
                raise ValueError("SMILE block-W mode requires exactly one dense genotype source.")
            from .smile_block_w import SmileBlockWeightedOperator, SmileMultiBlockWeightedOperator

            if cfg.smile_identity:
                self._smile_operators = (
                    SmileBlockWeightedOperator.identity(
                        self.streamers[0],
                        block_size=cfg.smile_identity_block_size,
                        normalization=cfg.smile_normalization,
                        diag_mode=cfg.smile_diag_mode,
                        strict_coverage=cfg.smile_strict_coverage,
                    ),
                )
            elif cfg.smile_weight_matrix_groups is not None:
                multi_op = SmileMultiBlockWeightedOperator.from_weight_matrix_groups(
                    self.streamers[0],
                    list(cfg.smile_weight_matrix_groups),
                    normalization=cfg.smile_normalization,
                    strict_coverage=cfg.smile_strict_coverage,
                    check_psd=cfg.smile_check_psd,
                    diag_mode=cfg.smile_diag_mode,
                )
                self._smile_operators = multi_op.operators
            elif cfg.smile_w_file_groups is not None:
                multi_op = SmileMultiBlockWeightedOperator.from_weight_file_groups(
                    self.streamers[0],
                    list(cfg.smile_w_file_groups),
                    normalization=cfg.smile_normalization,
                    strict_coverage=cfg.smile_strict_coverage,
                    check_psd=cfg.smile_check_psd,
                    diag_mode=cfg.smile_diag_mode,
                )
                self._smile_operators = multi_op.operators
            elif cfg.smile_weight_matrices is not None:
                self._smile_operators = (
                    SmileBlockWeightedOperator(
                        self.streamers[0],
                        list(cfg.smile_weight_matrices),
                        normalization=cfg.smile_normalization,
                        strict_coverage=cfg.smile_strict_coverage,
                        check_psd=cfg.smile_check_psd,
                        diag_mode=cfg.smile_diag_mode,
                    ),
                )
            else:
                self._smile_operators = (
                    SmileBlockWeightedOperator.from_weight_files(
                        self.streamers[0],
                        list(cfg.smile_w_files or ()),
                        normalization=cfg.smile_normalization,
                        strict_coverage=cfg.smile_strict_coverage,
                        check_psd=cfg.smile_check_psd,
                        diag_mode=cfg.smile_diag_mode,
                    ),
                )
            self._smile_operator = self._smile_operators[0] if self._smile_operators else None

        if self._n_dense_streamers > 1:
            self._dense_call_plan = tuple(
                (g_idx, c_idx)
                for g_idx, st in enumerate(self.streamers[:self._n_dense_streamers])
                for c_idx in range(st._n_calls)
            )

        if len(self.streamers) > 1 and not self._has_sparse:
            self._multi_call_plan = tuple(
                (g_idx, c_idx)
                for g_idx, st in enumerate(self.streamers)
                for c_idx in range(st._n_calls)
            )

        if cfg.verbose and self.streamers:
            st0 = self.streamers[0]
            logger.info(
                "init streamers=%d (dense=%d sparse=%d) n=%d m=%d call_width=%d device=%s",
                len(self.streamers), n_dense, len(self.streamers) - n_dense,
                st0.n, st0.m, cfg.call_width, cfg.device)
            if self._partitioned_streamer is not None:
                part_mode = (
                    "arbitrary_variant_groups"
                    if getattr(self._partitioned_streamer, "has_arbitrary_component_partition", False)
                    else "contiguous_blocks"
                )
                logger.info(
                    "single-stream multi_grm mode=%s n_components=%d block_sizes=%s",
                    part_mode,
                    self._partitioned_streamer.n_components,
                    list(self._partitioned_streamer._component_block_sizes),
                )
            if len(self.streamers) > 1 and not self._has_sparse:
                logger.info(
                    "multi_grm call plan: total_calls=%d streamers=%d "
                    "mode=fused_cross_stream_prefetch build_parallel=%d build_threads=%d",
                    len(self._multi_call_plan), len(self.streamers), n_parallel, build_threads)
            elif len(self.streamers) > 1 and self._has_sparse:
                logger.info(
                    "multi_grm mode=hybrid_dense_fused_sparse_append streamers=%d "
                    "(dense=%d sparse=%d)",
                    len(self.streamers), n_dense, len(self.streamers) - n_dense,
                )
                if self._n_dense_streamers > 1:
                    logger.info(
                        "dense_subplan: total_calls=%d dense_streamers=%d",
                        len(self._dense_call_plan), self._n_dense_streamers,
                    )
            if t0 is not None:
                logger.info("build streamers done elapsed=%.1fs", time.time() - t0)
        # Preconditioner
        self.precond_conf = None

    def _theta_ref_from_var_components(
        self,
        *,
        n_components: int,
        var_components_init: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        if var_components_init is not None:
            theta0 = np.asarray(var_components_init, dtype=np.float32).reshape(-1)
            if theta0.size == n_components + 1:
                theta_g = theta0[:-1]
            elif theta0.size == n_components:
                theta_g = theta0
            else:
                raise ValueError(
                    f"var_components_init length mismatch: expected {n_components} or "
                    f"{n_components + 1}, got {theta0.size}."
                )
            theta_g = np.maximum(theta_g, 0.0)
        else:
            theta_g = np.ones((n_components,), dtype=np.float32)
        total = float(np.sum(theta_g))
        if total <= 0.0 or not np.isfinite(total):
            theta_g = np.full((n_components,), 1.0 / max(1, n_components), dtype=np.float32)
        else:
            theta_g = theta_g / total
        return jnp.asarray(theta_g, dtype=jnp.float32)

    def _projected_core_diag_atoms(
        self,
        diag_list: Sequence[jnp.ndarray],
    ) -> jnp.ndarray:
        atoms = []
        for d in diag_list:
            darr = jnp.asarray(d, dtype=jnp.float32)
            if darr.size == 0:
                atoms.append(jnp.asarray(0.0, dtype=jnp.float32))
            elif darr.ndim == 0:
                atoms.append(darr)
            else:
                atoms.append(jnp.mean(darr.reshape(-1)))
        if not atoms:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.stack(atoms).astype(jnp.float32)

    def _build_projected_core_precond(
        self,
        *,
        K_mvs: Sequence,
        diag_list: Sequence[jnp.ndarray],
        weighted_hv,
        stacked_kv,
        projected_core_atoms,
        var_components_init: Optional[jnp.ndarray],
    ) -> ProjectedCorePrecondConf:
        if not self.streamers:
            raise RuntimeError("Projected-core preconditioner requires at least one streamer.")
        cfg = self.cfg
        t_precond = time.time() if cfg.verbose else None
        if cfg.verbose:
            logger.info("build projected_core precond start @ %s",
                        datetime.now().isoformat(timespec='seconds'))

        G = len(K_mvs)
        theta_ref = self._theta_ref_from_var_components(
            n_components=G,
            var_components_init=var_components_init,
        )
        dev = self.streamers[0].dev
        theta_ref = _ensure_on_device(theta_ref, dev)

        if weighted_hv is not None:
            def K_ref_mv(V, theta_ref=theta_ref, weighted_hv=weighted_hv):
                return weighted_hv(theta_ref, None, V)
        else:
            def K_ref_mv(V, theta_ref=theta_ref, K_mvs=tuple(K_mvs)):
                acc = jnp.zeros_like(V)
                for g_idx, mv in enumerate(K_mvs):
                    acc = acc + theta_ref[g_idx] * mv(V)
                return acc

        key = jax.random.PRNGKey(cfg.seed + 4321)
        U, _ = build_lowrank_basis(
            K_mv=K_ref_mv,
            n=self.streamers[0].n,
            max_rank=cfg.precond_rank,
            key=key,
        )
        rank = int(U.shape[1])
        eye = jnp.eye(rank, dtype=U.dtype)
        diag_atoms = _ensure_on_device(self._projected_core_diag_atoms(diag_list), dev)
        if self._partitioned_streamer is not None:
            core_atoms = self._partitioned_streamer.build_projected_core_atoms(
                U,
                subtract_identity=True,
            )
        else:
            if projected_core_atoms is not None:
                core_atoms = projected_core_atoms(U)
            elif stacked_kv is not None:
                KU_stack = stacked_kv(U)
                core_atoms = jnp.einsum("nr,gns->grs", U, KU_stack)
                core_atoms = 0.5 * (core_atoms + jnp.swapaxes(core_atoms, -1, -2))
                core_atoms = core_atoms - diag_atoms.astype(U.dtype)[:, None, None] * eye[None, :, :]
            else:
                KU_stack = jnp.stack([mv(U) for mv in K_mvs], axis=0)
                core_atoms = jnp.einsum("nr,gns->grs", U, KU_stack)
                core_atoms = 0.5 * (core_atoms + jnp.swapaxes(core_atoms, -1, -2))
                core_atoms = core_atoms - diag_atoms.astype(U.dtype)[:, None, None] * eye[None, :, :]
        conf = ProjectedCorePrecondConf(
            U=U,
            core_atoms=core_atoms,
            total_rank=rank,
            n_grm=G,
            diag_mode="scalar_identity",
            diag_atoms=diag_atoms,
            identity=eye,
        )
        self.precond_conf = conf
        if cfg.verbose:
            logger.info(
                "projected_core precond built rank=%d n_components=%d theta_ref_mode=%s",
                rank,
                G,
                "param_init" if var_components_init is not None else "uniform",
            )
            if t_precond is not None:
                logger.info("build projected_core precond done elapsed=%.1fs",
                            time.time() - t_precond)
        return conf

    def _assemble_reml_operators(self) -> _OperatorBundle:
        if self._smile_operators:
            ops = tuple(self._smile_operators)

            K_mvs = tuple(
                (lambda V, op=op: op.kv(V))
                for op in ops
            )

            def weighted_hv(theta_g, theta_e, V, ops=ops):
                theta_g_arr = jnp.asarray(theta_g)
                if theta_g_arr.ndim != 1 or int(theta_g_arr.shape[0]) != len(ops):
                    raise ValueError(
                        f"theta_g must have length {len(ops)} for SMILE multi-GRM."
                    )
                squeeze = V.ndim == 1
                if squeeze:
                    V_work = V[:, None]
                else:
                    V_work = V
                st = ops[0].streamer
                V_dev = jax.device_put(jnp.asarray(V_work), st.dev)
                XtV = st.xtv(V_dev, normalize=False)
                if XtV.ndim == 1:
                    XtV = XtV[:, None]
                fp = V_dev.dtype
                scores = jnp.zeros((int(st.m), int(V_dev.shape[1])), dtype=fp)
                for idx, op in enumerate(ops):
                    scores = op._accumulate_weighted_scores_from_xtv(
                        scores,
                        XtV,
                        fp=fp,
                        scale=theta_g_arr[idx],
                    )
                from .smile_block_w import _zxm_impl_streamed

                out = _zxm_impl_streamed(
                    st,
                    ops[0]._scores_by_call(scores),
                    missing_val=int(st._missing_val),
                )
                if theta_e is not None:
                    out = out + jnp.asarray(theta_e, dtype=fp) * V_dev
                return out[:, 0] if squeeze else out

            def _kv_from_shared_xtv(op, XtV, rhs_cols, fp, squeeze):
                st = op.streamer
                scores = jnp.zeros((int(st.m), int(rhs_cols)), dtype=fp)
                scores = op._accumulate_weighted_scores_from_xtv(scores, XtV, fp=fp)
                from .smile_block_w import _zxm_impl_streamed

                out = _zxm_impl_streamed(
                    st,
                    op._scores_by_call(scores),
                    missing_val=int(st._missing_val),
                )
                return out[:, 0] if squeeze else out

            def stacked_kv(V, ops=ops):
                squeeze = V.ndim == 1
                V_work = V[:, None] if squeeze else V
                st = ops[0].streamer
                V_dev = jax.device_put(jnp.asarray(V_work), st.dev)
                XtV = st.xtv(V_dev, normalize=False)
                if XtV.ndim == 1:
                    XtV = XtV[:, None]
                fp = V_dev.dtype
                return jnp.stack(
                    [
                        _kv_from_shared_xtv(op, XtV, int(V_dev.shape[1]), fp, squeeze)
                        for op in ops
                    ],
                    axis=0,
                )

            def projected_core_atoms(U, ops=ops):
                atoms = []
                eye = jnp.eye(int(U.shape[1]), dtype=U.dtype)
                st = ops[0].streamer
                U_dev = jax.device_put(jnp.asarray(U), st.dev)
                XtU = st.xtv(U_dev, normalize=False)
                if XtU.ndim == 1:
                    XtU = XtU[:, None]
                fp = U_dev.dtype
                for op in ops:
                    KU = _kv_from_shared_xtv(op, XtU, int(U_dev.shape[1]), fp, False)
                    atom = jnp.asarray(U_dev, dtype=KU.dtype).T @ KU
                    atom = 0.5 * (atom + atom.T)
                    diag_atom = jnp.mean(jnp.asarray(op.diag(), dtype=U.dtype).reshape(-1))
                    atoms.append(atom - diag_atom.astype(U.dtype) * eye)
                return jnp.stack(atoms, axis=0)

            return _OperatorBundle(
                K_mvs=K_mvs,
                diag_list=tuple(op.diag() for op in ops),
                weighted_hv=weighted_hv,
                stacked_kv=stacked_kv,
                projected_core_atoms=projected_core_atoms,
            )

        if self._partitioned_streamer is not None:
            st = self._partitioned_streamer
            K_mvs = [
                (lambda V, component_idx=component_idx, st=st: st.component_kv(
                    V, component_idx, normalize=True
                ))
                for component_idx in range(st.n_components)
            ]
            diag_list = st.component_diag_list()

            def weighted_hv(theta_g, theta_e, V, st=st):
                return st.weighted_component_hv(theta_g, theta_e, V)

            def stacked_kv(V, st=st):
                return st.stacked_component_kv(V, normalize=True)

            return _OperatorBundle(
                K_mvs=tuple(K_mvs),
                diag_list=tuple(diag_list),
                weighted_hv=weighted_hv,
                stacked_kv=stacked_kv,
                projected_core_atoms=(
                    lambda U, st=st: st.build_projected_core_atoms(U, subtract_identity=True)
                ),
            )

        K_mvs = [lambda V, st=st: st.kv(V) for st in self.streamers]
        diag_list = [st.diag() for st in self.streamers]
        weighted_hv = None
        stacked_kv = None
        projected_core_atoms = None

        if len(self.streamers) > 1:
            streamers = tuple(self.streamers)
            if self._has_sparse:
                from .kv_impl import (
                    _stack_leading_axis,
                    build_projected_core_atoms_multi_streamed,
                    kv_impl_multi_streamed_stacked,
                    kv_impl_multi_streamed_weighted,
                )
                dense_streamers = tuple(streamers[:self._n_dense_streamers])
                sparse_streamers = self._sparse_streamers
                dense_call_plan = self._dense_call_plan

                def _sparse_sum_v(
                    V_mat,
                    sparse_streamers=sparse_streamers,
                ) -> jnp.ndarray | None:
                    if not sparse_streamers:
                        return None
                    return jnp.sum(V_mat, axis=0)

                def weighted_hv(
                    theta_g, theta_e, V,
                    streamers=streamers,
                    dense_streamers=dense_streamers,
                    sparse_streamers=sparse_streamers,
                    dense_call_plan=dense_call_plan,
                ):
                    V_dev = _ensure_on_device(V, streamers[0].dev)
                    squeeze = V_dev.ndim == 1
                    if squeeze:
                        V_dev = V_dev[:, None]
                    fp = V_dev.dtype
                    acc = (
                        theta_e * V_dev
                        if theta_e is not None
                        else jnp.zeros_like(V_dev, dtype=fp)
                    )

                    dense_count = len(dense_streamers)
                    if dense_count > 1:
                        for st in dense_streamers:
                            st._prepare_kv_pass()
                        acc = kv_impl_multi_streamed_weighted(
                            V_dev,
                            dense_streamers,
                            dense_call_plan,
                            theta_g[:dense_count],
                            theta_e=None,
                            missing_val=int(dense_streamers[0]._missing_val),
                        ) + acc
                    elif dense_count == 1:
                        acc = acc + theta_g[0] * dense_streamers[0].kv(
                            V_dev, normalize=True
                        )

                    if sparse_streamers:
                        sum_v = _sparse_sum_v(V_dev)
                        sparse_theta = theta_g[
                            self._n_dense_streamers : self._n_dense_streamers + len(sparse_streamers)
                        ]
                        for comp_idx, st in enumerate(sparse_streamers):
                            acc = acc + sparse_theta[comp_idx] * st.kv(
                                V_dev,
                                normalize=True,
                                sum_v=sum_v,
                            )
                    return acc[:, 0] if squeeze else acc

                def stacked_kv(
                    V,
                    streamers=streamers,
                    dense_streamers=dense_streamers,
                    sparse_streamers=sparse_streamers,
                    dense_call_plan=dense_call_plan,
                ):
                    V_dev = _ensure_on_device(V, streamers[0].dev)
                    squeeze = V_dev.ndim == 1
                    if squeeze:
                        V_dev = V_dev[:, None]
                    component_outs = []

                    dense_count = len(dense_streamers)
                    if dense_count > 1:
                        for st in dense_streamers:
                            st._prepare_kv_pass()
                        dense_stack = kv_impl_multi_streamed_stacked(
                            V_dev,
                            dense_streamers,
                            dense_call_plan,
                            missing_val=int(dense_streamers[0]._missing_val),
                            normalize=True,
                        )
                        component_outs.extend(
                            dense_stack[g_idx] for g_idx in range(dense_count)
                        )
                    elif dense_count == 1:
                        component_outs.append(dense_streamers[0].kv(V_dev, normalize=True))

                    if sparse_streamers:
                        sum_v = _sparse_sum_v(V_dev)
                        for st in sparse_streamers:
                            component_outs.append(
                                st.kv(V_dev, normalize=True, sum_v=sum_v)
                            )

                    out = _stack_leading_axis(component_outs)
                    return out[:, :, 0] if squeeze else out

                def projected_core_atoms(
                    U,
                    streamers=streamers,
                    dense_streamers=dense_streamers,
                    sparse_streamers=sparse_streamers,
                    dense_call_plan=dense_call_plan,
                ):
                    U_dev = _ensure_on_device(U, streamers[0].dev)
                    component_atoms = []

                    dense_count = len(dense_streamers)
                    if dense_count > 1:
                        for st in dense_streamers:
                            st._prepare_kv_pass()
                        dense_atoms = build_projected_core_atoms_multi_streamed(
                            U_dev,
                            dense_streamers,
                            dense_call_plan,
                            missing_val=int(dense_streamers[0]._missing_val),
                            subtract_identity=True,
                        )
                        component_atoms.extend(
                            dense_atoms[g_idx] for g_idx in range(dense_count)
                        )
                    elif dense_count == 1:
                        component_atoms.append(
                            dense_streamers[0].build_projected_core_atom(
                                U_dev,
                                subtract_identity=True,
                            )
                        )

                    for st in sparse_streamers:
                        component_atoms.append(
                            st.build_projected_core_atom(
                                U_dev,
                                subtract_identity=True,
                            )
                        )
                    return _stack_leading_axis(component_atoms)
            else:
                from .kv_impl import (
                    build_projected_core_atoms_multi_streamed,
                    kv_impl_multi_streamed_stacked,
                    kv_impl_multi_streamed_weighted,
                )
                call_plan = self._multi_call_plan
                missing_val = int(streamers[0]._missing_val)

                def weighted_hv(theta_g, theta_e, V, streamers=streamers, call_plan=call_plan):
                    for st in streamers:
                        st._prepare_kv_pass()
                    V_dev = _ensure_on_device(V, streamers[0].dev)
                    return kv_impl_multi_streamed_weighted(
                        V_dev, streamers, call_plan, theta_g,
                        theta_e=theta_e, missing_val=missing_val)

                def stacked_kv(V, streamers=streamers, call_plan=call_plan):
                    for st in streamers:
                        st._prepare_kv_pass()
                    V_dev = _ensure_on_device(V, streamers[0].dev)
                    return kv_impl_multi_streamed_stacked(
                        V_dev, streamers, call_plan,
                        missing_val=missing_val, normalize=True)

                def projected_core_atoms(U, streamers=streamers, call_plan=call_plan):
                    for st in streamers:
                        st._prepare_kv_pass()
                    U_dev = _ensure_on_device(U, streamers[0].dev)
                    return build_projected_core_atoms_multi_streamed(
                        U_dev,
                        streamers,
                        call_plan,
                        missing_val=missing_val,
                        subtract_identity=True,
                    )

        return _OperatorBundle(
            K_mvs=tuple(K_mvs),
            diag_list=tuple(diag_list),
            weighted_hv=weighted_hv,
            stacked_kv=stacked_kv,
            projected_core_atoms=projected_core_atoms,
        )

    def _ensure_projected_core_precond_ready(
        self,
        ops: _OperatorBundle,
        *,
        var_components_init: Optional[jnp.ndarray],
    ) -> None:
        if self.cfg.precond_rank > 0 and self.precond_conf is None:
            self._build_projected_core_precond(
                K_mvs=ops.K_mvs,
                diag_list=ops.diag_list,
                weighted_hv=ops.weighted_hv,
                stacked_kv=ops.stacked_kv,
                projected_core_atoms=ops.projected_core_atoms,
                var_components_init=var_components_init,
            )

    def _make_precond_refresh_fn(
        self,
        ops: _OperatorBundle,
    ):
        if self.cfg.precond_rank <= 0:
            return None
        if self.cfg.precond_refresh_reldp <= 0.0:
            return None
        if self._smile_operators:
            return None
        if len(ops.K_mvs) <= 1:
            return None

        def refresh_fn(param: jnp.ndarray):
            return self._build_projected_core_precond(
                K_mvs=ops.K_mvs,
                diag_list=ops.diag_list,
                weighted_hv=ops.weighted_hv,
                stacked_kv=ops.stacked_kv,
                projected_core_atoms=ops.projected_core_atoms,
                var_components_init=jnp.asarray(param, dtype=jnp.float32),
            )

        return refresh_fn

    def _make_hv(self, ops: _OperatorBundle, theta_g: jnp.ndarray, theta_e: jnp.ndarray):
        if ops.weighted_hv is not None:
            return lambda V: ops.weighted_hv(theta_g, theta_e, V)

        def Hv(V):
            acc = theta_e * V
            for g_idx, mv in enumerate(ops.K_mvs):
                acc = acc + theta_g[g_idx] * mv(V)
            return acc

        return Hv

    def _make_effect_precond(
        self,
        ops: _OperatorBundle,
        theta_g: jnp.ndarray,
        theta_e: jnp.ndarray,
    ):
        if self.precond_conf is None:
            return None
        diag_H_scalar = scalar_diag_from_precond_conf(self.precond_conf, theta_g, theta_e)
        if diag_H_scalar is not None:
            diag_H = diag_H_scalar
        else:
            diag_stack = jnp.stack(ops.diag_list, axis=0)
            diag_H = theta_e + jnp.tensordot(theta_g, diag_stack, axes=1)
        return make_precond(self.precond_conf, theta_g, diag_H, eps=self.cfg.pcg_ridge)

    def _estimate_snp_effects(
        self,
        alpha: jnp.ndarray,
        theta_g: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        if self._smile_operators:
            theta_g_arr = jnp.asarray(theta_g)
            if theta_g_arr.ndim != 1 or int(theta_g_arr.shape[0]) != len(self._smile_operators):
                raise ValueError(
                    f"theta_g must have length {len(self._smile_operators)} for SMILE effects."
                )
            return tuple(
                op.snp_effects(alpha, theta_g_arr[idx])
                for idx, op in enumerate(self._smile_operators)
            )

        if self._partitioned_streamer is not None:
            st = self._partitioned_streamer
            xt_alpha = st.xtv(alpha, normalize=False)
            snp_effects = []
            for comp_idx in range(st.n_components):
                start = int(st._component_snp_offsets[comp_idx])
                stop = int(st._component_snp_offsets[comp_idx + 1])
                eff = float(st._component_eff_m_host[comp_idx])
                if eff > 0.0:
                    scale = theta_g[comp_idx] / jnp.asarray(eff, dtype=xt_alpha.dtype)
                    snp_effects.append(scale * xt_alpha[start:stop])
                else:
                    snp_effects.append(jnp.zeros((stop - start,), dtype=xt_alpha.dtype))
            return tuple(snp_effects)

        if len(self.streamers) > 1:
            streamers = tuple(self.streamers)
            if self._has_sparse:
                xt_alpha_parts: list[jnp.ndarray] = []
                dense_streamers = tuple(streamers[:self._n_dense_streamers])
                if dense_streamers:
                    from .kv_impl import xtv_impl_multi_streamed

                    for st in dense_streamers:
                        st._prepare_kv_pass()
                    if len(dense_streamers) > 1:
                        xt_alpha_parts.extend(
                            xtv_impl_multi_streamed(
                                alpha,
                                dense_streamers,
                                self._dense_call_plan,
                                missing_val=int(dense_streamers[0]._missing_val),
                                normalize=True,
                            )
                        )
                    else:
                        xt_alpha_parts.append(dense_streamers[0].xtv(alpha, normalize=True))
                xt_alpha_parts.extend(
                    st.xtv(alpha, normalize=True)
                    for st in self._sparse_streamers
                )
                return tuple(
                    theta_g[g_idx] * xt_alpha_parts[g_idx]
                    for g_idx in range(len(xt_alpha_parts))
                )

            from .kv_impl import xtv_impl_multi_streamed

            for st in streamers:
                st._prepare_kv_pass()
            xt_alpha_parts = xtv_impl_multi_streamed(
                alpha,
                streamers,
                self._multi_call_plan,
                missing_val=int(streamers[0]._missing_val),
                normalize=True,
            )
            return tuple(
                theta_g[g_idx] * xt_alpha_parts[g_idx]
                for g_idx in range(len(xt_alpha_parts))
            )

        return (theta_g[0] * self.streamers[0].xtv(alpha, normalize=True),)

    def estimate_effects(
        self,
        y: jnp.ndarray,
        *,
        var_components: jnp.ndarray,
        covar: Optional[jnp.ndarray] = None,
    ) -> EffectEstimates:
        if not self.streamers:
            raise RuntimeError("estimate_effects requires initialized streamers.")

        ops = self._assemble_reml_operators()
        theta = jnp.asarray(var_components, dtype=jnp.float32).reshape(-1)
        G = len(ops.K_mvs)
        if theta.shape[0] != G + 1:
            raise ValueError(
                f"var_components length mismatch: expected {G + 1}, got {theta.shape[0]}."
            )
        theta_g = theta[:-1]
        theta_e = theta[-1]

        self._ensure_projected_core_precond_ready(ops, var_components_init=theta)

        y_std, y_mean, y_scale = standardize_response(y)
        xmat = None
        if covar is not None:
            xmat = jnp.asarray(covar, dtype=jnp.float32)
            if xmat.ndim == 1:
                xmat = xmat[:, None]
            if xmat.size == 0:
                xmat = None

        Hv = self._make_hv(ops, theta_g, theta_e)
        M_cur = self._make_effect_precond(ops, theta_g, theta_e)

        rhs_parts = [y_std[:, None]]
        if xmat is not None:
            rhs_parts.append(xmat)
        rhs = jnp.concatenate(rhs_parts, axis=1)
        X0 = M_cur(rhs) if M_cur is not None else jnp.zeros_like(rhs)

        sol, rel_res, iters = pcg_solve(
            Hv,
            rhs,
            M=M_cur,
            tol=self.cfg.effect_pcg_tol,
            maxiter=self.cfg.max_pcg_iters,
            X0=X0,
        )
        if not bool(jnp.all(jnp.isfinite(sol))):
            raise FloatingPointError("Non-finite PCG solution encountered in estimate_effects.")

        Hinv_y = sol[:, 0]
        if xmat is not None:
            Hinv_x = sol[:, 1:]
            XtHinvX = xmat.T @ Hinv_x
            fixed_effects = _solve_small_spd(
                XtHinvX,
                xmat.T @ Hinv_y,
            )
            alpha = Hinv_y - Hinv_x @ fixed_effects
        else:
            fixed_effects = jnp.empty((0,), dtype=sol.dtype)
            alpha = Hinv_y

        if ops.stacked_kv is not None:
            Kalpha_stack = ops.stacked_kv(alpha)
        else:
            Kalpha_stack = jnp.stack([mv(alpha) for mv in ops.K_mvs], axis=0)

        random_effect_components = tuple(
            theta_g[g_idx] * Kalpha_stack[g_idx]
            for g_idx in range(G)
        )
        random_effect = jnp.sum(theta_g[:, None] * Kalpha_stack, axis=0)
        snp_effects = self._estimate_snp_effects(alpha, theta_g)

        return EffectEstimates(
            fixed_effects=fixed_effects,
            random_effect=random_effect,
            random_effect_components=random_effect_components,
            snp_effects=snp_effects,
            pcg_rel_res=float(jax.device_get(rel_res)),
            pcg_iters=int(iters),
            y_mean=float(jax.device_get(y_mean)),
            y_scale=float(jax.device_get(y_scale)),
        )

    def predict(
        self,
        effects: EffectEstimates,
        *,
        test_fitter: "InfinitesimalREMLFitter",
        test_covar: Optional[jnp.ndarray] = None,
    ) -> PredictionEstimates:
        if self._has_sparse or test_fitter._has_sparse:
            raise NotImplementedError(
                "Prediction currently supports dense genotype paths only; sparse test prediction is not implemented."
            )
        if not self.streamers or not test_fitter.streamers:
            raise RuntimeError("Prediction requires initialized training and test streamers.")

        _validate_dense_prediction_streamers(self.streamers, test_fitter.streamers)
        for st_train, st_test in zip(self.streamers, test_fitter.streamers):
            _copy_training_standardization_to_test_streamer(st_train, st_test)

        beta = jnp.asarray(effects.fixed_effects, dtype=jnp.float32).reshape(-1)
        if test_covar is not None:
            X_test = jnp.asarray(test_covar, dtype=jnp.float32)
            if X_test.ndim == 1:
                X_test = X_test[:, None]
        else:
            X_test = None

        if beta.size > 0:
            if X_test is None:
                raise ValueError("Prediction requires test covariates aligned to the training design matrix.")
            if int(X_test.shape[1]) != int(beta.size):
                raise ValueError(
                    f"Prediction covariate width mismatch: expected {int(beta.size)}, got {int(X_test.shape[1])}."
                )
            fixed_effect = X_test @ beta
        else:
            if not test_fitter.streamers:
                raise RuntimeError("Prediction requires at least one test streamer.")
            fixed_effect = jnp.zeros((int(test_fitter.streamers[0].n),), dtype=jnp.float32)

        from .kv_impl import zxb_impl_streamed

        if self._partitioned_streamer is not None:
            train_st = self._partitioned_streamer
            test_st = test_fitter._partitioned_streamer
            if test_st is None:
                raise ValueError("Prediction requires a partitioned test streamer to match training.")
            test_st._prepare_kv_pass()
            b_concat = jnp.concatenate(
                [jnp.asarray(b, dtype=jnp.float32).reshape(-1) for b in effects.snp_effects],
                axis=0,
            )
            b_by_call = _build_b_by_call(train_st, b_concat)
            random_effect, random_components = zxb_impl_streamed(
                b_by_call,
                test_st._true_widths_dev,
                test_st._means_by_call,
                test_st._inv_by_call,
                n=int(test_st.n),
                n_calls=int(test_st._n_calls),
                pop_block=test_st._pop_cached,
                missing_val=int(test_st._missing_val),
                component_ids=np.asarray(test_st._call_component_ids, dtype=np.int32),
                n_components=int(test_st.n_components),
            )
        else:
            if len(self.streamers) > 1:
                from .kv_impl import zxb_impl_multi_streamed

                train_streamers = tuple(self.streamers)
                test_streamers = tuple(test_fitter.streamers)
                for st in test_streamers:
                    st._prepare_kv_pass()
                b_by_call_list = tuple(
                    _build_b_by_call(st_train, jnp.asarray(b, dtype=jnp.float32).reshape(-1))
                    for st_train, b in zip(train_streamers, effects.snp_effects)
                )
                call_plan = tuple(
                    (g_idx, c_idx)
                    for g_idx, st in enumerate(test_streamers)
                    for c_idx in range(int(st._n_calls))
                )
                random_effect, random_components = zxb_impl_multi_streamed(
                    b_by_call_list,
                    test_streamers,
                    call_plan,
                    missing_val=int(test_streamers[0]._missing_val),
                )
            else:
                st_train = self.streamers[0]
                st_test = test_fitter.streamers[0]
                st_test._prepare_kv_pass()
                b_by_call = _build_b_by_call(
                    st_train,
                    jnp.asarray(effects.snp_effects[0], dtype=jnp.float32).reshape(-1),
                )
                comp_pred, _ = zxb_impl_streamed(
                    b_by_call,
                    st_test._true_widths_dev,
                    st_test._means_by_call,
                    st_test._inv_by_call,
                    n=int(st_test.n),
                    n_calls=int(st_test._n_calls),
                    pop_block=st_test._pop_cached,
                    missing_val=int(st_test._missing_val),
                )
                random_components = (comp_pred,)
                random_effect = comp_pred

        y_pred_std = fixed_effect + random_effect
        y_pred = (
            jnp.asarray(effects.y_mean, dtype=y_pred_std.dtype)
            + jnp.asarray(effects.y_scale, dtype=y_pred_std.dtype) * y_pred_std
        )
        return PredictionEstimates(
            fixed_effect=fixed_effect,
            random_effect=random_effect,
            random_effect_components=tuple(random_components),
            y_pred_std=y_pred_std,
            y_pred=y_pred,
        )

    def close(self) -> None:
        for st in getattr(self, "streamers", ()):
            close_streamer = getattr(st, "close", None)
            if not callable(close_streamer):
                continue
            try:
                close_streamer()
            except (OSError, RuntimeError, ValueError, BufferError):
                logger.debug("Failed to close fitter streamer.", exc_info=True)
        self.streamers = []
        self._has_sparse = False
        self._sparse_streamers = ()
        self._partitioned_streamer = None
        self._smile_operator = None
        self._smile_operators = ()

    def fit_infinitesimal(
        self,
        y: jnp.ndarray,
        covar: Optional[jnp.ndarray] = None,
        h2_init: float = 0.5,
        var_components_init: Optional[jnp.ndarray] = None,
        estimate_effects: bool = False,
    ) -> FitResult:
        if self.cfg.verbose:
            logger.info("fit_infinitesimal start @ %s", datetime.now().isoformat(timespec='seconds'))
            t_fit_start = time.time()

        ops = self._assemble_reml_operators()
        self._ensure_projected_core_precond_ready(ops, var_components_init=var_components_init)

        reps = []
        n_reps = max(1, int(self.cfg.n_reml_reps))
        for r in range(n_reps):
            vc, history = fit_reml(
                y=jnp.asarray(y, dtype=jnp.float32),
                K_mvs=ops.K_mvs,
                diag_list=ops.diag_list,
                covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
                n_rand_vec=self.cfg.n_rand_vec,
                maxiter=self.cfg.max_pcg_iters,
                seed=self.cfg.seed + r * 9973,
                h2_init=h2_init,
                param_init=(
                    jnp.asarray(var_components_init, dtype=jnp.float32)
                    if var_components_init is not None else None),
                minq_iter=self.cfg.minq_iter,
                slq_samples=self.cfg.slq_samples,
                slq_m=self.cfg.slq_m,
                slq_mode=self.cfg.slq_mode,
                precond_conf=self.precond_conf,
                precond_refresh_fn=self._make_precond_refresh_fn(ops),
                precond_refresh_reldp=self.cfg.precond_refresh_reldp,
                precond_eps=self.cfg.pcg_ridge,
                weighted_hv=ops.weighted_hv,
                stacked_kv=ops.stacked_kv,
                verbose=self.cfg.verbose,
            )
            reps.append((vc, history))

        vc_mean = reps[0][0]
        history = reps[0][1]
        rep_var_components = None
        jackknife_se_var = None
        jackknife_se_h2 = None

        if n_reps > 1:
            vc_stack = jnp.stack([x[0] for x in reps], axis=0)
            vc_mean = jnp.mean(vc_stack, axis=0)
            vc_center = vc_stack - vc_mean
            jackknife_se_var = jnp.sqrt(
                jnp.maximum(0.0, jnp.sum(vc_center**2, axis=0) * (n_reps - 1) / n_reps)
            )
            rep_var_components = vc_stack
            h2_vals = 1.0 - vc_stack[:, -1] / jnp.maximum(jnp.sum(vc_stack, axis=1), 1e-8)
            h2_center = h2_vals - jnp.mean(h2_vals)
            jackknife_se_h2 = float(
                jnp.sqrt(jnp.maximum(0.0, jnp.sum(h2_center**2) * (n_reps - 1) / n_reps))
            )

        effects = None
        if estimate_effects:
            effects = self.estimate_effects(
                y=jnp.asarray(y, dtype=jnp.float32),
                var_components=vc_mean,
                covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            )

        result = FitResult(
            var_components=vc_mean, history=history,
            rep_var_components=rep_var_components,
            jackknife_se_var=jackknife_se_var,
            jackknife_se_h2=jackknife_se_h2,
            effects=effects,
        )
        if self.cfg.verbose:
            logger.info("fit_infinitesimal done @ %s elapsed=%.1fs",
                        datetime.now().isoformat(timespec='seconds'), time.time() - t_fit_start)
        return result
