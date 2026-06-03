#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import importlib
import logging
import os
import re
import sys
import time
from datetime import datetime

repo_root = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(repo_root)
if parent not in sys.path:
    sys.path.insert(0, parent)
pkg_name = os.path.basename(repo_root)
_runtime_mod = importlib.import_module(f"{pkg_name}.runtime_env")
_runtime_mod.configure_runtime_env()

import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

jax.config.update("jax_default_matmul_precision", "tensorfloat32")

_inf_mod = importlib.import_module(f"{pkg_name}.reml_model")
_data_mod = importlib.import_module(f"{pkg_name}.data_utils")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_component_spec_mod = importlib.import_module(f"{pkg_name}.component_spec")
_effect_io_mod = importlib.import_module(f"{pkg_name}.effect_io")
_pred_io_mod = importlib.import_module(f"{pkg_name}.prediction_io")
_io_utils_mod = importlib.import_module(f"{pkg_name}.io_utils")
_smile_mod = importlib.import_module(f"{pkg_name}.smile_block_w")
_smile_planner_mod = importlib.import_module(f"{pkg_name}.smile_planner")
InfinitesimalREMLFitter = _inf_mod.InfinitesimalREMLFitter
FitConfig = _inf_mod.FitConfig
load_component_specs = _component_spec_mod.load_component_specs
load_pheno_covar_aligned_with_transform = _data_mod.load_pheno_covar_aligned_with_transform
load_covar_aligned = _data_mod.load_covar_aligned
write_effect_outputs = _effect_io_mod.write_effect_outputs
write_prediction_outputs = _pred_io_mod.write_prediction_outputs
ensure_parent_dir = _io_utils_mod.ensure_parent_dir
load_weight_matrix_shape = _smile_mod.load_weight_matrix_shape
run_smile_planner = _smile_planner_mod.run_smile_planner
_source_mod = importlib.import_module(f"{pkg_name}.geno_source")
PgenGenoSource = _source_mod.PgenGenoSource
env = _common_mod.env
read_keep_ids = _common_mod.read_keep_ids
setup_gpu = _common_mod.setup_gpu
run_planner = _common_mod.run_planner
print_planner_info = _common_mod.print_planner_info
cleanup_path = _common_mod.cleanup_path
make_nonbed_input_fam = _common_mod.make_nonbed_input_fam
compute_sample_mask = _common_mod.compute_sample_mask
resolve_cpu_threads = _common_mod.resolve_cpu_threads

from bed_reader import open_bed


def _bed_count(path: str, attr: str) -> int:
    bed = open_bed(path)
    try:
        return int(getattr(bed, attr))
    finally:
        close = getattr(bed, "close", None)
        if close is not None:
            close()


def _load_component_variant_indices(npz_path: str) -> list[np.ndarray]:
    return [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in load_component_specs(npz_path)
    ]


def _parse_grm_groups(raw: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for group_raw in raw.split(";"):
        paths = [x.strip() for x in group_raw.split(",") if x.strip()]
        if paths:
            groups.append(paths)
    return groups


def _read_w_files_list(path: str) -> list[str]:
    path = path.strip()
    if not path:
        return []
    try:
        text = open(path, "r", encoding="utf-8").read()
    except OSError as exc:
        raise SystemExit(f"Failed to read --w-files-list {path!r}: {exc}") from exc
    paths: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.extend(x.strip() for x in line.split(",") if x.strip())
    return paths


def _matrix_size_from_shape(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or int(shape[0]) != int(shape[1]):
        raise SystemExit(f"SMILE W matrix must be square, got shape {shape}.")
    return int(shape[0])


def parse_args():
    p = argparse.ArgumentParser(description="Run GPU-accelerated REML pipeline.")
    # Genotype input — exactly one of the two groups must be supplied
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""),
                   help="PLINK1 BED file prefix (no extension); comma-separated for multiple GRMs")
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""),
                   help="PLINK2 PGEN file prefix (direct read, no conversion needed)")
    p.add_argument("--rare-bed-prefix", default=env("RARE_BED_PREFIX", ""),
                   help="Rare-variant BED prefix; comma-separated for multiple sparse GRMs")
    p.add_argument("--rare-pgen-prefix", default=env("RARE_PGEN_PREFIX", ""),
                   help="Rare-variant PGEN prefix for sparse GRM input")
    p.add_argument("--vc-block-sizes", default=env("VC_BLOCK_SIZES", ""),
                   help="Comma-separated contiguous SNP block sizes for single-file multi-GRM.")
    p.add_argument(
        "--component-indices-npz",
        default=env("COMPONENT_INDICES_NPZ", ""),
        help="Legacy NPZ file of per-component SNP index arrays for arbitrary single-file multi-GRM.",
    )
    p.add_argument(
        "--component-spec",
        default=env("COMPONENT_SPEC", ""),
        help="Structured component spec (.json or .npz) with optional names/metadata for arbitrary single-file multi-GRM.",
    )
    p.add_argument(
        "--smile",
        action="store_true",
        default=env("SMILE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Enable SMILE-method weighted GRM mode.",
    )
    p.add_argument(
        "--w-files",
        dest="w_files",
        default=env("SMILE_W_FILES", ""),
        help=(
            "Comma-separated W_i files for one SMILE GRM, in SNP-block order."
        ),
    )
    p.add_argument(
        "--w-files-list",
        dest="w_files_list",
        default=env("SMILE_W_FILES_LIST", ""),
        help=(
            "Text file containing W_i paths for one SMILE GRM. "
            "Entries may be comma-separated or one path per line."
        ),
    )
    p.add_argument(
        "--grm-groups",
        dest="grm_groups",
        default=env("SMILE_GRM_GROUPS", ""),
        help=(
            "Semicolon-separated GRM groups, with comma-separated W_i files inside each GRM. "
            "Example: 'W1.rds,W2.rds;W3.rds' makes two SMILE GRMs where W1/W2 are summed inside GRM1."
        ),
    )
    p.add_argument(
        "--identity-w",
        dest="identity_w",
        action="store_true",
        default=env("SMILE_IDENTITY_W", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Use W=I in SMILE mode without materializing the identity matrix.",
    )
    p.add_argument(
        "--w-normalization",
        dest="w_normalization",
        choices=["kernel_trace", "effective_rank"],
        default=env("SMILE_W_NORMALIZATION", "kernel_trace"),
        help=(
            "SMILE W-kernel normalization. kernel_trace computes tr(Z W Z^T)/n; "
            "effective_rank uses per-W sidecar JSON effective_rank values."
        ),
    )
    p.add_argument(
        "--no-w-psd-check",
        dest="no_w_psd_check",
        action="store_true",
        default=env("SMILE_NO_W_PSD_CHECK", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Disable eigenvalue PSD checks for W_i matrices.",
    )
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument("--prediction-bed-prefix", default=env("PREDICTION_BED_PREFIX", ""),
                   help="Prediction BED prefix (comma-separated for multi-GRM).")
    p.add_argument("--prediction-pgen-prefix", default=env("PREDICTION_PGEN_PREFIX", ""),
                   help="Prediction PGEN prefix.")
    p.add_argument("--prediction-covar-txt", default=env("PREDICTION_COVAR_TXT", ""),
                   help="Prediction covariate file aligned to the prediction genotype FAM order.")
    p.add_argument("--keep-path", default=env("KEEP_PATH", ""))
    p.add_argument("--keep-out", default=env("KEEP_OUT", ""))
    p.add_argument("--dropped-out", default=env("DROPPED_OUT", ""))
    p.add_argument("--device", default=env("DEVICE", "gpu"))
    p.add_argument("--call-width", type=int, default=0,
                   help="Call width w (0 = auto from planner)")
    p.add_argument(
        "--gpu-budget-gib",
        "--gpu-budget-gb",
        dest="gpu_budget_gib",
        type=float,
        default=float(env("GPU_BUDGET_GIB", env("GPU_BUDGET_GB", "0"))),
        help="GPU budget in GiB (`--gpu-budget-gb` is a legacy alias; 0 = use 85%% of current free)",
    )
    p.add_argument("--ring-depth", type=int, default=int(env("RING_DEPTH", "0")),
                   help="Pinned ring buffer depth (0 = auto, default 32)")
    p.add_argument("--cpu-threads", type=int, default=int(env("CPU_THREADS", "0")))
    p.add_argument("--n-rand-vec", type=int, default=100)
    p.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    p.add_argument("--slq-samples", type=int, default=int(env("SLQ_SAMPLES", "4")))
    p.add_argument("--slq-m", type=int, default=int(env("SLQ_M", "8")))
    p.add_argument(
        "--slq-mode",
        choices=["raw", "projected_core_residual"],
        default=env("SLQ_MODE", "projected_core_residual"),
        help="SLQ mode: raw Lanczos on H, or projected-core residual SLQ when a projected-core preconditioner is available.",
    )
    p.add_argument(
        "--precond-refresh-reldp",
        type=float,
        default=float(env("PRECOND_REFRESH_RELDP", "0.20")),
        help="Rebuild projected-core U/core after an accepted REML step when max relative parameter change exceeds this threshold (<=0 disables).",
    )
    p.add_argument("--precond-type", choices=["projected_core"], default="projected_core")
    p.add_argument("--minq-iter", type=int, default=int(env("MINQ_ITER", "10")))
    p.add_argument("--compute-effects", action="store_true",
                   default=env("COMPUTE_EFFECTS", "").strip().lower() in {"1", "true", "yes", "on"},
                   help="After variance-component estimation, compute and write fixed/random/SNP effects.")
    p.add_argument("--out-prefix", default=env("OUT_PREFIX", ""),
                   help="Output prefix for effect / prediction files.")
    p.add_argument(
        "--verbose",
        action="store_true",
        default=env("VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("start @ %s", datetime.now().isoformat(timespec='seconds'))
    t0 = time.time()

    bed_list = [b for b in args.bed_prefix.split(",") if b]
    pgen_prefix = args.pgen_prefix.strip()
    rare_bed_list = [b for b in args.rare_bed_prefix.split(",") if b]
    rare_pgen_prefix = args.rare_pgen_prefix.strip()
    vc_block_sizes = [int(x) for x in args.vc_block_sizes.split(",") if x.strip()]
    smile_w_files = [x.strip() for x in args.w_files.split(",") if x.strip()]
    smile_w_files_from_list = _read_w_files_list(args.w_files_list)
    if smile_w_files and smile_w_files_from_list:
        raise SystemExit("Use only one of --w-files or --w-files-list.")
    if smile_w_files_from_list:
        smile_w_files = smile_w_files_from_list
    smile_w_file_groups = _parse_grm_groups(args.grm_groups)
    smile_weight_matrices = None
    smile_weight_matrix_groups = None
    smile_w_block_sizes: list[int] = []
    if smile_w_files:
        smile_w_block_sizes = [
            _matrix_size_from_shape(load_weight_matrix_shape(path))
            for path in smile_w_files
        ]
    elif smile_w_file_groups:
        smile_w_block_sizes = [
            _matrix_size_from_shape(load_weight_matrix_shape(path))
            for group in smile_w_file_groups
            for path in group
        ]
    smile_inputs_present = bool(args.identity_w or smile_w_files or smile_w_file_groups)
    use_smile = bool(args.smile or smile_inputs_present)
    smile_n_grm = (
        len(smile_w_file_groups)
        if smile_w_file_groups
        else 1
        if (args.identity_w or smile_w_files)
        else 0
    )
    component_spec_path = args.component_spec.strip()
    legacy_component_npz = args.component_indices_npz.strip()
    if component_spec_path and legacy_component_npz:
        raise SystemExit("Use only one of --component-spec or --component-indices-npz.")
    component_spec_source = component_spec_path or legacy_component_npz
    component_specs = load_component_specs(component_spec_source)
    component_variant_indices = [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in component_specs
    ]
    prediction_active = bool(args.prediction_bed_prefix.strip() or args.prediction_pgen_prefix.strip())

    _n_dense = sum(bool(x) for x in [bed_list, pgen_prefix])
    _n_rare = sum(bool(x) for x in [rare_bed_list, rare_pgen_prefix])
    if _n_dense == 0 and _n_rare == 0:
        raise SystemExit(
            "No genotype input specified. "
            "Use --bed-prefix / --pgen-prefix and/or --rare-bed-prefix / --rare-pgen-prefix."
        )
    if _n_dense > 1:
        raise SystemExit(
            "Specify only one of --bed-prefix / --pgen-prefix."
        )
    if _n_rare > 1:
        raise SystemExit(
            "Specify only one of --rare-bed-prefix / --rare-pgen-prefix."
        )
    if vc_block_sizes and component_variant_indices:
        raise SystemExit("Use only one of --vc-block-sizes or --component-indices-npz.")
    if use_smile and not smile_inputs_present:
        raise SystemExit(
            "SMILE mode requires one of --identity-w, --w-files, --w-files-list, or --grm-groups."
        )
    if sum(bool(x) for x in (args.identity_w, smile_w_files, smile_w_file_groups)) > 1:
        raise SystemExit("Use only one of --identity-w, --w-files/--w-files-list, or --grm-groups.")
    if use_smile and (vc_block_sizes or component_variant_indices):
        raise SystemExit("SMILE block-W mode cannot be combined with component partitioning.")
    if use_smile and (rare_bed_list or rare_pgen_prefix):
        raise SystemExit("SMILE block-W mode is currently supported only for dense-only fits.")
    if use_smile and _n_dense != 1:
        raise SystemExit("SMILE block-W mode requires exactly one dense genotype input.")
    if use_smile and len(bed_list) > 1:
        raise SystemExit("SMILE block-W mode cannot be combined with multiple BED prefixes.")
    if vc_block_sizes or component_variant_indices:
        if _n_dense != 1:
            raise SystemExit("single-source component partitioning requires exactly one dense input.")
        if len(bed_list) > 1:
            raise SystemExit("single-source component partitioning cannot be combined with multiple BED prefixes.")
        if rare_bed_list or rare_pgen_prefix:
            raise SystemExit("single-source component partitioning is currently supported only for dense-only fits.")
    if not args.pheno_txt:
        raise SystemExit("--pheno-txt is required.")
    if (args.compute_effects or prediction_active) and not args.out_prefix:
        raise SystemExit("--compute-effects / prediction inputs require --out-prefix.")
    if prediction_active:
        _pred_dense = sum(bool(x) for x in [args.prediction_bed_prefix.strip(), args.prediction_pgen_prefix.strip()])
        if _pred_dense != 1:
            raise SystemExit(
                "Prediction requires exactly one of --prediction-bed-prefix or --prediction-pgen-prefix."
            )

    temp_paths: list[str] = []

    # ---- Sample alignment (BED or PGEN FAM) ---------------
    if pgen_prefix:
        fam_path = make_nonbed_input_fam(pgen_prefix=pgen_prefix)
        temp_paths.append(fam_path)
    elif bed_list:
        fam_path = bed_list[0] + ".fam"
    elif rare_pgen_prefix:
        fam_path = make_nonbed_input_fam(pgen_prefix=rare_pgen_prefix)
        temp_paths.append(fam_path)
    else:
        fam_path = rare_bed_list[0] + ".fam"

    for path in temp_paths:
        atexit.register(cleanup_path, path)

    keep_ids = None
    if args.keep_path and os.path.exists(args.keep_path):
        keep_ids = read_keep_ids(args.keep_path)

    y_np, X_np, fam_keep, dropped, covar_transform = load_pheno_covar_aligned_with_transform(
        fam_path=fam_path, pheno_path=args.pheno_txt,
        covar_path=args.covar_txt or None, add_intercept=True, keep_ids=keep_ids)
    y_np = y_np.astype(np.float32, copy=False)
    if X_np is not None:
        X_np = X_np.astype(np.float32, copy=False)
    logger.info("Loaded %d samples; dropped %d", y_np.shape[0], len(dropped))

    # ---- PGEN direct read or BED sample-mask ---------------------------------
    sources = None
    sample_mask = None
    rare_cfg_sources = None
    rare_cfg_bed = []
    p_list = []
    if pgen_prefix:
        sample_mask = compute_sample_mask(fam_path, fam_keep)
        sources = [PgenGenoSource(pgen_prefix, sample_mask=sample_mask)]
        p_list = [sources[0].m]
        logger.info("Direct read: PgenGenoSource n_source=%d m=%d n_keep=%d",
                    sources[0]._n_full, sources[0].m, sources[0].n)
        sample_mask = None  # handled inside source
    elif bed_list:
        n_bed = _bed_count(bed_list[0] + ".bed", "iid_count")
        if n_bed != len(fam_keep):
            sample_mask = compute_sample_mask(fam_path, fam_keep)
            logger.info("BED path: using sample_mask (n_bed=%d -> n_keep=%d)",
                        n_bed, len(fam_keep))
        p_list = [_bed_count(pref + ".bed", "sid_count") for pref in bed_list]

    if rare_pgen_prefix:
        rare_sm = sample_mask if sample_mask is not None else compute_sample_mask(fam_path, fam_keep)
        rare_cfg_sources = [PgenGenoSource(rare_pgen_prefix, sample_mask=rare_sm)]
        p_list.append(rare_cfg_sources[0].m)
        logger.info("Rare-variant PGEN sparse input: m=%d", rare_cfg_sources[0].m)
    elif rare_bed_list:
        n_rare_bed = _bed_count(rare_bed_list[0] + ".bed", "iid_count")
        if n_rare_bed != len(fam_keep) and sample_mask is None:
            sample_mask = compute_sample_mask(fam_path, fam_keep)
            logger.info("Rare BED path: using sample_mask (n_bed=%d -> n_keep=%d)",
                        n_rare_bed, len(fam_keep))
        rare_cfg_bed = rare_bed_list
        p_list.extend(_bed_count(pref + ".bed", "sid_count") for pref in rare_bed_list)
        logger.info("Rare-variant BED sparse input: %d file(s)", len(rare_bed_list))
    logger.info("Using %d samples after alignment.", len(fam_keep))

    if args.out_prefix:
        ensure_parent_dir(args.out_prefix)
    if args.keep_out:
        ensure_parent_dir(args.keep_out)
        with open(args.keep_out, "w") as f:
            for iid in fam_keep:
                f.write(f"{iid} {iid}\n")
    if args.dropped_out and dropped:
        ensure_parent_dir(args.dropped_out)
        with open(args.dropped_out, "w") as f:
            for iid in dropped:
                f.write(f"{iid}\n")

    # Planner
    gpu_name, gpu_total, gpu_free = setup_gpu()
    n_covar = int(X_np.shape[1]) if X_np is not None else 0
    cpu_threads, cpu_threads_src = resolve_cpu_threads(args.cpu_threads or None)

    planner_kwargs = dict(
        n_samples=y_np.shape[0], p_list=p_list,
        n_grm=(
            smile_n_grm
            if use_smile
            else len(component_variant_indices)
            if component_variant_indices
            else len(vc_block_sizes)
            if vc_block_sizes
            else len(p_list)
        ),
        component_block_sizes=(
            [int(len(group)) for group in component_variant_indices]
            if component_variant_indices
            else vc_block_sizes or None
        ),
        precond_type=args.precond_type,
        gpu_free=gpu_free,
        gpu_budget=(args.gpu_budget_gib * 1024**3) if args.gpu_budget_gib > 0 else None,
        n_covar=n_covar,
        n_rand_vec=args.n_rand_vec,
        slq_samples=args.slq_samples,
        gpu_name=gpu_name,
        ring_depth=args.ring_depth if args.ring_depth > 0 else None,
        source_format=(
            "bed"
            if bed_list
            else "pgen"
            if pgen_prefix
            else None
        ),
        arbitrary_component_partition=bool(component_variant_indices),
    )
    if use_smile:
        plan = run_smile_planner(
            **planner_kwargs,
            smile_w_block_sizes=smile_w_block_sizes or None,
        )
    else:
        plan = run_planner(**planner_kwargs)
    planned_source_build_chunk_width = (
        plan.source_build_chunk_width if plan.source_build_chunk_width > 0 else None
    )

    call_width = args.call_width or plan.call_width
    gpu_budget_bytes = (
        float(args.gpu_budget_gib) * 1024**3
        if args.gpu_budget_gib > 0
        else float(plan.gpu_budget_gib) * 1024**3
    )
    precond_rank = plan.precond_rank

    logger.info("call_width=%d, n_rand_vec=%d, seed=%d, precond_rank=%d, slq_samples=%d, slq_mode=%s, precond_refresh_reldp=%.3g, "
                "gpu_budget_gib=%s", call_width, args.n_rand_vec, args.seed, precond_rank,
                args.slq_samples, args.slq_mode, args.precond_refresh_reldp,
                args.gpu_budget_gib if args.gpu_budget_gib > 0 else 'auto')
    logger.info("cpu_threads=%d (source=%s)", cpu_threads, cpu_threads_src)
    print_planner_info(plan, gpu_name, gpu_free, call_width)
    logger.info("jax devices: %s", jax.devices())
    if vc_block_sizes or component_variant_indices:
        part_desc = (
            f"component_spec={component_spec_source}"
            if component_variant_indices
            else f"vc_block_sizes={vc_block_sizes}"
        )
        if args.precond_type == "projected_core" and precond_rank > 0:
            logger.info(
                "single-stream multi-GRM enabled via %s; "
                "projected_core preconditioner rank=%d",
                part_desc,
                precond_rank,
            )
        else:
            logger.info(
                "single-stream multi-GRM enabled via %s; preconditioner disabled",
                part_desc,
            )
    if use_smile:
        logger.info(
            "SMILE block-W mode enabled with %s; normalization=%s psd_check=%s",
            (
                "identity W"
                if args.identity_w
                else f"{len(smile_w_file_groups)} W file group(s)"
                if smile_w_file_groups
                else f"{len(smile_w_files)} W block file(s)"
            ),
            args.w_normalization,
            not args.no_w_psd_check,
        )

    if sources is not None:
        cfg = FitConfig(
            sources=sources, sample_mask=sample_mask, device=args.device,
            rare_sources=rare_cfg_sources, rare_bed_prefix=rare_cfg_bed,
            vc_block_sizes=vc_block_sizes or None,
            component_variant_indices=component_variant_indices or None,
            smile_w_files=smile_w_files or None,
            smile_w_file_groups=smile_w_file_groups or None,
            smile_weight_matrices=smile_weight_matrices,
            smile_weight_matrix_groups=smile_weight_matrix_groups,
            smile_identity=args.identity_w,
            smile_normalization=args.w_normalization,
            smile_check_psd=not args.no_w_psd_check,
            call_width=call_width, keep_host_stats=prediction_active,
            cpu_threads=cpu_threads,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            source_build_chunk_width=planned_source_build_chunk_width,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter, seed=args.seed,
            slq_samples=args.slq_samples, slq_m=args.slq_m, slq_mode=args.slq_mode,
            precond_refresh_reldp=args.precond_refresh_reldp,
            precond_type=args.precond_type, precond_rank=precond_rank,
            verbose=args.verbose)
    else:
        cfg = FitConfig(
            bed_prefix=bed_list, device=args.device,
            sample_mask=sample_mask,
            rare_sources=rare_cfg_sources, rare_bed_prefix=rare_cfg_bed,
            vc_block_sizes=vc_block_sizes or None,
            component_variant_indices=component_variant_indices or None,
            smile_w_files=smile_w_files or None,
            smile_w_file_groups=smile_w_file_groups or None,
            smile_weight_matrices=smile_weight_matrices,
            smile_weight_matrix_groups=smile_weight_matrix_groups,
            smile_identity=args.identity_w,
            smile_normalization=args.w_normalization,
            smile_check_psd=not args.no_w_psd_check,
            call_width=call_width, keep_host_stats=prediction_active,
            cpu_threads=cpu_threads,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            source_build_chunk_width=planned_source_build_chunk_width,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter, seed=args.seed,
            slq_samples=args.slq_samples, slq_m=args.slq_m, slq_mode=args.slq_mode,
            precond_refresh_reldp=args.precond_refresh_reldp,
            precond_type=args.precond_type, precond_rank=precond_rank,
            verbose=args.verbose)

    logger.info("fit start @ %s", datetime.now().isoformat(timespec='seconds'))
    model = InfinitesimalREMLFitter(cfg)
    close_model = model.close
    atexit.register(close_model)
    need_effects = args.compute_effects or prediction_active
    res = model.fit_infinitesimal(
        jnp.asarray(y_np),
        jnp.asarray(X_np) if X_np is not None else None,
        estimate_effects=need_effects,
    )
    logger.info("fit done @ %s, elapsed=%.1fs", datetime.now().isoformat(timespec='seconds'), time.time() - t0)
    print("var_components:", res.var_components)
    vc = np.asarray(res.var_components, dtype=float).reshape(-1)
    if vc.size >= 2:
        total = float(np.sum(vc))
        if total > 0:
            print(f"h2: {float(np.sum(vc[:-1]) / total):.6f}")
    if res.history:
        for it in res.history:
            pcg = it.get("pcg_iters", 0)
            print(f"  iter {int(it['iter']):03d}: ll={it.get('loglik',float('nan')):.4g} "
                  f"rel_dll={it.get('rel_dll',float('nan')):.3e} pcg={pcg}")

    if args.compute_effects:
        if res.effects is None:
            raise RuntimeError("compute-effects was requested but no effects were returned.")
        component_offsets = None
        component_source_variant_indices = None
        if getattr(model, "_partitioned_streamer", None) is not None:
            if getattr(model._partitioned_streamer, "has_arbitrary_component_partition", False):
                component_source_variant_indices = [
                    model._partitioned_streamer.component_source_variant_indices(g_idx)
                    for g_idx in range(int(model._partitioned_streamer.n_components))
                ]
            else:
                component_offsets = np.asarray(
                    model._partitioned_streamer._component_snp_offsets[:-1],
                    dtype=np.int64,
                )
        effect_paths = write_effect_outputs(
            out_prefix=args.out_prefix,
            effects=res.effects,
            sample_ids=fam_keep,
            component_global_offsets=component_offsets,
            component_source_variant_indices=component_source_variant_indices,
            component_names=(
                [str(spec.name) for spec in component_specs]
                if component_specs
                else None
            ),
            component_annotations=(
                [spec.annotation for spec in component_specs]
                if component_specs
                else None
            ),
            component_provenance=(
                [spec.provenance for spec in component_specs]
                if component_specs
                else None
            ),
        )
        logger.info("effect outputs written:")
        logger.info("  fixed_effects -> %s", effect_paths["fixed_effects"])
        logger.info("  random_effect -> %s", effect_paths["random_effect"])
        logger.info("  random_effect_components -> %s", effect_paths["random_effect_components"])
        logger.info("  effect_metadata -> %s", effect_paths["effect_metadata"])
        logger.info(
            "  snp_effects -> %d files under %s",
            len(res.effects.snp_effects),
            args.out_prefix + ".snp_effects.component_*.tsv",
        )

    if prediction_active:
        if res.effects is None:
            raise RuntimeError("Prediction inputs were provided but no effects were returned.")
        std_overrides = []
        for st in model.streamers:
            if st._means_host is None or st._inv_sds_host is None:
                raise RuntimeError(
                    "Prediction requires retained training SNP standardization stats."
                )
            std_overrides.append((st._means_host, st._inv_sds_host))
        pred_bed_list = [b for b in args.prediction_bed_prefix.split(",") if b]
        pred_pgen_prefix = args.prediction_pgen_prefix.strip()

        pred_temp_paths: list[str] = []
        if pred_pgen_prefix:
            pred_fam_path = make_nonbed_input_fam(pgen_prefix=pred_pgen_prefix)
            pred_temp_paths.append(pred_fam_path)
        else:
            pred_fam_path = pred_bed_list[0] + ".fam"
        for path in pred_temp_paths:
            atexit.register(cleanup_path, path)

        pred_X_np, pred_keep_ids, pred_dropped = load_covar_aligned(
            pred_fam_path,
            args.prediction_covar_txt or None,
            transform=covar_transform,
        )
        logger.info("Prediction set loaded %d samples; dropped %d", len(pred_keep_ids), len(pred_dropped))

        pred_sources = None
        pred_sample_mask = None
        if pred_pgen_prefix:
            pred_sample_mask = compute_sample_mask(pred_fam_path, pred_keep_ids)
            pred_sources = [PgenGenoSource(pred_pgen_prefix, sample_mask=pred_sample_mask)]
            pred_sample_mask = None
        else:
            n_pred_bed = _bed_count(pred_bed_list[0] + ".bed", "iid_count")
            if n_pred_bed != len(pred_keep_ids):
                pred_sample_mask = compute_sample_mask(pred_fam_path, pred_keep_ids)

        pred_cfg_kwargs = dict(
            device=args.device,
            sample_mask=pred_sample_mask,
            vc_block_sizes=vc_block_sizes or None,
            component_variant_indices=component_variant_indices or None,
            standardization_overrides=std_overrides,
            call_width=call_width,
            keep_host_stats=False,
            cpu_threads=cpu_threads,
            ring_depth=plan.ring_depth,
            source_build_chunk_width=(
                planned_source_build_chunk_width
                if (
                    (pred_bed_list and bed_list)
                    or (pred_pgen_prefix and pgen_prefix)
                )
                else None
            ),
            n_rand_vec=args.n_rand_vec,
            minq_iter=args.minq_iter,
            slq_samples=args.slq_samples,
            slq_m=args.slq_m,
            slq_mode=args.slq_mode,
            precond_type=args.precond_type,
            precond_rank=0,
            verbose=args.verbose,
        )
        if pred_sources is not None:
            pred_model = InfinitesimalREMLFitter(
                FitConfig(
                    sources=pred_sources,
                    **pred_cfg_kwargs,
                )
            )
        else:
            pred_model = InfinitesimalREMLFitter(
                FitConfig(
                    bed_prefix=pred_bed_list,
                    **pred_cfg_kwargs,
                )
            )
        try:
            preds = model.predict(
                res.effects,
                test_fitter=pred_model,
                test_covar=jnp.asarray(pred_X_np) if pred_X_np is not None else None,
            )
        finally:
            for st in pred_model.streamers:
                try:
                    st.close()
                except (OSError, RuntimeError, ValueError):
                    logger.debug("Failed to close prediction streamer.", exc_info=True)

        pred_paths = write_prediction_outputs(
            out_prefix=args.out_prefix,
            predictions=preds,
            sample_ids=pred_keep_ids,
        )
        logger.info("prediction outputs written:")
        for key, path in pred_paths.items():
            logger.info("  %s -> %s", key, path)

    close_model()
    atexit.unregister(close_model)


if __name__ == "__main__":
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    for _name in ("GPU_REML_v6", pkg_name, __name__):
        _lg = logging.getLogger(_name)
        _lg.addHandler(_h)
        _lg.setLevel(logging.INFO)
    main()
