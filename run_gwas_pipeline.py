#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import importlib
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

repo_root = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(repo_root)
if parent not in sys.path:
    sys.path.insert(0, parent)
pkg_name = os.path.basename(repo_root)
_runtime_mod = importlib.import_module(f"{pkg_name}.runtime_env")
_runtime_mod.configure_runtime_env()

logger = logging.getLogger(__name__)

_model_mod = importlib.import_module(f"{pkg_name}.reml_model")
_data_mod = importlib.import_module(f"{pkg_name}.data_utils")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_gwas_mod = importlib.import_module(f"{pkg_name}.gwas")
_source_mod = importlib.import_module(f"{pkg_name}.geno_source")

InfinitesimalREMLFitter = _model_mod.InfinitesimalREMLFitter
FitConfig = _model_mod.FitConfig
load_pheno_covar_aligned = _data_mod.load_pheno_covar_aligned
run_continuous_gwas = _gwas_mod.run_continuous_gwas
PgenGenoSource = _source_mod.PgenGenoSource
env = _common_mod.env
setup_gpu = _common_mod.setup_gpu
read_keep_ids = _common_mod.read_keep_ids
compute_sample_mask = _common_mod.compute_sample_mask
make_nonbed_input_fam = _common_mod.make_nonbed_input_fam
cleanup_path = _common_mod.cleanup_path
resolve_cpu_threads = _common_mod.resolve_cpu_threads


def parse_args():
    p = argparse.ArgumentParser(description="Run continuous-trait marginal GWAS on GPU-streamed genotypes.")
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""),
                   help="PLINK1 BED file prefix (comma-separated for multiple components).")
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""),
                   help="PLINK2 PGEN file prefix (comma-separated for multiple components).")
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument("--keep-path", default=env("KEEP_PATH", ""))
    p.add_argument("--keep-out", default=env("KEEP_OUT", ""))
    p.add_argument("--dropped-out", default=env("DROPPED_OUT", ""))
    p.add_argument("--out-prefix", default=env("OUT_PREFIX", ""))
    p.add_argument("--device", default=env("DEVICE", "gpu"))
    p.add_argument("--call-width", type=int, default=int(env("CALL_WIDTH", "131072")))
    p.add_argument("--ring-depth", type=int, default=int(env("RING_DEPTH", "32")))
    p.add_argument("--cpu-threads", type=int, default=int(env("CPU_THREADS", "0")))
    p.add_argument(
        "--verbose",
        action="store_true",
        default=env("VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("start @ %s", datetime.now().isoformat(timespec="seconds"))
    t0 = time.time()

    if not args.pheno_txt:
        raise ValueError("--pheno-txt is required.")
    if not args.out_prefix:
        raise ValueError("--out-prefix is required.")
    bed_list = [b for b in args.bed_prefix.split(",") if b]
    pgen_list = [p for p in args.pgen_prefix.split(",") if p]
    if bool(bed_list) == bool(pgen_list):
        raise ValueError("Provide exactly one of --bed-prefix or --pgen-prefix.")

    gpu_name, gpu_total, gpu_free = setup_gpu()
    cpu_threads, cpu_origin = resolve_cpu_threads(args.cpu_threads or None)
    logger.info("cpu_threads=%d (%s)", cpu_threads, cpu_origin)
    if gpu_name is not None:
        logger.info(
            "gpu=%s total=%.1fGiB free=%.1fGiB",
            gpu_name,
            float(gpu_total or 0.0) / (1024.0 ** 3),
            float(gpu_free or 0.0) / (1024.0 ** 3),
        )

    temp_fam_paths: list[str] = []
    if bed_list:
        fam_path = bed_list[0] + ".fam"
    else:
        fam_path = make_nonbed_input_fam(pgen_prefix=pgen_list[0])
        temp_fam_paths.append(fam_path)
        atexit.register(cleanup_path, fam_path)

    keep_ids = None
    if args.keep_path and os.path.exists(args.keep_path):
        keep_ids = read_keep_ids(args.keep_path)
    y_np, X_np, fam_keep, dropped = load_pheno_covar_aligned(
        fam_path=fam_path,
        pheno_path=args.pheno_txt,
        covar_path=args.covar_txt or None,
        add_intercept=True,
        keep_ids=keep_ids,
    )

    if args.keep_out:
        os.makedirs(os.path.dirname(args.keep_out) or ".", exist_ok=True)
        with open(args.keep_out, "w") as f:
            for iid in fam_keep:
                f.write(f"{iid} {iid}\n")
    if args.dropped_out:
        os.makedirs(os.path.dirname(args.dropped_out) or ".", exist_ok=True)
        with open(args.dropped_out, "w") as f:
            for iid in dropped:
                f.write(f"{iid}\n")

    gwas_keep_ids = fam_keep
    sample_mask = compute_sample_mask(fam_path, gwas_keep_ids)
    logger.info("samples kept=%d dropped=%d", int(sample_mask.sum()), len(dropped))

    if bed_list:
        cfg = FitConfig(
            bed_prefix=bed_list,
            device=args.device,
            sample_mask=sample_mask,
            call_width=max(256, int(args.call_width)),
            ring_depth=max(2, int(args.ring_depth)),
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            verbose=bool(args.verbose),
        )
    else:
        sources = [PgenGenoSource(pref, sample_mask=sample_mask) for pref in pgen_list]
        cfg = FitConfig(
            sources=sources,
            device=args.device,
            sample_mask=None,
            call_width=max(256, int(args.call_width)),
            ring_depth=max(2, int(args.ring_depth)),
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            verbose=bool(args.verbose),
        )

    fitter = InfinitesimalREMLFitter(cfg)
    try:
        summary = run_continuous_gwas(fitter, y_np, X_np, out_prefix=args.out_prefix)
    finally:
        fitter.close()
    logger.info(
        "GWAS done elapsed=%.1fs n=%d m=%d dof=%d out=%s",
        time.time() - t0,
        summary.n_samples,
        summary.n_variants,
        summary.dof,
        summary.out_path,
    )


if __name__ == "__main__":
    main()
