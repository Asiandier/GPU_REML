#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import importlib
import logging
import os
import sys
import time

repo_root = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(repo_root)
if parent not in sys.path:
    sys.path.insert(0, parent)
pkg_name = os.path.basename(repo_root)
_runtime_mod = importlib.import_module(f"{pkg_name}.runtime_env")
_runtime_mod.configure_runtime_env()

import jax
import jax.numpy as jnp
import numpy as np

_data_mod = importlib.import_module(f"{pkg_name}.data_utils")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_geno_stream_mod = importlib.import_module(f"{pkg_name}.geno_stream")
_geno_source_mod = importlib.import_module(f"{pkg_name}.geno_source")
_relax_mod = importlib.import_module(f"{pkg_name}.relaxation_grouping")

BedBlockStreamer = _geno_stream_mod.BedBlockStreamer
PgenGenoSource = _geno_source_mod.PgenGenoSource
GenoBlockStreamer = _geno_stream_mod.GenoBlockStreamer
load_pheno_covar_aligned_with_transform = _data_mod.load_pheno_covar_aligned_with_transform
compute_sample_mask = _common_mod.compute_sample_mask
make_nonbed_input_fam = _common_mod.make_nonbed_input_fam
read_keep_ids = _common_mod.read_keep_ids
setup_gpu = _common_mod.setup_gpu
resolve_cpu_threads = _common_mod.resolve_cpu_threads

RelaxationConfig = _relax_mod.RelaxationConfig
initialize_theta = _relax_mod.initialize_theta
run_relaxation_grouping = _relax_mod.run_relaxation_grouping
write_relaxation_outputs = _relax_mod.write_relaxation_outputs

logger = logging.getLogger(__name__)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run simplex-constrained soft SNP-grouping updates by alternating REML "
            "with SNP-level profiled-likelihood exponentiated-gradient ascent."
        )
    )
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""))
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""))
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument("--keep-path", default=env("KEEP_PATH", ""))
    p.add_argument("--out-prefix", default=env("OUT_PREFIX", "relaxation_grouping"))
    p.add_argument("--device", default=env("DEVICE", "gpu"))
    p.add_argument("--call-width", type=int, default=int(env("CALL_WIDTH", "65536")))
    p.add_argument("--cpu-threads", type=int, default=int(env("CPU_THREADS", "0")))
    p.add_argument("--n-groups", type=int, required=True)
    p.add_argument("--outer-iters", type=int, default=int(env("RELAX_OUTER_ITERS", "5")))
    p.add_argument("--theta-lr", type=float, default=float(env("RELAX_THETA_LR", "1e-3")))
    p.add_argument("--theta-init-npy", default=env("RELAX_THETA_INIT_NPY", ""))
    p.add_argument("--theta-init", choices=["random", "uniform"], default=env("RELAX_THETA_INIT", "random"))
    p.add_argument("--theta-random-low", type=float, default=float(env("RELAX_THETA_RANDOM_LOW", "0.0")))
    p.add_argument("--theta-random-high", type=float, default=float(env("RELAX_THETA_RANDOM_HIGH", "1.0")))
    p.add_argument("--n-rand-vec", type=int, default=int(env("N_RAND_VEC", "32")))
    p.add_argument("--max-pcg-iters", type=int, default=int(env("MAX_PCG_ITERS", "400")))
    p.add_argument("--minq-iter", type=int, default=int(env("MINQ_ITER", "20")))
    p.add_argument("--slq-samples", type=int, default=int(env("SLQ_SAMPLES", "4")))
    p.add_argument("--slq-m", type=int, default=int(env("SLQ_M", "8")))
    p.add_argument("--h2-init", type=float, default=float(env("H2_INIT", "0.5")))
    p.add_argument("--pcg-tol", type=float, default=float(env("RELAX_PCG_TOL", "1e-3")))
    p.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    p.add_argument("--no-final-refit", action="store_true")
    p.add_argument("--verbose", action="store_true", default=env("VERBOSE", "1").lower() not in {"0", "false", "no"})
    return p.parse_args()


def _build_streamer(args, fam_keep: list[str]):
    bed_prefix = args.bed_prefix.strip()
    pgen_prefix = args.pgen_prefix.strip()
    if bool(bed_prefix) == bool(pgen_prefix):
        raise SystemExit("Specify exactly one of --bed-prefix or --pgen-prefix.")

    cpu_threads, _ = resolve_cpu_threads(args.cpu_threads if args.cpu_threads > 0 else None)
    if pgen_prefix:
        fam_path = make_nonbed_input_fam(pgen_prefix=pgen_prefix)
    else:
        fam_path = bed_prefix + ".fam"
    sample_mask = compute_sample_mask(fam_path, fam_keep)

    if pgen_prefix:
        source = PgenGenoSource(pgen_prefix, sample_mask=sample_mask)
        logger.info(
            "Direct read: PgenGenoSource n_source=%d m=%d n_keep=%d",
            int(getattr(source, "_n_full", source.n)),
            int(source.m),
            int(source.n),
        )
        return GenoBlockStreamer(
            source=source,
            call_width=int(args.call_width),
            device=args.device,
            keep_host_stats=False,
            build_threads=cpu_threads,
        )
    logger.info(
        "BED path: using sample_mask (n_bed=%d -> n_keep=%d)",
        int(sample_mask.shape[0]),
        int(np.sum(sample_mask)),
    )
    return BedBlockStreamer(
        bed_prefix,
        call_width=int(args.call_width),
        device=args.device,
        keep_host_stats=False,
        build_threads=cpu_threads,
        sample_mask=sample_mask,
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger.info("start @ %s", time.strftime("%Y-%m-%dT%H:%M:%S"))
    setup_gpu()
    t0 = time.time()

    bed_prefix = args.bed_prefix.strip()
    pgen_prefix = args.pgen_prefix.strip()
    if bool(bed_prefix) == bool(pgen_prefix):
        raise SystemExit("Specify exactly one of --bed-prefix or --pgen-prefix.")
    if not args.pheno_txt:
        raise SystemExit("--pheno-txt is required.")
    fam_path = make_nonbed_input_fam(pgen_prefix=pgen_prefix) if pgen_prefix else bed_prefix + ".fam"
    keep_ids = read_keep_ids(args.keep_path) if args.keep_path else None
    y_np, X_np, fam_keep, dropped, _ = load_pheno_covar_aligned_with_transform(
        fam_path,
        args.pheno_txt,
        args.covar_txt or None,
        keep_ids=keep_ids,
    )
    logger.info("Loaded %d samples; dropped %d", y_np.shape[0], len(dropped))
    logger.info("Using %d samples after alignment.", len(fam_keep))

    streamer = _build_streamer(args, fam_keep)
    atexit.register(streamer.close)
    if args.theta_init_npy:
        theta0 = np.load(args.theta_init_npy).astype(np.float32, copy=False)
    else:
        theta0 = initialize_theta(
            m=int(streamer.m),
            n_groups=int(args.n_groups),
            seed=int(args.seed),
            mode=args.theta_init,
            random_low=float(args.theta_random_low),
            random_high=float(args.theta_random_high),
        )

    n_covar = int(X_np.shape[1]) if X_np is not None else 0
    logger.info(
        "relaxation_grouping: n_groups=%d outer_iters=%d theta_lr=%.6g seed=%d "
        "n_rand_vec=%d slq_samples=%d slq_m=%d minq_iter=%d final_refit=%s",
        int(args.n_groups),
        int(args.outer_iters),
        float(args.theta_lr),
        int(args.seed),
        int(args.n_rand_vec),
        int(args.slq_samples),
        int(args.slq_m),
        int(args.minq_iter),
        "no" if args.no_final_refit else "yes",
    )
    logger.info(
        "genotype: m=%d n=%d call_width=%d device=%s n_covar=%d",
        int(streamer.m),
        int(streamer.n),
        int(args.call_width),
        args.device,
        n_covar,
    )
    logger.info("jax devices: %s", jax.devices())
    logger.info("fit start @ %s", time.strftime("%Y-%m-%dT%H:%M:%S"))
    cfg = RelaxationConfig(
        n_groups=int(args.n_groups),
        outer_iters=int(args.outer_iters),
        theta_lr=float(args.theta_lr),
        n_rand_vec=int(args.n_rand_vec),
        max_pcg_iters=int(args.max_pcg_iters),
        minq_iter=int(args.minq_iter),
        seed=int(args.seed),
        h2_init=float(args.h2_init),
        slq_samples=int(args.slq_samples),
        slq_m=int(args.slq_m),
        pcg_tol=float(args.pcg_tol),
        verbose=bool(args.verbose),
        refit_final=not bool(args.no_final_refit),
    )
    result = run_relaxation_grouping(
        streamer=streamer,
        y=jnp.asarray(y_np, dtype=jnp.float32),
        covar=jnp.asarray(X_np, dtype=jnp.float32) if X_np is not None else None,
        theta_init=theta0,
        cfg=cfg,
    )
    logger.info("fit done @ %s, elapsed=%.1fs", time.strftime("%Y-%m-%dT%H:%M:%S"), time.time() - t0)
    paths = write_relaxation_outputs(
        out_prefix=args.out_prefix,
        result=result,
        config=cfg,
    )
    vc = np.asarray(jax.device_get(result.var_components), dtype=float).reshape(-1)
    print("var_components:", vc)
    if vc.size >= 2:
        total = float(np.sum(vc))
        if total > 0:
            print(f"h2: {float(np.sum(vc[:-1]) / total):.6f}")
    for row in result.history:
        outer = row.get("outer_iter")
        ll = row.get("loglik", float("nan"))
        step = row.get("theta_step_norm", float("nan"))
        grad = row.get("theta_grad_norm", float("nan"))
        tmin = row.get("theta_min", float("nan"))
        tmax = row.get("theta_max", float("nan"))
        stop = row.get("reml_stop_reason", "")
        print(
            f"  outer {outer}: ll={float(ll):.4g} theta_step={float(step):.3e} "
            f"theta_grad={float(grad):.3e} theta=[{float(tmin):.3e}, {float(tmax):.3e}] "
            f"reml_stop={stop}"
        )
    if args.no_final_refit:
        print(
            "note: --no-final-refit was used; var_components are from the last REML "
            "before the final theta update."
        )
    else:
        print("note: final refit completed; var_components correspond to the written theta.")
    print("theta:", paths["theta"])
    print("theta_grad:", paths["theta_grad"])
    print("history:", paths["history"])
    print(f"elapsed_sec: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
