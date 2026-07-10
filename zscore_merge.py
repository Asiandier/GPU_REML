#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent
parent = repo_root.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

try:
    from .component_spec import load_component_specs
except ImportError:  # pragma: no cover - direct script execution
    from component_spec import load_component_specs


# Boundary variance-component test:
#   0.5 * P(chi^2_1 >= T) = 0.05 -> T = qchisq(0.90, df=1).
# The equivalent one-sided z cutoff is sqrt(T).
DEFAULT_Z_CUTOFF = 1.6448536269514722


def write_component_spec(path: Path, components: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"components": components}, handle, indent=2)
        handle.write("\n")


def specs_to_components(spec_path: str) -> list[dict[str, object]]:
    components = []
    for idx, spec in enumerate(load_component_specs(spec_path)):
        components.append(
            {
                "name": str(spec.name or f"component_{idx:04d}"),
                "variant_indices": [
                    int(x)
                    for x in np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1).tolist()
                ],
                "annotation": dict(spec.annotation or {}),
                "provenance": dict(spec.provenance or {}),
            }
        )
    return components


def h2_from_theta(
    theta: np.ndarray,
    genetic_trace_atoms: np.ndarray | None = None,
    residual_trace_atoms: np.ndarray | None = None,
) -> float:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if genetic_trace_atoms is None:
        genetic = float(theta[:-1].sum())
        residual = float(theta[-1])
    else:
        genetic_atoms = np.asarray(genetic_trace_atoms, dtype=np.float64).reshape(-1)
        residual_atoms = np.asarray(
            [1.0] if residual_trace_atoms is None else residual_trace_atoms,
            dtype=np.float64,
        ).reshape(-1)
        if theta.size != genetic_atoms.size + residual_atoms.size:
            raise ValueError("Trace-atom lengths do not match theta.")
        genetic = float(np.dot(theta[: genetic_atoms.size], genetic_atoms))
        residual = float(np.dot(theta[genetic_atoms.size :], residual_atoms))
    denom = genetic + residual
    return genetic / denom if denom > 0.0 else float("nan")


def _h2_from_round(theta: np.ndarray, prefix: Path) -> float:
    meta_path = prefix.with_suffix(".ai_stats.json")
    if not meta_path.exists():
        return h2_from_theta(theta)
    meta = json.loads(meta_path.read_text())
    genetic_atoms = meta.get("genetic_trace_atoms")
    if genetic_atoms is None:
        return h2_from_theta(theta)
    return h2_from_theta(theta, genetic_atoms, meta.get("residual_trace_atoms"))


def zscore_components(
    theta: np.ndarray,
    ai: np.ndarray,
    *,
    n_eff: float,
    ai_scale_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    theta_g = np.asarray(theta[:-1], dtype=np.float64)
    ai_full = np.asarray(ai, dtype=np.float64)
    if ai_full.ndim != 2 or ai_full.shape[0] != theta.size or ai_full.shape[1] != theta.size:
        raise ValueError(f"AI shape {ai_full.shape} is incompatible with theta length {theta.size}.")
    ai_full = 0.5 * (ai_full + ai_full.T)
    if ai_scale_mode == "per_sample":
        ai_full = ai_full * float(n_eff)
    elif ai_scale_mode != "full":
        raise ValueError(f"Unsupported AI scale mode: {ai_scale_mode}")

    scale = float(np.mean(np.abs(np.diag(ai_full)))) if ai_full.size else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    ai_full = ai_full + np.eye(ai_full.shape[0], dtype=np.float64) * (1e-8 * scale)
    # Invert the full AI matrix, then take the genetic block. The residual
    # variance is jointly estimated and must be treated as a nuisance parameter.
    cov_full = np.linalg.pinv(ai_full, hermitian=True)
    cov_g = 0.5 * (cov_full[:-1, :-1] + cov_full[:-1, :-1].T)
    se = np.sqrt(np.maximum(np.diag(cov_g), 0.0))
    z = np.divide(theta_g, se, out=np.zeros_like(theta_g), where=se > 0.0)
    return se, z


def merged_component(
    block: list[tuple[int, dict[str, object]]],
    z: np.ndarray,
    z_cutoff: float,
    out_idx: int,
    status: str,
) -> dict[str, object]:
    variant_indices: list[int] = []
    source_names: list[str] = []
    source_z: list[float] = []
    source_annotations = []
    source_positions: list[int] = []
    for pos, comp in block:
        variant_indices.extend(int(x) for x in comp["variant_indices"])
        source_names.append(str(comp.get("name", f"component_{pos:04d}")))
        source_z.append(float(z[pos]))
        source_annotations.append(comp.get("annotation"))
        source_positions.append(int(pos))
    return {
        "name": f"zmerge_block_{out_idx:04d}",
        "variant_indices": variant_indices,
        "annotation": {
            "method": "zscore_one_shot_merge",
            "n_source_components": int(len(block)),
            "n_variants": int(len(variant_indices)),
            "z_cutoff": float(z_cutoff),
            "source_z_min": float(np.min(source_z)),
            "source_z_max": float(np.max(source_z)),
            "source_z_mean": float(np.mean(source_z)),
            "zscore_merge_status": status,
            "source_component_positions": source_positions,
            "source_annotations": source_annotations,
        },
        "provenance": {
            "method": "zscore_one_shot_merge",
            "source_components": source_names,
        },
    }


def merge_global_weak(
    components: list[dict[str, object]],
    z: np.ndarray,
    z_cutoff: float,
) -> tuple[list[dict[str, object]], int, int]:
    weak = np.asarray(z < z_cutoff, dtype=bool)
    n_weak = int(weak.sum())
    weak_block: list[tuple[int, dict[str, object]]] = []
    out: list[dict[str, object]] = []
    for idx, comp_in in enumerate(components):
        comp = dict(comp_in)
        ann = dict(comp.get("annotation") or {})
        if weak[idx]:
            weak_block.append((idx, comp))
            continue
        ann["zscore_merge_status"] = "kept_significant"
        ann["zscore"] = float(z[idx])
        comp["annotation"] = ann
        out.append(comp)

    if not weak_block:
        return out, n_weak, 0
    if len(weak_block) == 1:
        pos, comp = weak_block[0]
        ann = dict(comp.get("annotation") or {})
        ann["zscore_merge_status"] = "weak_singleton"
        ann["zscore"] = float(z[pos])
        comp["annotation"] = ann
        out.append(comp)
        return out, n_weak, 0

    out.append(merged_component(weak_block, z, z_cutoff, len(out), "merged_global_weak"))
    return out, n_weak, 1


def _validate_same_variant_coverage(before: list[dict[str, object]], after: list[dict[str, object]]) -> None:
    before_variants = [int(x) for comp in before for x in comp["variant_indices"]]
    after_variants = [int(x) for comp in after for x in comp["variant_indices"]]
    if len(after_variants) != len(set(after_variants)):
        raise ValueError("merged component spec contains overlapping variant indices.")
    if sorted(before_variants) != sorted(after_variants):
        raise ValueError("merged component spec does not cover the same variants as the fine model.")


def merge_components(
    components: list[dict[str, object]],
    z: np.ndarray,
    z_cutoff: float,
    merge_mode: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], int, int]:
    if merge_mode == "global_weak":
        out = merge_global_weak(components, z, z_cutoff)
    else:
        raise ValueError(f"Unsupported merge mode: {merge_mode}")
    _validate_same_variant_coverage(components, out[0])
    return out


def write_trace_row(path: Path, row: dict[str, object]) -> None:
    cols = [
        "round",
        "stage",
        "merge_mode",
        "n_components",
        "h2",
        "sum_theta_g",
        "theta_e",
        "n_weak_z",
        "n_reject_z",
        "n_merged_groups",
        "next_n_components",
        "z_cutoff",
        "z_min",
        "z_median",
        "z_max",
        "se_median",
        "component_spec",
        "theta_npy",
        "ai_npy",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in cols})


def _add_arg(cmd: list[str], flag: str, value: object, *, include_zero: bool = True) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    if isinstance(value, (int, float)) and not include_zero and float(value) == 0.0:
        return
    cmd.extend([flag, str(value)])


def _pipeline_command(args: argparse.Namespace, component_spec: Path, out_prefix: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "run_reml_pipeline.py"),
        "--component-spec",
        str(component_spec),
        "--pheno-txt",
        args.pheno_txt,
        "--out-prefix",
        str(out_prefix),
        "--export-ai",
    ]
    _add_arg(cmd, "--bed-prefix", args.bed_prefix)
    _add_arg(cmd, "--pgen-prefix", args.pgen_prefix)
    _add_arg(cmd, "--covar-txt", args.covar_txt)
    _add_arg(cmd, "--keep-path", args.keep_path)
    _add_arg(cmd, "--device", args.device)
    _add_arg(cmd, "--call-width", args.call_width, include_zero=False)
    _add_arg(cmd, "--gpu-budget-gib", args.gpu_budget_gib)
    _add_arg(cmd, "--ring-depth", args.ring_depth)
    _add_arg(cmd, "--cpu-threads", args.cpu_threads)
    _add_arg(cmd, "--n-rand-vec", args.n_rand_vec)
    _add_arg(cmd, "--seed", args.seed)
    _add_arg(cmd, "--slq-samples", args.slq_samples)
    _add_arg(cmd, "--slq-m", args.slq_m)
    _add_arg(cmd, "--slq-mode", args.slq_mode)
    _add_arg(cmd, "--precond-refresh-reldp", args.precond_refresh_reldp)
    _add_arg(cmd, "--precond-type", args.precond_type)
    _add_arg(cmd, "--minq-iter", args.minq_iter)
    if not args.verbose:
        cmd.append("--non-verbose")
    return cmd


def run_reml_round(args: argparse.Namespace, component_spec: Path, out_prefix: Path) -> None:
    theta_path = out_prefix.with_suffix(".theta.npy")
    ai_path = out_prefix.with_suffix(".ai.npy")
    if theta_path.exists() and ai_path.exists():
        return
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = _pipeline_command(args, component_spec, out_prefix)
    env = os.environ.copy()
    if int(args.cpu_threads or 0) > 0:
        threads = str(int(args.cpu_threads))
        env.update({"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads})
    stdout = out_prefix.parent / f"{out_prefix.name}.stdout.log"
    stderr = out_prefix.parent / f"{out_prefix.name}.stderr.log"
    time_txt = out_prefix.parent / f"{out_prefix.name}.time.txt"
    with stdout.open("w") as out, stderr.open("w") as err:
        subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_txt)] + cmd,
            check=True,
            stdout=out,
            stderr=err,
            env=env,
        )


def _read_n_eff(args: argparse.Namespace, stats_path: Path) -> float:
    if float(args.merge_n_eff) > 0:
        return float(args.merge_n_eff)
    if stats_path.exists():
        with stats_path.open() as handle:
            stats = json.load(handle)
        n_samples = float(stats.get("n_samples", 0.0))
        if n_samples > 0:
            return n_samples
    if args.keep_path:
        with open(args.keep_path) as handle:
            return float(sum(1 for line in handle if line.strip()))
    return 1.0


def _copy_final_outputs(final_prefix: Path, final_spec: Path, out_prefix: str) -> None:
    if not out_prefix:
        return
    target = Path(out_prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".theta.npy", ".ai.npy", ".grad.npy", ".ai_stats.json"):
        src = final_prefix.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, Path(str(target) + suffix))
    shutil.copy2(final_spec, Path(str(target) + ".component_spec.json"))


def _validate_merge_args(args: argparse.Namespace) -> Path:
    if args.smile or args.identity_w or args.w_files or args.w_files_list or args.grm_groups:
        raise SystemExit("--merge is separate from SMILE mode and cannot be combined with SMILE W inputs.")
    if args.rare_bed_prefix or args.rare_pgen_prefix:
        raise SystemExit("--merge currently supports dense single-source component-spec fits only.")
    if any(
        getattr(args, name, "")
        for name in ("admix_q", "admix_fam", "admix_component_names")
    ):
        raise SystemExit("--merge cannot be combined with ADMIXTURE covariance inputs.")
    if args.vc_block_sizes or args.component_indices_npz:
        raise SystemExit("--merge requires --component-spec; do not use --vc-block-sizes or --component-indices-npz.")
    if not args.component_spec:
        raise SystemExit("--merge requires --component-spec.")
    if not args.pheno_txt:
        raise SystemExit("--merge requires --pheno-txt.")
    if args.compute_effects or args.prediction_bed_prefix or args.prediction_pgen_prefix:
        raise SystemExit("--merge currently estimates variance components only; run effects/prediction after selecting the final component spec.")
    if not (args.bed_prefix or args.pgen_prefix):
        raise SystemExit("--merge requires --bed-prefix or --pgen-prefix.")
    if args.bed_prefix and args.pgen_prefix:
        raise SystemExit("--merge requires exactly one of --bed-prefix or --pgen-prefix.")
    if len([x for x in str(args.bed_prefix).split(",") if x.strip()]) > 1:
        raise SystemExit("--merge requires one BED prefix; component indices are relative to a single genotype source.")
    out_dir_raw = args.merge_out_dir or (args.out_prefix + ".merge" if args.out_prefix else "")
    if not out_dir_raw:
        raise SystemExit("--merge requires --merge-out-dir or --out-prefix.")
    return Path(out_dir_raw)


def run_from_pipeline_args(args: argparse.Namespace) -> dict[str, object]:
    out_dir = _validate_merge_args(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = specs_to_components(args.component_spec)
    if not components:
        raise SystemExit("--component-spec contains no components.")

    trace_path = out_dir / "zscore_merge_trace.tsv"
    summary_path = out_dir / "zscore_merge_summary.json"
    for stale_path in (trace_path, summary_path):
        if stale_path.exists():
            stale_path.unlink()

    round00_spec = out_dir / "round00.component_spec.json"
    write_component_spec(round00_spec, components)
    round00_prefix = out_dir / "round00" / "fit"
    run_reml_round(args, round00_spec, round00_prefix)

    fine_theta = np.asarray(np.load(round00_prefix.with_suffix(".theta.npy")), dtype=np.float64).reshape(-1)
    fine_ai = np.asarray(np.load(round00_prefix.with_suffix(".ai.npy")), dtype=np.float64)
    n_eff = _read_n_eff(args, round00_prefix.with_suffix(".ai_stats.json"))
    se, z = zscore_components(
        fine_theta,
        fine_ai,
        n_eff=n_eff,
        ai_scale_mode=args.merge_ai_scale_mode,
    )
    next_components, n_weak, n_merged_groups = merge_components(
        components,
        z,
        float(args.z_cutoff),
        args.merge_mode,
        args,
    )
    write_trace_row(
        trace_path,
        {
            "round": 0,
            "stage": "fine_model_zscore_merge",
            "merge_mode": args.merge_mode,
            "n_components": len(components),
            "h2": _h2_from_round(fine_theta, round00_prefix),
            "sum_theta_g": float(fine_theta[:-1].sum()),
            "theta_e": float(fine_theta[-1]),
            "n_weak_z": n_weak,
            "n_reject_z": int(z.size - n_weak),
            "n_merged_groups": n_merged_groups,
            "next_n_components": len(next_components),
            "z_cutoff": float(args.z_cutoff),
            "z_min": float(np.min(z)) if z.size else "",
            "z_median": float(np.median(z)) if z.size else "",
            "z_max": float(np.max(z)) if z.size else "",
            "se_median": float(np.median(se)) if se.size else "",
            "component_spec": str(round00_spec),
            "theta_npy": str(round00_prefix.with_suffix(".theta.npy")),
            "ai_npy": str(round00_prefix.with_suffix(".ai.npy")),
        },
    )
    np.save(out_dir / "round00" / "theta_se.npy", se)
    np.save(out_dir / "round00" / "theta_z.npy", z)

    final_prefix = round00_prefix
    final_spec = round00_spec
    final_components = components
    stop_reason = "no_more_weak_merges"
    merged_component_spec = ""
    if len(next_components) != len(components) and n_merged_groups > 0:
        round01_spec = out_dir / "round01.component_spec.json"
        write_component_spec(round01_spec, next_components)
        round01_prefix = out_dir / "round01" / "fit"
        run_reml_round(args, round01_spec, round01_prefix)
        merged_theta = np.asarray(np.load(round01_prefix.with_suffix(".theta.npy")), dtype=np.float64).reshape(-1)
        write_trace_row(
            trace_path,
            {
                "round": 1,
                "stage": "merged_refit",
                "merge_mode": args.merge_mode,
                "n_components": len(next_components),
                "h2": _h2_from_round(merged_theta, round01_prefix),
                "sum_theta_g": float(merged_theta[:-1].sum()),
                "theta_e": float(merged_theta[-1]),
                "next_n_components": len(next_components),
                "z_cutoff": float(args.z_cutoff),
                "component_spec": str(round01_spec),
                "theta_npy": str(round01_prefix.with_suffix(".theta.npy")),
                "ai_npy": str(round01_prefix.with_suffix(".ai.npy")),
            },
        )
        final_prefix = round01_prefix
        final_spec = round01_spec
        final_components = next_components
        stop_reason = "one_shot_merge_refit_complete"
        merged_component_spec = str(round01_spec)

    final_theta = np.asarray(np.load(final_prefix.with_suffix(".theta.npy")), dtype=np.float64).reshape(-1)
    summary = {
        "out_dir": str(out_dir),
        "stop_reason": stop_reason,
        "merge_strategy": "zscore_one_shot_merge_refit",
        "merge_mode": args.merge_mode,
        "z_cutoff": float(args.z_cutoff),
        "n_eff": float(n_eff),
        "ai_scale_mode": args.merge_ai_scale_mode,
        "fine_component_spec": str(round00_spec),
        "merged_component_spec": merged_component_spec,
        "final_component_spec": str(final_spec),
        "final_theta_npy": str(final_prefix.with_suffix(".theta.npy")),
        "final_h2": _h2_from_round(final_theta, final_prefix),
        "n_initial_components": len(components),
        "n_final_components": len(final_components),
        "n_weak_z_at_merge": int(n_weak),
        "n_reject_z_at_merge": int(z.size - n_weak),
        "n_merged_groups_at_merge": int(n_merged_groups),
        "trace_tsv": str(trace_path),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _copy_final_outputs(final_prefix, final_spec, args.out_prefix)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    try:
        from .run_reml_pipeline import parse_args
    except ImportError:  # pragma: no cover - direct script execution
        from run_reml_pipeline import parse_args

    args = parse_args()
    args.merge = True
    run_from_pipeline_args(args)


if __name__ == "__main__":
    main()
