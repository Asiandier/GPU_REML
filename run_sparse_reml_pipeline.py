#!/usr/bin/env python3
"""
Sparse REML + LASSO pipeline.

Performance notes (vs. previous version):
  • Warm-start dictionary: maps SNP index → previous PCG solution column.
    When the candidate set overlaps between iterations (common), the warm
    starts for those columns are reused — dramatically reducing PCG iterations
    for the Z portion.
  • Gram matrix inputs kept in float64 throughout to avoid precision loss.
  • Pre-built JAX arrays for y and covar avoid repeated device_put.
  • Shared planner: uses the same closed-form GPU-width rule as pure REML.
"""
from __future__ import annotations

import argparse
import atexit
import importlib
import json
import logging
import os
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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from bed_reader import open_bed

# Highest FP32 accumulation is the robust default; users can explicitly select
# a faster hardware-specific mode after validating it on their GPU.
jax.config.update(
    "jax_default_matmul_precision",
    os.environ.get("GPU_REML_MATMUL_PRECISION", "highest"),
)
logger = logging.getLogger(__name__)

_inf_mod = importlib.import_module(f"{pkg_name}.reml_model")
_data_mod = importlib.import_module(f"{pkg_name}.data_utils")
_lasso_mod = importlib.import_module(f"{pkg_name}.lasso_cd")
_pcg_mod = importlib.import_module(f"{pkg_name}.pcg")
_precond_mod = importlib.import_module(f"{pkg_name}.precond")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_io_utils_mod = importlib.import_module(f"{pkg_name}.io_utils")
_component_spec_mod = importlib.import_module(f"{pkg_name}.component_spec")

InfinitesimalREMLFitter = _inf_mod.InfinitesimalREMLFitter
FitConfig = _inf_mod.FitConfig
load_pheno_covar_aligned = _data_mod.load_pheno_covar_aligned
LassoPathConfig = _lasso_mod.LassoPathConfig
compute_projected_hinv_vector = _lasso_mod.compute_projected_hinv_vector
fit_weighted_lasso_with_covariates = _lasso_mod.fit_weighted_lasso_with_covariates
pcg_solve = _pcg_mod.pcg_solve
load_component_specs = _component_spec_mod.load_component_specs

_source_mod = importlib.import_module(f"{pkg_name}.geno_source")
PgenGenoSource = _source_mod.PgenGenoSource

# Shared pipeline utilities (eliminates duplication with run_reml_pipeline.py)
env = _common_mod.env
query_gpu = _common_mod.query_gpu
read_keep_ids = _common_mod.read_keep_ids
setup_gpu = _common_mod.setup_gpu
run_planner = _common_mod.run_planner
print_planner_info = _common_mod.print_planner_info
log_runtime_gpu_memory = _common_mod.log_runtime_gpu_memory
ensure_parent_dir = _io_utils_mod.ensure_parent_dir
solve_spd = _common_mod.solve_spd
cleanup_path = _common_mod.cleanup_path
fam_order_mismatch = _common_mod.fam_order_mismatch
make_nonbed_input_fam = _common_mod.make_nonbed_input_fam
compute_sample_mask = _common_mod.compute_sample_mask
write_keep_file = _common_mod.write_keep_file
resolve_cpu_threads = _common_mod.resolve_cpu_threads

LASSO_EBIC_ES_PATIENCE_FIXED = 10


def _bed_count(path: str, attr: str) -> int:
    bed = open_bed(path)
    try:
        return int(getattr(bed, attr))
    finally:
        close = getattr(bed, "close", None)
        if close is not None:
            close()


def _load_component_variant_indices(path: str) -> list[np.ndarray]:
    return [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in load_component_specs(path)
    ]


def _normalized_design_is_well_conditioned(
    design: np.ndarray,
    *,
    relative_tol: float,
) -> bool:
    """Match the normalized-Gram rank criterion used by REML."""
    if design.shape[1] == 0:
        return True
    norms = np.linalg.norm(design, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return False
    normalized = design / norms
    gram = normalized.T @ normalized
    eigvals = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    return bool(
        eigvals[0]
        > float(relative_tol) * max(float(eigvals[-1]), 1.0)
    )


def _merge_independent_fixed_effects(
    covar: np.ndarray | None,
    active_geno: np.ndarray,
    *,
    relative_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a numerically independent subset of active SNP fixed effects.

    LASSO may select perfectly linked SNPs. Their fixed-effect columns span the
    same space, but passing every duplicate to REML makes ``X'V^-1X`` singular.
    Pivoted QR finds a stable spanning subset while always retaining the base
    covariates. The returned indices refer to columns of ``active_geno``.
    """
    active = np.asarray(active_geno, dtype=np.float64)
    if active.ndim != 2:
        raise ValueError("active_geno must be a two-dimensional matrix.")
    n_samples = int(active.shape[0])

    if covar is None:
        base = np.empty((n_samples, 0), dtype=np.float64)
    else:
        base = np.asarray(covar, dtype=np.float64)
        if base.ndim != 2 or int(base.shape[0]) != n_samples:
            raise ValueError("covar and active_geno must have matching rows.")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(active)):
        raise ValueError("Fixed-effect columns must contain only finite values.")
    if not _normalized_design_is_well_conditioned(
        base, relative_tol=relative_tol
    ):
        raise ValueError(
            "covar is rank-deficient or numerically collinear; "
            "remove redundant fixed-effect columns."
        )

    active_norms = np.linalg.norm(active, axis=0)
    eligible = np.flatnonzero(np.isfinite(active_norms) & (active_norms > 0.0))
    if eligible.size == 0:
        return np.asarray(base, dtype=np.float32), np.empty((0,), dtype=np.int64)

    active_normalized = active[:, eligible] / active_norms[eligible]
    if base.shape[1] > 0:
        base_normalized = base / np.linalg.norm(base, axis=0)
        q_base, _ = sla.qr(
            base_normalized,
            mode="economic",
            check_finite=False,
        )
        active_residual = active_normalized - q_base @ (q_base.T @ active_normalized)
    else:
        active_residual = active_normalized

    _, r_active, piv = sla.qr(
        active_residual,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diag = np.abs(np.diag(r_active))
    if diag.size == 0:
        selected_order = np.empty((0,), dtype=np.int64)
    else:
        qr_tol = np.sqrt(float(relative_tol)) * max(float(diag[0]), 1.0)
        rank = int(np.count_nonzero(diag > qr_tol))
        selected_order = eligible[np.asarray(piv[:rank], dtype=np.int64)]

    while selected_order.size > 0:
        trial = np.concatenate([base, active[:, selected_order]], axis=1)
        if _normalized_design_is_well_conditioned(
            trial, relative_tol=relative_tol
        ):
            break
        selected_order = selected_order[:-1]

    selected = np.sort(selected_order)
    merged = np.concatenate([base, active[:, selected]], axis=1)
    return np.asarray(merged, dtype=np.float32), selected


# ---------------------------------------------------------------------------
# Multi-GRM helpers
# ---------------------------------------------------------------------------

class MultiGRMIndex:
    """
    Maps global SNP indices to (grm_index, local_snp_index) pairs.

    With G GRMs having m_0, m_1, … SNPs, the global index space is
    [0, m_0) for GRM 0, [m_0, m_0+m_1) for GRM 1, etc.
    """

    def __init__(self, streamers, call_plan=(), component_variant_indices=None):
        self.streamers = streamers
        self.call_plan = tuple(call_plan)
        self._partitioned_single_streamer = (
            component_variant_indices is not None
            and len(streamers) == 1
        )
        self._source_variant_indices = None
        if self._partitioned_single_streamer:
            groups = [
                np.asarray(group, dtype=np.int64).reshape(-1)
                for group in component_variant_indices
            ]
            self.n_grm = len(groups)
            self.m_per_grm = np.array([group.size for group in groups], dtype=np.int64)
            if groups:
                self._source_variant_indices = np.concatenate(groups, axis=0)
            else:
                self._source_variant_indices = np.empty((0,), dtype=np.int64)
        else:
            self.n_grm = len(streamers)
            self.m_per_grm = np.array([st.m for st in streamers], dtype=np.int64)
        self.offsets = np.zeros(self.n_grm + 1, dtype=np.int64)
        np.cumsum(self.m_per_grm, out=self.offsets[1:])
        self.m_total = int(self.offsets[-1])

    def global_to_local(
        self, global_idx: np.ndarray
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """
        Convert global SNP indices to per-GRM groups.

        Returns list of (grm_idx, local_indices, positions_in_input) tuples,
        where positions_in_input are the positions in the original global_idx
        array so results can be assembled back.
        """
        gidx = np.asarray(global_idx, dtype=np.int64)
        grm_ids = np.searchsorted(self.offsets[1:], gidx, side="right")
        grm_ids = np.clip(grm_ids, 0, self.n_grm - 1)
        groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        for g in range(self.n_grm):
            mask = grm_ids == g
            if not np.any(mask):
                continue
            positions = np.flatnonzero(mask)
            local = gidx[positions] - int(self.offsets[g])
            groups.append((g, local, positions))
        return groups

    def xtv_all(self, u_jax: jnp.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Compute X^T u across all GRMs, returning a global (m_total,) score.
        """
        if self._partitioned_single_streamer:
            return np.asarray(
                self.streamers[0].xtv(u_jax, normalize=normalize),
                dtype=np.float64,
            )
        if self.n_grm > 1 and self.call_plan:
            from .kv_impl import xtv_impl_multi_streamed_concat

            for st in self.streamers:
                st._prepare_kv_pass()
            return np.asarray(
                xtv_impl_multi_streamed_concat(
                    u_jax,
                    self.streamers,
                    self.call_plan,
                    missing_val=int(self.streamers[0]._missing_val),
                    normalize=normalize,
                ),
                dtype=np.float64,
            )

        scores = np.zeros(self.m_total, dtype=np.float64)
        for g, st in enumerate(self.streamers):
            off = int(self.offsets[g])
            block = np.asarray(
                st.xtv(u_jax, normalize=normalize), dtype=np.float64
            )
            scores[off : off + st.m] = block
        return scores

    def extract_standardized_columns(
        self, global_idx: np.ndarray
    ) -> np.ndarray:
        """
        Extract standardized genotype columns for global SNP indices.
        Dispatches to the correct streamer for each GRM and assembles
        columns in the original order.
        """
        gidx = np.asarray(global_idx, dtype=np.int64)
        if self._partitioned_single_streamer:
            return self.streamers[0].extract_standardized_columns(gidx)
        n = self.streamers[0].n
        out = np.empty((n, gidx.size), dtype=np.float32)
        for g, local, positions in self.global_to_local(gidx):
            cols = self.streamers[g].extract_standardized_columns(local)
            out[:, positions] = cols
        return out

    def source_variant_indices(self, global_idx: np.ndarray) -> np.ndarray:
        gidx = np.asarray(global_idx, dtype=np.int64)
        if self._source_variant_indices is None:
            return gidx.copy()
        return np.asarray(self._source_variant_indices[gidx], dtype=np.int64)

    def lookup_bim_rows(
        self, bed_prefixes: list[str], global_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        """
        Look up BIM info for global SNP indices, dispatching to the
        correct .bim file for each GRM.
        """
        result: dict[int, tuple[str, str, str, str, str, str]] = {}
        if self._partitioned_single_streamer:
            source_idx = self.source_variant_indices(global_idx)
            source_rows = _lookup_bim_rows(bed_prefixes[0] + ".bim", source_idx)
            for global_snp, src_idx in zip(global_idx.tolist(), source_idx.tolist()):
                if int(src_idx) in source_rows:
                    result[int(global_snp)] = source_rows[int(src_idx)]
            return result
        for g, local, positions in self.global_to_local(global_idx):
            bim_path = bed_prefixes[g] + ".bim"
            local_rows = _lookup_bim_rows(bim_path, local)
            for pos, loc_idx in zip(positions, local):
                global_snp = int(global_idx[pos])
                if int(loc_idx) in local_rows:
                    result[global_snp] = local_rows[int(loc_idx)]
        return result


def _lookup_bim_rows(bim_path: str, snp_indices: np.ndarray) -> dict[int, tuple[str, str, str, str, str, str]]:
    idx = np.asarray(snp_indices, dtype=np.int64)
    if idx.size == 0:
        return {}
    need = set(int(i) for i in idx.tolist())
    rows: dict[int, tuple[str, str, str, str, str, str]] = {}
    with open(bim_path, "r") as f:
        for i, line in enumerate(f):
            if i not in need:
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            rows[i] = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            if len(rows) == len(need):
                break
    return rows


def _lookup_pvar_rows(
    pvar_path: str, snp_indices: np.ndarray,
) -> dict[int, tuple[str, str, str, str, str, str]]:
    """Look up variant annotation from a .pvar file for given indices."""
    idx = np.asarray(snp_indices, dtype=np.int64)
    if idx.size == 0:
        return {}
    need = set(int(i) for i in idx.tolist())
    rows: dict[int, tuple[str, str, str, str, str, str]] = {}
    data_line = 0
    with open(pvar_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if data_line not in need:
                data_line += 1
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                parts = line.strip().split()
            chrom = parts[0] if len(parts) >= 1 else "NA"
            pos = parts[1] if len(parts) >= 2 else "NA"
            snp_id = parts[2] if len(parts) >= 3 else f"SNP_{data_line}"
            a1 = parts[3] if len(parts) >= 4 else "NA"
            a2 = parts[4] if len(parts) >= 5 else "NA"
            rows[data_line] = (chrom, snp_id, "0", pos, a1, a2)
            if len(rows) == len(need):
                break
            data_line += 1
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sparse REML + LASSO pipeline on real genotype data.")
    # Genotype input — exactly one of the two groups must be supplied
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""),
                   help="PLINK1 BED file prefix (no extension); comma-separated for multiple GRMs")
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""),
                   help="PLINK2 PGEN file prefix (direct read, no conversion needed)")
    p.add_argument(
        "--component-indices-npz",
        default=env("COMPONENT_INDICES_NPZ", ""),
        help="Legacy NPZ file of per-component SNP index arrays for single-file multi-GRM Lasso.",
    )
    p.add_argument(
        "--component-spec",
        default=env("COMPONENT_SPEC", ""),
        help="Structured component spec (.json or .npz) defining SNP-ID/index GRM partitions.",
    )
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument("--keep-path", default=env("KEEP_PATH", ""))
    p.add_argument("--keep-out", default=env("KEEP_OUT", ""))
    p.add_argument("--dropped-out", default=env("DROPPED_OUT", ""))
    p.add_argument("--out-prefix", default=env("OUT_PREFIX", "sparse_reml"))
    p.add_argument("--device", default=env("DEVICE", "gpu"))
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=int(env("CPU_THREADS", "0")),
        help="CPU threads for source/build work (0 = auto-detect).",
    )
    p.add_argument("--call-width", type=int, default=0,
                   help="Call width w (0 = auto from planner)")
    p.add_argument(
        "--gpu-budget-gib",
        "--gpu-budget-gb",
        dest="gpu_budget_gib",
        type=float,
        default=float(env("GPU_BUDGET_GIB", env("GPU_BUDGET_GB", "0"))),
        help=(
            "Planner budget for active GPU allocations in GiB "
            "(`--gpu-budget-gb` is a legacy alias; 0 = use 85%% of current "
            "free memory). JAX allocator reservation shown by nvidia-smi may "
            "be higher."
        ),
    )
    p.add_argument("--ring-depth", type=int, default=int(env("RING_DEPTH", "0")),
                   help="Pinned ring buffer depth (0 = auto, default 32)")
    p.add_argument("--n-rand-vec", type=int, default=100)
    p.add_argument("--slq-samples", type=int, default=100)
    p.add_argument("--slq-m", type=int, default=int(env("SLQ_M", "50")))
    p.add_argument("--precond-type", choices=["projected_core"], default=env("PRECOND_TYPE", "projected_core"))
    p.add_argument("--minq-iter", type=int, default=int(env("MINQ_ITER", "10")))
    p.add_argument("--pcg-tol", type=float, default=float(env("PCG_TOL", "5e-3")))
    p.add_argument("--pcg-ridge", type=float, default=float(env("PCG_RIDGE", "1e-6")))
    p.add_argument("--max-pcg-iters", type=int, default=int(env("MAX_PCG_ITERS", "400")))
    p.add_argument("--outer-max", type=int, default=6)
    p.add_argument("--screen-topk", type=int, default=2000)
    p.add_argument("--candidate-k", type=int, default=256)
    p.add_argument("--max-active", type=int, default=128)
    p.add_argument("--vc-rel-tol", type=float, default=1e-2)
    p.add_argument("--support-stable-rounds", type=int, default=1)
    p.add_argument("--lasso-lam-min-ratio", type=float, default=0.05)
    p.add_argument("--lasso-n-lambda", type=int, default=60)
    p.add_argument("--lasso-ebic-gamma", type=float, default=0.5)
    p.add_argument("--lasso-ebic-early-stop", action="store_true")
    p.add_argument("--no-lasso-ebic-early-stop", dest="lasso_ebic_early_stop", action="store_false")
    p.set_defaults(lasso_ebic_early_stop=True)
    p.add_argument("--lasso-ebic-es-min-delta", type=float, default=0.0)
    p.add_argument("--lasso-cd-max-iter", type=int, default=2000)
    p.add_argument("--lasso-cd-tol", type=float, default=1e-6)
    p.add_argument("--lasso-active-set-period", type=int, default=5)
    p.add_argument("--lasso-ridge", type=float, default=1e-6)
    p.add_argument("--proj-ridge", type=float, default=1e-6)
    p.add_argument("--ebic-p-mode", choices=["candidate", "full"], default="candidate")
    p.add_argument("--kkt-check", action="store_true")
    p.add_argument("--no-kkt-check", dest="kkt_check", action="store_false")
    p.set_defaults(kkt_check=True)
    p.add_argument(
        "--kkt-tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for global inactive-SNP LASSO KKT checks.",
    )
    p.add_argument(
        "--kkt-rel-tol",
        type=float,
        default=1e-4,
        help="Relative tolerance, multiplied by max(1, lambda), for global KKT checks.",
    )
    p.add_argument(
        "--kkt-add-topk",
        type=int,
        default=256,
        help="Maximum number of outside-candidate KKT violators added per refinement round.",
    )
    p.add_argument(
        "--kkt-max-rounds",
        type=int,
        default=20,
        help="Maximum candidate-expansion rounds used to certify global KKT optimality.",
    )
    p.add_argument(
        "--kkt-max-candidate",
        type=int,
        default=0,
        help="Optional hard cap for KKT-expanded candidate set size (0 = no explicit cap).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=env("VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    return p.parse_args()


def _max_rel_change(new_v: np.ndarray, old_v: np.ndarray) -> float:
    denom = np.maximum(np.abs(old_v), 1e-6)
    return float(np.max(np.abs(new_v - old_v) / denom))


def _fixed_point_skip_reml(
    *,
    support_same: bool,
    theta_stable_prev: bool,
    has_reml_refit: bool,
    stable_rounds: int,
    support_stable_rounds: int,
) -> tuple[bool, int]:
    """
    Decide whether a repeated support under already-stable theta can skip REML.

    Returns (stop_now, stable_rounds_next).  If ``stop_now`` is False but
    ``stable_rounds_next`` increased, the caller should continue to the next
    outer iteration without a redundant REML refit.
    """
    if not (support_same and theta_stable_prev and has_reml_refit):
        return False, stable_rounds
    stable_rounds_next = stable_rounds + 1
    return stable_rounds_next >= max(1, int(support_stable_rounds)), stable_rounds_next


def _chive_q_hat_given_active(
    z_active: np.ndarray,
    y: np.ndarray,
    beta_active: np.ndarray,
) -> tuple[float, float, float]:
    """
    CHIVE single-sample estimator on the active SNP set:
        Q = (1/n)||Z_S b_S||^2 + (2/n) b_S^T Z_S^T (y - Z_S b_S)
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta_active, dtype=np.float64).reshape(-1)
    if z_active.size == 0 or beta.size == 0:
        return 0.0, 0.0, 0.0

    Zs = np.asarray(z_active, dtype=np.float64)
    if Zs.ndim != 2 or Zs.shape[0] != y.size or Zs.shape[1] != beta.size:
        raise ValueError("CHIVE input shape mismatch.")

    n = float(y.size)
    g = Zs @ beta
    r = y - g
    term1 = float(g @ g / n)
    term2 = float(2.0 * (beta @ (Zs.T @ r)) / n)
    return term1 + term2, term1, term2


def _gls_refit_on_support(
    y: np.ndarray,
    covar: np.ndarray | None,
    z_active: np.ndarray,
    Hinv_y: np.ndarray,
    Hinv_covar: np.ndarray | None,
    Hinv_z_active: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GLS refit on the final active support under fixed variance components.
    Returns (beta_cov, beta_active).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Zs = np.asarray(z_active, dtype=np.float64)
    Hy = np.asarray(Hinv_y, dtype=np.float64).reshape(-1)
    HZ = np.asarray(Hinv_z_active, dtype=np.float64)

    if Zs.ndim != 2 or HZ.shape != Zs.shape or Zs.shape[0] != y.size:
        raise ValueError("GLS active-set inputs have incompatible shapes.")

    k = Zs.shape[1]
    if covar is None or covar.size == 0:
        if k == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        A = 0.5 * ((Zs.T @ HZ) + (HZ.T @ Zs))
        b = Zs.T @ Hy
        beta_active = np.asarray(solve_spd(A, b, ridge=ridge), dtype=np.float64).reshape(-1)
        return np.empty((0,), dtype=np.float64), beta_active

    C = np.asarray(covar, dtype=np.float64)
    HC = np.asarray(Hinv_covar, dtype=np.float64)
    if C.ndim != 2 or HC.ndim != 2 or C.shape != HC.shape or C.shape[0] != y.size:
        raise ValueError("GLS covariate inputs have incompatible shapes.")

    X = np.concatenate([C, Zs], axis=1) if k > 0 else C
    HX = np.concatenate([HC, HZ], axis=1) if k > 0 else HC
    A = X.T @ HX
    A = 0.5 * (A + A.T)
    b = X.T @ Hy
    beta = np.asarray(solve_spd(A, b, ridge=ridge), dtype=np.float64).reshape(-1)
    p_c = C.shape[1]
    return beta[:p_c], beta[p_c:]


def _lasso_residual(
    y: np.ndarray,
    covar: np.ndarray | None,
    geno: np.ndarray,
    beta_cov: np.ndarray,
    beta_snp: np.ndarray,
) -> np.ndarray:
    """Residual for the current weighted LASSO solution."""
    resid = np.asarray(y, dtype=np.float64).reshape(-1).copy()
    if covar is not None and covar.size > 0 and beta_cov.size > 0:
        resid -= np.asarray(covar, dtype=np.float64) @ np.asarray(beta_cov, dtype=np.float64).reshape(-1)
    if geno.size > 0 and beta_snp.size > 0:
        resid -= np.asarray(geno, dtype=np.float64) @ np.asarray(beta_snp, dtype=np.float64).reshape(-1)
    return resid


def _outside_kkt_violators(
    *,
    score_abs: np.ndarray,
    candidate: np.ndarray,
    lam: float,
    abs_tol: float,
    rel_tol: float,
) -> tuple[np.ndarray, float, float]:
    """
    Return outside-candidate SNPs violating inactive LASSO KKT conditions.

    The inactive full-genome condition is |x_j^T H^{-1} r| <= lambda for
    every SNP not included in the candidate LASSO system.
    """
    scores = np.asarray(score_abs, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.int64).reshape(-1)
    lam_f = float(lam)
    tol = max(float(abs_tol), float(rel_tol) * max(1.0, abs(lam_f)))
    threshold = lam_f + tol

    outside = np.ones(scores.size, dtype=bool)
    if cand.size > 0:
        outside[cand] = False
    outside_scores = scores[outside]
    max_outside = float(np.max(outside_scores)) if outside_scores.size > 0 else 0.0
    viol_mask = outside & (scores > threshold)
    violators = np.flatnonzero(viol_mask).astype(np.int64)
    return violators, max_outside, threshold


def main() -> None:
    args = parse_args()
    if int(args.candidate_k) < int(args.max_active):
        raise SystemExit("candidate-k must be >= max-active.")
    if int(args.screen_topk) < int(args.candidate_k):
        raise SystemExit("screen-topk must be >= candidate-k.")
    if int(args.outer_max) < 1:
        raise SystemExit("outer-max must be >= 1.")
    if int(args.kkt_max_rounds) < 1:
        raise SystemExit("kkt-max-rounds must be >= 1.")
    if int(args.kkt_add_topk) < 1:
        raise SystemExit("kkt-add-topk must be >= 1.")
    if float(args.kkt_tol) < 0.0 or float(args.kkt_rel_tol) < 0.0:
        raise SystemExit("kkt tolerances must be nonnegative.")

    logger.info("[INFO] sparse pipeline start @ %s", datetime.now().isoformat(timespec='seconds'))
    t0 = time.time()

    bed_list = [b.strip() for b in args.bed_prefix.split(",") if b.strip()]
    pgen_prefix = args.pgen_prefix.strip()
    component_spec_path = args.component_spec.strip()
    legacy_component_npz = args.component_indices_npz.strip()
    if component_spec_path and legacy_component_npz:
        raise SystemExit("Use only one of --component-spec or --component-indices-npz.")
    component_spec_source = component_spec_path or legacy_component_npz
    component_variant_indices = (
        _load_component_variant_indices(component_spec_source)
        if component_spec_source
        else []
    )

    _n_formats = sum(bool(x) for x in [bed_list, pgen_prefix])
    if _n_formats == 0:
        raise SystemExit(
            "No genotype input specified. "
            "Use --bed-prefix or --pgen-prefix."
        )
    if _n_formats > 1:
        raise SystemExit(
            "Specify only one of --bed-prefix / --pgen-prefix."
        )
    if component_variant_indices:
        if len(bed_list) > 1:
            raise SystemExit("single-source component partitioning cannot be combined with multiple BED prefixes.")
        if not (len(bed_list) == 1 or pgen_prefix):
            raise SystemExit("single-source component partitioning requires exactly one genotype input.")
    if not args.pheno_txt:
        raise SystemExit("--pheno-txt is required.")

    temp_paths: list[str] = []

    # ---- Sample alignment (BED or PGEN FAM) ----------------
    if pgen_prefix:
        fam_path = make_nonbed_input_fam(pgen_prefix=pgen_prefix)
        temp_paths.append(fam_path)
    else:
        fam_path = bed_list[0] + ".fam"

    for path in temp_paths:
        atexit.register(cleanup_path, path)

    keep_ids = None
    if args.keep_path and os.path.exists(args.keep_path):
        keep_ids = read_keep_ids(args.keep_path)

    # Use first GRM's FAM as the reference for sample alignment.
    y_np, covar_np, fam_keep, dropped = load_pheno_covar_aligned(
        fam_path=fam_path,
        pheno_path=args.pheno_txt,
        covar_path=args.covar_txt or None,
        add_intercept=True,
        keep_ids=keep_ids,
    )
    y_np = y_np.astype(np.float32, copy=False)
    if covar_np is not None:
        covar_np = covar_np.astype(np.float32, copy=False)

    logger.info("Loaded %s samples; dropped %s", y_np.shape[0], len(dropped))

    # ---- PGEN direct read or BED sample-mask ---------------------------------
    sources = None
    sample_mask = None
    if pgen_prefix:
        sample_mask = compute_sample_mask(fam_path, fam_keep)
        sources = [PgenGenoSource(pgen_prefix, sample_mask=sample_mask)]
        logger.info("[INFO] Direct read: PgenGenoSource "
              "n_source=%s m=%s "
              "n_keep=%s", sources[0]._n_full, sources[0].m, sources[0].n)
        sample_mask = None  # handled inside source
    else:
        n_bed = _bed_count(bed_list[0] + ".bed", "iid_count")
        if n_bed != len(fam_keep):
            sample_mask = compute_sample_mask(fam_path, fam_keep)
            logger.info(
                "[INFO] BED path: using sample_mask "
                "(n_bed=%s -> n_keep=%s) "
                "instead of writing a subset BED", n_bed, len(fam_keep)
            )

    if dropped:
        logger.info("Filtered out %s samples missing pheno/covar.", len(dropped))
    logger.info("Using %s samples after alignment.", len(fam_keep))

    out_prefix = args.out_prefix.strip() or "sparse_reml"
    ensure_parent_dir(out_prefix)
    if args.keep_out:
        ensure_parent_dir(args.keep_out)
        with open(args.keep_out, "w") as f:
            for iid in fam_keep:
                f.write(f"{iid} {iid}\n")
        logger.info("Wrote aligned keep list to %s", args.keep_out)
    if args.dropped_out and dropped:
        ensure_parent_dir(args.dropped_out)
        with open(args.dropped_out, "w") as f:
            for iid in dropped:
                f.write(f"{iid}\n")
        logger.info("Wrote dropped IDs to %s", args.dropped_out)

    gpu_name, gpu_total, gpu_free = setup_gpu()
    n_covar = int(covar_np.shape[1]) if covar_np is not None else 0
    cpu_threads, cpu_threads_src = resolve_cpu_threads(args.cpu_threads or None)
    if sources is not None:
        p_list = [src.m for src in sources]
    else:
        p_list = [_bed_count(pref + ".bed", "sid_count") for pref in bed_list]
    plan = run_planner(
        n_samples=y_np.shape[0], p_list=p_list,
        n_grm=(
            len(component_variant_indices)
            if component_variant_indices
            else len(p_list)
        ),
        component_block_sizes=(
            [int(len(group)) for group in component_variant_indices]
            if component_variant_indices
            else None
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
        requested_call_width=(args.call_width if args.call_width > 0 else None),
    )
    call_width = plan.call_width
    gpu_budget_bytes = (
        float(args.gpu_budget_gib) * 1024**3
        if args.gpu_budget_gib > 0
        else float(plan.gpu_budget_gib) * 1024**3
    )
    print_planner_info(
        plan, gpu_name, gpu_free, call_width,
    )
    logger.info(
        "GPU params: gpu=%s, free=%s GiB, "
        "call_width=%s, "
        "gpu_budget_gib=%s, "
        "n_rand_vec=%s, precond_rank=%s, "
        "slq_samples=%s, slq_m=%s, "
        "minq_iter=%s, pcg_tol=%s, "
        "max_pcg_iters=%s",
        gpu_name, gpu_free/1024**3 if gpu_free else 'unk',
        call_width,
        args.gpu_budget_gib if args.gpu_budget_gib > 0 else 'auto',
        args.n_rand_vec, plan.precond_rank,
        args.slq_samples, args.slq_m,
        args.minq_iter, args.pcg_tol,
        args.max_pcg_iters,
    )
    logger.info("[INFO] cpu_threads=%s (source=%s)", cpu_threads, cpu_threads_src)
    logger.info("jax devices: %s", jax.devices())
    if component_variant_indices:
        logger.info(
            "[INFO] single-source SNP-ID component partition enabled: "
            "component_spec=%s n_components=%s block_sizes=%s",
            component_spec_source,
            len(component_variant_indices),
            [int(len(group)) for group in component_variant_indices],
        )

    if sources is not None:
        fit_cfg = FitConfig(
            sources=sources, sample_mask=sample_mask, device=args.device,
            component_variant_indices=component_variant_indices or None,
            call_width=call_width,
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter,
            slq_samples=args.slq_samples, slq_m=args.slq_m,
            precond_type=args.precond_type, precond_rank=plan.precond_rank,
            max_pcg_iters=args.max_pcg_iters, pcg_ridge=args.pcg_ridge,
            verbose=args.verbose,
        )
    else:
        fit_cfg = FitConfig(
            bed_prefix=bed_list, device=args.device,
            sample_mask=sample_mask,
            component_variant_indices=component_variant_indices or None,
            call_width=call_width,
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter,
            slq_samples=args.slq_samples, slq_m=args.slq_m,
            precond_type=args.precond_type, precond_rank=plan.precond_rank,
            max_pcg_iters=args.max_pcg_iters, pcg_ridge=args.pcg_ridge,
            verbose=args.verbose,
        )
    logger.info(
        "[INFO] streamer config: "
        "call_width=%s keep_host_stats=%s",
        call_width, fit_cfg.keep_host_stats,
    )

    logger.info("[INFO] build fitter @ %s", datetime.now().isoformat(timespec='seconds'))
    fitter = InfinitesimalREMLFitter(fit_cfg)
    close_fitter = fitter.close
    atexit.register(close_fitter)
    ops = fitter._assemble_reml_operators()
    grm_index = MultiGRMIndex(
        fitter.streamers,
        call_plan=fitter._multi_call_plan,
        component_variant_indices=component_variant_indices or None,
    )
    logger.info(
        "[INFO] multi-GRM: n_grm=%s "
        "m_per_grm=%s m_total=%s",
        grm_index.n_grm, grm_index.m_per_grm.tolist(), grm_index.m_total,
    )

    y_jax = jnp.asarray(y_np, dtype=jnp.float32)
    n_grm = len(ops.K_mvs)
    genetic_trace_atoms = np.asarray(
        jax.device_get(fitter._projected_core_diag_atoms(ops.diag_list)),
        dtype=np.float64,
    )
    if (
        genetic_trace_atoms.shape != (n_grm,)
        or not np.all(np.isfinite(genetic_trace_atoms))
        or not np.all(genetic_trace_atoms >= 0.0)
    ):
        raise RuntimeError("Invalid genetic trace atoms for sparse REML initialization.")

    def _trace_weighted_h2(theta_values: np.ndarray) -> float:
        theta_arr = np.asarray(theta_values, dtype=np.float64).reshape(-1)
        genetic_var = float(np.dot(theta_arr[:n_grm], genetic_trace_atoms))
        residual_var = float(theta_arr[n_grm])
        return genetic_var / max(genetic_var + residual_var, 1e-8)

    def _trace_weighted_genetic_var(theta_values: np.ndarray) -> float:
        theta_arr = np.asarray(theta_values, dtype=np.float64).reshape(-1)
        return float(np.dot(theta_arr[:n_grm], genetic_trace_atoms))

    # Match fit_reml's trace-calibrated default initialization.
    h2_init_default = 0.5
    trace_sum = float(np.sum(genetic_trace_atoms))
    if trace_sum <= 0.0:
        raise RuntimeError("Sparse REML requires at least one positive-trace GRM.")
    theta_g0 = np.where(
        genetic_trace_atoms > 0.0,
        h2_init_default / trace_sum,
        0.0,
    )
    theta_e0 = np.array([1.0 - h2_init_default], dtype=np.float64)
    theta = np.concatenate([theta_g0, theta_e0], axis=0)
    fitter._ensure_projected_core_precond_ready(
        ops,
        var_components_init=jnp.asarray(theta, dtype=jnp.float32),
    )
    logger.info(
        "[INFO] init theta (fit_reml default) @ %s: "
        "%s",
        datetime.now().isoformat(timespec='seconds'), theta.tolist(),
    )

    path_cfg = LassoPathConfig(
        lam_min_ratio=args.lasso_lam_min_ratio, n_lambda=args.lasso_n_lambda,
        ebic_gamma=args.lasso_ebic_gamma, max_cd_iter=args.lasso_cd_max_iter,
        ebic_early_stop=args.lasso_ebic_early_stop,
        ebic_early_stop_patience=LASSO_EBIC_ES_PATIENCE_FIXED,
        ebic_early_stop_min_delta=args.lasso_ebic_es_min_delta,
        cd_tol=args.lasso_cd_tol, active_set_period=args.lasso_active_set_period,
        verbose=args.verbose,
    )

    support = np.array([], dtype=np.int64)
    stable_rounds = 0
    history: list[dict] = []
    theta_stable_prev = False

    warm_screen = None
    warm_z_dict: dict[int, np.ndarray] = {}

    final_candidate = np.array([], dtype=np.int64)
    final_lasso = None
    theta_lasso = theta.copy()
    n_samples = y_np.shape[0]
    has_reml_refit = False

    # ---- Precompute loop-invariant B_screen = [y | covar] on device --------
    screen_parts = [y_np[:, None]]
    if covar_np is not None:
        screen_parts.append(covar_np)
    B_screen_np = np.concatenate(screen_parts, axis=1).astype(np.float32, copy=False)
    B_screen_dev = jnp.asarray(B_screen_np, dtype=jnp.float32)
    n_screen = B_screen_np.shape[1]

    for outer in range(1, int(args.outer_max) + 1):
        iter_t0 = time.time()
        theta_g = jnp.asarray(theta[:-1], dtype=jnp.float32)
        theta_e = jnp.asarray(theta[-1], dtype=jnp.float32)
        hv = fitter._make_hv(ops, theta_g, theta_e)
        precond = fitter._make_effect_precond(ops, theta_g, theta_e)

        # ---- Step 1: screen PCG (y + covar) ----
        x0_screen = warm_screen if (
            warm_screen is not None and warm_screen.shape == (n_samples, n_screen)
        ) else None
        sol_screen, res_screen, it_screen = pcg_solve(
            hv, B_screen_dev,
            M=precond, tol=args.pcg_tol, maxiter=args.max_pcg_iters,
            X0=x0_screen,
        )
        warm_screen = sol_screen

        # ---- Step 2: xtv screening score ----
        Hinv_y_np = np.asarray(sol_screen[:, 0], dtype=np.float64)
        Hinv_covar_np = None
        if covar_np is not None and covar_np.shape[1] > 0:
            Hinv_covar_np = np.asarray(sol_screen[:, 1:n_screen], dtype=np.float64)

        u = compute_projected_hinv_vector(
            covar=covar_np, Hinv_covar=Hinv_covar_np,
            Hinv_target=Hinv_y_np, ridge=args.proj_ridge,
        )

        score = np.abs(grm_index.xtv_all(
            jnp.asarray(u, dtype=jnp.float32), normalize=False,
        ))
        topk = min(int(args.screen_topk), score.size)
        if topk <= 0:
            raise RuntimeError("screen_topk must be > 0.")

        top_idx_unsorted = np.argpartition(score, -topk)[-topk:]
        top_idx = top_idx_unsorted[np.argsort(score[top_idx_unsorted])[::-1]]

        candidate_target = min(int(args.candidate_k), score.size)
        if candidate_target <= 0:
            raise RuntimeError("candidate_k must be > 0.")

        candidate_seed = top_idx[:candidate_target]

        # Build candidate: prioritize previous support, then new screened SNPs
        keep_list = []
        keep_set_cand = set()
        for snp in support.tolist():
            snp_i = int(snp)
            if snp_i not in keep_set_cand:
                keep_list.append(snp_i)
                keep_set_cand.add(snp_i)
                if len(keep_list) >= candidate_target:
                    break
        if len(keep_list) < candidate_target:
            for snp in candidate_seed.tolist():
                snp_i = int(snp)
                if snp_i not in keep_set_cand:
                    keep_list.append(snp_i)
                    keep_set_cand.add(snp_i)
                    if len(keep_list) >= candidate_target:
                        break

        candidate = np.asarray(sorted(keep_list), dtype=np.int64)

        # ---- Step 3/4: candidate LASSO with global KKT refinement ----------
        kkt_trace: list[dict] = []
        Z_cand = np.empty((n_samples, 0), dtype=np.float32)
        sol_z_np = np.empty((n_samples, 0), dtype=np.float32)
        res_all = np.asarray(0.0, dtype=np.float32)
        it_all = 0
        lasso = None
        active_local = np.array([], dtype=np.int64)
        support_new = np.array([], dtype=np.int64)
        certified_kkt = False

        max_kkt_rounds = int(args.kkt_max_rounds) if bool(args.kkt_check) else 1
        for kkt_round in range(1, max_kkt_rounds + 1):
            # Z_cand PCG with dictionary warm-start.  Candidate may grow after
            # global KKT scans, so this solve is intentionally inside the loop.
            Z_cand = grm_index.extract_standardized_columns(candidate).astype(np.float32, copy=False)
            B_z = jnp.asarray(Z_cand, dtype=jnp.float32)

            x0_z = None
            if warm_z_dict:
                x0_arr = np.zeros((n_samples, candidate.size), dtype=np.float32)
                hit = 0
                for j, snp_idx in enumerate(candidate.tolist()):
                    snp_i = int(snp_idx)
                    if snp_i in warm_z_dict:
                        x0_arr[:, j] = warm_z_dict[snp_i]
                        hit += 1
                if hit > 0:
                    x0_z = jnp.asarray(x0_arr, dtype=jnp.float32)
                    if args.verbose:
                        logger.info(
                            "[outer %s kkt %s] Z warm-start: %s/%s columns reused",
                            outer, kkt_round, hit, candidate.size,
                        )

            sol_z, res_all, it_all = pcg_solve(
                hv, B_z, M=precond, tol=args.pcg_tol,
                maxiter=args.max_pcg_iters, X0=x0_z,
            )

            sol_z_np = np.asarray(sol_z, dtype=np.float32)
            warm_z_dict = {
                int(snp_idx): sol_z_np[:, j]
                for j, snp_idx in enumerate(candidate.tolist())
            }

            p_for_ebic = int(candidate.size) if args.ebic_p_mode == "candidate" else grm_index.m_total
            lasso = fit_weighted_lasso_with_covariates(
                y=y_np, covar=covar_np, geno=Z_cand,
                Hinv_y=Hinv_y_np, Hinv_covar=Hinv_covar_np,
                Hinv_geno=sol_z_np, p_total=p_for_ebic,
                cfg=path_cfg, ridge=args.lasso_ridge,
            )
            theta_lasso = theta.copy()

            best_path = min(
                lasso["path"],
                key=lambda row: abs(float(row["lam"]) - float(lasso["lam"])),
            )
            if not bool(best_path.get("converged", False)):
                raise RuntimeError(
                    "Selected LASSO solution did not converge; KKT optimality cannot be certified. "
                    "Increase --lasso-cd-max-iter or loosen --lasso-cd-tol."
                )

            active_local = np.asarray(lasso["active_idx"], dtype=np.int64)
            if active_local.size > int(args.max_active):
                raise RuntimeError(
                    "LASSO selected more active SNPs than max-active "
                    f"({active_local.size} > {int(args.max_active)}). "
                    "Refusing to truncate coefficients because that would violate KKT optimality; "
                    "increase --max-active or use stronger LASSO/EBIC settings."
                )
            support_new = np.sort(candidate[active_local])

            if not bool(args.kkt_check):
                certified_kkt = False
                kkt_trace.append({
                    "round": int(kkt_round),
                    "candidate_size": int(candidate.size),
                    "support_size": int(support_new.size),
                    "checked": False,
                })
                break

            beta_cov = np.asarray(lasso.get("beta_cov", np.empty((0,))), dtype=np.float64)
            beta_snp = np.asarray(lasso["beta_snp"], dtype=np.float64)
            resid_lasso = _lasso_residual(
                y=y_np, covar=covar_np, geno=Z_cand,
                beta_cov=beta_cov, beta_snp=beta_snp,
            )
            sol_resid, res_kkt, it_kkt = pcg_solve(
                hv,
                jnp.asarray(resid_lasso[:, None], dtype=jnp.float32),
                M=precond,
                tol=args.pcg_tol,
                maxiter=args.max_pcg_iters,
            )
            score_kkt = np.abs(grm_index.xtv_all(sol_resid[:, 0], normalize=False))
            violators, max_outside_score, kkt_threshold = _outside_kkt_violators(
                score_abs=score_kkt,
                candidate=candidate,
                lam=float(lasso["lam"]),
                abs_tol=float(args.kkt_tol),
                rel_tol=float(args.kkt_rel_tol),
            )

            n_viol = int(violators.size)
            kkt_trace.append({
                "round": int(kkt_round),
                "candidate_size": int(candidate.size),
                "support_size": int(support_new.size),
                "lambda": float(lasso["lam"]),
                "threshold": float(kkt_threshold),
                "max_outside_score": float(max_outside_score),
                "n_violators": n_viol,
                "pcg_kkt_iters": int(it_kkt),
                "pcg_kkt_res": float(np.asarray(res_kkt)),
                "checked": True,
            })
            logger.info(
                "[outer %s kkt %s] cand=%s active=%s lam=%.3e "
                "max_outside=%.3e threshold=%.3e violators=%s",
                outer, kkt_round, int(candidate.size), int(support_new.size),
                float(lasso["lam"]), max_outside_score, kkt_threshold, n_viol,
            )

            if n_viol == 0:
                certified_kkt = True
                break

            n_add = min(int(args.kkt_add_topk), n_viol)
            add_idx = violators[np.argsort(score_kkt[violators])[-n_add:]]
            candidate = np.unique(np.concatenate([candidate, add_idx])).astype(np.int64)
            candidate.sort()

            max_candidate = int(args.kkt_max_candidate)
            if max_candidate > 0 and candidate.size > max_candidate:
                raise RuntimeError(
                    "KKT refinement exceeded --kkt-max-candidate "
                    f"({candidate.size} > {max_candidate})."
                )

        if bool(args.kkt_check) and not certified_kkt:
            raise RuntimeError(
                "Failed to certify global LASSO KKT optimality within "
                f"{int(args.kkt_max_rounds)} refinement rounds. "
                "Increase --kkt-max-rounds/--kkt-add-topk or inspect the KKT trace."
            )

        if lasso is None:
            raise RuntimeError("Internal error: LASSO refinement loop did not run.")

        support_same = bool(np.array_equal(support_new, support))

        if args.verbose:
            logger.info(
                "[outer %s] lasso_select: p_mode=%s "
                "candidate=%s k_selected=%s kkt_certified=%s",
                outer, args.ebic_p_mode, int(candidate.size), int(active_local.size),
                bool(certified_kkt),
            )

        # If no SNP is selected and support is already empty, covariates-only
        # REML is redundant only after at least one REML refit has completed.
        if active_local.size == 0 and support.size == 0 and has_reml_refit:
            history.append({
                "outer": outer,
                "pcg_screen_iters": int(it_screen),
                "pcg_screen_res": float(np.asarray(res_screen)),
                "pcg_all_iters": int(it_all),
                "pcg_all_res": float(np.asarray(res_all)),
                "theta": theta.tolist(),
                "support_size": 0,
                "support_same": True,
                "vc_rel": 0.0,
                "lam": float(lasso["lam"]),
                "best_ebic": float(lasso["best_ebic"]),
                "kkt_certified": bool(certified_kkt),
                "kkt_trace": kkt_trace,
                "early_stop_no_snp": True,
            })
            support = support_new
            final_candidate = candidate
            final_lasso = lasso
            logger.info(
                "[INFO] stop at outer=%s: k_selected=0 and support already empty; "
                "skip redundant REML refit.", outer,
            )
            break

        # If theta was already stable from the previous outer iteration and
        # this round's screening/LASSO reproduces the same support, then this
        # support is a fixed point under the current theta and another REML
        # refit would be redundant.
        stop_now, stable_rounds_next = _fixed_point_skip_reml(
            support_same=support_same,
            theta_stable_prev=theta_stable_prev,
            has_reml_refit=has_reml_refit,
            stable_rounds=stable_rounds,
            support_stable_rounds=int(args.support_stable_rounds),
        )
        if stable_rounds_next != stable_rounds:
            stable_rounds = stable_rounds_next
            history.append({
                "outer": outer,
                "pcg_screen_iters": int(it_screen),
                "pcg_screen_res": float(np.asarray(res_screen)),
                "pcg_all_iters": int(it_all),
                "pcg_all_res": float(np.asarray(res_all)),
                "theta": theta.tolist(),
                "support_size": int(support_new.size),
                "support_same": True,
                "vc_rel": 0.0,
                "lam": float(lasso["lam"]),
                "best_ebic": float(lasso["best_ebic"]),
                "kkt_certified": bool(certified_kkt),
                "kkt_trace": kkt_trace,
                "early_stop_fixed_point": True,
            })
            support = support_new
            final_candidate = candidate
            final_lasso = lasso
            theta_stable_prev = True
            if stop_now:
                logger.info(
                    "[INFO] stop at outer=%s: support repeated under stable theta; "
                    "skip redundant REML refit.", outer,
                )
                break
            if args.verbose:
                logger.info(
                    "[outer %s] fixed-point support repeated under stable theta; "
                    "stable_rounds=%s/%s "
                    "skip REML refit and continue.",
                    outer, stable_rounds, int(args.support_stable_rounds),
                )
            continue

        # ---- Step 5: REML re-fit with active SNPs as fixed effects ----
        X_fixed = covar_np
        active_fixed_local = np.empty((0,), dtype=np.int64)
        if active_local.size > 0:
            Z_active = Z_cand[:, active_local]
            X_fixed, active_fixed_local = _merge_independent_fixed_effects(
                X_fixed,
                Z_active,
            )
            if active_fixed_local.size != active_local.size:
                logger.info(
                    "[outer %s] REML fixed effects retained %s/%s active SNP "
                    "columns after removing numerical dependencies.",
                    outer,
                    int(active_fixed_local.size),
                    int(active_local.size),
                )

        reml_res = fitter.fit_infinitesimal(
            y_jax,
            jnp.asarray(X_fixed, dtype=jnp.float32) if X_fixed is not None else None,
            h2_init=_trace_weighted_h2(theta),
            var_components_init=jnp.asarray(theta, dtype=jnp.float32),
        )
        if args.verbose:
            logger.info("[outer %s] reml_init_theta=%s", outer, theta.tolist())
        theta_new = np.asarray(reml_res.var_components, dtype=np.float64)
        has_reml_refit = True

        # ---- Convergence checks ----
        vc_rel = _max_rel_change(theta_new, theta)
        theta_stable_now = bool(vc_rel < float(args.vc_rel_tol))
        if support_same and vc_rel < float(args.vc_rel_tol):
            stable_rounds += 1
        else:
            stable_rounds = 0

        history.append({
            "outer": outer,
            "pcg_screen_iters": int(it_screen),
            "pcg_screen_res": float(np.asarray(res_screen)),
            "pcg_all_iters": int(it_all),
            "pcg_all_res": float(np.asarray(res_all)),
            "theta": theta_new.tolist(),
            "support_size": int(support_new.size),
            "active_fixed_effect_size": int(active_fixed_local.size),
            "support_same": support_same,
            "vc_rel": float(vc_rel),
            "lam": float(lasso["lam"]),
            "best_ebic": float(lasso["best_ebic"]),
            "kkt_certified": bool(certified_kkt),
            "kkt_trace": kkt_trace,
        })

        logger.info(
            "[outer %s] pcg_screen=%s pcg_all=%s "
            "cand=%s active=%s kkt_rounds=%s certified=%s "
            "lam=%.3e ebic=%.4e "
            "vc_rel=%.3e support_same=%s "
            "iter_time=%.1fs",
            outer, int(it_screen), int(it_all),
            int(candidate.size), int(support_new.size), len(kkt_trace), bool(certified_kkt),
            float(lasso['lam']), float(lasso['best_ebic']),
            vc_rel, support_same,
            time.time() - iter_t0,
        )

        theta = theta_new
        support = support_new
        theta_stable_prev = theta_stable_now
        final_candidate = candidate
        final_lasso = lasso

        if stable_rounds >= int(args.support_stable_rounds):
            logger.info("[INFO] stop at outer=%s: support+variance stabilized.", outer)
            break

    # ---- Output results ----
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    h2_reml = _trace_weighted_h2(theta)
    h2_chive = h2_reml
    h2_chive_reml = h2_reml
    q_chive = 0.0
    q_chive_reml = 0.0
    q_chive_term1 = 0.0
    q_chive_term2 = 0.0
    q_chive_reml_term1 = 0.0
    q_chive_reml_term2 = 0.0
    beta_cov_lasso = np.empty((0,), dtype=np.float64)
    beta_cov_gls = np.empty((0,), dtype=np.float64)
    beta_gls_active = np.empty((0,), dtype=np.float64)

    if final_lasso is not None and support.size > 0:
        Z_support = grm_index.extract_standardized_columns(support).astype(np.float32, copy=False)
        beta_snp_final = np.asarray(final_lasso["beta_snp"], dtype=np.float64)
        beta_cov_lasso = np.asarray(final_lasso.get("beta_cov", np.empty((0,))), dtype=np.float64).reshape(-1)
        cand_pos = {int(snp): i for i, snp in enumerate(final_candidate.tolist())}
        beta_lasso_active = np.asarray(
            [beta_snp_final[cand_pos[int(snp)]] for snp in support.tolist()],
            dtype=np.float64,
        )
        y_chive = np.asarray(y_np, dtype=np.float64)
        if covar_np is not None and covar_np.size > 0 and beta_cov_lasso.size > 0:
            y_chive = y_chive - np.asarray(covar_np, dtype=np.float64) @ beta_cov_lasso
        q_chive, q_chive_term1, q_chive_term2 = _chive_q_hat_given_active(
            Z_support,
            y_chive,
            beta_lasso_active,
        )
        theta_lasso_sum = _trace_weighted_genetic_var(theta_lasso)
        theta_lasso_e = float(theta_lasso[-1])
        h2_chive = float(
            (q_chive + theta_lasso_sum) /
            max(q_chive + theta_lasso_sum + theta_lasso_e, 1e-8)
        )

        theta_g = jnp.asarray(theta[:-1], dtype=jnp.float32)
        theta_e = jnp.asarray(theta[-1], dtype=jnp.float32)
        hv_final = fitter._make_hv(ops, theta_g, theta_e)
        precond_final = fitter._make_effect_precond(ops, theta_g, theta_e)

        solve_parts = [y_np[:, None]]
        n_covar = 0
        if covar_np is not None:
            solve_parts.append(covar_np)
            n_covar = int(covar_np.shape[1])
        solve_parts.append(Z_support)
        B_final = np.concatenate(solve_parts, axis=1).astype(np.float32, copy=False)
        sol_final, _, _ = pcg_solve(
            hv_final,
            jnp.asarray(B_final, dtype=jnp.float32),
            M=precond_final,
            tol=args.pcg_tol,
            maxiter=args.max_pcg_iters,
        )
        sol_final_np = np.asarray(sol_final, dtype=np.float64)
        Hinv_y_final = sol_final_np[:, 0]
        Hinv_covar_final = None
        if n_covar > 0:
            Hinv_covar_final = sol_final_np[:, 1 : 1 + n_covar]
        Hinv_Z_support = sol_final_np[:, 1 + n_covar :]

        beta_cov_gls, beta_gls_active = _gls_refit_on_support(
            y=y_np,
            covar=covar_np,
            z_active=Z_support,
            Hinv_y=Hinv_y_final,
            Hinv_covar=Hinv_covar_final,
            Hinv_z_active=Hinv_Z_support,
            ridge=args.proj_ridge,
        )
        y_chive_reml = np.asarray(y_np, dtype=np.float64)
        if covar_np is not None and covar_np.size > 0 and beta_cov_gls.size > 0:
            y_chive_reml = y_chive_reml - np.asarray(covar_np, dtype=np.float64) @ beta_cov_gls
        q_chive_reml, q_chive_reml_term1, q_chive_reml_term2 = _chive_q_hat_given_active(
            Z_support,
            y_chive_reml,
            beta_gls_active,
        )
        theta_sum = _trace_weighted_genetic_var(theta)
        theta_e_final = float(theta[-1])
        h2_chive_reml = float(
            (q_chive_reml + theta_sum) /
            max(q_chive_reml + theta_sum + theta_e_final, 1e-8)
        )

    h2 = h2_chive_reml
    print(f"[RESULT] var_components={theta.tolist()}")
    print(f"[RESULT] h2_reml={h2_reml:.6f}")
    print(f"[RESULT] h2_chive={h2_chive:.6f}")
    print(f"[RESULT] h2_chive_reml={h2_chive_reml:.6f}")
    print(f"[RESULT] support_size={int(support.size)}")

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.time() - t0),
        "n_samples": int(y_np.shape[0]),
        "n_covar": int(covar_np.shape[1]) if covar_np is not None else 0,
        "n_snps_total": grm_index.m_total,
        "n_grms": grm_index.n_grm,
        "m_per_grm": grm_index.m_per_grm.tolist(),
        "genetic_trace_atoms": genetic_trace_atoms.tolist(),
        "component_spec": component_spec_source or None,
        "component_partition_mode": (
            "snp_id" if component_variant_indices else "input_prefix"
        ),
        "var_components": theta.tolist(),
        "h2_reml": h2_reml,
        "h2_chive": h2_chive,
        "h2_chive_reml": h2_chive_reml,
        "h2": h2,
        "q_chive": q_chive,
        "q_chive_reml": q_chive_reml,
        "q_chive_components": {
            "term1_g2_over_n": q_chive_term1,
            "term2_cross": q_chive_term2,
        },
        "q_chive_reml_components": {
            "term1_g2_over_n": q_chive_reml_term1,
            "term2_cross": q_chive_reml_term2,
        },
        "support_size": int(support.size),
        "support_indices": support.tolist(),
        "support_source_indices": grm_index.source_variant_indices(support).tolist(),
        "kkt_check_enabled": bool(args.kkt_check),
        "kkt_certified": bool(
            args.kkt_check and history and bool(history[-1].get("kkt_certified", False))
        ),
        "outer_history": history,
    }

    with open(out_prefix + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_prefix + ".history.json", "w") as f:
        json.dump(history, f, indent=2)

    beta_map: dict[int, float] = {}
    beta_reml_map: dict[int, float] = {}
    if final_lasso is not None and final_candidate.size > 0:
        beta_snp = np.asarray(final_lasso["beta_snp"], dtype=np.float64)
        for snp_idx, beta_val in zip(final_candidate.tolist(), beta_snp.tolist()):
            if beta_val != 0.0:
                beta_map[int(snp_idx)] = float(beta_val)
    if support.size > 0 and beta_gls_active.size == support.size:
        for snp_idx, beta_val in zip(support.tolist(), beta_gls_active.tolist()):
            beta_reml_map[int(snp_idx)] = float(beta_val)

    if sources is None:
        bim_rows = grm_index.lookup_bim_rows(bed_list, support)
    elif support.size > 0 and pgen_prefix:
        source_support = grm_index.source_variant_indices(support)
        source_rows = _lookup_pvar_rows(pgen_prefix + ".pvar", source_support)
        bim_rows = {}
        for global_snp, source_snp in zip(support.tolist(), source_support.tolist()):
            if int(source_snp) in source_rows:
                bim_rows[int(global_snp)] = source_rows[int(source_snp)]
    else:
        bim_rows = {}
    source_index_map = {
        int(global_snp): int(source_snp)
        for global_snp, source_snp in zip(
            support.tolist(), grm_index.source_variant_indices(support).tolist()
        )
    }
    # Build global → grm_index map for output
    _snp_grm_map: dict[int, int] = {}
    if support.size > 0:
        for g, _local, _positions in grm_index.global_to_local(support):
            for pos in _positions:
                _snp_grm_map[int(support[pos])] = g

    with open(out_prefix + ".selected_snps.tsv", "w") as f:
        f.write("snp_index\tsource_snp_index\tgrm\tchr\tsnp_id\tcm\tbp\ta1\ta2\tbeta_lasso\tbeta_gls_reml\n")
        for snp_idx in support.tolist():
            chr_, snp_id, cm, bp, a1, a2 = bim_rows.get(
                int(snp_idx),
                ("NA", f"SNP_{int(snp_idx)}", "NA", "NA", "NA", "NA"),
            )
            grm_id = _snp_grm_map.get(int(snp_idx), -1)
            source_snp_idx = source_index_map.get(int(snp_idx), int(snp_idx))
            beta_val = beta_map.get(int(snp_idx), 0.0)
            beta_reml = beta_reml_map.get(int(snp_idx), 0.0)
            f.write(
                f"{int(snp_idx)}\t{source_snp_idx}\t{grm_id}\t{chr_}\t{snp_id}\t{cm}\t{bp}\t{a1}\t{a2}\t"
                f"{beta_val:.8e}\t{beta_reml:.8e}\n"
            )

    logger.info("[INFO] done @ %s elapsed=%.1fs", datetime.now().isoformat(timespec='seconds'), time.time() - t0)
    logger.info("[INFO] summary -> %s.summary.json", out_prefix)
    logger.info("[INFO] support -> %s.selected_snps.tsv", out_prefix)
    log_runtime_gpu_memory(plan)
    close_fitter()
    atexit.unregister(close_fitter)


if __name__ == "__main__":
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    for _name in ("GPU_REML_v6", __name__):
        _lg = logging.getLogger(_name)
        _lg.addHandler(_h)
        _lg.setLevel(logging.INFO)
    main()
