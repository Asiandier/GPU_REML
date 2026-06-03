"""pipeline_common.py — Shared utilities for REML and sparse REML pipelines."""
from __future__ import annotations
import logging
import os
import subprocess
import tempfile
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

logger = logging.getLogger(__name__)

from .suggest_params_v3 import suggest_call_width, PlanResult


# ---------------------------------------------------------------------------
# Environment / GPU helpers
# ---------------------------------------------------------------------------

def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def resolve_cpu_threads(explicit: int | None = None) -> tuple[int, str]:
    if explicit is not None:
        return max(1, int(explicit)), "explicit"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value, name
    try:
        return max(1, len(os.sched_getaffinity(0))), "sched_getaffinity"
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1)), "os.cpu_count"


def first_visible_gpu_id() -> Optional[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.lower() in {"none", "void"}:
        return None
    first = visible.split(",")[0].strip()
    return first or None


def query_gpu() -> tuple[Optional[str], Optional[float], Optional[float]]:
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free,name",
               "--format=csv,noheader,nounits"]
        gpu_id = first_visible_gpu_id()
        if gpu_id is not None:
            cmd.insert(1, f"--id={gpu_id}")
        out = subprocess.check_output(cmd).decode().strip().splitlines()[0].split(",")
        total_mib, free_mib, name = [s.strip() for s in out]
        return name, float(total_mib) * 1024**2, float(free_mib) * 1024**2
    except (FileNotFoundError, subprocess.SubprocessError, OSError, IndexError, ValueError):
        logger.debug("nvidia-smi GPU query failed; falling back to JAX device metadata.", exc_info=True)
        try:
            dev = jax.devices("gpu")[0]
            total = getattr(dev, "memory_limit", None)
            free = total * 0.7 if total is not None else None
            return getattr(dev, "device_kind", None), total, free
        except (RuntimeError, IndexError, AttributeError, TypeError):
            logger.debug("JAX GPU metadata query failed.", exc_info=True)
            return None, None, None


# ---------------------------------------------------------------------------
# Keep-file reading
# ---------------------------------------------------------------------------

def read_keep_ids(keep_path: str) -> list[str]:
    ids: list[str] = []
    with open(keep_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cols = ln.split(maxsplit=2)
            ids.append(cols[1] if len(cols) >= 2 else cols[0])
    return ids


def cleanup_path(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except FileNotFoundError:
        pass


def fam_order_mismatch(fam_path: str, ref_iids: list[str]) -> Optional[str]:
    ref_n = len(ref_iids)
    seen = 0
    with open(fam_path) as f:
        for line_no, line in enumerate(f, start=1):
            cols = line.split()
            if len(cols) < 2:
                return f"{fam_path}: line {line_no} has fewer than 2 columns"
            iid = cols[1]
            if seen >= ref_n:
                return f"{fam_path}: extra IID {iid!r} at line {line_no}"
            if iid != ref_iids[seen]:
                return f"{fam_path}: line {line_no} expected {ref_iids[seen]!r}, found {iid!r}"
            seen += 1
    if seen != ref_n:
        return f"{fam_path}: {seen} samples, expected {ref_n}"
    return None


def _read_psam_iids(psam_path: str) -> list[str]:
    with open(psam_path) as f:
        header = None
        for line in f:
            cols = line.split()
            if cols:
                header = cols
                break
        if header is None:
            raise ValueError(f"{psam_path}: empty file.")
        norm = [c.lstrip("#") for c in header]
        try:
            iid_idx = norm.index("IID")
        except ValueError:
            iid_idx = 1 if len(norm) > 1 else 0
        iids: list[str] = []
        for line in f:
            cols = line.split()
            if not cols:
                continue
            if iid_idx >= len(cols):
                raise ValueError(f"{psam_path}: line has fewer than {iid_idx + 1} columns: {line.rstrip()!r}")
            iid = cols[iid_idx]
            if iid:
                iids.append(iid)
    if not iids:
        raise ValueError(f"{psam_path}: no sample IDs found.")
    return iids


def make_nonbed_input_fam(
    *,
    pgen_prefix: str = "",
) -> str:
    """Create a temporary FAM file from a PGEN .psam sidecar."""
    if not pgen_prefix:
        raise ValueError("pgen_prefix must be provided.")
    psam_path = pgen_prefix + ".psam"
    if not os.path.isfile(psam_path):
        raise FileNotFoundError(f"PSAM file not found: {psam_path}")
    iids = _read_psam_iids(psam_path)
    stem = os.path.basename(pgen_prefix)

    fd, fam_path = tempfile.mkstemp(prefix=f"{stem}_", suffix=".fam")
    try:
        with os.fdopen(fd, "w") as f:
            for iid in iids:
                f.write(f"{iid}\t{iid}\t0\t0\t0\t-9\n")
    except OSError:
        cleanup_path(fam_path)
        raise
    return fam_path


def write_keep_file(iids: Sequence[str], prefix: str) -> str:
    keep_path = prefix + ".keep"
    with open(keep_path, "w") as f:
        for iid in iids:
            f.write(f"{iid} {iid}\n")
    return keep_path


def compute_sample_mask(fam_path: str, keep_iids: list[str]) -> np.ndarray:
    """Boolean mask selecting *keep_iids* from the FAM sample order.

    *fam_path* must list ALL samples in the source's original order
    (e.g. a temp FAM created from .sample / .psam sidecar files).
    *keep_iids* must be a subsequence of the FAM IIDs (same order,
    possibly with gaps).
    """
    keep_set = set(keep_iids)
    mask: list[bool] = []
    order_check: list[str] = []
    with open(fam_path) as f:
        for line in f:
            cols = line.split()
            if len(cols) < 2:
                continue
            iid = cols[1]
            hit = iid in keep_set
            mask.append(hit)
            if hit:
                order_check.append(iid)

    if order_check != list(keep_iids):
        raise ValueError(
            f"Sample mask order mismatch: FAM gives {len(order_check)} "
            f"matching IIDs but keep_iids has {len(keep_iids)}. "
            "Ensure keep_iids is a subsequence of FAM order."
        )
    return np.array(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Hardware profile + planner
# ---------------------------------------------------------------------------

def setup_gpu() -> tuple[Optional[str], Optional[float], Optional[float]]:
    gpu_name, gpu_total, gpu_free = query_gpu()
    return gpu_name, gpu_total, gpu_free


def run_planner(
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
    smile_mode: bool = False,
    smile_w_block_sizes: Optional[Sequence[int]] = None,
) -> PlanResult:
    """Run GPU planner and return a PlanResult."""
    plan = suggest_call_width(
        n_samples=n_samples,
        p_list=p_list,
        n_grm=n_grm,
        component_block_sizes=component_block_sizes,
        precond_type=precond_type,
        gpu_free_bytes=gpu_free,
        gpu_budget_bytes=gpu_budget,
        gpu_name=gpu_name,
        n_covar=n_covar,
        n_rand_vec=n_rand_vec,
        slq_samples=slq_samples,
        ring_depth=ring_depth,
        source_format=source_format,
        arbitrary_component_partition=arbitrary_component_partition,
        smile_mode=smile_mode,
        smile_w_block_sizes=smile_w_block_sizes,
    )
    if not plan.feasible:
        raise SystemExit(
            f"[FATAL] Planner found no feasible configuration.\n  {plan.note}")
    return plan


def print_planner_info(
    plan: PlanResult,
    gpu_name: Optional[str],
    gpu_free: Optional[float],
    call_width: int,
) -> None:
    logger.info(
        "Planner: call_width=%d precond_rank=%d gpu_budget=%.1fGiB "
        "gpu_live_peak=%.1fGiB ring_depth=%d "
        "host_anon~%.1fGiB (ring=%.1fGiB) %s",
        call_width, plan.precond_rank, plan.gpu_budget_gib,
        plan.gpu_peak_gib, plan.ring_depth,
        plan.host_anon_est_gib, plan.host_ring_gib, plan.note,
    )


# ---------------------------------------------------------------------------
# H·V construction
# ---------------------------------------------------------------------------

def build_hv(K_mvs, theta_g: jnp.ndarray, theta_e: jnp.ndarray):
    def hv(V: jnp.ndarray) -> jnp.ndarray:
        acc = theta_e * V
        for i, mv in enumerate(K_mvs):
            acc = acc + theta_g[i] * mv(V)
        return acc
    return hv


# ---------------------------------------------------------------------------
# SPD solver
# ---------------------------------------------------------------------------

def solve_spd(mat: np.ndarray, rhs: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    A = np.asarray(mat, dtype=np.float64)
    B = np.asarray(rhs, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("SPD solve expects a square matrix.")
    if A.shape[0] == 0:
        return np.zeros_like(B, dtype=np.float64)
    if ridge > 0.0:
        A = A + float(ridge) * np.eye(A.shape[0], dtype=np.float64)
    try:
        chol = sla.cholesky(A, lower=True, check_finite=False)
        y = sla.solve_triangular(chol, B, lower=True, check_finite=False)
        return sla.solve_triangular(chol.T, y, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, B)
