from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .geno_stream import _ensure_on_device
from .kv_impl import kv_impl_snp_weighted_streamed
from .pcg import pcg_solve
from .reml import _stable_cho_factor_spd, fit_reml, standardize_response

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RelaxationConfig:
    n_groups: int
    outer_iters: int = 5
    theta_lr: float = 1e-3
    n_rand_vec: int = 32
    max_pcg_iters: int = 400
    minq_iter: int = 20
    seed: int = 0
    h2_init: float = 0.5
    slq_samples: int = 4
    slq_m: int = 8
    pcg_tol: float = 1e-3
    verbose: bool = True
    refit_final: bool = True


@dataclasses.dataclass
class SoftGroupingGradient:
    theta_grad: jnp.ndarray
    zTPy: jnp.ndarray
    zTPz: jnp.ndarray
    yPKPy: jnp.ndarray
    trPK: jnp.ndarray
    pcg_iters: int


@dataclasses.dataclass
class RelaxationResult:
    theta: jnp.ndarray
    var_components: jnp.ndarray
    theta_grad: jnp.ndarray
    history: list[dict[str, object]]


def _theta_by_call(streamer, theta: np.ndarray) -> np.ndarray:
    theta_np = np.asarray(theta, dtype=np.float32)
    G = int(theta_np.shape[1])
    out = np.zeros((G, int(streamer._n_calls), int(streamer._max_unpack_width)), dtype=np.float32)
    for c in range(int(streamer._n_calls)):
        s0 = int(streamer._call_snp_starts[c])
        tw = int(streamer._call_true_widths[c])
        if tw > 0:
            out[:, c, :tw] = theta_np[s0 : s0 + tw, :].T
    return out


class SoftGroupingOperator:
    """Dynamic dense soft-grouping GRM operator for one genotype streamer."""

    def __init__(self, streamer, theta: np.ndarray | jnp.ndarray):
        theta_np = np.asarray(jax.device_get(jnp.asarray(theta)), dtype=np.float32)
        if theta_np.ndim != 2:
            raise ValueError(f"theta must be a 2D array with shape (m, G), got {theta_np.shape}.")
        if int(theta_np.shape[0]) != int(streamer.m):
            raise ValueError(
                f"theta SNP dimension mismatch: expected m={int(streamer.m)}, got {theta_np.shape[0]}."
            )
        if theta_np.shape[1] <= 0:
            raise ValueError("theta must contain at least one group.")
        if not np.isfinite(theta_np).all():
            raise ValueError("theta contains non-finite values.")
        denom_np = np.sum(theta_np, axis=0, dtype=np.float64).astype(np.float32)
        if np.any(denom_np == 0.0):
            raise ZeroDivisionError("At least one soft-group denominator sum_i theta_ik is exactly zero.")

        self.streamer = streamer
        self.theta = jax.device_put(jnp.asarray(theta_np, dtype=jnp.float32), streamer.dev)
        self.denoms = jax.device_put(jnp.asarray(denom_np, dtype=jnp.float32), streamer.dev)
        self.n_groups = int(theta_np.shape[1])
        self.weights_by_call = jax.device_put(
            jnp.asarray(_theta_by_call(streamer, theta_np), dtype=jnp.float32),
            streamer.dev,
        )

    def kv(self, component_idx: int, V: jnp.ndarray) -> jnp.ndarray:
        if component_idx < 0 or component_idx >= self.n_groups:
            raise IndexError(f"component_idx={component_idx} out of range for {self.n_groups} groups.")
        self.streamer._prepare_kv_pass()
        V = _ensure_on_device(V, self.streamer.dev)
        return kv_impl_snp_weighted_streamed(
            V,
            self.streamer._true_widths_dev,
            self.streamer._means_by_call,
            self.streamer._inv_by_call,
            self.weights_by_call[component_idx],
            self.denoms[component_idx],
            n=int(self.streamer.n),
            n_calls=int(self.streamer._n_calls),
            pop_block=self.streamer._pop_cached,
            missing_val=int(self.streamer._missing_val),
        )

    def stacked_kv(self, V: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([self.kv(g_idx, V) for g_idx in range(self.n_groups)], axis=0)

    def weighted_hv(
        self,
        theta_g: jnp.ndarray,
        theta_e: jnp.ndarray | None,
        V: jnp.ndarray,
    ) -> jnp.ndarray:
        V = _ensure_on_device(V, self.streamer.dev)
        squeeze = V.ndim == 1
        if squeeze:
            V = V[:, None]
        acc = jnp.zeros_like(V)
        if theta_e is not None:
            acc = acc + jnp.asarray(theta_e, dtype=V.dtype) * V
        for g_idx in range(self.n_groups):
            acc = acc + theta_g[g_idx] * self.kv(g_idx, V)
        return acc[:, 0] if squeeze else acc

    def diag_list(self) -> tuple[jnp.ndarray, ...]:
        diag_one = jax.device_put(jnp.ones((int(self.streamer.n),), dtype=jnp.float32), self.streamer.dev)
        return tuple(diag_one for _ in range(self.n_groups))


def _project_hinv_columns(
    *,
    Hinv_cols: jnp.ndarray,
    covar_mat: jnp.ndarray | None,
    covar_cols: int,
) -> tuple[jnp.ndarray, object | None]:
    if covar_mat is None or covar_cols == 0:
        return Hinv_cols, None
    HinvC = Hinv_cols[:, :covar_cols]
    CtHinvC = covar_mat.T @ HinvC
    chol = _stable_cho_factor_spd(CtHinvC)
    mid = jsp.linalg.cho_solve(chol, covar_mat.T @ Hinv_cols, check_finite=False)
    return Hinv_cols - HinvC @ mid, chol


def compute_soft_grouping_gradient(
    *,
    streamer,
    op: SoftGroupingOperator,
    y: jnp.ndarray,
    covar: Optional[jnp.ndarray],
    var_components: jnp.ndarray,
    n_rand_vec: int,
    seed: int,
    max_pcg_iters: int,
    pcg_tol: float,
) -> SoftGroupingGradient:
    """Compute the document's SNP-level soft-grouping gradient.

    The gradient is the ordinary partial derivative with respect to theta.
    Constraint handling is applied by the theta update rule, not here.
    """
    if n_rand_vec <= 0:
        raise ValueError("n_rand_vec must be positive.")
    y_std, _, _ = standardize_response(jnp.asarray(y, dtype=jnp.float32))
    covar_mat = (
        jnp.asarray(covar, dtype=jnp.float32)
        if covar is not None and jnp.asarray(covar).size > 0
        else None
    )
    if covar_mat is not None and covar_mat.ndim == 1:
        covar_mat = covar_mat[:, None]

    theta_vc = jnp.asarray(var_components, dtype=jnp.float32).reshape(-1)
    if int(theta_vc.shape[0]) != op.n_groups + 1:
        raise ValueError(
            f"var_components length mismatch: expected {op.n_groups + 1}, got {int(theta_vc.shape[0])}."
        )
    theta_g = theta_vc[:-1]
    theta_e = theta_vc[-1]

    key = jax.random.PRNGKey(seed + 7919)
    Vrand = jax.random.rademacher(
        key, (int(streamer.n), int(n_rand_vec)), dtype=jnp.int32
    ).astype(jnp.float32)

    rhs_parts = []
    covar_cols = 0
    if covar_mat is not None and covar_mat.shape[1] > 0:
        covar_cols = int(covar_mat.shape[1])
        rhs_parts.append(covar_mat)
    y_col = covar_cols
    rhs_parts.append(y_std[:, None])
    z_start = y_col + 1
    rhs_parts.append(Vrand)
    rhs = jnp.concatenate(rhs_parts, axis=1)

    def Hv(V):
        return op.weighted_hv(theta_g, theta_e, V)

    sol, _, k_pcg = pcg_solve(
        Hv,
        rhs,
        M=None,
        tol=float(pcg_tol),
        maxiter=int(max_pcg_iters),
        X0=jnp.zeros_like(rhs),
    )
    projected, _ = _project_hinv_columns(Hinv_cols=sol, covar_mat=covar_mat, covar_cols=covar_cols)
    Py = projected[:, y_col]
    PZ = projected[:, z_start:]

    zTPy = streamer.xtv(Py, normalize=False)
    xTZ = streamer.xtv(Vrand, normalize=False)
    xTPZ = streamer.xtv(PZ, normalize=False)
    zTPz = jnp.mean(xTZ * xTPZ, axis=1)

    yPKPy_terms = []
    trPK_terms = []
    for g_idx in range(op.n_groups):
        KPy = op.kv(g_idx, Py)
        KPZ = op.kv(g_idx, PZ)
        yPKPy_terms.append(jnp.dot(Py, KPy))
        trPK_terms.append(jnp.sum(Vrand * KPZ) / float(n_rand_vec))
    yPKPy = jnp.stack(yPKPy_terms)
    trPK = jnp.stack(trPK_terms)

    theta_grad = soft_grouping_theta_gradient_from_terms(
        zTPy=zTPy,
        zTPz=zTPz,
        yPKPy=yPKPy,
        trPK=trPK,
        tau2=theta_g,
        denoms=op.denoms,
    )

    return SoftGroupingGradient(
        theta_grad=theta_grad,
        zTPy=zTPy,
        zTPz=zTPz,
        yPKPy=yPKPy,
        trPK=trPK,
        pcg_iters=int(k_pcg),
    )


def soft_grouping_theta_gradient_from_terms(
    *,
    zTPy: jnp.ndarray,
    zTPz: jnp.ndarray,
    yPKPy: jnp.ndarray,
    trPK: jnp.ndarray,
    tau2: jnp.ndarray,
    denoms: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble ``d l / d theta_ik`` from the REML projection terms."""
    zTPy = jnp.asarray(zTPy)
    zTPz = jnp.asarray(zTPz)
    yPKPy = jnp.asarray(yPKPy)
    trPK = jnp.asarray(trPK)
    tau2 = jnp.asarray(tau2)
    denoms = jnp.asarray(denoms)
    common = zTPy * zTPy - zTPz
    columns = []
    for g_idx in range(int(tau2.shape[0])):
        bracket = common - (yPKPy[g_idx] - trPK[g_idx])
        columns.append(tau2[g_idx] * bracket / (2.0 * denoms[g_idx]))
    return jnp.stack(columns, axis=1)


def initialize_theta(
    *,
    m: int,
    n_groups: int,
    seed: int,
    mode: str = "random",
    random_low: float = 0.0,
    random_high: float = 1.0,
) -> np.ndarray:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    if mode == "uniform":
        return np.full((int(m), int(n_groups)), 1.0 / float(n_groups), dtype=np.float32)
    if mode != "random":
        raise ValueError("theta initialization mode must be 'random' or 'uniform'.")
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(float(random_low), float(random_high), size=(int(m), int(n_groups))).astype(np.float32)
    return theta / np.sum(theta, axis=1, keepdims=True)


def constrained_theta_update(theta: jnp.ndarray, theta_grad: jnp.ndarray, theta_lr: float) -> jnp.ndarray:
    """Exponentiated-gradient update that preserves per-SNP simplex constraints."""
    numer = theta * jnp.exp(float(theta_lr) * theta_grad)
    return numer / jnp.sum(numer, axis=1, keepdims=True)


def run_relaxation_grouping(
    *,
    streamer,
    y: jnp.ndarray,
    covar: Optional[jnp.ndarray],
    theta_init: np.ndarray | jnp.ndarray,
    cfg: RelaxationConfig,
) -> RelaxationResult:
    theta = jnp.asarray(theta_init, dtype=jnp.float32)
    if int(theta.shape[1]) != int(cfg.n_groups):
        raise ValueError(f"theta_init group mismatch: expected {cfg.n_groups}, got {int(theta.shape[1])}.")

    history: list[dict[str, object]] = []
    var_components_init = None
    last_vc = None
    last_grad = jnp.zeros_like(theta)

    n_updates = max(0, int(cfg.outer_iters))
    for outer in range(n_updates):
        iter_t0 = time.time()
        if cfg.verbose:
            logger.info(
                "[relaxation] outer %d/%d REML start @ %s",
                outer + 1,
                n_updates,
                datetime.now().isoformat(timespec="seconds"),
            )
        op = SoftGroupingOperator(streamer, theta)
        vc, reml_history, diagnostics = fit_reml(
            y=jnp.asarray(y, dtype=jnp.float32),
            K_mvs=tuple(lambda V, g_idx=g_idx, op=op: op.kv(g_idx, V) for g_idx in range(op.n_groups)),
            diag_list=op.diag_list(),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            n_rand_vec=int(cfg.n_rand_vec),
            maxiter=int(cfg.max_pcg_iters),
            seed=int(cfg.seed) + outer * 1009,
            h2_init=float(cfg.h2_init),
            param_init=var_components_init,
            minq_iter=int(cfg.minq_iter),
            slq_samples=int(cfg.slq_samples),
            slq_m=int(cfg.slq_m),
            slq_mode="raw",
            precond_conf=None,
            precond_eps=1e-6,
            weighted_hv=op.weighted_hv,
            stacked_kv=op.stacked_kv,
            return_diagnostics=True,
            verbose=bool(cfg.verbose),
        )
        grad_info = compute_soft_grouping_gradient(
            streamer=streamer,
            op=op,
            y=jnp.asarray(y, dtype=jnp.float32),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            var_components=vc,
            n_rand_vec=int(cfg.n_rand_vec),
            seed=int(cfg.seed) + outer * 1009,
            max_pcg_iters=int(cfg.max_pcg_iters),
            pcg_tol=float(cfg.pcg_tol),
        )
        last_vc = vc
        last_grad = grad_info.theta_grad
        theta_next = constrained_theta_update(theta, last_grad, float(cfg.theta_lr))
        step = theta_next - theta
        theta_host, grad_host, step_host, vc_host = jax.device_get(
            (theta_next, last_grad, step, vc)
        )
        hist_row = {
            "outer_iter": outer + 1,
            "loglik": float(jax.device_get(diagnostics["loglik"])),
            "theta_grad_norm": float(np.linalg.norm(np.asarray(grad_host).reshape(-1))),
            "theta_step_norm": float(np.linalg.norm(np.asarray(step_host).reshape(-1))),
            "theta_min": float(np.min(theta_host)),
            "theta_max": float(np.max(theta_host)),
            "theta_row_sum_min": float(np.min(np.sum(theta_host, axis=1))),
            "theta_row_sum_max": float(np.max(np.sum(theta_host, axis=1))),
            "denom_min": float(np.min(np.sum(theta_host, axis=0))),
            "denom_max": float(np.max(np.sum(theta_host, axis=0))),
            "var_components": [float(x) for x in np.asarray(vc_host).reshape(-1)],
            "reml_stop_reason": str(diagnostics.get("stop_reason", "")),
            "reml_iters": int(len(reml_history)),
            "gradient_pcg_iters": int(grad_info.pcg_iters),
            "elapsed_sec": float(time.time() - iter_t0),
        }
        history.append(hist_row)
        if cfg.verbose:
            logger.info(
                "[relaxation] outer %d done ll=%.6e grad=%.3e step=%.3e theta=[%.3e, %.3e]",
                outer + 1,
                hist_row["loglik"],
                hist_row["theta_grad_norm"],
                hist_row["theta_step_norm"],
                hist_row["theta_min"],
                hist_row["theta_max"],
            )
        theta = jnp.asarray(theta_next, dtype=jnp.float32)
        var_components_init = jnp.asarray(vc, dtype=jnp.float32)

    if cfg.refit_final:
        op = SoftGroupingOperator(streamer, theta)
        vc, _, diagnostics = fit_reml(
            y=jnp.asarray(y, dtype=jnp.float32),
            K_mvs=tuple(lambda V, g_idx=g_idx, op=op: op.kv(g_idx, V) for g_idx in range(op.n_groups)),
            diag_list=op.diag_list(),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            n_rand_vec=int(cfg.n_rand_vec),
            maxiter=int(cfg.max_pcg_iters),
            seed=int(cfg.seed) + 99991,
            h2_init=float(cfg.h2_init),
            param_init=var_components_init,
            minq_iter=int(cfg.minq_iter),
            slq_samples=int(cfg.slq_samples),
            slq_m=int(cfg.slq_m),
            slq_mode="raw",
            precond_conf=None,
            precond_eps=1e-6,
            weighted_hv=op.weighted_hv,
            stacked_kv=op.stacked_kv,
            return_diagnostics=True,
            verbose=bool(cfg.verbose),
        )
        grad_info = compute_soft_grouping_gradient(
            streamer=streamer,
            op=op,
            y=jnp.asarray(y, dtype=jnp.float32),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            var_components=vc,
            n_rand_vec=int(cfg.n_rand_vec),
            seed=int(cfg.seed) + 99991,
            max_pcg_iters=int(cfg.max_pcg_iters),
            pcg_tol=float(cfg.pcg_tol),
        )
        last_vc = vc
        last_grad = grad_info.theta_grad
        history.append(
            {
                "outer_iter": "final_refit",
                "loglik": float(jax.device_get(diagnostics["loglik"])),
                "theta_grad_norm": float(np.linalg.norm(np.asarray(jax.device_get(last_grad)).reshape(-1))),
                "theta_step_norm": 0.0,
                "theta_min": float(np.min(np.asarray(jax.device_get(theta)))),
                "theta_max": float(np.max(np.asarray(jax.device_get(theta)))),
                "theta_row_sum_min": float(np.min(np.sum(np.asarray(jax.device_get(theta)), axis=1))),
                "theta_row_sum_max": float(np.max(np.sum(np.asarray(jax.device_get(theta)), axis=1))),
                "denom_min": float(np.min(np.sum(np.asarray(jax.device_get(theta)), axis=0))),
                "denom_max": float(np.max(np.sum(np.asarray(jax.device_get(theta)), axis=0))),
                "var_components": [float(x) for x in np.asarray(jax.device_get(vc)).reshape(-1)],
                "reml_stop_reason": str(diagnostics.get("stop_reason", "")),
                "gradient_pcg_iters": int(grad_info.pcg_iters),
            }
        )

    if last_vc is None:
        raise RuntimeError("No REML fit was run; set outer_iters > 0 or refit_final=True.")

    return RelaxationResult(
        theta=theta,
        var_components=jnp.asarray(last_vc, dtype=jnp.float32),
        theta_grad=jnp.asarray(last_grad, dtype=jnp.float32),
        history=history,
    )


def write_relaxation_outputs(
    *,
    out_prefix: str,
    result: RelaxationResult,
    config: RelaxationConfig,
) -> dict[str, str]:
    theta_path = out_prefix + ".theta.npy"
    vc_path = out_prefix + ".var_components.npy"
    grad_path = out_prefix + ".theta_grad.npy"
    history_path = out_prefix + ".history.json"
    meta_path = out_prefix + ".meta.json"

    np.save(theta_path, np.asarray(jax.device_get(result.theta), dtype=np.float32))
    np.save(vc_path, np.asarray(jax.device_get(result.var_components), dtype=np.float32))
    np.save(grad_path, np.asarray(jax.device_get(result.theta_grad), dtype=np.float32))
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(result.history, f, indent=2, sort_keys=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "simplex_constrained_soft_grouping_exponentiated_gradient_ascent",
                "n_groups": int(config.n_groups),
                "outer_iters": int(config.outer_iters),
                "theta_lr": float(config.theta_lr),
                "n_rand_vec": int(config.n_rand_vec),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "outputs": {
                    "theta": theta_path,
                    "var_components": vc_path,
                    "theta_grad": grad_path,
                    "history": history_path,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return {
        "theta": theta_path,
        "var_components": vc_path,
        "theta_grad": grad_path,
        "history": history_path,
        "meta": meta_path,
    }


__all__ = [
    "RelaxationConfig",
    "RelaxationResult",
    "SoftGroupingGradient",
    "SoftGroupingOperator",
    "compute_soft_grouping_gradient",
    "constrained_theta_update",
    "initialize_theta",
    "run_relaxation_grouping",
    "soft_grouping_theta_gradient_from_terms",
    "write_relaxation_outputs",
]
