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
    theta_linesearch_trials: int = 3
    verbose: bool = True
    reml_verbose: bool = True
    reml_log_detail: str = "compact"


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
                "theta SNP dimension mismatch: "
                f"expected m={int(streamer.m)}, got {theta_np.shape[0]}."
            )
        if theta_np.shape[1] <= 0:
            raise ValueError("theta must contain at least one group.")
        if not np.isfinite(theta_np).all():
            raise ValueError("theta contains non-finite values.")
        denom_np = np.sum(theta_np, axis=0, dtype=np.float64).astype(np.float32)
        if np.any(denom_np == 0.0):
            raise ZeroDivisionError(
                "At least one soft-group denominator sum_i theta_ik is exactly zero."
            )

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
            raise IndexError(
                f"component_idx={component_idx} out of range for {self.n_groups} groups."
            )
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
        diag_one = jax.device_put(
            jnp.ones((int(self.streamer.n),), dtype=jnp.float32),
            self.streamer.dev,
        )
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
            "var_components length mismatch: "
            f"expected {op.n_groups + 1}, got {int(theta_vc.shape[0])}."
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
    theta = rng.uniform(
        float(random_low),
        float(random_high),
        size=(int(m), int(n_groups)),
    ).astype(np.float32)
    return theta / np.sum(theta, axis=1, keepdims=True)


def constrained_theta_update(
    theta: jnp.ndarray,
    theta_grad: jnp.ndarray,
    theta_lr: float,
) -> jnp.ndarray:
    """Exponentiated-gradient update that preserves per-SNP simplex constraints."""
    numer = theta * jnp.exp(float(theta_lr) * theta_grad)
    return numer / jnp.sum(numer, axis=1, keepdims=True)


def _h2_from_vc(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    return float(np.sum(arr[:-1]) / total) if arr.size >= 2 and total > 0.0 else float("nan")


def _format_vc(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v):.3e}" for v in arr) + "]"


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
        raise ValueError(
            f"theta_init group mismatch: expected {cfg.n_groups}, got {int(theta.shape[1])}."
        )

    history: list[dict[str, object]] = []
    var_components_init = None
    accepted_theta = theta
    accepted_vc = None
    accepted_grad = jnp.zeros_like(theta)
    accepted_loglik: float | None = None
    consecutive_rejections = 0
    proposal_scale = 0.0
    proposal_trial = 0

    n_updates = max(0, int(cfg.outer_iters))
    reject_patience = max(1, int(cfg.theta_linesearch_trials))
    for outer in range(n_updates):
        iter_t0 = time.time()
        reml_seed = int(cfg.seed)
        current_proposal_scale = float(proposal_scale)
        current_proposal_trial = int(proposal_trial)
        if cfg.verbose:
            logger.info(
                "[relaxation] outer %d/%d evaluate current theta @ %s%s",
                outer + 1,
                n_updates,
                datetime.now().isoformat(timespec="seconds"),
                ""
                if accepted_loglik is None
                else (
                    f" accepted_best_ll={accepted_loglik:.6e} "
                    f"proposal_scale={current_proposal_scale:.3e}"
                ),
            )
        op = SoftGroupingOperator(streamer, theta)
        k_mvs = tuple(
            lambda V, g_idx=g_idx, op=op: op.kv(g_idx, V)
            for g_idx in range(op.n_groups)
        )
        vc, reml_history, diagnostics = fit_reml(
            y=jnp.asarray(y, dtype=jnp.float32),
            K_mvs=k_mvs,
            diag_list=op.diag_list(),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            n_rand_vec=int(cfg.n_rand_vec),
            maxiter=int(cfg.max_pcg_iters),
            seed=reml_seed,
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
            verbose=bool(cfg.reml_verbose),
            log_detail=str(cfg.reml_log_detail),
        )
        current_loglik = float(jax.device_get(diagnostics["loglik"]))
        previous_best = current_loglik if accepted_loglik is None else accepted_loglik
        accepted = accepted_loglik is None or current_loglik >= accepted_loglik
        grad_info = compute_soft_grouping_gradient(
            streamer=streamer,
            op=op,
            y=jnp.asarray(y, dtype=jnp.float32),
            covar=jnp.asarray(covar, dtype=jnp.float32) if covar is not None else None,
            var_components=vc,
            n_rand_vec=int(cfg.n_rand_vec),
            seed=reml_seed,
            max_pcg_iters=int(cfg.max_pcg_iters),
            pcg_tol=float(cfg.pcg_tol),
        ) if accepted else None

        delta_vs_best = current_loglik - previous_best
        if accepted:
            accepted_theta_prev = accepted_theta
            accepted_theta = theta
            accepted_vc = vc
            accepted_loglik = current_loglik
            accepted_grad = grad_info.theta_grad
            var_components_init = jnp.asarray(accepted_vc, dtype=jnp.float32)
            consecutive_rejections = 0
            proposal_scale = 1.0
            proposal_trial = 1
            theta_next = constrained_theta_update(
                accepted_theta,
                accepted_grad,
                float(cfg.theta_lr) * proposal_scale,
            )
            decision = "accept current theta; gradient computed and next theta proposed"
            step = accepted_theta - accepted_theta_prev if outer > 0 else jnp.zeros_like(theta)
            row_vc = vc
            row_grad = accepted_grad
            row_reml_history = reml_history
        else:
            consecutive_rejections += 1
            decision = "reject current theta; next outer will test a smaller step"
            step = theta - accepted_theta
            row_vc = vc
            row_grad = accepted_grad
            row_reml_history = reml_history
            if consecutive_rejections >= reject_patience:
                theta_next = accepted_theta
            else:
                proposal_trial = consecutive_rejections + 1
                proposal_scale = 0.5 ** consecutive_rejections
                theta_next = constrained_theta_update(
                    accepted_theta,
                    accepted_grad,
                    float(cfg.theta_lr) * proposal_scale,
                )

        theta_host, grad_host, step_host, vc_host, accepted_vc_host = jax.device_get(
            (
                theta,
                row_grad,
                step,
                row_vc,
                accepted_vc if accepted_vc is not None else vc,
            )
        )
        hist_row = {
            "outer_iter": outer + 1,
            "loglik": float(current_loglik),
            "evaluated_loglik": float(current_loglik),
            "accepted_loglik": float(accepted_loglik),
            "previous_accepted_loglik": float(previous_best),
            "delta_vs_best": float(delta_vs_best),
            "theta_state_accepted": bool(accepted),
            "theta_update_accepted": bool(accepted and current_proposal_trial > 0),
            "theta_linesearch_accepted": bool(accepted),
            "theta_proposal_scale": float(current_proposal_scale),
            "theta_proposal_trial": int(current_proposal_trial),
            "theta_reject_streak": int(consecutive_rejections),
            "outer_stop_reason": "",
            "theta_grad_norm": float(np.linalg.norm(np.asarray(grad_host).reshape(-1))),
            "theta_step_norm": float(np.linalg.norm(np.asarray(step_host).reshape(-1))),
            "theta_min": float(np.min(theta_host)),
            "theta_max": float(np.max(theta_host)),
            "theta_row_sum_min": float(np.min(np.sum(theta_host, axis=1))),
            "theta_row_sum_max": float(np.max(np.sum(theta_host, axis=1))),
            "denom_min": float(np.min(np.sum(theta_host, axis=0))),
            "denom_max": float(np.max(np.sum(theta_host, axis=0))),
            "var_components": [float(x) for x in np.asarray(vc_host).reshape(-1)],
            "accepted_var_components": [float(x) for x in np.asarray(accepted_vc_host).reshape(-1)],
            "reml_stop_reason": str(diagnostics.get("stop_reason", "")),
            "reml_iters": int(len(row_reml_history)),
            "gradient_pcg_iters": int(grad_info.pcg_iters) if grad_info is not None else 0,
            "elapsed_sec": float(time.time() - iter_t0),
        }
        history.append(hist_row)
        if cfg.verbose:
            logger.info(
                "[relaxation] outer %d result: evaluated_ll=%.6e "
                "delta_vs_accepted=%+.3e %s",
                outer + 1,
                hist_row["loglik"],
                hist_row["delta_vs_best"],
                decision,
            )
            logger.info(
                "[relaxation] outer %d summary: proposal_scale=%.3e "
                "theta_step_norm=%.3e grad_norm=%.3e h2=%.6f",
                outer + 1,
                hist_row["theta_proposal_scale"],
                hist_row["theta_step_norm"],
                hist_row["theta_grad_norm"],
                _h2_from_vc(vc_host),
            )
            logger.info(
                "[relaxation] outer %d vc=%s theta_range=[%.3e, %.3e]",
                outer + 1,
                _format_vc(vc_host),
                hist_row["theta_min"],
                hist_row["theta_max"],
            )
        if not accepted and consecutive_rejections >= reject_patience:
            hist_row["outer_stop_reason"] = "theta_line_search_rejected"
            if cfg.verbose:
                logger.info(
                    "[relaxation] stop outer loop: %d consecutive rejected theta proposal(s); "
                    "theta_linesearch_trials=%d max_outer=%d",
                    consecutive_rejections,
                    reject_patience,
                    n_updates,
                )
            break
        theta = jnp.asarray(theta_next, dtype=jnp.float32)

    if accepted_vc is None:
        raise RuntimeError("No REML fit was run; set outer_iters > 0.")

    return RelaxationResult(
        theta=accepted_theta,
        var_components=jnp.asarray(accepted_vc, dtype=jnp.float32),
        theta_grad=jnp.asarray(accepted_grad, dtype=jnp.float32),
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
                "max_outer_iters": int(config.outer_iters),
                "theta_lr": float(config.theta_lr),
                "theta_linesearch_trials": int(config.theta_linesearch_trials),
                "reml_verbose": bool(config.reml_verbose),
                "reml_log_detail": str(config.reml_log_detail),
                "seed_policy": "fixed_across_outer",
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
