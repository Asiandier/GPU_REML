from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class AdmixtureQ:
    """Ancestry proportions aligned to the REML sample order."""

    weights: np.ndarray
    component_names: tuple[str, ...]


def _read_fam_iids(path: str) -> list[str]:
    iids: list[str] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line {line_no} has fewer than 2 columns.")
            iids.append(parts[1])
    if len(iids) != len(set(iids)):
        raise ValueError(f"{path}: duplicated IID values are not supported for ADMIXTURE Q alignment.")
    return iids


def load_admixture_q_aligned(
    *,
    q_path: str,
    q_fam_path: str,
    sample_ids: Sequence[str],
    component_names: Sequence[str] | None = None,
    row_sum_tol: float = 5e-3,
) -> AdmixtureQ:
    """Load an ADMIXTURE .Q file and align rows to the target sample IDs.

    ADMIXTURE writes .Q rows in the input .fam order.  GPU_REML may drop samples
    after phenotype/covariate alignment, so the Q matrix is aligned by IID
    before entering the REML model.
    """

    q_iids = _read_fam_iids(q_fam_path)
    q = np.loadtxt(q_path, dtype=np.float32)
    if q.ndim == 0:
        q = q.reshape(1, 1)
    elif q.ndim == 1:
        q = q[:, None] if len(q_iids) == int(q.shape[0]) else q[None, :]
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError(f"{q_path}: expected a non-empty 2D ADMIXTURE Q matrix.")
    if not np.isfinite(q).all():
        raise ValueError(f"{q_path}: ADMIXTURE Q matrix contains non-finite values.")
    if np.any(q < -1e-6):
        raise ValueError(f"{q_path}: ADMIXTURE Q matrix contains negative ancestry proportions.")
    q = np.maximum(q, 0.0).astype(np.float32, copy=False)

    if len(q_iids) != int(q.shape[0]):
        raise ValueError(
            f"Q/FAM row mismatch: {q_path} has {int(q.shape[0])} rows but "
            f"{q_fam_path} has {len(q_iids)} samples."
        )
    iid_to_row = {iid: idx for idx, iid in enumerate(q_iids)}
    missing = [iid for iid in sample_ids if iid not in iid_to_row]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} REML samples are missing from ADMIXTURE FAM. Examples: {preview}"
        )

    row_idx = np.asarray([iid_to_row[str(iid)] for iid in sample_ids], dtype=np.int64)
    aligned = q[row_idx, :].astype(np.float32, copy=True)
    row_sums = aligned.sum(axis=1, dtype=np.float64)
    bad = np.abs(row_sums - 1.0) > float(row_sum_tol)
    if np.any(bad):
        raise ValueError(
            f"{q_path}: {int(np.count_nonzero(bad))} aligned Q rows do not sum to 1 "
            f"within tolerance {row_sum_tol:g}."
        )

    if component_names is None:
        names = tuple(f"admix_{idx:03d}" for idx in range(int(aligned.shape[1])))
    else:
        names = tuple(str(name) for name in component_names)
        if len(names) != int(aligned.shape[1]):
            raise ValueError(
                f"component_names length mismatch: expected {int(aligned.shape[1])}, got {len(names)}."
            )
    return AdmixtureQ(weights=aligned, component_names=names)


__all__ = ["AdmixtureQ", "load_admixture_q_aligned"]
