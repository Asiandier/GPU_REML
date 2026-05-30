from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .io_utils import write_joined_rows
from .reml_model import PredictionEstimates


def write_prediction_outputs(
    *,
    out_prefix: str,
    predictions: PredictionEstimates,
    sample_ids: Sequence[str],
) -> dict[str, str]:
    """Write prediction outputs to TSV/JSON files."""
    if not out_prefix:
        raise ValueError("write_prediction_outputs requires a non-empty out_prefix.")

    sample_ids = list(sample_ids)
    y_pred = np.asarray(predictions.y_pred, dtype=np.float64).reshape(-1)
    y_pred_std = np.asarray(predictions.y_pred_std, dtype=np.float64).reshape(-1)
    fixed = np.asarray(predictions.fixed_effect, dtype=np.float64).reshape(-1)
    rand = np.asarray(predictions.random_effect, dtype=np.float64).reshape(-1)
    if len(sample_ids) != y_pred.size:
        raise ValueError(
            f"sample_ids length mismatch: expected {y_pred.size}, got {len(sample_ids)}."
        )

    pred_path = out_prefix + ".prediction.tsv"
    meta_path = out_prefix + ".prediction_metadata.json"
    rand_components = [
        np.asarray(g, dtype=np.float64).reshape(-1)
        for g in predictions.random_effect_components
    ]
    header = [
        "sample_index",
        "iid",
        "fixed_effect",
        "random_effect",
        "y_pred_std",
        "y_pred",
    ] + [f"random_component_{g:03d}" for g in range(len(rand_components))]
    write_joined_rows(
        pred_path,
        "\t".join(header) + "\n",
        (
            f"{idx}\t{iid}\t"
            + "\t".join(
                [
                    f"{fixed[idx]:.8e}",
                    f"{rand[idx]:.8e}",
                    f"{y_pred_std[idx]:.8e}",
                    f"{y_pred[idx]:.8e}",
                ]
                + [f"{comp[idx]:.8e}" for comp in rand_components]
            )
            + "\n"
            for idx, iid in enumerate(sample_ids)
        ),
    )

    meta = {
        "n_samples": int(y_pred.size),
        "n_components": int(len(rand_components)),
        "files": {
            "prediction": pred_path,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    return {
        "prediction": pred_path,
        "prediction_metadata": meta_path,
    }


__all__ = ["write_prediction_outputs"]
