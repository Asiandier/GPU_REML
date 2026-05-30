from __future__ import annotations

import importlib
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
RUN_REML = importlib.import_module(f"{PKG.__name__}.run_reml_pipeline")
RUN_SPARSE = importlib.import_module(f"{PKG.__name__}.run_sparse_reml_pipeline")

_load_component_variant_indices = RUN_REML._load_component_variant_indices
_load_sparse_component_variant_indices = RUN_SPARSE._load_component_variant_indices


def test_load_component_variant_indices_sorts_default_arr_keys_numerically(tmp_path):
    path = tmp_path / "components.npz"
    np.savez(
        path,
        arr_0=np.asarray([0, 1], dtype=np.int64),
        arr_1=np.asarray([2], dtype=np.int64),
        arr_10=np.asarray([10], dtype=np.int64),
        arr_2=np.asarray([3, 4], dtype=np.int64),
    )

    got = _load_component_variant_indices(str(path))

    assert [arr.tolist() for arr in got] == [[0, 1], [2], [3, 4], [10]]


def test_load_component_variant_indices_preserves_npz_order_for_named_keys(tmp_path):
    path = tmp_path / "named_components.npz"
    np.savez(
        path,
        ld_high=np.asarray([5, 6], dtype=np.int64),
        ld_low=np.asarray([1, 2], dtype=np.int64),
        rare=np.asarray([9], dtype=np.int64),
    )

    got = _load_component_variant_indices(str(path))

    assert [arr.tolist() for arr in got] == [[5, 6], [1, 2], [9]]


def test_load_component_variant_indices_accepts_json_component_spec(tmp_path):
    path = tmp_path / "components.json"
    path.write_text(
        """
        {
          "components": [
            {"name": "low", "variant_indices": [4, 1]},
            {"name": "high", "variant_indices": [3]}
          ]
        }
        """
    )

    got = _load_component_variant_indices(str(path))

    assert [arr.tolist() for arr in got] == [[4, 1], [3]]


def test_sparse_lasso_load_component_variant_indices_matches_reml(tmp_path):
    path = tmp_path / "components.json"
    path.write_text(
        """
        {
          "components": [
            {"name": "maf_low", "variant_indices": [0, 2]},
            {"name": "maf_high", "variant_indices": [1, 3]}
          ]
        }
        """
    )

    got = _load_sparse_component_variant_indices(str(path))

    assert [arr.tolist() for arr in got] == [[0, 2], [1, 3]]
