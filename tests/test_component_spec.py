from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
COMPONENT_SPEC = importlib.import_module(f"{PKG.__name__}.component_spec")

load_component_specs = COMPONENT_SPEC.load_component_specs


def test_load_component_specs_from_json_preserves_metadata(tmp_path):
    path = tmp_path / "components.json"
    path.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "ld_low",
                        "variant_indices": [1, 3, 5],
                        "annotation": {"ld_bin": "low"},
                        "provenance": {"source": "unit-test"},
                    },
                    {
                        "name": "ld_high",
                        "variant_indices": [0, 2],
                    },
                ]
            }
        )
    )

    specs = load_component_specs(str(path))

    assert [spec.name for spec in specs] == ["ld_low", "ld_high"]
    assert [spec.variant_indices.tolist() for spec in specs] == [[1, 3, 5], [0, 2]]
    assert specs[0].annotation == {"ld_bin": "low"}
    assert specs[0].provenance == {"source": "unit-test"}
    assert specs[1].annotation is None


def test_load_component_specs_from_npz_can_attach_names_and_metadata(tmp_path):
    path = tmp_path / "components.npz"
    np.savez(
        path,
        arr_0=np.asarray([0, 2], dtype=np.int64),
        arr_1=np.asarray([1, 3], dtype=np.int64),
        component_names=np.asarray(["bin_a", "bin_b"]),
        component_annotations_json=np.asarray(
            [json.dumps({"ld_bin": "a"}), json.dumps({"ld_bin": "b"})]
        ),
        component_provenance_json=np.asarray(
            [json.dumps({"method": "rank"}), json.dumps({"method": "rank"})]
        ),
    )

    specs = load_component_specs(str(path))

    assert [spec.name for spec in specs] == ["bin_a", "bin_b"]
    assert [spec.variant_indices.tolist() for spec in specs] == [[0, 2], [1, 3]]
    assert specs[0].annotation == {"ld_bin": "a"}
    assert specs[1].provenance == {"method": "rank"}
