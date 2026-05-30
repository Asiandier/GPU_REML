from __future__ import annotations

import importlib
import os
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
SPARSE_PIPELINE = importlib.import_module(f"{PKG}.run_sparse_reml_pipeline")


def test_sparse_pipeline_help_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        SPARSE_PIPELINE.parse_args()

    assert excinfo.value.code == 0
    assert "Run sparse REML + LASSO pipeline" in capsys.readouterr().out


def test_sparse_pipeline_accepts_component_spec_arg(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-reml-sparse",
            "--bed-prefix",
            "geno",
            "--component-spec",
            "components.npz",
            "--pheno-txt",
            "pheno.txt",
        ],
    )

    args = SPARSE_PIPELINE.parse_args()

    assert args.component_spec == "components.npz"
    assert args.component_indices_npz == ""
