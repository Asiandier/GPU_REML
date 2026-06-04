from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
ZMERGE = importlib.import_module(f"{PKG.__name__}.zscore_merge")


def _write_component_spec(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "components": [
                    {"name": "c0", "variant_indices": [0, 1]},
                    {"name": "c1", "variant_indices": [2]},
                    {"name": "c2", "variant_indices": [3, 4]},
                    {"name": "c3", "variant_indices": [5]},
                ]
            }
        )
    )


def _base_args(tmp_path: Path, component_spec: Path) -> SimpleNamespace:
    return SimpleNamespace(
        bed_prefix="geno",
        pgen_prefix="",
        rare_bed_prefix="",
        rare_pgen_prefix="",
        vc_block_sizes="",
        component_indices_npz="",
        component_spec=str(component_spec),
        smile=False,
        identity_w=False,
        w_files="",
        w_files_list="",
        grm_groups="",
        pheno_txt="pheno.txt",
        covar_txt="",
        keep_path="",
        compute_effects=False,
        prediction_bed_prefix="",
        prediction_pgen_prefix="",
        out_prefix=str(tmp_path / "final"),
        merge_out_dir=str(tmp_path / "merge"),
        merge_mode="global_weak",
        z_cutoff=ZMERGE.DEFAULT_Z_CUTOFF,
        merge_ai_scale_mode="full",
        merge_n_eff=1.0,
        device="gpu",
        call_width=0,
        gpu_budget_gib=0.0,
        ring_depth=0,
        cpu_threads=1,
        n_rand_vec=4,
        seed=0,
        slq_samples=2,
        slq_m=4,
        slq_mode="projected_core_residual",
        precond_refresh_reldp=0.2,
        precond_type="projected_core",
        minq_iter=2,
        verbose=False,
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_zscore_uses_full_ai_inverse_for_residual_nuisance_adjustment():
    theta = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    ai = np.asarray(
        [
            [4.0, 0.0, 3.0],
            [0.0, 9.0, 0.0],
            [3.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )

    se, z = ZMERGE.zscore_components(theta, ai, n_eff=1.0, ai_scale_mode="full")
    expected_cov = np.linalg.pinv(ai, hermitian=True)[:-1, :-1]
    expected_se = np.sqrt(np.diag(expected_cov))

    assert np.allclose(se, expected_se, rtol=1e-7, atol=1e-7)
    assert se[0] > np.sqrt(1.0 / ai[0, 0])
    assert np.allclose(z, theta[:-1] / expected_se, rtol=1e-7, atol=1e-7)


def test_global_weak_merge_keeps_active_and_merges_weak_components(tmp_path):
    component_spec = tmp_path / "components.json"
    _write_component_spec(component_spec)
    components = ZMERGE.specs_to_components(str(component_spec))

    merged, n_weak, n_groups = ZMERGE.merge_components(
        components,
        np.asarray([2.0, 0.2, 0.3, 2.1], dtype=np.float64),
        z_cutoff=1.0,
        merge_mode="global_weak",
        args=_base_args(tmp_path, component_spec),
    )

    assert n_weak == 2
    assert n_groups == 1
    assert [comp["name"] for comp in merged[:2]] == ["c0", "c3"]
    assert merged[-1]["annotation"]["zscore_merge_status"] == "merged_global_weak"
    assert merged[-1]["annotation"]["source_component_positions"] == [1, 2]
    assert sorted(x for comp in merged for x in comp["variant_indices"]) == list(range(6))


def test_run_from_pipeline_args_refits_only_once_after_one_shot_merge(tmp_path, monkeypatch):
    component_spec = tmp_path / "components.json"
    _write_component_spec(component_spec)
    args = _base_args(tmp_path, component_spec)
    calls: list[Path] = []

    def fake_run_reml_round(args_in, spec_path, prefix):
        del args_in
        calls.append(Path(prefix))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        n_components = len(json.loads(Path(spec_path).read_text())["components"])
        if n_components == 4:
            theta = np.asarray([2.5, 0.4, 0.3, 2.2, 7.0], dtype=np.float64)
        else:
            theta = np.asarray([2.4, 2.1, 0.8, 7.2], dtype=np.float64)
        np.save(prefix.with_suffix(".theta.npy"), theta)
        np.save(prefix.with_suffix(".ai.npy"), np.eye(theta.size, dtype=np.float64))
        np.save(prefix.with_suffix(".grad.npy"), np.zeros(theta.size, dtype=np.float64))
        prefix.with_suffix(".ai_stats.json").write_text(json.dumps({"n_samples": 2}))

    monkeypatch.setattr(ZMERGE, "run_reml_round", fake_run_reml_round)

    summary = ZMERGE.run_from_pipeline_args(args)

    assert [prefix.parent.name for prefix in calls] == ["round00", "round01"]
    assert summary["merge_strategy"] == "zscore_one_shot_merge_refit"
    assert summary["stop_reason"] == "one_shot_merge_refit_complete"
    assert summary["n_initial_components"] == 4
    assert summary["n_final_components"] == 3
    assert Path(args.out_prefix).with_suffix(".theta.npy").exists()

    trace = _read_tsv(Path(args.merge_out_dir) / "zscore_merge_trace.tsv")
    assert [row["stage"] for row in trace] == ["fine_model_zscore_merge", "merged_refit"]


def test_merge_rejects_multiple_bed_prefixes(tmp_path):
    component_spec = tmp_path / "components.json"
    _write_component_spec(component_spec)
    args = _base_args(tmp_path, component_spec)
    args.bed_prefix = "geno_a,geno_b"

    try:
        ZMERGE.run_from_pipeline_args(args)
    except SystemExit as exc:
        assert "one BED prefix" in str(exc)
    else:
        raise AssertionError("expected multiple BED prefixes to be rejected")
