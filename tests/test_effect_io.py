from __future__ import annotations

import importlib
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")
EFFECT_IO = importlib.import_module(f"{PKG.__name__}.effect_io")

EffectEstimates = REML_MODEL.EffectEstimates
write_effect_outputs = EFFECT_IO.write_effect_outputs


def test_write_effect_outputs_writes_expected_files(tmp_path):
    effects = EffectEstimates(
        fixed_effects=jnp.asarray([0.1, -0.2], dtype=jnp.float32),
        random_effect=jnp.asarray([0.3, -0.4], dtype=jnp.float32),
        random_effect_components=(
            jnp.asarray([0.1, -0.1], dtype=jnp.float32),
            jnp.asarray([0.2, -0.3], dtype=jnp.float32),
        ),
        snp_effects=(
            jnp.asarray([0.01, 0.02], dtype=jnp.float32),
            jnp.asarray([0.03], dtype=jnp.float32),
        ),
        pcg_rel_res=1e-4,
        pcg_iters=7,
        y_mean=1.25,
        y_scale=0.5,
    )

    out_prefix = str(tmp_path / "effects" / "demo")
    paths = write_effect_outputs(
        out_prefix=out_prefix,
        effects=effects,
        sample_ids=["iid1", "iid2"],
        component_global_offsets=[0, 2],
    )

    fixed_txt = (tmp_path / "effects" / "demo.fixed_effects.tsv").read_text()
    assert "covariate_index\tlabel\tbeta" in fixed_txt
    assert "0\tintercept\t1.00000001e-01" in fixed_txt

    random_txt = (tmp_path / "effects" / "demo.random_effect.tsv").read_text()
    assert "sample_index\tiid\trandom_effect" in random_txt
    assert "0\tiid1\t3.00000012e-01" in random_txt

    random_comp_txt = (tmp_path / "effects" / "demo.random_effect_components.tsv").read_text()
    assert "component_000" in random_comp_txt
    assert "component_001" in random_comp_txt

    snp0_txt = (tmp_path / "effects" / "demo.snp_effects.component_000.tsv").read_text()
    assert "global_snp_index" in snp0_txt
    assert "0\t0\t0\t9.99999978e-03" in snp0_txt

    meta = json.loads((tmp_path / "effects" / "demo.effect_metadata.json").read_text())
    assert meta["n_components"] == 2
    assert meta["n_samples"] == 2
    assert meta["pcg_iters"] == 7
    assert meta["components"][0]["component_name"] == "component_000"
    assert os.path.exists(paths["fixed_effects"])
    assert os.path.exists(paths["effect_metadata"])
    assert os.path.exists(paths["snp_effects_component_001"])


def test_write_effect_outputs_validates_sample_ids(tmp_path):
    effects = EffectEstimates(
        fixed_effects=jnp.asarray([], dtype=jnp.float32),
        random_effect=jnp.asarray([0.1], dtype=jnp.float32),
        random_effect_components=(jnp.asarray([0.1], dtype=jnp.float32),),
        snp_effects=(jnp.asarray([0.01], dtype=jnp.float32),),
        pcg_rel_res=1e-3,
        pcg_iters=3,
        y_mean=0.0,
        y_scale=1.0,
    )
    try:
        write_effect_outputs(
            out_prefix=str(tmp_path / "bad" / "demo"),
            effects=effects,
            sample_ids=[],
        )
    except ValueError as e:
        assert "sample_ids length mismatch" in str(e)
    else:
        raise AssertionError("write_effect_outputs should validate sample_ids length")


def test_write_effect_outputs_can_emit_source_snp_indices(tmp_path):
    effects = EffectEstimates(
        fixed_effects=jnp.asarray([], dtype=jnp.float32),
        random_effect=jnp.asarray([0.2], dtype=jnp.float32),
        random_effect_components=(jnp.asarray([0.2], dtype=jnp.float32),),
        snp_effects=(jnp.asarray([0.01, -0.03], dtype=jnp.float32),),
        pcg_rel_res=1e-3,
        pcg_iters=4,
        y_mean=0.0,
        y_scale=1.0,
    )

    out_prefix = str(tmp_path / "effects_src" / "demo")
    write_effect_outputs(
        out_prefix=out_prefix,
        effects=effects,
        sample_ids=["iid1"],
        component_source_variant_indices=[[3, 7]],
    )

    snp_txt = (tmp_path / "effects_src" / "demo.snp_effects.component_000.tsv").read_text()
    assert "source_snp_index" in snp_txt
    assert "0\t0\t3\t9.99999978e-03" in snp_txt
    assert "0\t1\t7\t-2.99999993e-02" in snp_txt


def test_write_effect_outputs_can_emit_component_metadata(tmp_path):
    effects = EffectEstimates(
        fixed_effects=jnp.asarray([], dtype=jnp.float32),
        random_effect=jnp.asarray([0.2], dtype=jnp.float32),
        random_effect_components=(jnp.asarray([0.2], dtype=jnp.float32),),
        snp_effects=(jnp.asarray([0.01, -0.03], dtype=jnp.float32),),
        pcg_rel_res=1e-3,
        pcg_iters=4,
        y_mean=0.0,
        y_scale=1.0,
    )

    out_prefix = str(tmp_path / "effects_meta" / "demo")
    write_effect_outputs(
        out_prefix=out_prefix,
        effects=effects,
        sample_ids=["iid1"],
        component_source_variant_indices=[[3, 7]],
        component_names=["ld_low"],
        component_annotations=[{"ld_bin": "low"}],
        component_provenance=[{"source": "simulation"}],
    )

    snp_txt = (tmp_path / "effects_meta" / "demo.snp_effects.component_000.tsv").read_text()
    assert "component_name" in snp_txt
    assert "0\tld_low\t0\t3\t9.99999978e-03" in snp_txt

    meta = json.loads((tmp_path / "effects_meta" / "demo.effect_metadata.json").read_text())
    assert meta["components"][0]["component_name"] == "ld_low"
    assert meta["components"][0]["annotation"] == {"ld_bin": "low"}
    assert meta["components"][0]["provenance"] == {"source": "simulation"}
