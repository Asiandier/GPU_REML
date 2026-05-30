from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
PGEN_PREFIX = os.path.join(DATA_DIR, "ukb22828_c1_b0_v3.n1000_p5000_simple")
pytestmark = pytest.mark.skipif(
    not os.path.exists(PGEN_PREFIX + ".pgen"),
    reason="local genotype test data are not included in the public repository",
)

PKG = os.path.basename(REPO_ROOT)
GWAS = importlib.import_module(f"{PKG}.gwas")
GWAS_PIPELINE = importlib.import_module(f"{PKG}.run_gwas_pipeline")
REML_MODEL = importlib.import_module(f"{PKG}.reml_model")
GENO_SOURCE = importlib.import_module(f"{PKG}.geno_source")

FitConfig = REML_MODEL.FitConfig
InfinitesimalREMLFitter = REML_MODEL.InfinitesimalREMLFitter
PgenGenoSource = GENO_SOURCE.PgenGenoSource


def _reference_stats(Z: np.ndarray, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Q, rank = GWAS._orthonormal_covariate_basis(X.astype(np.float32, copy=False))
    y_perp = y.astype(np.float32, copy=False) - Q @ (Q.T @ y.astype(np.float32, copy=False))
    y_ss = float(np.dot(y_perp.astype(np.float64), y_perp.astype(np.float64)))
    x_ty = Z.T @ y_perp
    x_t_q = Z.T @ Q
    x_var = np.sum(Z * Z, axis=0, dtype=np.float64) - np.sum(x_t_q * x_t_q, axis=1, dtype=np.float64)
    dof = int(Z.shape[0] - rank - 1)
    beta_std = x_ty / x_var
    rss = np.maximum(y_ss - beta_std * x_ty, 0.0)
    sigma2 = rss / float(dof)
    se_std = np.sqrt(sigma2 / x_var)
    t_stat = beta_std / se_std
    from scipy.stats import t as t_dist

    p_val = 2.0 * t_dist.sf(np.abs(t_stat), dof)
    return beta_std, se_std, t_stat, p_val


def test_run_continuous_gwas_matches_dense_reference(tmp_path):
    mask = np.zeros(1000, dtype=bool)
    mask[:128] = True
    source = PgenGenoSource(PGEN_PREFIX, sample_mask=mask)
    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[source],
            call_width=257,
            keep_host_stats=True,
            verbose=False,
        )
    )
    n = int(mask.sum())
    rng = np.random.RandomState(7)
    cov = rng.standard_normal((n, 2)).astype(np.float32)
    X = np.concatenate([np.ones((n, 1), dtype=np.float32), cov], axis=1)
    y = (
        0.2 * cov[:, 0]
        - 0.15 * cov[:, 1]
        + rng.standard_normal(n).astype(np.float32) * 0.5
    ).astype(np.float32)

    out_prefix = str(tmp_path / "gwas" / "demo")
    summary = GWAS.run_continuous_gwas(fitter, y, X, out_prefix=out_prefix)
    assert summary.n_samples == n
    assert summary.n_variants == 5000

    st = fitter.streamers[0]
    valid_idx = np.where(
        (np.asarray(st._inv_sds_host, dtype=np.float64) > 0.0)
        & (np.asarray(st._count_host, dtype=np.float64) > X.shape[1] + 2)
    )[0]
    idx = np.asarray(valid_idx[:5], dtype=np.int64)
    assert idx.size == 5
    Z = st.extract_standardized_columns(idx).astype(np.float64, copy=False)
    beta_std_ref, se_std_ref, t_ref, p_ref = _reference_stats(Z, y.astype(np.float64), X.astype(np.float64))
    inv = np.asarray(st._inv_sds_host[idx], dtype=np.float64)
    beta_ref = beta_std_ref * inv
    se_ref = se_std_ref * inv

    found: dict[int, dict[str, str]] = {}
    with open(summary.out_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            local_idx = int(row["variant_index_local"])
            if local_idx in set(idx.tolist()):
                found[local_idx] = row
    assert set(found) == set(idx.tolist())

    for pos, j in enumerate(idx.tolist()):
        row = found[j]
        assert row["component_name"] == os.path.basename(PGEN_PREFIX)
        assert abs(float(row["beta"]) - beta_ref[pos]) < 5e-5
        assert abs(float(row["se"]) - se_ref[pos]) < 5e-5
        assert abs(float(row["beta_std"]) - beta_std_ref[pos]) < 5e-5
        assert abs(float(row["se_std"]) - se_std_ref[pos]) < 5e-5
        assert abs(float(row["t"]) - t_ref[pos]) < 5e-4
        assert abs(float(row["p"]) - p_ref[pos]) < 5e-6
        assert int(row["n_obs"]) > 0

    meta = json.loads((tmp_path / "gwas" / "demo.gwas_metadata.json").read_text())
    assert meta["model"] == "marginal_ols_fwl"
    assert meta["n_variants"] == 5000
    assert meta["n_samples"] == n
    fitter.close()


def test_run_continuous_gwas_handles_empty_variant_iterator(tmp_path, monkeypatch):
    class _FakeStreamer:
        n = 8
        m = 0
        _means_host = np.zeros((0,), dtype=np.float32)
        _inv_sds_host = np.zeros((0,), dtype=np.float32)
        _count_host = np.zeros((0,), dtype=np.float32)
        _variant_prefix = "fake"
        _variant_format = "bed"

        def xtv(self, V, normalize=False):
            del normalize
            return jnp.zeros((0, int(V.shape[1])), dtype=jnp.float32)

    class _FakeFitter:
        _has_sparse = False
        _n_dense_streamers = 1
        streamers = (_FakeStreamer(),)

    monkeypatch.setattr(GWAS, "iter_variant_records_for_prefix", lambda prefix, fmt: iter(()))
    summary = GWAS.run_continuous_gwas(
        _FakeFitter(),
        np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        out_prefix=str(tmp_path / "gwas" / "empty"),
    )

    assert summary.n_variants == 0
    lines = (tmp_path / "gwas" / "empty.gwas.tsv").read_text().strip().splitlines()
    assert len(lines) == 1


def test_run_gwas_pipeline_passes_device_to_fit_config(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class _FakeFitter:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def close(self):
            return None

    monkeypatch.setattr(GWAS_PIPELINE, "InfinitesimalREMLFitter", _FakeFitter)
    monkeypatch.setattr(GWAS_PIPELINE, "setup_gpu", lambda: (None, None, None))
    monkeypatch.setattr(GWAS_PIPELINE, "resolve_cpu_threads", lambda explicit=None: (2, "explicit"))
    monkeypatch.setattr(
        GWAS_PIPELINE,
        "load_pheno_covar_aligned",
        lambda **kwargs: (
            np.arange(4, dtype=np.float32),
            np.ones((4, 1), dtype=np.float32),
            ["i1", "i2", "i3", "i4"],
            [],
        ),
    )
    monkeypatch.setattr(
        GWAS_PIPELINE,
        "compute_sample_mask",
        lambda fam_path, keep_ids: np.ones((len(keep_ids),), dtype=bool),
    )
    monkeypatch.setattr(
        GWAS_PIPELINE,
        "run_continuous_gwas",
        lambda fitter, y, X, out_prefix: SimpleNamespace(
            n_samples=int(y.shape[0]),
            n_variants=0,
            dof=int(y.shape[0]) - int(X.shape[1]),
            out_path=out_prefix + ".gwas.tsv",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gwas_pipeline.py",
            "--bed-prefix",
            str(tmp_path / "fake_bed"),
            "--pheno-txt",
            str(tmp_path / "fake.pheno"),
            "--out-prefix",
            str(tmp_path / "gwas" / "demo"),
            "--device",
            "cpu",
        ],
    )

    GWAS_PIPELINE.main()

    assert captured["cfg"].device == "cpu"
