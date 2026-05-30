from __future__ import annotations

from pathlib import Path


def _pyproject_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "pyproject.toml").read_text()


def test_pyproject_packages_root_gpu_reml_package():
    text = _pyproject_text()

    assert "[tool.setuptools]" in text
    assert "packages = [\"GPU_REML\"]" in text
    assert "package-dir = { \"GPU_REML\" = \".\" }" in text


def test_pyproject_exposes_cli_entrypoints():
    text = _pyproject_text()

    assert "[project.scripts]" in text
    assert "gpu-reml = \"GPU_REML.run_reml_pipeline:main\"" in text
    assert "gpu-reml-sparse = \"GPU_REML.run_sparse_reml_pipeline:main\"" in text
    assert "gpu-reml-gwas = \"GPU_REML.run_gwas_pipeline:main\"" in text
