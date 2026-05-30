from __future__ import annotations

import importlib


_EXPORTS: dict[str, str] = {
    "GenoBlockSource": ".geno_source",
    "BedGenoSource": ".geno_source",
    "PgenGenoSource": ".geno_source",
    "GenoBlockStreamer": ".geno_stream",
    "BedBlockStreamer": ".geno_stream",
    "SparseGenoBlockStreamer": ".sparse_stream",
    "DensePackedBlockDescriptor": ".block_backend",
    "Sparse12BlockDescriptor": ".block_backend",
    "ProjectedCorePrecondConf": ".precond",
    "ProjectedCoreRuntime": ".precond",
    "build_lowrank_basis": ".precond",
    "build_projected_core_runtime": ".precond",
    "make_precond": ".precond",
    "make_projected_core_precond": ".precond",
    "projected_core_apply_invsqrt": ".precond",
    "projected_core_logdet": ".precond",
    "InfinitesimalREMLFitter": ".reml_model",
    "FitConfig": ".reml_model",
    "FitResult": ".reml_model",
    "EffectEstimates": ".reml_model",
    "PredictionEstimates": ".reml_model",
    "pcg_solve": ".pcg",
    "fit_reml": ".reml",
    "CovariateTransform": ".data_utils",
    "load_pheno_covar_aligned": ".data_utils",
    "load_pheno_covar_aligned_with_transform": ".data_utils",
    "load_covar_aligned": ".data_utils",
    "suggest_call_width": ".suggest_params_v3",
    "PlanResult": ".suggest_params_v3",
    "LassoPathConfig": ".lasso_cd",
    "compute_projected_hinv_vector": ".lasso_cd",
    "fit_weighted_lasso_with_covariates": ".lasso_cd",
    "GWASSummary": ".gwas",
    "run_continuous_gwas": ".gwas",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
