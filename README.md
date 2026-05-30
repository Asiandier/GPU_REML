# GPU_REML

GPU_REML is a JAX-based toolkit for large-scale genomic variance-component
analysis. It fits single-GRM and multi-GRM infinitesimal linear mixed models
without materializing dense genomic relationship matrices, using streamed
genotype matrix-vector products, block PCG solves, stochastic log-determinant
estimation, and projected-core preconditioning.

The package is designed for biobank-scale REML workflows where the statistical
model is expressed through one or more genotype-defined similarity components:

```text
y = X beta + u_1 + ... + u_G + e
u_g ~ N(0, theta_g K_g)
e   ~ N(0, theta_e I)

K_g = Z_g Z_g^T / m_eff,g
H(theta) = sum_g theta_g K_g + theta_e I
```

## Highlights

- Streaming `K @ V` products for PLINK1 BED and PLINK2 PGEN inputs.
- Dense common-variant GRMs and sparse rare-variant streams.
- Single-GRM, multi-GRM, contiguous SNP block, and arbitrary component
  partitions.
- AI-REML / Fisher scoring with nonnegative genetic variance constraints.
- Block PCG for multiple right-hand sides.
- Stochastic Lanczos quadrature (SLQ) for `log|H|`.
- Projected-core low-rank preconditioner for faster PCG and residual SLQ.
- Optional fixed/random/SNP effect output and prediction.
- Continuous-trait marginal GWAS utilities.
- Sparse REML + weighted LASSO utilities with global KKT checks.

## Installation

Install from a local checkout:

```bash
python -m pip install /path/to/GPU_REML
```

Optional PGEN support requires `pgenlib`:

```bash
python -m pip install "/path/to/GPU_REML[pgen]"
```

For development:

```bash
python -m pip install "/path/to/GPU_REML[dev]"
```

GPU runs require a JAX/JAXLIB installation compatible with the local CUDA
driver. CPU execution works for tests and small examples, but large REML jobs
are intended for GPU execution.

## Command Line

### REML

Single-GRM BED input:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/reml \
  --verbose
```

PGEN input:

```bash
gpu-reml \
  --pgen-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/reml
```

Multi-GRM from multiple BED prefixes:

```bash
gpu-reml \
  --bed-prefix /path/to/grm1,/path/to/grm2,/path/to/grm3 \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt
```

Single genotype file partitioned into variance components:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --component-spec components.npz \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt
```

`--component-spec` accepts `.npz` or `.json` component definitions. Each
component contains zero-based variant indices in source genotype order.

### Sparse REML + LASSO

```bash
gpu-reml-sparse \
  --bed-prefix /path/to/common \
  --component-spec components.npz \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/sparse \
  --kkt-check
```

### Continuous GWAS

```bash
gpu-reml-gwas \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/gwas
```

The repository-local `run_gpu.sh` launcher remains available for
environment-heavy benchmark runs.

## Inputs

Genotype inputs:

- PLINK1 BED/BIM/FAM prefix via `--bed-prefix`.
- PLINK2 PGEN/PVAR/PSAM prefix via `--pgen-prefix`.
- Multiple comma-separated BED prefixes for separate GRM components.
- Optional rare-variant inputs for sparse workflows.

Phenotype and covariate inputs:

- `--pheno-txt`: phenotype table aligned by FID/IID when used through the
  pipeline scripts.
- `--covar-txt`: fixed-effect covariate table. Typical REML runs include an
  intercept and ancestry PCs.
- `--keep-path`: optional sample keep file.

Component inputs:

- `--vc-block-sizes`: contiguous block sizes for single-file multi-GRM fits.
- `--component-spec`: structured `.npz` or `.json` component definitions.
- `--component-indices-npz`: legacy component-index format.

## Outputs

The REML pipeline prints variance components and total heritability to stdout:

```text
var_components: [...]
h2: ...
```

When `--compute-effects` and `--out-prefix` are provided, the pipeline writes
fixed effects, random effects, SNP effects, and prediction-compatible output
files.

GWAS output is a TSV with per-variant marginal effect estimates, standard
errors, test statistics, p-values, allele frequency, and variant metadata.

## Python API

```python
import jax.numpy as jnp
from GPU_REML import FitConfig, InfinitesimalREMLFitter

cfg = FitConfig(
    bed_prefix="/path/to/data",
    n_rand_vec=100,
    minq_iter=10,
    slq_samples=4,
    slq_m=8,
    precond_rank=500,
    verbose=True,
)

fitter = InfinitesimalREMLFitter(cfg)
result = fitter.fit_infinitesimal(
    y=jnp.asarray(y),
    covar=jnp.asarray(covar),
)
print(result.var_components)
```

For lower-level use, `fit_reml` accepts a list of `K @ V` callables and diagonal
atoms, allowing custom similarity operators.

## Algorithmic Notes

GPU_REML avoids explicit GRM construction. For each component, the core
operation is:

```text
K_g @ V = Z_g @ (Z_g.T @ V) / m_eff,g
```

Genotype blocks are packed on the host, streamed to the accelerator, unpacked,
mean-imputed, standardized, multiplied, and accumulated. REML evaluation uses
PCG solves for `H^{-1}[X | y | random probes]`, Hutchinson trace estimates for
score terms, and SLQ for `log|H|`. A projected-core preconditioner approximates
the leading spectral structure with `dI + U C U^T`.

The implementation is optimized for large `n` and `m`; for small datasets,
fixed overheads from JAX compilation and Python-level streaming may dominate.

## Validation

Run the test suite:

```bash
python -m pytest -q
```

Build a wheel:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/gpu_reml_wheel /path/to/GPU_REML
```

The repository includes tests for REML updates, PCG/preconditioning, genotype
streaming, PGEN/BED sources, partitioned GRMs, sparse streams, effect
estimation, prediction, GWAS, CLI behavior, and packaging.

Large local genotype fixtures under `tests/data/` are intentionally not
included in the public repository. Tests that require those fixtures are skipped
when the files are absent.

## Practical Guidance

- Use `--gpu-budget-gib` to keep planner choices inside available memory.
- Increase `--slq-samples` and `--slq-m` for more stable log-determinant
  estimates when final numerical accuracy matters.
- Increase `--n-rand-vec` for lower-variance Hutchinson trace estimates.
- For high-dimensional multi-GRM models, monitor boundary components and
  convergence history.
- Keep genotype, phenotype, covariate, and component definitions fixed when
  comparing against GCTA or other REML software.

## Limitations

- REML likelihood terms use randomized approximations; results can vary with
  seed and SLQ/Hutchinson settings.
- The package focuses on continuous traits.
- GPU performance depends on JAX/CUDA versions, PCIe bandwidth, call width,
  sample size, SNP count, and component count.
- No public license has been selected yet. Do not redistribute until the
  project owner adds an explicit license.

## Project Status

This codebase is research software. It has an automated test suite and has
been used in large-scale internal experiments, but users should validate
settings against small exact references or matched external software before
using it for production scientific conclusions.
