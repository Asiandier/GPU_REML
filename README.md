# GPU_REML

Matrix-free GPU REML for genotype-defined variance components.

GPU_REML is a research toolkit for estimating SNP heritability and
variance-component models from large genotype cohorts. It targets a specific
pain point in genomic REML: the statistical model is naturally written in terms
of one or more genomic relationship matrices (GRMs), but explicitly building
and factorizing dense `n x n` GRMs becomes the bottleneck as cohorts grow.

GPU_REML keeps the REML model, but changes the computation. Each GRM is
represented by streamed genotype matrix products:

```text
K_g V = Z_g (Z_g.T V) / m_eff,g
```

where genotype blocks are decoded on the host, streamed to the accelerator, and
multiplied in batches. The REML solver then uses block PCG, stochastic trace and
log-determinant estimates, and a projected-core preconditioner instead of dense
linear algebra on an explicit GRM.

The result is a Python/JAX codebase for experiments where the object of
interest is not only a single SNP-heritability number, but a flexible variance
decomposition across chromosomes, annotations, MAF bins, common/rare variant
sets, or custom SNP partitions.

## Why This Project Exists

Many genomics tools are excellent at association testing, QC, or fixed sets of
mixed-model workflows. GPU_REML is narrower: it is built for researchers who
want to fit and inspect REML variance-component models while keeping the
genotype data in its native matrix form.

The design priorities are:

- **Matrix-free REML.** GRMs are operators, not stored dense matrices.
- **GPU-oriented streaming.** BED/PGEN genotype blocks are packed on the CPU
  and multiplied on the accelerator with tunable call width and memory budget.
- **Flexible variance decomposition.** A model can use one GRM, several input
  GRMs, contiguous SNP blocks, or arbitrary SNP-index components from a single
  genotype file.
- **Numerical scalability.** PCG solves, Hutchinson traces, SLQ log-determinants,
  and low-rank projected-core preconditioning are exposed as first-class
  algorithmic controls.
- **Research visibility.** The repository exposes intermediate choices,
  convergence behavior, component metadata, effect estimates, prediction paths,
  and sparse-model diagnostics instead of hiding the fit behind a black box.

GPU_REML is not meant to replace PLINK, REGENIE, SAIGE, GCTA, or BOLT-LMM. It is
closer to a method-development and large-scale REML workbench: use established
tools for standard production workflows, and use GPU_REML when the question is
"how should this genotype-defined covariance be represented, partitioned,
preconditioned, and estimated at scale?"

## Model

GPU_REML fits Gaussian linear mixed models with genotype-defined covariance
components:

```text
y = X beta + u_1 + ... + u_G + e
u_g ~ N(0, theta_g K_g)
e   ~ N(0, theta_e I)

K_g = Z_g Z_g.T / m_eff,g
H(theta) = sum_g theta_g K_g + theta_e I
```

The public API and command-line tools support:

- single-GRM and multi-GRM REML;
- partitioned heritability from one genotype file;
- PLINK1 BED/BIM/FAM and PLINK2 PGEN/PVAR/PSAM inputs;
- common-variant dense streams and sparse rare-variant streams;
- post-REML fixed, random, SNP-effect, and prediction outputs;
- continuous-trait marginal GWAS utilities;
- sparse REML plus weighted LASSO with global KKT checks.

See [docs/mathematical_overview.md](docs/mathematical_overview.md) for the REML
objective, score equations, randomized estimators, and preconditioner structure.

## When To Use It

GPU_REML is a good fit when you want to:

- estimate SNP heritability without materializing a dense GRM;
- compare one-GRM and multi-GRM REML fits on the same cohort;
- decompose variance by chromosome, annotation, MAF bin, or custom SNP sets;
- combine dense common-variant covariance with sparse rare-variant components;
- experiment with SLQ, Hutchinson, PCG, and preconditioning settings;
- inspect component-level random effects or SNP effects after fitting.

It is a poor fit when you need a complete genotype QC pipeline, binary-trait
mixed models, turn-key cloud orchestration, or a polished production CLI with
stable long-term output contracts.

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

## Quick Start

Single-GRM REML from PLINK1 BED:

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

Multiple GRMs from multiple BED prefixes:

```bash
gpu-reml \
  --bed-prefix /path/to/grm1,/path/to/grm2,/path/to/grm3 \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt
```

Arbitrary SNP components from one genotype file:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --component-spec components.json \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/partitioned
```

Sparse REML plus LASSO:

```bash
gpu-reml-sparse \
  --bed-prefix /path/to/common \
  --component-spec components.json \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/sparse \
  --kkt-check
```

Continuous-trait marginal GWAS:

```bash
gpu-reml-gwas \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/gwas
```

The repository-local `run_gpu.sh` launcher remains available for
environment-heavy benchmark runs.

## Component Specifications

Component specs define how SNPs are grouped into GRM components. A JSON spec can
name components and carry metadata:

```json
{
  "components": [
    {
      "name": "maf_0_01",
      "variant_indices": [0, 4, 9],
      "annotation": {"maf_bin": "0-1%"}
    },
    {
      "name": "maf_01_05",
      "variant_indices": [1, 2, 8],
      "annotation": {"maf_bin": "1-5%"}
    }
  ]
}
```

NPZ specs are also supported for compact programmatic construction. See
[docs/component_specs.md](docs/component_specs.md).

## Outputs

The REML pipeline prints estimated variance components and total heritability:

```text
var_components: [...]
h2: ...
```

With `--compute-effects` and `--out-prefix`, GPU_REML writes:

- fixed-effect estimates;
- per-sample random effects;
- component-level random effects;
- SNP-effect tables for each component;
- JSON metadata linking outputs to component definitions.

Prediction inputs can reuse fitted effects on a matched test genotype source.
GWAS output is a TSV with marginal effect estimates, standard errors, test
statistics, p-values, allele frequency, and variant metadata.

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

Lower-level users can call `fit_reml` with custom `K @ V` operators and diagonal
atoms. This makes it possible to prototype new covariance representations
without rewriting the REML optimizer.

## How It Works

At each REML step, GPU_REML needs repeated applications of:

```text
H(theta) V = theta_e V + sum_g theta_g K_g V
```

The implementation builds this product from streamed genotype blocks. REML
evaluation then combines:

- block PCG solves for `H^-1 [X | y | random probes]`;
- Hutchinson probes for trace terms in the score;
- stochastic Lanczos quadrature for `log|H|`;
- projected Fisher / AI-style variance-component updates with nonnegative
  genetic-variance constraints;
- a projected-core preconditioner `dI + U C(theta) U.T` that captures leading
  covariance structure and can also support residual SLQ.

This is why most tuning parameters control either memory movement
(`--call-width`, `--gpu-budget-gib`, `--ring-depth`) or randomized numerical
accuracy (`--n-rand-vec`, `--slq-samples`, `--slq-m`, `--pcg-tol`,
`--precond-rank`).

## Practical Guidance

- Start with one GRM and modest randomized settings, then increase
  `--n-rand-vec`, `--slq-samples`, and `--slq-m` for final runs.
- Use `--gpu-budget-gib` when sharing a GPU or when JAX memory preallocation is
  undesirable.
- Keep phenotype, covariate, sample filtering, and component definitions fixed
  when comparing against GCTA, BOLT-REML, or other REML software.
- For high-dimensional multi-GRM models, inspect boundary components and
  convergence history rather than relying only on the final `h2`.
- Treat sparse REML plus LASSO as an experimental workflow; use `--kkt-check`
  when interpreting selected sparse effects.

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

Large local genotype fixtures under `tests/data/` are intentionally not included
in the public repository. Tests that require those fixtures are skipped when the
files are absent.

## Limitations

- REML likelihood terms use randomized approximations; results can vary with
  seed and SLQ/Hutchinson settings.
- The package currently focuses on continuous traits.
- GPU performance depends on JAX/CUDA versions, PCIe bandwidth, call width,
  sample size, SNP count, and component count.
- This is research software. Validate settings against small exact references or
  matched external software before using it for production scientific
  conclusions.
- No public license has been selected yet. Do not redistribute until the project
  owner adds an explicit license.

