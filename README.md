# GPU_REML

GPU_REML is a GPU-accelerated statistical framework for SNP heritability
estimation, genetic-variance decomposition, and downstream mixed-model
inference at biobank scale.

The central statistical problem is restricted maximum likelihood (REML)
estimation in linear mixed models where the genetic covariance is defined by one
or more genomic relationship matrices (GRMs). These models are the standard
language for estimating SNP heritability and asking how heritable signal is
distributed across chromosomes, annotations, MAF bins, LD environments, or
user-defined genomic regions.

In the standard formulation, GPU_REML fits a linear mixed model

$$
\begin{aligned}
y &= X\beta + u_1 + \cdots + u_G + e, \\
u_g &\sim \mathcal{N}(0, \sigma_g^2 K_g), \\
e &\sim \mathcal{N}(0, \sigma_e^2 I).
\end{aligned}
$$

where each `K_g` is a genotype-defined covariance component. With components
normalized to a comparable per-sample scale, SNP heritability is estimated from
the fitted variance components, for example

$$
h^2 =
\frac{\sigma_1^2 + \cdots + \sigma_G^2}
{\sigma_1^2 + \cdots + \sigma_G^2 + \sigma_e^2}
$$

and the individual `sigma_g^2` terms describe how genetic variance is allocated
across the chosen genomic components.

The computational obstacle is that the natural GRM representation is dense:
constructing, storing, and repeatedly factorizing `n x n` kernels becomes the
bottleneck as cohorts, marker counts, and component counts increase. GPU_REML
therefore keeps the statistical REML model but changes how each `K_g` is applied
numerically. Instead of materializing a GRM, each covariance component is
represented as a matrix-free genotype operator:

$$
K_g v = \frac{Z_g (Z_g^T v)}{m_{\mathrm{eff},g}}
$$

Genotype blocks are decoded on the host, streamed to the GPU, and multiplied in
batches. REML evaluation is then built from block PCG solves,
Hutchinson trace estimates, SLQ log-determinant estimates, constrained
AI/Fisher updates, and projected-core preconditioning rather than dense GRM
linear algebra.

The goal is not only to produce one whole-genome heritability number. GPU_REML is
designed as a method-development workbench for comparing covariance
representations: **single-GRM models, multi-GRM models, and weighted kernels
that encode local SNP covariance or effect-correlation structure**.

The repository also includes a SMILE-inspired weighted-GRM path, with explicit
attribution to the original [JianqiaoWang/SMILE](https://github.com/JianqiaoWang/SMILE)
project. GPU_REML implements the part that matches its matrix-free REML engine:
a block-diagonal SNP-space weight matrix `W`, evaluated without materializing
the sample-space kernel:

$$K_g=\frac{X_gW_gX_g^T}{c_g},\quad W_g=\mathrm{blockdiag}(W_{g,1},\ldots,W_{g,B}),\quad c_g=\frac{\mathrm{tr}(X_gW_gX_g^T)}{n}$$

Each `W_{g,i}` is treated as an arbitrary dense block. Blocks inside one GRM are
summed into one variance component; multiple GRM groups can be supplied when a
multi-component REML model is desired.

## Scientific Problem

Large genotype cohorts make it possible to study genetic architecture in more
detail than a single genome-wide random effect. In practice, however, richer
covariance models are still expensive to fit repeatedly. This limits routine
comparison of questions such as:

- How much SNP heritability is explained by different genomic regions,
  annotations, LD environments, or MAF bins?
- How do alternative GRM definitions change variance-component estimates,
  random effects, SNP effects, or held-out prediction?
- Can weighted SNP-space kernels be tested without constructing an explicit
  dense sample-by-sample GRM?

GPU_REML is built for this regime: repeated REML fitting and inspection of
genotype-defined covariance models while keeping the genotype data in its native
streamed matrix form.

## Core Ideas

The framework is organized around a small number of design choices:

- **Matrix-free REML.** GRMs are operators, not stored dense matrices.
- **GPU-oriented streaming.** BED/PGEN genotype blocks are packed on the CPU
  and multiplied on the GPU with tunable call width and memory budget.
- **Flexible variance decomposition.** A model can use one GRM, several input
  GRMs, contiguous SNP blocks, or arbitrary SNP-index components from a single
  genotype file.
- **Block-diagonal weighted kernels.** The SMILE path supports weighted
  covariance terms of the form `sum_i Z_i W_i Z_i.T`, evaluated without forming
  an `n x n` kernel and normalized by the exact genotype-stream trace.
- **Sparse acceleration path.** Experimental sparse genotype streams are
  available for workflows where event-style storage is more efficient than dense
  genotype blocks.
- **Numerical scalability.** PCG solves, Hutchinson traces, SLQ log-determinants,
  and low-rank projected-core preconditioning are exposed as first-class
  algorithmic controls.
- **End-to-end inference.** Fitted covariance models can be reused for fixed
  effects, random effects, SNP effects, prediction, GWAS utilities, and sparse
  REML/LASSO experiments.
- **Research transparency.** The repository exposes intermediate choices,
  convergence behavior, component metadata, effect estimates, prediction paths,
  and sparse-model diagnostics instead of hiding the fit behind a black box.

GPU_REML is not meant to replace PLINK, REGENIE, SAIGE, GCTA, or BOLT-LMM. It is
closer to a method-development and large-scale REML workbench: use established
tools for standard production workflows, and use GPU_REML when the question is
"how should this genotype-defined covariance be represented, partitioned,
preconditioned, and estimated at scale?"

## Statistical Model

GPU_REML fits Gaussian linear mixed models with genotype-defined covariance
components:

$$
\begin{aligned}
y &= X\beta + u_1 + \cdots + u_G + e, \\
u_g &\sim \mathcal{N}(0, \theta_g K_g), \\
e &\sim \mathcal{N}(0, \theta_e I), \\
K_g &= \frac{Z_g Z_g^T}{m_{\mathrm{eff},g}}, \\
H(\theta) &= \sum_g \theta_g K_g + \theta_e I.
\end{aligned}
$$

The REML objective is evaluated without explicitly forming `K_g`, `H^-1`, or the
REML projection matrix. Component operators supply the products needed by the
score equations, average-information updates, log-determinant estimator, and
preconditioner.

The public API and command-line tools support:

- single-GRM and multi-GRM REML;
- partitioned heritability from one genotype file;
- SMILE-style block-diagonal weighted GRMs;
- PLINK1 BED/BIM/FAM and PLINK2 PGEN/PVAR/PSAM inputs;
- dense genotype streams and experimental sparse genotype streams;
- post-REML fixed, random, SNP-effect, and prediction outputs;
- continuous-trait marginal GWAS utilities;
- sparse REML plus weighted LASSO with global KKT checks.

See [docs/mathematical_overview.md](docs/mathematical_overview.md) for the REML
objective, score equations, randomized estimators, and preconditioner structure.

## Research Use Cases

GPU_REML is a good fit when you want to:

- estimate SNP heritability while avoiding explicit dense GRM construction;
- compare one-GRM and multi-GRM REML fits on the same cohort;
- decompose variance by chromosome, annotation, MAF bin, or custom SNP sets;
- test genetic architecture hypotheses by changing the covariance
  representation while holding phenotype, covariates, and sample filters fixed;
- fit block-diagonal weighted-GRM models where dense `W_i` blocks encode local
  SNP covariance or effect-correlation structure;
- use the experimental sparse path for genotype settings where event-style
  storage is computationally advantageous;
- experiment with SLQ, Hutchinson, PCG, and preconditioning settings;
- inspect component-level random effects, SNP effects, prediction outputs, and
  convergence diagnostics after fitting.

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

SMILE-style block-diagonal weighted GRM:

```bash
gpu-reml \
  --smile \
  --bed-prefix /path/to/data \
  --w-files W_block_1.npy,W_block_2.npy,W_block_3.npy \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/smile
```

Multiple weighted GRMs are supplied as semicolon-separated groups. Blocks within
a group are summed into one GRM, and each group receives its own variance
component:

```bash
gpu-reml \
  --smile \
  --bed-prefix /path/to/data \
  --grm-groups 'A_1.npy,A_2.npy;B_1.npy,B_2.npy' \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/smile_multi
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

## SMILE-Style Weighted GRMs

The SMILE-related path is inspired by the original
[JianqiaoWang/SMILE](https://github.com/JianqiaoWang/SMILE) R project. The
original repository provides SMILE estimation routines and a genetic application
interface using genotype data with a specified weight matrix `W` or block
division.

GPU_REML does not vendor or reimplement the full R package. Instead, it
implements the part that fits naturally into GPU_REML's operator-based REML
architecture: a matrix-free weighted kernel with block-diagonal `W`.

For one weighted GRM, the model component is:

$$
K V =
\frac{Z_1 W_1 Z_1^T V + \cdots + Z_B W_B Z_B^T V}{c},
\qquad
c =
\frac{\mathrm{tr}\left(
Z_1 W_1 Z_1^T + \cdots + Z_B W_B Z_B^T
\right)}{n}
$$

The normalization is computed exactly for the current genotype stream,
standardization, and sample filter. It is not approximated by `tr(W)` and does
not require precomputed trace metadata. The implementation never materializes
the `n x n` kernel; it streams genotype products, applies each dense `W_i` in
SNP space, and then streams the result back through `Z_i`.

For multi-GRM SMILE models, `--grm-groups` defines the grouping:

```text
--grm-groups 'W_1,W_2;W_3,W_4,W_5'
```

This creates two variance components:

$$
\begin{aligned}
K_1 &=
\frac{Z_1 W_1 Z_1^T + Z_2 W_2 Z_2^T}{c_1}, \\
K_2 &=
\frac{Z_3 W_3 Z_3^T + Z_4 W_4 Z_4^T + Z_5 W_5 Z_5^T}{c_2}.
\end{aligned}
$$

This path is intended for method development around local LD, block-wise
effect-correlation models, and weighted covariance structures. Large dense
`W_i` blocks are computationally expensive: exact trace initialization and each
`W_i @ (Z_i.T V)` operation scale with the block width, so practical block sizes
should be chosen with GPU memory and runtime in mind.

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

$$
H(\theta)V = \theta_e V + \sum_g \theta_g K_g V
$$

The implementation builds this product from streamed genotype blocks. REML
evaluation then combines:

- block PCG solves for `H^-1 [X | y | random probes]`;
- Hutchinson probes for trace terms in the score;
- stochastic Lanczos quadrature for `log|H|`;
- projected Fisher / AI-style variance-component updates with nonnegative
  genetic-variance constraints;
- a projected-core preconditioner `dI + U C(theta) U.T` that captures leading
  covariance structure and can also support residual SLQ.

For SMILE-style weighted kernels, the same REML loop is reused after replacing
the standard GRM operator by the block-diagonal weighted operator. This keeps the
new covariance representation isolated from the ordinary single-GRM, multi-GRM,
partitioned, and sparse paths.

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
- For SMILE-style weighted models, start with small `W_i` blocks and
  `--no-w-psd-check` only when the supplied matrices are already known to be
  symmetric positive semidefinite.
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
streaming, PGEN/BED sources, partitioned GRMs, SMILE-style weighted GRMs, sparse
streams, effect estimation, prediction, GWAS, CLI behavior, and packaging.

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
