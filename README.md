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
e &\sim \mathcal{N}(0, \sigma_e^2 I), \\
V(\theta) &= \sum_g \theta_g K_g + \theta_e I.
\end{aligned}
$$

The restricted log likelihood is

$$\ell_R(\theta)=-\frac{1}{2}\left[\log|V(\theta)|+\log|X^TV(\theta)^{-1}X|+y^TP(\theta)y\right]$$

where

$$P(\theta)=V(\theta)^{-1}-V(\theta)^{-1}X\left(X^TV(\theta)^{-1}X\right)^{-1}X^TV(\theta)^{-1}.$$

Each `K_g` is a genotype-defined covariance component. With components
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
batches. Evaluating and optimizing the REML likelihood without explicit GRMs
leads to the main numerical machinery in GPU_REML: block PCG solves, Hutchinson
trace estimates, SLQ log-determinant estimates, constrained AI/Fisher updates,
and projected-core preconditioning.

The goal is not only to produce one whole-genome heritability number. GPU_REML is
designed as a method-development workbench for comparing **single-GRM and
multi-GRM covariance representations**. It also includes a SMILE-inspired
weighted-GRM extension, with explicit attribution to the original
[JianqiaoWang/SMILE](https://github.com/JianqiaoWang/SMILE) project. This path
adapts the SMILE idea of introducing a SNP-space weight matrix `W` into the
genetic covariance, while implementing the form that matches GPU_REML's
matrix-free REML engine: a block-diagonal `W`, evaluated without materializing
the sample-space kernel:

$$K_g=\frac{X_gW_gX_g^T}{c_g},\quad W_g=\mathrm{blockdiag}(W_{g,1},\ldots,W_{g,B}),\quad c_g=\frac{\mathrm{tr}(X_gW_gX_g^T)}{n}$$

Each `W_{g,i}` is treated as an arbitrary dense block. Blocks inside one GRM are
summed into one variance component; multiple GRM groups can be supplied when a
multi-component REML model is desired.

The sparse fixed-effect path uses the fitted covariance `V(theta)` to define a
penalized GLS likelihood over candidate SNP effects:

$$
(\hat\alpha_\lambda,\hat b_\lambda)=\arg\min_{\alpha,b}\frac{1}{2}(y-C\alpha-Z_Sb)^TV(\theta)^{-1}(y-C\alpha-Z_Sb)+\lambda\|b\|_1.
$$

## Research Use Cases

GPU_REML is most useful when the scientific question requires more than a
single whole-genome GRM. It is designed to make multi-GRM REML practical by
keeping each covariance component as a streamed genotype operator rather than a
stored dense matrix. This is the main advantage of the project: users can expand
from one GRM to many GRMs while keeping wall time low through GPU batched
products and keeping CPU memory controlled by avoiding explicit `n x n` GRM
storage.

Typical use cases include:

- comparing single-GRM and multi-GRM heritability estimates on the same cohort;
- decomposing SNP heritability across chromosomes, LD environments, MAF bins,
  annotations, or custom SNP sets;
- fitting many covariance components without constructing and storing one dense
  GRM per component;
- benchmarking alternative covariance representations under matched phenotype,
  covariate, and sample filters;
- testing SMILE-style block-diagonal weighted GRMs where dense `W_i` blocks
  encode local SNP covariance or effect-correlation structure.

## Installation

GPU_REML requires Python 3.10 or newer. For large runs, install a GPU-enabled
JAX build before installing GPU_REML.

```bash
git clone https://github.com/Asiandier/GPU_REML.git
cd GPU_REML
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install JAX for the local CUDA driver following the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html). For
current NVIDIA CUDA pip wheels, the command is typically:

```bash
python -m pip install -U "jax[cuda13]"
```

Then install GPU_REML:

```bash
python -m pip install -e .
```

Optional PGEN support:

```bash
python -m pip install -e ".[pgen]"
```

Development install:

```bash
python -m pip install -e ".[dev]"
```

Check that JAX can see the GPU:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

CPU-only JAX is sufficient for tests and small examples. Large REML jobs are
intended for GPU execution.

## Quick Start

Single-GRM REML from PLINK1 BED:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/reml
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
  --covar-txt covar.txt \
  --out-prefix out/multi_grm
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

Sparse REML plus LASSO, single GRM:

```bash
gpu-reml-sparse \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/sparse_single
```

Sparse REML plus LASSO, multiple BED prefixes as multiple GRMs:

```bash
gpu-reml-sparse \
  --bed-prefix /path/to/grm1,/path/to/grm2,/path/to/grm3 \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/sparse_multi_bed
```

Sparse REML plus LASSO, one genotype file partitioned into multiple GRMs:

```bash
gpu-reml-sparse \
  --bed-prefix /path/to/data \
  --component-spec components.json \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/sparse_components
```

Continuous-trait marginal GWAS:

```bash
gpu-reml-gwas \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --out-prefix out/gwas
```

Add `--covar-txt covar.txt` when covariates should be included.

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

For routine runs, the most important user-facing resource controls are the GPU
budget and the genotype-streaming ring depth.

## Key Runtime Parameters

- `--gpu-budget-gib`: planner-side GPU memory budget. This is the main knob for
  controlling GPU memory peak. The planner uses it to choose the streamed SNP
  call width and the size of GPU-resident work arrays, including REML random
  probe blocks and projected-core state. If omitted or set to `0`, GPU_REML uses
  the currently available GPU memory estimate.
- `--ring-depth`: number of CPU-side staging buffers used for genotype
  streaming. This is the main knob for controlling CPU memory peak during data
  movement. Larger values can give smoother host-to-GPU streaming but allocate
  more pinned/staging memory on the CPU. The default `0` lets the planner choose
  a conservative value.

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
