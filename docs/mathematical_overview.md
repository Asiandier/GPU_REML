# Mathematical Overview

GPU_REML fits Gaussian linear mixed models with one or more genotype-defined
similarity components:

```text
y = X beta + u_1 + ... + u_G + e
u_g ~ N(0, theta_g K_g)
e   ~ N(0, theta_e I)
H(theta) = sum_g theta_g K_g + theta_e I
```

For a standardized genotype matrix `Z_g`, the component GRM is represented as:

```text
K_g = Z_g Z_g^T / m_eff,g
```

The implementation never needs to materialize `K_g`; it only needs the product:

```text
K_g V = Z_g (Z_g^T V) / m_eff,g
```

## SMILE-Style Weighted Kernels

GPU_REML also includes a SMILE-inspired weighted kernel path. The implementation
is inspired by the original
[JianqiaoWang/SMILE](https://github.com/JianqiaoWang/SMILE) R project, but the
form implemented here is specialized to GPU_REML's matrix-free REML engine.

For one weighted GRM, `W` is represented as a block diagonal matrix:

```text
W = blockdiag(W_1, ..., W_B)
```

with corresponding contiguous genotype blocks:

```text
Z = [Z_1, ..., Z_B]
```

The genetic covariance component is:

```text
K = (sum_i Z_i W_i Z_i^T) / c
```

where the normalizer is computed from the current standardized genotype stream:

```text
c = tr(sum_i Z_i W_i Z_i^T) / n
  = (1/n) sum_i tr(Z_i W_i Z_i^T)
```

This gives:

```text
tr(K) = n
```

The computation never forms the `n x n` matrix. For any right-hand side `V`:

```text
K V = (sum_i Z_i [W_i (Z_i^T V)]) / c
```

Blocks within one SMILE GRM are computational terms inside one variance
component. Multiple SMILE GRMs are represented as groups of blocks:

```text
K_g = (sum_i Z_{g,i} W_{g,i} Z_{g,i}^T) / c_g
H(theta) = sum_g theta_g K_g + theta_e I
```

Each group receives its own genetic variance component `theta_g`.

## REML Objective

Let:

```text
P = H^-1 - H^-1 X (X^T H^-1 X)^-1 X^T H^-1
```

Ignoring constants, the restricted log-likelihood is:

```text
ell(theta) = -1/2 [ y^T P y + log|H| + log|X^T H^-1 X| ]
```

GPU_REML reports and optimizes a scaled version of this objective.

## Score and Average Information

For variance component `theta_i` with derivative matrix `K_i`, where the
residual component uses `K_e = I`:

```text
score_i = 1/2 [ y^T P K_i P y - tr(P K_i) ]
AI_ij   = 1/2 [ (K_i P y)^T P (K_j P y) ]
```

The solver uses projected Fisher / AI-REML updates with nonnegative genetic
variance constraints. Genetic components at the boundary are frozen according
to the KKT condition.

## Randomized Estimation

For large biobank datasets, exact traces and log determinants are too
expensive. GPU_REML uses:

- Hutchinson probes for trace terms,
- stochastic Lanczos quadrature for `log|H|`,
- block PCG solves for `H^-1` applications.

These approximations are controlled by parameters such as `n_rand_vec`,
`slq_samples`, `slq_m`, and the PCG tolerance.

## Projected-Core Preconditioning

The projected-core preconditioner approximates the leading spectral structure:

```text
M(theta) = d I + U C(theta) U^T
C(theta) = sum_g theta_g C_g
```

where `U` is a low-rank basis estimated by a randomized Nystrom-style sketch.
For orthonormal `U`:

```text
M^-1 v = d^-1 v + U [ (dI + C)^-1 - d^-1 I ] U^T v
```

This improves PCG convergence and can also be used as a residual basis for SLQ:

```text
log|H| = log|M| + log|M^-1/2 H M^-1/2|
```
