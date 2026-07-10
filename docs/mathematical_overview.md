# Mathematical Overview

GPU_REML fits a single-trait Gaussian linear mixed model without materializing
an `n x n` genomic relationship matrix (GRM). The general covariance used by
the solver is

```text
y = X beta + u_1 + ... + u_G + e_1 + ... + e_E
u_g ~ N(0, theta_g K_g)
e_j ~ N(0, eta_j R_j)
H(theta, eta) = sum_g theta_g K_g + sum_j eta_j R_j
```

The standard model has one residual component, `E = 1` and `R_1 = I`.
Per-ancestry residuals use diagonal `R_j` matrices. For REML, PCG, and SLQ to
have their stated interpretation, every `K_g` and `R_j` must be symmetric
positive semidefinite and `H` must be positive definite at every evaluated
parameter vector. When several diagonal residual components are supplied, they
must jointly provide positive residual support for every sample.

## Matrix-Free Genetic Kernels

For a standardized genotype matrix `Z_g`, a standard component is

```text
K_g = Z_g Z_g^T / m_eff,g
K_g V = Z_g (Z_g^T V) / m_eff,g
```

Only the second expression is evaluated. Packed genotype blocks are decoded,
standardized, multiplied, and discarded while the result is accumulated on the
GPU. The sample-space `K_g` matrix is never constructed.

For ancestry proportions `q_a`, the admixed component implemented by
`reml_model.py` is

```text
D_a = diag(sqrt(q_a))
K_a = D_a K D_a
K_a V = D_a K (D_a V)
```

Several ancestry right-hand sides are fused into one streamed `K` pass when
the configured RHS budget permits it.

## SMILE-Style Weighted Kernels

The SMILE path represents a block diagonal SNP-space weight matrix

```text
W = blockdiag(W_1, ..., W_B)
Z = [Z_1, ..., Z_B]
K = (sum_b Z_b W_b Z_b^T) / c
```

Each `W_b` must be finite, symmetric, and positive semidefinite. GPU_REML checks
the inexpensive shape/finiteness conditions but treats symmetry and PSD as a
trusted-input contract, avoiding a cubic-time eigenvalue check on large W
blocks. Supplying an asymmetric or indefinite W invalidates the covariance,
PCG, and REML interpretation.

For any right-hand side `V`,

```text
K V = (sum_b Z_b [W_b (Z_b^T V)]) / c
```

Under `kernel_trace` normalization,

```text
c = tr(sum_b Z_b W_b Z_b^T) / n
tr(K) / n = 1
```

Under `effective_rank` normalization, `c` is the sum of supplied effective
ranks. In that mode `tr(K) / n` is generally not one. The implementation still
computes the actual mean diagonal for initialization, preconditioning, and
heritability reporting.

Blocks within one SMILE group form one `K_g`. Semicolon-separated groups form
separate variance components.

## REML Projection and Objective

For full-column-rank fixed effects `X`, define

```text
P = H^-1 - H^-1 X (X^T H^-1 X)^-1 X^T H^-1
```

Ignoring constants independent of the variance parameters,

```text
ell_R = -1/2 [y^T P y + log|H| + log|X^T H^-1 X|]
```

GPU_REML divides the objective, score, and average-information matrix by `n`.
This scaling changes neither the optimum nor the Fisher-scoring direction.
The high-level CLI always supplies an intercept. Low-level `fit_reml` callers
must include every intended fixed-effect column, including an intercept when
the model requires one.

## Score and Average Information

Let `A_i` denote either a genetic derivative `K_g` or a residual derivative
`R_j`. The exact REML score and average-information entries are

```text
s_i     = 1/2 [y^T P A_i P y - tr(P A_i)]
AI_ij   = 1/2 [(A_i P y)^T P (A_j P y)]
```

The implementation computes `Py`, applies all component operators to `Py`, and
solves `H^-1 [A_1 Py, ..., A_(G+E) Py]`. Projection of those solutions gives the
AI matrix. The small fixed-effect normal matrix is Cholesky-factorized without
jitter when it is already SPD; jitter is used only as a numerical fallback.

Variance updates solve the bound-constrained quadratic model

```text
maximize_d  s^T d - 1/2 d^T AI d
subject to  theta_g + d_g >= 0
            eta_j + d_j >= residual_floor
```

The regularized AI system is small and dense. GPU_REML Cholesky-factorizes it,
shifts the parameter step by its lower bounds, and converts the bound-constrained
quadratic subproblem exactly to nonnegative least squares (NNLS). SciPy's
compiled active-set NNLS solver then enforces primal feasibility and the
lower-bound KKT signs in one host-side solve. This avoids recompiling a separate
JAX Cholesky for every possible active-set dimension when hundreds of variance
components hit zero together. Strict mode then uses deterministic, fixed-probe
likelihood backtracking; a numerically invalid candidate is reduced and retried.
Convergence checks combine likelihood improvement with a KKT-projected gradient
or a small relative parameter step.

## Hutchinson Trace Identity

For fixed random probes `z_r` satisfying `E[z_r z_r^T] = I`,

```text
tr(P A_i) ~= (1/R) sum_r z_r^T A_i P z_r
```

For symmetric genetic kernels,

```text
z_r^T K_i P z_r = (P z_r)^T K_i z_r
```

GPU_REML therefore caches `K_i z_r` once. Every subsequent REML evaluation
needs a fresh genetic operator application only for `Py`, not for the full
`[Py | Pz_1 | ... | Pz_R]` block. This is algebraically equivalent to the
original estimator and reduces that component-operator RHS width from `R + 1`
to `1`.

## PCG and Stochastic Log Determinants

Block multi-RHS PCG supplies the `H^-1` applications. It uses periodic
convergence checks to avoid synchronizing the GPU after every iteration. A REML
evaluation is rejected with an explicit error if either the main projection
solve or the AI solve reports a non-finite or above-tolerance residual.

The log determinant is estimated with stochastic Lanczos quadrature (SLQ). The
same SLQ probes are reused across strict line-search trials, so candidate
objectives are directly comparable. `smile_scoring` may use a first-order
log-determinant update for small steps and is intentionally an approximate,
non-monotone scoring mode; `strict` always recomputes the fixed-probe SLQ
quadrature for candidates.

For the common single-GRM, identity-residual model,

```text
H(theta) = theta_g K + theta_e I.
```

Lanczos is invariant to an affine shift and scale of its operator. With fixed
starting probes, its tridiagonal matrices therefore satisfy

```text
T_H(theta) = theta_g T_K + theta_e I.
```

GPU_REML constructs `T_K` once and evaluates later candidate log determinants
from this small affine matrix. This is the same raw-SLQ quadrature as rerunning
Lanczos on `H(theta)`, up to floating-point recurrence roundoff; it is not a
Taylor approximation. Multi-GRM and non-identity residual models continue to
run operator SLQ because their covariance components do not generally commute.

## Projected-Core Preconditioning

Let `U` be an orthonormal low-rank basis and define the mean diagonal atoms

```text
a_g = tr(K_g) / n
b_j = tr(R_j) / n
C_g = U^T K_g U - a_g I
d(theta, eta) = sum_g theta_g a_g + sum_j eta_j b_j
```

The preconditioner is

```text
M = d I + U [sum_g theta_g C_g] U^T
```

Subtracting `a_g I` from each projected core is essential: the same average
diagonal contribution is already present in `d I`. For orthonormal `U`,

```text
M^-1 v = d^-1 v
         + U [(d I + C)^-1 - d^-1 I] U^T v
```

The basis is rebuilt only after an accepted step whose maximum relative
parameter change reaches `precond_refresh_reldp`. Rejected line-search trials
reuse the current preconditioner.

## Heritability and Initialization

Variance parameters are coefficients of kernels, not automatically comparable
variance contributions. The sample-average genetic and residual variances are

```text
V_G = sum_g theta_g a_g
V_E = sum_j eta_j b_j
h2  = V_G / (V_G + V_E)
```

Only when every atom equals one does this reduce to the familiar ratio of raw
parameter sums. Default initialization is trace-calibrated so that `h2_init`
has this meaning for standard, admixed, and effective-rank SMILE kernels.

Repeated `n_reml_reps` fits vary the randomized probes. Their spread estimates
Monte Carlo uncertainty of the replicate mean; it is not a sample-deletion
jackknife. The explicit result fields are `monte_carlo_se_var` and
`monte_carlo_se_h2`; legacy `jackknife_*` fields remain compatibility aliases.
