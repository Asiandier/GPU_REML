# Contributing

Thank you for considering a contribution to GPU_REML.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional PGEN support:

```bash
python -m pip install -e ".[dev,pgen]"
```

## Checks Before Submitting

Run:

```bash
python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/gpu_reml_wheel .
```

For changes touching genotype streaming, REML scoring, PCG, preconditioning,
or sparse paths, add focused tests that compare against a dense or analytically
simple reference.

## Coding Guidelines

- Preserve the linear-operator contract: `K @ V = Z @ (Z.T @ V) / m_eff`.
- Keep host-side streaming and traced JAX code separated. Avoid calling Python
  genotype-streaming functions inside `jax.jit`-traced regions.
- Prefer explicit validation for sample order, variant order, and component
  definitions.
- Do not commit benchmark outputs, logs, temporary genotype splits, or private
  data paths.

## Reporting Issues

Please include:

- command line or API call,
- input format and dimensions,
- JAX/JAXLIB/CUDA versions,
- GPU model and memory,
- full traceback or relevant log excerpt,
- a minimal reproducer when possible.
