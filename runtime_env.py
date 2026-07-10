"""Runtime environment bootstrap for CLI entrypoints.

This must run before importing JAX so allocator-related environment
variables take effect.
"""

from __future__ import annotations

import os


def configure_runtime_env() -> None:
    # Use the non-preallocating JAX client by default so CLI runs track the
    # project GPU budget more closely. Do not force the platform allocator: it
    # serializes allocation/free work and is substantially slower for the
    # repeated streamed kernels used here. Users can still opt into it through
    # XLA_PYTHON_CLIENT_ALLOCATOR for allocation diagnostics.
    # Limit backend discovery to CUDA/CPU by default. On non-TPU machines JAX
    # otherwise tries to initialize the TPU backend and emits a noisy libtpu
    # warning during every CLI run.
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


__all__ = ["configure_runtime_env"]
