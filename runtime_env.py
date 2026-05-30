"""Runtime environment bootstrap for CLI entrypoints.

This must run before importing JAX so allocator-related environment
variables take effect.
"""

from __future__ import annotations

import os


def configure_runtime_env() -> None:
    # Use the non-preallocating JAX client by default so CLI runs track the
    # project GPU budget more closely. Prefer the platform allocator so
    # nvidia-smi reflects live allocations more faithfully than the default
    # pooled allocator. Respect any explicit user override.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


__all__ = ["configure_runtime_env"]
