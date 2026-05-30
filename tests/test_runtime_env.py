import os
import sys
import importlib


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_REPO_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PKG = os.path.basename(_REPO_ROOT)
_RUNTIME = importlib.import_module(f"{_PKG}.runtime_env")

configure_runtime_env = _RUNTIME.configure_runtime_env


def test_configure_runtime_env_sets_non_preallocating_platform_defaults(monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    configure_runtime_env()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


def test_configure_runtime_env_respects_existing_override(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    configure_runtime_env()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
