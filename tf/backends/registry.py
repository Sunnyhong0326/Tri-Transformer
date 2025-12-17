from __future__ import annotations
from typing import Dict
from .types import BackendOps

from tf.backends.pytorch import PyTorchOps
from tf.backends.triton import TritonOps

# Backends can be registered as classes or instances.
_BACKENDS: Dict[str, BackendOps] = {
    "pytorch": PyTorchOps,
    "triton": TritonOps,
}

def register_backend(name: str, ops: BackendOps) -> None:
    _BACKENDS[name] = ops

def _ensure_instance(ops: BackendOps) -> BackendOps:
    # If a class was registered, instantiate it; otherwise return the instance as-is.
    return ops() if isinstance(ops, type) else ops

def get_backend(name: str) -> BackendOps:
    if name not in _BACKENDS:
        raise KeyError(f"Backend '{name}' not registered. Available: {list(_BACKENDS.keys())}")
    return _ensure_instance(_BACKENDS[name])
