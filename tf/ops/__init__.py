from __future__ import annotations
import torch
from tf.backends.registry import get_backend
from tf.backends.types import AttentionParams

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: AttentionParams, *, backend: str = "pytorch") -> torch.Tensor:
    return get_backend(backend).attention(q, k, v, p)

def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, *, backend: str = "pytorch") -> torch.Tensor:
    return get_backend(backend).layernorm(x, weight, bias, eps)