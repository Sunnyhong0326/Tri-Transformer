from __future__ import annotations
import torch
from tf.backends.types import AttentionParams

from .kernels.layernorm import TritonLayerNormFn

class TritonOps:
    def attention(self, q, k, v, p: AttentionParams):
        raise NotImplementedError("FlashAttention backend not implemented yet.")

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        return TritonLayerNormFn.apply(x, weight, bias, eps)
