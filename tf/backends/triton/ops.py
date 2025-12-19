from __future__ import annotations
import torch
from tf.backends.types import AttentionParams

from .kernels.layernorm import TritonLayerNormFn
from .kernels.flashattn import TritonFlashAttnFn

class TritonOps:
    def attention(self, q, k, v, p: AttentionParams):
        return TritonFlashAttnFn.apply(q, k, v, p)

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        return TritonLayerNormFn.apply(x, weight, bias, eps)
