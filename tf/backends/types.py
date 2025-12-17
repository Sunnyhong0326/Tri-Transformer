from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional
import torch

@dataclass(frozen=True)
class AttentionParams:
    causal: bool = True
    sm_scale: Optional[float] = None   # if None: 1/sqrt(D)
    dropout_p: float = 0.0
    training: bool = False

class BackendOps(Protocol):
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: AttentionParams) -> torch.Tensor: ...
    def layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor: ...
