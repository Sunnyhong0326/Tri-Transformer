from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from ..types import AttentionParams

def _causal_mask(Tq: int, Tk: int, device, dtype):
    # mask[i, j] = True if j > i (future positions)
    return torch.triu(torch.ones((Tq, Tk), device=device, dtype=torch.bool), diagonal=1)

class PyTorchOps:
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: AttentionParams) -> torch.Tensor:
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expect (B,H,T,D)"
        B, H, Tq, Dq = q.shape
        _, _, Tk, Dk = k.shape
        assert Dq == Dk == v.shape[-1]
        scale = p.sm_scale if p.sm_scale is not None else (1.0 / math.sqrt(Dq))

        # scores: (B,H,Tq,Tk)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if p.causal:
            mask = _causal_mask(Tq, Tk, device=scores.device, dtype=scores.dtype)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)

        if p.dropout_p > 0 and p.training:
            attn = F.dropout(attn, p=p.dropout_p)

        out = torch.matmul(attn, v)  # (B,H,Tq,D)
        return out

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mean) * torch.rsqrt(var + eps)
        return xhat * weight + bias
