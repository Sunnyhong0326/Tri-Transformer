from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tf.ops import attention, layernorm
from tf.backends.types import AttentionParams

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, *, backend: str = "pytorch", causal: bool = True) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.shape
        qkv = self.W_qkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B,H,T,D)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()

        p = AttentionParams(
            causal=causal,
            sm_scale=1.0 / math.sqrt(self.d_head),
            dropout_p=self.dropout,
            training=self.training,
        )
        o = attention(q, k, v, p, backend=backend)  # (B,H,T,D)

        # back to (B,T,C)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(o)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, ln_eps: float = 1e-5):
        super().__init__()
        self.ln_eps = ln_eps
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.mlp = MLP(d_model, d_ff, dropout=dropout)

        # LN params as raw tensors so we can call our LN op (later: Triton LN)
        self.ln1_w = nn.Parameter(torch.ones(d_model))
        self.ln1_b = nn.Parameter(torch.zeros(d_model))
        self.ln2_w = nn.Parameter(torch.ones(d_model))
        self.ln2_b = nn.Parameter(torch.zeros(d_model))

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, backend: str = "pytorch", causal: bool = True) -> torch.Tensor:
        # Pre-LN Transformer (GPT-style)
        y = layernorm(x, self.ln1_w, self.ln1_b, self.ln_eps, backend=backend)
        y = self.attn(y, backend=backend, causal=causal)
        x = x + self.drop(y)

        y = layernorm(x, self.ln2_w, self.ln2_b, self.ln_eps, backend=backend)
        y = self.mlp(y)
        x = x + self.drop(y)
        return x
