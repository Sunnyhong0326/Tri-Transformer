from __future__ import annotations
import torch
import torch.nn as nn
from tf.nn.blocks import TransformerBlock
from tf.ops import layernorm

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 max_seq_len: int, dropout: float = 0.0, ln_eps: float = 1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.ln_eps = ln_eps

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, ln_eps=ln_eps)
            for _ in range(n_layers)
        ])

        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.ln_f_b = nn.Parameter(torch.zeros(d_model))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (optional, common in GPTs)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, *, backend: str = "pytorch") -> torch.Tensor:
        # idx: (B,T) int64
        B, T = idx.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=idx.device, dtype=torch.long)[None, :]  # (1,T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B,T,C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x, backend=backend, causal=True)

        x = layernorm(x, self.ln_f_w, self.ln_f_b, self.ln_eps, backend=backend)
        logits = self.lm_head(x)  # (B,T,V)
        return logits
