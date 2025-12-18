# Refence from: 
# 1. https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/kernel.py
# 2. https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd(
    Q, K, V,
    O, L,
    BH,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    bh_idx = tl.program_id(1)
    m_idx = tl.program_id(0)
    
    base_offset = bh_idx * BH

class TritonFlashAttnFn(torch.autograd.Function):
    """This operator implements FlashAttention-2
    """
    @staticmethod
    def forward(
        ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
        
        BH = query.shape[0] * query.shape[1]
        BLOCK_M = 64
        BLOCK_N = 64
        
        attn_out = torch.empty_like(query, device=query.device)
        logsumexp = torch.empty((query.shape[-1]), device=query.device)
        
        _flash_attn_fwd[(BLOCK_M, BH)](
            query, key, value,
            attn_out, logsumexp,
            BH,
            BLOCK_M, BLOCK_N
        )
        
        ctx.save_for_backward()

    @staticmethod
    def backward(ctx):
        raise NotImplementedError()