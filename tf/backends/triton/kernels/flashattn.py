# Refence from: 
# 1. https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/kernel.py
# 2. https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
from __future__ import annotations
import torch
import triton
import triton.language as tl

from tf.backends.types import AttentionParams

@triton.jit
def _flash_attn_fwd(
    Q, K, V, QK_SCALE: tl.constexpr,
    AO, LSE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr
):
    bh_idx = tl.program_id(1)
    m_idx = tl.program_id(0)
    
    qkv_base = bh_idx * L * D
    offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M) # (BM,)
    offs_d = tl.arange(0, D) # (D, )
    
    Q_ptr = Q + qkv_base + offs_m[:, None] * D + offs_d[None, :] # (BM, D)
    AO_ptr = AO + qkv_base + offs_m[:, None] * D + offs_d[None, :] # (BM, D)
    LSE_ptr = LSE + bh_idx * L + offs_m # (BM, )
    
    q = tl.load(Q_ptr, mask=offs_m[:, None] < L).to(tl.float32) # (BM, D)
    o = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    l = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    m = tl.full((BLOCK_M, ), -float('inf'), dtype=tl.float32)
    
    for start_n in tl.range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N) # (BM, )
        mask_n = offs_n < L # (BN, )
        
        K_ptr = K + qkv_base + offs_n[None, :] * D + offs_d[:, None] # (D, BN)
        V_ptr = V + qkv_base + offs_n[:, None] * D + offs_d[None, :] # (BN, D)
        
        kt = tl.load(K_ptr, mask=mask_n[None, :], other=0.0).to(tl.float32) # (D, BN)
        v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0).to(tl.float32) # (BN, D)
        
        qk = tl.dot(q, kt) * QK_SCALE * 1.4426950408889634 # (BM, BN)
        if L - start_n * BLOCK_N < BLOCK_N:
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        m_new = tl.maximum(m, tl.max(qk, axis=1)) 
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m - m_new)
        l = alpha * l + tl.sum(p, axis=1)
        m = m_new
        o = alpha[:, None] * o + tl.dot(p, v)
        
    o = o / l[:, None]
    l = m + tl.log2(l)
    
    mask_m = offs_m < L # (BM, )
    tl.store(AO_ptr, o, mask=mask_m[:, None])
    tl.store(LSE_ptr, l, mask=mask_m)
    

class TritonFlashAttnFn(torch.autograd.Function):
    """This operator implements FlashAttention-2
    """
    @staticmethod
    def forward(
        ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        p: AttentionParams
    ):
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
        assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4
        
        # batchsize * num_heads, sequence length, head dimension
        BH, L, D = query.shape[0]*query.shape[1], query.shape[2], query.shape[-1]
        BLOCK_M, BLOCK_N = 64, 64
        
        M = triton.cdiv(query.shape[2], BLOCK_M)
        attn_out = torch.empty_like(query, device=query.device)
        logsumexp = torch.empty((query.shape[-1]), device=query.device)
        
        _flash_attn_fwd[(M, BH)](
            query, key, value, p.sm_scale,
            attn_out, logsumexp,
            BLOCK_M, BLOCK_N, L, D
        )
        ctx.save_for_backward(query, key, value, attn_out, logsumexp)
        ctx.scale = p.sm_scale
        return attn_out

    @staticmethod
    def backward(ctx):
        raise NotImplementedError()