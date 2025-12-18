from __future__ import annotations
import torch
import triton
import triton.language as tl



class TritonFlashAttnFn(torch.autograd.Function):
    """This operator implements FlashAttention-2
    """
    @staticmethod
    def forward(
        ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        ):
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
        
        
    
    @staticmethod
    def backward(ctx):
        raise NotImplementedError()