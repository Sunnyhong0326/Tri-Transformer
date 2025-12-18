# Refence from: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_fwd_kernel(
    X, W, B, 
    Y, Mean, Rstd,
    N, EPS: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * N
    
    sum_x  = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)
    for col_offset in tl.range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        offset = row_offset + cols
        mask = cols < N
        # reduction of fp16 may cause precision issue so cast to fp32
        x = tl.load(X + offset, mask, other=0.).to(tl.float32) 
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    var  = tl.maximum(var, 0.0)
    rstd = 1 / tl.sqrt(var + EPS)
    
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    
    for col_offset in tl.range(0, N, BLOCK_N):
        cols = col_offset + tl.arange(0, BLOCK_N)
        offset = row_offset + cols
        mask = cols < N
        x = tl.load(X + offset, mask, other=0.).to(tl.float32) 
        w = tl.load(W + cols, mask, other=0.)
        b = tl.load(B + cols, mask, other=0.)
        y = (x - mean) * rstd * w + b
        tl.store(Y + offset, y, mask)
    
@triton.jit
def _layernorm_bwd_kernel(X, W, DY, Mean, Rstd, DX, DW, DB,
                   M: tl.constexpr, N: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pass

class TritonLayerNormFn(torch.autograd.Function):
    """This operator normalize over the 'last' dimension of the tensor
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        assert x.is_cuda, "Triton LN requires CUDA tensor"
        assert x.is_contiguous(), "Tensor x must be contiguous"
        assert x.shape[-1] == weight.numel() == bias.numel()
        
        # x must be contiguous so don't need to reshape
        N = x.shape[-1]
        M = x.numel() // N
        
        y = torch.empty_like(x)
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        if N > BLOCK_N:
            raise RuntimeError("This fused kernel expects N <= BLOCK_N")
        
        num_warps = min(max(BLOCK_N // 256, 1), 8)
        
        _layernorm_fwd_kernel[(M,)](
            x, weight, bias,
            y, mean, rstd,
            N,
            EPS=eps,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N
        ctx.eps = eps
        
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        raise NotImplementedError()