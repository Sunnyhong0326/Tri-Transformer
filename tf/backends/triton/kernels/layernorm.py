from __future__ import annotations
import torch
import triton
import triton.language as tl

def _pick_block_n(n: int) -> int:
    # correctness-first heuristic; tweak later via autotune
    if n <= 64: return 64
    if n <= 128: return 128
    if n <= 256: return 256
    if n <= 512: return 512
    return 1024

def _pick_num_warps(block_n: int) -> int:
    if block_n <= 128: return 4
    if block_n <= 256: return 4
    if block_n <= 512: return 8
    return 8

@triton.jit
def _ln_fwd_kernel(X, W, B, Y, Mean, Rstd,
                   M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # mean/var in fp32
    mean = tl.sum(x, axis=0) / N
    x0 = x - mean
    var = tl.sum(x0 * x0, axis=0) / N
    rstd = tl.rsqrt(var + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = x0 * rstd * w + b
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)  # store fp16; adjust if you want bf16/fp32

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

@triton.jit
def _ln_bwd_kernel(X, W, DY, Mean, Rstd, DX, DW, DB,
                   M: tl.constexpr, N: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(Mean + row).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)

    xhat = (x - mean) * rstd
    dy_g = dy * w

    # sums over feature dim
    sum1 = tl.sum(dy_g, axis=0)
    sum2 = tl.sum(dy_g * xhat, axis=0)

    invN = 1.0 / N
    dx = (dy_g - sum1 * invN - xhat * sum2 * invN) * rstd
    tl.store(DX + row * N + cols, dx.to(tl.float16), mask=mask)

    # dw = sum(dy * xhat), db = sum(dy) over rows
    # correctness-first: atomic accumulate
    tl.atomic_add(DW + cols, dy * xhat, mask=mask)
    tl.atomic_add(DB + cols, dy, mask=mask)

class TritonLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        assert x.is_cuda, "Triton LN requires CUDA tensor"
        assert x.shape[-1] == weight.numel() == bias.numel()
        # For simplicity (and to match common LN usage), require contiguous last dim
        x_ = x.contiguous()

        orig_shape = x_.shape
        N = orig_shape[-1]
        M = x_.numel() // N
        x2d = x_.view(M, N)

        y = torch.empty_like(x2d, dtype=torch.float16)
        mean = torch.empty((M,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

        block_n = _pick_block_n(N)
        num_warps = _pick_num_warps(block_n)

        _ln_fwd_kernel[(M,)](
            x2d, weight, bias, y, mean, rstd,
            M=M, N=N, EPS=eps,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x2d, weight, mean, rstd)
        ctx.N = N
        ctx.M = M
        ctx.block_n = block_n
        ctx.num_warps = num_warps
        return y.view(orig_shape).to(x.dtype)  # return same dtype as input

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, w, mean, rstd = ctx.saved_tensors
        N, M = ctx.N, ctx.M
        block_n, num_warps = ctx.block_n, ctx.num_warps

        dy2d = dy.contiguous().view(M, N)

        dx = torch.empty_like(x2d, dtype=torch.float16)
        dw = torch.zeros((N,), device=dy.device, dtype=torch.float32)
        db = torch.zeros((N,), device=dy.device, dtype=torch.float32)

        _ln_bwd_kernel[(M,)](
            x2d, w, dy2d, mean, rstd, dx, dw, db,
            M=M, N=N,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

        # Cast gradients to match parameter dtypes
        dx = dx.view_as(dy).to(dy.dtype)
        dw = dw.to(w.dtype)
        db = db.to(w.dtype)
        return dx, dw, db, None  # None for eps