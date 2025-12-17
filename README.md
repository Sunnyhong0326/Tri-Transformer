# Tri-Transformer from Scratch

A learning repository for building a Transformer from scratch while practicing Triton kernel development.

This repo implements core Transformer ops as custom kernels and compares them against PyTorch for both correctness and efficiency

## Custom operations

- FlashAttention Forward / Backward
- LayerNorm Forward / Backward


## Baselines

Each custom op is validated and benchmarked against the corresponding PyTorch implementation to ensure the kernel is both correct and worth using.