from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing as tt

from tf.backends.types import AttentionParams
from tf.ops import attention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PLOT_NAME = "flashattention-backend-performance"


def _attention_flops(B: int, H: int, T: int, D: int) -> int:
    # Rough FLOP count for QK^T and Attn*V matmuls; softmax cost is ignored.
    return 4 * B * H * T * T * D


def _run_backend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: AttentionParams, backend: str) -> float:
    if backend == "torch-sdpa":
        return tt.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=p.causal)
        )
    if backend == "pytorch":
        return tt.do_bench(lambda: attention(q, k, v, p, backend="pytorch"))
    if backend == "triton":
        return tt.do_bench(lambda: attention(q, k, v, p, backend="triton"))
    raise ValueError(f"Unknown backend '{backend}'")


def make_benchmark(include_triton: bool = False):
    line_vals = ["torch-sdpa", "pytorch"]
    line_names = ["Torch SDPA", "PyTorch Ops"]
    styles = [("green", "-"), ("blue", "-")]

    if include_triton:
        line_vals.append("triton")
        line_names.append("Triton")
        styles.append(("red", "-"))

    bench = tt.Benchmark(
        x_names=["T"],
        x_vals=[128 * i for i in range(2, 17)],  # 256 -> 2048
        line_arg="backend",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="TFLOP/s",
        plot_name=PLOT_NAME,
        args={"B": 4, "H": 8, "D": 64},
    )

    @tt.perf_report(bench)
    def benchmark(B: int, H: int, T: int, D: int, backend: str) -> float:
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required to run Triton benchmarks.")

        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
        p = AttentionParams(causal=False, sm_scale=1.0 / math.sqrt(D), dropout_p=0.0, training=False)

        try:
            ms = _run_backend(q, k, v, p, backend)
        except NotImplementedError as exc:
            raise RuntimeError("Selected backend does not implement attention yet.") from exc

        flops = _attention_flops(B, H, T, D)
        return flops * 1e-9 / ms

    return benchmark


def main(save_dir: str, include_triton: bool) -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required to run this benchmark.")

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = make_benchmark(include_triton)

    print(f"Running flash attention benchmark on {DEVICE} with dtype={DTYPE}.")
    bench.run(print_data=True, show_plots=False, save_path=str(output_dir))

    plot_path = output_dir / f"{PLOT_NAME}.png"
    print(f"Benchmark plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark flash attention across backends.")
    parser.add_argument(
        "--save-dir",
        default="benchmark_output",
        help="Directory where benchmark plots will be saved.",
    )
    parser.add_argument(
        "--include-triton",
        action="store_true",
        help="Include the Triton backend in the benchmark (fails if not implemented).",
    )
    args = parser.parse_args()
    main(args.save_dir, args.include_triton)
