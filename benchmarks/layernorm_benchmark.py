from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing as tt

from tf.ops import layernorm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
PLOT_NAME = "layernorm-backend-performance"


def _bytes_moved(x: torch.Tensor) -> int:
    """Approximate bytes touched by layernorm (read + write x)."""
    return 2 * x.numel() * x.element_size()


@tt.perf_report(
    tt.Benchmark(
        x_names=["D"],
        x_vals=[64 * i for i in range(2, 65)],
        line_arg="backend",
        line_vals=["Pytorch Ops", "pytorch", "triton"],
        line_names=["PyTorch Ops", "pytorch", "Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name=PLOT_NAME,
        args={"B": 8, "T": 2048},
    )
)
def benchmark(B: int, T: int, D: int, backend: str) -> float:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required to run Triton benchmarks.")

    x = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D, device=DEVICE, dtype=DTYPE)
    b = torch.randn(D, device=DEVICE, dtype=DTYPE)
    eps = 1e-5

    if backend == "Pytorch Ops":
        ms = tt.do_bench(lambda: F.layer_norm(x, (D,), w, b, eps=eps))
    else:
        ms = tt.do_bench(lambda: layernorm(x, w, b, eps, backend=backend))

    # bytes / (ms -> s) -> GB/s
    bytes_per_call = _bytes_moved(x)
    return bytes_per_call * 1e-6 / ms


def main(save_dir: str) -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required to run this benchmark.")

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running layernorm benchmark on {DEVICE} with dtype={DTYPE}.")
    benchmark.run(print_data=True, show_plots=False, save_path=str(output_dir))

    plot_path = output_dir / f"{PLOT_NAME}.png"
    print(f"Benchmark plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark layernorm across backends.")
    parser.add_argument(
        "--save-dir",
        default="benchmarks",
        help="Directory where benchmark plots will be saved.",
    )
    args = parser.parse_args()
    main(args.save_dir)
