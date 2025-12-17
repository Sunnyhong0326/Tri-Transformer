import pytest
import torch
import torch.nn.functional as F

from tf.ops import layernorm


@pytest.fixture()
def device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


@pytest.mark.parametrize(
    "B,T,D",
    [
        (1, 4, 16),
        (2, 5, 32),
        (3, 2, 24),
    ],
)
def test_layernorm_matches_torch(device_and_dtype, B, T, D):
    device, dtype = device_and_dtype
    torch.manual_seed(0)

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype)
    b = torch.randn(D, device=device, dtype=dtype)
    eps = 1e-5

    y_ref = F.layer_norm(x, (x.shape[-1],), w, b, eps=eps)
    y = layernorm(x, w, b, eps, backend="pytorch")

    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)
