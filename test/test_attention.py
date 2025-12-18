import math

import pytest
import torch
import torch.nn.functional as F

from tf.backends.types import AttentionParams
from tf.ops import attention


@pytest.fixture()
def device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


@pytest.mark.parametrize(
    "B,H,T,D,causal",
    [
        (1, 2, 8, 16, True),
        (2, 4, 16, 32, True),
        (2, 4, 16, 32, False),
        (3, 2, 12, 24, False),
    ],
)
def test_attention_matches_sdpa(device_and_dtype, B, H, T, D, causal):
    device, dtype = device_and_dtype
    torch.manual_seed(0)

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    # torch SDPA expects (B,H,T,D) and uses is_causal flag
    y_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)

    p = AttentionParams(causal=causal, sm_scale=scale, dropout_p=0.0, training=False)
    y = attention(q, k, v, p, backend="pytorch")

    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)
