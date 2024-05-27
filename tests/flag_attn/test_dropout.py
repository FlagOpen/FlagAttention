import torch
import pytest
from flag_attn.dropout import dropout_mask

@pytest.mark.parametrize('x', [torch.empty(1, device='cuda')])
@pytest.mark.parametrize('B, H, M, N', [
    (2, 4, 512, 612),
    (2, 4, 1024, 1034),
    (2, 4, 2048, 2048),
    (2, 4, 4096, 4096),
    (2, 4, 4001, 4001),
    (2, 4, 4001, 4096),
    (2, 4, 4096, 4000),
    (1, 2, 8192, 8202),
    (1, 2, 8192, 8192),
])
@pytest.mark.parametrize('p', [0.5, 0.8])
def test_dropout_mask(x, B, H, M, N, p):
    import math
    seed = 123456789
    offset = 123456789123456789
    mask = dropout_mask(x, B, H, M, N, p, seed, offset)
    # zeros indicate to drop
    # k follows Binomial distributio B(k; n, p)
    n = mask.numel()
    k = torch.sum(mask == 0)
    p_cap = k / n
    tol = 0.01
    assert math.fabs(p_cap - p) < tol * p
