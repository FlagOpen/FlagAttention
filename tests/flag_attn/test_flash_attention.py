import math
import torch
import pytest
import logging

import flag_attn
import flash_attn

torch.random.manual_seed(10086)

def max_diff(a, b):
    return (a - b).abs().max().item()

def zero_percent(a, b):
    diff = (a - b).abs()
    num_non_zeros = diff.nonzero().shape[0]
    return (1.0 - num_non_zeros/ diff.numel()) * 100.0

def report(name, actual, expected):
    print(f"{name}: \tmax_difference: {max_diff(actual, expected):0.6f}\tzero_diff elements: {zero_percent(actual, expected):0.3f}%")


@pytest.mark.parametrize('device_id', list(range(torch.cuda.device_count())))
@pytest.mark.parametrize('scale', [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize('B, H, T, D, P_SEQ', [
    (2, 4, 512, 128, 100),
    (2, 4, 1024, 64, 10), 
    (2, 4, 2048, 32, 0),
    (2, 4, 4096, 16, 0),
    (1, 2, 8192, 16, 0),
    (1, 2, 8192, 32, 0),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_standalone(B, H, T, D, P_SEQ, causal, dtype, scale, device_id):
    device = f"cuda:{device_id}"
    q = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    k = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale)

    o_ref = flag_attn.testing.flash_attention(q, k, v, causal, upcast=True)
    o_torch = flag_attn.testing.flash_attention(q, k, v, causal, upcast=False)
    o_hyp = flag_attn.flash_attention(q, k, v, causal)
    

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    logging.info("torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(torch_max_diff, triton_max_diff))
    assert triton_max_diff <= torch_max_diff
    # assert torch.testing.assert_close(o_hyp, o_ref)


@pytest.mark.parametrize('device_id', list(range(torch.cuda.device_count())))
@pytest.mark.parametrize('scale', [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize('B, H, T, D, P_SEQ', [
    (2, 4, 512, 128, 100),
    (2, 4, 1024, 64, 10), 
    (2, 4, 2048, 32, 0),
    (2, 4, 4096, 16, 0),
    (1, 2, 8192, 16, 0),
    (1, 2, 8192, 32, 0),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_fwd_bwd(B, H, T, D, P_SEQ, causal, dtype, scale, device_id):
    device = f"cuda:{device_id}"
    dtype = torch.bfloat16
    q = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    k = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    do = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()
    causal=False

    o_ref = flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=True)
    o_hyp = flag_attn.flash_attention(q, k, v, causal=causal)
    o_fla = flash_attn.flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=causal,
    ).transpose(1, 2)

    gq_ref, gk_ref, gv_ref = torch.autograd.grad(o_ref, (q, k, v), do)
    gq_hyp, gk_hyp, gv_hyp = torch.autograd.grad(o_hyp, (q, k, v), do)
    gq_fla, gk_fla, gv_fla = torch.autograd.grad(o_fla, (q, k, v), do)

    print("flash attention (triton) vs flash attention(cuda)")
    report("o", o_hyp, o_fla)
    report("gq", gq_hyp, gq_fla)
    report("gk", gk_hyp, gk_fla)
    report("gv", gv_hyp, gv_fla)

    print("flash attention (triton) vs pytorch(fp32)")
    report("o", o_hyp, o_ref)
    report("gq", gq_hyp, gq_ref)
    report("gk", gk_hyp, gk_ref)
    report("gv", gv_hyp, gv_ref)

    print("flash attention (cuda) vs pytorch(fp32)")
    report("o", o_fla, o_ref)
    report("gq", gq_fla, gq_ref)
    report("gk", gk_fla, gk_ref)
    report("gv", gv_fla, gv_ref)
