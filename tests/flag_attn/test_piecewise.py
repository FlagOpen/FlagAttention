import math
import torch
import pytest
import logging

from ref_impl.piecewise import attention as attention_torch
from ref_impl.piecewise import attention_grad as attention_grad_torch

from flag_attn.piecewise import standalone_forward as attention_triton
from flag_attn.piecewise import standalone_backward as attention_grad_triton
from flag_attn.piecewise import attention as piecewise_attn

def max_diff(a, b):
    return (a - b).abs().max().item()

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
    q1 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    q2 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale)
    sm_scale = 1. / math.sqrt(D)
    w = 780

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale, upcast=True)
    o_torch = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    o_hyp, _ = attention_triton(q1, k1, q2, k2, v, w, causal, sm_scale)
    

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    logging.info("torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(torch_max_diff, triton_max_diff))
    assert triton_max_diff <= torch_max_diff * 2 + 1e-5
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
def test_attention_grad_standalone(B, H, T, D, P_SEQ, causal, dtype, scale, device_id):
    device = f"cuda:{device_id}"
    q1 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    q2 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    sm_scale = 1. / math.sqrt(D)
    w = 780
    do = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale)

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale, upcast=True)
    dq1_ref, dk1_ref, dq2_ref, dk2_ref, dv_ref = attention_grad_torch(q1, k1, q2, k2, v, w, causal, sm_scale, o_ref, do, upcast=True)

    o_torch = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_torch, dk1_torch, dq2_torch, dk2_torch, dv_torch = attention_grad_torch(q1, k1, q2, k2, v, w, causal, sm_scale, o_torch, do)

    o_hyp, L = attention_triton(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = attention_grad_triton(q1, k1, q2, k2, v, w, causal, sm_scale, o_hyp, L, do)
    
    dq1_torch_max_diff = max_diff(dq1_torch, dq1_ref)
    dq2_torch_max_diff = max_diff(dq2_torch, dq2_ref)
    dk1_torch_max_diff = max_diff(dk1_torch, dk1_ref)
    dk2_torch_max_diff = max_diff(dk2_torch, dk2_ref)
    dv_torch_max_diff = max_diff(dv_torch, dv_ref)

    dq1_triton_max_diff = max_diff(dq1_hyp, dq1_ref)
    dq2_triton_max_diff = max_diff(dq2_hyp, dq2_ref)
    dk1_triton_max_diff = max_diff(dk1_hyp, dk1_ref)
    dk2_triton_max_diff = max_diff(dk2_hyp, dk2_ref)
    dv_triton_max_diff = max_diff(dv_hyp, dv_ref)

    logging.info("dq1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dq1_torch_max_diff, dq1_triton_max_diff))
    logging.info("dq2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dq2_torch_max_diff, dq2_triton_max_diff))
    logging.info("dk1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dk1_torch_max_diff, dk1_triton_max_diff))
    logging.info("dk2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dk2_torch_max_diff, dk2_triton_max_diff))
    logging.info("dv torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dv_torch_max_diff, dv_triton_max_diff))

    assert dq1_triton_max_diff <= 2 * dq1_torch_max_diff + 1e-5
    assert dq2_triton_max_diff <= 2 * dq2_torch_max_diff + 1e-5
    assert dk1_triton_max_diff <= 2 * dk1_torch_max_diff + 1e-5
    assert dk2_triton_max_diff <= 2 * dk2_torch_max_diff + 1e-5
    assert dv_triton_max_diff <= 2 * dv_torch_max_diff + 1e-5


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
    q1 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    q2 = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device=device).normal_(mean=0., std=scale).requires_grad_()
    sm_scale = 1. / math.sqrt(D)
    w = 780
    do = torch.empty((B, H, T, D), dtype=dtype, device=device).normal_(mean=0., std=scale)

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale, upcast=True)
    dq1_ref, dk1_ref, dq2_ref, dk2_ref, dv_ref = torch.autograd.grad(o_ref, (q1, k1, q2, k2, v), do)

    o_torch = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_torch, dk1_torch, dq2_torch, dk2_torch, dv_torch = torch.autograd.grad(o_torch, (q1, k1, q2, k2, v), do)

    o_hyp = piecewise_attn(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = torch.autograd.grad(o_hyp, (q1, k1, q2, k2, v), do)

    o_torch_max_diff = max_diff(o_torch, o_ref)
    dq1_torch_max_diff = max_diff(dq1_torch, dq1_ref)
    dq2_torch_max_diff = max_diff(dq2_torch, dq2_ref)
    dk1_torch_max_diff = max_diff(dk1_torch, dk1_ref)
    dk2_torch_max_diff = max_diff(dk2_torch, dk2_ref)
    dv_torch_max_diff = max_diff(dv_torch, dv_ref)

    o_triton_max_diff = max_diff(o_hyp, o_ref)
    dq1_triton_max_diff = max_diff(dq1_hyp, dq1_ref)
    dq2_triton_max_diff = max_diff(dq2_hyp, dq2_ref)
    dk1_triton_max_diff = max_diff(dk1_hyp, dk1_ref)
    dk2_triton_max_diff = max_diff(dk2_hyp, dk2_ref)
    dv_triton_max_diff = max_diff(dv_hyp, dv_ref)

    logging.info("o torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(o_torch_max_diff, o_triton_max_diff))
    logging.info("dq1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dq1_torch_max_diff, dq1_triton_max_diff))
    logging.info("dq2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dq2_torch_max_diff, dq2_triton_max_diff))
    logging.info("dk1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dk1_torch_max_diff, dk1_triton_max_diff))
    logging.info("dk2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dk2_torch_max_diff, dk2_triton_max_diff))
    logging.info("dv torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(dv_torch_max_diff, dv_triton_max_diff))

    assert o_triton_max_diff <= 2 * o_torch_max_diff + 1e-5
    assert dq1_triton_max_diff <= 2 * dq1_torch_max_diff + 1e-5
    assert dq2_triton_max_diff <= 2 * dq2_torch_max_diff + 1e-5
    assert dk1_triton_max_diff <= 2 * dk1_torch_max_diff + 1e-5
    assert dk2_triton_max_diff <= 2 * dk2_torch_max_diff + 1e-5
    assert dv_triton_max_diff <= 2 * dv_torch_max_diff + 1e-5
