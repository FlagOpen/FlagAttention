import torch
import pytest

from ref_impl.piecewise import attention as attention_torch
from ref_impl.piecewise import attention_grad as attention_grad_torch

from flag_attn.piecewise import standalone_forward as attention_triton
from flag_attn.piecewise import standalone_backward as attention_grad_triton
from flag_attn.piecewise import attention as piecewise_attn


@pytest.mark.parametrize('B, H, T, D, P_SEQ', [
    (2, 4, 1024, 32, 100), 
    (2, 4, 1024, 32, 0)])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_standalone(B, H, T, D, P_SEQ, causal, dtype):
    q1 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    q2 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    sm_scale = 0.5
    w = 780

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    o_hyp, _ = attention_triton(q1, k1, q2, k2, v, w, causal, sm_scale)
    torch.testing.assert_close(o_hyp, o_ref, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize('B, H, T, D, P_SEQ', [
    (2, 4, 1024, 32, 100), 
    (2, 4, 1024, 32, 0)])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_grad_standalone(B, H, T, D, P_SEQ, causal, dtype):
    q1 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    q2 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    w = 780
    do = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_ref, dk1_ref, dq2_ref, dk2_ref, dv_ref = attention_grad_torch(q1, k1, q2, k2, v, w, causal, sm_scale, o_ref, do)

    o_hyp, L = attention_triton(q1, k1, q2, k2, v, w, causal, sm_scale)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = attention_grad_triton(q1, k1, q2, k2, v, w, causal, sm_scale, o_hyp, L, do)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = [item.to(dtype) for item in [dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp]]
    
    torch.testing.assert_close(dv_hyp, dv_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dq1_hyp, dq1_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dq2_hyp, dq2_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dk1_hyp, dk1_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dk2_hyp, dk2_ref, atol=1e-2, rtol=1e-3) 


@pytest.mark.parametrize('B, H, T, D, P_SEQ', [
    (2, 4, 1024, 32, 100), 
    (2, 4, 1024, 32, 0)])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_fwd_bwd(B, H, T, D, P_SEQ, causal, dtype):
    q1 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    q2 = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k1 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k2 = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    w = 780
    do = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)

    o_ref = attention_torch(q1, k1, q2, k2, v, w, causal, sm_scale)
    o_ref.backward(do)
    dq1_ref, dk1_ref, dq2_ref, dk2_ref, dv_ref = q1.grad.clone(), k1.grad.clone(), k2.grad.clone(), k2.grad.clone(), v.grad.clone()
    q1.grad, k1.grad, k2.grad, k2.grad, v.grad = None, None, None, None, None

    o_hyp = piecewise_attn(q1, k1, q2, k2, v, w, causal, sm_scale)
    o_hyp.backward(do)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = q1.grad.clone(), k1.grad.clone(), k2.grad.clone(), k2.grad.clone(), v.grad.clone()
    q1.grad, k1.grad, k2.grad, k2.grad, v.grad = None, None, None, None, None

    torch.testing.assert_close(o_hyp, o_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dv_hyp, dv_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dq1_hyp, dq1_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dq2_hyp, dq2_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dk1_hyp, dk1_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(dk2_hyp, dk2_ref, atol=1e-2, rtol=1e-3) 