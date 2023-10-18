"""
Piecewise Attention
====================

This is a extension to Flash Attention v2 algorithm from Tri Dao 
(https://tridao.me/publications/flash2/flash2.pdf) that performs piecewise computation 
of attention scores(The scores to which softmax is applied). This design originates from 
the need to make better predictions when the predicted sequence is longer than sequences 
in the training set.

It takes as input two q's and two k's as inputs. The attention score is the dot product 
of (q1, k1) or (q2, k2) depending on whether the distance between q & k exceeds a threshold.

The code is adapted from triton's [tutorial](https://github.com/openai/triton/blob/5162871c6cae01a8508a309cf21a8e6b68a4c091/python/tutorials/06-fused-attention.py).
"""

import math
import torch
import triton
import triton.language as tl

__all__ = ["attention"]

class PiecewiseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1, k1, q2, k2, v, w, causal, sm_scale):
        """
        PiecewiseAttention

        Piecewise deviates from standard scaled dot product attention in that takes 
        as inputs two q's and two k's as inputs. The attention score is dot product 
        of (q1, k1) or (q2, k2) depending on whether the distance between q & k 
        exceeds a threshold.

        Arguments:
            q1(torch.Tensor): The first queries. The shape is (batch_size, nheads, seqlen_q, headdim).
            k1(torch.Tensor): The first keys. The shape is (batch_size, nheads, seqlen_k, headdim).
            q2(torch.Tensor): The second queries. The shape is (batch_size, nheads, seqlen_q, headdim).
            k2(torch.Tensor): The second keys. The shape is (batch_size, nheads, seqlen_k, headdim).
            v(torch.Tensor): The values. The shape is (batch_size, nheads, seqlen_k, headdim).
            w(int): The threshold of distance between q and k. When the distance is not greater than w, the attention score is dot(q1, k1); otherwise dot(q2, k2).
            causal(bool): Whether causal masking is applied to attention scores before applying softmax.
            sm_scale(float): The scaling of attention scores before applying softmax.

        Returns:
            out: (torch.Tensor): The output. The shape is (batch_size, nheads, seqlen_q, headdim).
        """
        o, L = standalone_forward(q1, k1, q2, k2, v, w, causal, sm_scale)

        ctx.save_for_backward(q1, k1, q2, k2, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.w = w
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()  

        q1, k1, q2, k2, v, o, L = ctx.saved_tensors
        w = ctx.w
        causal = ctx.causal
        sm_scale = ctx.sm_scale

        dq1, dk1, dq2, dk2, dv = standalone_backward(q1, k1, q2, k2, v, w, causal, sm_scale, o, L, do)

        return dq1, dk1, dq2, dk2, dv, None, None, None

def standalone_forward(q1, k1, q2, k2, v, w, causal, sm_scale):
    # to work around https://github.com/openai/triton/issues/2441
    orginal_device_index = torch.cuda.current_device()
    device_index = q1.device.index
    torch.cuda.set_device(device_index)

    # shape constraints
    Dq1, Dk1, Dq2, Dk2, Dv = q1.shape[-1], k1.shape[-1], q2.shape[-1], k2.shape[-1], v.shape[-1]
    assert Dq1 == Dk1 == Dq2 == Dk2 == Dv
    assert Dk1 in {16, 32, 64, 128}
    o = torch.empty_like(q1)
    
    # tune for A100, device_capability(8, 0)
    if torch.cuda.get_device_capability(device_index) == (8, 0): 
        BLOCK_M = 128
        BLOCK_N = 32 if Dk1 <=64 else 64
        # piecewise attention use more shm than flash attention
        num_stages = 3
        num_warps = 4 if Dk1 <=64 else 8
    else: # tune for RTX-3090, device_capability(8, 6)
        BLOCK_M = 128 if Dk1 <=64 else 64
        BLOCK_N = 64
        # piecewise attention use more shm than flash attention
        num_stages = 2
        num_warps = 4

    B, H, M, D = q1.shape
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    N = k1.shape[2]

    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    L = torch.empty((B * H, M), device=q1.device, dtype=torch.float32)
    P_SEQ = N - M
    # assume rotated q & rotated k
    _fwd_kernel[grid](
        q1, k1, q2, k2, v, sm_scale,
        L,
        o,
        q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
        k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
        q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
        k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, M, P_SEQ,
        w = w, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    torch.cuda.set_device(orginal_device_index)
    return o, L

def standalone_backward(q1, k1, q2, k2, v, w, causal, sm_scale, o, L, do):
    # to work around https://github.com/openai/triton/issues/2441
    orginal_device_index = torch.cuda.current_device()
    device_index = q1.device.index
    torch.cuda.set_device(device_index)

    B, H, M, D = q1.shape
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # tune for A100, device_capability(8, 0)
    if torch.cuda.get_device_capability(device_index) == (8, 0): 
        BLOCK_M = 64 if D<=64 else 128
        BLOCK_N = 64
        num_stages = 1 if D<=64 else (2 if not causal else 1)
        num_warps = 4 if D <=64 else 8
    else: # tune for RTX-3090, device_capability(8, 6)
        BLOCK_M = 64
        BLOCK_N = 64 if D <=64 else 32
        num_stages = 1
        num_warps = 4

    N = k1.shape[2]
    P_SEQ = N - M
    do = do.contiguous()

    delta = torch.empty_like(L)
    _bwd_preprocess[(triton.cdiv(M, BLOCK_M) * B * H, 1, 1)](
        o, do,
        delta,
        B * H * M,
        BLOCK_M=BLOCK_M, D_HEAD=D,
    )

    dk1 = torch.empty_like(k1)
    dk2 = torch.empty_like(k2)
    dv = torch.empty_like(v)
    grid = (triton.cdiv(N, BLOCK_N), B * H, 1)
    _bwd_kv_kernel[grid](
        q1, k1, q2, k2, v, sm_scale, do, 
        dk1,dk2, dv,
        L, delta,
        q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
        k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
        q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
        k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk1.stride(0), dk1.stride(1), dk1.stride(2), dk1.stride(3),
        dk2.stride(0), dk2.stride(1), dk2.stride(2), dk2.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        q1.shape[0], q1.shape[1], q1.shape[2], P_SEQ, 
        w=w,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        num_stages=num_stages,
        num_warps=num_warps,
        )
    
    dq1 = torch.zeros_like(q1)
    dq2 = torch.zeros_like(q2)
    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    _bwd_q_kernel[grid](
        q1, k1, q2, k2, v, sm_scale, do, 
        dq1, dq2,
        L, delta,
        q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
        k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
        q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
        k2.stride(0), k2.stride(1), k2.stride(2), k2.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq1.stride(0), dq1.stride(1), dq1.stride(2), dq1.stride(3),
        dq2.stride(0), dq2.stride(1), dq2.stride(2), dq2.stride(3),
        q1.shape[0], q1.shape[1], q1.shape[2], P_SEQ, 
        w=w,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        num_stages=num_stages,
        num_warps=num_warps,
        )
    torch.cuda.set_device(orginal_device_index)
    return dq1, dk1, dq2, dk2, dv


def attention(q1, k1, q2, k2, v, dist_threshold, causal=False, sm_scale=None):
    """
    PiecewiseAttention

    Piecewise deviates from standard scaled dot product attention in that takes 
    as inputs two q's and two k's as inputs. The attention score is dot product 
    of (q1, k1) or (q2, k2) depending on whether the distance between q & k 
    exceeds a threshold.

    Arguments:
        q1(torch.Tensor): The first queries. The shape is (batch_size, nheads, seqlen_q, headdim).
        k1(torch.Tensor): The first keys. The shape is (batch_size, nheads, seqlen_k, headdim).
        q2(torch.Tensor): The second queries. The shape is (batch_size, nheads, seqlen_q, headdim).
        k2(torch.Tensor): The second keys. The shape is (batch_size, nheads, seqlen_k, headdim).
        v(torch.Tensor): The values. The shape is (batch_size, nheads, seqlen_k, headdim).
        dist_threshold(int): The threshold of distance between q and k. When the distance is not greater than w, the attention score is dot(q1, k1); otherwise dot(q2, k2).
        causal(bool): Whether causal masking is applied to attention scores before applying softmax.
        sm_scale(float): The scaling of attention scores before applying softmax.

    Returns:
        out: (torch.Tensor): The output. The shape is (batch_size, nheads, seqlen_q, headdim).
    """
    return PiecewiseAttention.apply(q1, k1, q2, k2, v, dist_threshold, causal, sm_scale)

@triton.jit
def _fwd_kernel(
    Q1, K1, Q2, K2, V, sm_scale,
    L,
    O,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_k1z, stride_k1h, stride_k1n, stride_k1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_k2z, stride_k2h, stride_k2n, stride_k2k,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, P_SEQ,
    w: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    # initialize pointers to value-like data 
    q1_ptrs = Q1 + (offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k) # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k) # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k) # (BLOCK_N, BLOCK_DMODEL)
    k2_ptrs = K2 + (offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
    # q1 = (q1 * qk_scale).to(input_dtype)
    q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
    # q2 = (q2 * qk_scale).to(input_dtype)
    # Dot I trick
    # better way to generate a eye matrix. avoid casting
    I = tl.where(offs_k[:, None] == offs_k, 
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype), 
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    q1 = tl.dot(q1, I).to(input_dtype)
    q2 = tl.dot(q2, I).to(input_dtype)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < (N_CTX + P_SEQ)
        valid_mask = mask_m[:, None] & mask_n
        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :] + w)
        # -- load k, v --
        k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
        k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.where(piecewise_mask, 
                       tl.dot(q2, tl.trans(k2), out_dtype=tl.float32), 
                       tl.dot(q1, tl.trans(k1), out_dtype=tl.float32))
        qk *= qk_scale
        if IS_CAUSAL:
            qk = tl.where(causal_mask & valid_mask, qk, float("-inf"))
        else:
            qk = tl.where(valid_mask, qk, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        # Plus 0 trick
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        # acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v, out_dtype=tl.float32)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn

    # write back l
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=mask_m)
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None])


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    M, 
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < M
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :], mask=mask_m[:, None]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :], mask=mask_m[:, None]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta, mask=mask_m)

@triton.jit
def _bwd_kv_kernel(
    Q1, K1, Q2, K2, V, sm_scale, DO,
    DK1, DK2, DV,
    L,
    D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_k1z, stride_k1h, stride_k1n, stride_k1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_k2z, stride_k2h, stride_k2n, stride_k2k,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dk1z, stride_dk1h, stride_dk1n, stride_dk1k,
    stride_dk2z, stride_dk2h, stride_dk2n, stride_dk2k,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, N_CTX, P_SEQ, 
    w: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DK1 += off_z * stride_dk1z + off_h * stride_dk1h
    DK2 += off_z * stride_dk2z + off_h * stride_dk2h
    DV += off_z * stride_dvz + off_h * stride_dvh


    if CAUSAL:
        lo = tl.math.max(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < (N_CTX + P_SEQ)

    # initialize pointers to value-like data 
    q1_ptrs = Q1 + (offs_m_init[:, None] * stride_q1m + offs_k[None, :] * stride_q1k) # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (offs_m_init[:, None] * stride_q2m + offs_k[None, :] * stride_q2k) # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (offs_k[:, None] * stride_k1k + offs_n[None, :] * stride_k1n) # (BLOCK_DMODEL, BLOCK_N)
    k2_ptrs = K2 + (offs_k[:, None] * stride_k2k + offs_n[None, :] * stride_k2n) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
    dk1_ptrs = DK1 + (offs_n[:, None] * stride_dk1n + offs_k[None, :] * stride_dk1k) # (BLOCK_N, BLOCK_DMODEL)
    dk2_ptrs = DK2 + (offs_n[:, None] * stride_dk2n + offs_k[None, :] * stride_dk2k) # (BLOCK_N, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX

    # k and v stay in SRAM throughout
    v = tl.load(v_ptrs, mask=mask_n[:, None])
    k1 = tl.load(k1_ptrs, mask=mask_n[None, :])
    k2 = tl.load(k2_ptrs, mask=mask_n[None, :])

    # initialize dk amd dv
    dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # loop over a column
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + offs_m_base
        mask_m = offs_m < N_CTX
        # load q1, k1, q2, k2, v, do on-chip
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])

        # recompute p = softmax(qk, dim=-1).T
        valid_mask = mask_m[:, None] & mask_n
        causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :] + w) # (BLOCK_M, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.where(piecewise_mask, 
                       tl.dot(q2, k2, out_dtype=tl.float32), 
                       tl.dot(q1, k1, out_dtype=tl.float32))
        qk *= qk_scale
        if CAUSAL:
            qk = tl.where(causal_mask & valid_mask, qk, float("-inf"))
        else:
            qk = tl.where(valid_mask, qk, float("-inf"))

        # -- recompute p ---
        l = tl.load(l_ptrs + offs_m, mask=mask_m)
        p = tl.math.exp2(qk - l[:, None]) # (BLOCK_M, BLOCK_N)

        do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do, out_dtype=tl.float32) # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do.to(input_dtype), tl.trans(v), out_dtype=tl.float32)

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale # (BLOCK_M, BLOCK_N)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0).to(input_dtype)
        ds1 = tl.where(piecewise_mask, 0.0, ds).to(input_dtype)

        # compute dk = dot(ds.T, q) masking
        dk1 += tl.dot(tl.trans(ds1), q1, out_dtype=tl.float32)
        dk2 += tl.dot(tl.trans(ds2), q2, out_dtype=tl.float32)

        # increment pointers
        q1_ptrs += BLOCK_M * stride_q1m
        q2_ptrs += BLOCK_M * stride_q2m
        do_ptrs += BLOCK_M * stride_dom

    tl.store(dk1_ptrs, dk1.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
    tl.store(dk2_ptrs, dk2.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
    tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel(
    Q1, K1, Q2, K2, V, sm_scale, DO,
    DQ1, DQ2,
    L,
    D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_k1z, stride_k1h, stride_k1n, stride_k1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_k2z, stride_k2h, stride_k2n, stride_k2k,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dq1z, stride_dq1h, stride_dq1m, stride_dq1k,
    stride_dq2z, stride_dq2h, stride_dq2m, stride_dq2k,
    Z, H, N_CTX, P_SEQ, 
    w: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DQ1 += off_z * stride_dq1z + off_h * stride_dq1h
    DQ2 += off_z * stride_dq2z + off_h * stride_dq2h



    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    # initialize pointers to value-like data 
    q1_ptrs = Q1 + (offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k) # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k) # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k) # (BLOCK_N, BLOCK_DMODEL)
    k2_ptrs = K2 + (offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)

    dq1_ptrs = DQ1 + (offs_m[:, None] * stride_dq1m + offs_k[None, :] * stride_dq1k) # (BLOCK_M, BLOCK_DMODEL)
    dq2_ptrs = DQ2 + (offs_m[:, None] * stride_dq2m + offs_k[None, :] * stride_dq2k) # (BLOCK_M, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX

    # load q: it will stay in SRAM throughout
    q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
    q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
    do = tl.load(do_ptrs, mask=mask_m[:, None])
    D = tl.load(D_ptrs + offs_m, mask=mask_m)
    l = tl.load(l_ptrs + offs_m, mask=mask_m)

    # initialize dq 
    dq1 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)


    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if CAUSAL else N_CTX + P_SEQ

    # loop over a row
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < (N_CTX + P_SEQ)

        # load k1, k2, v on chip
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
        k2 = tl.load(k2_ptrs, mask=mask_n[:, None])

        # recompute p = softmax(qk, dim=-1).T
        valid_mask = mask_m[:, None] & mask_n
        causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :] + w) # (BLOCK_M, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.where(piecewise_mask, 
                       tl.dot(q2, tl.trans(k2), out_dtype=tl.float32), 
                       tl.dot(q1, tl.trans(k1), out_dtype=tl.float32))
        qk *= qk_scale
        if CAUSAL:
            qk = tl.where(causal_mask & valid_mask, qk, float("-inf"))
        else:
            qk = tl.where(valid_mask, qk, float("-inf"))

        # -- recompute p ---
        p = tl.math.exp2(qk - l[:, None]) # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D[:, None]
        dp += tl.dot(do.to(input_dtype), tl.trans(v), out_dtype=tl.float32)

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale # (BLOCK_M, BLOCK_N)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0).to(input_dtype)
        ds1 = tl.where(piecewise_mask, 0.0, ds).to(input_dtype)

        dq1 += tl.dot(ds1, k1, out_dtype=tl.float32)
        dq2 += tl.dot(ds2, k2, out_dtype=tl.float32)

        # increment pointers
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn

    tl.store(dq1_ptrs, dq1.to(input_dtype), mask=mask_m[:, None])
    tl.store(dq2_ptrs, dq2.to(input_dtype), mask=mask_m[:, None])
    
