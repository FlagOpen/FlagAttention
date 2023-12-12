import math
import torch
import triton
import triton.language as tl
from typing import Optional

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        sm_scale: float,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # switch device context
        orginal_device_index = torch.cuda.current_device()
        device_index = q.device.index
        torch.cuda.set_device(device_index)

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv
        assert Dk in {16, 32, 64, 128}

        B, H, M, D = q.shape
        N = k.shape[2]
        P_SEQ = N - M
        
        # tune for A100, device_capability(8, 0)
        if torch.cuda.get_device_capability(device_index) == (8, 0): 
            if not causal:
                if Dk <= 64:
                    BLOCK_M = 128 
                    BLOCK_N = 64
                    num_stages = 3
                    num_warps = 4
                else:
                    if M <= 1024:
                        BLOCK_M = 128
                        BLOCK_N = 32
                        num_stages = 3
                        num_warps = 4
                    else:
                        BLOCK_M = 128 
                        BLOCK_N = 128
                        num_stages = 3
                        num_warps = 8
            else:
                if Dk <= 64:
                    BLOCK_M = 128 
                    BLOCK_N = 64
                    num_stages = 4
                    num_warps = 4
                else:
                    if M <= 1024:
                        BLOCK_M = 128
                        BLOCK_N = 32
                        num_stages = 2
                        num_warps = 4
                    else:
                        BLOCK_M = 128 
                        BLOCK_N = 128
                        num_stages = 3
                        num_warps = 8
        else: # tune for RTX-3090, device_capability(8, 6)
            if not causal:
                if Dk <= 64:
                    BLOCK_M = 128 
                    BLOCK_N = 64
                    num_stages = 3
                    num_warps = 4
                else:
                    BLOCK_M = 128
                    BLOCK_N = 32
                    num_stages = 2
                    num_warps = 4
            else: # causal
                if Dk <= 64:
                    BLOCK_M = 64
                    BLOCK_N = 64
                    num_stages = 3
                    num_warps = 4
                else:
                    BLOCK_M = 128
                    BLOCK_N = 32
                    num_stages = 2
                    num_warps = 4

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # consider using 3d grid to avoid div & rem
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        _fwd_kernel[grid](
            # TODO: add bias
            q, k, v, sm_scale,
            L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            # TODO: add bias.stride
            B, H, M, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D, IS_CAUSAL=causal,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n, 
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, bias, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = D
        ctx.P_SEQ = P_SEQ
        ctx.causal = causal

        # restore device context
        torch.cuda.set_device(orginal_device_index)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, bias, o, L = ctx.saved_tensors

        # switching device context
        orginal_device_index = torch.cuda.current_device()
        device_index = q.device.index
        torch.cuda.set_device(device_index)

        B, H, M, D = q.shape
        N = k.shape[2]
        P_SEQ = N - M
        sm_scale = ctx.sm_scale
        causal = ctx.causal

        # tune for A100, device_capability(8, 0)
        if torch.cuda.get_device_capability(device_index) == (8, 0):
            if not causal:
                BLOCK_M = 128 if D <= 64 else 64
                BLOCK_N = 64
                num_stages = 2
                num_warps = 4
            else:
                BLOCK_M = 64
                BLOCK_N = 64
                num_stages = 3 if D <= 64 else 2
                num_warps = 4
        else: # tune for RTX-3090, device_capability(8, 6)
            if not causal:
                if D <= 64:
                    BLOCK_M = 64
                    BLOCK_N = 64
                    num_stages = 2
                    num_warps = 4
                else:
                    BLOCK_M = 64
                    BLOCK_N = 64
                    num_stages = 2
                    num_warps = 8
            else:
                if D <= 64:
                    BLOCK_M = 64
                    BLOCK_N = 64
                    num_stages = 2
                    num_warps = 4
                else:
                    BLOCK_M = 32
                    BLOCK_N = 32
                    num_stages = 2
                    num_warps = 4
        
        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0

        delta = torch.empty_like(L)
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        _bwd_preprocess[grid](
            o, do,
            delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            M,
            BLOCK_M=BLOCK_M, D_HEAD=D,
            DIVISIBLE_M=divisible_m,
        )

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = None
        if bias is not None:
            dbias = torch.empty_like(bias)

        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _bwd_kv_kernel[grid](
            # TODO: add bias (needed to calc Softmax)
            q, k, v, sm_scale, do, 
            dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            # TODO: add bias.stride
            q.shape[0], q.shape[1], q.shape[2], P_SEQ, 
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            num_stages=num_stages, num_warps=num_warps,
        )

        dq = torch.zeros_like(q) # us float32 for atomic updates
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        _bwd_q_kernel[grid](
            # TODO: add bias (needed to calc Softmax)
            q, k, v, sm_scale, do,
            dq,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            # TODO: add bias.stride
            q.shape[0], q.shape[1], q.shape[2], P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            num_stages=num_stages, num_warps = num_warps,
        )

        torch.cuda.set_device(orginal_device_index)
        return dq, dk, dv, None, None, None, dbias

def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    bias: Optional[torch.Tensor] = None
):
    return FlashAttention.apply(q, k, v, causal, sm_scale)


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, 
    IS_CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * N_CTX # l's shape is (B, H, N_CTX)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data 
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < N_CTX
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
    
    #Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q).to(input_dtype)

    if IS_CAUSAL:
        hi = P_SEQ + (start_m + 1) * BLOCK_M
    else:
        hi = N_CTX + P_SEQ

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        
        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < (N_CTX + P_SEQ)
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)
        
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    acc = acc * (1.0 / l_i[:, None])
    l = m_i * sm_scale + tl.log(l_i) # log(normalizer)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")



@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)        
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    if not DIVISIBLE_M:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)
    
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, sm_scale, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, N_CTX, P_SEQ, 
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    # offset pointers for batch/head
    D += (off_z * H + off_h) * N_CTX
    L += (off_z * H + off_h) * N_CTX

    if CAUSAL:
        lo = tl.math.max(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # initialize pointers to value-like data 
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk) # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < (N_CTX + P_SEQ)
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # loop over a col
    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < N_CTX
            valid_mask = mask_m[:, None] # & mask_n
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # s = tl.where(valid_mask, s , float("-inf"))
        # if CAUSAL:
        #     s = tl.where(causal_mask, s, float("-inf"))

        # -- recompute p ---
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e) # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do) # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds.to(input_dtype)

        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(tl.trans(ds), q)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    dk *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL,)
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, sm_scale, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    Z, H, N_CTX, P_SEQ, 
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, 
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * N_CTX
    L += (off_z * H + off_h) * N_CTX

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data 
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk) # (BLOCK_M, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
    else:
        mask_m = offs_m < N_CTX
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)

    # initialize dq 
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    hi = P_SEQ + (start_m + 1) * BLOCK_M if CAUSAL else N_CTX + P_SEQ

    # loop over a row
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        
        # load k1, k2, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
        else:
            mask_n = offs_n < (N_CTX + P_SEQ)
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])


        # recompute p = softmax(qk * sm_scale, dim=-1)
        if not DIVISIBLE_N:
            valid_mask = mask_n # & mask_m[:, None] 
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e) # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))
        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dq at last
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    
    dq *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
