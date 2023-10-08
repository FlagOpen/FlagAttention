import math
import pathlib
import torch
import triton

from flag_attn import piecewise_attn

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64

configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(10, 16)],
    line_arg='provider',
    line_vals=['triton', ] + (['flash'] if HAS_FLASH else []),
    line_names=['triton', ] + ([f'flash-{FLASH_VER}'] if HAS_FLASH else []),
    styles=[('red', '-'), ('green', '-')],
    ylabel='tflop/s',
    plot_name=f'piecewise_attention_batch-{BATCH}_head-{N_HEADS}_d-{D_HEAD}_mode-{mode}_caucal-{causal}_dtype-{dtype}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': dtype, 'mode': mode, 'causal': causal}
) for mode in ['fwd', 'bwd'] for causal in [False, True] for dtype in [torch.float16, torch.bfloat16]] 

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    w = N_CTX // 2 # dist thresold
    warmup = 25
    rep = 100
    if provider == "triton":
        q1 = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k1 = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        q2 = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k2 = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1. / math.sqrt(D_HEAD)
        fn = lambda: piecewise_attn(q1, k1, q2, k2, v, w, causal, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fn = lambda: flash_attn_func(qkv, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # total TFLOPS: following Flash Attention v2, only gemms are counted.
    # NOTE: It is not a fair play here, the total amount of flops and the elapsed time are different,
    # the tflop/s is a used as a metric of the performance of the operator, for refernce only.
    if provider == "flash":
        macs = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD # Q@K, P@V
        if mode == 'bwd':
            macs *= 2.5  # Q@K, dO@V, dO@P, dS@Q dS@K 
    else:
        macs = 3. * BATCH * H * N_CTX * N_CTX * D_HEAD # Q1@K1, Q2@K2, P@V
        if mode == 'bwd':
            macs *= 8/3.0 # Q1@K1, Q2@K2, dO@V, dO@P, dS1@@Q1, dS1@K1, dS2@@Q2, dS2@K2
    total_flops = 2 * macs

    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9

# only works on post-Ampere GPUs right now
output_dir = pathlib.Path("results")
output_dir.mkdir(exist_ok=True)
bench_flash_attention.run(save_path=output_dir, print_data=True)
