import math
import logging
import pathlib
import torch
import triton

import flag_attn


try:
    from flash_attn import flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None


configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(9, 16)],
    line_arg='provider',
    line_vals=['flag_attn', 'torch', ] + (['flash'] if HAS_FLASH else []),
    line_names=['flag_attn', 'torch', ] + ([f'flash-{FLASH_VER}'] if HAS_FLASH else []),
    styles=[('red', '-'), ('green', '-'), ('blue', '-'), ('cyan', '-')],
    ylabel='tflop/s',
    plot_name=f'attention_d-{D_HEAD}_mode-{mode}_causal-{causal}_dtype-{dtype}',
    args={'D_HEAD': D_HEAD, 'dtype': dtype, 'mode': mode, 'causal': causal}
) for mode in ['fwd', 'bwd'] 
    for causal in [False, True]
    for D_HEAD in [64, 128]
    for dtype in [torch.float16, torch.bfloat16]]

@triton.testing.perf_report(configs)
def bench_flash_attention(N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    w = N_CTX // 2 # dist thresold
    warmup = 25
    rep = 100

    BATCH = 32768 // N_CTX
    H = 2048 // D_HEAD
    if provider == "flag_attn":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: flag_attn.flash_attention(q, k, v, causal=causal)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "torch":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)

        try:
            fn = lambda: flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=False)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except torch.cuda.OutOfMemoryError as e:
            logging.info(f"torch OOM for batch_size: {BATCH}, num_heads: {H}, seqlen: {N_CTX}, headdim: {D_HEAD}")
            ms = float("inf")
    if provider == "flash":
        if FLASH_VER == 1:
            qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            q = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            fn = lambda: flash_attn_func(q, k, v, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # total TFLOPS: following Flash Attention v2, only gemms are counted.
    macs = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD # Q@K, P@V
    if mode == 'bwd':
        macs *= 2.5  # Q@K, dO@V, dO@P, dS@Q dS@K 
    total_flops = 2 * macs

    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9

# only works on post-Ampere GPUs right now
output_dir = pathlib.Path("results_flash_attention")
output_dir.mkdir(exist_ok=True)
bench_flash_attention.run(save_path=output_dir, print_data=True)
