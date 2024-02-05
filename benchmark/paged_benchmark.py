import torch

import triton
import triton.language as tl

import flag_attn


NUM_BLOCKS = 1000
warmup = 200
rep = 200

try:
    from vllm._C import ops as vllm_ops

    HAS_VLLM = True
    import vllm

    print("vllm.__version__", vllm.__version__)
except BaseException:
    HAS_VLLM = False


def vllm_paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    PARTITION_SIZE: int = 512,
    version: int = 1,
):

    if version == 1:
        # block_size 8, 16, 32
        vllm_ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            None,  # alibi_slopes
            "auto",  # kv_cache_dtype for vllm 0.3.0
        )
    elif version == 2:
        num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        vllm_ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            None,
            "auto",  # vllm 0.3.0
        )
    else:
        raise AssertionError(f"Unknown version: {version}")


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["context_len"],
            x_vals=[2**i for i in range(9, 15)],
            line_arg="provider",
            line_vals=["triton"] + (["vllm"] if HAS_VLLM else []),
            line_names=["triton"] + ([f"vllm-{vllm.__version__}"] if HAS_VLLM else []),
            styles=[("red", "-"), ("blue", "-")],
            ylabel="tflop/s",
            plot_name=f"vllm_paged_attention-B{num_seqs}-G{query_group_size}-D{head_size}-block_size{block_size}-v{version}",
            args={
                "num_seqs": num_seqs,
                "num_query_heads": 64,
                "query_group_size": query_group_size,
                "head_size": head_size,
                "block_size": block_size,
                "version": version,
                "dtype": dtype,
            },
        )
        for num_seqs in [1, 32]
        for query_group_size in [1, 8]
        for head_size in [64, 128]
        for block_size in [16, 32]
        for version in [1, 2]
        for dtype in [torch.float16]
    ]
)
def paged_attention_benchmark_with_vllm(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    context_len,
    version,
    provider,
    dtype=torch.float16,
    device="cuda",
):
    num_kv_heads = num_query_heads // query_group_size

    context_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device) + context_len
    max_num_blocks_per_seq = (context_len + block_size - 1) // block_size

    attn_scale = head_size**-0.5
    q = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype, device=device)
    q.uniform_(-attn_scale, attn_scale)
    out = torch.empty_like(q)

    k_cache = torch.empty(
        NUM_BLOCKS, num_kv_heads, block_size, head_size, dtype=dtype, device=device
    )
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(
        0,
        NUM_BLOCKS,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    if provider == "triton":
        fn = lambda: flag_attn.paged_attention(
            out,
            q,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            attn_scale,
            context_len,
            version=version,
        )
        print(f"vllm_paged_attention-B{num_seqs}-G{query_group_size}-D{head_size}-block_size{block_size}-v{version}: ",fn())
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "vllm":
        # Correctness error, does not affect performance results
        fn = lambda: vllm_paged_attention(
            out,
            q,
            k_cache,
            v_cache,
            num_kv_heads,
            attn_scale,
            block_tables,
            context_lens,
            block_size,
            context_len,
            PARTITION_SIZE=512,
            version=1,
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    total_flops = (
        2.0 * num_seqs * num_query_heads * 2 * context_len * head_size
    )
    return total_flops / ms * 1e-9

if HAS_VLLM:
    paged_attention_benchmark_with_vllm.run(
        save_path="./vllm_result", print_data=True
    )  # show_plots=True,


try:
    from flash_attn import flash_attn_with_kvcache

    HAS_FLASH = True
    import flash_attn

    print("flash_attn.__version__", flash_attn.__version__)
except BaseException:
    HAS_FLASH = False


# MAX_SEQ_LEN = 4096
@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["context_len"],
            x_vals=[2**i for i in range(9, 15)],
            line_arg="provider",
            line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
            line_names=["triton"] + ([f"flash-{flash_attn.__version__}"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-")],
            ylabel="tflop/s",
            plot_name=f"flash_paged_attention-B{num_seqs}-G{query_group_size}-D{head_size}-block_size{block_size}-v{version}",
            args={
                "num_seqs": num_seqs,
                "num_query_heads": 64,
                "query_group_size": query_group_size,
                "head_size": head_size,
                "block_size": block_size,
                "version": version,
                "dtype": dtype,
            },
        )
        for num_seqs in [1, 32]
        for query_group_size in [1, 8]
        for head_size in [64, 128]
        for block_size in [256]
        for version in [1, 2]
        for dtype in [torch.float16]
    ]
)
def paged_attention_benchmark_with_flash_attn(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    context_len,
    version,
    provider,
    dtype=torch.float16,
    device="cuda",
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    # torch.cuda.manual_seed(seed)

    num_kv_heads = num_query_heads // query_group_size
    # QUERY_GROUP_SIZE = num_query_heads // num_kv_heads
    
    # 初始化每个 seq 的长度
    context_lens = torch.randint(1, context_len, [num_seqs],dtype=torch.int32)
    # context_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device) + context_len
    max_context_len = context_lens.max().item()
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    attn_scale = head_size ** -0.5
    q = torch.empty(num_seqs, num_query_heads, head_size)
    q.uniform_(-attn_scale, attn_scale)
    out = torch.empty_like(q)

    k_cache = torch.empty(NUM_BLOCKS, num_kv_heads,block_size, head_size)
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq),dtype=torch.int32)

    total_flops = (
        2.0 * num_seqs * num_query_heads * 2 * context_len * head_size
    )

    if provider == "triton":
        fn = lambda: flag_attn.paged_attention(
            out,
            q,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            attn_scale,
            max_context_len,#context_len,
            version=version,
        )
        print(f"flash_paged_attention-B{num_seqs}-G{query_group_size}-D{head_size}-block_size{block_size}-v{version}: ",fn())
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return total_flops / ms * 1e-9
    
    if provider == "flash":
        q_flash = q.unsqueeze(1)
        k_cache_flash = k_cache.permute(0,2,1,3).contiguous()
        v_cache_flash = v_cache.permute(0,2,1,3).contiguous()

        fn = lambda: flash_attn_with_kvcache(
            q_flash,
            k_cache_flash,
            v_cache_flash,
            k = None,
            v = None,
            rotary_cos = None,
            rotary_sin = None,
            cache_seqlens = context_lens,
            cache_batch_idx = None,
            block_table = block_tables,
            softmax_scale = attn_scale,
            causal = False,
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return total_flops / ms * 1e-9


if HAS_FLASH:
    paged_attention_benchmark_with_flash_attn.run(
        save_path="./flash_result", print_data=True
    )  # show_plots=True,

