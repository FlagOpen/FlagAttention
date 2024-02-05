import torch
import triton
import triton.language as tl
from typing import Optional


def paged_attention(
    out: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    partition_size: int = 512,
    version: int = 1,
) -> None:
    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads
    max_num_blocks_per_seq = block_tables.shape[1]
    query_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = key_cache.stride(1)

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)
    # FIXME: Remove these constraints.
    assert head_size in [64, 128, 256], f"head_size={head_size}"
    assert kv_block_size >= 16
    assert query_group_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # use_v1 = False
    # dev_prop = torch.cuda.get_device_properties(0)
    # num_sms = dev_prop.multi_processor_count
    # if num_seqs * num_kv_heads > num_sms: # or partition_size != 0:
    #   use_v1 = True
    use_v1 = version == 1

    if use_v1:
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_v1_kernel[grid](
            out,
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            max_num_blocks_per_seq,
            attn_scale,
            query_stride,
            kv_block_stride,
            kv_head_stride,
            head_size,
            query_group_size,
            padded_group_size,
            num_kv_heads,
            kv_block_size,
        )
        # print_config_v1()
        return _paged_attn_v1_kernel.best_config

    else:
        # max_num_partitions = triton.cdiv(max_context_len, v2_partition_size)
        max_num_partitions = triton.cdiv(max_context_len, 64)
        grid = lambda META: (
            num_seqs,
            num_kv_heads,
            triton.cdiv(max_context_len, META["PARTITION_SIZE"]),
        )
        # grid = (num_seqs, num_kv_heads, max_num_partitions)
        m_i = torch.empty(
            size=(num_seqs, num_kv_heads, max_num_partitions, query_group_size),
            dtype=torch.float32,
            device=out.device,
        )
        l_i = torch.empty_like(m_i)
        tmp_out = torch.empty(
            size=(
                num_seqs,
                num_kv_heads,
                max_num_partitions,
                query_group_size,
                head_size,
            ),
            dtype=out.dtype,
            device=out.device,
        )
        _paged_attn_v2_kernel[grid](
            m_i,
            l_i,
            tmp_out,
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            max_num_blocks_per_seq,
            attn_scale,
            query_stride,
            kv_block_stride,
            kv_head_stride,
            head_size,
            query_group_size,
            padded_group_size,
            num_kv_heads,
            kv_block_size,
            # v2_partition_size,
        )
        # print_config_v2()

        v2_partition_size = _paged_attn_v2_kernel.best_config.kwargs["PARTITION_SIZE"]
        # WARNING
        assert(v2_partition_size >= kv_block_size)
        max_num_partitions = triton.cdiv(max_context_len, v2_partition_size)
        reduce_grid = (num_seqs, num_kv_heads)
        _paged_attn_v2_reduce_kernel[reduce_grid](
            out,
            m_i,
            l_i,
            tmp_out,
            context_lens,
            max_num_partitions,
            head_size,
            query_group_size,
            num_kv_heads,
            v2_partition_size,
            triton.next_power_of_2(max_num_partitions),
        )
        return _paged_attn_v2_kernel.best_config


def run_once(func):
    def wrapper():
        if not wrapper.has_run:
            func()
            wrapper.has_run = True

    wrapper.has_run = False
    return wrapper


@run_once
def print_config_v1():
    print("triton v1: ", _paged_attn_v1_kernel.best_config)


@run_once
def print_config_v2():
    print("triton v2: ", _paged_attn_v2_kernel.best_config)


def get_v1_configs():
    configs = []
    for num_stages in [1, 2, 3, 4]:
        for num_warps in [1, 2, 4, 8]:
            configs.append(
                triton.Config({}, num_stages=num_stages, num_warps=num_warps)
            )
    return configs


@triton.autotune(
    configs=get_v1_configs(),
    key=["KV_BLOCK_SIZE", "HEAD_SIZE", "PADDED_QUERY_GROUP_SIZE"],
)
@triton.jit
def _paged_attn_v1_kernel(
    out_ptr,
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,
    context_lens_ptr,
    block_tables_ptr,
    max_num_blocks_per_seq,
    attn_scale,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    _paged_attn_kernel(
        out_ptr,
        out_ptr,
        out_ptr,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        context_lens_ptr,
        block_tables_ptr,
        max_num_blocks_per_seq,
        attn_scale,
        Q_STRIDE,
        KV_BLOCK_STRIDE,
        KV_HEAD_STRIDE,
        HEAD_SIZE,
        QUERY_GROUP_SIZE,
        PADDED_QUERY_GROUP_SIZE,
        NUM_KV_HEADS,
        KV_BLOCK_SIZE,
        PARTITION_SIZE=0,
    )

# todo dynamic config base on KV_BLOCK_SIZE
def get_v2_configs():
    configs = []
    for partition_size in [256, 512, 1024]:
        for num_stages in [1, 2, 3, 4]:
            for num_warps in [1, 2, 4, 8]:
                configs.append(
                    triton.Config(
                        {"PARTITION_SIZE": partition_size},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs

def get_v2_one_configs():
    configs = []
    configs.append(triton.Config({"PARTITION_SIZE": 128}, num_stages=3, num_warps=8))
    return configs

@triton.autotune(
    configs=get_v2_configs(),
    key=["KV_BLOCK_SIZE", "HEAD_SIZE", "PADDED_QUERY_GROUP_SIZE"],
)
@triton.jit
def _paged_attn_v2_kernel(
    m_i_ptr,
    l_i_ptr,
    out_ptr,
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,
    context_lens_ptr,
    block_tables_ptr,
    max_num_blocks_per_seq,
    attn_scale,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    _paged_attn_kernel(
        m_i_ptr,
        l_i_ptr,
        out_ptr,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        context_lens_ptr,
        block_tables_ptr,
        max_num_blocks_per_seq,
        attn_scale,
        Q_STRIDE,
        KV_BLOCK_STRIDE,
        KV_HEAD_STRIDE,
        HEAD_SIZE,
        QUERY_GROUP_SIZE,
        PADDED_QUERY_GROUP_SIZE,
        NUM_KV_HEADS,
        KV_BLOCK_SIZE,
        PARTITION_SIZE,
    )


@triton.jit
def _paged_attn_kernel(
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    max_num_blocks_per_seq,
    attn_scale,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = partition_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            # Early exit.
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx, KV_BLOCK_SIZE)
    else:
        context_start_idx = 0
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    kv_offset = kv_head_idx * KV_HEAD_STRIDE
    kv_offset += block_offset[:, None] * HEAD_SIZE + head_offset[None, :]

    # Load queries.
    query_offset = seq_idx * Q_STRIDE + kv_head_idx * QUERY_GROUP_SIZE * HEAD_SIZE
    query_offset += (
        tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    group_mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] < QUERY_GROUP_SIZE
    # query: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    query = tl.load(q_ptr + query_offset, mask=group_mask, other=0.0)

    # Initialize accumulators.
    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    # NOTE: KV_BLOCK_SIZE must be >= 16.
    NUM_BLOCKS_PER_PARTITION = PARTITION_SIZE // KV_BLOCK_SIZE
    num_prev_blocks = partition_idx * NUM_BLOCKS_PER_PARTITION
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks_per_seq + block_idx
        )

        # Load a key block.
        kv_block_offset = block_number * KV_BLOCK_STRIDE + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        kv_mask = mask_offset[:, None] < context_len
        # key: [KV_BLOCK_SIZE, HEAD_SIZE]
        key = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        # Compute attention.
        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        if PADDED_QUERY_GROUP_SIZE == 1:
            query = tl.reshape(query, (1, HEAD_SIZE)) # tl.view for triton 2.1.0
            qk = tl.sum(query * key, axis=1)[None, :]
            # qk = tl.sum(query[:, None, :] * key[None, :, :], axis=2) # hang error for block_size 256
        else:
            qk = tl.dot(query, key.T, out_dtype=tl.float32)

        qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float("-inf"))

        # Compute m, l, and p.
        # m_ij: [PADDED_QUERY_GROUP_SIZE]
        m_ij = tl.max(qk, axis=1)
        # m_i_new: [PADDED_QUERY_GROUP_SIZE]
        m_i_new = tl.maximum(m_i, m_ij)

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.exp(qk - m_i_new[:, None])
        # alpha: [PADDED_QUERY_GROUP_SIZE]
        alpha = tl.exp(m_i - m_i_new)
        acc *= alpha[:, None]

        # Load a value block.
        # value: [KV_BLOCK_SIZE, HEAD_SIZE]
        value = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            p = tl.reshape(p, (1, KV_BLOCK_SIZE))
            acc += tl.sum(p.T * value, axis=0)[None, :]
            # acc += tl.sum(p.T[:, :, None] * value[:, None, :], axis=0)
        else:
            p = p.to(value.dtype)
            acc += tl.dot(p, value, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    # Store the current partition's m and l for later reduction.
    if USE_PARTITIONING:
        partition_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx)
            * max_num_partitions
            * QUERY_GROUP_SIZE
        )
        partition_offset += partition_idx * QUERY_GROUP_SIZE
        partition_offset += tl.arange(0, PADDED_QUERY_GROUP_SIZE)
        mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE) < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + partition_offset, m_i, mask=mask)
        tl.store(l_i_ptr + partition_offset, l_i, mask=mask)

    # NOTE: Unlike the query tensor, we assume the out tensor is contiguous.
    out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx)
        * max_num_partitions
        * QUERY_GROUP_SIZE
        * HEAD_SIZE
    )
    out_offset += partition_idx * QUERY_GROUP_SIZE * HEAD_SIZE
    out_offset += (
        tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    group_mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(
    out_ptr,  # [num_seqs, NUM_KV_HEADS, QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    max_num_partitions,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = (
        tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    if num_partitions == 1:
        # No reduction needed. Only copy tmp_out to out.
        tmp_out_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx)
            * max_num_partitions
            * QUERY_GROUP_SIZE
            * HEAD_SIZE
        )
        tmp_out_offset += group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)
        out_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx) * QUERY_GROUP_SIZE * HEAD_SIZE
        )
        out_offset += group_head_offset
        tl.store(out_ptr + out_offset, tmp_out)
        return

    # Get the global max logit.
    offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
    )
    offset += (
        tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE
        + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    )
    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    # m_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + offset, mask=mask, other=float("-inf"))
    # m: [QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [QUERY_GROUP_SIZE]
    l = tl.sum(l_i, axis=0)
    # r: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    r = l_i / l[None, :]

    # Aggregate tmp_out to out.
    tmp_out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx)
        * max_num_partitions
        * QUERY_GROUP_SIZE
        * HEAD_SIZE
    )
    tmp_out_offset += (
        tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE
    )
    tmp_out_offset += tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE
    tmp_out_offset += tl.arange(0, HEAD_SIZE)[None, None, :]
    # tmp_out: [NUM_PARTITIONS, QUERY_GROUP_SIZE, HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    # out: [QUERY_GROUP_SIZE, HEAD_SIZE]
    out = tl.sum((tmp_out * r[:, :, None]).to(tl.float32), axis=0)

    # Store output.
    out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * QUERY_GROUP_SIZE * HEAD_SIZE
    out_offset += group_head_offset
    tl.store(out_ptr + out_offset, out)
