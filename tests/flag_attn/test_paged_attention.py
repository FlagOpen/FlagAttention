import torch
import pytest

import triton
import triton.language as tl

import flag_attn

# from paged import paged_attention
# from testing.paged import paged_attention as torch_paged_attention


NUM_BLOCKS = 1000

@pytest.mark.parametrize('num_seqs', [1, 32])
@pytest.mark.parametrize('num_query_heads', [64])
@pytest.mark.parametrize('query_group_size', [1, 8])
@pytest.mark.parametrize('head_size', [64])
@pytest.mark.parametrize('block_size', [16, 256])
@pytest.mark.parametrize('max_seq_len', [4096])
@pytest.mark.parametrize('version', [1, 2])
def test_paged_attention(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    version,
    dtype=torch.float16,
    device="cuda",
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    # torch.cuda.manual_seed(seed)

    num_kv_heads = num_query_heads // query_group_size
    # QUERY_GROUP_SIZE = num_query_heads // num_kv_heads
    
    # 初始化每个 seq 的长度
    context_lens = torch.randint(1, max_seq_len, [num_seqs],dtype=torch.int32)
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
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq))
    
    print("version: ",version)
    flag_attn.paged_attention(
        out,
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        version = version,
    )

    ref_out = torch.empty_like(out)
    flag_attn.testing.paged_attention(
        ref_out,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
    )
    print(torch.abs(out - ref_out).max())
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-5)
