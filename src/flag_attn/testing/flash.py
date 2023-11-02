import math
import torch

import math
import torch
import pytest

def attention(q, k, v, causal, sm_scale=None, upcast=False):
    input_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    # (B, H, T, D)
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    kv_seq_len = k.shape[-2]
    q_seq_len = q.shape[-2]
    p_seq = kv_seq_len - q_seq_len
    device = q.device

    ms = torch.arange(q_seq_len, device=device).unsqueeze(-1)
    ns = torch.arange(kv_seq_len, device=device)
    
    S = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        S = torch.where(ms + p_seq >= ns, S, float("-inf"))

    # upcast attention to fp32
    P = torch.softmax(S, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(P.to(v.dtype), v)
    return attn_output.to(input_dtype)
