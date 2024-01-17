import math
import torch

import math
import torch

def attention(q,
              k,
              v,
              causal,
              sm_scale=None,
              return_log_normalizer=False,
              return_total_attention=False,
              upcast=False):
    input_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    # (B, H, T, D)
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    kv_seq_len = k.shape[-2]
    q_seq_len = q.shape[-2]
    p_seq = kv_seq_len - q_seq_len # if kv_seq_len > q_seq_len else 0
    device = q.device

    ms = torch.arange(q_seq_len, device=device).unsqueeze(-1)
    ns = torch.arange(kv_seq_len, device=device)

    S = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        S = torch.where(ms + p_seq >= ns, S, float("-inf"))

    S = S.to(torch.float32)
    if return_log_normalizer:
        log_normalizer = torch.logsumexp(S, dim=-1)

    # upcast attention to fp32
    P = torch.softmax(S, dim=-1, dtype=torch.float32)
    if causal:
        P = torch.where(ms + p_seq >= ns, P, 0.0)

    if return_total_attention:
        tot_attn = torch.sum(P, dim=-2)

    attn_output = torch.matmul(P.to(v.dtype), v).to(input_dtype)

    has_extra_return = return_log_normalizer or return_total_attention
    if has_extra_return:
        outs = (attn_output, 
                 log_normalizer if return_log_normalizer else None, 
                 tot_attn if return_total_attention else None)
        return outs
    else:
        return attn_output
