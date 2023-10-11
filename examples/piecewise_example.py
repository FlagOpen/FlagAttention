import math
import torch
from flag_attn import piecewise_attn

B, H, T, D = 2, 16, 8192, 128
sm_scale = 1. / math.sqrt(D)
dist_threshold = T // 2

q1 = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_().requires_grad_()
q2 = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_().requires_grad_()
k1 = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_().requires_grad_()
k2 = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_().requires_grad_()
v = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_().requires_grad_()
o = piecewise_attn(q1, k1, q2, k2, v, dist_threshold, causal=True, sm_scale=sm_scale)
print(o)

go = torch.empty((B, H, T, D), dtype=torch.float16, device="cuda:0").normal_()
gq1, gk1, gq2, gk2, gv = torch.autograd.grad(
    o, (q1, k1, q2, k2, v), go
)
print(gq1)


