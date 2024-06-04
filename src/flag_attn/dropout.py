import torch
import triton
import triton.language as tl

def philox_cuda_seed_offset(increment, device=None):
    device = device or torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state_copy = gen.get_state()
    c0, c1 = state_copy.view(torch.int64)
    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset

@triton.jit
def dropout_mask_kernel(dropout_mask, B, H, M, N, dropout_p, seed, offset):
    row, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = b * H * M * N + h * M * N + row * N
    BLOCK: tl.constexpr = 1024
    offs_base += tl.arange(0, BLOCK)
    for start_n in range(0, N, BLOCK):
        offs = start_n + offs_base
        rng_offs = offset + offs
        pmask = tl.rand(seed, rng_offs, n_rounds=6) > dropout_p
        row_mask = start_n + tl.arange(0, BLOCK) < N
        tl.store(dropout_mask + offs, pmask, mask=row_mask)

def dropout_mask(x, B, H, M, N, dropout_p, seed, offset):
    dropout_mask = torch.full((B, H, M, N), True, dtype=torch.bool, device=x.device)
    if dropout_p == 0:
        return dropout_mask
    grid = (M, B, H)
    with torch.cuda.device(x.device):
        dropout_mask_kernel[grid](dropout_mask, B, H, M, N, dropout_p, seed, offset)
    return dropout_mask
