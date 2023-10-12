# FlagAttention

[中文版](./README_cn.md)

FlagAttention is a project for memory-efficient attention operators implemented in the Triton language. It is inspired by [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention v2](https://tridao.me/publications/flash2/flash2.pdf) and extends them to satisfy the needs for research on large language modeling. FlashAttention and FlashAttention-2 save memory footprint and traffic to improve memory efficiency, but to modify them and add more options and functionalities requires precision in cuda programming. Thus, Flag Attention is implemented in the Triton language, which is easier to use to write custom GPU kernels.

The operators provided by FlagAttention is memory-efficient and fasts fast, like FlashAttention, which scales large language models to longer sequences. As a out-of-the-box collection of efficient  attention operators, FlagAttention balances efficiency and generality. FlagAttention makes extensions to the basic functionalities rather than tailor an operator for every detail of a specific model. PiecewiseAttention is currently used for inference in the [Aquila 34B](https://github.com/FlagAI-Open/Aquila2) model, but it can also be used by other models.

When further customization is needed, FlagAttention can also a reference or starting point.

## Requirements

FlagAttention requires Pytorch and Triton. To use new features of Triton, Triton nightly is recommended.

Instructions to install Pytorch nightly can be found at https://pytorch.org/get-started/locally/ . Triton is now a dependency of torch nightly, so it can be installed automatically.

FlagAttention requires Ampere Nvidia GPUs(e.g. A100, RTX-3090, ...) and CUDA Toolkit 11.6 and above. Other GPUs may work but not been tested yet.

## Installation

FlagAttention can be installed in either way below.

1. Editable Installation. Changes to the code in local source tree are effective without re-installation.
2. Build a distribution and then install. Only the package is installed.

### Editable Installation

Editable installation with pip.

```sh
git clone https://github.com/FlagOpen/FlagAttention && cd FlagAttention
pip install -e .
```

### Build a Distribution & Install

Following modern python packaging convention, FlagAttention is configured by [`pyproject.toml`](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/), and no `setup.py` is provided. To build a distribution, either a source distribution or a binary distribution, python package `build` is recommended.

First, install `build` package via pip.

```sh
pip install build
```

Then build the package.

```sh
git clone https://github.com/FlagOpen/FlagAttention && cd FlagAttention
python -m build --no-isolation
```

The built package is in `dist/` for installation.

```sh
pip install dist/flag_attn-xxx.whl
```

## Usage

FlagAttention provides customized attention operators. When an operator is equivalent to a torch function, it can be used as a drop-in replacement.

## Run the Tests

A recent version of `pytest`(>=7.1.0) is required to run the tests in `tests/`. Operators in `FlagAttention` are tested against a [reference implementation](tests/flag_attn/ref_impl) in pytorch, both forward and backward. For operators with support for inputs of `float16` or `bfloat16`, three different implementations are included for numerical accuracy testing.

1. The implementation used as reference is an implementation in PyTorch which upcasts the inputs to `float32` and performs the computations in `float32` all the way through before casting the outputs to `float16` or `bfloat16`. 
2. The implementation in Triton usually uses `float16` or `bfloat16` for mma(matrix multiplication accumulation) inputs, and `float32` for mma outputs and other computations.
3. The implementation for comparison is an implementation in PyTorch, with the same computation as the reference, except that the precision is the same as the Triton implementation.

The tests for numerical accuracy enforces that the max difference between the Triton implementation and reference implementation is not greater than twice of the max difference between the Pytorch implementation and reference implementation.

```sh
pytest .
```

## Run the Benchmark

Benchmarks are provided to measure the TFLOPs/s achieved. FLOPs/s is used as a metric for speed of the operator. To calculate the FLOPs of an operator, only matmul is counted. The FLOPs is divided by the median runtime to get the achieved FLOPs/s.

We benchmark operators in Triton implementation against a reference implementation in Pytorch. When the input size is large, the reference implementation in Pytorch runs out of memory. In such cases, the FLOP/s is treated as zero.

The speed of `Flash Attention v2` (https://github.com/Dao-AILab/flash-attention, v2.2.3) with the same size of inputs is also provided as a reference. But since operators in `FlagAttention` deviates from Flash Attention, the total amount of computation is different even when batch size, sequence length, number of heads, head dimension, and other configurations are the same. 

## Operators

### Piecewise Attention

The first extension of Flash Attention is [piecewise attention](src/flag_attn/piecewise.py).

The interface is show below.

```python
piecewise_attention(q1, k1, q2, k2, v, dist_threshold, softmax_scale=None, causal=False)
```

It is named `piecewise_attention` in that it takes two `q`'s and two `k`'s to compute attention scores (S) before applying softmax to get the attention weights (P). The design originates from the fact that a transformer with rotary position embedding is not good at predicting sequences longer than the longest sequence that it is trained on. Pair of (q, k) gets unexpectedly high attention scores when the distance is greater the max sequence length in training set. A proposal to solve the problem is to compute the attention score in different ways, depending on whether the distance between `q` and `k` is greater than a threshold.

In practice, `q` and `k` can be preprocessed in two different ways to get `q1, q2` and `k1, k2`. Then then attention score is computed as the dot product of `q1, k1` or `q2, k2` depending on the distance between `q` and `k`.

![piecewise attention](assets/piecewise_attention.png)

#### Usage

```python
from flag_attn import piecewise_attn

B, H, T, D = 2, 16, 8192, 128
sm_scale = 1. / math.sqrt(D)
dist_threshold = T // 2

q1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
q2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
k1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
k2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
v = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
o = piecewise_attn(q1, k1, q2, k2, v, dist_threshold, causal=True, sm_scale=sm_scale)
print(o)
```

#### Performance

Performance for piecewise_attention with causal masking on A100 is show below. Testing parameters are

1. seqlen in `[512, 1k, 2k, 4k, 16k, 32k]`;
2. batch size: `32k / seqlen`;
3. headdim in`[64, 128]`；
4. num_heads: 2048 / headdim.

Headdim=64
![headdim64, A100, causal](./assets/headdim64-causal-A100.png)

---

Headdim=128
![headdim128, A100, causal](./assets/headdim128-causal-A100.png)

#### Features

- support for [Nvidia](https://www.nvidia.com/) Ampere GPU(Tested on RTX-3090 and A100)；
- support for [Iluvatar CoreX](https://www.iluvatar.com/) GPU(Tested on Iluvatar CoreX MR-V100)；
- data type support, float16 and bfloat16 for Ampere Nvidia GPUs;
- support causal and non-causal modes.
- support forward & backward modes;
- the sequence length of k/v can be larger than that of q.

#### Limitations

- headdim should be in `[16, 32, 64, 128]`.
- dropout of attention weights is not supported yet.

## TODOs

1. Test on other GPUs;
2. Test on more triton versions 
3. Improve performance of attention operators.
2. Support other extensions to flash attention.
