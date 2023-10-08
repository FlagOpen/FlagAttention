# FlagAttention

FlagAttention is a project for memory-efficient attention operators implemented in the Triton language. It is inspired by [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlahAttention v2](https://tridao.me/publications/flash2/flash2.pdf) and extends them to satisfy the needs for research on large language modeling. FlashAttention and FlashAttention-2 save memory footprint and traffic to improve memory efficiency, but to modify them and add more options and functionalities requires precision in cuda programming. Thus, Flag Attention is implemented in the Triton language, which is easier to use to write custom GPU kernels.

## Installation

## Requirements

FlagAttention requires Torch and Triton. To use new features of Triton, Triton nightly is recommended.

Instructions for installing Torch nightly can be found at https://pytorch.org/get-started/locally/ . Triton is now a dependency of torch nightly, so it can be installed automatically.

FlagAttention requires Ampere Nvidia GPUs(e.g. A100, RTX-3090, ...) and CUDA Toolkit 11.6 and above. Other GPUs may work but not been tested yet.

FlagAttention can be installed in either way below.

1. Editable Installation. This includes tests and benchmarks. Changes to the code in local source tree are effective without re-installation.
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

A recent version of `pytest`(>=7.1.0) is required to run the tests in `tests/`. Operators in `FlagAttention` are tested against a [reference implementation](tests/flag_attn/ref_impl) in pytorch, both forward and backward. For `float16` and `bfloat16`, we set absolute and relative tolerance to `1e-2` and `1e-3`, respectively.

```sh
pytest .
```

## Run the Benchmark

Benchmarks are provided to measure the TFLOPs/s achieved. Since operators in `FlagAttention` deviates from Flash Attention, the total amount of computation is different even when batch size, sequence length, number of heads, head dimension, and other configurations are the same. To calculate the FLOPs of an operator, only matmuls are counted. The FLOPs is divided by the median runtime to get the achieved FLOPs/s.

## Operators

### Piecewise Attention

The first extension of Flash Attention is [piecewise attention](src/flag_attn/piecewise.py).

The interface is show below.

```python
piecewise_attention(q1, k1, q2, k2, v, dist_threshold, softmax_scale=None, causal=False)
```

It is named `piecewise_attention` in that it takes two `q`'s and two `k`'s to compute attention scores (S) before applying softmax to get the attention weights (P). The design originates from the fact that a transformer with rotary position embedding is not good at predicting sequences longer than the longest sequence that it is trained on. Pair of (q, k) gets unexpectedly high attention scores when the distance is greater the max sequence length in traing set. A proposal to solve the problem is to compute the attention score in different ways, depending on whether the distance between `q` and `k` is greater than a threshold.

In practice, `q` and `k` can be preprocessed in two different ways to get `q1, q2` and `k1, k2`. Then then attention score is computed as the dot product of `q1, k1` or `q2, k2` depending on the distance between `q` and `k`.

![piecewise attention](assets/piecewise_attention.png)

Features:

- the sequence length of k/v can be larger than that of q;
- data type support, float16 and bfloat16 for Ampere Nvidia GPUs;
- support causal and non-causal modes.
- support forward & backward modes.

Limitations:

- headdim should be in `[16, 32, 64, 128]`.
- dropout of attention weights is not supported yet.

## TODOs

1. Test on other GPUs;
2. Test on more triton versions 
3. Improve performance of attention operators.
2. Support other extensions to flash attention.
