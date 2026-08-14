#!/usr/bin/env python3
"""Run the verified FP32-cache WeLMv4 prefill kernel for msprof-op."""

from __future__ import annotations

import argparse

import torch
import torch_npu
import triton
import triton.language as tl


M = 16384
NUM_Q_HEADS = 6
NUM_K_HEADS = 1
HEAD_DIM = 256
ROPE_DIM = 64
ROPE_BASE = 100000.0
TOKEN_BLOCK = 64
NUM_STAGES = 1
PROGRAMS_PER_VECTOR_CORE = 8


@triton.jit
def _welmv4_apply_token_block_rope_npu(
    data_ptr: tl.tensor,
    token_offsets: tl.tensor,
    token_stride: tl.constexpr,
    cos: tl.tensor,
    sin: tl.tensor,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    rope_offsets = tl.arange(0, half_rope_dim)
    base = data_ptr + token_offsets[:, None] * token_stride
    x1 = tl.load(base + rope_offsets[None, :], care_padding=False)
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, :],
        care_padding=False,
    )
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, :], out1)
    tl.store(base + half_rope_dim + rope_offsets[None, :], out2)


@triton.jit
def _welmv4_apply_token_head_block_rope_npu(
    data_ptr: tl.tensor,
    token_offsets: tl.tensor,
    token_stride: tl.constexpr,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    head_offsets = tl.arange(0, num_heads)
    rope_offsets = tl.arange(0, half_rope_dim)
    base = (
        data_ptr
        + token_offsets[:, None, None] * token_stride
        + head_offsets[None, :, None] * head_dim
    )
    x1 = tl.load(base + rope_offsets[None, None, :], care_padding=False)
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, None, :],
        care_padding=False,
    )
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, None, :], out1)
    tl.store(base + half_rope_dim + rope_offsets[None, None, :], out2)


@triton.jit(do_not_specialize=["num_token_blocks", "N"])
def _welmv4_inplace_rope_prefill_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    num_token_blocks: int,
    N: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    num_stages: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)

    for block_id in tl.range(
        tl.program_id(0),
        num_token_blocks,
        tl.num_programs(0),
        num_stages=num_stages,
    ):
        token_base = block_id * token_block
        position_ids = tl.load(
            position_ptr + token_base + token_offsets
        ).to(tl.int32)
        cos = tl.load(
            cos_sin_cache_ptr
            + position_ids[:, None] * rope_dim
            + cos_offsets[None, :],
            care_padding=False,
        )
        sin = tl.load(
            cos_sin_cache_ptr
            + position_ids[:, None] * rope_dim
            + sin_offsets[None, :],
            care_padding=False,
        )

        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim
        _welmv4_apply_token_block_rope_npu(
            k_data,
            token_offsets,
            k_token_stride,
            cos,
            sin,
            head_dim,
            rope_dim,
        )

        # The verified A5 FP32-cache path: 2+2+2 Q heads leaves enough UB
        # for ping-pong multibuffering while preserving FP32 RoPE arithmetic.
        q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
        _welmv4_apply_token_head_block_rope_npu(
            q_data,
            token_offsets,
            q_token_stride,
            cos,
            sin,
            2,
            head_dim,
            rope_dim,
        )
        _welmv4_apply_token_head_block_rope_npu(
            q_data + 2 * head_dim,
            token_offsets,
            q_token_stride,
            cos,
            sin,
            2,
            head_dim,
            rope_dim,
        )
        _welmv4_apply_token_head_block_rope_npu(
            q_data + 4 * head_dim,
            token_offsets,
            q_token_stride,
            cos,
            sin,
            2,
            head_dim,
            rope_dim,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:5")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    torch.manual_seed(0)

    positions = torch.arange(M, dtype=torch.int64, device=device)
    query = torch.randn(
        (M, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        (M, NUM_K_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    frequencies = torch.outer(torch.arange(M, dtype=torch.float32), inv_freq)
    cos_sin_cache = torch.cat(
        (frequencies.cos(), frequencies.sin()), dim=-1
    ).to(device=device)

    assert query.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert positions.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.float32
    assert M % TOKEN_BLOCK == 0

    num_token_blocks = M // TOKEN_BLOCK
    device_index = int(torch_npu.npu.current_device())
    properties = triton.runtime.driver.active.utils.get_device_properties(
        device_index
    )
    num_vector_cores = int(
        properties.get("num_vectorcore", properties.get("num_aicore", -1))
    )
    if num_vector_cores <= 0:
        raise RuntimeError("could not determine the visible NPU vector-core count")
    num_programs = min(
        num_token_blocks, num_vector_cores * PROGRAMS_PER_VECTOR_CORE
    )

    for _ in range(args.iters):
        _welmv4_inplace_rope_prefill_kernel_npu[(num_programs,)](
            query,
            key,
            positions,
            cos_sin_cache,
            num_token_blocks,
            M,
            query.stride(0),
            key.stride(0),
            HEAD_DIM,
            ROPE_DIM,
            TOKEN_BLOCK,
            NUM_STAGES,
            multibuffer=True,
        )
    torch_npu.npu.synchronize()

    print(
        "completed "
        f"{args.iters} launches: M={M}, query={tuple(query.shape)} "
        f"{query.dtype}, key={tuple(key.shape)} {key.dtype}, "
        f"cache={cos_sin_cache.dtype}, token_block={TOKEN_BLOCK}, "
        f"q_split=2+2+2, num_stages={NUM_STAGES}, "
        f"multibuffer=True, programs={num_programs}, device={device}"
    )


if __name__ == "__main__":
    main()
