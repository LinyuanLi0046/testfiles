#!/usr/bin/env python3
"""Run the production WeLMv4 NPU RoPE operator for msprof-op capture."""

from __future__ import annotations

import argparse
import os
import sys


M = 16384
NUM_Q_HEADS = 6
NUM_K_HEADS = 1
HEAD_DIM = 256
ROPE_DIM = 64
ROPE_BASE = 100000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:5")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--sglang-python-path",
        default=os.environ.get(
            "SGLANG_PYTHON_PATH", "/data2/hw_lly/sglang/python"
        ),
        help="Directory containing the deployed sglang Python package",
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.sglang_python_path))

    import torch
    import torch_npu

    from sglang.srt.layers.welmv4_npu_op import welmv4_inplace_rope_npu

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
    frequencies = torch.outer(
        torch.arange(M, dtype=torch.float32), inv_freq
    )
    cos_sin_cache = torch.cat(
        (frequencies.cos(), frequencies.sin()), dim=-1
    ).to(device=device)

    assert query.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert positions.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.float32

    for _ in range(args.iters):
        welmv4_inplace_rope_npu(
            query,
            key,
            positions,
            cos_sin_cache,
            last_index=None,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        )
    torch_npu.npu.synchronize()

    print(
        "completed "
        f"{args.iters} launches: M={M}, query={tuple(query.shape)} "
        f"{query.dtype}, key={tuple(key.shape)} {key.dtype}, "
        f"cache={cos_sin_cache.dtype}, device={device}"
    )


if __name__ == "__main__":
    main()
