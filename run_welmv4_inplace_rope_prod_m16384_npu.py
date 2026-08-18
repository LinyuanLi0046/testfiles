#!/usr/bin/env python3
"""Launch the formal ordinary Prefill JIT at M=16384 for msprof op."""

from __future__ import annotations

import argparse

import torch
import torch_npu

from bench_welmv4_inplace_rope_npu import (
    ALL_CASES,
    Harness,
    audit_candidate_prefill_jit_contract,
    make_inputs,
)


CASE_NAME = "prefill_m16384"
KERNEL_NAME = "_candidate_welmv4_inplace_rope_prefill_kernel"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:5")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    audit_candidate_prefill_jit_contract()
    device = torch.device(args.device)
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    case = next(item for item in ALL_CASES if item.name == CASE_NAME)
    harness = Harness(device, seed=20260814)
    query, key, positions, last_index, segment_tile_starts = make_inputs(
        case, device, harness.seed
    )
    launch = harness.bind(
        "candidate",
        case,
        query,
        key,
        positions,
        last_index,
        segment_tile_starts,
    )

    for _ in range(args.iters):
        launch()
    torch_npu.npu.synchronize()
    print(
        f"completed {args.iters} launches: case={CASE_NAME}, "
        f"kernel_name={KERNEL_NAME}, device={device}; use "
        f"msprof op --kernel-name={KERNEL_NAME}"
    )


if __name__ == "__main__":
    main()
