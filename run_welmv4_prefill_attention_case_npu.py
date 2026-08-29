#!/usr/bin/env python3
"""Launch one named Full/SWA case repeatedly for external NPU tooling."""

from __future__ import annotations

import argparse

import torch
import torch_npu

from attention_contract import DEFAULT_TP_SIZE, find_case
from bench_welmv4_prefill_attention_npu import DEFAULT_SEED, Harness, run_compile_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--provider", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    device = torch.device(args.device)
    if device.type != "npu":
        parser.error("--device must be npu:N")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    case = find_case(args.case_name, tp_size=args.tp_size)
    harness = Harness(device, args.seed, "manual")
    run_compile_only(harness, case, args.provider, args.iterations)


if __name__ == "__main__":
    main()

