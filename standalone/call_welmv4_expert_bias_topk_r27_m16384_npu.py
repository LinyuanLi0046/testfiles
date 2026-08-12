#!/usr/bin/env python3
"""Invoke the exact WeLM-v4 R27 expert-bias TopK once for M=16384.

Input:
    scores:      FP32 [16384, 512]
    expert_bias: FP32 [512]
Output:
    weights:     FP32 [16384, 10]
    expert_ids:  INT64 [16384, 10]

The kernel selects IDs by clean_score + expert_bias, returns the selected
unbiased clean scores, maps score NaN to -inf, and preserves the original MMQ
tie priority.  This file intentionally contains no benchmark/timing loop.
"""

import argparse

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.utils import is_npu


if not is_npu():
    raise RuntimeError("This script must be run in an Ascend NPU environment.")

init_npu_backend()

M = 16384
NUM_EXPERTS = 512
TOPK = 10
OUTPUT_WIDTH = 16

# Triton 3.2 requires captured numeric globals to be explicit constexpr values.
_JIT_NUM_EXPERTS = tl.constexpr(NUM_EXPERTS)
_JIT_TOPK = tl.constexpr(TOPK)
_JIT_OUTPUT_WIDTH = tl.constexpr(OUTPUT_WIDTH)


@triton.jit
def _r27_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, out_lanes, 0)
    next_lanes = tl.minimum(out_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        out_lanes.to(tl.float32) < 9.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    if ~has_duplicate:
        matches = top_values[:, None] == routing_scores[None, :]
        selected_rank = tl.min(
            tl.where(matches, tie_rank[None, :], _JIT_NUM_EXPERTS + 1),
            axis=1,
        )
        selected_lane = selected_rank >> 4
        selected_local = selected_rank & 15
        selected_ids = (
            ((selected_local >> 2) << 7)
            + (selected_lane << 2)
            + (selected_local & 3)
        ).to(tl.int32)
    else:
        invalid_rank = _JIT_NUM_EXPERTS + 1
        previous_value = float("inf")
        previous_rank = -1.0
        for k in tl.static_range(0, _JIT_TOPK):
            kth_routing_score = al.get_element(sorted_routing, indice=[k])
            same_group = kth_routing_score == previous_value
            eligible = (routing_scores == kth_routing_score) & (
                ~same_group | (tie_rank.to(tl.float32) > previous_rank)
            )
            selected_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank),
                axis=0,
            )
            selected_lane = selected_rank >> 4
            selected_local = selected_rank & 15
            selected_idx = (
                ((selected_local >> 2) << 7)
                + (selected_lane << 2)
                + (selected_local & 3)
            )
            selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
            previous_value = kth_routing_score
            previous_rank = selected_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r27_m16384_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)

    offs = tl.arange(0, _JIT_NUM_EXPERTS)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _r27_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


def _query_num_vector_cores() -> int:
    device_index = torch_npu.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    for key in ("num_vectorcore", "num_vectorcores", "num_vector_cores"):
        if key in props and int(props[key]) > 0:
            return int(props[key])
    raise RuntimeError(
        "Triton device properties do not expose the AIV count; "
        f"available keys: {sorted(props.keys())}"
    )


class ModelNew(torch.nn.Module):
    """Fixed-shape, pure-Triton R27 launcher."""

    def __init__(self) -> None:
        super().__init__()
        self.num_vector_cores = _query_num_vector_cores()

    def forward(
        self,
        scores: torch.Tensor,
        expert_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scores.shape != (M, NUM_EXPERTS):
            raise ValueError(f"scores must have shape {(M, NUM_EXPERTS)}")
        if expert_bias.shape != (NUM_EXPERTS,):
            raise ValueError(f"expert_bias must have shape {(NUM_EXPERTS,)}")
        if scores.dtype != torch.float32 or expert_bias.dtype != torch.float32:
            raise TypeError("scores and expert_bias must both be FP32")
        if not scores.is_contiguous() or not expert_bias.is_contiguous():
            raise ValueError("scores and expert_bias must be contiguous")
        if scores.device != expert_bias.device or scores.device.type != "npu":
            raise ValueError("scores and expert_bias must be on the same NPU")

        weights = torch.empty((M, TOPK), dtype=torch.float32, device=scores.device)
        expert_ids = torch.empty((M, TOPK), dtype=torch.int64, device=scores.device)

        program_count = min(M, self.num_vector_cores)
        rows_per_program, extra_rows = divmod(M, program_count)
        _r27_m16384_kernel[(program_count,)](
            scores,
            expert_bias,
            weights,
            expert_ids,
            rows_per_program,
            extra_rows,
            multibuffer=False,
            unit_flag=False,
        )
        return weights, expert_ids


def _parse_device(device_text: str) -> tuple[torch.device, int]:
    device = torch.device(device_text)
    if device.type != "npu" or device.index is None:
        raise ValueError("--device must be explicit, for example npu:5")
    return device, int(device.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call WeLM-v4 R27 expert-bias TopK once at M=16384."
    )
    parser.add_argument("--device", default="npu:5")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device, device_index = _parse_device(args.device)
    torch_npu.npu.set_device(device_index)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scores_cpu = torch.sigmoid(
        torch.randn((M, NUM_EXPERTS), generator=generator, dtype=torch.float32)
    )
    bias_cpu = 0.02 * torch.randn(
        NUM_EXPERTS, generator=generator, dtype=torch.float32
    )
    scores = scores_cpu.to(device)
    expert_bias = bias_cpu.to(device)

    model = ModelNew()
    weights, expert_ids = model(scores, expert_bias)
    torch_npu.npu.synchronize()

    print(
        "R27 call completed: "
        f"device={device}, physical_AIV={model.num_vector_cores}, "
        f"weights={tuple(weights.shape)}/{weights.dtype}, "
        f"expert_ids={tuple(expert_ids.shape)}/{expert_ids.dtype}"
    )


if __name__ == "__main__":
    main()
