#!/usr/bin/env python3
"""A5-only accuracy and latency study for WeLM-v4 expert-bias TopK.

This is deliberately a manual, standalone experiment.  It does not replace or
register any production SGLang operator.  R0 is a frozen 2026-08-11 copy of the
old production kernel from ``sglang.srt.layers.welmv4_op``; R1--R6 are
cumulative experimental variants kept here so that every round remains
measurable after a later optimized kernel is integrated into the model path.

Semantic contract (the part that must stay bit-exact):

* input ``scores`` is FP32 [M, 512] and ``expert_bias`` is finite FP32 [512];
* score NaN is replaced by -inf;
* expert IDs are selected by ``clean_score + expert_bias``;
* returned weights are the selected *clean, unbiased* scores;
* ties use the original MMQ lane/copy priority, not smaller expert ID;
* output is FP32 [M, 10] weights and INT64 [M, 10] ordered expert IDs.

Rounds:

* R0: frozen exact copy of the old production kernel.
* R1: specialize the common contiguous N=512/K=10 path.
* R2: cap grid at the physical AIV count and give each AIV a contiguous row
  range (bias is still loaded once per row).
* R3: replace 10 x 4 full-width reductions with one FP32 vector sort plus ten
  exact tie-rank reductions.  The first draft's uint64 packed key is avoided on
  A5 because int64 vector arithmetic can scalar-lower.  Selected weights are
  gathered once from the row-resident FP32 vector.
* R4: hoist the invariant 512-element bias load outside the per-AIV row loop.
* R5: host dispatch: one-row/no-loop kernel for a small grid, partitioned
  kernel for a grid larger than the physical AIV count.  The large-grid body is
  intentionally the R4 body, so R5's incremental gain is expected on decode.
* R6: empirical two-path dispatch from the R0--R5 A5 measurements.  M <= 256
  reuses R3 (the winner at every measured small/medium shape); larger M reuses
  the R5 large partition path.  This round changes only host dispatch.

Run from the NEWSGLANG ``sglang`` directory on an A5 machine, for example:

    PYTHONPATH=python python sglang/test/manual/layers/moe/\
      bench_welmv4_expert_bias_topk_npu.py --mode both --cases all

Useful shorter runs:

    # Default: highlighted regimes plus representative decode batches.
    ... --cases common

    # Exhaustive M=1..64 decode sweep plus prefill/boundary shapes.
    ... --cases all

    # Exact semantic stress tests only.
    ... --mode correctness --cases common

    # Kernel-only timing, all M=1..64 plus prefill cases, CSV output.
    ... --mode performance --scope kernel --cases all --output-csv result.csv

The script intentionally uses ``torch_npu.npu.Event``.  Do not replace it with
``torch.cuda.Event`` merely because torch-npu exposes ``Tensor.is_cuda=True``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.utils import is_npu


if not is_npu():
    raise RuntimeError("This manual benchmark must be run on an Ascend NPU host.")

init_npu_backend()

NUM_EXPERTS = 512
TOPK = 10
SORT_WIDTH = 512
OUTPUT_WIDTH = 16  # power-of-two local buffer; only the first TOPK lanes store
R6_DISPATCH_CUTOFF_M = 256

# Triton 3.2 rejects ordinary module globals referenced from a @triton.jit
# function.  Keep plain ints for the host-side harness and expose separate,
# explicitly typed compile-time constants to the experimental kernels.
_JIT_NUM_EXPERTS = tl.constexpr(NUM_EXPERTS)
_JIT_TOPK = tl.constexpr(TOPK)
_JIT_SORT_WIDTH = tl.constexpr(SORT_WIDTH)
_JIT_OUTPUT_WIDTH = tl.constexpr(OUTPUT_WIDTH)

VARIANTS = (
    "r0_legacy",
    "r1_constexpr",
    "r2_core_partition",
    "r3_sort_tie",
    "r4_bias_hoist",
    "r5_dispatch",
    "r6_empirical_dispatch",
)

VARIANT_DESCRIPTIONS = {
    "r0_legacy": "production kernel",
    "r1_constexpr": "N512/K10 contiguous specialization",
    "r2_core_partition": "physical-AIV contiguous row partition",
    "r3_sort_tie": "FP32 sort + exact tie recovery",
    "r4_bias_hoist": "bias load hoisted outside row loop",
    "r5_dispatch": "small direct / large partition dispatch",
    "r6_empirical_dispatch": "R3 through M=256 / R5-large above M=256",
}

COMMON_SHAPES = [1, 2, 4, 8, 16, 32, 56, 63, 64, 65, 128, 512, 9616, 16384]
DECODE_SHAPES = list(range(1, 65))
PREFILL_SHAPES = [
    65,
    127,
    128,
    129,
    256,
    257,
    511,
    512,
    513,
    1024,
    9616,
    16384,
]
ALL_CORRECTNESS_SHAPES = sorted(set(DECODE_SHAPES + PREFILL_SHAPES))
ALL_PERFORMANCE_SHAPES = sorted(
    set(
        DECODE_SHAPES
        + [
            65,
            128,
            256,
            257,
            320,
            384,
            448,
            480,
            511,
            512,
            1024,
            9616,
            16384,
        ]
    )
)

EDGE_PATTERNS = (
    "random_zero_bias",
    "all_tie",
    "discrete_ties",
    "bias_induced_tie",
    "nan_at_least_k_finite",
    "nan_less_than_k_finite",
    "all_nan",
    "score_infinities",
    "one_ulp",
    "signed_zero",
)


# ---------------------------------------------------------------------------
# R0: frozen 2026-08-11 copy of the old production implementation from
# python/sglang/srt/layers/welmv4_op.py::mmq_style_expert_bias_topk_kernel.
# Keep this body unchanged; it is the long-lived before/after baseline.
# ---------------------------------------------------------------------------


@triton.jit
def _r0_legacy_kernel(
    scores_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    M,
    N: tl.constexpr,
    score_stride_m,
    score_stride_n,
    TOPK_ARG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    scores = tl.load(
        scores_ptr + row * score_stride_m + offs * score_stride_n,
        mask=mask,
        other=-float("inf"),
    )
    scores = tl.where(scores == scores, scores, -float("inf"))
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    routing_scores = tl.where(mask, scores + bias, -float("inf"))
    candidate_mask = mask

    elems_per_copy = 4
    copy_stride = 32 * elems_per_copy
    lane_idx = (offs % copy_stride) // elems_per_copy
    local_idx = (offs // copy_stride) * elems_per_copy + (offs % elems_per_copy)
    tie_rank = lane_idx * (N // 32) + local_idx
    invalid_rank = N + 1

    for k in tl.static_range(0, TOPK_ARG):
        max_routing_score = tl.max(routing_scores, axis=0)
        selected_rank = tl.min(
            tl.where(
                (routing_scores == max_routing_score) & candidate_mask,
                tie_rank,
                invalid_rank,
            ),
            axis=0,
        )
        selected_idx = tl.min(
            tl.where(
                (tie_rank == selected_rank) & candidate_mask,
                offs,
                invalid_rank,
            ),
            axis=0,
        )
        selected_score = tl.max(
            tl.where(
                (offs == selected_idx) & candidate_mask,
                scores,
                -float("inf"),
            ),
            axis=0,
        )
        tl.store(topk_weights_ptr + row * TOPK_ARG + k, selected_score)
        tl.store(topk_ids_ptr + row * TOPK_ARG + k, selected_idx)
        candidate_mask = candidate_mask & (offs != selected_idx)
        routing_scores = tl.where(candidate_mask, routing_scores, -float("inf"))


# ---------------------------------------------------------------------------
# R1/R2: exact legacy selection, progressively specialized/partitioned.
# ---------------------------------------------------------------------------


@triton.jit
def _r1_constexpr_kernel(scores_ptr, bias_ptr, weights_ptr, ids_ptr):
    row = tl.program_id(0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    valid = offs < _JIT_NUM_EXPERTS

    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    bias = tl.load(bias_ptr + offs)
    routing_scores = scores + bias
    candidate_mask = valid

    # Keep the production tie-rank expression in R1 so this round changes only
    # shape/stride specialization.  Constants are compile-time values here.
    elems_per_copy = 4
    copy_stride = 128
    lane_idx = (offs % copy_stride) // elems_per_copy
    local_idx = (offs // copy_stride) * elems_per_copy + (offs % elems_per_copy)
    tie_rank = lane_idx * 16 + local_idx
    invalid_rank = _JIT_NUM_EXPERTS + 1

    for k in tl.static_range(0, _JIT_TOPK):
        max_routing_score = tl.max(routing_scores, axis=0)
        selected_rank = tl.min(
            tl.where(
                (routing_scores == max_routing_score) & candidate_mask,
                tie_rank,
                invalid_rank,
            ),
            axis=0,
        )
        selected_idx = tl.min(
            tl.where(
                (tie_rank == selected_rank) & candidate_mask,
                offs,
                invalid_rank,
            ),
            axis=0,
        )
        selected_score = tl.max(
            tl.where(
                (offs == selected_idx) & candidate_mask,
                scores,
                -float("inf"),
            ),
            axis=0,
        )
        tl.store(weights_ptr + row * _JIT_TOPK + k, selected_score)
        tl.store(ids_ptr + row * _JIT_TOPK + k, selected_idx)
        candidate_mask = candidate_mask & (offs != selected_idx)
        routing_scores = tl.where(candidate_mask, routing_scores, -float("inf"))


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r2_core_partition_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    pid = tl.program_id(0)
    # Balanced contiguous partition.  ceil(M / P) chunks create empty AIVs for
    # shapes such as M=65/P=56; quotient/remainder gives every AIV work and the
    # row counts differ by at most one.  Division/modulo stay on the host.
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)

    offs = tl.arange(0, _JIT_SORT_WIDTH)
    valid = offs < _JIT_NUM_EXPERTS
    elems_per_copy = 4
    copy_stride = 128
    lane_idx = (offs % copy_stride) // elems_per_copy
    local_idx = (offs // copy_stride) * elems_per_copy + (offs % elems_per_copy)
    tie_rank = lane_idx * 16 + local_idx
    invalid_rank = _JIT_NUM_EXPERTS + 1

    for row in tl.range(row_start, row_end):
        scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
        scores = tl.where(scores == scores, scores, -float("inf"))
        # Intentionally still inside the row loop; R4 isolates the hoist.
        bias = tl.load(bias_ptr + offs)
        routing_scores = scores + bias
        candidate_mask = valid

        for k in tl.static_range(0, _JIT_TOPK):
            max_routing_score = tl.max(routing_scores, axis=0)
            selected_rank = tl.min(
                tl.where(
                    (routing_scores == max_routing_score) & candidate_mask,
                    tie_rank,
                    invalid_rank,
                ),
                axis=0,
            )
            selected_idx = tl.min(
                tl.where(
                    (tie_rank == selected_rank) & candidate_mask,
                    offs,
                    invalid_rank,
                ),
                axis=0,
            )
            selected_score = tl.max(
                tl.where(
                    (offs == selected_idx) & candidate_mask,
                    scores,
                    -float("inf"),
                ),
                axis=0,
            )
            tl.store(weights_ptr + row * _JIT_TOPK + k, selected_score)
            tl.store(ids_ptr + row * _JIT_TOPK + k, selected_idx)
            candidate_mask = candidate_mask & (offs != selected_idx)
            routing_scores = tl.where(
                candidate_mask, routing_scores, -float("inf")
            )


# ---------------------------------------------------------------------------
# R3+: rank-1 Ascend vector sort and exact MMQ tie recovery.
# ---------------------------------------------------------------------------


@triton.jit
def _sort_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """Select one row; all local vectors stay rank-1 for A5 vsort/vgather."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    candidate_mask = offs < _JIT_NUM_EXPERTS
    invalid_rank = _JIT_NUM_EXPERTS + 1
    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)

    for k in tl.static_range(0, _JIT_TOPK):
        # k is compile-time, so get_element is a fixed vector-element extract.
        kth_routing_score = al.get_element(sorted_routing, indice=[k])
        selected_rank = tl.min(
            tl.where(
                (routing_scores == kth_routing_score) & candidate_mask,
                tie_rank,
                invalid_rank,
            ),
            axis=0,
        )

        # tie_rank is a bijection for N=512.  Invert it algebraically and avoid
        # the production kernel's separate selected-index reduction.
        lane = selected_rank >> 4
        local = selected_rank & 15
        selected_idx = ((local >> 2) << 7) + (lane << 2) + (local & 3)

        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        candidate_mask = candidate_mask & (offs != selected_idx)

    # F32 rank-1 UB gather is supported on A5; ids are explicitly I32.  This
    # preserves the exact unbiased score bits, including signed zero.
    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r3_sort_tie_kernel(
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

    offs = tl.arange(0, _JIT_SORT_WIDTH)
    # Bitwise form is exactly the production copy/lane priority for N=512.
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)

    for row in tl.range(row_start, row_end):
        # Intentionally repeated per row; R4 is the isolated bias-hoist round.
        bias = tl.load(bias_ptr + offs)
        _sort_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r4_bias_hoist_kernel(
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

    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _sort_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit
def _r5_small_direct_kernel(scores_ptr, bias_ptr, weights_ptr, ids_ptr):
    row = tl.program_id(0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)
    _sort_select_store_row(
        scores_ptr,
        weights_ptr,
        ids_ptr,
        row,
        offs,
        tie_rank,
        bias,
    )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r5_large_partition_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Kept separate from R4 so CANN traces identify the R5 provider by name.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)

    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _sort_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


# ---------------------------------------------------------------------------
# Host launchers.  No production call site is modified.
# ---------------------------------------------------------------------------


def _query_num_vector_cores() -> int:
    device_index = torch_npu.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    for key in ("num_vectorcore", "num_vectorcores", "num_vector_cores"):
        if key in props:
            value = int(props[key])
            if value > 0:
                return value
    raise RuntimeError(
        "Triton device properties do not expose the physical AIV count; "
        f"available keys: {sorted(props.keys())}"
    )


class ModelNew(torch.nn.Module):
    """Experimental launcher suite required by the standalone coding contract."""

    def __init__(self) -> None:
        super().__init__()
        self.device_index = int(torch_npu.npu.current_device())
        self.device_name = str(torch_npu.npu.get_device_name(self.device_index))
        self.num_vector_cores = _query_num_vector_cores()

    def runtime_metadata(self, seed: int) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "physical_aiv": self.num_vector_cores,
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(
                getattr(torch_npu, "__version__", "unknown")
            ),
            "triton_version": str(getattr(triton, "__version__", "unknown")),
            "cann_version": str(getattr(torch.version, "cann", "unknown")),
            "seed": seed,
        }

    @staticmethod
    def _validate(scores: torch.Tensor, bias: torch.Tensor) -> None:
        assert scores.ndim == 2 and scores.shape[1] == NUM_EXPERTS
        assert scores.dtype == torch.float32
        assert scores.is_contiguous(), (
            "R1-R6 specialize the actual WeLM common path: contiguous [M,512]"
        )
        assert bias.shape == (NUM_EXPERTS,) and bias.dtype == torch.float32
        assert bias.is_contiguous()
        assert scores.device == bias.device

    @staticmethod
    def allocate_outputs(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = scores.shape[0]
        weights = torch.empty((m, TOPK), dtype=torch.float32, device=scores.device)
        ids = torch.empty((m, TOPK), dtype=torch.int64, device=scores.device)
        return weights, ids

    def _partition(self, m: int) -> tuple[tuple[int], int, int]:
        program_count = min(m, self.num_vector_cores)
        rows_per_program, extra_rows = divmod(m, program_count)
        assert rows_per_program >= 1
        return (program_count,), rows_per_program, extra_rows

    def bind_kernel(
        self,
        variant: str,
        scores: torch.Tensor,
        bias: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
    ) -> Callable[[], None]:
        """Validate once and bind one concrete JIT launch for kernel timing."""
        self._validate(scores, bias)
        m = scores.shape[0]
        assert weights.shape == (m, TOPK) and weights.dtype == torch.float32
        assert ids.shape == (m, TOPK) and ids.dtype == torch.int64

        if variant == "r0_legacy":

            def launch() -> None:
                # Literal old production launch: no experimental hints.
                _r0_legacy_kernel[(m,)](
                    scores,
                    bias,
                    weights,
                    ids,
                    m,
                    NUM_EXPERTS,
                    scores.stride(0),
                    scores.stride(1),
                    TOPK_ARG=TOPK,
                    BLOCK_SIZE=SORT_WIDTH,
                )

            return launch

        if variant == "r1_constexpr":

            def launch() -> None:
                _r1_constexpr_kernel[(m,)](
                    scores,
                    bias,
                    weights,
                    ids,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        if variant == "r2_core_partition":
            grid, rows_per_program, extra_rows = self._partition(m)

            def launch() -> None:
                _r2_core_partition_kernel[grid](
                    scores,
                    bias,
                    weights,
                    ids,
                    rows_per_program,
                    extra_rows,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        if variant == "r3_sort_tie":
            grid, rows_per_program, extra_rows = self._partition(m)

            def launch() -> None:
                _r3_sort_tie_kernel[grid](
                    scores,
                    bias,
                    weights,
                    ids,
                    rows_per_program,
                    extra_rows,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        if variant == "r4_bias_hoist":
            grid, rows_per_program, extra_rows = self._partition(m)

            def launch() -> None:
                _r4_bias_hoist_kernel[grid](
                    scores,
                    bias,
                    weights,
                    ids,
                    rows_per_program,
                    extra_rows,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        if variant == "r5_dispatch":
            if m <= self.num_vector_cores:

                def launch() -> None:
                    _r5_small_direct_kernel[(m,)](
                        scores,
                        bias,
                        weights,
                        ids,
                        multibuffer=False,
                        unit_flag=False,
                    )

                return launch

            grid, rows_per_program, extra_rows = self._partition(m)

            def launch() -> None:
                _r5_large_partition_kernel[grid](
                    scores,
                    bias,
                    weights,
                    ids,
                    rows_per_program,
                    extra_rows,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        if variant == "r6_empirical_dispatch":
            # Optimization point 12 (grid/multipath specialization), isolated:
            # bind the already-validated winning provider for each measured
            # regime.  No JIT body, grid, compiler hint, or math is changed.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= R6_DISPATCH_CUTOFF_M:

                def launch() -> None:
                    _r3_sort_tie_kernel[grid](
                        scores,
                        bias,
                        weights,
                        ids,
                        rows_per_program,
                        extra_rows,
                        multibuffer=False,
                        unit_flag=False,
                    )

                return launch

            def launch() -> None:
                _r5_large_partition_kernel[grid](
                    scores,
                    bias,
                    weights,
                    ids,
                    rows_per_program,
                    extra_rows,
                    multibuffer=False,
                    unit_flag=False,
                )

            return launch

        raise ValueError(f"Unknown variant: {variant}")

    def launch_into(
        self,
        variant: str,
        scores: torch.Tensor,
        bias: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
    ) -> None:
        self.bind_kernel(variant, scores, bias, weights, ids)()

    def forward(
        self,
        scores: torch.Tensor,
        bias: torch.Tensor,
        variant: str = "r6_empirical_dispatch",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = self.allocate_outputs(scores)
        self.launch_into(variant, scores, bias, weights, ids)
        return weights, ids


# ---------------------------------------------------------------------------
# Independent CPU oracle and adversarial inputs.
# ---------------------------------------------------------------------------


def mmq_tie_rank(num_experts: int = NUM_EXPERTS) -> torch.Tensor:
    expert_id = torch.arange(num_experts, dtype=torch.int64)
    return (
        ((expert_id % 128) // 4) * (num_experts // 32)
        + (expert_id // 128) * 4
        + (expert_id % 4)
    )


def mmq_oracle(
    scores_cpu: torch.Tensor,
    bias_cpu: torch.Tensor,
    *,
    chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch oracle; it never calls the tested Triton implementation."""
    assert scores_cpu.device.type == "cpu" and bias_cpu.device.type == "cpu"
    assert scores_cpu.dtype == bias_cpu.dtype == torch.float32
    assert torch.isfinite(bias_cpu).all(), (
        "The production kernel has no stable contract for NaN/Inf expert bias"
    )

    priority = torch.argsort(mmq_tie_rank(scores_cpu.shape[1]), stable=True)
    all_weights: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []

    for scores in scores_cpu.split(chunk_rows, dim=0):
        clean = torch.where(
            torch.isnan(scores),
            torch.full_like(scores, -float("inf")),
            scores,
        )
        routing = clean + bias_cpu
        positions = torch.argsort(
            routing[:, priority],
            dim=1,
            descending=True,
            stable=True,
        )[:, :TOPK]
        chosen_ids = priority[positions]
        all_weights.append(clean.gather(1, chosen_ids))
        all_ids.append(chosen_ids)

    return torch.cat(all_weights, dim=0), torch.cat(all_ids, dim=0)


def _case_seed(base_seed: int, pattern: str, m: int) -> int:
    return base_seed + 1009 * m + sum((i + 1) * ord(c) for i, c in enumerate(pattern))


def make_cpu_case(
    pattern: str,
    m: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        _case_seed(seed, pattern, m)
    )
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32)

    if pattern in ("random_bias", "random_zero_bias"):
        scores = torch.sigmoid(
            torch.randn((m, NUM_EXPERTS), generator=generator, dtype=torch.float32)
        )
        if pattern == "random_bias":
            bias = 0.02 * torch.randn(
                NUM_EXPERTS, generator=generator, dtype=torch.float32
            )
    elif pattern == "all_tie":
        scores = torch.full((m, NUM_EXPERTS), 0.5, dtype=torch.float32)
    elif pattern == "discrete_ties":
        raw = torch.rand((m, NUM_EXPERTS), generator=generator)
        scores = torch.floor(raw * 8.0) * 0.125
    elif pattern == "bias_induced_tie":
        expert_id = torch.arange(NUM_EXPERTS, dtype=torch.int64)
        one_row = ((expert_id & 3).to(torch.float32) * 0.125).contiguous()
        scores = one_row.repeat(m, 1)
        bias = 1.0 - one_row
    elif pattern in (
        "nan_at_least_k_finite",
        "nan_less_than_k_finite",
        "all_nan",
    ):
        scores = torch.full((m, NUM_EXPERTS), float("nan"), dtype=torch.float32)
        priority = torch.argsort(mmq_tie_rank(), stable=True)
        if pattern == "nan_at_least_k_finite":
            finite_count = 12
        elif pattern == "nan_less_than_k_finite":
            finite_count = 8
        else:
            finite_count = 0
        if finite_count:
            finite_ids = priority[:finite_count]
            finite_values = torch.linspace(
                0.2, 0.8, finite_count, dtype=torch.float32
            )
            scores[:, finite_ids] = finite_values
    elif pattern == "score_infinities":
        scores = torch.sigmoid(
            torch.randn((m, NUM_EXPERTS), generator=generator, dtype=torch.float32)
        )
        scores[:, 0] = float("inf")
        scores[:, 128] = float("inf")
        scores[:, 1] = -float("inf")
        scores[:, 129] = -float("inf")
    elif pattern == "one_ulp":
        scores = torch.full((m, NUM_EXPERTS), 0.5, dtype=torch.float32)
        half = torch.tensor(0.5, dtype=torch.float32)
        scores[:, 511] = torch.nextafter(half, torch.tensor(float("inf")))
        scores[:, 510] = torch.nextafter(half, torch.tensor(-float("inf")))
    elif pattern == "signed_zero":
        scores = torch.zeros((m, NUM_EXPERTS), dtype=torch.float32)
        scores[:, 1::2] = -0.0
    else:
        raise ValueError(f"Unknown input pattern: {pattern}")

    assert scores.is_contiguous() and bias.is_contiguous()
    assert torch.isfinite(bias).all()
    return scores, bias


@dataclass
class ExactComparison:
    ids_equal: bool
    weights_bitwise_equal: bool
    ids_dtype_ok: bool
    first_id_mismatch: str = ""
    first_weight_mismatch: str = ""

    @property
    def ok(self) -> bool:
        return self.ids_equal and self.weights_bitwise_equal and self.ids_dtype_ok


def compare_exact(
    actual_weights: torch.Tensor,
    actual_ids: torch.Tensor,
    expected_weights: torch.Tensor,
    expected_ids: torch.Tensor,
) -> ExactComparison:
    actual_weights = actual_weights.cpu().contiguous()
    actual_ids = actual_ids.cpu().contiguous()
    expected_weights = expected_weights.cpu().contiguous()
    expected_ids = expected_ids.cpu().contiguous()

    ids_dtype_ok = actual_ids.dtype == torch.int64
    ids_equal = ids_dtype_ok and torch.equal(actual_ids, expected_ids)
    actual_bits = actual_weights.view(torch.int32)
    expected_bits = expected_weights.view(torch.int32)
    weights_equal = torch.equal(actual_bits, expected_bits)

    id_detail = ""
    if not ids_equal:
        mismatch = (actual_ids != expected_ids).nonzero(as_tuple=False)
        if mismatch.numel():
            r, c = mismatch[0].tolist()
            id_detail = (
                f"[{r},{c}] actual={actual_ids[r, c].item()} "
                f"expected={expected_ids[r, c].item()}"
            )
        elif not ids_dtype_ok:
            id_detail = f"dtype actual={actual_ids.dtype}, expected=torch.int64"

    weight_detail = ""
    if not weights_equal:
        mismatch = (actual_bits != expected_bits).nonzero(as_tuple=False)
        if mismatch.numel():
            r, c = mismatch[0].tolist()
            weight_detail = (
                f"[{r},{c}] actual={actual_weights[r, c].item()} "
                f"(0x{actual_bits[r, c].item() & 0xFFFFFFFF:08x}) "
                f"expected={expected_weights[r, c].item()} "
                f"(0x{expected_bits[r, c].item() & 0xFFFFFFFF:08x})"
            )

    return ExactComparison(
        ids_equal=ids_equal,
        weights_bitwise_equal=weights_equal,
        ids_dtype_ok=ids_dtype_ok,
        first_id_mismatch=id_detail,
        first_weight_mismatch=weight_detail,
    )


# ---------------------------------------------------------------------------
# Correctness runner.
# ---------------------------------------------------------------------------


def build_correctness_cases(shapes: Sequence[int]) -> list[tuple[str, int]]:
    cases = [("random_bias", m) for m in shapes]
    for m in (1, 7, 64):
        if m in shapes:
            cases.extend((pattern, m) for pattern in EDGE_PATTERNS)
    return cases


def run_correctness(
    model: ModelNew,
    *,
    device: torch.device,
    shapes: Sequence[int],
    variants: Sequence[str],
    seed: int,
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    failures = 0
    cases = build_correctness_cases(shapes)
    print(f"\nCorrectness: {len(cases)} input cases x {len(variants)} variants")

    for case_index, (pattern, m) in enumerate(cases, start=1):
        scores_cpu, bias_cpu = make_cpu_case(pattern, m, seed=seed)
        reference_weights, reference_ids = mmq_oracle(scores_cpu, bias_cpu)

        if pattern == "all_tie":
            expected_tie_ids = torch.tensor(
                [0, 1, 2, 3, 128, 129, 130, 131, 256, 257],
                dtype=torch.int64,
            )
            assert torch.equal(reference_ids[0], expected_tie_ids)

        scores = scores_cpu.to(device)
        bias = bias_cpu.to(device)
        legacy_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
        case_ok = True

        for variant in variants:
            weights, ids = model(scores, bias, variant)
            torch_npu.npu.synchronize()
            weights_cpu = weights.cpu()
            ids_cpu = ids.cpu()
            against_oracle = compare_exact(
                weights_cpu, ids_cpu, reference_weights, reference_ids
            )

            if variant == "r0_legacy":
                legacy_cpu = (weights_cpu, ids_cpu)

            if legacy_cpu is not None:
                against_legacy = compare_exact(
                    weights_cpu, ids_cpu, legacy_cpu[0], legacy_cpu[1]
                )
            else:
                against_legacy = ExactComparison(True, True, True)

            status = against_oracle.ok and against_legacy.ok
            case_ok = case_ok and status
            if not status:
                failures += 1
                print(
                    f"  [FAIL] M={m} pattern={pattern} variant={variant}: "
                    f"oracle(ids={against_oracle.ids_equal}, "
                    f"weights={against_oracle.weights_bitwise_equal}); "
                    f"legacy(ids={against_legacy.ids_equal}, "
                    f"weights={against_legacy.weights_bitwise_equal})"
                )
                if against_oracle.first_id_mismatch:
                    print(f"         oracle id: {against_oracle.first_id_mismatch}")
                if against_oracle.first_weight_mismatch:
                    print(
                        "         oracle weight: "
                        f"{against_oracle.first_weight_mismatch}"
                    )

            records.append(
                {
                    **model.runtime_metadata(seed),
                    "record_type": "correctness",
                    "case": pattern,
                    "m": m,
                    "n": NUM_EXPERTS,
                    "k": TOPK,
                    "variant": variant,
                    "status": "PASS" if status else "FAIL",
                    "ids_equal_oracle": against_oracle.ids_equal,
                    "weights_bitwise_equal_oracle": (
                        against_oracle.weights_bitwise_equal
                    ),
                    "ids_equal_legacy": against_legacy.ids_equal,
                    "weights_bitwise_equal_legacy": (
                        against_legacy.weights_bitwise_equal
                    ),
                    "scope": "wrapper",
                }
            )

        if case_ok:
            print(
                f"  [PASS {case_index:>3}/{len(cases)}] "
                f"M={m:<5} pattern={pattern}"
            )

    return records, failures


# ---------------------------------------------------------------------------
# NPU event timing and performance runner.
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def auto_inner_repeat(m: int) -> int:
    if m <= 64:
        return 200
    if m <= 512:
        return 100
    if m <= 1024:
        return 50
    if m <= 9616:
        return 10
    return 5


def event_sample_us(fn: Callable[[], object], inner_repeat: int) -> float:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(inner_repeat):
        fn()
    end.record()
    torch_npu.npu.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / inner_repeat


def make_timed_callable(
    model: ModelNew,
    variant: str,
    scores: torch.Tensor,
    bias: torch.Tensor,
    scope: str,
) -> Callable[[], object]:
    if scope == "wrapper":
        return lambda: model(scores, bias, variant)

    weights, ids = model.allocate_outputs(scores)
    # Bind variant/grid/partition and validate once.  The timed closure contains
    # only the concrete JIT launch, avoiding Python string dispatch and repeated
    # metadata assertions in short decode measurements.
    return model.bind_kernel(variant, scores, bias, weights, ids)


def run_performance(
    model: ModelNew,
    *,
    device: torch.device,
    shapes: Sequence[int],
    variants: Sequence[str],
    seed: int,
    scope: str,
    warmup: int,
    rounds: int,
    inner_repeat_override: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    print(
        f"\nPerformance: scope={scope}, rounds={rounds}, "
        f"physical_AIV={model.num_vector_cores}"
    )

    for m in shapes:
        scores_cpu, bias_cpu = make_cpu_case("random_bias", m, seed=seed)
        scores = scores_cpu.to(device)
        bias = bias_cpu.to(device)
        functions = {
            variant: make_timed_callable(
                model, variant, scores, bias, scope
            )
            for variant in variants
        }

        # Compilation and warmup are outside all recorded events.
        for variant in variants:
            for _ in range(warmup):
                functions[variant]()
        torch_npu.npu.synchronize()

        inner_repeat = (
            inner_repeat_override
            if inner_repeat_override > 0
            else auto_inner_repeat(m)
        )
        samples = {variant: [] for variant in variants}
        for round_index in range(rounds):
            order: Iterable[str]
            order = variants if round_index % 2 == 0 else reversed(variants)
            for variant in order:
                samples[variant].append(
                    event_sample_us(functions[variant], inner_repeat)
                )

        stats: dict[str, dict[str, float]] = {}
        for variant in variants:
            values = samples[variant]
            stats[variant] = {
                "p20": percentile(values, 0.20),
                "p50": statistics.median(values),
                "p80": percentile(values, 0.80),
                "mean": statistics.fmean(values),
            }

        legacy_p50 = stats.get("r0_legacy", {}).get("p50", float("nan"))
        print(
            f"\nM={m}, N={NUM_EXPERTS}, K={TOPK}, "
            f"inner_repeat={inner_repeat}"
        )
        print(
            "  variant                    p20(us)   p50(us)   p80(us)  "
            "speedup/R0  speedup/prev-selected"
        )
        previous_p50 = float("nan")
        for variant in variants:
            current = stats[variant]
            speedup_r0 = legacy_p50 / current["p50"]
            speedup_previous = previous_p50 / current["p50"]
            print(
                f"  {variant:<26} {current['p20']:>9.3f} "
                f"{current['p50']:>9.3f} {current['p80']:>9.3f} "
                f"{speedup_r0:>10.3f}x {speedup_previous:>12.3f}x"
            )
            records.append(
                {
                    **model.runtime_metadata(seed),
                    "record_type": "performance",
                    "case": "random_bias",
                    "m": m,
                    "n": NUM_EXPERTS,
                    "k": TOPK,
                    "variant": variant,
                    "status": "MEASURED",
                    "scope": scope,
                    "p20_us": current["p20"],
                    "p50_us": current["p50"],
                    "p80_us": current["p80"],
                    "mean_us": current["mean"],
                    "speedup_vs_r0": speedup_r0,
                    "speedup_vs_previous_selected": speedup_previous,
                    "rounds": rounds,
                    "inner_repeat": inner_repeat,
                }
            )
            previous_p50 = current["p50"]

    return records


# ---------------------------------------------------------------------------
# CLI and CSV.
# ---------------------------------------------------------------------------


def parse_shapes(spec: str, *, correctness: bool) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return ALL_CORRECTNESS_SHAPES if correctness else ALL_PERFORMANCE_SHAPES
    if spec == "common":
        return COMMON_SHAPES.copy()
    if spec == "decode":
        return DECODE_SHAPES.copy()
    if spec == "prefill":
        return PREFILL_SHAPES.copy()

    result: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start <= 0 or end < start:
                raise ValueError(f"Invalid shape range: {item}")
            result.update(range(start, end + 1))
        else:
            value = int(item)
            if value <= 0:
                raise ValueError(f"M must be positive: {value}")
            result.add(value)
    if not result:
        raise ValueError("No shapes selected")
    return sorted(result)


def parse_variants(spec: str) -> list[str]:
    normalized = spec.strip().lower()
    if normalized in ("all", "0-6", "r0-r6"):
        return list(VARIANTS)
    if normalized in ("0-5", "r0-r5"):
        return list(VARIANTS[:6])

    aliases = {f"r{i}": name for i, name in enumerate(VARIANTS)}
    aliases.update({str(i): name for i, name in enumerate(VARIANTS)})
    selected: list[str] = []
    for item in spec.split(","):
        item = item.strip().lower()
        variant = aliases.get(item, item)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {item}")
        if variant not in selected:
            selected.append(variant)
    if not selected:
        raise ValueError("No variants selected")
    # This tool's purpose is old-vs-new comparison.  Keep the production
    # baseline even when the user asks for only one optimized round.
    if "r0_legacy" not in selected:
        selected.append("r0_legacy")
    # Preserve round order even when the CLI list is shuffled; this guarantees
    # that R0 is available before any requested optimized-vs-legacy comparison.
    return [variant for variant in VARIANTS if variant in selected]


def write_csv(path_text: str, records: Sequence[dict[str, object]]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV written to: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeLM-v4 A5 expert-bias TopK R0-R6 accuracy/latency study"
    )
    parser.add_argument(
        "--mode",
        choices=("both", "correctness", "performance"),
        default="both",
    )
    parser.add_argument(
        "--cases",
        default="common",
        help="all|common|decode|prefill or comma/range list, e.g. 1-64,9616,16384",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="all, 0-6, or comma list such as r3,r5,r6 (R0 is always added)",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--scope",
        choices=("kernel", "wrapper"),
        default="kernel",
        help=(
            "kernel, or harness allocation+launch device timeline; NPU Event "
            "does not include Python wall time"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--inner-repeat",
        type=int,
        default=0,
        help="0 selects an M-dependent value; positive forces one value",
    )
    parser.add_argument("--output-csv", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 1 or args.rounds < 1 or args.inner_repeat < 0:
        raise ValueError("warmup/rounds must be positive; inner-repeat must be >= 0")

    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError(f"--device must select an NPU, got: {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    model = ModelNew()
    variants = parse_variants(args.variants)

    print("WeLM-v4 expert-bias TopK manual A5 study")
    print(
        f"device={device} ({model.device_name}), "
        f"physical_AIV={model.num_vector_cores}"
    )
    print(
        f"torch={torch.__version__}, "
        f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}, "
        f"triton={getattr(triton, '__version__', 'unknown')}, "
        f"CANN={getattr(torch.version, 'cann', 'unknown')}"
    )
    for variant in variants:
        print(f"  {variant}: {VARIANT_DESCRIPTIONS[variant]}")

    records: list[dict[str, object]] = []
    failures = 0

    if args.mode in ("both", "correctness"):
        correctness_shapes = parse_shapes(args.cases, correctness=True)
        correctness_records, failures = run_correctness(
            model,
            device=device,
            shapes=correctness_shapes,
            variants=variants,
            seed=args.seed,
        )
        records.extend(correctness_records)
        print(
            f"\nCorrectness summary: "
            f"{'PASS' if failures == 0 else 'FAIL'}, failures={failures}"
        )

    if failures:
        print("Performance is skipped because exact correctness failed.")
    elif args.mode in ("both", "performance"):
        performance_shapes = parse_shapes(args.cases, correctness=False)
        records.extend(
            run_performance(
                model,
                device=device,
                shapes=performance_shapes,
                variants=variants,
                seed=args.seed,
                scope=args.scope,
                warmup=args.warmup,
                rounds=args.rounds,
                inner_repeat_override=args.inner_repeat,
            )
        )

    write_csv(args.output_csv, records)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
