#!/usr/bin/env python3
"""A5-only accuracy and latency study for WeLM-v4 expert-bias TopK.

This is deliberately a manual, standalone experiment.  It does not replace or
register any production SGLang operator.  R0 is a frozen 2026-08-11 copy of the
old production kernel from ``sglang.srt.layers.welmv4_op``; R1--R29 are
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
* R6: empirical two-path dispatch from the R0--R5 A5 measurements.  M <= 320
  reuses R3 (the winner at every measured small/medium shape); larger M reuses
  the R5 large partition path.  This round changes only host dispatch.
* R7: replace R3's ten serial scalar tie-recovery reductions with one
  priority-ordered [512,16] FP32 match matrix and a non-last-axis vector
  cumsum.  The hardware FP32 sort remains the ranking primitive, while all ten
  exact MMQ tie occurrences are recovered in parallel.  R6's M=320 dispatch
  and bias-placement choices are otherwise unchanged.
* R8: dispatch R7 only while every program processes exactly one row
  (M <= physical AIV); above that boundary, reuse R6 unchanged.  This isolates
  the measured one-row vectorization win without carrying R7's rank-2 UB
  matrix across multiple row-loop iterations.
* R9: for M > physical AIV, recover two sorted positions at a time with a
  [512,2] priority-match matrix.  Five vector scans replace ten serial tie
  reductions while avoiding R7's [512,16] multi-row UB pressure.  The R8
  one-row path remains unchanged.
* R10: return to R8's rank-1 path and replace its full candidate-mask state
  with a scalar threshold inside each equal-value group.  Sorted equal values
  are contiguous, so the next exact MMQ tie only needs a larger tie-rank than
  the previous selection when the routing value is unchanged.  This removes
  the per-round 512-lane candidate-mask update without any rank-2 tensor.
* R11: encode the descending FP32 routing order and ascending MMQ tie-rank in
  one signed INT64 key, then use one native A5 rank-1 vector sort.  This removes
  R10's remaining ten 512-wide tie reductions; ten scalar extracts recover the
  already ordered expert IDs.  Routing signed zero is canonicalized so +0/-0
  remain an exact legacy tie.  A5 testing rejected this path because its I64
  sort did not preserve one-ULP FP32 key order; the harness retains R11 as an
  explicit R10 fallback so later full runs stay exact.
* R12: keep R10's exact group-rank threshold but process two adjacent rank-1
  rows per loop iteration once each AIV owns at least two rows.  This halves
  dynamic row-loop control and exposes two row bodies for scheduling while
  respecting the A5 ``vsort`` rank-1 limit and avoiding R7/R9's [512,K] matrix.
* R13: return to R10's row schedule and represent the exact 0..511 tie-rank as
  FP32 during each min reduction.  This tests A5's native FP32 Vector reduction
  path while retaining exact integer IDs after a scalar cast.
* R14: gather routing scores once into MMQ-priority order, then use A5's
  low-rank FP32 ``argmax(tie_break_left=True)`` on each eligible 0/1 mask.
  The returned lane is directly the exact tie-rank, replacing R10's integer
  min reduction while retaining its scalar equal-group threshold.
* R15: detect whether the sorted top-10 routing values are unique.  The common
  unique-value path recovers all IDs with one [512,16] equality/min pass; any
  duplicate value falls back to exact R10.  Unlike R7/R9, the fast path has no
  cumsum and performs only one rank-2 scan.
* R16: keep R15's exact unique/fallback dispatch, but recover the ten unique
  IDs with one [512,8] tile plus two rank-1 tail reductions.  This reduces
  comparison lanes from 8192 to 5120 without a second rank-2 temporary.  A5
  rejected both drafts with runtime 507035 at M=57, so R16 is retained as an
  explicit R15 fallback.
* R17: retain R15's legal [512,16] unique-value matrix, gather routing scores
  into exact MMQ-priority order, and recover all unique IDs with a parallel
  leftmost FP32 argmax instead of the parallel integer min reduction.  A5
  scalar-lowered this rank-2 argmax and made M=16384 roughly 35 ms, so R17 is
  retained as an explicit R15 fallback.
* R18: replace R15's full 512-element value sort with four 128-element sorts,
  keep 16 values per chunk, and sort the resulting 64 candidates.  Exact R15
  ID recovery and duplicate fallback remain unchanged.  It was about 17%
  slower at M=16384, so R18 is retained as an explicit R15 fallback.
* R19: on R15's unique path, recover the single matching expert per column by
  an exact FP32 sum of ``match * expert_id``.  This replaces integer tie-rank
  min plus inverse-rank arithmetic; duplicate values still use exact R10.  It
  was about 27% slower at M=16384, so R19 is retained as an R15 fallback.
* R20: remove sorting entirely.  Reorder scores once into exact MMQ priority,
  then repeat a rank-1 max-value reduction plus a leftmost equality argmax for
  each of the ten selections.  A candidate mask removes each selected rank.
  It was about 2.14x slower at M=16384, so R20 is retained as an R15 fallback.
* R21: specialize R15's rows-per-program and extra-row partition constants for
  the two common fixed prefill shapes M=9616 and M=16384.  Every other shape
  remains an exact R15 dispatch to avoid a general-shape JIT cache explosion.
  It showed no gain, so R21 is retained as an explicit R15 fallback.
* R22: for M >= 512, launch one program per physical core (28 programs on the
  test A5) instead of one per AIV (56).  This tests whether two simultaneous
  sort-heavy AIV programs per core contend for Vector/UB resources.  It nearly
  halved large-shape throughput, so R22 is retained as an R15 fallback.
* R23: device capability gate for an index-returning public ``al.sort`` form.
  A tiny JIT probe tries ``return_indices=True`` and records the compile/run
  result in the same CSV.  The benchmark provider itself is an exact R15
  fallback; a paired value/index TopK will only be attempted in a later round
  if this gate succeeds and returns exact values and indices.
* R24: AscendC ``WholeReduceMax(ORDER_VALUE_INDEX)`` analogue.  Starting from
  R20's exact MMQ-priority order, each selection uses one public
  ``tl.max(..., return_indices=True)`` paired reduction instead of a value max
  followed by a separate equality argmax.  A remaining-candidate fallback is
  used only when the maximum reaches ``-inf`` so all-NaN and fewer-than-K
  finite rows preserve the exact no-duplicate legacy contract.  The extra
  reduction is branchlessly paid on every round and regressed M=16384 beyond
  R0, so later full runs retain R24 as an explicit R15 fallback.
* R25: replace R24's sentinel plus unconditional fallback reduction with one
  public tuple ``tl.reduce`` over ``(routing_score, MMQ_priority_rank)``.
  Candidate validity is encoded exactly as rank 0..511 versus invalid rank
  513, fitting A5's two-source reduction limit.  Its associative lexicographic
  combine selects valid lanes first, then the larger FP32 score, then the
  smaller exact FP32 rank.  Consequently genuine ``-inf`` values need no
  sentinel special case and each round has one value/index-preserving reduce.
  Current A5 BiShengIR fails to lower the two-result reduction inside the row
  loop, so later full runs retain R25 as an explicit R15 fallback.
* R26: on R15's common unique-top10 path, reduce the matching original expert
  ID directly.  A unique value has exactly one matching expert, so the MMQ
  rank reduction and inverse-rank arithmetic are redundant.  Duplicate values
  still execute R15's exact R10 fallback unchanged.  A5 made this rank-2 FP32
  ID reduction about 1.9x slower for large M, so later full runs retain R26 as
  an explicit R15 fallback.
* R27: retain R15's exact integer MMQ-rank reduction and inverse mapping, but
  transpose its unique-path match tile from [512,16] to [16,512].  The same
  values are reduced along the contiguous last axis instead of axis 0; sort,
  duplicate detection, fallback, grid and bias schedule are unchanged.
* R28: keep R27's contiguous last-axis reduction but split its padded
  [16,512] unique tile into [8,512] plus [2,512].  This reduces active match
  and min-reduction lanes from 8192 to 5120 while preserving the same ten
  values, exact integer MMQ ranks, duplicate fallback and output order.  The
  second rank-2 reduction made large M about 16% slower, so later full runs
  retain R28 as an explicit R27 fallback.
* R29: keep one [8,512] contiguous last-axis reduction for positions 0..7,
  then recover positions 8 and 9 with two rank-1 exact MMQ-rank reductions.
  This retains only one rank-2 reduction and halves its active lanes versus
  R27; ranking, duplicate fallback, weights and output order remain unchanged.

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
import inspect
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
R6_DISPATCH_CUTOFF_M = 320

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
    "r7_vector_tie_recovery",
    "r8_safe_vector_dispatch",
    "r9_pair_tie_recovery",
    "r10_group_rank_threshold",
    "r11_packed_exact_sort",
    "r12_two_row_tile",
    "r13_fp32_rank_reduce",
    "r14_priority_argmax",
    "r15_unique_top10_fastpath",
    "r16_split_unique_recovery",
    "r17_parallel_priority_argmax",
    "r18_hierarchical_partial_sort",
    "r19_unique_id_sum",
    "r20_iterative_rank1_maxarg",
    "r21_prefill_partition_constexpr",
    "r22_physical_core_grid",
    "r23_sort_index_capability",
    "r24_paired_max_index",
    "r25_lexicographic_tuple_reduce",
    "r26_unique_direct_expert_id",
    "r27_last_axis_rank_reduce",
    "r28_split_last_axis_rank_reduce",
    "r29_eight_plus_rank1_tail",
)

VARIANT_DESCRIPTIONS = {
    "r0_legacy": "production kernel",
    "r1_constexpr": "N512/K10 contiguous specialization",
    "r2_core_partition": "physical-AIV contiguous row partition",
    "r3_sort_tie": "FP32 sort + exact tie recovery",
    "r4_bias_hoist": "bias load hoisted outside row loop",
    "r5_dispatch": "small direct / large partition dispatch",
    "r6_empirical_dispatch": "R3 through M=320 / R5-large above M=320",
    "r7_vector_tie_recovery": "FP32 2D vector tie recovery on R6 dispatch",
    "r8_safe_vector_dispatch": "R7 through physical AIV / R6 above it",
    "r9_pair_tie_recovery": "R8 small path / five [512,2] tie scans above it",
    "r10_group_rank_threshold": (
        "R8 small path / scalar equal-group rank threshold above it"
    ),
    "r11_packed_exact_sort": "rejected I64 sort experiment; exact R10 fallback",
    "r12_two_row_tile": "R10 with two adjacent rank-1 rows per loop iteration",
    "r13_fp32_rank_reduce": "R10 with exact FP32 tie-rank min reductions",
    "r14_priority_argmax": "R10 with priority gather + FP32 leftmost argmax",
    "r15_unique_top10_fastpath": "one-pass unique top-10 recovery / exact R10 fallback",
    "r16_split_unique_recovery": "rejected [512,8] experiment; exact R15 fallback",
    "r17_parallel_priority_argmax": "rejected rank-2 argmax; exact R15 fallback",
    "r18_hierarchical_partial_sort": "rejected hierarchical sort; exact R15 fallback",
    "r19_unique_id_sum": "rejected FP32 ID sum; exact R15 fallback",
    "r20_iterative_rank1_maxarg": "rejected iterative maxarg; exact R15 fallback",
    "r21_prefill_partition_constexpr": "rejected prefill constexpr; exact R15 fallback",
    "r22_physical_core_grid": "rejected 28-program grid; exact R15 fallback",
    "r23_sort_index_capability": "device al.sort values+indices probe; exact R15 fallback",
    "r24_paired_max_index": "rejected paired max-index; exact R15 fallback",
    "r25_lexicographic_tuple_reduce": "A5 lowering rejected; exact R15 fallback",
    "r26_unique_direct_expert_id": "rejected direct-ID reduce; exact R15 fallback",
    "r27_last_axis_rank_reduce": (
        "R15 [16,512] unique match tile with contiguous last-axis rank min"
    ),
    "r28_split_last_axis_rank_reduce": (
        "rejected two rank-2 tiles; exact R27 fallback"
    ),
    "r29_eight_plus_rank1_tail": (
        "R27 [8,512] unique-rank tile plus two rank-1 tail reductions"
    ),
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
    384,
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


@triton.jit
def _group_rank_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """Exact ties via a scalar rank threshold within each equal-value group."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    invalid_rank = _JIT_NUM_EXPERTS + 1
    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    previous_value = float("inf")
    previous_rank = -1.0

    for k in tl.static_range(0, _JIT_TOPK):
        kth_routing_score = al.get_element(sorted_routing, indice=[k])
        same_group = kth_routing_score == previous_value
        rank_after_previous = tie_rank.to(tl.float32) > previous_rank
        eligible = (routing_scores == kth_routing_score) & (
            ~same_group | rank_after_previous
        )
        selected_rank = tl.min(
            tl.where(eligible, tie_rank, invalid_rank),
            axis=0,
        )

        lane = selected_rank >> 4
        local = selected_rank & 15
        selected_idx = ((local >> 2) << 7) + (lane << 2) + (local & 3)
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        previous_value = kth_routing_score
        previous_rank = selected_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _packed_exact_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """One native I64 sort for FP32 descending plus MMQ-priority ties."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    # Legacy FP32 equality treats +0 and -0 as one tie group.  Canonicalize
    # before bit-key construction so their sign bits cannot affect ordering.
    routing_scores = tl.where(routing_scores == 0.0, 0.0, routing_scores)

    # Anti-monotone IEEE FP32 key: signed ascending order equals FP32
    # descending order.  Put it in the high 32 bits and the ascending MMQ
    # tie-rank in the low bits.  I64 is necessary here to preserve all 32
    # routing bits plus the 9-bit exact priority; A5 vsort natively supports it.
    min_i32 = -2147483648
    routing_bits = routing_scores.to(tl.int32, bitcast=True)
    sign = routing_bits >> 31
    value_key = tl.where(
        sign == 0,
        routing_bits ^ -1,
        routing_bits ^ min_i32,
    )
    key_u32 = value_key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (key_u32 << 32) | tie_rank.to(tl.int64)
    sorted_packed = al.sort(packed, dim=-1, descending=False)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    for k in tl.static_range(0, _JIT_TOPK):
        kth_packed = al.get_element(sorted_packed, indice=[k])
        selected_rank = (kth_packed & 511).to(tl.int32)
        lane = selected_rank >> 4
        local = selected_rank & 15
        selected_idx = ((local >> 2) << 7) + (lane << 2) + (local & 3)
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _fp32_rank_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R10 selection with exact FP32 tie-rank reduction on A5 Vector."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    tie_rank_fp32 = tie_rank.to(tl.float32)
    invalid_rank_fp32 = 513.0
    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    previous_value = float("inf")
    previous_rank = -1.0

    for k in tl.static_range(0, _JIT_TOPK):
        kth_routing_score = al.get_element(sorted_routing, indice=[k])
        same_group = kth_routing_score == previous_value
        eligible = (routing_scores == kth_routing_score) & (
            ~same_group | (tie_rank_fp32 > previous_rank)
        )
        selected_rank_fp32 = tl.min(
            tl.where(eligible, tie_rank_fp32, invalid_rank_fp32),
            axis=0,
        )
        selected_rank = selected_rank_fp32.to(tl.int32)

        lane = selected_rank >> 4
        local = selected_rank & 15
        selected_idx = ((local >> 2) << 7) + (lane << 2) + (local & 3)
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        previous_value = kth_routing_score
        previous_rank = selected_rank_fp32

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _priority_argmax_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """R10 selection via MMQ-priority gather and leftmost FP32 argmax."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    priority_rank = offs
    lane = priority_rank >> 4
    local = priority_rank & 15
    priority_ids = ((local >> 2) << 7) + (lane << 2) + (local & 3)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    previous_value = float("inf")
    previous_rank = -1.0

    for k in tl.static_range(0, _JIT_TOPK):
        kth_routing_score = al.get_element(sorted_routing, indice=[k])
        same_group = kth_routing_score == previous_value
        eligible = (routing_by_priority == kth_routing_score) & (
            ~same_group | (priority_rank.to(tl.float32) > previous_rank)
        )
        selected_rank = tl.argmax(
            eligible.to(tl.float32),
            axis=0,
            tie_break_left=True,
        ).to(tl.int32)
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


@triton.jit
def _unique_top10_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """One-pass exact ID recovery when top-10 values are unique, else R10."""
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
        matches = routing_scores[:, None] == top_values[None, :]
        fast_selected_rank = tl.min(
            tl.where(
                matches,
                tie_rank[:, None],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=0,
        )
        fast_lane = fast_selected_rank >> 4
        fast_local = fast_selected_rank & 15
        selected_ids = (
            ((fast_local >> 2) << 7)
            + (fast_lane << 2)
            + (fast_local & 3)
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
            fallback_selected_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank),
                axis=0,
            )
            fallback_lane = fallback_selected_rank >> 4
            fallback_local = fallback_selected_rank & 15
            fallback_selected_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(
                out_lanes == k, fallback_selected_idx, selected_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_selected_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _unique_direct_id_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R15 with direct original-expert-ID recovery on the unique path."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, out_lanes, 0)
    next_lanes = tl.minimum(out_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        out_lanes.to(tl.float32) < 10.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    if ~has_duplicate:
        # Each selected routing value occurs once, so the matching original
        # expert ID is already the complete result.  FP32 is exact for 0..511
        # and avoids R15's integer min-rank plus inverse-permutation math.
        output_columns = out_lanes.to(tl.float32) < _JIT_TOPK
        matches = (routing_scores[:, None] == top_values[None, :]) & (
            output_columns[None, :]
        )
        selected_ids = tl.min(
            tl.where(
                matches,
                offs[:, None].to(tl.float32),
                _JIT_NUM_EXPERTS + 1.0,
            ),
            axis=0,
        ).to(tl.int32)
        selected_ids = tl.where(output_columns, selected_ids, 0)
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
            fallback_selected_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank),
                axis=0,
            )
            fallback_lane = fallback_selected_rank >> 4
            fallback_local = fallback_selected_rank & 15
            fallback_selected_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(
                out_lanes == k, fallback_selected_idx, selected_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_selected_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _last_axis_rank_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R15 with the unique match reduction on contiguous axis 1."""
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
        fast_selected_rank = tl.min(
            tl.where(
                matches,
                tie_rank[None, :],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=1,
        )
        fast_lane = fast_selected_rank >> 4
        fast_local = fast_selected_rank & 15
        selected_ids = (
            ((fast_local >> 2) << 7)
            + (fast_lane << 2)
            + (fast_local & 3)
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
            fallback_selected_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank),
                axis=0,
            )
            fallback_lane = fallback_selected_rank >> 4
            fallback_local = fallback_selected_rank & 15
            fallback_selected_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(
                out_lanes == k, fallback_selected_idx, selected_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_selected_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _split_last_axis_rank_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R27 exact semantics with 8+2 rather than 16 output columns."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    detection_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, detection_lanes, 0)
    next_lanes = tl.minimum(detection_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        detection_lanes.to(tl.float32) < 9.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    if ~has_duplicate:
        first_lanes = tl.arange(0, 8)
        first_values = tl.gather(top_values, first_lanes, 0)
        first_matches = first_values[:, None] == routing_scores[None, :]
        first_ranks = tl.min(
            tl.where(
                first_matches,
                tie_rank[None, :],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=1,
        )
        first_lane = first_ranks >> 4
        first_local = first_ranks & 15
        first_ids = (
            ((first_local >> 2) << 7)
            + (first_lane << 2)
            + (first_local & 3)
        ).to(tl.int32)
        first_weights = tl.gather(scores, first_ids, 0)
        first_offsets = row * _JIT_TOPK + first_lanes
        tl.store(weights_ptr + first_offsets, first_weights)
        tl.store(ids_ptr + first_offsets, first_ids)

        tail_lanes = tl.arange(0, 2)
        tail_values = tl.gather(top_values, tail_lanes + 8, 0)
        tail_matches = tail_values[:, None] == routing_scores[None, :]
        tail_ranks = tl.min(
            tl.where(
                tail_matches,
                tie_rank[None, :],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=1,
        )
        tail_lane = tail_ranks >> 4
        tail_local = tail_ranks & 15
        tail_ids = (
            ((tail_local >> 2) << 7)
            + (tail_lane << 2)
            + (tail_local & 3)
        ).to(tl.int32)
        tail_weights = tl.gather(scores, tail_ids, 0)
        tail_offsets = row * _JIT_TOPK + tail_lanes + 8
        tl.store(weights_ptr + tail_offsets, tail_weights)
        tl.store(ids_ptr + tail_offsets, tail_ids)
    else:
        fallback_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
        fallback_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
        invalid_rank = _JIT_NUM_EXPERTS + 1
        previous_value = float("inf")
        previous_rank = -1.0
        for k in tl.static_range(0, _JIT_TOPK):
            kth_routing_score = al.get_element(sorted_routing, indice=[k])
            same_group = kth_routing_score == previous_value
            eligible = (routing_scores == kth_routing_score) & (
                ~same_group | (tie_rank.to(tl.float32) > previous_rank)
            )
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            fallback_ids = tl.where(
                fallback_lanes == k, fallback_idx, fallback_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

        fallback_weights = tl.gather(scores, fallback_ids, 0)
        fallback_mask = fallback_lanes < _JIT_TOPK
        fallback_offsets = row * _JIT_TOPK + fallback_lanes
        tl.store(
            weights_ptr + fallback_offsets,
            fallback_weights,
            mask=fallback_mask,
        )
        tl.store(ids_ptr + fallback_offsets, fallback_ids, mask=fallback_mask)


@triton.jit
def _eight_plus_rank1_tail_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R27 exact semantics with one [8,512] tile and two rank-1 tails."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    detection_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, detection_lanes, 0)
    next_lanes = tl.minimum(detection_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        detection_lanes.to(tl.float32) < 9.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    if ~has_duplicate:
        first_lanes = tl.arange(0, 8)
        first_values = tl.gather(top_values, first_lanes, 0)
        first_matches = first_values[:, None] == routing_scores[None, :]
        first_ranks = tl.min(
            tl.where(
                first_matches,
                tie_rank[None, :],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=1,
        )
        first_lane = first_ranks >> 4
        first_local = first_ranks & 15
        first_ids = (
            ((first_local >> 2) << 7)
            + (first_lane << 2)
            + (first_local & 3)
        ).to(tl.int32)
        first_weights = tl.gather(scores, first_ids, 0)
        first_offsets = row * _JIT_TOPK + first_lanes
        tl.store(weights_ptr + first_offsets, first_weights)
        tl.store(ids_ptr + first_offsets, first_ids)

        tail_lanes = tl.arange(0, 2)
        tail_ids = tl.zeros((2,), dtype=tl.int32)
        for tail_k in tl.static_range(8, _JIT_TOPK):
            tail_value = al.get_element(sorted_routing, indice=[tail_k])
            tail_rank = tl.min(
                tl.where(
                    routing_scores == tail_value,
                    tie_rank,
                    _JIT_NUM_EXPERTS + 1,
                ),
                axis=0,
            )
            tail_lane = tail_rank >> 4
            tail_local = tail_rank & 15
            tail_idx = (
                ((tail_local >> 2) << 7)
                + (tail_lane << 2)
                + (tail_local & 3)
            )
            tail_ids = tl.where(tail_lanes == tail_k - 8, tail_idx, tail_ids)
        tail_weights = tl.gather(scores, tail_ids, 0)
        tail_offsets = row * _JIT_TOPK + tail_lanes + 8
        tl.store(weights_ptr + tail_offsets, tail_weights)
        tl.store(ids_ptr + tail_offsets, tail_ids)
    else:
        fallback_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
        fallback_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
        invalid_rank = _JIT_NUM_EXPERTS + 1
        previous_value = float("inf")
        previous_rank = -1.0
        for k in tl.static_range(0, _JIT_TOPK):
            kth_routing_score = al.get_element(sorted_routing, indice=[k])
            same_group = kth_routing_score == previous_value
            eligible = (routing_scores == kth_routing_score) & (
                ~same_group | (tie_rank.to(tl.float32) > previous_rank)
            )
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            fallback_ids = tl.where(
                fallback_lanes == k, fallback_idx, fallback_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

        fallback_weights = tl.gather(scores, fallback_ids, 0)
        fallback_mask = fallback_lanes < _JIT_TOPK
        fallback_offsets = row * _JIT_TOPK + fallback_lanes
        tl.store(
            weights_ptr + fallback_offsets,
            fallback_weights,
            mask=fallback_mask,
        )
        tl.store(ids_ptr + fallback_offsets, fallback_ids, mask=fallback_mask)


@triton.jit
def _split_unique_top10_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R15 semantics with [512,8] + two rank-1 unique-value reductions."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    detection_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, detection_lanes, 0)
    next_lanes = tl.minimum(detection_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        detection_lanes.to(tl.float32) < 9.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    if ~has_duplicate:
        first_lanes = tl.arange(0, 8)
        first_values = tl.gather(top_values, first_lanes, 0)
        first_matches = routing_scores[:, None] == first_values[None, :]
        first_ranks = tl.min(
            tl.where(first_matches, tie_rank[:, None], _JIT_NUM_EXPERTS + 1),
            axis=0,
        )
        first_lane = first_ranks >> 4
        first_local = first_ranks & 15
        first_ids = (
            ((first_local >> 2) << 7)
            + (first_lane << 2)
            + (first_local & 3)
        ).to(tl.int32)
        first_weights = tl.gather(scores, first_ids, 0)
        first_offsets = row * _JIT_TOPK + first_lanes
        tl.store(weights_ptr + first_offsets, first_weights)
        tl.store(ids_ptr + first_offsets, first_ids)

        # A second rank-2 [512,2] temporary triggered device error 507035 on
        # A5 at the first multi-row shape (M=57).  Keep the same 5120 total
        # comparison lanes, but recover positions 8 and 9 as rank-1 vectors.
        tail_lanes = tl.arange(0, 2)
        tail_ids = tl.zeros((2,), dtype=tl.int32)
        for tail_k in tl.static_range(8, _JIT_TOPK):
            tail_value = al.get_element(sorted_routing, indice=[tail_k])
            tail_rank = tl.min(
                tl.where(
                    routing_scores == tail_value,
                    tie_rank,
                    _JIT_NUM_EXPERTS + 1,
                ),
                axis=0,
            )
            tail_lane = tail_rank >> 4
            tail_local = tail_rank & 15
            tail_idx = (
                ((tail_local >> 2) << 7)
                + (tail_lane << 2)
                + (tail_local & 3)
            )
            tail_ids = tl.where(tail_lanes == tail_k - 8, tail_idx, tail_ids)
        tail_weights = tl.gather(scores, tail_ids, 0)
        tail_offsets = row * _JIT_TOPK + tail_lanes + 8
        tl.store(weights_ptr + tail_offsets, tail_weights)
        tl.store(ids_ptr + tail_offsets, tail_ids)
    else:
        fallback_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
        fallback_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
        invalid_rank = _JIT_NUM_EXPERTS + 1
        previous_value = float("inf")
        previous_rank = -1.0
        for k in tl.static_range(0, _JIT_TOPK):
            kth_routing_score = al.get_element(sorted_routing, indice=[k])
            same_group = kth_routing_score == previous_value
            eligible = (routing_scores == kth_routing_score) & (
                ~same_group | (tie_rank.to(tl.float32) > previous_rank)
            )
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            fallback_ids = tl.where(
                fallback_lanes == k, fallback_idx, fallback_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

        fallback_weights = tl.gather(scores, fallback_ids, 0)
        fallback_mask = fallback_lanes < _JIT_TOPK
        fallback_offsets = row * _JIT_TOPK + fallback_lanes
        tl.store(
            weights_ptr + fallback_offsets,
            fallback_weights,
            mask=fallback_mask,
        )
        tl.store(ids_ptr + fallback_offsets, fallback_ids, mask=fallback_mask)


@triton.jit
def _parallel_priority_argmax_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R15 unique dispatch with parallel leftmost MMQ-priority argmax."""
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
        priority_lane = offs >> 4
        priority_local = offs & 15
        priority_ids = (
            ((priority_local >> 2) << 7)
            + (priority_lane << 2)
            + (priority_local & 3)
        ).to(tl.int32)
        routing_by_priority = tl.gather(routing_scores, priority_ids, 0)
        matches = routing_by_priority[:, None] == top_values[None, :]
        selected_rank = tl.argmax(
            matches.to(tl.float32),
            axis=0,
            tie_break_left=True,
        ).to(tl.int32)
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
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(
                out_lanes == k, fallback_idx, selected_ids
            )
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _hierarchical_partial_sort_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """Exact R15 recovery after four sort128 plus one sort64 value merge."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias

    chunk_lanes = tl.arange(0, 128)
    chunk0 = tl.gather(routing_scores, chunk_lanes, 0)
    chunk1 = tl.gather(routing_scores, chunk_lanes + 128, 0)
    chunk2 = tl.gather(routing_scores, chunk_lanes + 256, 0)
    chunk3 = tl.gather(routing_scores, chunk_lanes + 384, 0)
    sorted0 = al.sort(chunk0, dim=-1, descending=True)
    sorted1 = al.sort(chunk1, dim=-1, descending=True)
    sorted2 = al.sort(chunk2, dim=-1, descending=True)
    sorted3 = al.sort(chunk3, dim=-1, descending=True)

    candidate_lanes = tl.arange(0, 64)
    candidate_group = candidate_lanes >> 4
    candidate_pos = candidate_lanes & 15
    values0 = tl.gather(sorted0, candidate_pos, 0)
    values1 = tl.gather(sorted1, candidate_pos, 0)
    values2 = tl.gather(sorted2, candidate_pos, 0)
    values3 = tl.gather(sorted3, candidate_pos, 0)
    candidates = tl.where(
        candidate_group == 0,
        values0,
        tl.where(candidate_group == 1, values1, tl.where(candidate_group == 2, values2, values3)),
    )
    sorted_candidates = al.sort(candidates, dim=-1, descending=True)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_candidates, out_lanes, 0)
    next_lanes = tl.minimum(out_lanes + 1, _JIT_OUTPUT_WIDTH - 1)
    next_values = tl.gather(top_values, next_lanes, 0)
    adjacent_duplicate = (top_values == next_values) & (
        out_lanes.to(tl.float32) < 9.0
    )
    has_duplicate = tl.sum(adjacent_duplicate.to(tl.float32), axis=0) > 0.0

    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    if ~has_duplicate:
        matches = routing_scores[:, None] == top_values[None, :]
        fast_rank = tl.min(
            tl.where(matches, tie_rank[:, None], _JIT_NUM_EXPERTS + 1),
            axis=0,
        )
        fast_lane = fast_rank >> 4
        fast_local = fast_rank & 15
        selected_ids = (
            ((fast_local >> 2) << 7)
            + (fast_lane << 2)
            + (fast_local & 3)
        ).to(tl.int32)
    else:
        invalid_rank = _JIT_NUM_EXPERTS + 1
        previous_value = float("inf")
        previous_rank = -1.0
        for k in tl.static_range(0, _JIT_TOPK):
            kth_routing_score = al.get_element(sorted_candidates, indice=[k])
            same_group = kth_routing_score == previous_value
            eligible = (routing_scores == kth_routing_score) & (
                ~same_group | (tie_rank.to(tl.float32) > previous_rank)
            )
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(out_lanes == k, fallback_idx, selected_ids)
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _unique_id_sum_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    tie_rank,
    bias,
):
    """R15 unique path with exact one-hot FP32 expert-ID sum."""
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
        matches = routing_scores[:, None] == top_values[None, :]
        # Every top-10 routing value is unique in this branch, hence each
        # column contains exactly one nonzero ID.  Expert IDs 0..511 are exact
        # in FP32, so this sum introduces no numerical approximation.
        selected_ids = tl.sum(
            matches.to(tl.float32) * offs[:, None].to(tl.float32),
            axis=0,
        ).to(tl.int32)
        # Only lanes 0..9 are outputs.  Cleaned NaNs can make local lanes
        # 10..15 equal -inf, so their one-hot assumption does not hold and the
        # sum may exceed 511.  Sanitize those non-output lanes before the
        # unmasked local gather below; the actual top-10 IDs are unchanged.
        selected_ids = tl.where(out_lanes < _JIT_TOPK, selected_ids, 0)
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
            fallback_rank = tl.min(
                tl.where(eligible, tie_rank, invalid_rank), axis=0
            )
            fallback_lane = fallback_rank >> 4
            fallback_local = fallback_rank & 15
            fallback_idx = (
                ((fallback_local >> 2) << 7)
                + (fallback_lane << 2)
                + (fallback_local & 3)
            )
            selected_ids = tl.where(out_lanes == k, fallback_idx, selected_ids)
            previous_value = kth_routing_score
            previous_rank = fallback_rank.to(tl.float32)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _iterative_rank1_maxarg_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """Exact MMQ top-10 via rank-1 max then leftmost argmax per round."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias

    priority_rank = offs
    priority_lane = priority_rank >> 4
    priority_local = priority_rank & 15
    priority_ids = (
        ((priority_local >> 2) << 7)
        + (priority_lane << 2)
        + (priority_local & 3)
    ).to(tl.int32)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)
    scores_by_priority = tl.gather(scores, priority_ids, 0)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    selected_weights = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.float32)
    candidate = tl.full((_JIT_SORT_WIDTH,), 1, dtype=tl.int1)

    for k in tl.static_range(0, _JIT_TOPK):
        active_scores = tl.where(candidate, routing_by_priority, -float("inf"))
        selected_value = tl.max(active_scores, axis=0)
        first_match = candidate & (routing_by_priority == selected_value)
        selected_rank = tl.argmax(
            first_match.to(tl.float32), axis=0, tie_break_left=True
        ).to(tl.int32)
        selected_rank_vec = selected_rank + tl.arange(0, 1)
        # Ascend tl.gather does not accept an int32 source.  selected_rank is
        # already the MMQ priority rank, so invert the exact rank bijection
        # directly instead of gathering priority_ids.
        selected_lane = selected_rank >> 4
        selected_local = selected_rank & 15
        selected_idx = (
            ((selected_local >> 2) << 7)
            + (selected_lane << 2)
            + (selected_local & 3)
        )
        selected_weight = tl.gather(
            scores_by_priority, selected_rank_vec, 0
        )
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        selected_weights = tl.where(
            out_lanes == k, selected_weight, selected_weights
        )
        candidate = candidate & (priority_rank != selected_rank)

    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _paired_max_index_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """Exact MMQ top-10 via ten fused value/index FP32 reductions."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias

    # Leftmost paired-reduction tie breaking is the MMQ tie contract after
    # this exact permutation.  All subsequent state remains in priority order.
    priority_rank = offs
    priority_lane = priority_rank >> 4
    priority_local = priority_rank & 15
    priority_ids = (
        ((priority_local >> 2) << 7)
        + (priority_lane << 2)
        + (priority_local & 3)
    ).to(tl.int32)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)
    scores_by_priority = tl.gather(scores, priority_ids, 0)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    selected_weights = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.float32)
    candidate = tl.full((_JIT_SORT_WIDTH,), 1, dtype=tl.int1)

    for k in tl.static_range(0, _JIT_TOPK):
        active_scores = tl.where(
            candidate, routing_by_priority, -float("inf")
        )
        selected_value, selected_rank = tl.max(
            active_scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
            propagate_nan=False,
        )
        selected_rank = selected_rank.to(tl.int32)

        # With fewer than K non--inf values, a -inf sentinel cannot distinguish
        # selected lanes from genuine cleaned -inf inputs.  Recover the exact
        # leftmost remaining MMQ-priority lane for this semantic tail.  This is
        # deliberately separate from the common paired-reduction result so no
        # finite routing value is perturbed by an epsilon or packed key.
        remaining_rank = tl.argmax(
            candidate.to(tl.float32), axis=0, tie_break_left=True
        ).to(tl.int32)
        selected_rank = tl.where(
            selected_value == -float("inf"),
            remaining_rank,
            selected_rank,
        )
        selected_rank_vec = selected_rank + tl.arange(0, 1)
        selected_lane = selected_rank >> 4
        selected_local = selected_rank & 15
        selected_idx = (
            ((selected_local >> 2) << 7)
            + (selected_lane << 2)
            + (selected_local & 3)
        )
        selected_weight = tl.gather(
            scores_by_priority, selected_rank_vec, 0
        )
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        selected_weights = tl.where(
            out_lanes == k, selected_weight, selected_weights
        )

        candidate = candidate & (priority_rank != selected_rank)

    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _lexicographic_candidate_combine(
    a_score,
    a_rank,
    b_score,
    b_rank,
):
    """Associative max over valid, score, then inverse MMQ rank."""
    a_valid = a_rank < _JIT_NUM_EXPERTS
    b_valid = b_rank < _JIT_NUM_EXPERTS
    take_b = (b_valid & ~a_valid) | (
        (b_valid == a_valid)
        & (
            (b_score > a_score)
            | ((b_score == a_score) & (b_rank < a_rank))
        )
    )
    return (
        tl.where(take_b, b_score, a_score),
        tl.where(take_b, b_rank, a_rank),
    )


@triton.jit
def _lexicographic_tuple_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """Exact MMQ top-10 via ten public tuple reductions.

    Candidate validity is encoded by an out-of-range FP32 rank, so selected
    lanes never need a score sentinel.  This stays within the A5 backend's
    maximum of two reduction sources.  Genuine cleaned -inf scores retain
    their distinct MMQ ranks and the all-NaN/fewer-than-K-finite cases remain
    exact without R24's second argmax reduction.
    """
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias

    priority_rank = offs
    priority_lane = priority_rank >> 4
    priority_local = priority_rank & 15
    priority_ids = (
        ((priority_local >> 2) << 7)
        + (priority_lane << 2)
        + (priority_local & 3)
    ).to(tl.int32)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)
    scores_by_priority = tl.gather(scores, priority_ids, 0)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)
    selected_weights = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.float32)
    candidate = tl.full((_JIT_SORT_WIDTH,), 1, dtype=tl.int1)
    rank_fp32 = priority_rank.to(tl.float32)

    for k in tl.static_range(0, _JIT_TOPK):
        candidate_rank = tl.where(
            candidate, rank_fp32, _JIT_NUM_EXPERTS + 1.0
        )
        _, selected_rank_fp32 = tl.reduce(
            (
                routing_by_priority,
                candidate_rank,
            ),
            axis=0,
            combine_fn=_lexicographic_candidate_combine,
        )
        selected_rank = selected_rank_fp32.to(tl.int32)
        selected_rank_vec = selected_rank + tl.arange(0, 1)
        selected_lane = selected_rank >> 4
        selected_local = selected_rank & 15
        selected_idx = (
            ((selected_local >> 2) << 7)
            + (selected_lane << 2)
            + (selected_local & 3)
        )
        selected_weight = tl.gather(scores_by_priority, selected_rank_vec, 0)
        selected_ids = tl.where(out_lanes == k, selected_idx, selected_ids)
        selected_weights = tl.where(
            out_lanes == k, selected_weight, selected_weights
        )
        candidate = candidate & (priority_rank != selected_rank)

    output_mask = out_lanes < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _vector_tie_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """Recover all ten MMQ-priority ties with one rank-2 vector scan."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    top_values = tl.gather(sorted_routing, out_lanes, 0)

    # Reorder the row into the legacy MMQ tie-priority sequence.  The inverse
    # mapping is a bijection for N=512, so equal routing values become a simple
    # stable occurrence problem along axis 0.
    priority_rank = offs
    lane = priority_rank >> 4
    local = priority_rank & 15
    priority_ids = ((local >> 2) << 7) + (lane << 2) + (local & 3)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)

    matches = routing_by_priority[:, None] == top_values[None, :]
    match_prefix = tl.cumsum(matches.to(tl.float32), axis=0)

    # For sorted position k, identify whether it is the 1st/2nd/... occurrence
    # of that value.  All comparisons and sums are vector FP32 operations.
    prior_position = (
        out_lanes[:, None].to(tl.float32)
        <= out_lanes[None, :].to(tl.float32)
    )
    same_top_value = top_values[:, None] == top_values[None, :]
    desired_occurrence = tl.sum(
        tl.where(prior_position & same_top_value, 1.0, 0.0), axis=0
    )

    selected_match = matches & (
        match_prefix == desired_occurrence[None, :]
    )
    selected_priority_rank = tl.min(
        tl.where(
            selected_match,
            priority_rank[:, None],
            _JIT_NUM_EXPERTS + 1,
        ),
        axis=0,
    )

    selected_lane = selected_priority_rank >> 4
    selected_local = selected_priority_rank & 15
    selected_ids = (
        ((selected_local >> 2) << 7)
        + (selected_lane << 2)
        + (selected_local & 3)
    ).to(tl.int32)
    selected_weights = tl.gather(scores, selected_ids, 0)

    output_mask = out_lanes.to(tl.float32) < _JIT_TOPK
    output_offsets = row * _JIT_TOPK + out_lanes
    tl.store(weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(ids_ptr + output_offsets, selected_ids, mask=output_mask)


@triton.jit
def _pair_tie_select_store_row(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    row,
    offs,
    bias,
):
    """Recover two sorted positions per vector scan; five scans for K=10."""
    scores = tl.load(scores_ptr + row * _JIT_NUM_EXPERTS + offs)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing_scores = scores + bias
    sorted_routing = al.sort(routing_scores, dim=-1, descending=True)

    priority_rank = offs
    lane = priority_rank >> 4
    local = priority_rank & 15
    priority_ids = ((local >> 2) << 7) + (lane << 2) + (local & 3)
    routing_by_priority = tl.gather(routing_scores, priority_ids, 0)

    out_lanes = tl.arange(0, _JIT_OUTPUT_WIDTH)
    pair_lanes = tl.arange(0, 2)
    selected_ids = tl.zeros((_JIT_OUTPUT_WIDTH,), dtype=tl.int32)

    for pair in tl.static_range(0, _JIT_TOPK // 2):
        sorted_positions = pair * 2 + pair_lanes
        pair_values = tl.gather(sorted_routing, sorted_positions, 0)
        matches = routing_by_priority[:, None] == pair_values[None, :]
        match_prefix = tl.cumsum(matches.to(tl.float32), axis=0)

        # Count identical earlier sorted values.  This is exact for arbitrary
        # duplicate groups, including groups that cross pair boundaries.
        prior_positions = tl.arange(0, _JIT_OUTPUT_WIDTH)
        prior_values = tl.gather(sorted_routing, prior_positions, 0)
        prior_mask = (
            prior_positions[:, None].to(tl.float32)
            <= sorted_positions[None, :].to(tl.float32)
        )
        same_value = prior_values[:, None] == pair_values[None, :]
        desired_occurrence = tl.sum(
            tl.where(prior_mask & same_value, 1.0, 0.0), axis=0
        )

        selected_match = matches & (
            match_prefix == desired_occurrence[None, :]
        )
        selected_priority_rank = tl.min(
            tl.where(
                selected_match,
                priority_rank[:, None],
                _JIT_NUM_EXPERTS + 1,
            ),
            axis=0,
        )
        selected_lane = selected_priority_rank >> 4
        selected_local = selected_priority_rank & 15
        pair_ids = (
            ((selected_local >> 2) << 7)
            + (selected_lane << 2)
            + (selected_local & 3)
        ).to(tl.int32)
        output_match = (
            out_lanes[:, None].to(tl.float32)
            == sorted_positions[None, :].to(tl.float32)
        )
        pair_update = tl.sum(
            tl.where(output_match, pair_ids[None, :], 0), axis=1
        )
        has_pair_update = (
            tl.sum(output_match.to(tl.float32), axis=1) > 0.0
        )
        selected_ids = tl.where(has_pair_update, pair_update, selected_ids)

    selected_weights = tl.gather(scores, selected_ids, 0)
    output_mask = out_lanes.to(tl.float32) < _JIT_TOPK
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


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r7_vector_tie_small_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R6's M<=320 R3 path: bias remains inside the per-row loop.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _vector_tie_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r7_vector_tie_large_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R6's M>320 R5-large path: bias is invariant and hoisted once.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _vector_tie_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r9_pair_tie_medium_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R8/R6 for physical_AIV < M <= 320: load bias per row.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _pair_tie_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r9_pair_tie_large_kernel(
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
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _pair_tie_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r10_group_rank_medium_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R8/R6 for physical_AIV < M <= 320: load bias per row.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _group_rank_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r10_group_rank_large_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R8/R6 above M=320: keep the invariant bias hoisted.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _group_rank_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r11_packed_exact_medium_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R10 for physical_AIV < M <= 320: load bias per row.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _packed_exact_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r11_packed_exact_large_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R10 above M=320: keep the invariant bias hoisted.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _packed_exact_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r12_two_row_medium_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R10 for physical_AIV < M <= 320: bias remains per row.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    pair_end = row_end - ((row_end - row_start) & 1)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)

    for row in tl.range(row_start, pair_end, 2):
        bias0 = tl.load(bias_ptr + offs)
        _group_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias0
        )
        bias1 = tl.load(bias_ptr + offs)
        _group_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row + 1, offs, tie_rank, bias1
        )

    for tail_row in tl.range(pair_end, row_end):
        tail_bias = tl.load(bias_ptr + offs)
        _group_rank_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            tail_row,
            offs,
            tie_rank,
            tail_bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r12_two_row_large_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    rows_per_program,
    extra_rows,
):
    # Match R10 above M=320: keep one invariant bias vector per AIV.
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, extra_rows)
    row_start = pid * rows_per_program + extra_before
    row_end = row_start + rows_per_program + tl.where(pid < extra_rows, 1, 0)
    pair_end = row_end - ((row_end - row_start) & 1)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, pair_end, 2):
        _group_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )
        _group_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row + 1, offs, tie_rank, bias
        )

    for tail_row in tl.range(pair_end, row_end):
        _group_rank_select_store_row(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            tail_row,
            offs,
            tie_rank,
            bias,
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r13_fp32_rank_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _fp32_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r13_fp32_rank_large_kernel(
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
        _fp32_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r14_priority_argmax_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _priority_argmax_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r14_priority_argmax_large_kernel(
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
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _priority_argmax_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r15_unique_top10_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _unique_top10_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r15_unique_top10_large_kernel(
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
        _unique_top10_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r16_split_unique_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _split_unique_top10_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r16_split_unique_large_kernel(
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
        _split_unique_top10_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r17_parallel_argmax_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _parallel_priority_argmax_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r17_parallel_argmax_large_kernel(
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
        _parallel_priority_argmax_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r18_hierarchical_sort_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _hierarchical_partial_sort_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r18_hierarchical_sort_large_kernel(
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
        _hierarchical_partial_sort_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r19_unique_id_sum_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _unique_id_sum_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r19_unique_id_sum_large_kernel(
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
        _unique_id_sum_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r20_iterative_maxarg_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _iterative_rank1_maxarg_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r20_iterative_maxarg_large_kernel(
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
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _iterative_rank1_maxarg_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r24_paired_max_index_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _paired_max_index_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r24_paired_max_index_large_kernel(
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
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _paired_max_index_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r25_lexicographic_tuple_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _lexicographic_tuple_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r25_lexicographic_tuple_large_kernel(
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
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _lexicographic_tuple_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r26_unique_direct_id_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _unique_direct_id_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r26_unique_direct_id_large_kernel(
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
        _unique_direct_id_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r27_last_axis_rank_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _last_axis_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r27_last_axis_rank_large_kernel(
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
        _last_axis_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r28_split_last_axis_rank_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _split_last_axis_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r28_split_last_axis_rank_large_kernel(
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
        _split_last_axis_rank_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r29_eight_plus_rank1_medium_kernel(
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

    for row in tl.range(row_start, row_end):
        bias = tl.load(bias_ptr + offs)
        _eight_plus_rank1_tail_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit(do_not_specialize=["rows_per_program", "extra_rows"])
def _r29_eight_plus_rank1_large_kernel(
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
        _eight_plus_rank1_tail_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit
def _r21_prefill_partition_constexpr_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    ROWS_PER_PROGRAM: tl.constexpr,
    EXTRA_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    extra_before = tl.minimum(pid, EXTRA_ROWS)
    row_start = pid * ROWS_PER_PROGRAM + extra_before
    row_end = row_start + ROWS_PER_PROGRAM + tl.where(pid < EXTRA_ROWS, 1, 0)
    offs = tl.arange(0, _JIT_SORT_WIDTH)
    tie_rank = ((offs & 127) >> 2) * 16 + ((offs >> 7) << 2) + (offs & 3)
    bias = tl.load(bias_ptr + offs)

    for row in tl.range(row_start, row_end):
        _unique_top10_select_store_row(
            scores_ptr, weights_ptr, ids_ptr, row, offs, tie_rank, bias
        )


@triton.jit
def _r23_sort_index_probe_kernel(input_ptr, values_ptr, indices_ptr):
    """Capability-only kernel; failure is caught and reported by the harness."""
    offs = tl.arange(0, 32)
    values = tl.load(input_ptr + offs)
    sorted_values, sorted_indices = al.sort(
        values, dim=-1, descending=True, return_indices=True
    )
    tl.store(values_ptr + offs, sorted_values)
    tl.store(indices_ptr + offs, sorted_indices.to(tl.int32))


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
            "R1-R15 specialize the actual WeLM common path: contiguous [M,512]"
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

        if variant == "r7_vector_tie_recovery":
            # Optimization point 5 (scalar-to-vector), isolated from R6:
            # keep its grid, cutoff, and bias placement; only replace the ten
            # serial scalar tie-recovery reductions inside each row.
            grid, rows_per_program, extra_rows = self._partition(m)
            kernel = (
                _r7_vector_tie_small_kernel
                if m <= R6_DISPATCH_CUTOFF_M
                else _r7_vector_tie_large_kernel
            )

            def launch() -> None:
                kernel[grid](
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

        if variant == "r8_safe_vector_dispatch":
            # Optimization point 12 (grid/multipath specialization): R7 is a
            # win only while each program owns one row.  Once M exceeds the
            # physical AIV count, bind the exact R6 winner instead.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r3_sort_tie_kernel
            else:
                kernel = _r5_large_partition_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r9_pair_tie_recovery":
            # Optimization point 5 (scalar-to-vector), isolated from R8.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r9_pair_tie_medium_kernel
            else:
                kernel = _r9_pair_tie_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r10_group_rank_threshold":
            # Optimization point 7 (redundant state/pass elimination), based
            # on R8 rather than the regressed R9.  Keep R8's single-row R7
            # path, grid, cutoff, and bias placement; above the physical-AIV
            # boundary only replace candidate-mask maintenance with one scalar
            # threshold inside each contiguous equal-routing-value group.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r10_group_rank_medium_kernel
            else:
                kernel = _r10_group_rank_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r11_packed_exact_sort":
            # Rejected experiment: A5 I64 sort failed one-ULP ordering.  Keep
            # the provider in full-run history as an exact R10 fallback.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r10_group_rank_medium_kernel
            else:
                kernel = _r10_group_rank_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r12_two_row_tile":
            # Optimization point 2 (UB-aware tiling), based on exact R10.
            # Preserve the one-row R7 path.  Once each AIV owns at least two
            # rows, expose two adjacent rank-1 row bodies per loop iteration;
            # the sort primitive itself remains rank-1 as required by A5.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif rows_per_program < 2:
                kernel = _r10_group_rank_medium_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r12_two_row_medium_kernel
            else:
                kernel = _r12_two_row_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r13_fp32_rank_reduce":
            # Optimization point 6 (avoid scalar/weak integer lowering), based
            # on R10.  Keep its grid, schedule, cutoff, small R7 path and bias
            # placement; only the exact 0..511 tie-rank min uses FP32 Vector.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r13_fp32_rank_medium_kernel
            else:
                kernel = _r13_fp32_rank_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r14_priority_argmax":
            # Optimization point 6 (A5 low-rank FP32 indexed reduction), based
            # on R10.  Preserve its grid/schedule/dispatch/bias choices; only
            # replace integer min by priority-order gather + leftmost argmax.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r14_priority_argmax_medium_kernel
            else:
                kernel = _r14_priority_argmax_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r15_unique_top10_fastpath":
            # Optimization point 12 (data-dependent multipath), based on R10.
            # Preserve its grid/schedule/small path/bias placement.  Above the
            # AIV boundary, unique top-10 rows use one parallel equality/min;
            # duplicate rows execute the exact R10 group-threshold fallback.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r16_split_unique_recovery":
            # The A5 backend rejected both [512,8] drafts with runtime 507035
            # at M=57.  Keep this rejected round measurable as exact R15.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r17_parallel_priority_argmax":
            # A5 scalar-lowered rank-2 argmax (M=16384 reached ~35 ms).  Keep
            # this rejected round measurable as the exact R15 implementation.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r18_hierarchical_partial_sort":
            # Hierarchical sort was ~17% slower at M=16384.  Keep this rejected
            # round measurable as the exact R15 implementation.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r19_unique_id_sum":
            # FP32 ID sum was ~27% slower at M=16384.  Keep this rejected round
            # measurable as the exact R15 implementation.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r20_iterative_rank1_maxarg":
            # Iterative rank-1 maxarg was ~2.14x slower at M=16384.  Keep this
            # rejected round measurable as the exact R15 implementation.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r21_prefill_partition_constexpr":
            # Fixed prefill constexpr showed no gain.  Keep this rejected round
            # measurable as the exact R15 implementation.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant in (
            "r22_physical_core_grid",
            "r23_sort_index_capability",
        ):
            # R22's 28-program grid nearly halved large-shape throughput.  R23
            # is a capability-only experiment.  Keep both historical provider
            # rows exact and measurable as the accepted R15 dispatch.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r24_paired_max_index":
            # R24's unconditional -inf-tail argmax made M=16384 slower than
            # R0.  Retain the historical row as the exact accepted R15 path.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r25_lexicographic_tuple_reduce":
            # Public tuple reduction is expressible, but current A5 BiShengIR
            # fails dominance verification on its two results in the row loop.
            # Keep the historical row exact and measurable as accepted R15.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r26_unique_direct_expert_id":
            # R26's rank-2 FP32 ID reduce was ~1.9x slower for large M.  Keep
            # the historical row measurable as the exact accepted R15 path.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r15_unique_top10_medium_kernel
            else:
                kernel = _r15_unique_top10_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r27_last_axis_rank_reduce":
            # Optimization point 2 (tiling/reduction-axis layout), isolated
            # from R15.  Preserve all math and only transpose [512,16] to
            # [16,512] so the exact rank min reduces along the last axis.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r27_last_axis_rank_medium_kernel
            else:
                kernel = _r27_last_axis_rank_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r28_split_last_axis_rank_reduce":
            # R28's second rank-2 reduction regressed large M by ~16%.  Keep
            # the historical row measurable as the exact accepted R27 path.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r27_last_axis_rank_medium_kernel
            else:
                kernel = _r27_last_axis_rank_large_kernel

            def launch() -> None:
                kernel[grid](
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

        if variant == "r29_eight_plus_rank1_tail":
            # Optimization point 2 (tiling), isolated from R27/R28.  Retain
            # one [8,512] last-axis rank min and recover only the final two
            # values with proven rank-1 exact reductions.
            grid, rows_per_program, extra_rows = self._partition(m)
            if m <= self.num_vector_cores:
                kernel = _r7_vector_tie_small_kernel
            elif m <= R6_DISPATCH_CUTOFF_M:
                kernel = _r29_eight_plus_rank1_medium_kernel
            else:
                kernel = _r29_eight_plus_rank1_large_kernel

            def launch() -> None:
                kernel[grid](
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
# Device capability probe.  This is diagnostic, never a correctness oracle.
# ---------------------------------------------------------------------------


def run_sort_index_capability_probe(
    model: ModelNew,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    """Try the natural public values+indices extension without aborting a run."""
    try:
        sort_signature = str(inspect.signature(al.sort))
    except (TypeError, ValueError) as exc:
        sort_signature = f"unavailable: {type(exc).__name__}: {exc}"

    # All values are finite and unique, so index semantics have an unambiguous
    # independent reference.  This probe intentionally does not claim MMQ tie
    # compatibility; that requires the full adversarial suite in a later round.
    input_cpu = torch.tensor(
        [
            3.0, -7.0, 12.0, 0.5, -1.0, 8.0, 4.0, 15.0,
            -9.0, 6.0, 1.0, 14.0, -3.0, 10.0, 2.0, 13.0,
            7.0, -5.0, 11.0, -0.5, 5.0, -8.0, 9.0, -2.0,
            16.0, -6.0, 0.0, 17.0, -4.0, 18.0, 19.0, 20.0,
        ],
        dtype=torch.float32,
    )
    input_npu = input_cpu.to(device)
    values_npu = torch.empty((32,), dtype=torch.float32, device=device)
    indices_npu = torch.empty((32,), dtype=torch.int32, device=device)
    status = "UNAVAILABLE"
    values_exact = False
    indices_exact = False
    error_type = ""
    error_excerpt = ""

    try:
        _r23_sort_index_probe_kernel[(1,)](
            input_npu,
            values_npu,
            indices_npu,
            multibuffer=False,
            unit_flag=False,
        )
        torch_npu.npu.synchronize()
        expected_values, expected_indices = torch.sort(
            input_cpu, descending=True, stable=True
        )
        actual_values = values_npu.cpu()
        actual_indices = indices_npu.cpu().to(torch.int64)
        values_exact = torch.equal(
            actual_values.contiguous().view(torch.int32),
            expected_values.contiguous().view(torch.int32),
        )
        indices_exact = torch.equal(actual_indices, expected_indices)
        status = "SUPPORTED" if values_exact and indices_exact else "SEMANTIC_MISMATCH"
    except Exception as exc:  # The unsupported public signature is expected.
        error_type = type(exc).__name__
        error_excerpt = " ".join(str(exc).split())[:1000]

    print("\nR23 al.sort values+indices capability probe")
    print(f"  signature: {sort_signature}")
    print(
        f"  status={status}, values_exact={values_exact}, "
        f"indices_exact={indices_exact}"
    )
    if error_type:
        print(f"  expected probe failure: {error_type}: {error_excerpt}")

    return {
        **model.runtime_metadata(seed),
        "record_type": "capability",
        "case": "al_sort_return_indices",
        "m": 1,
        "n": 32,
        "k": 32,
        "variant": "r23_sort_index_capability",
        "status": status,
        "scope": "device_compile_and_run_probe",
        "al_sort_signature": sort_signature,
        "probe_values_bitwise_exact": values_exact,
        "probe_indices_exact": indices_exact,
        "probe_error_type": error_type,
        "probe_error_excerpt": error_excerpt,
    }


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
    for m in (1, 7, 64, 384):
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
    if normalized in ("all", "0-29", "r0-r29"):
        return list(VARIANTS)
    if normalized in ("0-28", "r0-r28"):
        return list(VARIANTS[:29])
    if normalized in ("0-27", "r0-r27"):
        return list(VARIANTS[:28])
    if normalized in ("0-26", "r0-r26"):
        return list(VARIANTS[:27])
    if normalized in ("0-25", "r0-r25"):
        return list(VARIANTS[:26])
    if normalized in ("0-24", "r0-r24"):
        return list(VARIANTS[:25])
    if normalized in ("0-23", "r0-r23"):
        return list(VARIANTS[:24])
    if normalized in ("0-22", "r0-r22"):
        return list(VARIANTS[:23])
    if normalized in ("0-21", "r0-r21"):
        return list(VARIANTS[:22])
    if normalized in ("0-20", "r0-r20"):
        return list(VARIANTS[:21])
    if normalized in ("0-19", "r0-r19"):
        return list(VARIANTS[:20])
    if normalized in ("0-18", "r0-r18"):
        return list(VARIANTS[:19])
    if normalized in ("0-17", "r0-r17"):
        return list(VARIANTS[:18])
    if normalized in ("0-16", "r0-r16"):
        return list(VARIANTS[:17])
    if normalized in ("0-15", "r0-r15"):
        return list(VARIANTS[:16])
    if normalized in ("0-14", "r0-r14"):
        return list(VARIANTS[:15])
    if normalized in ("0-13", "r0-r13"):
        return list(VARIANTS[:14])
    if normalized in ("0-12", "r0-r12"):
        return list(VARIANTS[:13])
    if normalized in ("0-11", "r0-r11"):
        return list(VARIANTS[:12])
    if normalized in ("0-10", "r0-r10"):
        return list(VARIANTS[:11])
    if normalized in ("0-9", "r0-r9"):
        return list(VARIANTS[:10])
    if normalized in ("0-8", "r0-r8"):
        return list(VARIANTS[:9])
    if normalized in ("0-7", "r0-r7"):
        return list(VARIANTS[:8])
    if normalized in ("0-6", "r0-r6"):
        return list(VARIANTS[:7])
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
        description="WeLM-v4 A5 expert-bias TopK R0-R29 accuracy/latency study"
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
        help="all, 0-29, or comma list such as r27,r28,r29 (R0 is always added)",
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

    if "r23_sort_index_capability" in variants:
        records.append(
            run_sort_index_capability_probe(
                model,
                device=device,
                seed=args.seed,
            )
        )

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
