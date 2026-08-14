#!/usr/bin/env python3
"""Standalone Ascend benchmark for ``welmv4_inplace_rope_npu``.

The file intentionally contains two independent Triton implementations:

* ``baseline`` is a frozen copy of the current NEWSGLANG NPU implementation.
* ``candidate`` is the only section that should change during optimization.

The benchmark does not import NEWSGLANG, so the remote NPU worker only needs
this small Git repository plus its normal torch/torch_npu/Triton environment.
It checks both ordinary Q/K RoPE and the KV-mirror path where Q has ``BS`` rows
while K/positions have ``N`` rows.

Default model-local shape (WeLM v4.5 TP rank): 6 Q heads, 1 KV head,
head_dim=256, rope_dim=64, BF16.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch_npu
import triton
import triton.language as tl


HEAD_DIM = 256
ROPE_DIM = 64
HALF_ROPE_DIM = ROPE_DIM // 2
NUM_Q_HEADS = 6
NUM_K_HEADS = 1
MAX_POSITION = 32768
ROPE_BASE = 100000.0
NUM_STAGES = 4
PROGRAMS_PER_VECTOR_CORE = 8
PREFILL_TOKEN_BLOCK = 64
PREFILL_TOKEN_BLOCK_FALLBACK = 32
PREFILL_TOKEN_BLOCK_MIN = 16
DTYPE = torch.bfloat16
ATOL = 2.0e-2
RTOL = 2.0e-2
IR_CAPTURE_SCRIPT = "capture_welmv4_rope_ir.sh"
AUTO_OUTPUT_CSV = "welmv4_inplace_rope_npu_all.csv"
IR_CAPTURE_CASE = "prefill_m8192"
PROFILE_CAPTURE_CASE = "prefill_m16384"


@dataclass(frozen=True)
class Case:
    name: str
    phase: str
    n_tokens: int
    batch_size: int = 0

    @property
    def is_mirror(self) -> bool:
        return self.batch_size > 0

    @property
    def q_rows(self) -> int:
        return self.batch_size if self.is_mirror else self.n_tokens


# Decode latency is launch-bound and can vary at every batch size.  Keep the
# complete production concurrency range instead of sampling powers of two.
DECODE_CASES = tuple(Case(f"decode_m{m}", "decode", m) for m in range(1, 65))
PREFILL_CASES = tuple(
    Case(f"prefill_m{m}", "prefill", m)
    for m in (128, 256, 512, 1024, 2048, 4096, 8192, 9616, 16384)
)
MIRROR_CASES = (
    Case("mirror_m8192_bs4", "prefill_mirror", 8192, 4),
    Case("mirror_m16384_bs8", "prefill_mirror", 16384, 8),
)
ALL_CASES = DECODE_CASES + PREFILL_CASES + MIRROR_CASES


# ---------------------------------------------------------------------------
# Frozen R0 baseline: copied from NEWSGLANG welmv4_npu_op.py on 2026-08-14.
# Do not edit this section during optimization rounds.
# ---------------------------------------------------------------------------


@triton.jit
def _baseline_apply_tail_rope(
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    head_offsets = tl.arange(0, num_heads_blocked)
    rope_offsets = tl.arange(0, half_rope_dim)
    mask = head_offsets[:, None] < num_heads
    base = data_ptr + head_offsets[:, None] * head_dim
    x1 = tl.load(
        base + rope_offsets[None, :], mask=mask, care_padding=False
    )
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, :],
        mask=mask,
        care_padding=False,
    )
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, :], out1, mask=mask)
    tl.store(base + half_rope_dim + rope_offsets[None, :], out2, mask=mask)


@triton.jit(do_not_specialize=["N", "BS"])
def _baseline_welmv4_inplace_rope_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    N: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads_blocked: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(
        tl.program_id(0), N, tl.num_programs(0), num_stages=num_stages
    ):
        position_id = tl.load(position_ptr + token_id)
        cos = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + cos_offsets,
            care_padding=False,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + sin_offsets,
            care_padding=False,
        )
        q_data = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _baseline_apply_tail_rope(
            k_data,
            cos,
            sin,
            num_k_heads,
            num_k_heads_blocked,
            head_dim,
            rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                q_position_id = tl.load(last_index_ptr + token_id)
                q_position_id = tl.load(position_ptr + q_position_id)
                q_cos = tl.load(
                    cos_sin_cache_ptr
                    + q_position_id * rope_dim
                    + cos_offsets,
                    care_padding=False,
                )
                q_sin = tl.load(
                    cos_sin_cache_ptr
                    + q_position_id * rope_dim
                    + sin_offsets,
                    care_padding=False,
                )
                _baseline_apply_tail_rope(
                    q_data,
                    q_cos,
                    q_sin,
                    num_q_heads,
                    num_q_heads_blocked,
                    head_dim,
                    rope_dim,
                )
        else:
            _baseline_apply_tail_rope(
                q_data,
                cos,
                sin,
                num_q_heads,
                num_q_heads_blocked,
                head_dim,
                rope_dim,
            )


# ---------------------------------------------------------------------------
# Optimization candidate (R1 initially equals the frozen R0 baseline).
# Edit only this section in later optimization rounds.
# ---------------------------------------------------------------------------


@triton.jit
def _candidate_apply_tail_rope(
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    head_offsets = tl.arange(0, num_heads_blocked)
    rope_offsets = tl.arange(0, half_rope_dim)
    mask = head_offsets[:, None] < num_heads
    base = data_ptr + head_offsets[:, None] * head_dim
    x1 = tl.load(
        base + rope_offsets[None, :], mask=mask, care_padding=False
    )
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, :],
        mask=mask,
        care_padding=False,
    )
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, :], out1, mask=mask)
    tl.store(base + half_rope_dim + rope_offsets[None, :], out2, mask=mask)


@triton.jit(do_not_specialize=["N", "BS"])
def _candidate_welmv4_inplace_rope_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    N: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads_blocked: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(
        tl.program_id(0), N, tl.num_programs(0), num_stages=num_stages
    ):
        position_id = tl.load(position_ptr + token_id).to(tl.int32)
        cos = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + cos_offsets,
            care_padding=False,
        )
        sin = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + sin_offsets,
            care_padding=False,
        )
        q_data = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _candidate_apply_tail_rope(
            k_data,
            cos,
            sin,
            num_k_heads,
            num_k_heads_blocked,
            head_dim,
            rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                q_position_id = tl.load(last_index_ptr + token_id).to(
                    tl.int32
                )
                q_position_id = tl.load(
                    position_ptr + q_position_id
                ).to(tl.int32)
                q_cos = tl.load(
                    cos_sin_cache_ptr
                    + q_position_id * rope_dim
                    + cos_offsets,
                    care_padding=False,
                )
                q_sin = tl.load(
                    cos_sin_cache_ptr
                    + q_position_id * rope_dim
                    + sin_offsets,
                    care_padding=False,
                )
                _candidate_apply_tail_rope(
                    q_data,
                    q_cos,
                    q_sin,
                    num_q_heads,
                    num_q_heads_blocked,
                    head_dim,
                    rope_dim,
                )
        else:
            _candidate_apply_tail_rope(
                q_data,
                cos,
                sin,
                num_q_heads,
                num_q_heads_blocked,
                head_dim,
                rope_dim,
            )


@triton.jit
def _candidate_apply_token_block_rope(
    data_ptr: tl.tensor,
    token_offsets: tl.tensor,
    token_stride: tl.constexpr,
    cos: tl.tensor,
    sin: tl.tensor,
    token_mask: tl.tensor,
    masked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    rope_offsets = tl.arange(0, half_rope_dim)
    base = data_ptr + token_offsets[:, None] * token_stride
    mask = token_mask[:, None]
    if masked:
        x1 = tl.load(
            base + rope_offsets[None, :],
            mask=mask,
            other=0.0,
            care_padding=False,
        )
        x2 = tl.load(
            base + half_rope_dim + rope_offsets[None, :],
            mask=mask,
            other=0.0,
            care_padding=False,
        )
    else:
        x1 = tl.load(
            base + rope_offsets[None, :], care_padding=False
        )
        x2 = tl.load(
            base + half_rope_dim + rope_offsets[None, :],
            care_padding=False,
        )
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    if masked:
        tl.store(base + rope_offsets[None, :], out1, mask=mask)
        tl.store(
            base + half_rope_dim + rope_offsets[None, :], out2, mask=mask
        )
    else:
        tl.store(base + rope_offsets[None, :], out1)
        tl.store(base + half_rope_dim + rope_offsets[None, :], out2)


@triton.jit
def _candidate_apply_token_head_block_rope(
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
    x1 = tl.load(
        base + rope_offsets[None, None, :], care_padding=False
    )
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, None, :],
        care_padding=False,
    )
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, None, :], out1)
    tl.store(
        base + half_rope_dim + rope_offsets[None, None, :], out2
    )


@triton.jit
def _candidate_apply_masked_token_head_block_rope(
    data_ptr: tl.tensor,
    token_offsets: tl.tensor,
    token_stride: tl.constexpr,
    cos: tl.tensor,
    sin: tl.tensor,
    token_mask: tl.tensor,
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
    mask = token_mask[:, None, None]
    x1 = tl.load(
        base + rope_offsets[None, None, :],
        mask=mask,
        other=0.0,
        care_padding=False,
    )
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, None, :],
        mask=mask,
        other=0.0,
        care_padding=False,
    )
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(
        base + rope_offsets[None, None, :], out1, mask=mask
    )
    tl.store(
        base + half_rope_dim + rope_offsets[None, None, :],
        out2,
        mask=mask,
    )


@triton.jit(do_not_specialize=["num_token_blocks", "N"])
def _candidate_welmv4_inplace_rope_prefill_kernel(
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
    masked: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
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
        token_mask = token_base + token_offsets < N
        if masked:
            position_ids = tl.load(
                position_ptr + token_base + token_offsets,
                mask=token_mask,
                other=0,
            ).to(tl.int32)
            cos = tl.load(
                cos_sin_cache_ptr
                + position_ids[:, None] * rope_dim
                + cos_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
                care_padding=False,
            )
            sin = tl.load(
                cos_sin_cache_ptr
                + position_ids[:, None] * rope_dim
                + sin_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
                care_padding=False,
            )
        else:
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

        k_data = (
            k_ptr
            + token_base * k_token_stride
            + head_dim
            - rope_dim
        )
        _candidate_apply_token_block_rope(
            k_data,
            token_offsets,
            k_token_stride,
            cos,
            sin,
            token_mask,
            masked,
            head_dim,
            rope_dim,
        )

        if masked:
            q_data = (
                q_ptr
                + token_base * q_token_stride
                + head_dim
                - rope_dim
            )
            _candidate_apply_masked_token_head_block_rope(
                q_data,
                token_offsets,
                q_token_stride,
                cos,
                sin,
                token_mask,
                4,
                head_dim,
                rope_dim,
            )
            _candidate_apply_masked_token_head_block_rope(
                q_data + 4 * head_dim,
                token_offsets,
                q_token_stride,
                cos,
                sin,
                token_mask,
                2,
                head_dim,
                rope_dim,
            )
        else:
            q_data = (
                q_ptr
                + token_base * q_token_stride
                + head_dim
                - rope_dim
            )
            _candidate_apply_token_head_block_rope(
                q_data,
                token_offsets,
                q_token_stride,
                cos,
                sin,
                4,
                head_dim,
                rope_dim,
            )
            _candidate_apply_token_head_block_rope(
                q_data + 4 * head_dim,
                token_offsets,
                q_token_stride,
                cos,
                sin,
                2,
                head_dim,
                rope_dim,
            )

PROVIDERS = {
    "baseline": _baseline_welmv4_inplace_rope_kernel,
    "candidate": _candidate_welmv4_inplace_rope_kernel,
}


# ---------------------------------------------------------------------------
# Inputs, reference, launch binding, and measurement.
# ---------------------------------------------------------------------------


def repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class Harness:
    def __init__(self, device: torch.device, seed: int) -> None:
        self.device = device
        self.seed = seed
        self.device_index = int(torch_npu.npu.current_device())
        properties = triton.runtime.driver.active.utils.get_device_properties(
            self.device_index
        )
        self.num_vector_cores = int(
            properties.get("num_vectorcore", properties.get("num_aicore", -1))
        )
        if self.num_vector_cores <= 0:
            raise RuntimeError("could not determine the visible NPU vector-core count")
        self.device_name = str(torch_npu.npu.get_device_name(self.device_index))
        self.cache = make_cos_sin_cache(device)
        self.commit = repository_head()

    def metadata(self) -> dict[str, object]:
        return {
            "benchmark_commit": self.commit,
            "device": str(self.device),
            "device_name": self.device_name,
            "device_index": self.device_index,
            "num_vector_cores": self.num_vector_cores,
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
            "triton_version": str(getattr(triton, "__version__", "unknown")),
            "cann_version": str(getattr(torch.version, "cann", "unknown")),
            "python_version": platform.python_version(),
            "seed": self.seed,
            "dtype": str(DTYPE).removeprefix("torch."),
            "head_dim": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "num_q_heads": NUM_Q_HEADS,
            "num_k_heads": NUM_K_HEADS,
        }

    def bind(
        self,
        provider: str,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
        last_index: torch.Tensor | None,
    ) -> Callable[[], object]:
        n_tokens = int(positions.shape[0])
        batch_size = int(last_index.numel()) if last_index is not None else 0
        prefill_token_block = 0
        prefill_masked = False
        if provider == "candidate" and last_index is None:
            if n_tokens >= 8192 and n_tokens % PREFILL_TOKEN_BLOCK != 0:
                prefill_token_block = PREFILL_TOKEN_BLOCK
                prefill_masked = True
            elif (
                n_tokens >= PREFILL_TOKEN_BLOCK
                and n_tokens % PREFILL_TOKEN_BLOCK == 0
            ):
                prefill_token_block = PREFILL_TOKEN_BLOCK
            elif (
                n_tokens >= PREFILL_TOKEN_BLOCK_FALLBACK
                and n_tokens % PREFILL_TOKEN_BLOCK_FALLBACK == 0
            ):
                prefill_token_block = PREFILL_TOKEN_BLOCK_FALLBACK
            elif (
                n_tokens >= PREFILL_TOKEN_BLOCK_MIN
                and n_tokens % PREFILL_TOKEN_BLOCK_MIN == 0
            ):
                prefill_token_block = PREFILL_TOKEN_BLOCK_MIN
        use_blocked_prefill = prefill_token_block > 0
        kernel = (
            _candidate_welmv4_inplace_rope_prefill_kernel
            if use_blocked_prefill
            else PROVIDERS[provider]
        )
        if use_blocked_prefill:
            work_items = (
                triton.cdiv(n_tokens, prefill_token_block)
                if prefill_masked
                else n_tokens // prefill_token_block
            )
        else:
            work_items = n_tokens
        num_programs = min(
            work_items, self.num_vector_cores * PROGRAMS_PER_VECTOR_CORE
        )
        q_stride = int(query.stride(0))
        k_stride = int(key.stride(0))
        q_heads_blocked = triton.next_power_of_2(NUM_Q_HEADS)
        k_heads_blocked = triton.next_power_of_2(NUM_K_HEADS)

        def launch() -> object:
            if use_blocked_prefill:
                return kernel[(num_programs,)](
                    query,
                    key,
                    positions,
                    self.cache,
                    work_items,
                    n_tokens,
                    q_stride,
                    k_stride,
                    HEAD_DIM,
                    ROPE_DIM,
                    prefill_token_block,
                    prefill_masked,
                    NUM_STAGES,
                    NUM_Q_HEADS,
                )
            return kernel[(num_programs,)](
                query,
                key,
                positions,
                self.cache,
                last_index,
                n_tokens,
                batch_size,
                q_stride,
                k_stride,
                HEAD_DIM,
                ROPE_DIM,
                NUM_STAGES,
                NUM_Q_HEADS,
                NUM_K_HEADS,
                q_heads_blocked,
                k_heads_blocked,
            )

        return launch


def make_cos_sin_cache(device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    positions = torch.arange(MAX_POSITION, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    return cache.to(device=device, dtype=DTYPE)


def make_last_index(case: Case, device: torch.device) -> torch.Tensor | None:
    if not case.is_mirror:
        return None
    values = [
        max(0, min(case.n_tokens - 1, (i + 1) * case.n_tokens // case.batch_size - 1))
        for i in range(case.batch_size)
    ]
    return torch.tensor(values, device=device, dtype=torch.int64)


def make_inputs(
    case: Case, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    case_seed = seed + case.n_tokens * 17 + case.batch_size * 101
    torch.manual_seed(case_seed)
    positions = (
        torch.arange(case.n_tokens, device=device, dtype=torch.int64) * 17 + 11
    ) % MAX_POSITION
    query = torch.randn(
        (case.q_rows, NUM_Q_HEADS, HEAD_DIM), device=device, dtype=DTYPE
    )
    key = torch.randn(
        (case.n_tokens, NUM_K_HEADS, HEAD_DIM), device=device, dtype=DTYPE
    )
    return query, key, positions, make_last_index(case, device)


def apply_reference(
    data: torch.Tensor,
    rope_positions: torch.Tensor,
    cache: torch.Tensor,
) -> torch.Tensor:
    output = data.clone()
    cos_sin = cache.index_select(0, rope_positions)
    cos = cos_sin[:, :HALF_ROPE_DIM].float().unsqueeze(1)
    sin = cos_sin[:, HALF_ROPE_DIM:].float().unsqueeze(1)
    rotary = data[..., HEAD_DIM - ROPE_DIM :]
    x1 = rotary[..., :HALF_ROPE_DIM].float()
    x2 = rotary[..., HALF_ROPE_DIM:].float()
    output[..., HEAD_DIM - ROPE_DIM : HEAD_DIM - HALF_ROPE_DIM] = (
        x1 * cos - x2 * sin
    ).to(DTYPE)
    output[..., HEAD_DIM - HALF_ROPE_DIM :] = (x1 * sin + x2 * cos).to(DTYPE)
    return output


def reference_outputs(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    last_index: torch.Tensor | None,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_positions = (
        positions if last_index is None else positions.index_select(0, last_index)
    )
    return (
        apply_reference(query, query_positions, cache),
        apply_reference(key, positions, cache),
    )


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def case_fields(case: Case) -> dict[str, object]:
    return {
        "case": case.name,
        "phase": case.phase,
        "path": "kv_mirror" if case.is_mirror else "normal",
        "n_tokens": case.n_tokens,
        "q_rows": case.q_rows,
        "batch_size": case.batch_size,
    }


def run_correctness(
    harness: Harness, cases: Sequence[Case]
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    failures = 0
    print("\nCorrectness (BF16 tail-RoPE reference):")
    for case in cases:
        query, key, positions, last_index = make_inputs(
            case, harness.device, harness.seed
        )
        query_ref, key_ref = reference_outputs(
            query, key, positions, last_index, harness.cache
        )
        for provider in PROVIDERS:
            query_out = query.clone()
            key_out = key.clone()
            launch = harness.bind(
                provider, query_out, key_out, positions, last_index
            )
            launch()
            torch_npu.npu.synchronize()
            status = "PASS"
            detail = ""
            try:
                torch.testing.assert_close(
                    query_out, query_ref, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(key_out, key_ref, atol=ATOL, rtol=RTOL)
                nope_dim = HEAD_DIM - ROPE_DIM
                if not torch.equal(query_out[..., :nope_dim], query[..., :nope_dim]):
                    raise AssertionError("query non-RoPE prefix was modified")
                if not torch.equal(key_out[..., :nope_dim], key[..., :nope_dim]):
                    raise AssertionError("key non-RoPE prefix was modified")
            except AssertionError as exc:
                status = "FAIL"
                detail = str(exc).replace("\n", " | ")
                failures += 1

            q_error = max_abs_error(query_out, query_ref)
            k_error = max_abs_error(key_out, key_ref)
            print(
                f"  {case.name:<24} {provider:<10} {status:<4} "
                f"max_abs_q={q_error:.6g} max_abs_k={k_error:.6g}"
            )
            records.append(
                {
                    **harness.metadata(),
                    **case_fields(case),
                    "record_type": "correctness",
                    "variant": provider,
                    "status": status,
                    "detail": detail,
                    "atol": ATOL,
                    "rtol": RTOL,
                    "max_abs_q": q_error,
                    "max_abs_k": k_error,
                }
            )
    return records, failures


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def automatic_inner_repeat(n_tokens: int) -> int:
    if n_tokens <= 64:
        return 200
    if n_tokens <= 512:
        return 100
    if n_tokens <= 2048:
        return 50
    if n_tokens <= 9616:
        return 10
    return 5


def event_sample_us(launch: Callable[[], object], inner_repeat: int) -> float:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(inner_repeat):
        launch()
    end.record()
    torch_npu.npu.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / inner_repeat


def run_performance(
    harness: Harness,
    cases: Sequence[Case],
    *,
    scope: str,
    warmup: int,
    rounds: int,
    inner_repeat_override: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    print(
        f"\nPerformance: scope={scope}, warmup={warmup}, rounds={rounds}, "
        f"physical_AIV={harness.num_vector_cores}"
    )
    for case in cases:
        query, key, positions, last_index = make_inputs(
            case, harness.device, harness.seed
        )
        launches: dict[str, Callable[[], object]] = {}
        for provider in PROVIDERS:
            launches[provider] = harness.bind(
                provider, query.clone(), key.clone(), positions, last_index
            )

        for provider in PROVIDERS:
            for _ in range(warmup):
                launches[provider]()
        torch_npu.npu.synchronize()

        inner_repeat = (
            inner_repeat_override
            if inner_repeat_override > 0
            else automatic_inner_repeat(case.n_tokens)
        )
        samples = {provider: [] for provider in PROVIDERS}
        providers = tuple(PROVIDERS)
        for round_index in range(rounds):
            order: Iterable[str] = (
                providers if round_index % 2 == 0 else reversed(providers)
            )
            for provider in order:
                samples[provider].append(
                    event_sample_us(launches[provider], inner_repeat)
                )

        stats: dict[str, dict[str, float]] = {}
        for provider, values in samples.items():
            stats[provider] = {
                "p20_us": percentile(values, 0.20),
                "p50_us": statistics.median(values),
                "p80_us": percentile(values, 0.80),
                "mean_us": statistics.fmean(values),
            }

        baseline_p50 = stats["baseline"]["p50_us"]
        print(
            f"\n  {case.name}: N={case.n_tokens}, Q_rows={case.q_rows}, "
            f"BS={case.batch_size}, inner_repeat={inner_repeat}"
        )
        print("    variant      p20(us)   p50(us)   p80(us)   speedup/R0")
        for provider in PROVIDERS:
            current = stats[provider]
            speedup = baseline_p50 / current["p50_us"]
            print(
                f"    {provider:<10} {current['p20_us']:>9.3f} "
                f"{current['p50_us']:>9.3f} {current['p80_us']:>9.3f} "
                f"{speedup:>10.4f}x"
            )
            records.append(
                {
                    **harness.metadata(),
                    **case_fields(case),
                    "record_type": "performance",
                    "variant": provider,
                    "status": "MEASURED",
                    "scope": scope,
                    "warmup": warmup,
                    "rounds": rounds,
                    "inner_repeat": inner_repeat,
                    **current,
                    "speedup_vs_baseline": speedup,
                }
            )
    return records


# ---------------------------------------------------------------------------
# CLI and CSV.
# ---------------------------------------------------------------------------


def parse_cases(spec: str) -> list[Case]:
    normalized = spec.strip().lower()
    if normalized in ("all", "common"):
        return list(ALL_CASES)
    if normalized == "decode":
        return list(DECODE_CASES)
    if normalized == "prefill":
        return list(PREFILL_CASES)
    if normalized in ("mirror", "prefill_mirror"):
        return list(MIRROR_CASES)

    by_name = {case.name: case for case in ALL_CASES}
    by_tokens = {
        str(case.n_tokens): case for case in DECODE_CASES + PREFILL_CASES
    }
    selected: list[Case] = []
    for raw_item in spec.split(","):
        item = raw_item.strip().lower()
        case = by_name.get(item, by_tokens.get(item))
        if case is None:
            raise ValueError(
                f"unknown case {raw_item!r}; use all|decode|prefill|mirror, "
                "a case name, or a comma-separated normal-path token count"
            )
        if case not in selected:
            selected.append(case)
    if not selected:
        raise ValueError("no benchmark cases were selected")
    return selected


def run_compile_only(
    harness: Harness, case: Case, provider: str
) -> None:
    """Compile and launch one provider once for the external IR extractor."""
    query, key, positions, last_index = make_inputs(
        case, harness.device, harness.seed
    )
    launch = harness.bind(provider, query, key, positions, last_index)
    launch()
    torch_npu.npu.synchronize()
    print(f"IR compile-only launch completed: {provider}, {case.name}")


def capture_ir_records(
    harness: Harness, device: str
) -> list[dict[str, object]]:
    """Capture A5 compiler IR and encode it into the auto-committed CSV."""
    script = Path(__file__).resolve().with_name(IR_CAPTURE_SCRIPT)
    case = next(item for item in ALL_CASES if item.name == IR_CAPTURE_CASE)
    common = {
        **harness.metadata(),
        **case_fields(case),
        "record_type": "ir_artifact",
        "variant": "candidate",
        "scope": "compiler_ir",
    }
    if not script.is_file():
        return [{**common, "status": "ERROR", "capture_log": f"missing {script.name}"}]

    with tempfile.TemporaryDirectory(prefix="welmv4_rope_ir_") as output_dir:
        env = os.environ.copy()
        env.update(
            {
                "BENCH_PYTHON": sys.executable,
                "IR_OUTPUT_DIR": output_dir,
                "BISHENGIR_TARGET": harness.device_name,
            }
        )
        command = [
            "bash",
            str(script),
            str(Path(__file__).resolve()),
            "--compile-only-provider",
            "candidate",
            "--cases",
            case.name,
            "--device",
            device,
        ]
        print(f"\nCapturing compiler IR: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            return [{**common, "status": "ERROR", "capture_log": str(exc)}]

        output = result.stdout or ""
        files = sorted(Path(output_dir).glob("*.mlir"))
        if result.returncode != 0 or not files:
            print(output)
            return [
                {
                    **common,
                    "status": "ERROR",
                    "capture_returncode": result.returncode,
                    "capture_log": output[-60000:],
                }
            ]

        records: list[dict[str, object]] = []
        for path in files:
            content = path.read_bytes()
            records.append(
                {
                    **common,
                    "status": "CAPTURED",
                    "artifact_name": path.name,
                    "artifact_encoding": "gzip+base64",
                    "artifact_size_bytes": len(content),
                    "artifact_sha256": hashlib.sha256(content).hexdigest(),
                    "artifact_content": base64.b64encode(
                        gzip.compress(content, compresslevel=9)
                    ).decode("ascii"),
                }
            )
        print(
            "Captured compiler IR: "
            + ", ".join(f"{path.name} ({path.stat().st_size} B)" for path in files)
        )
        return records


def capture_profile_records(harness: Harness) -> list[dict[str, object]]:
    """Capture A5 pipe-utilization profiler CSVs for one accepted prefill case."""
    case = next(item for item in ALL_CASES if item.name == PROFILE_CAPTURE_CASE)
    common = {
        **harness.metadata(),
        **case_fields(case),
        "record_type": "profile_artifact",
        "variant": "candidate",
        "scope": "npu_pipe_profile",
    }
    query, key, positions, last_index = make_inputs(
        case, harness.device, harness.seed
    )
    launch = harness.bind("candidate", query, key, positions, last_index)
    for _ in range(5):
        launch()
    torch_npu.npu.synchronize()

    with tempfile.TemporaryDirectory(prefix="welmv4_rope_profile_") as output_dir:
        print(f"\nCapturing NPU pipe profile: {case.name} -> {output_dir}")
        try:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                l2_cache=False,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ):
                for _ in range(3):
                    launch()
                torch_npu.npu.synchronize()
        except Exception as exc:
            return [{**common, "status": "ERROR", "capture_log": repr(exc)}]

        wanted_names = {
            "kernel_details.csv",
            "operator_details.csv",
            "op_statistic.csv",
            "step_trace_time.csv",
        }
        files = sorted(
            path
            for path in Path(output_dir).rglob("*")
            if path.is_file()
            and (path.name in wanted_names or path.name.startswith("profiler_info"))
            and path.stat().st_size <= 10_000_000
        )
        if not files:
            discovered = sorted(
                str(path.relative_to(output_dir))
                for path in Path(output_dir).rglob("*")
                if path.is_file()
            )
            return [
                {
                    **common,
                    "status": "ERROR",
                    "capture_log": "no profiler summary files; discovered="
                    + repr(discovered[:100]),
                }
            ]

        records: list[dict[str, object]] = []
        for path in files:
            content = path.read_bytes()
            records.append(
                {
                    **common,
                    "status": "CAPTURED",
                    "artifact_name": str(path.relative_to(output_dir)).replace(
                        os.sep, "/"
                    ),
                    "artifact_encoding": "gzip+base64",
                    "artifact_size_bytes": len(content),
                    "artifact_sha256": hashlib.sha256(content).hexdigest(),
                    "artifact_content": base64.b64encode(
                        gzip.compress(content, compresslevel=9)
                    ).decode("ascii"),
                }
            )
        print(
            "Captured NPU profile: "
            + ", ".join(
                f"{path.name} ({path.stat().st_size} B)" for path in files
            )
        )
        return records


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
        description=(
            "WeLMv4 A5 inplace tail-RoPE correctness and kernel-latency study"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("both", "correctness", "performance"),
        default="both",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "all|decode|prefill|mirror, a case name, or comma-separated "
            "normal-path token counts"
        ),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--scope",
        choices=("kernel",),
        default="kernel",
        help="time only pre-bound Triton kernel launches on the NPU timeline",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--inner-repeat",
        type=int,
        default=0,
        help="0 selects an N-dependent repeat count; positive forces one value",
    )
    parser.add_argument(
        "--compile-only-provider",
        choices=tuple(PROVIDERS),
        default="",
        help="launch one selected provider/case once, for compiler IR capture",
    )
    parser.add_argument(
        "--capture-ir",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "capture candidate compiler IR; auto enables it only for the "
            f"standard {AUTO_OUTPUT_CSV} run"
        ),
    )
    parser.add_argument(
        "--capture-profile",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "capture candidate A5 pipe-utilization profiler summaries; use on "
            "for an explicit diagnostic run"
        ),
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
    cases = parse_cases(args.cases)
    harness = Harness(device, args.seed)

    print("WeLMv4 inplace tail-RoPE manual Ascend A5 study")
    print(
        f"device={device} ({harness.device_name}), "
        f"physical_AIV={harness.num_vector_cores}, commit={harness.commit[:12]}"
    )
    print(
        f"shape: q_heads={NUM_Q_HEADS}, kv_heads={NUM_K_HEADS}, "
        f"head_dim={HEAD_DIM}, rope_dim={ROPE_DIM}, dtype={DTYPE}"
    )

    if args.compile_only_provider:
        if len(cases) != 1:
            raise ValueError("--compile-only-provider requires exactly one case")
        run_compile_only(harness, cases[0], args.compile_only_provider)
        return 0

    records: list[dict[str, object]] = []
    failures = 0
    if args.mode in ("both", "correctness"):
        correctness_records, failures = run_correctness(harness, cases)
        records.extend(correctness_records)
        print(
            f"\nCorrectness summary: "
            f"{'PASS' if failures == 0 else 'FAIL'}, failures={failures}"
        )

    if failures:
        print("Performance skipped because correctness failed.")
    elif args.mode in ("both", "performance"):
        records.extend(
            run_performance(
                harness,
                cases,
                scope=args.scope,
                warmup=args.warmup,
                rounds=args.rounds,
                inner_repeat_override=args.inner_repeat,
            )
        )

    capture_ir = args.capture_ir == "on" or (
        args.capture_ir == "auto"
        and Path(args.output_csv).name == AUTO_OUTPUT_CSV
        and args.mode == "both"
        and args.cases.strip().lower() in ("all", "common")
    )
    if failures == 0 and capture_ir:
        records.extend(capture_ir_records(harness, str(device)))

    capture_profile = args.capture_profile == "on" or (
        args.capture_profile == "auto"
        and Path(args.output_csv).name == AUTO_OUTPUT_CSV
        and args.mode == "both"
        and args.cases.strip().lower() in ("all", "common")
    )
    if failures == 0 and capture_profile:
        records.extend(capture_profile_records(harness))

    write_csv(args.output_csv, records)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
