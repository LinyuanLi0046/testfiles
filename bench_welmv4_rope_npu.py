#!/usr/bin/env python3
"""Standalone Ascend benchmark for WeLMv4 RoPE DP-Attention layouts.

The worker compares the frozen production baseline with the optimization
candidate for TP4, TP4/DP2 and TP4/DP4 local head layouts.  Correctness uses a
FP32 torch reference.  Performance acceptance uses ``msprof op`` task duration;
NPU Event timing is intentionally forbidden by the workspace contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch_npu

import welmv4_rope_baseline as baseline_ops
import welmv4_rope_candidate as candidate_ops


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "rope_workspace_config.json"
BASELINE_PATH = ROOT / "welmv4_rope_baseline.py"
CANDIDATE_PATH = ROOT / "welmv4_rope_candidate.py"
IR_CAPTURE_SCRIPT = ROOT / "capture_welmv4_rope_ir.sh"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONTRACT = CONFIG["model_contract"]
VALIDATION = CONFIG["validation"]
DEFAULT_SEED = 20260831


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf_file(path: Path) -> str:
    """Hash source text after normalizing Git's CRLF/LF checkout detail."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def audit_frozen_baseline() -> None:
    expected = str(CONFIG["production_source"]["standalone_baseline_sha256"])
    actual = sha256_lf_file(BASELINE_PATH)
    if actual != expected:
        raise RuntimeError(
            "frozen RoPE baseline changed unexpectedly: "
            f"expected={expected}, actual={actual}"
        )
    if bool(VALIDATION.get("enable_npu_event_timing", False)):
        raise RuntimeError("NPU Event timing is forbidden for this workspace")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gzip_copy(source: Path, destination: Path, max_bytes: int) -> None:
    if source.stat().st_size > max_bytes:
        raise RuntimeError(f"artifact is too large: {source} ({source.stat().st_size})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, gzip.open(destination, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)


@dataclass(frozen=True)
class RopeCase:
    family: str
    topology: str
    n: int
    batch_size: int
    local_q_heads: int
    local_kv_heads: int
    dp_attention_size: int
    context_length: int

    @property
    def is_mirror(self) -> bool:
        return self.family.endswith("mirror")

    @property
    def is_segmented(self) -> bool:
        return self.family.startswith("segmented")

    @property
    def speculative_width(self) -> int:
        if self.family.startswith("mtp_d"):
            return int(self.family.removeprefix("mtp_d"))
        return 1

    @property
    def positions_are_contiguous(self) -> bool:
        return self.family.startswith("contiguous")

    @property
    def q_rows(self) -> int:
        return self.batch_size if self.is_mirror else self.n

    @property
    def name(self) -> str:
        return (
            f"{self.family}_n{self.n}_b{self.batch_size}_"
            f"ctx{self.context_length}_{self.topology}"
        )

    @property
    def workload_id(self) -> str:
        return (
            f"{self.family}_n{self.n}_b{self.batch_size}_ctx{self.context_length}"
        )

    @property
    def rotated_values(self) -> int:
        rope_dim = int(CONTRACT["rope_dim"])
        return rope_dim * (
            self.q_rows * self.local_q_heads + self.n * self.local_kv_heads
        )


def expand_specs(specs: Sequence[dict[str, object]]) -> list[RopeCase]:
    cases: list[RopeCase] = []
    topologies = CONTRACT["topologies"]
    for spec in specs:
        family = str(spec["family"])
        shapes: list[tuple[int, int]] = []
        if family == "decode":
            for m in spec.get("m_values", [spec.get("m")]):
                if m is None:
                    raise ValueError(f"decode case has no m/m_values: {spec}")
                value = int(m)
                shapes.append((value, value))
        elif family.startswith("mtp_d"):
            width = int(family.removeprefix("mtp_d"))
            if width not in (2, 3, 4):
                raise ValueError(f"unsupported MTP width: {family}")
            for batch in spec.get("batch_sizes", [spec.get("batch_size")]):
                if batch is None:
                    raise ValueError(f"MTP case has no batch_size(s): {spec}")
                value = int(batch)
                shapes.append((value * width, value))
        else:
            for n_value in spec.get("n_values", [spec.get("n")]):
                if n_value is None:
                    raise ValueError(f"case has no n/n_values: {spec}")
                n = int(n_value)
                default_batch = 1 if family == "contiguous_mirror" else 0
                shapes.append((n, int(spec.get("batch_size", default_batch))))

        raw_contexts = spec.get("context_lengths")
        if raw_contexts is None:
            raw_contexts = [spec.get("context_length")]
        contexts = list(raw_contexts)
        pairing = str(spec.get("pairing", "cross"))
        if pairing == "cycle":
            count = max(len(shapes), len(contexts))
            shape_context_pairs = [
                (shapes[index % len(shapes)], contexts[index % len(contexts)])
                for index in range(count)
            ]
        elif pairing == "cross":
            shape_context_pairs = [
                (shape, context) for shape in shapes for context in contexts
            ]
        else:
            raise ValueError(f"unknown case pairing {pairing!r}")

        for (n, batch_size), raw_context in shape_context_pairs:
            context_length = int(raw_context) if raw_context is not None else n
            if family.startswith("segmented") and batch_size <= 1:
                raise ValueError(f"segmented case requires batch_size>1: {spec}")
            if family.endswith("mirror") and batch_size <= 0:
                raise ValueError(f"mirror case requires batch_size>0: {spec}")
            if batch_size > n:
                raise ValueError(f"batch_size exceeds token count: {spec}")
            for topology_name in spec["topologies"]:
                topology = topologies[str(topology_name)]
                cases.append(
                    RopeCase(
                        family=family,
                        topology=str(topology_name),
                        n=n,
                        batch_size=batch_size,
                        local_q_heads=int(topology["local_q_heads"]),
                        local_kv_heads=int(topology["local_kv_heads"]),
                        dp_attention_size=int(topology["dp_attention_size"]),
                        context_length=context_length,
                    )
                )
    by_name = {case.name: case for case in cases}
    if len(by_name) != len(cases):
        raise ValueError("suite expands to duplicate case names")
    return cases


def phase_cases(suite: str, phase: str) -> list[RopeCase]:
    return expand_specs(CONFIG["suites"][suite].get(phase, []))


def all_suite_cases(suite: str) -> list[RopeCase]:
    merged: dict[str, RopeCase] = {}
    for phase in ("correctness", "performance", "ir", "profile"):
        for case in phase_cases(suite, phase):
            merged[case.name] = case
    return list(merged.values())


def select_named(cases: Sequence[RopeCase], names: Sequence[str]) -> list[RopeCase]:
    if not names:
        return list(cases)
    by_name = {case.name: case for case in cases}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown case name(s): {missing}")
    return [by_name[name] for name in names]


def balanced_lengths(total: int, batch_size: int) -> list[int]:
    if batch_size <= 0:
        return []
    base, remainder = divmod(total, batch_size)
    lengths = [base + int(index < remainder) for index in range(batch_size)]
    # Deterministically make the requests ragged while preserving positive
    # lengths and the exact total.
    for index in range(batch_size // 2):
        peer = batch_size - 1 - index
        delta = min((index % 5) + 1, lengths[index] - 1)
        lengths[index] -= delta
        lengths[peer] += delta
    assert all(length > 0 for length in lengths)
    assert sum(lengths) == total
    return lengths


def segment_tile_starts(lengths: Sequence[int]) -> list[int]:
    starts: list[int] = []
    offset = 0
    block = int(CONTRACT["token_block"])
    for length in lengths:
        starts.extend(range(offset, offset + int(length), block))
        offset += int(length)
    starts.append(offset)
    return starts


@dataclass
class RopeInputs:
    q_seed: torch.Tensor
    k_seed: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    positions: torch.Tensor
    cache: torch.Tensor
    last_index: torch.Tensor | None
    segment_tile_starts: torch.Tensor | None
    expected_q: torch.Tensor | None
    expected_k: torch.Tensor | None


def _reference_one(
    source: torch.Tensor,
    row_positions: torch.Tensor,
    cache_cpu: torch.Tensor,
) -> torch.Tensor:
    head_dim = int(CONTRACT["head_dim"])
    rope_dim = int(CONTRACT["rope_dim"])
    half = rope_dim // 2
    output = source.clone()
    tail = source[..., head_dim - rope_dim :].float()
    x1 = tail[..., :half]
    x2 = tail[..., half:]
    rows = cache_cpu.index_select(0, row_positions.to(torch.int64))
    cos = rows[:, None, :half]
    sin = rows[:, None, half:]
    rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    output[..., head_dim - rope_dim :] = rotated.to(output.dtype)
    return output


class Harness:
    def __init__(self, device_name: str, seed: int) -> None:
        self.device = torch.device(device_name)
        torch.npu.set_device(self.device)
        self.device_index = int(torch_npu.npu.current_device())
        self.device_name = str(torch_npu.npu.get_device_name(self.device_index))
        self.seed = seed
        max_position = int(CONTRACT["max_position"])
        rope_dim = int(CONTRACT["rope_dim"])
        half = rope_dim // 2
        positions = torch.arange(max_position, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        angles = positions[:, None] * inv_freq[None, :]
        self.cache_cpu = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
        self.cache = self.cache_cpu.to(self.device)
        self.baseline_hash = sha256_file(BASELINE_PATH)
        self.candidate_hash = sha256_file(CANDIDATE_PATH)

    def make_inputs(self, case: RopeCase, *, reference: bool) -> RopeInputs:
        generator = torch.Generator(device="cpu")
        case_seed = self.seed + int(hashlib.sha256(case.name.encode()).hexdigest()[:8], 16)
        generator.manual_seed(case_seed)
        head_dim = int(CONTRACT["head_dim"])
        q_cpu = torch.randn(
            (case.q_rows, case.local_q_heads, head_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        k_cpu = torch.randn(
            (case.n, case.local_kv_heads, head_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)

        lengths = balanced_lengths(case.n, case.batch_size) if case.is_segmented else []
        if case.family.startswith("mtp_d"):
            width = case.speculative_width
            pieces = []
            for request_id in range(case.batch_size):
                # Each request owns D consecutive speculative positions, while
                # different requests sit near the requested context band.
                request_end = case.context_length + (request_id % 7)
                pieces.append(
                    torch.arange(
                        request_end - width + 1,
                        request_end + 1,
                        dtype=torch.int32,
                    )
                )
            positions_cpu = torch.cat(pieces)
        elif case.family == "decode":
            positions_cpu = (
                case.context_length
                + torch.arange(case.n, dtype=torch.int32).remainder(17)
            )
        elif case.is_segmented:
            pieces: list[torch.Tensor] = []
            for index, length in enumerate(lengths):
                max_start = int(CONTRACT["max_position"]) - length - 1
                desired_start = max(0, case.context_length - length)
                prefix = (desired_start + index * 3079) % max_start
                pieces.append(torch.arange(prefix, prefix + length, dtype=torch.int32))
            positions_cpu = torch.cat(pieces)
        elif case.positions_are_contiguous:
            prefix = max(0, case.context_length - case.n)
            positions_cpu = torch.arange(prefix, prefix + case.n, dtype=torch.int32)
        else:
            modulus = int(CONTRACT["max_position"]) - 1
            positions_cpu = (
                torch.arange(case.n, dtype=torch.int64) * 997 + 123
            ).remainder(modulus).to(torch.int32)

        last_index_cpu: torch.Tensor | None = None
        if case.is_mirror:
            if case.is_segmented:
                cumulative = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
                last_index_cpu = (cumulative - 1).to(torch.int32)
            else:
                last_index_cpu = torch.tensor([case.n - 1], dtype=torch.int32)

        tile_starts_cpu: torch.Tensor | None = None
        if case.is_segmented:
            tile_starts_cpu = torch.tensor(
                segment_tile_starts(lengths), dtype=torch.int32
            )

        expected_q = expected_k = None
        if reference:
            q_positions = (
                positions_cpu.index_select(0, last_index_cpu.to(torch.int64))
                if last_index_cpu is not None
                else positions_cpu
            )
            expected_q = _reference_one(q_cpu, q_positions, self.cache_cpu)
            expected_k = _reference_one(k_cpu, positions_cpu, self.cache_cpu)

        q_seed = q_cpu.to(self.device)
        k_seed = k_cpu.to(self.device)
        return RopeInputs(
            q_seed=q_seed,
            k_seed=k_seed,
            query=torch.empty_like(q_seed),
            key=torch.empty_like(k_seed),
            positions=positions_cpu.to(self.device),
            cache=self.cache,
            last_index=(last_index_cpu.to(self.device) if last_index_cpu is not None else None),
            segment_tile_starts=(
                tile_starts_cpu.to(self.device) if tile_starts_cpu is not None else None
            ),
            expected_q=expected_q,
            expected_k=expected_k,
        )

    def bind(self, provider: str, case: RopeCase, inputs: RopeInputs) -> "BoundLaunch":
        module = baseline_ops if provider == "baseline" else candidate_ops

        def launch() -> None:
            inputs.query.copy_(inputs.q_seed)
            inputs.key.copy_(inputs.k_seed)
            module.welmv4_inplace_rope_npu(
                inputs.query,
                inputs.key,
                inputs.positions,
                inputs.cache,
                last_index=inputs.last_index,
                head_dim=int(CONTRACT["head_dim"]),
                rope_dim=int(CONTRACT["rope_dim"]),
                positions_are_contiguous=case.positions_are_contiguous,
                segment_tile_starts=inputs.segment_tile_starts,
            )

        return BoundLaunch(
            case=case,
            provider=provider,
            kernel_name=primary_kernel_name(provider, case),
            launch=launch,
        )


@dataclass(frozen=True)
class BoundLaunch:
    case: RopeCase
    provider: str
    kernel_name: str
    launch: Callable[[], None]


def provider_supports_optimized_layout(provider: str, case: RopeCase) -> bool:
    layouts = {
        (int(layout[0]), int(layout[1]))
        for layout in CONTRACT["optimized_head_layouts"][provider]
    }
    return (case.local_q_heads, case.local_kv_heads) in layouts


def primary_kernel_name(provider: str, case: RopeCase) -> str:
    supported = provider_supports_optimized_layout(provider, case)
    candidate_head_parallel = (
        provider == "candidate" and supported and case.local_q_heads != 6
    )
    threshold_all = int(CONTRACT["optimized_all_m_threshold"])
    threshold_exact = int(CONTRACT["optimized_exact64_threshold"])
    block = int(CONTRACT["token_block"])
    blocked = supported and (
        case.n >= threshold_all
        or (case.n >= threshold_exact and case.n % block == 0)
    )
    candidate_small_segmented_generic = (
        candidate_head_parallel
        and case.family == "segmented_prefill"
        and case.local_q_heads == 24
        and case.local_kv_heads == 2
        and 2
        * (
            len(segment_tile_starts(balanced_lengths(case.n, case.batch_size)))
            - 1
        )
        <= candidate_ops._get_num_sms()
    )
    if candidate_small_segmented_generic:
        return "_welmv4_inplace_rope_kernel_npu"
    if candidate_head_parallel and (
        case.family == "contiguous_mirror"
        and case.n >= threshold_all
        and case.n % block == 0
    ):
        return "_welmv4_inplace_rope_head_parallel_mirror_kernel_npu"
    if (
        not case.is_mirror
        and case.family != "segmented_prefill"
        and blocked
        and candidate_head_parallel
        and case.n % block == 0
    ):
        return "_welmv4_inplace_rope_head_parallel_prefill_kernel_npu"
    if case.family == "contiguous_mirror" and supported and case.n >= threshold_all:
        return "_welmv4_inplace_rope_contiguous_mirror_kernel_npu"
    if case.family == "segmented_mirror" and supported:
        return "_welmv4_inplace_rope_segmented_mirror_kernel_npu"
    if not case.is_mirror and blocked:
        if case.family == "segmented_prefill":
            return "_welmv4_inplace_rope_segmented_prefill_kernel_npu"
        if case.positions_are_contiguous:
            return "_welmv4_inplace_rope_contiguous_prefill_kernel_npu"
        return "_welmv4_inplace_rope_prefill_kernel_npu"
    return "_welmv4_inplace_rope_kernel_npu"


def case_record(case: RopeCase) -> dict[str, object]:
    return {
        "case": case.name,
        "workload_id": case.workload_id,
        "family": case.family,
        "topology": case.topology,
        "dp_attention_size": case.dp_attention_size,
        "context_length": case.context_length,
        "speculative_width": case.speculative_width,
        "n": case.n,
        "batch_size": case.batch_size,
        "q_rows": case.q_rows,
        "local_q_heads": case.local_q_heads,
        "local_kv_heads": case.local_kv_heads,
        "rotated_values": case.rotated_values,
    }


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    return float(delta.max().item()), float(delta.mean().item())


def run_correctness(
    harness: Harness,
    cases: Sequence[RopeCase],
    *,
    record_type: str,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    ref_atol = float(VALIDATION["reference_atol"])
    ref_rtol = float(VALIDATION["reference_rtol"])
    pair_atol = float(VALIDATION["baseline_candidate_atol"])
    pair_rtol = float(VALIDATION["baseline_candidate_rtol"])
    for index, case in enumerate(cases, 1):
        print(f"[{record_type} {index}/{len(cases)}] {case.name}", flush=True)
        inputs = harness.make_inputs(case, reference=True)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        kernels: dict[str, str] = {}
        for provider in ("baseline", "candidate"):
            bound = harness.bind(provider, case, inputs)
            bound.launch()
            torch_npu.npu.synchronize()
            outputs[provider] = (inputs.query.cpu(), inputs.key.cpu())
            kernels[provider] = bound.kernel_name

        assert inputs.expected_q is not None and inputs.expected_k is not None
        baseline_q_ok = torch.allclose(
            outputs["baseline"][0].float(), inputs.expected_q.float(),
            atol=ref_atol, rtol=ref_rtol,
        )
        baseline_k_ok = torch.allclose(
            outputs["baseline"][1].float(), inputs.expected_k.float(),
            atol=ref_atol, rtol=ref_rtol,
        )
        candidate_q_ok = torch.allclose(
            outputs["candidate"][0].float(), inputs.expected_q.float(),
            atol=ref_atol, rtol=ref_rtol,
        )
        candidate_k_ok = torch.allclose(
            outputs["candidate"][1].float(), inputs.expected_k.float(),
            atol=ref_atol, rtol=ref_rtol,
        )
        pair_q_ok = torch.allclose(
            outputs["candidate"][0].float(), outputs["baseline"][0].float(),
            atol=pair_atol, rtol=pair_rtol,
        )
        pair_k_ok = torch.allclose(
            outputs["candidate"][1].float(), outputs["baseline"][1].float(),
            atol=pair_atol, rtol=pair_rtol,
        )
        passed = all(
            (baseline_q_ok, baseline_k_ok, candidate_q_ok, candidate_k_ok,
             pair_q_ok, pair_k_ok)
        )
        failures += int(not passed)
        bq_max, bq_mean = tensor_error(outputs["baseline"][0], inputs.expected_q)
        bk_max, bk_mean = tensor_error(outputs["baseline"][1], inputs.expected_k)
        cq_max, cq_mean = tensor_error(outputs["candidate"][0], inputs.expected_q)
        ck_max, ck_mean = tensor_error(outputs["candidate"][1], inputs.expected_k)
        rows.append(
            {
                **case_record(case),
                "record_type": record_type,
                "status": "PASS" if passed else "FAIL",
                "baseline_kernel": kernels["baseline"],
                "candidate_kernel": kernels["candidate"],
                "baseline_q_vs_reference": baseline_q_ok,
                "baseline_k_vs_reference": baseline_k_ok,
                "candidate_q_vs_reference": candidate_q_ok,
                "candidate_k_vs_reference": candidate_k_ok,
                "candidate_q_vs_baseline": pair_q_ok,
                "candidate_k_vs_baseline": pair_k_ok,
                "baseline_q_max_abs": bq_max,
                "baseline_q_mean_abs": bq_mean,
                "baseline_k_max_abs": bk_max,
                "baseline_k_mean_abs": bk_mean,
                "candidate_q_max_abs": cq_max,
                "candidate_q_mean_abs": cq_mean,
                "candidate_k_max_abs": ck_max,
                "candidate_k_mean_abs": ck_mean,
            }
        )
        del inputs, outputs
        gc.collect()
    return rows, failures


def _parse_msprof_durations(
    output_dir: Path,
    stdout: str,
    kernel_name: str,
) -> tuple[list[float], list[str], str | None]:
    indexed: list[tuple[int, float]] = []
    names: list[str] = []
    launch_id: int | None = None
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if "Start analyze kernel" in line and "name is:" in line:
            try:
                launch_id = int(line.rpartition("name is:")[2].strip())
            except ValueError:
                launch_id = None
            continue
        if not line.strip().startswith("Op Name:"):
            continue
        name = line.partition(":")[2].strip()
        names.append(name)
        for detail in lines[index + 1 : index + 8]:
            if detail.strip().startswith("Task Duration(us):"):
                if kernel_name.lower() in name.lower():
                    if launch_id is None:
                        return [], names, "matching task has no numeric launch id"
                    try:
                        indexed.append(
                            (launch_id, float(detail.partition(":")[2].strip()))
                        )
                    except ValueError:
                        pass
                break
    if indexed:
        ids = sorted(item[0] for item in indexed)
        if ids != list(range(len(indexed))):
            return [], names, f"launch ids are not contiguous: {ids[:200]}"
        indexed.sort(key=lambda item: item[0])
        return [item[1] for item in indexed], names, None
    csv_files = sorted(output_dir.rglob("*.csv"))
    if csv_files:
        return [], names, "stdout had no ordered task durations; CSV is non-authoritative"
    return [], names, "no matching task durations found"


def run_msprof_child(
    harness: Harness,
    cases: Sequence[RopeCase],
    provider: str,
    kernel_name: str,
    warmup: int,
    samples: int,
) -> None:
    for case in cases:
        inputs = harness.make_inputs(case, reference=False)
        bound = harness.bind(provider, case, inputs)
        if bound.kernel_name != kernel_name:
            raise RuntimeError(
                f"kernel mismatch for {case.name}: {bound.kernel_name} != {kernel_name}"
            )
        for _ in range(warmup + samples):
            bound.launch()
        torch_npu.npu.synchronize()
        del inputs, bound
        gc.collect()


def capture_msprof(
    harness: Harness,
    cases: Sequence[RopeCase],
    output_dir: Path,
    suite: str,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    warmup = int(VALIDATION["msprof_warmup_per_case"])
    samples = int(VALIDATION["msprof_samples_per_case"])
    grouped: dict[tuple[str, str], list[RopeCase]] = {}
    for provider in ("baseline", "candidate"):
        for case in cases:
            grouped.setdefault((provider, primary_kernel_name(provider, case)), []).append(case)

    for (provider, kernel_name), group in grouped.items():
        print(
            f"[msprof] provider={provider}, kernel={kernel_name}, cases={len(group)}",
            flush=True,
        )
        launches = len(group) * (warmup + samples)
        with tempfile.TemporaryDirectory(prefix="welm_rope_msprof_") as tmp:
            command = [
                "msprof", "op", "--warm-up=0",
                f"--launch-count={launches}",
                f"--kernel-name={kernel_name}",
                f"--output={tmp}",
                sys.executable, str(Path(__file__).resolve()),
                "--suite", suite,
                "--device", str(harness.device),
                "--msprof-batch-provider", provider,
                "--msprof-batch-kernel", kernel_name,
                "--msprof-batch-warmup", str(warmup),
                "--msprof-batch-samples", str(samples),
            ]
            for case in group:
                command.extend(("--case-name", case.name))
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    check=False,
                )
                stdout = result.stdout or ""
                returncode = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                stdout = traceback.format_exc()
                returncode = -1
                parse_error = repr(exc)
                durations: list[float] = []
                names: list[str] = []
            else:
                durations, names, parse_error = _parse_msprof_durations(
                    Path(tmp), stdout, kernel_name
                )

            artifact = output_dir / "msprof" / provider / f"{kernel_name}.log.gz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(artifact, "wt", encoding="utf-8", compresslevel=9) as handle:
                handle.write(stdout)
            if returncode != 0 or parse_error or len(durations) != launches:
                failures += len(group)
                detail = (
                    f"returncode={returncode}, expected={launches}, got={len(durations)}, "
                    f"parse_error={parse_error!r}, discovered={sorted(set(names))[:100]}\n"
                    + stdout[-20000:]
                )
                for case in group:
                    rows.append(
                        {
                            **case_record(case),
                            "record_type": "msprof_task_duration",
                            "provider": provider,
                            "kernel_name": kernel_name,
                            "status": "ERROR",
                            "capture_log_tail": detail,
                            "artifact": str(artifact.relative_to(output_dir)),
                        }
                    )
                continue

            span = warmup + samples
            for index, case in enumerate(group):
                measured = durations[index * span + warmup : (index + 1) * span]
                rows.append(
                    {
                        **case_record(case),
                        "record_type": "msprof_task_duration",
                        "provider": provider,
                        "kernel_name": kernel_name,
                        "timing_authority": "msprof_op_task_duration_acceptance",
                        "status": "MEASURED",
                        "sample_count": len(measured),
                        "task_min_us": min(measured),
                        "task_p50_us": statistics.median(measured),
                        "task_mean_us": statistics.fmean(measured),
                        "task_max_us": max(measured),
                        "artifact": str(artifact.relative_to(output_dir)),
                    }
                )
    return rows, failures


def evaluate_performance(
    cases: Sequence[RopeCase],
    msprof_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], int, dict[str, object]]:
    baseline_snapshot = sha256_lf_file(BASELINE_PATH) == sha256_lf_file(
        CANDIDATE_PATH
    )
    measured: dict[tuple[str, str], dict[str, object]] = {
        (str(row["case"]), str(row["provider"])): row
        for row in msprof_rows
        if row.get("status") == "MEASURED"
    }
    rows: list[dict[str, object]] = []
    regressions = 0
    minimum_speedup = float(VALIDATION["minimum_case_speedup"])
    minimum_control_speedup = float(
        VALIDATION.get("minimum_control_layout_speedup", minimum_speedup)
    )
    candidate_costs: dict[tuple[str, str], float] = {}
    for case in cases:
        baseline = measured.get((case.name, "baseline"))
        candidate = measured.get((case.name, "candidate"))
        if baseline is None or candidate is None:
            regressions += 1
            rows.append(
                {**case_record(case), "status": "ERROR", "reason": "missing msprof measurement"}
            )
            continue
        baseline_us = float(baseline["task_p50_us"])
        candidate_us = float(candidate["task_p50_us"])
        speedup = baseline_us / candidate_us
        effective_minimum_speedup = (
            minimum_control_speedup if case.topology == "tp4" else minimum_speedup
        )
        passed = baseline_snapshot or speedup >= effective_minimum_speedup
        regressions += int(not passed)
        normalized = candidate_us / case.rotated_values
        candidate_costs[(case.workload_id, case.topology)] = normalized
        rows.append(
            {
                **case_record(case),
                "status": (
                    "BASELINE_SNAPSHOT"
                    if baseline_snapshot
                    else ("PASS" if passed else "PERF_REGRESSION")
                ),
                "baseline_kernel": baseline["kernel_name"],
                "candidate_kernel": candidate["kernel_name"],
                "baseline_p50_us": baseline_us,
                "candidate_p50_us": candidate_us,
                "speedup_vs_baseline": speedup,
                "candidate_us_per_rotated_value": normalized,
                "minimum_case_speedup": minimum_speedup,
                "effective_minimum_case_speedup": effective_minimum_speedup,
            }
        )

    max_ratio = float(VALIDATION["maximum_normalized_cost_ratio_vs_tp4"])
    efficiency_failures = 0
    for row in rows:
        if row.get("status") == "ERROR":
            continue
        key = (str(row["workload_id"]), "tp4")
        baseline_cost = candidate_costs.get(key)
        current_cost = candidate_costs.get((str(row["workload_id"]), str(row["topology"])))
        if baseline_cost is None or current_cost is None:
            continue
        ratio = current_cost / baseline_cost
        row["normalized_cost_ratio_vs_tp4"] = ratio
        row["maximum_normalized_cost_ratio_vs_tp4"] = max_ratio
        efficiency_ok = ratio <= max_ratio
        row["normalized_efficiency_pass"] = efficiency_ok
        if not efficiency_ok:
            efficiency_failures += 1
        if not baseline_snapshot and not efficiency_ok and row["status"] == "PASS":
            row["status"] = "PERF_REGRESSION"
            regressions += 1

    speedups = [float(row["speedup_vs_baseline"]) for row in rows if "speedup_vs_baseline" in row]
    geomean = (
        math.exp(statistics.fmean(math.log(value) for value in speedups))
        if speedups else 0.0
    )
    summary = {
        "timing_authority": "msprof_op_task_duration_acceptance",
        "case_count": len(cases),
        "baseline_snapshot": baseline_snapshot,
        "regression_count": regressions,
        "normalized_efficiency_failure_count": efficiency_failures,
        "speedup_geomean": geomean,
    }
    return rows, regressions, summary


def capture_ir(
    harness: Harness,
    cases: Sequence[RopeCase],
    output_dir: Path,
    suite: str,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    max_bytes = int(VALIDATION["max_artifact_bytes"])
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="welm_rope_ir_") as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "BENCH_PYTHON": sys.executable,
                    "IR_OUTPUT_DIR": tmp,
                    "BISHENGIR_TARGET": harness.device_name,
                }
            )
            command = [
                "bash", str(IR_CAPTURE_SCRIPT), str(Path(__file__).resolve()),
                "--suite", suite,
                "--device", str(harness.device),
                "--compile-only-provider", "candidate",
                "--case-name", case.name,
            ]
            print(f"[ir] {case.name}", flush=True)
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
                log_tail = (result.stdout or "")[-20000:]
                paths = sorted(Path(tmp).glob("*.mlir"))
                ok = result.returncode == 0 and bool(paths)
            except (OSError, subprocess.TimeoutExpired):
                log_tail = traceback.format_exc()[-20000:]
                paths = []
                ok = False
            if not ok:
                failures += 1
                rows.append(
                    {**case_record(case), "record_type": "compiler_ir", "status": "ERROR", "capture_log_tail": log_tail}
                )
                continue
            for source in paths:
                destination = output_dir / "ir" / case.topology / case.name / f"{source.name}.gz"
                try:
                    gzip_copy(source, destination, max_bytes)
                except RuntimeError as exc:
                    failures += 1
                    rows.append(
                        {**case_record(case), "record_type": "compiler_ir", "status": "SKIPPED_TOO_LARGE", "capture_log_tail": str(exc)}
                    )
                else:
                    rows.append(
                        {**case_record(case), "record_type": "compiler_ir", "status": "CAPTURED", "artifact": str(destination.relative_to(output_dir)), "uncompressed_size_bytes": source.stat().st_size}
                    )
    return rows, failures


def capture_profiles(
    harness: Harness,
    cases: Sequence[RopeCase],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    max_bytes = int(VALIDATION["max_artifact_bytes"])
    wanted_names = {
        "kernel_details.csv", "operator_details.csv", "op_statistic.csv",
        "step_trace_time.csv", "trace_view.json",
    }
    metrics = (
        ("pipe_utilization", torch_npu.profiler.AiCMetrics.PipeUtilization),
        ("memory", torch_npu.profiler.AiCMetrics.Memory),
    )
    for case in cases:
        inputs = harness.make_inputs(case, reference=False)
        bound = harness.bind("candidate", case, inputs)
        for _ in range(int(VALIDATION["profile_warmup"])):
            bound.launch()
        torch_npu.npu.synchronize()
        for metric_name, metric in metrics:
            print(f"[profile:{metric_name}] {case.name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="welm_rope_profile_") as tmp:
                try:
                    experimental = torch_npu.profiler._ExperimentalConfig(
                        export_type=[torch_npu.profiler.ExportType.Text],
                        aic_metrics=metric,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                        l2_cache=True,
                        data_simplification=False,
                    )
                    with torch_npu.profiler.profile(
                        activities=[
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU,
                        ],
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(tmp),
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=False,
                        with_flops=False,
                        with_modules=False,
                        experimental_config=experimental,
                    ):
                        for _ in range(int(VALIDATION["profile_iterations"])):
                            bound.launch()
                        torch_npu.npu.synchronize()
                except Exception:
                    failures += 1
                    rows.append(
                        {**case_record(case), "record_type": "profile", "profile_metric": metric_name, "status": "ERROR", "capture_log_tail": traceback.format_exc()[-20000:]}
                    )
                    continue
                sources = sorted(
                    path for path in Path(tmp).rglob("*")
                    if path.is_file()
                    and (path.name in wanted_names or path.name.startswith("l2_cache") or path.name.startswith("profiler_info"))
                )
                if not sources:
                    failures += 1
                    rows.append(
                        {**case_record(case), "record_type": "profile", "profile_metric": metric_name, "status": "ERROR", "capture_log_tail": "profiler emitted no selected artifact"}
                    )
                for source in sources:
                    relative = str(source.relative_to(tmp)).replace(os.sep, "__")
                    destination = output_dir / "profile" / case.topology / case.name / metric_name / f"{relative}.gz"
                    try:
                        gzip_copy(source, destination, max_bytes)
                    except RuntimeError as exc:
                        failures += 1
                        rows.append(
                            {**case_record(case), "record_type": "profile", "profile_metric": metric_name, "status": "SKIPPED_TOO_LARGE", "capture_log_tail": str(exc)}
                        )
                    else:
                        rows.append(
                            {**case_record(case), "record_type": "profile", "profile_metric": metric_name, "status": "CAPTURED", "artifact": str(destination.relative_to(output_dir)), "uncompressed_size_bytes": source.stat().st_size}
                        )
        del inputs, bound
        gc.collect()
    return rows, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--suite", choices=tuple(CONFIG["suites"]), default="smoke")
    parser.add_argument("--mode", choices=("both", "correctness", "performance"), default="both")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--case-name", action="append", default=[])
    parser.add_argument("--capture-ir", choices=("on", "off"), default="off")
    parser.add_argument("--capture-profile", choices=("on", "off"), default="off")
    parser.add_argument("--capture-msprof-op", choices=("on", "off"), default="off")
    parser.add_argument("--compile-only-provider", choices=("", "baseline", "candidate"), default="")
    parser.add_argument("--msprof-batch-provider", choices=("", "baseline", "candidate"), default="")
    parser.add_argument("--msprof-batch-kernel", default="")
    parser.add_argument("--msprof-batch-warmup", type=int, default=1)
    parser.add_argument("--msprof-batch-samples", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    requested_suite = args.suite
    candidate_is_baseline = sha256_lf_file(BASELINE_PATH) == sha256_lf_file(
        CANDIDATE_PATH
    )
    if (
        args.suite == "remote"
        and not candidate_is_baseline
        and os.environ.get("WELMV4_ROPE_FORCE_FULL", "0") != "1"
    ):
        # A long-running monitor may still pass the historical ``remote``
        # argument after pulling a new candidate. Fresh benchmark children
        # select the bounded iteration gate automatically; set the environment
        # variable only for periodic/final full validation.
        args.suite = "iteration"
        print(
            "candidate differs from baseline: using iteration suite "
            "(set WELMV4_ROPE_FORCE_FULL=1 for the full remote suite)",
            flush=True,
        )
    audit_frozen_baseline()
    harness = Harness(args.device, args.seed)
    all_cases = select_named(all_suite_cases(args.suite), args.case_name)

    if args.compile_only_provider:
        if len(all_cases) != 1:
            raise ValueError("compile-only mode requires exactly one --case-name")
        inputs = harness.make_inputs(all_cases[0], reference=False)
        bound = harness.bind(args.compile_only_provider, all_cases[0], inputs)
        bound.launch()
        torch_npu.npu.synchronize()
        return 0

    if args.msprof_batch_provider:
        if not args.msprof_batch_kernel or not args.case_name:
            raise ValueError("msprof child requires case names and kernel name")
        run_msprof_child(
            harness, all_cases, args.msprof_batch_provider,
            args.msprof_batch_kernel, args.msprof_batch_warmup,
            args.msprof_batch_samples,
        )
        return 0

    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "welmv4_rope_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    phases: dict[str, object] = {}
    correctness_failures = 0
    performance_regressions = 0
    capture_failures = 0

    if args.mode in ("both", "correctness"):
        cases = select_named(phase_cases(args.suite, "correctness"), args.case_name)
        rows, failures = run_correctness(harness, cases, record_type="correctness")
        write_csv(output_dir / "correctness.csv", rows)
        correctness_failures += failures
        phases["correctness"] = {"status": "PASS" if failures == 0 else "FAIL", "case_count": len(cases), "failure_count": failures}

    perf_cases = select_named(phase_cases(args.suite, "performance"), args.case_name)
    if args.mode in ("both", "performance") and correctness_failures == 0:
        validation_rows, failures = run_correctness(
            harness, perf_cases, record_type="performance_shape_validation"
        )
        write_csv(output_dir / "performance_shape_validation.csv", validation_rows)
        correctness_failures += failures
        phases["performance_shape_validation"] = {"status": "PASS" if failures == 0 else "FAIL", "case_count": len(perf_cases), "failure_count": failures}

        if failures == 0 and args.capture_msprof_op == "on":
            msprof_rows, msprof_failures = capture_msprof(
                harness, perf_cases, output_dir, args.suite
            )
            write_csv(output_dir / "msprof_task_duration.csv", msprof_rows)
            capture_failures += msprof_failures
            phases["msprof"] = {"status": "PASS" if msprof_failures == 0 else "ERROR", "failure_count": msprof_failures}
            perf_rows, performance_regressions, summary = evaluate_performance(
                perf_cases, msprof_rows
            )
            write_csv(output_dir / "performance.csv", perf_rows)
            phases["performance"] = {
                "status": "PASS" if performance_regressions == 0 and msprof_failures == 0 else "PERF_REGRESSION",
                **summary,
            }
        elif failures == 0:
            capture_failures += 1
            phases["performance"] = {"status": "ERROR", "reason": "performance requires --capture-msprof-op=on; NPU Event timing is forbidden"}

    if correctness_failures == 0 and args.capture_ir == "on":
        ir_cases = select_named(phase_cases(args.suite, "ir"), args.case_name)
        rows, failures = capture_ir(harness, ir_cases, output_dir, args.suite)
        write_csv(output_dir / "ir.csv", rows)
        capture_failures += failures
        phases["ir"] = {"status": "PASS" if failures == 0 else "ERROR", "failure_count": failures}

    if correctness_failures == 0 and args.capture_profile == "on":
        profile_cases = select_named(phase_cases(args.suite, "profile"), args.case_name)
        rows, failures = capture_profiles(harness, profile_cases, output_dir)
        write_csv(output_dir / "profile.csv", rows)
        capture_failures += failures
        phases["profile"] = {"status": "PASS" if failures == 0 else "ERROR", "failure_count": failures}

    if correctness_failures:
        status = "FAIL"
        returncode = 1
    elif capture_failures:
        status = "ERROR"
        returncode = 1
    elif performance_regressions:
        status = "PERF_REGRESSION"
        returncode = 2
    else:
        status = "PASS"
        returncode = 0

    manifest = {
        "schema_version": 1,
        "status": status,
        "created_at": utc_now(),
        "repository_head": repository_head(),
        "workspace": CONFIG["workspace"],
        "suite": args.suite,
        "requested_suite": requested_suite,
        "mode": args.mode,
        "device": str(harness.device),
        "timing_authority": "msprof_op_task_duration_acceptance",
        "npu_event_timing": False,
        "baseline_sha256": harness.baseline_hash,
        "candidate_sha256": harness.candidate_hash,
        "correctness_failure_count": correctness_failures,
        "performance_regression_count": performance_regressions,
        "capture_failure_count": capture_failures,
        "phases": phases,
    }
    write_json(output_dir / "result.json", manifest)
    print(f"result={status}; manifest={output_dir / 'result.json'}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
