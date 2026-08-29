#!/usr/bin/env python3
"""Standalone WeLMv4 Full/SWA paged-prefill Attention benchmark for Ascend.

The benchmark imports no NEWSGLANG package.  Its baseline and initial candidate
are byte-for-byte snapshots of the production ``sink_full_attention.py`` at the
commit recorded in ``workspace_config.json``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch_npu
import triton

import welmv4_prefill_attention_baseline as baseline_ops
import welmv4_prefill_attention_candidate as candidate_ops
from attention_contract import (
    BASELINE_PATH,
    CANDIDATE_PATH,
    CONFIG,
    DEFAULT_TP_SIZE,
    HEAD_DIM,
    M_MAX,
    M_MIN,
    PAGE_SIZE,
    SOFTMAX_SCALE,
    SWA_GLOBAL_WINDOW,
    SWA_LEFT_WINDOW,
    VALIDATION,
    AttentionCase,
    audit_frozen_baseline,
    find_case,
    make_manual_cases,
    parse_int_set,
    sha256_file,
    suite_cases,
)
from attention_inputs import AttentionInputs, make_inputs
from attention_reference import error_metrics, reference_prefill_attention


ROOT = Path(__file__).resolve().parent
IR_CAPTURE_SCRIPT = ROOT / "capture_welmv4_prefill_attention_ir.sh"
SCHEMA_VERSION = 1
DEFAULT_SEED = 20260829


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def file_record(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(root)).replace(os.sep, "/"),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def gzip_copy(
    source: Path,
    destination: Path,
    *,
    max_input_bytes: int | None = None,
) -> None:
    if max_input_bytes is not None and source.stat().st_size > max_input_bytes:
        raise ValueError(
            f"artifact exceeds {max_input_bytes} bytes: {source} "
            f"({source.stat().st_size} bytes)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, gzip.open(destination, "wb", compresslevel=9) as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)


@dataclass
class BoundLaunch:
    launch: Callable[[], torch.Tensor]
    host_prepare_submit_ms: float
    kernel_name: str


def primary_kernel_name(provider: str, case: AttentionCase) -> str:
    candidate_small_q = provider == "candidate" and case.max_q_len <= 4
    if case.attention == "full":
        return (
            "paged_prefill_small_q_grouped_kernel"
            if candidate_small_q
            else "paged_prefill_page_aggregation_kernel"
        )
    candidate_swa_small_q = candidate_small_q and (
        case.real_batch_size > 1 or case.max_q_len <= 2
    )
    if not candidate_swa_small_q:
        return "_swa_paged_prefill_aggregation_sink_kernel"
    return (
        "_swa_paged_prefill_single_q_grouped_sink_kernel"
        if case.max_q_len == 1
        else "_swa_paged_prefill_small_q_grouped_sink_kernel"
    )


@dataclass
class CapturedGraphBundle:
    """Own every tensor/object whose address can be retained by NPUGraph."""

    provider: str
    case: AttentionCase
    inputs: AttentionInputs
    bound: BoundLaunch
    graph: object
    graph_pool: object
    capture_stream: object
    static_output: torch.Tensor
    runtime_cu_q_lens: torch.Tensor
    runtime_kv_lens: torch.Tensor
    runtime_block_table: torch.Tensor


class Harness:
    def __init__(self, device: torch.device, seed: int, benchmark_base: str) -> None:
        self.device = device
        self.seed = seed
        self.benchmark_base = benchmark_base
        self.device_index = int(torch_npu.npu.current_device())
        properties = triton.runtime.driver.active.utils.get_device_properties(
            self.device_index
        )
        self.num_cube_cores = int(properties.get("num_aicore", -1))
        self.num_vector_cores = int(properties.get("num_vectorcore", -1))
        self.device_name = str(torch_npu.npu.get_device_name(self.device_index))
        self.commit = repository_head()
        self.baseline_hash = sha256_file(BASELINE_PATH)
        self.candidate_hash = sha256_file(CANDIDATE_PATH)
        # NEWSGLANG owns one persistent causal template on the backend.  Passing
        # None would make the copied public wrapper allocate a 1024x3072 mask
        # on every call and would no longer represent production latency.
        self.full_aux_mask = torch.ones(
            (1024, 1024 * 3), device=device, dtype=torch.bool
        ).tril_(diagonal=1024)
        self.graph_pool = None
        self.graph_capture_stream = None

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_base": self.benchmark_base,
            "workspace_commit": self.commit,
            "baseline_sha256": self.baseline_hash,
            "candidate_sha256": self.candidate_hash,
            "device": str(self.device),
            "device_name": self.device_name,
            "device_index": self.device_index,
            "num_cube_cores": self.num_cube_cores,
            "num_vector_cores": self.num_vector_cores,
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
            "triton_version": str(getattr(triton, "__version__", "unknown")),
            "cann_version": str(getattr(torch.version, "cann", "unknown")),
            "python_version": platform.python_version(),
            "seed": self.seed,
            "dtype": "bfloat16",
            "sink_dtype": "float32",
            "cache_layout": "native_NHD_then_page_head_token_dim_permute",
            "causal": True,
            "gqa_interleave": False,
        }

    def bind(
        self,
        provider: str,
        case: AttentionCase,
        inputs: AttentionInputs,
    ) -> BoundLaunch:
        for metadata_name, metadata in (
            ("runtime_cu_q_lens", inputs.runtime_cu_q_lens),
            ("runtime_kv_lens", inputs.runtime_kv_lens),
            ("block_table", inputs.block_table),
        ):
            if metadata.dtype != torch.int32 or not metadata.is_contiguous():
                raise RuntimeError(
                    f"{metadata_name} must match production contiguous int32 metadata; "
                    f"got dtype={metadata.dtype}, stride={metadata.stride()}"
                )
        if provider == "baseline":
            module = baseline_ops
        elif provider == "candidate":
            module = candidate_ops
        else:
            raise ValueError(f"unknown provider: {provider}")

        if case.attention == "full":
            prepare_start = time.perf_counter()
            schedule = module.paged_attention_prefill_prepare(
                inputs.capture_cu_q_lens_cpu,
                inputs.capture_kv_lens_cpu,
                case.local_num_q_heads,
                case.local_num_kv_heads,
                False,
                PAGE_SIZE,
                device=self.device,
            )
            host_prepare_submit_ms = (time.perf_counter() - prepare_start) * 1000.0

            def launch_full() -> torch.Tensor:
                return module.paged_attention_prefill_impl(
                    q=inputs.q,
                    key_cache=inputs.key_cache,
                    value_cache=inputs.value_cache,
                    cu_q_lens=inputs.runtime_cu_q_lens,
                    seqlens_kv=inputs.runtime_kv_lens,
                    block_tables=inputs.block_table,
                    gqa_interleave=False,
                    task_b=schedule[0],
                    task_q_block=schedule[1],
                    task_q_head=schedule[2],
                    core_task_offsets=schedule[3],
                    softmax_scale=SOFTMAX_SCALE,
                    aux_mask=self.full_aux_mask,
                    max_q_len=case.max_q_len,
                    sinks=inputs.sinks,
                )

            return BoundLaunch(
                launch_full,
                host_prepare_submit_ms,
                primary_kernel_name(provider, case),
            )

        swa_kwargs = {
            "q": inputs.q,
            "k_cache": inputs.key_cache,
            "v_cache": inputs.value_cache,
            "cu_q_lens": inputs.runtime_cu_q_lens,
            "kvlens": inputs.runtime_kv_lens,
            "block_table": inputs.block_table,
            "is_causal": True,
            "local_window_size": SWA_LEFT_WINDOW,
            "global_window_size": SWA_GLOBAL_WINDOW,
            "softmax_scale": SOFTMAX_SCALE,
            "gqa_interleave": False,
            "sinks": inputs.sinks,
        }
        if provider == "candidate":
            # The production backend already owns CPU query lengths. Passing
            # their maximum keeps dispatch shape-based and avoids a device
            # read/synchronization in the operator wrapper.
            swa_kwargs["max_q_len"] = case.max_q_len

        def launch_swa() -> torch.Tensor:
            return module.swa_paged_prefill_impl(
                **swa_kwargs,
            )

        return BoundLaunch(
            launch_swa,
            0.0,
            primary_kernel_name(provider, case),
        )


def common_case_record(
    harness: Harness,
    case: AttentionCase,
    inputs: AttentionInputs | None = None,
) -> dict[str, object]:
    record = {**harness.metadata(), **case.as_record()}
    record["runtime_q_lens"] = json.dumps(case.runtime_q_lens)
    record["runtime_kv_lens"] = json.dumps(case.runtime_kv_lens)
    record["capture_q_lens"] = json.dumps(case.capture_q_lens)
    record["capture_kv_lens"] = json.dumps(case.capture_kv_lens)
    if inputs is not None:
        record.update(
            {
                "q_stride": json.dumps(inputs.q.stride()),
                "kv_cache_stride": json.dumps(inputs.key_cache.stride()),
                "block_table_stride": json.dumps(inputs.block_table.stride()),
                "physical_pages": inputs.key_cache.shape[0],
            }
        )
    return record


def run_correctness(
    harness: Harness,
    cases: Sequence[AttentionCase],
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    reference_atol = float(VALIDATION["reference_atol"])
    reference_rtol = float(VALIDATION["reference_rtol"])
    pair_atol = float(VALIDATION["baseline_candidate_atol"])
    pair_rtol = float(VALIDATION["baseline_candidate_rtol"])

    for index, case in enumerate(cases, 1):
        print(f"[correctness {index}/{len(cases)}] {case.name}", flush=True)
        inputs = make_inputs(case, harness.device, seed=harness.seed)
        expected = reference_prefill_attention(
            attention=case.attention,
            q=inputs.q,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            runtime_q_lens=case.runtime_q_lens,
            runtime_kv_lens=case.runtime_kv_lens,
            block_table=inputs.block_table,
            sinks=inputs.sinks,
        )
        outputs: dict[str, torch.Tensor] = {}
        prepare_times: dict[str, float] = {}
        for provider in ("baseline", "candidate"):
            bound = harness.bind(provider, case, inputs)
            outputs[provider] = bound.launch()
            prepare_times[provider] = bound.host_prepare_submit_ms
        torch_npu.npu.synchronize()

        baseline_metrics = error_metrics(outputs["baseline"], expected)
        candidate_metrics = error_metrics(outputs["candidate"], expected)
        pair_metrics = error_metrics(outputs["candidate"], outputs["baseline"])
        baseline_ok = bool(
            torch.allclose(
                outputs["baseline"],
                expected,
                atol=reference_atol,
                rtol=reference_rtol,
            )
        )
        candidate_ok = bool(
            torch.allclose(
                outputs["candidate"],
                expected,
                atol=reference_atol,
                rtol=reference_rtol,
            )
        )
        pair_ok = bool(
            torch.allclose(
                outputs["candidate"],
                outputs["baseline"],
                atol=pair_atol,
                rtol=pair_rtol,
            )
        )
        passed = baseline_ok and candidate_ok and pair_ok
        failures += int(not passed)
        row = {
            **common_case_record(harness, case, inputs),
            "record_type": "correctness",
            "status": "PASS" if passed else "FAIL",
            "baseline_vs_reference": baseline_ok,
            "candidate_vs_reference": candidate_ok,
            "candidate_vs_baseline": pair_ok,
            "baseline_host_prepare_submit_ms": prepare_times["baseline"],
            "candidate_host_prepare_submit_ms": prepare_times["candidate"],
        }
        row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        row.update({f"candidate_{k}": v for k, v in candidate_metrics.items()})
        row.update({f"pair_{k}": v for k, v in pair_metrics.items()})
        rows.append(row)
        print(
            f"  {'PASS' if passed else 'FAIL'}: "
            f"candidate max_abs={candidate_metrics['max_abs_error']:.6g}, "
            f"pair max_abs={pair_metrics['max_abs_error']:.6g}",
            flush=True,
        )
        del expected, outputs, inputs
        gc.collect()
    return rows, failures


def _device_prefix_sum(values: Sequence[int], device: torch.device) -> torch.Tensor:
    cpu = torch.tensor(tuple(values), dtype=torch.int32, device="cpu")
    return torch.nn.functional.pad(
        torch.cumsum(cpu, dim=0, dtype=torch.int32), (1, 0)
    ).to(device)


def _capture_replay_one_provider(
    harness: Harness,
    provider: str,
    case: AttentionCase,
    inputs: AttentionInputs,
) -> tuple[torch.Tensor, float, int, CapturedGraphBundle]:
    """Capture one production-shaped wrapper and replay underfilled metadata.

    WeLM's all-Triton Graph path does not call ``NPUGraph.update``.  The graph
    retains device tensor addresses and replay copies current metadata into
    those tensors in place.  Start capture with a full Bcap x D bucket, then
    overwrite the same buffers with the configured real request count.
    """

    if not case.topology.startswith("graph_"):
        raise ValueError("NPUGraph capture requires a graph_d2/graph_d3 case")
    if inputs.q.shape[0] != case.q_buffer_rows:
        raise RuntimeError("Graph Q buffer does not match Bcap x D")
    if inputs.block_table.dtype != torch.int32 or not inputs.block_table.is_contiguous():
        raise RuntimeError("Graph block table must start as contiguous int32")
    normalized_table = inputs.block_table.to(dtype=torch.int32).contiguous()
    if normalized_table.data_ptr() != inputs.block_table.data_ptr():
        raise RuntimeError(
            "public wrapper would capture a temporary block-table address"
        )

    runtime_cu = inputs.runtime_cu_q_lens.clone()
    runtime_kv = inputs.runtime_kv_lens.clone()
    runtime_table = inputs.block_table.clone()
    capture_cu = _device_prefix_sum(case.capture_q_lens, harness.device)
    capture_kv = torch.tensor(
        case.capture_kv_lens, dtype=torch.int32, device=harness.device
    )

    inputs.runtime_cu_q_lens.copy_(capture_cu)
    inputs.runtime_kv_lens.copy_(capture_kv)
    # Capture with a deliberately different, but valid, page mapping.  Replay
    # must observe the in-place restoration below; otherwise reference checks
    # expose a wrapper-created temporary or stale Graph metadata.
    inputs.block_table.zero_()
    bound = harness.bind(provider, case, inputs)

    current_stream = torch.npu.current_stream()
    if harness.graph_capture_stream is None:
        harness.graph_capture_stream = torch.npu.Stream()
    capture_stream = harness.graph_capture_stream
    capture_stream.wait_stream(current_stream)
    with torch.npu.stream(capture_stream):
        for _ in range(2):
            bound.launch()
    torch_npu.npu.synchronize()

    if harness.graph_pool is None:
        harness.graph_pool = torch.npu.graph_pool_handle()
    graph_pool = harness.graph_pool
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        pool=graph_pool,
        stream=capture_stream,
        auto_dispatch_capture=True,
    ):
        static_output = bound.launch()
    torch_npu.npu.synchronize()

    bundle = CapturedGraphBundle(
        provider=provider,
        case=case,
        inputs=inputs,
        bound=bound,
        graph=graph,
        graph_pool=graph_pool,
        capture_stream=capture_stream,
        static_output=static_output,
        runtime_cu_q_lens=runtime_cu,
        runtime_kv_lens=runtime_kv,
        runtime_block_table=runtime_table,
    )
    output, replay_us, output_data_ptr = _replay_graph_bundle(bundle)
    return output, replay_us, output_data_ptr, bundle


def _replay_graph_bundle(
    bundle: CapturedGraphBundle,
) -> tuple[torch.Tensor, float, int]:
    """Replay while preserving captured addresses and refreshing live metadata."""

    bundle.inputs.runtime_cu_q_lens.copy_(bundle.runtime_cu_q_lens)
    bundle.inputs.runtime_kv_lens.copy_(bundle.runtime_kv_lens)
    bundle.inputs.block_table.copy_(bundle.runtime_block_table)
    bundle.static_output.fill_(7)
    torch_npu.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    bundle.graph.replay()
    end.record()
    torch_npu.npu.synchronize()
    replay_us = float(start.elapsed_time(end)) * 1000.0
    output = bundle.static_output.clone()
    torch_npu.npu.synchronize()
    return output, replay_us, int(bundle.static_output.data_ptr())


def run_graph_correctness(
    harness: Harness,
    cases: Sequence[AttentionCase],
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    reference_atol = float(VALIDATION["reference_atol"])
    reference_rtol = float(VALIDATION["reference_rtol"])
    pair_atol = float(VALIDATION["baseline_candidate_atol"])
    pair_rtol = float(VALIDATION["baseline_candidate_rtol"])

    initial_outputs: dict[tuple[str, str], torch.Tensor] = {}
    retained_outputs: dict[tuple[str, str], torch.Tensor] = {}
    initial_rows: dict[tuple[str, str], dict[str, object]] = {}
    retained_rows: dict[tuple[str, str], dict[str, object]] = {}

    # Capture every configured graph for one provider before replaying them in
    # reverse order. Production keeps many bucket/variant graphs alive at once;
    # this catches module-level mask/schedule caches that accidentally replace a
    # tensor whose device address is still referenced by an older graph.
    for provider in ("baseline", "candidate"):
        retained: list[tuple[CapturedGraphBundle, torch.Tensor]] = []
        for index, case in enumerate(cases, 1):
            print(
                f"[npu-graph:capture {provider} {index}/{len(cases)}] {case.name}",
                flush=True,
            )
            inputs = make_inputs(case, harness.device, seed=harness.seed)
            expected = reference_prefill_attention(
                attention=case.attention,
                q=inputs.q,
                key_cache=inputs.key_cache,
                value_cache=inputs.value_cache,
                runtime_q_lens=case.runtime_q_lens,
                runtime_kv_lens=case.runtime_kv_lens,
                block_table=inputs.block_table,
                sinks=inputs.sinks,
            )
            try:
                output, replay_us, output_data_ptr, bundle = (
                    _capture_replay_one_provider(harness, provider, case, inputs)
                )
                metrics = error_metrics(output, expected)
                reference_ok = bool(
                    torch.allclose(
                        output,
                        expected,
                        atol=reference_atol,
                        rtol=reference_rtol,
                    )
                )
                padding = output[case.m :]
                padding_nonzero = int(torch.count_nonzero(padding).item())
                passed = reference_ok and padding_nonzero == 0
                failures += int(not passed)
                key = (case.name, provider)
                initial_outputs[key] = output
                initial_rows[key] = {
                    **common_case_record(harness, case, inputs),
                    "record_type": "npu_graph_capture_replay",
                    "provider": provider,
                    "status": "PASS" if passed else "FAIL",
                    "reference_match": reference_ok,
                    "padding_nonzero_count": padding_nonzero,
                    "replay_latency_us": replay_us,
                    "static_output_data_ptr": output_data_ptr,
                    **metrics,
                }
                retained.append((bundle, expected))
            except Exception:
                failures += 1
                initial_rows[(case.name, provider)] = {
                    **common_case_record(harness, case, inputs),
                    "record_type": "npu_graph_capture_replay",
                    "provider": provider,
                    "status": "ERROR",
                    "capture_log_tail": traceback.format_exc()[-20000:],
                }
                del inputs, expected
                gc.collect()

        for replay_index, (bundle, expected) in enumerate(
            reversed(retained), 1
        ):
            case = bundle.case
            key = (case.name, provider)
            print(
                f"[npu-graph:retained-replay {provider} "
                f"{replay_index}/{len(retained)}] {case.name}",
                flush=True,
            )
            try:
                output, replay_us, output_data_ptr = _replay_graph_bundle(bundle)
                metrics = error_metrics(output, expected)
                reference_ok = bool(
                    torch.allclose(
                        output,
                        expected,
                        atol=reference_atol,
                        rtol=reference_rtol,
                    )
                )
                padding_nonzero = int(
                    torch.count_nonzero(output[case.m :]).item()
                )
                passed = reference_ok and padding_nonzero == 0
                failures += int(not passed)
                retained_outputs[key] = output
                retained_rows[key] = {
                    **common_case_record(harness, case, bundle.inputs),
                    "record_type": "npu_graph_retained_reverse_replay",
                    "provider": provider,
                    "status": "PASS" if passed else "FAIL",
                    "reference_match": reference_ok,
                    "padding_nonzero_count": padding_nonzero,
                    "replay_latency_us": replay_us,
                    "static_output_data_ptr": output_data_ptr,
                    "retained_graph_count": len(retained),
                    **metrics,
                }
            except Exception:
                failures += 1
                retained_rows[key] = {
                    **common_case_record(harness, case, bundle.inputs),
                    "record_type": "npu_graph_retained_reverse_replay",
                    "provider": provider,
                    "status": "ERROR",
                    "retained_graph_count": len(retained),
                    "capture_log_tail": traceback.format_exc()[-20000:],
                }

        # All graph/input/schedule/mask owners stayed live through the reverse
        # replay. Release this provider's potentially large long-context caches
        # before capturing the other provider.
        bundle = None
        expected = None
        inputs = None
        output = None
        del retained
        gc.collect()

    for case in cases:
        for replay_kind, output_map, row_map in (
            ("initial", initial_outputs, initial_rows),
            ("retained", retained_outputs, retained_rows),
        ):
            baseline_key = (case.name, "baseline")
            candidate_key = (case.name, "candidate")
            if baseline_key not in output_map or candidate_key not in output_map:
                continue
            pair_metrics = error_metrics(
                output_map[candidate_key], output_map[baseline_key]
            )
            pair_ok = bool(
                torch.allclose(
                    output_map[candidate_key],
                    output_map[baseline_key],
                    atol=pair_atol,
                    rtol=pair_rtol,
                )
            )
            if not pair_ok:
                failures += 1
            for key in (baseline_key, candidate_key):
                row = row_map[key]
                row["candidate_vs_baseline"] = pair_ok
                row["pair_replay_kind"] = replay_kind
                row.update(
                    {f"pair_{metric}": value for metric, value in pair_metrics.items()}
                )
                if not pair_ok and row["status"] == "PASS":
                    row["status"] = "FAIL"

    for case in cases:
        for row_map in (initial_rows, retained_rows):
            for provider in ("baseline", "candidate"):
                row = row_map.get((case.name, provider))
                if row is not None:
                    rows.append(row)
        print(
            "  "
            + ", ".join(
                f"{provider}="
                f"{initial_rows.get((case.name, provider), {}).get('status', 'MISSING')}"
                f"/retained="
                f"{retained_rows.get((case.name, provider), {}).get('status', 'MISSING')}"
                for provider in ("baseline", "candidate")
            ),
            flush=True,
        )

    del initial_outputs, retained_outputs
    gc.collect()
    return rows, failures


def _measure_event_group(
    launch: Callable[[], torch.Tensor], iterations: int
) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iterations


def run_performance(
    harness: Harness,
    cases: Sequence[AttentionCase],
) -> tuple[list[dict[str, object]], int, int, dict[str, object]]:
    rows: list[dict[str, object]] = []
    regressions = 0
    validation_failures = 0
    warmup = int(VALIDATION["latency_warmup"])
    groups = int(VALIDATION["latency_groups"])
    iterations = int(VALIDATION["latency_iterations_per_group"])
    minimum_speedup = float(VALIDATION["minimum_speedup"])
    minimum_case_speedup = float(VALIDATION.get("minimum_case_speedup", 1.0))
    gain_topologies = set(VALIDATION.get("gain_topologies", []))
    gain_speedups: list[float] = []
    event_timing_enabled = bool(
        VALIDATION.get("enable_npu_event_timing", False)
    )

    for index, case in enumerate(cases, 1):
        print(f"[performance {index}/{len(cases)}] {case.name}", flush=True)
        inputs = make_inputs(case, harness.device, seed=harness.seed)
        bound = {
            provider: harness.bind(provider, case, inputs)
            for provider in ("baseline", "candidate")
        }
        # Performance-only runs must not let a fast but wrong candidate bypass
        # correctness. Validate the exact timed shape before any warmup/event.
        expected = reference_prefill_attention(
            attention=case.attention,
            q=inputs.q,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            runtime_q_lens=case.runtime_q_lens,
            runtime_kv_lens=case.runtime_kv_lens,
            block_table=inputs.block_table,
            sinks=inputs.sinks,
        )
        probe = {
            provider: bound[provider].launch()
            for provider in ("baseline", "candidate")
        }
        torch_npu.npu.synchronize()
        reference_atol = float(VALIDATION["reference_atol"])
        reference_rtol = float(VALIDATION["reference_rtol"])
        pair_atol = float(VALIDATION["baseline_candidate_atol"])
        pair_rtol = float(VALIDATION["baseline_candidate_rtol"])
        baseline_ok = bool(
            torch.allclose(
                probe["baseline"], expected, atol=reference_atol, rtol=reference_rtol
            )
        )
        candidate_ok = bool(
            torch.allclose(
                probe["candidate"], expected, atol=reference_atol, rtol=reference_rtol
            )
        )
        pair_ok = bool(
            torch.allclose(
                probe["candidate"],
                probe["baseline"],
                atol=pair_atol,
                rtol=pair_rtol,
            )
        )
        if not (baseline_ok and candidate_ok and pair_ok):
            validation_failures += 1
            for provider in ("baseline", "candidate"):
                rows.append(
                    {
                        **common_case_record(harness, case, inputs),
                        "record_type": "logical_wrapper_latency",
                        "timing_authority": "not_measured_after_validation_failure",
                        "provider": provider,
                        "status": "FAIL",
                        "baseline_vs_reference": baseline_ok,
                        "candidate_vs_reference": candidate_ok,
                        "candidate_vs_baseline": pair_ok,
                    }
                )
            print("  FAIL: timed-shape correctness gate", flush=True)
            del expected, probe, bound, inputs
            gc.collect()
            continue
        del expected, probe
        if not event_timing_enabled:
            for provider in ("baseline", "candidate"):
                rows.append(
                    {
                        **common_case_record(harness, case, inputs),
                        "record_type": "performance_shape_validation",
                        "timing_authority": "msprof_op_task_duration_acceptance",
                        "provider": provider,
                        "status": "VALIDATED_FOR_MSPROF",
                        "host_prepare_submit_ms": bound[provider].host_prepare_submit_ms,
                        "kernel_name": bound[provider].kernel_name,
                    }
                )
            del bound, inputs
            gc.collect()
            continue
        for provider in ("baseline", "candidate"):
            for _ in range(warmup):
                bound[provider].launch()
        torch_npu.npu.synchronize()

        samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
        # Alternate provider order so temperature/drift does not always favor
        # one side. Each sample is total NPU timeline time for a logical public
        # wrapper call, so future helper-kernel splits remain accounted for.
        for group in range(groups):
            order = (
                ("baseline", "candidate")
                if group % 2 == 0
                else ("candidate", "baseline")
            )
            for provider in order:
                samples[provider].append(
                    _measure_event_group(bound[provider].launch, iterations)
                )

        medians = {
            provider: statistics.median(values)
            for provider, values in samples.items()
        }
        speedup = medians["baseline"] / medians["candidate"]
        accepted = speedup >= minimum_case_speedup
        if not gain_topologies or case.topology in gain_topologies:
            gain_speedups.append(speedup)
        regressions += int(not accepted)
        decode_targets = VALIDATION.get("decode_reference_targets_us", {})
        batch_targets = decode_targets.get(str(case.real_batch_size), {})
        decode_target_us = batch_targets.get(case.attention)
        for provider in ("baseline", "candidate"):
            values = samples[provider]
            mean = statistics.fmean(values)
            rows.append(
                {
                    **common_case_record(harness, case, inputs),
                    "record_type": "logical_wrapper_latency",
                    "timing_authority": "acceptance",
                    "provider": provider,
                    "status": "PASS" if accepted else "PERF_REGRESSION",
                    "sample_count": len(values),
                    "iterations_per_sample": iterations,
                    "latency_min_us": min(values),
                    "latency_p50_us": medians[provider],
                    "latency_mean_us": mean,
                    "latency_max_us": max(values),
                    "latency_cv": (
                        statistics.pstdev(values) / mean if mean else 0.0
                    ),
                    "speedup_vs_baseline": speedup if provider == "candidate" else 1.0,
                    "minimum_case_speedup": minimum_case_speedup,
                    "decode_reference_target_us": decode_target_us,
                    "decode_target_ratio": (
                        medians[provider] / float(decode_target_us)
                        if decode_target_us is not None
                        else None
                    ),
                    "host_prepare_submit_ms": bound[provider].host_prepare_submit_ms,
                    "kernel_name": bound[provider].kernel_name,
                }
            )
        print(
            f"  baseline={medians['baseline']:.3f} us, "
            f"candidate={medians['candidate']:.3f} us, speedup={speedup:.4f}, "
            f"{'PASS' if accepted else 'PERF_REGRESSION'}",
            flush=True,
        )
        del bound, inputs
        gc.collect()
    gain_geomean = (
        statistics.geometric_mean(gain_speedups) if gain_speedups else 1.0
    )
    gain_gate_passed = (
        gain_geomean >= minimum_speedup if event_timing_enabled else True
    )
    regressions += int(not gain_gate_passed)
    print(
        f"[performance gate] gain_geomean={gain_geomean:.4f}, "
        f"minimum={minimum_speedup:.4f}, "
        f"{'PASS' if gain_gate_passed else 'PERF_REGRESSION'}",
        flush=True,
    )
    summary = {
        "gain_case_count": len(gain_speedups),
        "gain_geomean_speedup": gain_geomean,
        "minimum_gain_speedup": minimum_speedup,
        "minimum_case_speedup": minimum_case_speedup,
        "gain_gate_passed": gain_gate_passed,
        "timing_authority": (
            "npu_event" if event_timing_enabled else "msprof_op_task_duration"
        ),
    }
    return rows, regressions, validation_failures, summary


def run_compile_only(
    harness: Harness,
    case: AttentionCase,
    provider: str,
    iterations: int,
) -> str:
    inputs = make_inputs(case, harness.device, seed=harness.seed)
    bound = harness.bind(provider, case, inputs)
    for _ in range(iterations):
        bound.launch()
    torch_npu.npu.synchronize()
    print(
        f"compile-only launch complete: provider={provider}, case={case.name}, "
        f"kernel={bound.kernel_name}, iterations={iterations}",
        flush=True,
    )
    return bound.kernel_name


def capture_ir(
    harness: Harness,
    cases: Sequence[AttentionCase],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    max_bytes = int(VALIDATION["max_artifact_bytes"])
    if not IR_CAPTURE_SCRIPT.is_file():
        raise FileNotFoundError(IR_CAPTURE_SCRIPT)

    for case in cases:
        destination = output_dir / "ir" / case.attention / case.name
        with tempfile.TemporaryDirectory(prefix=f"welm_attn_ir_{case.attention}_") as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "BENCH_PYTHON": sys.executable,
                    "IR_OUTPUT_DIR": tmp,
                    "BISHENGIR_TARGET": harness.device_name,
                }
            )
            command = [
                "bash",
                str(IR_CAPTURE_SCRIPT),
                str(Path(__file__).resolve()),
                "--compile-only-provider",
                "candidate",
                "--case-name",
                case.name,
                "--device",
                str(harness.device),
                "--tp-size",
                str(case.tp_size),
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
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures += 1
                rows.append(
                    {
                        **common_case_record(harness, case),
                        "record_type": "compiler_ir",
                        "status": "ERROR",
                        "capture_log_tail": repr(exc),
                    }
                )
                continue
            paths = sorted(Path(tmp).glob("*.mlir"))
            status = "CAPTURED" if result.returncode == 0 and paths else "ERROR"
            failures += int(status == "ERROR")
            common = {
                **common_case_record(harness, case),
                "record_type": "compiler_ir",
                "status": status,
                "capture_returncode": result.returncode,
                "capture_log_tail": (result.stdout or "")[-20000:],
            }
            if not paths:
                rows.append(common)
                continue
            for source in paths:
                destination_file = destination / f"{source.name}.gz"
                if source.stat().st_size > max_bytes:
                    failures += 1
                    rows.append(
                        {
                            **common,
                            "status": "SKIPPED_TOO_LARGE",
                            "source_name": source.name,
                            "uncompressed_size_bytes": source.stat().st_size,
                            "max_artifact_bytes": max_bytes,
                        }
                    )
                    continue
                gzip_copy(source, destination_file, max_input_bytes=max_bytes)
                rows.append(
                    {
                        **common,
                        **file_record(destination_file, output_dir),
                        "uncompressed_size_bytes": source.stat().st_size,
                    }
                )
    return rows, failures


def capture_profiles(
    harness: Harness,
    cases: Sequence[AttentionCase],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    failures = 0
    max_bytes = int(VALIDATION["max_artifact_bytes"])
    wanted_names = {
        "kernel_details.csv",
        "operator_details.csv",
        "op_statistic.csv",
        "step_trace_time.csv",
        "trace_view.json",
    }

    metrics = (
        ("pipe_utilization", torch_npu.profiler.AiCMetrics.PipeUtilization),
        ("memory", torch_npu.profiler.AiCMetrics.Memory),
    )

    for case in cases:
        inputs = make_inputs(case, harness.device, seed=harness.seed)
        bound = harness.bind("candidate", case, inputs)
        for _ in range(5):
            bound.launch()
        torch_npu.npu.synchronize()

        for metric_name, metric in metrics:
            print(f"[profile:{metric_name}] {case.name}", flush=True)
            with tempfile.TemporaryDirectory(
                prefix=f"welm_attn_profile_{case.attention}_{metric_name}_"
            ) as tmp:
                try:
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
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
                        experimental_config=experimental_config,
                    ):
                        for _ in range(3):
                            bound.launch()
                        torch_npu.npu.synchronize()
                except Exception:
                    failures += 1
                    rows.append(
                        {
                            **common_case_record(harness, case, inputs),
                            "record_type": "profile_artifact",
                            "profile_metric": metric_name,
                            "status": "ERROR",
                            "capture_log_tail": traceback.format_exc()[-20000:],
                        }
                    )
                    continue

                wanted = sorted(
                    path
                    for path in Path(tmp).rglob("*")
                    if path.is_file()
                    and (
                        path.name in wanted_names
                        or path.name.startswith("l2_cache")
                        or path.name.startswith("profiler_info")
                    )
                )
                oversized = [path for path in wanted if path.stat().st_size > max_bytes]
                sources = [path for path in wanted if path.stat().st_size <= max_bytes]
                for source in oversized:
                    failures += 1
                    rows.append(
                        {
                            **common_case_record(harness, case, inputs),
                            "record_type": "profile_artifact",
                            "profile_metric": metric_name,
                            "status": "SKIPPED_TOO_LARGE",
                            "source_name": source.name,
                            "uncompressed_size_bytes": source.stat().st_size,
                            "max_artifact_bytes": max_bytes,
                        }
                    )
                if not sources:
                    failures += 1
                    rows.append(
                        {
                            **common_case_record(harness, case, inputs),
                            "record_type": "profile_artifact",
                            "profile_metric": metric_name,
                            "status": "ERROR",
                            "capture_log_tail": "profiler produced no selected artifact",
                        }
                    )
                    continue
                for source in sources:
                    relative = source.relative_to(tmp)
                    destination = (
                        output_dir
                        / "profile"
                        / case.attention
                        / case.name
                        / metric_name
                    )
                    destination = destination / (
                        f"{str(relative).replace(os.sep, '__')}.gz"
                    )
                    gzip_copy(source, destination)
                    rows.append(
                        {
                            **common_case_record(harness, case, inputs),
                            "record_type": "profile_artifact",
                            "profile_metric": metric_name,
                            "status": "CAPTURED",
                            **file_record(destination, output_dir),
                            "uncompressed_size_bytes": source.stat().st_size,
                        }
                    )
        del bound, inputs
        gc.collect()
    return rows, failures


def _parse_msprof_durations(
    output_dir: Path,
    stdout: str,
    kernel_name: str,
) -> tuple[list[float], list[str], str | None]:
    durations: list[float] = []
    names: list[str] = []
    indexed_durations: list[tuple[int, float]] = []
    # msprof labels captured tasks with numeric launch ids, but renders the
    # per-task reports in lexicographic id order (0, 1, 10, 100, ..., 2, ...).
    # Recover and numerically sort those ids before assigning spans to cases.
    # CSV file ordering is not a stable timeline contract across versions.
    lines = stdout.splitlines()
    launch_id: int | None = None
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
                    try:
                        duration = float(detail.partition(":")[2].strip())
                    except ValueError:
                        pass
                    else:
                        if launch_id is None:
                            return [], names, "matching task has no numeric launch id"
                        indexed_durations.append((launch_id, duration))
                break
    if indexed_durations:
        launch_ids = [item[0] for item in indexed_durations]
        expected_ids = list(range(len(indexed_durations)))
        if sorted(launch_ids) != expected_ids:
            return (
                [],
                names,
                "numeric launch ids are not unique and contiguous: "
                f"got={sorted(launch_ids)[:200]}",
            )
        indexed_durations.sort(key=lambda item: item[0])
        return [item[1] for item in indexed_durations], names, None

    csv_files = sorted(output_dir.rglob("*.csv"))
    basic_files = [path for path in csv_files if "opbasicinfo" in path.name.lower()]
    for path in basic_files:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("Op Name", ""))
                names.append(name)
                if kernel_name.lower() not in name.lower():
                    continue
                try:
                    durations.append(float(row.get("Task Duration(us)", "")))
                except (TypeError, ValueError):
                    pass
    if durations:
        return (
            [],
            names,
            "CSV fallback has no authoritative numeric launch order",
        )
    return [], names, "no matching task durations found"


def capture_msprof_op(
    harness: Harness,
    cases: Sequence[AttentionCase],
    output_dir: Path,
) -> tuple[list[dict[str, object]], int]:
    """Measure primary Triton latency with batched authoritative msprof runs."""

    rows: list[dict[str, object]] = []
    failures = 0
    warmup = int(VALIDATION.get("msprof_warmup_per_case", 1))
    samples = int(VALIDATION.get("msprof_samples_per_case", 3))
    grouped: dict[tuple[str, str], list[AttentionCase]] = {}
    for provider in ("baseline", "candidate"):
        for case in cases:
            key = (provider, primary_kernel_name(provider, case))
            grouped.setdefault(key, []).append(case)

    for (provider, kernel_name), group_cases in grouped.items():
        print(
            f"[msprof-op batch] provider={provider}, kernel={kernel_name}, "
            f"cases={len(group_cases)}",
            flush=True,
        )
        total_launches = len(group_cases) * (warmup + samples)
        with tempfile.TemporaryDirectory(prefix="welm_attn_msprof_") as tmp:
            command = [
                "msprof",
                "op",
                "--warm-up=0",
                f"--launch-count={total_launches}",
                f"--kernel-name={kernel_name}",
                f"--output={tmp}",
                sys.executable,
                str(Path(__file__).resolve()),
                "--msprof-batch-provider",
                provider,
                "--msprof-batch-kernel",
                kernel_name,
                "--msprof-batch-warmup",
                str(warmup),
                "--msprof-batch-samples",
                str(samples),
                "--device",
                str(harness.device),
                "--tp-size",
                str(group_cases[0].tp_size),
            ]
            for case in group_cases:
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
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures += len(group_cases)
                for case in group_cases:
                    rows.append(
                        {
                            **common_case_record(harness, case),
                            "record_type": "msprof_primary_kernel",
                            "provider": provider,
                            "kernel_name": kernel_name,
                            "status": "ERROR",
                            "capture_log_tail": repr(exc),
                        }
                    )
                continue

            durations, names, ordering_error = _parse_msprof_durations(
                Path(tmp), stdout, kernel_name
            )
            artifact = (
                output_dir / "msprof" / provider / f"{kernel_name}.log.gz"
            )
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(artifact, "wt", encoding="utf-8", compresslevel=9) as handle:
                handle.write(stdout)
            expected_count = total_launches
            if result.returncode or ordering_error or len(durations) != expected_count:
                failures += len(group_cases)
                detail = (
                    f"expected {expected_count} ordered durations, got "
                    f"{len(durations)}; ordering_error={ordering_error!r}; "
                    f"discovered={sorted(set(names))[:100]}\n"
                    + stdout[-20000:]
                )
                for case in group_cases:
                    rows.append(
                        {
                            **common_case_record(harness, case),
                            "record_type": "msprof_primary_kernel",
                            "provider": provider,
                            "kernel_name": kernel_name,
                            "timing_authority": "msprof_op_task_duration",
                            "capture_returncode": result.returncode,
                            "status": "ERROR",
                            "capture_log_tail": detail,
                            **file_record(artifact, output_dir),
                        }
                    )
                continue

            span = warmup + samples
            for index, case in enumerate(group_cases):
                measured = durations[index * span + warmup : (index + 1) * span]
                rows.append(
                    {
                        **common_case_record(harness, case),
                        "record_type": "msprof_primary_kernel",
                        "provider": provider,
                        "kernel_name": kernel_name,
                        "timing_authority": "msprof_op_task_duration_acceptance",
                        "capture_returncode": result.returncode,
                        "status": "MEASURED",
                        "sample_count": len(measured),
                        "task_min_us": min(measured),
                        "task_p50_us": statistics.median(measured),
                        "task_mean_us": statistics.fmean(measured),
                        "task_max_us": max(measured),
                        **file_record(artifact, output_dir),
                    }
                )
    return rows, failures


def run_msprof_batch_child(
    harness: Harness,
    cases: Sequence[AttentionCase],
    provider: str,
    kernel_name: str,
    warmup: int,
    samples: int,
) -> None:
    for case in cases:
        inputs = make_inputs(case, harness.device, seed=harness.seed)
        bound = harness.bind(provider, case, inputs)
        if bound.kernel_name != kernel_name:
            raise RuntimeError(
                f"msprof batch kernel mismatch for {case.name}: "
                f"{bound.kernel_name} != {kernel_name}"
            )
        for _ in range(warmup + samples):
            bound.launch()
        torch_npu.npu.synchronize()
        del bound, inputs
        gc.collect()


def build_manual_cases(args: argparse.Namespace) -> list[AttentionCase] | None:
    if not args.m_values:
        return None
    m_values = parse_int_set(args.m_values, minimum=M_MIN, maximum=M_MAX)
    kv_lengths = parse_int_set(
        args.kv_lengths, minimum=1, maximum=int(CONFIG["model_contract"]["max_context_length"])
    )
    return make_manual_cases(
        attentions=[item.strip() for item in args.attention.split(",") if item.strip()],
        topologies=[item.strip() for item in args.topology.split(",") if item.strip()],
        m_values=m_values,
        kv_lengths=kv_lengths,
        length_pattern=args.length_pattern,
        table_width=args.table_width,
        bucket_batch_size=args.bucket_batch_size,
        prefill_batch_size=args.prefill_batch_size,
        tp_size=args.tp_size,
    )


def select_phase_cases(
    args: argparse.Namespace,
    phase: str,
    manual: list[AttentionCase] | None,
) -> list[AttentionCase]:
    if args.case_name:
        return [find_case(name, tp_size=args.tp_size) for name in args.case_name]
    if manual is not None:
        if phase == "graph":
            return [case for case in manual if case.topology.startswith("graph_")]
        return manual
    cases = suite_cases(args.suite, phase, tp_size=args.tp_size)
    if phase == "performance":
        focus = set(VALIDATION.get("performance_focus", []))
        if focus:
            cases = [case for case in cases if case.attention in focus]
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--benchmark-base", default="")
    parser.add_argument("--suite", choices=tuple(CONFIG["suites"]), default="smoke")
    parser.add_argument(
        "--mode", choices=("both", "correctness", "performance"), default="both"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--case-name", action="append", default=[])
    parser.add_argument(
        "--m-values",
        default="",
        help="manual M values/ranges, e.g. 1:1024 or 1,2,63:129",
    )
    parser.add_argument("--kv-lengths", default="4096")
    parser.add_argument("--attention", default="full,swa")
    parser.add_argument(
        "--topology", default="dense,ragged_prefill,verify_d2,verify_d3"
    )
    parser.add_argument(
        "--length-pattern", choices=("uniform", "ragged"), default="uniform"
    )
    parser.add_argument(
        "--table-width", choices=("compact", "graph"), default="compact"
    )
    parser.add_argument("--bucket-batch-size", type=int, default=0)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument(
        "--capture-graph", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--capture-ir", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--capture-profile", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--capture-msprof-op", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--compile-only-provider", choices=("", "baseline", "candidate"), default=""
    )
    parser.add_argument("--compile-only-iterations", type=int, default=1)
    parser.add_argument(
        "--msprof-batch-provider", choices=("", "baseline", "candidate"), default=""
    )
    parser.add_argument("--msprof-batch-kernel", default="")
    parser.add_argument("--msprof-batch-warmup", type=int, default=1)
    parser.add_argument("--msprof-batch-samples", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit_frozen_baseline()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError(f"--device must select an NPU, got {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    benchmark_base = args.benchmark_base or repository_head()
    harness = Harness(device, args.seed, benchmark_base)
    manual = build_manual_cases(args)

    if args.msprof_batch_provider:
        if not args.case_name or not args.msprof_batch_kernel:
            raise ValueError(
                "msprof batch mode requires --case-name and --msprof-batch-kernel"
            )
        cases = [find_case(name, tp_size=args.tp_size) for name in args.case_name]
        run_msprof_batch_child(
            harness,
            cases,
            args.msprof_batch_provider,
            args.msprof_batch_kernel,
            args.msprof_batch_warmup,
            args.msprof_batch_samples,
        )
        return 0

    if args.compile_only_provider:
        if len(args.case_name) != 1:
            raise ValueError("compile-only requires exactly one --case-name")
        case = find_case(args.case_name[0], tp_size=args.tp_size)
        run_compile_only(
            harness,
            case,
            args.compile_only_provider,
            args.compile_only_iterations,
        )
        return 0

    output_dir = Path(args.output_dir or f"manual_results/{utc_now().replace(':', '')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    print("WeLMv4 NPU Full/SWA prefill Attention optimization workspace")
    print(
        f"device={device} ({harness.device_name}), base={benchmark_base[:12]}, "
        f"baseline={harness.baseline_hash[:12]}, candidate={harness.candidate_hash[:12]}"
    )

    phase_state: dict[str, dict[str, object]] = {}
    correctness_failures = 0
    performance_regressions = 0
    artifact_failures = 0

    if args.mode in ("both", "correctness"):
        cases = select_phase_cases(args, "correctness", manual)
        rows, correctness_failures = run_correctness(harness, cases)
        write_csv(output_dir / "correctness.csv", rows)
        phase_state["correctness"] = {
            "status": "PASS" if correctness_failures == 0 else "FAIL",
            "case_count": len(cases),
            "failure_count": correctness_failures,
        }

    if correctness_failures == 0 and args.capture_graph == "on":
        cases = select_phase_cases(args, "graph", manual)
        rows, failures = run_graph_correctness(harness, cases)
        correctness_failures += failures
        write_csv(output_dir / "graph.csv", rows)
        phase_state["graph"] = {
            "status": "PASS" if failures == 0 else "FAIL",
            "case_count": len(cases),
            "failure_count": failures,
            "validation_kind": "real_torch_npu_npugraph_capture_replay",
        }
    elif args.capture_graph == "on":
        phase_state["graph"] = {
            "status": "SKIPPED",
            "reason": "eager correctness failed",
        }

    if args.mode in ("both", "performance") and correctness_failures == 0:
        cases = select_phase_cases(args, "performance", manual)
        rows, performance_regressions, validation_failures, performance_summary = run_performance(
            harness, cases
        )
        correctness_failures += validation_failures
        write_csv(output_dir / "performance.csv", rows)
        phase_state["performance"] = {
            "status": (
                "FAIL"
                if validation_failures
                else "PASS"
                if performance_regressions == 0
                else "PERF_REGRESSION"
            ),
            "case_count": len(cases),
            "regression_count": performance_regressions,
            "validation_failure_count": validation_failures,
            **performance_summary,
        }

        if args.capture_msprof_op == "on" and validation_failures == 0:
            rows, failures = capture_msprof_op(harness, cases, output_dir)
            artifact_failures += failures
            write_csv(output_dir / "msprof_primary_kernel.csv", rows)
            phase_state["msprof_primary_kernel"] = {
                "status": "PASS" if failures == 0 else "ERROR",
                "failure_count": failures,
            }
        elif args.capture_msprof_op == "on":
            phase_state["msprof_primary_kernel"] = {
                "status": "SKIPPED",
                "reason": "timed-shape correctness failed",
            }
    elif args.mode in ("both", "performance"):
        phase_state["performance"] = {
            "status": "SKIPPED",
            "reason": "correctness failed",
        }
        if args.capture_msprof_op == "on":
            phase_state["msprof_primary_kernel"] = {
                "status": "SKIPPED",
                "reason": "correctness failed",
            }

    if correctness_failures == 0 and args.capture_ir == "on":
        cases = select_phase_cases(args, "ir", manual)
        rows, failures = capture_ir(harness, cases, output_dir)
        artifact_failures += failures
        write_csv(output_dir / "ir.csv", rows)
        phase_state["ir"] = {
            "status": "PASS" if failures == 0 else "ERROR",
            "case_count": len(cases),
            "failure_count": failures,
        }
    elif args.capture_ir == "on":
        phase_state["ir"] = {
            "status": "SKIPPED",
            "reason": "correctness failed",
        }

    if correctness_failures == 0 and args.capture_profile == "on":
        cases = select_phase_cases(args, "profile", manual)
        rows, failures = capture_profiles(harness, cases, output_dir)
        artifact_failures += failures
        write_csv(output_dir / "profile.csv", rows)
        phase_state["profile"] = {
            "status": "PASS" if failures == 0 else "ERROR",
            "case_count": len(cases),
            "failure_count": failures,
        }
    elif args.capture_profile == "on":
        phase_state["profile"] = {
            "status": "SKIPPED",
            "reason": "correctness failed",
        }

    if correctness_failures:
        status = "FAIL"
        exit_code = 1
    elif artifact_failures:
        status = "ERROR"
        exit_code = 3
    elif performance_regressions:
        status = "PERF_REGRESSION"
        exit_code = 2
    else:
        status = "PASS"
        exit_code = 0

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workspace": CONFIG["workspace"],
        "status": status,
        "started_at": started,
        "finished_at": utc_now(),
        "benchmark_base": benchmark_base,
        "workspace_commit": harness.commit,
        "suite": args.suite,
        "mode": args.mode,
        "device": harness.metadata(),
        "production_source": CONFIG["production_source"],
        "model_contract": CONFIG["model_contract"],
        "validation": CONFIG["validation"],
        "phases": phase_state,
        "files": [file_record(path, output_dir) for path in files],
    }
    write_json(output_dir / "result.json", manifest)
    print(f"result={status}; manifest={output_dir / 'result.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
