#!/usr/bin/env python3
"""Standalone WeLMv4 Full/SWA DP-Attention benchmark for Ascend.

The benchmark imports no NEWSGLANG package. Prefill/target-verify snapshots come
from ``welmv4_sink_prefill_attention.py``; eager decode and decode-like mirror
snapshots come from ``sink_full_attention.py``.
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

import welmv4_dp_decode_attention_baseline as decode_baseline_ops
import welmv4_dp_decode_attention_candidate as decode_candidate_ops
import welmv4_dp_prefill_attention_baseline as prefill_baseline_ops
import welmv4_dp_prefill_attention_candidate as prefill_candidate_ops
from dp_attention_contract import (
    CONFIG,
    DECODE_BASELINE_PATH,
    DECODE_CANDIDATE_PATH,
    HEAD_DIM,
    MAX_CONTEXT_LENGTH,
    PAGE_SIZE,
    PREFILL_BASELINE_PATH,
    PREFILL_CANDIDATE_PATH,
    SOFTMAX_SCALE,
    SWA_GLOBAL_WINDOW,
    SWA_LEFT_WINDOW,
    VALIDATION,
    AttentionCase,
    audit_frozen_baselines,
    find_case,
    sha256_file,
    suite_cases,
)
from dp_attention_inputs import AttentionInputs, make_inputs
from dp_attention_reference import error_metrics, reference_attention


ROOT = Path(__file__).resolve().parent
IR_CAPTURE_SCRIPT = ROOT / "capture_welmv4_dp_attention_ir.sh"
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
    kernel_names: tuple[str, ...]

    @property
    def kernel_name(self) -> str:
        return self.kernel_names[0]


def expected_kernel_names(
    provider: str,
    case: AttentionCase,
    num_cube_cores: int | None = None,
) -> tuple[str, ...]:
    del provider  # Baseline and candidate start with the same production dispatch.
    if num_cube_cores is None:
        num_cube_cores = 24

    if case.family == "decode":
        if case.attention == "swa":
            return ("_swa_paged_decode_sink_kernel",)
        group_size = case.local_num_q_heads // case.local_num_kv_heads
        loop_times = case.real_batch_size * case.local_num_kv_heads
        use_fd = (
            case.max_kv_len_hint >= 256
            and loop_times < 0.4 * num_cube_cores
            and (group_size == 1 or case.max_kv_len_hint >= 2048)
        )
        return (
            ("paged_decode_fd_kernel", "paged_decode_fd_reduce_kernel")
            if use_fd
            else ("paged_decode_kernel",)
        )

    if case.attention == "full":
        use_small_q6 = case.local_num_q_heads == 6 and case.max_q_len <= 4
        use_mid_q6 = case.local_num_q_heads == 6 and 5 <= case.max_q_len <= 128
        if case.local_num_q_heads == 6 and 128 < case.max_q_len <= 256:
            num_q_blocks = (case.max_q_len + 63) // 64
            use_mid_q6 = (
                case.real_batch_size * num_q_blocks * case.local_num_q_heads
                <= num_cube_cores
            )
        if use_small_q6:
            return ("paged_prefill_small_q_grouped_kernel",)
        if use_mid_q6:
            return ("paged_prefill_mid_q_grouped_kernel",)
        return ("paged_prefill_page_aggregation_kernel",)

    use_grouped_q6 = (
        case.local_num_q_heads == 6
        and case.local_num_kv_heads == 1
        and case.max_q_len <= 4
        and (case.real_batch_size > 1 or case.max_q_len <= 2)
    )
    if not use_grouped_q6:
        return ("_swa_paged_prefill_aggregation_sink_kernel",)
    if case.max_q_len == 1:
        return ("_swa_paged_prefill_single_q_grouped_sink_kernel",)
    if case.max_q_len == 4:
        return ("_swa_paged_prefill_four_q_grouped_sink_kernel",)
    return ("_swa_paged_prefill_small_q_grouped_sink_kernel",)


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
        self.baseline_hashes = {
            "prefill": sha256_file(PREFILL_BASELINE_PATH),
            "decode": sha256_file(DECODE_BASELINE_PATH),
        }
        self.candidate_hashes = {
            "prefill": sha256_file(PREFILL_CANDIDATE_PATH),
            "decode": sha256_file(DECODE_CANDIDATE_PATH),
        }
        # NEWSGLANG owns one persistent causal template on the backend.  Passing
        # None would make the copied public wrapper allocate a 1024x3072 mask
        # on every call and would no longer represent production latency.
        self.full_aux_mask = torch.ones(
            (1024, 1024 * 3), device=device, dtype=torch.bool
        ).tril_(diagonal=1024)

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_base": self.benchmark_base,
            "workspace_commit": self.commit,
            "baseline_sha256": self.baseline_hashes,
            "candidate_sha256": self.candidate_hashes,
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
        modules = {
            ("prefill", "baseline"): prefill_baseline_ops,
            ("prefill", "candidate"): prefill_candidate_ops,
            ("decode", "baseline"): decode_baseline_ops,
            ("decode", "candidate"): decode_candidate_ops,
        }
        try:
            module = modules[(case.family, provider)]
        except KeyError as exc:
            raise ValueError(f"unknown provider: {provider}")

        kernel_names = expected_kernel_names(
            provider, case, self.num_cube_cores
        )

        if case.family == "decode":
            if case.attention == "full":

                def launch_full_decode() -> torch.Tensor:
                    return module.paged_attention_decode_impl(
                        q=inputs.q,
                        key_cache=inputs.key_cache,
                        value_cache=inputs.value_cache,
                        seqlens=inputs.runtime_kv_lens,
                        block_tables=inputs.block_table,
                        gqa_interleave=False,
                        softmax_scale=SOFTMAX_SCALE,
                        sinks=inputs.sinks,
                        max_kv_len_hint=case.max_kv_len_hint,
                    )

                return BoundLaunch(launch_full_decode, 0.0, kernel_names)

            def launch_swa_decode() -> torch.Tensor:
                return module.swa_paged_decode_impl(
                    q=inputs.q,
                    key_cache=inputs.key_cache,
                    value_cache=inputs.value_cache,
                    seqlens=inputs.runtime_kv_lens,
                    block_tables=inputs.block_table,
                    local_window_size=SWA_LEFT_WINDOW,
                    global_window_size=SWA_GLOBAL_WINDOW,
                    gqa_interleave=False,
                    softmax_scale=SOFTMAX_SCALE,
                    sinks=inputs.sinks,
                )

            return BoundLaunch(launch_swa_decode, 0.0, kernel_names)

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
                kernel_names,
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
        # The production backend owns CPU query lengths. Passing their maximum
        # avoids a device read/synchronization and matches the real call site.
        swa_kwargs["max_q_len"] = case.max_q_len

        def launch_swa() -> torch.Tensor:
            return module.swa_paged_prefill_impl(
                **swa_kwargs,
            )

        return BoundLaunch(
            launch_swa,
            0.0,
            kernel_names,
        )


def common_case_record(
    harness: Harness,
    case: AttentionCase,
    inputs: AttentionInputs | None = None,
) -> dict[str, object]:
    record = {**harness.metadata(), **case.as_record()}
    record["runtime_q_lens"] = json.dumps(case.runtime_q_lens)
    record["runtime_kv_lens"] = json.dumps(case.runtime_kv_lens)
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
        expected, checked_rows = reference_attention(
            attention=case.attention,
            q=inputs.q,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            runtime_q_lens=case.runtime_q_lens,
            runtime_kv_lens=case.runtime_kv_lens,
            block_table=inputs.block_table,
            sinks=inputs.sinks,
            max_query_rows=int(VALIDATION["reference_max_query_rows"]),
        )
        outputs: dict[str, torch.Tensor] = {}
        prepare_times: dict[str, float] = {}
        for provider in ("baseline", "candidate"):
            bound = harness.bind(provider, case, inputs)
            outputs[provider] = bound.launch()
            prepare_times[provider] = bound.host_prepare_submit_ms
        torch_npu.npu.synchronize()

        expected_checked = expected.index_select(0, checked_rows)
        baseline_checked = outputs["baseline"].index_select(0, checked_rows)
        candidate_checked = outputs["candidate"].index_select(0, checked_rows)
        baseline_metrics = error_metrics(baseline_checked, expected_checked)
        candidate_metrics = error_metrics(candidate_checked, expected_checked)
        pair_metrics = error_metrics(outputs["candidate"], outputs["baseline"])
        baseline_ok = bool(
            torch.allclose(
                baseline_checked,
                expected_checked,
                atol=reference_atol,
                rtol=reference_rtol,
            )
        )
        candidate_ok = bool(
            torch.allclose(
                candidate_checked,
                expected_checked,
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
            "reference_checked_rows": int(checked_rows.numel()),
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
        del expected, expected_checked, checked_rows, outputs, inputs
        gc.collect()
    return rows, failures


def run_performance(
    harness: Harness,
    cases: Sequence[AttentionCase],
) -> tuple[list[dict[str, object]], int, int, dict[str, object]]:
    rows: list[dict[str, object]] = []
    regressions = 0
    validation_failures = 0

    for index, case in enumerate(cases, 1):
        print(f"[performance {index}/{len(cases)}] {case.name}", flush=True)
        inputs = make_inputs(case, harness.device, seed=harness.seed)
        bound = {
            provider: harness.bind(provider, case, inputs)
            for provider in ("baseline", "candidate")
        }
        # Performance-only runs must not let a fast but wrong candidate bypass
        # correctness. Validate the exact timed shape before any warmup/event.
        expected, checked_rows = reference_attention(
            attention=case.attention,
            q=inputs.q,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            runtime_q_lens=case.runtime_q_lens,
            runtime_kv_lens=case.runtime_kv_lens,
            block_table=inputs.block_table,
            sinks=inputs.sinks,
            max_query_rows=int(VALIDATION["reference_max_query_rows"]),
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
        expected_checked = expected.index_select(0, checked_rows)
        baseline_checked = probe["baseline"].index_select(0, checked_rows)
        candidate_checked = probe["candidate"].index_select(0, checked_rows)
        baseline_ok = bool(
            torch.allclose(
                baseline_checked,
                expected_checked,
                atol=reference_atol,
                rtol=reference_rtol,
            )
        )
        candidate_ok = bool(
            torch.allclose(
                candidate_checked,
                expected_checked,
                atol=reference_atol,
                rtol=reference_rtol,
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
    summary = {
        "shape_validation_case_count": len(cases),
        "timing_authority": "msprof_op_task_duration",
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
    """Measure every Triton component with authoritative msprof task durations."""

    rows: list[dict[str, object]] = []
    failures = 0
    warmup = int(VALIDATION.get("msprof_warmup_per_case", 1))
    samples = int(VALIDATION.get("msprof_samples_per_case", 3))
    grouped: dict[tuple[str, str], list[AttentionCase]] = {}
    for provider in ("baseline", "candidate"):
        for case in cases:
            for kernel_name in expected_kernel_names(
                provider, case, harness.num_cube_cores
            ):
                grouped.setdefault((provider, kernel_name), []).append(case)

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
                            "record_type": "msprof_kernel_component",
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
                            "record_type": "msprof_kernel_component",
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
                        "record_type": "msprof_kernel_component",
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


def evaluate_msprof_performance(
    harness: Harness,
    cases: Sequence[AttentionCase],
    component_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], int, dict[str, object]]:
    """Apply no-regression and DP head-scaling gates to msprof task time."""

    baseline_snapshot = all(
        harness.baseline_hashes[family] == harness.candidate_hashes[family]
        for family in ("prefill", "decode")
    )
    measured = {
        (str(row["case"]), str(row["provider"]), str(row["kernel_name"])): row
        for row in component_rows
        if row.get("status") == "MEASURED"
    }
    minimum_speedup = float(VALIDATION["minimum_case_speedup"])
    control_minimum = float(VALIDATION["minimum_control_layout_speedup"])
    maximum_normalized_ratio = float(
        VALIDATION["maximum_normalized_cost_ratio_vs_tp4"]
    )
    totals: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    regressions = 0

    for case in cases:
        provider_totals: dict[str, float] = {}
        provider_components: dict[str, dict[str, float]] = {}
        missing: list[str] = []
        for provider in ("baseline", "candidate"):
            names = expected_kernel_names(provider, case, harness.num_cube_cores)
            components: dict[str, float] = {}
            for name in names:
                component = measured.get((case.name, provider, name))
                if component is None:
                    missing.append(f"{provider}:{name}")
                    continue
                components[name] = float(component["task_p50_us"])
            provider_components[provider] = components
            if len(components) == len(names):
                provider_totals[provider] = sum(components.values())

        if missing:
            regressions += 1
            rows.append(
                {
                    **common_case_record(harness, case),
                    "record_type": "msprof_logical_attention_latency",
                    "status": "ERROR",
                    "reason": f"missing msprof components: {missing}",
                }
            )
            continue

        baseline_us = provider_totals["baseline"]
        candidate_us = provider_totals["candidate"]
        speedup = baseline_us / candidate_us
        effective_minimum = control_minimum if case.layout == "tp4" else minimum_speedup
        passed = baseline_snapshot or speedup >= effective_minimum
        regressions += int(not passed)
        normalized_cost = candidate_us / case.q_head_scale
        totals[(case.workload_id, case.layout)] = normalized_cost
        target = VALIDATION.get("decode_reference_targets_us", {}).get(
            str(case.real_batch_size), {}
        ).get(case.attention)
        rows.append(
            {
                **common_case_record(harness, case),
                "record_type": "msprof_logical_attention_latency",
                "timing_authority": "sum_of_msprof_op_task_duration_p50",
                "status": (
                    "BASELINE_SNAPSHOT"
                    if baseline_snapshot
                    else ("PASS" if passed else "PERF_REGRESSION")
                ),
                "baseline_components_us": json.dumps(provider_components["baseline"]),
                "candidate_components_us": json.dumps(provider_components["candidate"]),
                "baseline_p50_us": baseline_us,
                "candidate_p50_us": candidate_us,
                "speedup_vs_baseline": speedup,
                "effective_minimum_case_speedup": effective_minimum,
                "candidate_us_per_qhead_scale": normalized_cost,
                "decode_reference_target_us": target if case.family == "decode" else None,
                "decode_target_ratio": (
                    candidate_us / float(target)
                    if case.family == "decode" and target is not None
                    else None
                ),
            }
        )

    efficiency_failures = 0
    for row in rows:
        if row.get("status") == "ERROR":
            continue
        workload_id = str(row["workload_id"])
        control = totals.get((workload_id, "tp4"))
        current = totals.get((workload_id, str(row["layout"])))
        if control is None or current is None:
            continue
        ratio = current / control
        efficient = ratio <= maximum_normalized_ratio
        row["normalized_cost_ratio_vs_tp4"] = ratio
        row["maximum_normalized_cost_ratio_vs_tp4"] = maximum_normalized_ratio
        row["normalized_efficiency_pass"] = efficient
        if not efficient:
            efficiency_failures += 1
            if not baseline_snapshot and row["status"] == "PASS":
                row["status"] = "PERF_REGRESSION"
                regressions += 1

    speedups = [
        float(row["speedup_vs_baseline"])
        for row in rows
        if "speedup_vs_baseline" in row
    ]
    summary = {
        "timing_authority": "sum_of_msprof_op_task_duration_p50",
        "baseline_snapshot": baseline_snapshot,
        "case_count": len(cases),
        "regression_count": regressions,
        "normalized_efficiency_failure_count": efficiency_failures,
        "speedup_geomean": (
            statistics.geometric_mean(speedups) if speedups else 0.0
        ),
    }
    return rows, regressions, summary


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
        if kernel_name not in bound.kernel_names:
            raise RuntimeError(
                f"msprof batch kernel mismatch for {case.name}: "
                f"{bound.kernel_names} does not contain {kernel_name}"
            )
        for _ in range(warmup + samples):
            bound.launch()
        torch_npu.npu.synchronize()
        del bound, inputs
        gc.collect()


def select_phase_cases(
    args: argparse.Namespace,
    phase: str,
) -> list[AttentionCase]:
    if args.case_name:
        return [find_case(name) for name in args.case_name]
    return suite_cases(args.suite, phase)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--benchmark-base", default="")
    parser.add_argument(
        "--suite", choices=tuple(CONFIG["suites"]) + ("remote",), default="smoke"
    )
    parser.add_argument(
        "--mode", choices=("both", "correctness", "performance"), default="both"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--case-name", action="append", default=[])
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
    audit_frozen_baselines()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError(f"--device must select an NPU, got {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    benchmark_base = args.benchmark_base or repository_head()
    harness = Harness(device, args.seed, benchmark_base)

    if args.msprof_batch_provider:
        if not args.case_name or not args.msprof_batch_kernel:
            raise ValueError(
                "msprof batch mode requires --case-name and --msprof-batch-kernel"
            )
        cases = [find_case(name) for name in args.case_name]
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
        case = find_case(args.case_name[0])
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
    print("WeLMv4 NPU Full/SWA DP-Attention optimization workspace")
    print(
        f"device={device} ({harness.device_name}), base={benchmark_base[:12]}, "
        f"baseline={harness.baseline_hashes}, candidate={harness.candidate_hashes}"
    )

    phase_state: dict[str, dict[str, object]] = {}
    correctness_failures = 0
    performance_regressions = 0
    artifact_failures = 0

    # Fail fast on msprof command/kernel-name/parser problems before compiling
    # and validating the much larger iteration matrix.
    if args.capture_msprof_op == "on":
        # Exercise both production source families and both attention kinds
        # before entering the long matrix. This also covers Full decode's
        # two-component FD path, so msprof/parser failures stop within minutes.
        preflight_cases = suite_cases("smoke", "correctness")[:4]
        preflight_dir = output_dir / "msprof_preflight"
        preflight_rows, preflight_failures = capture_msprof_op(
            harness, preflight_cases, preflight_dir
        )
        for row in preflight_rows:
            if row.get("path"):
                row["path"] = f"msprof_preflight/{row['path']}"
        write_csv(output_dir / "msprof_preflight.csv", preflight_rows)
        artifact_failures += preflight_failures
        phase_state["msprof_preflight"] = {
            "status": "PASS" if preflight_failures == 0 else "ERROR",
            "case_count": len(preflight_cases),
            "failure_count": preflight_failures,
        }

    if artifact_failures == 0 and args.mode in ("both", "correctness"):
        cases = select_phase_cases(args, "correctness")
        rows, correctness_failures = run_correctness(harness, cases)
        write_csv(output_dir / "correctness.csv", rows)
        phase_state["correctness"] = {
            "status": "PASS" if correctness_failures == 0 else "FAIL",
            "case_count": len(cases),
            "failure_count": correctness_failures,
        }

    if (
        args.mode in ("both", "performance")
        and correctness_failures == 0
        and artifact_failures == 0
    ):
        cases = select_phase_cases(args, "performance")
        rows, performance_regressions, validation_failures, performance_summary = run_performance(
            harness, cases
        )
        correctness_failures += validation_failures
        write_csv(output_dir / "performance_shape_validation.csv", rows)
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
            component_rows, failures = capture_msprof_op(harness, cases, output_dir)
            artifact_failures += failures
            write_csv(output_dir / "msprof_components.csv", component_rows)
            phase_state["msprof_components"] = {
                "status": "PASS" if failures == 0 else "ERROR",
                "failure_count": failures,
            }
            if failures == 0:
                perf_rows, regressions, msprof_summary = evaluate_msprof_performance(
                    harness, cases, component_rows
                )
                performance_regressions += regressions
                write_csv(output_dir / "performance.csv", perf_rows)
                phase_state["performance"].update(msprof_summary)
                phase_state["performance"]["status"] = (
                    "PASS" if regressions == 0 else "PERF_REGRESSION"
                )
        elif args.capture_msprof_op == "on":
            phase_state["msprof_components"] = {
                "status": "SKIPPED",
                "reason": "timed-shape correctness failed",
            }
    elif args.mode in ("both", "performance"):
        phase_state["performance"] = {
            "status": "SKIPPED",
            "reason": "correctness failed",
        }
        if args.capture_msprof_op == "on":
            phase_state["msprof_components"] = {
                "status": "SKIPPED",
                "reason": "correctness failed",
            }

    if correctness_failures == 0 and artifact_failures == 0 and args.capture_ir == "on":
        cases = select_phase_cases(args, "ir")
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

    if correctness_failures == 0 and artifact_failures == 0 and args.capture_profile == "on":
        cases = select_phase_cases(args, "profile")
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
        "production_sources": CONFIG["production_sources"],
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
