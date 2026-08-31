"""Case contract for WeLMv4 Full/SWA attention under Attention-DP.

The module intentionally has no torch/NPU dependency so the monitor can audit
the workspace and expand case names before entering the accelerator runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "dp_attention_workspace_config.json"

PREFILL_BASELINE_PATH = ROOT / "welmv4_dp_prefill_attention_baseline.py"
PREFILL_CANDIDATE_PATH = ROOT / "welmv4_dp_prefill_attention_candidate.py"
DECODE_BASELINE_PATH = ROOT / "welmv4_dp_decode_attention_baseline.py"
DECODE_CANDIDATE_PATH = ROOT / "welmv4_dp_decode_attention_candidate.py"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()
MODEL = CONFIG["model_contract"]
VALIDATION = CONFIG["validation"]

PAGE_SIZE = int(MODEL["page_size"])
HEAD_DIM = int(MODEL["head_dim"])
GLOBAL_NUM_Q_HEADS = int(MODEL["global_num_q_heads"])
GLOBAL_NUM_KV_HEADS = int(MODEL["global_num_kv_heads"])
TP_SIZE = int(MODEL["tp_size"])
SOFTMAX_SCALE = float(MODEL["softmax_scale"])
SWA_LEFT_WINDOW = int(MODEL["swa_left_window"])
SWA_GLOBAL_WINDOW = int(MODEL["swa_global_window"])
MAX_CONTEXT_LENGTH = int(MODEL["max_context_length"])
M_MAX = int(MODEL["m_max"])


def sha256_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def audit_frozen_baselines() -> None:
    paths = {
        "prefill": PREFILL_BASELINE_PATH,
        "decode": DECODE_BASELINE_PATH,
    }
    for family, path in paths.items():
        expected = str(CONFIG["production_sources"][family]["baseline_sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen {family} baseline changed: expected={expected}, actual={actual}."
            )


@dataclass(frozen=True)
class HeadLayout:
    name: str
    dp_size: int
    local_num_q_heads: int
    local_num_kv_heads: int

    @property
    def q_head_scale(self) -> float:
        return self.local_num_q_heads / (GLOBAL_NUM_Q_HEADS / TP_SIZE)


def head_layout(name: str) -> HeadLayout:
    try:
        spec = CONFIG["head_layouts"][name]
    except KeyError as exc:
        raise ValueError(f"unknown head layout: {name!r}") from exc
    layout = HeadLayout(
        name=name,
        dp_size=int(spec["dp_size"]),
        local_num_q_heads=int(spec["local_num_q_heads"]),
        local_num_kv_heads=int(spec["local_num_kv_heads"]),
    )
    if layout.local_num_q_heads % layout.local_num_kv_heads:
        raise ValueError(f"invalid GQA layout: {layout}")
    return layout


def topology_width(topology: str) -> int | None:
    match = re.fullmatch(r"verify_d([234])", topology)
    if match:
        return int(match.group(1))
    if topology in ("prefill_dense", "prefill_ragged", "decode", "mirror"):
        return None
    raise ValueError(f"unknown topology: {topology!r}")


@dataclass(frozen=True)
class AttentionCase:
    attention: str
    topology: str
    layout: str
    m: int
    kv_length: int
    batch_size: int = 0
    length_pattern: str = "uniform"

    def __post_init__(self) -> None:
        if self.attention not in ("full", "swa"):
            raise ValueError(f"unknown attention kind: {self.attention!r}")
        head_layout(self.layout)
        width = topology_width(self.topology)
        if not 1 <= self.m <= M_MAX:
            raise ValueError(f"M must be in [1, {M_MAX}], got {self.m}")
        if self.kv_length < 1 or self.kv_length > MAX_CONTEXT_LENGTH:
            raise ValueError(f"invalid kv_length={self.kv_length}")
        if self.length_pattern not in ("uniform", "ragged"):
            raise ValueError("length_pattern must be uniform or ragged")
        if self.topology == "prefill_dense":
            if self.batch_size not in (0, 1):
                raise ValueError("prefill_dense contains one request")
        elif self.topology == "prefill_ragged":
            if not 2 <= self.batch_size <= self.m:
                raise ValueError("prefill_ragged requires 2 <= batch_size <= M")
        elif self.topology in ("decode", "mirror"):
            if self.batch_size < 1 or self.m != self.batch_size:
                raise ValueError("decode/mirror require M == batch_size")
        elif width is not None:
            if self.batch_size < 1 or self.m != self.batch_size * width:
                raise ValueError(
                    f"{self.topology} requires M=batch_size*{width}, got {self.m}"
                )

    @property
    def family(self) -> str:
        return "decode" if self.topology in ("decode", "mirror") else "prefill"

    @property
    def draft_width(self) -> int | None:
        return topology_width(self.topology)

    @property
    def real_batch_size(self) -> int:
        if self.topology == "prefill_dense":
            return 1
        return self.batch_size

    @property
    def scheduled_batch_size(self) -> int:
        """Number of request rows represented by host-side metadata.

        This workspace benchmarks eager operator launches only, so there is no
        larger captured Graph bucket: scheduled and live batch sizes coincide.
        """
        return self.real_batch_size

    @property
    def q_buffer_rows(self) -> int:
        """Physical Q rows; eager launches contain no Graph padding rows."""
        return self.m

    @property
    def capture_q_lens(self) -> tuple[int, ...]:
        """Host query lengths used to prepare the production prefill schedule."""
        return self.runtime_q_lens

    @property
    def capture_kv_lens(self) -> tuple[int, ...]:
        """Host KV lengths used to prepare the production prefill schedule."""
        return self.runtime_kv_lens

    @property
    def local_num_q_heads(self) -> int:
        return head_layout(self.layout).local_num_q_heads

    @property
    def local_num_kv_heads(self) -> int:
        return head_layout(self.layout).local_num_kv_heads

    @property
    def q_head_scale(self) -> float:
        return head_layout(self.layout).q_head_scale

    @property
    def runtime_q_lens(self) -> tuple[int, ...]:
        if self.topology == "prefill_dense":
            return (self.m,)
        if self.topology == "prefill_ragged":
            base, remainder = divmod(self.m, self.batch_size)
            values = [base] * self.batch_size
            for index in range(remainder):
                values[(index * 5 + 1) % self.batch_size] += 1
            return tuple(values)
        if self.topology in ("decode", "mirror"):
            return (1,) * self.batch_size
        assert self.draft_width is not None
        return (self.draft_width,) * self.batch_size

    @property
    def max_q_len(self) -> int:
        return max(self.runtime_q_lens)

    @property
    def runtime_kv_lens(self) -> tuple[int, ...]:
        values: list[int] = []
        for request_id, q_len in enumerate(self.runtime_q_lens):
            if self.length_pattern == "uniform":
                length = self.kv_length
            else:
                variation = (request_id * 37 + 17) % (4 * PAGE_SIZE + 1)
                length = self.kv_length - variation
            values.append(max(q_len, length))
        return tuple(values)

    @property
    def block_table_width(self) -> int:
        return max(1, math.ceil(max(self.runtime_kv_lens) / PAGE_SIZE))

    @property
    def max_kv_len_hint(self) -> int:
        # The backend uses max_context_len for mirror/graph decode and the live
        # CPU maximum for ordinary eager decode.
        return MAX_CONTEXT_LENGTH if self.topology == "mirror" else max(self.runtime_kv_lens)

    @property
    def workload_id(self) -> str:
        return "_".join(
            (
                self.attention,
                self.topology,
                f"m{self.m}",
                f"b{self.real_batch_size}",
                f"kv{self.kv_length}",
                self.length_pattern,
            )
        )

    @property
    def name(self) -> str:
        return f"{self.workload_id}_{self.layout}"

    def as_record(self) -> dict[str, object]:
        return {
            "case": self.name,
            "workload_id": self.workload_id,
            "attention": self.attention,
            "family": self.family,
            "topology": self.topology,
            "layout": self.layout,
            "dp_size": head_layout(self.layout).dp_size,
            "m": self.m,
            "batch_size": self.real_batch_size,
            "draft_width": self.draft_width or "",
            "max_q_len": self.max_q_len,
            "kv_length": self.kv_length,
            "min_runtime_kv_length": min(self.runtime_kv_lens),
            "length_pattern": self.length_pattern,
            "block_table_width": self.block_table_width,
            "num_q_heads": self.local_num_q_heads,
            "num_kv_heads": self.local_num_kv_heads,
            "q_head_scale": self.q_head_scale,
            "page_size": PAGE_SIZE,
            "head_dim": HEAD_DIM,
            "softmax_scale": SOFTMAX_SCALE,
            "swa_left_window": SWA_LEFT_WINDOW if self.attention == "swa" else "",
            "max_kv_len_hint": self.max_kv_len_hint if self.family == "decode" else "",
        }


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _case_from_spec(spec: dict) -> AttentionCase:
    topology = str(spec["topology"])
    width = topology_width(topology)
    batch_size = int(spec.get("batch_size", 0))
    if "m" in spec:
        m = int(spec["m"])
    elif topology in ("decode", "mirror"):
        m = batch_size
    elif width is not None:
        m = batch_size * width
    else:
        raise ValueError(f"case has no M: {spec}")
    kv_raw = spec.get("kv_length", m)
    kv_length = m if kv_raw == "match_m" else int(kv_raw)
    return AttentionCase(
        attention=str(spec["attention"]),
        topology=topology,
        layout=str(spec["layout"]),
        m=m,
        kv_length=kv_length,
        batch_size=batch_size,
        length_pattern=str(spec.get("length_pattern", "uniform")),
    )


def _expand_group(group: dict) -> Iterator[AttentionCase]:
    attentions = _as_list(group["attention"])
    layouts = _as_list(group.get("layouts", group.get("layout")))
    topologies = _as_list(group["topology"])
    m_values = _as_list(group.get("m_values", group.get("m")))
    batch_sizes = _as_list(group.get("batch_sizes", group.get("batch_size", 0)))
    kv_lengths = _as_list(group.get("kv_lengths", group.get("kv_length", None)))
    if layouts == [None] or kv_lengths == [None]:
        raise ValueError(f"incomplete group: {group}")
    for attention in attentions:
        for layout in layouts:
            for topology in topologies:
                width = topology_width(str(topology))
                if str(topology) in ("decode", "mirror") or width is not None:
                    shape_axis = [(None, int(batch)) for batch in batch_sizes]
                else:
                    shape_axis = [(int(m), int(batch_sizes[0])) for m in m_values]
                for m, batch_size in shape_axis:
                    for kv_length in kv_lengths:
                        spec = {
                            **group,
                            "attention": attention,
                            "layout": layout,
                            "topology": topology,
                            "batch_size": batch_size,
                            "kv_length": kv_length,
                        }
                        if m is not None:
                            spec["m"] = m
                        yield _case_from_spec(spec)


def deduplicate_cases(cases: Iterable[AttentionCase]) -> list[AttentionCase]:
    return list(dict.fromkeys(cases))


def suite_cases(suite: str, phase: str) -> list[AttentionCase]:
    if suite == "remote":
        suite = "full" if os.environ.get("WELMV4_DP_ATTENTION_FORCE_FULL") == "1" else "iteration"
    try:
        suite_config = CONFIG["suites"][suite]
    except KeyError as exc:
        raise ValueError(f"unknown suite {suite!r}") from exc
    cases = [_case_from_spec(spec) for spec in suite_config.get(phase, [])]
    for group in suite_config.get(f"{phase}_groups", []):
        cases.extend(_expand_group(group))
    if suite == "full" and phase in ("ir", "profile") and not cases:
        return suite_cases("iteration", phase)
    return deduplicate_cases(cases)


def find_case(name: str) -> AttentionCase:
    for suite in CONFIG["suites"]:
        for phase in ("correctness", "performance", "ir", "profile"):
            for case in suite_cases(suite, phase):
                if case.name == name:
                    return case
    pattern = re.compile(
        r"^(full|swa)_(prefill_dense|prefill_ragged|verify_d[234]|decode|mirror)_"
        r"m([0-9]+)_b([0-9]+)_kv([0-9]+)_(uniform|ragged)_(tp4(?:_dp[24])?)$"
    )
    match = pattern.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid or unknown case name: {name}")
    attention, topology, m, batch, kv, pattern_name, layout = match.groups()
    return AttentionCase(
        attention=attention,
        topology=topology,
        layout=layout,
        m=int(m),
        kv_length=int(kv),
        batch_size=int(batch),
        length_pattern=pattern_name,
    )
