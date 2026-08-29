"""WeLMv4 production constants and reproducible Attention case expansion.

This module deliberately has no torch/NPU imports.  The long-running remote
worker can inspect the manifest and case names even when it is launched from a
plain system Python environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "workspace_config.json"
BASELINE_PATH = ROOT / "welmv4_prefill_attention_baseline.py"
CANDIDATE_PATH = ROOT / "welmv4_prefill_attention_candidate.py"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()
MODEL = CONFIG["model_contract"]
VALIDATION = CONFIG["validation"]

PAGE_SIZE = int(MODEL["page_size"])
HEAD_DIM = int(MODEL["head_dim"])
GLOBAL_NUM_Q_HEADS = int(MODEL["global_num_q_heads"])
GLOBAL_NUM_KV_HEADS = int(MODEL["global_num_kv_heads"])
DEFAULT_TP_SIZE = int(MODEL["default_tp_size"])
SOFTMAX_SCALE = float(MODEL["softmax_scale"])
SWA_LEFT_WINDOW = int(MODEL["swa_left_window"])
SWA_GLOBAL_WINDOW = int(MODEL["swa_global_window"])
GRAPH_BLOCK_TABLE_WIDTH = int(MODEL["graph_block_table_width"])
M_MIN = int(MODEL["m_min"])
M_MAX = int(MODEL["m_max"])


def sha256_file(path: Path) -> str:
    # Git checks this workspace out on Windows and Linux. Hash the canonical
    # LF form so core.autocrlf cannot make a frozen source appear modified.
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def audit_frozen_baseline() -> None:
    expected = str(CONFIG["production_source"]["baseline_sha256"])
    actual = sha256_file(BASELINE_PATH)
    if actual != expected:
        raise RuntimeError(
            "The frozen production baseline changed. Restore it or update "
            f"workspace_config.json deliberately: expected={expected}, actual={actual}."
        )


def local_head_counts(tp_size: int) -> tuple[int, int]:
    if tp_size <= 0 or GLOBAL_NUM_Q_HEADS % tp_size:
        raise ValueError(
            f"tp_size must divide {GLOBAL_NUM_Q_HEADS}, got {tp_size}."
        )
    if GLOBAL_NUM_KV_HEADS >= tp_size:
        if GLOBAL_NUM_KV_HEADS % tp_size:
            raise ValueError(
                f"tp_size={tp_size} cannot partition {GLOBAL_NUM_KV_HEADS} KV heads."
            )
    elif tp_size % GLOBAL_NUM_KV_HEADS:
        raise ValueError(
            f"tp_size={tp_size} cannot replicate {GLOBAL_NUM_KV_HEADS} KV heads."
        )
    return GLOBAL_NUM_Q_HEADS // tp_size, max(1, GLOBAL_NUM_KV_HEADS // tp_size)


def topology_width(topology: str) -> int | None:
    if topology in ("dense", "ragged_prefill"):
        return None
    if topology in ("verify_d2", "graph_d2"):
        return 2
    if topology in ("verify_d3", "graph_d3"):
        return 3
    raise ValueError(f"unknown topology: {topology!r}")


@dataclass(frozen=True)
class AttentionCase:
    attention: str
    topology: str
    m: int
    kv_length: int
    length_pattern: str = "uniform"
    table_width: str = "compact"
    bucket_batch_size: int = 0
    prefill_batch_size: int = 0
    tp_size: int = DEFAULT_TP_SIZE

    def __post_init__(self) -> None:
        if self.attention not in ("full", "swa"):
            raise ValueError(f"attention must be full or swa, got {self.attention!r}")
        width = topology_width(self.topology)
        if not M_MIN <= self.m <= M_MAX:
            raise ValueError(f"M must be in [{M_MIN}, {M_MAX}], got {self.m}")
        if self.kv_length < 1:
            raise ValueError("kv_length must be positive")
        if self.length_pattern not in ("uniform", "ragged"):
            raise ValueError("length_pattern must be uniform or ragged")
        if self.table_width not in ("compact", "graph"):
            raise ValueError("table_width must be compact or graph")
        if self.topology == "dense" and self.prefill_batch_size not in (0, 1):
            raise ValueError("dense topology contains exactly one request")
        if self.topology == "ragged_prefill":
            if not 2 <= self.prefill_batch_size <= self.m:
                raise ValueError(
                    "ragged_prefill requires 2 <= prefill_batch_size <= M"
                )
        elif self.prefill_batch_size:
            raise ValueError(
                "prefill_batch_size is valid only for ragged_prefill"
            )
        if width is not None and self.m % width:
            raise ValueError(
                f"{self.topology} is a fixed D={width} production topology; "
                f"M={self.m} must be divisible by D."
            )
        if self.topology.startswith("graph_"):
            if self.bucket_batch_size < self.real_batch_size:
                raise ValueError(
                    "graph bucket_batch_size must be at least the real batch size"
                )
            if self.table_width != "graph":
                raise ValueError(
                    "graph topologies require the production-width graph block table"
                )
        elif self.bucket_batch_size:
            raise ValueError("bucket_batch_size is valid only for graph topologies")
        local_head_counts(self.tp_size)

    @property
    def draft_width(self) -> int | None:
        return topology_width(self.topology)

    @property
    def real_batch_size(self) -> int:
        if self.topology == "ragged_prefill":
            return self.prefill_batch_size
        return 1 if self.draft_width is None else self.m // self.draft_width

    @property
    def scheduled_batch_size(self) -> int:
        return self.bucket_batch_size or self.real_batch_size

    @property
    def q_buffer_rows(self) -> int:
        """Static Q/O row count seen by the public wrapper.

        Graph replay keeps the captured ``Bcap x D`` model buffer even when
        only the first ``M`` rows belong to live requests.  Eager paths have no
        such tail and therefore use exactly ``M`` rows.
        """

        if self.topology.startswith("graph_"):
            assert self.draft_width is not None
            return self.scheduled_batch_size * self.draft_width
        return self.m

    @property
    def local_num_q_heads(self) -> int:
        return local_head_counts(self.tp_size)[0]

    @property
    def local_num_kv_heads(self) -> int:
        return local_head_counts(self.tp_size)[1]

    @property
    def runtime_q_lens(self) -> tuple[int, ...]:
        if self.topology == "ragged_prefill":
            # Deterministic positive, non-uniform request lengths whose exact
            # sum is M.  Rotating the remainder prevents request 0 from always
            # being the unique longest request.
            batch_size = self.prefill_batch_size
            base, remainder = divmod(self.m, batch_size)
            real = [base] * batch_size
            for index in range(remainder):
                real[(index * 5 + 1) % batch_size] += 1
        elif self.draft_width is None:
            real = [self.m]
        else:
            real = [self.draft_width] * self.real_batch_size
        return tuple(real + [0] * (self.scheduled_batch_size - len(real)))

    @property
    def capture_q_lens(self) -> tuple[int, ...]:
        if not self.topology.startswith("graph_"):
            return self.runtime_q_lens
        assert self.draft_width is not None
        return (self.draft_width,) * self.scheduled_batch_size

    @property
    def runtime_kv_lens(self) -> tuple[int, ...]:
        result: list[int] = []
        for request_id, q_len in enumerate(self.runtime_q_lens):
            if q_len == 0:
                # TARGET_VERIFY graph replay adds D to every static seq-len
                # slot, including padded requests. q_len=0 makes the value
                # numerically irrelevant today, but preserving D prevents a
                # future dispatch from learning the wrong production shape.
                result.append(self.draft_width or 0)
                continue
            if self.length_pattern == "uniform":
                length = self.kv_length
            else:
                # Exercise page/window tails without turning the configured
                # maximum into an unrelated tiny-context workload.
                variation = (request_id * 37 + 17) % (4 * PAGE_SIZE + 1)
                length = self.kv_length - variation
            result.append(max(q_len, length))
        return tuple(result)

    @property
    def capture_kv_lens(self) -> tuple[int, ...]:
        if not self.topology.startswith("graph_"):
            return self.runtime_kv_lens
        assert self.draft_width is not None
        # Decode Graph capture fills pre-verify seq_lens with zero. WeLM target
        # verify then adds D, so the captured post-write KV length is exactly D;
        # replay updates the same device tensor to the live (possibly 26K+) KV.
        return (self.draft_width,) * self.scheduled_batch_size

    @property
    def block_table_width(self) -> int:
        if self.table_width == "graph":
            return GRAPH_BLOCK_TABLE_WIDTH
        max_len = max(self.runtime_kv_lens, default=0)
        return max(1, math.ceil(max_len / PAGE_SIZE))

    @property
    def name(self) -> str:
        pieces = [
            self.attention,
            self.topology,
            f"m{self.m}",
            f"kv{self.kv_length}",
            self.length_pattern,
            self.table_width,
            f"tp{self.tp_size}",
        ]
        if self.bucket_batch_size:
            pieces.insert(2, f"bcap{self.bucket_batch_size}")
        elif self.prefill_batch_size:
            pieces.insert(2, f"b{self.prefill_batch_size}")
        return "_".join(pieces)

    def as_record(self) -> dict[str, object]:
        return {
            "case": self.name,
            "attention": self.attention,
            "topology": self.topology,
            "m": self.m,
            "q_buffer_rows": self.q_buffer_rows,
            "draft_width": self.draft_width or "",
            "real_batch_size": self.real_batch_size,
            "scheduled_batch_size": self.scheduled_batch_size,
            "kv_length": self.kv_length,
            "min_runtime_kv_length": min(
                (x for x in self.runtime_kv_lens if x), default=0
            ),
            "length_pattern": self.length_pattern,
            "table_width_mode": self.table_width,
            "block_table_width": self.block_table_width,
            "tp_size": self.tp_size,
            "num_q_heads": self.local_num_q_heads,
            "num_kv_heads": self.local_num_kv_heads,
            "page_size": PAGE_SIZE,
            "head_dim": HEAD_DIM,
            "softmax_scale": SOFTMAX_SCALE,
            "swa_left_window": SWA_LEFT_WINDOW if self.attention == "swa" else "",
        }


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _case_from_spec(spec: dict, *, tp_size: int) -> AttentionCase:
    topology = str(spec["topology"])
    width = topology_width(topology)
    if "m" in spec:
        m = int(spec["m"])
    elif "batch_size" in spec:
        if width is None:
            raise ValueError("batch_size requires a verify topology")
        m = int(spec["batch_size"]) * width
    elif "real_batch_size" in spec:
        if width is None:
            raise ValueError("real_batch_size requires a graph topology")
        m = int(spec["real_batch_size"]) * width
    else:
        raise ValueError(f"case has no M or batch size: {spec}")
    return AttentionCase(
        attention=str(spec["attention"]),
        topology=topology,
        m=m,
        kv_length=int(spec["kv_length"]),
        length_pattern=str(spec.get("length_pattern", "uniform")),
        table_width=str(spec.get("table_width", "compact")),
        bucket_batch_size=int(spec.get("bucket_batch_size", 0)),
        prefill_batch_size=int(spec.get("prefill_batch_size", 0)),
        tp_size=tp_size,
    )


def _expand_group(group: dict, *, tp_size: int) -> Iterator[AttentionCase]:
    attentions = _as_list(group["attention"])
    topology = str(group["topology"])
    width = topology_width(topology)
    kv_lengths = _as_list(group.get("kv_lengths", group.get("kv_length")))
    if kv_lengths == [None]:
        raise ValueError(f"group has no kv_length: {group}")

    if "m_values" in group:
        m_values = [int(value) for value in group["m_values"]]
    elif "batch_sizes" in group:
        if width is None:
            raise ValueError("batch_sizes requires a verify topology")
        m_values = [int(value) * width for value in group["batch_sizes"]]
    elif "real_batch_sizes" in group:
        if width is None:
            raise ValueError("real_batch_sizes requires a graph topology")
        m_values = [int(value) * width for value in group["real_batch_sizes"]]
    else:
        raise ValueError(f"group has no M axis: {group}")

    for attention in attentions:
        for kv_length in kv_lengths:
            for m in m_values:
                spec = {
                    **group,
                    "attention": attention,
                    "m": m,
                    "kv_length": kv_length,
                }
                yield _case_from_spec(spec, tp_size=tp_size)


def deduplicate_cases(cases: Iterable[AttentionCase]) -> list[AttentionCase]:
    return list(dict.fromkeys(cases))


def suite_cases(
    suite: str,
    phase: str,
    *,
    tp_size: int = DEFAULT_TP_SIZE,
) -> list[AttentionCase]:
    try:
        suite_config = CONFIG["suites"][suite]
    except KeyError as exc:
        raise ValueError(f"unknown suite {suite!r}") from exc

    direct = suite_config.get(phase, [])
    groups = suite_config.get(f"{phase}_groups", [])
    cases = [_case_from_spec(spec, tp_size=tp_size) for spec in direct]
    for group in groups:
        cases.extend(_expand_group(group, tp_size=tp_size))
    return deduplicate_cases(cases)


def parse_int_set(spec: str, *, minimum: int, maximum: int) -> list[int]:
    """Parse ``1,4,8:16,32:64:8`` into a sorted unique integer list."""
    values: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            values.add(int(parts[0]))
            continue
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid integer range: {item!r}")
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0 or end < start:
            raise ValueError(f"invalid integer range: {item!r}")
        values.update(range(start, end + 1, step))
    invalid = sorted(value for value in values if not minimum <= value <= maximum)
    if invalid:
        raise ValueError(
            f"values must be in [{minimum}, {maximum}], got {invalid[:10]}"
        )
    return sorted(values)


def make_manual_cases(
    *,
    attentions: Sequence[str],
    topologies: Sequence[str],
    m_values: Sequence[int],
    kv_lengths: Sequence[int],
    length_pattern: str,
    table_width: str,
    bucket_batch_size: int,
    prefill_batch_size: int,
    tp_size: int,
) -> list[AttentionCase]:
    cases: list[AttentionCase] = []
    for attention in attentions:
        for topology in topologies:
            width = topology_width(topology)
            for m in m_values:
                # A range such as 1:1024 remains convenient for fixed-D
                # topologies: non-production totals are skipped, not rounded.
                if width is not None and m % width:
                    continue
                if topology == "ragged_prefill" and m < 2:
                    continue
                for kv_length in kv_lengths:
                    cases.append(
                        AttentionCase(
                            attention=attention,
                            topology=topology,
                            m=m,
                            kv_length=kv_length,
                            length_pattern=length_pattern,
                            table_width=(
                                "graph"
                                if topology.startswith("graph_")
                                else table_width
                            ),
                            bucket_batch_size=(
                                bucket_batch_size or (m // int(width))
                                if topology.startswith("graph_")
                                else 0
                            ),
                            prefill_batch_size=(
                                min(prefill_batch_size, m)
                                if topology == "ragged_prefill"
                                else 0
                            ),
                            tp_size=tp_size,
                        )
                    )
    return deduplicate_cases(cases)


def find_case(name: str, *, tp_size: int = DEFAULT_TP_SIZE) -> AttentionCase:
    for suite in CONFIG["suites"]:
        for phase in ("correctness", "performance", "graph", "ir", "profile"):
            for case in suite_cases(suite, phase, tp_size=tp_size):
                if case.name == name:
                    return case
    pattern = re.compile(
        r"^(full|swa)_(dense|ragged_prefill|verify_d[23]|graph_d[23])_"
        r"(?:(?:bcap([0-9]+))|(?:b([0-9]+)))?_?m([0-9]+)_kv([0-9]+)_"
        r"(uniform|ragged)_(compact|graph)_tp([0-9]+)$"
    )
    match = pattern.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid or unknown case name: {name}")
    (
        attention,
        topology,
        bucket,
        prefill_batch,
        m,
        kv,
        length_pattern,
        table_width,
        encoded_tp,
    ) = match.groups()
    encoded_tp_int = int(encoded_tp)
    if encoded_tp_int != tp_size:
        raise ValueError(
            f"case name encodes tp{encoded_tp_int}, but --tp-size is {tp_size}"
        )
    return AttentionCase(
        attention=attention,
        topology=topology,
        m=int(m),
        kv_length=int(kv),
        length_pattern=length_pattern,
        table_width=table_width,
        bucket_batch_size=int(bucket or 0),
        prefill_batch_size=int(prefill_batch or 0),
        tp_size=tp_size,
    )
