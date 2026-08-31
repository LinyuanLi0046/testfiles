"""Production-shaped paged KV inputs for the standalone NPU benchmark."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import torch

from dp_attention_contract import AttentionCase, HEAD_DIM, PAGE_SIZE, SWA_LEFT_WINDOW


@dataclass
class AttentionInputs:
    q: torch.Tensor
    key_native: torch.Tensor
    value_native: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    sinks: torch.Tensor
    block_table: torch.Tensor
    runtime_cu_q_lens: torch.Tensor
    runtime_kv_lens: torch.Tensor
    capture_cu_q_lens_cpu: torch.Tensor
    capture_kv_lens_cpu: torch.Tensor


def _case_seed(case: AttentionCase, seed: int) -> int:
    digest = hashlib.sha256(case.name.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:4], "little")


def _prefix_sum(values: tuple[int, ...], *, device: torch.device | str) -> torch.Tensor:
    # NEWSGLANG constructs this metadata on CPU and then copies it to NPU.
    tensor = torch.tensor(values, dtype=torch.int32, device="cpu")
    result = torch.nn.functional.pad(
        torch.cumsum(tensor, dim=0, dtype=torch.int32), (1, 0)
    )
    return result.to(device=device)


def _retained_logical_pages(case: AttentionCase, q_len: int, kv_len: int) -> range:
    num_pages = math.ceil(kv_len / PAGE_SIZE)
    if case.attention == "full":
        return range(num_pages)

    # Across all query rows, the union of visible SWA tokens starts 511 tokens
    # before the first query.  Pages older than that are evicted in the real
    # hybrid SWA pool and map to a harmless poison page in this harness.
    first_query_position = kv_len - q_len
    first_visible_position = max(0, first_query_position - SWA_LEFT_WINDOW)
    first_page = first_visible_position // PAGE_SIZE
    return range(first_page, num_pages)


@torch.no_grad()
def make_inputs(
    case: AttentionCase,
    device: torch.device,
    *,
    seed: int,
) -> AttentionInputs:
    local_seed = _case_seed(case, seed)
    torch.manual_seed(local_seed)
    try:
        torch.npu.manual_seed_all(local_seed)
    except AttributeError:
        pass

    q_lens = case.runtime_q_lens
    kv_lens = case.runtime_kv_lens
    num_kv_heads = case.local_num_kv_heads

    retained_by_request: list[list[int]] = []
    total_retained_pages = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        retained = list(_retained_logical_pages(case, q_len, kv_len)) if q_len else []
        retained_by_request.append(retained)
        total_retained_pages += len(retained)

    # Page zero is a valid, allocated poison page.  Never use -1: an accidental
    # read should become a numerical failure, not an asynchronous NPU OOB kill.
    num_physical_pages = 1 + total_retained_pages
    physical_ids = list(range(1, num_physical_pages))
    random.Random(local_seed).shuffle(physical_ids)
    physical_iter = iter(physical_ids)

    block_table_cpu = torch.zeros(
        (case.scheduled_batch_size, case.block_table_width), dtype=torch.int32
    )
    for request_id, retained_pages in enumerate(retained_by_request):
        required_pages = math.ceil(kv_lens[request_id] / PAGE_SIZE) if q_lens[request_id] else 0
        if required_pages > case.block_table_width:
            raise ValueError(
                f"case {case.name}: block table width {case.block_table_width} "
                f"cannot hold {required_pages} logical pages"
            )
        for logical_page in retained_pages:
            block_table_cpu[request_id, logical_page] = next(physical_iter)

    # This is the native pool layout used by NEWSGLANG: [slot,Hkv,D].  The
    # public kernels receive the zero-copy [page,Hkv,64,D] permuted view.
    native_shape = (num_physical_pages * PAGE_SIZE, num_kv_heads, HEAD_DIM)
    key_native = torch.randn(native_shape, dtype=torch.bfloat16, device=device)
    value_native = torch.randn(native_shape, dtype=torch.bfloat16, device=device)
    key_native[:PAGE_SIZE].fill_(7.0)
    value_native[:PAGE_SIZE].fill_(-5.0)
    key_cache = key_native.view(
        num_physical_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM
    ).permute(0, 2, 1, 3)
    value_cache = value_native.view(
        num_physical_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM
    ).permute(0, 2, 1, 3)

    q = torch.randn(
        (case.q_buffer_rows, case.local_num_q_heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    sinks = torch.linspace(
        -1.25,
        1.25,
        case.local_num_q_heads,
        dtype=torch.float32,
        device=device,
    )

    return AttentionInputs(
        q=q,
        key_native=key_native,
        value_native=value_native,
        key_cache=key_cache,
        value_cache=value_cache,
        sinks=sinks,
        block_table=block_table_cpu.to(device=device),
        runtime_cu_q_lens=_prefix_sum(q_lens, device=device),
        runtime_kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        capture_cu_q_lens_cpu=_prefix_sum(case.capture_q_lens, device="cpu"),
        capture_kv_lens_cpu=torch.tensor(
            case.capture_kv_lens, dtype=torch.int32, device="cpu"
        ),
    )

