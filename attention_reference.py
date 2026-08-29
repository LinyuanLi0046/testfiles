"""Chunked FP32 oracle for WeLMv4 Full/SWA paged Attention with sinks."""

from __future__ import annotations

import torch

from attention_contract import PAGE_SIZE, SOFTMAX_SCALE, SWA_LEFT_WINDOW


@torch.no_grad()
def reference_prefill_attention(
    *,
    attention: str,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    runtime_q_lens: tuple[int, ...],
    runtime_kv_lens: tuple[int, ...],
    block_table: torch.Tensor,
    sinks: torch.Tensor,
    query_chunk_size: int = 32,
) -> torch.Tensor:
    """Evaluate the exact post-KV-write causal contract in FP32.

    ``key_cache`` and ``value_cache`` use the public production view
    ``[page, kv_head, 64, 256]``.  The oracle intentionally supports Hkv > 1
    even though the primary TP4 deployment has one local KV head.
    """

    if attention not in ("full", "swa"):
        raise ValueError(f"unknown attention kind: {attention}")
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("unexpected Q/K/V rank or mismatched paged caches")
    if len(runtime_q_lens) != len(runtime_kv_lens):
        raise ValueError("runtime Q/KV length vectors must have equal size")
    real_q_rows = sum(runtime_q_lens)
    if real_q_rows > q.shape[0]:
        raise ValueError(
            f"Q buffer is shorter than runtime q_lens: rows={q.shape[0]}, "
            f"real_rows={real_q_rows}"
        )

    num_q_heads = q.shape[1]
    num_kv_heads = key_cache.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError("GQA requires num_q_heads divisible by num_kv_heads")
    if sinks.shape != (num_q_heads,):
        raise ValueError("sink must contain one FP32 logit per local Q head")

    group_size = num_q_heads // num_kv_heads
    kv_head_index = torch.arange(num_q_heads, device=q.device) // group_size
    output = torch.zeros_like(q)
    q_offset = 0

    for request_id, (q_len, kv_len) in enumerate(
        zip(runtime_q_lens, runtime_kv_lens, strict=True)
    ):
        if q_len == 0:
            continue
        if kv_len < q_len:
            raise ValueError(
                f"request {request_id}: kv_len={kv_len} is smaller than q_len={q_len}"
            )
        num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        # A long-context SWA query can see only the union beginning 511 tokens
        # before its first query row.  Skip fully invisible pages in the FP32
        # oracle as the production SWA kernel does; otherwise a 26K request
        # needlessly materializes thousands of poison-page copies.
        prefix_len = kv_len - q_len
        first_visible_position = (
            max(0, prefix_len - SWA_LEFT_WINDOW) if attention == "swa" else 0
        )
        first_page = first_visible_position // PAGE_SIZE
        first_page_offset = first_visible_position - first_page * PAGE_SIZE
        page_ids = block_table[request_id, first_page:num_pages].to(torch.long)
        # [logical_page, kv_head, page_token, dim] -> [kv_token, kv_head, dim]
        keys = (
            key_cache.index_select(0, page_ids)
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, q.shape[2])[
                first_page_offset : first_page_offset
                + (kv_len - first_visible_position)
            ]
        )
        values = (
            value_cache.index_select(0, page_ids)
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, q.shape[2])[
                first_page_offset : first_page_offset
                + (kv_len - first_visible_position)
            ]
        )
        keys = keys.index_select(1, kv_head_index).float()
        values = values.index_select(1, kv_head_index).float()

        request_q = q[q_offset : q_offset + q_len]
        key_positions = torch.arange(
            first_visible_position, kv_len, device=q.device
        )

        for chunk_start in range(0, q_len, query_chunk_size):
            chunk_end = min(chunk_start + query_chunk_size, q_len)
            query = request_q[chunk_start:chunk_end].float()
            # [Q,H,D] x [K,H,D] -> [H,Q,K]
            logits = torch.einsum("qhd,khd->hqk", query, keys) * SOFTMAX_SCALE
            query_positions = prefix_len + torch.arange(
                chunk_start, chunk_end, device=q.device
            )
            visible = key_positions[None, :] <= query_positions[:, None]
            if attention == "swa":
                visible &= key_positions[None, :] >= (
                    query_positions[:, None] - SWA_LEFT_WINDOW
                )
            logits.masked_fill_(~visible.unsqueeze(0), -torch.inf)

            # The learned sink is a virtual key whose value is exactly zero.
            # It contributes to the stable softmax denominator but not to the
            # value-weighted numerator.
            sink_logits = sinks.float()[:, None]
            row_max = torch.maximum(logits.amax(dim=-1), sink_logits)
            exp_logits = torch.exp(logits - row_max[..., None])
            denominator = exp_logits.sum(dim=-1) + torch.exp(
                sink_logits - row_max
            )
            probabilities = exp_logits / denominator[..., None]
            chunk_output = torch.einsum("hqk,khd->qhd", probabilities, values)
            output[q_offset + chunk_start : q_offset + chunk_end].copy_(
                chunk_output.to(output.dtype)
            )

        q_offset += q_len

    if q_offset != real_q_rows:
        raise ValueError(
            "internal Q offset does not match runtime q_lens: "
            f"expected={real_q_rows}, actual={q_offset}"
        )
    # In an NPU Graph replay q may be the fixed Bcap*D capture buffer while
    # only the first M rows are live. output was initialized with zeros, so the
    # padding tail is also the production reference contract.
    return output


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    delta = (actual_fp32 - expected_fp32).abs()
    denominator = expected_fp32.abs().clamp_min(1.0e-6)
    flat_actual = actual_fp32.flatten()
    flat_expected = expected_fp32.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        flat_actual.unsqueeze(0), flat_expected.unsqueeze(0), dim=1
    )
    return {
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel_error": float((delta / denominator).max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "cosine_similarity": float(cosine.item()) if delta.numel() else 1.0,
        "nonfinite_count": float((~torch.isfinite(actual_fp32)).sum().item()),
    }
