"""Optimized WeLM paged prefill attention kernels for Ascend Triton.

The public wrappers in this module consume KV cache views in
``[num_pages, num_kv_heads, page_size, head_dim]`` order.  The Ascend backend
keeps the underlying allocation in its native page-major order and provides a
zero-copy permuted view before calling these wrappers.

Public dispatch wrappers:

* Full attention: :func:`paged_attention_prefill_impl`
* Sliding-window attention: :func:`swa_paged_prefill_impl`

The file intentionally excludes decode and KV-mirror kernels.  Kernel bodies
and dispatch decisions are copied from the final verified candidate so this
packaging step does not change numerical behavior or tiling.
"""

import heapq
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

__all__ = [
    "paged_attention_prefill_prepare",
    "paged_attention_prefill_impl",
    "swa_paged_prefill_impl",
]

@lru_cache(maxsize=3)
def get_num_cores(op_type="vector"):
    assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    return (
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        if op_type == "vector"
        else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
    )

def _tensor_to_cpu_list(tensor: torch.Tensor) -> List[int]:
    return [int(x) for x in tensor.detach().cpu().tolist()]


def _build_lpt_task_schedule(
        cu_q_lens: torch.Tensor,
        seqlens_kv: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        block_size_m: int,
        block_size_n: int,
        cube_num: int,
        gqa_interleave: bool,
        device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a shape-aware static schedule for the 1D Triton grid.

    Each task still computes one (batch, q block, q head). The estimated cost is
    the number of KV blocks scanned by that task. LPT keeps the kernel free of
    global atomics while avoiding the worst tail imbalance of round-robin.
    """
    cu_q_lens_host = _tensor_to_cpu_list(cu_q_lens)
    seqlens_kv_host = None if seqlens_kv is None else _tensor_to_cpu_list(seqlens_kv)
    batch_size = len(cu_q_lens_host) - 1

    weighted_tasks = []
    seq_no = 0
    for b_id in range(batch_size):
        q_seq_len = cu_q_lens_host[b_id + 1] - cu_q_lens_host[b_id]
        if q_seq_len <= 0:
            continue

        kv_seq_len = q_seq_len if seqlens_kv_host is None else seqlens_kv_host[b_id]
        kv_cache_len = kv_seq_len - q_seq_len
        if kv_cache_len < 0:
            raise ValueError(
                f"seqlens_kv[{b_id}] ({kv_seq_len}) must be >= q_seq_len ({q_seq_len})"
            )

        q_chunks = triton.cdiv(q_seq_len, block_size_m)
        for q_block_id in range(q_chunks):
            q_block_end = min((q_block_id + 1) * block_size_m, q_seq_len)
            cost = max(1, triton.cdiv(kv_cache_len + q_block_end, block_size_n))
            for q_head_id in range(num_q_heads):
                if gqa_interleave:
                    kv_head_id = q_head_id % num_kv_heads
                else:
                    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)
                weighted_tasks.append((b_id, kv_head_id, q_block_id, cost, q_head_id, seq_no))
                seq_no += 1

    heap = [(0, core_id) for core_id in range(cube_num)]
    per_core_tasks: List[List[Tuple[int, int, int]]] = [[] for _ in range(cube_num)]

    for b_id, kv_head_id, q_block_id, cost, q_head_id, seq_no in sorted(
        weighted_tasks, key=lambda task: (task[3], task[5]), reverse=True
    ):
        core_cost, core_id = heapq.heappop(heap)
        per_core_tasks[core_id].append((b_id, q_head_id, q_block_id))
        heapq.heappush(heap, (core_cost + cost, core_id))

    task_b = []
    task_q_block = []
    task_q_head = []
    core_task_offsets = [0]
    for tasks in per_core_tasks:
        for b_id, q_head_id, q_block_id in tasks:
            task_b.append(b_id)
            task_q_block.append(q_block_id)
            task_q_head.append(q_head_id)
        core_task_offsets.append(len(task_b))

    device = cu_q_lens.device if device is None else device
    return (
        torch.tensor(task_b, device=device, dtype=torch.int32),
        torch.tensor(task_q_block, device=device, dtype=torch.int32),
        torch.tensor(task_q_head, device=device, dtype=torch.int32),
        torch.tensor(core_task_offsets, device=device, dtype=torch.int32),
    )


@triton.jit
def causal_mask_fn(mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, kv_start, Q_BLOCK, KV_BLOCK):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (
        (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m
        + (mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]) * mask_stride_n
    )
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    return mask_causal


@triton.jit
def _sdpa_infer_single_block(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_T_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    qk_scale,
    mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    # -- Compute qk ----

    # Load (transposed) K block
    k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, k_T)
    # tl.extra.cann.extension.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)    # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.extra.cann.extension.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "stride_bt_batch",
    ]
)
def paged_prefill_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    task_b_ptr,
    task_q_block_ptr,
    task_q_head_ptr,
    core_task_offsets_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    stride_sink_head,
    softmax_scale,
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    task_begin = tl.load(core_task_offsets_ptr + pid)
    task_end = tl.load(core_task_offsets_ptr + pid + 1)

    for task_idx in range(task_begin, task_end):
        b_id = tl.load(task_b_ptr + task_idx)
        q_block_id = tl.load(task_q_block_ptr + task_idx)
        q_head_id = tl.load(task_q_head_ptr + task_idx)

        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc
        # Graph replay keeps a bucket-sized static task schedule. Padding
        # requests advertise q_len=0 through cu_q_lens and must not construct
        # zero-shaped block pointers from those otherwise-valid static tasks.
        if q_seq_len > 0:
            if seqlens_kv_ptr is None:
                kv_seq_len = q_seq_len
            else:
                kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
            kv_cache_len = kv_seq_len - q_seq_len

            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
            q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
            q_block_len = q_block_end_in_seq - q_block_start_in_seq

            Q_block_ptr = tl.make_block_ptr(
                base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )

            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # --- online-softmax init with optional sink ---
            m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            if SINK_ENABLED:
                s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
                m_i = m_i + s_h
                l_i = l_i + 1.0
            else:
                m_i = m_i - float("inf")
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

            num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
            num_no_mask_blocks = (kv_cache_len + q_block_start_in_seq) // BLOCK_SIZE_N
            for kv_block_id in range(0, num_no_mask_blocks):
                kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
                kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

                logical_page_id = kv_block_start_in_seq // PAGE_SIZE
                kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )

                K_T_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_infer_single_block(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_T_block_ptr,
                    V_block_ptr,
                    softmax_scale,
                    None,
                    HEAD_DIM,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    value_cache_ptr.dtype.element_ty == tl.float8e5,
                )


            for kv_block_id in range(num_no_mask_blocks, num_kv_blocks):
                kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
                kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

                logical_page_id = kv_block_start_in_seq // PAGE_SIZE
                kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )

                mask = causal_mask_fn(
                    aux_mask_ptr,
                    AUX_MASK_SIZE,
                    stride_mask_m,
                    stride_mask_n,
                    kv_cache_len + q_block_start_in_seq,
                    kv_block_start_in_seq,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )

                K_T_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )


                acc, l_i, m_i = _sdpa_infer_single_block(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_T_block_ptr,
                    V_block_ptr,
                    softmax_scale,
                    mask,
                    HEAD_DIM,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    value_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            m_i += tl.math.log(l_i)
            accumulator = acc / l_i[:, None]

            # NOTE(zhangjihang): for training
            # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
            # tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "stride_bt_batch",
    ]
)
def paged_prefill_page_aggregation_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    task_b_ptr,
    task_q_block_ptr,
    task_q_head_ptr,
    core_task_offsets_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    stride_sink_head,
    softmax_scale,
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    PAGE_AGGREGATION_NUM: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    pid = tl.program_id(0)

    task_begin = tl.load(core_task_offsets_ptr + pid)
    task_end = tl.load(core_task_offsets_ptr + pid + 1)

    for task_idx in range(task_begin, task_end):
        b_id = tl.load(task_b_ptr + task_idx)
        q_block_id = tl.load(task_q_block_ptr + task_idx)
        q_head_id = tl.load(task_q_head_ptr + task_idx)

        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc
        # The capture-time task schedule also contains replay-time padding
        # requests, whose q length is represented as zero.
        if q_seq_len > 0:
            if seqlens_kv_ptr is None:
                kv_seq_len = q_seq_len
            else:
                kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
            kv_cache_len = kv_seq_len - q_seq_len

            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
            q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
            q_block_len = q_block_end_in_seq - q_block_start_in_seq

            Q_block_ptr = tl.make_block_ptr(
                base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )

            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # --- online-softmax init with optional sink ---
            m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            if SINK_ENABLED:
                s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
                m_i = m_i + s_h
                l_i = l_i + 1.0
            else:
                m_i = m_i - float("inf")
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

            num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
            for kv_block_id in range(0, num_kv_blocks, PAGE_AGGREGATION_NUM):
                mask = causal_mask_fn(
                    aux_mask_ptr,
                    AUX_MASK_SIZE,
                    stride_mask_m,
                    stride_mask_n,
                    kv_cache_len + q_block_start_in_seq,
                    kv_block_id * BLOCK_SIZE_N,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N * PAGE_AGGREGATION_NUM,
                )


                # Load (transposed) K block
                k = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=key_cache_ptr.dtype.element_ty)
                for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                    kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                    kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                    kv_block_len = max(kv_block_end - kv_block_start, 0)
                    logical_page_id = min(kv_block_start // PAGE_SIZE, stride_bt_batch - 1)
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    physical_page_id = tl.load(
                        block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                    )
                    cur_k_block_ptr = tl.make_block_ptr(
                        base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                        shape=(kv_block_len, HEAD_DIM),
                        strides=(stride_k_blksz, stride_k_dim),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                        order=(1, 0),
                    )
                    k_slice = tl.load(cur_k_block_ptr, boundary_check=(
                        0, 1), padding_option="zero")
                    k = tl.extra.cann.extension.insert_slice(k, k_slice, offsets=(page_iter * BLOCK_SIZE_N, 0),
                                                             sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                                                             strides=(1, 1))
                k_T = tl.trans(k)
                qk = tl.dot(q, k_T)
                # tl.compile_hint(qk, "tile_cube_loop")

                qk = qk * softmax_scale
                if mask is not None:
                    qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)  # Scaled max
                qk = qk - m_ij[:, None]  # Stabilize

                # Softmax weights p = exp(qk)
                p = tl.math.exp(qk)

                p_cast = p.to(k_T.dtype)

                # Softmax denominator (sum of each row)
                l_ij = tl.sum(p, 1)
                # -- Update m_i and l_i
                alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
                l_i = l_i * alpha + l_ij  # Update softmax denominator
                # -- Update output accumulator --
                acc = acc * alpha[:, None]
                # Load corresponding V block
                v = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=value_cache_ptr.dtype.element_ty)
                for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                    kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                    kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                    kv_block_len = max(kv_block_end - kv_block_start, 0)
                    logical_page_id = min(kv_block_start // PAGE_SIZE, stride_bt_batch - 1)
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    physical_page_id = tl.load(
                        block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                    )
                    cur_v_block_ptr = tl.make_block_ptr(
                        base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                        shape=(kv_block_len, HEAD_DIM),
                        strides=(stride_v_blksz, stride_v_dim),
                        offsets=(0, 0),
                        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                        order=(1, 0),
                    )
                    v_slice = tl.load(cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    v = tl.extra.cann.extension.insert_slice(v, v_slice, offsets=(page_iter * BLOCK_SIZE_N, 0),
                                                             sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                                                             strides=(1, 1))
                acc = tl.dot(p_cast, v, acc)
                # tl.compile_hint(acc_ptr, "tile_cube_loop")

                # Update current block max
                m_i = m_ij

            m_i += tl.math.log(l_i)
            accumulator = acc / l_i[:, None]

            # NOTE(zhangjihang): for training
            # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
            # tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "batch_size",
        "stride_bt_batch",
    ]
)
def paged_prefill_small_q_grouped_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    batch_size,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_Q_LEN: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Group up to four query tokens and all six Q heads per request."""
    tl.static_assert(
        NUM_Q_HEADS == 6,
        "grouped WeLM small-Q prefill requires six local Q heads",
    )
    tl.static_assert(
        MAX_Q_LEN >= 1 and MAX_Q_LEN <= 4,
        "grouped WeLM small-Q prefill max query length must be in [1, 4]",
    )
    tl.static_assert(
        PAGE_SIZE == BLOCK_SIZE_N,
        "grouped WeLM small-Q prefill uses one page per KV tile",
    )
    GROUPED_ROWS: tl.constexpr = NUM_Q_HEADS * MAX_Q_LEN
    tl.static_assert(
        BLOCK_SIZE_M >= GROUPED_ROWS,
        "grouped WeLM small-Q prefill must pad all token/head rows",
    )
    tl.static_assert(
        BLOCK_SIZE_M == 16 or BLOCK_SIZE_M == 32,
        "grouped WeLM small-Q prefill uses Cube-aligned M=16/32",
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    b_begin = pid * batch_size // n_programs
    b_end = (pid + 1) * batch_size // n_programs

    for b_id in range(b_begin, b_end):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        if q_seq_len.to(tl.float32) > 0.0:
            row_ids = tl.arange(0, BLOCK_SIZE_M)
            row_tokens = row_ids // NUM_Q_HEADS
            row_heads = row_ids - row_tokens * NUM_Q_HEADS
            valid_rows = (row_ids < GROUPED_ROWS) & (
                row_tokens.to(tl.float32) < q_seq_len.to(tl.float32)
            )
            dim_offsets = tl.arange(0, BLOCK_SIZE_D)

            q_ptrs = (
                q_ptr
                + q_start * stride_qt
                + row_ids[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=valid_rows[:, None], other=0.0)

            if SINK_ENABLED:
                sink0 = tl.load(sinks_ptr).to(tl.float32)
                sink1 = tl.load(sinks_ptr + stride_sink_head).to(tl.float32)
                sink2 = tl.load(sinks_ptr + 2 * stride_sink_head).to(tl.float32)
                sink3 = tl.load(sinks_ptr + 3 * stride_sink_head).to(tl.float32)
                sink4 = tl.load(sinks_ptr + 4 * stride_sink_head).to(tl.float32)
                sink5 = tl.load(sinks_ptr + 5 * stride_sink_head).to(tl.float32)
                row_sink = tl.where(
                    row_heads < 1,
                    sink0,
                    tl.where(
                        row_heads < 2,
                        sink1,
                        tl.where(
                            row_heads < 3,
                            sink2,
                            tl.where(
                                row_heads < 4,
                                sink3,
                                tl.where(row_heads < 5, sink4, sink5),
                            ),
                        ),
                    ),
                )
                m_i = tl.where(valid_rows, row_sink, -float("inf"))
                l_i = tl.where(valid_rows, 1.0, 0.0).to(tl.float32)
            else:
                m_i = tl.full(
                    (BLOCK_SIZE_M,), -float("inf"), tl.float32
                )
                l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            acc = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32
            )

            kv_seq_len = tl.load(seqlens_kv_ptr + b_id).to(tl.int32)
            kv_cache_len = kv_seq_len - q_seq_len
            num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)
            for kv_block_id in range(0, num_kv_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(
                    kv_block_start + BLOCK_SIZE_N, kv_seq_len
                )
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = min(
                    kv_block_start // PAGE_SIZE,
                    stride_bt_batch - 1,
                )
                physical_page_id = tl.load(
                    block_tables_ptr
                    + b_id * stride_bt_batch
                    + logical_page_id * stride_bt_block
                )

                k_t_block_ptr = tl.make_block_ptr(
                    base=(
                        key_cache_ptr
                        + physical_page_id * stride_k_block
                    ),
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=(
                        value_cache_ptr
                        + physical_page_id * stride_v_block
                    ),
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                k_t = tl.load(
                    k_t_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                v = tl.load(
                    v_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )

                key_offsets = tl.arange(0, BLOCK_SIZE_N)
                key_positions = kv_block_start + key_offsets
                key_valid = (
                    key_positions.to(tl.float32)
                    < kv_seq_len.to(tl.float32)
                )
                query_positions = kv_cache_len + row_tokens
                mask = (
                    valid_rows[:, None]
                    & key_valid[None, :]
                    & (
                        key_positions[None, :].to(tl.float32)
                        <= query_positions[:, None].to(tl.float32)
                    )
                )
                qk = tl.dot(q, k_t) * softmax_scale
                qk = tl.where(mask, qk, -1e6)
                m_ij = tl.maximum(
                    m_i,
                    tl.max(qk, 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                p = tl.math.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(k_t.dtype), v, acc)
                m_i = m_ij

            o_ptrs = (
                o_ptr
                + q_start * stride_ot
                + row_ids[:, None] * stride_oh
                + dim_offsets[None, :] * stride_od
            )
            tl.store(
                o_ptrs,
                (acc / l_i[:, None]).to(o_ptr.type.element_ty),
                mask=valid_rows[:, None],
            )


@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "batch_size",
        "num_q_blocks",
        "stride_bt_batch",
    ]
)
def paged_prefill_mid_q_grouped_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    sinks_ptr,
    batch_size,
    num_q_blocks,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    NUM_HEAD_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    Q_BUCKET: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Run fixed M5-M128 tiles with grouped GQA heads and M64 Q blocks."""
    tl.static_assert(
        NUM_Q_HEADS == 6,
        "mid-Q WeLM prefill requires six local Q heads",
    )
    tl.static_assert(
        HEADS_PER_GROUP * NUM_HEAD_GROUPS == NUM_Q_HEADS,
        "mid-Q head groups must cover all local Q heads",
    )
    GROUPED_ROWS: tl.constexpr = Q_BUCKET * HEADS_PER_GROUP
    tl.static_assert(
        GROUPED_ROWS <= BLOCK_SIZE_M,
        "mid-Q grouped rows must fit the fixed M tile",
    )
    tl.static_assert(
        GROUPED_ROWS == 48 or GROUPED_ROWS == 64,
        "mid-Q buckets use 48 or 64 logical rows",
    )
    tl.static_assert(
        BLOCK_SIZE_M == 64,
        "mid-Q WeLM prefill uses one M64 Cube tile",
    )
    tl.static_assert(
        PAGE_SIZE == BLOCK_SIZE_N,
        "mid-Q WeLM prefill uses one page per KV tile",
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_tasks = batch_size * num_q_blocks * NUM_HEAD_GROUPS
    task_begin = pid * num_tasks // n_programs
    task_end = (pid + 1) * num_tasks // n_programs

    for task_id in range(task_begin, task_end):
        q_block_task = task_id // NUM_HEAD_GROUPS
        head_group_id = task_id - q_block_task * NUM_HEAD_GROUPS
        b_id = q_block_task // num_q_blocks
        q_block_id = q_block_task - b_id * num_q_blocks
        q_block_start = q_block_id * Q_BUCKET
        head_base = head_group_id * HEADS_PER_GROUP
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        if q_block_start.to(tl.float32) < q_seq_len.to(tl.float32):
            row_ids = tl.arange(0, BLOCK_SIZE_M)
            local_row_tokens = row_ids // HEADS_PER_GROUP
            row_tokens = q_block_start + local_row_tokens
            local_heads = (
                row_ids - local_row_tokens * HEADS_PER_GROUP
            )
            q_head_ids = head_base + local_heads
            valid_rows = (
                row_ids.to(tl.float32) < GROUPED_ROWS
            ) & (
                row_tokens.to(tl.float32) < q_seq_len.to(tl.float32)
            )
            dim_offsets = tl.arange(0, BLOCK_SIZE_D)
            q_ptrs = (
                q_ptr
                + (q_start + row_tokens[:, None]) * stride_qt
                + q_head_ids[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=valid_rows[:, None], other=0.0)

            if SINK_ENABLED:
                sink0 = tl.load(sinks_ptr).to(tl.float32)
                sink1 = tl.load(sinks_ptr + stride_sink_head).to(tl.float32)
                sink2 = tl.load(
                    sinks_ptr + 2 * stride_sink_head
                ).to(tl.float32)
                sink3 = tl.load(
                    sinks_ptr + 3 * stride_sink_head
                ).to(tl.float32)
                sink4 = tl.load(
                    sinks_ptr + 4 * stride_sink_head
                ).to(tl.float32)
                sink5 = tl.load(
                    sinks_ptr + 5 * stride_sink_head
                ).to(tl.float32)
                q_head_ids_fp32 = q_head_ids.to(tl.float32)
                row_sink = tl.where(
                    q_head_ids_fp32 < 1.0,
                    sink0,
                    tl.where(
                        q_head_ids_fp32 < 2.0,
                        sink1,
                        tl.where(
                            q_head_ids_fp32 < 3.0,
                            sink2,
                            tl.where(
                                q_head_ids_fp32 < 4.0,
                                sink3,
                                tl.where(
                                    q_head_ids_fp32 < 5.0,
                                    sink4,
                                    sink5,
                                ),
                            ),
                        ),
                    ),
                )
                m_i = tl.where(valid_rows, row_sink, -float("inf"))
                l_i = tl.where(valid_rows, 1.0, 0.0).to(tl.float32)
            else:
                m_i = tl.full(
                    (BLOCK_SIZE_M,), -float("inf"), tl.float32
                )
                l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            acc = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32
            )

            kv_seq_len = tl.load(seqlens_kv_ptr + b_id).to(tl.int32)
            kv_cache_len = kv_seq_len - q_seq_len
            q_block_end = min(q_block_start + Q_BUCKET, q_seq_len)
            visible_kv_end = kv_cache_len + q_block_end
            num_kv_blocks = tl.cdiv(visible_kv_end, BLOCK_SIZE_N)
            for kv_block_id in range(0, num_kv_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(
                    kv_block_start + BLOCK_SIZE_N, kv_seq_len
                )
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = min(
                    kv_block_start // PAGE_SIZE,
                    stride_bt_batch - 1,
                )
                physical_page_id = tl.load(
                    block_tables_ptr
                    + b_id * stride_bt_batch
                    + logical_page_id * stride_bt_block
                )
                k_t_block_ptr = tl.make_block_ptr(
                    base=(
                        key_cache_ptr
                        + physical_page_id * stride_k_block
                    ),
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=(
                        value_cache_ptr
                        + physical_page_id * stride_v_block
                    ),
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                k_t = tl.load(
                    k_t_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                v = tl.load(
                    v_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                key_offsets = tl.arange(0, BLOCK_SIZE_N)
                key_positions = kv_block_start + key_offsets
                key_valid = (
                    key_positions.to(tl.float32)
                    < kv_seq_len.to(tl.float32)
                )
                query_positions = kv_cache_len + row_tokens
                mask = (
                    valid_rows[:, None]
                    & key_valid[None, :]
                    & (
                        key_positions[None, :].to(tl.float32)
                        <= query_positions[:, None].to(tl.float32)
                    )
                )
                qk = tl.dot(q, k_t) * softmax_scale
                qk = tl.where(mask, qk, -1e6)
                m_ij = tl.maximum(
                    m_i,
                    tl.max(qk, 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                p = tl.math.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(k_t.dtype), v, acc)
                m_i = m_ij

            o_ptrs = (
                o_ptr
                + (q_start + row_tokens[:, None]) * stride_ot
                + q_head_ids[:, None] * stride_oh
                + dim_offsets[None, :] * stride_od
            )
            tl.store(
                o_ptrs,
                (acc / l_i[:, None]).to(o_ptr.type.element_ty),
                mask=valid_rows[:, None],
            )


def paged_attention_prefill_prepare(
    cu_q_lens,
    seqlens_kv,
    num_q_heads,
    num_kv_heads,
    gqa_interleave,
    page_size,
    device=None,
):
    cube_num = get_num_cores("cube")
    CHUNK_SIZE = 128
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))

    task_b, task_q_block, task_q_head, core_task_offsets = _build_lpt_task_schedule(
        cu_q_lens,
        seqlens_kv,
        num_q_heads,
        num_kv_heads,
        CHUNK_SIZE,
        BLOCK_SIZE_N,
        cube_num,
        gqa_interleave,
        device,
    )
    return task_b, task_q_block, task_q_head, core_task_offsets

def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    task_b: Optional[torch.Tensor],
    task_q_block: Optional[torch.Tensor],
    task_q_head: Optional[torch.Tensor],
    core_task_offsets: Optional[torch.Tensor],
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert task_b is not None
    assert task_q_block is not None
    assert task_q_head is not None
    assert core_task_offsets is not None

    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape
    batch_size = cu_q_lens.shape[0] - 1

    assert value_cache.shape == key_cache.shape
    assert head_dim == head_dim_cache
    assert num_q_heads % num_kv_heads == 0
    assert block_tables.shape[0] == batch_size

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if aux_mask is None:
        aux_mask = torch.ones(
            1024, 1024 * 3, device=q.device, dtype=torch.bool
        ).tril(1024)

    # --- sink setup ---
    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
    sinks_pass = sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)

    # A captured bucket may replay with fewer real requests.  The static task
    # schedule then skips zero-width padding rows, so initialize their output
    # deterministically instead of exposing stale graph-buffer contents to the
    # following projection/MoE layers.
    o = torch.zeros_like(q)
    block_tables_i32 = block_tables.to(dtype=torch.int32).contiguous()

    CHUNK_SIZE = 128
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    use_grouped_small_q = (
        q.dtype != torch.float32
        and seqlens_kv is not None
        and max_q_len is not None
        and 1 <= max_q_len <= 4
        and not gqa_interleave
        and num_kv_heads == 1
        and num_q_heads == 6
        and page_size == 64
        and q.stride(0) == num_q_heads * q.stride(1)
        and o.stride(0) == num_q_heads * o.stride(1)
    )
    if use_grouped_small_q:
        grouped_grid = (min(cube_num, batch_size),)
        paged_prefill_small_q_grouped_kernel[grouped_grid](
            q,
            key_cache,
            value_cache,
            o,
            cu_q_lens,
            seqlens_kv,
            block_tables_i32,
            sinks_pass,
            batch_size,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables_i32.stride(0),
            block_tables_i32.stride(1),
            sinks_pass.stride(0),
            softmax_scale,
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_M=16 if max_q_len <= 2 else 32,
            BLOCK_SIZE_D=head_dim,
            BLOCK_SIZE_N=page_size,
            MAX_Q_LEN=max_q_len,
            SINK_ENABLED=sink_enabled,
        )
        return o

    use_grouped_mid_q = (
        q.dtype != torch.float32
        and seqlens_kv is not None
        and max_q_len is not None
        and 5 <= max_q_len <= 256
        and not gqa_interleave
        and num_kv_heads == 1
        and num_q_heads == 6
        and page_size == 64
        and q.stride(0) == num_q_heads * q.stride(1)
        and o.stride(0) == num_q_heads * o.stride(1)
    )
    if use_grouped_mid_q:
        if max_q_len <= 8:
            q_bucket = 8
            heads_per_group = 6
        elif max_q_len <= 32:
            q_bucket = 32
            heads_per_group = 2
        else:
            q_bucket = 64
            heads_per_group = 1
        num_head_groups = num_q_heads // heads_per_group
        num_q_blocks = (max_q_len + q_bucket - 1) // q_bucket
        grouped_task_count = (
            batch_size * num_q_blocks * num_head_groups
        )
        if max_q_len > 128 and grouped_task_count > cube_num:
            use_grouped_mid_q = False

    if use_grouped_mid_q:
        grouped_grid = (
            min(cube_num, grouped_task_count),
        )
        paged_prefill_mid_q_grouped_kernel[grouped_grid](
            q,
            key_cache,
            value_cache,
            o,
            cu_q_lens,
            seqlens_kv,
            block_tables_i32,
            sinks_pass,
            batch_size,
            num_q_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables_i32.stride(0),
            block_tables_i32.stride(1),
            sinks_pass.stride(0),
            softmax_scale,
            NUM_Q_HEADS=num_q_heads,
            HEADS_PER_GROUP=heads_per_group,
            NUM_HEAD_GROUPS=num_head_groups,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_D=head_dim,
            BLOCK_SIZE_N=page_size,
            Q_BUCKET=q_bucket,
            SINK_ENABLED=sink_enabled,
        )
        return o

    if not (page_size < 128 and 128 % page_size == 0):
        paged_prefill_kernel[grid](
            q,
            key_cache,
            value_cache,
            o,
            aux_mask,
            task_b,
            task_q_block,
            task_q_head,
            core_task_offsets,
            cu_q_lens,
            seqlens_kv,
            block_tables_i32,
            sinks_pass,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables_i32.stride(0),
            block_tables_i32.stride(1),
            aux_mask.stride(0),
            aux_mask.stride(1),
            sinks_pass.stride(0),
            softmax_scale,
            aux_mask.shape[0],
            page_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_SIZE_M=CHUNK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_D=head_dim,
            SINK_ENABLED=sink_enabled,
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            intra_cache_num=3,
            inter_cache_num=2,
            enable_buffer_insert_optimization=True,
            enable_ub_refine_opt=True,
        )
    else:
        PAGE_AGGREGATION_NUM = (
            1 if page_size == 64 and head_dim == 256 else 128 // page_size
        )
        paged_prefill_page_aggregation_kernel[grid](
            q,
            key_cache,
            value_cache,
            o,
            aux_mask,
            task_b,
            task_q_block,
            task_q_head,
            core_task_offsets,
            cu_q_lens,
            seqlens_kv,
            block_tables_i32,
            sinks_pass,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables_i32.stride(0),
            block_tables_i32.stride(1),
            aux_mask.stride(0),
            aux_mask.stride(1),
            sinks_pass.stride(0),
            softmax_scale,
            aux_mask.shape[0],
            page_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_SIZE_M=CHUNK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_D=head_dim,
            PAGE_AGGREGATION_NUM=PAGE_AGGREGATION_NUM,
            SINK_ENABLED=sink_enabled,
            enable_dynamic_cv_pipeline=True,
            enable_cube_block_merge=True,
            enable_buffer_insert_optimization=True,
            enable_ub_refine_opt=True,
        )

    return o

# -----------------------------------------------------------------------------
# WeLM paged Sliding-Window Attention with learnable zero-value sink
# -----------------------------------------------------------------------------

AUX_MASK_SIZE = 256
AUX_MASK = None

_GLOBAL_WINDOW_SIZE = None
_LOCAL_WINDOW_SIZE = None
_BLOCK_M = None
_BLOCK_N = None
_COMPRESSED_MASK = None



def get_mask_causal_with_window(
        BLOCK_M: int,
        BLOCK_N: int,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        device: str = "npu",
):
    global _GLOBAL_WINDOW_SIZE
    global _LOCAL_WINDOW_SIZE
    global _BLOCK_M
    global _BLOCK_N
    global _COMPRESSED_MASK
    if (
        _GLOBAL_WINDOW_SIZE == global_window_size
        and _LOCAL_WINDOW_SIZE == local_window_size
        and _BLOCK_M == BLOCK_M
        and _BLOCK_N == BLOCK_N
        and _COMPRESSED_MASK is not None
    ):
        return _COMPRESSED_MASK

    if local_window_size is None:
        local_window_size = 0
    if global_window_size is None:
        global_window_size = 0

    M = (global_window_size + local_window_size + 4 * max(BLOCK_M, BLOCK_N) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    N = (global_window_size + local_window_size + 5 * max(BLOCK_M, BLOCK_N) + BLOCK_N - 1) // BLOCK_N * BLOCK_N

    causal = torch.ones(M, N, dtype=torch.bool).tril()

    sink_band = torch.zeros(M, N, dtype=torch.bool)
    sink_band[:, :global_window_size] = True

    local_band = torch.ones(M, N, dtype=torch.bool).triu(diagonal=-local_window_size)

    mask = causal & (sink_band | local_band)

    M_boundary = M + AUX_MASK_SIZE
    N_boundary = N + AUX_MASK_SIZE
    mask_boundary = torch.zeros(M_boundary, N_boundary, dtype=torch.bool)
    mask_boundary[:M, :N] = mask
    mask_boundary = mask_boundary.to(device=device)

    _GLOBAL_WINDOW_SIZE = global_window_size
    _LOCAL_WINDOW_SIZE = local_window_size
    _BLOCK_M = BLOCK_M
    _BLOCK_N = BLOCK_N
    _COMPRESSED_MASK = mask_boundary
    return _COMPRESSED_MASK


@triton.jit
def gen_mask_causal_with_window(mask_ptr_causal, mask_size_m, mask_size_n, M_BLOCK, N_BLOCK, m_start, n_start,
                                global_window_size, local_windows_size, q_seq_len, kv_seq_len, AUX_MASK_SIZE=AUX_MASK_SIZE):
    if local_windows_size is None:
        local_windows_size = 0
    if global_window_size is None:
        global_window_size = 0

    actual_mask_m = mask_size_m - AUX_MASK_SIZE
    is_q_oob = (m_start >= kv_seq_len).to(tl.int32)
    valid_rows = max(0, min(kv_seq_len - m_start, M_BLOCK))
    is_tail = (valid_rows < M_BLOCK).to(tl.int32)

    m_pos_normal = min(m_start, actual_mask_m - M_BLOCK)
    m_pos = (1 - is_q_oob) * m_pos_normal + is_q_oob * actual_mask_m

    shift = m_start - m_pos
    need_adjust = (shift != 0).to(tl.int32)
    is_global_block = (n_start < global_window_size).to(tl.int32)
    can_compensate = ((m_start - n_start) <= local_windows_size).to(tl.int32)
    need_adjust = need_adjust * ((1 - is_global_block) + is_global_block * can_compensate)
    n_pos = ((1 - need_adjust) * n_start + need_adjust * max(global_window_size + 1, n_start - shift)) * (1 - is_q_oob)

    mask = tl.load(
        mask_ptr_causal
        + (m_pos + tl.arange(0, M_BLOCK)[:, None]) * mask_size_n
        + (n_pos + tl.arange(0, N_BLOCK))[None, :]
    )
    return mask


@triton.jit
def _swa_split_blocks(
    q_block_start_id,
    q_block_len,
    kv_seq_len,
    BLOCK_SIZE_N,
    IS_CAUSAL,
    GLOBAL_WINDOW_SIZE,
    LOCAL_WINDOW_SIZE
):
    if not IS_CAUSAL:
        return 0, 0, tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

    num_total_blocks = tl.cdiv(q_block_start_id + q_block_len, BLOCK_SIZE_N)
    if GLOBAL_WINDOW_SIZE is None and LOCAL_WINDOW_SIZE is None:
        return 0, 0, num_total_blocks

    if GLOBAL_WINDOW_SIZE is not None:
        num_global_window_blocks = min(tl.cdiv(GLOBAL_WINDOW_SIZE, BLOCK_SIZE_N), num_total_blocks)
    else:
        num_global_window_blocks = 0

    if LOCAL_WINDOW_SIZE is not None:
        local_window_start_id = max(q_block_start_id - LOCAL_WINDOW_SIZE, 0)
        local_window_start_block = local_window_start_id // BLOCK_SIZE_N
    else:
        local_window_start_block = num_total_blocks

    non_global_window_start_block = max(num_global_window_blocks, local_window_start_block)

    return num_global_window_blocks, non_global_window_start_block, num_total_blocks


@triton.jit
def _swa_grouped_token_page_update(
    acc,
    l_i,
    m_i,
    q,
    k_t,
    v,
    mask,
    scale,
):
    """Update one six-head query token from an already-loaded K/V page."""
    qk = tl.dot(q, k_t) * scale
    qk = tl.where(mask[None, :], qk, -1e6)
    m_ij = tl.maximum(
        m_i,
        tl.max(qk, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    p = tl.math.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(k_t.dtype), v, acc)
    return acc, l_i, m_ij


@triton.jit
def _sdpa_acc_fwd_MxN(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)
    # tl.extra.cann.extension.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1e6)  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.extra.cann.extension.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i



@triton.jit(
    do_not_specialize=[
        "block_table_ptr",
        "bsz",
        "stride_block_table_b",
    ]
)
def _swa_paged_prefill_sink_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    sinks_ptr,
    bsz,
    cu_q_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    stride_sink_head,
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)

        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS
        for q_task_id in range((n_programs - prev_q_tasks % n_programs + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # --- online-softmax init with optional sink ---
            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            if SINK_ENABLED:
                s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
                m_i = m_i + s_h
                l_i = l_i + 1.0
            else:
                m_i = m_i - float("inf")
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )
            num_calced_blocks = num_global_window_blocks + (num_total_blocks - non_global_window_start_block)
            for kv_block_iter in range(num_calced_blocks):
                cond = kv_block_iter < num_global_window_blocks
                kv_block_id = cond * kv_block_iter + (1 - cond) * (
                            non_global_window_start_block + kv_block_iter - num_global_window_blocks)
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )

                if IS_CAUSAL:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                    )
                else:
                    mask = tl.full((BLOCK_M, BLOCK_N), 1,  dtype=tl.int1)

                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh + kv_block_start_in_page * stride_vt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            cur_o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))



@triton.jit(
    do_not_specialize=[
        "block_table_ptr",
        "bsz",
        "stride_block_table_b",
    ]
)
def _swa_paged_prefill_aggregation_sink_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    sinks_ptr,
    bsz,
    cu_q_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    stride_sink_head,
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_AGGREGATION_NUM: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)

        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS
        for q_task_id in range((n_programs - prev_q_tasks % n_programs + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start

            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # --- online-softmax init with optional sink ---
            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            if SINK_ENABLED:
                s_h = tl.load(sinks_ptr + q_head_id * stride_sink_head).to(tl.float32)
                m_i = m_i + s_h
                l_i = l_i + 1.0
            else:
                m_i = m_i - float("inf")
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )
            num_global_window_blocks = tl.cdiv(num_global_window_blocks, PAGE_AGGREGATION_NUM) * PAGE_AGGREGATION_NUM
            non_global_window_start_block = max(num_global_window_blocks, non_global_window_start_block)
            num_calced_blocks = num_global_window_blocks + max(num_total_blocks - non_global_window_start_block, 0)
            num_calced_blocks = min(num_calced_blocks, num_total_blocks)
            for kv_block_iter in range(0, num_calced_blocks, PAGE_AGGREGATION_NUM):
                cond = kv_block_iter < num_global_window_blocks
                kv_block_id = cond * kv_block_iter + (1 - cond) * (
                        non_global_window_start_block + kv_block_iter - num_global_window_blocks)
                kv_block_start = kv_block_id * BLOCK_N

                if IS_CAUSAL:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N * PAGE_AGGREGATION_NUM,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                    )
                else:
                    mask = tl.full((BLOCK_M, BLOCK_N), 1,  dtype=tl.int1)

                k = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_N, BLOCK_D), dtype=k_ptr.dtype.element_ty)
                for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                    kv_block_start = (kv_block_id + page_iter) * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = max(kv_block_end - kv_block_start, 0)
                    logical_page_id = min(kv_block_start // PAGE_SIZE, stride_block_table_b - 1)
                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
                    cur_k_block_ptr = tl.make_block_ptr(
                        base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh,
                        shape=(kv_block_len, HEAD_DIM),
                        strides=(stride_kt, stride_kd),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    k_slice = tl.load(cur_k_block_ptr, boundary_check=(
                        0, 1), padding_option="zero")
                    k = tl.extra.cann.extension.insert_slice(k, k_slice, offsets=(page_iter * BLOCK_N, 0),
                                                             sizes=(BLOCK_N, BLOCK_D),
                                                             strides=(1, 1))
                k_T = tl.trans(k)
                qk = tl.dot(cur_q_block, k_T)
                qk = qk * scale
                if IS_CAUSAL:
                    qk = tl.where(mask, qk, -1e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

                qk = qk - m_ij[:, None]
                p = tl.math.exp(qk)
                p_cast = p.to(k_T.dtype)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_N, BLOCK_D), dtype=v_ptr.dtype.element_ty)
                for page_iter in tl.extra.cann.extension.parallel(0, PAGE_AGGREGATION_NUM):
                    kv_block_start = (kv_block_id + page_iter) * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = max(kv_block_end - kv_block_start, 0)
                    logical_page_id = min(kv_block_start // PAGE_SIZE, stride_block_table_b - 1)
                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
                    cur_v_block_ptr = tl.make_block_ptr(
                        base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh,
                        shape=(kv_block_len, HEAD_DIM),
                        strides=(stride_vt, stride_vd),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    v_slice = tl.load(cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    v = tl.extra.cann.extension.insert_slice(v, v_slice, offsets=(page_iter * BLOCK_N, 0),
                                                             sizes=(BLOCK_N, BLOCK_D),
                                                             strides=(1, 1))
                acc = tl.dot(p_cast, v, acc)

            # cur_o_block_ptr = tl.advance(o_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))




@triton.jit(
    do_not_specialize=[
        "block_table_ptr",
        "bsz",
        "stride_block_table_b",
    ]
)
def _swa_paged_prefill_single_q_grouped_sink_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    sinks_ptr,
    bsz,
    cu_q_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    stride_sink_head,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Preserve the proven six-head D1 path without merged-row padding."""
    tl.static_assert(
        PAGE_SIZE // BLOCK_N * BLOCK_N == PAGE_SIZE,
        "BLOCK_N must divide PAGE_SIZE",
    )
    tl.static_assert(
        NUM_Q_HEADS == 6,
        "grouped WeLM single-Q prefill requires six local Q heads",
    )
    tl.static_assert(
        BLOCK_N == PAGE_SIZE,
        "grouped WeLM single-Q prefill uses one page per KV tile",
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    b_begin = pid * bsz // n_programs
    b_end = (pid + 1) * bsz // n_programs

    for b_id in range(b_begin, b_end):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        if q_seq_len.to(tl.float32) > 0.0:
            kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
            kv_computed_len = kv_seq_len - q_seq_len
            q_head_ids = tl.arange(0, NUM_Q_HEADS)
            dim_offsets = tl.arange(0, BLOCK_D)
            q_ptrs = (
                q_ptr
                + q_start * stride_qt
                + q_head_ids[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
            q = tl.load(q_ptrs)

            if SINK_ENABLED:
                m_i = tl.load(
                    sinks_ptr + q_head_ids * stride_sink_head
                ).to(tl.float32)
                l_i = tl.full((NUM_Q_HEADS,), 1.0, tl.float32)
            else:
                m_i = tl.full(
                    (NUM_Q_HEADS,), -float("inf"), tl.float32
                )
                l_i = tl.zeros((NUM_Q_HEADS,), dtype=tl.float32)
            acc = tl.zeros(
                (NUM_Q_HEADS, BLOCK_D), dtype=tl.float32
            )

            query_position = kv_computed_len
            (
                num_global_blocks,
                local_start_block,
                num_total_blocks,
            ) = _swa_split_blocks(
                query_position,
                1,
                kv_seq_len,
                BLOCK_N,
                True,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )
            local_start_block = max(num_global_blocks, local_start_block)
            num_calced_blocks = num_global_blocks + max(
                num_total_blocks - local_start_block, 0
            )
            num_calced_blocks = min(num_calced_blocks, num_total_blocks)

            for kv_block_iter in range(0, num_calced_blocks):
                is_global = (
                    kv_block_iter.to(tl.float32)
                    < num_global_blocks.to(tl.float32)
                )
                kv_block_id = is_global * kv_block_iter + (
                    1 - is_global
                ) * (
                    local_start_block
                    + kv_block_iter
                    - num_global_blocks
                )
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = max(kv_block_end - kv_block_start, 0)

                key_offsets = tl.arange(0, BLOCK_N)
                key_positions = kv_block_start + key_offsets
                causal = (
                    key_positions.to(tl.float32)
                    <= query_position.to(tl.float32)
                )
                in_sink = key_positions.to(tl.float32) < GLOBAL_WINDOW
                in_local = (
                    key_positions.to(tl.float32) + LOCAL_WINDOW
                    >= query_position.to(tl.float32)
                )
                key_valid = (
                    key_positions.to(tl.float32)
                    < kv_seq_len.to(tl.float32)
                )
                mask = key_valid & causal & (in_sink | in_local)

                logical_page_id = min(
                    kv_block_start // PAGE_SIZE,
                    stride_block_table_b - 1,
                )
                physical_page_id = tl.load(
                    block_table_ptr
                    + b_id * stride_block_table_b
                    + logical_page_id * stride_block_table_p
                )
                k_t_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_kd, stride_kt),
                    offsets=(0, 0),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                k_t = tl.load(
                    k_t_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                v = tl.load(
                    v_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                qk = tl.dot(q, k_t) * scale
                qk = tl.where(mask[None, :], qk, -1e6)
                m_ij = tl.maximum(
                    m_i,
                    tl.max(qk, 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                p = tl.math.exp(qk - m_ij[:, None])
                pv = tl.dot(p.to(k_t.dtype), v)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None] + pv
                m_i = m_ij

            o_ptrs = (
                o_ptr
                + q_start * stride_ot
                + q_head_ids[:, None] * stride_oh
                + dim_offsets[None, :] * stride_od
            )
            tl.store(
                o_ptrs,
                (acc / l_i[:, None]).to(o_ptr.type.element_ty),
            )


@triton.jit(
    do_not_specialize=[
        "block_table_ptr",
        "bsz",
        "stride_block_table_b",
    ]
)
def _swa_paged_prefill_four_q_grouped_sink_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    sinks_ptr,
    bsz,
    cu_q_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    stride_sink_head,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_Q_LEN: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Run up to four query tokens while grouping six Q heads per KV head."""
    tl.static_assert(
        PAGE_SIZE // BLOCK_N * BLOCK_N == PAGE_SIZE,
        "BLOCK_N must divide PAGE_SIZE",
    )
    tl.static_assert(
        NUM_Q_HEADS == 6,
        "grouped WeLM small-Q prefill requires six local Q heads",
    )
    tl.static_assert(
        MAX_Q_LEN == 4,
        "grouped WeLM four-Q prefill requires max query length 4",
    )
    tl.static_assert(
        BLOCK_N == PAGE_SIZE,
        "grouped WeLM small-Q prefill uses one page per KV tile",
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    # Keep each program's requests contiguous, but distribute the remainder
    # evenly so B=48 uses all 28 Cube cores instead of only 24.
    b_begin = pid * bsz // n_programs
    b_end = (pid + 1) * bsz // n_programs

    for b_id in range(b_begin, b_end):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        if q_seq_len.to(tl.float32) > 0.0:
            kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
            kv_computed_len = kv_seq_len - q_seq_len
            for q_token_id in range(0, MAX_Q_LEN):
                active_token = q_token_id < q_seq_len.to(tl.float32)
                q_head_ids = tl.arange(0, NUM_Q_HEADS)
                dim_offsets = tl.arange(0, BLOCK_D)
                q_ptrs = (
                    q_ptr
                    + (q_start + q_token_id) * stride_qt
                    + q_head_ids[:, None] * stride_qh
                    + dim_offsets[None, :] * stride_qd
                )
                q = tl.load(q_ptrs, mask=active_token, other=0.0)

                if SINK_ENABLED:
                    m_i = tl.load(
                        sinks_ptr + q_head_ids * stride_sink_head
                    ).to(tl.float32)
                    l_i = tl.full((NUM_Q_HEADS,), 1.0, tl.float32)
                else:
                    m_i = tl.full(
                        (NUM_Q_HEADS,), -float("inf"), tl.float32
                    )
                    l_i = tl.zeros((NUM_Q_HEADS,), dtype=tl.float32)
                acc = tl.zeros(
                    (NUM_Q_HEADS, BLOCK_D), dtype=tl.float32
                )

                query_position = kv_computed_len + q_token_id
                (
                    num_global_blocks,
                    local_start_block,
                    num_total_blocks,
                ) = _swa_split_blocks(
                    query_position,
                    1,
                    kv_seq_len,
                    BLOCK_N,
                    True,
                    GLOBAL_WINDOW,
                    LOCAL_WINDOW,
                )
                local_start_block = max(
                    num_global_blocks, local_start_block
                )
                num_calced_blocks = num_global_blocks + max(
                    num_total_blocks - local_start_block, 0
                )
                num_calced_blocks = min(
                    num_calced_blocks, num_total_blocks
                )

                for kv_block_iter in range(0, num_calced_blocks):
                    is_global = (
                        kv_block_iter.to(tl.float32)
                        < num_global_blocks.to(tl.float32)
                    )
                    kv_block_id = is_global * kv_block_iter + (
                        1 - is_global
                    ) * (
                        local_start_block
                        + kv_block_iter
                        - num_global_blocks
                    )
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_block_end = min(
                        kv_block_start + BLOCK_N, kv_seq_len
                    )
                    kv_block_len = max(
                        kv_block_end - kv_block_start, 0
                    )

                    key_offsets = tl.arange(0, BLOCK_N)
                    key_positions = kv_block_start + key_offsets
                    causal = (
                        key_positions.to(tl.float32)
                        <= query_position.to(tl.float32)
                    )
                    in_sink = (
                        key_positions.to(tl.float32) < GLOBAL_WINDOW
                    )
                    in_local = (
                        key_positions.to(tl.float32) + LOCAL_WINDOW
                        >= query_position.to(tl.float32)
                    )
                    key_valid = (
                        key_positions.to(tl.float32)
                        < kv_seq_len.to(tl.float32)
                    )
                    mask = active_token & key_valid & causal & (in_sink | in_local)

                    logical_page_id = min(
                        kv_block_start // PAGE_SIZE,
                        stride_block_table_b - 1,
                    )
                    physical_page_id = tl.load(
                        block_table_ptr
                        + b_id * stride_block_table_b
                        + logical_page_id * stride_block_table_p
                    )
                    k_t_block_ptr = tl.make_block_ptr(
                        base=k_ptr + physical_page_id * stride_kp,
                        shape=(HEAD_DIM, kv_block_len),
                        strides=(stride_kd, stride_kt),
                        offsets=(0, 0),
                        block_shape=(BLOCK_D, BLOCK_N),
                        order=(0, 1),
                    )
                    v_block_ptr = tl.make_block_ptr(
                        base=v_ptr + physical_page_id * stride_vp,
                        shape=(kv_block_len, HEAD_DIM),
                        strides=(stride_vt, stride_vd),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    k_t = tl.load(
                        k_t_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    v = tl.load(
                        v_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    qk = tl.dot(q, k_t) * scale
                    qk = tl.where(mask[None, :], qk, -1e6)
                    m_ij = tl.maximum(
                        m_i,
                        tl.max(qk, 1, propagate_nan=True),
                        propagate_nan=tl.PropagateNan.ALL,
                    )
                    p = tl.math.exp(qk - m_ij[:, None])
                    pv = tl.dot(p.to(k_t.dtype), v)
                    l_ij = tl.sum(p, 1)
                    alpha = tl.math.exp(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None] + pv
                    m_i = m_ij

                output = acc / l_i[:, None]
                o_ptrs = (
                    o_ptr
                    + (q_start + q_token_id) * stride_ot
                    + q_head_ids[:, None] * stride_oh
                    + dim_offsets[None, :] * stride_od
                )
                tl.store(
                    o_ptrs,
                    output.to(o_ptr.type.element_ty),
                    mask=active_token,
                )


@triton.jit(
    do_not_specialize=[
        "block_table_ptr",
        "bsz",
        "stride_block_table_b",
    ]
)
def _swa_paged_prefill_small_q_grouped_sink_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    sinks_ptr,
    bsz,
    cu_q_lens_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    stride_sink_head,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_Q_LEN: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Run up to four query tokens while grouping six Q heads per KV head."""
    tl.static_assert(
        PAGE_SIZE // BLOCK_N * BLOCK_N == PAGE_SIZE,
        "BLOCK_N must divide PAGE_SIZE",
    )
    tl.static_assert(
        NUM_Q_HEADS == 6,
        "grouped WeLM small-Q prefill requires six local Q heads",
    )
    tl.static_assert(
        MAX_Q_LEN >= 2 and MAX_Q_LEN <= 3,
        "shared-page WeLM small-Q prefill max query length must be 2 or 3",
    )
    tl.static_assert(
        BLOCK_N == PAGE_SIZE,
        "grouped WeLM small-Q prefill uses one page per KV tile",
    )
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    # Keep each program's requests contiguous, but distribute the remainder
    # evenly so B=48 uses all 28 Cube cores instead of only 24.
    b_begin = pid * bsz // n_programs
    b_end = (pid + 1) * bsz // n_programs

    for b_id in range(b_begin, b_end):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        if q_seq_len.to(tl.float32) > 0.0:
            kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
            kv_computed_len = kv_seq_len - q_seq_len
            q_head_ids = tl.arange(0, NUM_Q_HEADS)
            dim_offsets = tl.arange(0, BLOCK_D)
            q_base_ptrs = (
                q_ptr
                + q_start * stride_qt
                + q_head_ids[:, None] * stride_qh
                + dim_offsets[None, :] * stride_qd
            )
            q0 = tl.load(q_base_ptrs)
            active1 = q_seq_len.to(tl.float32) > 1.0
            q1 = tl.load(
                q_base_ptrs + stride_qt,
                mask=active1,
                other=0.0,
            )
            if MAX_Q_LEN >= 3:
                active2 = q_seq_len.to(tl.float32) > 2.0
                q2 = tl.load(
                    q_base_ptrs + 2 * stride_qt,
                    mask=active2,
                    other=0.0,
                )
            if MAX_Q_LEN >= 4:
                active3 = q_seq_len.to(tl.float32) > 3.0
                q3 = tl.load(
                    q_base_ptrs + 3 * stride_qt,
                    mask=active3,
                    other=0.0,
                )

            if SINK_ENABLED:
                m_init = tl.load(
                    sinks_ptr + q_head_ids * stride_sink_head
                ).to(tl.float32)
                l_init = tl.full((NUM_Q_HEADS,), 1.0, tl.float32)
            else:
                m_init = tl.full(
                    (NUM_Q_HEADS,), -float("inf"), tl.float32
                )
                l_init = tl.zeros((NUM_Q_HEADS,), dtype=tl.float32)
            m0 = m_init
            l0 = l_init
            acc0 = tl.zeros((NUM_Q_HEADS, BLOCK_D), dtype=tl.float32)
            m1 = m_init
            l1 = l_init
            acc1 = tl.zeros((NUM_Q_HEADS, BLOCK_D), dtype=tl.float32)
            if MAX_Q_LEN >= 3:
                m2 = m_init
                l2 = l_init
                acc2 = tl.zeros(
                    (NUM_Q_HEADS, BLOCK_D), dtype=tl.float32
                )
            if MAX_Q_LEN >= 4:
                m3 = m_init
                l3 = l_init
                acc3 = tl.zeros(
                    (NUM_Q_HEADS, BLOCK_D), dtype=tl.float32
                )

            (
                num_global_blocks,
                local_start_block,
                num_total_blocks,
            ) = _swa_split_blocks(
                kv_computed_len,
                q_seq_len,
                kv_seq_len,
                BLOCK_N,
                True,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )
            local_start_block = max(num_global_blocks, local_start_block)
            num_calced_blocks = num_global_blocks + max(
                num_total_blocks - local_start_block, 0
            )
            num_calced_blocks = min(num_calced_blocks, num_total_blocks)

            query0 = kv_computed_len
            query1 = kv_computed_len + 1
            if MAX_Q_LEN >= 3:
                query2 = kv_computed_len + 2
            if MAX_Q_LEN >= 4:
                query3 = kv_computed_len + 3
            for kv_block_iter in range(0, num_calced_blocks):
                is_global = (
                    kv_block_iter.to(tl.float32)
                    < num_global_blocks.to(tl.float32)
                )
                kv_block_id = is_global * kv_block_iter + (
                    1 - is_global
                ) * (
                    local_start_block
                    + kv_block_iter
                    - num_global_blocks
                )
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = max(kv_block_end - kv_block_start, 0)

                key_offsets = tl.arange(0, BLOCK_N)
                key_positions = kv_block_start + key_offsets
                key_valid = (
                    key_positions.to(tl.float32)
                    < kv_seq_len.to(tl.float32)
                )
                in_sink = key_positions.to(tl.float32) < GLOBAL_WINDOW
                mask0 = (
                    key_valid
                    & (
                        key_positions.to(tl.float32)
                        <= query0.to(tl.float32)
                    )
                    & (
                        in_sink
                        | (
                            key_positions.to(tl.float32) + LOCAL_WINDOW
                            >= query0.to(tl.float32)
                        )
                    )
                )
                mask1 = (
                    active1
                    & key_valid
                    & (
                        key_positions.to(tl.float32)
                        <= query1.to(tl.float32)
                    )
                    & (
                        in_sink
                        | (
                            key_positions.to(tl.float32) + LOCAL_WINDOW
                            >= query1.to(tl.float32)
                        )
                    )
                )
                if MAX_Q_LEN >= 3:
                    mask2 = (
                        active2
                        & key_valid
                        & (
                            key_positions.to(tl.float32)
                            <= query2.to(tl.float32)
                        )
                        & (
                            in_sink
                            | (
                                key_positions.to(tl.float32) + LOCAL_WINDOW
                                >= query2.to(tl.float32)
                            )
                        )
                    )
                if MAX_Q_LEN >= 4:
                    mask3 = (
                        active3
                        & key_valid
                        & (
                            key_positions.to(tl.float32)
                            <= query3.to(tl.float32)
                        )
                        & (
                            in_sink
                            | (
                                key_positions.to(tl.float32) + LOCAL_WINDOW
                                >= query3.to(tl.float32)
                            )
                        )
                    )

                logical_page_id = min(
                    kv_block_start // PAGE_SIZE,
                    stride_block_table_b - 1,
                )
                physical_page_id = tl.load(
                    block_table_ptr
                    + b_id * stride_block_table_b
                    + logical_page_id * stride_block_table_p
                )
                k_t_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_kd, stride_kt),
                    offsets=(0, 0),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                k_t = tl.load(
                    k_t_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                v = tl.load(
                    v_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                acc0, l0, m0 = _swa_grouped_token_page_update(
                    acc0, l0, m0, q0, k_t, v, mask0, scale
                )
                acc1, l1, m1 = _swa_grouped_token_page_update(
                    acc1, l1, m1, q1, k_t, v, mask1, scale
                )
                if MAX_Q_LEN >= 3:
                    acc2, l2, m2 = _swa_grouped_token_page_update(
                        acc2, l2, m2, q2, k_t, v, mask2, scale
                    )
                if MAX_Q_LEN >= 4:
                    acc3, l3, m3 = _swa_grouped_token_page_update(
                        acc3, l3, m3, q3, k_t, v, mask3, scale
                    )

            o_base_ptrs = (
                o_ptr
                + q_start * stride_ot
                + q_head_ids[:, None] * stride_oh
                + dim_offsets[None, :] * stride_od
            )
            tl.store(
                o_base_ptrs,
                (acc0 / l0[:, None]).to(o_ptr.type.element_ty),
            )
            tl.store(
                o_base_ptrs + stride_ot,
                (acc1 / l1[:, None]).to(o_ptr.type.element_ty),
                mask=active1,
            )
            if MAX_Q_LEN >= 3:
                tl.store(
                    o_base_ptrs + 2 * stride_ot,
                    (acc2 / l2[:, None]).to(o_ptr.type.element_ty),
                    mask=active2,
                )
            if MAX_Q_LEN >= 4:
                tl.store(
                    o_base_ptrs + 3 * stride_ot,
                    (acc3 / l3[:, None]).to(o_ptr.type.element_ty),
                    mask=active3,
                )


def swa_paged_prefill_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,  # [bsz + 1]
    kvlens: torch.Tensor,  # [bsz + 1]
    block_table: torch.Tensor,  # [bsz, num_kv_blocks]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    sinks: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
) -> torch.Tensor:

    bsz = cu_q_lens.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    # --- sink setup ---
    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
        # WeLM stores attn_sink in FP32 even when Q/K/V use BF16/FP16. The
        # kernel loads the sink directly into its FP32 online-softmax state.
    sinks_pass = sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    small_q_per_request = max_q_len is not None and 1 <= max_q_len <= 4
    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = min(64, triton.next_power_of_2(page_size))
    else:
        # Verify/Draft-extend carries D=2/3 rows per request.  Its dedicated
        # M=16 path matches the Cube micro-tile and keeps the M16/N128/D256
        # live set below the 248 KiB UB limit.  Normal prefill retains M=128.
        BLOCK_M = 16 if small_q_per_request else 128
        BLOCK_N = min(128, triton.next_power_of_2(page_size))

    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")

    num_q_chunks = triton.cdiv(tot_q_toks, BLOCK_M)
    # Summing tokens before cdiv severely under-counts verify work: for
    # B=56,D=2,M=128 it reports one chunk (six head tasks), while every
    # request owns an independent chunk (336 head tasks).  Keep the dense
    # estimate and the per-request lower bound, then cap at physical Cube
    # cores below.  The kernel already distributes these tasks by pid.
    dense_tasks = num_q_chunks * num_q_heads
    per_request_tasks = bsz * num_q_heads
    num_tasks = max(dense_tasks, per_request_tasks)
    n_programs = min(cube_num, num_tasks)
    grid = (n_programs,)

    if global_window_size is None:
        global_window_size = 0

    # WeLM TP=4 has one local KV head shared by six local Q heads.  When every
    # request has at most four query rows, group them and scan each page once.
    # N64 keeps the peak live set around 96 KiB, comfortably below the 248 KiB
    # UB budget and close to the already-proven decode kernel structure.
    use_grouped_small_q = (
        q.dtype != torch.float32
        and is_causal
        and small_q_per_request
        # A single SWA request with q_len 3/4 has only one grouped program;
        # the generic page-aggregation path is faster at that tiny grid.
        # Keep q_len 1/2 grouped because it still wins for B=1, and keep all
        # multi-request small-q workloads grouped to share each KV scan.
        and (bsz > 1 or max_q_len <= 2)
        and local_window_size is not None
        and not gqa_interleave
        and num_kv_heads == 1
        and num_q_heads == 6
        and q.stride(0) == num_q_heads * q.stride(1)
        and o.stride(0) == num_q_heads * o.stride(1)
        and page_size == 64
    )
    if use_grouped_small_q:
        grouped_grid = (min(cube_num, bsz),)
        if max_q_len == 1:
            _swa_paged_prefill_single_q_grouped_sink_kernel[grouped_grid](
                o,
                q,
                k_cache,
                v_cache,
                sinks_pass,
                bsz,
                cu_q_lens,
                kvlens,
                block_table,
                softmax_scale,
                o.stride(0),
                o.stride(1),
                o.stride(2),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(3),
                block_table.stride(0),
                block_table.stride(1),
                sinks_pass.stride(0),
                GLOBAL_WINDOW=global_window_size,
                LOCAL_WINDOW=local_window_size,
                NUM_Q_HEADS=num_q_heads,
                HEAD_DIM=head_dim,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                PAGE_SIZE=page_size,
                SINK_ENABLED=sink_enabled,
            )
            return o

        grouped_small_q_kernel = (
            _swa_paged_prefill_four_q_grouped_sink_kernel
            if max_q_len == 4
            else _swa_paged_prefill_small_q_grouped_sink_kernel
        )
        grouped_small_q_kernel[grouped_grid](
            o,
            q,
            k_cache,
            v_cache,
            sinks_pass,
            bsz,
            cu_q_lens,
            kvlens,
            block_table,
            softmax_scale,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            sinks_pass.stride(0),
            GLOBAL_WINDOW=global_window_size,
            LOCAL_WINDOW=local_window_size,
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            PAGE_SIZE=page_size,
            MAX_Q_LEN=max_q_len,
            SINK_ENABLED=sink_enabled,
        )
        return o

    causal_mask = get_mask_causal_with_window(
        BLOCK_M,
        BLOCK_N,
        local_window_size,
        global_window_size
    )
    causal_mask_m_size, causal_mask_n_size = causal_mask.shape

    if page_size < 128 and 128 % page_size == 0:
        PAGE_AGGREGATION_NUM = 128 // page_size
        _swa_paged_prefill_aggregation_sink_kernel[grid](
            o,
            q,
            k_cache,
            v_cache,
            sinks_pass,
            bsz,
            cu_q_lens,
            kvlens,
            block_table,
            softmax_scale,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            sinks_pass.stride(0),
            causal_mask,
            causal_mask_m_size,
            causal_mask_n_size,
            is_causal,
            global_window_size,
            local_window_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            page_size,
            PAGE_AGGREGATION_NUM,
            SINK_ENABLED=sink_enabled,
            enable_dynamic_cv_pipeline=True,
            enable_cube_block_merge=True,
        )
    else:
        _swa_paged_prefill_sink_kernel[grid](
            o,
            q,
            k_cache,
            v_cache,
            sinks_pass,
            bsz,
            cu_q_lens,
            kvlens,
            block_table,
            softmax_scale,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            sinks_pass.stride(0),
            causal_mask,
            causal_mask_m_size,
            causal_mask_n_size,
            is_causal,
            global_window_size,
            local_window_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            page_size,
            SINK_ENABLED=sink_enabled,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            intra_cache_num=3,
            inter_cache_num=2,
        )
    return o
