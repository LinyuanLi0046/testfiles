"""Paged Full and Sliding-Window Attention with WeLM's zero-value sink.

The public wrappers in this module consume KV cache views in
``[num_pages, num_kv_heads, page_size, head_dim]`` order.  The Ascend backend
keeps the underlying allocation in its native page-major order and provides a
zero-copy permuted view before calling these wrappers.
"""

import heapq
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

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


@triton.jit
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


def _should_use_flash_decode(
    batch_size: int,
    num_kv_heads: int,
    group_size: int,
    max_kv_len: int,
    cube_num: int,
) -> bool:
    """
    FD is triggered when B*N_KV tasks are too few to saturate all cube cores
    and the KV sequence is long enough to benefit from S2 splitting.
    """
    _FD_BN_RATIO = 0.4
    if max_kv_len < 256:
        return False
    loop_times = batch_size * num_kv_heads
    if loop_times >= _FD_BN_RATIO * cube_num:
        return False
    # MHA / MQA: always FD once loop_times threshold is met
    if group_size == 1:
        return True
    # GQA: additionally require long-enough KV to amortise workspace overhead
    return max_kv_len >= 2048


def _compute_kv_split_parts(
    batch_size: int,
    num_kv_heads: int,
    max_kv_len: int,
    cube_num: int,
) -> int:
    """
    Start with aicNum/loopTimes and reduce until each split covers at least
    KV_SPLIT_LIMIT tokens (experience value matching sInnerFactor_=128 path).
    """
    KV_SPLIT_LIMIT = 256
    loop_times = batch_size * num_kv_heads
    max_by_cores = cube_num // loop_times
    max_by_len = max_kv_len // KV_SPLIT_LIMIT
    return max(1, min(max_by_cores, max_by_len))


# ---------------------------------------------------------------------------
# Flash Decode Phase-1 kernel: Cube cores compute partial attention per split
#
# Core assignment:
#   Each program handles one (batch, kv_head, s2_split) triple and processes
#   all GROUP_SIZE Q-heads simultaneously.
#   Tasks are: total = B * N_KV * KV_SPLIT_PARTS, cycled across cube_num cores.
#
# Workspace written per program:
#   acc_ws  [ws_task_idx, g, d]  – locally-normalised partial output (float32)
#   lse_ws  [ws_task_idx, g]     – m_i + log(l_i) per G-head (float32)
# ---------------------------------------------------------------------------
# B.3  paged_decode_fd_reduce_kernel  (FD Phase-2 — sink as "virtual split")
#
#   paged_decode_fd_kernel (Phase-1) is imported UNCHANGED from flash_attention.
#   Sink must NOT be added in Phase-1 (each split would double-count it).
#   Here in Phase-2, sink participates as a virtual split:
#     - pass 1: lse_max = max(lse_max, s_h)
#     - pass 2: exp_sum += exp(s_h - lse_max)   (sink acc=0, only enlarges denominator)
# ---------------------------------------------------------------------------

# Decode Graph uses a max-context block table while eager KV-mirror prefill
# uses a tight table derived from the current sequence length.  They share this
# performance-specialized kernel, so suppress only the table base/row-stride
# attributes; Q/K/V/workspace layouts and KV_SPLIT_PARTS stay specialized.
@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "stride_bt_batch",
    ]
)
def paged_decode_fd_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlens_ptr,
    block_tables_ptr,
    acc_ws_ptr,
    lse_ws_ptr,
    stride_qb,
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
    stride_bt_batch,
    stride_bt_block,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    softmax_scale,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_fd_tasks = BATCH_SIZE * NUM_KV_HEADS * KV_SPLIT_PARTS

    for fd_task_id in range(pid, total_fd_tasks, n_progs):
        split_idx = fd_task_id % KV_SPLIT_PARTS
        kv_task = fd_task_id // KV_SPLIT_PARTS
        b_id = kv_task // NUM_KV_HEADS
        kv_head_id = kv_task % NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        # Partition the KV sequence evenly across splits, aligning chunk_size to
        # PAGE_SIZE so that each split starts at a page boundary.  This avoids KV
        # blocks that cross page boundaries, which would produce incorrect loads
        # because each page maps to a different physical block via block_tables.
        raw_chunk = tl.cdiv(kv_seq_len, KV_SPLIT_PARTS)
        chunk_size = tl.cdiv(raw_chunk, PAGE_SIZE) * PAGE_SIZE
        # Graph capture can choose the split count from the maximum context
        # length while a replayed request is much shorter.  Clamp empty tail
        # splits so kv_end - kv_start never becomes negative.
        kv_start = tl.minimum(split_idx * chunk_size, kv_seq_len)
        kv_end = tl.minimum(kv_start + chunk_size, kv_seq_len)

        # Workspace slot for this (b, kv_head, split)
        ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        # Load Q for all G-heads: [G, D]
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        # Iterate over KV blocks within [kv_start, kv_end)
        num_kv_blocks = tl.cdiv(kv_end - kv_start, BLOCK_SIZE_N)

        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_start + kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_end)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)  # [G, BLOCK_N]
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(k_T.dtype), v)
            m_i = m_ij

        # Locally normalise acc by this split's softmax denominator (l_i).
        # The reduce kernel will re-weight each split's contribution using lse_i.
        # Empty splits (kv_start >= kv_seq_len) have l_i=0; guard against div-0.
        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), float("-inf"))

        # Write lse to workspace
        lse_ptrs = (
            lse_ws_ptr
            + ws_task_idx * stride_lse_task
            + g_offsets * stride_lse_g
        )
        tl.store(lse_ptrs, lse_i)

        # Write acc to workspace
        acc_ptrs = (
            acc_ws_ptr
            + ws_task_idx * stride_aws_task
            + g_offsets[:, None] * stride_aws_g
            + offs_d[None, :] * stride_aws_d
        )
        tl.store(acc_ptrs, acc, mask=offs_d[None, :] < HEAD_DIM)


# ---------------------------------------------------------------------------
# Flash Decode Phase-2 kernel: Vector cores merge partial results
#
# Each program handles one (batch, kv_head) pair and merges KV_SPLIT_PARTS
# partial outputs into the final attention output.
#
# Merge formula (online softmax correction):
#   lse_max_g  = max_i(lse_i_g)          per G-head
#   w_i_g      = exp(lse_i_g - lse_max_g)
#   W_g        = sum_i(w_i_g)
#   out_g      = sum_i (w_i_g / W_g) * acc_ws[i, g, :]
# ---------------------------------------------------------------------------

@triton.jit
def paged_decode_fd_reduce_kernel(
    acc_ws_ptr,
    lse_ws_ptr,
    o_ptr,
    seqlens_ptr,
    sinks_ptr,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    stride_ob,
    stride_oh,
    stride_od,
    stride_sink_head,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_reduce_tasks = BATCH_SIZE * NUM_KV_HEADS

    for reduce_task_id in range(pid, total_reduce_tasks, n_progs):
        b_id = reduce_task_id // NUM_KV_HEADS
        kv_head_id = reduce_task_id % NUM_KV_HEADS

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        # --- load sink logits once (used in both passes) ---
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)

        # Pass 1: find per-G-head max lse across all splits (numerical stability)
        lse_max = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx
            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse_max = tl.maximum(lse_max, tl.load(lse_ptrs))

        if SINK_ENABLED:
            lse_max = tl.maximum(lse_max, s_h)

        # Pass 2: weighted accumulation of partial outputs
        out = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)
        exp_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)

        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse = tl.load(lse_ptrs)
            w = tl.math.exp(lse - lse_max)  # [G,]; 0 for empty splits (lse=-inf)
            exp_sum += w

            acc_ptrs = (
                acc_ws_ptr
                + ws_task_idx * stride_aws_task
                + g_offsets[:, None] * stride_aws_g
                + offs_d[None, :] * stride_aws_d
            )
            acc_split = tl.load(acc_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)
            out += w[:, None] * acc_split

        # --- sink contributes 0 to numerator (V=0), only enlarges denominator ---
        if SINK_ENABLED:
            exp_sum += tl.math.exp(s_h - lse_max)

        # Normalise and write final output
        exp_sum_safe = tl.where(exp_sum > 0, exp_sum, 1.0)
        out = out / exp_sum_safe[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


@triton.jit(
    do_not_specialize=[
        "stride_bt_batch",
        "BATCH_SIZE",
        "KV_SPLIT_PARTS",
    ]
)
def mirror_paged_decode_fd_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlens_ptr,
    block_tables_ptr,
    acc_ws_ptr,
    lse_ws_ptr,
    stride_qb,
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
    stride_bt_batch,
    stride_bt_block,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    softmax_scale,
    BATCH_SIZE,
    KV_SPLIT_PARTS,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Request-shape-stable Flash Decode phase 1 for KV-mirror prefill."""
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_fd_tasks = BATCH_SIZE * NUM_KV_HEADS * KV_SPLIT_PARTS

    for fd_task_id in range(pid, total_fd_tasks, n_progs):
        kv_task = fd_task_id // KV_SPLIT_PARTS
        split_idx = fd_task_id - kv_task * KV_SPLIT_PARTS
        b_id = kv_task // NUM_KV_HEADS
        kv_head_id = kv_task - b_id * NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)
        raw_chunk = tl.cdiv(kv_seq_len, KV_SPLIT_PARTS)
        chunk_size = tl.cdiv(raw_chunk, PAGE_SIZE) * PAGE_SIZE
        kv_start = tl.minimum(split_idx * chunk_size, kv_seq_len)
        kv_end = tl.minimum(kv_start + chunk_size, kv_seq_len)

        ws_task_idx = (
            (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx
        )

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
            other=0.0,
        )

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_end - kv_start, BLOCK_SIZE_N)
        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_start + kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_end)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start - logical_page_id * PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = (
                tl.arange(0, BLOCK_SIZE_N).to(tl.float32)
                < kv_block_len.to(tl.float32)
            )
            k_T = tl.load(
                K_T_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            v = tl.load(
                V_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(k_T.dtype), v)
            m_i = m_ij

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), float("-inf"))

        lse_ptrs = (
            lse_ws_ptr
            + ws_task_idx * stride_lse_task
            + g_offsets * stride_lse_g
        )
        tl.store(lse_ptrs, lse_i)

        acc_ptrs = (
            acc_ws_ptr
            + ws_task_idx * stride_aws_task
            + g_offsets[:, None] * stride_aws_g
            + offs_d[None, :] * stride_aws_d
        )
        tl.store(
            acc_ptrs,
            acc,
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
        )


@triton.jit(
    do_not_specialize=[
        "BATCH_SIZE",
        "KV_SPLIT_PARTS",
    ]
)
def mirror_paged_decode_fd_reduce_kernel(
    acc_ws_ptr,
    lse_ws_ptr,
    o_ptr,
    seqlens_ptr,
    sinks_ptr,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    stride_ob,
    stride_oh,
    stride_od,
    stride_sink_head,
    BATCH_SIZE,
    KV_SPLIT_PARTS,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Request-shape-stable Flash Decode reduction for KV-mirror prefill."""
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_reduce_tasks = BATCH_SIZE * NUM_KV_HEADS

    for reduce_task_id in range(pid, total_reduce_tasks, n_progs):
        b_id = reduce_task_id // NUM_KV_HEADS
        kv_head_id = reduce_task_id - b_id * NUM_KV_HEADS

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids.to(tl.float32) < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)

        lse_max = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        for split_idx in tl.range(0, KV_SPLIT_PARTS, num_stages=1):
            ws_task_idx = (
                (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS
                + split_idx
            )
            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse_max = tl.maximum(lse_max, tl.load(lse_ptrs))

        if SINK_ENABLED:
            lse_max = tl.maximum(lse_max, s_h)

        out = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)
        exp_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        for split_idx in tl.range(0, KV_SPLIT_PARTS, num_stages=1):
            ws_task_idx = (
                (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS
                + split_idx
            )
            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse = tl.load(lse_ptrs)
            w = tl.math.exp(lse - lse_max)
            exp_sum += w

            acc_ptrs = (
                acc_ws_ptr
                + ws_task_idx * stride_aws_task
                + g_offsets[:, None] * stride_aws_g
                + offs_d[None, :] * stride_aws_d
            )
            acc_split = tl.load(
                acc_ptrs,
                mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
                other=0.0,
            )
            out += w[:, None] * acc_split

        if SINK_ENABLED:
            exp_sum += tl.math.exp(s_h - lse_max)

        exp_sum_safe = tl.where(exp_sum > 0, exp_sum, 1.0)
        out = out / exp_sum_safe[:, None]
        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            out.to(o_ptr.dtype.element_ty),
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
        )


# See paged_decode_fd_kernel: only the graph/eager block-table representation
# is request-dependent.  BATCH_SIZE and all data layouts remain specialized.
@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "stride_bt_batch",
    ]
)
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    sinks_ptr,
    BATCH_SIZE,
    stride_qb,
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
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        # Compute q_head_ids for this kv_head group
        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        # Load q for all heads in the group: [GROUP_SIZE, D]
        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        # --- online-softmax init with optional sink ---
        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block)

            # Load K transposed: [D, BLOCK_N] for tl.dot(q, k_T)
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            # Load V: [BLOCK_N, D] for tl.dot(p, v)
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)

            qk *= softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            qk = qk - m_ij[:, None]

            p = tl.math.exp(qk)
            p_cast = p.to(k_T.dtype)

            pv = tl.dot(p_cast, v)

            # Softmax denominator and update (Vector: parallel with Cube)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        m_i += tl.math.log(l_i)
        if kv_seq_len > 0:
            # avoid division by zero
            acc = acc / l_i[:, None]

        # Store output for all heads in the group
        o_ptrs = o_ptr + b_id * stride_ob + q_head_ids[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


@triton.jit(
    do_not_specialize=[
        "BATCH_SIZE",
        "stride_bt_batch",
    ]
)
def mirror_paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    sinks_ptr,
    BATCH_SIZE,
    stride_qb,
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
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Request-shape-stable non-FD kernel for KV-mirror prefill."""
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(
        HEAD_DIM <= BLOCK_SIZE_D,
        "HEAD_DIM should be less than BLOCK_SIZE_D",
    )
    tl.static_assert(
        PAGE_SIZE // BLOCK_SIZE_N * BLOCK_SIZE_N == PAGE_SIZE,
        "BLOCK_SIZE_N must be a divisor of PAGE_SIZE",
    )
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        b_id = kv_task_id // NUM_KV_HEADS
        kv_head_id = kv_task_id - b_id * NUM_KV_HEADS
        kv_seq_len = tl.load(seqlens_ptr + b_id)

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
            other=0.0,
        )

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids.to(tl.float32) < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)
        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = tl.minimum(
                kv_block_start_in_seq + BLOCK_SIZE_N,
                kv_seq_len,
            )
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = (
                kv_block_start_in_seq - logical_page_id * PAGE_SIZE
            )
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = (
                tl.arange(0, BLOCK_SIZE_N).to(tl.float32)
                < kv_block_len.to(tl.float32)
            )
            k_T = tl.load(
                K_T_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            v = tl.load(
                V_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            qk = tl.dot(q, k_T) * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))
            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            pv = tl.dot(p.to(k_T.dtype), v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        m_i += tl.math.log(l_i)
        if kv_seq_len.to(tl.float32) > 0.0:
            acc = acc / l_i[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            acc.to(o_ptr.dtype.element_ty),
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
        )


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    max_kv_len_hint: Optional[int] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert value_cache.shape == key_cache.shape
    assert head_dim == head_dim_cache
    assert num_q_heads % num_kv_heads == 0
    assert block_tables.shape[0] == batch_size
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # --- sink setup ---
    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
    sinks_pass = sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)

    o = torch.empty_like(q)
    block_tables_i32 = block_tables.to(dtype=torch.int32).contiguous()

    cube_num = get_num_cores("cube")
    vector_num = get_num_cores("vector")
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    group_size = num_q_heads // num_kv_heads
    max_kv_len = (
        int(max_kv_len_hint)
        if max_kv_len_hint is not None
        else int(seqlens.max().item())
    )

    if _should_use_flash_decode(batch_size, num_kv_heads, group_size, max_kv_len, cube_num):
        kv_split_parts = _compute_kv_split_parts(
            batch_size, num_kv_heads, max_kv_len, cube_num
        )

        # Workspace: one slot per (batch, kv_head, split)
        num_ws_tasks = batch_size * num_kv_heads * kv_split_parts
        acc_ws = torch.empty(
            (num_ws_tasks, group_size, head_dim),
            dtype=torch.float32, device=q.device,
        )
        lse_ws = torch.full(
            (num_ws_tasks, group_size),
            float("-inf"), dtype=torch.float32, device=q.device,
        )

        # Phase 1 – Cube cores: partial attention per (b, kv_head, split)
        #   NOTE: sink is NOT passed here (would be double-counted across splits)
        paged_decode_fd_kernel[(cube_num,)](
            q, key_cache, value_cache, seqlens, block_tables_i32,
            acc_ws, lse_ws,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            block_tables_i32.stride(0), block_tables_i32.stride(1),
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            softmax_scale,
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Phase 2 – Vector cores: online-softmax merge across splits (+ sink)
        paged_decode_fd_reduce_kernel[(vector_num,)](
            acc_ws, lse_ws, o, seqlens,
            sinks_pass,
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            o.stride(0), o.stride(1), o.stride(2),
            sinks_pass.stride(0),
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            SINK_ENABLED=sink_enabled,
        )
        return o

    # -----------------------------------------------------------------------
    # Non-FD path: single-kernel decode with sink
    # -----------------------------------------------------------------------
    paged_decode_kernel[(cube_num,)](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables_i32,
        sinks_pass,
        batch_size,
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
        sinks_pass.stride(0),
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        SINK_ENABLED=sink_enabled,
    )
    return o


def paged_attention_mirror_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    max_kv_len_hint: Optional[int] = None,
) -> torch.Tensor:
    """Run KV-mirror prefill without specializing request-shape scalars."""
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert value_cache.shape == key_cache.shape
    assert head_dim == head_dim_cache
    assert num_q_heads % num_kv_heads == 0
    assert block_tables.shape[0] == batch_size
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
    sinks_pass = (
        sinks
        if sink_enabled
        else torch.empty(1, dtype=q.dtype, device=q.device)
    )

    o = torch.empty_like(q)
    block_tables_i32 = block_tables.to(dtype=torch.int32).contiguous()

    cube_num = get_num_cores("cube")
    vector_num = get_num_cores("vector")
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    group_size = num_q_heads // num_kv_heads
    max_kv_len = (
        int(max_kv_len_hint)
        if max_kv_len_hint is not None
        else int(seqlens.max().item())
    )

    if _should_use_flash_decode(
        batch_size,
        num_kv_heads,
        group_size,
        max_kv_len,
        cube_num,
    ):
        kv_split_parts = _compute_kv_split_parts(
            batch_size,
            num_kv_heads,
            max_kv_len,
            cube_num,
        )
        num_ws_tasks = batch_size * num_kv_heads * kv_split_parts
        acc_ws = torch.empty(
            (num_ws_tasks, group_size, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        lse_ws = torch.full(
            (num_ws_tasks, group_size),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        mirror_paged_decode_fd_kernel[(cube_num,)](
            q,
            key_cache,
            value_cache,
            seqlens,
            block_tables_i32,
            acc_ws,
            lse_ws,
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
            block_tables_i32.stride(0),
            block_tables_i32.stride(1),
            acc_ws.stride(0),
            acc_ws.stride(1),
            acc_ws.stride(2),
            lse_ws.stride(0),
            lse_ws.stride(1),
            softmax_scale,
            batch_size,
            kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        mirror_paged_decode_fd_reduce_kernel[(vector_num,)](
            acc_ws,
            lse_ws,
            o,
            seqlens,
            sinks_pass,
            acc_ws.stride(0),
            acc_ws.stride(1),
            acc_ws.stride(2),
            lse_ws.stride(0),
            lse_ws.stride(1),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            sinks_pass.stride(0),
            batch_size,
            kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            SINK_ENABLED=sink_enabled,
        )
        return o

    mirror_paged_decode_kernel[(cube_num,)](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables_i32,
        sinks_pass,
        batch_size,
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
        sinks_pass.stride(0),
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=gqa_interleave,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        SINK_ENABLED=sink_enabled,
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


@lru_cache(maxsize=1)
def get_num_cores(op_type="vector"):
    assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    return (
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        if op_type == "vector"
        else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
    )


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



@triton.jit
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
    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = min(64, triton.next_power_of_2(page_size))
    else:
        BLOCK_M = 128
        BLOCK_N = min(128, triton.next_power_of_2(page_size))

    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")

    num_q_chunks = triton.cdiv(tot_q_toks, BLOCK_M)
    num_tasks = num_q_chunks * num_q_heads
    n_programs = min(cube_num, num_tasks)
    grid = (n_programs,)

    if global_window_size is None:
        global_window_size = 0

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



# SWA decode and BS=1 KV-mirror prefill share this kernel.  Tight eager tables
# may have a different base alignment and row stride from the graph table.
@triton.jit(
    do_not_specialize=[
        "block_tables_ptr",
        "stride_bt_batch",
    ]
)
def _swa_paged_decode_sink_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    sinks_ptr,
    BATCH_SIZE,
    stride_qb,
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
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        # --- online-softmax init with optional sink ---
        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
            kv_seq_len - 1,
            1,
            kv_seq_len,
            BLOCK_SIZE_N,
            True,
            GLOBAL_WINDOW,
            LOCAL_WINDOW,
        )


        for kv_block_id in range(num_global_window_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            gw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N)) < GLOBAL_WINDOW
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                gw_mask = gw_mask | sw_mask
            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            mask = gw_mask & kv_mask

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(mask[None, :], 0.0, -2.0**30)

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp(qk - m_ij[:, None])

            pv = tl.dot(p.to(k_T.dtype), v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        for kv_block_id in range(non_global_window_start_block, num_total_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                mask = kv_mask & sw_mask
            else:
                mask = kv_mask

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(mask[None, :], 0.0, -2.0**30)

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp(qk - m_ij[:, None])

            pv = tl.dot(p.to(k_T.dtype), v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        o_ptrs = o_ptr + b_id * stride_ob + q_head_ids[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)



@triton.jit(
    do_not_specialize=[
        "BATCH_SIZE",
        "stride_bt_batch",
    ]
)
def mirror_swa_paged_decode_sink_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    sinks_ptr,
    BATCH_SIZE,
    stride_qb,
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
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_sink_head,
    softmax_scale,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SINK_ENABLED: tl.constexpr,
):
    """Request-shape-stable SWA+Sink kernel for KV-mirror prefill."""
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(
        HEAD_DIM <= BLOCK_SIZE_D,
        "HEAD_DIM should be <= BLOCK_SIZE_D",
    )
    tl.static_assert(
        PAGE_SIZE // BLOCK_SIZE_N * BLOCK_SIZE_N == PAGE_SIZE,
        "BLOCK_SIZE_N must be a divisor of PAGE_SIZE",
    )

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        b_id = kv_task_id // NUM_KV_HEADS
        kv_head_id = kv_task_id - b_id * NUM_KV_HEADS
        kv_seq_len = tl.load(seqlens_ptr + b_id)

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
            other=0.0,
        )

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        if SINK_ENABLED:
            s_h = tl.load(
                sinks_ptr + q_head_ids * stride_sink_head,
                mask=q_head_ids.to(tl.float32) < NUM_Q_HEADS,
                other=-float("inf"),
            ).to(tl.float32)
            m_i = m_i + s_h
            l_i = l_i + 1.0
        else:
            m_i = m_i - float("inf")
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        (
            num_global_window_blocks,
            non_global_window_start_block,
            num_total_blocks,
        ) = _swa_split_blocks(
            kv_seq_len - 1,
            1,
            kv_seq_len,
            BLOCK_SIZE_N,
            True,
            GLOBAL_WINDOW,
            LOCAL_WINDOW,
        )

        for kv_block_id in range(num_global_window_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(
                kv_block_start + BLOCK_SIZE_N,
                kv_seq_len,
            )
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start - logical_page_id * PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )
            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            block_offsets = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
            if GLOBAL_WINDOW is not None:
                gw_mask = block_offsets.to(tl.float32) < GLOBAL_WINDOW
            else:
                gw_mask = block_offsets.to(tl.float32) < 0.0
            if LOCAL_WINDOW is not None:
                sw_mask = (
                    (block_offsets + LOCAL_WINDOW).to(tl.float32)
                    >= (kv_seq_len - 1).to(tl.float32)
                )
                gw_mask = gw_mask | sw_mask
            kv_mask = (
                tl.arange(0, BLOCK_SIZE_N).to(tl.float32)
                < kv_block_len.to(tl.float32)
            )
            mask = gw_mask & kv_mask

            k_T = tl.load(
                K_T_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            v = tl.load(
                V_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(
                mask[None, :],
                0.0,
                -2.0**30,
            )

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            p = tl.math.exp(qk - m_ij[:, None])
            pv = tl.dot(p.to(k_T.dtype), v)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        for kv_block_id in range(
            non_global_window_start_block,
            num_total_blocks,
        ):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(
                kv_block_start + BLOCK_SIZE_N,
                kv_seq_len,
            )
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start - logical_page_id * PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )
            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            kv_mask = (
                tl.arange(0, BLOCK_SIZE_N).to(tl.float32)
                < kv_block_len.to(tl.float32)
            )
            if LOCAL_WINDOW is not None:
                sw_mask = (
                    (
                        kv_block_start
                        + tl.arange(0, BLOCK_SIZE_N)
                        + LOCAL_WINDOW
                    ).to(tl.float32)
                    >= (kv_seq_len - 1).to(tl.float32)
                )
                mask = kv_mask & sw_mask
            else:
                mask = kv_mask

            k_T = tl.load(
                K_T_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            v = tl.load(
                V_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(
                mask[None, :],
                0.0,
                -2.0**30,
            )

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            p = tl.math.exp(qk - m_ij[:, None])
            pv = tl.dot(p.to(k_T.dtype), v)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if kv_seq_len.to(tl.float32) > 0.0:
            acc = acc / l_i[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            acc.to(o_ptr.dtype.element_ty),
            mask=offs_d[None, :].to(tl.float32) < HEAD_DIM,
        )


def swa_paged_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    # --- sink setup ---
    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
        # Keep the checkpoint's FP32 sink; casting it to Q dtype is unnecessary.
    sinks_pass = sinks if sink_enabled else torch.empty(1, dtype=q.dtype, device=q.device)

    o = torch.empty_like(q, memory_format=torch.contiguous_format)

    cube_num = get_num_cores("cube")
    grid = (cube_num, )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))

    # Note(chenyifan):
    #   under swa, the kv workload is rather evenly across diffrent queries,
    #   so we have low necessity to apply split-kv strategy

    _swa_paged_decode_sink_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables,
        sinks_pass,
        batch_size,
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
        block_tables.stride(0),
        block_tables.stride(1),
        sinks_pass.stride(0),
        softmax_scale,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        SINK_ENABLED=sink_enabled,
    )
    return o


def swa_paged_mirror_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run SWA KV-mirror prefill without request-shape specialization."""
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    sink_enabled = sinks is not None
    if sink_enabled:
        assert sinks.shape == (num_q_heads,), (
            f"sinks must have shape ({num_q_heads},), but got {tuple(sinks.shape)}"
        )
    sinks_pass = (
        sinks
        if sink_enabled
        else torch.empty(1, dtype=q.dtype, device=q.device)
    )

    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    cube_num = get_num_cores("cube")
    grid = (cube_num,)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))

    mirror_swa_paged_decode_sink_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables,
        sinks_pass,
        batch_size,
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
        block_tables.stride(0),
        block_tables.stride(1),
        sinks_pass.stride(0),
        softmax_scale,
        GLOBAL_WINDOW=global_window_size,
        LOCAL_WINDOW=local_window_size,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=gqa_interleave,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        SINK_ENABLED=sink_enabled,
    )
    return o
