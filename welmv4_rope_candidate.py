from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

try:
    from sglang.srt.utils import is_npu
except ModuleNotFoundError:
    # The remote optimization worker is intentionally standalone and does not
    # carry a NEWSGLANG checkout.  This is the only adaptation made to the
    # frozen production baseline.
    def is_npu() -> bool:
        return hasattr(torch, "npu") and torch.npu.is_available()


_WELMV4_ROPE_PREFILL_TOKEN_BLOCK = 64
_WELMV4_ROPE_PREFILL_EXACT64_THRESHOLD = 576
_WELMV4_ROPE_PREFILL_ALL_M_THRESHOLD = 640
_WELMV4_ROPE_PROGRAMS_PER_VECTOR_CORE = 8
_WELMV4_ROPE_PREFILL_NUM_STAGES = 1

_WELMV4_RMS_NORM_HIDDEN_SIZE = 2048
_WELMV4_RMS_NORM_BLOCK_SIZE = 2048
_WELMV4_RMS_NORM_TWO_ROW_MIN = 40
_WELMV4_RMS_NORM_FOUR_ROW_MIN = 224
_WELMV4_RMS_NORM_TWO_ROWS = 2
_WELMV4_RMS_NORM_FOUR_ROWS = 4


def build_welmv4_rope_segment_tile_starts(
    segment_lengths: Optional[Sequence[int]],
    *,
    batch_size: int,
    num_position_tokens: int,
    ordinary_prefill: bool,
) -> Optional[list[int]]:
    """Build 64-token tile starts for independently contiguous requests.

    The final entry is a sentinel.  Consequently, the next entry is both the
    end of a short tail tile and the start of the following tile/request.
    Return ``None`` whenever the framework cannot prove that the supplied
    positions contain exactly one concatenated segment per request.
    """
    if not ordinary_prefill or batch_size <= 1 or segment_lengths is None:
        return None
    if len(segment_lengths) != batch_size:
        return None

    lengths = [int(length) for length in segment_lengths]
    if any(length < 0 for length in lengths):
        return None
    if sum(lengths) != num_position_tokens:
        return None
    if num_position_tokens <= _WELMV4_ROPE_PREFILL_ALL_M_THRESHOLD:
        return None

    tile_starts: list[int] = []
    segment_start = 0
    for length in lengths:
        tile_starts.extend(
            range(
                segment_start,
                segment_start + length,
                _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            )
        )
        segment_start += length
    tile_starts.append(segment_start)
    return tile_starts


def _get_num_sms(multiplier: int = 1) -> int:
    if is_npu():
        device = torch.device("npu", torch.npu.current_device())
        return (
            _welmv4_vector_core_count(device) * multiplier
        )

    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        * multiplier
    )


@triton.jit
def _welmv4_fused_rms_norm_compute_npu(hidden, gamma, cols: int, eps: tl.constexpr):
    hidden = hidden.to(gamma.dtype).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma
    return out


@triton.jit
def _welmv4_fused_rms_norm_compute_2d_npu(
    hidden, gamma, cols: int, eps: tl.constexpr
):
    hidden = hidden.to(gamma.dtype).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms[:, None]
    out *= gamma[None, :]
    return out


@triton.jit(do_not_specialize=["rows"])
def _welmv4_fused_rms_norm_true_true_kernel_npu(
    hidden_states_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    fp32_out_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    row_begin = program_id * rows // num_programs
    row_end = (program_id + 1) * rows // num_programs
    cols_off = tl.arange(0, BLOCK_SIZE)
    gamma_shm = tl.load(gamma_ptr + cols_off)
    output_dtype = out_ptr.dtype.element_ty
    for row_id in tl.range(row_begin, row_end, num_stages=2):
        offsets = row_id * cols + cols_off
        hidden = tl.load(hidden_states_ptr + offsets).to(tl.float32)
        if residual_ptr is not None:
            residual = tl.load(residual_ptr + offsets).to(tl.float32)
            hidden = hidden + residual

        out = _welmv4_fused_rms_norm_compute_npu(
            hidden, gamma_shm, cols, eps
        )
        tl.store(fp32_out_ptr + offsets, out)
        tl.store(out_ptr + offsets, out.to(output_dtype))


@triton.jit(do_not_specialize=["rows"])
def _welmv4_fused_rms_norm_true_true_multirow_kernel_npu(
    hidden_states_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    fp32_out_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_blocks = tl.cdiv(rows, BLOCK_ROWS)
    block_begin = program_id * num_row_blocks // num_programs
    block_end = (program_id + 1) * num_row_blocks // num_programs
    row_lanes = tl.arange(0, BLOCK_ROWS)
    cols_off = tl.arange(0, BLOCK_SIZE)
    gamma_shm = tl.load(gamma_ptr + cols_off)
    output_dtype = out_ptr.dtype.element_ty
    for block_id in tl.range(block_begin, block_end, num_stages=2):
        row_ids = block_id * BLOCK_ROWS + row_lanes
        offsets = row_ids[:, None] * cols + cols_off[None, :]
        mask = row_ids[:, None] < rows
        hidden = tl.load(
            hidden_states_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        if residual_ptr is not None:
            residual = tl.load(
                residual_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            hidden = hidden + residual

        out = _welmv4_fused_rms_norm_compute_2d_npu(
            hidden, gamma_shm, cols, eps
        )
        tl.store(fp32_out_ptr + offsets, out, mask=mask)
        tl.store(out_ptr + offsets, out.to(output_dtype), mask=mask)


@triton.jit(do_not_specialize=["rows"])
def _welmv4_fused_rms_norm_false_false_kernel_npu(
    hidden_states_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    row_begin = program_id * rows // num_programs
    row_end = (program_id + 1) * rows // num_programs
    cols_off = tl.arange(0, BLOCK_SIZE)
    gamma_shm = tl.load(gamma_ptr + cols_off)
    output_dtype = out_ptr.dtype.element_ty
    for row_id in tl.range(row_begin, row_end, num_stages=2):
        offsets = row_id * cols + cols_off
        hidden = tl.load(hidden_states_ptr + offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        norm_input = hidden + residual
        tl.store(out_residual_ptr + offsets, norm_input)

        out = _welmv4_fused_rms_norm_compute_npu(
            norm_input, gamma_shm, cols, eps
        )
        tl.store(out_ptr + offsets, out.to(output_dtype))


@triton.jit(do_not_specialize=["rows"])
def _welmv4_fused_rms_norm_false_false_multirow_kernel_npu(
    hidden_states_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_blocks = tl.cdiv(rows, BLOCK_ROWS)
    block_begin = program_id * num_row_blocks // num_programs
    block_end = (program_id + 1) * num_row_blocks // num_programs
    row_lanes = tl.arange(0, BLOCK_ROWS)
    cols_off = tl.arange(0, BLOCK_SIZE)
    gamma_shm = tl.load(gamma_ptr + cols_off)
    output_dtype = out_ptr.dtype.element_ty
    for block_id in tl.range(block_begin, block_end, num_stages=2):
        row_ids = block_id * BLOCK_ROWS + row_lanes
        offsets = row_ids[:, None] * cols + cols_off[None, :]
        mask = row_ids[:, None] < rows
        hidden = tl.load(
            hidden_states_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        norm_input = hidden + residual
        tl.store(out_residual_ptr + offsets, norm_input, mask=mask)

        out = _welmv4_fused_rms_norm_compute_2d_npu(
            norm_input, gamma_shm, cols, eps
        )
        tl.store(out_ptr + offsets, out.to(output_dtype), mask=mask)


def welmv4_fused_rms_norm_true_true_npu(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
    num_vector_cores: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU-only True+True path; the unused BF16 residual aliases output."""
    rows = hidden_states.shape[0]
    output = torch.empty_like(hidden_states, dtype=output_dtype)
    fp32_out = torch.empty_like(hidden_states, dtype=torch.float32)

    if rows < _WELMV4_RMS_NORM_TWO_ROW_MIN:
        num_programs = min(rows, num_vector_cores)
        _welmv4_fused_rms_norm_true_true_kernel_npu[(num_programs,)](
            hidden_states,
            residual,
            weight,
            output,
            fp32_out,
            rows,
            _WELMV4_RMS_NORM_HIDDEN_SIZE,
            eps,
            _WELMV4_RMS_NORM_BLOCK_SIZE,
        )
    else:
        block_rows = (
            _WELMV4_RMS_NORM_TWO_ROWS
            if rows < _WELMV4_RMS_NORM_FOUR_ROW_MIN
            else _WELMV4_RMS_NORM_FOUR_ROWS
        )
        num_programs = min(triton.cdiv(rows, block_rows), num_vector_cores)
        _welmv4_fused_rms_norm_true_true_multirow_kernel_npu[(num_programs,)](
            hidden_states,
            residual,
            weight,
            output,
            fp32_out,
            rows,
            _WELMV4_RMS_NORM_HIDDEN_SIZE,
            eps,
            _WELMV4_RMS_NORM_BLOCK_SIZE,
            block_rows,
        )

    return output, output, fp32_out


def welmv4_fused_rms_norm_false_false_npu(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
    num_vector_cores: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU-only False+False path with the required FP32 pre-norm residual."""
    rows = hidden_states.shape[0]
    output = torch.empty_like(hidden_states, dtype=output_dtype)
    out_residual = torch.empty_like(residual)

    if rows < _WELMV4_RMS_NORM_TWO_ROW_MIN:
        num_programs = min(rows, num_vector_cores)
        _welmv4_fused_rms_norm_false_false_kernel_npu[(num_programs,)](
            hidden_states,
            residual,
            weight,
            output,
            out_residual,
            rows,
            _WELMV4_RMS_NORM_HIDDEN_SIZE,
            eps,
            _WELMV4_RMS_NORM_BLOCK_SIZE,
        )
    else:
        block_rows = (
            _WELMV4_RMS_NORM_TWO_ROWS
            if rows < _WELMV4_RMS_NORM_FOUR_ROW_MIN
            else _WELMV4_RMS_NORM_FOUR_ROWS
        )
        num_programs = min(triton.cdiv(rows, block_rows), num_vector_cores)
        _welmv4_fused_rms_norm_false_false_multirow_kernel_npu[(num_programs,)](
            hidden_states,
            residual,
            weight,
            output,
            out_residual,
            rows,
            _WELMV4_RMS_NORM_HIDDEN_SIZE,
            eps,
            _WELMV4_RMS_NORM_BLOCK_SIZE,
            block_rows,
        )

    return output, out_residual


@triton.autotune(
    configs=[
        triton.Config(
            {"GROUP_SIZE_M": group_size_m, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k},
        )
        for group_size_m in [4, 8, 16, 32, 64]
        for block_size_n in [512]
        for block_size_k in [32, 64, 128, 256, 512, 1024, 2048]
    ],
    key=["N", "K"],
)
@triton.jit(do_not_specialize=["M"])
def mmq_style_router_linear_kernel_npu(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * GROUP_SIZE_M + tl.arange(0, GROUP_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((GROUP_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def mmq_style_router_linear_npu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and weight.dim() == 2
    assert x.shape[1] == weight.shape[1], "hidden_size mismatch: x.shape[1] must equal weight.shape[1]"

    M, K = x.shape
    N = weight.shape[0]

    x = x.contiguous()
    weight = weight.to(dtype=x.dtype).contiguous()

    c = torch.empty((M, N), dtype=torch.float32, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["GROUP_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    mmq_style_router_linear_kernel_npu[grid](
        x,
        weight,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


@triton.jit
def _rope_npu(
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    """单 token 多 head 的 RoPE 辅助 kernel (就地修改)。

    [优化] 消除 trans/split/join/reshape 等 layout 变换,改为分两半
    load → 计算 → 分两半 store,所有访存行优先连续,避免 NPU store
    转置退化。计算逻辑 (GPT-NeoX rotate-half) 完全等价。
    """
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_head_offset = tl.arange(0, num_heads_blocked)
    half_rope_offset = tl.arange(0, half_rope_dim)
    mask = num_head_offset[:, None] < num_heads
    base = data_ptr + num_head_offset[:, None] * head_dim
    # 分两半加载: 前半 x1 与后半 x2,每半沿 rope 维度连续
    x1 = tl.load(
        base + half_rope_offset[None, :], mask=mask, care_padding=False
    )
    x2 = tl.load(
        base + (half_rope_dim + half_rope_offset)[None, :],
        mask=mask,
        care_padding=False,
    )
    # GPT-NeoX rotate-half
    x_out1 = x1 * cos - x2 * sin
    x_out2 = x1 * sin + x2 * cos
    # 分两半存储: 行优先连续写回,无转置退化
    tl.store(base + half_rope_offset[None, :], x_out1, mask=mask)
    tl.store(
        base + (half_rope_dim + half_rope_offset)[None, :], x_out2, mask=mask
    )


@triton.jit(do_not_specialize=["N", "BS"])
def _welmv4_inplace_rope_kernel_npu(
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
    """WeLMv4 尾部 RoPE 主 kernel (就地修改 Q/K)。

    与原始 kernel 的差异:
      - 所有 tl.load 添加 care_padding=False
      - 调用的 _rope_npu 已重写 (消除 trans/split/join, 分半 load/store)
    Grid (1D, 跨步循环)、计算逻辑、条件分支均保持不变。
    """
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_off = tl.arange(0, half_rope_dim)
    sin_off = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(
        tl.program_id(0), N, tl.num_programs(0), num_stages=num_stages
    ):
        position_id = tl.load(position_ptr + token_id).to(tl.int32)
        # [NPU 迁移] 添加 care_padding=False
        cos_sin_cache = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + cos_off, care_padding=False
        )
        sin_sin_cache = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + sin_off, care_padding=False
        )
        q_data_ptr = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data_ptr = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _rope_npu(
            k_data_ptr, cos_sin_cache, sin_sin_cache,
            num_k_heads, num_k_heads_blocked, head_dim, rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                position_id = tl.load(last_index_ptr + token_id).to(tl.int32)
                position_id = tl.load(position_ptr + position_id).to(tl.int32)
                # [NPU 迁移] 添加 care_padding=False
                cos_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + cos_off,
                    care_padding=False,
                )
                sin_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + sin_off,
                    care_padding=False,
                )
                _rope_npu(
                    q_data_ptr, cos_sin_cache, sin_sin_cache,
                    num_q_heads, num_q_heads_blocked, head_dim, rope_dim,
                )
        else:
            _rope_npu(
                q_data_ptr, cos_sin_cache, sin_sin_cache,
                num_q_heads, num_q_heads_blocked, head_dim, rope_dim,
            )


@triton.jit
def _welmv4_apply_token_block_rope_npu(
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
        x1 = tl.load(base + rope_offsets[None, :], care_padding=False)
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
def _welmv4_apply_token_head_block_rope_npu(
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
    x1 = tl.load(base + rope_offsets[None, None, :], care_padding=False)
    x2 = tl.load(
        base + half_rope_dim + rope_offsets[None, None, :],
        care_padding=False,
    )
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    tl.store(base + rope_offsets[None, None, :], out1)
    tl.store(base + half_rope_dim + rope_offsets[None, None, :], out2)


@triton.jit
def _welmv4_apply_masked_token_head_block_rope_npu(
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
    """Apply RoPE to a 2-head Q tile while preserving segment-tail masks."""
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
    tl.store(base + rope_offsets[None, None, :], out1, mask=mask)
    tl.store(
        base + half_rope_dim + rope_offsets[None, None, :],
        out2,
        mask=mask,
    )


@triton.jit(do_not_specialize=["num_token_blocks", "N"])
def _welmv4_inplace_rope_prefill_kernel_npu(
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
    num_k_heads: tl.constexpr,
):
    """A5 blocked prefill kernel for the WeLM head_dim=256/rope_dim=64 path."""
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
            # Runtime position rows are genuinely discrete on the generic
            # path.  Avoid the backend's scalarized per-row GM copies.
            tl.extra.cann.extension.compile_hint(cos, "mayDiscretememaccess")
            tl.extra.cann.extension.compile_hint(sin, "mayDiscretememaccess")

        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim
        if num_k_heads == 1:
            _welmv4_apply_token_block_rope_npu(
                k_data, token_offsets, k_token_stride, cos, sin, token_mask,
                masked, head_dim, rope_dim,
            )
        else:
            for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                _welmv4_apply_token_block_rope_npu(
                    k_data + k_head_id * head_dim,
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
            q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
            if num_q_heads == 6:
                for head_id in tl.static_range(0, num_q_heads):
                    _welmv4_apply_token_block_rope_npu(
                        q_data + head_id * head_dim,
                        token_offsets,
                        q_token_stride,
                        cos,
                        sin,
                        token_mask,
                        masked,
                        head_dim,
                        rope_dim,
                    )
            else:
                for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                    _welmv4_apply_masked_token_head_block_rope_npu(
                        q_data + q_head_base * head_dim,
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
            q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
            if num_q_heads == 6:
                # Preserve the production Q6/K1 instruction shape exactly.
                _welmv4_apply_token_head_block_rope_npu(
                    q_data, token_offsets, q_token_stride, cos, sin, 2,
                    head_dim, rope_dim,
                )
                _welmv4_apply_token_head_block_rope_npu(
                    q_data + 2 * head_dim, token_offsets, q_token_stride,
                    cos, sin, 2, head_dim, rope_dim,
                )
                _welmv4_apply_token_head_block_rope_npu(
                    q_data + 4 * head_dim, token_offsets, q_token_stride,
                    cos, sin, 2, head_dim, rope_dim,
                )
            else:
                # Q12/Q24 reuse the proven 2-head UB tile without padding to
                # Q16/Q32. Keep the head loop structured to bound live UB.
                for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                    _welmv4_apply_token_head_block_rope_npu(
                        q_data + q_head_base * head_dim,
                        token_offsets,
                        q_token_stride,
                        cos,
                        sin,
                        2,
                        head_dim,
                        rope_dim,
                    )


@triton.jit(do_not_specialize=["num_token_blocks", "N"])
def _welmv4_inplace_rope_contiguous_prefill_kernel_npu(
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
    num_k_heads: tl.constexpr,
):
    """A5 fast path for one ordinary-prefill request with contiguous positions."""
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

        # For one non-speculative extend request, positions are contiguous
        # from an arbitrary runtime prefix offset.  Express that fact so the
        # FP32 cache reads lower to regular 2D GM-to-UB transfers.
        position_base = tl.load(position_ptr + token_base).to(tl.int32)
        position_ids = position_base + token_offsets
        if masked:
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

        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim
        if num_k_heads == 1:
            _welmv4_apply_token_block_rope_npu(
                k_data, token_offsets, k_token_stride, cos, sin, token_mask,
                masked, head_dim, rope_dim,
            )
        else:
            for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                _welmv4_apply_token_block_rope_npu(
                    k_data + k_head_id * head_dim,
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
            q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
            if num_q_heads == 6:
                for head_id in tl.static_range(0, num_q_heads):
                    _welmv4_apply_token_block_rope_npu(
                        q_data + head_id * head_dim,
                        token_offsets,
                        q_token_stride,
                        cos,
                        sin,
                        token_mask,
                        masked,
                        head_dim,
                        rope_dim,
                    )
            else:
                for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                    _welmv4_apply_masked_token_head_block_rope_npu(
                        q_data + q_head_base * head_dim,
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
            q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
            if num_q_heads == 6:
                _welmv4_apply_token_head_block_rope_npu(
                    q_data, token_offsets, q_token_stride, cos, sin, 2,
                    head_dim, rope_dim,
                )
                _welmv4_apply_token_head_block_rope_npu(
                    q_data + 2 * head_dim, token_offsets, q_token_stride,
                    cos, sin, 2, head_dim, rope_dim,
                )
                _welmv4_apply_token_head_block_rope_npu(
                    q_data + 4 * head_dim, token_offsets, q_token_stride,
                    cos, sin, 2, head_dim, rope_dim,
                )
            else:
                for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                    _welmv4_apply_token_head_block_rope_npu(
                        q_data + q_head_base * head_dim,
                        token_offsets,
                        q_token_stride,
                        cos,
                        sin,
                        2,
                        head_dim,
                        rope_dim,
                    )


@triton.jit(do_not_specialize=["num_segment_tiles", "N"])
def _welmv4_inplace_rope_segmented_prefill_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    segment_tile_starts_ptr: tl.tensor,
    num_segment_tiles: int,
    N: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
):
    """A5 multi-request prefill path with contiguous positions per segment."""
    half_rope_dim: tl.constexpr = rope_dim // 2
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)
    for tile_id in tl.range(
        tl.program_id(0),
        num_segment_tiles,
        tl.num_programs(0),
        num_stages=num_stages,
    ):
        token_base = tl.load(segment_tile_starts_ptr + tile_id).to(tl.int32)
        token_end = tl.load(segment_tile_starts_ptr + tile_id + 1).to(tl.int32)
        token_mask = token_offsets < token_end - token_base

        # Positions advance regularly inside each request, although different
        # requests have unrelated runtime prefix lengths.
        position_base = tl.load(position_ptr + token_base).to(tl.int32)
        position_ids = position_base + token_offsets
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

        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim
        if num_k_heads == 1:
            _welmv4_apply_token_block_rope_npu(
                k_data, token_offsets, k_token_stride, cos, sin, token_mask,
                True, head_dim, rope_dim,
            )
        else:
            for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                _welmv4_apply_token_block_rope_npu(
                    k_data + k_head_id * head_dim,
                    token_offsets,
                    k_token_stride,
                    cos,
                    sin,
                    token_mask,
                    True,
                    head_dim,
                    rope_dim,
                )
        q_data = q_ptr + token_base * q_token_stride + head_dim - rope_dim
        if num_q_heads == 6:
            _welmv4_apply_masked_token_head_block_rope_npu(
                q_data, token_offsets, q_token_stride, cos, sin, token_mask,
                2, head_dim, rope_dim,
            )
            _welmv4_apply_masked_token_head_block_rope_npu(
                q_data + 2 * head_dim, token_offsets, q_token_stride, cos,
                sin, token_mask, 2, head_dim, rope_dim,
            )
            _welmv4_apply_masked_token_head_block_rope_npu(
                q_data + 4 * head_dim, token_offsets, q_token_stride, cos,
                sin, token_mask, 2, head_dim, rope_dim,
            )
        else:
            for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                _welmv4_apply_masked_token_head_block_rope_npu(
                    q_data + q_head_base * head_dim,
                    token_offsets,
                    q_token_stride,
                    cos,
                    sin,
                    token_mask,
                    2,
                    head_dim,
                    rope_dim,
                )


@triton.jit(do_not_specialize=["num_token_blocks", "N"])
def _welmv4_inplace_rope_contiguous_mirror_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    num_token_blocks: int,
    N: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads: tl.constexpr,
):
    """A5 BS=1 mirror path for a long globally contiguous K sequence."""
    half_rope_dim: tl.constexpr = rope_dim // 2
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)
    program_id = tl.program_id(0)

    # Q contains exactly one packed row.  Only program 0 writes it, while all
    # programs participate in the much larger K workload.
    if program_id == 0:
        q_source_token = tl.load(last_index_ptr).to(tl.int32)
        q_position = tl.load(position_ptr + q_source_token).to(tl.int32)
        q_cos = tl.load(
            cos_sin_cache_ptr + q_position * rope_dim + cos_offsets,
            care_padding=False,
        )
        q_sin = tl.load(
            cos_sin_cache_ptr + q_position * rope_dim + sin_offsets,
            care_padding=False,
        )
        q_data = q_ptr + head_dim - rope_dim
        if num_q_heads == 6:
            _rope_npu(
                q_data, q_cos, q_sin, num_q_heads, num_q_heads_blocked,
                head_dim, rope_dim,
            )
        else:
            for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                _rope_npu(
                    q_data + q_head_base * head_dim,
                    q_cos,
                    q_sin,
                    2,
                    2,
                    head_dim,
                    rope_dim,
                )

    # Give each physical AIV one consecutive K-block interval.  N and the
    # derived block count remain runtime values to avoid shape recompilation.
    blocks_per_program = tl.cdiv(num_token_blocks, tl.num_programs(0))
    block_start = program_id * blocks_per_program
    block_end = tl.minimum(block_start + blocks_per_program, num_token_blocks)
    for block_id in tl.range(
        block_start,
        block_end,
        num_stages=num_stages,
    ):
        token_base = block_id * token_block
        token_ids = token_base + token_offsets
        position_base = tl.load(position_ptr + token_base).to(tl.int32)
        position_ids = position_base + token_offsets
        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim
        block_end_token = token_base + token_block

        if block_end_token.to(tl.float32) <= N.to(tl.float32):
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
            if num_k_heads == 1:
                _welmv4_apply_token_block_rope_npu(
                    k_data, token_offsets, k_token_stride, cos, sin,
                    token_offsets == token_offsets, False, head_dim, rope_dim,
                )
            else:
                for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                    _welmv4_apply_token_block_rope_npu(
                        k_data + k_head_id * head_dim,
                        token_offsets,
                        k_token_stride,
                        cos,
                        sin,
                        token_offsets == token_offsets,
                        False,
                        head_dim,
                        rope_dim,
                    )
        else:
            token_mask = token_ids.to(tl.float32) < N.to(tl.float32)
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
            if num_k_heads == 1:
                _welmv4_apply_token_block_rope_npu(
                    k_data, token_offsets, k_token_stride, cos, sin,
                    token_mask, True, head_dim, rope_dim,
                )
            else:
                for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                    _welmv4_apply_token_block_rope_npu(
                        k_data + k_head_id * head_dim,
                        token_offsets,
                        k_token_stride,
                        cos,
                        sin,
                        token_mask,
                        True,
                        head_dim,
                        rope_dim,
                    )


@triton.jit(do_not_specialize=["num_segment_tiles", "BS"])
def _welmv4_inplace_rope_segmented_mirror_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    segment_tile_starts_ptr: tl.tensor,
    num_segment_tiles: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads: tl.constexpr,
):
    """A5 multi-request mirror path with request-local contiguous K tiles."""
    half_rope_dim: tl.constexpr = rope_dim // 2
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Mirror Q contains one row per request.  Each AIV owns one consecutive
    # request interval and obtains that row's position from last_index.
    requests_per_program = tl.cdiv(BS, num_programs)
    request_start = program_id * requests_per_program
    request_end = tl.minimum(request_start + requests_per_program, BS)
    for request_id in tl.range(request_start, request_end, num_stages=1):
        q_source_token = tl.load(last_index_ptr + request_id).to(tl.int32)
        q_position = tl.load(position_ptr + q_source_token).to(tl.int32)
        q_cos = tl.load(
            cos_sin_cache_ptr + q_position * rope_dim + cos_offsets,
            care_padding=False,
        )
        q_sin = tl.load(
            cos_sin_cache_ptr + q_position * rope_dim + sin_offsets,
            care_padding=False,
        )
        q_data = q_ptr + request_id * q_token_stride + head_dim - rope_dim
        if num_q_heads == 6:
            _rope_npu(
                q_data, q_cos, q_sin, num_q_heads, num_q_heads_blocked,
                head_dim, rope_dim,
            )
        else:
            for q_head_base in tl.range(0, num_q_heads, 2, num_stages=1):
                _rope_npu(
                    q_data + q_head_base * head_dim,
                    q_cos,
                    q_sin,
                    2,
                    2,
                    head_dim,
                    rope_dim,
                )

    # Tile boundaries are built independently per request, so no tile crosses
    # a position discontinuity.  Every AIV receives one consecutive tile range.
    tiles_per_program = tl.cdiv(num_segment_tiles, num_programs)
    tile_start_id = program_id * tiles_per_program
    tile_end_id = tl.minimum(
        tile_start_id + tiles_per_program, num_segment_tiles
    )
    for tile_id in tl.range(
        tile_start_id,
        tile_end_id,
        num_stages=num_stages,
    ):
        token_base = tl.load(segment_tile_starts_ptr + tile_id).to(tl.int32)
        token_end = tl.load(segment_tile_starts_ptr + tile_id + 1).to(tl.int32)
        token_count = token_end - token_base
        position_base = tl.load(position_ptr + token_base).to(tl.int32)
        position_ids = position_base + token_offsets
        k_data = k_ptr + token_base * k_token_stride + head_dim - rope_dim

        if token_count.to(tl.float32) >= token_block:
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
            if num_k_heads == 1:
                _welmv4_apply_token_block_rope_npu(
                    k_data, token_offsets, k_token_stride, cos, sin,
                    token_offsets == token_offsets, False, head_dim, rope_dim,
                )
            else:
                for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                    _welmv4_apply_token_block_rope_npu(
                        k_data + k_head_id * head_dim,
                        token_offsets,
                        k_token_stride,
                        cos,
                        sin,
                        token_offsets == token_offsets,
                        False,
                        head_dim,
                        rope_dim,
                    )
        else:
            token_mask = token_offsets.to(tl.float32) < token_count.to(
                tl.float32
            )
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
            if num_k_heads == 1:
                _welmv4_apply_token_block_rope_npu(
                    k_data, token_offsets, k_token_stride, cos, sin,
                    token_mask, True, head_dim, rope_dim,
                )
            else:
                for k_head_id in tl.range(0, num_k_heads, num_stages=1):
                    _welmv4_apply_token_block_rope_npu(
                        k_data + k_head_id * head_dim,
                        token_offsets,
                        k_token_stride,
                        cos,
                        sin,
                        token_mask,
                        True,
                        head_dim,
                        rope_dim,
                    )


@triton.jit(do_not_specialize=["num_tasks", "N"])
def _welmv4_inplace_rope_head_parallel_prefill_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    segment_tile_starts_ptr: tl.tensor,
    num_tasks: int,
    N: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    masked: tl.constexpr,
    position_mode: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
):
    """DP-attention prefill with token tiles and 2-head groups as tasks."""
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_q_head_groups: tl.constexpr = num_q_heads // 2
    num_head_groups: tl.constexpr = num_k_heads + num_q_head_groups
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)

    tasks_per_program = tl.cdiv(num_tasks, tl.num_programs(0))
    task_start = tl.program_id(0) * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    for task_id in tl.range(task_start, task_end, num_stages=1):
        task_id_i32 = task_id.to(tl.int32)
        tile_id = task_id_i32 // num_head_groups
        head_group_id = task_id_i32 - tile_id * num_head_groups
        if position_mode == 2:
            token_base = tl.load(segment_tile_starts_ptr + tile_id).to(tl.int32)
            token_end = tl.load(segment_tile_starts_ptr + tile_id + 1).to(
                tl.int32
            )
            token_mask = token_offsets.to(tl.float32) < (
                token_end - token_base
            ).to(tl.float32)
        else:
            token_base = tile_id * token_block
            token_mask = (token_base + token_offsets).to(tl.float32) < N.to(
                tl.float32
            )

        if position_mode == 0:
            position_ids = tl.load(
                position_ptr + token_base + token_offsets,
                mask=token_mask,
                other=0,
            ).to(tl.int32)
        else:
            position_base = tl.load(position_ptr + token_base).to(tl.int32)
            position_ids = position_base + token_offsets

        if masked:
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
        if position_mode == 0:
            tl.extra.cann.extension.compile_hint(cos, "mayDiscretememaccess")
            tl.extra.cann.extension.compile_hint(sin, "mayDiscretememaccess")

        if head_group_id.to(tl.float32) < num_k_heads:
            k_data = (
                k_ptr
                + token_base * k_token_stride
                + head_group_id * head_dim
                + head_dim
                - rope_dim
            )
            _welmv4_apply_token_block_rope_npu(
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
        else:
            q_head_base = (head_group_id - num_k_heads) * 2
            q_data = (
                q_ptr
                + token_base * q_token_stride
                + q_head_base * head_dim
                + head_dim
                - rope_dim
            )
            if masked:
                _welmv4_apply_masked_token_head_block_rope_npu(
                    q_data,
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
                _welmv4_apply_token_head_block_rope_npu(
                    q_data,
                    token_offsets,
                    q_token_stride,
                    cos,
                    sin,
                    2,
                    head_dim,
                    rope_dim,
                )


@triton.jit(
    do_not_specialize=[
        "num_q_tasks",
        "num_tasks",
        "N",
    ]
)
def _welmv4_inplace_rope_head_parallel_mirror_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    segment_tile_starts_ptr: tl.tensor,
    num_q_tasks: int,
    num_tasks: int,
    N: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    token_block: tl.constexpr,
    segmented: tl.constexpr,
    masked: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
):
    """DP-attention mirror RoPE with Q groups and K tiles as AIV tasks."""
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_q_head_groups: tl.constexpr = num_q_heads // 2
    token_offsets = tl.arange(0, token_block)
    cos_offsets = tl.arange(0, half_rope_dim)
    sin_offsets = tl.arange(half_rope_dim, rope_dim)

    tasks_per_program = tl.cdiv(num_tasks, tl.num_programs(0))
    task_start = tl.program_id(0) * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    for task_id in tl.range(task_start, task_end, num_stages=1):
        task_id_i32 = task_id.to(tl.int32)
        if task_id_i32.to(tl.float32) < num_q_tasks.to(tl.float32):
            request_id = task_id_i32 // num_q_head_groups
            q_head_group = task_id_i32 - request_id * num_q_head_groups
            q_source_token = tl.load(last_index_ptr + request_id).to(tl.int32)
            q_position = tl.load(position_ptr + q_source_token).to(tl.int32)
            q_cos = tl.load(
                cos_sin_cache_ptr + q_position * rope_dim + cos_offsets,
                care_padding=False,
            )
            q_sin = tl.load(
                cos_sin_cache_ptr + q_position * rope_dim + sin_offsets,
                care_padding=False,
            )
            q_data = (
                q_ptr
                + request_id * q_token_stride
                + q_head_group * 2 * head_dim
                + head_dim
                - rope_dim
            )
            _rope_npu(
                q_data, q_cos, q_sin, 2, 2, head_dim, rope_dim
            )
        else:
            k_task_id = task_id_i32 - num_q_tasks.to(tl.int32)
            tile_id = k_task_id // num_k_heads
            k_head_id = k_task_id - tile_id * num_k_heads
            if segmented:
                token_base = tl.load(segment_tile_starts_ptr + tile_id).to(
                    tl.int32
                )
                token_end = tl.load(segment_tile_starts_ptr + tile_id + 1).to(
                    tl.int32
                )
                token_mask = token_offsets.to(tl.float32) < (
                    token_end - token_base
                ).to(tl.float32)
            else:
                token_base = tile_id * token_block
                token_mask = (token_base + token_offsets).to(tl.float32) < N.to(
                    tl.float32
                )

            position_base = tl.load(position_ptr + token_base).to(tl.int32)
            position_ids = position_base + token_offsets
            if masked:
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
                + k_head_id * head_dim
                + head_dim
                - rope_dim
            )
            _welmv4_apply_token_block_rope_npu(
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


def welmv4_inplace_rope_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    last_index: torch.Tensor = None,
    head_dim: int = 128,
    rope_dim: int = 64,
    num_stages: int = 4,
    positions_are_contiguous: bool = False,
    segment_tile_starts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 就地应用尾部 RoPE。

    Args:
        query: (N, num_q_heads, head_dim)
        key:   (N, num_k_heads, head_dim)
        positions: (N,) int32
        cos_sin_cache: (max_pos, rope_dim) float32, 前半 cos 后半 sin
        last_index: (BS,) int32 或 None; KV-mirror 模式 Q 的源 token 索引
        head_dim, rope_dim: 维度参数
        num_stages: tl.range 软件流水线阶段数
        positions_are_contiguous: Host 已确认 positions 属于单请求、非
            speculative extend 的连续位置序列；仅用于选择专用 kernel
        segment_tile_starts: 多请求普通 prefill/mirror 的 64-token tile
            起点和末尾哨兵；长度相关参数保持运行时值，不参与 JIT 特化

    Returns:
        (query, key) 就地修改后返回
    """
    N = positions.shape[0]
    num_q_heads = query.shape[1]
    num_k_heads = key.shape[1]
    BS = last_index.numel() if last_index is not None else 0
    supports_optimized_head_layout = (
        (num_q_heads == 6 and num_k_heads == 1)
        or (num_q_heads == 12 and num_k_heads == 1)
        or (num_q_heads == 24 and num_k_heads == 2)
    )
    use_contiguous_mirror = (
        last_index is not None
        and supports_optimized_head_layout
        and head_dim == 256
        and rope_dim == 64
        and BS == 1
        and N >= _WELMV4_ROPE_PREFILL_ALL_M_THRESHOLD
        and positions_are_contiguous
        and query.shape[0] == 1
        and key.shape[0] == N
    )
    use_segmented_mirror = (
        last_index is not None
        and supports_optimized_head_layout
        and head_dim == 256
        and rope_dim == 64
        and BS > 1
        and query.shape[0] == BS
        and key.shape[0] == N
        and segment_tile_starts is not None
        and segment_tile_starts.ndim == 1
        and segment_tile_starts.dtype == torch.int32
        and segment_tile_starts.device == positions.device
        and segment_tile_starts.numel() > 1
    )
    supports_blocked_prefill = (
        last_index is None
        and supports_optimized_head_layout
        and head_dim == 256
        and rope_dim == 64
    )
    use_blocked_prefill = supports_blocked_prefill and (
        N >= _WELMV4_ROPE_PREFILL_ALL_M_THRESHOLD
        or (
            N >= _WELMV4_ROPE_PREFILL_EXACT64_THRESHOLD
            and N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK == 0
        )
    )
    use_segmented_prefill = (
        use_blocked_prefill
        and not positions_are_contiguous
        and segment_tile_starts is not None
        and segment_tile_starts.ndim == 1
        and segment_tile_starts.dtype == torch.int32
        and segment_tile_starts.device == positions.device
        and segment_tile_starts.numel() > 1
        and query.shape[0] == N
        and key.shape[0] == N
    )
    use_head_parallel = num_q_heads != 6
    num_q_head_groups = num_q_heads // 2
    num_head_groups = num_k_heads + num_q_head_groups
    if (
        use_contiguous_mirror
        and use_head_parallel
        and N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK == 0
    ):
        num_token_blocks = triton.cdiv(N, _WELMV4_ROPE_PREFILL_TOKEN_BLOCK)
        num_q_tasks = BS * num_q_head_groups
        num_tasks = num_q_tasks + num_token_blocks * num_k_heads
        num_sms = min(num_tasks, _get_num_sms())
        _welmv4_inplace_rope_head_parallel_mirror_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            last_index,
            None,
            num_q_tasks,
            num_tasks,
            N,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            False,
            N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK != 0,
            num_q_heads,
            num_k_heads,
            multibuffer=True,
        )
    elif (
        use_blocked_prefill
        and not use_segmented_prefill
        and use_head_parallel
        and N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK == 0
    ):
        prefill_masked = N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK != 0
        num_token_blocks = triton.cdiv(N, _WELMV4_ROPE_PREFILL_TOKEN_BLOCK)
        num_tasks = num_token_blocks * num_head_groups
        num_sms = min(num_tasks, _get_num_sms())
        position_mode = 1 if positions_are_contiguous else 0
        _welmv4_inplace_rope_head_parallel_prefill_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            None,
            num_tasks,
            N,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            prefill_masked,
            position_mode,
            num_q_heads,
            num_k_heads,
            multibuffer=True,
        )
    elif use_contiguous_mirror:
        num_token_blocks = triton.cdiv(N, _WELMV4_ROPE_PREFILL_TOKEN_BLOCK)
        num_sms = min(num_token_blocks, _get_num_sms())
        _welmv4_inplace_rope_contiguous_mirror_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            last_index,
            num_token_blocks,
            N,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            _WELMV4_ROPE_PREFILL_NUM_STAGES,
            num_q_heads,
            triton.next_power_of_2(num_q_heads),
            num_k_heads,
            multibuffer=True,
        )
    elif use_segmented_mirror:
        num_segment_tiles = segment_tile_starts.numel() - 1
        num_sms = min(
            max(num_segment_tiles, BS),
            _get_num_sms(),
        )
        _welmv4_inplace_rope_segmented_mirror_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            last_index,
            segment_tile_starts,
            num_segment_tiles,
            BS,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            _WELMV4_ROPE_PREFILL_NUM_STAGES,
            num_q_heads,
            triton.next_power_of_2(num_q_heads),
            num_k_heads,
            multibuffer=not use_head_parallel,
        )
    elif use_segmented_prefill:
        num_segment_tiles = segment_tile_starts.numel() - 1
        num_sms = min(
            num_segment_tiles,
            _get_num_sms(multiplier=_WELMV4_ROPE_PROGRAMS_PER_VECTOR_CORE),
        )
        _welmv4_inplace_rope_segmented_prefill_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            segment_tile_starts,
            num_segment_tiles,
            N,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            _WELMV4_ROPE_PREFILL_NUM_STAGES,
            num_q_heads,
            num_k_heads,
            multibuffer=True,
        )
    elif use_blocked_prefill:
        prefill_masked = N % _WELMV4_ROPE_PREFILL_TOKEN_BLOCK != 0
        num_token_blocks = triton.cdiv(N, _WELMV4_ROPE_PREFILL_TOKEN_BLOCK)
        num_sms = min(
            num_token_blocks,
            _get_num_sms(multiplier=_WELMV4_ROPE_PROGRAMS_PER_VECTOR_CORE),
        )
        prefill_kernel = (
            _welmv4_inplace_rope_contiguous_prefill_kernel_npu
            if positions_are_contiguous
            else _welmv4_inplace_rope_prefill_kernel_npu
        )
        prefill_kernel[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            num_token_blocks,
            N,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            _WELMV4_ROPE_PREFILL_TOKEN_BLOCK,
            prefill_masked,
            _WELMV4_ROPE_PREFILL_NUM_STAGES,
            num_q_heads,
            num_k_heads,
            multibuffer=True,
        )
    else:
        num_sms = min(N, _get_num_sms(multiplier=8))
        _welmv4_inplace_rope_kernel_npu[(num_sms,)](
            query,
            key,
            positions,
            cos_sin_cache,
            last_index,
            N,
            BS,
            query.stride(0),
            key.stride(0),
            head_dim,
            rope_dim,
            num_stages,
            num_q_heads,
            num_k_heads,
            triton.next_power_of_2(num_q_heads),
            triton.next_power_of_2(num_k_heads),
        )
    return query, key


# -----------------------------------------------------------------------------
# WeLMv4 over-encoding helpers for Ascend A5.
#
# These kernels intentionally stop at hashed OE ids and token-table updates.
# OE embedding lookup/concat remains on the framework's native PyTorch path.
# -----------------------------------------------------------------------------

_WELMV4_OE_HASH_MULTIPLIER = 2654435761
_WELMV4_OE_BRANCHES = 4
_WELMV4_VECTOR_CORE_CACHE: dict[tuple[str, int], int] = {}


@triton.jit
def _welmv4_u32_remainder(value, divisor: tl.constexpr):
    """Exact uint32 remainder supported efficiently by Triton Ascend."""
    quotient = value // divisor
    return value - quotient * divisor


@triton.jit(
    do_not_specialize=["num_tokens", "num_token_tiles", "num_tasks"]
)
def _welmv4_oe_hash_prefill_4way_kernel(
    input_ids_ptr,
    token_table_ptr,
    req_rows_ptr,
    token_offsets_ptr,
    req_lens_ptr,
    column_starts_ptr,
    hashed_out_ptr,
    num_tokens,
    context_len: tl.constexpr,
    num_token_tiles,
    num_tasks,
    VOCAB_SIZE: tl.constexpr,
    OE_V0: tl.constexpr,
    OE_V1: tl.constexpr,
    OE_V2: tl.constexpr,
    OE_V3: tl.constexpr,
    HASH_MUL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fuse request mapping, history gather, pack, hash and four mods."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_T)

    for task_idx in range(task_start, task_end):
        req_idx = task_idx // num_token_tiles
        tile_idx = task_idx - req_idx * num_token_tiles
        local_offsets = tile_idx * BLOCK_T + base_offsets

        req_len = tl.load(req_lens_ptr + req_idx).to(tl.int32)
        token_start = tl.load(token_offsets_ptr + req_idx).to(tl.int32)
        request_row = tl.load(req_rows_ptr + req_idx).to(tl.int32)
        column_start = tl.load(column_starts_ptr + req_idx).to(tl.int32)

        flat_indices = token_start + local_offsets
        token_mask = (local_offsets.to(tl.float32) < req_len.to(tl.float32)) & (
            flat_indices.to(tl.float32) < num_tokens
        )
        logical_positions = column_start + local_offsets
        current = tl.load(
            input_ids_ptr + flat_indices, mask=token_mask, other=0
        ).to(tl.uint32)

        # Current-chunk history is contiguous in input_ids. Only the first two
        # boundary tokens need an indexed request-token-table read.
        local_offsets_fp32 = local_offsets.to(tl.float32)
        logical_positions_fp32 = logical_positions.to(tl.float32)
        prev1_from_chunk_mask = token_mask & (local_offsets_fp32 >= 1.0)
        prev1_from_table_mask = (
            token_mask & (local_offsets_fp32 < 1.0) & (logical_positions_fp32 >= 1.0)
        )
        prev1_from_chunk = tl.load(
            input_ids_ptr + flat_indices - 1,
            mask=prev1_from_chunk_mask,
            other=0,
        ).to(tl.uint32)
        prev1_from_table = tl.load(
            token_table_ptr
            + request_row * context_len
            + logical_positions
            - 1,
            mask=prev1_from_table_mask,
            other=0,
        ).to(tl.uint32)
        previous1 = prev1_from_chunk + prev1_from_table

        prev2_from_chunk_mask = token_mask & (local_offsets_fp32 >= 2.0)
        prev2_from_table_mask = (
            token_mask & (local_offsets_fp32 < 2.0) & (logical_positions_fp32 >= 2.0)
        )
        prev2_from_chunk = tl.load(
            input_ids_ptr + flat_indices - 2,
            mask=prev2_from_chunk_mask,
            other=0,
        ).to(tl.uint32)
        prev2_from_table = tl.load(
            token_table_ptr
            + request_row * context_len
            + logical_positions
            - 2,
            mask=prev2_from_table_mask,
            other=0,
        ).to(tl.uint32)
        previous2 = prev2_from_chunk + prev2_from_table

        packed2 = current + previous1 * VOCAB_SIZE
        packed3 = packed2 + previous2 * VOCAB_SIZE * VOCAB_SIZE
        hash2 = (packed2 * HASH_MUL).to(tl.uint32)
        hash3 = (packed3 * HASH_MUL).to(tl.uint32)

        tl.store(
            hashed_out_ptr + flat_indices,
            _welmv4_u32_remainder(hash2, OE_V0).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + num_tokens + flat_indices,
            _welmv4_u32_remainder(hash2, OE_V1).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + 2 * num_tokens + flat_indices,
            _welmv4_u32_remainder(hash3, OE_V2).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + 3 * num_tokens + flat_indices,
            _welmv4_u32_remainder(hash3, OE_V3).to(tl.int32),
            mask=token_mask,
        )


@triton.jit(do_not_specialize=["batch_size", "num_tasks"])
def _welmv4_oe_hash_decode_4way_kernel(
    input_ids_ptr,
    token_table_ptr,
    req_rows_ptr,
    column_starts_ptr,
    hashed_out_ptr,
    batch_size,
    context_len: tl.constexpr,
    num_tasks,
    VOCAB_SIZE: tl.constexpr,
    OE_V0: tl.constexpr,
    OE_V1: tl.constexpr,
    OE_V2: tl.constexpr,
    OE_V3: tl.constexpr,
    HASH_MUL: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Decode specialization: one current token per real request."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_B)

    for task_idx in range(task_start, task_end):
        req_indices = task_idx * BLOCK_B + base_offsets
        request_mask = req_indices.to(tl.float32) < batch_size
        request_rows = tl.load(
            req_rows_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.int32)
        positions = tl.load(
            column_starts_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.int32)
        current = tl.load(
            input_ids_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.uint32)
        previous1 = tl.load(
            token_table_ptr + request_rows * context_len + positions - 1,
            mask=request_mask & (positions.to(tl.float32) >= 1.0),
            other=0,
        ).to(tl.uint32)
        previous2 = tl.load(
            token_table_ptr + request_rows * context_len + positions - 2,
            mask=request_mask & (positions.to(tl.float32) >= 2.0),
            other=0,
        ).to(tl.uint32)

        packed2 = current + previous1 * VOCAB_SIZE
        packed3 = packed2 + previous2 * VOCAB_SIZE * VOCAB_SIZE
        hash2 = (packed2 * HASH_MUL).to(tl.uint32)
        hash3 = (packed3 * HASH_MUL).to(tl.uint32)
        tl.store(
            hashed_out_ptr + req_indices,
            _welmv4_u32_remainder(hash2, OE_V0).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + batch_size + req_indices,
            _welmv4_u32_remainder(hash2, OE_V1).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + 2 * batch_size + req_indices,
            _welmv4_u32_remainder(hash3, OE_V2).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + 3 * batch_size + req_indices,
            _welmv4_u32_remainder(hash3, OE_V3).to(tl.int32),
            mask=request_mask,
        )


@triton.jit(do_not_specialize=["batch_size", "num_tasks"])
def _welmv4_oe_hash_explicit_history_4way_kernel(
    current_ptr,
    previous1_ptr,
    previous2_ptr,
    hashed_out_ptr,
    batch_size,
    num_tasks,
    VOCAB_SIZE: tl.constexpr,
    OE_V0: tl.constexpr,
    OE_V1: tl.constexpr,
    OE_V2: tl.constexpr,
    OE_V3: tl.constexpr,
    HASH_MUL: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Hash one draft token per request using draft-local history."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_B)

    for task_idx in range(task_start, task_end):
        indices = task_idx * BLOCK_B + base_offsets
        mask = indices.to(tl.float32) < batch_size
        current = tl.load(current_ptr + indices, mask=mask, other=0).to(tl.uint32)
        previous1 = tl.load(previous1_ptr + indices, mask=mask, other=0).to(
            tl.uint32
        )
        previous2 = tl.load(previous2_ptr + indices, mask=mask, other=0).to(
            tl.uint32
        )
        packed2 = current + previous1 * VOCAB_SIZE
        packed3 = packed2 + previous2 * VOCAB_SIZE * VOCAB_SIZE
        hash2 = (packed2 * HASH_MUL).to(tl.uint32)
        hash3 = (packed3 * HASH_MUL).to(tl.uint32)
        tl.store(
            hashed_out_ptr + indices,
            _welmv4_u32_remainder(hash2, OE_V0).to(tl.int32),
            mask=mask,
        )
        tl.store(
            hashed_out_ptr + batch_size + indices,
            _welmv4_u32_remainder(hash2, OE_V1).to(tl.int32),
            mask=mask,
        )
        tl.store(
            hashed_out_ptr + 2 * batch_size + indices,
            _welmv4_u32_remainder(hash3, OE_V2).to(tl.int32),
            mask=mask,
        )
        tl.store(
            hashed_out_ptr + 3 * batch_size + indices,
            _welmv4_u32_remainder(hash3, OE_V3).to(tl.int32),
            mask=mask,
        )


@triton.jit(
    do_not_specialize=["num_tokens", "num_token_tiles", "num_tasks"]
)
def _welmv4_token_table_ragged_update_kernel(
    token_table_ptr,
    tokens_ptr,
    row_indices_ptr,
    token_offsets_ptr,
    column_starts_ptr,
    req_lens_ptr,
    num_tokens,
    context_len: tl.constexpr,
    num_token_tiles,
    num_tasks,
    BLOCK_T: tl.constexpr,
):
    """Copy request-segmented tokens without materializing flat row/col ids."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_T)

    for task_idx in range(task_start, task_end):
        req_idx = task_idx // num_token_tiles
        tile_idx = task_idx - req_idx * num_token_tiles
        local_offsets = tile_idx * BLOCK_T + base_offsets
        req_len = tl.load(req_lens_ptr + req_idx).to(tl.int32)
        token_start = tl.load(token_offsets_ptr + req_idx).to(tl.int32)
        row = tl.load(row_indices_ptr + req_idx).to(tl.int32)
        column_start = tl.load(column_starts_ptr + req_idx).to(tl.int32)
        token_indices = token_start + local_offsets
        columns = column_start + local_offsets
        mask = (
            (local_offsets.to(tl.float32) < req_len.to(tl.float32))
            & (token_indices.to(tl.float32) < num_tokens)
            & (columns.to(tl.float32) < context_len)
        )
        values = tl.load(tokens_ptr + token_indices, mask=mask, other=0)
        tl.store(token_table_ptr + row * context_len + columns, values, mask=mask)


@triton.jit(do_not_specialize=["batch_size", "num_tasks"])
def _welmv4_token_table_decode_update_kernel(
    token_table_ptr,
    next_token_ids_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    skip_mask_ptr,
    batch_size,
    context_len: tl.constexpr,
    num_tasks,
    HAS_SKIP_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Write one sampled token per real request, honoring chunk skip state."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_B)

    for task_idx in range(task_start, task_end):
        req_indices = task_idx * BLOCK_B + base_offsets
        real_mask = req_indices.to(tl.float32) < batch_size
        if HAS_SKIP_MASK:
            skip = tl.load(
                skip_mask_ptr + req_indices, mask=real_mask, other=1
            ).to(tl.int1)
        else:
            skip = tl.zeros((BLOCK_B,), dtype=tl.int1)
        rows = tl.load(
            req_pool_indices_ptr + req_indices, mask=real_mask, other=0
        ).to(tl.int32)
        columns = tl.load(
            seq_lens_ptr + req_indices, mask=real_mask, other=0
        ).to(tl.int32)
        values = tl.load(next_token_ids_ptr + req_indices, mask=real_mask, other=0)
        update_mask = (
            real_mask & (~skip) & (columns.to(tl.float32) < context_len)
        )
        tl.store(
            token_table_ptr + rows * context_len + columns,
            values,
            mask=update_mask,
        )


@triton.jit(do_not_specialize=["batch_size", "width"])
def _welmv4_token_table_spec_accept_update_kernel(
    token_table_ptr,
    predict_ptr,
    accept_index_ptr,
    accept_lens_ptr,
    req_pool_indices_ptr,
    old_seq_lens_ptr,
    batch_size,
    width,
    context_len: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Commit accepted predictions after each request's incoming root."""
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_W)
    request_mask = req_idx < batch_size
    accept_len = tl.load(
        accept_lens_ptr + req_idx, mask=request_mask, other=0
    ).to(tl.int32)
    valid = request_mask & (offsets < width) & (offsets < accept_len)
    flat_index = req_idx * width + offsets
    source = tl.load(
        accept_index_ptr + flat_index, mask=valid, other=0
    ).to(tl.int64)
    source = tl.maximum(source, 0)
    values = tl.load(predict_ptr + source, mask=valid, other=0)
    row = tl.load(
        req_pool_indices_ptr + req_idx, mask=request_mask, other=0
    ).to(tl.int64)
    old_len = tl.load(
        old_seq_lens_ptr + req_idx, mask=request_mask, other=0
    ).to(tl.int64)
    columns = old_len + 1 + offsets
    valid = valid & (columns < context_len)
    tl.store(
        token_table_ptr + row * context_len + columns,
        values,
        mask=valid,
    )


def _welmv4_vector_core_count(device: torch.device) -> int:
    index = torch.npu.current_device() if device.index is None else int(device.index)
    key = (device.type, index)
    cached = _WELMV4_VECTOR_CORE_CACHE.get(key)
    if cached is not None:
        return cached
    properties = triton.runtime.driver.active.utils.get_device_properties(index)
    count = int(properties.get("num_vectorcore", properties.get("num_aicore", -1)))
    if count <= 0:
        raise RuntimeError("Failed to detect the Ascend Vector Core count")
    _WELMV4_VECTOR_CORE_CACHE[key] = count
    return count


def _welmv4_1d_grid(num_tasks: int, device: torch.device) -> tuple[int]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    return (min(num_tasks, _welmv4_vector_core_count(device)),)


def _validate_welmv4_oe_vocab_sizes(
    oe_vocab_sizes: Sequence[int],
) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in oe_vocab_sizes)
    if len(values) != _WELMV4_OE_BRANCHES or any(v <= 0 for v in values):
        raise ValueError("oe_vocab_sizes must contain four positive integers")
    if any(v > (1 << 31) for v in values):
        raise ValueError("int32 OE hash output requires every vocab size <= 2^31")
    return values


def welmv4_oe_hash_prefill_4way_npu(
    input_ids: torch.Tensor,
    token_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    req_lens: torch.Tensor,
    column_starts: torch.Tensor,
    *,
    max_req_len: int,
    vocab_size: int,
    oe_vocab_sizes: Sequence[int],
    block_t: int = 512,
) -> torch.Tensor:
    """Return int32 hashed ids with layout [4, num_tokens] for prefill."""
    oe_v0, oe_v1, oe_v2, oe_v3 = _validate_welmv4_oe_vocab_sizes(
        oe_vocab_sizes
    )
    num_tokens = input_ids.numel()
    output = torch.empty(
        (_WELMV4_OE_BRANCHES, num_tokens),
        dtype=torch.int32,
        device=input_ids.device,
    )
    if num_tokens == 0:
        return output
    num_token_tiles = triton.cdiv(max_req_len, block_t)
    num_tasks = req_lens.numel() * num_token_tiles
    _welmv4_oe_hash_prefill_4way_kernel[
        _welmv4_1d_grid(num_tasks, input_ids.device)
    ](
        input_ids,
        token_table,
        req_pool_indices,
        token_offsets,
        req_lens,
        column_starts,
        output,
        num_tokens,
        token_table.shape[1],
        num_token_tiles,
        num_tasks,
        VOCAB_SIZE=int(vocab_size),
        OE_V0=oe_v0,
        OE_V1=oe_v1,
        OE_V2=oe_v2,
        OE_V3=oe_v3,
        HASH_MUL=_WELMV4_OE_HASH_MULTIPLIER,
        BLOCK_T=block_t,
    )
    return output


def welmv4_oe_hash_decode_4way_npu(
    input_ids: torch.Tensor,
    token_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    column_starts: torch.Tensor,
    *,
    vocab_size: int,
    oe_vocab_sizes: Sequence[int],
    block_b: int = 128,
) -> torch.Tensor:
    """Return int32 hashed ids with layout [4, batch_size] for decode."""
    oe_v0, oe_v1, oe_v2, oe_v3 = _validate_welmv4_oe_vocab_sizes(
        oe_vocab_sizes
    )
    batch_size = input_ids.numel()
    if (
        req_pool_indices.numel() < batch_size
        or column_starts.numel() < batch_size
    ):
        raise ValueError("decode metadata must contain at least batch_size entries")
    output = torch.empty(
        (_WELMV4_OE_BRANCHES, batch_size),
        dtype=torch.int32,
        device=input_ids.device,
    )
    if batch_size == 0:
        return output
    num_tasks = triton.cdiv(batch_size, block_b)
    _welmv4_oe_hash_decode_4way_kernel[
        _welmv4_1d_grid(num_tasks, input_ids.device)
    ](
        input_ids,
        token_table,
        req_pool_indices,
        column_starts,
        output,
        batch_size,
        token_table.shape[1],
        num_tasks,
        VOCAB_SIZE=int(vocab_size),
        OE_V0=oe_v0,
        OE_V1=oe_v1,
        OE_V2=oe_v2,
        OE_V3=oe_v3,
        HASH_MUL=_WELMV4_OE_HASH_MULTIPLIER,
        BLOCK_B=block_b,
    )
    return output


def welmv4_oe_hash_explicit_history_4way_npu(
    current: torch.Tensor,
    previous1: torch.Tensor,
    previous2: torch.Tensor,
    *,
    vocab_size: int,
    oe_vocab_sizes: Sequence[int],
    block_b: int = 128,
) -> torch.Tensor:
    """Return [4, B] hashes for recurrent draft-local token history."""
    oe_v0, oe_v1, oe_v2, oe_v3 = _validate_welmv4_oe_vocab_sizes(
        oe_vocab_sizes
    )
    batch_size = current.numel()
    if previous1.numel() != batch_size or previous2.numel() != batch_size:
        raise ValueError("draft-local OE history must have one row per token")
    output = torch.empty(
        (_WELMV4_OE_BRANCHES, batch_size),
        dtype=torch.int32,
        device=current.device,
    )
    if batch_size == 0:
        return output
    num_tasks = triton.cdiv(batch_size, block_b)
    _welmv4_oe_hash_explicit_history_4way_kernel[
        _welmv4_1d_grid(num_tasks, current.device)
    ](
        current,
        previous1,
        previous2,
        output,
        batch_size,
        num_tasks,
        VOCAB_SIZE=int(vocab_size),
        OE_V0=oe_v0,
        OE_V1=oe_v1,
        OE_V2=oe_v2,
        OE_V3=oe_v3,
        HASH_MUL=_WELMV4_OE_HASH_MULTIPLIER,
        BLOCK_B=block_b,
    )
    return output


def welmv4_token_table_ragged_update_npu(
    token_table: torch.Tensor,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    *,
    max_req_len: int,
    block_t: int = 256,
) -> None:
    """Update prefill request segments directly in the request token table."""
    num_tokens = tokens.numel()
    if num_tokens == 0:
        return
    num_token_tiles = triton.cdiv(max_req_len, block_t)
    num_tasks = req_lens.numel() * num_token_tiles
    _welmv4_token_table_ragged_update_kernel[
        _welmv4_1d_grid(num_tasks, tokens.device)
    ](
        token_table,
        tokens,
        row_indices,
        token_offsets,
        column_starts,
        req_lens,
        num_tokens,
        token_table.shape[1],
        num_token_tiles,
        num_tasks,
        BLOCK_T=block_t,
    )


def welmv4_token_table_decode_update_npu(
    token_table: torch.Tensor,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    skip_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    block_b: int = 16,
) -> None:
    """Update one sampled token per real request; ignore graph padding rows."""
    if batch_size == 0:
        return
    num_tasks = triton.cdiv(batch_size, block_b)
    # A compile-time flag removes the skip load for normal decode. Triton still
    # needs a pointer argument, so next_token_ids is an unused safe placeholder.
    skip_mask_ptr = next_token_ids if skip_mask is None else skip_mask
    _welmv4_token_table_decode_update_kernel[
        _welmv4_1d_grid(num_tasks, next_token_ids.device)
    ](
        token_table,
        next_token_ids,
        req_pool_indices,
        seq_lens,
        skip_mask_ptr,
        batch_size,
        token_table.shape[1],
        num_tasks,
        HAS_SKIP_MASK=skip_mask is not None,
        BLOCK_B=block_b,
    )


def welmv4_token_table_spec_accept_update_npu(
    token_table: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    old_seq_lens: torch.Tensor,
) -> None:
    """Commit Spec V2 topk1 accepts at old_seq_lens + 1 + j."""
    batch_size, width = accept_index.shape
    if batch_size == 0 or width == 0:
        return
    block_w = triton.next_power_of_2(width)
    _welmv4_token_table_spec_accept_update_kernel[(batch_size,)](
        token_table,
        predict,
        accept_index,
        accept_lens,
        req_pool_indices,
        old_seq_lens,
        batch_size,
        width,
        token_table.shape[1],
        BLOCK_W=block_w,
    )

def sigmoid_mul_ref(
    x: torch.Tensor,
    y: torch.Tensor,
):
    return torch.mul(torch.sigmoid(x.to(torch.float32)).to(y.dtype), y)


@triton.jit(do_not_specialize=["rows"])
def sigmoid_mul_kernel_npu(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    x_row_stride: tl.constexpr,
    y_row_stride: tl.constexpr,
    y_col_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offset < rows
    col_offset = tl.arange(0, cols)

    x_off = row_offset[:, None] * x_row_stride
    y_off = row_offset[:, None] * y_row_stride + col_offset[None, :] * y_col_stride

    x_data = tl.load(x_ptr + x_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
    y_data = tl.load(y_ptr + y_off, mask=row_mask[:, None], other=0.0)

    out_data = tl.sigmoid(x_data).to(y_ptr.dtype.element_ty) * y_data
    tl.store(y_ptr + y_off, out_data, mask=row_mask[:, None])


def inplace_sigmoid_mul_npu(
    x: torch.Tensor,
    y: torch.Tensor,
    BLOCK_SIZE: int = 128,
) -> None:
    cols = y.shape[-1]
    rows = y.numel() // cols

    assert x.is_contiguous()
    assert y.is_contiguous()
    assert x.shape[-1] == 1 and y.shape[-1] == 256

    x_flattened = x.view(rows, -1)
    y_flattened = y.view(rows, -1)

    grid = (triton.cdiv(rows, BLOCK_SIZE),)
    sigmoid_mul_kernel_npu[grid](
        x_flattened,
        y_flattened,
        rows,
        cols,
        x_flattened.stride(0),
        y_flattened.stride(0),
        y_flattened.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
