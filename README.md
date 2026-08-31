# WeLMv4 NPU RoPE DP-Attention optimization workspace

This repository is a standalone Ascend worker for preserving the current
WeLMv4 RoPE performance while generalizing its optimized kernels from the
original TP4 local layout to DP-Attention layouts. The remote worker does not
need a NEWSGLANG checkout.

## Head-layout contract

The model has 24 global Q heads and 2 global KV heads:

| topology | attention TP | local Q | local KV |
|---|---:|---:|---:|
| `tp4` | 4 | 6 | 1 |
| `tp4_dp2` | 2 | 12 | 1 |
| `tp4_dp4` | 1 | 24 | 2 |

All cases use BF16 Q/K, FP32 cos/sin cache, `head_dim=256` and tail
`rope_dim=64`. The candidate must retain the Q6/K1 latency while making Q12/K1
and Q24/K2 scale with their real arithmetic and memory traffic instead of
falling back to padded Q16/Q32 execution.

`welmv4_rope_baseline.py` is a frozen production copy of
`python/sglang/srt/layers/welmv4_npu_op.py`; its only standalone adaptation is
the fallback `is_npu()` import. Its LF-normalized hash is pinned in
`rope_workspace_config.json`. Optimization changes belong in
`welmv4_rope_candidate.py`.

The previous Full/SWA Attention files and result history remain for
traceability. The active monitor now invokes `bench_welmv4_rope_npu.py` and
publishes `welmv4_rope_results/`.

## Covered execution families

- Decode with local M=`1/4/8/16/32/40/56`.
- MTP step1/2/3, represented by per-request widths D=`2/3/4`, across the same
  batch values.
- Contiguous and segmented ordinary prefill.
- Single-request and segmented KV-mirror prefill.
- Dispatch boundaries 575/576/577 and 639/640/641.

MTP/decode positions cover 4096, 8192 and 9616, plus the lower/central/upper
points around 11K, 16.5K and 26K:

```text
10240 / 11000 / 12288
15360 / 16500 / 18432
24576 / 26000 / 28672
```

RoPE does not read KV cache, but these context values materially exercise the
cos/sin-cache addressing pattern. Correctness uses a wider cross-product;
performance rotates representative `(M,D,context)` points instead of compiling
every possible Cartesian combination.

## Acceptance rules

1. Baseline and candidate are checked against an FP32 torch reference.
2. Candidate BF16 output must be bitwise-equal to the frozen baseline.
3. Every timed Q6, Q12 and Q24 case must have candidate/baseline speedup >=1.0.
4. Candidate microseconds per rotated value for DP2/DP4 may not exceed the
   matching TP4 workload by more than the configured 5% noise tolerance.
5. NPU Event timing is disabled and rejected. Acceptance uses ordered
   `msprof op` task durations.

The normalized rule does not require equal single-request latency when a rank
owns two or four times as many Q heads. It requires the latency increase to
match the actual rotated-value count, without extra loss from padding or a
generic kernel.

## Initial baseline snapshot

`welmv4_rope_candidate.py` initially matches the frozen baseline byte for byte.
The first remote cycle therefore records the current implementation, including
the existing generic fallback for DP-Attention layouts, without pretending that
an unverified optimization is ready.

Because baseline and candidate hashes are equal, the first run marks timing rows
as `BASELINE_SNAPSHOT`: timing noise cannot falsely fail the cycle. Normalized
Q6/Q12/Q24 cost ratios are still emitted so padded/fallback inefficiencies are
visible. Once the candidate changes, the strict no-regression and normalized
efficiency gates automatically become active.

After the baseline snapshot, a changed candidate automatically maps the legacy
monitor's `--suite remote` request to the bounded `iteration` suite. This keeps
each one-point experiment fast while retaining Q6 guards, DP2/DP4 boundaries,
long prefill and mirror cases. Set `WELMV4_ROPE_FORCE_FULL=1` for periodic or
final full-matrix validation.

The unchanged TP4 Q6/K1 control layout uses a 2% msprof noise band established
by the byte-identical baseline snapshot. Modified DP2/DP4 layouts retain the
strict `speedup>=1.0` gate. Decode and MTP remain in every iteration's full
correctness matrix, but enter performance timing only when their generic kernel
is the active optimization target.

Optimization must be driven by returned msprof/profile/MLIR evidence, and all
Q6 regression gates must remain green.

## Remote worker

Stop the previous worker once, pull this switch, and run:

```bash
python auto_bench_on_git_update.py --run-now --device npu:5
```

After that, no manual intervention is required. The existing protocol still
fast-forwards from origin, stages a run safely, retains PASS/failure/regression
artifacts, handles push races and commits only active generated artifacts.

Manual smoke run:

```bash
python bench_welmv4_rope_npu.py \
  --suite smoke --mode both --device npu:5 \
  --capture-msprof-op on --capture-ir on --capture-profile off
```

`BENCH_PYTHON=/path/to/python` selects the NPU interpreter.

## Generated artifacts

```text
welmv4_rope_results/
  result.json
  correctness.csv
  performance_shape_validation.csv
  performance.csv
  msprof_task_duration.csv
  ir.csv
  profile.csv
  ir/<topology>/<case>/*.mlir.gz
  profile/<topology>/<case>/{pipe_utilization,memory}/*.{csv,json}.gz
  msprof/{baseline,candidate}/*.log.gz
welmv4_rope_run_error.log
```

`result.json` is authoritative. Status is `PASS`, `FAIL`,
`PERF_REGRESSION`, or `ERROR`.
