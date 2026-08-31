# WeLMv4 NPU DP-Attention optimization workspace

This repository is the standalone remote-NPU loop for the WeLMv4 Full/SWA
attention kernels used by NEWSGLANG.

## Production split

- Prefill and Spec V2 target verify:
  `welmv4_sink_prefill_attention.py`, frozen as
  `welmv4_dp_prefill_attention_baseline.py`.
- Eager decode and decode-like KV-mirror prefill:
  `sink_full_attention.py`, frozen as
  `welmv4_dp_decode_attention_baseline.py`.
- Optimization changes belong only in the matching `*_candidate.py` file.

The initial baseline and candidate files are byte-identical. The benchmark
audits both frozen baseline hashes before touching the NPU.

## Head layouts

The global model has Q24/KV2 with TP=4:

| layout | Attention DP | local Q heads | local KV heads | role |
|---|---:|---:|---:|---|
| `tp4` | 1 | 6 | 1 | non-regression control and Q6-only dispatch |
| `tp4_dp2` | 2 | 12 | 1 | DP2 target |
| `tp4_dp4` | 4 | 24 | 2 | DP4 target |

The workspace explicitly exercises the production Q6-only grouped prefill
kernels and the generic Q12/Q24 branches whose DP-Attention performance is not
yet known.

## Shape contract

- Decode batch sizes: `1, 4, 8, 16, 32, 40, 56`.
- Spec V2 steps 1-3: target-verify widths `D=2,3,4`.
- Ordinary dense and ragged prefill.
- Decode-like KV-mirror calls, including the production max-context dispatch
  hint.
- Page size 64, head dimension 256.
- Context/prefill coverage: 4096, 8192, 9616, the 11K band
  (10880/11264/11648), and the 16.5K band (16000/16896/17408).
- The removed 26K cases are not part of this task.

Long prefill correctness uses full baseline/candidate comparison plus a
deterministic FP32 sample of query rows. This keeps correctness independent
without turning the oracle into an O(16K²) remote bottleneck.

## Performance gates

`msprof op` task duration is the only timing authority. NPU Event timing is
disabled.

- Every changed candidate case must be no slower than its frozen baseline.
- TP4/Q6 allows only the configured 2% measurement-noise band.
- Candidate time is normalized by the Q-head work scale. DP2 must be no worse
  than 2x raw TP4 latency and DP4 no worse than 4x, with a 2% noise allowance.
- Full flash-decode latency is the sum of
  `paged_decode_fd_kernel` and `paged_decode_fd_reduce_kernel`, not just the
  first kernel.

Before the long matrix, four small cases run an msprof preflight. A bad command,
wrong kernel name, unexpected launch count, or unparseable CSV stops the run
immediately.

## Iteration versus full validation

`--suite remote` normally maps to the bounded `iteration` suite. It retains
all required decode/MTP correctness shapes but times only representative
workloads. Set:

```bash
export WELMV4_DP_ATTENTION_FORCE_FULL=1
```

for initial, periodic, or final full-matrix validation. The monitor performs a
full run on its first benchmark and then every three iteration runs by default;
change this with `--full-every N`.

## Remote worker

```bash
python auto_bench_on_git_update.py --run-now --device npu:5
```

The worker publishes:

- `welmv4_dp_attention_results/result.json`
- correctness and performance CSV files
- per-component msprof task durations and compressed logs
- IR/MLIR captures
- profiler and pipeline artifacts
- `welmv4_dp_attention_run_error.log` on failure

The old RoPE and prefill-attention result trees remain historical artifacts;
the monitor no longer writes them.
