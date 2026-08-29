# WeLMv4 NPU Full/SWA Prefill Attention optimization workspace

This repository is a self-contained Ascend worker for optimizing the WeLMv4
NPU paged Prefill Attention used by normal Prefill and Spec-V2 MTP verify. It
does **not** import a NEWSGLANG checkout on the remote machine.

The frozen baseline and initial candidate are exact copies of:

```text
NEWSGLANG/sglang @ 58a1448cc30c5a21feebaee6980f5b3612ed914e
python/sglang/srt/hardware_backend/npu/attention/sink_full_attention.py
```

`workspace_config.json` pins the LF-normalized baseline SHA-256. A run fails
immediately if the frozen baseline is edited accidentally. Optimization commits
should edit only `welmv4_prefill_attention_candidate.py` plus an intentional
case/worker change when needed.

## Production contract represented here

- BF16 Q/K/V and KV cache; FP32 learned sink.
- Page size 64 and head dimension 256.
- Global heads 24/2. The default TP4 rank has local Hq=6/Hkv=1; TP1/2/8 can
  also be selected with `--tp-size`.
- Native KV allocation is `[P*64,Hkv,256]`; the kernels receive the same
  zero-copy `view(P,64,Hkv,256).permute(0,2,1,3)` view as NEWSGLANG.
- Contiguous GQA grouping (`gqa_interleave=false`).
- YaRN-adjusted softmax scale `0.09119556747428784`.
- Full Attention plus a zero-value learned sink.
- SWA Attention plus the same sink, left history 511 (visible span 512), and
  global window 0.
- `seq_lens_kv` is the post-write length: K/V for the current Query window is
  already in paged cache.

At page64/head256 the current Full implementation executes
`paged_prefill_page_aggregation_kernel` with BM/BN/BD=128/64/256 and page
aggregation 1. SWA executes `_swa_paged_prefill_aggregation_sink_kernel` with
the same blocks and page aggregation 2.

## M and topology

`M` always means total real Query rows, `sum(q_lens)`. It is not sufficient by
itself to describe performance, so every result also records the complete
`q_lens`, KV lengths, batch size, table stride, topology, and graph bucket.

- `dense`: one request with `q_len=M`; every integer M from 1 through 1024 is
  legal.
- `ragged_prefill`: multiple requests with positive, non-uniform `q_lens` and
  exact `sum(q_lens)=M`, covering the ordinary batched Prefill scheduler/LPT
  contract.
- `verify_d2`: step=1, fixed verify width D=2, B=M/2 requests.
- `verify_d3`: step=2, fixed verify width D=3, B=M/3 requests.

The automatic optimization gate measures the Attention operator itself. It does
not capture or replay NPU Graph; Graph integration remains in NEWSGLANG's model
test and does not affect selection of the Triton kernel candidate here.

SWA cases map evicted old logical pages to an allocated poison page (never
`-1`) and retain shuffled physical pages only for the union of tokens visible
to the Query window. An erroneous old-page read therefore becomes a numerical
failure instead of an asynchronous device OOB crash.

The remote matrix treats context length as a band, not three exact points. It
covers roughly 10–12K, 15–18K, and 24–28K tokens, including the observed 11K,
16.5K, and 26K regions. Ragged per-request lengths also create non-page-aligned
tails. Exact lengths remain selectable with `--kv-lengths`.

## Remote worker

Stop the old RoPE monitor once, pull this revision, then start the new worker
from the NPU Python environment:

```bash
python auto_bench_on_git_update.py --run-now --device npu:5
```

Useful worker flags:

```bash
# Run one polling cycle, publish the generated result, and exit.
python auto_bench_on_git_update.py --once --run-now --device npu:5

# Change the polling interval.
python auto_bench_on_git_update.py --interval 60 --device npu:5
```

`BENCH_PYTHON=/path/to/python` selects the NPU interpreter. The worker:

1. fetches and fast-forwards the current branch;
2. runs the remote Full/SWA correctness, performance, IR, profiler and msprof
   suites without any Graph phase;
3. writes `welmv4_prefill_attention_results/` on success, or a captured error
   log on failure;
4. commits only those generated artifacts and pushes them to the same branch.

Push races are detected before publication. A result produced from an obsolete
HEAD is discarded and rerun on the newest code. Old successful results are
never deleted by a later failed run.

## Automatic result layout

```text
welmv4_prefill_attention_results/
  result.json
  correctness.csv
  performance.csv
  msprof_primary_kernel.csv
  ir.csv
  profile.csv
  ir/full/.../*.mlir.gz
  ir/swa/.../*.mlir.gz
  profile/full/.../{pipe_utilization,memory}/*.csv.gz or *.json.gz
  profile/swa/.../{pipe_utilization,memory}/*.csv.gz or *.json.gz
  msprof/.../*.log.gz
welmv4_prefill_attention_run_error.log  # only when the run fails
```

`result.json` is the machine-readable authority. Status is one of `PASS`,
`FAIL`, `PERF_REGRESSION`, or `ERROR`, with per-phase state and
source/environment hashes.

Correctness is checked in FP32 against causal paged GQA with the virtual
zero-value sink. Candidate is also compared directly with the frozen baseline.
Logical wrapper latency is measured with grouped NPU events and is the
acceptance metric, so all device kernels introduced by a future split are
included. Every timed shape passes its own reference/baseline correctness gate
first, including `--mode performance`. Selected-name `msprof op` duration is a
non-authoritative diagnostic for the current primary Triton kernel; a legal
split/rename is recorded rather than rejected. Selected candidate calls export
PipeUtilization and Memory/L2 profiler pipelines plus TTIR, TTAdapter, and
last-pass MLIR as separate gzip artifacts.

## Manual use

Smoke test:

```bash
python bench_welmv4_prefill_attention_npu.py \
  --suite smoke --mode both --device npu:0 \
  --output-dir manual_results/smoke
```

Any dense M from 1 to 1024:

```bash
python bench_welmv4_prefill_attention_npu.py \
  --mode correctness --device npu:0 \
  --attention full,swa --topology dense,ragged_prefill \
  --m-values 1:1024 --kv-lengths 4096 \
  --prefill-batch-size 8 \
  --output-dir manual_results/dense_all_m
```

Actual step1/step2 verify shapes (invalid non-divisible M values are skipped):

```bash
python bench_welmv4_prefill_attention_npu.py \
  --mode both --device npu:0 \
  --attention full,swa --topology verify_d2,verify_d3 \
  --m-values 1:1024 --kv-lengths 511,512,513,4096 \
  --length-pattern ragged \
  --output-dir manual_results/verify_all_m
```

The three production context bands can be requested explicitly:

```bash
python bench_welmv4_prefill_attention_npu.py \
  --mode both --device npu:0 --attention full,swa \
  --topology verify_d2 --m-values 2,32,112 \
  --kv-lengths 10240,11264,12288,15360,16896,18432,24576,26624,28672 \
  --length-pattern ragged --output-dir manual_results/context_bands
```

Run one configured case repeatedly for `msprof`, debugger, or compiler work:

```bash
python run_welmv4_prefill_attention_case_npu.py \
  --case-name full_verify_d2_m112_kv4096_uniform_compact_tp4 \
  --provider candidate --iterations 20 --device npu:0
```

## Optimization discipline

- Keep `welmv4_prefill_attention_baseline.py` frozen.
- Make one optimization change at a time in the candidate.
- Preserve all production semantics; do not specialize on runtime M, sequence
  lengths, page IDs, or block-table contents.
- Bounded static variants such as D=2/D=3 or a finite BM set are allowed, but
  must be visible in source and reflected by IR/profile artifacts.
- Correctness must pass before performance is accepted.
- A candidate below the configured minimum speedup is published as
  `PERF_REGRESSION`, not silently promoted.

Two initial profiling targets are intentionally recorded in the sketches:

- Full verify uses only 2/3 valid Query rows inside BM=128.
- SWA additionally sizes its grid from `ceil(total_M/128)*Hq`, although its
  actual task count is `sum_i ceil(q_len_i/128)*Hq`. For B=56,D=2 this is 6
  programs serializing 336 request/head tasks.

These are hypotheses/observations to validate with the returned profiler and
MLIR evidence, not pre-applied optimizations.
