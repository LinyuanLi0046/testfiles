# WeLMv4 inplace RoPE NPU optimization workspace

This repository is a self-contained remote benchmark loop for
`welmv4_inplace_rope_npu` on Ascend A5.  It does not import the NEWSGLANG
checkout.

## Remote worker

Start or restart the monitor from the NPU Python environment in the repository
root.  `--run-now` benchmarks the current synchronized HEAD immediately and
then leaves the normal one-minute monitor running:

```bash
python auto_bench_on_git_update.py --run-now --device npu:5
```

An already-running copy of the old TopK monitor must be stopped and restarted;
a Python process does not reload its source after `git pull` rewrites the file.

Every 60 seconds it fetches `origin/<current-branch>`.  When the remote branch
has a newer commit, it fast-forwards, runs:

```bash
python bench_welmv4_inplace_rope_npu.py \
  --mode both \
  --cases all \
  --scope kernel \
  --device npu:5 \
  --output-csv welmv4_inplace_rope_npu_all.csv
```

During the benchmark it does not fetch or pull.  On success it removes any old
`welmv4_inplace_rope_npu_run_error.log`, commits the CSV, and pushes.  On
failure it removes any stale CSV, writes the combined stdout/stderr to that
error log, commits it, and pushes.  Automatic result commits contain
`Auto-Benchmark: true`, allowing the monitor to recover safely after a restart
or push failure.

The standard all-case run also recompiles only the candidate for a selected
representative case with Triton debug dumping enabled.  It lowers the adapter IR
with `bishengir-compile` for the detected A5 target and stores gzip+base64
TTIR, TTAdapter, and last-pass MLIR as `record_type=ir_artifact` rows in the
same CSV.  IR capture errors are diagnostic CSV rows and do not discard valid
correctness/performance measurements.  Use `--capture-ir off` for a manual
run that should skip this diagnostic step.

An explicit `--capture-profile on` run profiles the candidate at
`prefill_m16384` with A5 memory and L2-cache counters and stores profiler text
summaries as gzip+base64 `record_type=profile_artifact` rows. Profiling
failures remain diagnostic rows and do not discard valid benchmark results.

The standard remote run also invokes native `msprof op` for a small probe set
with `--warm-up=10 --launch-count=5` and an exact `--kernel-name` for the
selected baseline or candidate Triton kernel.  Parsed `OpBasicInfo.csv`
`Task Duration(us)` values are stored as `record_type=msprof_op`, and the raw
files are stored as gzip+base64 `record_type=msprof_op_artifact` rows.  These
device task durations, rather than the approximately 30-us event submission
floor, are authoritative for the small-M dispatch threshold.  The final
verification probe covers M=576/577/640/641.  Use
`--capture-msprof-op off` only for manual runs that intentionally skip them.

Use `BENCH_PYTHON=/path/to/python` to select a different interpreter.  Use
`--device npu:N` to select the NPU (`npu:5` by default) and `--interval
SECONDS` to change the poll interval.  `--once --run-now` runs and publishes
exactly one current-HEAD benchmark, while plain `--once` performs one poll
only.

## Benchmark coverage

The fixed production-local shape is BF16, 6 query heads, 1 KV head,
`head_dim=256`, and `rope_dim=64`.

- Decode: every concurrency/token count from `M=1` through `M=128`.
- Prefill: dense crossover probes from `M=128` through `M=1281`, the
  `M=2048..4097` threshold pairs, and `M=6145,7169,8191,8192,9616,16384`.
- KV mirror: `(N=8192, BS=4)` and `(N=16384, BS=8)`, where Q has only `BS`
  rows while K and positions have `N` rows.

The dedicated blocked kernel is selected only for the candidate's ordinary
`prefill` phase.  Native msprof-op establishes the all-M threshold at `M=640`:
M=640 exact64 improves 15.65% and M=641 masked64 improves 8.32%.  M=576 exact64
improves 6.23% and is retained as an aligned fast path, but M=577 masked64
regresses 1.8%, so other M below 640 use the shared kernel.  Decode and KV
mirror never select the dedicated prefill kernel.

The `baseline` kernel is frozen.  Later optimization rounds should edit only
the clearly marked `candidate` section in
`bench_welmv4_inplace_rope_npu.py`.  Correctness must pass before performance
is measured.
