#!/usr/bin/env bash
# Capture Triton frontend IR and the final BishengIR pass for one kernel launch.
# This is the latency-optimizer run_and_extract workflow narrowed to one WeLMv4
# candidate/case so that the normal remote benchmark remains affordable.

set -euo pipefail

PYTHON_SCRIPT="${1:-}"
if [[ -z "$PYTHON_SCRIPT" ]]; then
    echo "usage: $0 <python-script> [script arguments...]" >&2
    exit 2
fi
shift

PYTHON_BIN="${BENCH_PYTHON:-python}"
IR_OUTPUT_DIR="${IR_OUTPUT_DIR:-$(pwd)/welmv4_rope_ir}"
TARGET="${BISHENGIR_TARGET:-Ascend950PR_957b}"
BISHENGIR_BIN="${BISHENGIR_COMPILE:-}"
if [[ -z "$BISHENGIR_BIN" ]]; then
    BISHENGIR_BIN="$(command -v bishengir-compile 2>/dev/null || true)"
fi
if [[ -z "$BISHENGIR_BIN" ]]; then
    echo "bishengir-compile is unavailable; set BISHENGIR_COMPILE or PATH" >&2
    exit 3
fi

mkdir -p "$IR_OUTPUT_DIR"
RUN_LOG="$(mktemp /tmp/welmv4_rope_compile.XXXXXX.log)"
FULL_IR="$(mktemp /tmp/welmv4_rope_bishengir.XXXXXX.log)"
trap 'rm -f "$RUN_LOG" "$FULL_IR"' EXIT

export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0
export TRITON_DISABLE_FFTS=1

echo "Compiling one candidate launch with Triton IR dumps enabled"
"$PYTHON_BIN" "$PYTHON_SCRIPT" "$@" 2>&1 | tee "$RUN_LOG"

DUMP_DIR="$(awk '/Dumping intermediate results to/ {dir=$NF} END {print dir}' "$RUN_LOG")"
if [[ -z "$DUMP_DIR" || ! -f "$DUMP_DIR/kernel.ttadapter.mlir" ]]; then
    echo "Triton did not report a usable kernel.ttadapter.mlir dump" >&2
    exit 4
fi

KERNEL_NAME="$(sed -nE 's/.*(func\.func|tt\.func|module) @([A-Za-z0-9_]+).*/\2/p' "$DUMP_DIR/kernel.ttadapter.mlir" | head -n 1)"
KERNEL_NAME="${KERNEL_NAME:-welmv4_inplace_rope_candidate}"
if [[ -f "$DUMP_DIR/kernel.ttir.mlir" ]]; then
    cp "$DUMP_DIR/kernel.ttir.mlir" "$IR_OUTPUT_DIR/${KERNEL_NAME}_ttir.mlir"
fi
cp "$DUMP_DIR/kernel.ttadapter.mlir" "$IR_OUTPUT_DIR/${KERNEL_NAME}_ttadapter.mlir"

HELP_TEXT="$("$BISHENGIR_BIN" --help 2>&1 || true)"
if grep -q "mlir-print-ir-after-all" <<<"$HELP_TEXT"; then
    PRINT_FLAG="--mlir-print-ir-after-all"
    PRINT_MODE="markers"
elif grep -q "bishengir-print-ir-after" <<<"$HELP_TEXT"; then
    PRINT_FLAG="--bishengir-print-ir-after=hivm-inject-sync"
    PRINT_MODE="direct"
elif grep -q "print-after-all" <<<"$HELP_TEXT"; then
    PRINT_FLAG="--print-after-all"
    PRINT_MODE="markers"
else
    echo "bishengir-compile exposes no supported IR dump flag" >&2
    exit 5
fi

echo "Lowering $KERNEL_NAME for target $TARGET with $PRINT_FLAG"
(
    cd "$DUMP_DIR"
    "$BISHENGIR_BIN" \
        --target="$TARGET" \
        --enable-auto-multi-buffer=True \
        --enable-auto-bind-sub-block=True \
        --enable-hfusion-compile=true \
        --enable-hivm-compile=true \
        --enable-triton-kernel-compile=true \
        "$PRINT_FLAG" \
        kernel.ttadapter.mlir
) >"$FULL_IR" 2>&1

LAST_PASS="$IR_OUTPUT_DIR/${KERNEL_NAME}_last_pass.mlir"
if [[ "$PRINT_MODE" == "direct" ]]; then
    cp "$FULL_IR" "$LAST_PASS"
else
    LAST_LINE="$(grep -n "IR Dump After" "$FULL_IR" | tail -n 1 | cut -d: -f1)"
    if [[ -z "$LAST_LINE" ]]; then
        echo "compiler completed but emitted no 'IR Dump After' marker" >&2
        exit 6
    fi
    sed -n "${LAST_LINE},\$p" "$FULL_IR" >"$LAST_PASS"
fi

if [[ ! -s "$LAST_PASS" ]] || ! grep -Eq "hivm\.hir\.|llvm\." "$LAST_PASS"; then
    echo "last-pass IR is empty or stopped before HIVM/LLVM lowering" >&2
    exit 7
fi

echo "IR capture complete:"
find "$IR_OUTPUT_DIR" -maxdepth 1 -type f -name '*.mlir' -printf '  %f %s bytes\n'
