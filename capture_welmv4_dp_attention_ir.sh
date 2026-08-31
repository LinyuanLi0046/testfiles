#!/usr/bin/env bash
# Capture every Triton dump produced by one standalone Attention launch and
# lower each TTAdapter module through the last BishengIR pass.

set -euo pipefail

PYTHON_SCRIPT="${1:-}"
if [[ -z "$PYTHON_SCRIPT" ]]; then
    echo "usage: $0 <python-script> [script arguments...]" >&2
    exit 2
fi
shift

PYTHON_BIN="${BENCH_PYTHON:-python}"
IR_OUTPUT_DIR="${IR_OUTPUT_DIR:-$(pwd)/welmv4_prefill_attention_ir}"
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
RUN_LOG="$(mktemp /tmp/welmv4_attention_compile.XXXXXX.log)"
trap 'rm -f "$RUN_LOG"' EXIT

export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0
export TRITON_DISABLE_FFTS=1

echo "Compiling one candidate Attention call with Triton IR dumps enabled"
"$PYTHON_BIN" "$PYTHON_SCRIPT" "$@" 2>&1 | tee "$RUN_LOG"

mapfile -t DUMP_DIRS < <(
    awk '/Dumping intermediate results to/ {print $NF}' "$RUN_LOG" | awk '!seen[$0]++'
)
if [[ "${#DUMP_DIRS[@]}" -eq 0 ]]; then
    echo "Triton did not report an intermediate-results directory" >&2
    exit 4
fi

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

CAPTURED=0
INDEX=0
for DUMP_DIR in "${DUMP_DIRS[@]}"; do
    if [[ ! -f "$DUMP_DIR/kernel.ttadapter.mlir" ]]; then
        continue
    fi
    INDEX=$((INDEX + 1))
    KERNEL_NAME="$(
        sed -nE 's/.*(func\.func|tt\.func|module) @([A-Za-z0-9_]+).*/\2/p' \
            "$DUMP_DIR/kernel.ttadapter.mlir" | head -n 1
    )"
    KERNEL_NAME="${KERNEL_NAME:-kernel_${INDEX}}"
    SAFE_NAME="$(tr -cs 'A-Za-z0-9_.-' '_' <<<"$KERNEL_NAME" | sed 's/_$//')"
    PREFIX="$(printf '%02d_%s' "$INDEX" "$SAFE_NAME")"

    if [[ -f "$DUMP_DIR/kernel.ttir.mlir" ]]; then
        cp "$DUMP_DIR/kernel.ttir.mlir" "$IR_OUTPUT_DIR/${PREFIX}_ttir.mlir"
    fi
    cp "$DUMP_DIR/kernel.ttadapter.mlir" "$IR_OUTPUT_DIR/${PREFIX}_ttadapter.mlir"

    FULL_IR="$(mktemp /tmp/welmv4_attention_bishengir.XXXXXX.log)"
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

    LAST_PASS="$IR_OUTPUT_DIR/${PREFIX}_last_pass.mlir"
    if [[ "$PRINT_MODE" == "direct" ]]; then
        cp "$FULL_IR" "$LAST_PASS"
    else
        LAST_LINE="$(
            grep -n "IR Dump After" "$FULL_IR" | tail -n 1 | cut -d: -f1 || true
        )"
        if [[ -z "$LAST_LINE" ]]; then
            rm -f "$FULL_IR"
            echo "$KERNEL_NAME emitted no final IR dump marker" >&2
            exit 6
        fi
        sed -n "${LAST_LINE},\$p" "$FULL_IR" >"$LAST_PASS"
    fi
    rm -f "$FULL_IR"

    if [[ ! -s "$LAST_PASS" ]] || ! grep -Eq "hivm\.hir\.|llvm\." "$LAST_PASS"; then
        echo "$KERNEL_NAME last-pass IR is empty or stopped before HIVM/LLVM" >&2
        exit 7
    fi
    CAPTURED=$((CAPTURED + 1))
done

if [[ "$CAPTURED" -eq 0 ]]; then
    echo "No usable kernel.ttadapter.mlir was found in reported dump directories" >&2
    exit 8
fi

echo "IR capture complete: kernels=$CAPTURED"
find "$IR_OUTPUT_DIR" -maxdepth 1 -type f -name '*.mlir' -printf '  %f %s bytes\n'

