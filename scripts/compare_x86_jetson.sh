#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-artifacts/model.onnx}"
JETSON_BENCH="${JETSON_BENCH:-artifacts/bench_ort_jetson.json}"
X86_OUT="${X86_OUT:-artifacts/bench_x86.json}"
COMPARE_OUT="${COMPARE_OUT:-artifacts/x86_vs_jetson_compare.json}"
BACKEND="${BACKEND:-ort}"
MODE="${MODE:-e2e}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"

python src/benchmark.py \
  --backend "$BACKEND" \
  --model "$MODEL_PATH" \
  --mode "$MODE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --out "$X86_OUT" \
  --telemetry-jsonl artifacts/telemetry/x86_compare.jsonl

python src/compare_bench_json.py \
  --reference "$X86_OUT" \
  --candidate "$JETSON_BENCH" \
  --reference-label "x86_$BACKEND" \
  --candidate-label "jetson_$BACKEND" \
  --out "$COMPARE_OUT"

echo "Wrote comparison: $COMPARE_OUT"

