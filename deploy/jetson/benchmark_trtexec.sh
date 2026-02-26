#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${ENGINE_PATH:-artifacts/model.plan}"
WARMUP_MS="${WARMUP_MS:-${WARMUP:-200}}"
ITERS="${ITERS:-500}"
SHAPES="${SHAPES:-input:1x3x32x32}"
LOG_PATH="${LOG_PATH:-artifacts/trtexec.log}"
OUT_JSON="${OUT_JSON:-artifacts/bench_trt.json}"
BASELINE="${BASELINE:-artifacts/baseline_trt.json}"
THRESHOLD="${THRESHOLD:-1.10}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec is required. Install TensorRT on Jetson/Orin first."
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

echo "Benchmarking TensorRT engine with trtexec..."
trtexec \
  --loadEngine="$ENGINE_PATH" \
  --shapes="$SHAPES" \
  --warmUp="$WARMUP_MS" \
  --iterations="$ITERS" \
  --percentile=95,99 \
  > "$LOG_PATH" 2>&1

python src/parse_trtexec_log.py --log "$LOG_PATH" --out "$OUT_JSON"
python src/gate_regression.py --current "$OUT_JSON" --baseline "$BASELINE" --threshold "$THRESHOLD"
