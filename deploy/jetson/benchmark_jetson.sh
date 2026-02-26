#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-ort}"
MODEL_PATH="${MODEL_PATH:-artifacts/model.onnx}"
MODE="${MODE:-e2e}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-fp16}"
OUT_JSON="${OUT_JSON:-artifacts/bench_jetson.json}"
TELEMETRY="${TELEMETRY:-artifacts/telemetry/jetson.jsonl}"
BASELINE="${BASELINE:-artifacts/baseline_jetson_ort.json}"
THRESHOLD="${THRESHOLD:-1.10}"

if [ "${BACKEND}" = "tensorrt" ] || [ "${BACKEND}" = "trt" ]; then
  ENGINE_PATH="${ENGINE_PATH:-artifacts/model.plan}"
  LOG_PATH="${LOG_PATH:-artifacts/trtexec.log}"
  ./deploy/jetson/benchmark_trtexec.sh
  exit 0
fi

python src/benchmark.py \
  --backend "${BACKEND}" \
  --model "${MODEL_PATH}" \
  --mode "${MODE}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --out "${OUT_JSON}" \
  --telemetry-jsonl "${TELEMETRY}"

python src/gate_regression.py --current "${OUT_JSON}" --baseline "${BASELINE}" --threshold "${THRESHOLD}"
