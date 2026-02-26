#!/usr/bin/env bash
set -euo pipefail

# One-command Jetson matrix run:
# 1) ORT benchmark + gate
# 2) TensorRT engine build (optional) + benchmark + gate
# 3) ORT vs TRT comparison report

MODE="${MODE:-e2e}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
THRESHOLD_ORT="${THRESHOLD_ORT:-1.20}"
THRESHOLD_TRT="${THRESHOLD_TRT:-1.20}"

ONNX_PATH="${ONNX_PATH:-artifacts/model.onnx}"
ENGINE_PATH="${ENGINE_PATH:-artifacts/model.plan}"
PRECISION="${PRECISION:-fp16}"
SHAPES="${SHAPES:-input:1x3x32x32}"
BUILD_ENGINE="${BUILD_ENGINE:-1}"

ORT_OUT="${ORT_OUT:-artifacts/bench_ort_jetson.json}"
ORT_TELEMETRY="${ORT_TELEMETRY:-artifacts/telemetry_ort_jetson.jsonl}"
TRT_OUT="${TRT_OUT:-artifacts/bench_trt.json}"
TRT_LOG="${TRT_LOG:-artifacts/trtexec.log}"
COMPARE_OUT="${COMPARE_OUT:-artifacts/backend_compare.json}"

BASELINE_ORT="${BASELINE_ORT:-artifacts/baseline_real.json}"
BASELINE_TRT="${BASELINE_TRT:-artifacts/baseline_trt.json}"

echo "== ORT benchmark =="
BACKEND=ort \
MODEL_PATH="$ONNX_PATH" \
MODE="$MODE" \
WARMUP="$WARMUP" \
ITERS="$ITERS" \
OUT_JSON="$ORT_OUT" \
TELEMETRY="$ORT_TELEMETRY" \
BASELINE="$BASELINE_ORT" \
THRESHOLD="$THRESHOLD_ORT" \
./deploy/jetson/benchmark_jetson.sh

if [ "$BUILD_ENGINE" = "1" ]; then
  echo "== TensorRT engine build =="
  ONNX_PATH="$ONNX_PATH" \
  ENGINE_PATH="$ENGINE_PATH" \
  PRECISION="$PRECISION" \
  SHAPES="$SHAPES" \
  ./deploy/jetson/build_trt_engine.sh
fi

echo "== TensorRT benchmark =="
BACKEND=tensorrt \
ENGINE_PATH="$ENGINE_PATH" \
MODE="$MODE" \
WARMUP="$WARMUP" \
ITERS="$ITERS" \
OUT_JSON="$TRT_OUT" \
LOG_PATH="$TRT_LOG" \
BASELINE="$BASELINE_TRT" \
THRESHOLD="$THRESHOLD_TRT" \
./deploy/jetson/benchmark_jetson.sh

echo "== ORT vs TensorRT compare =="
python src/compare_bench_json.py \
  --reference "$ORT_OUT" \
  --candidate "$TRT_OUT" \
  --reference-label onnxruntime \
  --candidate-label tensorrt \
  --out "$COMPARE_OUT"

echo "Wrote:"
echo "  ORT:     $ORT_OUT"
echo "  TRT:     $TRT_OUT"
echo "  Compare: $COMPARE_OUT"

