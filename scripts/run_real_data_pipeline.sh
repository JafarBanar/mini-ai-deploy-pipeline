#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-artifacts/data}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"
EPOCHS="${EPOCHS:-3}"
GATE_THRESHOLD="${GATE_THRESHOLD:-1.20}"
BENCH_MODE="${BENCH_MODE:-e2e}"
BASELINE_PATH="${BASELINE_PATH:-artifacts/baseline_real.json}"
UPDATE_BASELINE="${UPDATE_BASELINE:-0}"
BACKEND="${BACKEND:-ort}"

python src/train.py \
  --dataset cifar10 \
  --data-dir "$DATA_DIR" \
  --download \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --out artifacts/model.pt

python src/export_onnx.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/parity_check.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/accuracy_compare.py \
  --checkpoint artifacts/model.pt \
  --onnx artifacts/model.onnx \
  --dataset cifar10 \
  --data-dir "$DATA_DIR" \
  --download \
  --val-samples "$VAL_SAMPLES"

python src/quantize_onnx.py --in-onnx artifacts/model.onnx --out-onnx artifacts/model.int8.onnx --per-channel
python src/benchmark.py --backend "$BACKEND" --model artifacts/model.onnx --mode "$BENCH_MODE" --out artifacts/bench.json --warmup 50 --iters 500 --telemetry-jsonl artifacts/telemetry_real.jsonl
python src/benchmark_compare.py --fp32-onnx artifacts/model.onnx --int8-onnx artifacts/model.int8.onnx --mode "$BENCH_MODE" --out artifacts/bench_compare.json
python src/experiment_grid.py --onnx artifacts/model.onnx --mode "$BENCH_MODE" --batch-sizes 1,4,8 --warmups 20,50 --iters-list 200,500 --out artifacts/experiments.json

if [ "$UPDATE_BASELINE" = "1" ]; then
  cp artifacts/bench.json "$BASELINE_PATH"
  echo "Updated baseline: $BASELINE_PATH"
fi

python src/gate_regression.py --current artifacts/bench.json --baseline "$BASELINE_PATH" --threshold "$GATE_THRESHOLD"
pytest -q
