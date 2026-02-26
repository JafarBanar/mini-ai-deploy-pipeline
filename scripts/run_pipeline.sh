#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-ort}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-fp16}"

python src/train.py --out artifacts/model.pt --epochs 3 --batch-size 64
python src/export_onnx.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/parity_check.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
pytest -q
python src/infer.py --backend "$BACKEND" --model artifacts/model.onnx --batch-size 1 --device "$DEVICE" --precision "$PRECISION"
python src/benchmark.py --backend "$BACKEND" --model artifacts/model.onnx --device "$DEVICE" --precision "$PRECISION" --out artifacts/bench.json --warmup 20 --iters 200 --telemetry-jsonl artifacts/telemetry/core.jsonl
python src/gate_regression.py --current artifacts/bench.json --baseline artifacts/baseline.json --threshold 1.05
