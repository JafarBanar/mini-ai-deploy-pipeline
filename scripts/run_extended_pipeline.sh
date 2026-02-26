#!/usr/bin/env bash
set -euo pipefail

GATE_THRESHOLD="${GATE_THRESHOLD:-1.20}"

python src/train.py --out artifacts/model.pt --epochs 3 --batch-size 64
python src/export_onnx.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/parity_check.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/accuracy_compare.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx --val-samples 1024
python src/quantize_onnx.py --in-onnx artifacts/model.onnx --out-onnx artifacts/model.int8.onnx --per-channel
python src/benchmark.py --onnx artifacts/model.onnx --out artifacts/bench.json --warmup 50 --iters 500
python src/benchmark_compare.py --fp32-onnx artifacts/model.onnx --int8-onnx artifacts/model.int8.onnx --out artifacts/bench_compare.json
python src/experiment_grid.py --onnx artifacts/model.onnx --batch-sizes 1,4,8 --warmups 20,50 --iters-list 200,500 --out artifacts/experiments.json
python src/gate_regression.py --current artifacts/bench.json --baseline artifacts/baseline.json --threshold "$GATE_THRESHOLD"
pytest -q
