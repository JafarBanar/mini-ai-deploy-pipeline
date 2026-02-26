#!/usr/bin/env bash
set -euo pipefail

python src/train.py --out artifacts/model.pt --epochs 3 --batch-size 64
python src/export_onnx.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
python src/parity_check.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
pytest -q
python src/infer_ort.py --onnx artifacts/model.onnx --batch-size 1
python src/benchmark.py --onnx artifacts/model.onnx --out artifacts/bench.json --warmup 20 --iters 200
python src/gate_regression.py --current artifacts/bench.json --baseline artifacts/baseline.json --threshold 1.05
