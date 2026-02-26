# Mini AI Deploy Pipeline

This project is a hands-on learning pipeline for model deployment:

`Train (PyTorch) -> Export (ONNX) -> Validate parity (PyTorch vs ONNX) -> Inference (ORT) -> Benchmark (p50/p95) -> Regression gate`

## Why this repo exists

If you want to learn deployment, this gives you a minimal but complete path:

- Train a real model (not just random weights)
- Convert to ONNX
- Run with ONNX Runtime
- Measure latency with warmup and percentiles
- Enforce performance policy in CI

## Project layout

```text
.
├── src/
│   ├── model.py              # TinyCNN
│   ├── train.py              # synthetic training + checkpoint save
│   ├── export_onnx.py        # checkpoint -> ONNX
│   ├── parity_check.py       # numerical output check (PyTorch vs ONNX)
│   ├── infer_ort.py          # single inference call
│   ├── benchmark.py          # p50/p90/p95/p99 + json output
│   └── gate_regression.py    # p95 threshold gate against baseline
├── scripts/
│   └── run_pipeline.sh       # one-command local run
├── tests/
│   └── test_smoke.py
├── artifacts/
│   ├── baseline.json         # committed reference benchmark
│   ├── model.pt              # generated
│   ├── model.onnx            # generated
│   └── bench.json            # generated
└── .github/workflows/ci.yml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Learn it step-by-step

### 1) Train a model and save checkpoint

```bash
python src/train.py --out artifacts/model.pt --epochs 3 --batch-size 64
```

### 2) Export checkpoint to ONNX

```bash
python src/export_onnx.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx --opset 17
```

### 3) Check numerical parity

```bash
python src/parity_check.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx
```

If parity fails, fix export/inference differences before trusting benchmark numbers.

### 4) Run ONNX Runtime inference

```bash
python src/infer_ort.py --onnx artifacts/model.onnx --batch-size 1
```

### 5) Benchmark latency and save JSON

```bash
python src/benchmark.py --onnx artifacts/model.onnx --warmup 20 --iters 200 --out artifacts/bench.json
```

### 6) Apply regression gate

```bash
python src/gate_regression.py --current artifacts/bench.json --baseline artifacts/baseline.json --threshold 1.05
```

Policy: fail if current `p95` is more than 5% slower than baseline.

## Run whole pipeline with one command

```bash
source .venv/bin/activate
./scripts/run_pipeline.sh
```

## CI behavior

On every push/PR, GitHub Actions runs:

1. training
2. export
3. parity check
4. smoke test
5. benchmark
6. p95 regression gate

## GitHub CLI token fix

If `gh auth status` says token is invalid:

```bash
gh auth login -h github.com
gh auth status
```
