# Mini AI Deploy Pipeline

A compact, interview-friendly deployment pipeline that demonstrates:

- PyTorch model definition
- ONNX export
- ONNX Runtime inference
- Latency benchmarking (`p50`, `p95`, etc.)
- CI smoke test + benchmark regression gate

## Project Layout

```text
mini-ai-deploy-pipeline/
  README.md
  requirements.txt
  src/
    model.py
    export_onnx.py
    infer_ort.py
    benchmark.py
    utils_time.py
  artifacts/
    .gitkeep
    # model.onnx (generated)
    # bench.json (generated)
    # baseline.json (optional)
  tests/
    test_smoke.py
  .github/workflows/ci.yml
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/export_onnx.py
python src/infer_ort.py
python src/benchmark.py
pytest -q
```

## Baseline + Regression Gate

CI compares current benchmark `p95` with `artifacts/baseline.json`:

- If no baseline exists, CI prints current latency and skips failing.
- If baseline exists, CI fails when:
  - `current_p95 > baseline_p95 * 1.05`

To create/update baseline locally:

```bash
python src/benchmark.py
cp artifacts/bench.json artifacts/baseline.json
```

## Interview Talking Point

This repo demonstrates a practical deploy path:

`PyTorch -> ONNX -> ONNX Runtime benchmark -> CI regression gate`

and is structured to extend into TensorRT and ROS2 integration.
