# Mini AI Deploy Pipeline

This project is a hands-on learning pipeline for model deployment:

`Train (PyTorch) -> Export (ONNX) -> Validate parity (PyTorch vs ONNX) -> Inference (ORT) -> Benchmark (p50/p95) -> Regression gate`

## Why this repo exists

If you want to learn deployment, this gives you a minimal but complete path:

- Train a real model (not just random weights)
- Convert to ONNX
- Run with ONNX Runtime
- Measure latency with warmup and percentiles
- Compare PyTorch vs ONNX task accuracy
- Run benchmark sweeps across settings
- Compare FP32 vs INT8 dynamic quantized ONNX
- Enforce performance policy in CI

## Project layout

```text
.
├── src/
│   ├── model.py              # TinyCNN
│   ├── backends/             # runtime abstraction (ort/trt/tvm)
│   ├── train.py              # synthetic or CIFAR-10 training + checkpoint save
│   ├── datasets.py           # dataset loaders/subsetting (synthetic/cifar10)
│   ├── export_onnx.py        # checkpoint -> ONNX
│   ├── parity_check.py       # numerical output check (PyTorch vs ONNX)
│   ├── accuracy_compare.py   # task accuracy comparison on same val set
│   ├── infer.py              # single inference call by backend
│   ├── infer_ort.py          # legacy ORT inference entrypoint
│   ├── benchmark.py          # p50/p90/p95/p99 + json output
│   ├── quantize_onnx.py      # dynamic INT8 quantization
│   ├── benchmark_compare.py  # fp32 vs int8 benchmark summary
│   ├── experiment_grid.py    # parameter sweeps
│   ├── telemetry.py          # jsonl telemetry logger
│   ├── parse_trtexec_log.py  # parse trtexec logs into bench.json schema
│   ├── compare_bench_json.py # compare two bench.json files (reference vs candidate)
│   └── gate_regression.py    # p95 threshold gate against baseline
├── deploy/jetson/            # Jetson/Orin scripts (power, engine build, benchmark, matrix)
├── ros2_node/                # ROS2 Python package + launch file
├── scripts/
│   └── run_pipeline.sh       # one-command local run
│   └── run_extended_pipeline.sh
│   └── run_real_data_pipeline.sh
│   └── compare_x86_jetson.sh
├── tests/
│   └── test_smoke.py
├── artifacts/
│   ├── baseline.json         # committed reference benchmark
│   ├── baseline_ci.json      # GitHub-hosted runner benchmark baseline
│   ├── baseline_real.json    # committed real-data reference benchmark
│   ├── baseline_jetson_ort.json
│   ├── baseline_trt.json
│   ├── telemetry/            # telemetry jsonl outputs
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

Real data (CIFAR-10):

```bash
python src/train.py \
  --dataset cifar10 \
  --data-dir artifacts/data \
  --download \
  --train-samples 20000 \
  --val-samples 5000 \
  --epochs 3 \
  --out artifacts/model.pt
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
python src/infer.py --backend ort --model artifacts/model.onnx --batch-size 1
```

### 5) Benchmark latency and save JSON

```bash
python src/benchmark.py --onnx artifacts/model.onnx --warmup 20 --iters 200 --out artifacts/bench.json
```

To include preprocessing + postprocessing in timing:

```bash
python src/benchmark.py --onnx artifacts/model.onnx --mode e2e --warmup 50 --iters 500 --out artifacts/bench.json
```

Backend switch examples:

```bash
python src/benchmark.py --backend ort --model artifacts/model.onnx --mode e2e --out artifacts/bench.json
python src/benchmark.py --backend tensorrt --model artifacts/model.onnx --device auto --precision fp16 --mode e2e --out artifacts/bench.json
```

### 6) Apply regression gate

```bash
python src/gate_regression.py --current artifacts/bench.json --baseline artifacts/baseline.json --threshold 1.05
```

Policy: fail if current `p95` is more than 5% slower than baseline.

### 7) Compare task accuracy (PyTorch vs ONNX)

```bash
python src/accuracy_compare.py --checkpoint artifacts/model.pt --onnx artifacts/model.onnx --val-samples 1024
```

### 8) Create INT8 ONNX and compare latency with FP32

```bash
python src/quantize_onnx.py --in-onnx artifacts/model.onnx --out-onnx artifacts/model.int8.onnx --per-channel
python src/benchmark_compare.py --fp32-onnx artifacts/model.onnx --int8-onnx artifacts/model.int8.onnx
```

### 9) Run benchmark sweeps

```bash
python src/experiment_grid.py --onnx artifacts/model.onnx --batch-sizes 1,4,8 --warmups 20,50 --iters-list 200,500 --out artifacts/experiments.json
```

## Run whole pipeline with one command

```bash
source .venv/bin/activate
./scripts/run_pipeline.sh
```

## Run full extended learning pipeline

```bash
source .venv/bin/activate
./scripts/run_extended_pipeline.sh
```

The extended runner uses a relaxed gate by default (`GATE_THRESHOLD=1.20`) to avoid false failures from local timing jitter.
You can override it:

```bash
GATE_THRESHOLD=1.05 ./scripts/run_extended_pipeline.sh
```

## Run full pipeline on real data (CIFAR-10)

```bash
source .venv/bin/activate
./scripts/run_real_data_pipeline.sh
```

Useful overrides:

```bash
TRAIN_SAMPLES=50000 VAL_SAMPLES=10000 EPOCHS=5 BENCH_MODE=e2e GATE_THRESHOLD=1.20 ./scripts/run_real_data_pipeline.sh
```

Initialize/update the real-data baseline from current run:

```bash
UPDATE_BASELINE=1 ./scripts/run_real_data_pipeline.sh
```

## Telemetry output

Benchmark scripts can emit per-iteration telemetry JSONL:

```bash
python src/benchmark.py --backend ort --model artifacts/model.onnx --mode e2e --telemetry-jsonl artifacts/telemetry/bench.jsonl
```

Each row includes iteration latencies (`preprocess_ms`, `infer_ms`, `postprocess_ms`, `e2e_ms`), plus `fps`, `dropped_frames`, and `queue_depth`.

## Jetson / Orin flow

```bash
./deploy/jetson/check_env.sh
./deploy/jetson/setup_power.sh 0
BACKEND=ort BENCH_MODE=e2e ./deploy/jetson/benchmark_jetson.sh
```

TensorRT flow with `trtexec`:

```bash
ONNX_PATH=artifacts/model.onnx ENGINE_PATH=artifacts/model.plan PRECISION=fp16 ./deploy/jetson/build_trt_engine.sh
BACKEND=tensorrt ENGINE_PATH=artifacts/model.plan THRESHOLD=1.20 ./deploy/jetson/benchmark_jetson.sh
```

This produces `artifacts/bench_trt.json` in the same format expected by the regression gate.

One-command ORT vs TensorRT matrix:

```bash
./deploy/jetson/run_backend_matrix.sh
```

Outputs:

- `artifacts/bench_ort_jetson.json`
- `artifacts/bench_trt.json`
- `artifacts/backend_compare.json`

Defaults use p95 regression gates (`THRESHOLD_ORT=1.10`, `THRESHOLD_TRT=1.10`) against:

- `artifacts/baseline_jetson_ort.json`
- `artifacts/baseline_trt.json`

Compare x86 and Jetson with the same benchmark config:

```bash
JETSON_BENCH=artifacts/bench_ort_jetson.json BACKEND=ort MODE=e2e ./scripts/compare_x86_jetson.sh
```

### Jetson Runner Setup (for nightly workflow)

If `Jetson Nightly Benchmark` stays queued, your repo likely has no self-hosted runner with labels:
`self-hosted`, `linux`, `arm64`, `jetson`.

On the Jetson device:

```bash
export RUNNER_TOKEN=<token-from-github-runner-ui>
./deploy/jetson/setup_actions_runner.sh
```

Then verify:

```bash
gh api repos/JafarBanar/mini-ai-deploy-pipeline/actions/runners
```

## ROS2 node package

`ros2_node/` is a proper ROS2 Python package (`edge_inference_node`) with a launch file.

From a ROS2 workspace root:

```bash
mkdir -p src
rsync -a <path-to-this-repo>/ros2_node/ src/edge_inference_node/
colcon build --packages-select edge_inference_node
source install/setup.bash
ros2 launch edge_inference_node inference.launch.py backend:=ort model_path:=<abs-path-to-model.onnx>
```

It subscribes to `/camera/image`, runs preprocess -> inference -> postprocess, and publishes JSON metrics on `/inference_metrics`.
Key params: `backend`, `model_path`, `batch_size`, `device`, `precision`, `image_topic`, `metrics_topic`.

## CI behavior

On every push/PR, GitHub Actions runs:

1. training
2. export
3. parity check
4. smoke test
5. benchmark (+ telemetry artifact)
6. p95 regression gate
7. PyTorch-vs-ONNX accuracy comparison
8. FP32-vs-INT8 benchmark comparison
9. benchmark sweep

The CI gate uses `artifacts/baseline_ci.json` with a 10% p95 threshold for hosted-runner stability.

Nightly self-hosted Jetson workflow:

```text
.github/workflows/jetson-nightly.yml
```

Nightly Jetson runs ORT and TensorRT benchmarks on-device via `deploy/jetson/run_backend_matrix.sh` with p95 gates.

## GitHub CLI token fix

If `gh auth status` says token is invalid:

```bash
gh auth login -h github.com
gh auth status
```

## Connection issue and SSH setup

If `gh` says token invalid in one terminal but valid in another, usually one of these is different:

- shell session/environment
- keychain access
- network route (VPN/proxy/firewall)

Check:

```bash
which gh
gh --version
gh auth status
```

Switch git remote to SSH:

```bash
git remote set-url origin git@github.com:JafarBanar/mini-ai-deploy-pipeline.git
git remote -v
ssh -T git@github.com
git push
```
