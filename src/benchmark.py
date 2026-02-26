import json
import time
import argparse

import numpy as np
import onnxruntime as ort

try:
    from .utils_time import percentile  # type: ignore[attr-defined]
except ImportError:
    from utils_time import percentile


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def _core_input(batch_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)


def _raw_input(batch_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Simulate camera-style uint8 NHWC frames.
    return rng.integers(0, 256, size=(batch_size, 32, 32, 3), dtype=np.uint8)


def _preprocess_cifar10(raw_nhwc: np.ndarray) -> np.ndarray:
    x = raw_nhwc.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    # Convert NHWC -> NCHW for model input.
    return np.transpose(x, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _postprocess_logits(outputs: list[np.ndarray]) -> np.ndarray:
    return np.argmax(outputs[0], axis=1)


def benchmark_ort(
    onnx_path: str,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    mode: str = "core",
    out_json: str = "artifacts/bench.json",
):
    providers = _select_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    if mode == "core":
        x = _core_input(batch_size=batch_size, seed=42)
    elif mode == "e2e":
        raw_x = _raw_input(batch_size=batch_size, seed=42)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for _ in range(warmup):
        if mode == "core":
            sess.run(None, {input_name: x})
        else:
            x = _preprocess_cifar10(raw_x)
            outputs = sess.run(None, {input_name: x})
            _postprocess_logits(outputs)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if mode == "core":
            sess.run(None, {input_name: x})
        else:
            x = _preprocess_cifar10(raw_x)
            outputs = sess.run(None, {input_name: x})
            _postprocess_logits(outputs)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    stats = {
        "backend": "onnxruntime",
        "mode": mode,
        "providers": providers,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "latency_ms": {
            "p50": percentile(times_ms, 50),
            "p90": percentile(times_ms, 90),
            "p95": percentile(times_ms, 95),
            "p99": percentile(times_ms, 99),
            "mean": float(np.mean(times_ms)),
            "min": float(np.min(times_ms)),
            "max": float(np.max(times_ms)),
        },
    }

    print(json.dumps(stats, indent=2))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime latency.")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mode", choices=["core", "e2e"], default="core")
    parser.add_argument("--out", default="artifacts/bench.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_ort(
        onnx_path=args.onnx,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        mode=args.mode,
        out_json=args.out,
    )
