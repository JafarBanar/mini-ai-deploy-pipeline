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


def benchmark_ort(
    onnx_path: str,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    out_json: str = "artifacts/bench.json",
):
    providers = _select_providers()
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(42)
    x = rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    stats = {
        "backend": "onnxruntime",
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
    parser.add_argument("--out", default="artifacts/bench.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_ort(
        onnx_path=args.onnx,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        out_json=args.out,
    )
