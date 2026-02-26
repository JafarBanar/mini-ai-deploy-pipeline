import argparse
import json
import time
from typing import Any

import numpy as np

try:
    from .backends import create_backend_session  # type: ignore[attr-defined]
    from .telemetry import JsonlTelemetryLogger  # type: ignore[attr-defined]
    from .utils_time import percentile  # type: ignore[attr-defined]
except ImportError:
    from backends import create_backend_session
    from telemetry import JsonlTelemetryLogger
    from utils_time import percentile


def _core_input(batch_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)


def _raw_input(batch_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch_size, 32, 32, 3), dtype=np.uint8)


def _preprocess_cifar10(raw_nhwc: np.ndarray) -> np.ndarray:
    x = raw_nhwc.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    return np.transpose(x, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _postprocess_logits(outputs: list[np.ndarray]) -> np.ndarray:
    return np.argmax(outputs[0], axis=1)


def _summarize_ms(times_ms: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(times_ms, 50),
        "p90": percentile(times_ms, 90),
        "p95": percentile(times_ms, 95),
        "p99": percentile(times_ms, 99),
        "mean": float(np.mean(times_ms)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
    }


def _fps_from_mean(mean_ms: float) -> float:
    return float(1000.0 / mean_ms) if mean_ms > 0 else 0.0


def benchmark_backend(
    backend: str,
    model_path: str,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    mode: str = "core",
    device: str = "auto",
    precision: str = "fp16",
    out_json: str = "artifacts/bench.json",
    telemetry_jsonl: str | None = None,
) -> dict[str, Any]:
    sess = create_backend_session(
        backend=backend,
        model_path=model_path,
        device=device,
        precision=precision,
    )
    info = sess.info()

    if mode == "core":
        x = _core_input(batch_size=batch_size, seed=42)
    elif mode == "e2e":
        raw_x = _raw_input(batch_size=batch_size, seed=42)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "core":
        sess.warmup(x, iters=warmup)
    else:
        for _ in range(warmup):
            x = _preprocess_cifar10(raw_x)
            outputs = sess.infer(x)
            _ = _postprocess_logits(outputs)

    e2e_times_ms: list[float] = []
    preprocess_ms: list[float] = []
    infer_ms: list[float] = []
    postprocess_ms: list[float] = []

    telemetry_logger = JsonlTelemetryLogger(telemetry_jsonl) if telemetry_jsonl else None
    for i in range(iters):
        if mode == "core":
            t0 = time.perf_counter()
            _ = sess.infer(x)
            t1 = time.perf_counter()
            e2e = (t1 - t0) * 1000.0
            e2e_times_ms.append(e2e)
            if telemetry_logger:
                telemetry_logger.log(
                    {
                        "type": "iter",
                        "iter": i,
                        "backend": info.name,
                        "mode": mode,
                        "batch_size": batch_size,
                        "e2e_ms": e2e,
                        "fps": _fps_from_mean(e2e),
                        "dropped_frames": 0,
                        "queue_depth": 0,
                    }
                )
        else:
            t0 = time.perf_counter()
            x_prep = _preprocess_cifar10(raw_x)
            t1 = time.perf_counter()
            outputs = sess.infer(x_prep)
            t2 = time.perf_counter()
            _ = _postprocess_logits(outputs)
            t3 = time.perf_counter()

            prep = (t1 - t0) * 1000.0
            inf = (t2 - t1) * 1000.0
            post = (t3 - t2) * 1000.0
            e2e = (t3 - t0) * 1000.0

            preprocess_ms.append(prep)
            infer_ms.append(inf)
            postprocess_ms.append(post)
            e2e_times_ms.append(e2e)

            if telemetry_logger:
                telemetry_logger.log(
                    {
                        "type": "iter",
                        "iter": i,
                        "backend": info.name,
                        "mode": mode,
                        "batch_size": batch_size,
                        "preprocess_ms": prep,
                        "infer_ms": inf,
                        "postprocess_ms": post,
                        "e2e_ms": e2e,
                        "fps": _fps_from_mean(e2e),
                        "dropped_frames": 0,
                        "queue_depth": 0,
                    }
                )
    latency = _summarize_ms(e2e_times_ms)
    stats: dict[str, Any] = {
        "backend": info.name,
        "backend_extra": info.extra,
        "model_path": info.model_path,
        "mode": mode,
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "latency_ms": latency,
        "fps": _fps_from_mean(latency["mean"]),
        "dropped_frames": 0,
        "queue_depth": 0,
    }
    if mode == "e2e":
        stats["stage_latency_ms"] = {
            "preprocess": _summarize_ms(preprocess_ms),
            "infer": _summarize_ms(infer_ms),
            "postprocess": _summarize_ms(postprocess_ms),
        }

    if telemetry_logger:
        telemetry_logger.log({"type": "summary", **stats})
        telemetry_logger.close()

    print(json.dumps(stats, indent=2))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def benchmark_ort(
    onnx_path: str,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    mode: str = "core",
    device: str = "auto",
    precision: str = "fp16",
    out_json: str = "artifacts/bench.json",
    telemetry_jsonl: str | None = None,
) -> dict[str, Any]:
    return benchmark_backend(
        backend="ort",
        model_path=onnx_path,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
        mode=mode,
        device=device,
        precision=precision,
        out_json=out_json,
        telemetry_jsonl=telemetry_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference latency.")
    parser.add_argument("--backend", choices=["ort", "tensorrt", "tvm"], default="ort")
    parser.add_argument("--model", default="artifacts/model.onnx")
    parser.add_argument("--onnx", default=None, help="Compatibility alias for --model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mode", choices=["core", "e2e"], default="core")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out", default="artifacts/bench.json")
    parser.add_argument("--telemetry-jsonl", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.onnx if args.onnx else args.model
    benchmark_backend(
        backend=args.backend,
        model_path=model_path,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        mode=args.mode,
        device=args.device,
        precision=args.precision,
        out_json=args.out,
        telemetry_jsonl=args.telemetry_jsonl,
    )
