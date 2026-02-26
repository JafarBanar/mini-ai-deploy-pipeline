import argparse
import json
import os

try:
    from .benchmark import benchmark_ort  # type: ignore[attr-defined]
except ImportError:
    from benchmark import benchmark_ort


def compare(
    fp32_onnx: str = "artifacts/model.onnx",
    int8_onnx: str = "artifacts/model.int8.onnx",
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    mode: str = "core",
    out_json: str = "artifacts/bench_compare.json",
) -> dict:
    fp32_stats = benchmark_ort(
        onnx_path=fp32_onnx,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
        mode=mode,
        out_json="artifacts/bench_fp32.json",
    )
    int8_stats = benchmark_ort(
        onnx_path=int8_onnx,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
        mode=mode,
        out_json="artifacts/bench_int8.json",
    )

    fp32_p95 = fp32_stats["latency_ms"]["p95"]
    int8_p95 = int8_stats["latency_ms"]["p95"]
    speedup = fp32_p95 / int8_p95 if int8_p95 > 0 else None

    result = {
        "fp32_onnx": fp32_onnx,
        "int8_onnx": int8_onnx,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "mode": mode,
        "fp32_p95_ms": fp32_p95,
        "int8_p95_ms": int8_p95,
        "p95_speedup_x": speedup,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP32 and INT8 ONNX benchmark.")
    parser.add_argument("--fp32-onnx", default="artifacts/model.onnx")
    parser.add_argument("--int8-onnx", default="artifacts/model.int8.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mode", choices=["core", "e2e"], default="core")
    parser.add_argument("--out", default="artifacts/bench_compare.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(
        fp32_onnx=args.fp32_onnx,
        int8_onnx=args.int8_onnx,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        mode=args.mode,
        out_json=args.out,
    )
