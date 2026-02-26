import argparse
import json
import os
from itertools import product

try:
    from .benchmark import benchmark_ort  # type: ignore[attr-defined]
except ImportError:
    from benchmark import benchmark_ort


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def run_grid(
    onnx_path: str,
    batch_sizes: list[int],
    warmups: list[int],
    iters_list: list[int],
    mode: str,
    out_json: str,
) -> dict:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    runs = []
    for batch_size, warmup, iters in product(batch_sizes, warmups, iters_list):
        stats = benchmark_ort(
            onnx_path=onnx_path,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            mode=mode,
            out_json=f"artifacts/bench_b{batch_size}_w{warmup}_i{iters}.json",
        )
        runs.append(
            {
                "batch_size": batch_size,
                "warmup": warmup,
                "iters": iters,
                "p50_ms": stats["latency_ms"]["p50"],
                "p95_ms": stats["latency_ms"]["p95"],
                "mean_ms": stats["latency_ms"]["mean"],
                "providers": stats["providers"],
            }
        )

    summary = {"onnx_path": onnx_path, "mode": mode, "runs": runs}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark parameter sweep.")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--warmups", default="20,50")
    parser.add_argument("--iters-list", default="200,500")
    parser.add_argument("--mode", choices=["core", "e2e"], default="core")
    parser.add_argument("--out", default="artifacts/experiments.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(
        onnx_path=args.onnx,
        batch_sizes=_parse_int_list(args.batch_sizes),
        warmups=_parse_int_list(args.warmups),
        iters_list=_parse_int_list(args.iters_list),
        mode=args.mode,
        out_json=args.out,
    )
