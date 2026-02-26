from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any


LATENCY_RE = re.compile(
    r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms,\s*median\s*=\s*([0-9.]+)\s*ms,\s*percentile\(95%\)\s*=\s*([0-9.]+)\s*ms,\s*percentile\(99%\)\s*=\s*([0-9.]+)\s*ms"
)
LATENCY_FALLBACK_RE = re.compile(
    r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms,\s*median\s*=\s*([0-9.]+)\s*ms,\s*percentile\(99%\)\s*=\s*([0-9.]+)\s*ms"
)
THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9.]+)\s*qps")


def parse_log(path: str) -> dict[str, Any]:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    m = LATENCY_RE.search(text)
    if m:
        min_ms, max_ms, mean_ms, median_ms, p95_ms, p99_ms = map(float, m.groups())
    else:
        mf = LATENCY_FALLBACK_RE.search(text)
        if not mf:
            raise RuntimeError("Could not parse TensorRT latency from trtexec log.")
        min_ms, max_ms, mean_ms, median_ms, p99_ms = map(float, mf.groups())
        p95_ms = p99_ms

    mt = THROUGHPUT_RE.search(text)
    throughput = float(mt.group(1)) if mt else None

    return {
        "backend": "tensorrt-trtexec",
        "latency_ms": {
            "p50": median_ms,
            "p90": None,
            "p95": p95_ms,
            "p99": p99_ms,
            "mean": mean_ms,
            "min": min_ms,
            "max": max_ms,
        },
        "throughput_qps": throughput,
        "fps": throughput if throughput is not None else (1000.0 / mean_ms if mean_ms > 0 else 0.0),
        "dropped_frames": 0,
        "queue_depth": 0,
        "source_log": path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse trtexec benchmark log into bench.json format.")
    parser.add_argument("--log", required=True, help="Path to trtexec log file")
    parser.add_argument("--out", default="artifacts/bench_trt.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = parse_log(args.log)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
