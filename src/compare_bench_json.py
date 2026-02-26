from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_bench(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_row(ref: float, cand: float) -> dict[str, float]:
    delta_ms = cand - ref
    change_pct = (delta_ms / ref * 100.0) if ref != 0 else 0.0
    speedup_x = (ref / cand) if cand != 0 else 0.0
    return {
        "reference_ms": ref,
        "candidate_ms": cand,
        "delta_ms": delta_ms,
        "change_pct": change_pct,
        "speedup_x": speedup_x,
    }


def compare(
    reference_path: str,
    candidate_path: str,
    reference_label: str,
    candidate_label: str,
    out_json: str,
) -> dict[str, Any]:
    ref = _load_bench(reference_path)
    cand = _load_bench(candidate_path)

    ref_latency = ref["latency_ms"]
    cand_latency = cand["latency_ms"]

    result = {
        "reference": {"label": reference_label, "path": reference_path, "backend": ref.get("backend")},
        "candidate": {"label": candidate_label, "path": candidate_path, "backend": cand.get("backend")},
        "summary": {
            "p50": _metric_row(float(ref_latency["p50"]), float(cand_latency["p50"])),
            "p95": _metric_row(float(ref_latency["p95"]), float(cand_latency["p95"])),
            "mean": _metric_row(float(ref_latency["mean"]), float(cand_latency["mean"])),
        },
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate benchmark against a reference benchmark."
    )
    parser.add_argument("--reference", required=True, help="Reference bench JSON (e.g., ORT).")
    parser.add_argument("--candidate", required=True, help="Candidate bench JSON (e.g., TensorRT).")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out", default="artifacts/backend_compare.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare(
        reference_path=args.reference,
        candidate_path=args.candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        out_json=args.out,
    )


if __name__ == "__main__":
    main()

