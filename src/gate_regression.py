import argparse
import json
import os


def gate_p95(
    current_path: str = "artifacts/bench.json",
    baseline_path: str = "artifacts/baseline.json",
    threshold: float = 1.05,
) -> bool:
    with open(current_path, "r", encoding="utf-8") as f:
        cur = json.load(f)
    cur_p95 = float(cur["latency_ms"]["p95"])

    if not os.path.exists(baseline_path):
        print("No baseline.json found. Skipping regression gate.")
        print(f"Current p95(ms): {cur_p95:.6f}")
        return True

    with open(baseline_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    base_p95 = float(base["latency_ms"]["p95"])

    allowed = base_p95 * threshold
    if cur_p95 > allowed:
        raise SystemExit(
            f"Regression: p95 {cur_p95:.6f}ms > baseline {base_p95:.6f}ms * {threshold}"
        )

    print(f"OK: p95 {cur_p95:.6f}ms <= baseline {base_p95:.6f}ms * {threshold}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if benchmark p95 regresses.")
    parser.add_argument("--current", default="artifacts/bench.json")
    parser.add_argument("--baseline", default="artifacts/baseline.json")
    parser.add_argument("--threshold", type=float, default=1.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gate_p95(current_path=args.current, baseline_path=args.baseline, threshold=args.threshold)
