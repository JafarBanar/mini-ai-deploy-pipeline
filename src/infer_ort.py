import argparse

try:
    from .infer import run_once  # type: ignore[attr-defined]
except ImportError:
    from infer import run_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy ORT one-shot inference entrypoint.")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_once(model_path=args.onnx, backend="ort", batch_size=args.batch_size, seed=args.seed)
