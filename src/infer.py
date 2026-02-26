import argparse

import numpy as np

try:
    from .backends import create_backend_session  # type: ignore[attr-defined]
except ImportError:
    from backends import create_backend_session


def run_once(
    model_path: str,
    backend: str,
    batch_size: int = 1,
    seed: int = 42,
    device: str = "auto",
    precision: str = "fp16",
):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)
    session = create_backend_session(
        backend=backend,
        model_path=model_path,
        device=device,
        precision=precision,
    )
    output = session.infer(x)
    info = session.info()
    print(f"Backend: {info.name}")
    print("Output shape:", output[0].shape)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one inference on a selected backend.")
    parser.add_argument("--backend", choices=["ort", "tensorrt", "tvm"], default="ort")
    parser.add_argument("--model", default="artifacts/model.onnx")
    parser.add_argument("--onnx", default=None, help="Compatibility alias for --model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.onnx if args.onnx else args.model
    run_once(
        model_path=model_path,
        backend=args.backend,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
    )
