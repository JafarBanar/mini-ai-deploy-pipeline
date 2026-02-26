import argparse

import numpy as np
import onnxruntime as ort


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def load_session(onnx_path: str) -> ort.InferenceSession:
    providers = _select_providers()
    return ort.InferenceSession(onnx_path, providers=providers)


def run_once(sess: ort.InferenceSession, batch_size: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    input_name = sess.get_inputs()[0].name
    x = rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)
    return sess.run(None, {input_name: x})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ONNX Runtime inference.")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    session = load_session(args.onnx)
    output = run_once(session, batch_size=args.batch_size, seed=args.seed)
    print("Output shape:", output[0].shape)
