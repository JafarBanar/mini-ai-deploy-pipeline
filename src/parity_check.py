import argparse

import numpy as np
import onnxruntime as ort
import torch

try:
    from .model import TinyCNN  # type: ignore[attr-defined]
except ImportError:
    from model import TinyCNN


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def check_parity(
    checkpoint_path: str = "artifacts/model.pt",
    onnx_path: str = "artifacts/model.onnx",
    batch_size: int = 4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 42,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 10))

    model = TinyCNN(num_classes=num_classes).eval()
    model.load_state_dict(ckpt["state_dict"])

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch_size, 3, 32, 32)).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=_select_providers())
    ort_in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {ort_in_name: x})[0]

    abs_diff = np.abs(torch_out - ort_out)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    all_close = bool(np.allclose(torch_out, ort_out, atol=atol, rtol=rtol))

    result = {
        "all_close": all_close,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "atol": atol,
        "rtol": rtol,
    }
    print(result)
    if not all_close:
        raise SystemExit(
            f"Parity failed: max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX outputs.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_parity(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        batch_size=args.batch_size,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
    )
