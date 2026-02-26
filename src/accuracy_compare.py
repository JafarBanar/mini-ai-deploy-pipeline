import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

try:
    from .model import TinyCNN  # type: ignore[attr-defined]
    from .train import build_synthetic_dataset  # type: ignore[attr-defined]
except ImportError:
    from model import TinyCNN
    from train import build_synthetic_dataset


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def compare_accuracy(
    checkpoint_path: str = "artifacts/model.pt",
    onnx_path: str = "artifacts/model.onnx",
    val_samples: int = 1024,
    batch_size: int = 128,
    seed: int = 43,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 10))

    model = TinyCNN(num_classes=num_classes).eval()
    model.load_state_dict(ckpt["state_dict"])

    val_ds = build_synthetic_dataset(num_samples=val_samples, num_classes=num_classes, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    sess = ort.InferenceSession(onnx_path, providers=_select_providers())
    input_name = sess.get_inputs()[0].name

    pytorch_correct = 0
    ort_correct = 0
    pred_match = 0
    total = 0

    with torch.no_grad():
        for x_t, y_t in val_loader:
            logits_t = model(x_t)
            pred_t = torch.argmax(logits_t, dim=1).numpy()

            x_np = x_t.numpy().astype(np.float32)
            logits_o = sess.run(None, {input_name: x_np})[0]
            pred_o = np.argmax(logits_o, axis=1)

            y = y_t.numpy()
            pytorch_correct += int((pred_t == y).sum())
            ort_correct += int((pred_o == y).sum())
            pred_match += int((pred_t == pred_o).sum())
            total += y.shape[0]

    result = {
        "total_samples": total,
        "pytorch_acc": pytorch_correct / total,
        "onnx_acc": ort_correct / total,
        "pred_match_rate": pred_match / total,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX accuracy.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_accuracy(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
