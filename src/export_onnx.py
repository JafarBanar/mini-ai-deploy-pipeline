import argparse
import os

import torch

try:
    from .model import TinyCNN  # type: ignore[attr-defined]
except ImportError:
    from model import TinyCNN


def main(
    checkpoint_path: str = "artifacts/model.pt",
    onnx_path: str = "artifacts/model.onnx",
    opset: int = 17,
) -> None:
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    num_classes = 10
    model = TinyCNN(num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        num_classes = int(ckpt.get("num_classes", num_classes))
        if num_classes != model.fc2.out_features:
            model = TinyCNN(num_classes=num_classes)
            model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Exporting randomly initialized model.")

    model = model.eval()
    dummy = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,
        external_data=False,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX to {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyCNN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="artifacts/model.pt", help="Input checkpoint path.")
    parser.add_argument("--onnx", default="artifacts/model.onnx", help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(checkpoint_path=args.checkpoint, onnx_path=args.onnx, opset=args.opset)
