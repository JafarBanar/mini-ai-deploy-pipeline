import os

import torch

try:
    from .model import TinyCNN  # type: ignore[attr-defined]
except ImportError:
    from model import TinyCNN


def main() -> None:
    os.makedirs("artifacts", exist_ok=True)

    model = TinyCNN(num_classes=10).eval()
    dummy = torch.randn(1, 3, 32, 32)
    onnx_path = "artifacts/model.onnx"

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,
        external_data=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX to {onnx_path}")


if __name__ == "__main__":
    main()
