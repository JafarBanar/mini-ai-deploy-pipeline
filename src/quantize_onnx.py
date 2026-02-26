import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(
    in_onnx: str = "artifacts/model.onnx",
    out_onnx: str = "artifacts/model.int8.onnx",
    per_channel: bool = True,
    reduce_range: bool = False,
) -> str:
    os.makedirs(os.path.dirname(out_onnx) or ".", exist_ok=True)
    quantize_dynamic(
        model_input=in_onnx,
        model_output=out_onnx,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {out_onnx}")
    return out_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dynamic INT8 ONNX model.")
    parser.add_argument("--in-onnx", default="artifacts/model.onnx")
    parser.add_argument("--out-onnx", default="artifacts/model.int8.onnx")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize(
        in_onnx=args.in_onnx,
        out_onnx=args.out_onnx,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )
