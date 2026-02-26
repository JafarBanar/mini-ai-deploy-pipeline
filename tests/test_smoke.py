import os

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def test_onnx_load_and_run():
    assert os.path.exists("artifacts/model.onnx"), "Run export first: python src/export_onnx.py"
    sess = ort.InferenceSession(
        "artifacts/model.onnx",
        providers=_select_providers(),
    )
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = sess.run(None, {input_name: x})
    assert y[0].shape == (1, 10)
