import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from backends import create_backend_session


def test_create_ort_backend_and_infer():
    sess = create_backend_session("ort", "artifacts/model.onnx")
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    sess.warmup(x, iters=1)
    y = sess.infer(x)
    assert y[0].shape == (1, 10)


def test_unsupported_backend_raises():
    with pytest.raises(ValueError):
        create_backend_session("bad_backend", "artifacts/model.onnx")
