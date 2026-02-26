from __future__ import annotations

import onnxruntime as ort
import numpy as np

from .base import BackendInfo


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


class OnnxRuntimeBackendSession:
    def __init__(self, model_path: str) -> None:
        providers = _select_providers()
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._model_path = model_path
        self._providers = providers

    def get_input_name(self) -> str:
        return self._sess.get_inputs()[0].name

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]:
        return self._sess.run(None, {self.get_input_name(): x_nchw})

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="onnxruntime",
            model_path=self._model_path,
            extra={"providers": self._providers},
        )
