from __future__ import annotations

import numpy as np

from .base import BackendInfo


class TensorRTBackendSession:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        raise RuntimeError(
            "TensorRT backend scaffold is present but full runtime binding is not implemented here. "
            "Use a prebuilt .plan engine on Jetson/Orin and integrate TensorRT execution bindings."
        )

    def get_input_name(self) -> str:
        return "input"

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def info(self) -> BackendInfo:
        return BackendInfo(name="tensorrt", model_path=self._model_path, extra={})
