from __future__ import annotations

import numpy as np

from .base import BackendInfo


class TVMBackendSession:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        raise RuntimeError(
            "TVM backend scaffold is present but runtime integration is not implemented in this repo yet."
        )

    def get_input_name(self) -> str:
        return "input"

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def info(self) -> BackendInfo:
        return BackendInfo(name="tvm", model_path=self._model_path, extra={})
