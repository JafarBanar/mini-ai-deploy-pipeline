from __future__ import annotations

import numpy as np

from .base import BackendInfo


class TVMBackendSession:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        raise RuntimeError(
            "TVM backend scaffold is present but runtime integration is not implemented in this repo yet."
        )

    def load(self, model_path: str) -> "TVMBackendSession":
        self._model_path = model_path
        raise RuntimeError(
            "TVM backend is not implemented yet. Add TVM runtime module after TensorRT path is stable."
        )

    def warmup(self, x_nchw: np.ndarray, iters: int = 20) -> None:  # noqa: ARG002
        raise NotImplementedError

    def get_input_name(self) -> str:
        return "input"

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def info(self) -> BackendInfo:
        return BackendInfo(name="tvm", model_path=self._model_path, extra={})
