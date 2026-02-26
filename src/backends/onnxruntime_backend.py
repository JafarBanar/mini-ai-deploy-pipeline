from __future__ import annotations

import onnxruntime as ort
import numpy as np

from .base import BackendInfo


def _select_providers(device: str = "auto") -> list[str]:
    available = ort.get_available_providers()
    dev = device.lower().strip()
    if dev == "cpu":
        preferred = ["CPUExecutionProvider"]
    elif dev in {"cuda", "gpu", "jetson"}:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


class OnnxRuntimeBackendSession:
    def __init__(self, model_path: str, device: str = "auto") -> None:
        self._model_path = model_path
        self._device = device
        self._providers = _select_providers(device=device)
        self._sess: ort.InferenceSession | None = None
        self.load(model_path)

    def load(self, model_path: str) -> "OnnxRuntimeBackendSession":
        self._model_path = model_path
        self._sess = ort.InferenceSession(model_path, providers=self._providers)
        return self

    def warmup(self, x_nchw: np.ndarray, iters: int = 20) -> None:
        for _ in range(iters):
            _ = self.infer(x_nchw)

    def get_input_name(self) -> str:
        if self._sess is None:
            raise RuntimeError("Backend session is not loaded.")
        return self._sess.get_inputs()[0].name

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]:
        if self._sess is None:
            raise RuntimeError("Backend session is not loaded.")
        return self._sess.run(None, {self.get_input_name(): x_nchw})

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="onnxruntime",
            model_path=self._model_path,
            extra={"providers": self._providers, "device": self._device},
        )
