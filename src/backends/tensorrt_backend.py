from __future__ import annotations

from typing import Any

import onnxruntime as ort
import numpy as np

from .base import BackendInfo


class TensorRTBackendSession:
    def __init__(self, model_path: str, fp16: bool = True) -> None:
        self._model_path = model_path
        self._fp16 = fp16
        self._sess: ort.InferenceSession | None = None
        self._providers: list[Any] = []
        self.load(model_path)

    def _select_tensorrt_providers(self) -> list[Any]:
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "TensorrtExecutionProvider is not available in this onnxruntime build. "
                "Use Jetson/Orin ORT with TensorRT EP, or benchmark TRT via deploy/jetson/benchmark_trtexec.sh."
            )

        trt_options: dict[str, Any] = {"trt_fp16_enable": self._fp16}
        providers: list[Any] = [("TensorrtExecutionProvider", trt_options)]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers

    def load(self, model_path: str) -> "TensorRTBackendSession":
        self._model_path = model_path
        self._providers = self._select_tensorrt_providers()
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
            name="tensorrt",
            model_path=self._model_path,
            extra={"providers": self._providers, "fp16": self._fp16},
        )
