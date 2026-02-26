from __future__ import annotations

from .base import BackendSession
from .onnxruntime_backend import OnnxRuntimeBackendSession
from .tensorrt_backend import TensorRTBackendSession
from .tvm_backend import TVMBackendSession


def create_backend_session(backend: str, model_path: str) -> BackendSession:
    backend = backend.lower().strip()
    if backend in {"ort", "onnxruntime"}:
        return OnnxRuntimeBackendSession(model_path=model_path)
    if backend in {"trt", "tensorrt"}:
        return TensorRTBackendSession(model_path=model_path)
    if backend == "tvm":
        return TVMBackendSession(model_path=model_path)
    raise ValueError(f"Unsupported backend: {backend}")
