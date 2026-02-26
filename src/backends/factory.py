from __future__ import annotations

from .base import BackendSession
from .onnxruntime_backend import OnnxRuntimeBackendSession
from .tensorrt_backend import TensorRTBackendSession
from .tvm_backend import TVMBackendSession


def create_backend_session(
    backend: str,
    model_path: str,
    device: str = "auto",
    precision: str = "fp16",
) -> BackendSession:
    backend = backend.lower().strip()
    if backend in {"ort", "onnxruntime"}:
        return OnnxRuntimeBackendSession(model_path=model_path, device=device)
    if backend in {"trt", "tensorrt"}:
        fp16 = precision.lower() == "fp16"
        if device not in {"auto", "cuda", "gpu", "jetson"}:
            raise ValueError(f"Unsupported device for TensorRT backend: {device}")
        return TensorRTBackendSession(model_path=model_path, fp16=fp16)
    if backend == "tvm":
        return TVMBackendSession(model_path=model_path)
    raise ValueError(f"Unsupported backend: {backend}")
