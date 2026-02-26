import numpy as np
import onnxruntime as ort


def _select_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected if selected else available


def load_session(onnx_path: str) -> ort.InferenceSession:
    providers = _select_providers()
    return ort.InferenceSession(onnx_path, providers=providers)


def run_once(sess: ort.InferenceSession, batch_size: int = 1):
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
    return sess.run(None, {input_name: x})


if __name__ == "__main__":
    session = load_session("artifacts/model.onnx")
    output = run_once(session, batch_size=1)
    print("Output shape:", output[0].shape)
