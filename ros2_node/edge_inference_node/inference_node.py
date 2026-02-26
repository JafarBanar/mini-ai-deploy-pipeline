#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


def _load_backend_factory() -> Callable:
    try:
        from src.backends import create_backend_session

        return create_backend_session
    except Exception:
        repo_src = Path(__file__).resolve().parents[2] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from backends import create_backend_session

        return create_backend_session


create_backend_session = _load_backend_factory()


def _preprocess_cifar10(raw_nhwc: np.ndarray) -> np.ndarray:
    x = raw_nhwc.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    return np.transpose(x, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _postprocess_logits(outputs: list[np.ndarray]) -> np.ndarray:
    return np.argmax(outputs[0], axis=1)


def _image_to_nhwc(msg: "Image", logger: "Node") -> np.ndarray | None:
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    h = int(msg.height)
    w = int(msg.width)

    if msg.encoding == "rgb8":
        expected = h * w * 3
        if raw.size < expected:
            logger.get_logger().warning("RGB image payload smaller than expected, dropping frame.")
            return None
        arr = raw[:expected].reshape(h, w, 3)
    elif msg.encoding == "bgr8":
        expected = h * w * 3
        if raw.size < expected:
            logger.get_logger().warning("BGR image payload smaller than expected, dropping frame.")
            return None
        arr = raw[:expected].reshape(h, w, 3)[:, :, ::-1]
    elif msg.encoding == "mono8":
        expected = h * w
        if raw.size < expected:
            logger.get_logger().warning("Mono image payload smaller than expected, dropping frame.")
            return None
        gray = raw[:expected].reshape(h, w, 1)
        arr = np.repeat(gray, 3, axis=2)
    else:
        logger.get_logger().warning(f"Unsupported encoding '{msg.encoding}', expected rgb8/bgr8/mono8.")
        return None

    crop_h = min(32, h)
    crop_w = min(32, w)
    cropped = np.zeros((32, 32, 3), dtype=np.uint8)
    cropped[:crop_h, :crop_w, :] = arr[:crop_h, :crop_w, :]
    return np.expand_dims(cropped, axis=0)


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
    except Exception as exc:
        raise SystemExit(
            "ROS2 dependencies not installed. Install rclpy + sensor_msgs + std_msgs in a ROS2 environment."
        ) from exc

    class InferenceNode(Node):
        def __init__(self) -> None:
            super().__init__("edge_inference_node")
            self.declare_parameter("backend", "ort")
            self.declare_parameter("model_path", "artifacts/model.onnx")
            self.declare_parameter("device", "auto")
            self.declare_parameter("precision", "fp16")
            self.declare_parameter("batch_size", 1)
            self.declare_parameter("image_topic", "/camera/image")
            self.declare_parameter("metrics_topic", "/inference_metrics")
            self.declare_parameter("qos_depth", 10)

            backend = str(self.get_parameter("backend").value)
            model_path = str(self.get_parameter("model_path").value)
            self._device = str(self.get_parameter("device").value)
            self._precision = str(self.get_parameter("precision").value)
            self._batch_size = max(1, int(self.get_parameter("batch_size").value))
            qos_depth = max(1, int(self.get_parameter("qos_depth").value))

            self._backend = create_backend_session(
                backend=backend,
                model_path=model_path,
                device=self._device,
                precision=self._precision,
            )

            image_topic = str(self.get_parameter("image_topic").value)
            metrics_topic = str(self.get_parameter("metrics_topic").value)
            self._metrics_pub = self.create_publisher(String, metrics_topic, 10)
            self._sub = self.create_subscription(Image, image_topic, self._on_image, qos_depth)
            self._queue_depth = 0
            self._dropped_frames = 0
            self._processed_frames = 0
            self._last_emit_ts: float | None = None

            self.get_logger().info(
                "Started node "
                f"backend={backend} device={self._device} precision={self._precision} "
                f"batch_size={self._batch_size} model={model_path} image_topic={image_topic}"
            )

        def _on_image(self, msg: "Image") -> None:
            self._queue_depth += 1
            try:
                t0 = time.perf_counter()
                arr = _image_to_nhwc(msg, self)
                if arr is None:
                    self._dropped_frames += 1
                    return
                t1 = time.perf_counter()

                if self._batch_size > 1:
                    arr = np.repeat(arr, self._batch_size, axis=0)

                x = _preprocess_cifar10(arr)
                outputs = self._backend.infer(x)
                t2 = time.perf_counter()

                pred = int(_postprocess_logits(outputs)[0])
                t3 = time.perf_counter()

                self._processed_frames += 1
                fps = 0.0
                if self._last_emit_ts is not None:
                    dt = t3 - self._last_emit_ts
                    fps = float(1.0 / dt) if dt > 0 else 0.0
                self._last_emit_ts = t3

                payload = {
                    "backend": self._backend.info().name,
                    "device": self._device,
                    "batch_size": self._batch_size,
                    "input_encoding": msg.encoding,
                    "predicted_class": pred,
                    "preprocess_ms": (t1 - t0) * 1000.0,
                    "infer_ms": (t2 - t1) * 1000.0,
                    "postprocess_ms": (t3 - t2) * 1000.0,
                    "e2e_ms": (t3 - t0) * 1000.0,
                    "fps": fps,
                    "dropped_frames": self._dropped_frames,
                    "queue_depth": max(0, self._queue_depth - 1),
                }
                out = String()
                out.data = json.dumps(payload)
                self._metrics_pub.publish(out)
            finally:
                self._queue_depth = max(0, self._queue_depth - 1)

    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
