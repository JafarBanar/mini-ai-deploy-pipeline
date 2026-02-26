# ROS2 Node

This directory is a ROS2 Python package named `edge_inference_node`.

## Build and run

From a ROS2 workspace root:

```bash
mkdir -p src
rsync -a <repo>/ros2_node/ src/edge_inference_node/
colcon build --packages-select edge_inference_node
source install/setup.bash
ros2 launch edge_inference_node inference.launch.py backend:=ort model_path:=<abs-path-to-model.onnx>
```

Metrics are published as JSON strings on `/inference_metrics`.

Useful launch params:

- `backend` (`ort|tensorrt|tvm`)
- `model_path`
- `device` (`auto|cuda|gpu|jetson`)
- `precision` (`fp16|fp32`)
- `batch_size`
- `image_topic`
- `metrics_topic`
