from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    backend_arg = DeclareLaunchArgument("backend", default_value="ort")
    model_path_arg = DeclareLaunchArgument("model_path", default_value="artifacts/model.onnx")
    device_arg = DeclareLaunchArgument("device", default_value="auto")
    precision_arg = DeclareLaunchArgument("precision", default_value="fp16")
    batch_size_arg = DeclareLaunchArgument("batch_size", default_value="1")
    image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/camera/image")
    metrics_topic_arg = DeclareLaunchArgument("metrics_topic", default_value="/inference_metrics")
    qos_depth_arg = DeclareLaunchArgument("qos_depth", default_value="10")

    node = Node(
        package="edge_inference_node",
        executable="edge_inference_node",
        name="edge_inference_node",
        output="screen",
        parameters=[
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "precision": LaunchConfiguration("precision"),
                "batch_size": LaunchConfiguration("batch_size"),
                "image_topic": LaunchConfiguration("image_topic"),
                "metrics_topic": LaunchConfiguration("metrics_topic"),
                "qos_depth": LaunchConfiguration("qos_depth"),
            }
        ],
    )

    return LaunchDescription(
        [
            backend_arg,
            model_path_arg,
            device_arg,
            precision_arg,
            batch_size_arg,
            image_topic_arg,
            metrics_topic_arg,
            qos_depth_arg,
            node,
        ]
    )
