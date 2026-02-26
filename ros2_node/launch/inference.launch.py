from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    backend_arg = DeclareLaunchArgument("backend", default_value="ort")
    model_path_arg = DeclareLaunchArgument("model_path", default_value="artifacts/model.onnx")
    image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/camera/image")
    metrics_topic_arg = DeclareLaunchArgument("metrics_topic", default_value="/inference_metrics")

    node = Node(
        package="edge_inference_node",
        executable="edge_inference_node",
        name="edge_inference_node",
        output="screen",
        parameters=[
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "metrics_topic": LaunchConfiguration("metrics_topic"),
            }
        ],
    )

    return LaunchDescription([backend_arg, model_path_arg, image_topic_arg, metrics_topic_arg, node])

