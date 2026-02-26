from setuptools import find_packages, setup

package_name = "edge_inference_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/inference.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jafar Banar",
    maintainer_email="jafar@example.com",
    description="ROS2 edge inference node for model deployment experiments.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "edge_inference_node = edge_inference_node.inference_node:main",
        ],
    },
)

