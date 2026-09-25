"""Parameterised launcher for the physics_amm node (D5.1 guideline #10)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    # Only override python_executable when the argument was actually given,
    # so a value from the params file is never clobbered by the default.
    parameters = [LaunchConfiguration("params_file")]
    python_executable = LaunchConfiguration("python_executable").perform(context)
    if python_executable:
        parameters.append(
            {"python_executable": os.path.expanduser(python_executable)})
    return [
        Node(
            package="physics_amm",
            executable="physics_amm_node",
            namespace=LaunchConfiguration("namespace"),
            name="physics_amm",
            parameters=parameters,
            output="screen",
        ),
    ]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("physics_amm"), "config", "params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="YAML file with physics_amm node parameters"),
        DeclareLaunchArgument(
            "namespace", default_value="physics_amm",
            description="Namespace for the node (package name per D5.1)"),
        DeclareLaunchArgument(
            "python_executable", default_value="",
            description="Interpreter for the ANSR subprocess (point this at "
                        "the venv holding the ANSR requirements, e.g. "
                        "~/venvs/ansr/bin/python)"),
        OpaqueFunction(function=_launch_node),
    ])
