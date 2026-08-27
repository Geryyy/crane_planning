#!/usr/bin/env python3
"""Launch the native crane planner node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    tool = LaunchConfiguration("tool")
    use_sim_time = LaunchConfiguration("use_sim_time")
    parameter_file = LaunchConfiguration("parameter_file")

    default_parameter_file = (
        PathSubstitution(FindPackageShare("crane_planning"))
        / "config"
        / "crane_planner.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tool",
                default_value="pzs100",
                choices=["pzs100", "epsilon7040"],
                description="Mounted crane tool used by the planner",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulated ROS time",
            ),
            DeclareLaunchArgument(
                "parameter_file",
                default_value=default_parameter_file,
                description="Planner parameter YAML file",
            ),
            Node(
                package="crane_planning",
                executable="crane_planner_node",
                name="crane_planner",
                output="screen",
                parameters=[
                    parameter_file,
                    {"tool": tool, "use_sim_time": use_sim_time},
                ],
            ),
        ]
    )
