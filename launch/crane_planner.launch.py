#!/usr/bin/env python3
"""Launch the native crane planner node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    parameter_file = LaunchConfiguration("parameter_file")
    joint_states_topic = LaunchConfiguration("joint_states_topic")

    default_parameter_file = (
        PathSubstitution(FindPackageShare("crane_planning"))
        / "config"
        / "crane_planner.yaml"
    )

    return LaunchDescription(
        [
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
            DeclareLaunchArgument(
                "joint_states_topic",
                default_value="/joint_states",
                description="Canonical joint-state input.",
            ),
            Node(
                package="crane_planning",
                executable="crane_planner_node",
                name="crane_planner",
                output="screen",
                parameters=[
                    parameter_file,
                    {"use_sim_time": use_sim_time},
                ],
                remappings=[("/joint_states", joint_states_topic)],
            ),
        ]
    )
