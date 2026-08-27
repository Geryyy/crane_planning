#!/usr/bin/env python3
"""Call the crane planner for one A-to-B move and visualize its answer.

The planner remains the single owner of IK, collision checking and timing.  This
script is only a small developer-facing client for ``/crane/plan_motion``.  A
planner node must already be running and have received the robot description,
joint state and (when collision checking is enabled) collision scene.

Example::

    ros2 run crane_planning plan_a2b --goal 1.5 0.0 1.2 --yaw 0.0 --show
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from crane_msgs.srv import PlanMotion
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


DEFAULT_OUTPUT = Path("plan_a2b.png")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan one Cartesian crane A-to-B move and plot the returned trajectory."
    )
    parser.add_argument(
        "--goal", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
        help="TCP goal position in K0_mounting_base, metres",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="TCP goal yaw, radians")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-collision-check", action="store_true",
        help="ask the planner to omit collision checking (the default checks it)",
    )
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument(
        "--payload-dimensions", nargs=3, type=float, default=(0.1, 0.1, 0.1),
        metavar=("X", "Y", "Z"), help="carried box dimensions in K8, metres",
    )
    parser.add_argument(
        "--payload-com", nargs=3, type=float, default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"), help="payload COM in K8, metres",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="PNG plot path; CSV is written beside it",
    )
    parser.add_argument("--show", action="store_true", help="also open the Matplotlib window")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PlannerClient(Node):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__("plan_a2b_visualizer")
        self.client = self.create_client(PlanMotion, "/crane/plan_motion")
        request = PlanMotion.Request()
        request.goal = PoseStamped()
        request.goal.header.frame_id = "K0_mounting_base"
        request.goal.pose.position.x, request.goal.pose.position.y, request.goal.pose.position.z = options.goal
        qx, qy, qz, qw = yaw_quaternion(options.yaw)
        request.goal.pose.orientation.x = qx
        request.goal.pose.orientation.y = qy
        request.goal.pose.orientation.z = qz
        request.goal.pose.orientation.w = qw
        request.speed_scale = options.speed_scale
        request.avoid_collisions = not options.no_collision_check
        request.payload.mass = options.payload_mass
        request.payload.com.x, request.payload.com.y, request.payload.com.z = options.payload_com
        if options.payload_mass > 0.0:
            request.payload.shape = request.payload.SHAPE_BOX
            request.payload.dimensions.x, request.payload.dimensions.y, request.payload.dimensions.z = options.payload_dimensions
        else:
            request.payload.shape = request.payload.SHAPE_NONE
        self.request = request


def write_csv(path: Path, response: PlanMotion.Response) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = list(response.trajectory.joint_names)
        writer.writerow(["time_s", *[f"q_{name}" for name in names],
                         *[f"dq_{name}" for name in names], "tcp_x", "tcp_y", "tcp_z"])
        for index, point in enumerate(response.trajectory.points):
            tcp = response.tcp_path[index].pose.position if index < len(response.tcp_path) else None
            xyz = (tcp.x, tcp.y, tcp.z) if tcp is not None else (float("nan"),) * 3
            seconds = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            writer.writerow([seconds, *point.positions, *point.velocities, *xyz])


def plot(path: Path, response: PlanMotion.Response, show: bool) -> None:
    trajectory = response.trajectory
    time = np.array([
        point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        for point in trajectory.points
    ])
    positions = np.asarray([point.positions for point in trajectory.points])
    velocities = np.asarray([point.velocities for point in trajectory.points])
    tcp = np.asarray([
        (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        for pose in response.tcp_path
    ])
    names = list(trajectory.joint_names)
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for index, name in enumerate(names):
        axes[0].plot(time, positions[:, index], label=name)
        axes[1].plot(time, velocities[:, index], label=name)
    axes[0].set_ylabel("joint position [rad / m]")
    axes[1].set_ylabel("joint velocity [rad/s / m/s]")
    axes[0].set_title("Crane planner output")
    axes[0].legend(ncol=3)
    axes[1].legend(ncol=3)
    if len(tcp):
        for index, label in enumerate(("x", "y", "z")):
            axes[2].plot(time[:len(tcp)], tcp[:, index], label=label)
        axes[2].legend()
    axes[2].set_ylabel("TCP position [m]")
    axes[2].set_xlabel("time [s]")
    figure.suptitle(response.message, fontsize=8, wrap=True)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    options = arguments()
    if options.speed_scale <= 0.0 or options.speed_scale > 1.0:
        raise SystemExit("--speed-scale must be in (0, 1]")
    rclpy.init()
    node = PlannerClient(options)
    try:
        if not node.client.wait_for_service(timeout_sec=options.timeout):
            raise RuntimeError("/crane/plan_motion did not become available")
        future = node.client.call_async(node.request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=options.timeout)
        if not future.done():
            raise RuntimeError("planner service call timed out")
        response = future.result()
        if response is None:
            raise RuntimeError("planner service returned no response")
        if not response.success:
            raise RuntimeError(response.message)
        output = options.output.resolve()
        write_csv(output.with_suffix(".csv"), response)
        plot(output, response, options.show)
        print(response.message)
        print(f"plot: {output}\nCSV:  {output.with_suffix('.csv')}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
