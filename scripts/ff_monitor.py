#!/usr/bin/env python3
"""
Republish the feedforward as a live time series, beside what the crane did.

C3 feedforward isn't observable from /trajectory_controllers/controller_state
(publish_state never fills reference.effort). Reconstructed by interpolating
JTC's reference.time_from_start against /crane/reference, which is latched
and carries effort.

Two JointState topics (PlotJuggler expands by joint name, and index orders of
/joint_states and controller_state disagree):

| topic | position | velocity | effort |
|---|---|---|---|
| /crane/debug/feedforward | reference | qdot_d + effort, whole ff branch | effort alone, the C3 correction |
| /crane/debug/achieved | feedback | feedback | the command actually written |

.../feedforward/<joint>/velocity vs .../achieved/<joint>/velocity is
feedforward vs what the machine did. Should differ by n_d: ff branch is
previewed by dead time, so achieved rate lags commanded on an axis the
inversion is right about.

Read-only: subscribes, publishes, commands nothing.

    ./scripts/ff_monitor.py
    plotjuggler   # Start -> ROS2 Topic Subscriber -> both /crane/debug topics
"""

from __future__ import annotations

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

REFERENCE_TOPIC = "/crane/reference"
STATE_TOPIC = "/trajectory_controllers/controller_state"
FEEDFORWARD_TOPIC = "/crane/debug/feedforward"
ACHIEVED_TOPIC = "/crane/debug/achieved"


def latched(depth: int = 1) -> QoSProfile:
    """Match the planner's own publisher, or the reference never arrives."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class Monitor(Node):
    def __init__(self):
        super().__init__("ff_monitor")
        self.stamps = np.zeros(0)
        self.effort: dict = {}
        self.create_subscription(
            JointTrajectory, REFERENCE_TOPIC, self._reference, latched()
        )
        self.create_subscription(
            JointTrajectoryControllerState, STATE_TOPIC, self._state, 20
        )
        self.feedforward = self.create_publisher(JointState, FEEDFORWARD_TOPIC, 20)
        self.achieved = self.create_publisher(JointState, ACHIEVED_TOPIC, 20)
        self.get_logger().info(
            f"waiting for {REFERENCE_TOPIC}; publishing {FEEDFORWARD_TOPIC} "
            f"and {ACHIEVED_TOPIC}"
        )

    def _reference(self, message: JointTrajectory) -> None:
        """Keep the effort column per joint *name*, not per index."""
        self.stamps = np.array(
            [
                point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                for point in message.points
            ]
        )
        width = len(message.joint_names)
        rows = np.array(
            [
                list(point.effort) if point.effort else [0.0] * width
                for point in message.points
            ]
        )
        self.effort = {name: rows[:, i] for i, name in enumerate(message.joint_names)}
        carried = any(np.any(column != 0.0) for column in self.effort.values())
        self.get_logger().info(
            f"reference: {len(message.points)} points over {self.stamps[-1]:.2f} s, "
            f"feedforward {'present' if carried else 'absent -- static arm'}"
        )

    def _state(self, message: JointTrajectoryControllerState) -> None:
        when = (
            message.reference.time_from_start.sec
            + message.reference.time_from_start.nanosec * 1e-9
        )
        names = list(message.joint_names)

        def ff(name: str) -> float:
            # Zero past end of plan, or for a joint the plan doesn't command.
            if self.stamps.size == 0 or name not in self.effort:
                return 0.0
            return float(np.interp(when, self.stamps, self.effort[name], right=0.0))

        def emit(publisher, position, velocity, effort) -> None:
            out = JointState()
            out.header.stamp = message.header.stamp
            out.name = names
            out.position = [float(v) for v in position]
            out.velocity = [float(v) for v in velocity]
            out.effort = [float(v) for v in effort]
            publisher.publish(out)

        reference = list(message.reference.velocities) or [0.0] * len(names)
        correction = [ff(name) for name in names]
        emit(
            self.feedforward,
            message.reference.positions,
            # Whole ff branch: ff_velocity_scale * qdot_d + effort, scale 1.0.
            [r + c for r, c in zip(reference, correction)],
            correction,
        )
        emit(
            self.achieved,
            message.feedback.positions,
            message.feedback.velocities or [0.0] * len(names),
            message.output.velocities or [0.0] * len(names),
        )


def main() -> None:
    rclpy.init()
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
