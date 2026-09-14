#!/usr/bin/env python3
r"""
Write what `/robot_description` carries, byte for byte, for the exporter to bake.

    ./scripts/dump_robot_description.py live.urdf
    ./scripts/export_timing_ocp.py --description live.urdf --compile-only

Bytes are the point. `crane_planning.ocp.cache_key` hashes the description
string, the node hashes the one the topic handed it; same string, or the
prebaked solver is missed and the node quits rather than plan for another
machine. `ros2 topic echo --field data` is unusable: it appends its own
`\\n---\\n` record separator.

Which description a deployment publishes there is a launch decision -- a sim may
publish the `rviz` one -- so dumping is also how you find what you plan for.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

TOPIC = "/robot_description"


def dump(output: Path, topic: str, timeout: float) -> int:
    """Return the description the topic latches, or nothing if it is silent."""
    rclpy.init()
    node = Node("dump_robot_description")
    received = []
    # Transient-local: publisher latched it once at start-up, a volatile
    # subscription joining later hears nothing.
    node.create_subscription(
        String,
        topic,
        lambda message: received.append(message.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    deadline = node.get_clock().now().nanoseconds * 1e-9 + timeout
    while not received and node.get_clock().now().nanoseconds * 1e-9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if not received:
        print(f"nothing published on {topic} within {timeout} s", file=sys.stderr)
        return 1
    output.write_text(received[0])
    # sha1 is what `cache_key` hashes and what the node reports, so this matches
    # a dump against the solver a run actually loaded.
    digest = hashlib.sha1(received[0].encode()).hexdigest()
    print(f"wrote {output}: {len(received[0])} bytes, sha1 {digest[:10]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", type=Path, help="file to write the description to")
    parser.add_argument(
        "--topic",
        default=TOPIC,
        help=(
            "which description to dump. A sim publishes several -- the planner "
            "reads this one, Gazebo's physics runs on `robot_description_full`, "
            "and they are not the same machine"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="seconds to wait for the topic"
    )
    arguments = parser.parse_args()
    return dump(arguments.output, arguments.topic, arguments.timeout)


if __name__ == "__main__":
    sys.exit(main())
