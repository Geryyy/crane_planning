#!/usr/bin/env python3
r"""
Time one `CalcMovement` server over a fixed request set. Server-agnostic.

Stage 0 of the C4 continuity work, and the acceptance measure itself: both
`a2b_ilqr_server` and `crane_planning` implement this service, so the same file
of requests through both is the apples-to-apples comparison. Requests come from
`bench_plan.py --emit-requests`, never from here -- a client that computes its
own goal is a client that hands each server a different one.

    ./scripts/bench_plan.py --emit-requests /tmp/a2b_goals.json
    ./scripts/bench_calc_movement.py --requests /tmp/a2b_goals.json --label ilqr
    ./scripts/bench_calc_movement.py --requests /tmp/a2b_goals.json \\
        --service /crane/a2b_movement --label crane_planning

Both servers claim `a2b_movement`, so run them one at a time or namespace one
of them; `--service` is what picks between them.

Wall time here is the served request end to end -- transport, planning and the
answer -- measured under whatever else is running. Say what was running when
quoting a number: `ocp.py` records the same solve at 0.53 s idle and 4.97 s
inside a Gazebo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from timber_crane_planning_interfaces.srv import CalcMovement

SERVICE = "a2b_movement"

#: `CraneNodeBase` builds its parameter base **lazily, in the `/joint_states`
#: callback** (`crane_node_base.cpp:115`), so an `a2b_ilqr_server` that has
#: never seen one segfaults on the first request -- a null
#: `crane_parameter_base_` dereferenced at `a2b_server_base.cpp:154`. Publishing
#: the topic is what makes the reference server answerable outside a full
#: bringup. The names are `mp_parameter_pzs100.yaml`'s `joints` plus its one
#: `aux_joints` entry; `crane_planning` reads `/joint_states` by name too and
#: ignores it entirely when a request carries `q0`.
JOINT_NAMES = (
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
    "pincer_cylinder_piston_in_barrel_linear_joint",
)


def build(move: dict, slow_down: float, avoid_collisions: bool) -> CalcMovement.Request:
    """One request. Every field the `.srv` leaves defaulted stays defaulted."""
    request = CalcMovement.Request()
    request.y_n.x, request.y_n.y, request.y_n.z = (float(v) for v in move["y_n"])
    request.phi_tool_n = float(move["phi_tool_n"])
    request.q0 = [float(value) for value in move["q0"]]
    request.slow_down = float(slow_down)
    request.check_log_collision = avoid_collisions
    request.check_gripper_collision = avoid_collisions
    # The path topic is a side effect with its own cost, and neither server is
    # being measured on how fast it draws.
    request.publish_path = False
    return request


def answer_shape(response) -> dict:
    """
    Report what came back beyond success: duration, sample count, which arrays.

    The acceleration column is the continuity evidence this whole task is
    about -- `a2b_server_base.cpp:615` sends velocities and no accelerations, so
    the JTC interpolates cubic and what executes is `C1`. Reported per server so
    a claim about that is measured rather than read off a source file.
    """
    points = response.trajectory.points
    if not points:
        return {"points": 0, "duration": 0.0, "velocities": 0, "accelerations": 0}
    last = points[-1].time_from_start
    return {
        "points": len(points),
        "duration": last.sec + last.nanosec * 1e-9,
        "velocities": len(points[0].velocities),
        "accelerations": len(points[0].accelerations),
    }


def joint_states(node: Node, q0) -> None:
    """Publish one measured state at `q0`, latched, and keep it up at 20 Hz."""
    publisher = node.create_publisher(JointState, "joint_states", 10)
    positions = [float(value) for value in q0] + [0.0] * (len(JOINT_NAMES) - len(q0))

    def tick():
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = positions
        message.velocity = [0.0] * len(JOINT_NAMES)
        message.effort = [0.0] * len(JOINT_NAMES)
        publisher.publish(message)

    node.create_timer(0.05, tick)


def collision_scene(node: Node) -> None:
    """
    Publish an empty, always-fresh collision scene on `/crane/collision_scene`.

    `crane_planning` refuses a collision-avoiding request when it has never seen
    one, and refuses again once the last is older than `max_scene_age`. Empty is
    what `bench_plan.py` plans against offline (`scene=[]`,
    `avoid_collisions=True`), so this is what makes the two numbers comparable.
    `a2b_ilqr_server` does not subscribe to it, so publishing it costs it
    nothing.
    """
    from crane_msgs.msg import CollisionScene
    from rclpy.qos import DurabilityPolicy, QoSProfile

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(CollisionScene, "/crane/collision_scene", latched)

    def tick():
        message = CollisionScene()
        message.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(message)

    node.create_timer(1.0, tick)


def call(node: Node, client, request, timeout: float) -> tuple:
    """`(elapsed_s, response or None)`, timing the served request end to end."""
    started = time.perf_counter()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    elapsed = time.perf_counter() - started
    return elapsed, (future.result() if future.done() else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument("--label", default=None, help="what to call this server")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--slow-down", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-collisions", action="store_true")
    parser.add_argument("--no-joint-states", action="store_true")
    parser.add_argument(
        "--settle", type=float, default=3.0, help="seconds of joint_states first"
    )
    parser.add_argument("--csv", type=Path, default=None)
    options = parser.parse_args()

    moves = json.loads(options.requests.read_text())["moves"]
    label = options.label or options.service

    rclpy.init()
    node = rclpy.create_node("bench_calc_movement")
    client = node.create_client(CalcMovement, options.service)
    if not client.wait_for_service(timeout_sec=30.0):
        node.get_logger().error(f"no server on {options.service} after 30 s")
        rclpy.shutdown()
        return 1
    if not options.no_joint_states:
        joint_states(node, moves[0]["q0"])
        collision_scene(node)
        # Spun, not slept: the server needs the topic *before* the first
        # request, and the timer only fires while this node is spinning.
        end = time.perf_counter() + options.settle
        while time.perf_counter() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    rows, samples = [], []
    print(f"{label}: {len(moves)} moves x {options.repeats}, on {options.service}")
    for move in moves:
        request = build(move, options.slow_down, not options.no_collisions)
        elapsed_s, shapes, failures = [], [], 0
        for _ in range(options.repeats):
            elapsed, response = call(node, client, request, options.timeout)
            if response is None or not response.success:
                failures += 1
                continue
            elapsed_s.append(elapsed)
            shapes.append(answer_shape(response))
            rows.append((label, move["name"], elapsed, shapes[-1]["duration"]))
        if not elapsed_s:
            print(
                f"  {move['name']:<12} refused or timed out {failures}/{options.repeats}"
            )
            continue
        samples.extend(elapsed_s)
        shape = shapes[0]
        print(
            f"  {move['name']:<12} {np.median(elapsed_s):6.3f} s median  "
            f"{np.percentile(elapsed_s, 90):6.3f} s p90  "
            f"{shape['duration']:5.2f} s trajectory, {shape['points']:4d} points, "
            f"vel {shape['velocities']} acc {shape['accelerations']}"
            + (f", {failures} refused" if failures else "")
        )

    if samples:
        print(
            f"{label}: {np.median(samples):.3f} s median, "
            f"{np.percentile(samples, 90):.3f} s p90, "
            f"{np.min(samples):.3f}-{np.max(samples):.3f} s over {len(samples)} calls"
        )
    if options.csv is not None and rows:
        with options.csv.open("a") as handle:
            if handle.tell() == 0:
                handle.write("server,move,wall_s,trajectory_s\n")
            for server, move_name, elapsed, duration in rows:
                handle.write(f"{server},{move_name},{elapsed:.6f},{duration:.6f}\n")
        print(f"appended {len(rows)} rows to {options.csv}")

    node.destroy_node()
    rclpy.shutdown()
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
