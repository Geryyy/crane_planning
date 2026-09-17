"""
Draw a plan, so an operator sees what the planner decided.

Path, tool swept along it (pivot-to-centre segment, so a swinging plan visibly
swings), goal, and the scene actually checked (expanded truck bodies, payload)
rather than what was published. Whole set rebuilt per request behind a
`DELETEALL` -- markers are cheap, this is a plan not a stream.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from crane_model import Frame
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

#: How many samples of the tool are drawn along the trajectory.
SWEEP_SAMPLES = 12

PATH_COLOR = ColorRGBA(r=1.0, g=0.67, b=0.0, a=1.0)
TOOL_COLOR = ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.65)
GOAL_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.8, a=0.9)
#: Structural = vehicle (fixed), perceived = world model, payload = gripper contents --
#: three colours so a refusal shows which kind was hit.
STRUCTURAL_COLOR = ColorRGBA(r=0.55, g=0.55, b=0.6, a=0.45)
PERCEIVED_COLOR = ColorRGBA(r=1.0, g=0.35, b=0.1, a=0.45)
PAYLOAD_COLOR = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.6)

SHAPES = {"box": Marker.CUBE, "cylinder": Marker.CYLINDER, "sphere": Marker.SPHERE}


def _marker(frame, stamp, namespace: str, index: int, kind: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = index
    marker.type = kind
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.frame_locked = True
    return marker


def _point(position) -> Point:
    return Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))


def _quaternion(rotation: np.ndarray) -> tuple:
    """`(x, y, z, w)` of a rotation matrix, scalar-last as ROS writes it."""
    return tuple(float(value) for value in pin.Quaternion(rotation).coeffs())


def clear(frame: str, stamp) -> MarkerArray:
    """Build a `DELETEALL`, so the previous plan doesn't linger."""
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.action = Marker.DELETEALL
    return MarkerArray(markers=[marker])


def scene(primitives: list, frame: str, stamp) -> list:
    """Draw the bodies the plan was checked against, coloured by kind."""
    markers = []
    for index, primitive in enumerate(primitives):
        kind = SHAPES.get(primitive.shape)
        if kind is None:
            continue
        marker = _marker(frame, stamp, "scene", index, kind)
        pose = primitive.pose_in_mounting_base
        marker.pose.position = _point(pose.translation)
        quaternion = _quaternion(pose.rotation)
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = quaternion
        extent = np.asarray(primitive.dimensions_m, dtype=float)
        marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in extent)
        if primitive.id == "payload":
            marker.color = PAYLOAD_COLOR
        elif primitive.structural:
            marker.color = STRUCTURAL_COLOR
        else:
            marker.color = PERCEIVED_COLOR
        marker.text = primitive.id
        markers.append(marker)
    return markers


def goal(position_m, yaw: float, frame: str, stamp) -> list:
    """Draw where the tool was asked to end up, and which way round."""
    marker = _marker(frame, stamp, "goal", 0, Marker.ARROW)
    marker.points = [
        _point(position_m),
        _point(
            np.asarray(position_m, dtype=float)
            + 0.6 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ),
    ]
    marker.scale.x, marker.scale.y, marker.scale.z = 0.05, 0.12, 0.12
    marker.color = GOAL_COLOR
    return [marker]


def plan(model, plan_, frame: str, stamp, payload_shape=None) -> list:
    """Draw the path the tool takes, plus tool and load along it."""
    path = _marker(frame, stamp, "path", 0, Marker.LINE_STRIP)
    path.scale.x = 0.04
    path.color = PATH_COLOR
    path.points = [_point(position) for position in plan_.tcp]

    # Pendulum, sampled: Frame.TILT is the hinge the tool hangs from, so this segment
    # leans by exactly the sway the OCP solved for.
    tool = _marker(frame, stamp, "tool", 0, Marker.LINE_LIST)
    tool.scale.x = 0.03
    tool.color = TOOL_COLOR
    step = max(1, len(plan_.q) // SWEEP_SAMPLES)
    for q in plan_.q[::step]:
        hinge = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TILT)
        tip = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        tool.points.append(_point(hinge.position_m))
        tool.points.append(_point(tip.position_m))

    # What it carries, drawn along the path -- a body drawn only at the start says
    # nothing about where it might hit something later.
    load = []
    if payload_shape is not None:
        from .planner import payload_primitive

        for index, q in enumerate(plan_.q[::step]):
            carried = payload_primitive(model, q, payload_shape)
            if carried is None:
                break
            load.extend(scene([carried], frame, stamp))
            load[-1].ns = "load"
            load[-1].id = index
            load[-1].color = TOOL_COLOR
    return [path, tool, *load]
