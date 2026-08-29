"""
Draw a plan, so an operator can see what the planner decided.

Four things, and each answers a question the numbers do not:

* **the path** the tool takes, as one line;
* **the tool itself**, swept along it -- the segment from the tip pivot to the
  tool centre at samples down the trajectory. That segment is the pendulum, and
  its lean is the sway the OCP *planned*, so a plan that swings is a plan that
  looks like it swings;
* **the goal**, where the request asked for the tool and which way round;
* **the scene the planner actually checked against**, which is not the scene
  anyone published. The reserved `truck` primitive has become a bed and six
  runges by this point, and what is in the gripper has been inserted as a body
  of its own. A refusal is very hard to read without them and obvious with them.

Markers are cheap and this is a plan, not a stream: the whole set is rebuilt and
republished per request, led by a `DELETEALL` so nothing from the previous plan
survives into this one.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from crane_model import Frame
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

#: Where the tool is drawn along the trajectory. Enough to read the swing,
#: few enough that the machine is not hidden behind its own preview.
SWEEP_SAMPLES = 12

PATH_COLOR = ColorRGBA(r=1.0, g=0.67, b=0.0, a=1.0)
TOOL_COLOR = ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.65)
GOAL_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.8, a=0.9)
#: Structural bodies are the vehicle and cannot move; perceived ones came from
#: the world model; the payload is what the gripper is holding. Three colours,
#: because "why was this refused" is usually answered by which kind it hit.
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
    """Return a `DELETEALL`, so the previous plan does not linger under this one."""
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.action = Marker.DELETEALL
    return MarkerArray(markers=[marker])


def scene(primitives: list, frame: str, stamp) -> list:
    """Draw the bodies the plan was checked against, coloured by what they are."""
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
    """Draw the path the tool takes, and the tool and its load along it."""
    path = _marker(frame, stamp, "path", 0, Marker.LINE_STRIP)
    path.scale.x = 0.04
    path.color = PATH_COLOR
    path.points = [_point(position) for position in plan_.tcp]

    # The pendulum, sampled. `Frame.TILT` is the hinge the tool hangs from, so
    # this segment leans by exactly the sway the timing OCP solved for.
    tool = _marker(frame, stamp, "tool", 0, Marker.LINE_LIST)
    tool.scale.x = 0.03
    tool.color = TOOL_COLOR
    step = max(1, len(plan_.q) // SWEEP_SAMPLES)
    for q in plan_.q[::step]:
        hinge = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TILT)
        tip = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        tool.points.append(_point(hinge.position_m))
        tool.points.append(_point(tip.position_m))

    # And what it is carrying, drawn where it will actually be. The payload
    # travels with the tool, so a body drawn only at the start says nothing
    # about the half of the path where it might hit something.
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
