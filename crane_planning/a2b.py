"""
Serve retained a2b_movement (CalcMovement) over the native planner: thin adapter.

Payload always goes to the legacy side as a cylinder (its LogShape); y_n targets the
tip K5, not the tool K8_tool_center_point, so tip_to_tcp_offset translates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from crane_model import Payload

from .planner import PASSIVE_INDICES, PLANNED_INDICES, PlanningError, Start

#: Name retained callers resolve exactly; renaming retires the service instead of keeping compat.
A2B_MOVEMENT_SERVICE = "/a2b_movement"

#: How far apart two float32 shape fields may be and still count as one body.
SHAPE_TOLERANCE = 1.0e-6


@dataclass
class A2bGoal:
    position_m: np.ndarray
    yaw: float
    payload: Payload | None
    payload_shape: tuple | None
    avoid_collisions: bool
    speed_scale: float
    start: Start | None


def _finite_point(point) -> bool:
    return bool(np.all(np.isfinite([point.x, point.y, point.z])))


def _same_length(left: float, right: float) -> bool:
    if not (np.isfinite(left) and np.isfinite(right)):
        return False
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= SHAPE_TOLERANCE * scale


def translate_start(request) -> Start | None:
    """q0/q0_dot to Start; all-zero q0 = measured start, so nonzero q0_dot then is refused."""
    q0 = np.asarray(request.q0, dtype=float)
    dq0 = np.asarray(request.q0_dot, dtype=float)
    if np.all(q0 == 0.0):
        if np.any(dq0 != 0.0):
            raise PlanningError(
                "`q0_dot` is non-zero while `q0` is the all-zero .srv default. That default "
                "means use the measured start, and combining rates supplied for one state "
                "with positions measured from another would not describe one initial condition"
            )
        return None
    if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(dq0))):
        raise PlanningError(
            "`q0`/`q0_dot` do not contain sixteen finite numbers, so they are not an initial "
            "condition this planner can use"
        )
    return Start(
        q=q0,
        dq_a=dq0[list(PLANNED_INDICES)],
        dq_u=dq0[list(PASSIVE_INDICES)],
        passive_measured=True,
    )


def translate_payload(request):
    """Log fields -> payload, or None if carries_log is false (approach leg sends a shape too)."""
    if not request.carries_log:
        return None, None

    # log_carrying is the inertia shape, coll_shape the collision one; native uses one for both.
    for field in ("length", "radius_top", "radius_bottom"):
        if not _same_length(
            getattr(request.log_carrying, field), getattr(request.coll_shape, field)
        ):
            raise PlanningError(
                "`log_carrying` and `coll_shape` describe two different bodies, and this "
                "planner carries one shape for the mass and for the collision check alike. "
                "Send the same cylinder in both, or plan the two separately"
            )

    radius = max(request.log_carrying.radius_top, request.log_carrying.radius_bottom)
    length = request.log_carrying.length
    if not (
        np.isfinite(length) and length > 0.0 and np.isfinite(radius) and radius > 0.0
    ):
        raise PlanningError(
            f"`carries_log` is true but `log_carrying` has no positive length and radius "
            f"(length = {length} m, radius_top = {request.log_carrying.radius_top} m, "
            f"radius_bottom = {request.log_carrying.radius_bottom} m); an unknown payload is "
            "not a zero-sized one, so send `carries_log` false instead"
        )
    if not (np.isfinite(request.m_log) and request.m_log > 0.0):
        raise PlanningError(
            f"`carries_log` is true but `m_log` is {request.m_log} kg; an unknown payload is "
            "not a massless one, so send `carries_log` false instead"
        )
    # p_cyl_8 is x-only (legacy only read .x); s_log_8 is measured w/ CBS's grasp-offset y.
    # Centre = s_log_8 where given else p_cyl_8; only x cross-checked (both sides write it).
    mass_centre = np.array([request.s_log_8.x, request.s_log_8.y, request.s_log_8.z])
    collision_centre = np.array(
        [request.p_cyl_8.x, request.p_cyl_8.y, request.p_cyl_8.z]
    )
    if np.isfinite(mass_centre[0]) and (
        not np.isfinite(collision_centre[0])
        or abs(mass_centre[0] - collision_centre[0]) > 1.0e-9
    ):
        raise PlanningError(
            f"`s_log_8.x` and `p_cyl_8.x` are two different points along the body "
            f"({mass_centre[0]} m against {collision_centre[0]} m), and this planner "
            "carries one centre for the mass and for the collision body alike. A "
            "homogeneous log's centre of mass is its geometric centre; send one point"
        )
    centre = np.where(np.isfinite(mass_centre), mass_centre, collision_centre)
    if not np.all(np.isfinite(centre)):
        raise PlanningError(
            "neither `s_log_8` nor `p_cyl_8` gives a finite centre on every axis, and "
            "it is the centre this planner carries; the collision body has to have "
            "somewhere to be"
        )

    payload = Payload(
        mass_kg=float(request.m_log),
        center_of_mass_k8_m=centre,
        inertia_k8_kg_m2=np.zeros((3, 3)),
        valid=True,
    )
    # Cylinder dims = own-frame extents (2r,2r,len); bounds taper, not legacy's mean radius.
    shape = ("cylinder", np.array([2.0 * radius, 2.0 * radius, float(length)]), centre)
    return payload, shape


def translate_request(request, tip_to_tcp_offset_m: np.ndarray) -> A2bGoal:
    """Map every field of a `CalcMovement` request, or refuse it by name."""
    start = translate_start(request)

    if not np.isfinite(request.t_end) or request.t_end != 0.0:
        raise PlanningError(
            f"`t_end` asks for a fixed end time of {request.t_end} s, and this planner derives "
            "the duration from the path and the machine's own force and flow limits. Send "
            "`t_end` 0.0, which every caller does today"
        )
    if not np.all(np.asarray(request.v_d_tip, dtype=float) == 0.0):
        raise PlanningError(
            "`v_d_tip` asks the tool to still be moving at the end of the trajectory, and the "
            "terminal condition brings it to rest hanging still. This is refused rather than "
            "silently dropped"
        )
    # Obstacles come from the scene topic; a request can't carry its own without risk.
    if len(request.logs_scene) > 0:
        raise PlanningError(
            f"`logs_scene` carries {len(request.logs_scene)} obstacles, and this planner takes "
            "the scene from the collision scene topic where the world model publishes it. "
            "Silently planning without them would be planning through them"
        )

    payload, shape = translate_payload(request)

    # Two flags -> one avoid_collisions; empty gripper decides alone, carried log needs both.
    if (
        request.carries_log
        and request.check_log_collision != request.check_gripper_collision
    ):
        raise PlanningError(
            "`check_log_collision` and `check_gripper_collision` disagree while a log is "
            "carried, and this planner has one collision check covering the crane and the "
            "payload it holds. Checking one and not the other is not something it can be "
            "asked for"
        )

    # kappa is the deployment reservation; slow_down<1 would ask to exceed it, so it's refused.
    if not np.isfinite(request.slow_down) or request.slow_down < 1.0:
        raise PlanningError(
            f"`slow_down` is {request.slow_down} and it maps to a speed scale of "
            "1 / slow_down, which is bounded to (0, 1]. A divider below one asks this "
            "planner to exceed the deployment's own reservation"
        )

    if not (_finite_point(request.y_n) and np.isfinite(request.phi_tool_n)):
        raise PlanningError(
            "`y_n` and `phi_tool_n` are not four finite numbers, so there is no goal to plan to"
        )
    return A2bGoal(
        position_m=np.array([request.y_n.x, request.y_n.y, request.y_n.z])
        + np.asarray(tip_to_tcp_offset_m, dtype=float),
        yaw=float(request.phi_tool_n),
        payload=payload,
        payload_shape=shape,
        avoid_collisions=bool(request.check_gripper_collision),
        speed_scale=1.0 / float(request.slow_down),
        start=start,
    )
