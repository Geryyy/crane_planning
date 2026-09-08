"""
Serve the retained `a2b_movement` contract over the native planner.

`timber_crane_planning_interfaces/CalcMovement`, as a thin adapter.

# Why the service is kept and the message is not

Both stacks call `a2b_movement` today, so this planner serves it as a drop-in
replacement for the legacy server. Its payload fields are
`wood_log_msgs/LogShape` and `Log[]` -- a cylinder -- which is why a concrete
block is currently declared to the legacy planner **as a cylinder**. Keeping the
service compatible is worth an adapter; carrying `LogShape` inward is not. The
fiction stops here: everything below speaks `crane_model.Payload`, and a block
on the native path stays a box.

# An adapter, not a planner

Nothing here plans, limits or checks a collision. `translate_request` produces
the arguments of `Planner.plan`, the node runs it, the answer comes back
unchanged. One start state, one scene, one kappa, one `/crane/reference`
publication. No second set of limits, no second collision configuration.

# What `CalcMovement` says and does not

Two things the `.srv` carries only as a comment, settled by the callers:

* **Frame.** No `header`, so no frame field. Callers convert into
  `K0_mounting_base` before calling and the legacy server answers in it, so this
  adapter **asserts** that frame and converts nothing.
* **Body.** `y_n` is the "target position of tip (K5)" -- the pivot the pendulum
  hangs from -- while the native goal is the **tool** pose,
  `K8_tool_center_point`. `Planner.tip_to_tcp_offset` reads the settled offset
  between them out of the description at the request's yaw rather than writing it
  down or assuming it vertical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from crane_model import Payload

from .planner import PASSIVE_INDICES, PLANNED_INDICES, PlanningError, Start

#: The name the retained callers resolve, kept exactly as they resolve it.
#:
#: New cross-node contracts go under `/crane/...`; this is not a new one. The
#: timber panels and behaviour trees ask for the relative `a2b_movement` from a
#: node in the root namespace and the feasibility script asks for
#: `/a2b_movement` outright, so the absolute name all of them land on is this.
#: Renaming it would be retiring the service rather than keeping it compatible.
A2B_MOVEMENT_SERVICE = "/a2b_movement"

#: How far apart two `float32` shape fields may be and still be one body. The
#: callers copy one shape into the other, so equal fields are bit-identical.
SHAPE_TOLERANCE = 1.0e-6


@dataclass
class A2bGoal:
    """A `CalcMovement` request, in the native planner's own terms."""

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
    """
    Map `q0`/`q0_dot` onto an explicit start, or fall back to the measurement.

    A feasibility caller fills all eight `q0` entries so it can probe a pose
    without moving the crane, and those canonical rows map exactly onto the
    native start: `[q1..q8]`, with the passive pair supplied rather than
    estimated. A non-zero `q0_dot` under the all-zero `q0` default is refused,
    because it would combine rates supplied for one state with positions
    measured from another.
    """
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
    """
    Map the six log fields onto one payload, or refuse to guess.

    `carries_log == false` reads none of them: the timber trees send a
    `log_carrying` shape on the *approach* leg too, describing the log they are
    about to pick up, and reading it there would hang a log in an open gripper.
    """
    if not request.carries_log:
        return None, None

    # `log_carrying` is the inertia shape and `coll_shape` the collision shape,
    # and the native payload is one shape for both. Every caller that carries
    # sends them equal, so a request where they differ is asking for two bodies.
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
    # The two centres. `p_cyl_8` is an **x-only** field: every caller writes
    # `p_cyl_8.x` and leaves y and z at the .srv zero default, because the legacy
    # server read nothing else -- `a2b_server_base.cpp` takes `p_cyl_8.x` and the
    # generated collision body carries the scalar `p_cyl_8_x`. `s_log_8` is the
    # measured point: CBS fills its y from the offset the block ended up gripped
    # at. So the centre is `s_log_8` where specified and `p_cyl_8` where NaN, and
    # only x is cross-checked -- the one component both sides really write, where
    # a disagreement does describe two bodies. Reading an unwritten y as a claim
    # refuses every off-centre grasp.
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
    # A cylinder stays a cylinder, and the dimensions are the extent along each
    # axis of the primitive's own frame -- (2r, 2r, length), never a radius in
    # the first entry. A tapered log is bounded by the *enclosing* cylinder and
    # not by the legacy mean of the two radii, which leaves the wider end
    # sticking out of its own collision body.
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
    # Obstacles. The world model publishes them on the collision scene topic
    # already converted into K0_mounting_base; a request cannot carry its own,
    # and dropping the ones this one carries would plan through them.
    if len(request.logs_scene) > 0:
        raise PlanningError(
            f"`logs_scene` carries {len(request.logs_scene)} obstacles, and this planner takes "
            "the scene from the collision scene topic where the world model publishes it. "
            "Silently planning without them would be planning through them"
        )

    payload, shape = translate_payload(request)

    # The two collision flags become one `avoid_collisions`, which covers the
    # crane and the payload primitive together. With an empty gripper there is
    # no log to check, so the gripper's flag decides alone; with a log in it the
    # two have to agree, because there is no way to check one and not the other.
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

    # kappa is the deployment's reservation and the speed scale is the caller's
    # own request, which is exactly what `slow_down` is: a divider on the legacy
    # limits. A divider below one asks to go *faster* than the deployment
    # allows, which is the one thing the reservation exists to prevent.
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
