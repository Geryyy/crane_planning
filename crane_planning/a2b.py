"""
Serve the retained `a2b_movement` contract over the native planner.

`timber_crane_planning_interfaces/CalcMovement`, as a thin adapter.

# Why the service is kept and the message is not

`a2b_movement` is the interface the timber workflow depends on and both stacks
call today, so this planner serves it and is a drop-in replacement for the
legacy server. Its payload fields are `wood_log_msgs/LogShape` and `Log[]` -- a
cylinder description -- which is why a concrete block is currently declared to
the legacy planner **as a cylinder**. Keeping the service compatible is worth an
adapter; carrying `LogShape` into the new stack is not. So the fiction stops
here: everything below this module speaks `crane_model.Payload`, and a block on
the native path stays a box.

# It is an adapter and not a planner

Nothing here plans, limits, or checks a collision. `translate_request` produces
the arguments of `Planner.plan`, the same call `/crane/plan_motion` is answered
by -- so the same start state, the same scene, the same kappa and the same
`/crane/reference` publication -- and the answer is carried back unchanged.
There is no second set of limits and no second collision configuration, because
there is no second planner.

# What `CalcMovement` says and what it does not

Two things the `.srv` carries only as a comment, and that the callers settle:

* **The frame.** `CalcMovement` has no `header` and therefore no frame field.
  The callers convert into `K0_mounting_base` before they call, and the legacy
  server answers in it and stamps its published path with it. This adapter
  therefore **asserts** that frame and converts nothing.
* **The body.** `y_n` is documented as the "target position of tip (K5)" -- the
  pivot the pendulum hangs from -- while the native goal is the **tool** pose,
  `K8_tool_center_point`. `Planner.tip_to_tcp_offset` reads the settled 3-D
  offset between them out of the description for the request's yaw and payload,
  rather than writing it down or assuming it is vertical: the PZS100's rail
  gripper and the 7040's jaw do not hang alike, and an off-axis centre of mass
  changes it again.
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
    if not _finite_point(request.p_cyl_8):
        raise PlanningError(
            "`p_cyl_8` is not three finite numbers, and it is the centre this planner "
            "carries; the collision body has to have somewhere to be"
        )

    # The two centres. Every caller fills all three components of `p_cyl_8`; the
    # behaviour tree leaves `s_log_8.z` at NaN on purpose, which is that caller
    # saying "not specified" rather than "at infinity". One centre carries both,
    # so the unspecified components come from `p_cyl_8` and the specified ones
    # have to agree with it -- a homogeneous log's mass centre *is* its
    # geometric centre, and a request where the two disagree describes two
    # bodies again.
    centre = np.array([request.p_cyl_8.x, request.p_cyl_8.y, request.p_cyl_8.z])
    mass_centre = np.array([request.s_log_8.x, request.s_log_8.y, request.s_log_8.z])
    for axis in range(3):
        if (
            np.isfinite(mass_centre[axis])
            and abs(mass_centre[axis] - centre[axis]) > 1.0e-9
        ):
            raise PlanningError(
                f"`s_log_8` and `p_cyl_8` are two different points on axis {axis} "
                f"({mass_centre[axis]} m against {centre[axis]} m), and this planner carries "
                "one centre for the mass and for the collision body alike. A homogeneous "
                "log's centre of mass is its geometric centre; send one point"
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
