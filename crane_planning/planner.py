"""
Plan a motion, and time it.

Three stages, no more: OMPL RRT-Connect searches the five planned joint
coordinates, a C2 spline turns the polyline into q_a(sigma) with derivatives,
and one CasADi/IPOPT optimal control problem over sigma decides how fast the
machine may traverse it. Everything the machine can do -- reach, hang, collide,
lift -- is asked of `crane_model`; nothing about the machine is written down
here.

The passive pair (tip/tilt sway) is never searched over: it hangs, so it is
solved with `passive_equilibrium` wherever geometry is checked, and it is a
*state* of the timing OCP so that the answer is a trajectory the tool arrives
at rest from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pinocchio as pin
from crane_model import (
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    CollisionPrimitive,
    CraneModel,
    CraneModelError,
    Frame,
    Payload,
    Tool,
    parse,
)
from crane_model import symbolic as symbolic_model
from ompl import base as ob
from ompl import geometric as og
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

#: The five coordinates a plan moves, in canonical indices. The tool axis (q8)
#: is held by the low-level controller and rides the path at a constant value.
PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]
PLANNED_DOF = len(PLANNED_INDICES)

#: `q_a | q_a' | q_a'' | q8` -- what the OCP needs to know about the path at one
#: value of sigma, and the only thing it is told about it.
PATH_BLOCK = 3 * PLANNED_DOF + 1


class PlanningError(RuntimeError):
    """A refusal. The message is what the service answer carries."""


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    """What the machine may do, read from the description and the hydraulics."""

    lower: np.ndarray  # position, per planned coordinate; -inf where continuous
    upper: np.ndarray
    bounded: np.ndarray  # False for a `continuous` joint, which has no range
    dq_max: np.ndarray  # velocity, per planned coordinate
    force_max: np.ndarray  # cylinder force, per planned axis, at the relief
    flow_max: float  # summed pump draw
    passive_lower: np.ndarray  # the range the sway pair may hang in
    passive_upper: np.ndarray


def read_limits(
    description_xml: str,
    tool: Tool,
    model: symbolic_model.CraneSymbolicModel,
    system_pressure_pa: float,
    pump_flow_max: float,
    pump_flow_planning_factor: float,
) -> Limits:
    """
    Position and velocity out of the description, force out of the hydraulics.

    A `continuous` joint occupies two configuration slots (a cos/sin pair) and
    carries no range; the rotator is one, so it is reported unbounded rather
    than given an invented range here.
    """
    description = parse(description_xml, tool)
    inner = description.model
    lower = np.empty(PLANNED_DOF)
    upper = np.empty(PLANNED_DOF)
    bounded = np.empty(PLANNED_DOF, dtype=bool)
    dq_max = np.empty(PLANNED_DOF)
    for axis, index in enumerate(PLANNED_INDICES):
        slot = description.drives[index][0].slot
        bounded[axis] = slot.nq == 1
        lower[axis] = inner.lowerPositionLimit[slot.idx_q] if slot.nq == 1 else -np.inf
        upper[axis] = inner.upperPositionLimit[slot.idx_q] if slot.nq == 1 else np.inf
        dq_max[axis] = inner.velocityLimit[slot.idx_v]
    if not np.all(np.isfinite(dq_max)) or np.any(dq_max <= 0.0):
        raise PlanningError(
            "the description gives a planned joint no finite positive velocity limit"
        )
    passive = [description.drives[index][0].slot for index in PASSIVE_INDICES]
    passive_lower = np.array([inner.lowerPositionLimit[slot.idx_q] for slot in passive])
    passive_upper = np.array([inner.upperPositionLimit[slot.idx_q] for slot in passive])

    # `hydraulics.md` 4's F = A_A p_A - A_B p_B with one chamber at the relief
    # setting at a time. The *smaller* of the two is the limit, because the
    # constraint is written symmetrically in |F| and a differential cylinder is
    # weaker retracting.
    pressure = np.full(len(ACTUATED_INDICES), float(system_pressure_pa))
    zero = np.zeros(len(ACTUATED_INDICES))
    extend = np.abs(np.array(ca.evalf(model.chamber_force(pressure, zero))).ravel())
    retract = np.abs(np.array(ca.evalf(model.chamber_force(zero, pressure))).ravel())
    force_max = np.minimum(extend, retract)[:PLANNED_DOF]
    if np.any(force_max <= 0.0):
        raise PlanningError("a cylinder carries no force at the relief pressure")
    return Limits(
        lower=lower,
        upper=upper,
        bounded=bounded,
        dq_max=dq_max,
        force_max=force_max,
        flow_max=float(pump_flow_max) * float(pump_flow_planning_factor),
        passive_lower=passive_lower,
        passive_upper=passive_upper,
    )


# --------------------------------------------------------------------- config


@dataclass
class PlannerConfig:
    """Properties of the solve. Machine numbers are not among them."""

    tool: Tool = Tool.PZS100

    # Reservation held back from every physical limit so the MPC has authority
    # left to correct with.
    kappa: float = 0.8

    # Endpoint IK acceptance, in metres and radians.
    eps_pos: float = 1.0e-3
    eps_yaw: float = 1.0e-3
    ik_restarts: int = 5

    # OMPL. The seed is a constant and not a clock: one request against one
    # scene is reproducible.
    ompl_seed: int = 20420042
    ompl_time_budget: float = 5.0
    ompl_extension_span: float = 2.0
    ompl_unbounded_margin: float = np.pi
    shortcut_attempts: int = 120

    # Collision, in metres. The path and the sway envelope share the step.
    q_sway_max: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2]))
    check_resolution: float = 0.10
    min_check_resolution: float = 0.01
    max_check_samples: int = 2048

    # Timing OCP.
    intervals: int = 40
    sway_weight: float = 2.0
    input_weight: float = 1.0e-3
    sigma_rate_min: float = 0.02
    sigma_rate_max: float = 2.0
    sigma_accel_max: float = 20.0
    dq_sway_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5]))
    ddq_a_max: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.7, 0.5, 1.0, 6.0])
    )
    max_wall_clock: float = 30.0
    max_iterations: int = 400
    tolerance: float = 1.0e-6

    # Hydraulics the description does not carry. See config/hydraulic_limits.yaml.
    pump_flow_max: float = 1.4e-3
    pump_flow_planning_factor: float = 0.95
    system_pressure_pa: float = 2.5e7

    # The emitted reference period.
    Ts: float = 0.04

    # The truck, as a property of the vehicle: the scene carries one primitive
    # with the reserved id `truck` and the bed and six runges are placed on it.
    # The stations are the posts of `post_setup:=134` and its subset `13`; keep
    # them equal to `world_model.vehicle_box.runge_stations_m`.
    truck_runge_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.28, 0.31, 2.12])
    )
    truck_runge_stations: np.ndarray = field(
        default_factory=lambda: np.array([-2.261, -1.049, 1.935])
    )
    truck_bed_thickness: float = 0.10
    # The headboard closing the cab end of the deck: 0.45 m of plate and side
    # rail standing 1.922 m off the bed surface, across its full width. It is
    # the box's +x end, which is the end the outermost runge station is at.
    truck_headboard_thickness: float = 0.45
    truck_headboard_height: float = 1.922


TRUCK_ID = "truck"
PAYLOAD_ID = "payload"


# ------------------------------------------------------------------ the scene


def expand_truck(scene: list, config: PlannerConfig) -> list:
    """
    Turn the reserved `truck` primitive into the bed, the runges and the headboard.

    The crane is bolted to the vehicle, so the box is not an obstacle; what the
    tool can hit is the bed's top face, the runges standing on it and the
    headboard closing its cab end. The dimensions are the vehicle's and are
    configured; the position is measured and arrives with the primitive.

    A runge is placed flush against the bed edge rather than centred on it,
    which is where the description puts the real post's outer face; the
    configured section is an inflation of that post, so it grows inboard. The
    headboard is flush against the box's +x face the same way, which is the end
    the outermost runge station is at.
    """
    expanded = []
    for primitive in scene:
        if primitive.id != TRUCK_ID:
            expanded.append(primitive)
            continue
        pose = primitive.pose_in_mounting_base
        box = np.asarray(primitive.dimensions_m, dtype=float)
        bed = float(config.truck_bed_thickness)
        top = 0.5 * box[2]
        expanded.append(
            CollisionPrimitive(
                id="truck_bed",
                shape="box",
                pose_in_mounting_base=pose
                * pin.SE3(np.eye(3), np.array([0.0, 0.0, top - 0.5 * bed])),
                dimensions_m=np.array([box[0], box[1], bed]),
                structural=True,
            )
        )
        head = float(config.truck_headboard_thickness)
        height = float(config.truck_headboard_height)
        if head > 0.0 and height > 0.0:
            centre = np.array([0.5 * (box[0] - head), 0.0, top + 0.5 * height])
            expanded.append(
                CollisionPrimitive(
                    id="truck_headboard",
                    shape="box",
                    pose_in_mounting_base=pose * pin.SE3(np.eye(3), centre),
                    dimensions_m=np.array([head, box[1], height]),
                    structural=True,
                )
            )
        runge = np.asarray(config.truck_runge_dimensions, dtype=float)
        for side, sign in (("right", 1.0), ("left", -1.0)):
            edge = sign * 0.5 * (box[1] - runge[1])
            for station, offset in enumerate(config.truck_runge_stations):
                centre = np.array([float(offset), edge, top + 0.5 * runge[2]])
                expanded.append(
                    CollisionPrimitive(
                        id=f"truck_runge_{side}_{station}",
                        shape="box",
                        pose_in_mounting_base=pose * pin.SE3(np.eye(3), centre),
                        dimensions_m=runge,
                        structural=True,
                    )
                )
    return expanded


def payload_primitive(
    model: CraneModel, q: np.ndarray, payload_shape
) -> CollisionPrimitive | None:
    """
    Return what is in the gripper, as a body at the K8 pose of `q`.

    The model's collision API takes no payload, so a carried block is handed to
    it as a scene body instead -- otherwise a plan is certified clear of
    everything except the thing the crane is holding. It is marked
    `attached_to_tool`, which is what keeps the grip itself from reading as a
    collision and what gets the body checked against the obstacles.

    **It is placed per configuration and not once.** A payload pinned at the
    pose the machine set off from is a ghost standing in the start pose while
    the real one rides the tool through the scene unchecked.
    """
    if payload_shape is None:
        return None
    shape, dimensions, offset = payload_shape
    pose = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.ROTATOR_LOWER_PART)
    placement = pin.SE3(
        pin.XYZQUATToSE3(np.concatenate([pose.position_m, pose.orientation_xyzw]))
    ) * pin.SE3(np.eye(3), np.asarray(offset, dtype=float))
    return CollisionPrimitive(
        id=PAYLOAD_ID,
        shape=shape,
        pose_in_mounting_base=placement,
        dimensions_m=np.asarray(dimensions, dtype=float),
        structural=False,
        attached_to_tool=True,
    )


# ---------------------------------------------------------------- the hanging pose


class Equilibrium:
    """
    Where the tool hangs, as Newton on a compiled residual.

    `crane_model` answers this too, and answers it more carefully -- it scans
    the whole passive range for a seed every time. That costs some 80 ms, and a
    sampling planner asks the question tens of thousands of times: once per
    state OMPL checks, once per residual evaluation of the inverse kinematics,
    once per node of the timing OCP. So the same equation is solved here off the
    shared CasADi graph, warm-started from the last answer, which is the right
    seed because consecutive questions are about neighbouring configurations.

    A two-hinge pendulum has four critical points on the torus and two of them
    are the tool standing *up*; they satisfy `h_u = 0` exactly as well. The sign
    of the stiffness is what separates them, so it is checked at every iterate
    and not only at the end.
    """

    #: The residual is of order 1e3 N m before it cancels; this is the floor it
    #: reaches, and it is worth under 1e-11 rad against the restoring stiffness.
    RESIDUAL_NM = 1.0e-8
    ITERATIONS = 32
    #: Samples per axis in the cold-start scan. Five over the pi of tip range
    #: puts a sample well inside the hanging well.
    SAMPLES = 5
    #: How far outside the description's passive range an answer may land and
    #: still be that range's own edge. At a canonical pose the tool hangs
    #: *exactly* on the tip limit, and Newton reaches it from below by two ulps;
    #: a bare comparison then refuses the pose the machine is actually in.
    RANGE_TOLERANCE = 1.0e-6

    def __init__(self, model: symbolic_model.CraneSymbolicModel, limits: Limits):
        q_a = ca.SX.sym("q_a", PLANNED_DOF)
        q_u = ca.SX.sym("q_u", 2)
        q_tool = ca.SX.sym("q_tool")
        payload = ca.SX.sym("payload", symbolic_model.NP - 1)
        residual = ca.substitute(
            model.bias_u,
            ca.vertcat(model.x, model.u, model.p),
            ca.vertcat(
                q_a,
                q_u,
                ca.SX.zeros(PLANNED_DOF + 2),  # x, at rest
                ca.SX.zeros(PLANNED_DOF),  # u
                q_tool,
                payload,  # p
            ),
        )
        self._step = ca.Function(
            "equilibrium_step",
            [q_a, q_u, q_tool, payload],
            [residual, ca.jacobian(residual, q_u)],
        )
        self._scan = [
            np.array([tip, tilt])
            for tip in np.linspace(
                limits.passive_lower[0], limits.passive_upper[0], self.SAMPLES
            )
            for tilt in np.linspace(
                limits.passive_lower[1], limits.passive_upper[1], self.SAMPLES
            )
        ]
        self._lower = limits.passive_lower
        self._upper = limits.passive_upper
        self._last = np.zeros(2)

    def solve(self, q_a: np.ndarray, q_tool: float, payload: np.ndarray) -> np.ndarray:
        """Return the passive pair the tool hangs at, or refuse that it hangs at all."""
        for seed in [self._last, *self._scan]:
            answer = self._newton(q_a, q_tool, payload, seed)
            if answer is not None:
                self._last = answer
                return answer
        raise PlanningError(
            "the tool reaches no hanging pose inside the range the description gives "
            "the passive joints at this configuration"
        )

    def _newton(self, q_a, q_tool, payload, seed):
        q_u = np.asarray(seed, dtype=float).copy()
        for _ in range(self.ITERATIONS):
            residual, stiffness = self._step(q_a, q_u, q_tool, payload)
            residual = np.array(residual).ravel()
            stiffness = np.array(stiffness)
            if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(stiffness))):
                return None
            try:
                # Positive definite is the hanging branch; the saddles and the
                # tool standing up are the other three roots.
                factor = np.linalg.cholesky(0.5 * (stiffness + stiffness.T))
            except np.linalg.LinAlgError:
                return None
            if np.linalg.norm(residual) <= self.RESIDUAL_NM:
                if np.any(q_u < self._lower - self.RANGE_TOLERANCE) or np.any(
                    q_u > self._upper + self.RANGE_TOLERANCE
                ):
                    return None
                return np.clip(q_u, self._lower, self._upper)
            q_u = q_u - np.linalg.solve(factor.T, np.linalg.solve(factor, residual))
            if not np.all(np.isfinite(q_u)):
                return None
        return None


# ------------------------------------------------------------------- geometry


class Geometry:
    """
    Whether a configuration and a path are clear, at the pose the tool hangs at.

    Every distance is a `crane_model` query -- there is no second collision
    model here. The sway envelope of `wiki/trajectory_planning.md` 4.3 is
    answered in two steps: one query at the nominal hanging pose settles the
    free-space majority, because no admissible sway can move the tool further
    than `l_tool * sin(q_u^+)`; only where that margin is not met is the sway
    box actually gridded.
    """

    def __init__(
        self,
        model: CraneModel,
        equilibrium: Equilibrium,
        scene: list,
        config: PlannerConfig,
        q_tool: float,
        payload: np.ndarray,
        payload_shape=None,
    ):
        self.model = model
        self.equilibrium = equilibrium
        self.scene = list(scene)
        self.config = config
        self.q_tool = float(q_tool)
        self.payload = payload
        self.payload_shape = payload_shape
        self.envelope = self._envelope()
        self.step_m = self._resolution()

    def bodies(self, q: np.ndarray) -> list:
        """Return the scene at `q`, with what the tool carries placed on it."""
        carried = payload_primitive(self.model, q, self.payload_shape)
        return self.scene if carried is None else self.scene + [carried]

    def _envelope(self) -> float:
        """How far the tool can swing, in metres, at the admissible sway bound."""
        q = np.zeros(GENERALIZED_DOF)
        q[TOOL_INDEX] = self.q_tool
        hinge = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TILT)
        tcp = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        length = float(np.linalg.norm(tcp.position_m - hinge.position_m))
        return length * float(np.sin(np.max(np.abs(self.config.q_sway_max))))

    def _resolution(self) -> float:
        """Return the step, tightened to half the thinnest primitive the scene carries."""
        step = float(self.config.check_resolution)
        for primitive in self.scene:
            step = min(step, 0.5 * float(np.min(primitive.dimensions_m)))
        if step < self.config.min_check_resolution:
            raise PlanningError(
                f"the scene needs a check step of {step:.4f} m, below the "
                f"{self.config.min_check_resolution:.4f} m floor: a path checked at a "
                "step it does not resolve is not checked"
            )
        return step

    def configuration(self, q_a: np.ndarray) -> np.ndarray:
        """Return the canonical eight at `q_a`, with the tool hanging."""
        q = np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)] = q_a
        q[TOOL_INDEX] = self.q_tool
        q[list(PASSIVE_INDICES)] = self.equilibrium.solve(
            q_a, self.q_tool, self.payload
        )
        return q

    def clearance(self, q: np.ndarray) -> float:
        """Smallest distance to anything, the machine against itself included."""
        return float(self.model.collision_query(q, self.bodies(q)).minimum_distance_m)

    def is_valid(self, q_a: np.ndarray) -> bool:
        """Clear at the hanging pose, and clear anywhere the tool may swing to."""
        try:
            q = self.configuration(q_a)
        except (CraneModelError, PlanningError):
            return False
        bodies = self.bodies(q)
        # One sweep, not two. `crane_model.collision.query` is defined as
        # `min(queries(...))`, so asking for the overall minimum and then for
        # the per-primitive row computed every distance pair twice -- and this
        # is the inner loop of the search.
        results = self.model.collision_queries(q, bodies)
        if min(result.minimum_distance_m for result in results) <= 0.0:
            return False
        if self.envelope <= 0.0 or not bodies:
            return True
        scene_only = min(
            (result.minimum_distance_m for result in results[:-1]),
            default=np.inf,
        )
        if scene_only > self.envelope:
            return True
        # The sufficient condition did not hold, so the box is checked. Corners
        # and edge midpoints: the extremes are where the tool actually reaches.
        equilibrium = q[list(PASSIVE_INDICES)]
        bound = np.asarray(self.config.q_sway_max, dtype=float)
        axis = (-1.0, 0.0, 1.0)
        swung = q.copy()
        for tip in axis:
            for tilt in axis:
                swung[list(PASSIVE_INDICES)] = equilibrium + bound * np.array(
                    [tip, tilt]
                )
                # The payload swings with the tool, so it is re-placed too.
                if (
                    self.model.collision_query(
                        swung, self.bodies(swung)
                    ).minimum_distance_m
                    <= 0.0
                ):
                    return False
        return True

    def travel(self, first: np.ndarray, second: np.ndarray) -> float:
        """How far the tool moves between two configurations, in metres."""
        try:
            one = self.model.forward_kinematics(
                self.configuration(first), Frame.MOUNTING_BASE, Frame.TCP
            )
            other = self.model.forward_kinematics(
                self.configuration(second), Frame.MOUNTING_BASE, Frame.TCP
            )
        except (CraneModelError, PlanningError):
            return np.inf
        return float(np.linalg.norm(other.position_m - one.position_m))

    def check_path(self, path) -> None:
        """Sample the fitted curve and refuse at the first blocked sigma."""
        samples = min(
            self.config.max_check_samples,
            max(16, int(np.ceil(path.travel_m / self.step_m)) + 1),
        )
        for sigma in np.linspace(0.0, 1.0, samples):
            if not self.is_valid(path.position(sigma)):
                raise PlanningError(
                    f"the fitted path is blocked at sigma = {sigma:.3f}"
                )


# -------------------------------------------------------------------- the IK


def yaw_of(rotation: np.ndarray) -> float:
    """Return the rotation about K0 z, which is the whole of what a goal fixes."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap(angle: float) -> float:
    """Fold an angle difference into (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def solve_ik(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    position_m: np.ndarray,
    yaw: float,
    seed: np.ndarray,
) -> np.ndarray:
    """
    Solve for the planned five that put the tool at `(position_m, yaw)`, hanging.

    One least-squares problem, not a closed form: the passive pair is re-settled
    at every configuration tested, so what is solved is where the tool actually
    ends up rather than where it would be if it did not hang. The redundancy the
    telescope leaves is taken up by a weak pull towards the seed, and the
    restarts spread over the telescope range because that is the coordinate the
    residual is flat in.
    """
    lower = np.where(limits.bounded, limits.lower, seed - 2.0 * np.pi)
    upper = np.where(limits.bounded, limits.upper, seed + 2.0 * np.pi)

    def residual(q_a):
        try:
            q = geometry.configuration(q_a)
        except (CraneModelError, PlanningError):
            return np.full(PLANNED_DOF + 4, 1.0e3)
        pose = geometry.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        rotation = pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
        angle = np.arctan2(
            np.sin(yaw_of(rotation) - yaw), np.cos(yaw_of(rotation) - yaw)
        )
        return np.concatenate(
            [pose.position_m - position_m, [angle], 1.0e-3 * (q_a - seed)]
        )

    best, best_error = None, np.inf
    telescope = np.linspace(lower[3], upper[3], max(1, config.ik_restarts))
    for extension in telescope:
        start = np.clip(seed.copy(), lower, upper)
        start[3] = extension
        answer = least_squares(
            residual, start, bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
        error = np.linalg.norm(answer.fun[:4])
        if error < best_error:
            best, best_error = answer.x, error
        # The restarts exist to escape a local minimum. A solve that already
        # meets the tolerance this function is about to check is not in one, and
        # the remaining restarts can only re-derive an answer already in hand --
        # measured across the workspace, every one of them returns the same
        # configuration to three decimals.
        if (
            float(np.linalg.norm(answer.fun[:3])) <= config.eps_pos
            and float(abs(answer.fun[3])) <= config.eps_yaw
        ):
            # This restart, not whichever had the smallest combined norm. The
            # selection above mixes three metres with one radian in a single
            # 4-norm while the acceptance test below is two separate scalars, so
            # the restart that satisfies the test need not be the one holding
            # `best` -- and returning the other refuses a goal just reached.
            best, best_error = answer.x, error
            break

    q_a = best
    final = residual(q_a)
    position_error = float(np.linalg.norm(final[:3]))
    yaw_error = float(abs(final[3]))
    if position_error > config.eps_pos or yaw_error > config.eps_yaw:
        raise PlanningError(
            f"the goal is not reachable: the closest configuration misses it by "
            f"{position_error * 1e3:.1f} mm and {np.degrees(yaw_error):.2f} deg "
            f"(tolerances {config.eps_pos * 1e3:.1f} mm, "
            f"{np.degrees(config.eps_yaw):.2f} deg)"
        )
    if not geometry.is_valid(q_a):
        raise PlanningError("the goal configuration is in collision")
    return q_a


# ---------------------------------------------------------------- the geometry search


@dataclass
class Path:
    """
    q_a(sigma) on [0, 1], twice differentiable everywhere.

    Twice, and that is the whole point of fitting at all: the timing stage
    writes `ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot`, so at a kink in the
    polyline q_a'' is unbounded, the admissible path rate collapses to zero and
    the machine stops dead at every waypoint.
    """

    spline: CubicSpline
    travel_m: float

    def position(self, sigma) -> np.ndarray:
        return self.spline(sigma)

    def rate(self, sigma) -> np.ndarray:
        return self.spline(sigma, 1)

    def curvature(self, sigma) -> np.ndarray:
        return self.spline(sigma, 2)

    def block(self, sigma, q_tool: float) -> np.ndarray:
        """Return the `q_a | q_a' | q_a'' | q8` the OCP is told about one sigma."""
        return np.concatenate(
            [self.position(sigma), self.rate(sigma), self.curvature(sigma), [q_tool]]
        )


def search(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """
    RRT-Connect over the five planned coordinates, then a shortcut pass.

    The search runs in coordinates scaled by each axis's own velocity limit, so
    the metric OMPL extends and interpolates in is *seconds at full speed*
    rather than a sum of radians and metres. Only exact solutions are taken.
    """
    scale = limits.dq_max
    margin = config.ompl_unbounded_margin
    lower = np.where(limits.bounded, limits.lower, np.minimum(start, goal) - margin)
    upper = np.where(limits.bounded, limits.upper, np.maximum(start, goal) + margin)
    if np.any(start < lower) or np.any(start > upper):
        raise PlanningError(
            "the measured start is outside the description's joint range"
        )
    if np.any(goal < lower) or np.any(goal > upper):
        raise PlanningError(
            "the goal configuration is outside the description's joint range"
        )

    space = ob.RealVectorStateSpace(PLANNED_DOF)
    bounds = ob.RealVectorBounds(PLANNED_DOF)
    for axis in range(PLANNED_DOF):
        bounds.setLow(axis, float(lower[axis] / scale[axis]))
        bounds.setHigh(axis, float(upper[axis] / scale[axis]))
    space.setBounds(bounds)
    space.setup()

    setup = og.SimpleSetup(space)
    information = setup.getSpaceInformation()

    class Valid(ob.StateValidityChecker):
        def isValid(self, state):
            return geometry.is_valid(
                np.array([state[axis] for axis in range(PLANNED_DOF)]) * scale
            )

    checker = Valid(information)
    setup.setStateValidityChecker(checker)

    # How finely a motion is subdivided. The search metric is time, not
    # distance, so the fraction is set from how far the tool actually travels
    # over the start-to-goal chord -- the fitted path is re-checked at `step_m`
    # afterwards, and that check is what certifies the answer.
    chord = float(np.linalg.norm((goal - start) / scale))
    travel = geometry.travel(start, goal)
    if chord > 1e-9 and travel > 1e-9:
        extent = float(space.getMaximumExtent())
        information.setStateValidityCheckingResolution(
            float(np.clip(geometry.step_m * chord / (travel * extent), 1e-4, 0.05))
        )

    def state(values):
        allocated = information.allocState()
        for axis in range(PLANNED_DOF):
            allocated[axis] = float(values[axis] / scale[axis])
        return allocated

    # The direct motion, before the sampler is asked for anything. In this
    # space -- every axis divided by its own velocity limit -- the straight line
    # between two configurations is the least joint travel there is, so when it
    # is clear there is nothing for a search to improve on and `shortcut` would
    # collapse to it anyway. Both endpoints are already known valid: `solve_ik`
    # refuses a goal in collision and `Planner.plan` refuses the start, so this
    # costs one motion check and nothing else.
    # `setStateValidityCheckingResolution` records a *fraction*; the segment
    # length a motion check actually subdivides by is recomputed only in
    # `setup()`, which `solve()` would not reach until after the check below.
    # Without this the direct motion is checked at OMPL's default one percent --
    # on this space seven times coarser than the step the scene asked for, which
    # is how a straight line past a runge gets accepted here and then refused by
    # `check_path`.
    information.setup()
    first, last = state(start), state(goal)
    if information.checkMotion(first, last):
        return np.array([start, goal])

    setup.setStartAndGoalStates(first, last)
    planner = og.RRTConnect(information)
    planner.setRange(float(config.ompl_extension_span))
    setup.setPlanner(planner)

    if not setup.solve(float(config.ompl_time_budget)):
        raise PlanningError(
            f"no collision-free path was found in {config.ompl_time_budget:.1f} s"
        )
    if not setup.haveExactSolutionPath():
        raise PlanningError("only an approximate path was found, which is not a plan")

    solution = setup.getSolutionPath()
    waypoints = [
        np.array([solution.getState(index)[axis] for axis in range(PLANNED_DOF)])
        * scale
        for index in range(solution.getStateCount())
    ]
    return shortcut(information, space, scale, waypoints, config)


def shortcut(
    information, space, scale, waypoints: list, config: PlannerConfig
) -> np.ndarray:
    """
    Replace random spans by their chord wherever the chord is clear.

    Its own seeded generator, so the smoothing is reproducible for the same
    search result, and its own check allowance -- how hard the search was is no
    reason to smooth its answer less.
    """
    generator = np.random.default_rng(config.ompl_seed + 1)

    def state(values):
        allocated = information.allocState()
        for axis in range(PLANNED_DOF):
            allocated[axis] = float(values[axis] / scale[axis])
        return allocated

    for _ in range(int(config.shortcut_attempts)):
        if len(waypoints) <= 2:
            break
        first, second = sorted(generator.choice(len(waypoints), size=2, replace=False))
        if second - first < 2:
            continue
        if information.checkMotion(state(waypoints[first]), state(waypoints[second])):
            waypoints = waypoints[: first + 1] + waypoints[second:]
    return np.array(waypoints)


def fit(
    geometry: Geometry,
    limits: Limits,
    waypoints: np.ndarray,
    dq_start: np.ndarray,
) -> tuple[Path, float | None]:
    """
    Fit `q_a(sigma)` through the waypoints, and say what sigma_dot(0) must be.

    sigma is distributed by how long each chord takes at the slowest axis's own
    limit, so a segment that is slow in one coordinate gets more of the
    parameter. That total is a duration, and it is also what pins the start
    rate: choosing `q_a'(0) = dq_a * L` makes `dq_a = q_a'(0) sigma_dot(0)` hold
    exactly at `sigma_dot(0) = 1/L`, rather than leaving a residual to accept or
    refuse.
    """
    spans = np.max(np.abs(np.diff(waypoints, axis=0)) / limits.dq_max, axis=1)
    spans = np.maximum(spans, 1.0e-3)
    total = float(np.sum(spans))
    nodes = np.concatenate([[0.0], np.cumsum(spans) / total])

    moving = float(np.linalg.norm(dq_start)) > 1.0e-9
    start_rate = (
        np.asarray(dq_start, dtype=float) * total if moving else np.zeros(PLANNED_DOF)
    )
    spline = CubicSpline(
        nodes,
        waypoints,
        bc_type=((1, start_rate), (1, np.zeros(PLANNED_DOF))),
    )

    dense = np.linspace(0.0, 1.0, 129)
    sampled = spline(dense)
    slack = 1.0e-6
    for axis in range(PLANNED_DOF):
        if not limits.bounded[axis]:
            continue
        if (
            np.min(sampled[:, axis]) < limits.lower[axis] - slack
            or np.max(sampled[:, axis]) > limits.upper[axis] + slack
        ):
            raise PlanningError(
                f"the fitted path overshoots joint {axis}'s range; the search returned "
                "waypoints too far apart to smooth inside it"
            )

    travel = 0.0
    for one, other in zip(sampled[:-1], sampled[1:]):
        travel += geometry.travel(one, other)
    return Path(spline=spline, travel_m=float(travel)), (
        1.0 / total if moving else None
    )


# ------------------------------------------------------------------- the timing OCP


@dataclass
class Timing:
    """What the OCP decided, on its own sigma grid."""

    sigma: np.ndarray
    sigma_dot: np.ndarray
    sigma_ddot: np.ndarray
    q_u: np.ndarray
    dq_u: np.ndarray
    q_u_eq: np.ndarray
    q_a: np.ndarray
    dq_a: np.ndarray
    ddq_a: np.ndarray
    cylinder_force: np.ndarray
    pump_flow: np.ndarray
    time: np.ndarray
    iterations: int
    solve_time_s: float

    @property
    def duration(self) -> float:
        return float(self.time[-1])


class TimingOcp:
    """
    How fast the machine may traverse a given path, as one NLP over sigma.

    The independent variable is the path parameter, not time. The state is
    `(sigma_dot, q_u, dq_u)`: the sway is *planned*, not merely tolerated, so
    the trajectory the machine is handed is one the tool arrives at rest from.
    The control is `sigma_ddot`, and the five joint accelerations follow from it
    by the chain rule -- `ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot` -- which
    is why the path has to be C2 and why nothing here re-solves geometry.

    Minimising traversal time is minimising the integral of `dsigma/sigma_dot`,
    so `sigma_dot` is bounded away from zero throughout; the path is fitted to
    leave and arrive at rest in *q_a*, which is what makes that admissible.
    """

    def __init__(
        self,
        model: symbolic_model.CraneSymbolicModel,
        limits: Limits,
        config: PlannerConfig,
    ):
        self.limits = limits
        self.config = config

        state = ca.SX.sym("y", 5)  # sigma_dot, q_u (2), dq_u (2)
        control = ca.SX.sym("a")
        block = ca.SX.sym("block", PATH_BLOCK)
        payload = ca.SX.sym("payload", symbolic_model.NP - 1)

        rate, curvature = block[5:10], block[10:15]
        sigma_dot, q_u, dq_u = state[0], state[1:3], state[3:5]
        ddq_a = curvature * sigma_dot**2 + rate * control

        substituted = ca.substitute(
            ca.vertcat(model.ddq_u, model.z),
            ca.vertcat(model.x, model.u, model.p),
            ca.vertcat(
                block[0:5],
                q_u,
                rate * sigma_dot,
                dq_u,  # x
                ddq_a,  # u
                block[15],
                payload,  # p
            ),
        )
        ddq_u, output = substituted[:2], substituted[2:]
        force = output[
            symbolic_model.K_CYLINDER_FORCE_OFFSET : symbolic_model.K_CYLINDER_FORCE_OFFSET
            + PLANNED_DOF
        ]
        flow = ca.sum1(
            output[
                symbolic_model.K_AXIS_FLOW_OFFSET : symbolic_model.K_AXIS_FLOW_OFFSET
                + PLANNED_DOF
            ]
        )

        arguments = [state, control, block, payload]
        #: d/dsigma of the state: every time derivative divided by sigma_dot,
        #: because sigma and not time is what is being integrated over.
        self.derivative = ca.Function(
            "derivative",
            arguments,
            [ca.vertcat(control / sigma_dot, dq_u / sigma_dot, ddq_u / sigma_dot)],
        )
        #: What the answer demands of the machine at one node.
        self.demand = ca.Function("demand", arguments, [ddq_a, force, flow])

    def _blocks(self, path: Path, q_tool: float, nodes: np.ndarray) -> np.ndarray:
        return np.array([path.block(sigma, q_tool) for sigma in nodes])

    def _ceiling(self, blocks: np.ndarray, speed_scale: float) -> np.ndarray:
        """
        Return the largest `sigma_dot` each node admits on joint velocity alone.

        `dq_a = q_a'(sigma) sigma_dot`, so a velocity limit is a bound on the
        path rate and not a nonlinear row. At the endpoints `q_a' = 0` and
        nothing bounds it, which is why `sigma_rate_max` exists.
        """
        rate = np.abs(blocks[:, 5:10])
        allowed = self.config.kappa * speed_scale * self.limits.dq_max
        ceiling = np.full(len(blocks), speed_scale * self.config.sigma_rate_max)
        for node in range(len(blocks)):
            moving = rate[node] > 1.0e-9
            if np.any(moving):
                ceiling[node] = min(
                    ceiling[node], float(np.min(allowed[moving] / rate[node][moving]))
                )
        return np.maximum(ceiling, 2.0 * self.config.sigma_rate_min)

    def _guess(
        self,
        blocks: np.ndarray,
        ceiling: np.ndarray,
        equilibrium: np.ndarray,
        payload: np.ndarray,
        sigma_dot_start: float | None,
        accel_max: np.ndarray,
        flow_max: float,
    ) -> np.ndarray:
        """
        Build a path-rate profile that is already close to admissible.

        Three passes, and all three earn their place. The velocity ceiling says
        nothing about acceleration, so the classical `sqrt(ddq_max / |q_a''|)`
        curve is imposed on top of it. Neither says anything about cylinder
        force or pump flow, which are what actually bind on a lift, so the rate
        is bisected down until the worst of them sits at 90% of its allowance --
        90% and not 100% because a barrier method started with a dozen rows at
        zero slack is a barrier method that does not start. And per-node
        feasibility is not reachability, so a forward and a backward sweep under
        `d(sigma_dot^2)/dsigma = 2 sigma_ddot` connect the nodes to each other.
        """
        guess = ceiling.copy()
        curvature = np.abs(blocks[:, 10:15])
        allowed = self.config.kappa * accel_max
        for node in range(len(blocks)):
            turning = curvature[node] > 1.0e-9
            if np.any(turning):
                guess[node] = min(
                    guess[node],
                    float(np.min(np.sqrt(allowed[turning] / curvature[node][turning]))),
                )

        def demand(node: int, rate: float) -> float:
            state = np.concatenate([[rate], equilibrium[node], np.zeros(2)])
            _, force, flow = self.demand(state, 0.0, blocks[node], payload)
            force = np.abs(np.array(force).ravel()) / (
                self.config.kappa * self.limits.force_max
            )
            return max(
                float(np.max(force)),
                float(abs(float(flow))) / (self.config.kappa * flow_max),
            )

        floor = 2.0 * self.config.sigma_rate_min
        for node in range(len(blocks)):
            if demand(node, guess[node]) <= 0.9:
                continue
            low, high = floor, guess[node]
            for _ in range(12):
                middle = 0.5 * (low + high)
                if demand(node, middle) <= 0.9:
                    low = middle
                else:
                    high = middle
            guess[node] = low

        step = 2.0 * self.config.sigma_accel_max / (len(blocks) - 1)
        if sigma_dot_start is not None:
            guess[0] = sigma_dot_start
        for node in range(1, len(guess)):
            guess[node] = min(guess[node], np.sqrt(guess[node - 1] ** 2 + step))
        # The backward sweep stops one short of a pinned start: sigma_dot(0) is
        # measured, not chosen.
        last = 0 if sigma_dot_start is not None else -1
        for node in range(len(guess) - 2, last, -1):
            guess[node] = min(guess[node], np.sqrt(guess[node + 1] ** 2 + step))
        guess = np.maximum(guess, floor)
        if sigma_dot_start is not None:
            guess[0] = sigma_dot_start
        return guess

    def solve(
        self,
        path: Path,
        q_tool: float,
        payload_vector: np.ndarray,
        equilibrium: np.ndarray,
        q_u_start: np.ndarray,
        dq_u_start: np.ndarray,
        sigma_dot_start: float | None,
        speed_scale: float,
    ) -> Timing:
        """Solve the OCP for this path, or refuse with what the solver said."""
        config = self.config
        intervals = int(config.intervals)
        span = 1.0 / intervals
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        blocks = self._blocks(path, q_tool, nodes)
        middles = self._blocks(path, q_tool, nodes[:-1] + 0.5 * span)
        # What a speed scale scales. It is the caller's own divider on the
        # machine's allowance, so it takes every *rate* budget with it: joint
        # velocity, the path rate itself, and the pump. Acceleration goes with
        # its square, because along a fixed path `ddq_a` is quadratic in the
        # rate. The cylinder force allowance is deliberately untouched --
        # gravity does not slow down, and a lift that needs the force standing
        # still needs it at half speed too.
        #
        # Scaling the pump is what makes the divider mean anything on this
        # machine: these moves are flow-limited long before they are
        # velocity-limited, so a scale that only touched the joint speeds would
        # hand a caller asking for half speed the very same trajectory back.
        accel_max = speed_scale**2 * config.ddq_a_max
        flow_max = speed_scale * self.limits.flow_max
        ceiling = self._ceiling(blocks, speed_scale)
        if sigma_dot_start is not None and sigma_dot_start > ceiling[0]:
            raise PlanningError(
                "the machine is already moving faster along this path than its joint "
                "velocity limits allow it to continue"
            )

        # The gravity load does not fall with speed, so a path that overloads a
        # cylinder standing still has no timing at all -- and saying so here is
        # worth more than a solver that reports infeasibility.
        for node in range(intervals + 1):
            state = np.concatenate(
                [[config.sigma_rate_min], equilibrium[node], np.zeros(2)]
            )
            _, force, _ = self.demand(state, 0.0, blocks[node], payload_vector)
            excess = (
                np.abs(np.array(force).ravel()) - config.kappa * self.limits.force_max
            )
            if np.any(excess > 0.0):
                axis = int(np.argmax(excess))
                raise PlanningError(
                    f"the path is not liftable: holding it at sigma = {nodes[node]:.2f} "
                    f"already asks cylinder {axis} for "
                    f"{float(np.abs(np.array(force).ravel())[axis]) * 1e-3:.1f} kN against "
                    f"{config.kappa * self.limits.force_max[axis] * 1e-3:.1f} kN allowed"
                )

        opti = ca.Opti()
        state = opti.variable(5, intervals + 1)
        control = opti.variable(intervals)
        sigma_dot, q_u, dq_u = state[0, :], state[1:3, :], state[3:5, :]

        cost = 0.0
        for node in range(intervals):
            step = span
            here, there = state[:, node], state[:, node + 1]
            k1 = self.derivative(here, control[node], blocks[node], payload_vector)
            k2 = self.derivative(
                here + 0.5 * step * k1, control[node], middles[node], payload_vector
            )
            k3 = self.derivative(
                here + 0.5 * step * k2, control[node], middles[node], payload_vector
            )
            k4 = self.derivative(
                here + step * k3, control[node], blocks[node + 1], payload_vector
            )
            opti.subject_to(there == here + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

            # Traversal time, sway and a regularisation on the input, all as
            # integrals over sigma: dt = dsigma / sigma_dot is the only reason
            # the first term is the objective it is.
            cost += span * (
                0.5 * (1.0 / sigma_dot[node] + 1.0 / sigma_dot[node + 1])
                + config.sway_weight * ca.sumsqr(dq_u[:, node]) / sigma_dot[node]
                + config.input_weight * control[node] ** 2
            )

        # What the answer demands of the machine, at **every** node including the
        # last. The terminal node carries no input of its own, but it does not
        # need one: the path arrives at rest in q_a, so `q_a'(1) sigma_ddot`
        # vanishes and `ddq_a = q_a''(1) sigma_dot^2` is well defined -- and it
        # is the row that decides how fast the machine may still be travelling
        # when it gets there. Leaving the last node out is how a plan ends with
        # nine times the admissible deceleration.
        for node in range(intervals + 1):
            here = state[:, node]
            input_ = control[min(node, intervals - 1)]
            ddq_a, force, flow = self.demand(here, input_, blocks[node], payload_vector)
            allowed = config.kappa * accel_max
            opti.subject_to(opti.bounded(-allowed, ddq_a, allowed))
            allowed = config.kappa * self.limits.force_max
            opti.subject_to(opti.bounded(-allowed, force, allowed))
            opti.subject_to(opti.bounded(0.0, flow, config.kappa * flow_max))

        opti.subject_to(
            opti.bounded(-config.sigma_accel_max, control, config.sigma_accel_max)
        )
        for node in range(intervals + 1):
            opti.subject_to(
                opti.bounded(config.sigma_rate_min, sigma_dot[node], ceiling[node])
            )
            opti.subject_to(
                opti.bounded(
                    equilibrium[node] - config.q_sway_max,
                    q_u[:, node],
                    equilibrium[node] + config.q_sway_max,
                )
            )
            opti.subject_to(
                opti.bounded(-config.dq_sway_max, dq_u[:, node], config.dq_sway_max)
            )

        opti.subject_to(q_u[:, 0] == q_u_start)
        opti.subject_to(dq_u[:, 0] == dq_u_start)
        if sigma_dot_start is not None:
            opti.subject_to(sigma_dot[0] == sigma_dot_start)
        # The tool arrives hanging and still, which is the whole reason the sway
        # is a state of this problem rather than something checked afterwards.
        opti.subject_to(q_u[:, intervals] == equilibrium[intervals])
        opti.subject_to(dq_u[:, intervals] == 0.0)

        opti.minimize(cost)
        guess = self._guess(
            blocks,
            ceiling,
            equilibrium,
            payload_vector,
            sigma_dot_start,
            accel_max,
            flow_max,
        )
        opti.set_initial(sigma_dot, guess)
        opti.set_initial(q_u, equilibrium.T)
        opti.set_initial(dq_u, np.zeros((2, intervals + 1)))
        opti.set_initial(control, np.zeros(intervals))
        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": int(config.max_iterations),
                "max_cpu_time": float(config.max_wall_clock),
                "tol": float(config.tolerance),
                "acceptable_tol": float(config.tolerance) * 1.0e2,
            },
        )

        started = time.monotonic()
        try:
            answer = opti.solve()
        except RuntimeError as failure:
            raise PlanningError(
                f"no admissible timing exists for this path: {opti.stats().get('return_status', failure)}"
            ) from None
        elapsed = time.monotonic() - started

        rate = np.array(answer.value(sigma_dot)).ravel()
        accel = np.array(answer.value(control)).ravel()
        sway = np.array(answer.value(q_u)).reshape(2, -1).T
        sway_rate = np.array(answer.value(dq_u)).reshape(2, -1).T

        ddq_a = np.zeros((intervals + 1, PLANNED_DOF))
        force = np.zeros((intervals + 1, PLANNED_DOF))
        flow = np.zeros(intervals + 1)
        for node in range(intervals + 1):
            here = np.concatenate([[rate[node]], sway[node], sway_rate[node]])
            values = self.demand(
                here, accel[min(node, intervals - 1)], blocks[node], payload_vector
            )
            ddq_a[node] = np.array(values[0]).ravel()
            force[node] = np.array(values[1]).ravel()
            flow[node] = float(values[2])

        time_of = np.concatenate(
            [[0.0], np.cumsum(span * 0.5 * (1.0 / rate[:-1] + 1.0 / rate[1:]))]
        )
        return Timing(
            sigma=nodes,
            sigma_dot=rate,
            sigma_ddot=np.append(accel, accel[-1]),
            q_u=sway,
            dq_u=sway_rate,
            q_u_eq=equilibrium,
            q_a=blocks[:, 0:5],
            dq_a=blocks[:, 5:10] * rate[:, None],
            ddq_a=ddq_a,
            cylinder_force=force,
            pump_flow=flow,
            time=time_of,
            iterations=int(opti.stats().get("iter_count", 0)),
            solve_time_s=elapsed,
        )


# ------------------------------------------------------------------ the planner


@dataclass
class Start:
    """The measured state a plan leaves from."""

    q: np.ndarray  # the canonical eight
    dq_a: np.ndarray  # the planned five
    dq_u: np.ndarray = field(default_factory=lambda: np.zeros(2))  # the sway rate
    passive_measured: bool = True

    @property
    def q_a(self) -> np.ndarray:
        return self.q[list(PLANNED_INDICES)]

    @property
    def q_tool(self) -> float:
        return float(self.q[TOOL_INDEX])

    @property
    def q_u(self) -> np.ndarray:
        return self.q[list(PASSIVE_INDICES)]


@dataclass
class Plan:
    """A trajectory, the geometry it was found on, and how it was arrived at."""

    time: np.ndarray
    q: np.ndarray  # the canonical eight, resampled at Ts, one row per sample
    dq: np.ndarray
    tcp: np.ndarray  # tool position in K0_mounting_base, one row per sample
    timing: Timing
    message: str

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def q_a(self) -> np.ndarray:
        """The actuated six, which is what the native reference carries."""
        return self.q[:, list(ACTUATED_INDICES)]

    @property
    def dq_a(self) -> np.ndarray:
        return self.dq[:, list(ACTUATED_INDICES)]


class Planner:
    """
    One description, one tool, one set of limits, many requests.

    Built once from the robot description -- parsing it, deriving the force
    limits from the hydraulics and code-generating the OCP's functions all
    happen here, not per request.
    """

    def __init__(self, robot_description_xml: str, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.model = CraneModel(robot_description_xml, self.config.tool)
        self.symbolic = symbolic_model.CraneSymbolicModel(
            robot_description_xml, self.config.tool.value
        )
        self.limits = read_limits(
            robot_description_xml,
            self.config.tool,
            self.symbolic,
            self.config.system_pressure_pa,
            self.config.pump_flow_max,
            self.config.pump_flow_planning_factor,
        )
        self.equilibrium = Equilibrium(self.symbolic, self.limits)
        self.ocp = TimingOcp(self.symbolic, self.limits, self.config)

    def plan(
        self,
        start: Start,
        goal_position_m: np.ndarray,
        goal_yaw: float,
        payload: Payload | None = None,
        payload_shape=None,
        scene: list | None = None,
        avoid_collisions: bool = True,
        speed_scale: float = 1.0,
    ) -> Plan:
        """
        Answer a placement goal in `K0_mounting_base` with a timed trajectory.

        The stages refuse; they do not degrade. A goal that cannot be reached, a
        path that cannot be found or smoothed, and a path that cannot be timed
        are three different answers, and each one names itself.
        """
        if not 0.0 < speed_scale <= 1.0:
            raise PlanningError(f"speed_scale must be in (0, 1], not {speed_scale}")
        if avoid_collisions and scene is None:
            raise PlanningError(
                "avoid_collisions was asked for with no collision scene: a plan "
                "certified against nothing is not certified"
            )

        primitives = self.prepare_scene(scene, avoid_collisions)
        payload_vector = payload_parameters(payload)
        geometry = Geometry(
            self.model,
            self.equilibrium,
            primitives,
            self.config,
            start.q_tool,
            payload_vector,
            payload_shape if avoid_collisions else None,
        )

        goal = solve_ik(
            geometry,
            self.limits,
            self.config,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            start.q_a,
        )
        if not geometry.is_valid(start.q_a):
            raise PlanningError("the measured start configuration is in collision")

        waypoints = search(geometry, self.limits, self.config, start.q_a, goal)
        path, sigma_dot_start = fit(geometry, self.limits, waypoints, start.dq_a)
        geometry.check_path(path)

        nodes = np.linspace(0.0, 1.0, int(self.config.intervals) + 1)
        equilibrium = np.array(
            [
                geometry.configuration(path.position(sigma))[list(PASSIVE_INDICES)]
                for sigma in nodes
            ]
        )
        q_u_start = start.q_u if start.passive_measured else equilibrium[0]
        dq_u_start = (
            np.asarray(start.dq_u, dtype=float)
            if start.passive_measured
            else np.zeros(2)
        )
        if np.any(np.abs(q_u_start - equilibrium[0]) > self.config.q_sway_max):
            raise PlanningError(
                "the tool is swinging further than the admissible sway bound, so "
                "there is no plan that keeps it inside one"
            )
        if np.any(np.abs(dq_u_start) > self.config.dq_sway_max):
            raise PlanningError(
                "the tool is swinging faster than the admissible sway rate, so there is "
                "no plan that starts from it"
            )

        timing = self.ocp.solve(
            path,
            start.q_tool,
            payload_vector,
            equilibrium,
            q_u_start,
            dq_u_start,
            sigma_dot_start,
            speed_scale,
        )
        return self._resample(geometry, path, timing, start)

    def prepare_scene(self, scene, avoid_collisions: bool) -> list:
        """
        Return the static bodies this plan is checked against.

        Not the same list the request carried: the reserved `truck` primitive
        has become a bed, six runges and a headboard by the time the planner
        looks at it, and that is invisible to anyone who only sees the scene
        topic. This is a
        method and not a private step because the node draws it, and a refusal
        is far easier to read beside the geometry that caused it.

        What the tool carries is **not** here. It moves, so it is placed at each
        configuration checked rather than pinned to the scene once.
        """
        if not avoid_collisions:
            return []
        primitives = expand_truck(list(scene or []), self.config)
        if any(primitive.id == PAYLOAD_ID for primitive in primitives):
            raise PlanningError(f"'{PAYLOAD_ID}' is a reserved scene id")
        return primitives

    def _settled(
        self, q_a: np.ndarray, q_tool: float, payload: np.ndarray
    ) -> np.ndarray:
        """Return the canonical eight at `q_a` with the passive pair hanging."""
        q = np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)] = q_a
        q[TOOL_INDEX] = q_tool
        q[list(PASSIVE_INDICES)] = self.equilibrium.solve(q_a, q_tool, payload)
        return q

    def _tcp_yaw(self, q: np.ndarray) -> float:
        pose = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        return yaw_of(
            pin.XYZQUATToSE3(
                np.concatenate([pose.position_m, pose.orientation_xyzw])
            ).rotation
        )

    def tip_to_tcp_offset(
        self, payload: Payload | None, yaw: float, q_tool: float
    ) -> np.ndarray:
        """
        Where the tool hangs relative to the tip pivot K5, for one payload and yaw.

        The retained `a2b_movement` goal names the pivot and the native goal
        names the tool, so the adapter needs the vector between them. It is read
        out of the model at the hanging equilibrium and never written down: the
        PZS100's rail gripper and the 7040's jaw do not hang at the same offset,
        and a centre of mass off the tool axis changes it again.

        The two passive joints make the settled offset independent of the boom
        and telescope pose -- once the pendulum is settled, only rotation about
        gravity moves this vector. So the description's own yaw convention is
        measured at a canonical pose, the slew is turned by the difference, the
        pendulum is settled again, and the result is *checked* to carry the
        requested yaw rather than assumed to.
        """
        if not (np.isfinite(yaw) and np.isfinite(q_tool)):
            raise PlanningError(
                "`phi_tool_n` or the tool coordinate q8 is not finite, so the hanging "
                "tip-to-tool offset cannot be evaluated"
            )
        vector = payload_parameters(payload)
        q_a = np.zeros(PLANNED_DOF)
        canonical = self._settled(q_a, q_tool, vector)
        q_a[0] = _wrap(yaw - self._tcp_yaw(canonical))
        q = self._settled(q_a, q_tool, vector)

        error = _wrap(self._tcp_yaw(q) - yaw)
        if abs(error) > 1.0e-9:
            raise PlanningError(
                f"the settled pose used to translate `y_n` misses `phi_tool_n` by "
                f"{error:.3e} rad, so its tip-to-tool offset cannot be applied without "
                "approximation"
            )
        tip = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TIP)
        tcp = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        offset = tcp.position_m - tip.position_m
        if not (np.all(np.isfinite(offset)) and np.linalg.norm(offset) > 0.0):
            raise PlanningError(
                "the description produced no finite non-zero offset from the tip pivot to "
                "the tool centre point, so `y_n` cannot be placed on the native tool goal"
            )
        return offset

    def _resample(self, geometry, path, timing, start: Start) -> Plan:
        """
        Put the answer on the emitted reference's own clock.

        The OCP works on a uniform sigma grid, which is not a uniform time grid;
        the consumer reads at `Ts`. Sigma is interpolated against the elapsed
        time the solve produced, and the joint rates are re-derived from
        `q_a'(sigma) sigma_dot` there rather than interpolated -- interpolating a
        derivative and its integral separately is how the two stop agreeing.

        The passive pair is carried too, and it is the sway the OCP **planned**
        rather than the pose the tool would settle to. It costs nothing -- the
        OCP solves for it either way -- and it is what the trajectory actually
        claims: on the way to the goal the tool is swinging, and a consumer
        tracking the passive joints as state wants the swing, not the rest pose.
        """
        Ts = float(self.config.Ts)
        stamps = np.arange(0.0, timing.duration + 0.5 * Ts, Ts)
        if stamps[-1] < timing.duration:
            stamps = np.append(stamps, timing.duration)
        sigma = np.interp(stamps, timing.time, timing.sigma)
        rate = np.interp(stamps, timing.time, timing.sigma_dot)

        q = np.zeros((len(stamps), GENERALIZED_DOF))
        dq = np.zeros_like(q)
        q[:, list(PLANNED_INDICES)] = path.position(sigma)
        q[:, TOOL_INDEX] = start.q_tool
        dq[:, list(PLANNED_INDICES)] = path.rate(sigma) * rate[:, None]
        for slot, index in enumerate(PASSIVE_INDICES):
            q[:, index] = np.interp(stamps, timing.time, timing.q_u[:, slot])
            dq[:, index] = np.interp(stamps, timing.time, timing.dq_u[:, slot])

        # The tool where the plan says it is, sway included -- not where it
        # would hang if the machine stopped at each sample.
        tcp = np.array(
            [
                geometry.model.forward_kinematics(
                    row, Frame.MOUNTING_BASE, Frame.TCP
                ).position_m
                for row in q
            ]
        )
        message = (
            f"{timing.duration:.2f} s over {path.travel_m:.2f} m of tool travel, "
            f"{len(stamps)} points at {Ts * 1e3:.0f} ms; IPOPT converged in "
            f"{timing.iterations} iterations and {timing.solve_time_s:.2f} s; "
            f"peak force {np.max(np.abs(timing.cylinder_force) / self.limits.force_max):.2f} "
            f"and peak pump draw {np.max(timing.pump_flow) / self.limits.flow_max:.2f} "
            f"of the physical limit, at kappa = {self.config.kappa}"
        )
        return Plan(time=stamps, q=q, dq=dq, tcp=tcp, timing=timing, message=message)


def payload_parameters(payload: Payload | None) -> np.ndarray:
    """Mass, centre of mass in K8 and the six independent entries of Theta_L."""
    values = np.zeros(symbolic_model.NP - 1)
    if payload is None or not payload.valid:
        return values
    values[0] = float(payload.mass_kg)
    values[1:4] = np.asarray(payload.center_of_mass_k8_m, dtype=float)
    inertia = np.asarray(payload.inertia_k8_kg_m2, dtype=float).reshape(3, 3)
    values[4:] = [
        inertia[row, column] for row, column in symbolic_model.INERTIA_ENTRIES
    ]
    return values
