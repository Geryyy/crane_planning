"""
Plan a motion, and time it.

Two stages. The tool centre point runs a straight line from where it is to where
the goal asks for it; every sample of that line is lifted into the five planned
joint coordinates by inverse kinematics, and the whole machine is checked there.
One CasADi/IPOPT optimal control problem over the path parameter then decides how
fast the machine may traverse the result.

Everything the machine can do -- reach, hang, collide -- is asked of
`crane_model`; nothing about the machine is written down here.

The passive pair (tip/tilt sway) is never searched over. Where it hangs is a
closed form -- `q_eq = (pi/2 - q_boom - q_arm, pi/2)`, which
`crane_mpc/src/mpc_node.cpp` measured to 1e-4 rad across the workspace and found
independent of slew, telescope, rotator, tool and payload -- and it is a *state*
of the timing OCP, so the answer is a trajectory the tool arrives nearly still
from.
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
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.optimize import least_squares

#: The five coordinates a plan moves, in canonical indices. The tool axis (q8)
#: is held by the low-level controller and rides the path at a constant value.
PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]
PLANNED_DOF = len(PLANNED_INDICES)

#: The frame each planned coordinate turns about, for the step bound below. The
#: telescope is prismatic and has no radius, which is what `None` marks.
PLANNED_FRAMES = (
    Frame.SLEWING_COLUMN,
    Frame.BOOM,
    Frame.ARM,
    None,
    Frame.ROTATOR,
)
TELESCOPE_AXIS = 3

#: Metres of tip travel per metre of `q4`. **Two, not one.** `q4` is stage-1
#: travel and `q5_small_telescope` mimics it at multiplier 1, so a chain couples
#: the second stage and the tip advances `2 q4` -- `wiki/nomenclature.md` 4. A
#: step bound built on 1.0 lets a telescope-dominated segment move the tool
#: twice as far as it is checked for, which is the whole margin.
TELESCOPE_TRAVEL_PER_UNIT = 2.0

#: `q_a | q_a' | q_a'' | q8` -- what the OCP needs to know about the path at one
#: value of sigma, and the only thing it is told about it.
PATH_BLOCK = 3 * PLANNED_DOF + 1

#: Where the adaptive march starts, and the step at which it gives up. The
#: first is only a guess -- it halves on the first violation and grows back
#: over a clear run -- and the second is what a branch change runs into.
INITIAL_LIFT_STEP = 1.0 / 32.0
MIN_LIFT_STEP = 1.0e-6

#: The two rows of the boom four-bar the hanging pose tracks, in planned axes.
BOOM_AXIS = 1
ARM_AXIS = 2


class PlanningError(RuntimeError):
    """A refusal. The message is what the service answer carries."""


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    """What the machine may do, read from the description."""

    lower: np.ndarray  # position, per planned coordinate; -inf where continuous
    upper: np.ndarray
    bounded: np.ndarray  # False for a `continuous` joint, which has no range
    dq_max: np.ndarray  # velocity, per planned coordinate
    tau_max: np.ndarray  # rated actuated effort, per planned coordinate
    flow_max: float  # summed pump draw
    tool_lower: float  # held tool-coordinate position
    tool_upper: float
    tool_bounded: bool


def read_limits(
    description_xml: str,
    tool: Tool,
    pump_flow_max: float,
    pump_flow_planning_factor: float,
) -> Limits:
    """
    Position and velocity out of the description; the pump out of the config.

    A `continuous` joint occupies two configuration slots (a cos/sin pair) and
    carries no range; the rotator is one, so it is reported unbounded rather than
    given an invented range here.

    There is deliberately no cylinder-force *constraint*. It was the smaller
    chamber area times a relief pressure nothing in this workspace has measured.
    `tau_max` is the description's own `effort` on each planned joint and is not
    a substitute for it: nothing is bounded by it, it is the scale the OCP's
    effort term is divided by, so that `tau_weight` means "a fraction of rated
    effort" on every axis alike.
    """
    description = parse(description_xml, tool)
    inner = description.model
    lower = np.empty(PLANNED_DOF)
    upper = np.empty(PLANNED_DOF)
    bounded = np.empty(PLANNED_DOF, dtype=bool)
    dq_max = np.empty(PLANNED_DOF)
    tau_max = np.empty(PLANNED_DOF)
    for axis, index in enumerate(PLANNED_INDICES):
        slot = description.drives[index][0].slot
        bounded[axis] = slot.nq == 1
        lower[axis] = inner.lowerPositionLimit[slot.idx_q] if slot.nq == 1 else -np.inf
        upper[axis] = inner.upperPositionLimit[slot.idx_q] if slot.nq == 1 else np.inf
        dq_max[axis] = inner.velocityLimit[slot.idx_v]
        tau_max[axis] = inner.effortLimit[slot.idx_v]
    if not np.all(np.isfinite(dq_max)) or np.any(dq_max <= 0.0):
        raise PlanningError(
            "the description gives a planned joint no finite positive velocity limit"
        )
    if not np.all(np.isfinite(tau_max)) or np.any(tau_max <= 0.0):
        raise PlanningError(
            "the description gives a planned joint no finite positive effort limit, "
            "which is what the OCP's effort term is measured against"
        )
    tool_slot = description.drives[TOOL_INDEX][0].slot
    tool_bounded = tool_slot.nq == 1
    tool_lower = (
        float(inner.lowerPositionLimit[tool_slot.idx_q]) if tool_bounded else -np.inf
    )
    tool_upper = (
        float(inner.upperPositionLimit[tool_slot.idx_q]) if tool_bounded else np.inf
    )
    return Limits(
        lower=lower,
        upper=upper,
        bounded=bounded,
        dq_max=dq_max,
        tau_max=tau_max,
        flow_max=float(pump_flow_max) * float(pump_flow_planning_factor),
        tool_lower=tool_lower,
        tool_upper=tool_upper,
        tool_bounded=tool_bounded,
    )


# --------------------------------------------------------------------- config


@dataclass
class PlannerConfig:
    """Properties of the solve. Machine numbers are not among them."""

    tool: Tool = Tool.PZS100

    # Reservation held back from every physical limit so the controller has
    # authority left to correct with.
    kappa: float = 0.8

    # Endpoint IK acceptance, in metres and radians. `ik_restarts` applies to
    # the goal alone; along the line the previous sample is the seed.
    eps_pos: float = 1.0e-3
    eps_yaw: float = 1.0e-3
    ik_restarts: int = 5

    # --- what "clear" means, in metres ---------------------------------------
    #
    # The three-way split of the margin. `margin_safety` is the clearance an
    # answer actually carries; `margin_interp` is what pays for the gap between
    # two checked configurations; the swing envelope is computed, not configured.
    # Each is spent once. See `Geometry`.
    margin_safety: float = 0.05
    margin_interp: float = 0.10
    #: How far the tool's collision geometry reaches past the tool centre point.
    #: It enters the step bound as an over-estimate of the machine's outermost
    #: point, so too large costs samples and too small is unsound.
    tool_radius: float = 1.0
    #: The cap on lifted samples. Reaching it means the arm reconfigures faster
    #: than the line resolves -- usually an IK branch jump -- and is a refusal.
    max_lift_samples: int = 4096

    # Bounded crane-specific alternatives to the direct tool line.
    corridor_clearance: float = 0.15
    corridor_height_step: float = 0.35
    corridor_height_samples: int = 3
    corridor_lateral_step: float = 0.50
    corridor_lateral_samples: int = 2

    q_sway_max: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2]))

    #: How many cubic segments the path is fitted with, whatever the sample
    #: count. This is the dial that decouples the two things the old
    #: interpolating fit welded together: the collision certificate wants
    #: samples dense, and the curve wants its end intervals long. Approximating
    #: with far fewer control points than samples gives both.
    path_segments: int = 12

    # --- the timing OCP ------------------------------------------------------
    intervals: int = 40
    sway_weight: float = 2.0
    tau_weight: float = 0.1
    input_weight: float = 1.0e-3
    #: Terminal sway, as a cost and not an equality. One control -- the path
    #: rate -- cannot in general drive four terminal quantities to zero along a
    #: path it may not leave, and asking it to as a hard row is a refusal rather
    #: than a slow trajectory. Normalised by the admissible bounds, so this
    #: weight is a multiple of "the whole box".
    terminal_sway_weight: float = 200.0
    terminal_q_sway_max: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02])
    )
    terminal_dq_sway_max: np.ndarray = field(
        default_factory=lambda: np.array([0.04, 0.04])
    )
    #: The pump row carries L1 slack for the same reason: a move that needs
    #: 5% more flow than the reservation allows should come back slower and say
    #: so, not come back as `Infeasible_Problem_Detected`.
    flow_slack_weight: float = 1.0e3
    sigma_rate_min: float = 0.02
    sigma_rate_max: float = 2.0
    sigma_accel_max: float = 20.0
    dq_sway_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5]))
    ddq_a_max: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.7, 0.5, 1.0, 6.0])
    )
    max_wall_clock: float = 6.0
    max_iterations: int = 400
    tolerance: float = 1.0e-6

    # The pump, which the description does not carry.
    pump_flow_max: float = 1.4e-3
    pump_flow_planning_factor: float = 0.95

    # The emitted reference period.
    Ts: float = 0.04

    # The truck, as a property of the vehicle: the scene carries one primitive
    # with the reserved id `truck` and the bed and six runges are placed on it.
    truck_runge_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.28, 0.31, 2.12])
    )
    truck_runge_stations: np.ndarray = field(
        default_factory=lambda: np.array([-2.261, -1.049, 1.935])
    )
    truck_bed_thickness: float = 0.10
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


def passive_equilibrium(q_a: np.ndarray) -> np.ndarray:
    """
    Where the tool hangs, in closed form.

    A two-hinge pendulum under gravity hangs straight down, so the tip joint
    takes up whatever the boom four-bar accumulated and the tilt joint does not
    move at all. `crane_mpc/src/mpc_node.cpp` measured this against the general
    5x5-grid-plus-Newton solve and found it good to 1e-4 rad across the
    workspace, and independent of slew, telescope, rotator, tool and payload --
    against a sway box half-width of 0.2 rad.

    That solve costs 22.6 ms and this costs two subtractions, which is what makes
    it affordable to settle the pendulum at every configuration the lift checks.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])


# ------------------------------------------------------------------- geometry


class Geometry:
    """
    Whether a configuration is clear, at the pose the tool hangs at.

    Every distance is a `crane_model` query -- there is no second collision model
    here.

    # How a finite set of checks certifies a continuous motion

    Sampling never proves clearance on its own. What makes it a proof is a
    margin paired with a bound on how far anything can move between two samples.
    Rotating planned joint `j` by `dq_j` displaces any point below it by at most
    `radii[j] * |dq_j|`, so between two configurations no point of the machine
    moves further than `step_bound`. Therefore:

        if every checked configuration clears the scene by more than
        `required`, and `step_bound <= margin_interp` for every consecutive
        pair, then no point of the machine touches anything anywhere on the
        continuous motion between them.

    `required` is spent three ways and each part is spent once:

        required = margin_safety + margin_interp + envelope

    `margin_interp` bridges the samples, `envelope` is how far the tool can swing
    inside the admissible sway box, and `margin_safety` is what is actually left
    over as clearance. The swing is therefore answered by one distance test
    rather than by gridding the sway box, and a thin obstacle cannot be tunnelled
    -- which is what the old "tighten the step to half the thinnest primitive"
    rule was reaching for without being able to prove.

    Self-collision keeps its own test at zero: the machine's links are near each
    other by design, and holding them a margin apart would refuse poses it is
    built to reach.
    """

    def __init__(
        self,
        model: CraneModel,
        limits: Limits,
        scene: list,
        config: PlannerConfig,
        q_tool: float,
        payload: np.ndarray,
        payload_shape=None,
    ):
        self.model = model
        self.limits = limits
        self.scene = list(scene)
        self.config = config
        self.q_tool = float(q_tool)
        self.payload = payload
        self.payload_shape = payload_shape
        self.envelope = self._envelope()
        self.required = (
            float(config.margin_safety) + float(config.margin_interp) + self.envelope
        )
        self.radii = self._radii()

    def bodies(self, q: np.ndarray) -> list:
        """Return the scene at `q`, with what the tool carries placed on it."""
        carried = payload_primitive(self.model, q, self.payload_shape)
        return self.scene if carried is None else self.scene + [carried]

    def _carried_reach(self) -> float:
        """How far past the tool centre point the carried body reaches, in metres."""
        if self.payload_shape is None:
            return 0.0
        _shape, dimensions, offset = self.payload_shape
        return float(
            np.linalg.norm(np.asarray(offset, dtype=float))
            + np.linalg.norm(np.asarray(dimensions, dtype=float))
        )

    def _envelope(self) -> float:
        """
        Return how far the farthest carried point can swing, in metres.

        Measured from the passive pivot to that point and **not to the tool
        centre point**: a gripped block hangs below the TCP, so it swings on a
        longer pendulum than the tool does and takes up more room for the same
        `q_sway_max`. Everything between the pivot and the TCP is closer to the
        pivot and so swings less, which is why only what is carried is added.

        The sway box is shared with the controller by design, so this is not a
        conservative approximation of a swing that will not happen -- it is a
        state the machine is entitled to reach, and geometry that was not
        checked for it was not checked.
        """
        q = np.zeros(GENERALIZED_DOF)
        q[TOOL_INDEX] = self.q_tool
        hinge = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TILT)
        tcp = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        length = float(np.linalg.norm(tcp.position_m - hinge.position_m))
        length += self._carried_reach()
        return length * float(np.sin(np.max(np.abs(self.config.q_sway_max))))

    def _radii(self) -> np.ndarray:
        """
        How far a body can move per unit of each planned coordinate.

        Measured at full telescope extension, which is where every radius is
        largest. The outermost point of the machine is taken to be the tool
        centre point plus `tool_radius` and the carried body's own reach, and the
        distance is taken from the joint's *origin* rather than its axis -- both
        over-estimate, which is the safe direction for a bound that has to hold.
        The telescope is prismatic, and displaces what it carries by
        `TELESCOPE_TRAVEL_PER_UNIT` -- two metres per metre of `q4`, not one.
        """
        q = np.zeros(GENERALIZED_DOF)
        q[TOOL_INDEX] = self.q_tool
        extension = self.limits.upper[TELESCOPE_AXIS]
        if np.isfinite(extension):
            q[PLANNED_INDICES[TELESCOPE_AXIS]] = extension
        q[list(PASSIVE_INDICES)] = passive_equilibrium(q[list(PLANNED_INDICES)])
        tcp = self.model.forward_kinematics(
            q, Frame.MOUNTING_BASE, Frame.TCP
        ).position_m

        reach = float(self.config.tool_radius) + self._carried_reach()

        radii = np.ones(PLANNED_DOF)
        radii[TELESCOPE_AXIS] = TELESCOPE_TRAVEL_PER_UNIT
        for axis, frame in enumerate(PLANNED_FRAMES):
            if frame is None:
                continue
            origin = self.model.forward_kinematics(
                q, Frame.MOUNTING_BASE, frame
            ).position_m
            radii[axis] = float(np.linalg.norm(tcp - origin)) + reach
        return radii

    def step_bound(self, first: np.ndarray, second: np.ndarray) -> float:
        """Return the furthest any point of the machine moves between two configurations."""
        return float(
            np.sum(self.radii * np.abs(np.asarray(second) - np.asarray(first)))
        )

    def configuration(self, q_a: np.ndarray) -> np.ndarray:
        """Return the canonical eight at `q_a`, with the tool hanging."""
        q = np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)] = q_a
        q[TOOL_INDEX] = self.q_tool
        q[list(PASSIVE_INDICES)] = passive_equilibrium(q_a)
        return q

    def clearance(self, q_a: np.ndarray) -> float:
        """
        Scene clearance at `q_a`, in metres, or `-inf` where it cannot be evaluated.

        The self-collision row is separated out and tested at zero; what is
        returned is the distance to the scene, which is what the margin is
        against.
        """
        try:
            q = self.configuration(q_a)
            bodies = self.bodies(q)
            results = self.model.collision_queries(q, bodies)
        except (CraneModelError, PlanningError):
            return -np.inf
        # `crane_model.collision.queries` answers one row per scene primitive in
        # scene order, then -- if the description has any self-pairs at all --
        # one for the machine against itself.
        scene_rows = results[: len(bodies)]
        self_rows = results[len(bodies) :]
        if any(row.minimum_distance_m <= 0.0 for row in self_rows):
            return -np.inf
        return min((row.minimum_distance_m for row in scene_rows), default=np.inf)

    def is_valid(self, q_a: np.ndarray) -> bool:
        """Clear of the scene by the whole margin, and not folded into itself."""
        return self.clearance(q_a) > self.required

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

    def tcp_pose(self, q_a: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the tool centre point's position and yaw at `q_a`, hanging."""
        pose = self.model.forward_kinematics(
            self.configuration(q_a), Frame.MOUNTING_BASE, Frame.TCP
        )
        rotation = pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
        return pose.position_m, yaw_of(rotation)

    def check_path(self, path) -> None:
        """
        Re-run the certificate on the fitted curve, which is what executes.

        The polyline satisfied it by construction; the spline through it is a
        different curve, so it earns the same two tests -- clearance at every
        sample, and the step bound between consecutive ones. It is walked with
        the same adaptive step for the same reason: a sample count read off the
        tool's travel says nothing about how far the arm moved between two of
        them.
        """
        previous = path.position(0.0)
        if not self.is_valid(previous):
            raise PlanningError("the fitted path is blocked at its start")
        sigma, step, count = 0.0, INITIAL_LIFT_STEP, 1
        while sigma < 1.0:
            step = min(step, 1.0 - sigma)
            candidate = path.position(sigma + step)
            if self.step_bound(previous, candidate) > self.config.margin_interp:
                step *= 0.5
                if step < MIN_LIFT_STEP:
                    raise PlanningError(
                        f"the fitted path moves the machine faster than the "
                        f"{self.config.margin_interp:.3f} m interpolation margin "
                        f"resolves near sigma = {sigma:.3f}, so it is not certified"
                    )
                continue
            if not self.is_valid(candidate):
                raise PlanningError(
                    f"the fitted path is blocked at sigma = {sigma + step:.3f}"
                )
            previous, sigma = candidate, sigma + step
            count += 1
            if count > int(self.config.max_lift_samples):
                raise PlanningError(
                    f"the fitted path needs more than {self.config.max_lift_samples} "
                    "samples to certify"
                )
            step *= 1.25


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
    restarts: int = 1,
) -> np.ndarray:
    """
    Solve for the planned five that put the tool at `(position_m, yaw)`, hanging.

    One least-squares problem, not a closed form: the passive pair is re-settled
    at every configuration tested, so what is solved is where the tool actually
    ends up rather than where it would be if it did not hang. The redundancy the
    telescope leaves is taken up by a weak pull towards the seed.

    `restarts` is what separates the two callers. The goal is solved cold and
    spreads its restarts over the telescope range, because that is the coordinate
    the residual is flat in. Every sample along the line is solved with one
    restart from its predecessor, which is the right seed -- consecutive
    questions are about neighbouring poses -- and is also what keeps the lifted
    path on one IK branch.

    This checks reachability and nothing else. Whether the answer is clear is the
    caller's question, because the caller knows which sample it is.
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
        return np.concatenate(
            [
                pose.position_m - position_m,
                [_wrap(yaw_of(rotation) - yaw)],
                1.0e-3 * (q_a - seed),
            ]
        )

    def jacobian(q_a):
        """
        Return the residual's derivative, analytically rather than by differencing.

        Worth the two dozen lines: a five-parameter finite difference costs six
        forward-kinematics evaluations per iteration and this costs one
        Jacobian, which is why the lift stopped being the slowest stage.

        Two corrections turn the model's answer into this residual's. The
        Jacobian is LOCAL, so the linear and angular blocks are rotated into
        `K0_mounting_base`. And the passive pair is not independent: the
        pendulum hangs at `q_eq = (pi/2 - q_boom - q_arm, pi/2)`, so moving the
        boom or the arm swings the tip joint back by exactly as much, and the
        tip column enters those two with a factor of -1.

        The yaw row takes the angular Jacobian's z entry, which is exact only
        while the tool hangs near-upright. That costs a little convergence and
        nothing in correctness: the residual it is a derivative of stays exact,
        and the acceptance test below is on the residual.
        """
        try:
            q = geometry.configuration(q_a)
        except (CraneModelError, PlanningError):
            return np.zeros((PLANNED_DOF + 4, PLANNED_DOF))
        pose = geometry.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        rotation = pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
        full = geometry.model.jacobian(q, Frame.TCP).value
        linear = rotation @ full[:3, :]
        angular = rotation @ full[3:, :]

        out = np.zeros((PLANNED_DOF + 4, PLANNED_DOF))
        tip = PASSIVE_INDICES[0]
        for axis, column in enumerate(PLANNED_INDICES):
            coupling = -1.0 if axis in (BOOM_AXIS, ARM_AXIS) else 0.0
            out[0:3, axis] = linear[:, column] + coupling * linear[:, tip]
            out[3, axis] = angular[2, column] + coupling * angular[2, tip]
        out[4:, :] = 1.0e-3 * np.eye(PLANNED_DOF)
        return out

    if restarts <= 1:
        starts = [np.clip(np.asarray(seed, dtype=float), lower, upper)]
    else:
        starts = []
        for extension in np.linspace(
            lower[TELESCOPE_AXIS], upper[TELESCOPE_AXIS], restarts
        ):
            candidate = np.clip(np.asarray(seed, dtype=float).copy(), lower, upper)
            candidate[TELESCOPE_AXIS] = extension
            starts.append(candidate)

    best, best_error = starts[0], np.inf
    for start in starts:
        answer = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        error = float(np.linalg.norm(answer.fun[:4]))
        if error < best_error:
            best, best_error = answer.x, error
        # A solve that already meets the tolerance this function is about to
        # check is not in a local minimum, and the remaining restarts can only
        # re-derive an answer already in hand. It is taken as `best` outright:
        # the selection above mixes three metres with one radian in a single
        # 4-norm while the acceptance test is two separate scalars, so the
        # restart that satisfies the test need not be the one holding `best`.
        if (
            float(np.linalg.norm(answer.fun[:3])) <= config.eps_pos
            and float(abs(answer.fun[3])) <= config.eps_yaw
        ):
            return answer.x

    final = residual(best)
    position_error = float(np.linalg.norm(final[:3]))
    yaw_error = float(abs(final[3]))
    if position_error > config.eps_pos or yaw_error > config.eps_yaw:
        raise PlanningError(
            f"the tool cannot be placed at "
            f"({position_m[0]:.3f}, {position_m[1]:.3f}, {position_m[2]:.3f}) m / "
            f"{np.degrees(yaw):.1f} deg: the closest configuration misses it by "
            f"{position_error * 1e3:.1f} mm and {np.degrees(yaw_error):.2f} deg "
            f"(tolerances {config.eps_pos * 1e3:.1f} mm, "
            f"{np.degrees(config.eps_yaw):.2f} deg)"
        )
    return best


# ---------------------------------------------------------- geometric candidates


@dataclass(frozen=True)
class CartesianCandidate:
    """A named TCP polyline in the mounting-base frame."""

    name: str
    positions_m: tuple[np.ndarray, ...]


def _primitive_top(primitive) -> float:
    """Highest mounting-base z point of an oriented primitive's bounding box."""
    pose = primitive.pose_in_mounting_base
    extent = np.asarray(primitive.dimensions_m, dtype=float)
    half_height = 0.5 * float(np.sum(np.abs(pose.rotation[2, :]) * extent))
    return float(pose.translation[2]) + half_height


def cartesian_candidates(
    geometry: Geometry,
    config: PlannerConfig,
    start_position_m: np.ndarray,
    goal_position_m: np.ndarray,
) -> list[CartesianCandidate]:
    """Return the bounded deterministic set of tool corridors to try."""
    start = np.asarray(start_position_m, dtype=float)
    goal = np.asarray(goal_position_m, dtype=float)
    candidates = [CartesianCandidate("direct tool line", (start, goal))]

    transfer_z = max(float(start[2]), float(goal[2])) + float(
        config.corridor_clearance
    )
    if geometry.scene:
        transfer_z = max(
            transfer_z,
            max(_primitive_top(body) for body in geometry.scene)
            + float(config.corridor_clearance)
            + float(geometry.required),
        )

    horizontal = goal[:2] - start[:2]
    horizontal_length = float(np.linalg.norm(horizontal))
    perpendicular = (
        np.array([-horizontal[1], horizontal[0]]) / horizontal_length
        if horizontal_length > 1.0e-9
        else np.array([0.0, 1.0])
    )

    for level in range(max(1, int(config.corridor_height_samples))):
        height = transfer_z + level * float(config.corridor_height_step)
        start_high = np.array([start[0], start[1], height])
        goal_high = np.array([goal[0], goal[1], height])
        candidates.append(
            CartesianCandidate(
                f"lift/traverse/descend at z={height:.2f} m",
                (start, start_high, goal_high, goal),
            )
        )
        for lateral in range(1, max(0, int(config.corridor_lateral_samples)) + 1):
            distance = lateral * float(config.corridor_lateral_step)
            for sign, side in ((1.0, "left"), (-1.0, "right")):
                offset = sign * distance * perpendicular
                start_side = start_high.copy()
                goal_side = goal_high.copy()
                start_side[:2] += offset
                goal_side[:2] += offset
                candidates.append(
                    CartesianCandidate(
                        f"{side} corridor {distance:.2f} m at z={height:.2f} m",
                        (start, start_high, start_side, goal_side, goal_high, goal),
                    )
                )
    return candidates


def _without_repeated_positions(positions) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for position in positions:
        point = np.asarray(position, dtype=float)
        if not unique or np.linalg.norm(point - unique[-1]) > 1.0e-9:
            unique.append(point)
    return unique


def lift_candidate(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_yaw: float,
    candidate: CartesianCandidate,
) -> np.ndarray:
    """Lift one TCP polyline into joint space using continuation IK."""
    positions = _without_repeated_positions(candidate.positions_m)
    if len(positions) < 2:
        raise PlanningError(f"{candidate.name} contains no motion")
    _start_position, start_yaw = geometry.tcp_pose(start_q_a)
    yaw_sweep = _wrap(float(goal_yaw) - start_yaw)
    lengths = np.array(
        [
            np.linalg.norm(right - left)
            for left, right in zip(positions[:-1], positions[1:])
        ],
        dtype=float,
    )
    total_length = float(np.sum(lengths))
    if total_length <= 1.0e-9:
        lengths[:] = 1.0
        total_length = float(len(lengths))
    segment_starts = np.concatenate([[0.0], np.cumsum(lengths[:-1])]) / total_length

    waypoints = [np.asarray(start_q_a, dtype=float)]
    segments = len(lengths)
    for segment, (left, right, segment_length, progress_start) in enumerate(
        zip(positions[:-1], positions[1:], lengths, segment_starts), start=1
    ):
        fraction, step = 0.0, INITIAL_LIFT_STEP
        while fraction < 1.0:
            step = min(step, 1.0 - fraction)
            along = fraction + step
            position = (1.0 - along) * left + along * right
            progress = progress_start + along * segment_length / total_length
            q_a = solve_ik(
                geometry,
                limits,
                config,
                position,
                start_yaw + progress * yaw_sweep,
                waypoints[-1],
                restarts=1,
            )
            if not geometry.is_valid(q_a):
                raise PlanningError(
                    f"{candidate.name} is blocked on segment {segment}/{segments} "
                    f"at {along * 100.0:.0f}%: clearance {geometry.clearance(q_a):.3f} m "
                    f"against {geometry.required:.3f} m required"
                )
            if geometry.step_bound(waypoints[-1], q_a) > config.margin_interp:
                step *= 0.5
                if step < MIN_LIFT_STEP:
                    raise PlanningError(
                        f"the arm reconfigures faster than {candidate.name} resolves on "
                        f"segment {segment}/{segments} -- a singularity or IK branch change"
                    )
                continue
            waypoints.append(q_a)
            fraction += step
            if len(waypoints) > int(config.max_lift_samples):
                raise PlanningError(
                    f"{candidate.name} needs more than {config.max_lift_samples} "
                    "configurations to certify"
                )
            step *= 1.25
    return np.array(waypoints)


def plan_geometric_path(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_position_m: np.ndarray,
    goal_yaw: float,
) -> tuple[np.ndarray, str]:
    """Try the direct path and bounded crane-specific corridors in order."""
    start_position, _ = geometry.tcp_pose(start_q_a)
    failures = []
    candidates = cartesian_candidates(
        geometry, config, start_position, np.asarray(goal_position_m, dtype=float)
    )
    for candidate in candidates:
        try:
            waypoints = lift_candidate(
                geometry, limits, config, start_q_a, goal_yaw, candidate
            )
            return waypoints, candidate.name
        except PlanningError as failure:
            failures.append(str(failure))
    summary = "; ".join(failures[:3])
    if len(failures) > 3:
        summary += f"; and {len(failures) - 3} more corridor refusals"
    raise PlanningError(
        f"none of the {len(candidates)} deterministic tool corridors is clear: {summary}"
    )


def lift(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_position_m: np.ndarray,
    goal_yaw: float,
) -> np.ndarray:
    """Compatibility wrapper around the complete deterministic geometric stage."""
    waypoints, _candidate_name = plan_geometric_path(
        geometry, limits, config, start_q_a, goal_position_m, goal_yaw
    )
    return waypoints


# ---------------------------------------------------------------------- the fit


@dataclass
class Path:
    """
    q_a(sigma) on [0, 1], twice differentiable everywhere.

    Twice, and that is the whole point of fitting at all: the timing stage writes
    `ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot`, so at a kink in the polyline
    q_a'' is unbounded, the admissible path rate collapses to zero and the machine
    stops dead at every waypoint.
    """

    spline: BSpline
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


def fit(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    waypoints: np.ndarray,
    dq_start: np.ndarray,
) -> tuple[Path, float | None]:
    """
    Fit `q_a(sigma)` to the waypoints, and say what sigma_dot(0) must be.

    sigma is distributed by how long each chord takes at the slowest axis's own
    limit, so a segment that is slow in one coordinate gets more of the
    parameter. That total is a duration, and it is also what pins the start
    rate: choosing `q_a'(0) = dq_a * L` makes `dq_a = q_a'(0) sigma_dot(0)` hold
    exactly at `sigma_dot(0) = 1/L`.

    # Why this approximates instead of interpolating

    A cubic spline forced through every sample ties the number of curve
    segments to the number of samples, and those two want opposite things. The
    collision certificate wants samples **dense** -- that is what makes it a
    proof. The curve wants its first and last intervals **long**, because the
    path is clamped to leave and arrive at rest in `q_a` and that flattening has
    to happen somewhere. Interpolating a densely sampled path squeezes the
    clamp into a sliver of sigma, and `q_a''` at the endpoint blows up: measured
    at 361 against an interior value of 0.4, a spike the timing OCP then reads
    at its very first node and refuses.

    A least-squares B-spline with its own knot count breaks the tie. `sigma` is
    sampled as densely as the certificate demands and the curve carries
    `path_segments` cubic pieces regardless, so the end intervals stay long and
    the curvature stays O(1). It also stops the fit chasing inverse-kinematics
    noise between neighbouring samples.

    The curve no longer passes exactly through the checked configurations, which
    would matter if the polyline were the thing being certified. It is not:
    `Geometry.check_path` re-runs the whole certificate on this curve, because
    this curve is what executes.
    """
    if len(waypoints) < 2:
        raise PlanningError("the lifted path carries fewer than two configurations")
    spans = np.max(np.abs(np.diff(waypoints, axis=0)) / limits.dq_max, axis=1)
    spans = np.maximum(spans, 1.0e-9)
    total = float(np.sum(spans))
    nodes = np.concatenate([[0.0], np.cumsum(spans) / total])
    nodes[-1] = 1.0

    degree = 3
    # One coefficient per segment plus the degree, and never more than the data
    # can determine.
    segments = int(np.clip(config.path_segments, 1, max(1, len(waypoints) - degree)))
    interior = np.linspace(0.0, 1.0, segments + 1)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    try:
        spline = make_lsq_spline(nodes, waypoints, knots, k=degree)
    except ValueError as failure:
        raise PlanningError(
            f"the lifted path cannot be fitted with {segments} segments: {failure}"
        ) from None

    # The endpoints and the end slopes are boundary conditions, not something to
    # be least-squares fitted. For a clamped B-spline the first and last
    # coefficients *are* the endpoint values, and the end slope is set by the
    # one coefficient next to each -- so both are imposed exactly by writing
    # four coefficients, and every remaining one is left as fitted.
    coefficients = np.array(spline.c, dtype=float)
    coefficients[0] = waypoints[0]
    coefficients[-1] = waypoints[-1]
    moving = float(np.linalg.norm(dq_start)) > 1.0e-9
    start_rate = (
        np.asarray(dq_start, dtype=float) * total if moving else np.zeros(PLANNED_DOF)
    )
    coefficients[1] = (
        coefficients[0] + start_rate * (knots[degree + 1] - knots[1]) / degree
    )
    # Arriving at rest in q_a is what lets sigma_dot stay above its floor while
    # dq_a = q_a' sigma_dot still reaches zero.
    coefficients[-2] = coefficients[-1]
    spline = BSpline(knots, coefficients, degree)

    dense = np.linspace(0.0, 1.0, max(129, len(waypoints)))
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
                f"the fitted path leaves joint {axis}'s range; the lifted "
                f"configurations cannot be followed with {segments} segments"
            )

    travel = 0.0
    for one, other in zip(sampled[:-1], sampled[1:]):
        travel += geometry.travel(one, other)
    return Path(spline=spline, travel_m=float(travel)), (
        1.0 / total if moving else None
    )


def plan_fitted_path(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_position_m: np.ndarray,
    goal_yaw: float,
    dq_start: np.ndarray,
) -> tuple[Path, float | None, int, str]:
    """Lift, fit and certify candidates until the executable curve is clear."""
    start_position, _ = geometry.tcp_pose(start_q_a)
    candidates = cartesian_candidates(
        geometry, config, start_position, np.asarray(goal_position_m, dtype=float)
    )
    failures = []
    for candidate in candidates:
        try:
            waypoints = lift_candidate(
                geometry, limits, config, start_q_a, goal_yaw, candidate
            )
            path, sigma_dot_start = fit(
                geometry, limits, config, waypoints, dq_start
            )
            # The spline is a different curve from its lifted polyline. Only a
            # successful second certificate makes it executable.
            geometry.check_path(path)
            return path, sigma_dot_start, len(waypoints), candidate.name
        except PlanningError as failure:
            failures.append(f"{candidate.name}: {failure}")
    summary = "; ".join(failures[:3])
    if len(failures) > 3:
        summary += f"; and {len(failures) - 3} more corridor refusals"
    raise PlanningError(
        f"none of the {len(candidates)} deterministic tool corridors survives "
        f"lifting, C2 fitting and collision certification: {summary}"
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
    tau: np.ndarray
    pump_flow: np.ndarray
    flow_slack: np.ndarray
    time: np.ndarray
    iterations: int
    solve_time_s: float

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def terminal_sway(self) -> float:
        """How far the tool still hangs off its rest pose at the goal, in radians."""
        return float(np.max(np.abs(self.q_u[-1] - self.q_u_eq[-1])))

    @property
    def terminal_sway_rate(self) -> float:
        """Largest passive rate remaining at the goal, in radians per second."""
        return float(np.max(np.abs(self.dq_u[-1])))


class TimingOcp:
    """
    How fast the machine may traverse a given path, as one NLP over sigma.

    The independent variable is the path parameter, not time. The state is
    `(sigma_dot, q_u, dq_u)`: the sway is *planned*, not merely tolerated. The
    control is `sigma_ddot`, and the five joint accelerations follow from it by
    the chain rule -- `ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot` -- which is
    why the path has to be C2 and why nothing here re-solves geometry.

    Minimising traversal time is minimising the integral of `dsigma/sigma_dot`,
    so `sigma_dot` is bounded away from zero throughout; the path is fitted to
    leave and arrive at rest in *q_a*, which is what makes that admissible. The
    end time is free -- that integral is the end time.

    # What binds and what is merely priced

    Joint velocity, joint acceleration and the sway box are hard. The pump is
    hard with L1 slack. `tau_a` is **priced and not bounded**: the cylinder-force
    row it replaces needed a relief pressure nothing here has measured, and a
    limit that is invented refuses moves the machine can make. Pricing effort
    still puts the machine's inertia in the problem -- the optimiser feels what
    it costs to accelerate a loaded telescope -- without asserting a number.

    The consequence is real and is the trade: this planner no longer refuses a
    move it cannot lift. It plans it, and the machine stalls on it.
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
        # `tau_a` of `wiki/robot_model.md` 3.3 -- the actuated generalized force,
        # which is what carries the machine's inertia into the cost. The cylinder
        # force rows sit further down the same output map and are not read.
        tau = output[
            symbolic_model.K_ACTUATED_FORCE_OFFSET : symbolic_model.K_ACTUATED_FORCE_OFFSET
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
        self.demand = ca.Function("demand", arguments, [ddq_a, tau, flow])

    def _blocks(self, path: Path, q_tool: float, nodes: np.ndarray) -> np.ndarray:
        return np.array([path.block(sigma, q_tool) for sigma in nodes])

    def _ceiling(self, blocks: np.ndarray, speed_scale: float) -> np.ndarray:
        """
        Return the largest `sigma_dot` each node admits on joint velocity alone.

        `dq_a = q_a'(sigma) sigma_dot`, so a velocity limit is a bound on the path
        rate and not a nonlinear row. At the endpoints `q_a' = 0` and nothing
        bounds it, which is why `sigma_rate_max` exists.
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
        curve is imposed on top of it. Neither says anything about pump flow,
        which is what actually binds on a lift, so the rate is bisected down until
        it sits at 90% of its allowance -- 90% and not 100% because a barrier
        method started with a dozen rows at zero slack is a barrier method that
        does not start. And per-node feasibility is not reachability, so a forward
        and a backward sweep under `d(sigma_dot^2)/dsigma = 2 sigma_ddot` connect
        the nodes to each other.
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

        def draw(node: int, rate: float) -> float:
            state = np.concatenate([[rate], equilibrium[node], np.zeros(2)])
            _, _, flow = self.demand(state, 0.0, blocks[node], payload)
            return float(abs(float(flow))) / (self.config.kappa * flow_max)

        floor = 2.0 * self.config.sigma_rate_min
        for node in range(len(blocks)):
            if draw(node, guess[node]) <= 0.9:
                continue
            low, high = floor, guess[node]
            for _ in range(12):
                middle = 0.5 * (low + high)
                if draw(node, middle) <= 0.9:
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
        # velocity, the path rate itself, and the pump. Acceleration goes with its
        # square, because along a fixed path `ddq_a` is quadratic in the rate.
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

        opti = ca.Opti()
        state = opti.variable(5, intervals + 1)
        control = opti.variable(intervals)
        flow_slack = opti.variable(intervals + 1)
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

        # What the answer demands of the machine, at **every** node including the
        # last. The terminal node carries no input of its own, but it does not
        # need one: the path arrives at rest in q_a, so `q_a'(1) sigma_ddot`
        # vanishes and `ddq_a = q_a''(1) sigma_dot^2` is well defined -- and it is
        # the row that decides how fast the machine may still be travelling when
        # it gets there. Leaving the last node out is how a plan ends with nine
        # times the admissible deceleration.
        for node in range(intervals + 1):
            here = state[:, node]
            input_ = control[min(node, intervals - 1)]
            ddq_a, tau, flow = self.demand(here, input_, blocks[node], payload_vector)
            allowed = config.kappa * accel_max
            opti.subject_to(opti.bounded(-allowed, ddq_a, allowed))
            opti.subject_to(flow_slack[node] >= 0.0)
            opti.subject_to(flow >= 0.0)
            opti.subject_to(flow <= config.kappa * flow_max + flow_slack[node])

            # Traversal time, sway, effort and a regularisation on the input, all
            # as integrals over sigma: `dt = dsigma / sigma_dot` is the only
            # reason the first term is the objective it is. The trapezoid weight
            # halves the two endpoints.
            weight = span * (0.5 if node in (0, intervals) else 1.0)
            cost += weight * (
                1.0 / sigma_dot[node]
                + config.sway_weight * ca.sumsqr(dq_u[:, node]) / sigma_dot[node]
                + config.tau_weight
                * ca.sumsqr(tau / self.limits.tau_max)
                / sigma_dot[node]
            )
            cost += config.flow_slack_weight * flow_slack[node] / flow_max
        cost += config.input_weight * span * ca.sumsqr(control)

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
        # The tool should arrive hanging and still, and that is a **cost**. One
        # control -- the rate along a path it may not leave -- cannot in general
        # drive four terminal quantities to zero, and demanding it as a hard row
        # is what turns a slow answer into `Infeasible_Problem_Detected`. The
        # normalisation is the admissible box, so the weight is a multiple of
        # "the whole allowance" and the number reported afterwards is in radians.
        cost += config.terminal_sway_weight * (
            ca.sumsqr((q_u[:, intervals] - equilibrium[intervals]) / config.q_sway_max)
            + ca.sumsqr(dq_u[:, intervals] / config.dq_sway_max)
        )

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
        opti.set_initial(flow_slack, np.zeros(intervals + 1))
        opti.solver(
            "ipopt",
            {"print_time": False, "expand": True},
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
                f"no admissible timing exists for this path: "
                f"{opti.stats().get('return_status', failure)}"
            ) from None
        elapsed = time.monotonic() - started

        rate = np.array(answer.value(sigma_dot)).ravel()
        accel = np.array(answer.value(control)).ravel()
        sway = np.array(answer.value(q_u)).reshape(2, -1).T
        sway_rate = np.array(answer.value(dq_u)).reshape(2, -1).T
        slack = np.array(answer.value(flow_slack)).ravel()

        ddq_a = np.zeros((intervals + 1, PLANNED_DOF))
        tau = np.zeros((intervals + 1, PLANNED_DOF))
        flow = np.zeros(intervals + 1)
        for node in range(intervals + 1):
            here = np.concatenate([[rate[node]], sway[node], sway_rate[node]])
            values = self.demand(
                here, accel[min(node, intervals - 1)], blocks[node], payload_vector
            )
            ddq_a[node] = np.array(values[0]).ravel()
            tau[node] = np.array(values[1]).ravel()
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
            tau=tau,
            pump_flow=flow,
            flow_slack=slack,
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

    Built once from the robot description -- parsing it and code-generating the
    OCP's functions both happen here, not per request.
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
            self.config.pump_flow_max,
            self.config.pump_flow_planning_factor,
        )
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
        straight tool path that is blocked or does not resolve, and a path that
        cannot be timed are three different answers, and each one names itself.
        """
        if not 0.0 < speed_scale <= 1.0:
            raise PlanningError(f"speed_scale must be in (0, 1], not {speed_scale}")
        self._validate_start(start)
        if avoid_collisions and scene is None:
            raise PlanningError(
                "avoid_collisions was asked for with no collision scene: a plan "
                "certified against nothing is not certified"
            )

        primitives = self.prepare_scene(scene, avoid_collisions)
        payload_vector = payload_parameters(payload)
        geometry = Geometry(
            self.model,
            self.limits,
            primitives,
            self.config,
            start.q_tool,
            payload_vector,
            payload_shape if avoid_collisions else None,
        )

        # Is the goal pose reachable at all? This solve is cold and spreads its
        # restarts over the telescope range, which is the coordinate the residual
        # is flat in -- so it answers "no configuration reaches this" cheaply and
        # with the miss in millimetres. The configuration it returns is
        # deliberately **not** kept: which point of the redundant family the arm
        # ends at is decided by marching there from the start, and pinning an
        # independently chosen one is a discontinuity no refinement can close.
        solve_ik(
            geometry,
            self.limits,
            self.config,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            start.q_a,
            restarts=int(self.config.ik_restarts),
        )
        if not geometry.is_valid(start.q_a):
            raise PlanningError(
                f"the measured start configuration clears the scene by only "
                f"{geometry.clearance(start.q_a):.3f} m against the "
                f"{geometry.required:.3f} m this plan requires"
            )

        path, sigma_dot_start, lifted, candidate_name = plan_fitted_path(
            geometry,
            self.limits,
            self.config,
            start.q_a,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            start.dq_a,
        )

        nodes = np.linspace(0.0, 1.0, int(self.config.intervals) + 1)
        equilibrium = np.array(
            [passive_equilibrium(path.position(sigma)) for sigma in nodes]
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
                "the tool is swinging faster than the admissible sway rate, so there "
                "is no plan that starts from it"
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
        terminal_offset = np.abs(timing.q_u[-1] - timing.q_u_eq[-1])
        terminal_rate = np.abs(timing.dq_u[-1])
        if np.any(terminal_offset > self.config.terminal_q_sway_max) or np.any(
            terminal_rate > self.config.terminal_dq_sway_max
        ):
            raise PlanningError(
                "the timing solve converged but did not arrive settled: terminal sway "
                f"offset {terminal_offset.tolist()} rad against "
                f"{self.config.terminal_q_sway_max.tolist()}, rate "
                f"{terminal_rate.tolist()} rad/s against "
                f"{self.config.terminal_dq_sway_max.tolist()}"
            )
        return self._resample(
            geometry, path, timing, start, lifted, candidate_name
        )

    def _validate_start(self, start: Start) -> None:
        """Refuse a measured state that is not one state of this description."""
        q = np.asarray(start.q, dtype=float).reshape(-1)
        dq_a = np.asarray(start.dq_a, dtype=float).reshape(-1)
        dq_u = np.asarray(start.dq_u, dtype=float).reshape(-1)
        if q.size != GENERALIZED_DOF or not np.all(np.isfinite(q)):
            raise PlanningError(
                f"the start must contain {GENERALIZED_DOF} finite canonical positions"
            )
        if dq_a.size != PLANNED_DOF or not np.all(np.isfinite(dq_a)):
            raise PlanningError(
                f"the start must contain {PLANNED_DOF} finite planned-joint rates"
            )
        if dq_u.size != len(PASSIVE_INDICES) or not np.all(np.isfinite(dq_u)):
            raise PlanningError("the start must contain two finite passive sway rates")
        q_a = q[list(PLANNED_INDICES)]
        outside = self.limits.bounded & (
            (q_a < self.limits.lower) | (q_a > self.limits.upper)
        )
        if np.any(outside):
            axis = int(np.flatnonzero(outside)[0])
            raise PlanningError(
                f"planned coordinate {axis} is measured at {q_a[axis]:.6f}, outside "
                f"[{self.limits.lower[axis]:.6f}, {self.limits.upper[axis]:.6f}]; "
                "planning from a projected state would not match the machine"
            )
        if self.limits.tool_bounded and not (
            self.limits.tool_lower <= start.q_tool <= self.limits.tool_upper
        ):
            raise PlanningError(
                f"the tool coordinate is measured at {start.q_tool:.6f}, outside "
                f"[{self.limits.tool_lower:.6f}, {self.limits.tool_upper:.6f}]; "
                "fix the simulated state or the description instead of projecting it"
            )

    def prepare_scene(self, scene, avoid_collisions: bool) -> list:
        """
        Return the static bodies this plan is checked against.

        Not the same list the request carried: the reserved `truck` primitive has
        become a bed, six runges and a headboard by the time the planner looks at
        it, and that is invisible to anyone who only sees the scene topic. This
        is a method and not a private step because the node draws it, and a
        refusal is far easier to read beside the geometry that caused it.

        What the tool carries is **not** here. It moves, so it is placed at each
        configuration checked rather than pinned to the scene once.
        """
        if not avoid_collisions:
            return []
        primitives = expand_truck(list(scene or []), self.config)
        if any(primitive.id == PAYLOAD_ID for primitive in primitives):
            raise PlanningError(f"'{PAYLOAD_ID}' is a reserved scene id")
        return primitives

    def _settled(self, q_a: np.ndarray, q_tool: float) -> np.ndarray:
        """Return the canonical eight at `q_a` with the passive pair hanging."""
        q = np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)] = q_a
        q[TOOL_INDEX] = q_tool
        q[list(PASSIVE_INDICES)] = passive_equilibrium(q_a)
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
        Where the tool hangs relative to the tip pivot K5, for one yaw.

        The retained `a2b_movement` goal names the pivot and the native goal names
        the tool, so the adapter needs the vector between them. It is read out of
        the model at the hanging equilibrium and never written down: the PZS100's
        rail gripper and the 7040's jaw do not hang at the same offset.

        The two passive joints make the settled offset independent of the boom and
        telescope pose -- once the pendulum is settled, only rotation about gravity
        moves this vector. So the description's own yaw convention is measured at a
        canonical pose, the slew is turned by the difference, the pendulum is
        settled again, and the result is *checked* to carry the requested yaw
        rather than assumed to.

        `payload` no longer enters it: the hanging pose is a closed form in the
        boom and arm angles alone, which is what `crane_mpc` measured it to be.
        The argument is kept so the adapter's call site stays honest about what it
        is asking for.
        """
        if not (np.isfinite(yaw) and np.isfinite(q_tool)):
            raise PlanningError(
                "`phi_tool_n` or the tool coordinate q8 is not finite, so the hanging "
                "tip-to-tool offset cannot be evaluated"
            )
        q_a = np.zeros(PLANNED_DOF)
        canonical = self._settled(q_a, q_tool)
        q_a[0] = _wrap(yaw - self._tcp_yaw(canonical))
        q = self._settled(q_a, q_tool)

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
                "the description produced no finite non-zero offset from the tip pivot "
                "to the tool centre point, so `y_n` cannot be placed on the native "
                "tool goal"
            )
        return offset

    def _resample(
        self,
        geometry,
        path,
        timing,
        start: Start,
        lifted: int,
        candidate_name: str,
    ) -> Plan:
        """
        Put the answer on the emitted reference's own clock.

        The OCP works on a uniform sigma grid, which is not a uniform time grid;
        the consumer reads at `Ts`. Sigma is interpolated against the elapsed time
        the solve produced, and the joint rates are re-derived from
        `q_a'(sigma) sigma_dot` there rather than interpolated -- interpolating a
        derivative and its integral separately is how the two stop agreeing.

        The passive pair is carried too, and it is the sway the OCP **planned**
        rather than the pose the tool would settle to. It costs nothing -- the OCP
        solves for it either way -- and it is what the trajectory actually claims:
        on the way to the goal the tool is swinging.
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

        # The tool where the plan says it is, sway included -- not where it would
        # hang if the machine stopped at each sample.
        tcp = np.array(
            [
                geometry.model.forward_kinematics(
                    row, Frame.MOUNTING_BASE, Frame.TCP
                ).position_m
                for row in q
            ]
        )
        slack = float(np.max(timing.flow_slack)) / self.limits.flow_max
        message = (
            f"{candidate_name}; {timing.duration:.2f} s over "
            f"{path.travel_m:.2f} m of tool travel, "
            f"{lifted} lifted configurations, {len(stamps)} points at "
            f"{Ts * 1e3:.0f} ms; IPOPT converged in {timing.iterations} iterations and "
            f"{timing.solve_time_s:.2f} s; peak pump draw "
            f"{np.max(timing.pump_flow) / self.limits.flow_max:.2f} of the physical "
            f"limit at kappa = {self.config.kappa}"
            + (f", exceeding the reservation by {slack:.3f}" if slack > 1e-9 else "")
            + f"; the tool arrives {np.degrees(timing.terminal_sway):.2f} deg off rest "
            + f"at {timing.terminal_sway_rate:.3f} rad/s"
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
