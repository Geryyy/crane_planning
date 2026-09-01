"""
Where the machine may go, and the curve fitted through it.

The scene, the inverse kinematics, the bounded family of candidate paths and the
fit. Clearance is *proved* rather than sampled: a margin paired with a bound on
how far the machine moves between two checked configurations. Nothing here knows
about time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from crane_model import (
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    CollisionPrimitive,
    CraneModel,
    CraneModelError,
    Frame,
)
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.optimize import least_squares

from .config import (
    ARM_AXIS,
    BOOM_AXIS,
    INITIAL_LIFT_STEP,
    MIN_LIFT_STEP,
    PLANNED_DOF,
    PLANNED_FRAMES,
    PLANNED_INDICES,
    TELESCOPE_AXIS,
    TELESCOPE_TRAVEL_PER_UNIT,
    TOOL_INDEX,
    Limits,
    PlannerConfig,
    PlanningError,
    passive_equilibrium,
)

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


def wrap(angle: float) -> float:
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
                [wrap(yaw_of(rotation) - yaw)],
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

    def accepted(values: np.ndarray) -> bool:
        return (
            float(np.linalg.norm(values[:3])) <= config.eps_pos
            and float(abs(values[3])) <= config.eps_yaw
        )

    # Continuation queries are deliberately close to their seed.  A bounded
    # Gauss--Newton step with the analytic Jacobian gets them home without
    # constructing and factorising a fresh scipy trust-region problem for every
    # collision-certificate sample.  Keep scipy below as the robust fallback
    # and for the cold, restarted endpoint query.
    if restarts <= 1:
        current = starts[0].copy()
        values = residual(current)
        for _ in range(8):
            if accepted(values):
                return current
            try:
                update = np.linalg.lstsq(jacobian(current), -values, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            baseline = float(np.linalg.norm(values[:4]))
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                trial = np.clip(current + scale * update, lower, upper)
                trial_values = residual(trial)
                if float(np.linalg.norm(trial_values[:4])) < baseline:
                    current, values = trial, trial_values
                    improved = True
                    break
            if not improved:
                break
        starts = [current]

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
        if accepted(answer.fun):
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

    transfer_z = max(float(start[2]), float(goal[2])) + float(config.corridor_clearance)
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
    certify_samples: bool = True,
) -> np.ndarray:
    """Lift one TCP polyline into joint space using continuation IK."""
    positions = _without_repeated_positions(candidate.positions_m)
    if len(positions) < 2:
        raise PlanningError(f"{candidate.name} contains no motion")
    _start_position, start_yaw = geometry.tcp_pose(start_q_a)
    yaw_sweep = wrap(float(goal_yaw) - start_yaw)
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
            if certify_samples and not geometry.is_valid(q_a):
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
    # These samples establish a continuous IK branch and the interpolation
    # bound; they are not executed.  Collision certification is performed once
    # on the fitted C2 curve below, avoiding a duplicate model query at every
    # continuation sample while preserving the certificate on the actual path.
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
                geometry,
                limits,
                config,
                start_q_a,
                goal_yaw,
                candidate,
                False,
            )
            path, sigma_dot_start = fit(geometry, limits, config, waypoints, dq_start)
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
