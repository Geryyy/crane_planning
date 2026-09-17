"""Where machine may go, and the curve fitted through it: scene, IK, candidate paths, fit."""

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

#: Two passive hinges tool hangs on, in `q_sway_max`/`PASSIVE_INDICES` order, 0.223 m apart.
PASSIVE_PIVOTS = (Frame.TIP, Frame.TILT)


# ------------------------------------------------------------------ the scene


def expand_truck(scene: list, config: PlannerConfig) -> list:
    """
    Turn `truck` primitive into bed, runges, headboard.

    Bolted to vehicle, so the box itself is no obstacle; dims configured, position
    measured. Runge/headboard flush to real edge/face, not centred.
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
    Payload as a scene body at the tool pose for `q`.

    Collision API can't see it, so it's added per config (not once -- pinned at
    start pose it's a ghost, real one rides unchecked). `attached_to_tool` keeps
    the grip from reading as self-collision.
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
    Whether config is clear, at the pose the tool hangs at.

    Every distance is a `crane_model` query. Certificate: margin plus Lipschitz
    bound, not sampling, proves clearance between checked configs. `radii[j]*|dq_j|`
    bounds how far joint j moves anything below it, summed into `step_bound`: every
    checked config clears `required`, and `step_bound <= margin_interp` between
    consecutive pairs => nothing touches in between.

        required = margin_safety + margin_interp + envelope   (each spent once)

    Envelope is owed only by what hangs on hinges (one hanging-pose query settles
    whole-machine; splitting at the upper hinge only charges the swinging half).
    Self-collision tested at zero, not `required`: links are near each other by design.
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
        self.required_rigid = float(config.margin_safety) + float(config.margin_interp)
        self.required = self.required_rigid + self.envelope
        self.radii = self._radii()

    def bodies(self, q: np.ndarray, scene=None) -> list:
        """Return scene at `q`, with what tool carries placed on it."""
        scene = self.scene if scene is None else list(scene)
        carried = payload_primitive(self.model, q, self.payload_shape)
        return scene if carried is None else scene + [carried]

    def _carried_reach(self) -> float:
        """How far past TCP carried body reaches, in metres."""
        if self.payload_shape is None:
            return 0.0
        _shape, dimensions, offset = self.payload_shape
        return float(
            np.linalg.norm(np.asarray(offset, dtype=float))
            + np.linalg.norm(np.asarray(dimensions, dtype=float))
        )

    def _envelope(self) -> float:
        """
        Farthest carried point can swing, in metres.

        Two hinges: `theta6_tip` at `Frame.TIP`, `theta7_tilt` 0.223 m below at
        `Frame.TILT`, orthogonal (Cardan joint), each bounded by `q_sway_max`,
        uncoupled -- both at bound at once, perpendicular so hypotenuse. Dropping
        the upper (longer) hinge understates swing: PZS100 0.153 m one hinge vs
        0.251 m two. Chord `2 L sin(theta/2)`, not `L sin theta` (~0.5% apart,
        undershoots by microns) -- estimate not bound.

        Levers at hanging equilibrium: `|TILT -> TCP|` rigid, `|TIP -> TCP|` bends
        with tilt (0.802 m vs 0.994 m at raw zero); independent of boom/arm/telescope
        so one config suffices, gripped block lengthens both. Sway box is shared with
        the controller -- machine may actually reach it, not a conservative allowance.
        """
        q = np.zeros(GENERALIZED_DOF)
        q[TOOL_INDEX] = self.q_tool
        q[list(PASSIVE_INDICES)] = passive_equilibrium(q[list(PLANNED_INDICES)])
        tcp = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        reach = self._carried_reach()
        swing = []
        for pivot, bound in zip(
            PASSIVE_PIVOTS, np.abs(self.config.q_sway_max), strict=True
        ):
            origin = self.model.forward_kinematics(q, Frame.MOUNTING_BASE, pivot)
            lever = float(np.linalg.norm(tcp.position_m - origin.position_m)) + reach
            swing.append(lever * 2.0 * float(np.sin(0.5 * bound)))
        return float(np.hypot(*swing))

    def _radii(self) -> np.ndarray:
        """
        Distance a body can move per unit of each planned coordinate.

        At full telescope extension (radii largest there): outermost point = TCP +
        `tool_radius` + carried reach, from each joint's *origin* not its axis --
        both over-estimate, safe direction. Telescope is prismatic: displaces what
        it carries by `TELESCOPE_TRAVEL_PER_UNIT`, two metres per metre of q4.
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
        """Furthest any machine point moves between two configs."""
        return float(
            np.sum(self.radii * np.abs(np.asarray(second) - np.asarray(first)))
        )

    def configuration(self, q_a: np.ndarray) -> np.ndarray:
        """Return canonical eight at `q_a`, tool hanging."""
        q = np.zeros(GENERALIZED_DOF)
        q[list(PLANNED_INDICES)] = q_a
        q[TOOL_INDEX] = self.q_tool
        q[list(PASSIVE_INDICES)] = passive_equilibrium(q_a)
        return q

    def _distances(self, q: np.ndarray, swinging=None, scene=None) -> dict | None:
        """
        Distance per scene id at `q`, over `scene` (default all) plus what tool carries.

        `None` if unevaluable or machine touches itself.
        """
        try:
            bodies = self.bodies(q, scene)
            results = self.model.collision_queries(q, bodies, swinging)
        except (CraneModelError, PlanningError):
            return None
        # queries(): one row per scene primitive, then a self-pair row if there is one
        scene_rows = results[: len(bodies)]
        if any(row.minimum_distance_m <= 0.0 for row in results[len(bodies) :]):
            return None
        return {
            body.id: row.minimum_distance_m
            for body, row in zip(bodies, scene_rows, strict=True)
        }

    def _scene_distance(self, q: np.ndarray, swinging=None, scene=None) -> float:
        distances = self._distances(q, swinging, scene)
        if distances is None:
            return -np.inf
        return min(distances.values(), default=np.inf)

    def clearance(self, q_a: np.ndarray) -> float:
        """
        Scene clearance at `q_a` in metres, `-inf` if unevaluable.

        Self-collision is tested separately at zero.
        """
        return self._scene_distance(self.configuration(q_a))

    def margin(self, q_a: np.ndarray, scene=None) -> tuple[float, float, str | None]:
        """
        `(clearance, required, body)` deciding `q_a`.

        `body` is the scene id measured against (`None` if no query possible) --
        refusals must name it. Clearing `required` whole-machine is sufficient but
        not necessary: rigid axes don't swing, so a config only inside `required`
        because of one of them isn't refused -- split at the upper hinge, swinging
        half owes envelope, rigid half just the two margins. `scene` restricts the
        query; `check_path` passes bodies its bounds can't vouch for.
        """
        q = self.configuration(q_a)
        return self._decide(q, self._distances(q, scene=scene), scene)

    def _decide(self, q: np.ndarray, distances: dict | None, scene) -> tuple:
        """
        `(clearance, required, body)` for one set of measured distances.

        Id comes from whichever query produced the number.
        """
        if distances is None:
            return -np.inf, self.required, None
        body, hanging = self._nearest(distances)
        if hanging > self.required:
            return hanging, self.required, body
        if hanging <= self.required_rigid:
            # inside both margins with something: no split can pass it
            return hanging, self.required_rigid, body
        # one more query not two: hanging is the smaller half, swinging clear of required leaves
        # rigid clear at hanging
        swinging_body, swinging = self._nearest(
            self._distances(q, swinging=True, scene=scene)
        )
        if swinging > self.required:
            return hanging, self.required_rigid, body
        return swinging, self.required, swinging_body

    @staticmethod
    def _nearest(distances: dict | None) -> tuple[str | None, float]:
        """Return nearest body and its distance: `-inf` no query, `inf` no body."""
        if distances is None:
            return None, -np.inf
        if not distances:
            return None, np.inf
        body = min(distances, key=distances.get)
        return body, distances[body]

    def is_valid(self, q_a: np.ndarray) -> bool:
        """Clear of scene by whole margin, not folded into itself."""
        clearance, required, _body = self.margin(q_a)
        return clearance > required

    def tcp_pose(self, q_a: np.ndarray) -> tuple[np.ndarray, float]:
        """Return TCP position and yaw at `q_a`, hanging."""
        pose = self.model.forward_kinematics(
            self.configuration(q_a), Frame.MOUNTING_BASE, Frame.TCP
        )
        rotation = pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
        return pose.position_m, yaw_of(rotation)

    def check_path(self, path) -> None:
        """
        Run the certificate on the fitted curve -- the only place it runs.

        The lifted polyline isn't checked (only fixes IK branch/interpolation
        bound); the spline is what executes, so both are tested: clearance at every
        sample, step bound between consecutive ones. Step bound doubles as broad
        phase: a body measured at `d` can't be nearer than `d - motion` after a step
        moving the machine by at most `motion`, so a body whose bound still exceeds
        `required` is just decremented, not re-queried -- every step for truck under
        crane, never for container across yard. Carried body always queried.
        """
        previous = path.position(0.0)
        q = self.configuration(previous)
        bounds = self._distances(q)
        clearance, required, body = self._decide(q, bounds, None)
        if not clearance > required:
            raise PlanningError(f"the fitted path is blocked at its start by '{body}'")
        sigma, step, count = 0.0, INITIAL_LIFT_STEP, 1
        while sigma < 1.0:
            step = min(step, 1.0 - sigma)
            candidate = path.position(sigma + step)
            motion = self.step_bound(previous, candidate)
            if motion > self.config.margin_interp:
                step *= 0.5
                if step < MIN_LIFT_STEP:
                    raise PlanningError(
                        f"the fitted path moves the machine faster than the "
                        f"{self.config.margin_interp:.3f} m interpolation margin "
                        f"resolves near sigma = {sigma:.3f}, so it is not certified"
                    )
                continue
            for body in bounds:
                bounds[body] -= motion
            near = [body for body in self.scene if bounds[body.id] <= self.required]
            if not near and self.scene:
                # self row needs query anyway; carry nearest body
                near = [min(self.scene, key=lambda body: bounds[body.id])]
            q = self.configuration(candidate)
            distances = self._distances(q, scene=near)
            clearance, required, body = self._decide(q, distances, near)
            if not clearance > required:
                raise PlanningError(
                    f"the fitted path is blocked at sigma = {sigma + step:.3f} "
                    f"by '{body}'"
                )
            bounds.update(distances)
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
    """
    `phi_tool`: world yaw of TCP local y (not x) about K0 z.

    `phi_tool_n` is the jaw-opening heading, on PZS100 TCP y, rails stroke along ±y -- x gives the
    gripper 90 deg off.
    """
    return float(np.arctan2(rotation[1, 1], rotation[0, 1]))


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
    Solve for the planned five putting tool at `(position_m, yaw)`, hanging.

    Least squares, not closed form: passive pair re-settles per config tested,
    telescope redundancy taken up by a weak pull to seed. `restarts`: cold goal
    spreads restarts over telescope range (residual flat there); corridor samples
    take one restart from predecessor, keeping lift on one IK branch. Reachability
    only -- clearance is the caller's question.
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
        Analytic derivative of residual, not finite difference.

        Worth it: finite-differencing costs six FK evals/iteration vs one Jacobian
        here. Two corrections: model Jacobian is LOCAL so linear/angular blocks
        rotate into `K0_mounting_base`; and the passive pair isn't independent --
        pendulum hangs at `q_eq = (pi/2 - q_boom - q_arm, pi/2)`, so boom/arm swing
        the tip joint back by as much, entering those two columns with factor -1.
        Yaw row uses the angular Jacobian z entry, exact only near-upright -- costs
        convergence, not correctness (acceptance is on the exact residual).
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

    # continuation queries sit near their seed: bounded Gauss-Newton w/ analytic Jacobian is cheaper
    # than a fresh scipy problem; scipy below is fallback, takes cold restarted endpoint
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
        # inside tolerance: not a local minimum. Returned outright, not via `best`,
        # which mixes metres/radians in one norm while acceptance is two scalars
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
    """Named TCP polyline in mounting-base frame."""

    name: str
    positions_m: tuple[np.ndarray, ...]


@dataclass
class JointCandidate:
    """Straight line in planned coordinates to cold goal solve."""

    name: str
    goal_q_a: np.ndarray


def _primitive_top(primitive) -> float:
    """Highest mounting-base z of oriented primitive's bounding box."""
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
    """Return bounded deterministic set of tool corridors to try."""
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
) -> np.ndarray:
    """
    Lift one TCP polyline into joint space via continuation IK.

    Samples fix IK branch/interpolation bound, not executed or collision-checked -- the certificate
    runs once, on the fitted C2 curve, since that curve executes.
    """
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


def lift_joint_line(
    geometry: Geometry,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_q_a: np.ndarray,
) -> np.ndarray:
    """
    Waypoints on a straight joint-space line, spaced inside step bound.

    No IK needed since the goal config is the cold solve's and the line to it is continuous by
    construction, so the branch-jump concern for Cartesian corridors doesn't apply. Only candidate
    that doesn't sweep the tool through a chord.
    """
    start = np.asarray(start_q_a, dtype=float)
    goal = np.asarray(goal_q_a, dtype=float)
    count = int(np.ceil(geometry.step_bound(start, goal) / config.margin_interp)) + 1
    count = min(max(count, 2), int(config.max_lift_samples))
    return np.linspace(start, goal, count)


# ---------------------------------------------------------------------- the fit


@dataclass
class Path:
    """
    q_a(sigma) on [0, 1], twice differentiable everywhere.

    A kink makes q_a'' unbounded there, so the certificate's step bound refuses it and any
    differentiating consumer sees a spike.
    """

    spline: BSpline

    def position(self, sigma) -> np.ndarray:
        return self.spline(sigma)


def _densify(waypoints: np.ndarray, count: int) -> np.ndarray:
    """Resample joint-space polyline to `count` points along own chords."""
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    nodes = np.concatenate([[0.0], np.cumsum(lengths)])
    if nodes[-1] <= 0.0:
        raise PlanningError("the lifted path does not move")
    target = np.linspace(0.0, nodes[-1], count)
    return np.column_stack(
        [
            np.interp(target, nodes, waypoints[:, axis])
            for axis in range(waypoints.shape[1])
        ]
    )


def fit(
    limits: Limits,
    config: PlannerConfig,
    waypoints: np.ndarray,
    dq_start: np.ndarray,
) -> Path:
    """
    Fit `q_a(sigma)` to the waypoints.

    Sigma is distributed by how long each chord takes at its slowest axis's limit,
    so a segment slow in one coordinate gets more parameter -- arclength in
    `dq_max` metric (varies 1.05 over path on shipped defaults, not
    reparametrised). Approximates, not interpolates: interpolation ties segment
    count to sample count, but the certificate wants dense samples and the curve
    wants long end intervals for clamp; interpolating a dense path spikes `q_a''`
    at the endpoint (361 vs 0.4 interior). Least-squares B-spline of
    `path_segments` pieces breaks that tie -- curve then misses checked configs,
    fine since the polyline isn't what's certified.
    """
    if len(waypoints) < 2:
        raise PlanningError("the lifted path carries fewer than two configurations")

    # degree 5: simple interior knots make degree-d B-spline C(d-1), quintic is C4 in
    # sigma -- must match ocp.ORDER = degree + 1
    degree = 5
    # exactly path_segments, never fewer: OCP is code-generated for that many
    # polynomials, so fewer pieces is a different problem. Resampling along own
    # chords is free -- not certified, only what fit targets
    segments = int(config.path_segments)
    # Schoenberg-Whitney: least-squares needs data in every knot interval, not just
    # enough points for the segments+degree coefficients, or make_lsq_spline refuses.
    # 4/segment is what degree 5 needs, free -- points already lie on the polyline
    minimum = 4 * segments + degree
    if len(waypoints) < minimum:
        waypoints = _densify(waypoints, minimum)

    # after densifying, never before: nodes indexes waypoints, lengths must match
    spans = np.max(np.abs(np.diff(waypoints, axis=0)) / limits.dq_max, axis=1)
    spans = np.maximum(spans, 1.0e-9)
    total = float(np.sum(spans))
    nodes = np.concatenate([[0.0], np.cumsum(spans) / total])
    nodes[-1] = 1.0

    interior = np.linspace(0.0, 1.0, segments + 1)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    try:
        spline = make_lsq_spline(nodes, waypoints, knots, k=degree)
    except ValueError as failure:
        raise PlanningError(
            f"the lifted path cannot be fitted with {segments} segments: {failure}"
        ) from None

    # endpoints are boundary conditions not fit targets: on a clamped B-spline the
    # outer coefficients *are* endpoint values
    coefficients = np.array(spline.c, dtype=float)
    coefficients[0] = waypoints[0]
    coefficients[-1] = waypoints[-1]
    # start slope is a boundary condition only while moving (dq_a = q_a'(sigma)*v,
    # q_a'(0) = dq_a*L); from rest no direction to honour, writing one would make
    # ddq_a singular. Goal end never clamped: arriving stopped is v=0, a timing
    # condition, not geometry's
    if np.any(np.asarray(dq_start, dtype=float) != 0.0):
        start_rate = np.asarray(dq_start, dtype=float) * total
        coefficients[1] = (
            coefficients[0] + start_rate * (knots[degree + 1] - knots[1]) / degree
        )
    spline = BSpline(knots, coefficients, degree)

    # load-bearing scan not a certificate: OCP has no q_a to box, so nothing else
    # catches an overshoot between samples; check_path's Lipschitz bound would
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

    return Path(spline=spline)


def plan_fitted_path(
    geometry: Geometry,
    limits: Limits,
    config: PlannerConfig,
    start_q_a: np.ndarray,
    goal_position_m: np.ndarray,
    goal_yaw: float,
    dq_start: np.ndarray,
    goal_q_a: np.ndarray | None = None,
) -> tuple[Path, int, str]:
    """
    Lift, fit and certify candidates until one executable curve is found.

    Direct tool line, then joint-space line (if goal config given), then transfer corridors. An
    unreachable corner refuses every corridor through it before any is lifted -- lifting towards one
    otherwise costs hundreds of IK solves per corridor first.
    """
    start_position, start_yaw = geometry.tcp_pose(start_q_a)
    candidates = cartesian_candidates(
        geometry, config, start_position, np.asarray(goal_position_m, dtype=float)
    )
    if goal_q_a is not None:
        candidates.insert(1, JointCandidate("joint line", np.asarray(goal_q_a)))
    failures = []
    unreachable: dict[tuple, str] = {}

    def corner_refusal(candidate: CartesianCandidate) -> str | None:
        corners = candidate.positions_m[1:-1]
        for index, corner in enumerate(corners):
            key = tuple(np.round(np.asarray(corner, dtype=float), 6))
            if key not in unreachable:
                yaw = start_yaw if index < len(corners) / 2 else float(goal_yaw)
                try:
                    solve_ik(
                        geometry, limits, config, corner, yaw, start_q_a, restarts=1
                    )
                    unreachable[key] = ""
                except PlanningError as failure:
                    unreachable[key] = str(failure)
            if unreachable[key]:
                return unreachable[key]
        return None

    for candidate in candidates:
        try:
            if isinstance(candidate, JointCandidate):
                waypoints = lift_joint_line(
                    geometry, config, start_q_a, candidate.goal_q_a
                )
            else:
                refusal = corner_refusal(candidate)
                if refusal is not None:
                    raise PlanningError(refusal)
                waypoints = lift_candidate(
                    geometry, limits, config, start_q_a, goal_yaw, candidate
                )
            path = fit(limits, config, waypoints, dq_start)
            # spline differs from its lifted polyline; only a passing certificate makes it
            # executable
            geometry.check_path(path)
            return path, len(waypoints), candidate.name
        except PlanningError as failure:
            failures.append(f"{candidate.name}: {failure}")
    summary = "; ".join(failures[:3])
    if len(failures) > 3:
        summary += f"; and {len(failures) - 3} more corridor refusals"
    raise PlanningError(
        f"none of the {len(candidates)} deterministic tool corridors survives "
        f"lifting, C2 fitting and collision certification: {summary}"
    )
