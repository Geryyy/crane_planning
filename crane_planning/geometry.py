"""
Where the machine may go, and the curve fitted through it.

Scene, IK, the bounded family of candidate paths, the fit. Clearance is *proved*
rather than sampled: a margin paired with a bound on how far the machine moves
between two checked configurations. Nothing here knows about time.
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

#: The two passive hinges the tool hangs on, in `q_sway_max` / `PASSIVE_INDICES`
#: order. Orthogonal axes 0.223 m apart, so their swings do not add -- see
#: `Geometry._envelope`.
PASSIVE_PIVOTS = (Frame.TIP, Frame.TILT)


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

    Every distance is a `crane_model` query -- no second collision model here.

    The certificate: sampling proves nothing alone, a margin paired with a
    Lipschitz bound does. `radii[j]*|dq_j|` bounds how far joint j's rotation
    moves anything below it, summed into `step_bound`. So if every checked
    configuration clears by more than `required` and `step_bound <=
    margin_interp` for every consecutive pair, nothing touches anything in
    between. Both halves are load-bearing; README "The collision certificate"
    has the argument.

        required = margin_safety + margin_interp + envelope   (each spent once)

    The envelope is owed by what hangs on the hinges and by nothing else. One
    hanging-pose query settles a configuration that clears by `required` with
    the whole machine; one inside it is split at the upper hinge, and only the
    swinging half has to clear by the envelope -- `margin`.

    Self-collision tests at zero, not against `required`: the links are near each
    other by design.
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
        """Return the scene at `q`, with what the tool carries placed on it."""
        scene = self.scene if scene is None else list(scene)
        carried = payload_primitive(self.model, q, self.payload_shape)
        return scene if carried is None else scene + [carried]

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

        The tool hangs on **two** hinges, not one: `theta6_tip` at `Frame.TIP`
        and `theta7_tilt` 0.223 m below it at `Frame.TILT`. Their axes are
        orthogonal -- a Cardan joint, which is what `K6_double_joint_link` is
        named for -- and `q_sway_max` bounds them separately with nothing
        coupling them, so both reach their bound at once.

        Each hinge swings the carried point about its own pivot and the two
        displacements are perpendicular, so they compose as a hypotenuse. The
        *upper* hinge has the *longer* lever, so dropping it understates the
        swing rather than erring safe: on the PZS100 one hinge gives 0.153 m
        where two give 0.251 m.

        Per hinge the term is the **chord** `2 L sin(theta/2)`, not `L sin
        theta`. The difference is 0.5%, and it is the difference between a bound
        and an estimate: `sin` undershoots the swept sway box by a few tens of
        microns, which a bound may not do.

        Levers are read **at the hanging equilibrium**, which is where the sway
        box is centred. `|TILT -> TCP|` is rigid and does not care, but
        `|TIP -> TCP|` bends with the tilt hinge and reads 0.802 m instead of
        0.994 m at a raw zero configuration. Neither depends on boom, arm or
        telescope, so one configuration is enough. A gripped block hangs below
        the TCP and lengthens both levers alike.

        The sway box is shared with the controller, so this is not a conservative
        allowance for a swing that will not happen -- it is a state the machine is
        entitled to reach.
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
        How far a body can move per unit of each planned coordinate.

        At full telescope extension, where every radius is largest. Outermost
        point taken as TCP + `tool_radius` + the carried body's reach; distance
        taken from each joint's *origin*, not its axis. Both over-estimate, the
        safe direction for a bound that has to hold. The prismatic telescope
        displaces what it carries by `TELESCOPE_TRAVEL_PER_UNIT` -- two metres per
        metre of q4, not one.
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
        """Furthest any point of the machine moves between two configurations."""
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

    def _distances(self, q: np.ndarray, swinging=None, scene=None) -> dict | None:
        """
        Return the distance per scene id at `q`.

        Over `scene` (default: all of it) plus what the tool carries. `None`
        when the query cannot be evaluated or the machine touches itself.
        """
        try:
            bodies = self.bodies(q, scene)
            results = self.model.collision_queries(q, bodies, swinging)
        except (CraneModelError, PlanningError):
            return None
        # queries(): one row per scene primitive in scene order, then -- if the
        # description has self-pairs and the machine is not split -- one for the
        # machine against itself.
        scene_rows = results[: len(bodies)]
        self_rows = results[len(bodies) :]
        if any(row.minimum_distance_m <= 0.0 for row in self_rows):
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
        Scene clearance at `q_a` in metres, `-inf` where it cannot be evaluated.

        Self-collision is split out and tested at zero; what is returned is the
        distance to the scene, which is what the margin is against.
        """
        return self._scene_distance(self.configuration(q_a))

    def margin(self, q_a: np.ndarray, scene=None) -> tuple[float, float, str | None]:
        """
        Return `(clearance, required, body)` as what decides `q_a`.

        `body` is the scene id the clearance was measured against -- what a
        refusal has to name to be worth reading -- and `None` where no query
        could be made.

        The whole machine clearing by `required` at the hanging pose is
        sufficient: no sway state can then reach anything. It is not necessary.
        The column, the boom and the arm do not swing, and a configuration
        inside `required` because of one of them is not refused on it: the
        machine is split at the upper hinge, the swinging half owes the
        envelope, the rigid half only the two margins, and the binding of the
        two is what is reported.

        `scene` restricts the query to a subset of the scene; `check_path` passes
        the bodies its bounds cannot already vouch for.
        """
        q = self.configuration(q_a)
        return self._decide(q, self._distances(q, scene=scene), scene)

    def _decide(self, q: np.ndarray, distances: dict | None, scene) -> tuple:
        """
        Return `(clearance, required, body)` for one set of measured distances.

        The id comes from whichever query produced the number, so the name and
        the number never say different things.
        """
        if distances is None:
            return -np.inf, self.required, None
        body, hanging = self._nearest(distances)
        if hanging > self.required:
            return hanging, self.required, body
        if hanging <= self.required_rigid:
            # inside the two margins with something: no split can pass it
            return hanging, self.required_rigid, body
        # One more query, not two: `hanging` is the smaller of the two halves,
        # so a swinging half clear of `required` leaves the rigid half at
        # `hanging`, which is inside the band and therefore clear of its own.
        swinging_body, swinging = self._nearest(
            self._distances(q, swinging=True, scene=scene)
        )
        if swinging > self.required:
            return hanging, self.required_rigid, body
        return swinging, self.required, swinging_body

    @staticmethod
    def _nearest(distances: dict | None) -> tuple[str | None, float]:
        """Return the nearest body and its distance: `-inf` no query, `inf` no body."""
        if distances is None:
            return None, -np.inf
        if not distances:
            return None, np.inf
        body = min(distances, key=distances.get)
        return body, distances[body]

    def is_valid(self, q_a: np.ndarray) -> bool:
        """Clear of the scene by the whole margin, and not folded into itself."""
        clearance, required, _body = self.margin(q_a)
        return clearance > required

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
        Run the certificate on the fitted curve. This is the only place it runs.

        The lifted polyline is not checked -- it only establishes the IK branch
        and the interpolation bound. The spline through it is a different curve,
        and it is the curve that executes, so it takes both tests: clearance at
        every sample, step bound between consecutive ones. Walked with the same
        adaptive step, because a sample count read off the tool's travel says
        nothing about how far the arm moved.

        The step bound is also a broad phase. A body measured at distance `d`
        cannot come nearer than `d - motion` over a step that moves the machine
        by at most `motion`, so a body whose bound still exceeds `required` is
        not queried again; its bound is just decremented. Bodies are re-measured
        only once the bound has been spent, which for the truck under the crane
        is every step and for a container across the yard is never. The carried
        body rides the tool and is always queried.
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
                # the self row needs a query anyway; carry the nearest body
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
    Return `phi_tool`: the world yaw of the TCP's local **y**, about K0 z.

    Not the x axis. `phi_tool_n` is the heading of the jaw-opening direction,
    which on the PZS100 is TCP y -- the rails stroke along -+y. The legacy
    planner spelled the same angle in joint space as `theta1 - theta8`, and the
    two agree exactly at every configuration. Reading the x axis instead is the
    perpendicular, and puts the gripper 90 deg off the requested angle.
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
    Solve for the planned five that put the tool at `(position_m, yaw)`, hanging.

    Least squares, not closed form: the passive pair is re-settled at every
    configuration tested, so what is solved is where the tool actually ends up.
    The telescope's redundancy is taken up by a weak pull towards the seed.

    `restarts` separates the two callers. The goal is cold and spreads restarts
    over the telescope range -- the coordinate the residual is flat in. Each
    sample along a corridor takes one restart from its predecessor, the right
    seed for neighbouring poses, and what keeps the lift on one IK branch.

    Reachability only. Whether the answer is clear is the caller's question.
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
        Analytic derivative of the residual, not a finite difference.

        Worth the lines: differencing five parameters costs six FK evaluations
        per iteration, this costs one Jacobian. That is why the lift stopped
        being the slowest stage.

        Two corrections. The model's Jacobian is LOCAL, so linear and angular
        blocks are rotated into `K0_mounting_base`. And the passive pair is not
        independent -- the pendulum hangs at `q_eq = (pi/2 - q_boom - q_arm,
        pi/2)`, so boom and arm swing the tip joint back by as much, and the tip
        column enters those two columns with factor -1.

        The yaw row takes the angular Jacobian's z entry, exact only while the
        tool hangs near-upright. Costs a little convergence, nothing in
        correctness: the residual stays exact and acceptance is on the residual.
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

    # Continuation queries sit close to their seed, so a bounded Gauss-Newton
    # step with the analytic Jacobian gets them home without building a fresh
    # scipy trust-region problem per sample. scipy below stays the fallback, and
    # handles the cold restarted endpoint query.
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
        # Already inside tolerance: not a local minimum, and further restarts can
        # only re-derive it. Returned outright rather than via `best`, because the
        # selection above mixes metres and radians in one 4-norm while acceptance
        # is two separate scalars -- so the passing restart need not hold `best`.
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


@dataclass
class JointCandidate:
    """A straight line in the planned coordinates to the cold goal solve."""

    name: str
    goal_q_a: np.ndarray


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
) -> np.ndarray:
    """
    Lift one TCP polyline into joint space using continuation IK.

    These samples establish a continuous IK branch and the interpolation bound;
    they are not executed and they are not collision-checked. The certificate
    runs once, on the fitted C2 curve, because that curve is what executes.
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
    Waypoints on the straight joint-space line, spaced inside the step bound.

    The line the reference iLQR planner effectively moves along. No IK: the goal
    configuration is the cold solve's, and a straight line to it is continuous
    by construction, so the branch-jump concern that discards that solve for the
    Cartesian corridors does not apply. Cheap, and the one candidate that does
    not sweep the tool through a chord the arm has to fold in to follow.
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

    Twice, because a kink is not a curve the machine can be asked to follow: at
    one q_a'' is unbounded, so the certificate's step bound refuses it and any
    consumer that differentiates the path reads a spike at the waypoint.
    """

    spline: BSpline

    def position(self, sigma) -> np.ndarray:
        return self.spline(sigma)


def _densify(waypoints: np.ndarray, count: int) -> np.ndarray:
    """Resample a joint-space polyline to `count` points along its own chords."""
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

    sigma is distributed by how long each chord takes at its slowest axis's own
    limit, so a segment slow in one coordinate gets more of the parameter. The
    OCP moves along this curve rather than beside it, so that distribution is
    also what its input sees: measured on the shipped defaults `|q_a'|` in the
    `dq_max` metric varies by 1.05 over the whole path, which is arclength in
    that metric without anyone having reparametrised anything.

    Approximates, does not interpolate: interpolation ties segment count to
    sample count, and the certificate wants samples dense while the curve wants
    its end intervals long for the clamp. Interpolating a dense path spikes
    `q_a''` at the endpoint -- measured 361 against 0.4 in the interior. A
    least-squares B-spline with `path_segments` pieces breaks the tie and stops
    the fit chasing IK noise. The curve then misses the checked configurations,
    which would matter if the polyline were what is certified; it is not.
    """
    if len(waypoints) < 2:
        raise PlanningError("the lifted path carries fewer than two configurations")

    # Degree 5, not 3. With simple interior knots a degree-`d` B-spline is
    # `C(d-1)`, so a quintic is `C4` in sigma -- the derivative count C3's flat
    # inversion needs, obtained from the parameterization rather than enforced.
    # It must match `ocp.ORDER`, which is `degree + 1`.
    degree = 5
    # Exactly `path_segments`, never fewer: the OCP is code-generated against a
    # parameter vector that many polynomials wide, so a short lift answered with
    # fewer pieces is a different problem, not a smaller one. Resampling the
    # polyline along its own chords is free -- it is not what gets certified,
    # only what the least-squares fit is pulled towards.
    segments = int(config.path_segments)
    # `segments + degree` coefficients, and a least-squares fit needs data in
    # every knot interval (Schoenberg-Whitney), not merely as many points as
    # coefficients: at degree 5 that many points is exactly determined and
    # `make_lsq_spline` refuses with "Need more x points". Four per segment is
    # the sample spacing the higher degree asks for, and it costs nothing --
    # the added points are on the polyline already, so the fit is pulled towards
    # the same shape, only sampled better.
    minimum = 4 * segments + degree
    if len(waypoints) < minimum:
        waypoints = _densify(waypoints, minimum)

    # After densifying, never before: `nodes` indexes the waypoints and the two
    # have to be the same length.
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

    # Endpoints are boundary conditions, not things to fit: for a clamped
    # B-spline the outer coefficients *are* the endpoint values, so two writes
    # impose them exactly.
    coefficients = np.array(spline.c, dtype=float)
    coefficients[0] = waypoints[0]
    coefficients[-1] = waypoints[-1]
    # The start slope is a boundary condition only while the machine is moving.
    # The OCP follows this curve, so `dq_a = q_a'(sigma) * v` can only leave
    # along the tangent and the tangent has to be the measured direction;
    # `q_a'(0) = dq_a * L` is that same motion measured in sigma. From rest
    # there is no direction to honour, and writing one anyway sets `q_a'(0) = 0`,
    # where `ddq_a = q_a'' v^2 + q_a' a` contains no `a`: a singular input, not
    # a slow one. The goal end is never clamped for the same reason -- arriving
    # stopped is `v = 0`, a condition on the timing, and asking the geometry for
    # it as well is asking twice and paying twice.
    if np.any(np.asarray(dq_start, dtype=float) != 0.0):
        start_rate = np.asarray(dq_start, dtype=float) * total
        coefficients[1] = (
            coefficients[0] + start_rate * (knots[degree + 1] - knots[1]) / degree
        )
    spline = BSpline(knots, coefficients, degree)

    # Load-bearing, and a scan rather than a certificate: the OCP has no `q_a`
    # to box, so nothing downstream would see an overshoot between two samples.
    # `check_path` carries the Lipschitz bound that would close that; range has
    # none.
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
    Lift, fit and certify candidates until the executable curve is clear.

    Direct tool line, then the joint-space line when a goal configuration is
    given, then the transfer corridors. A corridor corner the tool cannot be
    placed at refuses every corridor through it before any of them is lifted:
    the transfer height is set by the tallest scene body, which need not be
    anywhere near the move, and lifting towards an unreachable corner costs
    hundreds of IK solves per corridor before it fails.
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
            # The spline is a different curve from its lifted polyline. Only a
            # successful certificate on it makes it executable.
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
