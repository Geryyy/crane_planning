"""
Plan a motion, and time it.

Two stages. `geometry` walks a bounded family of TCP corridors, lifts one into
joint space and certifies the fitted curve; `ocp` is handed that curve and solves
for the time-optimal way *along* it. What executes is what was certified.

Everything the machine can do -- reach, hang, collide -- is asked of
`crane_model`; nothing about the machine is written down here.

The passive pair (tip/tilt sway) is never searched over. Where it hangs is a
closed form, `q_eq = (pi/2 - q_boom - q_arm, pi/2)`, good to 1e-4 rad across the
workspace and independent of slew, telescope, rotator, tool and payload. It is a
*state* of the OCP, so the answer is a trajectory the tool arrives nearly still
from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
from crane_model import (
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    CollisionPrimitive,
    CraneModel,
    Frame,
    Payload,
    Tool,
)
from crane_model import symbolic as symbolic_model

from . import weights as crane_weights

# One import site for a consumer, three modules for an author. `config` is what
# a plan is written in, `geometry` is where the machine may go, `timing` is how
# fast it may get there; this file is the stage that runs them in order.
from .config import (
    ARM_AXIS,
    BOOM_AXIS,
    PLANNED_DOF,
    PLANNED_FRAMES,
    PLANNED_INDICES,
    TELESCOPE_AXIS,
    TOOL_INDEX,
    Limits,
    PlannerConfig,
    PlanningError,
    passive_equilibrium,
    read_limits,
)
from .geometry import (
    PAYLOAD_ID,
    TRUCK_ID,
    CartesianCandidate,
    Geometry,
    Path,
    cartesian_candidates,
    expand_truck,
    fit,
    lift_candidate,
    payload_primitive,
    plan_fitted_path,
    solve_ik,
    wrap,
    yaw_of,
)
from .ocp import Trajectory, TrajectoryOcp, evaluate, power_coefficients

# ------------------------------------------------------------------ the planner


#: rad/s (m/s on the telescope). Below this a measured start velocity is rest.
#: An assumption about the encoder noise floor, not a measurement -- Gazebo
#: reports 1e-6-scale rates at standstill; read the real one off a
#: control_recordings/ bag before trusting it on hardware.
REST_VELOCITY = 1e-3

#: rad/s. What the last emitted sample may drift from the rest the OCP pinned
#: before the plan is refused. Deliberately not `REST_VELOCITY`: that one is a
#: claim about the encoder and is meant to be raised to a measured noise floor,
#: which must not quietly loosen what "the solve stopped" means.
TERMINAL_DRIFT_MAX = 1e-3


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


def actuated_samples(timing, stamps: np.ndarray) -> tuple:
    """
    `q_a` and `dq_a` at `stamps`, on the curve rather than on the chord.

    `q_a = c(sigma)` is an identity the solve never leaves, so `q_a` is not a row
    to interpolate: between two nodes the straight line joining them is off the
    curve, and off the curve the certificate was run on. Reconstruct `sigma`
    instead and evaluate.

    The reconstruction is exact, not an interpolation of its own. acados holds
    the input constant across an interval and `sigma'''' = s` there, so `sigma`
    is quartic in elapsed time on each interval.
    """
    node = np.clip(
        np.searchsorted(timing.time, stamps, side="right") - 1,
        0,
        len(timing.time) - 2,
    )
    dt = stamps - timing.time[node]
    v, a = timing.speed[node], timing.acceleration[node]
    j, s = timing.jerk[node], timing.snap[node]
    sigma = (
        timing.sigma[node]
        + v * dt
        + a * dt**2 / 2.0
        + j * dt**3 / 6.0
        + s * dt**4 / 24.0
    )
    speed = v + a * dt + j * dt**2 / 2.0 + s * dt**3 / 6.0
    return (
        evaluate(timing.coefficients, sigma),
        evaluate(timing.coefficients, sigma, order=1) * speed[:, None],
    )


@dataclass
class Plan:
    """A trajectory, the geometry it was found on, and how it was arrived at."""

    time: np.ndarray
    q: np.ndarray  # the canonical eight, resampled at Ts, one row per sample
    dq: np.ndarray
    tcp: np.ndarray  # tool position in K0_mounting_base, one row per sample
    timing: Trajectory
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
    One description, one set of limits, many requests.

    Built once: parsing the description and exporting the OCP solver both happen
    here, not per request.
    """

    def __init__(
        self,
        robot_description_xml: str,
        config: PlannerConfig | None = None,
        weights: dict | None = None,
    ):
        self.config = config or PlannerConfig()
        self.model = CraneModel(robot_description_xml, Tool.PZS100)
        self.limits = read_limits(
            robot_description_xml,
            self.config.pump_flow_max,
            self.config.pump_flow_planning_factor,
        )
        self.ocp = TrajectoryOcp(
            robot_description_xml,
            self.limits,
            self.config,
            weights or crane_weights.DEFAULTS,
        )

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

        Stages refuse, they do not degrade: unreachable goal, no corridor
        surviving its certificate, and untimeable path are three answers and each
        names itself.
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

        # Reachable at all? Cold solve, restarts spread over the telescope range
        # -- the coordinate the residual is flat in. The Cartesian corridors
        # discard its configuration: which member of the redundant family the
        # arm ends at is decided by marching there, and pinning an independently
        # chosen one is a discontinuity no refinement closes. The joint-space
        # line is the one candidate that goes to it directly.
        goal_q_a = solve_ik(
            geometry,
            self.limits,
            self.config,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            start.q_a,
            restarts=int(self.config.ik_restarts),
        )
        clearance, required = geometry.margin(start.q_a)
        if not clearance > required:
            raise PlanningError(
                f"the measured start configuration clears the scene by only "
                f"{clearance:.3f} m against the {required:.3f} m this plan requires"
            )

        # A measured velocity is never exactly zero. Below the noise floor it is
        # rest: fitting a tangent to it pins `c'(0)` at noise magnitude, which
        # the OCP then has to grow to the move's own scale inside one knot span,
        # and that acceleration exceeds the bound at a node where nothing can
        # help. Refused on every start with a live encoder before this.
        dq_a_start = np.asarray(start.dq_a, dtype=float)
        if np.max(np.abs(dq_a_start)) < REST_VELOCITY:
            dq_a_start = np.zeros_like(dq_a_start)

        path, lifted, candidate_name = plan_fitted_path(
            geometry,
            self.limits,
            self.config,
            start.q_a,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            dq_a_start,
            goal_q_a,
        )

        equilibrium_start = passive_equilibrium(path.position(0.0))
        q_u_start = start.q_u if start.passive_measured else equilibrium_start
        dq_u_start = (
            np.asarray(start.dq_u, dtype=float)
            if start.passive_measured
            else np.zeros(2)
        )
        if np.any(np.abs(q_u_start - equilibrium_start) > self.config.q_sway_max):
            raise PlanningError(
                "the tool is swinging further than the admissible sway bound, so "
                "there is no plan that keeps it inside one"
            )
        if np.any(np.abs(dq_u_start) > self.config.dq_sway_max):
            raise PlanningError(
                "the tool is swinging faster than the admissible sway rate, so there "
                "is no plan that starts from it"
            )

        # The certified curve itself, not its endpoints. The solve moves along it
        # and decides only how fast, so what executes is what was proved clear.
        timing = self.ocp.solve(
            coefficients=power_coefficients(path, self.ocp.segments),
            q_u_start=q_u_start,
            dq_a_start=dq_a_start,
            dq_u_start=dq_u_start,
            payload=payload_vector,
            q_tool=start.q_tool,
            speed_scale=speed_scale,
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
        return self._resample(geometry, path, timing, start, lifted, candidate_name)

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

        Not the list the request carried: `truck` has become a bed, six runges and
        a headboard, invisible to anyone watching the scene topic. Public because
        the node draws it -- a refusal reads far better beside the geometry that
        caused it.

        What the tool carries is **not** here: it moves, so it is placed at each
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

        `a2b_movement` names the pivot, the native goal names the tool, so the
        adapter needs the vector between. Read from the model at the hanging
        equilibrium, never written down.

        The passive pair makes the settled offset independent of boom and
        telescope pose: once settled, only rotation about gravity moves it. So
        measure the description's yaw at a canonical pose, turn the slew by the
        difference, settle again, and *check* the result carries the requested
        yaw.

        `payload` does not enter it -- the hanging pose is closed form in boom and
        arm angles alone. The argument is kept so the call site stays honest.
        """
        if not (np.isfinite(yaw) and np.isfinite(q_tool)):
            raise PlanningError(
                "`phi_tool_n` or the tool coordinate q8 is not finite, so the hanging "
                "tip-to-tool offset cannot be evaluated"
            )
        q_a = np.zeros(PLANNED_DOF)
        canonical = self._settled(q_a, q_tool)
        q_a[0] = wrap(yaw - self._tcp_yaw(canonical))
        q = self._settled(q_a, q_tool)

        error = wrap(self._tcp_yaw(q) - yaw)
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

        The OCP solves on a uniform grid over its own horizon, the consumer reads
        at `Ts`, so every row is interpolated against elapsed time. The passive
        pair is carried as the OCP **planned** it, not as the pose the tool would
        settle to: on the way to the goal the tool is swinging, and that is what
        the trajectory claims.

        `q_a` is the exception and it is the whole point of the method. It is not
        an independent row: `q_a = c(sigma)` is an identity the solve never
        leaves, so interpolating it between nodes reports the chord and not the
        curve -- configurations the plan never contained, off the curve the
        certificate was run on. So `sigma` is reconstructed and the curve is
        evaluated at it. The reconstruction is exact rather than interpolated:
        acados holds the input constant across an interval, and `sigma'''' = s`
        there, so `sigma` is quartic and `v` cubic on each interval.
        """
        Ts = float(self.config.Ts)
        stamps = np.arange(0.0, timing.duration + 0.5 * Ts, Ts)
        # Rounding to nearest puts the last stamp up to `Ts / 2` *past* the solve's
        # own end, and `actuated_samples` then evaluates the terminal interval
        # outside it: a sample off the end of the certified curve, moving at
        # `snap * overshoot^3 / 6` where the plan says it is stopped. 10 ms of it
        # is what the JTC rejected. The reference ends when the plan does.
        stamps[-1] = timing.duration

        def sample(rows):
            return np.array(
                [
                    np.interp(stamps, timing.time, rows[:, i])
                    for i in range(rows.shape[1])
                ]
            ).T

        q_a, dq_a = actuated_samples(timing, stamps)

        q = np.zeros((len(stamps), GENERALIZED_DOF))
        dq = np.zeros_like(q)
        q[:, list(PLANNED_INDICES)] = q_a
        dq[:, list(PLANNED_INDICES)] = dq_a
        # No closed form for the passive pair: it is an integrated state, not a
        # function of `sigma`, so interpolating it is the only option here.
        q[:, list(PASSIVE_INDICES)] = sample(timing.q_u)
        dq[:, list(PASSIVE_INDICES)] = sample(timing.dq_u)
        q[:, TOOL_INDEX] = start.q_tool
        # The last stamp is the terminal node, where the OCP pins `v = a = j = 0`
        # and `dq_u = 0`: the plan ends stopped. `actuated_samples` reconstructs
        # that sample from the interval before it, so what it reports there is
        # the multiple-shooting gap -- 1e-6 rad/s at `ocp_tolerance`. JTC rejects
        # a goal whose last point moves at all (`float` epsilon, 1.19e-7), so
        # write the row the solve pinned and refuse a solve that did not stop.
        drift = float(np.max(np.abs(dq[-1])))
        if drift > TERMINAL_DRIFT_MAX:
            raise PlanningError(
                f"the plan ends at {drift:.3e} rad/s, above the "
                f"{TERMINAL_DRIFT_MAX:.0e} terminal floor: the solve did not "
                "arrive stopped"
            )
        dq[-1] = 0.0
        # `c(1)` *is* the lifted goal, `geometry.fit` writes it there. What the
        # reconstruction lands on is `c(1 - gap)`, so take the certified end.
        q[-1, list(PLANNED_INDICES)] = evaluate(timing.coefficients, 1.0)[0]

        # The tool where the plan says it is, sway included -- not where it would
        # hang if the machine stopped at each sample. This is visualization, not
        # the control reference, so evaluate a bounded number of poses.
        visual_count = min(len(q), int(self.config.visualization_samples))
        visual_indices = np.unique(np.linspace(0, len(q) - 1, visual_count, dtype=int))
        tcp = np.array(
            [
                geometry.model.forward_kinematics(
                    row, Frame.MOUNTING_BASE, Frame.TCP
                ).position_m
                for row in q[visual_indices]
            ]
        )
        message = (
            f"{candidate_name}; {timing.duration:.2f} s, {lifted} lifted "
            f"configurations, {len(stamps)} points at {Ts * 1e3:.0f} ms; "
            f"{timing.iterations} SQP iterations in {timing.solve_time_s:.2f} s; "
            f"peak pump draw {np.max(timing.pump_flow):.2f} of the physical limit "
            f"at kappa = {self.config.kappa}"
            + (f", on {timing.slack:.3f} of slack" if timing.slack > 1e-9 else "")
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


#: Re-exported so `crane_planning.planner` stays the one import site it has been.
__all__ = [
    "ACTUATED_INDICES",
    "ARM_AXIS",
    "BOOM_AXIS",
    "CartesianCandidate",
    "CollisionPrimitive",
    "GENERALIZED_DOF",
    "Geometry",
    "Limits",
    "PASSIVE_INDICES",
    "PAYLOAD_ID",
    "PLANNED_DOF",
    "PLANNED_FRAMES",
    "PLANNED_INDICES",
    "Path",
    "Plan",
    "Planner",
    "PlannerConfig",
    "PlanningError",
    "Start",
    "TELESCOPE_AXIS",
    "TOOL_INDEX",
    "TRUCK_ID",
    "Trajectory",
    "TrajectoryOcp",
    "cartesian_candidates",
    "expand_truck",
    "fit",
    "lift_candidate",
    "passive_equilibrium",
    "payload_parameters",
    "payload_primitive",
    "pin",
    "plan_fitted_path",
    "read_limits",
    "solve_ik",
    "wrap",
    "yaw_of",
]
