"""Two-stage planner: geometry certifies a curve, ocp times it; passive pair is closed-form."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
from scipy.interpolate import CubicSpline

from . import weights as crane_weights
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
from .ocp import SLACK_SPENT, Trajectory, TrajectoryOcp, evaluate, power_coefficients

#: rad/s (m/s telescope). Below = rest. Measured standing noise: rms .002-.005, max
#: .0126 rad/s; old 1e-3 was under that floor (noise read as motion, plans died in QP1).
REST_VELOCITY = 2e-2

#: m, rad. How far past a limit a start coordinate may sit and still be planned from.
#: Telescope creeps past its stop each move; clamped (not tolerated) since IK/fit/OCP
#: need start inside the box. Equals eps_pos, so the projection reads as an accepted IK.
LIMIT_DEADBAND = 1e-3

#: rad/s. Terminal drift above which a plan is refused; deliberately not REST_VELOCITY
#: (that's encoder noise, this is what "solve stopped" means -- keep the two separate).
TERMINAL_DRIFT_MAX = 1e-3


@dataclass
class Start:
    q: np.ndarray  # canonical eight
    dq_a: np.ndarray  # planned five
    dq_u: np.ndarray = field(default_factory=lambda: np.zeros(2))  # sway rate
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
    """q_a/dq_a/ddq_a at stamps, on the certified curve -- not the interpolated chord."""
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
    accel = a + j * dt + s * dt**2 / 2.0
    tangent = evaluate(timing.coefficients, sigma, order=1)
    curvature = evaluate(timing.coefficients, sigma, order=2)
    return (
        evaluate(timing.coefficients, sigma),
        tangent * speed[:, None],
        curvature * speed[:, None] ** 2 + tangent * accel[:, None],
    )


@dataclass
class Plan:
    """Trajectory, geometry it was found on, how it was arrived at."""

    time: np.ndarray
    q: np.ndarray  # canonical eight, resampled at Ts, one row per sample
    dq: np.ndarray
    ddq: np.ndarray  # actuated five only, so JTC interpolates quintic
    #: C3 inversion correction u(t+dead time)-dq_d(t); zero on passive pair & tool.
    effort: np.ndarray
    tcp: np.ndarray  # tool position in K0_mounting_base, one row per sample
    timing: Trajectory
    message: str

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def q_a(self) -> np.ndarray:
        """Actuated six -- what native reference carries."""
        return self.q[:, list(ACTUATED_INDICES)]

    @property
    def dq_a(self) -> np.ndarray:
        return self.dq[:, list(ACTUATED_INDICES)]


class Planner:
    """One description, one set of limits, many requests; built once, not per request."""

    def __init__(
        self,
        robot_description_xml: str,
        config: PlannerConfig | None = None,
        weights: dict | None = None,
        build_missing: bool = True,
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
            build_missing,
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
        """Answer a placement goal with a timed trajectory; every stage refuses, not degrades."""
        if not 0.0 < speed_scale <= 1.0:
            raise PlanningError(f"speed_scale must be in (0, 1], not {speed_scale}")
        start = self._validate_start(start)
        # Plans use this box (relaxed to the measured pose), not self.limits directly.
        limits = self.limits.relaxed_to(start.q_a)
        if avoid_collisions and scene is None:
            raise PlanningError(
                "avoid_collisions was asked for with no collision scene: a plan "
                "certified against nothing is not certified"
            )

        primitives = self.prepare_scene(scene, avoid_collisions)
        payload_vector = payload_parameters(payload)
        geometry = Geometry(
            self.model,
            limits,
            primitives,
            self.config,
            start.q_tool,
            payload_vector,
            payload_shape if avoid_collisions else None,
        )

        # Redundant arm config picked by restarts marching to the goal, not pinned
        # independently -- that's a discontinuity no refinement closes.
        goal_q_a = solve_ik(
            geometry,
            limits,
            self.config,
            np.asarray(goal_position_m, dtype=float),
            float(goal_yaw),
            start.q_a,
            restarts=int(self.config.ik_restarts),
        )
        clearance, required, body = geometry.margin(start.q_a)
        if not clearance > required:
            if body is None:
                raise PlanningError(
                    "the measured start configuration cannot be measured against "
                    "the scene at all: the machine is folded into itself there"
                )
            raise PlanningError(
                f"the measured start configuration clears '{body}' by only "
                f"{clearance:.3f} m against the {required:.3f} m this plan requires"
            )

        # Below REST_VELOCITY treat as rest (else fit pins c'(0) to noise, breaks accel bound).
        dq_a_start = np.asarray(start.dq_a, dtype=float)
        if np.max(np.abs(dq_a_start)) < REST_VELOCITY:
            dq_a_start = np.zeros_like(dq_a_start)

        path, lifted, candidate_name = plan_fitted_path(
            geometry,
            limits,
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

        # Solve only times the certified curve; what executes is what geometry proved clear.
        timing = self.ocp.solve(
            coefficients=power_coefficients(path, self.ocp.segments),
            q_u_start=q_u_start,
            dq_a_start=dq_a_start,
            dq_u_start=dq_u_start,
            payload=payload_vector,
            q_tool=start.q_tool,
            speed_scale=speed_scale,
        )
        # Sway box is soft in the OCP (priced at ocp_slack_price) though geometry
        # proved the corridor clear against q_sway_max; a converged solve could buy
        # outside the certified envelope unnoticed, so refused here, not priced higher.
        if timing.sway_slack > SLACK_SPENT:
            raise PlanningError(
                "the timing solve bought the sway box instead of meeting it: "
                f"{timing.sway_slack:.3g} of slack on the sway rows "
                f"({timing.sway_slack * 100.0:.1f}% of q_sway_max) out of "
                f"{timing.slack:.3g} total, so the tool leaves the clearance "
                "envelope the corridor was certified against",
                stats=timing.report(),
            )
        # Flow row is soft for the same reason and refused for the same reason: a converged
        # solve that bought it draws more than the pump delivers, and the answer reads as valid.
        # Surfaced by the tuning sweep -- baseline refused these paths before reaching here.
        pump_peak = float(np.max(timing.pump_flow))
        if pump_peak > timing.pump_flow_bound + SLACK_SPENT:
            raise PlanningError(
                "the timing solve bought the pump limit instead of meeting it: peak "
                f"draw {pump_peak:.3g} of the pump against the {timing.pump_flow_bound:.3g} "
                f"reserved, out of {timing.slack:.3g} total slack",
                stats=timing.report(),
            )
        terminal_offset = np.abs(timing.q_u[-1] - timing.q_u_eq[-1])
        terminal_rate = np.abs(timing.dq_u[-1])
        if np.any(terminal_offset > self.config.terminal_q_sway_max) or np.any(
            terminal_rate > self.config.terminal_dq_sway_max
        ):
            # stats included: clean iterations w/ failed sway means model's wrong, solver isn't.
            raise PlanningError(
                "the timing solve converged but did not arrive settled: terminal sway "
                f"offset {terminal_offset.tolist()} rad against "
                f"{self.config.terminal_q_sway_max.tolist()}, rate "
                f"{terminal_rate.tolist()} rad/s against "
                f"{self.config.terminal_dq_sway_max.tolist()}",
                stats=timing.report(),
            )
        return self._resample(geometry, path, timing, start, lifted, candidate_name)

    def _validate_start(self, start: Start) -> Start:
        """Refuse a state outside this description; clamp within LIMIT_DEADBAND onto it."""
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
        # Description's box, not control-safe one; plan() relaxes to wherever the machine is.
        outside = self.limits.bounded & (
            (q_a < self.limits.description_lower - LIMIT_DEADBAND)
            | (q_a > self.limits.description_upper + LIMIT_DEADBAND)
        )
        if np.any(outside):
            axis = int(np.flatnonzero(outside)[0])
            raise PlanningError(
                f"planned coordinate {axis} is measured at {q_a[axis]:.6f}, outside "
                f"[{self.limits.description_lower[axis]:.6f}, "
                f"{self.limits.description_upper[axis]:.6f}] by "
                f"more than {LIMIT_DEADBAND:.0e}; planning from a projected state "
                "would not match the machine"
            )
        if self.limits.tool_bounded and not (
            self.limits.tool_lower - LIMIT_DEADBAND
            <= start.q_tool
            <= self.limits.tool_upper + LIMIT_DEADBAND
        ):
            raise PlanningError(
                f"the tool coordinate is measured at {start.q_tool:.6f}, outside "
                f"[{self.limits.tool_lower:.6f}, {self.limits.tool_upper:.6f}] by "
                f"more than {LIMIT_DEADBAND:.0e}; fix the simulated state or the "
                "description instead of projecting it"
            )

        projected = q.copy()
        projected[list(PLANNED_INDICES)] = np.where(
            self.limits.bounded,
            np.clip(q_a, self.limits.description_lower, self.limits.description_upper),
            q_a,
        )
        if self.limits.tool_bounded:
            projected[TOOL_INDEX] = np.clip(
                start.q_tool, self.limits.tool_lower, self.limits.tool_upper
            )
        if np.array_equal(projected, q):
            return start
        return replace(start, q=projected)

    def prepare_scene(self, scene, avoid_collisions: bool) -> list:
        """Scene bodies, truck expanded; tool excluded here, placed per checked pose."""
        if not avoid_collisions:
            return []
        primitives = expand_truck(list(scene or []), self.config)
        if any(primitive.id == PAYLOAD_ID for primitive in primitives):
            raise PlanningError(f"'{PAYLOAD_ID}' is a reserved scene id")
        return primitives

    def _settled(self, q_a: np.ndarray, q_tool: float) -> np.ndarray:
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
        Tool's offset from tip pivot K5, for one yaw.

        Settle, turn by yaw, settle again; payload unused (closed form in boom/arm).
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
        """Resample onto the emitted reference's clock; q_a stays on the certified curve."""
        Ts = float(self.config.Ts)
        stamps = np.arange(0.0, timing.duration + 0.5 * Ts, Ts)
        # Rounding puts last stamp up to Ts/2 past solve end; clamp so ref ends when plan does.
        stamps[-1] = timing.duration

        def sample(rows):
            return np.array(
                [
                    np.interp(stamps, timing.time, rows[:, i])
                    for i in range(rows.shape[1])
                ]
            ).T

        q_a, dq_a, ddq_a = actuated_samples(timing, stamps)

        q = np.zeros((len(stamps), GENERALIZED_DOF))
        dq = np.zeros_like(q)
        # JTC: cubic w/o accel, quintic with; feedforward is off the C4 curve so cubic
        # reconstruction would diverge from it.
        ddq = np.zeros_like(q)
        q[:, list(PLANNED_INDICES)] = q_a
        dq[:, list(PLANNED_INDICES)] = dq_a
        ddq[:, list(PLANNED_INDICES)] = ddq_a
        # Passive pair has no closed form (integrated state); interpolation is the only option.
        q[:, list(PASSIVE_INDICES)] = sample(timing.q_u)
        dq[:, list(PASSIVE_INDICES)] = sample(timing.dq_u)
        q[:, TOOL_INDEX] = start.q_tool
        # Last sample rebuilt from the prior interval carries the shooting gap (~1e-6
        # rad/s); write the row the solve pinned instead, since JTC rejects any motion.
        drift = float(np.max(np.abs(dq[-1])))
        if drift > TERMINAL_DRIFT_MAX:
            raise PlanningError(
                f"the plan ends at {drift:.3e} rad/s, above the "
                f"{TERMINAL_DRIFT_MAX:.0e} terminal floor: the solve did not "
                "arrive stopped",
                stats=timing.report(),
            )
        dq[-1] = 0.0
        ddq[-1] = 0.0
        # c(1) is the lifted goal; reconstruction lands on c(1-gap), so use the certified end.
        q[-1, list(PLANNED_INDICES)] = evaluate(timing.coefficients, 1.0)[0]

        # C3 feedforward from the already-solved/bounded u, not a second derivation:
        # effort = u(t+n_d) - dq_d(t); np.interp holds the terminal (zero) value past end.
        effort = np.zeros_like(q)
        preview = stamps + float(self.config.command_dead_time_s)
        commanded = np.array(
            [
                np.interp(preview, timing.time, timing.command[:, i])
                for i in range(timing.command.shape[1])
            ]
        ).T
        # Block2 PT1: u_f chases u via tau_v; lag_s=0 leaves commanded untouched (shipped law).
        #
        # Cubic through OCP nodes, differentiated analytically -- central diff/np.interp
        # flatten u' peaks: lost 83% of slewing peak, 93% of rotator's on `stow` bench,
        # worse than none. Preview clamped (not extrapolated); command clipped to domain.
        lag = np.asarray(self.config.command_lag_s, dtype=float)
        lag_saturation = 0.0
        if np.any(lag != 0.0):
            slope = CubicSpline(
                timing.time,
                timing.command,
                axis=0,
                # Natural, not not-a-knot: not-a-knot ran u' to the plan's largest value in
                # the final 3 samples; natural has zero end curvature, reproduces the ramp.
                bc_type="natural",
            ).derivative()
            rate = slope(np.clip(preview, timing.time[0], timing.time[-1]))
            # Fade to zero over final interval (smoothstep, C1): stepping instead climbed
            # to 0.0312 rad/s at T then cut in one 40ms sample -- ringing seen on the lag
            # arm, not the static one. u least trustworthy there (v=a=j=0 pinned, 1-node kink).
            span = timing.time[-1] - timing.time[-2]
            x = np.clip((timing.time[-1] - preview) / span, 0.0, 1.0)
            rate = rate * (x * x * (3.0 - 2.0 * x))[:, None]
            raw = commanded + lag * rate
            commanded = np.clip(
                raw, self.config.command_u_min, self.config.command_u_max
            )
            lag_saturation = float(np.mean(raw != commanded))
        effort[:, list(PLANNED_INDICES)] = commanded - dq[:, list(PLANNED_INDICES)]
        # Held feedforward isn't transient (JTC samples it forever): -0.0276 rad/s held
        # bought a standing 0.39 rad error the trim loop never removed (measured 0.355 rad).
        effort[-1] = 0.0

        # Tool per plan incl. sway, not settled pose; viz only, bounded sample count.
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
            + (
                f", on {timing.slack:.3g} of slack"
                if timing.slack > SLACK_SPENT
                else ""
            )
            + f"; the tool arrives {np.degrees(timing.terminal_sway):.2f} deg off rest "
            + f"at {timing.terminal_sway_rate:.3f} rad/s"
            + (
                f"; PT1 inversion on, clipped on {lag_saturation:.1%} of samples"
                if np.any(lag != 0.0)
                else ""
            )
            # Named, not just carried in `stats`: every row is met but the duration is not
            # proven minimal, and an operator reading the log cannot tell that from the rest.
            + (
                "; admitted at the iteration cap on stationarity "
                f"{timing.residuals[0]:.1e} -- every row met, duration not proven minimal"
                if timing.stats["acados_status"] != 0
                else ""
            )
        )
        return Plan(
            time=stamps,
            q=q,
            dq=dq,
            ddq=ddq,
            effort=effort,
            tcp=tcp,
            timing=timing,
            message=message,
        )


def payload_parameters(payload: Payload | None) -> np.ndarray:
    """Mass, centre of mass in K8, six independent entries of Theta_L."""
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


#: Re-exported so crane_planning.planner stays the one import site.
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
