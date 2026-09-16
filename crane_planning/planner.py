"""
Plan a motion, and time it.

Two stages. `geometry` walks bounded family of TCP corridors, lifts one to joint
space, certifies fitted curve; `ocp` gets that curve, solves time-optimal way
*along* it. Executes what was certified.

Everything machine can do -- reach, hang, collide -- asked of `crane_model`; no
machine knowledge here.

Passive pair (tip/tilt sway) never searched. Where it hangs is closed form,
`q_eq = (pi/2 - q_boom - q_arm, pi/2)`, good to 1e-4 rad workspace-wide, independent
of slew, telescope, rotator, tool, payload. It is OCP *state*, so answer is
trajectory tool arrives nearly still from.
"""

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

# One import site for consumer, three modules for author. `config` = what plan is
# written in, `geometry` = where machine may go, `timing` = how fast. This file runs
# them in order.
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

# ------------------------------------------------------------------ the planner


#: rad/s (m/s on telescope). Below this, measured start velocity is rest.
#: **Measured on standing machine** (live `/joint_states`, 993 samples over 10 s):
#: per-axis rms 0.002-0.005, max 0.0098 actuated five, 0.0126 passive pair. Old 1e-3
#: was a Gazebo-scale guess, 3-10x *under* that floor, so deadband never fired on
#: hardware: noise read as motion, `start_speed` pinned `v(0)` on it against a path
#: clamped to leave at rest, every plan died in the first QP -- acados status 4, one
#: iteration, stationarity 1e3. 2e-2 = twice measured max, 2-4% of `dq_max`: creep
#: this slow is rest.
REST_VELOCITY = 2e-2

#: m, rad. How far outside its limit a measured coordinate may sit and still be
#: planned from. Joint on its stop reads a hair past it -- Gazebo settles retracted
#: telescope at -8e-5, excursion *creeps* every move, so without this first plan
#: ending fully retracted is last plan that succeeds. Same argument as
#: `REST_VELOCITY`, one derivative down.
#:
#: Clamped into box, not merely tolerated: everything downstream -- IK, fit range
#: check, OCP state box -- needs start inside it. Equals `eps_pos`, planner's
#: positional acceptance, so projection this small is indistinguishable from IK
#: solution it would accept. Past it, refusal stands: state grossly out of range
#: means wrong description, projecting hides it.
LIMIT_DEADBAND = 1e-3

#: rad/s. Drift of last emitted sample from rest OCP pinned, above which plan is
#: refused. Deliberately not `REST_VELOCITY`: that claims something about encoder,
#: meant to rise to measured noise floor, which must not quietly loosen what "solve
#: stopped" means.
TERMINAL_DRIFT_MAX = 1e-3


@dataclass
class Start:
    """Measured state a plan leaves from."""

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
    """
    `q_a`, `dq_a`, `ddq_a` at `stamps`, on curve not chord.

    `q_a = c(sigma)` is identity solve never leaves, so `q_a` is no row to
    interpolate: chord between nodes is off the certified curve. Reconstruct `sigma`
    and evaluate instead -- exactly, not by interpolation: acados holds input
    constant per interval and `sigma'''' = s` there, so `sigma` quartic in elapsed
    time.
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
    #: C3 inversion as correction JTC effort field carries,
    #: `u(t + dead time) - dq_d(t)`, zero on passive pair and tool.
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
    """
    One description, one set of limits, many requests.

    Built once: parsing description, exporting OCP solver -- here, not per request.
    """

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
        """
        Answer a placement goal in `K0_mounting_base` with a timed trajectory.

        Stages refuse, never degrade: unreachable goal, no corridor surviving its
        certificate, untimeable path -- three answers, each names itself.
        """
        if not 0.0 < speed_scale <= 1.0:
            raise PlanningError(f"speed_scale must be in (0, 1], not {speed_scale}")
        start = self._validate_start(start)
        # The control-safe box may never exclude the measured pose; everything
        # below plans inside this one rather than `self.limits`.
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

        # Reachable at all? Cold solve, restarts spread over telescope range -- the
        # coordinate residual is flat in. Cartesian corridors discard its
        # configuration: which member of redundant family arm ends at is set by
        # marching there; pinning an independently chosen one is discontinuity no
        # refinement closes. Joint-space line goes there directly.
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
            # Named: "it said no" vs "it said no, and the block it holds is 0.6 m
            # inside the tool" go to different people.
            if body is None:
                raise PlanningError(
                    "the measured start configuration cannot be measured against "
                    "the scene at all: the machine is folded into itself there"
                )
            raise PlanningError(
                f"the measured start configuration clears '{body}' by only "
                f"{clearance:.3f} m against the {required:.3f} m this plan requires"
            )

        # Measured velocity never exactly zero. Below noise floor it is rest: fitting
        # tangent pins `c'(0)` at noise magnitude, OCP must grow that to move scale
        # within one knot span, and that acceleration breaks bound at a node where
        # nothing helps. Before this every start with live encoder was refused.
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

        # Certified curve itself, not endpoints. Solve moves along it, decides only
        # how fast, so what executes is what was proved clear.
        timing = self.ocp.solve(
            coefficients=power_coefficients(path, self.ocp.segments),
            q_u_start=q_u_start,
            dq_a_start=dq_a_start,
            dq_u_start=dq_u_start,
            payload=payload_vector,
            q_tool=start.q_tool,
            speed_scale=speed_scale,
        )
        # **Certificate rests on this row, no other.** `Geometry` builds clearance
        # envelope from `q_sway_max` and geometric stage proves corridor clear against
        # it; in OCP that box is *soft*, priced at `ocp_slack_price`. On a request it
        # cannot otherwise meet, solve may buy its way out instead of refusing and
        # return a plan swinging outside the certified envelope -- converged, every
        # residual clean, so nothing upstream notices.
        #
        # No provoking request known: on shipped fixture every over-constrained
        # request tried non-converges instead of paying. Row is purchasable either
        # way, and two near-misses stalled at stationarity 1e-4 to 1e-3 -- a solve
        # that pays given iterations.
        #
        # Refused here, not priced higher: raising price trades violation for
        # non-convergence -- same plan missing, worse diagnosis. `slack` rides the
        # message so flow row's share shows too; that one is over-draw, not breached
        # proof, stays a log line.
        if timing.sway_slack > SLACK_SPENT:
            raise PlanningError(
                "the timing solve bought the sway box instead of meeting it: "
                f"{timing.sway_slack:.3g} of slack on the sway rows "
                f"({timing.sway_slack * 100.0:.1f}% of q_sway_max) out of "
                f"{timing.slack:.3g} total, so the tool leaves the clearance "
                "envelope the corridor was certified against",
                stats=timing.report(),
            )
        terminal_offset = np.abs(timing.q_u[-1] - timing.q_u_eq[-1])
        terminal_rate = np.abs(timing.dq_u[-1])
        if np.any(terminal_offset > self.config.terminal_q_sway_max) or np.any(
            terminal_rate > self.config.terminal_dq_sway_max
        ):
            # `timing.stats`, not nothing: this refusal most needs numbers on wire.
            # Clean iteration count, every residual under `ocp_tolerance`, beside a
            # sway solve did not close => solver fine, model it got is not. Empty
            # report says opposite.
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
        """
        Refuse a measured state that is not one state of this description.

        Returns state to plan from: measured one, any coordinate resting within
        `LIMIT_DEADBAND` of its stop clamped onto it.
        """
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
        # The *description's* box, not the control-safe one: this asks whether
        # the measurement can be real. A machine parked outside the control-safe
        # box -- the boom folds below its 0.02 rad -- is normal, and `plan`
        # relaxes the box to wherever it is rather than refusing.
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
        """
        Return the static bodies this plan is checked against.

        Not the list the request carried: `truck` expands to bed, six runges,
        headboard -- invisible on the scene topic. Public because node draws it;
        refusal reads far better beside geometry causing it.

        What the tool carries is **not** here: it moves, so placed at each checked
        configuration instead of pinned to scene once.
        """
        if not avoid_collisions:
            return []
        primitives = expand_truck(list(scene or []), self.config)
        if any(primitive.id == PAYLOAD_ID for primitive in primitives):
            raise PlanningError(f"'{PAYLOAD_ID}' is a reserved scene id")
        return primitives

    def _settled(self, q_a: np.ndarray, q_tool: float) -> np.ndarray:
        """Return canonical eight at `q_a` with the passive pair hanging."""
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

        `a2b_movement` names pivot, native goal names tool, so adapter needs vector
        between. Read from model at hanging equilibrium, never written down.

        Passive pair makes settled offset independent of boom and telescope pose:
        once settled, only rotation about gravity moves it. So measure description's
        yaw at canonical pose, turn slew by difference, settle again, *check* result
        carries requested yaw.

        `payload` does not enter -- hanging pose is closed form in boom and arm
        angles alone. Argument kept so call site stays honest.
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

        OCP solves on uniform grid over its horizon, consumer reads at `Ts`, so every
        row is interpolated against elapsed time. Passive pair carried as OCP
        **planned** it, not as pose tool would settle to: en route tool swings, and
        that is what trajectory claims.

        `q_a` is the exception, and the whole point of the method: `q_a = c(sigma)`,
        so interpolating between nodes reports the chord -- configurations plan never
        held, off the certified curve. `actuated_samples` reconstructs `sigma`
        exactly and evaluates the curve there.
        """
        Ts = float(self.config.Ts)
        stamps = np.arange(0.0, timing.duration + 0.5 * Ts, Ts)
        # Rounding to nearest puts last stamp up to `Ts / 2` *past* solve's own end,
        # and `actuated_samples` then evaluates terminal interval outside it: sample
        # off end of certified curve, moving at `snap * overshoot^3 / 6` where plan
        # says stopped. 10 ms of it is what JTC rejected. Reference ends when plan
        # does.
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
        # JTC interpolates cubic between knots without accelerations, quintic with
        # them. Feedforward comes off C4 curve, so cubic reconstruction of tracked
        # reference would put the two branches on different curves. Passive pair gets
        # none: nothing commands it, not a function of `sigma`.
        ddq = np.zeros_like(q)
        q[:, list(PLANNED_INDICES)] = q_a
        dq[:, list(PLANNED_INDICES)] = dq_a
        ddq[:, list(PLANNED_INDICES)] = ddq_a
        # No closed form for passive pair: integrated state, not a function of
        # `sigma`, so interpolating is the only option.
        q[:, list(PASSIVE_INDICES)] = sample(timing.q_u)
        dq[:, list(PASSIVE_INDICES)] = sample(timing.dq_u)
        q[:, TOOL_INDEX] = start.q_tool
        # Last stamp is terminal node, where OCP pins `v = a = j = 0` and
        # `dq_u = 0`: plan ends stopped. `actuated_samples` rebuilds that sample from
        # interval before, so it reports multiple-shooting gap -- 1e-6 rad/s at
        # `ocp_tolerance`. JTC rejects a goal whose last point moves at all (`float`
        # epsilon, 1.19e-7), so write the row solve pinned and refuse a solve that
        # did not stop.
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
        # `c(1)` *is* the lifted goal, `geometry.fit` writes it there. Reconstruction
        # lands on `c(1 - gap)`, so take certified end.
        q[-1, list(PLANNED_INDICES)] = evaluate(timing.coefficients, 1.0)[0]

        # C3 feedforward, from the `u` OCP already solved for and bounded, not a
        # second derivation. Effort field carries *correction* `u(t + n_d) - dq_d(t)`,
        # so plugin's `dq_d + effort` is inversion advanced by dead time, and a
        # controller reading a reference without one degrades to static feedforward,
        # not to none. `np.interp` holds terminal value past plan end, which is zero:
        # `u = dq_a + tau_dot_a / k`, plan ends at rest.
        effort = np.zeros_like(q)
        preview = stamps + float(self.config.command_dead_time_s)
        commanded = np.array(
            [
                np.interp(preview, timing.time, timing.command[:, i])
                for i in range(timing.command.shape[1])
            ]
        ).T
        # Block 2, command PT1, when an arm asks for it. `u_f` chases `u` through
        # `tau_v`, so commanding `u + tau_v du/dt` lands `u_f` on inversion;
        # `command_lag_s` = 0 leaves `commanded` untouched, the shipped law. Extra
        # derivative is `q_a''''` -- why path is C4.
        #
        # **Cubic through OCP nodes, differentiated analytically -- not a central
        # difference on that grid.** `np.interp` is piecewise linear, so
        # differentiating it gives a staircase (why old code differentiated first);
        # `np.gradient` on OCP grid is no better. Grid runs ~183 ms on a nominal move
        # vs 40 ms emitted samples, and `u` carries curvature between nodes (snap
        # piecewise constant on `path_segments` breakpoints). Central difference
        # flattens `u'` peaks and smears them onto neighbours: vs this spline on
        # `stow` bench move it lost 83% of slewing peak, 93% of rotator's. Lead term
        # *is* `tau_v u'`, so attenuated, displaced `u'` pushes where it should not
        # and barely where it should -- worse than no feedforward, as machine showed.
        #
        # Preview clamped into plan, not extrapolated: past `T` command is held, so
        # slope held too. Not cosmetic -- preview reaches `command_dead_time_s` past
        # last stamp on every plan, and a cubic out there runs away.
        #
        # Clipped to identified domain: feedforward branch is bounded at its source
        # and this term is likeliest to ask for command machine lacks. Saturation
        # count reported, not swallowed.
        lag = np.asarray(self.config.command_lag_s, dtype=float)
        lag_saturation = 0.0
        if np.any(lag != 0.0):
            slope = CubicSpline(
                timing.time,
                timing.command,
                axis=0,
                # Natural, not scipy's not-a-knot default, and not clamped.
                # Not-a-knot *extrapolates* curvature through last interval and ran
                # `u'` to plan's largest value in final three samples. Clamping
                # `u'(T)` to zero fixes tail, but cubic spline is **not local** --
                # end condition perturbs every interval, so a ramp stops being a ramp
                # long before boundary and block-2 law is no longer `tau_v u'`
                # anywhere. Natural claims no curvature at both ends, reproduces ramp
                # exactly; tail handled locally by fade below.
                bc_type="natural",
            ).derivative()
            rate = slope(np.clip(preview, timing.time[0], timing.time[-1]))
            # Faded to zero across final OCP interval, smoothstep so lead stays C1.
            # Feedforward must *land* at zero (JTC holds last point, `effort[-1]`
            # pinned below); question is smooth or step. It stepped: on `stow` bench
            # move slewing lead climbed to 0.0312 rad/s at `T`, cut to zero in one
            # 40 ms sample -- 0.03 rad/s impulse into axis as it arrives, the
            # end-of-move ringing the lag arm showed and static arm did not. Fade is
            # local, so `tau_v u'` untouched wherever last interval does not reach --
            # why ramp fixture still asserts law exactly.
            #
            # Interval is right width, not tuned: `u` is least trustworthy there
            # anyway. Solve pins `v = a = j = 0` at terminal node, and node before
            # carries one-node kink (u: -0.0102, -0.0363, -0.00003 on slewing) no
            # derivative of `u` can tell from signal.
            span = timing.time[-1] - timing.time[-2]
            x = np.clip((timing.time[-1] - preview) / span, 0.0, 1.0)
            rate = rate * (x * x * (3.0 - 2.0 * x))[:, None]
            raw = commanded + lag * rate
            commanded = np.clip(
                raw, self.config.command_u_min, self.config.command_u_max
            )
            lag_saturation = float(np.mean(raw != commanded))
        effort[:, list(PLANNED_INDICES)] = commanded - dq[:, list(PLANNED_INDICES)]
        # Last point held, so its feedforward is not transient: JTC keeps sampling it
        # after plan ends, PID plugin keeps adding it, forever. Fatal on slewing --
        # `p: 0.07`, `i: 0` means loop settles where `p e_pos` cancels it, so held
        # -0.0276 rad/s buys standing 0.39 rad position error the 28 s trim loop never
        # removes. Measured 0.355 rad and still moving. Plan ends at rest, so answer
        # is zero; pinned like `dq[-1]` and `ddq[-1]`.
        effort[-1] = 0.0

        # Tool where plan says it is, sway included -- not where it would hang if
        # machine stopped at each sample. Visualization, not control reference, so
        # evaluate bounded number of poses.
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


#: Re-exported so `crane_planning.planner` stays the one import site.
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
