"""
Plan a motion, and time it.

Two stages. The tool centre point runs a straight line from where it is to where
the goal asks for it; every sample of that line is lifted into the five planned
joint coordinates by inverse kinematics, and the whole machine is checked there.
A bounded shaped-duration search simulates the passive dynamics and certifies how
fast the machine may traverse the result.

Everything the machine can do -- reach, hang, collide -- is asked of
`crane_model`; nothing about the machine is written down here.

The passive pair (tip/tilt sway) is never searched over. Where it hangs is a
closed form -- `q_eq = (pi/2 - q_boom - q_arm, pi/2)`, which
`crane_mpc/src/mpc_node.cpp` measured to 1e-4 rad across the workspace and found
independent of slew, telescope, rotator, tool and payload -- and it is a *state*
of the timing rollout, so the answer is a trajectory the tool arrives nearly
still from.
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
)
from crane_model import symbolic as symbolic_model

from . import weights as crane_weights

# One import site for a consumer, three modules for an author. `config` is what
# a plan is written in, `geometry` is where the machine may go, `timing` is how
# fast it may get there; this file is the stage that runs them in order.
from .config import (
    ARM_AXIS,
    BOOM_AXIS,
    PATH_BLOCK,
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
    lift,
    lift_candidate,
    payload_primitive,
    plan_fitted_path,
    plan_geometric_path,
    solve_ik,
    wrap,
    yaw_of,
)
from .ocp import Trajectory, TrajectoryOcp

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
    One description, one tool, one set of limits, many requests.

    Built once from the robot description -- parsing it and constructing the
    vectorized dynamics rollout both happen here, not per request.
    """

    def __init__(
        self,
        robot_description_xml: str,
        config: PlannerConfig | None = None,
        weights: dict | None = None,
    ):
        self.config = config or PlannerConfig()
        self.model = CraneModel(robot_description_xml, self.config.tool)
        self.limits = read_limits(
            robot_description_xml,
            self.config.tool,
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

        # The OCP is handed the geometric stage's endpoints and nothing else: it
        # is free to move between them however the dynamics prefer, which is where
        # its speed comes from and why the corridor is still owed.
        timing = self.ocp.solve(
            q_a_start=start.q_a,
            q_a_goal=path.position(1.0),
            q_u_start=q_u_start,
            dq_a_start=np.asarray(start.dq_a, dtype=float),
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

        The OCP solves on a uniform grid over its own horizon and the consumer
        reads at `Ts`, so every row is interpolated against elapsed time. The
        passive pair is carried as the OCP **planned** it rather than as the pose
        the tool would settle to: on the way to the goal the tool is swinging, and
        that is what the trajectory claims.
        """
        Ts = float(self.config.Ts)
        stamps = np.arange(0.0, timing.duration + 0.5 * Ts, Ts)
        if stamps[-1] < timing.duration:
            stamps = np.append(stamps, timing.duration)

        def sample(rows):
            return np.array(
                [
                    np.interp(stamps, timing.time, rows[:, i])
                    for i in range(rows.shape[1])
                ]
            ).T

        q = np.zeros((len(stamps), GENERALIZED_DOF))
        dq = np.zeros_like(q)
        q[:, list(PLANNED_INDICES)] = sample(timing.q_a)
        dq[:, list(PLANNED_INDICES)] = sample(timing.dq_a)
        q[:, list(PASSIVE_INDICES)] = sample(timing.q_u)
        dq[:, list(PASSIVE_INDICES)] = sample(timing.dq_u)
        q[:, TOOL_INDEX] = start.q_tool

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
    "PATH_BLOCK",
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
    "lift",
    "lift_candidate",
    "passive_equilibrium",
    "payload_parameters",
    "payload_primitive",
    "pin",
    "plan_fitted_path",
    "plan_geometric_path",
    "read_limits",
    "solve_ik",
    "wrap",
    "yaw_of",
]
