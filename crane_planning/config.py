"""Coordinates, limits, preferences a plan is written in; imports nothing else here."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from crane_model import ACTUATED_INDICES, Frame, Tool, hydraulic_limits, parse
from crane_model.conventions import (
    CONTROL_SAFE_AXES,
    canonical_joints,
    control_safe_limits,
)
from crane_model.symbolic import K_ACTUATOR_FIT
from crane_model.velocity_loop import load_velocity_loop

#: Five planned coordinates; tool axis (q8) held constant by the low-level controller.
PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]
PLANNED_DOF = len(PLANNED_INDICES)

#: Frame each planned coordinate turns about; telescope is prismatic (`None` = no radius).
PLANNED_FRAMES = (
    Frame.SLEWING_COLUMN,
    Frame.BOOM,
    Frame.ARM,
    None,
    Frame.ROTATOR,
)
TELESCOPE_AXIS = 3

#: Tip travel per `q4`: two, not one (`q5_small_telescope` mirrors `q4` at multiplier 1).
#: A bound of 1.0 here would let a telescope segment move 2x as far as checked.
TELESCOPE_TRAVEL_PER_UNIT = 2.0

#: Adaptive march start/give-up: guess that halves on violation, grows back over a clear run.
INITIAL_LIFT_STEP = 1.0 / 32.0
MIN_LIFT_STEP = 1.0e-6

BOOM_AXIS = 1
ARM_AXIS = 2


@lru_cache(maxsize=1)
def _command_domain() -> tuple[np.ndarray, np.ndarray]:
    """Psi's identified domain per planned axis, off `crane_model`'s velocity loop."""
    gains, _rate_hz = load_velocity_loop()
    joints = canonical_joints()
    axes = [gains[joints[index]] for index in PLANNED_INDICES]
    return (
        np.array([axis.u_clamp_min for axis in axes], dtype=float),
        np.array([axis.u_clamp_max for axis in axes], dtype=float),
    )


class PlanningError(RuntimeError):
    """Refusal. Message is what the service answer carries."""

    def __init__(self, message: str, stats: dict | None = None):
        super().__init__(message)
        #: Solver numbers when refused from a solve (Trajectory carries none otherwise); empty else.
        self.stats = dict(stats or {})


@dataclass(frozen=True)
class Limits:
    """What the machine may do, read from the description."""

    lower: np.ndarray  # position, per planned coordinate; -inf where continuous
    upper: np.ndarray
    bounded: np.ndarray  # False for a `continuous` joint -- no range
    #: Same two rows pre control-safe box; start-state check alone reads these (outside-box normal).
    description_lower: np.ndarray
    description_upper: np.ndarray
    dq_max: np.ndarray  # velocity, per planned coordinate
    tau_max: np.ndarray  # rated actuated effort, per planned coordinate
    flow_max: float  # summed pump draw
    tool_lower: float  # held tool-coordinate position
    tool_upper: float
    tool_bounded: bool

    def relaxed_to(self, q_a: np.ndarray) -> Limits:
        """Widen rows to contain `q_a`; box gives way to the measured pose, never excludes it."""
        q_a = np.asarray(q_a, dtype=float)
        return replace(
            self,
            lower=np.where(self.bounded, np.minimum(self.lower, q_a), self.lower),
            upper=np.where(self.bounded, np.maximum(self.upper, q_a), self.upper),
        )


def read_limits(
    description_xml: str,
    pump_flow_max: float,
    pump_flow_planning_factor: float,
) -> Limits:
    """
    Read from the description, intersect with `crane_model`'s control-safe box.

    Box is narrower on boom/arm and four velocity rows -- not merely read:
    until intersected this planner certified poses/speeds the MPC's
    constraint 1 refuses (boom below 0.02 rad, arm above 1.3039, velocity
    refs up to 2.7x the MPC bound). Rotator and tool rows stay unintersected
    (continuous joint; tool row is a retired jaw-gripper angle, not this
    rail). No cylinder-force constraint (unmeasured relief pressure);
    `tau_max` is the description's per-joint effort, the scale `tau_weight`
    divides by.

    The two pump numbers come from the caller's `PlannerConfig`, which reads
    them off `crane_model`; they are never defaulted here.
    """
    description = parse(description_xml, Tool.PZS100)
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
    description_lower = lower.copy()
    description_upper = upper.copy()
    box = control_safe_limits()
    axis_of = {joint: axis for axis, joint in CONTROL_SAFE_AXES.items()}
    joints = canonical_joints()
    for axis, index in enumerate(PLANNED_INDICES):
        name = axis_of[joints[index]]
        dq_max[axis] = min(dq_max[axis], box["dq_a_max"][name])
        if not bounded[axis]:
            continue
        lower[axis] = max(lower[axis], box["q_a_lower"][name] + box["q_a_margin"][name])
        upper[axis] = min(upper[axis], box["q_a_upper"][name] - box["q_a_margin"][name])
        if lower[axis] >= upper[axis]:
            raise PlanningError(
                f"{joints[index]} has an empty box: the description and "
                "crane_model's control-safe limits do not overlap"
            )

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
        description_lower=description_lower,
        description_upper=description_upper,
        bounded=bounded,
        dq_max=dq_max,
        tau_max=tau_max,
        flow_max=float(pump_flow_max) * float(pump_flow_planning_factor),
        tool_lower=tool_lower,
        tool_upper=tool_upper,
        tool_bounded=tool_bounded,
    )


@dataclass
class PlannerConfig:
    """Properties of the solve. No machine numbers."""

    # Held back from every physical limit so the controller retains correction authority.
    kappa: float = 0.8

    # IK acceptance, m and rad; ik_restarts is for the goal, line samples reuse the seed.
    eps_pos: float = 1.0e-3
    eps_yaw: float = 1.0e-3
    ik_restarts: int = 5

    # margin_safety is the clearance the answer carries; margin_interp pays checked-config gap.
    margin_safety: float = 0.05
    margin_interp: float = 0.10
    #: Tool collision reach past centre point; over-estimate for the step bound (too small unsound).
    tool_radius: float = 1.0
    #: Cap on lifted samples; reaching it means an IK branch jump, and is a refusal.
    max_lift_samples: int = 4096

    corridor_clearance: float = 0.15
    corridor_height_step: float = 0.35
    corridor_height_samples: int = 3
    corridor_lateral_step: float = 0.50
    corridor_lateral_samples: int = 2

    q_sway_max: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2]))

    #: Cubic segments in fitted path, independent of sample count (certificate dense, curve long).
    path_segments: int = 12

    # ocp_horizon is a *scaling*: solver carries theta = T/ocp_horizon, duration box limits answer.
    ocp_intervals: int = 40
    ocp_horizon: float = 7.0
    #: ERK or IRK (ocp.INTEGRATORS), both order 4; IRK (Gauss-Legendre) damps swing where ERK4 not.
    ocp_integrator: str = "ERK"
    ocp_duration_min: float = 2.0
    ocp_duration_max: float = 20.0
    #: Command row stiffer than dddq_a it replaced: ax_slew needs 70 iters (was 48); sh_boom 29.9s.
    ocp_max_iterations: int = 100
    #: acados default 1e-6 on all residuals; resampled onto 25 Hz closed loop, so decades waste.
    ocp_tolerance: float = 1.0e-4
    levenberg_marquardt: float = 1.0e-6
    #: L1 price on soft rows -- sway, pump, settled box; linear+large exact, quadratic leaks.
    ocp_slack_price: float = 1.0e3

    terminal_q_sway_max: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02])
    )
    terminal_dq_sway_max: np.ndarray = field(
        default_factory=lambda: np.array([0.04, 0.04])
    )
    dq_sway_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5]))
    ddq_a_max: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.7, 0.5, 1.0, 6.0])
    )
    #: No longer a limit -- scale of the jerk preference in cost; H_COMMAND bounds the command.
    dddq_a_max: np.ndarray = field(
        default_factory=lambda: np.array([21.2, 2.72, 12.5, 236.0, 44.4])
    )
    #: C3 stiffness per axis, N m/rad (N/m on the telescope); the same constant
    #: AddC3Feedforward inverts with. Value lives in
    #: `crane_model/config/c3_full_model.json`, which carries the provenance
    #: (telescope a geometry prior, rotator four bags -- indicative, not
    #: measured); read here, never repeated.
    command_k: np.ndarray = field(
        default_factory=lambda: np.array(K_ACTUATOR_FIT.k, dtype=float)
    )
    #: Psi's identified domain; outside it the compensator extrapolates a monotone
    #: spline past its data. Value lives in `crane_model/config/velocity_loop.yaml`
    #: (`u_clamp_min`/`u_clamp_max`), which carries the provenance; read here,
    #: never repeated.
    command_u_min: np.ndarray = field(
        default_factory=lambda: _command_domain()[0].copy()
    )
    command_u_max: np.ndarray = field(
        default_factory=lambda: _command_domain()[1].copy()
    )
    #: s; feedforward advanced by this much (open loop, late plant): inversion+preview worth
    #: x25-x52. Common to every axis -- the fit pinned the dead time common. Value lives in
    #: `crane_model/config/c3_full_model.json`; read here, never repeated.
    command_dead_time_s: float = field(
        default_factory=lambda: float(K_ACTUATOR_FIT.dead_time_s)
    )
    #: s/axis, C3 block 2 PT1. Zero is shipped: feedforward inverts block3/RNEA/block1, not block2.
    #: Non-zero adds tau_v du/dt (path C4); the arm that turns it on is
    #: `crane_model.symbolic.K_ACTUATOR_FIT.tau_v`, not restated here -- only the
    #: lag/dead-time sum is identified, so it is a choice, not a measurement.
    command_lag_s: np.ndarray = field(default_factory=lambda: np.zeros(5))
    visualization_samples: int = 25

    #: The only actuator limit the planner enforces. Value lives in
    #: `crane_model/config/hydraulics.yaml`, which carries the provenance; read
    #: here, never repeated. A deployment overrides it through the node's
    #: parameters, not by typing the number in again.
    pump_flow_max: float = field(
        default_factory=lambda: hydraulic_limits()["pump_flow_max"]
    )
    pump_flow_planning_factor: float = field(
        default_factory=lambda: hydraulic_limits()["pump_flow_planning_factor"]
    )

    Ts: float = 0.04

    #: Truck: scene has one primitive id truck; bed and six runges placed on it.
    truck_runge_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.28, 0.31, 2.12])
    )
    truck_runge_stations: np.ndarray = field(
        default_factory=lambda: np.array([-2.261, -1.049, 1.935])
    )
    truck_bed_thickness: float = 0.10
    truck_headboard_thickness: float = 0.45
    truck_headboard_height: float = 1.922


def passive_equilibrium(q_a: np.ndarray) -> np.ndarray:
    """
    Return where the tool hangs, in closed form.

    Tip absorbs boom accumulation, tilt fixed; good to 1e-4 rad vs the
    grid-Newton solve replaced, 22.6 ms vs two subtractions.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])
