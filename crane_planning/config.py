"""Coordinates, limits, preferences a plan is written in; imports nothing else here."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from crane_model import (
    ACTUATED_INDICES,
    CraneModelError,
    Frame,
    Tool,
    hydraulic_limits,
    parse,
)
from crane_model.conventions import (
    CONTROL_SAFE_AXES,
    PLANNED_DOF,
    PLANNED_INDICES,
    TOOL_AXIS,
    canonical_joints,
    control_safe_limits,
)
from crane_model.symbolic import ACTUATOR_FIT
from crane_model.velocity_loop import load_velocity_loop

#: Five planned coordinates; tool axis (q8) held constant by the low-level
#: controller. `PLANNED_*` are `crane_model.conventions`', re-exported here
#: because the rest of this package reads them off config.
TOOL_INDEX = ACTUATED_INDICES[TOOL_AXIS]

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

#: `q_a_margin` multiples the plan keeps clear of a control-safe row. One is what
#: `crane_mpc`'s `position_box` already enforces, so a plan retreating only that far
#: sits *on* the MPC's hard bound and leaves its first interval no interior -- which
#: is what `--goal out` did (telescope pinned to 0.038000, clearance 0.000000; 25% of
#: cycles refused, 226 §3). The second is one `q_a_margin` of room for the machine to
#: be off the plan and still leave the MPC's box an interior -- no new constant, and
#: `q_a_margin` is the travel its own row derives (231).
#: Two caveats, neither closed: the retreat is what `read_limits` returns and
#: `relaxed_to` gives both margins back on an axis whose measurement is inside them,
#: so the clearance holds for a start inside the retreat and not for a parked one;
#: and the margin is one 60 ms dead time of pre-128 travel, where what has to fit is
#: dead time *plus* measurement latency -- at kappa 0.8 and 26 ms that is 5-10 % short
#: on the arm and the slewing column. Sufficient where measured, not derived.
PLAN_MARGINS = 2.0

#: m, rad. How far past a limit a start coordinate may sit and still be planned from.
#: Telescope creeps past its stop each move; against the *description* it is clamped
#: (not tolerated) since IK/fit/OCP need start inside the box. Equals eps_pos, so the
#: projection reads as an accepted IK. `relaxed_to` reuses it as the slack on the
#: control-safe row: axes park on their own stops and sim crept 7.9e-5..1.4e-4 rad past
#: one (`test_start_limit_deadband`), which must not read as a machine out of the box.
LIMIT_DEADBAND = 1e-3


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
    #: The control-safe row itself, margins dropped: how far `relaxed_to` may give way.
    control_safe_lower: np.ndarray
    control_safe_upper: np.ndarray
    dq_max: np.ndarray  # velocity, per planned coordinate
    tau_max: np.ndarray  # rated actuated effort, per planned coordinate
    flow_max: float  # summed pump draw
    tool_lower: float  # held tool-coordinate position
    tool_upper: float
    tool_bounded: bool

    def relaxed_to(self, q_a: np.ndarray) -> Limits:
        """
        Widen rows to contain `q_a`, as far as the control-safe row and no further.

        The box gives way to where the machine is: axes park on their stops, inside
        `PLAN_MARGINS * q_a_margin` of the row, and refusing there would refuse to
        plan at all. Widening rather than clamping is load-bearing -- `plan_fitted_path`
        scans containment against this box at a slack of 1e-6, so a start outside it is
        a refusal however small the excess, which is also why `LIMIT_DEADBAND` past the
        row is tolerated here and not projected onto it. **The cost is that a parked
        axis hands `PLAN_MARGINS`' clearance straight back**: the plan may then use the
        whole band down to the measurement, goal included, so the clearance is the
        box's and not every plan's. Bounding it is the point. Further out the
        measurement is no longer a margin artefact -- `crane_mpc`'s `position_box` pins
        its hard box to exactly that state, so the first interval has no interior and a
        quarter of cycles refuse (226 §2, the arm 0.24-0.41 rad over for a whole move).
        Refused by name rather than planned: the only answer this package can give to a
        machine outside its control-safe range.
        """
        q_a = np.asarray(q_a, dtype=float)
        outside = self.bounded & (
            (q_a < self.control_safe_lower - LIMIT_DEADBAND)
            | (q_a > self.control_safe_upper + LIMIT_DEADBAND)
        )
        if np.any(outside):
            axis = int(np.flatnonzero(outside)[0])
            raise PlanningError(
                f"planned coordinate {axis} is measured at {q_a[axis]:.6f}, outside "
                "crane_model's control-safe range "
                f"[{self.control_safe_lower[axis]:.6f}, "
                f"{self.control_safe_upper[axis]:.6f}] by more than "
                f"{LIMIT_DEADBAND:.0e}; crane_mpc pins its hard position box to that "
                "state, so no plan from here is one it can hold"
            )
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
    `tau_max` is the description's per-joint effort, the scale the
    `dynamic_tau` residual row divides by.

    The position rows retreat `PLAN_MARGINS` margins, not one, so a plan built in
    *this* box never lies on a bound `crane_mpc` also enforces -- boom 0.04..1.523,
    arm -0.868..1.2829, telescope 0.076..2.160, each exactly one `q_a_margin` inside
    `position_box`. `relaxed_to` is what `plan` actually builds in and it gives the
    retreat back where the machine stands inside it, so this is the box's clearance,
    not every plan's. `control_safe_lower/upper` keep the row without any margin,
    which is the furthest `relaxed_to` gives way.

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
    control_safe_lower = lower.copy()
    control_safe_upper = upper.copy()
    for axis, index in enumerate(PLANNED_INDICES):
        name = axis_of[joints[index]]
        dq_max[axis] = min(dq_max[axis], box["dq_a_max"][name])
        if not bounded[axis]:
            continue
        control_safe_lower[axis] = max(lower[axis], box["q_a_lower"][name])
        control_safe_upper[axis] = min(upper[axis], box["q_a_upper"][name])
        retreat = PLAN_MARGINS * box["q_a_margin"][name]
        lower[axis] = control_safe_lower[axis] + retreat
        upper[axis] = control_safe_upper[axis] - retreat
        if lower[axis] >= upper[axis]:
            raise PlanningError(
                f"{joints[index]} has an empty box: the description and "
                "crane_model's control-safe limits do not overlap by more than "
                f"{2.0 * retreat:g}"
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
        control_safe_lower=control_safe_lower,
        control_safe_upper=control_safe_upper,
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

    #: Quintic segments in fitted path, independent of sample count (certificate dense, curve long).
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
        default_factory=lambda: np.array(ACTUATOR_FIT.k, dtype=float)
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
        default_factory=lambda: float(ACTUATOR_FIT.dead_time_s)
    )
    #: s/axis, C3 block 2 PT1. Zero is shipped: feedforward inverts block3/RNEA/block1, not block2.
    #: Non-zero adds tau_v du/dt (path C4); the arm that turns it on is
    #: `crane_model.symbolic.ACTUATOR_FIT.tau_v`, not restated here -- only the
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

    Tip absorbs boom accumulation, tilt fixed. Exact for no payload or a centre
    of mass on the tool axis -- 2e-12 against `CraneModel.passive_equilibrium`
    over the whole box -- for two subtractions against its 1.7 ms. An off-axis
    centre moves the real equilibrium (0.03 rad for a measured crooked grip, as
    wide as the terminal settled box), which is what `settled_offset` adds.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])


def settled_offset(model, payload, q_tool: float):
    """
    Return `(q_a) -> q_u shift` the payload moves the hanging pose by.

    Zero without a valid payload, so the no-payload path stays the closed form.
    The shift turns with the rotator and with nothing else (measured: slewing,
    boom, arm, telescope and the tool coordinate move it by 0), so it is not one
    constant along a path that rotates.

    Zero too where the numeric solve finds no hanging pose inside the passive
    range -- the boom-low/arm-low corner, ~2% of the box, where the closed form
    puts the tip past its own stop. Refusing there would retire poses that plan
    today; the closed form stands, as it did before.
    """
    if payload is None or not getattr(payload, "valid", False):
        return lambda q_a: np.zeros(2)

    def offset(q_a: np.ndarray) -> np.ndarray:
        q_a = np.asarray(q_a, dtype=float)
        actuated = np.concatenate([q_a[:PLANNED_DOF], [q_tool]])
        try:
            settled = model.passive_equilibrium(actuated, payload)
        except CraneModelError:
            return np.zeros(2)
        return settled - passive_equilibrium(q_a)

    return offset
