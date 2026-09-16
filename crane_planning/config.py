"""
What a plan is written in: coordinates, limits, preferences.

Machine numbers, solve numbers, canonical indices both index by. Imported by
every other module here, imports none of them -- keeps `geometry` and `ocp`
from knowing about each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crane_model import ACTUATED_INDICES, Frame, Tool, parse
from crane_model.conventions import (
    CONTROL_SAFE_AXES,
    canonical_joints,
    control_safe_limits,
)

#: Five coordinates a plan moves, canonical indices. Tool axis (q8) held by
#: low-level controller, rides the path at constant value.
PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]
PLANNED_DOF = len(PLANNED_INDICES)

#: Frame each planned coordinate turns about, for the step bound below.
#: Telescope is prismatic, no radius -- that is what `None` marks.
PLANNED_FRAMES = (
    Frame.SLEWING_COLUMN,
    Frame.BOOM,
    Frame.ARM,
    None,
    Frame.ROTATOR,
)
TELESCOPE_AXIS = 3

#: Metres tip travel per metre `q4`. **Two, not one:** `q4` is stage-1 travel
#: and `q5_small_telescope` mimics it at multiplier 1, so tip advances `2 q4`.
#: Step bound built on 1.0 lets a telescope-dominated segment move the tool
#: twice as far as it is checked for -- the whole margin.
TELESCOPE_TRAVEL_PER_UNIT = 2.0

#: Where the adaptive march starts and where it gives up. First is a guess --
#: halves on violation, grows back over a clear run. Second is what an IK
#: branch change runs into.
INITIAL_LIFT_STEP = 1.0 / 32.0
MIN_LIFT_STEP = 1.0e-6

#: Two rows of the boom four-bar the hanging pose tracks, in planned axes.
BOOM_AXIS = 1
ARM_AXIS = 2


class PlanningError(RuntimeError):
    """Refusal. Message is what the service answer carries."""

    def __init__(self, message: str, stats: dict | None = None):
        super().__init__(message)
        #: Solver's own numbers where the refusal came out of a solve, so a
        #: non-convergence -- the one outcome building no `Trajectory` to carry
        #: them -- still reports numbers, not only prose. Empty on every
        #: refusal raised before the solve.
        self.stats = dict(stats or {})


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    """What the machine may do, read from the description."""

    lower: np.ndarray  # position, per planned coordinate; -inf where continuous
    upper: np.ndarray
    bounded: np.ndarray  # False for a `continuous` joint -- no range
    dq_max: np.ndarray  # velocity, per planned coordinate
    tau_max: np.ndarray  # rated actuated effort, per planned coordinate
    flow_max: float  # summed pump draw
    tool_lower: float  # held tool-coordinate position
    tool_upper: float
    tool_bounded: bool


def read_limits(
    description_xml: str,
    pump_flow_max: float,
    pump_flow_planning_factor: float,
) -> Limits:
    """
    Read position and velocity from the description, the pump from the config.

    Then intersect with `crane_model`'s control-safe box, which is narrower
    than the description on the boom and the arm and on four velocity rows. It
    has to be intersected and not merely read: the description admits poses the
    four-bar cannot reach, and until it was, this planner certified poses and
    speeds the MPC's constraint 1 hard-refuses -- the boom below 0.02 rad, the
    arm above 1.3039, and references up to 2.7x the MPC's own velocity bound.
    The box carries `q_a_margin` because that is what constraint 1 enforces.

    Two rows are deliberately not intersected. An unbounded axis keeps its
    unboundedness: the rotator is `continuous`, the box bounds it by turn
    counting, and a finite range on a cos/sin slot would mean something else
    here. The tool keeps the description's travel, because the box's tool row
    is an angle on a retired jaw gripper and does not describe this rail --
    that file's `known_gaps` is the long form.

    A `continuous` joint occupies two configuration slots (cos/sin pair) and
    carries no range; the rotator is one, so reported unbounded rather than
    given an invented range here.

    Deliberately no cylinder-force *constraint*: it was the smaller chamber
    area times a relief pressure nothing here has measured. `tau_max` is the
    description's own `effort` per planned joint, not a substitute -- nothing
    is bounded by it, it is the scale the OCP's effort term is divided by, so
    `tau_weight` means "fraction of rated effort" on every axis alike.
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
    """Properties of the solve. No machine numbers."""

    # Reservation held back from every physical limit, so the controller keeps
    # authority to correct with.
    kappa: float = 0.8

    # Endpoint IK acceptance, m and rad. `ik_restarts` for the goal alone;
    # along the line the previous sample is the seed.
    eps_pos: float = 1.0e-3
    eps_yaw: float = 1.0e-3
    ik_restarts: int = 5

    # --- what "clear" means, in metres ---------------------------------------
    #
    # Three-way split, each part spent once: `margin_safety` is the clearance
    # the answer carries, `margin_interp` pays for the gap between two checked
    # configurations, swing envelope is computed. See `Geometry`.
    margin_safety: float = 0.05
    margin_interp: float = 0.10
    #: How far tool collision geometry reaches past the tool centre point.
    #: Enters the step bound as over-estimate of the machine's outermost point:
    #: too large costs samples, too small is unsound.
    tool_radius: float = 1.0
    #: Cap on lifted samples. Reaching it means the arm reconfigures faster
    #: than the line resolves -- usually an IK branch jump -- and is a refusal.
    max_lift_samples: int = 4096

    # Bounded crane-specific alternatives to the direct tool line.
    corridor_clearance: float = 0.15
    corridor_height_step: float = 0.35
    corridor_height_samples: int = 3
    corridor_lateral_step: float = 0.50
    corridor_lateral_samples: int = 2

    q_sway_max: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2]))

    #: Cubic segments in the fitted path, whatever the sample count. Decouples
    #: what interpolation welds together: certificate wants samples dense,
    #: curve wants long end intervals.
    path_segments: int = 12

    # --- the trajectory OCP ---------------------------------------------------
    #
    # Generated acados solver of `ocp.py`. `ocp_horizon` is a *scaling*, not a
    # bound: solver carries `theta = T / ocp_horizon` as a state, so the horizon
    # enters the Jacobian at the same magnitude as every other state. What
    # limits the answer is the duration box below.
    ocp_intervals: int = 40
    ocp_horizon: float = 7.0
    #: `ERK` or `IRK`, see `ocp.INTEGRATORS`. Both order 4. ERK4 is the cheap
    #: default, accurate at the durations the solve lands on; IRK is
    #: Gauss-Legendre, symplectic, does not damp the swing at the long end of
    #: the duration box where ERK4 does.
    ocp_integrator: str = "ERK"
    ocp_duration_min: float = 2.0
    ocp_duration_max: float = 20.0
    #: Command row is a stiffer direction than the `dddq_a` row it replaced:
    #: `ax_slew` needs 70 iterations against 48 before, 60 refuses it. Not
    #: higher: a refusal costs the whole budget, and `sh_boom` spends 29.9 s
    #: failing at 200 against a 30 s call timeout in the tree.
    ocp_max_iterations: int = 100
    #: acados defaults to 1e-6 on all four residuals, a control-loop number.
    #: This answer is resampled onto a 25 Hz reference and tracked by a
    #: controller closing the loop on it, so the last two decades buy nothing
    #: and cost every remaining iteration.
    ocp_tolerance: float = 1.0e-4
    levenberg_marquardt: float = 1.0e-6
    #: `L1` price on every soft row -- sway, pump, settled box. Linear and
    #: large is exact; quadratic leaks a little violation everywhere.
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
    #: No longer a limit -- the scale the jerk *preference* in the cost is
    #: measured in. `H_COMMAND` bounds what the command actually is.
    dddq_a_max: np.ndarray = field(
        default_factory=lambda: np.array([21.2, 2.72, 12.5, 236.0, 44.4])
    )
    #: C3 actuator stiffness per axis and the domain the compensator was
    #: identified over. Same three lists `AddC3Feedforward` reads; planner and
    #: feedforward must not carry two derivations of one number.
    command_k: np.ndarray = field(
        default_factory=lambda: np.array(
            [319074.23, 1834000.0, 578000.0, 3500000.0, 7296.0]
        )
    )
    command_u_min: np.ndarray = field(
        default_factory=lambda: np.array([-0.9635, -0.3134, -0.3342, -0.5551, -2.3969])
    )
    command_u_max: np.ndarray = field(
        default_factory=lambda: np.array([0.9357, 0.2977, 0.3059, 0.5849, 2.4636])
    )
    #: s. Feedforward the reference carries is advanced by this much: the
    #: branch it feeds is open loop against a plant that answers late --
    #: inversion and preview are worth x25-x52 together, a fraction of that
    #: apart. One number for every axis; dead time pinned common by the fit.
    command_dead_time_s: float = 0.06
    #: s, per axis. Block 2 of C3, the command PT1. **Zero is the shipped law**:
    #: emitted feedforward then inverts block 3 and the rigid body (RNEA) and
    #: block 1 by preview, block 2 not at all. Non-zero adds `tau_v du/dt` on
    #: top -- the block 2 inversion, which is why that path is C4. Set it to the
    #: fitted split (sw .100, ha .025, ka 0, sa .075, ro .125) to run that arm;
    #: the Gazebo URDF carries the same numbers as `tau_v`.
    command_lag_s: np.ndarray = field(default_factory=lambda: np.zeros(5))
    visualization_samples: int = 25

    # Pump -- the description does not carry it.
    pump_flow_max: float = 1.4e-3
    pump_flow_planning_factor: float = 0.95

    # Emitted reference period.
    Ts: float = 0.04

    # Truck, as a property of the vehicle: scene carries one primitive with
    # reserved id `truck`; bed and six runges are placed on it.
    truck_runge_dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.28, 0.31, 2.12])
    )
    truck_runge_stations: np.ndarray = field(
        default_factory=lambda: np.array([-2.261, -1.049, 1.935])
    )
    truck_bed_thickness: float = 0.10
    truck_headboard_thickness: float = 0.45
    truck_headboard_height: float = 1.922


# ---------------------------------------------------------------- the hanging pose


def passive_equilibrium(q_a: np.ndarray) -> np.ndarray:
    """
    Return where the tool hangs, in closed form.

    Two-hinge pendulum hangs straight down: tip joint takes up whatever the
    boom four-bar accumulated, tilt joint does not move. Good to 1e-4 rad across
    the workspace against the general grid-plus-Newton solve it replaces, and
    independent of slew, telescope, rotator, tool and payload. 22.6 ms against
    two subtractions -- what makes settling the pendulum at every configuration
    the lift checks affordable.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])
