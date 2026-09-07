"""
What a plan is written in: coordinates, limits and preferences.

The machine's numbers, the solve's numbers, and the canonical indices both are
indexed by. Imported by every other module here and importing none of them,
which is what keeps `geometry` and `ocp` from knowing about each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crane_model import ACTUATED_INDICES, Frame, Tool, parse

#: The five coordinates a plan moves, in canonical indices. The tool axis (q8)
#: is held by the low-level controller and rides the path at a constant value.
PLANNED_INDICES = ACTUATED_INDICES[:5]
TOOL_INDEX = ACTUATED_INDICES[5]
PLANNED_DOF = len(PLANNED_INDICES)

#: The frame each planned coordinate turns about, for the step bound below. The
#: telescope is prismatic and has no radius, which is what `None` marks.
PLANNED_FRAMES = (
    Frame.SLEWING_COLUMN,
    Frame.BOOM,
    Frame.ARM,
    None,
    Frame.ROTATOR,
)
TELESCOPE_AXIS = 3

#: Metres of tip travel per metre of `q4`. **Two, not one:** `q4` is stage-1
#: travel and `q5_small_telescope` mimics it at multiplier 1, so the tip advances
#: `2 q4`. A step bound built on 1.0 lets a telescope-dominated segment move the
#: tool twice as far as it is checked for, which is the whole margin.
TELESCOPE_TRAVEL_PER_UNIT = 2.0

#: Where the adaptive march starts and where it gives up. The first is a guess --
#: it halves on violation, grows back over a clear run. The second is what an IK
#: branch change runs into.
INITIAL_LIFT_STEP = 1.0 / 32.0
MIN_LIFT_STEP = 1.0e-6

#: The two rows of the boom four-bar the hanging pose tracks, in planned axes.
BOOM_AXIS = 1
ARM_AXIS = 2


class PlanningError(RuntimeError):
    """A refusal. The message is what the service answer carries."""


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    """What the machine may do, read from the description."""

    lower: np.ndarray  # position, per planned coordinate; -inf where continuous
    upper: np.ndarray
    bounded: np.ndarray  # False for a `continuous` joint, which has no range
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
    Position and velocity out of the description; the pump out of the config.

    A `continuous` joint occupies two configuration slots (a cos/sin pair) and
    carries no range; the rotator is one, so it is reported unbounded rather than
    given an invented range here.

    There is deliberately no cylinder-force *constraint*. It was the smaller
    chamber area times a relief pressure nothing in this workspace has measured.
    `tau_max` is the description's own `effort` on each planned joint and is not
    a substitute for it: nothing is bounded by it, it is the scale the OCP's
    effort term is divided by, so that `tau_weight` means "a fraction of rated
    effort" on every axis alike.
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
    """Properties of the solve. Machine numbers are not among them."""

    # Reservation held back from every physical limit so the controller has
    # authority left to correct with.
    kappa: float = 0.8

    # Endpoint IK acceptance, in metres and radians. `ik_restarts` applies to
    # the goal alone; along the line the previous sample is the seed.
    eps_pos: float = 1.0e-3
    eps_yaw: float = 1.0e-3
    ik_restarts: int = 5

    # --- what "clear" means, in metres ---------------------------------------
    #
    # The margin's three-way split, each part spent once: `margin_safety` is the
    # clearance the answer carries, `margin_interp` pays for the gap between two
    # checked configurations, the swing envelope is computed. See `Geometry`.
    margin_safety: float = 0.05
    margin_interp: float = 0.10
    #: How far the tool's collision geometry reaches past the tool centre point.
    #: It enters the step bound as an over-estimate of the machine's outermost
    #: point, so too large costs samples and too small is unsound.
    tool_radius: float = 1.0
    #: The cap on lifted samples. Reaching it means the arm reconfigures faster
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
    #: the two things interpolation welds together: the certificate wants samples
    #: dense, the curve wants its end intervals long.
    path_segments: int = 12

    # --- the trajectory OCP ---------------------------------------------------
    #
    # The generated acados solver of `ocp.py`. `ocp_horizon` is a *scaling*, not a
    # bound: the solver carries `theta = T / ocp_horizon` as a state, so the
    # horizon enters the Jacobian at the same magnitude as every other state, and
    # what actually limits the answer is the duration box below.
    ocp_intervals: int = 40
    ocp_horizon: float = 7.0
    #: `ERK` or `IRK`, see `ocp.INTEGRATORS`. Both order 4. ERK4 is the cheap
    #: default and is accurate at the durations the solve actually lands on;
    #: IRK is Gauss-Legendre, symplectic, and does not damp the swing at the
    #: long end of the duration box where ERK4 does.
    ocp_integrator: str = "ERK"
    ocp_duration_min: float = 2.0
    ocp_duration_max: float = 20.0
    #: The command row is a stiffer direction than the `dddq_a` row it replaced:
    #: `ax_slew` needs 70 iterations against 48 before, and 60 refuses it. Not
    #: higher: a refusal costs the whole budget, and `sh_boom` spends 29.9 s
    #: failing at 200 against a 30 s call timeout in the tree.
    ocp_max_iterations: int = 100
    #: acados defaults to 1e-6 on all four residuals, a control-loop number. This
    #: answer is resampled onto a 25 Hz reference and tracked by a controller that
    #: closes the loop on it, so the last two decades buy nothing and cost every
    #: remaining iteration.
    ocp_tolerance: float = 1.0e-4
    levenberg_marquardt: float = 1.0e-6
    #: The `L1` price on every soft row -- sway, the pump, the settled box. Linear
    #: and large is exact; quadratic leaks a little violation everywhere.
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
    #: C3's actuator stiffness per axis and the domain the compensator was
    #: identified over. The same three lists `AddC3Feedforward` reads from
    #: `bt_server_override.yaml`; planner and feedforward must not carry two
    #: derivations of one number.
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
    #: s. The feedforward the reference carries is advanced by this much, because
    #: the branch it feeds is open loop against a plant that answers late --
    #: inversion and preview are worth x25-x52 together and a fraction of that
    #: apart (controller_design.md 4.3). One number for every axis: the dead time
    #: was pinned common by the fit.
    command_dead_time_s: float = 0.06
    #: s, per axis. Block 2 of C3, the command PT1. **Zero is the shipped law**:
    #: the emitted feedforward then inverts blocks 3 and the rigid body (RNEA,
    #: hydraulic_actuator_model.md 5.3) and block 1 by preview, and block 2 not
    #: at all. Non-zero adds `tau_v du/dt` on top -- the block 2 inversion
    #: controller_design.md 2.4 names, which is why the path is C4. Set it to the
    #: fitted split (sw .100, ha .025, ka 0, sa .075, ro .125) to run that arm;
    #: the Gazebo URDF carries the same numbers as `tau_v`.
    command_lag_s: np.ndarray = field(default_factory=lambda: np.zeros(5))
    visualization_samples: int = 25

    # The pump, which the description does not carry.
    pump_flow_max: float = 1.4e-3
    pump_flow_planning_factor: float = 0.95

    # The emitted reference period.
    Ts: float = 0.04

    # The truck, as a property of the vehicle: the scene carries one primitive
    # with the reserved id `truck` and the bed and six runges are placed on it.
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
    Where the tool hangs, in closed form.

    A two-hinge pendulum hangs straight down: the tip joint takes up whatever the
    boom four-bar accumulated, the tilt joint does not move. Good to 1e-4 rad
    across the workspace against the general grid-plus-Newton solve it replaces,
    and independent of slew, telescope, rotator, tool and payload. 22.6 ms
    against two subtractions -- which is what makes settling the pendulum at
    every configuration the lift checks affordable.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])
