"""
What a plan is written in: coordinates, limits and preferences.

The machine's own numbers, the solve's own numbers, and the canonical indices
both are indexed by. Imported by every other module here and importing none of
them, which is what keeps `geometry` and `timing` from knowing about each other.
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

#: Metres of tip travel per metre of `q4`. **Two, not one.** `q4` is stage-1
#: travel and `q5_small_telescope` mimics it at multiplier 1, so a chain couples
#: the second stage and the tip advances `2 q4` -- `wiki/nomenclature.md` 4. A
#: step bound built on 1.0 lets a telescope-dominated segment move the tool
#: twice as far as it is checked for, which is the whole margin.
TELESCOPE_TRAVEL_PER_UNIT = 2.0

#: `q_a | q_a' | q_a'' | q8` -- what the OCP needs to know about the path at one
#: value of sigma, and the only thing it is told about it.
PATH_BLOCK = 3 * PLANNED_DOF + 1

#: Where the adaptive march starts, and the step at which it gives up. The
#: first is only a guess -- it halves on the first violation and grows back
#: over a clear run -- and the second is what a branch change runs into.
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
    tool: Tool,
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
    description = parse(description_xml, tool)
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

    tool: Tool = Tool.PZS100

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
    # The three-way split of the margin. `margin_safety` is the clearance an
    # answer actually carries; `margin_interp` is what pays for the gap between
    # two checked configurations; the swing envelope is computed, not configured.
    # Each is spent once. See `Geometry`.
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

    #: How many cubic segments the path is fitted with, whatever the sample
    #: count. This is the dial that decouples the two things the old
    #: interpolating fit welded together: the collision certificate wants
    #: samples dense, and the curve wants its end intervals long. Approximating
    #: with far fewer control points than samples gives both.
    path_segments: int = 12

    # --- the trajectory OCP ---------------------------------------------------
    #
    # The generated acados solver of `ocp.py`. `ocp_horizon` is a *scaling*, not a
    # bound: the solver carries `theta = T / ocp_horizon` as a state, so the
    # horizon enters the Jacobian at the same magnitude as every other state, and
    # what actually limits the answer is the duration box below.
    ocp_intervals: int = 40
    ocp_horizon: float = 7.0
    ocp_duration_min: float = 2.0
    ocp_duration_max: float = 20.0
    ocp_max_iterations: int = 60
    #: acados' own default is 1e-6 on all four residuals, which is a control-loop
    #: number: this answer is resampled onto a 25 Hz reference and tracked by a
    #: controller that closes the loop on it, so the last two decades buy a plan
    #: nothing and cost it every remaining iteration.
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

    A two-hinge pendulum under gravity hangs straight down, so the tip joint
    takes up whatever the boom four-bar accumulated and the tilt joint does not
    move at all. `crane_mpc/src/mpc_node.cpp` measured this against the general
    5x5-grid-plus-Newton solve and found it good to 1e-4 rad across the
    workspace, and independent of slew, telescope, rotator, tool and payload --
    against a sway box half-width of 0.2 rad.

    That solve costs 22.6 ms and this costs two subtractions, which is what makes
    it affordable to settle the pendulum at every configuration the lift checks.
    """
    return np.array([0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi])
