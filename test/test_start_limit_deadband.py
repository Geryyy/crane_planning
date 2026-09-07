"""
A joint parked on its stop reads a hair past it, and that is not a bad state.

Observed in sim, four requests in a row after one successful move left the
telescope retracted: `planned coordinate 3 is measured at -0.000079, outside
[0.000000, 2.236000]`, then -0.000091, -0.000104, -0.000144. The excursion
creeps, so the first plan that ends fully retracted was the last plan that
succeeded. `REST_VELOCITY` already carries this argument one derivative up.
"""

import numpy as np
import pytest
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_INDICES, TOOL_INDEX, PlannerConfig
from crane_planning.planner import LIMIT_DEADBAND, Planner, PlanningError, Start
from test_jerk_bound import START, TOOL_POSITION, description


def planner() -> Planner:
    return Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))


def start_at(built: Planner, axis: int, position: float) -> Start:
    """The bench start with one planned coordinate moved to `position`."""
    q = np.zeros(8)
    q[list(PLANNED_INDICES)] = START
    q[TOOL_INDEX] = TOOL_POSITION
    q[list(PLANNED_INDICES)[axis]] = position
    return Start(q=q, dq_a=np.zeros(5))


def test_a_joint_resting_on_its_stop_is_planned_from_the_stop():
    built = planner()
    axis = int(np.flatnonzero(built.limits.bounded)[0])
    lower = float(built.limits.lower[axis])

    checked = built._validate_start(start_at(built, axis, lower - 1.44e-4))

    assert checked.q_a[axis] == pytest.approx(lower)


def test_a_state_well_outside_its_range_is_still_refused():
    """The deadband is for the stop, not for the wrong description."""
    built = planner()
    axis = int(np.flatnonzero(built.limits.bounded)[0])
    lower = float(built.limits.lower[axis])

    with pytest.raises(PlanningError, match="planned coordinate"):
        built._validate_start(start_at(built, axis, lower - 10.0 * LIMIT_DEADBAND))


def test_a_start_inside_its_range_is_handed_back_untouched():
    built = planner()
    start = start_at(built, 0, START[0])

    assert built._validate_start(start) is start
