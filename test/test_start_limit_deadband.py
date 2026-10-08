"""Stop is the description's, narrower control-safe box; sim crept -0.000079 to -0.000144 rad."""

import numpy as np
import pytest
from crane_planning import weights as crane_weights
from crane_planning.config import ARM_AXIS, PLANNED_INDICES, TOOL_INDEX, PlannerConfig
from crane_planning.planner import LIMIT_DEADBAND, Planner, PlanningError, Start
from test_jerk_bound import START, TOOL_POSITION, description


def planner() -> Planner:
    return Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))


def start_at(built: Planner, axis: int, position: float) -> Start:
    """Bench start with one planned coordinate moved to `position`."""
    q = np.zeros(8)
    q[list(PLANNED_INDICES)] = START
    q[TOOL_INDEX] = TOOL_POSITION
    q[list(PLANNED_INDICES)[axis]] = position
    return Start(q=q, dq_a=np.zeros(5))


def test_a_joint_resting_on_its_stop_is_planned_from_the_stop():
    built = planner()
    axis = int(np.flatnonzero(built.limits.bounded)[0])
    lower = float(built.limits.description_lower[axis])

    checked = built._validate_start(start_at(built, axis, lower - 1.44e-4))

    assert checked.q_a[axis] == pytest.approx(lower)


def test_a_state_well_outside_its_range_is_still_refused():
    """Deadband is for the stop, not for a wrong description."""
    built = planner()
    axis = int(np.flatnonzero(built.limits.bounded)[0])
    lower = float(built.limits.description_lower[axis])

    with pytest.raises(PlanningError, match="planned coordinate"):
        built._validate_start(start_at(built, axis, lower - 10.0 * LIMIT_DEADBAND))


def test_a_start_inside_its_range_is_handed_back_untouched():
    built = planner()
    start = start_at(built, 0, START[0])

    assert built._validate_start(start) is start


def test_plan_refuses_a_start_outside_the_control_safe_box():
    """
    231: the description admits the arm to 4.6, so `_validate_start` passes
    `initial_pose:=1`'s 1.5708 -- 0.246 rad above the control-safe 1.324864, and
    mechanically fine (stroke 2.1494 of 2.172) but outside what `crane_mpc` can hold.
    Relaxing the row to it let the planner keep the arm outside the box for a whole
    move: 226 §2 measured 0.24-0.41 rad over on all 45 dumped live ticks, with a
    quarter of cycles refusing. `plan` is where that is caught, and only because it
    still calls `relaxed_to`.
    """
    built = planner()
    start = start_at(built, ARM_AXIS, 1.5708)

    with pytest.raises(PlanningError, match="control-safe range"):
        built.plan(start, np.zeros(3), 0.0, scene=[], avoid_collisions=True)
