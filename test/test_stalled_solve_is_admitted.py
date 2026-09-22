"""A solve stopped at the iteration cap is a feasible answer, not a broken one."""

import numpy as np
import pinocchio as pin
import pytest
from crane_model import Frame, presets
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_INDICES, PlannerConfig
from crane_planning.geometry import yaw_of
from crane_planning.ocp import SLACK_SPENT, row_excess
from crane_planning.planner import Planner, Start
from test_jerk_bound import description


def test_row_excess_refuses_what_the_bookkeeping_would_not_see():
    """The gate is not decorative.

    `res_ineq` is one scalar acados may have computed at another iterate, and
    `sway_slack` is a slack variable, not a state. This is the row itself.
    """
    lh, uh = -np.ones(3), np.ones(3)
    sigma, speed = np.linspace(0.0, 1.0, 4), np.zeros(4)

    # endpoints sit on the sigma box, so a clean iterate reaches zero, never above it
    assert row_excess(np.zeros((4, 3)), lh, uh, sigma, speed) <= 0.0

    rows = np.zeros((4, 3))
    rows[2, 1] = 1.05  # one row, one node, 5% past its bound
    assert row_excess(rows, lh, uh, sigma, speed) == pytest.approx(0.05)

    # sigma past 1 runs the answer off the end of the certified curve
    over = np.array([0.0, 0.3, 0.7, 1.2])
    assert row_excess(np.zeros((4, 3)), lh, uh, over, speed) == pytest.approx(0.2)

    # v < 0 runs the machine backwards along its own curve
    back = np.array([0.0, -0.4, 0.1, 0.0])
    assert row_excess(np.zeros((4, 3)), lh, uh, sigma, back) == pytest.approx(0.4)


def test_the_here_preset_plans_and_names_how_it_was_admitted():
    """`goal_here` from `OUTSIDE` is the move that made this a bug.

    A near-vertical 0.99 m lift: TOPP floor 1.04 s against an answer of 6.80 s,
    so no rate row binds and the pendulum alone sets the duration. Minimum time
    goes flat, the SQP cycles between two active sets to the cap, and refusing
    that discarded an answer meeting every row with 3x margin.
    """
    planner = Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))
    start = Start(q=presets.OUTSIDE.copy(), dq_a=np.zeros(len(PLANNED_INDICES)))
    pose = planner.model.forward_kinematics(start.q, Frame.MOUNTING_BASE, Frame.TCP)
    yaw = yaw_of(
        pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
    )

    plan = planner.plan(
        start,
        presets.goal_here(planner.model, start.q),
        yaw,
        scene=[],
        avoid_collisions=True,
    )

    timing = plan.timing
    config = planner.config
    # what makes the answer usable is the rows, not the stationarity residual
    sway = np.max(np.abs(timing.q_u - timing.q_u_eq) / config.q_sway_max)
    assert sway <= 1.0
    assert np.max(np.abs(timing.dq_u) / config.dq_sway_max) <= 1.0
    assert np.max(timing.pump_flow) <= timing.pump_flow_bound + SLACK_SPENT
    assert timing.terminal_sway <= np.max(config.terminal_q_sway_max)
    assert timing.terminal_sway_rate <= np.max(config.terminal_dq_sway_max)

    # and an answer admitted at the cap says so, or an operator reads it as time-optimal
    admitted = timing.stats["acados_status"] != 0
    assert admitted == ("admitted at the iteration cap" in plan.message)
