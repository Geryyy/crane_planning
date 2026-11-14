"""Node 0 is where the machine is, not a row the plan has to meet."""

import numpy as np
from crane_planning.config import PlannerConfig, passive_equilibrium
from crane_planning.ocp import evaluate
from crane_planning.planner import payload_parameters
from test_sway_certificate import planner_and_goal

#: Path units per second along the fitted tangent, i.e. a replan issued mid-move. 0.38 is the
#: most the bench move carries: past it the sway rows at nodes 2-4 and the pump row at node 1
#: refuse for reasons of their own, which no node-0 rule reaches.
START_SPEED = 0.38


def test_a_moving_start_outside_the_command_box_still_gets_a_trajectory():
    """Issue 344: the first QP was infeasible on `command slew at node 0` and the goal
    aborted. `x0` pins the velocity and starts accel/jerk flat, so the command there follows
    from the measurement -- what the plan owns are the rows from node 1 on."""
    planner, start, position, yaw = planner_and_goal()
    coefficients = planner.plan(
        start, position, yaw, scene=[], avoid_collisions=True
    ).timing.coefficients

    timing = planner.ocp.solve(
        coefficients=coefficients,
        q_u_start=passive_equilibrium(evaluate(coefficients, 0.0)[0]),
        dq_a_start=evaluate(coefficients, 0.0, order=1)[0] * START_SPEED,
        dq_u_start=np.zeros(2),
        payload=payload_parameters(None),
        q_tool=start.q_tool,
    )

    config = PlannerConfig()
    box = config.kappa * np.maximum(np.abs(config.command_u_min), config.command_u_max)
    outside = np.max(
        np.maximum(timing.command - box, -box - timing.command) / box, axis=1
    )
    assert outside[0] > 0.1, f"start only {outside[0]:+.3f} outside the command box"
    assert np.max(outside[1:]) <= 1.0e-6, (
        f"command row broken at node {int(np.argmax(outside[1:])) + 1}, "
        f"{np.max(outside[1:]):+.3e} of its bound"
    )
