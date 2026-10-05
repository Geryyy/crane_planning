"""Flow row is soft too -- a converged solve can buy the pump instead of meeting it."""

import dataclasses

import numpy as np
import pytest
from test_sway_certificate import planner_and_goal

from crane_planning.config import PlanningError


def test_a_bought_pump_limit_is_refused_and_a_met_one_is_not():
    """Surfaced by the tuning sweep: an answer drawing 1.18 of the pump on 0.419 of
    slack converged, met the sway box, and was published. `plan` refused a sway box
    bought with slack and let this through."""
    planner, start, position, yaw = planner_and_goal()
    request = dict(scene=[], avoid_collisions=True)

    # clean solve first: the gate must not fire on the shipped config's own answer
    plan = planner.plan(start, position, yaw, **request)
    assert np.max(plan.timing.pump_flow) <= plan.timing.pump_flow_bound

    # same answer, having overdrawn the pump to get it
    over = np.full_like(plan.timing.pump_flow, plan.timing.pump_flow_bound)
    over[3] = plan.timing.pump_flow_bound * 1.5
    bought = dataclasses.replace(plan.timing, pump_flow=over, slack=0.419)
    planner.ocp.solve = lambda **_: bought

    with pytest.raises(PlanningError) as refusal:
        planner.plan(start, position, yaw, **request)

    assert "pump" in str(refusal.value)
    # the refusal names both numbers, so "how far over" is not a second query
    assert f"{plan.timing.pump_flow_bound:.3g}" in str(refusal.value)


def test_a_bought_pump_limit_is_solved_again_slower_once():
    """From a swinging start the SQP stopped at 7.7 s on 1.47 of the pump
    while 10.2 s met it. The retry floors the duration past the bought one, by the
    overshoot: flow goes as 1/T, and 1.25x left a 2.1x overdraw refused."""
    planner, start, position, yaw = planner_and_goal()
    request = dict(scene=[], avoid_collisions=True)
    clean = planner.plan(start, position, yaw, **request).timing
    over = np.full_like(clean.pump_flow, clean.pump_flow_bound)
    over[3] = clean.pump_flow_bound * 1.5
    bought = dataclasses.replace(clean, pump_flow=over, slack=0.419)

    floors = []

    def solve(duration_min=None, **_):
        floors.append(duration_min)
        return bought if duration_min is None else clean

    planner.ocp.solve = solve
    plan = planner.plan(start, position, yaw, **request)

    assert np.max(plan.timing.pump_flow) <= plan.timing.pump_flow_bound
    assert floors == [None, pytest.approx(1.5 * bought.duration)]
