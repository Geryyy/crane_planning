"""Flow row is soft too -- a converged solve can buy the pump instead of meeting it."""

import dataclasses

import numpy as np
import pytest
from crane_planning.config import PlanningError
from test_sway_certificate import planner_and_goal


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
