"""Weights must reach the *built* solver, not just the export -- W enters generated
code at export time, cached `.so` is loaded not regenerated, so an unbuilt weight is
inert. Before the fix: weights.time swept 0.3-10 gave identical durations/iterations."""

from pathlib import Path

import numpy as np
import pytest
from crane_planning import weights as crane_weights
from crane_planning.config import PlannerConfig, read_limits
from crane_planning.ocp import TrajectoryOcp

#: terminal residual's last row is theta -- weights.time is W_e's corner
TIME_ROW = -1


def description() -> str:
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    return path.read_text()


def build(weights: dict) -> TrajectoryOcp:
    config = PlannerConfig()
    limits = read_limits(
        description(), config.pump_flow_max, config.pump_flow_planning_factor
    )
    return TrajectoryOcp(description(), limits, config, weights)


@pytest.mark.parametrize("time_weight", [0.3, 10.0])
def test_time_weight_reaches_the_built_solver(time_weight):
    weights = {**crane_weights.DEFAULTS, "time": time_weight}
    ocp = build(weights)
    terminal = np.asarray(ocp.solver.cost_get(ocp.N, "W"))
    assert terminal[TIME_ROW, TIME_ROW] == pytest.approx(time_weight)


def test_stage_weights_reach_every_stage():
    weights = {**crane_weights.DEFAULTS, "sway": 7.0}
    ocp = build(weights)
    stage, _terminal = crane_weights.matrices(weights)
    for node in (0, ocp.N // 2, ocp.N - 1):
        assert np.allclose(np.asarray(ocp.solver.cost_get(node, "W")), stage)
