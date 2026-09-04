"""The weights a caller passes must reach the *built* solver, not just the export.

`W` enters the generated code at code-export time, and `TrajectoryOcp` loads a
cached `.so` rather than regenerating it -- so a weight that is only written into
the `AcadosOcp` object is inert on every run but the one that built the tree.
`BAKED` excludes the weights on purpose, so a changed weight does not invalidate
the cache either, and the whole of `config/crane_planner.yaml`'s `weights` block
goes quietly dead. Measured before the fix: `weights.time` swept 0.3 to 10 gave
durations equal to the centisecond and identical iteration counts.

Costs one solver build on a cold cache. That is the price of covering a
configuration knob that fails silently.
"""

from pathlib import Path

import numpy as np
import pytest
from crane_planning import weights as crane_weights
from crane_planning.config import PlannerConfig, read_limits
from crane_planning.ocp import TrajectoryOcp

#: The terminal residual's last row is `theta`, so `weights.time` is the corner
#: of `W_e` -- one number that says whether the block arrived.
TIME_ROW = -1


def description() -> str:
    """The expanded PZS100 description crane_model keeps as a fixture."""
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
