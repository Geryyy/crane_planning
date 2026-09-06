"""The jerk the C3 feedforward can pay for is a row in the OCP, and a hard one.

`wiki/controller_design.md` section 4.4: the inversion is
`u = v + (2 zeta / w_n) v' + v'' / w_n^2`, so its excursion grows as the cube of the
inverse rise time, and a reference whose third derivative the command cannot afford
saturates the valve -- at which point the feedforward is no longer one. That page
ends "nothing on this page is shippable without it", and says the bound belongs to
the planner rather than to a post-hoc check.

Two properties, and they fail differently. If the row is missing the solve answers
with a reference the machine cannot track. If the row is made *soft* the solve
answers with a priced violation, which is the same failure wearing a price tag --
so the hardness is asserted, not just the presence.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from crane_model import ACTUATED_INDICES, Frame
from crane_planning import weights as crane_weights
from crane_planning.config import (
    PLANNED_INDICES,
    TOOL_INDEX,
    PlannerConfig,
)
from crane_planning.geometry import yaw_of
from crane_planning.ocp import H_JERK, NH, baked_parameters, build_ocp
from crane_planning.planner import Planner, Start

#: `stow`, the `bench_plan.py` move that comes closest to the bound -- 0.75 of it
#: on the boom, measured before the row existed.
START = (0.6, 0.7, 0.9, 1.4, 0.4)
GOAL = (0.0, 0.25, 0.35, 0.7, 0.0)
TOOL_POSITION = 0.45


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


def test_the_jerk_rows_are_present_and_hard():
    config = PlannerConfig()
    baked = {**baked_parameters(config), "weights": crane_weights.DEFAULTS}
    ocp, _scale, _model = build_ocp(
        description(), baked, {"pump_flow_max": config.pump_flow_max}
    )

    assert ocp.model.con_h_expr.shape[0] == NH
    jerk_rows = set(range(H_JERK, NH))
    # Soft rows are the ones a caller would rather have late than refused. A jerk
    # the command cannot pay for is not a slower plan, it is an untrackable one.
    assert jerk_rows.isdisjoint(set(np.atleast_1d(ocp.constraints.idxsh).tolist()))


def test_a_solved_trajectory_respects_the_jerk_bound():
    config = PlannerConfig()
    planner = Planner(description(), config, dict(crane_weights.DEFAULTS))

    def configuration(coordinates):
        q = np.zeros(8)
        q[list(PLANNED_INDICES)] = coordinates
        q[TOOL_INDEX] = TOOL_POSITION
        q[[4, 5]] = planner.model.passive_equilibrium(q[list(ACTUATED_INDICES)])
        return q

    goal = configuration(GOAL)
    pose = planner.model.forward_kinematics(goal, Frame.MOUNTING_BASE, Frame.TCP)
    yaw = yaw_of(
        pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
    )
    plan = planner.plan(
        Start(q=configuration(START), dq_a=np.zeros(5)),
        pose.position_m,
        yaw,
        scene=[],
        avoid_collisions=True,
    )

    peak = np.max(np.abs(plan.timing.dddq_a), axis=0)
    assert np.all(peak <= config.dddq_a_max + 1.0e-6), (
        f"peak jerk {peak.tolist()} against {config.dddq_a_max.tolist()}"
    )
