"""C3 feedforward command (u_d = dq_a + tau_dot_a/k) is a hard OCP row, not the old dddq_a proxy."""

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
from crane_planning.ocp import H_COMMAND, NH, baked_parameters, build_ocp
from crane_planning.planner import Planner, Start

#: stow, bench_plan.py's move closest to the bound (0.75 of it on the boom)
START = (0.6, 0.7, 0.9, 1.4, 0.4)
GOAL = (0.0, 0.25, 0.35, 0.7, 0.0)
TOOL_POSITION = 0.45


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


def test_the_command_rows_are_present_and_hard():
    config = PlannerConfig()
    baked = {**baked_parameters(config), "weights": crane_weights.DEFAULTS}
    ocp, _scale, _model = build_ocp(
        description(), baked, {"pump_flow_max": config.pump_flow_max}
    )

    assert ocp.model.con_h_expr.shape[0] == NH
    command_rows = set(range(H_COMMAND, NH))
    # a command the valve cannot deliver is not a slower plan, it's an untrackable one
    assert command_rows.isdisjoint(set(np.atleast_1d(ocp.constraints.idxsh).tolist()))


def test_a_solved_trajectory_respects_the_command_bound():
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

    reservation = config.kappa
    assert np.all(plan.timing.command <= reservation * config.command_u_max + 1e-6), (
        f"command {plan.timing.command.max(axis=0).tolist()} over "
        f"{(reservation * config.command_u_max).tolist()}"
    )
    assert np.all(plan.timing.command >= reservation * config.command_u_min - 1e-6), (
        f"command {plan.timing.command.min(axis=0).tolist()} under "
        f"{(reservation * config.command_u_min).tolist()}"
    )
