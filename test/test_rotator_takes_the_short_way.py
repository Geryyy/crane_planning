"""A continuous axis comes back on the turn nearest the seed, not the long way round."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from crane_model import CraneModel, Frame, Tool
from crane_planning.config import PlannerConfig, read_limits
from crane_planning.geometry import Geometry, solve_ik, wrap, yaw_of

ROTATOR_AXIS = 4
Q_TOOL = 0.30

#: `(seed rotator, goal yaw, TCP position)`. Every row is a goal the full-turn
#: search window used to answer past pi off the seed (-3.55, -3.50, -3.48 rad),
#: which is the surplus turn issue 213's 4.761 rad goal accumulated.
CASES = [
    (1.9, np.radians(115.0), (2.80, 1.05, 1.6)),
    (-1.9, np.radians(-30.0), (2.80, 1.05, 1.6)),
    (0.0, np.radians(-140.0), (2.00, 0.75, 1.6)),
]


def stage() -> tuple[Geometry, object, PlannerConfig]:
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    xml = path.read_text()
    config = PlannerConfig()
    limits = read_limits(xml, config.pump_flow_max, config.pump_flow_planning_factor)
    model = CraneModel(xml, Tool.PZS100)
    geometry = Geometry(
        model, limits, [], config, Q_TOOL, np.zeros(4), None, np.zeros(2)
    )
    return geometry, limits, config


@pytest.mark.parametrize(("seed_rotator", "goal_yaw", "position_m"), CASES)
def test_the_goal_solve_stays_within_half_a_turn(seed_rotator, goal_yaw, position_m):
    """
    The rotator's turn count walks otherwise: each move seeds on the last one's
    answer, so a surplus turn is never given back and the MPC ends up tracking a
    reference a full turn off the pose every other run lifts the same block at.
    """
    geometry, limits, config = stage()
    seed = np.array([0.785, 0.5236, 0.5236, 0.25, seed_rotator])
    position_m = np.asarray(position_m, dtype=float)

    answer = solve_ik(
        geometry,
        limits,
        config,
        position_m,
        goal_yaw,
        seed,
        restarts=config.ik_restarts,
    )

    # Still the pose that was asked for -- unwinding is exact, not a compromise.
    pose = geometry.model.forward_kinematics(
        geometry.configuration(answer), Frame.MOUNTING_BASE, Frame.TCP
    )
    rotation = pin.XYZQUATToSE3(
        np.concatenate([pose.position_m, pose.orientation_xyzw])
    ).rotation
    assert np.linalg.norm(pose.position_m - position_m) <= config.eps_pos
    assert abs(wrap(yaw_of(rotation) - goal_yaw)) <= config.eps_yaw
    assert abs(answer[ROTATOR_AXIS] - seed_rotator) <= np.pi
