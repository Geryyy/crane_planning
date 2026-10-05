"""An occupied goal is refused before any corridor is certified (7 min under load otherwise)."""

import time

import numpy as np
import pinocchio as pin
import pytest
from crane_model import ACTUATED_INDICES, CollisionPrimitive, Frame
from test_jerk_bound import GOAL, START, TOOL_POSITION, description

from crane_planning import weights as crane_weights
from crane_planning.config import (
    PLANNED_INDICES,
    TOOL_INDEX,
    PlannerConfig,
    PlanningError,
)
from crane_planning.geometry import yaw_of
from crane_planning.planner import Planner, Start


def test_an_occupied_goal_is_refused_at_once():
    planner = Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))

    def configuration(coordinates):
        q = np.zeros(8)
        q[list(PLANNED_INDICES)] = coordinates
        q[TOOL_INDEX] = TOOL_POSITION
        q[[4, 5]] = planner.model.passive_equilibrium(q[list(ACTUATED_INDICES)])
        return q

    pose = planner.model.forward_kinematics(
        configuration(GOAL), Frame.MOUNTING_BASE, Frame.TCP
    )
    yaw = yaw_of(
        pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
    )
    block = CollisionPrimitive(
        id="block_at_goal",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), pose.position_m),
        dimensions_m=np.array([0.5, 0.5, 0.5]),
    )
    began = time.monotonic()
    with pytest.raises(
        PlanningError, match="goal configuration clears 'block_at_goal'"
    ):
        planner.plan(
            Start(q=configuration(START), dq_a=np.zeros(5)),
            pose.position_m,
            yaw,
            scene=[block],
            avoid_collisions=True,
        )
    assert time.monotonic() - began < 5.0
