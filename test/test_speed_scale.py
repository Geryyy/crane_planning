"""`slow_down` re-times the full-speed answer; it does not re-solve with tighter ceilings.

Scaling the rate/accel/flow rows by `speed_scale` instead bought 1.3-1.6x the duration
for a factor 2 and 1.7-2.3x for a factor 3, and refused `across` (16 s at full speed) at
both -- what sets that duration is the sway rows, not a kinematic ceiling, so tightening
the ceilings moves the answer somewhere else rather than slowing it down.
"""

import numpy as np
import pinocchio as pin
import pytest
from crane_model import ACTUATED_INDICES, Frame
from crane_planning import weights as crane_weights
from crane_planning.config import (
    PLANNED_INDICES,
    TOOL_INDEX,
    PlannerConfig,
    PlanningError,
)
from crane_planning.geometry import yaw_of
from crane_planning.planner import Planner, Start
from test_jerk_bound import GOAL, START, TOOL_POSITION, description

SLOW_DOWN = 2.0


def plans():
    config = PlannerConfig()
    planner = Planner(description(), config, dict(crane_weights.DEFAULTS))

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

    def plan(speed_scale):
        return planner.plan(
            Start(q=configuration(START), dq_a=np.zeros(5)),
            pose.position_m,
            yaw,
            scene=[],
            avoid_collisions=True,
            speed_scale=speed_scale,
        )

    return planner, plan


def test_a_slowed_plan_is_the_fast_one_stretched():
    planner, plan = plans()

    fast = plan(1.0)
    slow = plan(1.0 / SLOW_DOWN)

    assert slow.duration == pytest.approx(SLOW_DOWN * fast.duration)
    assert slow.timing.duration == pytest.approx(SLOW_DOWN * fast.timing.duration)
    peak = np.max(np.abs(fast.timing.dq_a), axis=0)
    assert np.max(np.abs(slow.timing.dq_a), axis=0) == pytest.approx(peak / SLOW_DOWN)
    # The row every rate ceiling is written on, so the stretch cannot leave the box the
    # full-speed answer met: half of what met it, met twice over.
    assert np.all(peak <= planner.config.kappa * planner.limits.dq_max + 1.0e-6)
    # Emitted reference too, not just the solve's grid: `time_from_start` is what the JTC
    # paces the move by, and leaving it at the full-speed clock would run the machine fast
    # while every velocity row claimed otherwise.
    assert slow.time[-1] == pytest.approx(SLOW_DOWN * fast.time[-1])
    emitted = slow.dq[:, list(PLANNED_INDICES)]
    assert np.max(np.abs(emitted)) == pytest.approx(
        np.max(np.abs(fast.dq[:, list(PLANNED_INDICES)])) / SLOW_DOWN, rel=1.0e-2
    )

    # `kappa` is the deployment's reservation and a scale above 1 would spend it.
    with pytest.raises(PlanningError, match="speed_scale"):
        plan(1.5)
