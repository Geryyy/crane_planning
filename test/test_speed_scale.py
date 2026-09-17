"""`slow_down` divides rate/accel/jerk linearly, not by `s**k`: at s=0.5 on 21 bench
moves, linear answered 3/5 core moves vs 1/5 under `s**k` (rules coincide at s=1)."""

import numpy as np
import pinocchio as pin
from crane_model import ACTUATED_INDICES, Frame
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_INDICES, TOOL_INDEX, PlannerConfig
from crane_planning.geometry import yaw_of
from crane_planning.planner import Planner, Start
from test_jerk_bound import GOAL, START, TOOL_POSITION, description

#: at 0.25 refuses in the QP first iteration (start-guess failure, not a ceiling)
SPEED_SCALE = 0.5


def test_slow_down_divides_every_ceiling_once():
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
    plan = planner.plan(
        Start(q=configuration(START), dq_a=np.zeros(5)),
        pose.position_m,
        yaw,
        scene=[],
        avoid_collisions=True,
        speed_scale=SPEED_SCALE,
    )

    reservation = config.kappa * SPEED_SCALE
    peak_rate = np.max(np.abs(plan.timing.dq_a), axis=0)
    peak_accel = np.max(np.abs(plan.timing.ddq_a), axis=0)
    assert np.all(peak_rate <= reservation * planner.limits.dq_max + 1.0e-6), (
        f"peak rate {peak_rate.tolist()} over {reservation} of "
        f"{planner.limits.dq_max.tolist()}"
    )
    assert np.all(peak_accel <= reservation * config.ddq_a_max + 1.0e-6), (
        f"peak acceleration {peak_accel.tolist()} over {reservation} of "
        f"{config.ddq_a_max.tolist()}"
    )

    # discriminating: under s**k unavailable (accel reaches 1.03-1.48x of s**2's allowance)
    # no jerk assertion: dddq_a_max is a cost scale now, bounds nothing
    assert np.any(peak_accel > config.kappa * SPEED_SCALE**2 * config.ddq_a_max)
