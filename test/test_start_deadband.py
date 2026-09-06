"""A measured start velocity at the noise floor is rest, not a tangent to fit."""

import numpy as np
import pinocchio as pin
from crane_model import ACTUATED_INDICES, Frame
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_INDICES, TOOL_INDEX, PlannerConfig
from crane_planning.geometry import yaw_of
from crane_planning.planner import REST_VELOCITY, Planner, Start
from test_jerk_bound import GOAL, START, TOOL_POSITION, description


def test_noise_level_start_velocity_plans_as_rest():
    """
    `fit` pins the curve's start tangent to the measured velocity, so a 1e-5
    encoder reading is a tangent of magnitude 1e-5 that the OCP has to grow to
    the move's scale inside one knot span -- refused at node 0, on every start
    with a live encoder. Below `REST_VELOCITY` the plan is the rest plan.
    """
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
    plans = [
        planner.plan(
            Start(q=configuration(START), dq_a=np.full(5, dq)),
            pose.position_m,
            yaw,
            scene=[],
            avoid_collisions=True,
        )
        for dq in (0.0, -0.1 * REST_VELOCITY)
    ]
    np.testing.assert_allclose(plans[1].timing.dddq_a, plans[0].timing.dddq_a)
