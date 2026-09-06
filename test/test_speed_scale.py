"""`slow_down` divides rate, acceleration and jerk alike -- it is not a time scaling.

`CalcMovement.slow_down` is documented as a "divider to reduce max
speed/acceleration", and the reference server divides `qDotMax`, `qDDotMax` and
`qDDDotMax` by it each (`mp_crane_lib_helpers.hpp`, `apply_slow_down`). Scaling the
k-th derivative by `speed_scale**k` instead would be right for a uniform time
reparametrisation, but that is not a symmetry of this problem: the pendulum period
is fixed, so an `s**2` acceleration ceiling takes away the authority a solve needs
to cancel sway at the swing frequency, and it refuses where it should have slowed
down. Measured on the 21 bench moves at `s = 0.5`: 1 of 5 core moves answered under
`s**k`, 3 of 5 under the linear rule.

At `s = 1` the two rules coincide, so the ceilings are asserted at `s = 0.5`, and
against both rules -- an answer that stays inside the linear ceilings while
exceeding the old ones is what says which rule shipped.
"""

import numpy as np
import pinocchio as pin
from crane_model import ACTUATED_INDICES, Frame
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_INDICES, TOOL_INDEX, PlannerConfig
from crane_planning.geometry import yaw_of
from crane_planning.planner import Planner, Start
from test_jerk_bound import GOAL, START, TOOL_POSITION, description

#: Half speed. `stow` answers here; at 0.25 it refuses in the QP at the first
#: iteration, which is a start-guess failure and not a ceiling.
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
    peak_accel = np.max(np.abs(plan.timing.ddq_a), axis=0)
    peak_jerk = np.max(np.abs(plan.timing.dddq_a), axis=0)
    assert np.all(peak_accel <= reservation * config.ddq_a_max + 1.0e-6), (
        f"peak acceleration {peak_accel.tolist()} over {reservation} of "
        f"{config.ddq_a_max.tolist()}"
    )
    assert np.all(peak_jerk <= reservation * config.dddq_a_max + 1.0e-6), (
        f"peak jerk {peak_jerk.tolist()} over {reservation} of "
        f"{config.dddq_a_max.tolist()}"
    )

    # The discriminating half: under `s**k` this answer was not available.
    assert np.any(peak_accel > config.kappa * SPEED_SCALE**2 * config.ddq_a_max)
    assert np.any(peak_jerk > config.kappa * SPEED_SCALE**3 * config.dddq_a_max)
