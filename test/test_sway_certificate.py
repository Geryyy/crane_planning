"""A plan that bought the sway box with slack must be refused, not published.

`Geometry` builds the clearance envelope from `q_sway_max` and the geometric stage
certifies the corridor against it. In the OCP the same box is *soft*, priced at
`ocp_slack_price`, so on a request it cannot otherwise meet the solve converges,
every KKT residual clean, having paid its way past the bound -- and the plan it
returns swings outside the envelope the certificate covers. Nothing downstream
re-checks, so the breach is silent.

The gate is asserted on a doctored `Trajectory` rather than on a request that
provokes one, deliberately: on the shipped fixture every over-constrained request
tried -- the duration box cut to 4-6 s, a start already swinging at 0.95 of the
box with and without rate -- fails to *converge* instead of buying slack. That the
QP would rather die than pay here is a property of this conditioning, not of the
formulation, and the row stays soft and purchasable either way.

`slack_spent` gets its own test because it is where the breach hid: it read the
lower slacks only, so a swing past the upper bound reported zero.
"""

import dataclasses

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
from crane_planning.ocp import NH_E, NH_SWAY, SLACK_SPENT, slack_spent
from crane_planning.planner import Planner, Start
from test_jerk_bound import GOAL, START, TOOL_POSITION, description


class FakeSolver:
    """`get(node, 'sl'|'su')` and nothing else, which is all `slack_spent` reads."""

    def __init__(self, slacks: dict):
        self.slacks = slacks

    def get(self, node, field):
        return np.asarray(self.slacks.get((node, field), np.zeros(NH_SWAY + 1)))


def test_slack_on_the_upper_bound_is_not_invisible():
    """The row the tool overshoots is `su`, and it counted for nothing."""
    solver = FakeSolver({(3, "su"): np.array([0.0, 0.07, 0.0])})

    slack, sway = slack_spent(solver, N=10)

    assert slack == pytest.approx(0.07)
    assert sway == pytest.approx(0.07)


def test_the_settled_box_is_not_reported_as_an_envelope_breach():
    """Terminal sway rows are the settled box; `Planner.plan` owns that refusal."""
    solver = FakeSolver({(10, "sl"): np.array([0.4, 0.0, 0.0, 0.0])})

    slack, sway = slack_spent(solver, N=10)

    assert slack == pytest.approx(0.4)
    assert sway == 0.0


def planner_and_goal():
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
    return (
        planner,
        Start(q=configuration(START), dq_a=np.zeros(5)),
        pose.position_m,
        yaw,
    )


def test_a_bought_sway_box_is_refused_and_a_met_one_is_not():
    planner, start, position, yaw = planner_and_goal()
    request = dict(scene=[], avoid_collisions=True)

    # The clean solve first, so the gate is shown not to fire on the answer the
    # shipped config actually returns -- a refusal that fires on every plan is the
    # same defect wearing the opposite sign.
    plan = planner.plan(start, position, yaw, **request)
    assert plan.timing.sway_slack <= SLACK_SPENT

    # `slack_spent` slices `[:NH_SWAY]`, which is the sway pair only while `h` is
    # the *only* softened constraint: acados orders slacks `[sbu, sbx, sg, sh]`,
    # so softening any box -- `dq_sway_max` is the one a stubborn solve would
    # want -- prepends entries and the slice silently starts reading the box.
    # Asserted against the built solver rather than a fixture, because a fixture
    # would be updated by the same edit that breaks the gate.
    solver = planner.ocp.solver
    assert len(solver.get(0, "sl")) == NH_SWAY + 1, "a stage node is sway, sway, flow"
    assert len(solver.get(len(plan.timing.time) - 1, "sl")) == NH_E

    # The same answer, with the solve having paid 5% of `q_sway_max` to get it.
    bought = dataclasses.replace(plan.timing, slack=0.05, sway_slack=0.05)
    planner.ocp.solve = lambda **_: bought

    with pytest.raises(PlanningError) as refusal:
        planner.plan(start, position, yaw, **request)

    assert "sway" in str(refusal.value)
    assert refusal.value.stats["sway_slack"] == pytest.approx(0.05)


def test_a_settled_box_looser_than_the_envelope_is_refused():
    """Node N is skipped because its box is tighter -- so it has to stay tighter.

    Both are declared parameters, and relaxing `terminal_q_sway_max` is the
    natural move when a goal will not settle. Past `q_sway_max` it puts the goal
    pose outside the clearance envelope with neither check watching: `sway_slack`
    skips node N by construction and the terminal check now passes it.
    """
    planner, start, position, yaw = planner_and_goal()
    planner.ocp.config.terminal_q_sway_max = planner.config.q_sway_max * 1.5

    with pytest.raises(PlanningError, match="settled box is a tightening"):
        planner.plan(start, position, yaw, scene=[], avoid_collisions=True)
