"""The pose the tool hangs at: closed form without a payload, solved with one."""

from itertools import product

import numpy as np
import pytest
from crane_model import (
    ACTUATED_INDICES,
    PASSIVE_INDICES,
    CraneModel,
    CraneModelError,
    Frame,
    Payload,
    Tool,
)
from crane_planning import weights as crane_weights
from crane_planning.config import (
    PlannerConfig,
    passive_equilibrium,
    read_limits,
    settled_offset,
)
from crane_planning.planner import Planner, Start
from test_jerk_bound import description

Q_TOOL = 0.22

#: Crooked grip on a block resting tilted, measured 2026-10-05 on a Gazebo wall
#: build: 800 kg, centre 6.4 cm back and 4.0 cm across the tool axis. The planner
#: refused every move from it as "did not arrive settled".
CROOKED = np.array([-0.064, -0.040, 1.04])


def machine():
    xml = description()
    config = PlannerConfig()
    return CraneModel(xml, Tool.PZS100), read_limits(
        xml, config.pump_flow_max, config.pump_flow_planning_factor
    )


def poses(limits) -> list:
    """Planned five across the box, extreme telescope and rotator included."""
    rows = [np.array([-3.0, 1.2])]
    for axis in (1, 2, 3):
        rows.append(np.linspace(limits.lower[axis], limits.upper[axis], 3))
    rows.append(np.array([-3.1, 0.0, 2.6]))
    return [np.array(q_a) for q_a in product(*rows)]


def payload(centre) -> Payload:
    return Payload(
        mass_kg=800.0,
        center_of_mass_k8_m=np.asarray(centre, dtype=float),
        inertia_k8_kg_m2=np.zeros((3, 3)),
        valid=True,
    )


def test_the_closed_form_is_the_numeric_solve_without_a_payload():
    """The cheap form every consumer reads is the model's own answer, not an
    approximation of it -- so a difference from it is the payload and nothing else."""
    model, limits = machine()
    unanswered = set()
    for q_a in poses(limits):
        try:
            settled = model.passive_equilibrium(np.append(q_a, Q_TOOL))
        except CraneModelError:
            unanswered.add((q_a[1], q_a[2]))
            continue
        assert settled == pytest.approx(passive_equilibrium(q_a), abs=1e-9)
    # The solve gives up at one corner only -- boom and arm both at their floor, where
    # the closed form puts the tip 0.87 rad past its own stop and there is nothing
    # inside the passive range to converge to. That is where `settled_offset` falls back.
    assert unanswered == {(limits.lower[1], limits.lower[2])}


def test_an_off_axis_grip_moves_the_settled_pose_past_the_settled_box():
    """0.03 rad against a 0.02 box: the bias is the size of the thing it is measured
    against, which is why the terminal gate refused a plan that had arrived."""
    model, limits = machine()
    offset = settled_offset(model, payload(CROOKED), Q_TOOL)
    worst = max(float(np.max(np.abs(offset(q_a)))) for q_a in poses(limits))
    assert worst > float(np.min(PlannerConfig().terminal_q_sway_max))


def test_a_centre_on_the_tool_axis_leaves_the_closed_form_alone():
    """Mass alone does not move the equilibrium; only a lever across the axis does."""
    model, limits = machine()
    for carried in (None, payload([0.0, 0.0, 1.04])):
        offset = settled_offset(model, carried, Q_TOOL)
        for q_a in poses(limits):
            assert offset(q_a) == pytest.approx(np.zeros(2), abs=1e-9)


def test_a_crooked_grip_plans_and_arrives_where_the_block_hangs():
    """
    Boom and telescope, machine settled, block gripped 6.4 cm off centre.

    Refused before this: terminal sway [0.020, 0.022] rad against the [0.02, 0.02]
    box, every row of it the payload's own shift. The assertion is not the gate's
    own number but the model's: the tool must arrive where it will still be once
    the operator lets go.
    """
    config = PlannerConfig()
    planner = Planner(description(), config, dict(crane_weights.DEFAULTS))
    carried = payload(CROOKED)
    q = np.array([0.4345, 1.1108, -0.2041, 0.5072, 0.0, 0.0, 2.0, Q_TOOL])
    q[list(PASSIVE_INDICES)] = planner.model.passive_equilibrium(
        q[list(ACTUATED_INDICES)], carried
    )
    start = Start(q=q, dq_a=np.zeros(5))
    goal = planner._settled(start.q_a + np.array([0.0, -0.3, 0.25, 0.6, 0.0]), Q_TOOL)
    tcp = planner.model.forward_kinematics(goal, Frame.MOUNTING_BASE, Frame.TCP)

    timing = planner.plan(
        start,
        tcp.position_m,
        planner._tcp_yaw(goal),
        payload=carried,
        avoid_collisions=False,
    ).timing

    hangs = planner.model.passive_equilibrium(
        np.append(timing.q_a[-1], Q_TOOL), carried
    )
    assert np.abs(timing.q_u[-1] - hangs) == pytest.approx(
        np.zeros(2), abs=float(np.min(config.terminal_q_sway_max))
    )
