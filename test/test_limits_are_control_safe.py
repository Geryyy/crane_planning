"""Planner may not certify what the MPC refuses -- both read
crane_model/config/control_safe_limits.yaml now, this pins them in sync."""

import os

import pytest
from ament_index_python.packages import get_package_share_directory
from crane_model.conventions import (
    CONTROL_SAFE_AXES,
    canonical_joints,
    control_safe_limits,
)
from crane_planning import weights as crane_weights
from crane_planning.config import (
    PLANNED_DOF,
    PLANNED_INDICES,
    PlannerConfig,
    read_limits,
)
from crane_planning.ocp import H_RATE, baked_parameters, build_ocp

DESCRIPTION = os.path.join(
    get_package_share_directory("crane_model"), "description", "pzs100.urdf"
)


def limits():
    config = PlannerConfig()
    with open(DESCRIPTION, encoding="utf-8") as handle:
        return read_limits(
            handle.read(), config.pump_flow_max, config.pump_flow_planning_factor
        )


def test_the_rate_rows_ceiling_is_the_control_safe_one():
    """
    The rate row is bounded at `kappa`, so its ceiling is that times
    the divisor the solver was built with. Read off the built solver's `scale` --
    the same array `row_excess` mirrors and the exporter writes into the header --
    not off `read_limits`: the two used to disagree, the solver dividing by the
    description instead, which put the boom's ceiling 1.76x over the box
    `crane_mpc` enforces and made this planner's certificate false.
    """
    config = PlannerConfig()
    read = limits()
    with open(DESCRIPTION, encoding="utf-8") as handle:
        _ocp, scale, _model = build_ocp(
            handle.read(),
            {**baked_parameters(config, read), "weights": crane_weights.DEFAULTS},
            {"pump_flow_max": config.pump_flow_max},
        )

    ceiling = config.kappa * scale[H_RATE : H_RATE + PLANNED_DOF]
    assert ceiling == pytest.approx(config.kappa * read.dq_max)
    # discriminating: the boom's description row is 0.5257, which is what it read before
    assert scale[H_RATE + 1] == pytest.approx(0.239067)


def test_no_planned_axis_leaves_constraint_ones_box():
    """
    231: `>=` is not enough -- a plan allowed *onto* the MPC's bound leaves its first
    interval no interior, because `position_box` then pins the hard box to the
    measurement. Every row must keep a whole `q_a_margin` of clearance to it.

    This is `read_limits`' box, not `plan`'s: `relaxed_to` hands the retreat back on
    an axis the machine is standing inside it, and the next two tests are that half.
    """
    read = limits()
    box = control_safe_limits()
    axis_of = {joint: axis for axis, joint in CONTROL_SAFE_AXES.items()}
    joints = canonical_joints()
    for axis, index in enumerate(PLANNED_INDICES):
        name = axis_of[joints[index]]
        assert read.dq_max[axis] <= box["dq_a_max"][name] + 1e-12, name
        if not read.bounded[axis]:
            continue
        margin = box["q_a_margin"][name]
        assert read.lower[axis] >= box["q_a_lower"][name] + 2.0 * margin - 1e-12, name
        assert read.upper[axis] <= box["q_a_upper"][name] - 2.0 * margin + 1e-12, name


def test_the_intersection_bites_where_the_description_is_wider():
    read = limits()
    # boom: -1.2 admits 1.22 rad four-bar can't reach; arm: 4.6 past dead point 1.8466647
    assert read.lower[1] == 0.02 + 0.020
    assert read.upper[2] == 1.324864 - 2.0 * 0.021
    assert read.dq_max[1] == 0.239067
    # the row itself, margins dropped: how far relaxed_to gives way
    assert (read.control_safe_lower[1], read.control_safe_upper[2]) == (0.0, 1.324864)


def test_the_tool_keeps_the_descriptions_travel():
    # box's tool row is an angle on a retired jaw gripper, not this rail
    read = limits()
    assert (read.tool_lower, read.tool_upper) == (0.0, 0.538)


def test_the_box_gives_way_to_a_machine_parked_inside_the_margins():
    # sim parks the boom and the telescope on their own stops, i.e. inside the retreat
    # this planner takes, and creeps a fraction of a mrad past them
    # (test_start_limit_deadband); crane_mpc's position_box relaxes the same way, and
    # refusing here would refuse to plan at all
    read = limits()
    parked = read.lower.copy()
    parked[1] = -1.44e-4
    parked[3] = 0.0
    relaxed = read.relaxed_to(parked)

    assert (relaxed.lower[1], relaxed.lower[3]) == (-1.44e-4, 0.0)
    assert relaxed.upper[1] == read.upper[1]
    assert list(relaxed.lower[[0, 2, 4]]) == list(read.lower[[0, 2, 4]])
    # and the description still says where the stop actually is
    assert read.description_lower[1] == -1.2
