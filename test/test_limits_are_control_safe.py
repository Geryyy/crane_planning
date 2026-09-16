"""
The planner may not certify what the MPC refuses.

`read_limits` used to read the description alone, so it admitted the boom below
0.02 rad and the arm above 1.3039 -- 3.3 rad of arm travel constraint 1
hard-refuses -- and velocity rows up to 2.7x the MPC's own bound. Both packages
now read `crane_model/config/control_safe_limits.yaml` and this is what keeps
the planner's side of that inside the MPC's.
"""

import os

from ament_index_python.packages import get_package_share_directory
from crane_model.conventions import (
    CONTROL_SAFE_AXES,
    canonical_joints,
    control_safe_limits,
)
from crane_planning.config import PLANNED_INDICES, PlannerConfig, read_limits

DESCRIPTION = os.path.join(
    get_package_share_directory("crane_model"), "description", "pzs100.urdf"
)


def limits():
    config = PlannerConfig()
    with open(DESCRIPTION, encoding="utf-8") as handle:
        return read_limits(
            handle.read(), config.pump_flow_max, config.pump_flow_planning_factor
        )


def test_no_planned_axis_leaves_constraint_ones_box():
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
        assert read.lower[axis] >= box["q_a_lower"][name] + margin - 1e-12, name
        assert read.upper[axis] <= box["q_a_upper"][name] - margin + 1e-12, name


def test_the_intersection_bites_where_the_description_is_wider():
    read = limits()
    # boom: the description's -1.2 admits 1.22 rad the four-bar cannot reach;
    # arm: its 4.6 is past the transmission dead point at 1.8466647.
    assert read.lower[1] == 0.02
    assert read.upper[2] == 1.324864 - 0.021
    assert read.dq_max[1] == 0.239067


def test_the_tool_keeps_the_descriptions_travel():
    # The box's tool row is an angle on a retired jaw gripper, not this rail.
    read = limits()
    assert (read.tool_lower, read.tool_upper) == (0.0, 0.538)


def test_the_box_gives_way_to_where_the_machine_is():
    # The machine parks with the boom folded below the control-safe 0.02, which
    # is a normal thing to plan out of. crane_mpc's position_box relaxes the
    # same way; refusing here would refuse to plan at all.
    read = limits()
    folded = read.lower.copy()
    folded[1] = -0.2
    relaxed = read.relaxed_to(folded)

    assert relaxed.lower[1] == -0.2
    assert relaxed.upper[1] == read.upper[1]
    assert list(relaxed.lower[2:]) == list(read.lower[2:])
    # and the description still says where the stop actually is
    assert read.description_lower[1] == -1.2
