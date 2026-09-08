"""`phi_tool_n` is `theta1 - theta8`, and `yaw_of` has to read that same angle."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from crane_model import GENERALIZED_DOF, PASSIVE_INDICES, CraneModel, Frame, Tool
from crane_planning.config import PLANNED_INDICES, TOOL_INDEX, passive_equilibrium
from crane_planning.geometry import wrap, yaw_of

Q_TOOL = 0.35

#: `q_a = [q1 slew, q2 boom, q3 arm, q4 telescope, q8 rotator]`, spread so a
#: formula that ignores either of the two angles that make `phi_tool` fails.
CONFIGURATIONS = [
    (0.0, 0.2, -0.5, 0.1, 0.0),
    (0.3, 0.2, -0.5, 0.1, 0.0),
    (0.0, 0.2, -0.5, 0.1, 0.4),
    (0.3, 0.2, -0.5, 0.1, 0.4),
    (-0.2, 0.1, -0.8, 0.6, 0.7),
]


def model() -> CraneModel:
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    return CraneModel(path.read_text(), Tool.PZS100)


@pytest.mark.parametrize("q_a", CONFIGURATIONS)
def test_yaw_of_reads_the_legacy_tool_angle(q_a):
    """
    The retained `a2b_movement` contract, pinned against the machine.

    `a2b_ilqr_server` closed the goal yaw as `theta8 = theta1 - phiTool`, so
    `phi_tool = theta1 - theta8` exactly. It is the heading of the jaw-opening
    direction, TCP local y. This is the only test that would catch the planner
    answering the perpendicular axis, which is 90 deg off and still a
    plausible-looking plan.
    """
    crane = model()
    q_a = np.asarray(q_a, dtype=float)
    q = np.zeros(GENERALIZED_DOF)
    q[list(PLANNED_INDICES)] = q_a
    q[TOOL_INDEX] = Q_TOOL
    q[list(PASSIVE_INDICES)] = passive_equilibrium(q_a)

    pose = crane.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    rotation = pin.XYZQUATToSE3(
        np.concatenate([pose.position_m, pose.orientation_xyzw])
    ).rotation
    assert abs(wrap(yaw_of(rotation) - (q_a[0] - q_a[4]))) < 1.0e-9
