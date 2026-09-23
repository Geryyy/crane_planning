"""`/crane/joint_path` is the planner's own curve: wrong sampling or a transposed reshape."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from crane_msgs.msg import JointPath  # noqa: F401
except ImportError:  # crane_msgs not (re)built with JointPath yet
    pytest.skip("crane_msgs/JointPath is not built", allow_module_level=True)

from builtin_interfaces.msg import Time
from crane_model import canonical_joints
from crane_planning.config import PLANNED_DOF, PLANNED_INDICES
from crane_planning.node import PATH_SAMPLES, CranePlanner
from crane_planning.ocp import ORDER, evaluate

# curved, so a transposed reshape cannot pass the endpoint rows by accident
COEFFICIENTS = np.zeros((1, ORDER, PLANNED_DOF))
COEFFICIENTS[0, 0] = [0.0, 0.10, -0.20, 0.30, 0.05]
COEFFICIENTS[0, 1] = [1.20, -0.60, 0.90, 0.40, -0.30]
COEFFICIENTS[0, 2] = [-0.40, 0.70, 0.30, -0.50, 0.20]


def test_the_path_carries_the_curve_endpoints_and_the_plans_own_duration():
    plan = SimpleNamespace(
        timing=SimpleNamespace(
            coefficients=COEFFICIENTS, time=np.linspace(0.0, 4.03, 9)
        )
    )
    node = SimpleNamespace(joint_names=list(canonical_joints()))

    message = CranePlanner._joint_path(node, plan, Time(sec=7, nanosec=250000000))

    assert message.joint_names == [node.joint_names[i] for i in PLANNED_INDICES]
    assert len(message.q_path) == PATH_SAMPLES * PLANNED_DOF
    q_path = np.asarray(message.q_path).reshape(PATH_SAMPLES, PLANNED_DOF)
    assert q_path[0] == pytest.approx(evaluate(COEFFICIENTS, 0.0)[0])
    assert q_path[-1] == pytest.approx(evaluate(COEFFICIENTS, 1.0)[0])
    assert message.duration.sec + message.duration.nanosec * 1e-9 == pytest.approx(4.03)
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (7, 250000000)
