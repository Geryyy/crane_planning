"""The stamp rule: zero in the answer, publish time on the pair. A planning-start stamp sits
seconds in the past and the JTC skips that far into the plan -- 4 s of a 7.4 s move."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from crane_msgs.msg import JointPath  # noqa: F401
except ImportError:  # crane_msgs not (re)built with JointPath yet
    pytest.skip("crane_msgs/JointPath is not built", allow_module_level=True)

from builtin_interfaces.msg import Time
from crane_model import GENERALIZED_DOF, canonical_joints
from crane_planning.config import PLANNED_DOF
from crane_planning.node import CranePlanner
from crane_planning.ocp import ORDER

from timber_crane_planning_interfaces.srv import CalcMovement

POINTS = 3
COEFFICIENTS = np.zeros((1, ORDER, PLANNED_DOF))
COEFFICIENTS[0, 1] = [1.20, -0.60, 0.90, 0.40, -0.30]

PARAMETERS = {"command_lag_s": [0.0] * PLANNED_DOF, "c3_feedforward": True}

PUBLISHERS = (
    "reference",
    "joint_path",
    "planned_path",
    "markers",
    "solver_stats",
    "legacy_tcp_path",
    "plan_status",  # _run publishes it on success, _a2b on the refusals
)


class _Capture:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Clock:
    """A later stamp on every read, so a stamp taken before the solve is visible."""

    def __init__(self):
        self.reads = 0

    def now(self):
        self.reads += 1
        return SimpleNamespace(to_msg=lambda sec=self.reads: Time(sec=sec))


def plan() -> SimpleNamespace:
    rows = np.zeros((POINTS, GENERALIZED_DOF))
    return SimpleNamespace(
        time=np.linspace(0.0, 2.0, POINTS),
        q=rows,
        dq=rows,
        ddq=rows,
        effort=rows,
        tcp=np.zeros((POINTS, 3)),
        message="planned",
        timing=SimpleNamespace(
            coefficients=COEFFICIENTS,
            time=np.linspace(0.0, 2.0, POINTS),
            s=np.linspace(0.0, 1.0, POINTS),
            slack=0.0,
            stats={"acados_status": 0.0},
            report=dict,
        ),
    )


def node(**extra) -> CranePlanner:
    """Real node, stubbed collaborators: `_trajectory`/`_joint_path` are the code under test."""
    standing = CranePlanner.__new__(CranePlanner)
    clock = _Clock()
    standing.clock = clock
    # Clock reading the solve started at, so "stamped after the solve" is checkable.
    standing.solved = []
    standing.planner = SimpleNamespace(
        config=SimpleNamespace(command_lag_s=np.zeros(PLANNED_DOF)),
        model=SimpleNamespace(
            forward_kinematics=lambda *_: SimpleNamespace(position_m=np.zeros(3))
        ),
        prepare_scene=lambda *_: [],
        plan=lambda *_, **__: standing.solved.append(clock.reads) or plan(),
        tip_to_tcp_offset=lambda *_: np.zeros(3),
    )
    standing.joint_names = list(canonical_joints())
    standing.get_clock = lambda: clock
    standing.get_name = lambda: "crane_planner"
    standing.get_parameter = lambda name: SimpleNamespace(value=PARAMETERS[name])
    standing.get_logger = lambda: SimpleNamespace(info=print, warn=print)
    standing._primitives = lambda *_: []
    for name in PUBLISHERS:
        setattr(standing, name, _Capture())
    for name, value in extra.items():
        setattr(standing, name, value)
    return standing


def test_the_published_plan_is_stamped_at_publish_and_both_forms_alike():
    standing = node()

    CranePlanner._run(
        standing, SimpleNamespace(q_tool=0.3), np.zeros(3), 0.0, None, None, False, 1.0
    )

    (reference,) = standing.reference.messages
    (path,) = standing.joint_path.messages
    # One plan in two forms: `Cycle.resolved_path` pairs them by exact stamp equality.
    assert reference.header.stamp == path.header.stamp
    # Read after the solve returned, not before it: that is the whole fix.
    (solved,) = standing.solved
    assert reference.header.stamp.sec > solved


def test_the_answer_is_stamped_start_now():
    """Zero is the JTC's "start at the first point"; `pid_move.py` forwards the answer as is."""
    standing = node(
        _run=lambda *_: (plan(), None, SimpleNamespace(poses=[])),
        _start=lambda: SimpleNamespace(q_tool=0.3),
    )
    request = CalcMovement.Request()
    request.slow_down = 1.0

    response = CranePlanner._a2b(standing, request, CalcMovement.Response())

    assert response.success
    # Canonical eight: `trajectory_controller_a2b` tracks the passive pair as well.
    assert len(response.trajectory.joint_names) == GENERALIZED_DOF
    assert len(response.trajectory.points) == POINTS
    stamp = response.trajectory.header.stamp
    assert (stamp.sec, stamp.nanosec) == (0, 0)
