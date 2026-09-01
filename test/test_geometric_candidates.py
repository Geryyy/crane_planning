"""The deterministic geometric stage goes around a blocked direct corridor."""

import numpy as np
import pinocchio as pin
import pytest
from crane_model.collision import CollisionPrimitive
from crane_planning import planner as planning


def limits() -> planning.Limits:
    return planning.Limits(
        lower=np.full(5, -10.0),
        upper=np.full(5, 10.0),
        bounded=np.ones(5, dtype=bool),
        dq_max=np.ones(5),
        tau_max=np.ones(5),
        flow_max=1.0,
        tool_lower=0.2,
        tool_upper=0.7,
        tool_bounded=True,
    )


class FakeGeometry:
    """A Cartesian identity arm facing one low obstacle at the origin."""

    required = 0.2

    def __init__(self, blocked=True):
        self.blocked = blocked
        self.scene = [
            CollisionPrimitive(
                id="wall",
                shape="box",
                pose_in_mounting_base=pin.SE3.Identity(),
                dimensions_m=np.array([0.4, 0.4, 0.4]),
            )
        ]

    def tcp_pose(self, q_a):
        return np.asarray(q_a[:3], dtype=float), float(q_a[3])

    def is_valid(self, q_a):
        return not (
            self.blocked and abs(float(q_a[0])) < 0.2 and float(q_a[2]) < 0.45
        )

    def clearance(self, q_a):
        return 1.0 if self.is_valid(q_a) else 0.0

    def step_bound(self, first, second):
        return float(np.linalg.norm(np.asarray(second) - np.asarray(first)))


@pytest.fixture
def identity_ik(monkeypatch):
    def solve(_geometry, _limits, _config, position, yaw, _seed, restarts=1):
        assert restarts == 1
        return np.array([position[0], position[1], position[2], yaw, 0.0])

    monkeypatch.setattr(planning, "solve_ik", solve)


def config() -> planning.PlannerConfig:
    return planning.PlannerConfig(
        margin_interp=0.2,
        corridor_clearance=0.1,
        corridor_height_step=0.2,
        corridor_height_samples=2,
        corridor_lateral_step=0.3,
        corridor_lateral_samples=1,
    )


def test_clear_direct_line_is_selected_first(identity_ik):
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    waypoints, name = planning.plan_geometric_path(
        FakeGeometry(blocked=False), limits(), config(), start, [1.0, 0.0, 0.0], 0.0
    )
    assert name == "direct tool line"
    assert np.allclose(waypoints[0], start)
    assert np.allclose(waypoints[-1, :3], [1.0, 0.0, 0.0])


def test_blocked_direct_line_uses_overhead_corridor(identity_ik):
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    waypoints, name = planning.plan_geometric_path(
        FakeGeometry(), limits(), config(), start, [1.0, 0.0, 0.0], 0.0
    )
    assert name.startswith("lift/traverse/descend")
    assert np.max(waypoints[:, 2]) >= 0.5
    assert np.allclose(waypoints[-1, :3], [1.0, 0.0, 0.0])


def test_smoothing_failure_advances_to_the_next_candidate(monkeypatch):
    geometry = FakeGeometry(blocked=False)
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    checks = 0

    monkeypatch.setattr(
        planning,
        "lift_candidate",
        lambda *_args: np.array([start, [1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(planning, "fit", lambda *_args: (object(), None))

    def check_path(_path):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise planning.PlanningError("the smoothed curve cut the corner")

    geometry.check_path = check_path
    _path, _rate, _lifted, name = planning.plan_fitted_path(
        geometry, limits(), config(), start, [1.0, 0.0, 0.0], 0.0, np.zeros(5)
    )
    assert checks == 2
    assert name.startswith("lift/traverse/descend")


def test_out_of_range_tool_start_is_not_projected():
    planner = planning.Planner.__new__(planning.Planner)
    planner.limits = limits()
    q = np.zeros(8)
    q[planning.TOOL_INDEX] = 1.261858
    with pytest.raises(planning.PlanningError, match="instead of projecting"):
        planner._validate_start(planning.Start(q=q, dq_a=np.zeros(5)))
