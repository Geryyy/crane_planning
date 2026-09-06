"""The deterministic geometric stage goes around a blocked direct corridor."""

import numpy as np
import pinocchio as pin
import pytest
from crane_model.collision import CollisionPrimitive
from crane_planning import geometry as geometry_stage
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
        return not (self.blocked and abs(float(q_a[0])) < 0.2 and float(q_a[2]) < 0.45)

    def clearance(self, q_a):
        return 1.0 if self.is_valid(q_a) else 0.0

    def step_bound(self, first, second):
        return float(np.linalg.norm(np.asarray(second) - np.asarray(first)))

    def check_path(self, path):
        """The real certificate's contract: the fitted curve, not the polyline."""
        for sigma in np.linspace(0.0, 1.0, 201):
            if not self.is_valid(path.position(sigma)):
                raise planning.PlanningError(f"blocked at sigma = {sigma:.3f}")


@pytest.fixture
def identity_ik(monkeypatch):
    def solve(_geometry, _limits, _config, position, yaw, _seed, restarts=1):
        assert restarts == 1
        return np.array([position[0], position[1], position[2], yaw, 0.0])

    monkeypatch.setattr(geometry_stage, "solve_ik", solve)


def config() -> planning.PlannerConfig:
    return planning.PlannerConfig(
        margin_interp=0.2,
        corridor_clearance=0.1,
        corridor_height_step=0.2,
        corridor_height_samples=2,
        corridor_lateral_step=0.3,
        corridor_lateral_samples=1,
    )


def certify(geometry, start, goal):
    goal_q_a = np.array([goal[0], goal[1], goal[2], 0.0, 0.0])
    path, _lifted, name = planning.plan_fitted_path(
        geometry, limits(), config(), start, goal, 0.0, np.zeros(5), goal_q_a
    )
    return path, name


def test_joint_line_answers_when_the_tool_chord_does_not(identity_ik):
    """A blocked tool chord with a clear joint line is answered without a corridor."""
    geometry = FakeGeometry()
    # the identity arm's joint line is the tool chord, so give the goal a
    # configuration whose line passes above the block
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    goal_q_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    geometry.is_valid = lambda q_a: (
        not (abs(float(q_a[0])) < 0.2 and float(q_a[4]) < 0.1)
    )
    goal_q_a[4] = 1.0
    path, _lifted, name = planning.plan_fitted_path(
        geometry, limits(), config(), start, [1.0, 0.0, 0.0], 0.0, np.zeros(5), goal_q_a
    )
    assert name == "joint line"
    assert np.allclose(path.position(1.0), goal_q_a)


def test_unreachable_corner_refuses_its_corridors_without_lifting(monkeypatch):
    """One IK per corner, not a lift per corridor, when the transfer plane is out of reach."""
    lifts, solves = [], []

    def solve(_geometry, _limits, _config, position, yaw, _seed, restarts=1):
        solves.append(np.asarray(position, dtype=float))
        if position[2] > 0.3:
            raise planning.PlanningError("the tool cannot be placed there")
        return np.array([position[0], position[1], position[2], yaw, 0.0])

    real_lift = geometry_stage.lift_candidate

    def lift(*args):
        lifts.append(args[-1].name)
        return real_lift(*args)

    monkeypatch.setattr(geometry_stage, "solve_ik", solve)
    monkeypatch.setattr(geometry_stage, "lift_candidate", lift)
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(planning.PlanningError, match="cannot be placed"):
        planning.plan_fitted_path(
            geometry_stage and FakeGeometry(),
            limits(),
            config(),
            start,
            [1.0, 0.0, 0.0],
            0.0,
            np.zeros(5),
        )
    assert lifts == ["direct tool line"]
    # two planes, two corners each; the second corner of a plane is never asked
    # for once the first refused
    assert sum(point[2] > 0.3 for point in solves) == 2


def test_clear_direct_line_is_selected_first(identity_ik):
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    path, name = certify(FakeGeometry(blocked=False), start, [1.0, 0.0, 0.0])
    assert name == "direct tool line"
    assert np.allclose(path.position(0.0), start)
    assert np.allclose(path.position(1.0)[:3], [1.0, 0.0, 0.0])


def test_blocked_direct_line_uses_overhead_corridor(identity_ik):
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    path, name = certify(FakeGeometry(), start, [1.0, 0.0, 0.0])
    assert name.startswith("lift/traverse/descend")
    sampled = np.array([path.position(sigma) for sigma in np.linspace(0.0, 1.0, 201)])
    assert np.max(sampled[:, 2]) >= 0.45
    assert np.allclose(path.position(1.0)[:3], [1.0, 0.0, 0.0])


def test_smoothing_failure_advances_to_the_next_candidate(monkeypatch, identity_ik):
    geometry = FakeGeometry(blocked=False)
    start = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    checks = 0

    monkeypatch.setattr(
        geometry_stage,
        "lift_candidate",
        lambda *_args: np.array([start, [1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(geometry_stage, "fit", lambda *_args: object())

    def check_path(_path):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise planning.PlanningError("the smoothed curve cut the corner")

    geometry.check_path = check_path
    _path, _lifted, name = planning.plan_fitted_path(
        geometry, limits(), config(), start, [1.0, 0.0, 0.0], 0.0, np.zeros(5), start
    )
    assert checks == 2
    assert name == "joint line"


def test_out_of_range_tool_start_is_not_projected():
    planner = planning.Planner.__new__(planning.Planner)
    planner.limits = limits()
    q = np.zeros(8)
    q[planning.TOOL_INDEX] = 1.261858
    with pytest.raises(planning.PlanningError, match="instead of projecting"):
        planner._validate_start(planning.Start(q=q, dq_a=np.zeros(5)))
