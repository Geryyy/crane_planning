"""A continuous axis comes back on the nearest turn inside its box, not the long way round."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from crane_model import CraneModel, Frame, Tool
from crane_planning.config import PlannerConfig, PlanningError, read_limits
from crane_planning.geometry import Geometry, solve_ik, wrap, yaw_of

ROTATOR_AXIS = 4
Q_TOOL = 0.30

#: `(seed rotator, goal yaw, TCP position)`. Every row is a goal the full-turn
#: search window used to answer past pi off the seed (-3.55, -3.50, -3.48 rad),
#: which is the surplus turn issue 213's 4.761 rad goal accumulated.
#: The last two are the machine's own convention: it reports the rotator
#: unwrapped, and a lab day started a run at 2pi + 0.25 (issue 293).
CASES = [
    (1.9, np.radians(115.0), (2.80, 1.05, 1.6)),
    (-1.9, np.radians(-30.0), (2.80, 1.05, 1.6)),
    (0.0, np.radians(-140.0), (2.00, 0.75, 1.6)),
    (2.0 * np.pi + 0.25, np.radians(115.0), (2.80, 1.05, 1.6)),
    (-2.0 * np.pi - 0.25, np.radians(115.0), (2.80, 1.05, 1.6)),
]


def stage() -> tuple[Geometry, object, PlannerConfig]:
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    xml = path.read_text()
    config = PlannerConfig()
    limits = read_limits(xml, config.pump_flow_max, config.pump_flow_planning_factor)
    model = CraneModel(xml, Tool.PZS100)
    geometry = Geometry(
        model, limits, [], config, Q_TOOL, np.zeros(4), None, np.zeros(2)
    )
    return geometry, limits, config


@pytest.mark.parametrize(("seed_rotator", "goal_yaw", "position_m"), CASES)
def test_the_goal_solve_stays_within_half_a_turn(seed_rotator, goal_yaw, position_m):
    """
    The rotator's turn count walks otherwise: each move seeds on the last one's
    answer, so a surplus turn is never given back and the MPC ends up tracking a
    reference a full turn off the pose every other run lifts the same block at.
    """
    geometry, limits, config = stage()
    seed = np.array([0.785, 0.5236, 0.5236, 0.25, seed_rotator])
    position_m = np.asarray(position_m, dtype=float)

    answer = solve_ik(
        geometry,
        limits,
        config,
        position_m,
        goal_yaw,
        seed,
        restarts=config.ik_restarts,
    )

    # Still the pose that was asked for -- unwinding is exact, not a compromise.
    pose = geometry.model.forward_kinematics(
        geometry.configuration(answer), Frame.MOUNTING_BASE, Frame.TCP
    )
    rotation = pin.XYZQUATToSE3(
        np.concatenate([pose.position_m, pose.orientation_xyzw])
    ).rotation
    assert np.linalg.norm(pose.position_m - position_m) <= config.eps_pos
    assert abs(wrap(yaw_of(rotation) - goal_yaw)) <= config.eps_yaw
    assert abs(answer[ROTATOR_AXIS] - seed_rotator) <= np.pi


@pytest.mark.parametrize("turns", [1, -1])
def test_a_whole_turn_on_the_start_moves_the_goal_by_exactly_that_turn(turns):
    """
    The machine reports the rotator unwrapped (lab 2026-11-04, issue 293).

    Fold the start to [-pi, pi] instead and the goal is a whole turn off the
    pose the crane is in, so the move undoes the turn. Gazebo, issue 293: the
    same move planned +4.4466 from a 2pi + 0.25 start and -1.8366 from a 0.0
    one -- one angle, one turn apart.
    """
    geometry, limits, config = stage()
    position_m, goal_yaw = np.array([2.80, 1.05, 1.6]), np.radians(115.0)
    seed = np.array([0.785, 0.5236, 0.5236, 0.25, 0.25])

    def solve(rotator: float) -> np.ndarray:
        start = seed.copy()
        start[ROTATOR_AXIS] = rotator
        return solve_ik(
            geometry,
            limits,
            config,
            position_m,
            goal_yaw,
            start,
            restarts=config.ik_restarts,
        )

    here = solve(seed[ROTATOR_AXIS])
    turned = solve(seed[ROTATOR_AXIS] + turns * 2.0 * np.pi)
    # The rotator enters the model as cos/sin, so the shift is exact, not within
    # eps_yaw: measured 9e-16 here, and 1e-9 still catches a turn-dependent bias.
    assert turned[ROTATOR_AXIS] == pytest.approx(
        here[ROTATOR_AXIS] + turns * 2.0 * np.pi, abs=1e-9
    )


def tcp(geometry: Geometry, q_a: np.ndarray) -> tuple[np.ndarray, float]:
    """Where the tool is and which way it points, at a planned five."""
    pose = geometry.model.forward_kinematics(
        geometry.configuration(q_a), Frame.MOUNTING_BASE, Frame.TCP
    )
    rotation = pin.XYZQUATToSE3(
        np.concatenate([pose.position_m, pose.orientation_xyzw])
    ).rotation
    return pose.position_m, yaw_of(rotation)


def test_a_goal_one_turn_past_four_pi_takes_the_other_turn():
    """
    297: the nearest turn is not always inside the box. Before this, only
    `crane_mpc`'s hard box stopped the count, and the PID fallback skips it.
    """
    geometry, limits, config = stage()
    seed = np.array([0.785, 0.5236, 0.5236, 0.25, 4.0 * np.pi - 0.3])
    # the pose one radian further round than the seed, asked for as a pose: its exact
    # answer is the rotator at 4pi + 0.7, the turn the fold used to take. Read off the
    # wound configuration rather than built from `yaw + 1`, which turns the other way
    wound = seed.copy()
    wound[ROTATOR_AXIS] = wound[ROTATOR_AXIS] + 1.0
    position_m, goal_yaw = tcp(geometry, wound)

    answer = solve_ik(
        geometry,
        limits,
        config,
        position_m,
        goal_yaw,
        seed,
        restarts=config.ik_restarts,
    )

    reached, reached_yaw = tcp(geometry, answer)
    assert np.linalg.norm(reached - position_m) <= config.eps_pos
    assert abs(wrap(reached_yaw - goal_yaw)) <= config.eps_yaw
    assert answer[ROTATOR_AXIS] <= limits.control_safe_upper[ROTATOR_AXIS]
    # one turn back, not folded onto the seed: eps_yaw is what the solve converges to
    assert answer[ROTATOR_AXIS] == pytest.approx(2.0 * np.pi + 0.7, abs=config.eps_yaw)


def test_a_rotator_already_wound_past_the_box_is_refused():
    """
    Both turns outside: nothing this package can plan, and `relaxed_to` exempts it.

    Needs more than *half a turn* past the row, not merely past it -- the fold is
    within pi of the seed, so from 4pi..5pi the turn one back still lands inside and
    the plan unwinds instead of refusing. 6pi + 1 is 2.5 turns out.
    """
    geometry, limits, config = stage()
    seed = np.array([0.785, 0.5236, 0.5236, 0.25, 6.0 * np.pi + 1.0])
    position_m, yaw_here = tcp(geometry, seed)

    with pytest.raises(PlanningError, match="wound more than half a turn"):
        solve_ik(
            geometry,
            limits,
            config,
            position_m,
            yaw_here,
            seed,
            restarts=config.ik_restarts,
        )
