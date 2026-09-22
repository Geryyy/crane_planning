"""JTC rejects nonzero last-point velocity (float eps 1.19e-7); shooting gap can leave it so."""

from types import SimpleNamespace

import numpy as np
import pytest
from crane_planning.config import (
    PLANNED_DOF,
    PLANNED_INDICES,
    PlannerConfig,
    PlanningError,
)
from crane_planning.ocp import ORDER, evaluate
from crane_planning.planner import TERMINAL_DRIFT_MAX, Planner

COEFFICIENTS = np.zeros((1, ORDER, PLANNED_DOF))
COEFFICIENTS[0, 0] = [0.0, 0.10, -0.20, 0.30, 0.05]
COEFFICIENTS[0, 1] = [1.20, -0.60, 0.90, 0.40, -0.30]


def timing(gap: float, duration: float, nodes: int = 8) -> SimpleNamespace:
    """Solve stopping at `T`, whose last interval misses it by `gap` rad/s."""
    time = np.linspace(0.0, duration, nodes + 1)
    # sigma(t) = 3 (t/T)^2 - 2 (t/T)^3: starts and ends at rest, ends at 1.
    tau = time / duration
    speed = (6.0 * tau - 6.0 * tau**2) / duration
    speed[-2] += gap  # reconstructs last sample
    return SimpleNamespace(
        time=time,
        duration=duration,
        sigma=3.0 * tau**2 - 2.0 * tau**3,
        speed=speed,
        acceleration=(6.0 - 12.0 * tau) / duration**2,
        jerk=np.full(nodes + 1, -12.0 / duration**3),
        snap=np.zeros(nodes + 1),
        coefficients=COEFFICIENTS,
        # ramp per axis: dropped/backwards preview fails, u differs by known slope
        command=np.outer(time, [0.01, 0.02, 0.03, 0.04, 0.05]),
        q_u=np.zeros((nodes + 1, 2)),
        dq_u=np.zeros((nodes + 1, 2)),
        pump_flow=np.zeros(nodes + 1),
        iterations=3,
        solve_time_s=0.1,
        report=dict,  # stand-in; _resample's drift refusal reads solve numbers off it
        # converged, so the message's "admitted at the cap" clause stays off
        stats={"acados_status": 0.0},
        residuals=np.zeros(4),
        slack=0.0,
        terminal_sway=0.0,
        terminal_sway_rate=0.0,
    )


def resample(gap: float, duration: float = 4.0, lag=None):
    """`_resample` alone: reads `self.config` and geometry's FK, nothing else."""
    pose = SimpleNamespace(position_m=np.zeros(3))
    geometry = SimpleNamespace(
        model=SimpleNamespace(forward_kinematics=lambda *_: pose)
    )
    config = PlannerConfig()
    if lag is not None:
        config.command_lag_s = np.asarray(lag, dtype=float)
    planner = SimpleNamespace(config=config)
    return Planner._resample(
        planner,
        geometry,
        None,
        timing(gap, duration),
        SimpleNamespace(q_tool=0.3),
        1,
        "test",
    )


# duration is a solver output, lands anywhere on the 40ms grid: short/past/on a multiple
@pytest.mark.parametrize("duration", [4.0, 4.001, 4.03])
def test_the_reference_ends_where_the_plan_does(duration):
    plan = resample(1.0e-6, duration)  # the gap that failed in sim

    assert plan.time[-1] == pytest.approx(duration)
    assert np.all(plan.dq[-1] == 0.0)
    assert plan.q[-1, list(PLANNED_INDICES)] == pytest.approx(
        evaluate(COEFFICIENTS, 1.0)[0]
    )


def test_a_solve_that_did_not_stop_is_refused():
    with pytest.raises(PlanningError, match="did not arrive stopped"):
        resample(10.0 * TERMINAL_DRIFT_MAX, 4.0)


def test_the_effort_field_carries_the_previewed_command():
    """Effort is correction: plugin sums dq_d + effort = u(t + n_d); OCP already bounded u."""
    config = PlannerConfig()
    plan = resample(1.0e-6, 4.03)
    solved = timing(1.0e-6, 4.03)

    planned = list(PLANNED_INDICES)
    commanded = plan.effort[:, planned] + plan.dq[:, planned]
    previewed = np.array(
        [
            np.interp(
                plan.time + config.command_dead_time_s,
                solved.time,
                solved.command[:, axis],
            )
            for axis in range(solved.command.shape[1])
        ]
    ).T

    # last sample pinned to zero (JTC holds it after plan ends); see test_the_held_last_point_*
    assert commanded[:-1] == pytest.approx(previewed[:-1])
    assert np.all(plan.effort[-1] == 0.0)
    # pendulum/tool not commanded but field is width-checked against joint names
    passive_and_tool = [i for i in range(plan.effort.shape[1]) if i not in planned]
    assert np.all(plan.effort[:, passive_and_tool] == 0.0)


def test_the_pt1_arm_adds_tau_v_du_dt_and_leaves_a_zero_lag_axis_alone():
    """Block 2 PT1 inversion (bench_track.py's third arm); ramp command makes du/dt exact."""
    lag = np.array([0.100, 0.025, 0.000, 0.075, 0.125])
    slope = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    planned = list(PLANNED_INDICES)
    without = resample(1.0e-6, 4.03).effort[:, planned]
    with_lag = resample(1.0e-6, 4.03, lag=lag).effort[:, planned]

    # law exact away from end; final OCP interval tapers to zero (JTC holds last point)
    # so assert constant only where taper does not reach, landing separately
    solved = timing(1.0e-6, 4.03)
    interior = resample(1.0e-6, 4.03).time + PlannerConfig().command_dead_time_s
    interior = interior < solved.time[-2]
    assert with_lag[interior] == pytest.approx(without[interior] + lag * slope)
    assert with_lag[-1] == pytest.approx(np.zeros(len(lag)))
    # ka's fitted lag is zero, so both arms bit-identical on arm axis (ax_arm)
    assert np.all(with_lag[:, 2] == without[:, 2])


def test_the_held_last_point_carries_no_command():
    """A held feedforward is a standing bias: sim before this pin held -0.0276 rad/s,
    left axis settled 0.355 rad off goal (predicted 0.394 rad, p:0.07 i:0, tau 28s)."""
    lag = np.array([0.100, 0.025, 0.000, 0.075, 0.125])

    assert np.all(resample(1.0e-6, 4.03).effort[-1] == 0.0)
    assert np.all(resample(1.0e-6, 4.03, lag=lag).effort[-1] == 0.0)


def test_the_reference_carries_accelerations():
    """Without them JTC interpolates a cubic while feedforward comes off the C4 curve."""
    plan = resample(1.0e-6, 4.03)

    planned = list(PLANNED_INDICES)
    interior = slice(1, -1)
    difference = np.gradient(plan.dq[:, planned], plan.time, axis=0)[interior]
    assert difference == pytest.approx(plan.ddq[interior][:, planned], abs=1e-3)

    not_commanded = [i for i in range(plan.ddq.shape[1]) if i not in planned]
    assert np.all(plan.ddq[:, not_commanded] == 0.0)
