"""What the emitted reference must satisfy: it stops where the plan stops, and
its effort field carries the command the OCP solved for.

The last point first.

`joint_trajectory_controller` rejects a `FollowJointTrajectory` goal whose last
point moves at all -- `fabs(v) > numeric_limits<float>::epsilon()`, 1.19e-7 --
and the `a2b_movement` answer is fed straight to it. The OCP pins `v = a = j = 0`
and `dq_u = 0` at the terminal node, but `actuated_samples` reconstructs that
sample from the interval before it, so the multiple-shooting gap (1e-6 rad/s at
`ocp_tolerance = 1e-4`) shows up as a moving last point. Observed in sim:
"Velocity of last trajectory point of joint theta1_slewing_joint is not zero:
0.000001194866194".
"""

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
    """A solve that stops at `T`, whose last interval misses it by `gap` rad/s."""
    time = np.linspace(0.0, duration, nodes + 1)
    # sigma(t) = 3 (t/T)^2 - 2 (t/T)^3: starts and ends at rest, ends at 1.
    tau = time / duration
    speed = (6.0 * tau - 6.0 * tau**2) / duration
    speed[-2] += gap  # the node the last sample is reconstructed from
    return SimpleNamespace(
        time=time,
        duration=duration,
        sigma=3.0 * tau**2 - 2.0 * tau**3,
        speed=speed,
        acceleration=(6.0 - 12.0 * tau) / duration**2,
        jerk=np.full(nodes + 1, -12.0 / duration**3),
        snap=np.zeros(nodes + 1),
        coefficients=COEFFICIENTS,
        # A ramp per axis, so a preview that is dropped or applied backwards
        # cannot pass: `u` at `t` and at `t + dead time` differ by a known slope.
        command=np.outer(time, [0.01, 0.02, 0.03, 0.04, 0.05]),
        q_u=np.zeros((nodes + 1, 2)),
        dq_u=np.zeros((nodes + 1, 2)),
        pump_flow=np.zeros(nodes + 1),
        iterations=3,
        solve_time_s=0.1,
        slack=0.0,
        terminal_sway=0.0,
        terminal_sway_rate=0.0,
    )


def resample(gap: float, duration: float = 4.0):
    """`_resample` alone: it reads `self.config` and the geometry's FK, nothing else."""
    pose = SimpleNamespace(position_m=np.zeros(3))
    geometry = SimpleNamespace(
        model=SimpleNamespace(forward_kinematics=lambda *_: pose)
    )
    planner = SimpleNamespace(config=PlannerConfig())
    return Planner._resample(
        planner,
        geometry,
        None,
        timing(gap, duration),
        SimpleNamespace(q_tool=0.3),
        1,
        "test",
    )


# `duration` is a solver output, so it lands anywhere against the 40 ms grid:
# just short of a multiple (the stub), just past one (what rounding to nearest
# extrapolated past the end of the plan), and exactly on one (measure-zero).
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
    """
    `jtc_fork.md` delta 3: effort is the *correction*, so what the plugin adds up
    is `dq_d + effort = u(t + n_d)`. The OCP already bounded that `u`; nothing
    downstream may derive it a second time.
    """
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

    assert commanded == pytest.approx(previewed)
    # The pendulum and the tool are not commanded, and the field is width-checked
    # against the joint names, so they carry a zero rather than nothing.
    passive_and_tool = [i for i in range(plan.effort.shape[1]) if i not in planned]
    assert np.all(plan.effort[:, passive_and_tool] == 0.0)
