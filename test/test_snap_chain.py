"""`sigma` is integrated four times, so the reference is `C4` in time.

With the acceleration as the input, acados held it constant across an interval and
`ddq_a` jumped at every node: `q_a(t)` was `C1` however smooth the curve was. With
the snap as the input, `sigma` is quartic on each interval and `sigma`, `v`, `a`,
`j` are all continuous across the join, so `q_a(t) = c(sigma(t))` is `C4` -- the
derivative count `wiki/controller_design.md` section 2.4 asks for.

Two things can break silently. The row order is written once in `X_*` and read back
positionally everywhere, so a reorder that misses one reader gives plausible wrong
numbers rather than an error. And the resampler reconstructs `sigma` from the chain
in closed form; if it keeps a lower-order formula the samples stop being on the
plan, which is exactly what issue 110 fixed for the previous layout.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from crane_planning import weights as crane_weights
from crane_planning.config import PLANNED_DOF, PlannerConfig
from crane_planning.ocp import (
    NU,
    NX,
    ORDER,
    X_ACCEL,
    X_HORIZON,
    X_JERK,
    X_SIGMA,
    X_SPEED,
    build_ocp,
    evaluate,
)
from crane_planning.planner import actuated_samples

COEFFICIENTS = np.zeros((1, ORDER, PLANNED_DOF))
COEFFICIENTS[0, 0] = [0.0, 0.10, -0.20, 0.30, 0.05]
COEFFICIENTS[0, 1] = [1.20, -0.60, 0.90, 0.40, -0.30]
COEFFICIENTS[0, 2] = [-0.80, 0.50, -1.10, 0.20, 0.70]
COEFFICIENTS[0, 3] = [0.60, -0.30, 0.40, -0.50, 0.25]


def description() -> str:
    """The expanded PZS100 description crane_model keeps as a fixture."""
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    return path.read_text()


def test_the_state_layout_is_what_the_model_carries():
    config = PlannerConfig()
    baked = {
        "ocp_intervals": config.ocp_intervals,
        "ocp_horizon": config.ocp_horizon,
        "ocp_integrator": config.ocp_integrator,
        "ocp_max_iterations": config.ocp_max_iterations,
        "ocp_tolerance": config.ocp_tolerance,
        "levenberg_marquardt": config.levenberg_marquardt,
        "q_sway_max": config.q_sway_max,
        "dq_sway_max": config.dq_sway_max,
        "ddq_a_max": config.ddq_a_max,
        "path_segments": config.path_segments,
        "weights": crane_weights.DEFAULTS,
    }
    ocp, _scale, _model = build_ocp(
        description(), baked, {"pump_flow_max": config.pump_flow_max}
    )

    assert ocp.model.x.shape[0] == NX
    assert ocp.model.u.shape[0] == NU
    names = [str(ocp.model.x[row]) for row in range(NX)]
    assert names[X_SIGMA] == "sigma"
    assert names[X_SPEED] == "v"
    assert names[X_ACCEL] == "a"
    assert names[X_JERK] == "j"
    assert names[X_HORIZON] == "theta"


def chain(snap: np.ndarray, duration: float = 2.0) -> SimpleNamespace:
    """A solved chain: integrate a per-interval snap exactly, node to node."""
    nodes = len(snap)
    time = np.linspace(0.0, duration, nodes + 1)
    step = duration / nodes
    sigma, speed, accel, jerk = (np.zeros(nodes + 1) for _ in range(4))
    speed[0] = 0.05
    for k, s in enumerate(snap):
        sigma[k + 1] = (
            sigma[k]
            + speed[k] * step
            + accel[k] * step**2 / 2.0
            + jerk[k] * step**3 / 6.0
            + s * step**4 / 24.0
        )
        speed[k + 1] = (
            speed[k] + accel[k] * step + jerk[k] * step**2 / 2.0 + s * step**3 / 6.0
        )
        accel[k + 1] = accel[k] + jerk[k] * step + s * step**2 / 2.0
        jerk[k + 1] = jerk[k] + s * step
    return SimpleNamespace(
        time=time,
        sigma=sigma,
        speed=speed,
        acceleration=accel,
        jerk=jerk,
        snap=np.append(snap, snap[-1]),
        coefficients=COEFFICIENTS,
    )


def test_the_resampler_reproduces_the_chain_at_the_next_node():
    """
    A lower-order reconstruction lands on the right value only at `dt = 0`.

    Sampling exactly on the node grid would pass whatever formula the resampler
    used, because every interval is entered at `dt = 0`. So the check is that
    integrating interval `k` all the way across reproduces node `k+1`, which is
    the property a quadratic or cubic formula does not have.
    """
    plan = chain(np.array([3.0, -2.0, 1.5, -4.0, 0.5]))
    edges = plan.time[1:] - 1.0e-12  # inside interval k, at its right edge
    q_a, dq_a = actuated_samples(plan, edges)

    expected_q = evaluate(COEFFICIENTS, plan.sigma[1:])
    expected_dq = evaluate(COEFFICIENTS, plan.sigma[1:], order=1) * plan.speed[1:, None]
    assert q_a == pytest.approx(expected_q, abs=1e-9)
    assert dq_a == pytest.approx(expected_dq, abs=1e-9)


def test_a_quadratic_reconstruction_would_fail_that():
    """The oracle discriminates: the pre-112 formula misses by a wide margin."""
    plan = chain(np.array([3.0, -2.0, 1.5, -4.0, 0.5]))
    step = plan.time[1] - plan.time[0]
    quadratic = (
        plan.sigma[:-1]
        + plan.speed[:-1] * step
        + 0.5 * plan.acceleration[:-1] * step**2
    )
    assert np.max(np.abs(quadratic - plan.sigma[1:])) > 1e-3
