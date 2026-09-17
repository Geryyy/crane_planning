"""Snap as input makes sigma quartic per interval, C4 in time -- accel-as-input
held ddq_a constant per node and only gave C1."""

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
    baked_parameters,
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
    baked = {**baked_parameters(config), "weights": crane_weights.DEFAULTS}
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
    """Solved chain: integrate a per-interval snap exactly, node to node."""
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
    """Sampling on the node grid passes any formula (dt=0 every interval); this
    integrates interval k all the way across, which only the exact formula survives."""
    plan = chain(np.array([3.0, -2.0, 1.5, -4.0, 0.5]))
    edges = plan.time[1:] - 1.0e-12  # inside interval k, at its right edge
    q_a, dq_a, _ = actuated_samples(plan, edges)

    expected_q = evaluate(COEFFICIENTS, plan.sigma[1:])
    expected_dq = evaluate(COEFFICIENTS, plan.sigma[1:], order=1) * plan.speed[1:, None]
    assert q_a == pytest.approx(expected_q, abs=1e-9)
    assert dq_a == pytest.approx(expected_dq, abs=1e-9)


def test_a_quadratic_reconstruction_would_fail_that():
    """Oracle discriminates: the old quadratic formula misses by a wide margin."""
    plan = chain(np.array([3.0, -2.0, 1.5, -4.0, 0.5]))
    step = plan.time[1] - plan.time[0]
    quadratic = (
        plan.sigma[:-1]
        + plan.speed[:-1] * step
        + 0.5 * plan.acceleration[:-1] * step**2
    )
    assert np.max(np.abs(quadratic - plan.sigma[1:])) > 1e-3
