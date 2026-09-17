"""Emitted reference must lie on the certified curve, not the chord np.interp over nodes gives."""

from types import SimpleNamespace

import numpy as np
import pytest
from crane_planning.config import PLANNED_DOF
from crane_planning.ocp import ORDER, evaluate
from crane_planning.planner import actuated_samples

#: one segment (evaluate's lookup not under test); cubic term makes the chord miss
COEFFICIENTS = np.zeros((1, ORDER, PLANNED_DOF))
COEFFICIENTS[0, 0] = [0.0, 0.10, -0.20, 0.30, 0.05]
COEFFICIENTS[0, 1] = [1.20, -0.60, 0.90, 0.40, -0.30]
COEFFICIENTS[0, 2] = [-0.80, 0.50, -1.10, 0.20, 0.70]
COEFFICIENTS[0, 3] = [0.60, -0.30, 0.40, -0.50, 0.25]


def timing(nodes: int = 4, duration: float = 2.0) -> SimpleNamespace:
    """Solved chain over `[0, duration]`: constant acceleration, zero above it."""
    time = np.linspace(0.0, duration, nodes + 1)
    a = 2.0 / duration**2  # sigma(0) = 0, sigma(T) = 1, from rest
    return SimpleNamespace(
        time=time,
        sigma=0.5 * a * time**2,
        speed=a * time,
        acceleration=np.full(nodes + 1, a),
        jerk=np.zeros(nodes + 1),
        snap=np.zeros(nodes + 1),
        coefficients=COEFFICIENTS,
    )


def recovered_sigma(row: np.ndarray) -> float:
    """`sigma` read back off coordinate 0, monotone over `[0, 1]`."""
    dense = np.linspace(0.0, 1.0, 200001)
    return float(np.interp(row[0], evaluate(COEFFICIENTS, dense)[:, 0], dense))


def test_every_emitted_configuration_lies_on_the_curve():
    plan = timing()
    stamps = np.linspace(0.0, plan.time[-1], 37)  # deliberately off the node grid
    q_a, _, _ = actuated_samples(plan, stamps)

    for row in q_a:
        on_curve = evaluate(COEFFICIENTS, recovered_sigma(row))[0]
        assert np.allclose(row, on_curve, atol=1e-6)


def test_the_chord_would_fail_this():
    """Oracle discriminates: linear interpolation over the nodes does not pass."""
    plan = timing()
    stamps = np.linspace(0.0, plan.time[-1], 37)
    at_nodes = evaluate(COEFFICIENTS, plan.sigma)
    chord = np.array(
        [np.interp(stamps, plan.time, at_nodes[:, i]) for i in range(PLANNED_DOF)]
    ).T

    worst = max(
        float(np.max(np.abs(row - evaluate(COEFFICIENTS, recovered_sigma(row))[0])))
        for row in chord
    )
    assert worst > 1e-3


def test_velocities_are_the_derivative_of_the_positions_emitted():
    plan = timing()
    stamps = np.linspace(0.0, plan.time[-1], 2001)
    q_a, dq_a, _ = actuated_samples(plan, stamps)

    interior = slice(1, -1)
    difference = np.gradient(q_a, stamps, axis=0)[interior]
    assert difference == pytest.approx(dq_a[interior], abs=1e-4)
