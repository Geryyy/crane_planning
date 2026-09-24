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


def unclamped(coefficients, sigma) -> np.ndarray:
    """`evaluate` as it was before the `sigma` clamp: segment index clipped, `sigma` not."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    segments = coefficients.shape[0]
    index = np.clip(np.floor(sigma * segments).astype(int), 0, segments - 1)
    local = sigma - index / segments
    return sum(local[:, None] ** m * coefficients[index, m, :] for m in range(ORDER))


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


def test_a_reconstruction_past_the_end_stops_at_the_curves_endpoint():
    """`sigma` is reconstructed, so it overshoots 1; past the end is the end, not more curve."""
    past = np.array([1.0, 1.0 + 1e-9, 1.01, 1.3])
    endpoint = evaluate(COEFFICIENTS, 1.0)[0]
    assert evaluate(COEFFICIENTS, past) == pytest.approx(
        np.tile(endpoint, (past.size, 1))
    )
    assert evaluate(COEFFICIENTS, [-0.2, 0.0]) == pytest.approx(
        np.tile(evaluate(COEFFICIENTS, 0.0)[0], (2, 1))
    )
    # Oracle discriminates: extrapolating the final quintic leaves the curve outright.
    assert np.max(np.abs(unclamped(COEFFICIENTS, 1.3) - endpoint)) > 0.1


def test_the_published_joint_path_is_unchanged_by_the_clamp():
    """`/crane/joint_path` samples inside [0, 1], so the clamp may not move one of them."""
    node = pytest.importorskip("crane_planning.node")
    places = np.linspace(0.0, 1.0, node.PATH_SAMPLES)
    assert evaluate(COEFFICIENTS, places) == pytest.approx(
        unclamped(COEFFICIENTS, places), abs=0.0
    )


def test_velocities_are_the_derivative_of_the_positions_emitted():
    plan = timing()
    stamps = np.linspace(0.0, plan.time[-1], 2001)
    q_a, dq_a, _ = actuated_samples(plan, stamps)

    interior = slice(1, -1)
    difference = np.gradient(q_a, stamps, axis=0)[interior]
    assert difference == pytest.approx(dq_a[interior], abs=1e-4)
