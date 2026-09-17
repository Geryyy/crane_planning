"""Path must be C4 in sigma (PT1 inversion needs one derivative beyond qdddot_d);
quintic gives that by construction where a C2 cubic B-spline would not."""

import casadi as ca
import numpy as np
import pytest
from crane_model import symbolic as cs
from crane_planning.config import PLANNED_DOF, Limits, PlannerConfig
from crane_planning.geometry import fit
from crane_planning.ocp import (
    ORDER,
    PATH_DERIVATIVES,
    evaluate,
    path_expression,
    power_coefficients,
)

#: curves in every coordinate so a straight-line fit would not pass by accident
WAYPOINTS = np.column_stack(
    [
        np.linspace(0.0, 0.9, 40) + 0.15 * np.sin(np.linspace(0.0, 3.0, 40)),
        0.40 + 0.20 * np.cos(np.linspace(0.0, 2.0, 40)),
        0.60 + 0.10 * np.sin(np.linspace(0.0, 4.0, 40)),
        np.linspace(0.8, 1.6, 40),
        0.30 * np.sin(np.linspace(0.0, 2.5, 40)),
    ]
)


def limits() -> Limits:
    ones = np.ones(PLANNED_DOF)
    return Limits(
        lower=-10.0 * ones,
        upper=10.0 * ones,
        bounded=np.ones(PLANNED_DOF, dtype=bool),
        description_lower=-10.0 * ones,
        description_upper=10.0 * ones,
        dq_max=ones,
        tau_max=1000.0 * ones,
        flow_max=1.0,
        tool_lower=0.2,
        tool_upper=0.7,
        tool_bounded=True,
    )


def fitted():
    return fit(limits(), PlannerConfig(), WAYPOINTS, np.zeros(PLANNED_DOF))


def piece(coefficients: np.ndarray, segment: int, local: float, order: int):
    """One segment's polynomial and its derivatives, evaluated off the edge."""
    total = np.zeros(coefficients.shape[2])
    for m in range(order, ORDER):
        scale = float(np.prod([m - k for k in range(order)]))
        total += scale * local ** (m - order) * coefficients[segment, m]
    return total


def test_the_fitted_path_is_c4_at_every_interior_breakpoint():
    """Reads the two polynomials, not a finite difference across the join -- a
    one-sided difference can't separate a genuine jump from the next derivative."""
    coefficients = power_coefficients(fitted(), PlannerConfig().path_segments)
    segments = coefficients.shape[0]
    width = 1.0 / segments

    for segment in range(segments - 1):
        for order in range(PATH_DERIVATIVES):
            left = piece(coefficients, segment, width, order)
            right = piece(coefficients, segment + 1, 0.0, order)
            size = np.maximum(np.abs(left), 1.0)
            assert np.max(np.abs(right - left) / size) < 1e-9, (
                f"jump at order {order}, breakpoint {segment + 1}"
            )


def test_evaluate_and_path_expression_agree_at_every_level():
    coefficients = power_coefficients(fitted(), PlannerConfig().path_segments)
    segments = coefficients.shape[0]
    assert coefficients.shape[1] == ORDER

    sigma = ca.SX.sym("sigma")
    symbols = ca.SX.sym("c", segments * ORDER * cs.K_PLANNED_DOF)
    levels = ca.Function(
        "levels",
        [sigma, symbols],
        path_expression(sigma, symbols, segments),
    )

    flat = coefficients.reshape(-1)
    # breakpoints included: where a disagreeing segment lookup would show
    query = np.unique(
        np.concatenate([np.linspace(0.0, 1.0, 97), np.linspace(0.0, 1.0, segments + 1)])
    )
    for value in query:
        symbolic = levels(value, flat)
        for order in range(PATH_DERIVATIVES):
            numeric = evaluate(coefficients, value, order=order)[0]
            assert np.asarray(symbolic[order]).ravel() == pytest.approx(
                numeric, abs=1e-9
            )
