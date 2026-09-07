"""A refused solve reports its numbers; `CalcMovement` has nowhere to put them.

`CalcMovement.Response` carries no message field, so before `~/solver_stats` a
non-convergence left exactly one trace: a line in the planner's own log. The
numbers that diagnose it -- which of the four KKT residuals stalled -- are also
the ones no `Trajectory` exists to carry, so they ride `PlanningError.stats`.

The two things that can break quietly are asserted here and not much else.
`DiagnosticStatus.level` is a `byte` field, so rclpy types it `bytes` and an
`int` fails its check -- on the refusal path only, which is the path nobody
exercises until the day it matters. And a refusal raised before the solve has no
numbers at all: it must still report, or a silent stream reads as a healthy one.
"""

from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from crane_planning.config import PlanningError
from crane_planning.node import CranePlanner
from crane_planning.ocp import PLAN_ROWS, RESIDUALS, solver_stats
from diagnostic_msgs.msg import DiagnosticStatus


class _Solver:
    """The four `get_stats` fields `solver_stats` reads, and nothing else."""

    def __init__(self, **fields):
        self._fields = fields

    def get_stats(self, field):
        return self._fields[field]


class _Capture:
    def __init__(self):
        self.reports = []

    def publish(self, message):
        self.reports.append(message)


def report(level, message, stats):
    """`_report` against a stand-in node -- the formatting, without a graph."""
    node = SimpleNamespace(
        solver_stats=_Capture(),
        get_name=lambda: "crane_planner",
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
    )
    CranePlanner._report(node, level, message, stats)
    (published,) = node.solver_stats.reports
    (status,) = published.status
    return status


def test_a_non_convergence_reports_its_residuals():
    # `qp_iter` and `qp_stat` come back one entry per SQP iteration, so the
    # report carries the total and the worst; a per-iteration vector is not a
    # row anything can plot beside the scalars.
    solver = _Solver(
        sqp_iter=100,
        qp_iter=np.array([0.0, 16.0, 17.0]),
        qp_stat=np.array([0.0, 0.0, 3.0]),
        time_tot=8.7,
        time_qp=0.21,
        residuals=np.array([1e-3, 2e-9, 3e-7, 4e-12]),
    )
    refusal = PlanningError(
        "the trajectory OCP did not converge", stats=solver_stats(solver, 4, 9.1)
    )
    status = report(DiagnosticStatus.ERROR, str(refusal), refusal.stats)
    assert status.level == DiagnosticStatus.ERROR
    values = {entry.key: entry.value for entry in status.values}
    assert values["acados_status"] == "4"
    assert values["sqp_iterations"] == "100"
    assert values["qp_iterations"] == "33"
    assert values["qp_status_worst"] == "3"
    assert {f"residual_{name}" for name in RESIDUALS} <= set(values)


def test_the_plan_rows_are_there_either_way():
    """A consumer reads one set of keys; `nan` is how "no plan was built" reads.

    The alternative -- omitting them where there is no trajectory -- makes the
    key set depend on the outcome, so every reader has to branch before it can
    plot, and a missing key and a zero become the same thing on a plot.
    """
    solver = _Solver(
        sqp_iter=9,
        qp_iter=np.array([0.0, 16.0]),
        qp_stat=np.zeros(2),
        time_tot=1.0,
        time_qp=0.2,
        residuals=np.zeros(4),
    )
    stats = solver_stats(solver, 0, 1.0)
    assert set(PLAN_ROWS) <= set(stats)
    assert all(np.isnan(stats[name]) for name in PLAN_ROWS)
    values = {
        entry.key: entry.value
        for entry in report(DiagnosticStatus.ERROR, "no plan", stats).values
    }
    assert values["slack"] == "nan"


def test_a_refusal_from_before_the_solve_still_reports():
    status = report(DiagnosticStatus.ERROR, "no robot description", {})
    assert status.level == DiagnosticStatus.ERROR
    assert status.values == []
    assert status.message == "no robot description"
