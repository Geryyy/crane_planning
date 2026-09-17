"""
Trajectory OCP cost weights, and the row layout they sit on.

Shared by `export_timing_ocp.py` (solver defaults), the node (ROS parameters), and
residual readers. `W` is a runtime cost field, not an acados parameter: parameters
get copied onto every stage before every solve, `W` is set once at build, so
per-stage copies would be wasted on weights that don't change within a run.

Every residual row is dimensionless (exporter divides by the limit it's measured
against, so 1.0 is the bound); weight is preference only, 1.0 everywhere means one
bound violated costs the same wherever it happens.
"""

from __future__ import annotations

import numpy as np

#: Stage residual order: y=[dq_a, sway, dq_u, dynamic_tau, ddq_a, dddq_a]; ddq_a was
#: `u` when acceleration was the input, now a state since input is snap.
STAGE_BLOCKS = (
    ("dq_a", 5),
    ("sway", 2),
    ("dq_u", 2),
    ("dynamic_tau", 5),
    ("ddq_a", 5),
    ("dddq_a", 5),
)
#: Terminal residual: no input, so no effort row. `theta` carries minimum time.
TERMINAL_BLOCKS = (("dq_a", 5), ("sway", 2), ("dq_u", 2), ("time", 1))

NY = sum(width for _, width in STAGE_BLOCKS)
NY_E = sum(width for _, width in TERMINAL_BLOCKS)


#: What shipped `config/crane_planner.yaml` carries, and what a node falls back to.
#: `time` is the measured ceiling above which solve asks for more pump than exists.
DEFAULTS = {
    "dq_a": 1.0,
    "sway": 1.0,
    "dq_u": 1.0,
    "dynamic_tau": 1.0,
    "ddq_a": 1.0,
    "dddq_a": 1.0,
    "terminal_scale": 1.0,
    "time": 0.3,
}


def _block(weights: dict, name: str, width: int) -> np.ndarray:
    """Build one diagonal block, scalar or per-row. Refuse anything else."""
    value = np.atleast_1d(np.asarray(weights[name], dtype=float))
    if value.size == 1:
        value = np.full(width, float(value[0]))
    if value.size != width:
        raise ValueError(
            f"weights.{name} is {value.size} entries, but the residual block is {width}"
        )
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"weights.{name} is not finite and non-negative: {value}")
    return value


def stage_diagonal(weights: dict) -> np.ndarray:
    """Return `W`'s diagonal, in `STAGE_BLOCKS` order."""
    return np.concatenate([_block(weights, n, w) for n, w in STAGE_BLOCKS])


def terminal_diagonal(weights: dict) -> np.ndarray:
    """`W_e`'s diagonal: blocks shared with a stage node are the same weights scaled by `terminal_scale`; `time` is the terminal stage's own minimum-time objective."""
    scale = float(weights["terminal_scale"])
    shared = np.concatenate(
        [_block(weights, n, w) for n, w in TERMINAL_BLOCKS if n != "time"]
    )
    return np.concatenate([scale * shared, [float(weights["time"])]])


def matrices(weights: dict) -> tuple:
    """Return `(W, W_e)` as acados wants them, square and dense."""
    return np.diag(stage_diagonal(weights)), np.diag(terminal_diagonal(weights))
