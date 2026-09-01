"""
The trajectory OCP's cost weights, and the row layout they sit on.

One place, because three things must agree: `export_timing_ocp.py` bakes these as
the solver's defaults, the node writes its own ROS parameters onto that solver at
construction, and anything reading a residual back needs to know which row is
which.

`W` is a **runtime cost field**, not an acados parameter. Parameters carry values
entering the model *expressions* and are copied onto every stage before every
solve; `W` is set once when the solver is built, so putting weights in `p` would
pay per-stage copies for numbers that never change within a run.

Every residual row is dimensionless -- the exporter divides each by the limit it
is measured against, so 1.0 is the bound. A weight is preference and nothing
else, and 1.0 everywhere is the honest default: one bound violated costs the same
wherever it happens.
"""

from __future__ import annotations

import numpy as np

#: Stage residual, in row order: `y = [dq_a, sway, dq_u, dynamic_tau, u]`.
STAGE_BLOCKS = (("dq_a", 5), ("sway", 2), ("dq_u", 2), ("dynamic_tau", 5), ("u", 5))
#: Terminal residual: no input, so no effort row; `theta` carries minimum time.
TERMINAL_BLOCKS = (("dq_a", 5), ("sway", 2), ("dq_u", 2), ("time", 1))

NY = sum(width for _, width in STAGE_BLOCKS)
NY_E = sum(width for _, width in TERMINAL_BLOCKS)


#: What the shipped `config/crane_planner.yaml` carries, and what a node falls
#: back to. Preferences are 1.0 because the rows are dimensionless; `time` is the
#: measured ceiling above which the solve asks for more pump than exists.
DEFAULTS = {
    "dq_a": 1.0,
    "sway": 1.0,
    "dq_u": 1.0,
    "dynamic_tau": 1.0,
    "u": 1.0,
    "terminal_scale": 1.0,
    "time": 0.3,
}


def _block(weights: dict, name: str, width: int) -> np.ndarray:
    """One block of the diagonal, scalar or per-row, refused if it is neither."""
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
    """`W`'s diagonal, in `STAGE_BLOCKS` order."""
    return np.concatenate([_block(weights, n, w) for n, w in STAGE_BLOCKS])


def terminal_diagonal(weights: dict) -> np.ndarray:
    """
    `W_e`'s diagonal.

    The blocks the terminal stage shares with a stage node are the same weights
    scaled by `terminal_scale`, which is what keeps a preference expressed once;
    `time` is the terminal stage's own and is the whole minimum-time objective.
    """
    scale = float(weights["terminal_scale"])
    shared = np.concatenate(
        [_block(weights, n, w) for n, w in TERMINAL_BLOCKS if n != "time"]
    )
    return np.concatenate([scale * shared, [float(weights["time"])]])


def matrices(weights: dict) -> tuple:
    """`(W, W_e)` as acados wants them, square and dense."""
    return np.diag(stage_diagonal(weights)), np.diag(terminal_diagonal(weights))
