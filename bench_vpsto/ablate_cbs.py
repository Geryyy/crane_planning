#!/usr/bin/env python3
"""
Where CBS's duration goes: drop one guarantee at a time, same cases as tune_cbs.py.

    ./ablate_cbs.py        # -> results/ablate_cbs.md
"""

import common  # noqa: F401

# isort: split
import numpy as np
import tune_cbs

from bench_vpsto import RESULTS
from crane_planning.weights import DEFAULTS

OPEN_BOX = dict(terminal_q_sway_max=[0.2, 0.2], terminal_dq_sway_max=[1.0, 0.5])
WIDE = dict(q_sway_max=[0.6, 0.6], dq_sway_max=[3.0, 3.0])  # recompiles the solver
NO_JERK = {**DEFAULTS, "dddq_a": 1e-3}
NO_SWAY = {**DEFAULTS, "sway": 0.0, "dq_u": 0.0}
VARIANTS = (
    ("shipped", {}, DEFAULTS),
    ("settled box removed", OPEN_BOX, DEFAULTS),
    ("jerk preference off", {}, NO_JERK),
    ("settled box removed, jerk off", OPEN_BOX, NO_JERK),
    ("sway + damping cost off", OPEN_BOX, NO_SWAY),
    (
        "... + sway bound in transit 0.6 rad",
        {**WIDE, "terminal_q_sway_max": [0.6, 0.6], "terminal_dq_sway_max": [3.0, 3.0]},
        NO_SWAY,
    ),
)


def main():
    base, out = tune_cbs.PlannerConfig, {}
    for name, cfg, weights in VARIANTS:
        arrays = {k: np.array(v) for k, v in cfg.items()}
        tune_cbs.PlannerConfig = lambda **k: base(**k, **arrays)  # noqa: B023
        out[name] = {r["move"]: r for r in tune_cbs.evaluate(dict(weights))}
        print(name, flush=True)
    moves = list(out["shipped"])
    med = lambda rows, k: np.nanmedian([r.get(k, np.nan) for r in rows])  # noqa: E731
    lines = [
        "# CBS duration ablation (ablate_cbs.py)",
        "",
        "| variant | median T s | planned | settled at T | median residual sway deg |",
        "|---|---|---|---|---|",
    ]
    for name, rows in out.items():
        rows = list(rows.values())
        lines.append(
            f"| {name} | {med(rows, 'duration_s'):.2f} | "
            f"{sum('duration_s' in r for r in rows)}/{len(rows)} | "
            f"{sum(r.get('settled_at_T', 0) for r in rows)}/{len(rows)} | "
            f"{med(rows, 'residual_sway_deg'):.2f} |"
        )
    lines += [
        "",
        "## Duration per move, s",
        "",
        "| move | " + " | ".join(out) + " |",
        "|---" * (len(out) + 1) + "|",
    ]
    for m in moves:
        lines.append(
            f"| {m} | "
            + " | ".join(f"{out[n][m].get('duration_s', np.nan):.2f}" for n in out)
            + " |"
        )
    (RESULTS / "ablate_cbs.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
