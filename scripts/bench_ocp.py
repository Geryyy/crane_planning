#!/usr/bin/env python3
"""
Plan a corpus of moves and log what the OCP did on each, for tuning it.

    ./scripts/bench_ocp.py                          # 100 moves, seeded
    ./scripts/bench_ocp.py --moves 20 --seed 3
    ./scripts/bench_ocp.py --out build/after.json   # then diff two logs

Start and goal are sampled inside the control-safe box, the goal through FK so
IK has an answer. This sweeps nothing: change the OCP, re-export it, re-run,
compare the logs. The log carries the baked parameters it ran under, so a run
whose solver has moved cannot be mistaken for one that has not.

Refusals are split -- a goal geometry could not reach is a corpus artefact, a
solve that did not converge is the thing being tuned. `--moves` counts moves
sampled, not solved; the geometric stages run before the solver and do not
depend on it, so the same seed refuses the same moves across OCP changes and
two logs stay comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import bench_plan  # noqa: E402
import plan_example  # noqa: E402
from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402
from crane_planning.ocp import BAKED  # noqa: E402

#: Fraction of each axis' range kept clear of the box edge, both ends. Sampled
#: onto a bound, a move is about the bound and not about the solver.
INSET = 0.10

#: What the rotator is sampled over; it is continuous, so it has no range.
ROTATOR = np.pi


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--paths", type=int, default=100, help="paths to sample (default: %(default)s)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGE / "build" / "bench_ocp.json",
        help="log written here (default: %(default)s)",
    )
    return parser.parse_args()


def corpus(planner: Planner, paths: int, seed: int) -> list:
    """Draw `paths` start/goal pairs in planned coordinates, reproducible from `seed`."""
    limits = planner.limits
    lower, upper = limits.lower.copy(), limits.upper.copy()
    lower[~limits.bounded], upper[~limits.bounded] = -ROTATOR, ROTATOR
    span = upper - lower
    lower, upper = lower + INSET * span, upper - INSET * span
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(lower, upper), rng.uniform(lower, upper)) for _ in range(paths)
    ]


def solve(planner: Planner, pair) -> dict:
    """One move: the solve's own report, or why it was refused."""
    start = Start(q=bench_plan.configuration(planner, pair[0]), dq_a=np.zeros(5))
    position, yaw = bench_plan.goal_pose(planner, pair[1])
    clock = time.perf_counter()
    try:
        plan = planner.plan(start, position, yaw, scene=[], avoid_collisions=True)
    except PlanningError as refusal:
        # A refusal carrying solver stats is one the solve reached -- it did not
        # converge, or it converged onto something inadmissible. Everything else
        # was refused on geometry before the solver was reached, and only those
        # are the same every run.
        kind = "ocp" if refusal.stats else "geometry"
        return {"refused": kind, "why": str(refusal)}
    # `report()`'s own keys are left as the node publishes them; the one added
    # here is the whole `plan` call, solver and geometry together.
    return plan.timing.report() | {"planning_time_s": time.perf_counter() - clock}


#: Summary name against the per-path key it reduces. What the run is tuned on.
COST = {
    "sqp_iterations": "sqp_iterations",
    "qp_iterations": "qp_iterations",
    "solver_time_s": "acados_time_s",  # acados' own time_tot, load-bound
    "planning_time_s": "planning_time_s",  # the whole plan call, geometry included
}

#: Same, for what must not get worse when the cost above gets better.
QUALITY = {
    "trajectory_duration_s": "duration_s",
    "terminal_sway_rad": "terminal_sway",
    "slack_spent": "slack",
}


def summary(records: list) -> dict:
    """Reduce the run to what it is read off: cost first, then what it bought."""
    solved = [row for row in records if "refused" not in row]
    out = {
        "paths_sampled": len(records),
        "paths_solved": len(solved),
        "refused_by_ocp": sum(row.get("refused") == "ocp" for row in records),
        "refused_by_geometry": sum(row.get("refused") == "geometry" for row in records),
    }
    if not solved:
        return out
    # A solve acados calls successful can still carry a QP that merely ran out of
    # iterations, so the status is counted apart from the iteration spread.
    # `nan` is "this solver does not report it"; counting it as nonzero would read as a fault.
    out["qp_status_nonzero"] = sum(
        np.isfinite(row["qp_status_worst"]) and row["qp_status_worst"] != 0
        for row in solved
    )
    out["qp_status_unreported"] = sum(
        not np.isfinite(row["qp_status_worst"]) for row in solved
    )
    out["acados_status_nonzero"] = sum(row["acados_status"] != 0 for row in solved)
    for name, key in {**COST, **QUALITY}.items():
        values = np.array([row[key] for row in solved], dtype=float)
        out[name] = {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }
    return out


def main() -> int:
    options = arguments()
    planner = Planner(plan_example.description(), PlannerConfig())
    baked = {key: getattr(planner.config, key) for key in BAKED}

    records = []
    print(
        f"{'path':>4}  {'sqp':>5} {'qp iter':>8} {'qp':>3}"
        f"  {'planning':>9}  {'trajectory':>11}"
    )
    for index, pair in enumerate(corpus(planner, options.paths, options.seed)):
        row = solve(planner, pair)
        records.append(row)
        if "refused" in row:
            print(f"{index:4d}  refused ({row['refused']})  {row['why'][:70]}")
        else:
            # A solver that does not report one of these leaves `nan`; that is a missing
            # measurement, not a zero, and it must not end the run.
            def count(key: str) -> str:
                value = row[key]
                return "  -" if not np.isfinite(value) else f"{int(value):3d}"

            print(
                f"{index:4d}  {int(row['sqp_iterations']):5d} {int(row['qp_iterations']):8d}"
                f" {count('qp_status_worst')}"
                f"  {row['planning_time_s']:7.2f} s  {row['duration_s']:9.2f} s"
            )

    totals = summary(records)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(
        json.dumps(
            {
                "seed": options.seed,
                # Without these a log cannot be attributed to a solver.
                "baked": {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in baked.items()
                },
                "summary": totals,
                "moves": records,
            },
            indent=1,
            default=float,
        )
    )

    print(
        f"\npaths sampled        {totals['paths_sampled']}"
        f"\npaths solved         {totals['paths_solved']}"
        f"\nrefused by the OCP   {totals['refused_by_ocp']}   <- the tuning signal"
        f"\nrefused by geometry  {totals['refused_by_geometry']}"
        "   <- corpus artefact, same every run"
    )
    if totals["paths_solved"]:
        print(
            f"\nqp status nonzero    {totals['qp_status_nonzero']}"
            f" of {totals['paths_solved']} solved"
            f"\nacados status nonzero {totals['acados_status_nonzero']}"
        )
        print(f"\n{'cost':<22}{'median':>10}{'p90':>10}{'max':>10}")
        for name in COST:
            row = totals[name]
            print(
                f"{name:<22}{row['median']:10.3f}{row['p90']:10.3f}{row['max']:10.3f}"
            )
        print(f"\n{'what it bought':<22}{'median':>10}{'p90':>10}{'max':>10}")
        for name in QUALITY:
            row = totals[name]
            print(
                f"{name:<22}{row['median']:10.3f}{row['p90']:10.3f}{row['max']:10.3f}"
            )
        print("-- these must not get worse when the cost above gets better")
    print(f"\nwrote {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
