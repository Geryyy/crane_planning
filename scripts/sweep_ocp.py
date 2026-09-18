#!/usr/bin/env python3
"""
Run `bench_ocp.py` once per acados setting variant and table the results.

    ./scripts/sweep_ocp.py --variants variants.json --paths 100

Each variant is a name against a `CRANE_PLANNING_OCP_OPTIONS` patch; the empty
patch is the shipped solver and is always run first, as the baseline every other
row is read against. One subprocess per variant, because the setting is compiled
into the `.so` and a process holds one.

Compiling is the cost -- minutes, once per variant, cached by `ocp.cache_key`
afterwards, so a re-run of the same sweep is the bench alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent

#: Cost keys first, then what must not get worse. Names as `bench_ocp.summary` writes them.
COLUMNS = (
    ("solved", None),
    ("ocp_refusals", None),
    ("sqp_med", ("sqp_iterations", "median")),
    ("sqp_p90", ("sqp_iterations", "p90")),
    ("qp_med", ("qp_iterations", "median")),
    ("acados_med", ("solver_time_s", "median")),
    ("acados_p90", ("solver_time_s", "p90")),
    ("dur_med", ("trajectory_duration_s", "median")),
    ("sway_p90", ("terminal_sway_rad", "p90")),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=PACKAGE / "build" / "sweep_ocp.json"
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--only",
        nargs="*",
        help="run just these variant names (baseline always included)",
    )
    return parser.parse_args()


def run(name: str, patch: dict, options) -> dict:
    """One variant, in its own process; returns its summary plus what it cost."""
    log = options.out.parent / "sweep" / f"{name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    # A variant is acados settings, or `{"options": ..., "env": ...}` when it also moves something
    # that is not compiled in -- the initial guess is the one that matters.
    shaped = "options" in patch or "env" in patch
    acados = patch.get("options", {}) if shaped else patch
    environment = dict(
        os.environ,
        CRANE_PLANNING_OCP_OPTIONS=json.dumps(acados),
        **{k: str(v) for k, v in patch.get("env", {}).items()},
    )
    clock = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "scripts" / "bench_ocp.py"),
            "--paths",
            str(options.paths),
            "--seed",
            str(options.seed),
            "--out",
            str(log),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=options.timeout,
    )
    wall = time.monotonic() - clock
    if result.returncode != 0 or not log.is_file():
        return {"name": name, "patch": patch, "failed": result.stderr[-1500:]}
    record = json.loads(log.read_text())
    return {
        "name": name,
        "patch": patch,
        "wall_s": wall,
        # A cold variant compiles first, so wall time is not comparable across rows; the
        # per-solve numbers inside the summary are.
        "compiled": "compiling, which takes minutes" in result.stderr,
        "summary": record["summary"],
    }


def cell(summary: dict, key) -> float:
    if key is None:
        return float("nan")
    block = summary.get(key[0])
    return float(block[key[1]]) if isinstance(block, dict) else float("nan")


def table(rows: list) -> str:
    head = f"{'variant':<26}" + "".join(f"{name:>11}" for name, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for row in rows:
        if "failed" in row:
            lines.append(f"{row['name']:<26}  FAILED")
            continue
        summary = row["summary"]
        cells = [
            f"{summary['paths_solved']:>11d}",
            f"{summary['refused_by_ocp']:>11d}",
        ] + [f"{cell(summary, key):>11.3f}" for _, key in COLUMNS[2:]]
        lines.append(f"{row['name']:<26}" + "".join(cells))
    return "\n".join(lines)


def main() -> int:
    options = arguments()
    variants = json.loads(options.variants.read_text())
    variants = {"baseline": {}, **variants}
    if options.only:
        wanted = set(options.only) | {"baseline"}
        variants = {k: v for k, v in variants.items() if k in wanted}

    rows = []
    for name, patch in variants.items():
        print(f"== {name}  {json.dumps(patch)}", flush=True)
        row = run(name, patch, options)
        rows.append(row)
        if "failed" in row:
            print(f"   FAILED\n{row['failed']}", flush=True)
        else:
            summary = row["summary"]
            print(
                f"   solved {summary['paths_solved']}  ocp-refused "
                f"{summary['refused_by_ocp']}  sqp med "
                f"{cell(summary, ('sqp_iterations', 'median')):.0f}  "
                f"acados med {cell(summary, ('solver_time_s', 'median')):.3f} s"
                f"  [{row['wall_s']:.0f} s{', compiled' if row['compiled'] else ''}]",
                flush=True,
            )
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(
            json.dumps(
                {"paths": options.paths, "seed": options.seed, "rows": rows}, indent=1
            )
        )

    print(f"\n{table(rows)}\n\nwrote {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
