#!/usr/bin/env python3
"""
How fast can CBS get by cost weights alone, still settled and clear? Optuna over `weights`.

Fixed, as in bench_vpsto.py: limits, kappa, margins, settled box, scene, payload. Free: the
cost weights (and duration floor 0.5 s). Objective: median duration over the bench cases;
constraint: every case planned, settled at T, clean on the dense check, limit use <= UTIL_MAX.

    ./tune_cbs.py --trials 40 --jobs 4      # sqlite study in results/, resumable
    ./tune_cbs.py --report                  # best trial on every case -> results/tune_cbs.md
"""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"  # acados OpenMP; set before it loads

import common  # noqa: F401

# isort: split
import numpy as np
import optuna
from common import description
from export_model import build

from bench_vpsto import EXCLUDED, RESULTS, Evaluator, cases, clean, cost_model
from crane_planning import Planner, PlannerConfig, PlanningError
from crane_planning.planner import actuated_samples
from crane_planning.weights import DEFAULTS

STORAGE = f"sqlite:///{RESULTS / 'tune_cbs.db'}"
FREE = ("time", "dq_a", "sway", "dq_u", "dynamic_tau", "ddq_a", "dddq_a")
#: time-weight ladder at default shape weights, enqueued first: the 1-D answer.
#: limit use allowed; default weights reach 1.063 (pump row between OCP nodes).
UTIL_MAX = 1.07
LADDER = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def evaluate(weights, only=None):
    planner = Planner(description(), PlannerConfig(ocp_duration_min=0.5), weights)
    ev = Evaluator(planner, build(description())[0])
    rows = []
    for case in [c for c in cases(planner, only) if c.name not in EXCLUDED]:
        scene = planner.prepare_scene(case.scene, True)
        model = cost_model(planner, case, scene)
        model.has_scene = bool(scene)
        try:
            plan = case.plan(planner)
        except PlanningError as refusal:
            rows.append(dict(move=case.name, ok=0, note=str(refusal)[:80]))
            continue
        timing = plan.timing
        m = ev.row(
            case, scene, model, lambda t: actuated_samples(timing, t), timing.time[-1]
        )
        util = max(m["util_vel"], m["util_acc"], m["util_pump"])
        ok = clean(m) and m["settled_at_T"] and util <= UTIL_MAX
        rows.append(dict(move=case.name, ok=int(ok), iterations=timing.iterations, **m))
    return rows


def objective(trial):
    w = dict(DEFAULTS)
    for name in FREE:
        lo, hi = (0.3, 300.0) if name == "time" else (0.01, 30.0)
        w[name] = trial.suggest_float(name, lo, hi, log=True)
    rows = evaluate(w)
    bad = sum(1 - r["ok"] for r in rows)
    trial.set_user_attr("failed", [r["move"] for r in rows if not r["ok"]])
    trial.set_user_attr("max_iter", max(r.get("iterations", 0) for r in rows))
    trial.set_user_attr(
        "residual_deg_max", max(r.get("residual_sway_deg", 0) for r in rows)
    )
    trial.set_user_attr("constraint", [float(bad)])
    durations = [r.get("duration_s", 20.0) for r in rows]
    return float(np.median(durations))


def sampler(seed):
    return optuna.samplers.TPESampler(
        seed=seed, constraints_func=lambda t: t.user_attrs["constraint"]
    )


def study():
    s = optuna.create_study(
        study_name="cbs_duration",
        storage=STORAGE,
        sampler=sampler(0),
        load_if_exists=True,
    )
    if not s.trials:
        for time_weight in LADDER:
            s.enqueue_trial({**{k: DEFAULTS[k] for k in FREE}, "time": time_weight})
    return s


def report(s, path):
    feasible = [
        t
        for t in s.trials
        if t.value is not None and t.user_attrs["constraint"] == [0.0]
    ]
    best = min(feasible, key=lambda t: t.value)
    lines = [
        "# CBS duration by cost weights (tune_cbs.py)",
        "",
        "## Trials",
        "",
        "| trial | " + " | ".join(FREE) + " | median T s | failed | max it |",
        "|" + "---|" * (len(FREE) + 4),
    ]
    for t in sorted(s.trials, key=lambda t: t.value or 99):
        if t.value is None:
            continue
        lines.append(
            f"| {t.number} | "
            + " | ".join(f"{t.params[k]:.3g}" for k in FREE)
            + f" | {t.value:.2f} | {','.join(t.user_attrs['failed']) or '-'}"
            f" | {t.user_attrs['max_iter']} |"
        )
    rows = evaluate({**DEFAULTS, **best.params})
    lines += [
        "",
        f"## Best feasible: trial {best.number}",
        "",
        "| move | ok | duration s | residual sway deg | settled at T | iterations | util pump |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['move']} | {r['ok']} | {r.get('duration_s', np.nan):.2f} | "
            f"{r.get('residual_sway_deg', np.nan):.2f} | {r.get('settled_at_T', '-')} | "
            f"{r.get('iterations', '-')} | {r.get('util_pump', np.nan):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--report", action="store_true")
    options = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    s = study()
    if options.report:
        return report(s, RESULTS / "tune_cbs.md")
    per = -(-options.trials // options.jobs)
    children = []
    for job in range(options.jobs):
        pid = os.fork()
        if pid == 0:
            optuna.load_study(
                study_name="cbs_duration", storage=STORAGE, sampler=sampler(os.getpid())
            ).optimize(objective, n_trials=per)
            os._exit(0)
        children.append(pid)
    for pid in children:
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
