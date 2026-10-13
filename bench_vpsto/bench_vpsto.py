#!/usr/bin/env python3
"""
CBS planner (crane_planning) vs VP-STO (EcB25 cost, ros2_vpsto) on the PZS100.

Same limits, scene and payload; both trajectories replay through one passive-dynamics sim.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    ./build.sh && ./bench_vpsto.py            # whole benchmark, idle machine
    ./bench_vpsto.py --only slew --runs 3     # smoke
"""

import argparse
import csv
import os
import platform
import time
from collections import defaultdict

import common  # noqa: F401  (puts the workspace packages on sys.path)

# isort: split
import bench_plan as bp
import carry_move as wall
import numpy as np
from common import HERE, description
from crane_model import Payload
from crane_planning import Planner, PlannerConfig, PlanningError, Start
from crane_planning.config import passive_equilibrium, settled_offset
from crane_planning.geometry import payload_primitive
from crane_planning.planner import (
    PASSIVE_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    actuated_samples,
    payload_parameters,
)
from export_model import build
from lib import CraneVpsto
from replay import Replay

N_VIA, N_EVAL, N_SAMPLES, THREADS = (
    6,
    80,
    50,
    8,
)  # EcB25 via/population; old server nEval/threads
RESULTS = HERE / "results"
FIELDS = (
    "move planner run success plan_s solver_ms iterations duration_s terminal_sway_deg "
    "terminal_rate peak_sway_deg settled_at_T settle_s util_vel util_acc util_pump "
    "clearance_eq_m clearance_sway_m cost"
).split()


class Case:
    def __init__(
        self,
        planner,
        name,
        start,
        position,
        yaw,
        goal_q=None,
        payload=None,
        shape=None,
        scene=(),
    ):
        self.name, self.start, self.position, self.yaw = name, start, position, yaw
        self.goal_q, self.payload, self.shape, self.scene = (
            goal_q,
            payload,
            shape,
            list(scene),
        )
        self.vector = payload_parameters(payload)
        self.offset = (
            settled_offset(planner.model, payload, start.q_tool)
            if payload is not None
            else None
        )

    def plan(self, planner):
        return planner.plan(
            self.start,
            self.position,
            self.yaw,
            payload=self.payload,
            payload_shape=self.shape,
            scene=self.scene,
            avoid_collisions=True,
        )


def cases(planner, only):
    out = []
    for name, a, b in (*bp.MOVES, *bp.WIDE_MOVES):
        position, yaw = bp.goal_pose(planner, b)
        start = Start(q=bp.configuration(planner, a), dq_a=np.zeros(5))
        out.append(Case(planner, name, start, position, yaw, goal_q=np.array(b)))
    name, a, b = bp.CARRY_MOVES[0]
    position, yaw = bp.goal_pose(planner, b)
    out.append(
        Case(
            planner,
            "bench_carry",
            Start(q=bp.configuration(planner, a), dq_a=np.zeros(5)),
            position,
            yaw,
            goal_q=np.array(b),
            payload=Payload(
                mass_kg=bp.CARRY_MASS, center_of_mass_k8_m=bp.CARRY_COM, valid=True
            ),
            shape=bp.CARRY_SHAPE,
            scene=bp.CARRY_SCENE,
        )
    )
    out.append(
        Case(
            planner,
            "example_wall",
            Start(q=np.array(wall.START_Q), dq_a=np.zeros(5)),
            wall.PLACE,
            wall.YAW,
            payload=Payload(mass_kg=wall.MASS, valid=True),
            scene=wall.scene(),
        )
    )
    return [c for c in out if only in (None, c.name)]


class Evaluator:
    """Replay + limit utilisation + clearance, identical for both planners."""

    def __init__(self, planner, pump):
        self.planner, self.replay = planner, Replay()
        self.pump = pump.map(1)  # re-mapped per call length below
        self.pump_fn = pump

    def utilisation(self, sampler, T):
        cfg, lim = self.planner.config, self.planner.limits
        q, dq, ddq = sampler(np.linspace(0.0, T, int(np.ceil(T / 0.01)) + 1))
        flow = np.asarray(self.pump_fn.map(len(q))(q.T, dq.T)).ravel()
        return (
            float(np.max(np.abs(dq) / (cfg.kappa * lim.dq_max))),
            float(np.max(np.abs(ddq) / (cfg.kappa * np.asarray(cfg.ddq_a_max)))),
            float(np.max(np.abs(flow)) / (cfg.kappa * lim.flow_max)),
        )

    def clearance(self, case, rep, scene):
        """Min scene distance along the path at the equilibrium and at the replayed sway."""
        if not scene and case.shape is None:
            return np.nan, np.nan
        model, best = self.planner.model, [np.inf, np.inf]
        for i in np.linspace(0, len(rep["t"]) - 1, 80).astype(int):
            a = rep["q_a"][i]
            eq = passive_equilibrium(a) + (case.offset(a) if case.offset else 0.0)
            for slot, q_u in enumerate((eq, rep["q_u"][i])):
                q = np.zeros(8)
                q[list(PLANNED_INDICES)], q[list(PASSIVE_INDICES)] = a, q_u
                q[TOOL_INDEX] = case.start.q_tool
                carried = payload_primitive(model, q, case.shape)
                bodies = scene + ([carried] if carried else [])
                rows = model.collision_queries(q, bodies)[: len(bodies)]
                best[slot] = min(best[slot], min(r.minimum_distance_m for r in rows))
        return tuple(best)

    def row(self, case, scene, sampler, T):
        params = np.concatenate([[case.start.q_tool], case.vector])
        rep = self.replay.run(sampler, T, case.start.q_u, params, case.offset)
        vel, acc, pump = self.utilisation(sampler, T)
        clr = self.clearance(case, rep, scene)
        return dict(
            duration_s=T,
            terminal_sway_deg=np.degrees(rep["terminal_sway"]),
            terminal_rate=rep["terminal_rate"],
            peak_sway_deg=np.degrees(rep["peak_sway"]),
            settled_at_T=int(rep["settled_at_T"]),
            settle_s=rep["settle_s"],
            util_vel=vel,
            util_acc=acc,
            util_pump=pump,
            clearance_eq_m=clr[0],
            clearance_sway_m=clr[1],
        )


def run_cbs(planner, ev, case, scene, repeats):
    rows, plan = [], None
    for repeat in range(repeats):
        began = time.perf_counter()
        try:
            candidate = case.plan(planner)
        except PlanningError as refusal:
            rows.append(
                dict(
                    move=case.name,
                    planner="cbs",
                    run=repeat,
                    success=0,
                    plan_s=time.perf_counter() - began,
                    note=str(refusal)[:120],
                )
            )
            continue
        wall_s = time.perf_counter() - began
        plan = plan or candidate
        rows.append(
            dict(
                move=case.name,
                planner="cbs",
                run=repeat,
                success=1,
                plan_s=wall_s,
                iterations=candidate.timing.iterations,
                solver_ms=1e3 * candidate.timing.acados_time_s,
            )
        )
    if plan is not None:
        timing = plan.timing
        metrics = ev.row(
            case, scene, lambda t: actuated_samples(timing, t), timing.time[-1]
        )
        for row in rows:
            if row["success"]:
                row.update(metrics)
    return rows, plan


def run_vpsto(planner, ev, case, scene, q_goal, runs):
    cfg, lim = planner.config, planner.limits.relaxed_to(case.start.q_a)
    big = 1e9
    model = CraneVpsto(
        cfg.kappa * lim.dq_max,
        cfg.kappa * np.asarray(cfg.ddq_a_max),
        np.where(np.isfinite(lim.lower), lim.lower, -big),
        np.where(np.isfinite(lim.upper), lim.upper, big),
        cfg.kappa * lim.flow_max,
        case.start.q_tool,
        scene,
        case.shape,
    )
    q0 = case.start.q_a
    rows = []
    for run in range(runs):
        began = time.perf_counter()
        sol = model.solve(q0, q_goal, N_VIA, N_EVAL, N_SAMPLES, THREADS)
        wall_s = time.perf_counter() - began
        row = dict(
            move=case.name,
            planner="vpsto",
            run=run,
            plan_s=wall_s,
            solver_ms=sol["ms"],
            cost=sol["cost"],
            success=int(sol["cost"] - sol["T"] < 1e-6),
        )
        T = sol["T"]
        _, q, dq, ddq = model.sample(q0, q_goal, sol["via"], T, int(np.ceil(T / 1e-3)))
        grid = np.linspace(0, T, len(q))

        def sampler(t, q=q, dq=dq, ddq=ddq, grid=grid):
            return tuple(
                np.stack([np.interp(t, grid, a[:, i]) for i in range(5)], 1)
                for a in (q, dq, ddq)
            )

        row.update(ev.row(case, scene, sampler, T))
        rows.append(row)
    return rows


NOTES = """
Columns: plan s = wall of the whole call (CBS Planner.plan incl. geometry; VP-STO vc_solve incl.
basis setup); duration = plan T; sway/settle from the shared passive-dynamics replay (start at
rest, q_u0 = the planner's start; settle = time after T to enter and stay in the 0.02 rad /
0.04 rad/s box with actuators held, ">20" = not within 20 s); clearance = worst run, 80 samples
along the path, scene bodies only (empty for the 21 bench moves). VP-STO success = final cost
has no penalty. Both planners stay inside their limits (see runs.csv util_*, fractions of the
kappa-scaled bound). lim_boom_high / lim_arm_out: CBS refuses (goal folds the machine into
itself); VP-STO has no self-collision term and was given the bench goal joints. Plan times are
load-bound; an unrelated Gazebo session ran during a first pass, which was discarded.
"""


def summarise(rows, path, header):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["move"], r["planner"])].append(r)

    def med(rs, key):
        v = [
            r[key]
            for r in rs
            if r["success"]
            and key in r
            and np.isfinite(r.get(key, np.nan))
            or r["success"]
            and r.get(key) == np.inf
        ]
        return (
            "-"
            if not v
            else (">20" if np.median(v) == np.inf else f"{np.median(v):.2f}")
        )

    lines = [
        header,
        "",
        "| move | planner | n | success | plan s med / p95 | duration s | "
        "term sway deg med / max | peak sway deg | settled at T | settle s | "
        "min clearance eq / sway m |",
        "|" + "---|" * 11,
    ]
    for (move, name), rs in groups.items():
        wall = [r["plan_s"] for r in rs]
        ok = [r for r in rs if r["success"]]
        sway = [r["terminal_sway_deg"] for r in ok]
        settled = f"{sum(r['settled_at_T'] for r in ok)}/{len(ok)}" if ok else "-"
        clr = [
            (r["clearance_eq_m"], r["clearance_sway_m"])
            for r in ok
            if np.isfinite(r.get("clearance_eq_m", np.nan))
        ]
        clearance = (
            f"{min(c[0] for c in clr):.2f} / {min(c[1] for c in clr):.2f}"
            if clr
            else "-"
        )
        lines.append(
            f"| {move} | {name} | {len(rs)} | {len(ok)}/{len(rs)} | "
            f"{np.median(wall):.2f} / {np.percentile(wall, 95):.2f} | {med(rs, 'duration_s')} | "
            f"{med(rs, 'terminal_sway_deg')} / {max(sway):.2f} | {med(rs, 'peak_sway_deg')} | "
            f"{settled} | {med(rs, 'settle_s')} | {clearance} |"
            if ok
            else f"| {move} | {name} | {len(rs)} | 0/{len(rs)} | "
            f"{np.median(wall):.2f} / {np.percentile(wall, 95):.2f} | - | - | - | - | - | - |"
        )
    path.write_text("\n".join(lines) + "\n" + NOTES)


def cpu() -> str:
    model = next(
        (l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l),
        "?",
    )
    return f"{model}, nproc {os.cpu_count()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", default="runs.csv")
    options = parser.parse_args()

    planner = Planner(description(), PlannerConfig())
    ev = Evaluator(planner, build(description())[0])
    RESULTS.mkdir(exist_ok=True)
    all_rows = []
    todo = cases(planner, options.only)
    bp.run(planner, bp.MOVES[0], 1.0)  # discarded: first call loads solver lazily
    for case in todo:
        scene = planner.prepare_scene(case.scene, True)
        cbs_rows, plan = run_cbs(planner, ev, case, scene, options.repeats)
        q_goal = plan.timing.q_a[-1] if plan is not None else case.goal_q
        print(
            f"load {os.getloadavg()[0]:4.1f} {case.name:<15} cbs {'T=%.2f s' % plan.timing.duration if plan else 'REFUSED'}",
            end="  ",
            flush=True,
        )
        all_rows += cbs_rows
        if q_goal is None:
            print("vpsto skipped: no goal joints")
            continue
        rows = run_vpsto(planner, ev, case, scene, q_goal, options.runs)
        all_rows += rows
        ok = [r for r in rows if r["success"]]
        print(
            f"vpsto {len(ok)}/{len(rows)} ok, T~{np.median([r['duration_s'] for r in rows]):.2f} s, "
            f"{np.median([r['plan_s'] for r in rows]):.2f} s wall",
            flush=True,
        )
    keys = FIELDS + ["note"]
    with open(RESULTS / options.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    header = (
        f"# CBS vs VP-STO (generated {time.strftime('%Y-%m-%d')})\n\nCPU: {cpu()}; "
        f"VP-STO {THREADS} OpenMP threads, nVia {N_VIA}, nEval {N_EVAL}, population "
        f"{N_SAMPLES}, {options.runs} runs/move; CBS acados single thread, "
        f"{options.repeats} repeats/move. {platform.platform()}"
    )
    summarise(
        all_rows,
        RESULTS / ("summary.md" if options.out == "runs.csv" else "smoke.md"),
        header,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
