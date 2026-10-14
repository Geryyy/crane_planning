#!/usr/bin/env python3
"""
CBS planner (crane_planning) vs VP-STO (EcB25 cost, ros2_vpsto) on the PZS100.

Same limits, scene and payload; both trajectories replay through one passive-dynamics sim
and get the same dense (1 kHz) limit / collision check.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    ./build.sh
    ./bench_vpsto.py --cpus 0-7                        # whole benchmark, idle machine
    ./bench_vpsto.py --only slew,bench_carry --repeats-cbs 2 --runs 3 --out smoke.csv
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
from lib import CraneVpsto, set_threads
from replay import Replay

N_EVAL, N_SAMPLES = 80, 50
RESULTS = HERE / "results"
LOAD_FLAG = 1.5
#: CBS refuses (goal folds the machine into itself); VP-STO has no self-collision term.
EXCLUDED = ("lim_boom_high", "lim_arm_out")
FIELDS = (
    "move planner threads run success plan_s solver_ms iterations t_ocp_s t_acados_s t_geom_s "
    "t_rest_s ms_per_iter load load_flag duration_s peak_sway_deg residual_sway_deg terminal_sway_deg "
    "settled_at_T settle_s acc0 accT util_vel util_acc util_pump util_over col_cost box_viol "
    "clearance_eq_m clearance_sway_m cost note"
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
        self.offset = None
        if payload is not None:  # start at the payload's settled hanging pose
            self.offset = settled_offset(planner.model, payload, start.q_tool)
            start.q[list(PASSIVE_INDICES)] = passive_equilibrium(
                start.q_a
            ) + self.offset(start.q_a)

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
    _, a, b = bp.CARRY_MOVES[0]
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
    # carry_move plans without a carried body; the same move with the 0.9x0.6x0.6 block hanging
    # at the BT-published CoM (as bench_carry) gives both planners and the dynamics one body.
    # A block centred on K8 is refused by CBS (0.094 m < 0.150 m "payload against the machine").
    start = np.array(wall.START_Q)
    block = ("box", wall.BLOCK, bp.CARRY_COM)
    for name, mass_com, shape in (
        ("example_wall_nobody", np.zeros(3), None),
        ("example_wall_body", bp.CARRY_COM, block),
    ):
        payload = Payload(mass_kg=wall.MASS, center_of_mass_k8_m=mass_com, valid=True)
        out.append(
            Case(
                planner,
                name,
                Start(q=start.copy(), dq_a=np.zeros(5)),
                wall.PLACE,
                wall.YAW,
                payload=payload,
                shape=shape,
                scene=wall.scene(),
            )
        )
    return [c for c in out if only is None or c.name in only]


def cost_model(planner, case, scene):
    """VP-STO's cost with the CBS planner's own effective limits, scene and payload."""
    cfg, lim = planner.config, planner.limits.relaxed_to(case.start.q_a)
    big = 1e9
    return CraneVpsto(
        cfg.kappa * lim.dq_max,
        cfg.kappa * np.asarray(cfg.ddq_a_max),
        np.where(np.isfinite(lim.lower), lim.lower, -big),
        np.where(np.isfinite(lim.upper), lim.upper, big),
        cfg.kappa * lim.flow_max,
        case.start.q_tool,
        scene,
        case.shape,
    )


class Evaluator:
    """Replay + dense check + clearance, identical for both planners."""

    def __init__(self, planner, pump):
        self.planner, self.replay, self.pump = planner, Replay(), pump

    def dense(self, model, sampler, T):
        """1 kHz: utilisation of the kappa-scaled bounds, W-SDF collision, joint box."""
        cfg, lim = self.planner.config, self.planner.limits
        q, dq, ddq = sampler(np.linspace(0.0, T, int(np.ceil(T / 1e-3)) + 1))
        flow = np.asarray(self.pump.map(len(q))(q.T, dq.T)).ravel()
        ddq_bound = cfg.kappa * np.asarray(cfg.ddq_a_max)
        util = (
            float(np.max(np.abs(dq) / (cfg.kappa * lim.dq_max))),
            float(np.max(np.abs(ddq) / ddq_bound)),
            float(np.max(np.abs(flow)) / (cfg.kappa * lim.flow_max)),
        )
        col = (
            np.array([model.collision(x) for x in q])
            if model.has_scene
            else np.zeros(1)
        )
        return dict(
            util_vel=util[0],
            util_acc=util[1],
            util_pump=util[2],
            util_over=int(max(util) > 1.0 + 1e-3),
            acc0=float(np.max(np.abs(ddq[0]) / ddq_bound)),
            accT=float(np.max(np.abs(ddq[-1]) / ddq_bound)),
            col_cost=float(col.max()),
            box_viol=float(max(np.max(model.lower - q), np.max(q - model.upper), 0.0)),
        )

    def clearance(self, case, rep, scene, stride=2):
        """Min scene distance on the replay grid (100 Hz): at equilibrium, at replayed sway."""
        if not scene and case.shape is None:
            return np.nan, np.nan
        model, best = self.planner.model, [np.inf, np.inf]
        for i in range(0, len(rep["t"]), stride):
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

    def row(self, case, scene, model, sampler, T):
        params = np.concatenate([[case.start.q_tool], case.vector])
        rep = self.replay.run(sampler, T, case.start.q_u, params, case.offset)
        clr = self.clearance(case, rep, scene)
        return dict(
            duration_s=T,
            peak_sway_deg=np.degrees(rep["peak_sway"]),
            residual_sway_deg=np.degrees(rep["residual_sway"]),
            terminal_sway_deg=np.degrees(rep["terminal_sway"]),
            settled_at_T=int(rep["settled_at_T"]),
            settle_s=rep["settle_s"],
            clearance_eq_m=clr[0],
            clearance_sway_m=clr[1],
            **self.dense(model, sampler, T),
        )


def clean(row):
    return int(row["col_cost"] == 0.0 and row["box_viol"] <= 1e-6)


def run_cbs(planner, label, threads, ev, case, scene, model, repeats):
    rows, plan = [], None
    for repeat in range(repeats):
        record = defaultdict(float)
        set_threads(threads)  # VP-STO's ctor changes libgomp's count for acados too
        try:
            with bp.timed(record):
                began = time.perf_counter()
                candidate = case.plan(planner)
                wall_s = time.perf_counter() - began
        except PlanningError as refusal:
            rows.append(
                dict(
                    planner=label,
                    threads=threads,
                    run=repeat,
                    success=0,
                    note=str(refusal)[:100],
                )
            )
            continue
        plan = plan or candidate
        geom = sum(record[k] for k in ("prepare_scene", "geometry", "ik", "path"))
        rows.append(
            dict(
                planner=label,
                threads=threads,
                run=repeat,
                success=1,
                plan_s=wall_s,
                iterations=candidate.timing.iterations,
                t_ocp_s=record["ocp"],
                t_acados_s=candidate.timing.acados_time_s,
                t_geom_s=geom,
                t_rest_s=wall_s - record["ocp"] - geom,
                ms_per_iter=1e3
                * candidate.timing.acados_time_s
                / max(candidate.timing.iterations, 1),
            )
        )
    if plan is not None:
        timing = plan.timing
        metrics = ev.row(
            case, scene, model, lambda t: actuated_samples(timing, t), timing.time[-1]
        )
        for row in rows:
            if row["success"]:
                row.update(metrics, success=clean(metrics))
    return rows, plan


def run_vpsto(ev, case, scene, model, q_goal, runs, threads, n_via, sway=False):
    q0, rows = case.start.q_a, []
    for run in range(runs):
        began = time.perf_counter()
        sol = model.solve(
            q0, q_goal, n_via, 100 if sway else N_EVAL, N_SAMPLES, threads
        )
        row = dict(
            planner=f"{'ecb25a' if sway else 'vpsto'}_t{threads}",  # EcB25a: nEval 100
            threads=threads,
            run=run,
            plan_s=time.perf_counter() - began,
            solver_ms=sol["ms"],
            cost=sol["cost"],
        )
        T = sol["T"]
        _, q, dq, ddq = model.sample(q0, q_goal, sol["via"], T, int(np.ceil(T / 1e-3)))
        grid = np.linspace(0, T, len(q))

        def sampler(t, q=q, dq=dq, ddq=ddq, grid=grid):
            return tuple(
                np.stack([np.interp(t, grid, a[:, i]) for i in range(5)], 1)
                for a in (q, dq, ddq)
            )

        row.update(ev.row(case, scene, model, sampler, T))
        row["success"] = int(
            (sway or sol["cost"] - T < 1e-6) and clean(row)
        )  # no penalty (sway cost is not one), dense clean
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------- summary

NOTES = """
## What each planner enforces

| | CBS (`crane_planning`) | VP-STO (EcB25 cost, ros2_vpsto) |
|---|---|---|
| rate / accel | kappa-scaled dq, ddq rows in the OCP | same bounds, at 80 eval points only |
| pump | flow row, soft, refused if bought | Q(q,q')/Q_max time bound at eval points |
| command | u_c3 domain (Psi range), jerk-bounded C4 command | none |
| sway | 0.2 rad box at every node; settled terminal box 0.02 rad / 0.04 rad/s | none (free end accelerations) |
| collision | fixed TCP corridor, margins 0.05 + 0.10 + swing envelope, certificate, self-collision | W-SDF (sd < 0 only) vs scene, boom down, at passive equilibrium, 80 eval points; no self-collision |
| joint box | planner box | penalty 1e2 (1 + violation) |
| time | min duration 2 s, weighted LSQ + time cost | pure min time |
| goal | TCP pose, IK inside plan time | goal joints given (no IK in its time) |

VP-STO settings: ros2_vpsto as is (full-covariance ccmaes, sigma0 0.5 vs paper 1, stopTolFunHist 1e-2
vs old sepaCMA ftol/xtol 1e-2, time-based seed: passed as 0, not settable without editing Marc's
code); nVia and threads on the command line (default 8 = 6 free via points; old server nVia 8).

Columns: plan s = wall of the whole call (CBS Planner.plan; VP-STO vc_solve incl. basis setup;
one-off warm-up plan not counted). CBS split: ocp = solver phase, geom = scene + geometry + IK +
corridor/certificate, rest = remainder. Sway/settle from the shared replay (start at the settled
hanging pose, actuators held after T): peak = max during the move, residual = max over
[T, T+20 s], settle = time after T to enter and stay in the 0.02 rad / 0.04 rad/s box (">20" never).
acc0/accT = max |ddq| / bound at t = 0 / T. Success = (VP-STO: solver cost has no penalty and) the
1 kHz check finds no penetration and no joint-box violation; util > 1 (+1e-3) is listed apart
(over-limit column). Clearance: 100 Hz grid, scene bodies only (the 21 bench moves have none).
`example_wall_body` carries the 0.9x0.6x0.6 block (CoM as bench_carry) for CBS, VP-STO and the
dynamics; `example_wall_nobody` is carry_move as is (no carried body collision-checked).
`cbs_min0.5` = CBS with ocp_duration_min 0.5 s (separates the 2 s floor from sway effects).
Thread counts are per row (`planner` suffix _aN = acados threads, `vpsto_tN` = VP-STO threads).
Rows with load flag (1-min load > 1.5 at move start) have unreliable timings.
lim_boom_high / lim_arm_out: CBS refuses (self-collision); VP-STO has no self-collision term;
rows kept, excluded from the aggregate.

## What this supports

Supported: open loop, a min-time spline with free end accelerations leaves large residual sway;
a timing OCP with sway as state ends settled without a downstream damper.
Not supported: system-level sway of EcB25 (VP-STO + iLQR), "CBS is X times slower than EcB25".
"""


def med(rows, key):
    values = [r[key] for r in rows if key in r and not np.isnan(r[key])]
    if not values:
        return "-"
    value = np.median(values)
    return ">20" if np.isinf(value) else f"{value:.2f}"


def summarise(rows, path, header):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["move"], r["planner"])].append(r)
    main = [
        "| move | planner | n | success | plan s med / p95 | duration s | peak sway | "
        "residual sway med / max | settled at T | settle s | acc0 / accT | over limit | "
        "min clearance eq / sway | load flag |",
        "|" + "---|" * 14,
    ]
    split = [
        "| move | planner | acados threads | ocp s | geom s | rest s | iterations | ms / iteration |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (move, name), rs in groups.items():
        ok = [r for r in rs if r["success"]]
        wall_s = [r["plan_s"] for r in rs if "plan_s" in r]
        plan = (
            f"{np.median(wall_s):.2f} / {np.percentile(wall_s, 95):.2f}"
            if wall_s
            else "-"
        )
        clr = [
            (r["clearance_eq_m"], r["clearance_sway_m"])
            for r in ok
            if not np.isnan(r.get("clearance_eq_m", np.nan))
        ]
        clearance = (
            f"{min(c[0] for c in clr):.2f} / {min(c[1] for c in clr):.2f}"
            if clr
            else "-"
        )
        flagged = sum(r.get("load_flag", 0) for r in rs)
        done = [r for r in rs if "duration_s" in r]
        residual_max = (
            f"{max(r['residual_sway_deg'] for r in done):.2f}" if done else "-"
        )
        label = f"{move}" + (" (excluded)" if move in EXCLUDED else "")
        main.append(
            f"| {label} | {name} | {len(rs)} | {len(ok)}/{len(rs)} | {plan} | "
            f"{med(done, 'duration_s')} | {med(done, 'peak_sway_deg')} | "
            f"{med(done, 'residual_sway_deg')} / {residual_max} | "
            f"{sum(r['settled_at_T'] for r in done)}/{len(done)} | {med(done, 'settle_s')} | "
            f"{med(done, 'acc0')} / {med(done, 'accT')} | {sum(r['util_over'] for r in done)} | "
            f"{clearance} | {flagged} |"
        )
        if name.startswith("cbs") and wall_s:
            split.append(
                f"| {move} | {name} | {rs[0]['threads']} | {med(rs, 't_ocp_s')} | "
                f"{med(rs, 't_geom_s')} | {med(rs, 't_rest_s')} | {med(rs, 'iterations')} | "
                f"{med(rs, 'ms_per_iter')} |"
            )
    kept = [r for r in rows if r["move"] not in EXCLUDED and "duration_s" in r]
    aggregate = [
        "| planner | runs | settled at T | median residual sway deg | median duration s "
        "| median plan s |",
        "|---|---|---|---|---|---|",
    ]
    for name in dict.fromkeys(r["planner"] for r in kept):
        rs = [r for r in kept if r["planner"] == name]
        aggregate.append(
            f"| {name} | {len(rs)} | {sum(r['settled_at_T'] for r in rs)}/{len(rs)} | "
            f"{med(rs, 'residual_sway_deg')} | {med(rs, 'duration_s')} | {med(rs, 'plan_s')} |"
        )
    text = [
        header,
        "",
        "## Aggregate (without the excluded moves)",
        "",
        *aggregate,
        "",
        "## Per move",
        "",
        *main,
        "",
        "## CBS time split",
        "",
        *split,
        NOTES,
    ]
    path.write_text("\n".join(text))


def cpu() -> str:
    model = next(
        (l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l),
        "?",
    )
    return f"{model}, nproc {os.cpu_count()}"


def parse_cpus(text):
    out = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.update(range(int(low), int(high or low) + 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=int, default=20, help="VP-STO runs per thread setting"
    )
    parser.add_argument("--repeats-cbs", type=int, default=20)
    parser.add_argument("--threads", default="1,8", help="VP-STO OpenMP threads, list")
    parser.add_argument(
        "--acados-threads", default="1", help="CBS acados OpenMP threads, list"
    )
    parser.add_argument("--n-via", type=int, default=8)
    parser.add_argument("--cpus", default=None, help="pin the process, e.g. 0-7")
    parser.add_argument("--only", default=None, help="comma list of move names")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument(
        "--cooldown", type=float, default=60.0, help="max s to wait for load"
    )
    parser.add_argument(
        "--force", action="store_true", help="start although load > 1.0"
    )
    parser.add_argument("--out", default="runs.csv")
    parser.add_argument(
        "--sway",
        default=None,
        help="EcB25a sway weights w_vI,w_qT,w_vT (paper: 1,10,1)",
    )
    options = parser.parse_args()

    if options.cpus:
        os.sched_setaffinity(0, parse_cpus(options.cpus))
    if os.getloadavg()[0] > 1.0 and not options.force:
        raise SystemExit(
            f"load {os.getloadavg()[0]:.1f} > 1.0: not idle (--force to run anyway)"
        )
    thread_list = [int(t) for t in options.threads.split(",")]
    acados_list = [int(t) for t in options.acados_threads.split(",")]
    built = {"cbs": Planner(description(), PlannerConfig())}
    if not options.no_ablation:
        built["cbs_min0.5"] = Planner(
            description(), PlannerConfig(ocp_duration_min=0.5)
        )
    planner = built["cbs"]
    planners = {
        f"{name}_a{n}": (each, n) for name, each in built.items() for n in acados_list
    }
    ev = Evaluator(planner, build(description())[0])
    RESULTS.mkdir(exist_ok=True)
    todo = cases(planner, options.only and options.only.split(","))
    for each in built.values():  # discarded: first call loads the solver lazily
        bp.run(each, bp.MOVES[0], 1.0)

    all_rows, goals = [], {}
    for case in todo:
        scene = planner.prepare_scene(case.scene, True)
        model = cost_model(planner, case, scene)
        model.has_scene = bool(scene)
        waited = time.time()
        while (
            os.getloadavg()[0] > LOAD_FLAG and time.time() - waited < options.cooldown
        ):
            time.sleep(2.0)
        load = os.getloadavg()[0]
        rows, goal = [], None
        for label, (each, n) in planners.items():
            part, plan = run_cbs(
                each, label, n, ev, case, scene, model, options.repeats_cbs
            )
            rows += part
            goal = goal if goal is not None else (plan.timing.q_a[-1] if plan else None)
        goal = (
            goal
            if goal is not None
            else goals.get(case.name.replace("_body", "_nobody"), case.goal_q)
        )
        if goal is not None:
            goals[case.name] = goal
        if options.sway and goal is not None:
            q_end = passive_equilibrium(goal) + (
                case.offset(goal) if case.offset else 0.0
            )
            params = np.concatenate([[case.start.q_tool], case.vector])
            weights = [float(w) for w in options.sway.split(",")]
            model.set_sway(weights, case.start.q_u, q_end, params)
        for threads in thread_list:
            rows += run_vpsto(
                ev,
                case,
                scene,
                model,
                goal,
                options.runs,
                threads,
                options.n_via,
                bool(options.sway),
            )
        for r in rows:
            r.update(move=case.name, load=load, load_flag=int(load > LOAD_FLAG))
        all_rows += rows
        line = ", ".join(
            f"{name} T={np.median([r['duration_s'] for r in rs if 'duration_s' in r] or [np.nan]):.2f}"
            f" {sum(r['success'] for r in rs)}/{len(rs)}"
            for name in dict.fromkeys(r["planner"] for r in rows)
            for rs in [[r for r in rows if r["planner"] == name]]
        )
        print(f"load {load:4.1f} {case.name:<15} {line}", flush=True)
    with open(RESULTS / options.out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["move", *FIELDS[1:]], extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(all_rows)
    header = (
        f"# CBS vs VP-STO (generated {time.strftime('%Y-%m-%d')})\n\nCPU: {cpu()}, cpus "
        f"{options.cpus or 'unpinned'}; VP-STO OpenMP threads {thread_list}, nVia {options.n_via}, "
        f"nEval {N_EVAL}, population {N_SAMPLES}, {options.runs} runs/move/setting; CBS acados "
        f"OpenMP threads {acados_list} (set explicitly before every plan), "
        f"{options.repeats_cbs} repeats/move. {platform.platform()}"
    )
    summarise(
        all_rows,
        RESULTS
        / (
            "summary.md"
            if options.out == "runs.csv"
            else options.out.replace(".csv", ".md")
        ),
        header,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
