#!/usr/bin/env python3
"""
Open-loop execution of CBS and VP-STO plans through the wire_chain plant: JTC playback
(PI + feed-forward, 100 Hz) -> Psi (identity) -> C3 incl. 60 ms dead time -> MuJoCo. No MPC.

    nice python3 sens/mj.py --only slew --seeds 5 --out sens/mj_slew.json
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: F401,E402

# isort: split
WC = HERE.parents[2] / "crane_mpc" / "scripts" / "wire_chain"
sys.path.insert(0, str(WC))
import bench_vpsto as bv  # noqa: E402
import harness  # noqa: E402
from common import description  # noqa: E402
from crane_model.model import Payload  # noqa: E402
from crane_model.mujoco_plant import NX_RIGID, MujocoPlant  # noqa: E402
from crane_model.symbolic import PAYLOAD_MOUNT_LINK  # noqa: E402
from crane_mpc.horizon import _jtc_cubic  # noqa: E402
from crane_planning import Planner, PlannerConfig  # noqa: E402
from harness import ACTUATED, PASSIVE, PLANNED  # noqa: E402

H, HOLD, DT = 0.01, 20.0, 5e-4
OPT = dict(
    control_rate=None,
    pi_scale=1.0,
    psi_gain_positive=None,
    psi_gain_negative=None,
    dead_time=None,
    no_clamp=False,
)


def carry(plant, mass, com):
    if not mass:
        return
    mj, m = plant._mj, plant.model
    b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, PAYLOAD_MOUNT_LINK)
    m.body_mass[b], m.body_ipos[b] = mass, com
    m.body_iquat[b] = (1.0, 0.0, 0.0, 0.0)
    m.body_inertia[b] = 1e-4 * mass
    mj.mj_setConst(m, plant.data)
    plant.forward()


def sampler(t, p, v, e):
    def sample(x):
        if x >= t[-1]:
            return p[-1], v[-1], e[-1]
        i = np.searchsorted(t, x, side="right") - 1
        q, dq = _jtc_cubic(t[i], t[i + 1], p[i], v[i], p[i + 1], v[i + 1], x)
        return q, dq, e[i] + (e[i + 1] - e[i]) * (x - t[i]) / (t[i + 1] - t[i])

    return sample


def execute(case, planner, payload, traj, rung):
    t, p, v, e = traj
    T = t[-1]
    plant = MujocoPlant(description(), timestep=DT)
    carry(
        plant,
        case.payload.mass_kg if case.payload else 0.0,
        np.asarray(case.payload.center_of_mass_k8_m) if case.payload else np.zeros(3),
    )
    chain = harness.Chain(
        plant, case.start.q_tool, SimpleNamespace(pi_rung=rung, **OPT)
    )
    q0 = np.zeros(8)
    q0[ACTUATED] = np.append(p[0], case.start.q_tool)
    q0[PASSIVE] = planner.model.passive_equilibrium(q0[ACTUATED], payload)
    x = np.zeros(harness.cs.NX)
    x[:5] = p[0]
    x[5:7] = q0[PASSIVE]
    chain.seed(x[:NX_RIGID])
    sample = sampler(t, p, v, e)
    ff, n = chain.ff_scale, int(round((T + HOLD) / H))
    ts, sway, qa = [], [], []
    for k in range(n):
        now = k * H
        q_ref, dq_ref, _ = sample(now)
        _, dq_next, eff_next = sample(now + H)
        cmd = chain.loop.step(
            q_ref,
            dq_ref,
            plant.q[PLANNED],
            plant.dq[PLANNED],
            feedforward=ff * (dq_next - dq_ref) + eff_next,
        )
        plant.drive(chain.actuator, chain.psi.apply(cmd), H)
        if k % 5 == 0:
            ts.append(now + H)
            qa.append(plant.q[ACTUATED].copy())
            sway.append(plant.q[PASSIVE].copy())
    ts, sway, qa = np.array(ts), np.array(sway), np.array(qa)
    eq = np.array([planner.model.passive_equilibrium(q, payload) for q in qa])
    mag = np.abs(sway - eq).max(1)
    i = np.searchsorted(ts, T)
    return dict(
        res=float(np.degrees(mag[i:].max())),
        atT=float(np.degrees(mag[i])),
        peak=float(np.degrees(mag[:i].max())),
        T=float(T),
        miss=float(np.abs(plant.q[PLANNED][:4] - p[-1][:4]).max()),
        t=ts.round(3).tolist(),
        sway=np.degrees(mag).round(3).tolist(),
    )  # trace, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="slew")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rungs", default="feedforward,full")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    planner = Planner(description(), PlannerConfig())
    todo = bv.cases(planner, o.only.split(","))
    out = {}
    for case in todo:
        scene = planner.prepare_scene(case.scene, True)
        cm = bv.cost_model(planner, case, scene)
        cm.has_scene = bool(scene)
        plan = case.plan(planner)
        q_goal = plan.timing.q_a[-1]
        pay = case.payload
        payload = (
            Payload(pay.mass_kg, pay.center_of_mass_k8_m, valid=True)
            if pay
            else Payload(0.0, np.zeros(3), valid=True)
        )
        trajs = {
            "cbs": (
                plan.time,
                plan.q[:, PLANNED],
                plan.dq[:, PLANNED],
                plan.effort[:, PLANNED],
            )
        }
        for k in range(o.seeds):
            sol = cm.solve(case.start.q_a, q_goal, 8, bv.N_EVAL, bv.N_SAMPLES, 1)
            T = sol["T"]
            _, q, dq, _ = cm.sample(
                case.start.q_a, q_goal, sol["via"], T, int(round(T / 0.04))
            )
            trajs[f"vp{k}"] = (np.linspace(0, T, len(q)), q, dq, np.zeros_like(q))
        out[case.name] = {}
        for rung in o.rungs.split(","):
            for name, tr in trajs.items():
                r = execute(case, planner, payload, tr, rung)
                out[case.name].setdefault(rung, {})[name] = r
                print(
                    case.name,
                    rung,
                    name,
                    {a: round(b, 2) for a, b in r.items() if a not in ("t", "sway")},
                    flush=True,
                )
        Path(o.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
