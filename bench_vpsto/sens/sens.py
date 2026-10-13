#!/usr/bin/env python3
"""
Sensitivity of the replayed residual sway to the replay itself.

    nice python3 sens/sens.py --only slew,lift --seeds 5 --out sens/out_x.json

Engines: ideal (committed replay), lp (2nd-order low-pass of ddq_a), c3 (dead time + lag + force
state, optional JTC velocity loop). Perturbations: passive damping/Coulomb, payload inertia.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: F401,E402

# isort: split
import bench_vpsto as bv  # noqa: E402
import casadi as ca  # noqa: E402
import crane_model.symbolic as S  # noqa: E402
from common import description  # noqa: E402
from crane_model.conventions import canonical_joints  # noqa: E402
from crane_model.velocity_loop import VelocityLoop, load_velocity_loop  # noqa: E402
from crane_planning import Planner, PlannerConfig  # noqa: E402
from crane_planning.config import passive_equilibrium  # noqa: E402
from crane_planning.planner import actuated_samples  # noqa: E402
from export_model import build  # noqa: E402

DT, HOLD, TICK, PASS = 5e-3, 20.0, 2, (4, 5)
_parse = S.parse
_models = {}


def model(damp=1.0, fric=1.0, c3=False):
    """CraneSymbolicModel with passive-pair damping / Coulomb scaled (parse patched)."""
    key = (damp, fric, c3)
    if key not in _models:

        def parse(*a, **k):
            d = _parse(*a, **k)
            for i in PASS:
                d.damping[i] *= damp
                d.friction[i] *= fric
            return d

        S.parse = parse
        m = S.CraneSymbolicModel(
            description(), "pzs100", actuator=S.ACTUATOR_FIT if c3 else None
        )
        S.parse = _parse
        if c3:
            fn = (
                ca.Function("f", [m.x, m.u, m.p], [m.xdot]),
                ca.Function("h", [m.x, m.u, m.p], [m.h_eff]),
            )
        else:
            fn = ca.Function("f", [m.x, m.u, m.p], [m.ddq_u])
        _models[key] = (m, fn)
    return _models[key]


def tcp_function():
    m, _ = model()
    q = ca.SX.sym("q", 8)
    return ca.Function("tcp", [q], [m.tool_position(q)])


def block_inertia(dims, mass):
    a, b, c = dims
    return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def with_inertia(params, dims):
    p = params.copy()
    p[1 + 4 :] = 0.0
    for k, v in zip((0, 3, 5), block_inertia(dims, p[1])):  # entries xx, yy, zz
        p[5 + k] = v
    return p


class Plan:
    """A plan as the replay needs it: sampler(t) -> q,dq,ddq and optional C3 command."""

    def __init__(self, sampler, T, command=None):
        self.sampler, self.T, self.command = sampler, T, command


def simulate(case, plan, params, engine, damp=1.0, fric=1.0, fc=None, ff="dq", loop=False):
    """
    engine: ideal | lp (fc Hz) | c3.  c3 ff: dq (naive, delayed 60 ms), dq_pre (dead time
    previewed), u (plan.command previewed).  loop: JTC velocity PI (gains from velocity_loop).
    Returns sway history relative to the equilibrium of the *actual* q_a.
    """
    T, n_hold = plan.T, int(HOLD / DT)
    N = int(np.ceil(T / DT)) + n_hold
    half = np.arange(2 * N + 1) * DT / 2

    def ref(t):
        tc = np.minimum(t, T)
        q, dq, ddq = plan.sampler(tc)
        dq, ddq = dq * (t <= T)[:, None], ddq * (t <= T)[:, None]
        return q, dq, ddq

    qr, dqr, ddqr = ref(half)
    qu0 = case.start.q_u
    s_out = np.zeros((N + 1, 4 + 5))  # q_u(2), dq_u(2), q_a(5)
    m, fn = model(damp, fric, engine == "c3")
    if engine in ("ideal", "lp"):
        wn = 2 * np.pi * (fc or 1.0)
        # state: q_u2 dq_u2 q_a5 dq_a5 a5 da5
        s = np.concatenate([qu0, np.zeros(2), qr[0], np.zeros(5), np.zeros(10)])

        def rhs(s, i):
            if engine == "ideal":
                qa, va, aa = qr[i], dqr[i], ddqr[i]
                x = np.concatenate([qa, s[:2], va, s[2:4]])
                ddu = np.asarray(fn(x, aa, params)).ravel()
                return np.concatenate([s[2:4], ddu, va * 0, va * 0, np.zeros(10)])
            qa, va, a, da = s[4:9], s[9:14], s[14:19], s[19:24]
            x = np.concatenate([qa, s[:2], va, s[2:4]])
            ddu = np.asarray(fn(x, a, params)).ravel()
            dd = wn * wn * (ddqr[i] - a) - 2 * wn * da
            return np.concatenate([s[2:4], ddu, va, a, da, dd])

        for k in range(N + 1):
            if engine == "ideal":
                s[4:9] = qr[2 * k]
            s_out[k] = np.concatenate([s[:4], s[4:9]])
            if k == N:
                break
            k1 = rhs(s, 2 * k)
            k2 = rhs(s + DT / 2 * k1, 2 * k + 1)
            k3 = rhs(s + DT / 2 * k2, 2 * k + 1)
            k4 = rhs(s + DT * k3, 2 * k + 2)
            s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if engine == "ideal":
                s[4:9] = qr[2 * k + 2]
    else:
        f, h = fn
        x = np.zeros(S.NX)
        x[:5], x[5:7] = qr[0], qu0
        x[S.X_ACTUATED_FORCE : S.X_ACTUATED_FORCE + 5] = np.asarray(
            h(x, np.zeros(S.NU_PROGRESS), params)
        ).ravel()
        names = canonical_joints()
        gains, _ = load_velocity_loop()
        vl = VelocityLoop([gains[names[i]] for i in S.ACTUATED_INDICES[:5]], TICK * DT)
        if not loop:  # feedforward only
            vl._p[:] = vl._d[:] = 0.0
        pre = S.ACTUATOR_FIT.dead_time_s
        queue = [np.zeros(5)] * int(round(pre / (TICK * DT)))
        uc = np.zeros(S.NU_PROGRESS)
        for k in range(N + 1):
            if k % TICK == 0:
                t = k * DT
                q_r, dq_r, _ = ref(np.array([t]))
                if ff == "u":
                    feed = plan.command(min(t + pre, T)) - dq_r[0]
                elif ff == "dq_pre":
                    feed = ref(np.array([t + pre]))[1][0] - dq_r[0]
                else:
                    feed = np.zeros(5)
                cmd = vl.step(q_r[0], dq_r[0], x[:5], x[7:12], feed)
                queue.append(cmd)
                uc[:5] = queue.pop(0)
            s_out[k] = np.concatenate([x[5:7], x[12:14], x[:5]])
            if k == N:
                break
            k1 = np.asarray(f(x, uc, params)).ravel()
            k2 = np.asarray(f(x + DT / 2 * k1, uc, params)).ravel()
            k3 = np.asarray(f(x + DT / 2 * k2, uc, params)).ravel()
            k4 = np.asarray(f(x + DT * k3, uc, params)).ravel()
            x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s_out


_tcp = None


def metrics(case, plan, s_out):
    global _tcp
    _tcp = _tcp or tcp_function()
    t = np.arange(len(s_out)) * DT
    ia = int(np.ceil(plan.T / DT))
    keep = np.arange(0, len(t), 4)
    offs = {}
    stride = max(1, ia // 40)
    idx = list(range(0, ia + 1, stride)) + [len(t) - 1]
    if case.offset is not None:
        sh = np.array([case.offset(s_out[i, 4:]) for i in idx])
        off = np.stack([np.interp(t, t[idx], sh[:, c]) for c in range(2)], 1)
    else:
        off = 0.0
    eq = np.array([passive_equilibrium(a) for a in s_out[:, 4:]]) + off
    sway = s_out[:, :2] - eq
    mag = np.abs(sway).max(1)
    res, atT, peak = mag[ia:].max(), np.interp(plan.T, t, mag), mag[: ia + 1].max()
    # horizontal TCP displacement vs hanging at equilibrium, tail only
    tail = np.arange(ia, len(t), 20)
    d = 0.0
    for i in tail:
        q = np.zeros(8)
        q[[0, 1, 2, 3, 6]] = s_out[i, 4:]
        q[7] = case.start.q_tool
        q[4:6] = s_out[i, :2]
        a = np.asarray(_tcp(q)).ravel()
        q[4:6] = eq[i]
        b = np.asarray(_tcp(q)).ravel()
        d = max(d, float(np.hypot(*(a - b)[:2])))
    # dominant tail frequency (pendulum), for the spread analysis
    seg = sway[ia : ia + int(10 / DT), 0] - sway[ia : ia + int(10 / DT), 0].mean()
    fs = np.fft.rfftfreq(len(seg), DT)
    f_p = float(fs[1:][np.argmax(np.abs(np.fft.rfft(seg))[1:])]) if seg.std() > 1e-6 else 0.0
    return dict(
        res=float(np.degrees(res)),
        atT=float(np.degrees(atT)),
        peak=float(np.degrees(peak)),
        disp=d,
        f_p=f_p,
    )


VARIANTS = [  # name, kwargs for simulate, inertia?
    ("ideal", dict(engine="ideal"), False),
    ("ideal_d0.5", dict(engine="ideal", damp=0.5), False),
    ("ideal_d1.5", dict(engine="ideal", damp=1.5), False),
    ("ideal_f0", dict(engine="ideal", fric=0.0), False),
    ("ideal_f2", dict(engine="ideal", fric=2.0), False),
    ("ideal_inertia", dict(engine="ideal"), True),
    ("lp2Hz", dict(engine="lp", fc=2.0), False),
    ("lp4Hz", dict(engine="lp", fc=4.0), False),
    ("c3_ff", dict(engine="c3", ff="dq"), False),
    ("c3_ff_pre", dict(engine="c3", ff="dq_pre"), False),
    ("c3_loop", dict(engine="c3", ff="dq", loop=True), False),
    ("c3_loop_pre", dict(engine="c3", ff="dq_pre", loop=True), False),
    ("c3_loop_pre_d0.5", dict(engine="c3", ff="dq_pre", loop=True, damp=0.5), False),
    ("c3_loop_pre_d1.5", dict(engine="c3", ff="dq_pre", loop=True, damp=1.5), False),
    ("c3_loop_pre_f0", dict(engine="c3", ff="dq_pre", loop=True, fric=0.0), False),
    ("c3_loop_pre_f2", dict(engine="c3", ff="dq_pre", loop=True, fric=2.0), False),
    ("c3_loop_pre_inertia", dict(engine="c3", ff="dq_pre", loop=True), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="example_wall_body")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    planner = Planner(description(), PlannerConfig())
    ev = bv.Evaluator(planner, build(description())[0])
    todo = bv.cases(planner, o.only.split(","))
    out = {}
    for case in todo:
        scene = planner.prepare_scene(case.scene, True)
        cm = bv.cost_model(planner, case, scene)
        cm.has_scene = bool(scene)
        cbs = case.plan(planner)
        timing = cbs.timing
        q_goal = timing.q_a[-1]
        plans = {
            "cbs": Plan(
                lambda t, tm=timing: actuated_samples(tm, np.atleast_1d(t)),
                timing.time[-1],
                lambda t, tm=timing: np.array(
                    [np.interp(t, tm.time, tm.command[:, i]) for i in range(5)]
                ),
            )
        }
        for k in range(o.seeds):
            sol = cm.solve(case.start.q_a, q_goal, 8, bv.N_EVAL, bv.N_SAMPLES, 1)
            T = sol["T"]
            _, q, dq, ddq = cm.sample(case.start.q_a, q_goal, sol["via"], T, int(np.ceil(T / 1e-3)))
            grid = np.linspace(0, T, len(q))
            plans[f"vp{k}"] = Plan(
                lambda t, q=q, dq=dq, ddq=ddq, grid=grid: tuple(
                    np.stack([np.interp(np.atleast_1d(t), grid, a[:, i]) for i in range(5)], 1)
                    for a in (q, dq, ddq)
                ),
                T,
            )
            plans[f"vp{k}"].ddq = ddq
            plans[f"vp{k}"].via = sol["via"].tolist()
        params = np.concatenate([[case.start.q_tool], case.vector])
        has_payload = case.payload is not None
        res = {}
        for name, kw, inertia in VARIANTS:
            if o.variants and name not in o.variants.split(","):
                continue
            if inertia and not has_payload:
                continue
            p = with_inertia(params, bv.wall.BLOCK) if inertia else params
            for pname, plan in plans.items():
                runs = [("u", "cbs_u")] if pname == "cbs" and kw["engine"] == "c3" else []
                modes = [(kw, pname)]
                if pname == "cbs" and kw["engine"] == "c3":
                    if kw.get("ff") == "dq_pre":
                        modes = [(dict(kw, ff="u"), "cbs")]  # shipped law
                        modes.append((dict(kw, ff="dq"), "cbs_dqnaive"))
                    elif kw.get("ff") == "dq":
                        modes = [(dict(kw, ff="u"), "cbs")]
                for kw2, label in modes:
                    t0 = time.time()
                    s = simulate(case, plan, p, **kw2)
                    res.setdefault(name, {})[label] = metrics(case, plan, s)
                    res[name][label]["T"] = plan.T
                    print(case.name, name, label, f"{res[name][label]['res']:.2f}", f"{time.time()-t0:.1f}s", flush=True)
        res["_plans"] = {
            k: dict(T=v.T, via=getattr(v, "via", None)) for k, v in plans.items()
        }
        res["_ddq"] = {k: np.asarray(v.ddq)[::10].tolist() for k, v in plans.items() if hasattr(v, "ddq")}
        out[case.name] = res
        Path(o.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
