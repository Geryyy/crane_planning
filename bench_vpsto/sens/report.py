#!/usr/bin/env python3
"""Tables from sens/out/replay_*.json and mj_*.json."""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
CASES = "example_wall_body example_wall_nobody bench_carry slew lift ax_slew".split()
L_PEND = 2.0


def load(prefix):
    return {c: json.loads(p.read_text())[c] for c in CASES if (p := OUT / f"{prefix}_{c}.json").exists()}


def vp(d):
    return [v for k, v in d.items() if k.startswith("vp")]


def rng(v, key):
    a = np.array([x[key] for x in v])
    return f"{np.median(a):.1f} [{a.min():.0f}-{a.max():.0f}]"


def replay_table():
    data = load("replay")
    names = list(next(iter(data.values())).keys())
    names = [n for n in names if not n.startswith("_")]
    print("| case | variant | VP res med [min-max] | VP atT med | VP peak med | CBS res | CBS(dq-naive) res | VP disp m |")
    print("|---|---|---|---|---|---|---|---|")
    for c, r in data.items():
        for n in names:
            if n not in r:
                continue
            v = vp(r[n])
            cb, nv = r[n].get("cbs"), r[n].get("cbs_dqnaive")
            print(f"| {c} | {n} | {rng(v,'res')} | {np.median([x['atT'] for x in v]):.1f} | "
                  f"{np.median([x['peak'] for x in v]):.1f} | {cb['res']:.2f} | "
                  f"{(nv['res'] if nv else float('nan')):.2f} | {np.median([x['disp'] for x in v]):.2f} |")


def mj_table():
    data = load("mj")
    print("| case | JTC | VP res med [min-max] | VP atT med | CBS res | CBS atT | CBS miss rad |")
    print("|---|---|---|---|---|---|---|")
    for c, r in data.items():
        for rung, d in r.items():
            v = vp(d)
            print(f"| {c} | {rung} | {rng(v,'res')} | {np.median([x['atT'] for x in v]):.1f} | "
                  f"{d['cbs']['res']:.2f} | {d['cbs']['atT']:.2f} | {d['cbs']['miss']:.3f} |")


def spread():
    from scipy.stats import spearmanr

    data = load("replay")
    for c, r in data.items():
        res = [r["ideal"][f"vp{k}"] for k in range(5)]
        f = np.median([x["f_p"] for x in res])
        meas = []
        for k in range(5):
            a = np.array(r["_ddq"][f"vp{k}"])  # 10 ms grid of 1 kHz samples
            t = np.linspace(0, r["_plans"][f"vp{k}"]["T"], len(a))
            w = 2 * np.pi * f
            meas.append(float(sum(abs(np.trapz(a[:, i] * np.exp(-1j * w * t), t)) for i in range(5))))
        y = [x["res"] for x in res]
        T = [x["T"] for x in res]
        rho = spearmanr(meas, y)[0] if np.std(y) > 0 else float("nan")
        print(f"{c}: f_p {f:.2f} Hz, T {np.round(T,2)}, res {np.round(y,1)}, |F(f_p)| {np.round(meas,2)}, spearman {rho:.2f}")


if __name__ == "__main__":
    print("## replay variants"); replay_table()
    print("## mujoco open-loop"); mj_table()
    print("## spread"); spread()
