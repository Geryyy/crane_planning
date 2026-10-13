#!/usr/bin/env python3
"""Sanity checks, asserts only: generated C vs crane_model numerics, sampler vs solver, sim vs CBS."""

import json

import common  # noqa: F401  (puts the workspace packages on sys.path)

# isort: split
import bench_plan as bp
import casadi as ca
import numpy as np
import pinocchio as pin
from common import GEN, description
from crane_model import CraneModel, Frame, Tool
from crane_model.collision import LinkGeometry
from crane_model.symbolic import CraneSymbolicModel
from crane_planning import Planner, PlannerConfig, Start
from crane_planning.config import passive_equilibrium
from crane_planning.planner import PLANNED_INDICES, TOOL_INDEX, actuated_samples
from lib import CraneVpsto, casadi_call
from replay import Replay

rng = np.random.default_rng(0)
xml = description()
planner = Planner(xml, PlannerConfig())
body_table = LinkGeometry(planner.model._model, Tool.PZS100)


def configs(n):
    lo = np.where(np.isfinite(planner.limits.lower), planner.limits.lower, -1.0)
    hi = np.where(np.isfinite(planner.limits.upper), planner.limits.upper, 1.0)
    return [lo + rng.random(5) * (hi - lo) for _ in range(n)]


# (a) generated pump / frames vs the model they came from
sym = CraneSymbolicModel(xml, "pzs100")
flow = ca.Function("f", [sym.x, sym.u, sym.p], [ca.sum1(sym.z[18:23])])
mesh = CraneModel(xml, Tool.PZS100)
bodies = json.loads((GEN / "link_table.json").read_text())["bodies"]
names = [b[0] for b in body_table.bodies]
n_out = 12 * (len(bodies) + 1)
for q_a, dq in zip(configs(5), configs(5)):
    state = np.concatenate([q_a, passive_equilibrium(q_a), dq, np.zeros(2)])
    expected = float(flow(state, np.zeros(5), np.zeros(11)))
    got = casadi_call("vc_pump", q_a, dq, n_out=1)[0]
    assert abs(got - expected) < 1e-12 + 1e-9 * abs(expected), (got, expected)

    q = np.zeros(8)
    q[list(PLANNED_INDICES)], q[[4, 5]], q[TOOL_INDEX] = (
        q_a,
        passive_equilibrium(q_a),
        0.3,
    )
    frames = casadi_call("vc_frames", q_a, [0.3], n_out=n_out)
    data = mesh._model.createData()
    pin.forwardKinematics(mesh._model, data, mesh._description.configuration(q))
    base = mesh._model.getFrameId(Frame.MOUNTING_BASE.value)
    pin.updateFramePlacement(mesh._model, data, base)
    for i, entry in enumerate(bodies):
        _, frame, _, offset = body_table.bodies[names.index(entry["name"])]
        pin.updateFramePlacement(mesh._model, data, frame)
        pose = data.oMf[base].actInv(data.oMf[frame]) * offset
        assert np.allclose(
            frames[12 * i : 12 * i + 9].reshape(3, 3), pose.rotation, atol=1e-9
        )
        assert np.allclose(
            frames[12 * i + 9 : 12 * i + 12], pose.translation, atol=1e-9
        )
    k8 = mesh.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.ROTATOR_LOWER_PART)
    assert np.allclose(frames[-3:], k8.position_m, atol=1e-9)

# (b) VP-STO: dense sampler == solver's own path at eval points, bounds hold, collision cost bites
cfg, lim = planner.config, planner.limits
args = (
    cfg.kappa * lim.dq_max,
    cfg.kappa * np.asarray(cfg.ddq_a_max),
    lim.lower,
    lim.upper,
    cfg.kappa * lim.flow_max,
    0.3,
)
model = CraneVpsto(*args, [], None)
q0, qd = np.array(bp.MOVES[0][1]), np.array(bp.MOVES[0][2])
sol = model.solve(q0, qd)
_, q, dq, ddq = model.sample(q0, qd, sol["via"], sol["T"], 79)
assert np.allclose(q, sol["q"], atol=1e-9)
assert np.allclose(dq * sol["T"], sol["qp"], atol=1e-9)
assert np.allclose(ddq * sol["T"] ** 2, sol["qpp"], atol=1e-9)
assert abs(model.cost(q0, qd, sol["via"]) - sol["cost"]) < 1e-9
assert sol["cost"] == sol["T"], "slew move should be penalty-free"
assert np.all(np.abs(dq) <= args[0] * (1 + 1e-6)) and np.all(
    np.abs(ddq) <= args[1] * (1 + 1e-6)
)
body = casadi_call("vc_frames", q0, [0.3], n_out=n_out)[:12]  # first body
box = bp.CollisionPrimitive(
    "x",
    "box",
    pin.SE3(np.asarray(body[:9]).reshape(3, 3), body[9:]),
    np.array([0.1, 0.1, 0.1]),
)
hit = CraneVpsto(*args, [box], None).collision(q0)
assert hit >= 1e2, hit
far = bp.CollisionPrimitive(
    "y", "box", pin.SE3(np.eye(3), np.array([50.0, 0, 0])), np.ones(3)
)
assert CraneVpsto(*args, [far], None).collision(q0) == 0.0

# (c) replay of a CBS plan reproduces its own terminal sway and peak sway
a, b = bp.MOVES[2][1], bp.MOVES[2][2]
start = Start(q=bp.configuration(planner, a), dq_a=np.zeros(5))
position, yaw = bp.goal_pose(planner, b)
plan = planner.plan(start, position, yaw, scene=[], avoid_collisions=True)
timing = plan.timing
rep = Replay().run(
    lambda t: actuated_samples(timing, t),
    timing.time[-1],
    start.q_u,
    np.concatenate([[start.q_tool], np.zeros(10)]),
)
peak = np.abs(timing.q_u - timing.q_u_eq).max()
assert abs(rep["terminal_sway"] - timing.terminal_sway) < 0.005, (
    rep["terminal_sway"],
    timing.terminal_sway,
)
assert abs(rep["peak_sway"] - peak) < 0.1 * peak + 0.005, (rep["peak_sway"], peak)
print(
    f"check_model ok: replay terminal {np.degrees(rep['terminal_sway']):.2f} vs plan "
    f"{np.degrees(timing.terminal_sway):.2f} deg, peak {np.degrees(rep['peak_sway']):.2f} vs "
    f"{np.degrees(peak):.2f} deg"
)
