#!/usr/bin/env python3
"""
Codegen for the C++ VP-STO cost: pump flow and collision-link poses as casadi C.

vc_pump(q_a5, dq_a5) -> Q_P         axis flows summed, tool still (as the CBS OCP row)
vc_ddqu(x14, ddq_a5, p11) -> ddq_u   passive dynamics, as replay.py integrates them (EcB25a)
vc_frames(q_a5, q_tool) -> 12/body  R (row-major 9) + t (3) of each link primitive in K0,
                                    passive pair at the closed-form equilibrium;
                                    last body = K8_rotator_lower_part (payload mount)
Writes gen/vpsto_crane_gen.{c,h} and gen/link_table.json (shape, dims, W-SDF weight).
"""

import json

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import yaml
from common import GEN, description
from crane_model.collision import default_collision_model_path
from crane_model.symbolic import CraneSymbolicModel
from crane_planning.planner import PASSIVE_INDICES, PLANNED_INDICES, TOOL_INDEX

# EcB25 Eq 13-14 link weights (K5 inner telescope down incl. gripper: 1e2); links upstream of the boom (column, base, boxes) are absent.
WEIGHTS = {"K2_boom": 1e5, "K3_arm": 1e4, "K4_outer_telescope": 1e3}  # rest 1e2
UPSTREAM = {
    "K0_mounting_base",
    "K1_slewing_column",
    "control_box",
    "control_box_sensor_mount",
    "hydraulic_block",
}
MOUNT = "K8_rotator_lower_part"


def rotation(xyzw):
    quaternion = np.asarray(xyzw, dtype=float)
    return np.array(pin.Quaternion(quaternion[3], *quaternion[:3]).matrix())


def build(xml: str):
    model = CraneSymbolicModel(xml, "pzs100")
    inner = model.description.model
    cmodel = cpin.Model(inner)
    qa, dqa, qt = ca.SX.sym("qa", 5), ca.SX.sym("dqa", 5), ca.SX.sym("qt")

    # pump: the OCP's own flow row, state reduced to (q_a, dq_a)
    x = ca.vertcat(qa, ca.SX.zeros(2), dqa, ca.SX.zeros(2))
    symbols = ca.vertcat(model.x, model.u, model.p)
    values = ca.vertcat(x, ca.SX.zeros(5), ca.SX.zeros(11))
    flow = ca.sum1(ca.substitute(model.z[18:23], symbols, values))
    pump = ca.Function("vc_pump", [qa, dqa], [flow])
    ddqu = ca.Function("vc_ddqu", [model.x, model.u, model.p], [model.ddq_u])

    q = ca.SX.zeros(8)
    for axis, row in enumerate(PLANNED_INDICES):
        q[row] = qa[axis]
    q[PASSIVE_INDICES[0]] = 0.5 * np.pi - qa[1] - qa[2]
    q[PASSIVE_INDICES[1]] = 0.5 * np.pi
    q[TOOL_INDEX] = qt
    configuration, _ = model._expand(q, ca.SX.zeros(8))
    data = cmodel.createData()
    cpin.framesForwardKinematics(cmodel, data, configuration)
    base = data.oMf[cmodel.getFrameId("K0_mounting_base")]

    with open(default_collision_model_path(), encoding="utf-8") as handle:
        table = yaml.safe_load(handle)["primitives"]
    bodies, poses = [], []
    for entry in table:
        link = entry["link"]
        if entry["tool"] != "pzs100" or link in UPSTREAM or not inner.existFrame(link):
            continue
        extents = entry["extents_m"]
        capsule = entry["shape"] == "capsule"
        bodies.append(
            {
                "name": link,
                "shape": "capsule" if capsule else "box",
                # coal params: capsule (radius, length); box full sides
                "dims": [extents[0], 2 * extents[1]]
                if capsule
                else [2 * e for e in extents],
                "weight": WEIGHTS.get(link, 1e2),
            }
        )
        poses.append((link, rotation(entry["orientation_xyzw"]), entry["origin_m"]))
    poses.append((MOUNT, np.eye(3), [0.0, 0.0, 0.0]))

    out = []
    for link, offset_r, offset_t in poses:
        placed = base.actInv(data.oMf[cmodel.getFrameId(link)])
        rot = ca.mtimes(placed.rotation, ca.DM(offset_r))
        trans = ca.mtimes(placed.rotation, ca.DM(offset_t)) + placed.translation
        out += [ca.vec(rot.T), trans]
    frames = ca.Function("vc_frames", [qa, qt], [ca.vertcat(*out)])
    return pump, frames, bodies, ddqu


def main() -> None:
    pump, frames, bodies, ddqu = build(description())
    GEN.mkdir(exist_ok=True)
    generator = ca.CodeGenerator("vpsto_crane_gen", {"with_header": True})
    generator.add(pump)
    generator.add(frames)
    generator.add(ddqu)
    generator.generate(str(GEN) + "/")
    (GEN / "link_table.json").write_text(json.dumps({"bodies": bodies}, indent=1))
    print(
        f"{len(bodies)} bodies; pump {pump.n_nodes()} nodes, frames {frames.n_nodes()} nodes"
    )


if __name__ == "__main__":
    main()
