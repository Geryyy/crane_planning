#!/usr/bin/env python3
"""
Where a `Planner.plan` call spends its time. No ROS, no graph, no clock.

Answers what bench_calc_movement.py can't -- which stage costs what, how much
OCP time is acados vs Python.

    ./scripts/bench_plan.py --repeats 5

Wall clock is load-bound (ocp.py: 0.53 s idle vs 4.97 s under running Gazebo):
run idle, read regressions off sqp_iter and acados time, not totals.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import pinocchio as pin  # noqa: E402
from crane_model import CollisionPrimitive, Frame, Payload  # noqa: E402

from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402
from crane_planning import geometry as geometry_module  # noqa: E402
from crane_planning import planner as planner_module  # noqa: E402
from crane_planning import weights as crane_weights  # noqa: E402
from crane_planning.planner import (  # noqa: E402
    ACTUATED_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    yaw_of,
)

DESCRIPTION = "pzs100.urdf"

# Servers disagree on joint limits; a request both answer lives in their
# intersection, smaller than either. Per planned axis (slew,boom,arm,tele,rot):
#     crane_planning (PZS100 description):
#         [-3.71,3.71] [-1.20,1.563] [-0.91,4.60] [0.00,2.236]  unbounded
#     a2b_ilqr_server qMinCtrl/qMaxCtrl minus 0.087 rad qLimSafetyBuffer:
#         [-3.71,3.71] [0.00,1.562] [-0.91,1.325] [0.00,2.236] [-12.6,12.6]
# Boom binds (a2b_ilqr_server refuses negative boom), so these moves keep boom
# positive -- not plan_example's moves, not comparable.

#: Tool coordinate q8. crane_planning bounds it [0.20,0.70], mp_parameter_pzs100
#: [0.00,0.538]; 0.45 inside both, room for safety buffer.
TOOL_POSITION = 0.45

#: (name, start, goal) in planned joint coords, inside the intersection above.
#: Goal is joint config pushed through FK, so the IK stage always has an answer.
MOVES = (
    ("slew", (0.0, 0.4, 0.6, 1.0, 0.0), (0.9, 0.4, 0.6, 1.0, 0.0)),
    ("telescope", (0.0, 0.4, 0.6, 0.8, 0.0), (0.0, 0.4, 0.6, 1.6, 0.0)),
    ("lift", (0.0, 0.4, 0.6, 1.0, 0.0), (0.6, 0.7, 0.9, 1.4, 0.4)),
    ("stow", (0.6, 0.7, 0.9, 1.4, 0.4), (0.0, 0.25, 0.35, 0.7, 0.0)),
    ("across", (-0.7, 0.3, 0.4, 0.9, -0.3), (0.8, 0.8, 1.0, 1.5, 0.5)),
)

#: Pose every wide move leaves from/arrives at, interior to the intersection,
#: so a move off it is about its named axis, not a limit.
BASE = (0.0, 0.5, 0.6, 1.0, 0.0)

#: Spread MOVES lacks: those are long multi-axis moves where a C4 jerk bound
#: costs least (measured: reaches only 0.75 of itself on the worst). Regimes:
#: single axis (one axis decides duration); short 0.05-0.3 rad (jerk bound
#: actually binds, feedforward may ask for command it lacks); near limit
#: (shared box, boom floor is a2b_ilqr_server's); long (MOVES itself, anchor).
WIDE_MOVES = (
    # single axis, from one pose
    ("ax_slew", BASE, (2.5, 0.5, 0.6, 1.0, 0.0)),
    ("ax_boom", BASE, (0.0, 1.2, 0.6, 1.0, 0.0)),
    ("ax_arm", BASE, (0.0, 0.5, 1.2, 1.0, 0.0)),
    ("ax_tele", BASE, (0.0, 0.5, 0.6, 2.0, 0.0)),
    ("ax_rot", BASE, (0.0, 0.5, 0.6, 1.0, 1.5)),
    # short -- regime the jerk bound is for
    ("sh_slew", BASE, (0.1, 0.5, 0.6, 1.0, 0.0)),
    ("sh_boom", BASE, (0.0, 0.55, 0.6, 1.0, 0.0)),
    ("sh_tele", BASE, (0.0, 0.5, 0.6, 1.1, 0.0)),
    ("sh_multi", BASE, (0.1, 0.55, 0.65, 1.1, 0.1)),
    # near shared box, boom especially
    ("lim_boom_low", BASE, (0.0, 0.12, 0.6, 1.0, 0.0)),
    ("lim_boom_high", BASE, (0.0, 1.45, 0.6, 1.0, 0.0)),
    ("lim_tele_out", BASE, (0.0, 0.5, 0.6, 2.15, 0.0)),
    ("lim_arm_out", BASE, (0.0, 1.3, 1.2, 1.8, 0.0)),
    # pairs and reconfigurations
    ("pair_slew_lift", BASE, (1.2, 1.0, 0.9, 1.0, 0.0)),
    ("pair_reach", BASE, (0.0, 0.3, 0.2, 2.0, 0.0)),
    ("pair_tuck", (1.0, 1.2, 1.0, 2.0, 0.5), (0.0, 0.2, 0.1, 0.2, 0.0)),
)

#: Timed separately, in call order; plan runs them as module functions/methods
#: so wrapping names suffices -- no production edit, the point of a baseline.
#: `swept` is inside `ik`/`path` (`Geometry._swept`), not beside them -- it is the
#: sway branch-and-bound those two pay for per configuration, and the only stage
#: a carried body changes by an order of magnitude, so it is reported separately.
PHASES = ("prepare_scene", "geometry", "ik", "path", "coefficients", "ocp", "resample")
NESTED = ("swept",)

#: What `--carry` plans with. The block as the behaviour tree publishes it
#: (`sim_captures/2026-10-09-pp-check/run1`: r=0.54, l=0.60; 880 kg block),
#: the `truck` primitive as `config/crane_planner.yaml` documents the vehicle box,
#: and five blocks on the ground inside reach -- 13 bodies after `expand_truck`,
#: the order the live wall-assembly scene carries.
#:
#: Poses are in K0_mounting_base, so the vehicle box is *behind* the column and the
#: bed top sits at K0's own height: ground is 1.26 m below K0 (the yaml's bed
#: height), not 0. Placing it at its world-frame z stands the headboard through the
#: machine and every pose refuses.
CARRY_MASS = 880.0
CARRY_COM = np.array(
    [-0.010, -0.021, 1.081]
)  # captured 1.032 below TCP, + TCP 0.049 below K8
CARRY_SHAPE = ("cylinder", np.array([1.08, 1.08, 0.60]), CARRY_COM)
CARRY_SCENE = [
    CollisionPrimitive(
        id="truck",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([-5.2, 0.0, -0.33])),
        dimensions_m=np.array([5.554, 2.518, 0.663]),
        structural=True,
    )
] + [
    CollisionPrimitive(
        id=f"block_{index}",
        shape="cylinder",
        # Ground is ~1.26 m below K0 (the yaml's bed height), blocks 0.60 tall.
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([x, y, -0.96])),
        dimensions_m=np.array([1.08, 1.08, 0.60]),
    )
    for index, (x, y) in enumerate(
        ((5.4, -1.2), (5.4, 0.0), (5.4, 1.2), (6.6, -0.6), (6.6, 0.6))
    )
]

#: Carry from over one block to over a stack place, both inside the shared box.
CARRY_MOVES = (("carry", (0.6, 0.4, 0.6, 1.0, 0.0), (-0.2, 0.5, 0.7, 1.3, 0.3)),)


def description() -> str:
    path = PACKAGE.parent / "crane_model" / "test" / "description" / DESCRIPTION
    if not path.is_file():
        raise SystemExit(f"no machine description at {path}")
    return path.read_text()


@contextlib.contextmanager
def timed(record: dict):
    """Wrap every stage `plan` calls, writing elapsed seconds into `record`."""

    def wrap(owner, name, phase):
        original = getattr(owner, name)

        def instrumented(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                record[phase] += time.perf_counter() - started

        setattr(owner, name, instrumented)
        return owner, name, original

    patched = [
        wrap(planner_module.Planner, "prepare_scene", "prepare_scene"),
        wrap(planner_module, "Geometry", "geometry"),
        wrap(planner_module, "solve_ik", "ik"),
        wrap(planner_module, "plan_fitted_path", "path"),
        wrap(planner_module, "power_coefficients", "coefficients"),
        wrap(planner_module.TrajectoryOcp, "solve", "ocp"),
        wrap(planner_module.Planner, "_resample", "resample"),
        wrap(geometry_module.Geometry, "_swept", "swept"),
    ]
    try:
        yield
    finally:
        for owner, name, original in patched:
            setattr(owner, name, original)


def configuration(planner: Planner, coordinates) -> np.ndarray:
    """Build the canonical eight from five planned coordinates, tool hanging."""
    q = np.zeros(8)
    q[list(PLANNED_INDICES)] = coordinates
    q[TOOL_INDEX] = TOOL_POSITION
    q[[4, 5]] = planner.model.passive_equilibrium(q[list(ACTUATED_INDICES)])
    return q


def goal_pose(planner: Planner, coordinates, frame=Frame.TCP) -> tuple:
    """`(position_m, yaw)` of `frame` at a joint configuration."""
    import pinocchio as pin

    q = configuration(planner, coordinates)
    pose = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, frame)
    tcp = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    rotation = pin.XYZQUATToSE3(
        np.concatenate([tcp.position_m, tcp.orientation_xyzw])
    ).rotation
    return pose.position_m, yaw_of(rotation)


def emit_requests(planner: Planner, path: Path, moves) -> None:
    """
    Write moves as CalcMovement fields for the service benchmark.

    Client replays this file, computes no goal, so both servers see
    byte-identical requests. y_n is the tip pivot K5 (what
    a2b.translate_request converts from), not the TCP.
    """
    import json

    requests = []
    for name, start_q, goal_q in moves:
        position, yaw = goal_pose(planner, goal_q, Frame.TIP)
        requests.append(
            {
                "name": name,
                "q0": configuration(planner, start_q).tolist(),
                "y_n": [float(value) for value in position],
                "phi_tool_n": float(yaw),
            }
        )
    path.write_text(
        json.dumps({"description": DESCRIPTION, "moves": requests}, indent=2)
    )
    print(f"wrote {len(requests)} requests to {path}")


def utilisation(planner: Planner, timing) -> dict:
    """
    How close the answer came to each row, as a fraction of the row's own limit.

    Fraction of the *limit*, not of the bound the solve was given: those differ
    by `kappa` (and `speed_scale`), so 0.8 is a rate row sitting on its bound.
    Pump is already carried as a fraction, with its bound beside it.
    """
    config = planner.config
    return {
        "dq": float(np.max(np.abs(timing.dq_a) / planner.limits.dq_max)),
        "ddq": float(np.max(np.abs(timing.ddq_a) / np.asarray(config.ddq_a_max))),
        "sway": float(
            np.max(np.abs(timing.q_u - timing.q_u_eq) / np.asarray(config.q_sway_max))
        ),
        "pump": float(np.max(timing.pump_flow)),
        "pump_bound": float(timing.pump_flow_bound),
        "terminal_sway": timing.terminal_sway,
        "status": float(timing.stats["acados_status"]),
    }


def run(planner: Planner, move, speed_scale: float, carry: bool = False) -> dict | None:
    """One planned move, per-phase seconds plus what the solve reported."""
    _name, start_q, goal_q = move
    start = Start(q=configuration(planner, start_q), dq_a=np.zeros(5))
    position, yaw = goal_pose(planner, goal_q)
    payload = (
        Payload(mass_kg=CARRY_MASS, center_of_mass_k8_m=CARRY_COM, valid=True)
        if carry
        else None
    )
    record: dict = defaultdict(float)
    started = time.perf_counter()
    try:
        with timed(record):
            plan = planner.plan(
                start,
                position,
                yaw,
                payload=payload,
                payload_shape=CARRY_SHAPE if carry else None,
                scene=CARRY_SCENE if carry else [],
                avoid_collisions=True,
                speed_scale=speed_scale,
            )
    except PlanningError as refusal:
        return {"refused": str(refusal)}
    record["total"] = time.perf_counter() - started
    record["acados"] = plan.timing.acados_time_s
    record["iterations"] = plan.timing.iterations
    record["duration"] = plan.timing.duration
    record.update(utilisation(planner, plan.timing))
    return dict(record)


def report(samples: dict) -> None:
    """Median and worst per phase, over every repeat of every move."""
    print(f"\n{'phase':<14} {'median s':>10} {'p90 s':>10} {'share':>7}")
    total = float(np.median(samples["total"]))
    for phase in (*PHASES, *NESTED, "total"):
        values = np.asarray(samples[phase], dtype=float)
        if not values.size:
            continue
        median = float(np.median(values))
        share = "" if phase == "total" else f"{100.0 * median / total:6.1f}%"
        label = f"{phase} *" if phase in NESTED else phase
        print(
            f"{label:<14} {median:10.3f} "
            f"{float(np.percentile(values, 90)):10.3f} {share:>7}"
        )
    print("* nested inside ik/path, so its share is already counted there")
    acados = np.asarray(samples["acados"], dtype=float)
    ocp = np.asarray(samples["ocp"], dtype=float)
    print(
        f"\nacados        {float(np.median(acados)):.3f} s median, "
        f"{100.0 * float(np.median(acados)) / float(np.median(ocp)):.0f}% of the "
        f"`ocp` phase -- the rest is Python: guess, readback, flow"
    )
    print(
        f"iterations    {int(np.median(samples['iterations']))} median, "
        f"{int(np.max(samples['iterations']))} worst, "
        f"{1e3 * float(np.median(acados)) / max(float(np.median(samples['iterations'])), 1.0):.0f} "
        "ms each -- the load figure to quote (261)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one cost weight, repeatable (e.g. --weight dq_a=0.3); "
        "only has effect since the W cost_set fix",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="override the deployment reservation, to attribute what it costs",
    )
    parser.add_argument(
        "--only", default=None, help="one move name from the shipped set"
    )
    parser.add_argument(
        "--set",
        dest="move_set",
        choices=("core", "wide", "carry", "all"),
        default="core",
        help="`core` is the five long moves, `wide` adds single-axis, short and "
        "near-limit ones; `all` is both. `carry` is the one move planned with a "
        "block hanging against `CARRY_SCENE`, which `core`/`wide` never are",
    )
    parser.add_argument(
        "--emit-requests",
        type=Path,
        default=None,
        help="write the moves as CalcMovement fields and exit, without planning",
    )
    parser.add_argument(
        "--stage", choices=("corridors", "ces"), default=None, help="geometric stage"
    )
    options = parser.parse_args()

    catalogue = {
        "core": MOVES,
        "wide": WIDE_MOVES,
        "carry": CARRY_MOVES,
        "all": MOVES + WIDE_MOVES,
    }
    carry = options.move_set == "carry"
    moves = [m for m in catalogue[options.move_set] if options.only in (None, m[0])]
    if not moves:
        raise SystemExit(f"no move named {options.only!r}")

    config = PlannerConfig()
    if options.kappa is not None:
        config.kappa = options.kappa
    if options.stage is not None:
        config.geometric_stage = options.stage
    weights = dict(crane_weights.DEFAULTS)
    for override in options.weight:
        key, _, value = override.partition("=")
        if key not in weights:
            raise SystemExit(f"no cost weight named {key!r}")
        weights[key] = float(value)
    built = time.perf_counter()
    planner = Planner(description(), config, weights)
    print(f"planner built in {time.perf_counter() - built:.1f} s")

    if options.emit_requests is not None:
        emit_requests(planner, options.emit_requests, moves)
        return 0

    # Discarded: first call pays for whatever the solver loaded lazily; a
    # baseline carrying that once is no baseline of a served request.
    run(planner, moves[0], options.speed_scale, carry)
    if carry:
        print(
            f"carrying {CARRY_MASS:.0f} kg against "
            f"{len(planner.prepare_scene(CARRY_SCENE, True))} bodies after truck expansion"
        )

    samples: dict = defaultdict(list)
    # Utilisation beside duration: a move that got faster by spending a row it
    # was nowhere near is a different answer from one that hit the row.
    print(
        f"\n{'move':<14} {'dur s':>6} {'iter':>5} {'st':>3} "
        f"{'dq':>5} {'ddq':>5} {'pump':>5} {'sway':>5} {'term':>6}"
    )
    for move in moves:
        per_move: dict = defaultdict(list)
        for _ in range(options.repeats):
            answer = run(planner, move, options.speed_scale, carry)
            if "refused" in answer:
                print(f"{move[0]:<14} refused: {answer['refused']}")
                break
            for key, value in answer.items():
                per_move[key].append(value)
                samples[key].append(value)
        if per_move:

            def median(key: str, rows=per_move) -> float:
                return float(np.median(rows[key]))

            print(
                f"{move[0]:<14} {median('duration'):6.2f} "
                f"{int(median('iterations')):5d} {int(median('status')):3d} "
                f"{median('dq'):5.2f} {median('ddq'):5.2f} "
                f"{median('pump'):5.2f} {median('sway'):5.2f} "
                f"{np.degrees(median('terminal_sway')):6.2f}"
            )
    if samples:
        print(
            "dq/ddq/pump/sway are fractions of the limit; the solve's own bound "
            f"is kappa = {planner.config.kappa} on the rate rows, "
            f"{float(np.median(samples['pump_bound'])):.2f} on the pump. "
            "`term` is terminal sway in deg, `st` the acados status"
        )
        report(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
