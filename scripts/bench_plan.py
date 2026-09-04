#!/usr/bin/env python3
"""
Where a `Planner.plan` call spends its time. No ROS, no graph, no clock.

Stage 0 of the C4 continuity work: the baseline to attribute against, taken
before the spline degree or the OCP state vector move. Answers two questions
the service-level benchmark (`bench_calc_movement.py`) cannot -- which stage
costs what, and how much of the OCP's wall time is acados rather than Python.

    ./scripts/bench_plan.py --repeats 5

Wall clock is load-bound (`ocp.py`: 0.53 s idle against 4.97 s under a running
Gazebo), so run this idle and read a regression off `sqp_iter` and the acados
time, not off the totals.
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

from crane_model import Frame  # noqa: E402
from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402
from crane_planning import planner as planner_module  # noqa: E402
from crane_planning.planner import (  # noqa: E402
    ACTUATED_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    yaw_of,
)

DESCRIPTION = "pzs100.urdf"

# The two servers do not agree on what a joint may do, so a request set both can
# answer lives in the **intersection** of their boxes, which is much smaller than
# either. Measured, per planned axis (slew, boom, arm, telescope, rotator):
#
#     crane_planning, off the PZS100 description it reads:
#         [-3.71, 3.71] [-1.20, 1.563] [-0.91, 4.60] [0.00, 2.236]  unbounded
#     a2b_ilqr_server, `qMinCtrl`/`qMaxCtrl` in `mp_parameter_pzs100.yaml`,
#     less its 0.087 rad `qLimSafetyBuffer`:
#         [-3.71, 3.71] [ 0.00, 1.562] [-0.91, 1.325] [0.00, 2.236] [-12.6, 12.6]
#
# The boom is the binding one: `a2b_ilqr_server` refuses any negative boom angle
# ("q0[1] is not feasible: 0.00 < -0.30 < 1.56"), which is where `plan_example`'s
# shipped -0.2 sits. So these moves keep the boom positive; they are not
# `plan_example`'s and the two are not comparable.

#: The tool coordinate q8. `crane_planning`'s description bounds it to
#: [0.20, 0.70], `mp_parameter_pzs100.yaml` to [0.00, 0.538]; 0.45 is inside
#: both with room for the safety buffer.
TOOL_POSITION = 0.45

#: `(name, start, goal)` in planned joint coordinates, all inside the
#: intersection above. Spread over the axes a duration is decided by: slew
#: alone, telescope alone, and three that move everything. Reachability is not
#: assumed -- a goal is a joint configuration pushed through FK, so the IK stage
#: has an answer.
MOVES = (
    ("slew", (0.0, 0.4, 0.6, 1.0, 0.0), (0.9, 0.4, 0.6, 1.0, 0.0)),
    ("telescope", (0.0, 0.4, 0.6, 0.8, 0.0), (0.0, 0.4, 0.6, 1.6, 0.0)),
    ("lift", (0.0, 0.4, 0.6, 1.0, 0.0), (0.6, 0.7, 0.9, 1.4, 0.4)),
    ("stow", (0.6, 0.7, 0.9, 1.4, 0.4), (0.0, 0.25, 0.35, 0.7, 0.0)),
    ("across", (-0.7, 0.3, 0.4, 0.9, -0.3), (0.8, 0.8, 1.0, 1.5, 0.5)),
)

#: Timed separately, in call order. `plan` runs them as module-level functions
#: and two methods, so wrapping the names is enough -- no production edit, which
#: is the point of a baseline.
PHASES = ("prepare_scene", "geometry", "ik", "path", "coefficients", "ocp", "resample")


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


def emit_requests(planner: Planner, path: Path) -> None:
    """
    Write the same moves as `CalcMovement` fields, for the service benchmark.

    Generated here rather than in the ROS client so both servers are handed
    byte-identical requests off one description: the client replays a file, it
    does not compute a goal. `y_n` is the **tip pivot K5**, which is what the
    `.srv` names and what `a2b.translate_request` converts from -- not the TCP.
    """
    import json

    requests = []
    for name, start_q, goal_q in MOVES:
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


def run(planner: Planner, move, speed_scale: float) -> dict | None:
    """One planned move, per-phase seconds plus what the solve reported."""
    _name, start_q, goal_q = move
    start = Start(q=configuration(planner, start_q), dq_a=np.zeros(5))
    position, yaw = goal_pose(planner, goal_q)
    record: dict = defaultdict(float)
    started = time.perf_counter()
    try:
        with timed(record):
            plan = planner.plan(
                start,
                position,
                yaw,
                scene=[],
                avoid_collisions=True,
                speed_scale=speed_scale,
            )
    except PlanningError as refusal:
        return {"refused": str(refusal)}
    record["total"] = time.perf_counter() - started
    record["acados"] = plan.timing.acados_time_s
    record["iterations"] = plan.timing.iterations
    record["duration"] = plan.timing.duration
    return dict(record)


def report(samples: dict) -> None:
    """Median and worst per phase, over every repeat of every move."""
    print(f"\n{'phase':<14} {'median s':>10} {'p90 s':>10} {'share':>7}")
    total = float(np.median(samples["total"]))
    for phase in (*PHASES, "total"):
        values = np.asarray(samples[phase], dtype=float)
        if not values.size:
            continue
        median = float(np.median(values))
        share = "" if phase == "total" else f"{100.0 * median / total:6.1f}%"
        print(
            f"{phase:<14} {median:10.3f} "
            f"{float(np.percentile(values, 90)):10.3f} {share:>7}"
        )
    acados = np.asarray(samples["acados"], dtype=float)
    ocp = np.asarray(samples["ocp"], dtype=float)
    print(
        f"\nacados        {float(np.median(acados)):.3f} s median, "
        f"{100.0 * float(np.median(acados)) / float(np.median(ocp)):.0f}% of the "
        f"`ocp` phase -- the rest is Python: guess, readback, flow"
    )
    print(
        f"iterations    {int(np.median(samples['iterations']))} median, "
        f"{int(np.max(samples['iterations']))} worst"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--only", default=None, help="one move name from the shipped set"
    )
    parser.add_argument(
        "--emit-requests",
        type=Path,
        default=None,
        help="write the moves as CalcMovement fields and exit, without planning",
    )
    options = parser.parse_args()

    moves = [m for m in MOVES if options.only in (None, m[0])]
    if not moves:
        raise SystemExit(f"no move named {options.only!r}")

    built = time.perf_counter()
    planner = Planner(description(), PlannerConfig())
    print(f"planner built in {time.perf_counter() - built:.1f} s")

    if options.emit_requests is not None:
        emit_requests(planner, options.emit_requests)
        return 0

    # Discarded: the first call pays for whatever the solver loaded lazily, and
    # a baseline that carries it once is not a baseline of a served request.
    run(planner, moves[0], options.speed_scale)

    samples: dict = defaultdict(list)
    for move in moves:
        per_move: dict = defaultdict(list)
        for _ in range(options.repeats):
            answer = run(planner, move, options.speed_scale)
            if "refused" in answer:
                print(f"{move[0]:<12} refused: {answer['refused']}")
                break
            for key, value in answer.items():
                per_move[key].append(value)
                samples[key].append(value)
        if per_move:
            print(
                f"{move[0]:<12} {float(np.median(per_move['total'])):6.3f} s over "
                f"{len(per_move['total'])} runs, "
                f"{float(np.median(per_move['duration'])):5.2f} s trajectory, "
                f"{int(np.median(per_move['iterations']))} iterations"
            )
    if samples:
        report(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
