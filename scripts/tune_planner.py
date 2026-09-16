#!/usr/bin/env python3
"""
Plan one motion and roll it on MuJoCo, so the sway claim gets a second opinion.

`plan_example.py` answers "what did the OCP decide"; this answers "and was it
right". The planner's sway trace is its own model's prediction, so a plan that
arrives at rest on paper proves nothing about the plan -- only about the
agreement between the OCP and the model it was exported from. MuJoCo is an
independent articulated-body solver on the same URDF, so the gap between the two
traces is the modelling error the planner is actually exposed to.

    ./scripts/tune_planner.py --goal out --show
    ./scripts/tune_planner.py --goal here --kappa 0.6 --settle 8

Start pose is `initialization_outside.yaml` (`crane_model.presets.OUTSIDE`).
`--goal out` is a TCP placement at (4, 0, 2) m in `K0_mounting_base`; `--goal
here` lifts the start TCP to z = 2 m and leaves x and y alone. Both are goals
that could have been typed into the RViz panel, because that is the frame and
the shape `crane_msgs/srv/PlanMotion` carries.

What this does **not** simulate is the controller: the planned axes are driven
onto the reference with gravity, damping and the mimic's pull cancelled, which
is as close to an ideal drive as the plant allows. Tracking error is reported so
a run says when that assumption is the thing being measured. The MPC's own
tracking is `crane_mpc/scripts/tune_mpc.py`, and the hydraulics are in neither
(`wiki/hydraulics.md` §5.1).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import plan_example  # noqa: E402
from crane_model import Frame, Payload, presets  # noqa: E402
from crane_model.conventions import ACTUATED_INDICES, PASSIVE_INDICES  # noqa: E402
from crane_model.mujoco_plant import MujocoPlant, viewer_was_opened  # noqa: E402
from crane_planning import Planner, PlanningError, Start  # noqa: E402
from crane_planning.planner import PLANNED_INDICES, TOOL_INDEX, yaw_of  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--goal",
        choices=("out", "here"),
        default="out",
        help="'out' = (4, 0, 2) m; 'here' = start x, y lifted to z = 2 m",
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "YAW"),
        help="TCP pose in K0_mounting_base instead of a preset goal",
    )
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--resettle-start",
        action="store_true",
        help=(
            "replace the preset's passive pair with its own equilibrium. The "
            "yaml's is about 20 mrad off, and that transient adds to every sway "
            "number a run prints"
        ),
    )

    tuning = parser.add_argument_group("timing")
    tuning.add_argument("--kappa", type=float, default=None)
    tuning.add_argument("--ocp-duration-max", type=float, default=None)
    tuning.add_argument("--ocp-integrator", choices=("ERK", "IRK"), default=None)

    clearance = parser.add_argument_group('what "clear" means, in metres')
    clearance.add_argument("--margin-safety", type=float, default=None)
    clearance.add_argument("--margin-interp", type=float, default=None)

    plant = parser.add_argument_group("the plant")
    plant.add_argument(
        "--settle",
        type=float,
        default=6.0,
        help="seconds to hold the final pose after the plan ends, watching",
    )
    plant.add_argument(
        "--timestep",
        type=float,
        default=5.0e-4,
        help=(
            "MuJoCo step. Pairs with --bandwidth: the drive is explicit, so "
            "raising one without lowering the other diverges, and the plant "
            "says so rather than quietly resetting a body"
        ),
    )
    plant.add_argument(
        "--bandwidth",
        type=float,
        default=100.0,
        help=(
            "rad/s of the drive that puts the axes on the reference. The "
            "arrival sway moves with it until it converges -- 30 rad/s still "
            "reports half a degree of drive lag as if it were sway -- so a "
            "number worth quoting is one that survives doubling this"
        ),
    )
    plant.add_argument(
        "--viewer",
        action="store_true",
        help="watch it in MuJoCo's passive viewer while it runs",
    )
    plant.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help=(
            "viewer playback factor: 1.0 is a simulated second per second, 0 is "
            "as fast as it computes. Pacing only, the run is the same either way"
        ),
    )
    plant.add_argument(
        "--collision",
        action="store_true",
        help="let MuJoCo resolve contacts too; off by default, the planner "
        "already proved clearance against Coal",
    )

    parser.add_argument("--output", type=Path, default=Path("tune_planner.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def start_of(planner: Planner, options) -> Start:
    """Take the preset, its passive pair either as written or re-settled."""
    q = presets.OUTSIDE.copy()
    if options.resettle_start:
        q[list(PASSIVE_INDICES)] = planner.model.passive_equilibrium(
            q[list(ACTUATED_INDICES)]
        )
    return Start(q=q, dq_a=np.zeros(len(ACTUATED_INDICES) - 1))


def goal_of(planner: Planner, start: Start, options) -> tuple[np.ndarray, float]:
    """Where the tool goes, and the yaw it keeps, both in K0_mounting_base."""
    pose = planner.model.forward_kinematics(start.q, Frame.MOUNTING_BASE, Frame.TCP)
    import pinocchio as pin

    yaw = yaw_of(
        pin.XYZQUATToSE3(
            np.concatenate([pose.position_m, pose.orientation_xyzw])
        ).rotation
    )
    if options.goal_pose is not None:
        return np.array(options.goal_pose[:3]), options.goal_pose[3]
    if options.goal == "out":
        return presets.GOAL_OUT.copy(), yaw
    return presets.goal_here(planner.model, start.q), yaw


def roll(planner: Planner, description: str, plan, start: Start, options):
    """
    Drive the plan on MuJoCo and record what the load did; return both.

    The plan is a sampled reference at its own `Ts`, so the plant is advanced
    one sample at a time on the same grid; after the last sample the final pose
    is held for `--settle` seconds, which is where a plan that merely *arrives*
    quiet separates from one that *leaves* the load quiet.
    """
    plant = MujocoPlant(
        description, timestep=options.timestep, collision=options.collision
    )
    plant.set_state(start.q, np.zeros(len(start.q)))
    if options.viewer:
        plant.open_viewer(realtime=options.realtime)

    times, states, tracking = [0.0], [plant.state], [np.zeros(len(PLANNED_INDICES))]
    sample = float(plan.time[1] - plan.time[0])
    q_tool = float(start.q[TOOL_INDEX])

    def advance(q_ref, dq_ref, ddq_ref, duration, stamp) -> None:
        plant.follow(
            q_ref,
            dq_ref,
            ddq_ref,
            duration,
            bandwidth_rad_s=options.bandwidth,
            q_tool_ref=q_tool,
        )
        times.append(stamp)
        states.append(plant.state)
        tracking.append(plant.q[list(PLANNED_INDICES)] - q_ref)

    for index in range(1, plan.time.size):
        advance(
            plan.q[index, list(PLANNED_INDICES)],
            plan.dq[index, list(PLANNED_INDICES)],
            plan.ddq[index, list(PLANNED_INDICES)],
            float(plan.time[index] - plan.time[index - 1]),
            float(plan.time[index]),
        )
    held = plan.q[-1, list(PLANNED_INDICES)]
    still = np.zeros(len(PLANNED_INDICES))
    for step in range(int(round(options.settle / sample))):
        advance(held, still, still, sample, float(plan.time[-1] + (step + 1) * sample))

    state = np.array(states)
    # Sway is an offset from rest, and rest moves with the machine: the passive
    # pair's equilibrium is a function of where the actuated axes are, so it is
    # re-solved per sample rather than taken at the start pose.
    equilibrium = np.array(
        [
            planner.model.passive_equilibrium(
                _canonical(row, q_tool)[list(ACTUATED_INDICES)]
            )
            for row in state
        ]
    )
    return {
        "time": np.array(times),
        "q_a": state[:, 0:5],
        "q_u": state[:, 5:7],
        "dq_a": state[:, 7:12],
        "dq_u": state[:, 12:14],
        "q_u_eq": equilibrium,
        "tracking": np.array(tracking),
    }, plant


def _canonical(rigid: np.ndarray, q_tool: float) -> np.ndarray:
    q = np.zeros(8)
    q[list(PLANNED_INDICES)] = rigid[0:5]
    q[list(PASSIVE_INDICES)] = rigid[5:7]
    q[TOOL_INDEX] = q_tool
    return q


def report(plan, rolled: dict, planning_elapsed: float) -> list[str]:
    """Say what the two models make of the same motion, side by side."""
    timing = plan.timing
    predicted = timing.q_u - timing.q_u_eq
    actual = rolled["q_u"] - rolled["q_u_eq"]
    arrival = int(np.searchsorted(rolled["time"], plan.duration))

    def pair(values) -> str:
        tip, tilt = np.degrees(values)
        return f"{tip:9.3f}  {tilt:9.3f}"

    return [
        f"plan           {plan.duration:.2f} s, {timing.iterations} SQP iterations",
        f"planner call   {planning_elapsed:.3f} s end to end",
        "",
        "                       tip       tilt   [deg]",
        f"peak sway  OCP   {pair(np.max(np.abs(predicted), axis=0))}",
        f"peak sway  mjc   {pair(np.max(np.abs(actual), axis=0))}",
        f"at arrival OCP   {pair(predicted[-1])}",
        f"at arrival mjc   {pair(actual[arrival])}",
        f"after settle     {pair(actual[-1])}",
        f"settled rate     {pair(rolled['dq_u'][-1])}  [deg/s]",
        "",
        f"drive error    {np.max(np.abs(rolled['tracking'])):.2e} worst axis, "
        "worst instant",
        "               (raise --bandwidth if this is the size of the sway)",
    ]


def figure(plan, rolled: dict, planning_elapsed: float):
    import matplotlib.pyplot as plt

    timing = plan.timing
    predicted = timing.q_u - timing.q_u_eq
    actual = rolled["q_u"] - rolled["q_u_eq"]
    t = rolled["time"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for index, label in enumerate(("tip", "tilt")):
        colour = f"C{index}"
        axes[0, 0].plot(
            timing.time, predicted[:, index], colour, ls="--", label=f"{label} OCP"
        )
        axes[0, 0].plot(t, actual[:, index], colour, label=f"{label} MuJoCo")
        axes[0, 1].plot(
            timing.time, timing.dq_u[:, index], colour, ls="--", label=f"{label} OCP"
        )
        axes[0, 1].plot(t, rolled["dq_u"][:, index], colour, label=f"{label} MuJoCo")
    axes[0, 0].set_ylabel(r"sway offset $q_u - q_u^{eq}$ [rad]")
    axes[0, 1].set_ylabel("sway rate [rad/s]")
    for cell in (axes[0, 0], axes[0, 1]):
        cell.axvline(plan.duration, color="k", lw=0.8, ls=":")

    names = [f"q{index + 1}" for index in PLANNED_INDICES]
    for axis, name in enumerate(names):
        axes[1, 0].plot(t, rolled["tracking"][:, axis], label=name)
    axes[1, 0].set_ylabel("drive error, plant - reference [rad, m]")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(report(plan, rolled, planning_elapsed)),
        family="monospace",
        fontsize=8,
        va="top",
        transform=axes[1, 1].transAxes,
    )

    for row in axes:
        for cell in row:
            cell.grid(alpha=0.3)
            cell.set_xlabel("time [s]")
            if cell.get_legend_handles_labels()[0]:
                cell.legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"plan of {plan.duration:.2f} s rolled on MuJoCo; dotted = the OCP's own "
        "prediction, solid = the plant"
    )
    fig.tight_layout()
    return fig


def main() -> int:
    options = arguments()
    description = plan_example.description()
    planner = Planner(description, plan_example.configure(options))

    start = start_of(planner, options)
    offset = presets.settled(planner.model, start.q)
    print(f"start passive pair sits {np.degrees(offset)} deg off rest")
    position, yaw = goal_of(planner, start, options)
    print(
        f"goal: [{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}] m, "
        f"yaw {np.degrees(yaw):.1f} deg"
    )

    payload = Payload(mass_kg=options.payload_mass, valid=options.payload_mass > 0.0)
    began = time.monotonic()
    try:
        plan = planner.plan(
            start,
            position,
            yaw,
            payload=payload if payload.valid else None,
            scene=[],
            avoid_collisions=True,
            speed_scale=options.speed_scale,
        )
    except PlanningError as refusal:
        print(f"refused: {refusal}")
        return 1
    planning_elapsed = time.monotonic() - began
    print(plan.message)

    rolled, plant = roll(planner, description, plan, start, options)
    for line in report(plan, rolled, planning_elapsed):
        print(line)

    if options.csv is not None:
        header = "time_s," + ",".join(
            [f"q{index + 1}" for index in PLANNED_INDICES]
            + ["q_tip", "q_tilt", "q_tip_eq", "q_tilt_eq", "dq_tip", "dq_tilt"]
        )
        np.savetxt(
            options.csv,
            np.column_stack(
                [
                    rolled["time"],
                    rolled["q_a"],
                    rolled["q_u"],
                    rolled["q_u_eq"],
                    rolled["dq_u"],
                ]
            ),
            delimiter=",",
            header=header,
            comments="",
        )
        print(f"wrote {options.csv}")

    if not options.show:
        matplotlib.use("Agg")
    fig = figure(plan, rolled, planning_elapsed)
    fig.savefig(options.output, dpi=150)
    print(f"wrote {options.output}")
    if options.show:
        import matplotlib.pyplot as plt

        plt.show()
    # The window outlives the run: the interesting part of a lift is often the
    # pose it ends in, and the report and the figure are worth having first.
    if options.viewer:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 0


if __name__ == "__main__":
    status = main()
    # A viewer run would otherwise exit 139: MuJoCo's viewer segfaults on
    # interpreter teardown here, after every file is written. `os._exit` leaves
    # without tearing down.
    if viewer_was_opened():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)
