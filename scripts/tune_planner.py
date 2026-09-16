#!/usr/bin/env python3
"""
Plan one motion and roll it on MuJoCo, so the sway claim has a second opinion.

`plan_example.py` says what the OCP decided; its sway trace is the OCP's own
model predicting itself. This drives the same plan through an independent solver
on the same URDF and plots the two against each other.

    ./scripts/tune_planner.py --goal out --viewer
    ./scripts/tune_planner.py --goal here --kappa 0.6 --settle 8

Start is `crane_model.presets.OUTSIDE`. The planned axes are driven onto the
reference with the resistance cancelled, as close to an ideal drive as the plant
allows -- the controller is `crane_mpc/scripts/tune_mpc.py`, the hydraulics are
in neither.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import plan_example  # noqa: E402
from crane_model import Frame, presets  # noqa: E402
from crane_model.conventions import ACTUATED_INDICES, PASSIVE_INDICES  # noqa: E402
from crane_model.mujoco_plant import MujocoPlant, leave  # noqa: E402
from crane_planning import Planner, PlanningError, Start  # noqa: E402
from crane_planning.planner import PLANNED_INDICES, TOOL_INDEX, yaw_of  # noqa: E402


def plan_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add every flag `plan_for` and `plan_example.configure` read, and no other.

    `tune_mpc.py` calls this too, so the two tools cannot drift into meaning
    different things by the same flag. Its own `--output` and `--show` stay out:
    it forwards unknown flags to `mpc_a2b`, which owns those.
    """
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
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--resettle-start",
        action="store_true",
        help="start the passive pair at its own equilibrium, not the preset's",
    )

    tuning = parser.add_argument_group("timing")
    tuning.add_argument("--kappa", type=float, default=None)
    tuning.add_argument("--ocp-duration-max", type=float, default=None)
    tuning.add_argument("--ocp-integrator", choices=("ERK", "IRK"), default=None)

    clearance = parser.add_argument_group('what "clear" means, in metres')
    clearance.add_argument("--margin-safety", type=float, default=None)
    clearance.add_argument("--margin-interp", type=float, default=None)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    plan_arguments(parser)

    plant = parser.add_argument_group("the plant")
    plant.add_argument(
        "--settle", type=float, default=6.0, help="seconds to hold the final pose"
    )
    plant.add_argument("--timestep", type=float, default=5.0e-4)
    plant.add_argument(
        "--bandwidth",
        type=float,
        default=100.0,
        help="rad/s of the drive. The arrival sway moves with it until it "
        "converges, so a number worth quoting survives doubling this",
    )
    plant.add_argument("--viewer", action="store_true")
    plant.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer playback factor; 0 is as fast as it computes. Pacing only",
    )

    parser.add_argument(
        "--output", type=Path, default=PACKAGE / "build" / "tune_planner.png"
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def start_of(planner: Planner, options) -> Start:
    """Take the preset, its passive pair either as written or re-settled."""
    q = presets.OUTSIDE.copy()
    if options.resettle_start:
        q[list(PASSIVE_INDICES)] = planner.model.passive_equilibrium(
            q[list(ACTUATED_INDICES)]
        )
    return Start(q=q, dq_a=np.zeros(len(PLANNED_INDICES)))


def goal_of(planner: Planner, start: Start, options) -> tuple[np.ndarray, float]:
    """Where the tool goes and the yaw it keeps, in K0_mounting_base."""
    import pinocchio as pin

    pose = planner.model.forward_kinematics(start.q, Frame.MOUNTING_BASE, Frame.TCP)
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


def plan_for(options) -> tuple[str, Planner, Start, object]:
    """
    Build the planner, solve the preset move, say what came out.

    Shared with `crane_mpc/scripts/tune_mpc.py`, which tunes the controller
    against the same plan this tunes the shaping of.
    """
    description = plan_example.description()
    planner = Planner(description, plan_example.configure(options))
    start = start_of(planner, options)
    offset = start.q[list(PASSIVE_INDICES)] - planner.model.passive_equilibrium(
        start.q[list(ACTUATED_INDICES)]
    )
    print(f"start passive pair sits {np.degrees(offset)} deg off rest")
    position, yaw = goal_of(planner, start, options)
    print(
        f"goal: [{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}] m, "
        f"yaw {np.degrees(yaw):.1f} deg"
    )
    try:
        plan = planner.plan(
            start,
            position,
            yaw,
            scene=[],
            avoid_collisions=True,
            speed_scale=options.speed_scale,
        )
    except PlanningError as refusal:
        raise SystemExit(f"refused: {refusal}") from refusal
    print(plan.message)
    return description, planner, start, plan


def roll(planner: Planner, description: str, plan, start: Start, options):
    """Drive the plan on MuJoCo, record what the load did, return both."""
    plant = MujocoPlant(description, timestep=options.timestep)
    plant.set_state(start.q)
    q_tool = float(start.q[TOOL_INDEX])
    sample = float(plan.time[1] - plan.time[0])
    still = np.zeros(len(PLANNED_INDICES))
    rows = list(PLANNED_INDICES)

    times, states, tracking = [0.0], [plant.state], [still]

    def advance(q_ref, dq_ref, ddq_ref, duration, stamp) -> None:
        plant.follow(q_ref, dq_ref, ddq_ref, duration, options.bandwidth)
        times.append(stamp)
        states.append(plant.state)
        tracking.append(plant.q[rows] - q_ref)

    if options.viewer:
        plant.open_viewer(options.realtime)
    for index in range(1, plan.time.size):
        advance(
            plan.q[index, rows],
            plan.dq[index, rows],
            plan.ddq[index, rows],
            float(plan.time[index] - plan.time[index - 1]),
            float(plan.time[index]),
        )
    # A plan that merely *arrives* quiet separates here from one that leaves the
    # load quiet.
    for step in range(int(round(options.settle / sample))):
        advance(
            plan.q[-1, rows],
            still,
            still,
            sample,
            float(plan.time[-1] + (step + 1) * sample),
        )

    state = np.array(states)
    # Rest moves with the machine, so the equilibrium is re-solved per sample.
    equilibrium = []
    for row in state:
        q = np.zeros(8)
        q[rows], q[list(PASSIVE_INDICES)], q[TOOL_INDEX] = row[0:5], row[5:7], q_tool
        equilibrium.append(planner.model.passive_equilibrium(q[list(ACTUATED_INDICES)]))
    return {
        "time": np.array(times),
        "q_u": state[:, 5:7],
        "dq_u": state[:, 12:14],
        "q_u_eq": np.array(equilibrium),
        "tracking": np.array(tracking),
    }, plant


def report(plan, rolled: dict) -> list[str]:
    """Say what the two models make of the same motion, side by side."""
    predicted = plan.timing.q_u - plan.timing.q_u_eq
    actual = rolled["q_u"] - rolled["q_u_eq"]
    arrival = int(np.searchsorted(rolled["time"], plan.duration))

    def pair(values) -> str:
        return "{:9.3f}  {:9.3f}".format(*np.degrees(values))

    return [
        f"plan           {plan.duration:.2f} s, {plan.timing.iterations} SQP"
        f" iterations, {plan.timing.solve_time_s:.3f} s in the solver",
        "",
        "                       tip       tilt   [deg]",
        f"peak sway  OCP   {pair(np.max(np.abs(predicted), axis=0))}",
        f"peak sway  mjc   {pair(np.max(np.abs(actual), axis=0))}",
        f"at arrival OCP   {pair(predicted[-1])}",
        f"at arrival mjc   {pair(actual[arrival])}",
        f"after settle     {pair(actual[-1])}",
        f"settled rate     {pair(rolled['dq_u'][-1])}  [deg/s]",
        "",
        f"drive error    {np.max(np.abs(rolled['tracking'])):.2e} worst axis and"
        " instant; raise --bandwidth if that is the size of the sway",
    ]


def figure(plan, rolled: dict):
    import matplotlib.pyplot as plt

    predicted = plan.timing.q_u - plan.timing.q_u_eq
    actual = rolled["q_u"] - rolled["q_u_eq"]
    t = rolled["time"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for index, label in enumerate(("tip", "tilt")):
        colour = f"C{index}"
        for cell, ocp, mjc in (
            (axes[0, 0], predicted[:, index], actual[:, index]),
            (axes[0, 1], plan.timing.dq_u[:, index], rolled["dq_u"][:, index]),
        ):
            cell.plot(plan.timing.time, ocp, colour, ls="--", label=f"{label} OCP")
            cell.plot(t, mjc, colour, label=f"{label} MuJoCo")
    axes[0, 0].set_ylabel(r"sway offset $q_u - q_u^{eq}$ [rad]")
    axes[0, 1].set_ylabel("sway rate [rad/s]")
    for cell in axes[0]:
        cell.axvline(plan.duration, color="k", lw=0.8, ls=":")

    for axis, index in enumerate(PLANNED_INDICES):
        axes[1, 0].plot(t, rolled["tracking"][:, axis], label=f"q{index + 1}")
    axes[1, 0].set_ylabel("drive error, plant - reference [rad, m]")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(report(plan, rolled)),
        family="monospace",
        fontsize=8,
        va="top",
        transform=axes[1, 1].transAxes,
    )
    for cell in axes.flat:
        cell.grid(alpha=0.3)
        cell.set_xlabel("time [s]")
        if cell.get_legend_handles_labels()[0]:
            cell.legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"plan of {plan.duration:.2f} s on MuJoCo; dotted = the OCP's own"
        " prediction, solid = the plant"
    )
    fig.tight_layout()
    return fig


def main() -> int:
    options = arguments()
    description, planner, start, plan = plan_for(options)
    rolled, plant = roll(planner, description, plan, start, options)
    for line in report(plan, rolled):
        print(line)

    if not options.show:
        matplotlib.use("Agg")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    figure(plan, rolled).savefig(options.output, dpi=150)
    print(f"wrote {options.output}")
    if options.show:
        import matplotlib.pyplot as plt

        plt.show()
    if options.viewer:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(leave(main()))
