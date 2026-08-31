#!/usr/bin/env python3
"""
Plan one motion offline and plot what it asks of the machine.

No ROS, no graph, no clock: this runs the deployment's own `Planner` against
the checked-in machine descriptions, so what it tunes is what the node solves.
It exists because the timing OCP has weights, and a weight is tuned by looking
at the profile it produces -- traversal time against sway against the actuated
force it asks for and how close the pump comes to its limit.

    ./scripts/plan_example.py --tool pzs100 --sway-weight 8 --show

The goal defaults to a joint-space target that is put through the real inverse
kinematics as a Cartesian pose, so a run exercises every stage rather than
skipping to the one being tuned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

from crane_model import Frame, Payload, Tool  # noqa: E402
from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402
from crane_planning.planner import (  # noqa: E402
    ACTUATED_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    yaw_of,
)

#: The two expanded descriptions `crane_model` keeps as its own fixtures.
DESCRIPTIONS = {
    Tool.PZS100: "pzs100.urdf",
    Tool.EPSILON_7040: "epsilon_7040.urdf",
}
#: A lift: out, up and further out, with the rotator following.
DEFAULT_START = (0.0, -0.2, 0.4, 1.0, 0.0)
DEFAULT_GOAL = (0.6, -0.2, 0.9, 1.4, 0.4)
DEFAULT_TOOL_POSITION = {Tool.PZS100: 0.30, Tool.EPSILON_7040: 0.20}


def description(tool: Tool) -> str:
    path = PACKAGE.parent / "crane_model" / "test" / "description" / DESCRIPTIONS[tool]
    if not path.is_file():
        raise SystemExit(f"no machine description at {path}")
    return path.read_text()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool", choices=[tool.value for tool in Tool], default=Tool.PZS100.value
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=5,
        default=DEFAULT_START,
        help="the five planned joint coordinates to leave from",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=5,
        default=DEFAULT_GOAL,
        help="a joint configuration; its tool pose is what the IK is asked for",
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "YAW"),
        help="a tool pose in K0_mounting_base instead, which may be unreachable",
    )
    parser.add_argument("--tool-position", type=float, default=None)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)

    tuning = parser.add_argument_group("the timing OCP")
    tuning.add_argument("--kappa", type=float, default=None)
    tuning.add_argument("--intervals", type=int, default=None)
    tuning.add_argument("--sway-weight", type=float, default=None)
    tuning.add_argument("--tau-weight", type=float, default=None)
    tuning.add_argument("--terminal-sway-weight", type=float, default=None)
    tuning.add_argument("--input-weight", type=float, default=None)
    tuning.add_argument("--sigma-accel-max", type=float, default=None)
    tuning.add_argument("--sigma-rate-max", type=float, default=None)
    tuning.add_argument("--max-iterations", type=int, default=None)

    clearance = parser.add_argument_group('what "clear" means, in metres')
    clearance.add_argument("--margin-safety", type=float, default=None)
    clearance.add_argument("--margin-interp", type=float, default=None)

    parser.add_argument("--output", type=Path, default=Path("plan_example.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def configure(options) -> PlannerConfig:
    config = PlannerConfig(tool=Tool(options.tool))
    for name in (
        "kappa",
        "intervals",
        "sway_weight",
        "tau_weight",
        "terminal_sway_weight",
        "input_weight",
        "sigma_accel_max",
        "sigma_rate_max",
        "max_iterations",
        "margin_safety",
        "margin_interp",
    ):
        value = getattr(options, name)
        if value is not None:
            setattr(config, name, value)
    return config


def figure(planner: Planner, plan, options):
    """Eight panels, all against time: what the machine does and what it costs."""
    import matplotlib.pyplot as plt

    timing = plan.timing
    limits, config = planner.limits, planner.config
    t = timing.time
    names = [f"q{index + 1}" for index in PLANNED_INDICES]

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)

    for axis in range(len(PLANNED_INDICES)):
        axes[0, 0].plot(t, timing.q_a[:, axis], label=names[axis])
        axes[0, 1].plot(t, timing.dq_a[:, axis], label=names[axis])
        axes[1, 0].plot(t, timing.ddq_a[:, axis], label=names[axis])
        axes[3, 0].plot(t, timing.tau[:, axis] * 1e-3, label=names[axis])
        colour = axes[0, 1].lines[-1].get_color()
        bound = config.kappa * options.speed_scale * limits.dq_max[axis]
        for sign in (1.0, -1.0):
            axes[0, 1].axhline(sign * bound, color=colour, ls=":", lw=0.7)
            axes[1, 0].axhline(
                sign * config.kappa * config.ddq_a_max[axis],
                color=colour,
                ls=":",
                lw=0.7,
            )
    axes[0, 0].set_ylabel("position [rad, m]")
    axes[0, 1].set_ylabel("velocity [rad/s, m/s]")
    axes[1, 0].set_ylabel("acceleration")
    # No limit lines: `tau_a` is priced by the OCP's cost and not bounded, so
    # there is no force limit left to draw it against.
    axes[3, 0].set_ylabel(r"actuated force $\tau_a$ [kN$\,$m]")
    axes[3, 0].set_title("priced, not bounded", fontsize=8, loc="left")

    axes[1, 1].plot(t, timing.sigma_dot, label=r"$\dot\sigma$")
    axes[1, 1].plot(t, timing.sigma_ddot, label=r"$\ddot\sigma$")
    twin = axes[1, 1].twinx()
    twin.plot(t, timing.sigma, color="0.6", ls="--")
    twin.set_ylabel(r"$\sigma$", color="0.5")
    axes[1, 1].set_ylabel("path rate")

    offset = timing.q_u - timing.q_u_eq
    for index, label in enumerate(("tip", "tilt")):
        axes[2, 0].plot(t, offset[:, index], label=label)
        axes[2, 1].plot(t, timing.dq_u[:, index], label=label)
        colour = axes[2, 0].lines[-1].get_color()
        for sign in (1.0, -1.0):
            axes[2, 0].axhline(
                sign * config.q_sway_max[index], color=colour, ls=":", lw=0.7
            )
            axes[2, 1].axhline(
                sign * config.dq_sway_max[index], color=colour, ls=":", lw=0.7
            )
    axes[2, 0].set_ylabel(r"sway offset $q_u - q_u^{eq}$ [rad]")
    axes[2, 1].set_ylabel("sway rate [rad/s]")

    axes[3, 1].plot(t, timing.pump_flow * 1e3, color="tab:red", label="sum")
    axes[3, 1].plot(
        t, timing.flow_slack * 1e3, color="tab:orange", ls="--", label="slack"
    )
    axes[3, 1].axhline(
        config.kappa * limits.flow_max * 1e3, color="tab:red", ls=":", lw=0.7
    )
    axes[3, 1].set_ylabel("pump flow [L/s]")

    for row in axes:
        for cell in row:
            cell.grid(alpha=0.3)
            if cell.get_legend_handles_labels()[0]:
                cell.legend(fontsize=7, ncol=3)
    for cell in axes[3]:
        cell.set_xlabel("time [s]")

    fig.suptitle(
        f"{planner.config.tool.value}: {timing.duration:.2f} s, "
        f"{timing.iterations} IPOPT iterations in {timing.solve_time_s:.2f} s, "
        f"sway weight {config.sway_weight}, tau weight {config.tau_weight}, "
        f"kappa {config.kappa}; arrives "
        f"{np.degrees(timing.terminal_sway):.2f} deg off rest"
    )
    fig.tight_layout()
    return fig


def main() -> int:
    options = arguments()
    tool = Tool(options.tool)
    config = configure(options)
    planner = Planner(description(tool), config)

    q_tool = (
        options.tool_position
        if options.tool_position is not None
        else DEFAULT_TOOL_POSITION[tool]
    )
    q = np.zeros(8)
    q[list(PLANNED_INDICES)] = options.start
    q[TOOL_INDEX] = q_tool
    q[[4, 5]] = planner.model.passive_equilibrium(q[list(ACTUATED_INDICES)])
    start = Start(q=q, dq_a=np.zeros(5))

    if options.goal_pose is not None:
        position, yaw = np.array(options.goal_pose[:3]), options.goal_pose[3]
    else:
        goal = np.zeros(8)
        goal[list(PLANNED_INDICES)] = options.goal
        goal[TOOL_INDEX] = q_tool
        goal[[4, 5]] = planner.model.passive_equilibrium(goal[list(ACTUATED_INDICES)])
        pose = planner.model.forward_kinematics(goal, Frame.MOUNTING_BASE, Frame.TCP)
        position = pose.position_m
        import pinocchio as pin

        yaw = yaw_of(
            pin.XYZQUATToSE3(
                np.concatenate([pose.position_m, pose.orientation_xyzw])
            ).rotation
        )
    print(
        f"goal: [{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}] m, "
        f"yaw {np.degrees(yaw):.1f} deg"
    )

    payload = Payload(mass_kg=options.payload_mass, valid=options.payload_mass > 0.0)
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

    print(plan.message)
    print(
        f"terminal sway {np.degrees(plan.timing.terminal_sway):.2f} deg, "
        f"peak pump slack {max(0.0, np.max(plan.timing.flow_slack)) * 1e3:.3f} L/s"
    )
    if options.csv is not None:
        header = "time_s," + ",".join(
            [f"q{index + 1}" for index in ACTUATED_INDICES]
            + [f"dq{index + 1}" for index in ACTUATED_INDICES]
        )
        np.savetxt(
            options.csv,
            np.column_stack([plan.time, plan.q_a, plan.dq_a]),
            delimiter=",",
            header=header,
            comments="",
        )
        print(f"wrote {options.csv}")

    if not options.show:
        matplotlib.use("Agg")
    fig = figure(planner, plan, options)
    fig.savefig(options.output, dpi=150)
    print(f"wrote {options.output}")
    if options.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
