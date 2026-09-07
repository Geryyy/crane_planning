#!/usr/bin/env python3
"""
Plan one motion offline and plot it. No ROS, no graph, no clock.

Runs the deployment's own `Planner` against the checked-in description, so
what is tuned here is what the node solves. Judge timing and clearance
settings from the profile: traversal time, sway, actuated force, pump draw.

    ./scripts/plan_example.py --duration-step 0.5 --show

Goal defaults to a joint target pushed through the real IK as a Cartesian
pose, so a run exercises every stage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

from crane_model import Frame, Payload  # noqa: E402
from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402
from crane_planning.planner import (  # noqa: E402
    ACTUATED_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    yaw_of,
)

#: the expanded description crane_model keeps as a fixture
DESCRIPTION = "pzs100.urdf"
#: a lift: out, up, further out, rotator following
DEFAULT_START = (0.0, -0.2, 0.4, 1.0, 0.0)
DEFAULT_GOAL = (0.6, -0.2, 0.9, 1.4, 0.4)
DEFAULT_TOOL_POSITION = 0.30


def description() -> str:
    path = PACKAGE.parent / "crane_model" / "test" / "description" / DESCRIPTION
    if not path.is_file():
        raise SystemExit(f"no machine description at {path}")
    return path.read_text()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=float,
        nargs=5,
        default=DEFAULT_START,
        help="five planned joint coords to start from",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=5,
        default=DEFAULT_GOAL,
        help="joint config; its tool pose is the IK target",
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "YAW"),
        help="tool pose in K0_mounting_base instead; may be unreachable",
    )
    parser.add_argument("--tool-position", type=float, default=None)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)

    tuning = parser.add_argument_group("timing")
    tuning.add_argument("--kappa", type=float, default=None)
    tuning.add_argument("--ocp-duration-max", type=float, default=None)
    tuning.add_argument(
        "--ocp-integrator",
        choices=("ERK", "IRK"),
        default=None,
        help="ERK4 or 2-stage Gauss-Legendre IRK; rebuilds the solver",
    )

    clearance = parser.add_argument_group('what "clear" means, in metres')
    clearance.add_argument("--margin-safety", type=float, default=None)
    clearance.add_argument("--margin-interp", type=float, default=None)

    parser.add_argument("--output", type=Path, default=Path("plan_example.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def configure(options) -> PlannerConfig:
    config = PlannerConfig()
    for name in (
        "kappa",
        "ocp_duration_max",
        "ocp_integrator",
        "margin_safety",
        "margin_interp",
    ):
        value = getattr(options, name)
        if value is not None:
            setattr(config, name, value)
    return config


def bounds(planner: Planner, speed_scale: float) -> dict:
    """
    Every bound `TrajectoryOcp.solve` writes onto the solver, in one place.

    Limit lines and normalisation both read it, so no plot can claim a bound
    the solve did not use. `speed_scale` enters every row once: it is the
    reference server's `slow_down` divider, which divides rate, acceleration and
    jerk alike, and not a time reparametrisation -- see `TrajectoryOcp.solve`.
    The command row is the exception and carries `kappa` alone: `slow_down` asks
    for a gentler move, it does not shrink the valve's domain.

    No `dddq_a` entry: nothing bounds the third derivative on its own any more.
    What it is spent on is the command, which is a row of its own.
    """
    limits, config = planner.limits, planner.config
    return {
        "dq_a": config.kappa * speed_scale * limits.dq_max,
        "ddq_a": config.kappa * speed_scale * config.ddq_a_max,
        "command": (
            config.kappa * config.command_u_min,
            config.kappa * config.command_u_max,
        ),
        "flow": config.kappa * speed_scale * limits.flow_max / config.pump_flow_max,
    }


def usage(planner: Planner, plan, speed_scale: float) -> dict:
    """
    Each constrained row over its own bound, worst axis per instant.

    `ocp.py` divides every residual and `h` row by its own limit, so 1.0 means
    "at the bound" everywhere and the rows are mutually comparable (the
    physical panels are not). The row touching 1.0 decided the duration; all
    rows well under means `ocp_duration_max` or the weights decided, not the
    machine.
    """
    limits, config, timing = planner.limits, planner.config, plan.timing
    limit = bounds(planner, speed_scale)
    box = limits.bounded  # continuous joint: no range to be a fraction of
    centre = 0.5 * (limits.upper[box] + limits.lower[box])
    half = 0.5 * (limits.upper[box] - limits.lower[box])
    # Last node is not on the stage box: `solve` writes `h_e` there with
    # `terminal_q_sway_max`, 10x tighter. Normalising the whole trace against the
    # stage bound would plot the deciding sample at a tenth of where it sits.
    sway_bound = np.tile(config.q_sway_max, (timing.time.size, 1))
    rate_bound = np.tile(config.dq_sway_max, (timing.time.size, 1))
    sway_bound[-1] = config.terminal_q_sway_max
    rate_bound[-1] = config.terminal_dq_sway_max
    return {
        "q_a in box": np.max(np.abs((timing.q_a[:, box] - centre) / half), axis=1),
        "dq_a": np.max(np.abs(timing.dq_a / limit["dq_a"]), axis=1),
        r"u = $\ddot q_a$": np.max(np.abs(timing.ddq_a / limit["ddq_a"]), axis=1),
        "sway": np.max(np.abs((timing.q_u - timing.q_u_eq) / sway_bound), axis=1),
        "dq_u": np.max(np.abs(timing.dq_u / rate_bound), axis=1),
        "pump": timing.pump_flow / limit["flow"],
        "C3 command": np.max(
            np.abs(
                timing.command
                / np.where(
                    timing.command < 0, -limit["command"][0], limit["command"][1]
                )
            ),
            axis=1,
        ),
    }


def solver_report(planner: Planner, plan, elapsed: float) -> list:
    """
    Report what the solve did, not what it answered.

    Wall clock is no regression signal: same problem, 0.53 s idle vs 4.97 s
    inside a running Gazebo. Iterations and the four residuals against
    `ocp_tolerance` do not move with load.
    """
    config, timing = planner.config, plan.timing
    stat, eq, ineq, comp = timing.residuals
    # Element-wise, then worst: tip and tilt are different axes, tolerances need
    # not stay equal. max offset over max tolerance would compare tip excursion
    # against tilt allowance.
    terminal = max(
        np.max(np.abs(timing.q_u[-1] - timing.q_u_eq[-1]) / config.terminal_q_sway_max),
        np.max(np.abs(timing.dq_u[-1]) / config.terminal_dq_sway_max),
    )
    return [
        f"acados SQP, Gauss-Newton, HPIPM, {config.ocp_integrator},"
        f" N = {config.ocp_intervals}",
        f"iterations     {timing.iterations} of {config.ocp_max_iterations}",
        f"residuals      stat {stat:.1e}   eq {eq:.1e}",
        f"               ineq {ineq:.1e}   comp {comp:.1e}",
        f"tolerance      {config.ocp_tolerance:.0e} on each of the four",
        f"soft slack     {timing.slack:.2e} at {config.ocp_slack_price:.0e} per unit",
        f"horizon        theta {timing.duration / config.ocp_horizon:.3f}"
        f" of {config.ocp_horizon:.1f} s, duration in"
        f" [{config.ocp_duration_min:.0f}, {config.ocp_duration_max:.0f}] s",
        f"duration       {timing.duration:.2f} s",
        f"arrives        {np.degrees(timing.terminal_sway):.2f} deg off rest,"
        f" worst terminal row {terminal:.2f} of its own bound",
        f"solve          {timing.acados_time_s:.3f} s in acados,"
        f" {timing.solve_time_s:.3f} s wall",
        f"planner call   {elapsed:.3f} s end to end",
    ]


def figure(planner: Planner, plan, options, elapsed: float):
    """Physical units, the same rows over their own bounds, and the solve."""
    import matplotlib.pyplot as plt

    timing = plan.timing
    config = planner.config
    limit = bounds(planner, options.speed_scale)
    t = timing.time
    names = [f"q{index + 1}" for index in PLANNED_INDICES]

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)

    for axis in range(len(PLANNED_INDICES)):
        axes[0, 0].plot(t, timing.q_a[:, axis], label=names[axis])
        axes[0, 1].plot(t, timing.dq_a[:, axis], label=names[axis])
        axes[1, 0].plot(t, timing.ddq_a[:, axis], label=names[axis])
        colour = axes[0, 1].lines[-1].get_color()
        for sign in (1.0, -1.0):
            axes[0, 1].axhline(sign * limit["dq_a"][axis], color=colour, ls=":", lw=0.7)
            axes[1, 0].axhline(
                sign * limit["ddq_a"][axis], color=colour, ls=":", lw=0.7
            )
    # A far-off limit line must not set the scale: rotator bounds sit an order of
    # magnitude above anything these plans use and flatten the other four axes.
    # Closeness to a bound is read off the normalised panel, not here.
    for cell, rows in ((axes[0, 1], timing.dq_a), (axes[1, 0], timing.ddq_a)):
        span = max(float(np.max(np.abs(rows))), 1.0e-6)
        cell.set_ylim(-1.2 * span, 1.2 * span)
    axes[0, 0].set_ylabel("position [rad, m]")
    axes[0, 1].set_ylabel("velocity [rad/s, m/s]")
    # `u` is the OCP's input, not a derived quantity; its box is hard.
    axes[1, 0].set_ylabel(r"input $u = \ddot q_a$")
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

    # `pump_flow` is already a fraction of the physical pump (the row the OCP
    # constrains); the reservation is what it may actually use.
    axes[1, 1].plot(t, timing.pump_flow, color="tab:red", label="sum")
    axes[1, 1].axhline(
        limit["flow"], color="tab:red", ls=":", lw=0.7, label="reservation"
    )
    axes[1, 1].set_ylabel("pump draw / physical limit")

    for label, fraction in usage(planner, plan, options.speed_scale).items():
        axes[3, 0].plot(t, fraction, label=label)
    axes[3, 0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[3, 0].set_ylabel("worst axis / its own bound")
    axes[3, 0].set_ylim(0.0, max(1.05, axes[3, 0].get_ylim()[1]))

    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.0,
        1.0,
        "\n".join(solver_report(planner, plan, elapsed)),
        family="monospace",
        fontsize=8,
        va="top",
        transform=axes[3, 1].transAxes,
    )

    for row in axes:
        for cell in row:
            cell.grid(alpha=0.3)
            if cell.get_legend_handles_labels()[0]:
                cell.legend(fontsize=7, ncol=3)
    # Solver panel is text, so the sway-rate panel above it ends its column and
    # keeps its own axis.
    axes[2, 1].tick_params(labelbottom=True)
    for cell in (axes[2, 1], axes[3, 0]):
        cell.set_xlabel("time [s]")

    fig.suptitle(
        f"{timing.duration:.2f} s, "
        f"{timing.iterations} SQP iterations, kappa {config.kappa}; arrives "
        f"{np.degrees(timing.terminal_sway):.2f} deg off rest"
    )
    fig.tight_layout()
    return fig


def main() -> int:
    options = arguments()
    config = configure(options)
    planner = Planner(description(), config)

    q_tool = (
        options.tool_position
        if options.tool_position is not None
        else DEFAULT_TOOL_POSITION
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
    planning_started = time.monotonic()
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
    planning_elapsed = time.monotonic() - planning_started

    print(plan.message)
    for line in solver_report(planner, plan, planning_elapsed):
        print(line)
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
    fig = figure(planner, plan, options, planning_elapsed)
    fig.savefig(options.output, dpi=150)
    print(f"wrote {options.output}")
    if options.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
