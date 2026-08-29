#!/usr/bin/env python3
"""
Solve one crane timing OCP offline and plot it.

This is a ROS-free developer tuning tool for the time parametrization of
``wiki/trajectory_planning.md`` §5.2.

**It contains no optimal control problem.** It builds an argument list, runs
``crane_planning_timing_cli`` -- which calls the same ``solve_timing_ocp`` the
planner node calls -- and plots the CSV that comes back. There is deliberately
no second solver setup here: the scaled bounds, the per-node passive
equilibrium, the stage parameter blocks and the bisected warm start all live in
``src/timing_ocp.cpp`` and are reached, not reimplemented. A weight tuned here
is a weight the deployment uses, and every default below is the C++ default
because the CLI fills in whatever this script does not pass.

The path is fitted by the planner's own ``fit_c2_path``, so what is timed is the
shape the deployment actually produces rather than a hand-written polynomial.

Needs numpy, matplotlib and pyyaml. It does **not** need casadi or acados: the
solver it drives is the compiled one.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
MODEL_PACKAGE = PACKAGE.parent / "crane_model"
BINARY = "crane_planning_timing_cli"

AXIS_NAMES = ("slew", "boom", "arm", "telescope", "rotator")
PASSIVE_NAMES = ("sway_1", "sway_2")
DESCRIPTIONS = {"pzs100": "pzs100.urdf", "epsilon7040": "epsilon_7040.urdf"}
DEFAULT_TOOL_POSITION = {"pzs100": 0.30, "epsilon7040": 0.20}
DEFAULT_A = (0.0, -0.2, 0.4, 1.0, 0.0)
DEFAULT_B = (0.0, -0.2, 0.9, 1.4, 0.0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and plot the crane timing OCP on one fitted example path.",
        epilog="Every tuning option left unset takes its default from "
        "TimingOcpSettings in include/crane_planning/timing_ocp.hpp, which is "
        "also where crane_planner.yaml's own defaults come from.",
    )
    parser.add_argument("--tool", choices=sorted(DESCRIPTIONS), default="pzs100")
    parser.add_argument(
        "--description",
        type=Path,
        default=None,
        help="machine description; defaults to crane_model's checked-in one for --tool",
    )
    parser.add_argument(
        "--a",
        nargs=5,
        type=float,
        default=DEFAULT_A,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR"),
        help="start pose; radians except telescope in metres",
    )
    parser.add_argument(
        "--b",
        nargs=5,
        type=float,
        default=DEFAULT_B,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR"),
        help="goal pose; radians except telescope in metres",
    )
    parser.add_argument(
        "--via",
        nargs=5,
        type=float,
        action="append",
        default=None,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR"),
        help="an interior waypoint, repeatable; the fit is C2 across it",
    )
    parser.add_argument(
        "--tool-position",
        type=float,
        default=None,
        help="fixed tool coordinate; defaults to 0.30 for PZS100 and 0.20 for Epsilon",
    )
    parser.add_argument("--payload-mass", type=float, default=None)
    parser.add_argument(
        "--payload-com",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="payload COM in K8, metres",
    )

    # Every one of these is passed through only when given, so an unset option is
    # the C++ default rather than a copy of it living in this file.
    tuning = parser.add_argument_group("OCP tuning (unset means the deployed default)")
    tuning.add_argument("--kappa", type=float)
    tuning.add_argument("--intervals", type=int)
    tuning.add_argument("--sway-weight", type=float)
    tuning.add_argument("--input-weight", type=float)
    tuning.add_argument("--sigma-rate-min", type=float)
    tuning.add_argument("--sigma-rate-max", type=float)
    tuning.add_argument("--sigma-accel-max", type=float)
    tuning.add_argument(
        "--q-u-max",
        nargs=2,
        type=float,
        metavar=("Q1", "Q2"),
        help="passive position bounds, rad",
    )
    tuning.add_argument(
        "--dq-u-max",
        nargs=2,
        type=float,
        metavar=("DQ1", "DQ2"),
        help="passive velocity bounds, rad/s",
    )
    tuning.add_argument(
        "--ddq-a-max",
        nargs=6,
        type=float,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR", "TOOL"),
        help="actuated acceleration limits",
    )
    tuning.add_argument("--max-iterations", type=int)
    tuning.add_argument("--max-wall-clock", type=float)
    tuning.add_argument("--levenberg-marquardt", type=float)
    tuning.add_argument("--tolerance-stationarity", type=float)
    tuning.add_argument("--tolerance-feasibility", type=float)
    tuning.add_argument("--sample-period", type=float)
    tuning.add_argument("--speed-scale", type=float, help="velocity margin in (0, 1]")

    start = parser.add_argument_group(
        "measured start (§7's re-plan from a moving machine; unset is the stopped start)"
    )
    start.add_argument(
        "--start-q-u",
        nargs=2,
        type=float,
        metavar=("Q1", "Q2"),
        help="measured sway angle at the seam, rad. **Absolute passive "
        "coordinates**, as the passive pair on /joint_states carries them and as "
        "the node passes them -- not an offset from the hanging pose. They must lie inside "
        "q_u_max of the equilibrium at the start pose, which the node CSV's "
        "q_eq_sway_* columns report.",
    )
    start.add_argument(
        "--start-dq-u",
        nargs=2,
        type=float,
        metavar=("DQ1", "DQ2"),
        help="measured sway rate at the seam, rad/s",
    )
    start.add_argument(
        "--start-sigma-rate",
        type=float,
        help="pin sigma_dot(0) to this, reproducing the measured actuated velocity",
    )

    shape = parser.add_argument_group("path fit shape")
    shape.add_argument("--path-rate-headroom", type=float)
    shape.add_argument("--path-acceleration-span", type=float)
    shape.add_argument("--path-jerk-span", type=float)

    hydraulics = parser.add_argument_group("hydraulic limit overrides")
    hydraulics.add_argument("--pump-flow-max", type=float)
    hydraulics.add_argument("--pump-flow-planning-factor", type=float)
    hydraulics.add_argument("--system-pressure-pa", type=float)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timing_a2b.png"),
        help="plot path; CSV is written beside it. Relative to the working directory.",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="delete the CSV after plotting"
    )
    parser.add_argument(
        "--reference-csv",
        action="store_true",
        help="also write the emitted 25 Hz reference -- what actually reaches "
        "/crane/reference -- beside the node CSV as <output>.reference.csv",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help=f"path to {BINARY}; found on AMENT_PREFIX_PATH when unset",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the CLI invocation before running it",
    )
    return parser.parse_args()


def find_binary(override: Path | None) -> Path:
    """Locate the compiled CLI without depending on ament_index_python."""
    if override is not None:
        if not os.access(override, os.X_OK):
            raise RuntimeError(f"{override} is not an executable file")
        return override
    from_env = os.environ.get("CRANE_PLANNING_TIMING_CLI")
    if from_env:
        return Path(from_env)
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "lib" / "crane_planning" / BINARY
        if os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(BINARY)
    if found:
        return Path(found)
    raise RuntimeError(
        f"{BINARY} not found. Build the package and source the workspace, or pass "
        f"--binary. It is installed to lib/crane_planning/{BINARY}."
    )


def hydraulic_defaults() -> dict:
    """Return the two hydraulic numbers, from the file that carries their evidence."""
    import yaml

    path = PACKAGE / "config" / "hydraulic_limits.yaml"
    with open(path) as stream:
        root = yaml.safe_load(stream)
    return root["crane_planner"]["ros__parameters"]


def build_command(arguments: argparse.Namespace, binary: Path) -> list[str]:
    tool_position = arguments.tool_position
    if tool_position is None:
        tool_position = DEFAULT_TOOL_POSITION[arguments.tool]
    description = arguments.description
    if description is None:
        description = (
            MODEL_PACKAGE / "test" / "description" / DESCRIPTIONS[arguments.tool]
        )
    if not Path(description).is_file():
        raise RuntimeError(f"machine description not found: {description}")

    command = [
        str(binary),
        "--description",
        str(description),
        "--tool",
        arguments.tool,
        "--q8",
        repr(float(tool_position)),
    ]
    for waypoint in [arguments.a, *(arguments.via or []), arguments.b]:
        command += ["--waypoint", ",".join(repr(float(value)) for value in waypoint)]

    hydraulics = hydraulic_defaults()
    passthrough = {
        "kappa": arguments.kappa,
        "intervals": arguments.intervals,
        "sway-weight": arguments.sway_weight,
        "input-weight": arguments.input_weight,
        "sigma-rate-min": arguments.sigma_rate_min,
        "sigma-rate-max": arguments.sigma_rate_max,
        "sigma-accel-max": arguments.sigma_accel_max,
        "max-iterations": arguments.max_iterations,
        "max-wall-clock": arguments.max_wall_clock,
        "levenberg-marquardt": arguments.levenberg_marquardt,
        "tolerance-stationarity": arguments.tolerance_stationarity,
        "tolerance-feasibility": arguments.tolerance_feasibility,
        "sample-period": arguments.sample_period,
        "speed-scale": arguments.speed_scale,
        "payload-mass": arguments.payload_mass,
        "start-sigma-rate": arguments.start_sigma_rate,
        "path-rate-headroom": arguments.path_rate_headroom,
        "path-acceleration-span": arguments.path_acceleration_span,
        "path-jerk-span": arguments.path_jerk_span,
        # An override wins; otherwise the number and its provenance come from
        # config/hydraulic_limits.yaml rather than from a default in this file.
        "pump-flow-max": (
            arguments.pump_flow_max
            if arguments.pump_flow_max is not None
            else hydraulics["pump_flow_max"]
        ),
        "pump-flow-planning-factor": (
            arguments.pump_flow_planning_factor
            if arguments.pump_flow_planning_factor is not None
            else hydraulics["pump_flow_planning_factor"]
        ),
        "system-pressure-pa": (
            arguments.system_pressure_pa
            if arguments.system_pressure_pa is not None
            else hydraulics["system_pressure_pa"]
        ),
    }
    for name, value in passthrough.items():
        if value is not None:
            command += [
                f"--{name}",
                repr(value) if isinstance(value, float) else str(value),
            ]

    for name, values in (
        ("q-u-max", arguments.q_u_max),
        ("dq-u-max", arguments.dq_u_max),
        ("ddq-a-max", arguments.ddq_a_max),
        ("payload-com", arguments.payload_com),
        ("start-q-u", arguments.start_q_u),
        ("start-dq-u", arguments.start_dq_u),
    ):
        if values is not None:
            command += [f"--{name}", ",".join(repr(float(value)) for value in values)]
    return command


def run_solver(command: list[str], csv_path: Path, reference_path: Path | None) -> dict:
    """Run the CLI and return its `key=value` summary. Its refusal is ours."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["--output", str(csv_path)]
    if reference_path is not None:
        extra += ["--reference-output", str(reference_path)]
    finished = subprocess.run(
        command + extra,
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            finished.stderr.strip() or f"{BINARY} exited {finished.returncode}"
        )
    summary = {}
    for line in finished.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            summary[key.strip()] = value.strip()
    return summary


def read_csv(path: Path) -> dict:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        headings = next(reader)
        rows = [[float(field) for field in row] for row in reader]
    if not rows:
        raise RuntimeError(f"{path} holds no samples")
    table = np.asarray(rows)
    return {name: table[:, index] for index, name in enumerate(headings)}


def block(data: dict, prefix: str, names) -> np.ndarray:
    return np.column_stack([data[f"{prefix}{name}"] for name in names])


def plot(path: Path, data: dict, summary: dict, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    force_limit = np.asarray(
        [float(value) for value in summary["cylinder_force_limit_N"].split(",")]
    )
    flow_limit = float(summary["pump_flow_limit_m3_s"])
    time = data["time_s"]
    q_a = block(data, "q_", AXIS_NAMES)
    dq_a = block(data, "dq_", AXIS_NAMES)
    ddq_a = block(data, "ddq_", AXIS_NAMES)
    q_u = block(data, "q_", PASSIVE_NAMES)
    dq_u = block(data, "dq_", PASSIVE_NAMES)
    q_u_eq = block(data, "q_eq_", PASSIVE_NAMES)
    force = np.column_stack([data[f"cylinder_force_{name}_N"] for name in AXIS_NAMES])

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)

    for index, name in enumerate(AXIS_NAMES):
        axes[0, 0].plot(time, q_a[:, index], color=colors[index], label=name)
        axes[0, 1].plot(time, dq_a[:, index], color=colors[index], label=name)
        axes[1, 0].plot(time, ddq_a[:, index], color=colors[index], label=name)
        axes[3, 0].plot(time, force[:, index] / 1000.0, color=colors[index], label=name)
        for sign in (1.0, -1.0):
            axes[3, 0].axhline(
                sign * force_limit[index] / 1000.0,
                color=colors[index],
                linestyle="--",
                alpha=0.45,
            )
    axes[0, 0].set_title("Timed actuated trajectory")
    axes[0, 0].set_ylabel("position [rad / m]")
    axes[0, 1].set_title("Actuated velocity")
    axes[0, 1].set_ylabel("velocity [rad/s / m/s]")
    axes[1, 0].set_title("Actuated acceleration")
    axes[1, 0].set_ylabel("acceleration [rad/s² / m/s²]")
    axes[3, 0].set_title("Cylinder force (dashed: kappa × limit)")
    axes[3, 0].set_ylabel("force [kN]")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[3, 0]):
        axis.legend(ncol=3, fontsize=8)

    axes[1, 1].plot(time, data["sigma_rate"], label="sigma_dot")
    axes[1, 1].plot(time, data["sigma_accel"], label="sigma_ddot (differentiated)")
    axes[1, 1].set_title("Timing variables")
    axes[1, 1].set_ylabel("rate [1/s], accel [1/s^2]")
    # sigma itself is dimensionless 0..1 and would sit flat against the rates, so
    # it gets its own axis. It is the node grid: where it flattens, the solver
    # spent time; where it steepens, the path was cheap.
    sigma_axis = axes[1, 1].twinx()
    sigma_axis.plot(
        time, data["sigma"], color="tab:gray", linestyle=":", lw=2, label="sigma"
    )
    sigma_axis.set_ylabel("sigma [-]")
    sigma_axis.set_ylim(0.0, 1.0)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    sigma_handles, sigma_labels = sigma_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + sigma_handles, labels + sigma_labels)

    for index, name in enumerate(PASSIVE_NAMES):
        axes[2, 0].plot(
            time,
            q_u[:, index] - q_u_eq[:, index],
            color=colors[index],
            label=f"{name} offset",
        )
        axes[2, 1].plot(time, dq_u[:, index], color=colors[index], label=f"{name} rate")
    axes[2, 0].set_title("Passive sway offset")
    axes[2, 0].set_ylabel("angle [rad]")
    axes[2, 0].legend()
    axes[2, 1].set_title("Passive sway rate")
    axes[2, 1].set_ylabel("rate [rad/s]")
    axes[2, 1].legend()

    axes[3, 1].plot(
        time, data["pump_flow_m3_s"] * 1.0e3, color="black", lw=2, label="pump flow"
    )
    axes[3, 1].axhline(
        flow_limit * 1.0e3, color="tab:red", linestyle="--", label="planning limit"
    )
    axes[3, 1].set_title("Pump flow")
    axes[3, 1].set_ylabel("flow [L/s]")
    axes[3, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time [s]")
    figure.suptitle(
        f"crane_planning timing OCP — duration {float(summary['duration_s']):.3f} s, "
        f"{summary['status']}, {summary['iterations']} SQP iterations"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    print(f"Wrote plot: {path}")
    if show:
        plt.show()
    plt.close(figure)


def print_summary(data: dict, summary: dict) -> None:
    sway = block(data, "q_", PASSIVE_NAMES) - block(data, "q_eq_", PASSIVE_NAMES)
    dq_u = block(data, "dq_", PASSIVE_NAMES)
    force_use = block(data, "force_use_", AXIS_NAMES)
    print("\nSummary")
    print(f"  duration:               {float(summary['duration_s']):.6f} s")
    print(f"  solver status:          {summary['status']}")
    print(f"  SQP iterations:         {summary['iterations']}")
    print(f"  solve time:             {float(summary['solve_time_s']) * 1e3:.2f} ms")
    print(f"  kappa:                  {float(summary['kappa']):.4f}")
    print(f"  peak sway offset:       {np.max(np.abs(sway), axis=0)} rad")
    print(f"  peak sway rate:         {np.max(np.abs(dq_u), axis=0)} rad/s")
    # Against the **kappa-scaled** allowance, i.e. against the bound the solver
    # actually imposed: 1.0 here means a row sat on its constraint.
    print(f"  peak force use, of kappa x limit: {np.max(force_use, axis=0)}")
    print(f"  peak pump use,  of kappa x limit: {np.max(data['pump_use']):.6f}")
    # Reported by the solve against the **unscaled** limit, so "the peak demand
    # sits at kappa and not at one" is readable rather than trusted.
    # And against the **unscaled** limit, which is what makes "the peak demand
    # sits at kappa and not at one" readable. The two blocks differ by 1/kappa on
    # force; pump_flow additionally carries the 0.95 planning factor in its
    # denominator, so it is not on the same footing as the other three.
    print("  peak demand, as a fraction of the physical limit:")
    for key in (
        "peak_joint_velocity",
        "peak_joint_acceleration",
        "peak_cylinder_force",
        "peak_pump_flow",
    ):
        print(f"    {key[5:]:<20} {float(summary[key]):.6f}")


def main() -> int:
    arguments = parse_arguments()
    try:
        binary = find_binary(arguments.binary)
        command = build_command(arguments, binary)
        if arguments.print_command:
            print(" ".join(command))
        output = arguments.output.resolve()
        csv_path = output.with_suffix(".csv")
        reference_path = (
            output.with_suffix(".reference.csv") if arguments.reference_csv else None
        )
        summary = run_solver(command, csv_path, reference_path)
        data = read_csv(csv_path)
        plot(output, data, summary, arguments.show)
        if arguments.no_csv:
            csv_path.unlink()
        else:
            print(f"Wrote data: {csv_path}")
        if reference_path is not None:
            print(f"Wrote emitted reference: {reference_path}")
        print_summary(data, summary)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
