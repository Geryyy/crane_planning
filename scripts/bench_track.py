#!/usr/bin/env python3
r"""
Drive bench moves under each feedforward law and measure tracking.

Unlike bench_calc_movement.py, which never drives the machine. Three arms run
the identical OCP solve; c3_feedforward/command_lag_s are read per-request
after the solve, so only the reference's effort field differs. static: no
effort (JTC uses ff_velocity_scale*qdot_d). inverse: inverts block 3 + rigid
body via RNEA + preview. inverse_lag: also inverts the PT1 lag.

Reads /trajectory_controllers/controller_state; nothing new instrumented.

    ./scripts/bench_plan.py --emit-requests /tmp/a2b_goals.json --set all
    ./scripts/bench_track.py --requests /tmp/a2b_goals.json \\
        --only ax_slew,ax_arm,sh_multi,across --repeats 3 --out /tmp/ablation

--out writes one wide CSV per move (all arms on one grid) plus metrics.csv.
Run sim headless alongside, tree idle:

    ros2 launch concrete_block_behavior_tree gazebo_wall_assembly_pzs100.launch.py \\
        controller:=pid planner:=cbs enable_livox_sim:=off gui:=false

Warning: Gazebo plant IS C3 (same k, d, dead time, tau_v), so inverse_lag
flatters itself here -- not evidence it transfers to hardware.

ax_arm has fitted lag 0: inverse and inverse_lag must match there exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import SwitchController
from crane_model import PASSIVE_INDICES
from crane_planning.config import PLANNED_INDICES
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from timber_crane_planning_interfaces.srv import CalcMovement

#: Fitted lag/dead-time split, planned-axis order (sw,ha,ka,sa,ro); same as
#: Gazebo URDF tau_v, so inverse_lag inverts that plant exactly.
FITTED_LAG = [0.100, 0.025, 0.000, 0.075, 0.125]
NO_LAG = [0.0] * 5

#: `(c3_feedforward, command_lag_s)`. Both read per request by planner, so
#: switching arms costs a service call, not a relaunch.
ARMS = {
    "static": (False, NO_LAG),
    "inverse": (True, NO_LAG),
    "inverse_lag": (True, FITTED_LAG),
}

#: Resample grid for the wide CSV; finer than 25 Hz reference and 100 Hz loop.
GRID_S = 0.01

#: rad, m on telescope. Checked over the five planned axes only -- passive pair
#: and tool have no feedforward/fitted lag, not evidence for this ablation.
#: Achieved offset also goes into every metrics row, not just gated on.
START_TOLERANCE = 0.05

#: Settle wait after park goal returns: slew loop is a trim integrator with
#: 28 s dominant time constant, so "goal succeeded" != "axis stopped moving".
PARK_SETTLE_S = 40.0

#: rad, rad/s. q0 assumes passive pair hanging still at passive_equilibrium;
#: measured parking left tilt 0.224 rad off vertical (planner's q_sway_max is
#: 0.2), so a run starting mid-swing is a run the OCP never planned. Nothing
#: commands the pair, so waiting is the only fix.
PASSIVE_TOLERANCE = 0.03
REST_RATE = 0.05


def parked(names: list, q0, seconds: float) -> JointTrajectory:
    """Build a one-point goal at `q0`."""
    trajectory = JointTrajectory()
    trajectory.joint_names = list(names)
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in q0]
    point.velocities = [0.0] * len(names)
    point.time_from_start.sec = int(seconds)
    point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
    trajectory.points.append(point)
    return trajectory


def request_of(move: dict) -> CalcMovement.Request:
    """One bench request; same fields bench_calc_movement.build sets."""
    request = CalcMovement.Request()
    request.y_n.x, request.y_n.y, request.y_n.z = (float(v) for v in move["y_n"])
    request.phi_tool_n = float(move["phi_tool_n"])
    request.q0 = [float(value) for value in move["q0"]]
    request.slow_down = 1.0
    request.check_log_collision = False
    request.check_gripper_collision = False
    request.publish_path = False
    return request


def sampled(trajectory: JointTrajectory) -> tuple:
    """`(t, effort)`, zero-filled on an arm that carries none."""
    stamps = np.array(
        [
            point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            for point in trajectory.points
        ]
    )
    width = len(trajectory.joint_names)
    effort = np.array(
        [
            list(point.effort) if point.effort else [0.0] * width
            for point in trajectory.points
        ]
    )
    return stamps, effort


class Bench(Node):
    """One node for the whole campaign: plan, park, execute, record."""

    def __init__(self, controller: str, service: str):
        super().__init__("bench_track")
        self.controller = controller
        self.samples: list = []
        self.recording = False
        self.state_names: list = []
        self.latest: JointTrajectoryControllerState | None = None
        self.plan_client = self.create_client(CalcMovement, service)
        self.params = self.create_client(SetParameters, "/crane_planner/set_parameters")
        self.switch = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self.follow = ActionClient(
            self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory"
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/trajectory_controllers/controller_state",
            self._state,
            50,
        )

    # -- recording --

    def _state(self, message: JointTrajectoryControllerState) -> None:
        self.state_names = list(message.joint_names)
        self.latest = message
        if not self.recording:
            return
        width = len(message.joint_names)

        def row(values) -> list:
            # output carries commanded joints only (6 of 8 here); padded not
            # dropped -- absent command isn't a zero one.
            padded = list(values) + [float("nan")] * (width - len(values))
            return padded[:width]

        stamp = message.header.stamp
        self.samples.append(
            (
                stamp.sec + stamp.nanosec * 1e-9,
                row(message.reference.positions),
                row(message.feedback.positions),
                row(message.error.positions),
                row(message.output.velocities or message.output.positions),
            )
        )

    def spin(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait(self, future, timeout: float):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    # -- the four things this script does --

    def arm(self, name: str) -> bool:
        """Set the two parameters that select a feedforward law. Live."""
        feedforward, lag = ARMS[name]
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="c3_feedforward",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL, bool_value=feedforward
                ),
            ),
            Parameter(
                name="command_lag_s",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    double_array_value=[float(value) for value in lag],
                ),
            ),
        ]
        answer = self.wait(self.params.call_async(request), 10.0)
        return answer is not None and all(r.successful for r in answer.results)

    def switch_controller(self, activate: bool) -> bool:
        request = SwitchController.Request()
        if activate:
            request.activate_controllers = [self.controller]
        else:
            request.deactivate_controllers = [self.controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        answer = self.wait(self.switch.call_async(request), 10.0)
        return answer is not None and answer.ok

    def execute(self, trajectory: JointTrajectory, record: bool, timeout: float):
        """Send one goal with the stamp zeroed. Returns `(status, samples)`."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        # Zero the stamp: planner stamps with the /joint_states it planned
        # from, seconds old by send time; JTC would otherwise sample mid-plan.
        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0
        self.samples = []
        self.recording = record
        handle = self.wait(self.follow.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            self.recording = False
            return None, []
        result = self.wait(handle.get_result_async(), timeout)
        self.recording = False
        return (result.result.error_code if result is not None else None), self.samples

    def offset_from(self, q0, order: list) -> float:
        """Largest deviation from `q0` over the planned axes, by name."""
        if self.latest is None:
            return float("inf")
        feedback = np.array(self.latest.feedback.positions)[order]
        axes = list(PLANNED_INDICES)
        return float(np.max(np.abs(feedback[axes] - np.asarray(q0, dtype=float)[axes])))

    def at_rest(self, q0, order: list) -> tuple:
        """`(planned offset, passive offset, largest rate)` right now."""
        if self.latest is None:
            return float("inf"), float("inf"), float("inf")
        position = np.array(self.latest.feedback.positions)[order]
        rate = np.array(self.latest.feedback.velocities)
        target = np.asarray(q0, dtype=float)
        axes = list(PLANNED_INDICES)
        passive = list(PASSIVE_INDICES)
        return (
            float(np.max(np.abs(position[axes] - target[axes]))),
            float(np.max(np.abs(position[passive] - target[passive]))),
            float(np.max(np.abs(rate[order]))) if rate.size else float("inf"),
        )

    def park(self, names: list, q0, order: list, rate: float) -> tuple:
        """
        Drive to q0, then wait for it to stop.

        Success gates on commanded axes only, not the pendulum.
        """
        travel = self.offset_from(q0, order)
        status, _ = self.execute(
            parked(names, q0, max(6.0, travel / rate)), record=False, timeout=240.0
        )
        end = time.perf_counter() + PARK_SETTLE_S
        while True:
            offset, sway, speed = self.at_rest(q0, order)
            settled = (
                offset <= START_TOLERANCE
                and sway <= PASSIVE_TOLERANCE
                and speed <= REST_RATE
            )
            if settled or time.perf_counter() > end:
                return status, offset, sway, speed
            self.spin(0.25)


# -- the numbers --


def channels(samples: list, order: list) -> tuple:
    """`(t from zero, reference, feedback, error, command)` in plan order."""
    t = np.array([row[0] for row in samples])
    blocks = [
        np.array([row[column] for row in samples])[:, order] for column in (1, 2, 3, 4)
    ]
    return (t - t[0], *blocks)


def metrics(samples: list, duration: float, names: list, order: list) -> list:
    """Per joint: NRMSE, peak error, overshoot, terminal error, settle time."""
    if len(samples) < 10:
        return []
    t, reference, feedback, error, _command = channels(samples, order)
    rows = []
    for j, name in enumerate(names):
        span = float(np.ptp(reference[:, j]))
        if span < 1e-6:
            continue
        goal = float(reference[-1, j])
        travel = goal - float(reference[0, j])
        # Signed by travel direction, so undershoot reads negative not
        # zero-like-perfect.
        past = np.sign(travel) * (feedback[:, j] - goal)
        # Last instant outside 1% of the move, measured from end of
        # reference, not of the recording.
        outside = np.nonzero(np.abs(error[:, j]) > 0.01 * span)[0]
        settle = float(t[outside[-1]] - duration) if outside.size else float("-inf")
        rows.append(
            {
                "joint": name,
                "span": span,
                "nrmse": float(np.sqrt(np.mean(error[:, j] ** 2)) / span),
                "peak_error": float(np.max(np.abs(error[:, j]))),
                "overshoot": float(np.max(past) / abs(travel)) if travel else 0.0,
                "terminal_error": float(error[-1, j]),
                "settle_s": settle,
            }
        )
    return rows


def write_move_csv(path: Path, runs: dict, names: list, order: list) -> None:
    """One wide CSV per move: every arm on one time base, for PlotJuggler."""
    horizon = max(run["duration"] for run in runs.values()) + 2.0
    grid = np.arange(0.0, horizon, GRID_S)
    columns: dict = {"t": grid}
    for arm, run in runs.items():
        t, *blocks = channels(run["samples"], order)
        stamps, effort = sampled(run["trajectory"])
        for label, block in zip(("ref", "fb", "err", "cmd"), blocks):
            for j, name in enumerate(names):
                columns[f"{arm}/{name}/{label}"] = np.interp(grid, t, block[:, j])
        for j, name in enumerate(names):
            # ff held at zero past end of plan, matching the reference.
            columns[f"{arm}/{name}/ff"] = np.interp(
                grid, stamps, effort[:, j], right=0.0
            )
    table = np.column_stack(list(columns.values()))
    path.write_text(
        ",".join(columns)
        + "\n"
        + "\n".join(",".join(f"{value:.6g}" for value in row) for row in table)
        + "\n"
    )


def summarise(rows: list, arms: list) -> None:
    """Median NRMSE and overshoot per (move, joint), one column per arm."""
    keys = sorted({(row["move"], row["joint"]) for row in rows})
    width = max(len(f"{move}/{joint}") for move, joint in keys) + 2
    print(f"\n{'move/joint':<{width}}" + "".join(f"{arm:>22}" for arm in arms))
    print(f"{'':<{width}}" + "".join(f"{'NRMSE | overshoot':>22}" for _ in arms))
    for move, joint in keys:
        line = f"{move + '/' + joint:<{width}}"
        for arm in arms:
            picked = [
                row
                for row in rows
                if row["move"] == move and row["joint"] == joint and row["arm"] == arm
            ]
            if not picked:
                line += f"{'-':>22}"
                continue
            nrmse = float(np.median([row["nrmse"] for row in picked]))
            over = float(np.median([row["overshoot"] for row in picked]))
            line += f"{nrmse:>13.4f} |{over:>7.3f}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="comma-separated move names")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--service", default="/a2b_movement")
    parser.add_argument("--controller", default="trajectory_controller_a2b")
    parser.add_argument("--plan-timeout", type=float, default=120.0)
    parser.add_argument("--park-rate", type=float, default=0.08, help="rad/s, slow")
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument(
        "--deactivate", action="store_true", help="release the controller on exit"
    )
    options = parser.parse_args()

    moves = json.loads(options.requests.read_text())["moves"]
    if options.only:
        wanted = set(options.only.split(","))
        moves = [move for move in moves if move["name"] in wanted]
        if not moves:
            raise SystemExit(f"no move named in {options.only!r}")
    arms = [arm for arm in options.arms.split(",") if arm in ARMS]
    options.out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Bench(options.controller, options.service)
    for client, what in (
        (node.plan_client, options.service),
        (node.params, "/crane_planner/set_parameters"),
        (node.switch, "/controller_manager/switch_controller"),
    ):
        if not client.wait_for_service(timeout_sec=30.0):
            node.get_logger().error(f"no {what} after 30 s")
            return 1
    if not node.follow.wait_for_server(timeout_sec=30.0):
        node.get_logger().error(f"no {options.controller} action server after 30 s")
        return 1

    # Activated once for whole campaign (tree idle, nothing else claims these
    # interfaces).
    if not node.switch_controller(True):
        node.get_logger().error(f"could not activate {options.controller}")
        return 1
    node.spin(1.0)
    if not node.state_names:
        node.get_logger().error("no controller_state arrived; is the sim running?")
        return 1

    names: list = []
    order: list = []
    rows, dropped = [], 0
    for move in moves:
        per_arm: dict = {}
        for arm in arms:
            if not node.arm(arm):
                node.get_logger().error(f"could not set arm {arm}")
                continue
            started = time.perf_counter()
            answer = node.wait(
                node.plan_client.call_async(request_of(move)), options.plan_timeout
            )
            if answer is None or not answer.success:
                print(f"  {move['name']:<14} {arm:<12} refused")
                continue
            trajectory = answer.trajectory
            if not names:
                # Plan order != controller order; map built by name. Indexing
                # across the two lists by position is the standing footgun.
                names = list(trajectory.joint_names)
                missing = [n for n in names if n not in node.state_names]
                if missing:
                    node.get_logger().error(f"controller does not report {missing}")
                    return 1
                order = [node.state_names.index(n) for n in names]
            duration = float(sampled(trajectory)[0][-1])
            print(
                f"  {move['name']:<14} {arm:<12} planned {duration:5.2f} s in "
                f"{time.perf_counter() - started:5.2f} s, "
                f"{len(trajectory.points)} points, effort "
                f"{'yes' if trajectory.points[0].effort else 'no'}"
            )
            for repeat in range(options.repeats):
                # Park first and check it landed -- a run started elsewhere is
                # no run of this plan.
                park_status, offset, sway, speed = node.park(
                    names, move["q0"], order, options.park_rate
                )
                node.spin(options.settle)
                if (
                    offset > START_TOLERANCE
                    or sway > PASSIVE_TOLERANCE
                    or speed > REST_RATE
                ):
                    node.get_logger().warn(
                        f"{move['name']}/{arm}/{repeat}: dropped -- parked "
                        f"{offset:.3f} off q0, tool {sway:.3f} off hanging, "
                        f"moving at {speed:.3f} rad/s "
                        f"(park error_code {park_status})"
                    )
                    dropped += 1
                    continue
                status, samples = node.execute(
                    trajectory, record=True, timeout=duration + 60.0
                )
                if status != FollowJointTrajectory.Result.SUCCESSFUL:
                    node.get_logger().warn(
                        f"{move['name']}/{arm}/{repeat}: error_code {status}"
                    )
                for row in metrics(samples, duration, names, order):
                    rows.append(
                        {
                            "move": move["name"],
                            "arm": arm,
                            "repeat": repeat,
                            "start_offset": offset,
                            "start_sway": sway,
                            "start_rate": speed,
                            **row,
                        }
                    )
                if repeat == 0 and samples:
                    per_arm[arm] = {
                        "samples": samples,
                        "trajectory": trajectory,
                        "duration": duration,
                    }
        if per_arm:
            path = options.out / f"{move['name']}.csv"
            write_move_csv(path, per_arm, names, order)
            print(f"  -> {path}")

    # Left active on purpose: deactivating stops controller_state, breaking
    # post-run debugging.
    if options.deactivate:
        node.switch_controller(False)

    if rows:
        keys = list(rows[0])
        table = options.out / "metrics.csv"
        table.write_text(
            ",".join(keys)
            + "\n"
            + "\n".join(
                ",".join(
                    f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
                    for key in keys
                )
                for row in rows
            )
            + "\n"
        )
        print(f"\nwrote {len(rows)} rows to {table}", end="")
        print(f", {dropped} runs dropped" if dropped else "")
        summarise(rows, arms)

    node.destroy_node()
    rclpy.shutdown()
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
