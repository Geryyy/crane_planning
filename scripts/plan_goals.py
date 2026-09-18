#!/usr/bin/env python3
"""
Plan from the OUTSIDE preset to each named goal and watch every one in MuJoCo.

One planner, one plant, one window: each goal is planned from
crane_model.presets.OUTSIDE again, its TCP path is drawn into the scene, and
the plan is then rolled so the crane walks the line that was just drawn.

    ./scripts/plan_goals.py
    ./scripts/plan_goals.py --goals out across --settle 3
    ./scripts/plan_goals.py --headless        # numbers only, no window

The planner is the deployment's own, the rollout is tune_planner.roll's and
the plant is crane_model.mujoco_plant -- nothing is simulated here. What this
adds is the goal set, the overlay, and the row per goal.

What it does not show is clearance: the scene is empty, so `avoid_collisions`
certifies self-collision alone, and the plant carries no contacts (clearance is
proven against Coal, and MuJoCo must not re-decide it). Five clean sweeps mean
reachable and self-clear, not that the goals are clear of a truck.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))
if (PACKAGE / "crane_planning" / "planner.py").is_file():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import plan_example  # noqa: E402
import tune_planner  # noqa: E402
from crane_model import Frame, presets  # noqa: E402
from crane_model.mujoco_plant import MujocoPlant, leave  # noqa: E402
from crane_planning import Planner, PlannerConfig, PlanningError, Start  # noqa: E402

#: Goal names in run order. `goals_for` is the one place they mean something.
GOAL_NAMES = ("out", "up", "in", "across", "low")

#: Drawn path, in metres and RGBA. The tube is thin enough to read against the
#: telescope it runs beside; the goal ball is the placement tolerance's size.
PATH_WIDTH = 0.04
PATH_RGBA = (0.10, 0.85, 0.95, 1.0)
GOAL_RADIUS = 0.12
GOAL_RGBA = (1.0, 0.35, 0.05, 0.9)


def arguments() -> argparse.Namespace:
    """
    Take what changes what you see, and nothing else.

    Shipped defaults throughout: shaping this move is tune_planner.py's job,
    and a knob offered in both is a knob the two can disagree about.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--goals",
        nargs="+",
        choices=GOAL_NAMES,
        default=list(GOAL_NAMES),
        help="which goals to plan, in the order given (default: all)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=5.0,
        help="seconds to hold each arrival. Measured: the sway is damped, not "
        "ringing, and `end` is still mid-decay at 3 s (`across` reads 0.26 deg "
        "against its converged 0.07); it is flat from 4 s on",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer playback factor; 0 is as fast as it computes. Pacing only",
    )
    parser.add_argument(
        "--headless", action="store_true", help="skip the window and just plan and roll"
    )
    return parser.parse_args()


def goals_for(planner: Planner, start: Start) -> dict[str, np.ndarray]:
    """
    TCP goals in K0_mounting_base, each reachable from `start`.

    Named for the axis they ask for, not for a pose: what makes a goal worth
    keeping here is that it moves something the others leave alone.
    """
    return {
        "out": presets.GOAL_OUT.copy(),  # slew in 45 deg and lift
        "up": presets.goal_here(planner.model, start.q),  # lift alone, bearing kept
        "in": np.array([2.90, 2.90, 1.50]),  # retract, same bearing
        "across": np.array([0.00, 5.00, 1.50]),  # slew out the other way
        "low": np.array([5.00, 0.00, 0.30]),  # out and down
    }


def draw(viewer, path: np.ndarray, goal: np.ndarray) -> None:
    """
    Put the planned TCP path and its goal into the viewer's own scene.

    `user_scn` survives `sync`, so this is drawn once per goal and then stands
    while the crane walks it. Coordinates go in untransformed because
    K0_mounting_base *is* MuJoCo's world origin for the description this runs:
    it is that URDF's root link, and `with_scenery` attaches at an identity
    frame. A truck-mounted description carries `world` above it and would need
    the offset -- checked, not assumed.
    """
    import mujoco

    scene = viewer.user_scn
    scene.ngeom = 0

    def claim(kind, size, position, rgba) -> int:
        """Take the next scene slot for `kind`; -1 once the scene is full."""
        if scene.ngeom >= scene.maxgeom:
            return -1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            kind,
            np.asarray(size, dtype=float),
            np.asarray(position, dtype=float),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return scene.ngeom - 1

    claim(mujoco.mjtGeom.mjGEOM_SPHERE, [GOAL_RADIUS] * 3, goal, GOAL_RGBA)
    capsule = mujoco.mjtGeom.mjGEOM_CAPSULE
    for first, second in zip(path[:-1], path[1:]):
        slot = claim(capsule, np.zeros(3), np.zeros(3), PATH_RGBA)
        if slot < 0:
            break
        mujoco.mjv_connector(
            scene.geoms[slot],
            capsule,
            PATH_WIDTH,
            np.asarray(first, dtype=float),
            np.asarray(second, dtype=float),
        )


def tcp_path(planner: Planner, plan) -> np.ndarray:
    """
    Where the tool goes, one point per emitted sample.

    Not `plan.tcp`: that is the geometric path at `visualization_samples`, 25
    points, which facets visibly on a slew. This is the joint trajectory the
    plant is about to be driven along, so the drawn line is the one the crane
    walks rather than one it merely passes near.
    """
    return np.array(
        [
            planner.model.forward_kinematics(
                q, Frame.MOUNTING_BASE, Frame.TCP
            ).position_m
            for q in plan.q
        ]
    )


def row(name: str, plan, rolled: dict) -> str:
    """One goal's line of the summary: what it cost and what the load did."""
    predicted = plan.timing.q_u - plan.timing.q_u_eq
    actual = rolled["q_u"] - rolled["q_u_eq"]
    arrival = int(np.searchsorted(rolled["time"], plan.duration))
    worst = np.degrees(
        [
            np.max(np.abs(predicted)),
            np.max(np.abs(actual)),
            np.max(np.abs(actual[arrival])),
            np.max(np.abs(actual[-1])),
        ]
    )
    return (
        f"{name:<8}{plan.duration:8.2f}{plan.timing.iterations:6d}"
        f"{plan.timing.solve_time_s:9.3f}" + "".join(f"{value:9.2f}" for value in worst)
    )


HEADING = (
    f"{'goal':<8}{'dur [s]':>8}{'SQP':>6}{'solve':>9}"
    f"{'OCP pk':>9}{'mjc pk':>9}{'arrival':>9}{'end':>9}"
    "   sway, worst of tip/tilt [deg];"
    "\n" + " " * 58 + "`end` is the decay at --settle, one instant"
)


def main() -> int:
    options = arguments()
    description = plan_example.description()
    planner = Planner(description, PlannerConfig())
    start = tune_planner.start_of(planner)
    yaw = tune_planner.start_yaw(planner, start)
    table = goals_for(planner, start)

    plant = MujocoPlant(description)
    viewer = None if options.headless else plant.open_viewer(options.realtime)

    lines = []
    for name in options.goals:
        goal = table[name]
        print(f"\n{name}: [{goal[0]:.2f} {goal[1]:.2f} {goal[2]:.2f}] m")
        try:
            plan = planner.plan(
                start,
                goal,
                yaw,
                scene=[],
                avoid_collisions=True,
            )
        except PlanningError as refusal:
            print(f"  refused: {refusal}")
            lines.append(f"{name:<8}  refused: {refusal}")
            continue
        print(f"  {plan.message}")
        if viewer is not None:
            if not viewer.is_running():
                break
            draw(viewer, plan.tcp, goal)
        lines.append(
            row(
                name,
                plan,
                tune_planner.roll(planner, plant, plan, start, options.settle),
            )
        )

    print(f"\n{HEADING}")
    for line in lines:
        print(line)
    if viewer is not None:
        print("\nclose the viewer window to finish")
        plant.hold_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(leave(main()))
