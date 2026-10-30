"""A long carried reach (radius 4.5 -> 8.9 m, 880 kg) plans within the pump, in one short solve."""

import numpy as np
import pinocchio as pin
from crane_model import CollisionPrimitive, Payload
from crane_planning import weights as crane_weights
from crane_planning.config import PlannerConfig
from crane_planning.planner import Planner, Start
from test_jerk_bound import description

YAW_PI = (0.0, 0.0, 1.0, 0.0)


def box(name, world, size, quat=(0.0, 0.0, 0.0, 1.0)):
    """Box from the wall_yawm90 world, in K0: x = -wx - 6.94, y = -wy + 0.35, z = wz - 1.14."""
    k0 = [-world[0] - 6.94, -world[1] + 0.35, world[2] - 1.14]
    return CollisionPrimitive(
        id=name,
        shape="box",
        pose_in_mounting_base=pin.XYZQUATToSE3(np.array(k0 + list(quat))),
        dimensions_m=np.array(size, dtype=float),
        structural=False,
    )


def test_block_4_of_the_rotated_wall_plans_within_the_pump_in_one_solve():
    """First solve bought 1.6x the pump in 56 SQP iterations, the retry took 96 more
    (28.9 s in the sim). Now one solve: guess at the pump's duration, no 0 flow floor."""
    planner = Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))
    q = np.array([0.4345, 1.1108, -0.2041, 0.5072, 0.6391, 1.5546, 2.0023, 0.2199])
    start = Start(q=q, dq_a=np.zeros(5))
    centre = np.array([0.030, 0.018, 1.020])
    payload = Payload(
        mass_kg=880.0,
        center_of_mass_k8_m=centre,
        inertia_k8_kg_m2=np.zeros((3, 3)),
        valid=True,
    )
    shape = ("cylinder", np.array([1.082, 1.082, 0.60]), centre)
    scene = [
        box("truck", (-3.332, 0.0, 0.933), (5.554, 2.518, 0.663), YAW_PI),
        box("container", (-9.439, -10.65, 2.461), (7.0, 4.0, 5.0), YAW_PI),
        box("hedge", (-6.939, 6.85, 1.461), (20.0, 5.0, 3.0), YAW_PI),
        box("block_4", (-13.0, -1.5, 0.3), (0.9, 0.6, 0.6)),
    ] + [
        box(f"block_{i}", (-15.0, y, 0.3), (0.6, 0.9, 0.6))
        for i, y in enumerate((-3.0, -3.92, -4.84))
    ]
    goal = np.array([8.06, 3.65, 2.701]) + planner.tip_to_tcp_offset(
        payload, 0.0, start.q_tool
    )

    solves, solve = [], planner.ocp.solve
    planner.ocp.solve = lambda **kw: solves.append(solve(**kw)) or solves[-1]

    timing = planner.plan(
        start, goal, 0.0, payload=payload, payload_shape=shape, scene=scene
    ).timing

    assert np.max(timing.pump_flow) <= timing.pump_flow_bound + 1.0e-6
    assert len(solves) == 1  # no pump retry
    # iterations, not seconds: 20 iterations took 6 s idle and 14 s under a running sim
    assert solves[0].iterations <= 30
