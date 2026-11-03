"""CES's sphere distances, to the scene and to itself, never read clearer than Coal's."""

import numpy as np
from crane_planning import weights as crane_weights
from crane_planning.config import PlannerConfig
from crane_planning.geometry import Geometry
from crane_planning.planner import Planner
from test_jerk_bound import description
from test_long_carried_reach import YAW_PI, box


def test_spheres_are_conservative_where_coal_says_clear():
    planner = Planner(description(), PlannerConfig(), dict(crane_weights.DEFAULTS))
    scene = [
        box("truck", (-3.332, 0.0, 0.933), (5.554, 2.518, 0.663), YAW_PI),
        box("container", (-9.439, -10.65, 7.5), (8.0, 5.0, 15.0), YAW_PI),
        box("block_4", (-13.0, -1.5, 0.3), (0.9, 0.6, 0.6)),
    ]
    centre = np.array([0.030, 0.018, 1.020])
    geometry = Geometry(
        planner.model,
        planner.limits,
        planner.prepare_scene(scene, True),
        planner.config,
        0.2199,
        None,
        ("cylinder", np.array([1.082, 1.082, 0.60]), centre),
    )
    limits = planner.limits
    lower = np.where(limits.bounded, limits.lower, -np.pi)
    upper = np.where(limits.bounded, limits.upper, np.pi)
    q = np.random.default_rng(1).uniform(lower, upper, size=(100, 5))
    rows = [
        geometry.model.collision_queries(
            geometry.configuration(row), geometry.bodies(geometry.configuration(row))
        )
        for row in q
    ]
    exact = np.array([min(r.minimum_distance_m for r in row[:-1]) for row in rows])
    exact_self = np.array([row[-1].minimum_distance_m for row in rows])
    spheres, spheres_self = geometry.sphere_clearance(q)
    clear, clear_self = exact > 0.0, exact_self > 0.0
    assert np.count_nonzero(clear) > 30 and np.count_nonzero(clear_self) > 30
    assert np.all(spheres[clear] <= exact[clear] + 1.0e-9)
    assert np.all(spheres_self[clear_self] <= exact_self[clear_self] + 1.0e-9)
