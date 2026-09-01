"""`expand_truck` must cover what the description actually puts on the truck."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from crane_model.collision import CollisionPrimitive
from crane_planning.planner import PlannerConfig, expand_truck

# The truck, measured in `world` off `epsilon_crane_description`'s
# `timber_loader_AIT.urdf.xacro` expanded with `post_setup:=134` -- what
# `crane_bringup/launch/sim.launch.py` launches, and a superset of the `13` the
# hardware bringup uses. The box is the one `concrete_block_world_model`'s
# `vehicle_box` ships, whose own agreement with these numbers that package
# asserts; here it is the input, and what matters is what comes out of it.
DECK_CENTRE = np.array([-3.332, 0.0, 0.932575])
DECK_EXTENT = np.array([5.554, 2.518, 0.662850])
DECK_SURFACE_Z = 1.264
# Posts: 0.080 x 0.108 x 1.750, standing on the cross struts at z 1.332.
POST_X = (-5.593, -4.381, -1.397)
POST_HALF_EXTENT = np.array([0.040, 0.054])
POST_CENTRE_Y = 1.2025
POST_Z = (1.332, 3.082)
# The headboard closing the cab end, from `lkw_ladeflaeche_mit_rammschutz_seitlich_col.stl`.
HEADBOARD_MIN = np.array([-1.005, -1.254, DECK_SURFACE_Z])
HEADBOARD_MAX = np.array([-0.555, 1.259, 3.186])

TOLERANCE = 0.01


def shipped_config() -> PlannerConfig:
    """The planner config as launched, not as defaulted."""
    document = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "crane_planner.yaml").read_text()
    )
    parameters = next(iter(document.values()))["ros__parameters"]
    config = PlannerConfig()
    for name, value in parameters.items():
        if hasattr(config, name):
            setattr(
                config,
                name,
                np.asarray(value, dtype=float) if isinstance(value, list) else value,
            )
    return config


def expanded_boxes() -> dict:
    """The truck primitive at the measured deck, expanded, as {id: (min, max)}."""
    truck = CollisionPrimitive(
        id="truck",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), DECK_CENTRE),
        dimensions_m=DECK_EXTENT,
        structural=True,
    )
    boxes = {}
    for primitive in expand_truck([truck], shipped_config()):
        assert primitive.structural, primitive.id
        centre = np.asarray(primitive.pose_in_mounting_base.translation, dtype=float)
        half = 0.5 * np.asarray(primitive.dimensions_m, dtype=float)
        boxes[primitive.id] = (centre - half, centre + half)
    return boxes


def envelops(box, lower, upper) -> bool:
    return bool(
        np.all(box[0] <= lower + TOLERANCE) and np.all(box[1] >= upper - TOLERANCE)
    )


def test_the_truck_is_replaced_by_the_bed_at_its_surface():
    boxes = expanded_boxes()
    assert "truck" not in boxes
    assert boxes["truck_bed"][1][2] == DECK_SURFACE_Z


def test_every_post_the_description_stands_is_inside_a_runge():
    boxes = expanded_boxes()
    runges = [box for name, box in boxes.items() if name.startswith("truck_runge_")]
    assert len(runges) == 2 * len(shipped_config().truck_runge_stations)

    for x in POST_X:
        for side in (1.0, -1.0):
            lower = np.array(
                [
                    x - POST_HALF_EXTENT[0],
                    side * POST_CENTRE_Y - POST_HALF_EXTENT[1],
                    POST_Z[0],
                ]
            )
            upper = np.array(
                [
                    x + POST_HALF_EXTENT[0],
                    side * POST_CENTRE_Y + POST_HALF_EXTENT[1],
                    POST_Z[1],
                ]
            )
            assert any(envelops(runge, lower, upper) for runge in runges), (
                f"no runge covers the post at x={x}, y={side * POST_CENTRE_Y}"
            )


def test_the_headboard_closing_the_cab_end_is_an_obstacle():
    boxes = expanded_boxes()
    assert envelops(boxes["truck_headboard"], HEADBOARD_MIN, HEADBOARD_MAX)
