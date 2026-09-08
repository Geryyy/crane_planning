"""A start refused on clearance must say which scene body it is refused against."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
import yaml
from crane_model import CollisionPrimitive, CraneModel, Tool
from crane_planning.config import PLANNED_DOF, PlannerConfig, read_limits
from crane_planning.geometry import Geometry
from crane_planning.planner import payload_parameters

Q_TOOL = 0.35
Q_A = np.zeros(PLANNED_DOF)


def description() -> str:
    path = (
        Path(__file__).parents[2]
        / "crane_model"
        / "test"
        / "description"
        / "pzs100.urdf"
    )
    if not path.is_file():
        pytest.skip(f"no machine description at {path}")
    return path.read_text()


def shipped_config() -> PlannerConfig:
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


def box(name: str, centre: np.ndarray) -> CollisionPrimitive:
    return CollisionPrimitive(
        id=name,
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.asarray(centre, dtype=float)),
        dimensions_m=np.array([0.9, 0.6, 0.6]),
        structural=False,
    )


def geometry(scene) -> Geometry:
    xml = description()
    config = shipped_config()
    model = CraneModel(xml, Tool.PZS100)
    return Geometry(
        model,
        read_limits(xml, config.pump_flow_max, config.pump_flow_planning_factor),
        scene,
        config,
        Q_TOOL,
        payload_parameters(None),
        None,
    )


def test_the_body_the_machine_is_inside_is_the_one_named():
    """The block sitting on the tool is named, not the one across the site."""
    tcp, _yaw = geometry([]).tcp_pose(Q_A)
    scene = geometry([box("held_block", tcp), box("far_block", tcp + [0.0, 12.0, 0.0])])

    clearance, _required, body = scene.margin(Q_A)

    assert body == "held_block"
    assert clearance < 0.0


def test_no_body_is_named_when_no_query_can_be_made():
    """Self-collision or a failed query blames no scene body."""
    scene = geometry([box("far_block", np.array([0.0, 12.0, 0.0]))])
    clearance, _required, body = scene.margin(np.full(PLANNED_DOF, np.nan))

    assert body is None
    assert clearance == -np.inf
