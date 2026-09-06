"""`Geometry.envelope` must bound the swing of both passive hinges, not one."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
import yaml
from crane_model import (
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    CollisionPrimitive,
    CraneModel,
    Frame,
    Tool,
)
from crane_planning.config import (
    PLANNED_DOF,
    TOOL_INDEX,
    PlannerConfig,
    passive_equilibrium,
    read_limits,
)
from crane_planning.geometry import Geometry
from crane_planning.planner import payload_parameters

Q_TOOL = 0.35
#: A block roughly the size of the ones CBS sets, as `(shape, dimensions, offset)`.
PAYLOAD_SHAPE = ("box", np.array([0.40, 0.20, 0.20]), np.array([0.0, 0.0, -0.30]))


def description() -> str:
    """The expanded PZS100 description crane_model keeps as a fixture."""
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


def geometry(payload_shape) -> Geometry:
    xml = description()
    config = shipped_config()
    return Geometry(
        CraneModel(xml, Tool.PZS100),
        read_limits(xml, config.pump_flow_max, config.pump_flow_planning_factor),
        [],
        config,
        Q_TOOL,
        payload_parameters(None),
        payload_shape,
    )


def carried_point(model: CraneModel, sway, reach: float) -> np.ndarray:
    """
    Where the farthest carried point sits, for a given pair of hinge angles.

    The block hangs below the TCP along the tool's own z axis, so it rides the
    tool's rotation -- which is the whole reason the upper hinge matters. Sway
    is an offset from the hanging equilibrium, which is where the box is centred.
    """
    q = np.zeros(GENERALIZED_DOF)
    q[TOOL_INDEX] = Q_TOOL
    q[list(PASSIVE_INDICES)] = passive_equilibrium(np.zeros(PLANNED_DOF)) + np.asarray(
        sway
    )
    pose = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    x, y, z, w = pose.orientation_xyzw
    axis = pin.Quaternion(w, x, y, z).matrix()[:, 2]
    return pose.position_m + reach * axis


def measured_swing(model: CraneModel, bounds, reach: float) -> float:
    """Farthest the carried point gets from rest, swept over the sway box."""
    rest = carried_point(model, (0.0, 0.0), reach)
    grid = [np.linspace(-bound, bound, 41) for bound in bounds]
    return max(
        float(np.linalg.norm(carried_point(model, (a, b), reach) - rest))
        for a in grid[0]
        for b in grid[1]
    )


@pytest.mark.parametrize("payload_shape", [None, PAYLOAD_SHAPE])
def test_envelope_bounds_both_hinges(payload_shape):
    """
    The envelope is an upper bound on the real swing, and a tight one.

    Sweeping the sway box through the model is the oracle. The one-hinge
    formula this replaced returns 0.153 m against a real 0.250 m, so it fails
    the bound; a formula that adds the two hinges instead of composing them
    returns 0.352 m and fails the tightness.
    """
    scene = geometry(payload_shape)
    swing = measured_swing(
        scene.model, np.abs(scene.config.q_sway_max), scene._carried_reach()
    )
    assert scene.envelope >= swing
    assert scene.envelope <= 1.05 * swing


def test_upper_hinge_carries_the_longer_lever():
    """
    Why the omission was unsafe rather than conservative.

    `theta6_tip` pivots further from the tool than `theta7_tilt` does, so
    leaving it out of the envelope drops the larger of the two terms.
    """
    scene = geometry(None)
    q = np.zeros(GENERALIZED_DOF)
    q[TOOL_INDEX] = Q_TOOL
    q[list(PASSIVE_INDICES)] = passive_equilibrium(np.zeros(PLANNED_DOF))
    tcp = scene.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    levers = [
        np.linalg.norm(
            tcp.position_m
            - scene.model.forward_kinematics(q, Frame.MOUNTING_BASE, pivot).position_m
        )
        for pivot in (Frame.TIP, Frame.TILT)
    ]
    assert levers[0] > levers[1]


def test_required_spends_the_envelope_once():
    """`required` is the three margins summed, with the corrected envelope."""
    scene = geometry(None)
    assert scene.required == pytest.approx(
        scene.config.margin_safety + scene.config.margin_interp + scene.envelope
    )


def _shifted(
    scene: Geometry, body: CollisionPrimitive, along: np.ndarray, target: float
):
    """Move `body` along `along` so the hanging pose clears it by `target`."""
    scene.scene = [body]
    clearance = scene.clearance(np.zeros(PLANNED_DOF))
    assert np.isfinite(clearance)
    body.pose_in_mounting_base.translation += along * (clearance - target)
    assert scene.clearance(np.zeros(PLANNED_DOF)) == pytest.approx(target, abs=1e-3)
    return scene


def _tcp(scene: Geometry) -> np.ndarray:
    return scene.model.forward_kinematics(
        scene.configuration(np.zeros(PLANNED_DOF)), Frame.MOUNTING_BASE, Frame.TCP
    ).position_m


def test_inside_the_envelope_only_the_swinging_half_owes_it():
    """
    The envelope is owed by what hangs on the hinges. A wall the same distance
    from the machine, inside `required` and outside the two margins, is
    accepted when the base is what stands near it and refused when the tool is.
    """
    scene = geometry(None)
    target = 0.5 * (scene.required + scene.required_rigid)
    tcp = _tcp(scene)

    # beside the base: the outriggers reach 2.62 m in y, the tool is 2.9 m out
    # in x and 0.9 m wide, so a wall over the base is the base's alone
    near_base = CollisionPrimitive(
        id="near_base",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([0.0, 5.0, 0.0])),
        dimensions_m=np.array([1.0, 0.1, 3.0]),
    )
    _shifted(scene, near_base, np.array([0.0, -1.0, 0.0]), target)
    clearance, required = scene.margin(np.zeros(PLANNED_DOF))
    assert required == pytest.approx(scene.required_rigid)
    assert clearance > required
    assert scene.is_valid(np.zeros(PLANNED_DOF))

    near_tool = CollisionPrimitive(
        id="near_tool",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), tcp + np.array([0.0, 1.5, 0.0])),
        dimensions_m=np.array([3.0, 0.1, 3.0]),
    )
    _shifted(scene, near_tool, np.array([0.0, -1.0, 0.0]), target)
    clearance, required = scene.margin(np.zeros(PLANNED_DOF))
    assert required == pytest.approx(scene.required)
    assert clearance < required
    assert not scene.is_valid(np.zeros(PLANNED_DOF))
