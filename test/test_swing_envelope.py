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
#: (shape, dimensions, offset), roughly CBS block size
PAYLOAD_SHAPE = ("box", np.array([0.40, 0.20, 0.20]), np.array([0.0, 0.0, -0.30]))


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
    """Planner config as launched, not as defaulted."""
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
    """Block hangs below TCP along tool's own z axis, so it rides tool rotation --
    why upper hinge matters. Sway offsets hanging equilibrium, where box is centred."""
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
    rest = carried_point(model, (0.0, 0.0), reach)
    grid = [np.linspace(-bound, bound, 41) for bound in bounds]
    return max(
        float(np.linalg.norm(carried_point(model, (a, b), reach) - rest))
        for a in grid[0]
        for b in grid[1]
    )


@pytest.mark.parametrize("payload_shape", [None, PAYLOAD_SHAPE])
def test_envelope_bounds_both_hinges(payload_shape):
    """One-hinge formula this replaced returned 0.153m vs real 0.250m (fails bound);
    adding hinges instead of composing returns 0.352m (fails tightness)."""
    scene = geometry(payload_shape)
    swing = measured_swing(
        scene.model, np.abs(scene.config.q_sway_max), scene._carried_reach()
    )
    assert scene.envelope >= swing
    assert scene.envelope <= 1.05 * swing


def test_upper_hinge_carries_the_longer_lever():
    """theta6_tip pivots further from tool than theta7_tilt, so dropping it drops
    the larger term -- the omission was unsafe, not conservative."""
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
    """`required` is three margins summed, with corrected envelope."""
    scene = geometry(None)
    assert scene.required == pytest.approx(
        scene.config.margin_safety + scene.config.margin_interp + scene.envelope
    )


def _shifted(
    scene: Geometry, body: CollisionPrimitive, along: np.ndarray, target: float
):
    """Move `body` along `along` so hanging pose clears it by `target`."""
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
    """One wall distance, inside required and outside both margins: accepted near
    base, refused near tool -- envelope is owed only by what hangs on the hinges."""
    scene = geometry(None)
    target = 0.5 * (scene.required + scene.required_rigid)
    tcp = _tcp(scene)

    # beside base: outriggers reach 2.62m in y, tool 2.9m out in x -- wall over base is base's alone
    near_base = CollisionPrimitive(
        id="near_base",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([0.0, 5.0, 0.0])),
        dimensions_m=np.array([1.0, 0.1, 3.0]),
    )
    _shifted(scene, near_base, np.array([0.0, -1.0, 0.0]), target)
    clearance, required, body = scene.margin(np.zeros(PLANNED_DOF))
    assert required == pytest.approx(scene.required_rigid)
    assert clearance > required
    assert body == "near_base"
    assert scene.is_valid(np.zeros(PLANNED_DOF))

    near_tool = CollisionPrimitive(
        id="near_tool",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), tcp + np.array([0.0, 1.5, 0.0])),
        dimensions_m=np.array([3.0, 0.1, 3.0]),
    )
    _shifted(scene, near_tool, np.array([0.0, -1.0, 0.0]), target)
    clearance, required, body = scene.margin(np.zeros(PLANNED_DOF))
    assert required == pytest.approx(scene.required)
    assert clearance < required
    assert body == "near_tool"
    assert not scene.is_valid(np.zeros(PLANNED_DOF))


class _Line:
    """Straight path in planned coordinates, for the certificate alone."""

    def __init__(self, start, goal):
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)

    def position(self, sigma):
        return self.start + sigma * (self.goal - self.start)


def test_check_path_stops_querying_bodies_its_bounds_vouch_for(monkeypatch):
    """Body at distance d cannot come nearer than d - motion per step, so far bodies
    get queried once then ride their bound -- second half shows bound is spent."""
    scene = geometry(None)
    tcp = _tcp(scene)
    far = CollisionPrimitive(
        id="far",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), tcp + np.array([0.0, 4.0, 0.0])),
        dimensions_m=np.array([10.0, 0.1, 3.0]),
    )
    # wall 0.6m behind base stays close so far one is skipped while it's still queried
    near = CollisionPrimitive(
        id="near",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([-3.0, 0.0, 0.0])),
        dimensions_m=np.array([0.1, 3.0, 3.0]),
    )
    scene.scene = [near, far]
    queried = []
    real = scene.model.collision_queries

    def counting(q, bodies, swinging=None):
        queried.append([body.id for body in bodies])
        return real(q, bodies, swinging)

    monkeypatch.setattr(scene.model, "collision_queries", counting)

    # slew away from the wall: 0.3 rad on the slewing axis
    scene.check_path(_Line(np.zeros(PLANNED_DOF), [-0.3, 0.0, 0.0, 0.0, 0.0]))
    assert queried[0] == ["near", "far"]
    # one body per step after: self row carries the smallest bound, other rides its bound
    assert all(len(bodies) == 1 for bodies in queried[1:])
    assert len(queried) > 5

    # slew towards it: wall is 4 m out, tool arcs to within the margin
    queried.clear()
    with pytest.raises(Exception, match="blocked at sigma"):
        scene.check_path(_Line(np.zeros(PLANNED_DOF), [1.3, 0.0, 0.0, 0.0, 0.0]))
    assert any("far" in bodies for bodies in queried[1:])
