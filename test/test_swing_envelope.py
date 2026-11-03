"""`Geometry.envelope` must bound the swing of both passive hinges, not one."""

from itertools import product
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
from crane_model import PASSIVE_INDICES as SWAY
from crane_planning.config import (
    PLANNED_DOF,
    TOOL_INDEX,
    PlannerConfig,
    passive_equilibrium,
    read_limits,
)
from crane_planning.geometry import Geometry, payload_primitive
from crane_planning.planner import payload_parameters

Q_TOOL = 0.35
#: (shape, dimensions, offset from K8), small block
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
    """One-hinge formula returned 0.153m vs real 0.250m at the carried point."""
    scene = geometry(payload_shape)
    swing = measured_swing(
        scene.model, np.abs(scene.config.q_sway_max), scene._carried_reach()
    )
    assert scene.envelope >= swing


def test_a_gripped_block_owes_its_own_extent_not_the_k8_drop():
    """Reach is the farthest corner from TCP, half a diagonal past the centre: the
    full one refused a lift clearing 0.99 m against 1.05 m."""
    shape = ("cylinder", np.array([0.6, 0.6, 0.9]), np.array([-0.003, 0.001, 1.048]))
    scene = geometry(shape)
    carried = payload_primitive(
        scene.model, scene.configuration(np.zeros(PLANNED_DOF)), shape
    )
    corners = [
        carried.pose_in_mounting_base.act(0.5 * np.asarray(signs) * shape[1])
        for signs in product((-1.0, 1.0), repeat=3)
    ]
    farthest = max(np.linalg.norm(corner - _tcp(scene)) for corner in corners)
    assert scene._carried_reach() >= farthest
    # the old |offset| + full diagonal, whatever the sway box makes of it
    assert scene._carried_reach() < np.linalg.norm(shape[2]) + np.linalg.norm(shape[1])


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
    target = (
        scene.required_rigid + 0.1
    )  # inside required; the tool swings 0.19 m sideways
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
    # refused on the swept bound against the rigid margins: the wall is beside the tool
    assert required == pytest.approx(scene.required_rigid)
    assert clearance < required
    assert body == "near_tool"
    assert not scene.is_valid(np.zeros(PLANNED_DOF))


class _Line:
    """Straight path in planned coordinates, for the certificate alone."""

    def __init__(self, start, goal):
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)

    def position(self, s):
        return self.start + s * (self.goal - self.start)


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

    def counting(q, bodies, *args):
        queried.append([body.id for body in bodies])
        return real(q, bodies, *args)

    monkeypatch.setattr(scene.model, "collision_queries", counting)

    # slew away from the wall: 0.3 rad on the slewing axis
    scene.check_path(_Line(np.zeros(PLANNED_DOF), [-0.3, 0.0, 0.0, 0.0, 0.0]))
    assert queried[0] == ["near", "far"]
    # one body per step after: self row carries the smallest bound, other rides its bound
    assert all(len(bodies) == 1 for bodies in queried[1:])
    assert len(queried) > 5

    # slew towards it: wall is 4 m out, tool arcs to within the margin
    queried.clear()
    with pytest.raises(Exception, match="blocked at s"):
        scene.check_path(_Line(np.zeros(PLANNED_DOF), [1.3, 0.0, 0.0, 0.0, 0.0]))
    assert any("far" in bodies for bodies in queried[1:])


CBS_BLOCK = (
    "cylinder",
    np.array([1.082, 1.082, 0.6]),
    np.array([-0.003, 0.001, 1.048]),
)


def _block_near(direction, target: float) -> Geometry:
    """Gripped CBS block, a second one `target` from it along `direction` at hanging."""
    scene = geometry(CBS_BLOCK)
    carried = payload_primitive(
        scene.model, scene.configuration(np.zeros(PLANNED_DOF)), CBS_BLOCK
    )
    along = np.asarray(direction, dtype=float)
    neighbour = CollisionPrimitive(
        id="block_1",
        shape="box",
        pose_in_mounting_base=pin.SE3(
            np.eye(3), carried.pose_in_mounting_base.translation + 3.0 * along
        ),
        dimensions_m=np.array([0.6, 0.6, 0.6]),
    )
    scene.scene = [neighbour]
    # payload-to-machine (0.99 m) is the nearest row until the neighbour closes in
    for _ in range(20):
        gap = scene.clearance(np.zeros(PLANNED_DOF)) - target
        if abs(gap) < 1e-4:
            return scene
        neighbour.pose_in_mounting_base.translation -= along * gap
    raise AssertionError("neighbour never settled at the target clearance")


@pytest.mark.parametrize(
    ("direction", "clear"), [((0.0, 0.0, -1.0), True), ((1.0, 0.0, 0.0), False)]
)
def test_sway_is_charged_where_it_swings_not_as_a_ball(direction, clear):
    """0.33 m from a neighbour: below, a swing only lifts the block (live lift refused
    against the 0.86 m ball); beside, the box swings it 0.3 m across and must refuse."""
    scene = _block_near(direction, 0.33)
    assert scene.clearance(np.zeros(PLANNED_DOF)) < scene.required
    assert scene.is_valid(np.zeros(PLANNED_DOF)) is clear
    if not clear:
        assert "block_1" in scene.margin(np.zeros(PLANNED_DOF))[2]


def test_swept_bound_never_exceeds_the_swept_distance():
    """The certificate: no sway inside the box comes nearer than `_swept` claims."""
    scene = _block_near((0.7, 0.0, -0.7), 0.4)
    q = scene.configuration(np.zeros(PLANNED_DOF))
    bound, _ = scene._swept(q, scene.scene)
    box = np.abs(scene.config.q_sway_max)
    nearest = np.inf
    for a in np.linspace(-box[0], box[0], 21):
        for b in np.linspace(-box[1], box[1], 21):
            swung = q.copy()
            swung[list(SWAY)] += (a, b)
            distances = scene._distances(swung, swinging=True, scene=scene.scene)
            nearest = min(nearest, min(distances.values()))
    assert bound <= nearest


@pytest.mark.parametrize("payload_shape", [None, CBS_BLOCK])
@pytest.mark.parametrize(
    ("direction", "distance"),
    [((-0.6, 0.79, -0.06), 2.22), ((0.69, -0.71, -0.12), 2.22)],
)
def test_envelope_bounds_how_near_the_sway_box_brings_any_point(
    payload_shape, direction, distance
):
    """`_decide` accepts on `envelope` alone, skipping `_swept`. Composing on-axis TCP
    chords allowed 0.25 m where an empty gripper's rails swing 0.53 m nearer."""
    scene = geometry(payload_shape)
    probe = CollisionPrimitive(
        id="probe",
        shape="sphere",
        pose_in_mounting_base=pin.SE3(
            np.eye(3), _tcp(scene) + distance * np.asarray(direction)
        ),
        dimensions_m=np.array([0.02, 0.02, 0.02]),
    )
    q = scene.configuration(np.zeros(PLANNED_DOF))
    hanging = scene._distances(q, swinging=True, scene=[probe])["probe"]
    box = np.abs(scene.config.q_sway_max)
    nearest = hanging
    for a in np.linspace(-box[0], box[0], 21):
        for b in np.linspace(-box[1], box[1], 21):
            swung = q.copy()
            swung[list(SWAY)] += (a, b)
            distances = scene._distances(swung, swinging=True, scene=[probe])
            nearest = min(nearest, distances["probe"])
    assert hanging - nearest <= scene.envelope


def test_a_far_body_does_not_ride_the_near_ones_search(monkeypatch):
    """Live: 14 bodies queried in every cell the nearest one split, 80% of a carry plan."""
    scene = _block_near((1.0, 0.0, 0.0), 0.5)
    q = scene.configuration(np.zeros(PLANNED_DOF))
    far = CollisionPrimitive(
        id="far",
        shape="box",
        pose_in_mounting_base=pin.SE3(np.eye(3), np.array([-20.0, 0.0, 0.0])),
        dimensions_m=np.array([1.0, 1.0, 1.0]),
    )
    queried = []
    real = scene.model.collision_queries

    def counting(q, bodies, *args):
        queried.extend(body.id for body in bodies)
        return real(q, bodies, *args)

    monkeypatch.setattr(scene.model, "collision_queries", counting)
    alone = scene._swept(q, scene.scene)
    # block-against-machine is its own search; the far body costs one query, its root cell
    scene._swept(q, scene.scene + [far])
    assert queried.count("far") == 1
    assert scene._swept(q, scene.scene + [far]) == alone


def test_carried_swept_bounds_refuse_what_a_fresh_search_refuses():
    """Slew onto a neighbour beside the block, refused on the swept bound only (a bound
    ridden without spending it passes this path)."""
    scene = _block_near((1.0, 0.0, 0.0), 0.33)
    start = np.zeros(PLANNED_DOF)
    start[0] = -0.3
    line = _Line(start, np.zeros(PLANNED_DOF))
    with pytest.raises(Exception, match="blocked at s") as carried:
        scene.check_path(line)
    scene.swept_step_bound = lambda first, second: np.inf
    with pytest.raises(Exception, match="blocked at s") as fresh:
        scene.check_path(line)
    assert str(carried.value) == str(fresh.value)
