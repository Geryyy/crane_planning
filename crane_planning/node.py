"""
The `crane_planner` node: `/a2b_movement`, and the reference it answered with.

**Not a second writer of the machine.** `crane_velocity_controller` is the sole
claimant of the six velocity command interfaces, and which controller holds that
claim is the supervisor's decision alone. So this node publishes a *reference*,
holds no `controller_manager` client, and is composed beside the manager rather
than loaded into it.

Every name below is an absolute cross-node contract except `/robot_description`,
which a deployment remaps -- the description composition publishes several and
the profile picks one. A planner solving the kinematics of a different
description from the one the controllers were configured against would place the
tool of a different crane.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from crane_model import (
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    Payload,
    Tool,
    canonical_joints,
)
from crane_msgs.msg import CollisionScene, PayloadEstimate
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

from timber_crane_planning_interfaces.srv import CalcMovement

from . import markers as viz
from . import weights as crane_weights
from .a2b import (
    A2B_MOVEMENT_SERVICE,
    translate_payload,
    translate_request,
    translate_start,
)
from .planner import (
    PASSIVE_INDICES,
    PLANNED_INDICES,
    TOOL_INDEX,
    CollisionPrimitive,
    Planner,
    PlannerConfig,
    PlanningError,
    Start,
    pin,
)

REFERENCE_TOPIC = "/crane/reference"
PLANNED_PATH_TOPIC = "/crane_planner/planned_path"
#: The Cartesian path the legacy A2B server published beside its service
#: answer, on the name it published it: relative, from a node in the root
#: namespace, so that an operator's RViz shows the same thing it always did.
#:
#: The legacy server also latched the trajectory itself on `joint_trajectory`,
#: and that one is **deliberately not carried over**. Nothing in the workspace
#: subscribes to it -- the trajectory controller listens on its own
#: `~/joint_trajectory` -- so it is a debug artefact, and publishing it would
#: give this node a second `JointTrajectory` publisher. That count is what the
#: launch contract reads as evidence that the planner cannot be a second
#: command producer, and spending it on a topic nobody reads is a bad trade.
LEGACY_TCP_PATH_TOPIC = "tcp_path"
#: Everything a plan looks like, in one array: the path, the tool swept along
#: it, the goal, and the bodies the plan was actually checked against.
MARKERS_TOPIC = "/crane_planner/markers"
JOINT_STATES_TOPIC = "/joint_states"
COLLISION_SCENE_TOPIC = "/crane/collision_scene"
PAYLOAD_ESTIMATE_TOPIC = "/crane/payload_estimate"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"

#: The frame planning geometry is in. The assembly planner converts `world` to
#: this before it calls; nothing downstream converts, so a goal that arrives in
#: any other frame is refused rather than assumed.
PLANNING_FRAME = "K0_mounting_base"

SHAPES = {1: "box", 2: "cylinder", 3: "sphere"}


def latched(depth: int = 1) -> QoSProfile:
    """Reliable, keep-last, transient-local: the reference and the scene rows."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def volatile(depth: int = 1) -> QoSProfile:
    """Reliable, keep-last, volatile: the measured state this plans from."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


class CranePlanner(Node):
    def __init__(self) -> None:
        super().__init__("crane_planner")
        self._declare()
        #: The OCP's cost weights, read and checked at construction.
        self.weights = self._weights()
        self.planner: Planner | None = None
        # Both producers land here and are routed when they are read, not when
        # they arrive: the description is what says which joint name is which,
        # and it does not always arrive first.
        self.joint_states: deque = deque(maxlen=8)
        self.scene: CollisionScene | None = None
        self.estimate: PayloadEstimate | None = None

        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self._description, latched()
        )
        # Depth 10 and not 1: `/joint_states` carries **two** partial messages
        # from two producers, published back to back. A depth-1 queue can drop
        # one of them for good -- the second overwrites the first before the
        # subscription is served -- and the planner then never sees a state
        # carrying the actuated six. The legacy A2B server read this topic at
        # depth 5 for the same reason.
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._joint_states, volatile(10)
        )
        self.create_subscription(
            CollisionScene, COLLISION_SCENE_TOPIC, self._collision_scene, latched()
        )
        self.create_subscription(
            PayloadEstimate, PAYLOAD_ESTIMATE_TOPIC, self._payload_estimate, latched()
        )
        self.reference = self.create_publisher(
            JointTrajectory, REFERENCE_TOPIC, latched()
        )
        self.planned_path = self.create_publisher(Path, PLANNED_PATH_TOPIC, latched())
        self.legacy_tcp_path = self.create_publisher(Path, LEGACY_TCP_PATH_TOPIC, 10)
        # Transient-local, because an RViz started after the plan should still
        # see it -- a plan is a standing decision, not a stream.
        self.markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, latched())
        # The retained timber contract, on the same node and over the same
        # planner: one adapter, no second set of limits.
        self.create_service(CalcMovement, A2B_MOVEMENT_SERVICE, self._a2b)
        self.get_logger().info(f"waiting for {ROBOT_DESCRIPTION_TOPIC}")

    # -- parameters -----------------------------------------------------------

    def _declare(self) -> None:
        """
        Declare every knob `config/crane_planner.yaml` carries.

        The machine's numbers -- how far and how fast each joint may go -- are
        deliberately absent, read from the description instead: a limit written
        beside a node drifts away from the description every controller in the
        same deployment was configured against.
        """
        defaults = PlannerConfig()
        for name in (
            "kappa",
            "eps_pos",
            "eps_yaw",
            "margin_safety",
            "margin_interp",
            "tool_radius",
            "corridor_clearance",
            "corridor_height_step",
            "corridor_lateral_step",
            "ocp_horizon",
            "ocp_duration_min",
            "ocp_duration_max",
            "ocp_tolerance",
            "ocp_slack_price",
            "levenberg_marquardt",
            "pump_flow_max",
            "pump_flow_planning_factor",
            "Ts",
            "truck_bed_thickness",
            "truck_headboard_thickness",
            "truck_headboard_height",
        ):
            self.declare_parameter(name, float(getattr(defaults, name)))
        for name in (
            "ik_restarts",
            "max_lift_samples",
            "corridor_height_samples",
            "corridor_lateral_samples",
            "path_segments",
            "ocp_intervals",
            "ocp_max_iterations",
            "visualization_samples",
        ):
            self.declare_parameter(name, int(getattr(defaults, name)))
        self.declare_parameter("ocp_integrator", str(defaults.ocp_integrator))
        for name in (
            "q_sway_max",
            "terminal_q_sway_max",
            "terminal_dq_sway_max",
            "dq_sway_max",
            "ddq_a_max",
            "dddq_a_max",
            "truck_runge_dimensions",
            "truck_runge_stations",
        ):
            self.declare_parameter(
                name, [float(value) for value in getattr(defaults, name)]
            )
        # The OCP's cost weights, nested under `weights` as the yaml writes them.
        # Every residual row is dimensionless, so these are preferences and a
        # scalar sets a whole block; `weights.time` prices the horizon row and is
        # the minimum-time objective.
        for name, value in crane_weights.DEFAULTS.items():
            self.declare_parameter(f"weights.{name}", float(value))
        self.declare_parameter("max_input_age", 0.5)
        self.declare_parameter("max_scene_age", 10.0)
        self.declare_parameter("pendulum_state_deadline", 0.15)
        self.declare_parameter("payload_estimate_deadline", 1.0)

    def _weights(self) -> dict:
        """
        Read the cost weights once, and check them once.

        `weights.matrices` is called and its answer discarded on purpose: a width
        that does not match the residual, or a negative price, is a configuration
        error and belongs at construction, not on the first plan a caller asks
        for.
        """
        weights = {
            name: self.get_parameter(f"weights.{name}").value
            for name in crane_weights.DEFAULTS
        }
        crane_weights.matrices(weights)
        return weights

    def _config(self) -> PlannerConfig:
        config = PlannerConfig()
        for field in vars(config):
            if not self.has_parameter(field):
                continue
            value = self.get_parameter(field).value
            setattr(
                config,
                field,
                np.asarray(value, dtype=float) if isinstance(value, list) else value,
            )
        return config

    def _age(self, header) -> float:
        stamp = rclpy.time.Time.from_msg(header.stamp)
        return float((self.get_clock().now() - stamp).nanoseconds) * 1e-9

    # -- inputs ---------------------------------------------------------------

    def _description(self, message: String) -> None:
        try:
            self.planner = Planner(message.data, self._config(), self.weights)
        except Exception as failure:  # a bad description is not a crash
            self.planner = None
            self.get_logger().error(f"the robot description was refused: {failure}")
            return
        self.joint_names = list(canonical_joints())
        self.get_logger().info(f"planning for {Tool.PZS100.value}")

    def _joint_states(self, message: JointState) -> None:
        """
        Keep the recent messages; which is which is decided when they are read.

        `/joint_states` carries actuated and passive joints alike, but telling
        them apart needs the joint names, which come from the description --
        routing here would silently drop every state that arrived before it.
        """
        self.joint_states.append(message)

    def _newest(self, names: set) -> JointState | None:
        """Return the newest cached message carrying all of `names`."""
        for message in reversed(self.joint_states):
            if names <= set(message.name):
                return message
        return None

    def _collision_scene(self, message: CollisionScene) -> None:
        self.scene = message

    def _payload_estimate(self, message: PayloadEstimate) -> None:
        self.estimate = message

    # -- the start state ------------------------------------------------------

    def _start(self) -> Start:
        actuated = self._newest({self.joint_names[index] for index in ACTUATED_INDICES})
        if actuated is None:
            raise PlanningError(
                f"no message on {JOINT_STATES_TOPIC} carries the actuated six"
            )
        age = self._age(actuated.header)
        if age > self.get_parameter("max_input_age").value:
            raise PlanningError(
                f"the newest joint state is {age:.2f} s old, past the "
                f"{self.get_parameter('max_input_age').value:.2f} s this plans from"
            )
        if not actuated.velocity:
            raise PlanningError(
                "the joint state carries no velocity array, and a measurement that "
                "does not say whether the machine is moving is not one to plan from"
            )

        names = list(actuated.name)
        q = np.zeros(8)
        dq_a = np.zeros(len(PLANNED_INDICES))
        for slot, index in enumerate(ACTUATED_INDICES):
            where = names.index(self.joint_names[index])
            q[index] = actuated.position[where]
            if index != TOOL_INDEX:
                dq_a[slot] = actuated.velocity[where]
        self.start_stamp = actuated.header.stamp

        # The passive half, and what happens when it is not there. Reading an
        # absent sway estimate as zero is the stopped-start convention this
        # refusal exists to remove; there is no third answer.
        deadline = self.get_parameter("pendulum_state_deadline").value
        passive = self._newest({self.joint_names[index] for index in PASSIVE_INDICES})
        if passive is None:
            raise PlanningError(
                "the passive joint state has never arrived, so the sway the plan "
                "would have to close on is unknown"
            )
        age = self._age(passive.header)
        if age > deadline:
            raise PlanningError(
                f"the passive joint state is {age:.3f} s old, past its {deadline:.3f} s "
                "deadline: the sway the plan would have to close on is stale"
            )
        names = list(passive.name)
        dq_u = np.zeros(len(PASSIVE_INDICES))
        for slot, index in enumerate(PASSIVE_INDICES):
            where = names.index(self.joint_names[index])
            q[index] = passive.position[where]
            # A broadcaster that publishes the sway without its rate leaves the
            # OCP no boundary condition for it; zero is the honest reading only
            # when the array is genuinely absent.
            if passive.velocity:
                dq_u[slot] = passive.velocity[where]
        return Start(q=q, dq_a=dq_a, dq_u=dq_u)

    def _primitives(self, avoid_collisions: bool) -> list:
        if self.scene is None:
            return None if avoid_collisions else []
        age = self._age(self.scene.header)
        if age > self.get_parameter("max_scene_age").value:
            if avoid_collisions:
                raise PlanningError(
                    f"the collision scene is {age:.1f} s old: a trajectory certified "
                    "clear of geometry that may have moved is a false statement"
                )
            self.get_logger().warn(f"planning without the {age:.1f} s old scene")
            return []
        primitives = []
        for entry in self.scene.primitives:
            if entry.shape not in SHAPES:
                raise PlanningError(
                    f"scene primitive '{entry.id}' has an unknown shape"
                )
            primitives.append(
                CollisionPrimitive(
                    id=entry.id,
                    shape=SHAPES[entry.shape],
                    pose_in_mounting_base=pin.XYZQUATToSE3(
                        np.array(
                            [
                                entry.pose.position.x,
                                entry.pose.position.y,
                                entry.pose.position.z,
                                entry.pose.orientation.x,
                                entry.pose.orientation.y,
                                entry.pose.orientation.z,
                                entry.pose.orientation.w,
                            ]
                        )
                    ),
                    dimensions_m=np.array(
                        [entry.dimensions.x, entry.dimensions.y, entry.dimensions.z]
                    ),
                    structural=entry.structural,
                )
            )
        return primitives

    def _payload(self, declared):
        """Return the declared payload, overridden by a fresh and valid estimate."""
        mass = float(declared.mass)
        com = np.array([declared.com.x, declared.com.y, declared.com.z])
        if self.estimate is not None and self.estimate.valid:
            age = self._age(self.estimate.header)
            if age <= self.get_parameter("payload_estimate_deadline").value:
                mass = float(self.estimate.payload.mass)
                com = np.array(
                    [
                        self.estimate.payload.com.x,
                        self.estimate.payload.com.y,
                        self.estimate.payload.com.z,
                    ]
                )
        payload = Payload(
            mass_kg=mass,
            center_of_mass_k8_m=com,
            inertia_k8_kg_m2=np.zeros((3, 3)),
            valid=mass > 0.0,
        )
        shape = None
        if declared.shape in SHAPES:
            dimensions = np.array(
                [declared.dimensions.x, declared.dimensions.y, declared.dimensions.z]
            )
            if np.all(dimensions > 0.0):
                shape = (SHAPES[declared.shape], dimensions, com)
        return payload, shape

    # -- the service ----------------------------------------------------------

    def _a2b(self, request, response):
        """
        Answer the retained `a2b_movement` contract over the same planner.

        Nothing here plans: `a2b` maps the request onto `Planner.plan`, this runs
        it. The node's only service -- the native `/crane/plan_motion` had no
        caller and was removed -- so also the only path publishing
        `/crane/reference`.

        `CalcMovement.Response` has **no message field**, so a refusal cannot say
        why. It goes to the log and the trajectory is left empty, as the legacy
        server left it.
        """
        response.success = False
        response.trajectory = JointTrajectory()
        response.tcp_path = []
        if self.planner is None:
            self.get_logger().warn(
                f"a2b_movement refused: no robot description has arrived on "
                f"{ROBOT_DESCRIPTION_TOPIC}"
            )
            return response

        try:
            # The payload first, because the offset from the tip pivot to the
            # tool is read at the pose the tool hangs at *with it on*.
            payload, _shape = translate_payload(request)
            start = translate_start(request)
            if start is None:
                start = self._start()
            else:
                # A supplied feasibility state, not a measurement: it is stamped
                # now, because there is no measurement whose clock to use.
                self.start_stamp = self.get_clock().now().to_msg()
            offset = self.planner.tip_to_tcp_offset(
                payload, request.phi_tool_n, start.q_tool
            )
            goal = translate_request(request, offset)
            plan, _trajectory, path = self._run(
                start,
                goal.position_m,
                goal.yaw,
                goal.payload,
                goal.payload_shape,
                goal.avoid_collisions,
                goal.speed_scale,
            )
        except PlanningError as refusal:
            self.get_logger().warn(f"a2b_movement refused: {refusal}")
            return response

        # The canonical eight, because that is what the trajectory controller
        # this answer is fed to is configured with. The passive pair costs
        # nothing to supply -- the OCP solved for it -- and it is the sway the
        # plan actually claims rather than the pose the tool would settle to.
        legacy = self._trajectory(plan, self.start_stamp, range(GENERALIZED_DOF))
        # `publish_path` gates the topic and nothing else: the answer carries
        # `tcp_path` either way, exactly as the legacy server answered it.
        if request.publish_path:
            self.legacy_tcp_path.publish(path)

        response.success = True
        response.trajectory = legacy
        response.tcp_path = path.poses
        return response

    def _run(
        self, start, position_m, yaw, payload, shape, avoid_collisions, speed_scale
    ):
        """Plan, then publish the reference and the drawing."""
        primitives = self._primitives(avoid_collisions)
        # The goal and the geometry are drawn **before** the solve, so that a
        # refusal leaves them on screen. "It said no" and "it said no, and here
        # is the runge it would have hit" are very different messages.
        checked = self.planner.prepare_scene(primitives, avoid_collisions)
        stamp = self.get_clock().now().to_msg()
        standing = viz.scene(checked, PLANNING_FRAME, stamp) + viz.goal(
            position_m, yaw, PLANNING_FRAME, stamp
        )
        self._draw(standing)

        plan = self.planner.plan(
            start,
            position_m,
            yaw,
            payload=payload,
            payload_shape=shape,
            scene=primitives,
            avoid_collisions=avoid_collisions,
            speed_scale=speed_scale,
        )
        self._draw(
            standing
            + viz.plan(
                self.planner.model,
                plan,
                PLANNING_FRAME,
                stamp,
                shape if avoid_collisions else None,
            )
        )
        trajectory = self._trajectory(plan, self.start_stamp, ACTUATED_INDICES)
        path = self._path(plan)
        self.reference.publish(trajectory)
        self.planned_path.publish(path)
        self.get_logger().info(plan.message)
        return plan, trajectory, path

    def _trajectory(self, plan, stamp, indices) -> JointTrajectory:
        """
        Build a trajectory over `indices` of the canonical eight.

        Two callers, two widths, and the difference is not cosmetic. The native
        reference carries the **actuated six**, what `crane_velocity_controller`
        claims. The `a2b_movement` answer carries the **canonical eight**: the
        trajectory controller consuming it lists the passive tip and tilt joints
        as state -- it commands six and tracks eight -- so a six-wide goal is
        rejected.

        `header.frame_id` is empty on purpose: joint space has no frame.
        """
        trajectory = JointTrajectory()
        trajectory.header.stamp = stamp
        trajectory.joint_names = [self.joint_names[index] for index in indices]
        columns = list(indices)
        for when, position, velocity in zip(plan.time, plan.q, plan.dq):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in position[columns]]
            point.velocities = [float(value) for value in velocity[columns]]
            point.time_from_start.sec = int(when)
            point.time_from_start.nanosec = int((when - int(when)) * 1e9)
            trajectory.points.append(point)
        return trajectory

    def _draw(self, markers: list) -> None:
        """Replace the drawing wholesale: a `DELETEALL`, then this plan's set."""
        self.markers.publish(viz.clear(PLANNING_FRAME, self.get_clock().now().to_msg()))
        self.markers.publish(MarkerArray(markers=markers))

    def _path(self, plan) -> Path:
        """Build the tool's own geometry, for the operator. Visualization only."""
        path = Path()
        path.header.frame_id = PLANNING_FRAME
        path.header.stamp = self.get_clock().now().to_msg()
        for position in plan.tcp:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
                float(value) for value in position
            )
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CranePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
