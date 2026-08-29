"""
The `crane_planner` node: `/crane/plan_motion`, and the reference it answered with.

It is **not a second writer of the machine**. `crane_velocity_controller` is the
sole claimant of the six velocity command interfaces and which controller holds
that claim is the supervisor's decision alone, so this node publishes a
*reference* and not a command, holds no `controller_manager` client of any kind,
and is composed beside the manager rather than loaded into it.

Every name below is an absolute cross-node contract except `/robot_description`,
which a deployment remaps: the description composition publishes several and the
profile decides which one this stack runs on. A planner that solved the
kinematics of a different description from the one the controllers were
configured against would place the tool of a different crane.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from crane_model import ACTUATED_INDICES, Payload, Tool, canonical_joints
from crane_msgs.msg import CollisionScene, PayloadEstimate
from crane_msgs.srv import PlanMotion
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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

PLAN_MOTION_SERVICE = "/crane/plan_motion"
REFERENCE_TOPIC = "/crane/reference"
PLANNED_PATH_TOPIC = "/crane_planner/planned_path"
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
        self.planner: Planner | None = None
        # Both producers land here and are routed when they are read, not when
        # they arrive: the description is what says which joint name is which,
        # and it does not always arrive first.
        self.joint_states: deque = deque(maxlen=8)
        self.scene: CollisionScene | None = None
        self.estimate: PayloadEstimate | None = None
        self.standing = JointTrajectory()

        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self._description, latched()
        )
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._joint_states, volatile()
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
        self.create_service(PlanMotion, PLAN_MOTION_SERVICE, self._plan)
        self.get_logger().info(f"waiting for {ROBOT_DESCRIPTION_TOPIC}")

    # -- parameters -----------------------------------------------------------

    def _declare(self) -> None:
        """
        Declare every knob `config/crane_planner.yaml` carries.

        The machine's own numbers -- how far and how fast each joint may go --
        are deliberately absent: they are read out of the description the
        profile remaps this node onto, because a limit written down beside a
        node is a limit that drifts away from the description every controller
        in the same deployment was configured against.
        """
        defaults = PlannerConfig()
        self.declare_parameter("tool", defaults.tool.value)
        for name in (
            "kappa",
            "eps_pos",
            "eps_yaw",
            "ompl_time_budget",
            "ompl_extension_span",
            "ompl_unbounded_margin",
            "check_resolution",
            "min_check_resolution",
            "sway_weight",
            "input_weight",
            "sigma_rate_min",
            "sigma_rate_max",
            "sigma_accel_max",
            "max_wall_clock",
            "tolerance",
            "pump_flow_max",
            "pump_flow_planning_factor",
            "system_pressure_pa",
            "Ts",
            "truck_bed_thickness",
        ):
            self.declare_parameter(name, float(getattr(defaults, name)))
        for name in (
            "ik_restarts",
            "ompl_seed",
            "shortcut_attempts",
            "max_check_samples",
            "intervals",
            "max_iterations",
        ):
            self.declare_parameter(name, int(getattr(defaults, name)))
        for name in (
            "q_sway_max",
            "dq_sway_max",
            "ddq_a_max",
            "truck_runge_dimensions",
            "truck_runge_stations",
        ):
            self.declare_parameter(
                name, [float(value) for value in getattr(defaults, name)]
            )
        self.declare_parameter("max_input_age", 0.5)
        self.declare_parameter("max_scene_age", 10.0)
        self.declare_parameter("pendulum_state_deadline", 0.15)
        self.declare_parameter("payload_estimate_deadline", 1.0)

    def _config(self) -> PlannerConfig:
        config = PlannerConfig(tool=Tool(self.get_parameter("tool").value))
        for field in vars(config):
            if field == "tool" or not self.has_parameter(field):
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
            self.planner = Planner(message.data, self._config())
        except Exception as failure:  # a bad description is not a crash
            self.planner = None
            self.get_logger().error(f"the robot description was refused: {failure}")
            return
        self.joint_names = list(canonical_joints(self.planner.config.tool))
        self.get_logger().info(f"planning for {self.planner.config.tool.value}")

    def _joint_states(self, message: JointState) -> None:
        """
        Keep the recent messages; which is which is decided when they are read.

        `/joint_states` carries **partial** messages: `joint_state_broadcaster`
        publishes the actuated six and `tip_tilt_state_broadcaster` the passive
        pair, at different rates and from different hardware. Telling them apart
        needs the joint names, which come from the description -- so routing
        here would silently drop every state that arrived before it.
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
        for index in PASSIVE_INDICES:
            q[index] = passive.position[names.index(self.joint_names[index])]
        return Start(q=q, dq_a=dq_a)

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

    def _plan(self, request, response):
        """
        Answer a placement goal, or refuse and say what is still standing.

        A refusal hands back the trajectory that is still on the reference topic
        rather than an empty one, so a caller can tell "nothing changed" from
        "the machine has no plan".
        """
        response.trajectory = self.standing
        response.tcp_path = []
        if self.planner is None:
            response.success = False
            response.message = (
                f"no robot description has arrived on {ROBOT_DESCRIPTION_TOPIC}"
            )
            return response
        if request.goal.header.frame_id != PLANNING_FRAME:
            response.success = False
            response.message = (
                f"the goal is in '{request.goal.header.frame_id}'; this planner converts "
                f"nothing and plans only in '{PLANNING_FRAME}'"
            )
            return response

        try:
            start = self._start()
            primitives = self._primitives(request.avoid_collisions)
            payload, shape = self._payload(request.payload)
            orientation = request.goal.pose.orientation
            rotation = pin.Quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            ).toRotationMatrix()
            plan = self.planner.plan(
                start,
                np.array(
                    [
                        request.goal.pose.position.x,
                        request.goal.pose.position.y,
                        request.goal.pose.position.z,
                    ]
                ),
                float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                payload=payload if payload.valid else None,
                payload_shape=shape,
                scene=primitives,
                avoid_collisions=request.avoid_collisions,
                speed_scale=request.speed_scale,
            )
        except PlanningError as refusal:
            response.success = False
            response.message = f"{refusal}; the standing reference is unchanged"
            self.get_logger().warn(response.message)
            return response

        trajectory = self._trajectory(plan, self.start_stamp)
        path = self._path(plan)
        self.standing = trajectory
        self.reference.publish(trajectory)
        self.planned_path.publish(path)
        response.success = True
        response.message = plan.message
        response.trajectory = trajectory
        response.tcp_path = path.poses
        self.get_logger().info(plan.message)
        return response

    def _trajectory(self, plan, stamp) -> JointTrajectory:
        """
        Build the actuated six, in canonical order, on the measurement's own clock.

        `header.frame_id` is empty on purpose: joint space has no frame.
        """
        trajectory = JointTrajectory()
        trajectory.header.stamp = stamp
        trajectory.joint_names = [self.joint_names[index] for index in ACTUATED_INDICES]
        for when, position, velocity in zip(plan.time, plan.q_a, plan.dq_a):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in position]
            point.velocities = [float(value) for value in velocity]
            point.time_from_start.sec = int(when)
            point.time_from_start.nanosec = int((when - int(when)) * 1e9)
            trajectory.points.append(point)
        return trajectory

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
