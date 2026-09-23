"""crane_planner: reference-only for `/a2b_movement`, no `controller_manager` client."""

from __future__ import annotations

import hashlib
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
from crane_msgs.msg import CollisionScene, JointPath, PayloadEstimate
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
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
from .ocp import SLACK_SPENT, SolverNotExported, evaluate
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
#: The same plan as geometry, for a path-following cost; paired to the reference by stamp.
JOINT_PATH_TOPIC = "/crane/joint_path"
PLANNED_PATH_TOPIC = "/crane_planner/planned_path"
#: Legacy A2B's Cartesian path, old name kept for RViz; unused `joint_trajectory` latch dropped.
LEGACY_TCP_PATH_TOPIC = "tcp_path"
MARKERS_TOPIC = "/crane_planner/markers"
JOINT_STATES_TOPIC = "/joint_states"
COLLISION_SCENE_TOPIC = "/crane/collision_scene"
PAYLOAD_ESTIMATE_TOPIC = "/crane/payload_estimate"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
#: Last-request OCP result; private diagnostics like crane_mpc's `~/shadow_comparison`, latched.
SOLVER_STATS_TOPIC = "~/solver_stats"

#: Planning frame; assembly planner converts `world` to this, nothing downstream converts.
PLANNING_FRAME = "K0_mounting_base"

#: Samples of the planner's curve on `/crane/joint_path`. 30 clamped-cubic B-spline control
#: points fitted to 120 uniform samples reproduce the curved candidate to 1.3 mm at the tool
#: (measured, docs/features/path-following-mpc/brief.md), and the consumer refits anyway -- so
#: this only has to not be the limiting error.
PATH_SAMPLES = 120

SHAPES = {1: "box", 2: "cylinder", 3: "sphere"}


def latched(depth: int = 1) -> QoSProfile:
    """Reliable, keep-last, transient-local: reference and scene rows."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def volatile(depth: int = 1) -> QoSProfile:
    """Reliable, keep-last, volatile: measured state this plans from."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


#: `PlannerConfig` fields the yaml may set; three were once undeclared, silent defaults (footgun).
FLOAT_PARAMETERS = (
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
    "command_dead_time_s",
    "pump_flow_max",
    "pump_flow_planning_factor",
    "Ts",
    "truck_bed_thickness",
    "truck_headboard_thickness",
    "truck_headboard_height",
)
INT_PARAMETERS = (
    "ik_restarts",
    "max_lift_samples",
    "corridor_height_samples",
    "corridor_lateral_samples",
    "path_segments",
    "ocp_intervals",
    "ocp_max_iterations",
    "visualization_samples",
)
ARRAY_PARAMETERS = (
    "q_sway_max",
    "terminal_q_sway_max",
    "terminal_dq_sway_max",
    "dq_sway_max",
    "ddq_a_max",
    "dddq_a_max",
    "command_k",
    "command_u_min",
    "command_u_max",
    "command_lag_s",
    "truck_runge_dimensions",
    "truck_runge_stations",
)

OTHER_PARAMETERS = (
    "ocp_integrator",
    "max_input_age",
    "max_scene_age",
    "pendulum_state_deadline",
    "payload_estimate_deadline",
    "c3_feedforward",
)


class CranePlanner(Node):
    def __init__(self) -> None:
        super().__init__("crane_planner")
        self._declare()
        self.weights = self._weights()
        self.planner: Planner | None = None
        self.joint_states: deque = deque(maxlen=8)
        self.scene: CollisionScene | None = None
        self.estimate: PayloadEstimate | None = None

        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self._description, latched()
        )
        # Depth 10: two partial `/joint_states` messages back-to-back; depth-1 drops one unread.
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
        self.joint_path = self.create_publisher(JointPath, JOINT_PATH_TOPIC, latched())
        self.planned_path = self.create_publisher(Path, PLANNED_PATH_TOPIC, latched())
        self.legacy_tcp_path = self.create_publisher(Path, LEGACY_TCP_PATH_TOPIC, 10)
        self.markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, latched())
        self.solver_stats = self.create_publisher(
            DiagnosticArray, SOLVER_STATS_TOPIC, latched()
        )
        self.create_service(CalcMovement, A2B_MOVEMENT_SERVICE, self._a2b)
        self.get_logger().info(f"waiting for {ROBOT_DESCRIPTION_TOPIC}")

    def _declare(self) -> None:
        """Declare yaml knobs; constants (not literals) let a test catch an undeclared override."""
        defaults = PlannerConfig()
        for name in FLOAT_PARAMETERS:
            self.declare_parameter(name, float(getattr(defaults, name)))
        for name in INT_PARAMETERS:
            self.declare_parameter(name, int(getattr(defaults, name)))
        for name in ARRAY_PARAMETERS:
            self.declare_parameter(
                name, [float(value) for value in getattr(defaults, name)]
            )
        self.declare_parameter("ocp_integrator", str(defaults.ocp_integrator))
        for name, value in crane_weights.DEFAULTS.items():
            self.declare_parameter(f"weights.{name}", float(value))
        self.declare_parameter("max_input_age", 0.5)
        self.declare_parameter("max_scene_age", 10.0)
        # Test rig only: without it a scene-less collision request is refused (else empty-world).
        self.declare_parameter("allow_missing_scene", True)
        self.declare_parameter("pendulum_state_deadline", 0.15)
        self.declare_parameter("payload_estimate_deadline", 1.0)
        # C3 inversion in effort field; controllers lacking effort_field_is_feedforward reject it.
        self.declare_parameter("c3_feedforward", True)

    def _weights(self) -> dict:
        """Validate cost weights at construction: `weights.matrices` is called and discarded."""
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

    def _description(self, message: String) -> None:
        try:
            self.planner = Planner(
                message.data, self._config(), self.weights, build_missing=False
            )
        except SolverNotExported as missing:
            # Quit, not degrade: solver was hand-exported for this exact description.
            self.planner = None
            self.get_logger().fatal(str(missing))
            raise SystemExit(1) from missing
        except Exception as failure:  # bad description is not a crash
            self.planner = None
            self.get_logger().error(f"the robot description was refused: {failure}")
            return
        self.joint_names = list(canonical_joints())
        digest = hashlib.sha1(message.data.encode()).hexdigest()
        self.get_logger().info(
            f"planning for {Tool.PZS100.value} on the description at sha1 "
            f"{digest[:10]}, solver {self.planner.ocp.tree.name}"
        )

    def _joint_states(self, message: JointState) -> None:
        """Cache, don't route: routing needs description's joint names, which may lag."""
        self.joint_states.append(message)

    def _newest(self, names: set) -> JointState | None:
        for message in reversed(self.joint_states):
            if names <= set(message.name):
                return message
        return None

    def _collision_scene(self, message: CollisionScene) -> None:
        self.scene = message

    def _payload_estimate(self, message: PayloadEstimate) -> None:
        self.estimate = message

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

        # Passive half: missing-sway-as-zero is the stopped-start convention; no third answer.
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
            # Sway without rate: zero is honest only when the array is truly absent.
            if passive.velocity:
                dq_u[slot] = passive.velocity[where]
        return Start(q=q, dq_a=dq_a, dq_u=dq_u)

    def _primitives(self, avoid_collisions: bool) -> list:
        if self.scene is None:
            if avoid_collisions and self.get_parameter("allow_missing_scene").value:
                self.get_logger().warn(
                    "allow_missing_scene: planning against an empty world. The "
                    "trajectory is certified clear of nothing but the machine "
                    "itself -- do not run this on hardware."
                )
                return []
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

    def _a2b(self, request, response):
        """Answer a2b_movement over `Planner.plan`; `CalcMovement.Response` has no message field."""
        response.success = False
        response.trajectory = JointTrajectory()
        response.tcp_path = []
        if self.planner is None:
            refusal = f"no robot description has arrived on {ROBOT_DESCRIPTION_TOPIC}"
            self.get_logger().warn(f"a2b_movement refused: {refusal}")
            self._report(DiagnosticStatus.ERROR, refusal, {})
            return response

        try:
            # Payload first: tip-to-tool offset is read at the pose the tool hangs at, with it on.
            payload, _shape = translate_payload(request)
            start = translate_start(request)
            if start is None:
                start = self._start()
            else:
                # Supplied feasibility state, not a measurement: stamped now, no clock to use.
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
            # Log every refusal, not only the solver's: no numbers reads as "solve was fine".
            self._report(DiagnosticStatus.ERROR, str(refusal), refusal.stats)
            return response

        # Canonical eight; passive pair is free (OCP-solved) -- claimed sway, not settle point.
        legacy = self._trajectory(plan, self.start_stamp, range(GENERALIZED_DOF))
        # `publish_path` gates the topic only: response always carries `tcp_path`.
        if request.publish_path:
            self.legacy_tcp_path.publish(path)

        response.success = True
        response.trajectory = legacy
        response.tcp_path = path.poses
        return response

    def _run(
        self, start, position_m, yaw, payload, shape, avoid_collisions, speed_scale
    ):
        # Re-read live: lag only enters post-solve reconstruction, safe to change between requests.
        self.planner.config.command_lag_s = np.asarray(
            self.get_parameter("command_lag_s").value, dtype=float
        )
        primitives = self._primitives(avoid_collisions)
        # Goal+geometry drawn before solve, so a refusal still leaves them on screen.
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
        # Same stamp as the reference: that is what pairs the two forms of one plan.
        self.joint_path.publish(self._joint_path(plan, self.start_stamp))
        self.planned_path.publish(path)
        self.get_logger().info(plan.message)
        # WARN not OK when slack bought the answer: priced violation executes silently otherwise.
        # Same for an answer admitted at the iteration cap -- every row is met, but the duration
        # is not proven minimal and the cycle behind it is worth seeing from outside.
        level = (
            DiagnosticStatus.WARN
            if plan.timing.slack > SLACK_SPENT
            or plan.timing.stats["acados_status"] != 0
            else DiagnosticStatus.OK
        )
        self._report(level, plan.message, plan.timing.report())
        return plan, trajectory, path

    def _report(self, level: bytes, message: str, stats: dict) -> None:
        """Publish solver result; `level` is a `DiagnosticStatus` constant, `bytes` not int."""
        report = DiagnosticArray()
        report.header.stamp = self.get_clock().now().to_msg()
        report.status = [
            DiagnosticStatus(
                level=level,
                name=f"{self.get_name()}: trajectory OCP",
                message=message,
                values=[
                    KeyValue(key=key, value=f"{value:.6g}")
                    for key, value in stats.items()
                ],
            )
        ]
        self.solver_stats.publish(report)

    def _trajectory(self, plan, stamp, indices) -> JointTrajectory:
        """Two widths: actuated six for the reference, canonical eight for a2b_movement."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = stamp
        trajectory.joint_names = [self.joint_names[index] for index in indices]
        columns = list(indices)
        feedforward = self.get_parameter("c3_feedforward").value
        for when, position, velocity, acceleration, command in zip(
            plan.time, plan.q, plan.dq, plan.ddq, plan.effort
        ):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in position[columns]]
            point.velocities = [float(value) for value in velocity[columns]]
            point.accelerations = [float(value) for value in acceleration[columns]]
            if feedforward:
                point.effort = [float(value) for value in command[columns]]
            point.time_from_start.sec = int(when)
            point.time_from_start.nanosec = int((when - int(when)) * 1e9)
            trajectory.points.append(point)
        return trajectory

    def _joint_path(self, plan, stamp) -> JointPath:
        """Geometry off the planner's own curve, not the resample: no timing law in the path."""
        places = np.linspace(0.0, 1.0, PATH_SAMPLES)
        message = JointPath()
        message.header.stamp = stamp
        message.joint_names = [self.joint_names[index] for index in PLANNED_INDICES]
        # Row-major, one row per place: `evaluate` already returns (places, planned).
        message.q_path = evaluate(plan.timing.coefficients, places).reshape(-1).tolist()
        total = float(plan.timing.time[-1])
        message.duration.sec = int(total)
        message.duration.nanosec = int((total - int(total)) * 1e9)
        return message

    def _draw(self, markers: list) -> None:
        self.markers.publish(viz.clear(PLANNING_FRAME, self.get_clock().now().to_msg()))
        self.markers.publish(MarkerArray(markers=markers))

    def _path(self, plan) -> Path:
        """Visualization only."""
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
