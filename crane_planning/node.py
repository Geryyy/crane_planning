"""
`crane_planner` node: `/a2b_movement` + the reference it answers with.

**Not a second writer of the machine.** `crane_velocity_controller` is sole
claimant of six velocity command interfaces; supervisor alone grants that claim.
So: reference only, no `controller_manager` client, composed beside manager.

Names below are absolute cross-node contracts except `/robot_description`,
remapped per deployment (several published, profile picks one). Wrong
description -> tool of a different crane.
"""

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
from crane_msgs.msg import CollisionScene, PayloadEstimate
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
from .ocp import SLACK_SPENT, SolverNotExported
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
#: Cartesian path legacy A2B server published beside its service answer. Kept
#: on its old relative name so operator RViz is unchanged.
#:
#: Legacy also latched trajectory on `joint_trajectory` -- **not carried over**:
#: nothing subscribes (trajectory controller uses its own `~/joint_trajectory`),
#: and it would make this a second `JointTrajectory` publisher -- the count the
#: launch contract reads as proof planner is no second command producer.
LEGACY_TCP_PATH_TOPIC = "tcp_path"
#: Whole plan in one array: path, swept tool, goal, bodies checked against.
MARKERS_TOPIC = "/crane_planner/markers"
JOINT_STATES_TOPIC = "/joint_states"
COLLISION_SCENE_TOPIC = "/crane/collision_scene"
PAYLOAD_ESTIMATE_TOPIC = "/crane/payload_estimate"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
#: What OCP did last request, converged or not. **Private**, like `crane_mpc`'s
#: `~/shadow_comparison`: diagnostics only, nothing decides on it, must not sit
#: one remap from looking like a contract. Latched -- asked after the fact.
SOLVER_STATS_TOPIC = "~/solver_stats"

#: Frame of planning geometry. Assembly planner converts `world` to this before
#: calling; nothing downstream converts, so other frames refused, not assumed.
PLANNING_FRAME = "K0_mounting_base"

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


#: Every `PlannerConfig` field `config/crane_planner.yaml` may set, by type.
#: `command_k`, `command_u_min`, `command_u_max` were once absent here: yaml set
#: them, nothing declared them, dataclass defaults ran silently. They matched,
#: no number moved.
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

#: Declared outside the three loops; the yaml check needs them too.
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
        #: OCP cost weights, read + checked at construction.
        self.weights = self._weights()
        self.planner: Planner | None = None
        # Both producers land here, routed on read: description says which joint
        # name is which and need not arrive first.
        self.joint_states: deque = deque(maxlen=8)
        self.scene: CollisionScene | None = None
        self.estimate: PayloadEstimate | None = None

        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self._description, latched()
        )
        # Depth 10 not 1: `/joint_states` gets **two** partial messages from two
        # producers back to back. Depth-1 loses one for good (second overwrites
        # first before serving) -> planner never sees actuated six. Legacy used
        # depth 5.
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
        # Transient-local: RViz started after the plan still sees it -- standing
        # decision, not stream.
        self.markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, latched())
        self.solver_stats = self.create_publisher(
            DiagnosticArray, SOLVER_STATS_TOPIC, latched()
        )
        # Retained timber contract, same node + planner: one adapter, one set of
        # limits.
        self.create_service(CalcMovement, A2B_MOVEMENT_SERVICE, self._a2b)
        self.get_logger().info(f"waiting for {ROBOT_DESCRIPTION_TOPIC}")

    # -- parameters -----------------------------------------------------------

    def _declare(self) -> None:
        """
        Declare every knob `config/crane_planner.yaml` carries.

        Machine numbers -- joint range and speed -- deliberately absent, read
        from description: a limit beside a node drifts from the one every
        controller in the deployment uses.

        Name lists are module constants, not literals, because an override for an
        undeclared name is dropped silently -- yaml reads authoritative but is
        not. `test_every_yaml_key_is_declared` compares them; only thing that
        notices.
        """
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
        # OCP cost weights, nested under `weights` as yaml writes them. Residual
        # rows dimensionless -> preferences, one scalar sets a block;
        # `weights.time` prices horizon row = minimum-time objective.
        for name, value in crane_weights.DEFAULTS.items():
            self.declare_parameter(f"weights.{name}", float(value))
        self.declare_parameter("max_input_age", 0.5)
        self.declare_parameter("max_scene_age", 10.0)
        # Test rig only: no perception -> no scene, and a request wanting
        # collision checks is refused. True -> plan against empty world;
        # self-collision still checked, geometry not.
        self.declare_parameter("allow_missing_scene", True)
        self.declare_parameter("pendulum_state_deadline", 0.15)
        self.declare_parameter("payload_estimate_deadline", 1.0)
        # C3's inversion in the effort field. A controller not setting
        # `effort_field_is_feedforward` rejects a trajectory carrying it, so
        # profiles without that JTC fork set this false.
        self.declare_parameter("c3_feedforward", True)

    def _weights(self) -> dict:
        """
        Read the cost weights once, check them once.

        `weights.matrices` called, answer discarded on purpose: width mismatch
        vs residual, or negative price, is a config error -- belongs here, not on
        the first plan.
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
            self.planner = Planner(
                message.data, self._config(), self.weights, build_missing=False
            )
        except SolverNotExported as missing:
            # Quit, not degrade: solver hand-exported against the published
            # description, so a miss means planning for another machine. Dying
            # says so while launch is still up to dump from.
            self.planner = None
            self.get_logger().fatal(str(missing))
            raise SystemExit(1) from missing
        except Exception as failure:  # bad description is not a crash
            self.planner = None
            self.get_logger().error(f"the robot description was refused: {failure}")
            return
        self.joint_names = list(canonical_joints())
        # Which machine, not just which tool: several descriptions published,
        # this node remapped onto one; sha1 names solver compiled for it. Else
        # only the launch file answers "which URDF?".
        digest = hashlib.sha1(message.data.encode()).hexdigest()
        self.get_logger().info(
            f"planning for {Tool.PZS100.value} on the description at sha1 "
            f"{digest[:10]}, solver {self.planner.ocp.tree.name}"
        )

    def _joint_states(self, message: JointState) -> None:
        """
        Keep recent messages; which is which is decided on read.

        `/joint_states` carries actuated and passive alike; telling them apart
        needs joint names from description, so routing here would drop states
        arriving before it.
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

        # Passive half, and what if absent. Missing sway read as zero is the
        # stopped-start convention this refusal removes; no third answer.
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
            # Broadcaster publishing sway without its rate leaves the OCP no
            # boundary condition; zero honest only when the array is truly absent.
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
        """Return the declared payload, overridden by a fresh valid estimate."""
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

        Nothing here plans: `a2b` maps request onto `Planner.plan`, this runs it.
        Only service here (native `/crane/plan_motion` had no caller, removed),
        so only path publishing `/crane/reference`.

        `CalcMovement.Response` has **no message field** -- refusal cannot say
        why. Goes to log; trajectory left empty, like legacy.
        """
        response.success = False
        response.trajectory = JointTrajectory()
        response.tcp_path = []
        if self.planner is None:
            refusal = f"no robot description has arrived on {ROBOT_DESCRIPTION_TOPIC}"
            self.get_logger().warn(f"a2b_movement refused: {refusal}")
            self._report(DiagnosticStatus.ERROR, refusal, {})
            return response

        try:
            # Payload first: tip-to-tool offset is read at the pose the tool
            # hangs at *with it on*.
            payload, _shape = translate_payload(request)
            start = translate_start(request)
            if start is None:
                start = self._start()
            else:
                # Supplied feasibility state, not a measurement: stamped now, no
                # measurement clock to use.
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
            # Every refusal, not only the solver's: silence on refusals with no
            # numbers reads as "solve was fine".
            self._report(DiagnosticStatus.ERROR, str(refusal), refusal.stats)
            return response

        # Canonical eight: what the trajectory controller fed this is configured
        # with. Passive pair free (OCP solved it) and is the sway the plan
        # claims, not where the tool would settle.
        legacy = self._trajectory(plan, self.start_stamp, range(GENERALIZED_DOF))
        # `publish_path` gates the topic only: answer carries `tcp_path` either
        # way, as legacy did.
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
        # Re-read live, like `c3_feedforward` below. Safe between requests where
        # rest is not: lag enters only post-solve effort-field reconstruction,
        # never exported solver, geometry or cache. Lets an ablation switch
        # feedforward laws without relaunch.
        self.planner.config.command_lag_s = np.asarray(
            self.get_parameter("command_lag_s").value, dtype=float
        )
        primitives = self._primitives(avoid_collisions)
        # Goal + geometry drawn **before** solve, so a refusal leaves them on
        # screen. "No" vs "no, and here is the runge it would have hit" are very
        # different messages.
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
        # `WARN` not `OK` when slack bought the answer: a priced violation of the
        # sway box, settled box or pump share still executes and residuals say
        # nothing. Threshold is `plan.message`'s own -- log and topic cannot
        # disagree.
        level = (
            DiagnosticStatus.WARN
            if plan.timing.slack > SLACK_SPENT
            else DiagnosticStatus.OK
        )
        self._report(level, plan.message, plan.timing.report())
        return plan, trajectory, path

    def _report(self, level: bytes, message: str, stats: dict) -> None:
        """
        Publish what the solver did, converged or not, one topic one shape.

        Non-convergence is why this exists and returns nothing else: no message
        field on `CalcMovement.Response`, so a failed solve used to leave only a
        log line. `level` is a `DiagnosticStatus` constant -- those are `bytes`,
        not ints; an int fails the field's type check.
        """
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
        """
        Build a trajectory over `indices` of the canonical eight.

        Two callers, two widths, not cosmetic. Native reference: **actuated
        six**, what `crane_velocity_controller` claims. `a2b_movement` answer:
        **canonical eight** -- its trajectory controller commands six but tracks
        passive tip and tilt too, so a six-wide goal is rejected.

        `header.frame_id` empty on purpose: joint space has no frame.
        """
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

    def _draw(self, markers: list) -> None:
        """Replace the drawing wholesale: `DELETEALL`, then this plan's set."""
        self.markers.publish(viz.clear(PLANNING_FRAME, self.get_clock().now().to_msg()))
        self.markers.publish(MarkerArray(markers=markers))

    def _path(self, plan) -> Path:
        """Build the tool geometry for the operator. Visualization only."""
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
