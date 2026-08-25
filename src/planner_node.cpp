#include "crane_planning/planner_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "crane_msgs/msg/collision_primitive.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_planning/crane_planner_parameters.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace
{

constexpr int kWarnPeriodMs = 5000;

/// The tool spelling a deployment names, and the model's enum for it.
bool tool_from_string(const std::string & name, crane_model::Tool & tool)
{
  if (name == "pzs100") {
    tool = crane_model::Tool::Pzs100;
    return true;
  }
  if (name == "epsilon7040") {
    tool = crane_model::Tool::Epsilon7040;
    return true;
  }
  return false;
}

/// What a deployment says to do when `/crane/pendulum_state` is not usable.
bool passive_policy_from_string(
  const std::string & name, crane_planning::PassiveEstimatePolicy & policy)
{
  if (name == "refuse") {
    policy = crane_planning::PassiveEstimatePolicy::Refuse;
    return true;
  }
  if (name == "conservative") {
    policy = crane_planning::PassiveEstimatePolicy::Conservative;
    return true;
  }
  return false;
}

/// Which end of the tool axis's range a deployment says is a closed gripper.
bool tool_end_from_string(const std::string & name, crane_planning::ToolEnd & end)
{
  if (name == "lower") {
    end = crane_planning::ToolEnd::Lower;
    return true;
  }
  if (name == "upper") {
    end = crane_planning::ToolEnd::Upper;
    return true;
  }
  return false;
}

/// The message's shape enumeration, as `crane_model`'s.
/**
 * The two lists are the same three shapes in the same order and are still
 * converted rather than cast: a shape the message adds later would then be a
 * refusal here instead of whatever the cast happened to land on.
 */
bool shape_from_message(std::uint8_t shape, crane_model::CollisionShape & out)
{
  switch (shape) {
    case crane_msgs::msg::CollisionScene::SHAPE_BOX:
      out = crane_model::CollisionShape::Box;
      return true;
    case crane_msgs::msg::CollisionScene::SHAPE_CYLINDER:
      out = crane_model::CollisionShape::Cylinder;
      return true;
    case crane_msgs::msg::CollisionScene::SHAPE_SPHERE:
      out = crane_model::CollisionShape::Sphere;
      return true;
    default:
      return false;
  }
}

/// Every axis of a primitive's own frame carries a finite, positive extent.
/**
 * `dimensions` is the extent per axis and never a half-extent or a radius
 * (`ros2_interfaces` 6), so a zero anywhere is a primitive with no thickness in
 * that direction rather than a shorthand for something else.
 */
bool has_positive_extent(const Eigen::Vector3d & dimensions)
{
  return dimensions.allFinite() && (dimensions.array() > 0.0).all();
}

/// A `geometry_msgs/Pose` as the isometry the model's scene is expressed in.
bool pose_from_message(const geometry_msgs::msg::Pose & pose, Eigen::Isometry3d & out)
{
  // ROS carries the quaternion scalar-last and Eigen scalar-first
  // (wiki/nomenclature.md 5); the reorder is this boundary's job.
  const Eigen::Quaterniond orientation(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (!std::isfinite(orientation.norm()) || !(orientation.norm() > 0.0)) {
    return false;
  }
  out = Eigen::Isometry3d::Identity();
  out.linear() = orientation.normalized().toRotationMatrix();
  out.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return out.matrix().allFinite();
}

}  // namespace

namespace crane_planning
{

rclcpp::QoS reference_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS input_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

rclcpp::QoS collision_scene_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS payload_estimate_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS robot_description_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

const char * passive_policy_name(PassiveEstimatePolicy policy) noexcept
{
  switch (policy) {
    case PassiveEstimatePolicy::Refuse: return "refuse";
    case PassiveEstimatePolicy::Conservative: return "conservative";
  }
  return "unknown";
}

PlannerNode::PlannerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("crane_planner", options)
{
  const crane_planner::ParamListener listener(this);
  const crane_planner::Params parameters = listener.get_params();

  if (!tool_from_string(parameters.tool, tool_)) {
    // `one_of<>` on the parameter already refuses anything else, so reaching
    // here means the two lists have drifted apart rather than that a deployment
    // asked for a tool that does not exist.
    throw std::runtime_error("crane_planner: unknown tool '" + parameters.tool + "'");
  }

  settings_.ik.eps_pos = parameters.eps_pos;
  settings_.ik.eps_yaw = parameters.eps_yaw;
  settings_.ik.d45_samples = static_cast<std::size_t>(parameters.d45_samples);
  settings_.ik.fixed_point_iterations =
    static_cast<std::size_t>(parameters.fixed_point_iterations);
  settings_.ik.refinement_iterations =
    static_cast<std::size_t>(parameters.refinement_iterations);
  settings_.ik.geometry.planarity_m = parameters.geometry_tolerance.length;
  settings_.ik.geometry.structure_m = parameters.geometry_tolerance.length;
  settings_.ik.geometry.bearing_rad = parameters.geometry_tolerance.bearing;
  settings_.equilibrium.eps_equilibrium = parameters.eps_equilibrium;
  settings_.equilibrium.max_iterations =
    static_cast<std::size_t>(parameters.nlp_max_iterations);
  settings_.equilibrium.max_wall_clock_s = parameters.nlp_max_wall_clock;
  // An empty list is "no floor" and one element is the floor. The bound is
  // optional in the derivation itself (structured_primitive.hpp), so it is
  // optional here too rather than being a sentinel number a reader has to know
  // about; `size_lt<>` on the parameter is what keeps the list from carrying a
  // second one.
  if (!parameters.transfer_altitude_floor.empty()) {
    settings_.primitive.altitude.floor_m = parameters.transfer_altitude_floor.front();
  }
  if (!parameters.transfer_altitude_ceiling.empty()) {
    settings_.primitive.altitude.ceiling_m = parameters.transfer_altitude_ceiling.front();
  }
  settings_.primitive.fit.rate_headroom = parameters.path_rate_headroom;
  settings_.primitive.fit.acceleration_span = parameters.path_acceleration_span;
  settings_.primitive.fit.jerk_span = parameters.path_jerk_span;

  // The fallback of trajectory_planning 4.4, which runs only when the primitive
  // above has been generated, checked and refused. The seed is a parameter and a
  // constant: a stochastic planner answering two identical requests with two
  // different paths is not something an operator or a test can reason about.
  settings_.sampling.seed = static_cast<std::uint32_t>(parameters.sampling_seed);
  settings_.sampling.time_budget_s = parameters.sampling_time_budget;
  settings_.sampling.max_validity_checks =
    static_cast<std::size_t>(parameters.sampling_max_validity_checks);
  settings_.sampling.extension_span = parameters.sampling_extension_span;
  settings_.sampling.shortcut_attempts =
    static_cast<std::size_t>(parameters.shortcut_attempts);
  settings_.sampling.unbounded_margin_rad = parameters.sampling_unbounded_margin;

  // The sway envelope of trajectory_planning 4.3, at the bound mpc 3 constraint
  // 3 imposes. `fixed_size<>` on the parameter is what keeps this pair a pair.
  settings_.primitive.collision.sway.q_sway_max =
    Eigen::Vector2d(parameters.q_sway_max.at(0), parameters.q_sway_max.at(1));
  settings_.primitive.collision.resolution_m = parameters.check_resolution;
  settings_.primitive.collision.min_resolution_m = parameters.min_check_resolution;
  settings_.primitive.collision.max_samples =
    static_cast<std::size_t>(parameters.max_check_samples);
  settings_.primitive.collision.max_sway_samples =
    static_cast<std::size_t>(parameters.max_sway_samples);
  truck_.runge_dimensions_m = Eigen::Vector3d(
    parameters.truck.runge_dimensions.at(0), parameters.truck.runge_dimensions.at(1),
    parameters.truck.runge_dimensions.at(2));
  truck_.station_offsets_m = parameters.truck.runge_stations;
  truck_.bed_thickness_m = parameters.truck.bed_thickness;
  settings_.primitive.collision.truck = truck_;

  // The second half of robot_model 2.2 step 3's redundancy score. Both routes
  // spend the leftover freedom the same way, so both carry the same weights.
  settings_.ik.redundancy.clearance = parameters.clearance_weight;
  settings_.ik.redundancy.clearance_reference_m = parameters.clearance_reference;
  settings_.equilibrium.redundancy = settings_.ik.redundancy;

  settings_.ramp.Ts = parameters.Ts;
  settings_.ramp.min_duration = parameters.min_duration;

  // trajectory_planning 5.5 and 5.2. kappa is the deployment's reservation and
  // is deliberately not reachable from a request: `speed_scale` arrives per call
  // on the service and multiplies the velocity bound only. The two hydraulic
  // numbers and their evidence live in config/hydraulic_limits.yaml.
  settings_.timing.kappa = parameters.kappa;
  settings_.timing.sample_period = parameters.Ts;
  settings_.timing.actuation.pump_flow_max = parameters.pump_flow_max;
  settings_.timing.actuation.pump_flow_planning_factor =
    parameters.pump_flow_planning_factor;
  settings_.system_pressure_pa = parameters.system_pressure_pa;
  settings_.geometry_samples = static_cast<std::size_t>(parameters.geometry_samples);
  max_input_age_ = parameters.max_input_age;

  // trajectory_planning 7, and the whole of issue 045. The rest thresholds decide
  // whether a request is a re-plan from a moving machine at all; the deadlines are
  // the freshness of the two state topics this planner closes on; the policy is
  // what control_architecture 5.3 asks for -- a defined consequence when an input
  // stops arriving -- and the budget is the latency bound, enforced here rather
  // than in whatever timeout the caller happened to set.
  settings_.start.at_rest_dq_a = parameters.at_rest.dq_a;
  settings_.start.at_rest_dq_u = parameters.at_rest.dq_u;
  settings_.latency.total_s = parameters.latency_budget;
  pendulum_deadline_ = parameters.pendulum_state_deadline;
  payload_deadline_ = parameters.payload_estimate_deadline;
  conservative_sway_rad_ = parameters.conservative_sway.q_u;
  conservative_sway_rate_ = parameters.conservative_sway.dq_u;
  if (!passive_policy_from_string(parameters.passive_estimate_policy, passive_policy_)) {
    // `one_of<>` on the parameter already refuses anything else, so reaching here
    // means the two lists have drifted apart.
    throw std::runtime_error(
            "crane_planner: unknown passive_estimate_policy '" +
            parameters.passive_estimate_policy + "'");
  }

  // The one fact about the mounted tool the description does not carry: which
  // end of the range it gives the tool axis is a closed gripper. `one_of<>` on
  // the parameter already refuses anything else, so reaching the throw means the
  // two lists have drifted apart. `tool_axis.hpp` records what the evidence for
  // the default is on each machine, and that it is measured on one of them and
  // assumed on the other.
  if (!tool_end_from_string(parameters.gripper_closed_end, settings_.tool_axis.closed_end)) {
    throw std::runtime_error(
            "crane_planner: unknown gripper_closed_end '" + parameters.gripper_closed_end + "'");
  }
  settings_.tool_axis.transmission_samples =
    static_cast<std::size_t>(parameters.gripper_transmission_samples);
  settings_.tool_axis.transmission_floor = parameters.gripper_transmission_floor;

  reference_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
    kReferenceTopic, reference_qos());
  joint_states_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
    kJointStatesTopic, input_qos(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
      joint_states_ = std::move(message);
    });
  robot_description_subscription_ = create_subscription<std_msgs::msg::String>(
    kRobotDescriptionTopic, robot_description_qos(),
    [this](std_msgs::msg::String::ConstSharedPtr message) {on_robot_description(message);});
  collision_scene_subscription_ = create_subscription<crane_msgs::msg::CollisionScene>(
    kCollisionSceneTopic, collision_scene_qos(),
    [this](crane_msgs::msg::CollisionScene::ConstSharedPtr message) {
      on_collision_scene(message);
    });
  // The passive half of 7's start state, and what is in the gripper. Both are
  // kept as they arrive and judged at the request rather than in the callback:
  // a freshness check evaluated in a subscription callback cannot fire, because
  // the case it exists for is the one where no callback runs again
  // (control_architecture 5.3).
  pendulum_state_subscription_ = create_subscription<crane_msgs::msg::PendulumState>(
    kPendulumStateTopic, input_qos(),
    [this](crane_msgs::msg::PendulumState::ConstSharedPtr message) {
      pendulum_state_ = std::move(message);
    });
  payload_estimate_subscription_ = create_subscription<crane_msgs::msg::PayloadEstimate>(
    kPayloadEstimateTopic, payload_estimate_qos(),
    [this](crane_msgs::msg::PayloadEstimate::ConstSharedPtr message) {
      payload_estimate_ = std::move(message);
    });
  plan_motion_ = create_service<crane_msgs::srv::PlanMotion>(
    kPlanMotionService,
    [this](
      crane_msgs::srv::PlanMotion::Request::SharedPtr request,
      crane_msgs::srv::PlanMotion::Response::SharedPtr response) {
      plan(*request, *response);
    });
  plan_grip_ = create_service<crane_msgs::srv::PlanGrip>(
    kPlanGripService,
    [this](
      crane_msgs::srv::PlanGrip::Request::SharedPtr request,
      crane_msgs::srv::PlanGrip::Response::SharedPtr response) {
      grip(*request, *response);
    });

  RCLCPP_INFO(
    get_logger(),
    "crane_planner: %s answered with a timed joint trajectory, republished on %s reliable and "
    "transient-local. This is the slice-5 tracer: a goal here is a placement goal, so the "
    "endpoint is the equilibrium-constrained IK of wiki/robot_model.md 2.2 -- the passive pair "
    "is a decision variable under g_u(q) = 0 rather than a pinning, so the tool arrives at rest "
    "-- the geometry is the structured lift/traverse/descend primitive of trajectory_planning "
    "4.4, built C2 in sigma with its transfer altitude derived from the endpoints and the tool's "
    "own reach rather than hard-coded. "
    "The path is checked against %s -- the scene, the truck bed and the runges of "
    "trajectory_planning 4.2 keyed to the measured truck pose, and the crane against itself -- "
    "over the sway envelope of 4.3 at the q_sway_max of mpc 3 constraint 3. A primitive that is "
    "blocked falls back to the RRT-Connect sampling planner of 4.4 over the five path coordinates "
    "of 4.1, shortcut, refitted C2 and re-checked as 4.5 makes mandatory, from a fixed seed and "
    "inside a per-call budget -- and which of the two answered is in every reply. The timing is "
    "the path-constrained OCP of trajectory_planning 5.2, solved with acados over crane_model's "
    "symbolic graph: the sway is a state, so 5.4's terminal condition makes the tool arrive "
    "hanging still, and the cylinder force and pump flow are constrained by the same expressions "
    "mpc 3 constrains them with rather than by a second copy of them. kappa = %.2f of 5.5 is held "
    "back from every physical limit for the MPC to correct with, and a caller's speed_scale "
    "cannot reach it. A solve that does not converge is a refusal carrying the solver's own "
    "status word, never a clipped trajectory.",
    kPlanMotionService, kReferenceTopic, kCollisionSceneTopic, settings_.timing.kappa);

  RCLCPP_INFO(
    get_logger(),
    "crane_planner: %s answers the four phases of crane_msgs/PlanGrip on the same six actuated "
    "joints, the same stamp semantics and the same %s republication. Descend and lift are arm "
    "motions and are planned by the very same call /crane/plan_motion is -- same endpoint, same "
    "primitive, same collision check, same sway envelope, same kappa -- with the transfer "
    "altitude's ceiling lowered to the phase's own endpoints so a descend goes across and down "
    "rather than up, across and down. Close and open drive q8 alone, on the retained cosine "
    "primitive of trajectory_planning 8 carried on the rate so it is C2 at both ends, inside the "
    "tool axis's own velocity, acceleration and pump-flow limits, and emitted on the arm's own "
    "sample period beside the five held path coordinates -- 4.1's one clock. The %s end of the "
    "range the description gives the tool axis is taken to be the closed gripper. A travel that "
    "spans a reversal of that axis's transmission ratio is refused naming the crossing, never "
    "planned through: at the crossing the cylinder moves the tool through no distance at all.",
    kPlanGripService, kReferenceTopic, tool_end_name(settings_.tool_axis.closed_end));
}

void PlannerNode::on_robot_description(std_msgs::msg::String::ConstSharedPtr message)
{
  if (model_.has_value()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), kWarnPeriodMs,
      "A second robot description arrived on %s and was ignored. A description that changes under "
      "a running planner is a different crane, and rebuilding silently would answer the next "
      "request with the geometry of the old one.",
      kRobotDescriptionTopic);
    return;
  }

  crane_model::ModelConfig config;
  config.robot_description_xml = message->data;
  config.tool = tool_;
  auto model = crane_model::Model::create(config);
  if (!model.ok()) {
    RCLCPP_ERROR(
      get_logger(),
      "The robot description on %s does not describe this machine, so nothing can be planned "
      "against it: %s",
      kRobotDescriptionTopic, model.status().message.c_str());
    return;
  }
  model_.emplace(std::move(model).value());

  auto context = build_planner(*model_, config, settings_);
  if (!context.ok()) {
    RCLCPP_ERROR(
      get_logger(),
      "The description on %s is not the planar two-link arm robot_model 2.2 poses the inverse "
      "kinematics on, or does not state its own joint limits: %s",
      kRobotDescriptionTopic, context.status().message.c_str());
    model_.reset();
    return;
  }
  context_.emplace(std::move(context).value());

  RCLCPP_INFO(
    get_logger(),
    "Model built from %s. The arm closes as a2 = %.6f m, a3 = %.6f m, d45 = %.6f + %.6f q4 m "
    "(robot_model 2.2), probed out of the description rather than written down.",
    kRobotDescriptionTopic, context_->geometry.a2, context_->geometry.a3,
    context_->geometry.d45_0, context_->geometry.d45_gain);
}

void PlannerNode::on_collision_scene(crane_msgs::msg::CollisionScene::ConstSharedPtr message)
{
  // The frame is checked and never assumed, on the same terms as the goal pose.
  // ROS 2 Interfaces 4 gives this row `K0_mounting_base` and names the world
  // model as the element that has already converted `world` into it; a scene in
  // any other frame is a scene about somewhere else, and taking it as K0 would
  // put the truck wherever the vehicle happens to be parked.
  if (message->header.frame_id != kPlanningFrame) {
    scene_note_ = std::string("the newest ") + kCollisionSceneTopic + " is in frame '" +
      message->header.frame_id + "' and this planner plans in '" + kPlanningFrame +
      "', so it was refused rather than reinterpreted";
    RCLCPP_ERROR(get_logger(), "%s", scene_note_.c_str());
    return;
  }

  crane_model::CollisionScene scene;
  std::vector<std::string> ids;
  scene.primitives.reserve(message->primitives.size());
  ids.reserve(message->primitives.size());

  for (const crane_msgs::msg::CollisionPrimitive & incoming : message->primitives) {
    std::string why;
    crane_model::CollisionPrimitive primitive;
    primitive.id = incoming.id;
    primitive.structural = incoming.structural;
    primitive.dimensions_m = Eigen::Vector3d(
      incoming.dimensions.x, incoming.dimensions.y, incoming.dimensions.z);

    if (incoming.id.empty()) {
      why = "a primitive arrived with no id, and crane_model needs one to name it in a refusal";
    } else if (incoming.id == kPayloadId) {
      why = "a primitive arrived with the reserved id '" + std::string(kPayloadId) +
        "', which crane_model reads as geometry the tool is carrying and does not check against "
        "the links that carry it. The planner places the payload; the scene does not";
    } else if (std::find(ids.begin(), ids.end(), incoming.id) != ids.end()) {
      why = "the id '" + incoming.id + "' arrived twice";
    } else if (!shape_from_message(incoming.shape, primitive.shape)) {
      why = "primitive '" + incoming.id + "' carries shape " +
        std::to_string(static_cast<int>(incoming.shape)) + ", which is none of the three";
    } else if (!pose_from_message(incoming.pose, primitive.pose_in_mounting_base)) {
      why = "primitive '" + incoming.id + "' does not carry a usable pose";
    } else if (!has_positive_extent(primitive.dimensions_m)) {
      why = "primitive '" + incoming.id +
        "' has no positive extent along every axis of its own frame; dimensions are the extent "
        "per axis, so a box carries its three side lengths and a cylinder (2r, 2r, length)";
    }
    if (!why.empty()) {
      scene_note_ = "the newest scene was refused whole rather than silently shortened: " + why;
      RCLCPP_ERROR(get_logger(), "%s", scene_note_.c_str());
      return;
    }
    ids.push_back(incoming.id);

    // The truck is not an obstacle, it is a *pose*: trajectory_planning 4.2's
    // structural geometry is keyed to it here, so that the bed and the runges
    // move when the vehicle does and nothing hard-codes where they are.
    if (incoming.id == kTruckId) {
      auto expanded = expand_truck(primitive, truck_);
      if (!expanded.ok()) {
        scene_note_ = "the truck model does not fit the truck the scene measured: " +
          expanded.status().message;
        RCLCPP_ERROR(get_logger(), "%s", scene_note_.c_str());
        return;
      }
      for (const crane_model::CollisionPrimitive & piece : expanded.value()) {
        scene.primitives.push_back(piece);
      }
      continue;
    }
    scene.primitives.push_back(std::move(primitive));
  }

  const std::size_t structural = static_cast<std::size_t>(
    std::count_if(
      scene.primitives.begin(), scene.primitives.end(),
      [](const crane_model::CollisionPrimitive & primitive) {return primitive.structural;}));
  scene_note_ = std::to_string(message->primitives.size()) + " primitives arrived on " +
    kCollisionSceneTopic + " and became " + std::to_string(scene.primitives.size()) + ", " +
    std::to_string(structural) + " of them structural";
  scene_note_ += (std::find(ids.begin(), ids.end(), kTruckId) != ids.end()) ?
    ". The truck pose was expanded into the bed and the runges of trajectory_planning 4.2, at the "
    "legacy dimensions and keyed to that pose" :
    ". No primitive carried the reserved id 'truck', so no bed and no runges were placed";
  scene_.emplace(std::move(scene));
  RCLCPP_INFO(get_logger(), "%s", scene_note_.c_str());
}

bool PlannerNode::read_start(MeasuredStart & start, std::string & why) const
{
  if (joint_states_ == nullptr) {
    why = std::string("nothing has been received on ") + kJointStatesTopic +
      ", so the configuration this plan would start from is unknown";
    return false;
  }
  const double age = (now() - rclcpp::Time(joint_states_->header.stamp)).seconds();
  if (!(age <= max_input_age_)) {
    why = std::string("the newest ") + kJointStatesTopic + " is " + std::to_string(age) +
      " s old, past max_input_age = " + std::to_string(max_input_age_) +
      " s; a plan whose first point was valid that long ago is a plan for a crane that has moved";
    return false;
  }

  const std::array<std::string, crane_model::kGeneralizedDof> & names =
    model_->urdf_joint_names();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const std::string & joint = names[kActuatedRows[row]];
    const auto found =
      std::find(joint_states_->name.begin(), joint_states_->name.end(), joint);
    if (found == joint_states_->name.end()) {
      why = joint + " is not in " + kJointStatesTopic;
      return false;
    }
    const std::size_t index =
      static_cast<std::size_t>(std::distance(joint_states_->name.begin(), found));
    if (index >= joint_states_->position.size() ||
      !std::isfinite(joint_states_->position[index]))
    {
      why = joint + " carries no finite position in " + kJointStatesTopic;
      return false;
    }
    start.q_a[static_cast<Eigen::Index>(row)] = joint_states_->position[index];

    // The rate, on the same terms, and its absence is a refusal rather than a
    // zero: trajectory_planning 7 asks for `(q, dq)` **as measured**, and reading
    // a message with no velocities as a machine standing still is the
    // stopped-start convention the page removes -- the one that makes a re-plan
    // issued while the tool is still moving mis-predict the sway from the first
    // step.
    if (index >= joint_states_->velocity.size() ||
      !std::isfinite(joint_states_->velocity[index]))
    {
      why = joint + " carries no finite velocity in " + kJointStatesTopic +
        ". trajectory_planning 7 asks for (q, dq) as measured, and a measurement with no rate in "
        "it does not say whether the machine is moving; taking that as a standstill is the "
        "stopped-start convention 7 exists to remove, so it is refused instead. "
        "joint_state_broadcaster publishes the velocity state interface";
      return false;
    }
    start.dq_a[static_cast<Eigen::Index>(row)] = joint_states_->velocity[index];
  }
  return true;
}

bool PlannerNode::read_passive(PassiveStart & passive, std::string & why) const
{
  passive = PassiveStart{};

  // The three absences control_architecture 5.3 asks to be told apart. "Never
  // connected" and "died" are different things for an operator to chase, and a
  // producer that says `valid == false` is a third thing again -- it is running,
  // it is talking, and it is telling the truth about itself.
  std::string absence;
  if (pendulum_state_ == nullptr) {
    absence = std::string("nothing has ever been received on ") + kPendulumStateTopic +
      " -- this is 'never connected' and not 'died': no producer has been seen at all";
  } else {
    const double age = (now() - rclcpp::Time(pendulum_state_->header.stamp)).seconds();
    if (!(age <= pendulum_deadline_)) {
      absence = std::string("the newest ") + kPendulumStateTopic + " is " + std::to_string(age) +
        " s old against its own " + std::to_string(pendulum_deadline_) +
        " s deadline (control_architecture 5.3) -- this is 'died' and not 'never connected': the "
        "estimate arrived and then stopped";
    } else if (!pendulum_state_->valid) {
      absence = std::string("the newest ") + kPendulumStateTopic +
        " is fresh and reports valid == false, which is the producer's own verdict on itself and "
        "is carried rather than restated: \"" + pendulum_state_->status + "\"";
    } else if (!std::isfinite(pendulum_state_->position[0]) ||
      !std::isfinite(pendulum_state_->position[1]) ||
      !std::isfinite(pendulum_state_->velocity[0]) ||
      !std::isfinite(pendulum_state_->velocity[1]))
    {
      absence = std::string("the newest ") + kPendulumStateTopic +
        " is fresh and valid but does not carry four finite numbers";
    }
  }

  if (absence.empty()) {
    passive.measured = true;
    passive.q_u = crane_model::QU(
      pendulum_state_->position[0], pendulum_state_->position[1]);
    passive.dq_u = crane_model::DQU(
      pendulum_state_->velocity[0], pendulum_state_->velocity[1]);
    passive.note = std::string("the passive pair came off ") + kPendulumStateTopic +
      ", valid, inside its " + std::to_string(pendulum_deadline_) + " s deadline: \"" +
      pendulum_state_->status + "\"";
    return true;
  }

  if (passive_policy_ == PassiveEstimatePolicy::Refuse) {
    why = absence +
      ". This deployment's passive_estimate_policy is 'refuse', so the request is refused rather "
      "than planned from an assumed still tool -- trajectory_planning 7 is explicit that a "
      "re-plan whose sway is assumed mis-predicts the first step, and control_architecture 5.3 "
      "asks an absent input to end in a defined consequence rather than in a silent zero";
    return false;
  }

  passive.measured = false;
  passive.sway_reserve_rad = conservative_sway_rad_;
  passive.sway_rate_reserve = conservative_sway_rate_;
  passive.note = absence +
    ". This deployment's passive_estimate_policy is 'conservative', so the plan is built from the "
    "hanging pose of the measured actuated configuration with " +
    std::to_string(conservative_sway_rad_) + " rad and " +
    std::to_string(conservative_sway_rate_) +
    " rad/s of the sway allowance held back for the sway that may be there anyway. That is a "
    "stated bound and not a measurement: it says the plan leaves room for sway it cannot see, not "
    "that there is none";
  return true;
}

void PlannerNode::read_payload_estimate(crane_model::Payload & payload, std::string & note) const
{
  if (payload_estimate_ == nullptr) {
    note = std::string("nothing has ever been received on ") + kPayloadEstimateTopic +
      " -- 'never connected' -- so the payload is the one the request declared and no estimate "
      "entered this plan";
    return;
  }
  const double age = (now() - rclcpp::Time(payload_estimate_->header.stamp)).seconds();
  if (!(age <= payload_deadline_)) {
    note = std::string("the newest ") + kPayloadEstimateTopic + " is " + std::to_string(age) +
      " s old against a " + std::to_string(payload_deadline_) +
      " s deadline -- 'died' -- so the payload is the one the request declared";
    return;
  }
  if (!payload_estimate_->valid) {
    // The branch that actually runs: both CBS profiles publish `valid == false`
    // today, so this is the tested path and not the exceptional one. `valid` on
    // this message means "estimated at rest", and an estimate taken while the
    // machine was moving is not one this planner may substitute for a caller's
    // own declaration.
    note = std::string("the newest ") + kPayloadEstimateTopic +
      " is fresh and reports valid == false -- the estimator says it has no estimate taken at "
      "rest -- so the payload is the one the request declared and not this message's numbers";
    return;
  }
  if (!std::isfinite(payload_estimate_->mass) || payload_estimate_->mass < 0.0 ||
    !std::isfinite(payload_estimate_->m_r_x) || !std::isfinite(payload_estimate_->m_r_y) ||
    !std::isfinite(payload_estimate_->r_z))
  {
    note = std::string("the newest ") + kPayloadEstimateTopic +
      " is fresh and valid but does not carry finite numbers, so the payload is the one the "
      "request declared";
    return;
  }

  // `wiki/robot_model.md` 5.3: for a gravity moment the payload is a point mass,
  // so the estimate's first moment divided by its mass *is* the centre of mass in
  // K8 and nothing else about the shape enters. A zero mass carries no first
  // moment either, and the centre is then meaningless rather than at the origin.
  payload.valid = true;
  payload.mass_kg = payload_estimate_->mass;
  payload.inertia_k8_kg_m2.setZero();
  if (payload_estimate_->mass > 0.0) {
    payload.center_of_mass_k8_m = Eigen::Vector3d(
      payload_estimate_->m_r_x / payload_estimate_->mass,
      payload_estimate_->m_r_y / payload_estimate_->mass,
      payload_estimate_->r_z);
  } else {
    payload.center_of_mass_k8_m.setZero();
  }
  note = std::string("the payload is ") + kPayloadEstimateTopic + "'s, valid and " +
    std::to_string(age) + " s old: " + std::to_string(payload.mass_kg) + " kg at (" +
    std::to_string(payload.center_of_mass_k8_m.x()) + ", " +
    std::to_string(payload.center_of_mass_k8_m.y()) + ", " +
    std::to_string(payload.center_of_mass_k8_m.z()) +
    ") m in K8. It replaces what the request declared, because the estimator measured the machine "
    "and the request described it";
}

std::string PlannerNode::describe_standing() const
{
  if (!standing_.has_value()) {
    return std::string("Nothing has been published on ") + kReferenceTopic +
      " by this planner yet, so no trajectory is standing and this refusal displaced nothing";
  }
  const double duration = standing_->points.empty() ?
    0.0 : rclcpp::Duration(standing_->points.back().time_from_start).seconds();
  return std::string("The trajectory standing on ") + kReferenceTopic +
    " is unchanged and is the one still being executed: " +
    std::to_string(standing_->points.size()) + " points over " + std::to_string(duration) +
    " s, stamped " + std::to_string(rclcpp::Time(standing_->header.stamp).seconds()) +
    ". It is carried on this response so a caller can tell 'I kept planning' from 'here is "
    "something new'; success is false, so it is not something new";
}

bool PlannerNode::read_payload(
  const crane_msgs::msg::Payload & message, crane_model::Payload & payload, PayloadShape & shape,
  std::string & why)
{
  // The payload as the equilibrium needs it. `wiki/robot_model.md` 5.3 is
  // explicit that for a gravity moment the payload is a point mass -- shape,
  // size and inertia do not enter the equilibrium condition at all -- so the
  // inertia goes over as zero, which crane_model accepts as a point mass.
  payload = crane_model::Payload{};
  shape = PayloadShape{};
  payload.valid = true;
  payload.inertia_k8_kg_m2.setZero();
  if (message.shape == crane_msgs::msg::Payload::SHAPE_NONE) {
    payload.mass_kg = 0.0;
    payload.center_of_mass_k8_m.setZero();
    return true;
  }
  if (!std::isfinite(message.mass) || message.mass <= 0.0) {
    why = "a payload shape was declared with a mass of " + std::to_string(message.mass) +
      " kg; an unknown payload is not a zero-mass payload, so declare SHAPE_NONE for an empty "
      "gripper instead";
    return false;
  }
  payload.mass_kg = message.mass;
  payload.center_of_mass_k8_m = Eigen::Vector3d(message.com.x, message.com.y, message.com.z);

  // The other half of the same message: the shape and the extent, which the
  // equilibrium has no use for and the collision check cannot do without. It
  // becomes the scene primitive with the reserved id `payload`, placed at the
  // K8 pose of every configuration the path is checked at.
  shape.declared = true;
  shape.dimensions_m =
    Eigen::Vector3d(message.dimensions.x, message.dimensions.y, message.dimensions.z);
  shape.center_k8_m = payload.center_of_mass_k8_m;
  if (!shape_from_message(message.shape, shape.shape)) {
    why = "the payload carries shape " + std::to_string(static_cast<int>(message.shape)) +
      ", which is no shape this planner can place in the collision scene";
    return false;
  }
  if (!has_positive_extent(shape.dimensions_m)) {
    why = "a payload shape was declared with no positive extent along every axis; dimensions are "
      "the extent per axis, so a box carries its three side lengths and a cylinder (2r, 2r, "
      "length)";
    return false;
  }
  return true;
}

trajectory_msgs::msg::JointTrajectory PlannerNode::as_message(
  const TimedTrajectory & trajectory, const rclcpp::Time & origin) const
{
  trajectory_msgs::msg::JointTrajectory message;
  // ROS 2 Interfaces 1: the stamp is the absolute time the first point is valid
  // for. The first point *is* the measured start configuration, so the stamp is
  // that measurement's own stamp and not the moment this reply was built.
  message.header.stamp = origin;
  message.header.frame_id = "";  // joint space has no geometric frame (1)
  const std::array<std::string, crane_model::kGeneralizedDof> & names =
    model_->urdf_joint_names();
  message.joint_names.reserve(crane_model::kActuatedDof);
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    message.joint_names.push_back(names[kActuatedRows[row]]);
  }
  message.points.reserve(trajectory.time_from_start.size());
  for (std::size_t index = 0; index < trajectory.time_from_start.size(); ++index) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(crane_model::kActuatedDof);
    point.velocities.resize(crane_model::kActuatedDof);
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      point.positions[row] = trajectory.q_a_ref[index][axis];
      point.velocities[row] = trajectory.dq_a_ref[index][axis];
    }
    point.time_from_start = rclcpp::Duration::from_seconds(trajectory.time_from_start[index]);
    message.points.push_back(std::move(point));
  }
  return message;
}

void PlannerNode::plan(
  const crane_msgs::srv::PlanMotion::Request & request,
  crane_msgs::srv::PlanMotion::Response & response)
{
  response.success = false;
  response.trajectory = trajectory_msgs::msg::JointTrajectory{};
  response.tcp_path.clear();

  // The frame is checked and never assumed. ROS 2 Interfaces 1 and 4 put
  // planning geometry in K0_mounting_base and 5 names the assembly planner as
  // the one element that converts `world` into it. Nothing downstream of that
  // call converts, so a goal in any other frame -- `world` included -- is a goal
  // this planner cannot place, and silently treating it as K0 would put the tool
  // wherever the truck happens to be parked.
  // Every refusal below leaves `/crane/reference` alone and says what is still
  // standing on it. The topic is transient-local, so a refused re-plan that
  // republished anything would leave a late subscriber latching onto a plan
  // nobody is executing -- and a caller has to be able to tell "I kept planning"
  // from "here is something new" (trajectory_planning 7's fallback).
  const auto refuse = [this, &response](std::string message) {
      response.success = false;
      response.trajectory = standing_.value_or(trajectory_msgs::msg::JointTrajectory{});
      response.tcp_path.clear();
      response.message = std::move(message) + ". " + describe_standing();
      RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
    };

  if (request.goal.header.frame_id != kPlanningFrame) {
    refuse(
      "the goal is in frame '" + request.goal.header.frame_id +
      "', and this planner plans in '" + std::string(kPlanningFrame) +
      "'. ROS 2 Interfaces 5 makes the assembly planner the one element that converts world into "
      + kPlanningFrame + "; nothing downstream of it converts, and this node does not either");
    return;
  }
  if (!ready()) {
    refuse(
      std::string("no usable robot description has arrived on ") + kRobotDescriptionTopic +
      " yet, so there is no model to plan against");
    return;
  }

  MeasuredStart start;
  std::string why;
  if (!read_start(start, why)) {
    refuse("no start state: " + why);
    return;
  }
  // The passive half, and this deployment's answer to its absence. Refused here
  // rather than deeper down, because whether an unusable estimate is a refusal at
  // all is a *deployment's* decision and not the algorithm's.
  if (!read_passive(start.passive, why)) {
    refuse("no passive start state: " + why);
    return;
  }

  MotionRequest motion;
  motion.p_tcp_0 = Eigen::Vector3d(
    request.goal.pose.position.x, request.goal.pose.position.y, request.goal.pose.position.z);
  // ROS messages carry the quaternion scalar-last, Eigen scalar-first
  // (wiki/nomenclature.md 5); the reorder is this boundary's job.
  const Eigen::Quaterniond orientation(
    request.goal.pose.orientation.w, request.goal.pose.orientation.x,
    request.goal.pose.orientation.y, request.goal.pose.orientation.z);
  if (!(orientation.norm() > 0.0)) {
    refuse("the goal orientation is a zero quaternion");
    return;
  }
  motion.phi_z_d = phi_z_of(orientation);
  motion.start = start;
  motion.speed_scale = request.speed_scale;
  motion.avoid_collisions = request.avoid_collisions;

  // The two halves of `crane_msgs/Payload`, read the one way both services read
  // it. The first issue that evaluates a *dynamics* call for a carried payload
  // owes a real inertia tensor there; issue 043 is that issue.
  if (!read_payload(request.payload, motion.payload, motion.payload_shape, why)) {
    refuse(why);
    return;
  }
  // ...and then `/crane/payload_estimate` where it is valid, on the same terms
  // the pendulum's estimate is read: stated, never assumed.
  std::string payload_note;
  read_payload_estimate(motion.payload, payload_note);

  // The scene, if one has arrived. Whether its absence is fatal is the core's
  // decision and depends on `avoid_collisions`, so the note travels either way.
  motion.scene = scene_.has_value() ? &scene_.value() : nullptr;

  LatencyLedger ledger(settings_.latency);
  auto solved = plan_motion(*model_, *context_, motion, &ledger);
  if (!solved.ok()) {
    std::string message = solved.status().message;
    if (!ledger.overrun().empty()) {
      // 7's fallback, in as many words: the budget fired, no partial plan was
      // built, and the previous trajectory is what is still standing.
      message += ". This is the latency bound of trajectory_planning 7 firing and not the machine "
        "refusing: nothing about the goal has been shown to be wrong";
    }
    if (!payload_note.empty()) {
      message += ". As for the payload: " + payload_note;
    }
    if (!scene_note_.empty()) {
      // What the planner knows about the world it just refused to plan in. A
      // refusal that says "blocked" without saying which scene it was blocked
      // against leaves an operator nothing to act on.
      message += ". As for the scene: " + scene_note_;
    }
    refuse(std::move(message));
    return;
  }
  const MotionPlan & motion_plan = solved.value();

  const rclcpp::Time origin(joint_states_->header.stamp);
  response.trajectory = as_message(motion_plan.trajectory, origin);
  const trajectory_msgs::msg::JointTrajectory & trajectory = response.trajectory;

  response.tcp_path.reserve(motion_plan.tcp_path.size());
  for (std::size_t index = 0; index < motion_plan.tcp_path.size(); ++index) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = kPlanningFrame;
    pose.header.stamp =
      origin + rclcpp::Duration::from_seconds(motion_plan.trajectory.time_from_start[index]);
    pose.pose.position.x = motion_plan.tcp_path[index].position_m.x();
    pose.pose.position.y = motion_plan.tcp_path[index].position_m.y();
    pose.pose.position.z = motion_plan.tcp_path[index].position_m.z();
    pose.pose.orientation.w = motion_plan.tcp_path[index].orientation.w();
    pose.pose.orientation.x = motion_plan.tcp_path[index].orientation.x();
    pose.pose.orientation.y = motion_plan.tcp_path[index].orientation.y();
    pose.pose.orientation.z = motion_plan.tcp_path[index].orientation.z();
    response.tcp_path.push_back(std::move(pose));
  }

  // The one place `/crane/reference` is written, and it is reached only by a plan
  // that was adopted: a refusal returns above without touching it.
  standing_ = trajectory;
  reference_->publish(trajectory);

  response.success = true;
  // The residuals travel with the answer. wiki/implementation/style_guide.md 4
  // asks a solver to expose them rather than only its output, and here they are
  // also the evidence that the acceptance test of robot_model 2.2 was run.
  response.message = "planned in " + std::to_string(motion_plan.trajectory.duration) +
    " s over " + std::to_string(trajectory.points.size()) +
    " points; the endpoint IK closed to " + std::to_string(motion_plan.endpoint.residual_p) +
    " m and " + std::to_string(motion_plan.endpoint.residual_phi_z) +
    " rad against forward kinematics, resolving the telescope at d45 = " +
    std::to_string(motion_plan.endpoint.d45) + " m in " +
    std::to_string(motion_plan.endpoint.nlp_iterations) +
    " equilibrium-constrained iterations and " + std::to_string(motion_plan.endpoint.elapsed_s) +
    " s, leaving the passive pair " +
    std::to_string(motion_plan.endpoint.residual_equilibrium) +
    " rad off Model::passive_equilibrium, so the tool arrives at rest. The peak velocity is " +
    std::to_string(motion_plan.trajectory.limiting_fraction) +
    " of the scaled limit. The geometry came from " + mechanism_name(motion_plan.mechanism) +
    ", over " + std::to_string(motion_plan.path.segment_count()) + " segments in sigma. ";
  // Which mechanism answered, in as many words, because a caller cannot
  // otherwise tell a deterministic plan from a sampled one -- and 4.4's whole
  // ordering argument is that the two are not the same product.
  response.message += (motion_plan.mechanism == PathMechanism::SamplingFallback) ?
    describe(motion_plan.sampled) :
    describe(motion_plan.primitive.altitude) + ". " + motion_plan.primitive.check.note;
  // What the timing was, and what it cost the machine. `PeakDemand` is a fraction
  // of the **physical** limit, so a caller reads the margin directly rather than
  // taking on trust that kappa was applied: every number below sits at or under
  // kappa, and the gap between it and one is what the MPC has left to correct
  // with (trajectory_planning 5.5).
  const TimingSolution & timing = motion_plan.timing;
  response.message += ". The timing is the path-constrained OCP of trajectory_planning 5.2, "
    "solved with acados over crane_model's symbolic graph in " +
    std::to_string(timing.solve_time) + " s and " + std::to_string(timing.iterations) +
    " SQP iterations (" + timing.solver_status +
    "): the sway is a state, so 5.4's terminal condition holds and the tool arrives hanging "
    "still. Of the physical limits the peak demand is " +
    std::to_string(timing.peak_demand.joint_velocity) + " of joint velocity, " +
    std::to_string(timing.peak_demand.joint_acceleration) + " of joint acceleration, " +
    std::to_string(timing.peak_demand.cylinder_force) + " of cylinder force and " +
    std::to_string(timing.peak_demand.pump_flow) +
    " of pump flow -- the force and flow expressions being the ones mpc 3 constrains with, "
    "read off the same output map rather than restated. kappa = " +
    std::to_string(timing.kappa) + " of 5.5 bounds all four and is the deployment's "
    "reservation; speed_scale = " + std::to_string(timing.speed_scale) +
    " is this caller's own request and scales the velocity bound alone, so no value of it "
    "reaches into the margin";
  // What this plan started from, and what it cost. Both belong on the wire:
  // `crane_msgs/PlanMotion` is frozen, so `message` is the only surface a caller
  // has, and "planned from a measurement" against "planned from the hanging pose"
  // is the difference trajectory_planning 7 is about.
  response.message += ". " + motion_plan.start_note + describe(ledger) +
    ", and this trajectory is now the one standing on " + kReferenceTopic;
  if (!payload_note.empty()) {
    response.message += ". As for the payload: " + payload_note;
  }
  if (!scene_note_.empty()) {
    response.message += ". As for the scene: " + scene_note_;
  }
  RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
}

void PlannerNode::grip(
  const crane_msgs::srv::PlanGrip::Request & request,
  crane_msgs::srv::PlanGrip::Response & response)
{
  response.success = false;
  response.trajectory = trajectory_msgs::msg::JointTrajectory{};

  // The same fallback `/crane/plan_motion` uses, for the same reason: a refused
  // phase leaves the standing reference alone and says which one it is.
  const auto refuse = [this, &response](std::string message) {
      response.success = false;
      response.trajectory = standing_.value_or(trajectory_msgs::msg::JointTrajectory{});
      response.message = std::move(message) + ". " + describe_standing();
      RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
    };

  GripRequest grip_request;
  if (!grip_phase_from_message(request.phase, grip_request.phase)) {
    refuse(
      "phase " + std::to_string(static_cast<int>(request.phase)) +
      " is none of the four crane_msgs/PlanGrip defines -- PHASE_DESCEND, PHASE_CLOSE, "
      "PHASE_OPEN, PHASE_LIFT. The .srv is frozen (PRD 15) and a fifth phase is a slice of its "
      "own, so this is refused rather than mapped onto whichever of the four is nearest");
    return;
  }
  const bool arm_phase = phase_moves_the_arm(grip_request.phase);

  // The frame, on exactly the terms `/crane/plan_motion` checks it -- and only
  // on the phases that read a pose. A close or an open moves the tool
  // coordinate to the end of the range the description gives it and reads no
  // goal at all, so demanding a frame of a field the phase does not use would
  // refuse a well-formed request for a field it was right to leave empty.
  if (arm_phase && request.goal.header.frame_id != kPlanningFrame) {
    refuse(
      "the goal of this " + std::string(grip_phase_name(grip_request.phase)) +
      " phase is in frame '" + request.goal.header.frame_id + "', and this planner plans in '" +
      std::string(kPlanningFrame) +
      "'. ROS 2 Interfaces 5 makes the assembly planner the one element that converts world into "
      + kPlanningFrame + "; nothing downstream of it converts, and this node does not either");
    return;
  }
  if (!ready()) {
    refuse(
      std::string("no usable robot description has arrived on ") + kRobotDescriptionTopic +
      " yet, so there is no model to plan against");
    return;
  }

  MeasuredStart start;
  std::string why;
  if (!read_start(start, why)) {
    refuse("no start state: " + why);
    return;
  }
  if (!read_passive(start.passive, why)) {
    refuse("no passive start state: " + why);
    return;
  }
  grip_request.start = start;
  grip_request.speed_scale = request.speed_scale;

  if (arm_phase) {
    grip_request.p_tcp_0 = Eigen::Vector3d(
      request.goal.pose.position.x, request.goal.pose.position.y, request.goal.pose.position.z);
    // ROS carries the quaternion scalar-last and Eigen scalar-first
    // (wiki/nomenclature.md 5); the reorder is this boundary's job.
    const Eigen::Quaterniond orientation(
      request.goal.pose.orientation.w, request.goal.pose.orientation.x,
      request.goal.pose.orientation.y, request.goal.pose.orientation.z);
    if (!(orientation.norm() > 0.0)) {
      refuse("the goal orientation is a zero quaternion");
      return;
    }
    grip_request.phi_z_d = phi_z_of(orientation);
    // `crane_msgs/PlanGrip` has no `avoid_collisions` row, and the frozen .srv
    // is not amended for one (PRD 15). A grip's arm phase is the least
    // forgiving move the machine makes -- it puts a tool between the runges --
    // so the absent field is read as the checked plan and never as the blind
    // one, and a request with no scene behind it is refused by `plan_motion`
    // naming the topic.
    grip_request.avoid_collisions = true;
    grip_request.scene = scene_.has_value() ? &scene_.value() : nullptr;
  }

  if (!read_payload(request.payload, grip_request.payload, grip_request.payload_shape, why)) {
    refuse(why);
    return;
  }
  std::string payload_note;
  read_payload_estimate(grip_request.payload, payload_note);

  LatencyLedger ledger(settings_.latency);
  auto solved = plan_grip(*model_, *context_, grip_request, &ledger);
  if (!solved.ok()) {
    std::string message = solved.status().message;
    if (!ledger.overrun().empty()) {
      message += ". This is the latency bound of trajectory_planning 7 firing and not the machine "
        "refusing: nothing about the goal has been shown to be wrong";
    }
    if (!payload_note.empty()) {
      message += ". As for the payload: " + payload_note;
    }
    if (arm_phase && !scene_note_.empty()) {
      message += ". As for the scene: " + scene_note_;
    }
    refuse(std::move(message));
    return;
  }
  const GripPlan & grip_plan = solved.value();

  const rclcpp::Time origin(joint_states_->header.stamp);
  response.trajectory = as_message(grip_plan.trajectory, origin);

  // The reference the planner answered with, on the row ROS 2 Interfaces 4 gives
  // it. A grip phase belongs on it for the reason trajectory_planning 4.1 gives:
  // the MPC tracks all six actuated coordinates, so q8_ref has to have a
  // producer while a grip is running, and the producer is this. Reached only by a
  // phase that was adopted -- every refusal above returned without touching it.
  standing_ = response.trajectory;
  reference_->publish(response.trajectory);

  response.success = true;
  response.message = describe(grip_plan);
  if (grip_plan.arm_phase) {
    const TimingSolution & timing = grip_plan.motion.timing;
    response.message += ". The endpoint IK closed to " +
      std::to_string(grip_plan.motion.endpoint.residual_p) + " m and " +
      std::to_string(grip_plan.motion.endpoint.residual_phi_z) +
      " rad against forward kinematics, leaving the passive pair " +
      std::to_string(grip_plan.motion.endpoint.residual_equilibrium) +
      " rad off Model::passive_equilibrium -- so this phase ends at a genuine steady state of the "
      "passive subsystem and not at a configuration with the passive joints pinned where they "
      "happened to be measured. Of the physical limits the peak demand is " +
      std::to_string(timing.peak_demand.joint_velocity) + " of joint velocity, " +
      std::to_string(timing.peak_demand.joint_acceleration) + " of joint acceleration, " +
      std::to_string(timing.peak_demand.cylinder_force) + " of cylinder force and " +
      std::to_string(timing.peak_demand.pump_flow) + " of pump flow, under kappa = " +
      std::to_string(timing.kappa) + " with speed_scale = " + std::to_string(timing.speed_scale);
    response.message += ". " + grip_plan.motion.start_note;
    if (!scene_note_.empty()) {
      response.message += ". As for the scene: " + scene_note_;
    }
  }
  response.message += ". " + describe(ledger) + ", and this trajectory is now the one standing on "
    + kReferenceTopic;
  if (!payload_note.empty()) {
    response.message += ". As for the payload: " + payload_note;
  }
  RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
}

}  // namespace crane_planning
