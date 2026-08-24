#include "crane_planning/planner_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

rclcpp::QoS robot_description_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
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
  settings_.ik.redundancy_passes = static_cast<std::size_t>(parameters.redundancy_passes);
  settings_.ik.fixed_point_iterations =
    static_cast<std::size_t>(parameters.fixed_point_iterations);
  settings_.ik.refinement_iterations =
    static_cast<std::size_t>(parameters.refinement_iterations);
  settings_.ik.geometry.planarity_m = parameters.geometry_tolerance.length;
  settings_.ik.geometry.structure_m = parameters.geometry_tolerance.length;
  settings_.ik.geometry.bearing_rad = parameters.geometry_tolerance.bearing;
  settings_.ramp.Ts = parameters.Ts;
  settings_.ramp.min_duration = parameters.min_duration;
  settings_.geometry_samples = static_cast<std::size_t>(parameters.geometry_samples);
  max_input_age_ = parameters.max_input_age;

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
  plan_motion_ = create_service<crane_msgs::srv::PlanMotion>(
    kPlanMotionService,
    [this](
      crane_msgs::srv::PlanMotion::Request::SharedPtr request,
      crane_msgs::srv::PlanMotion::Response::SharedPtr response) {
      plan(*request, *response);
    });

  RCLCPP_INFO(
    get_logger(),
    "crane_planner: %s answered with a timed joint trajectory, republished on %s reliable and "
    "transient-local. This is the slice-5 tracer: the endpoint is the semi-analytic IK of "
    "wiki/robot_model.md 2.2 with the passive pair pinned at its equilibrium, and the timing is "
    "one velocity-limited ramp. There is no collision check of any kind (issue 041), no "
    "structured lift/traverse/descend primitive (issue 040) and no path-constrained OCP, so no "
    "force, flow or kappa margin (issue 043). Each of those is refused or named rather than "
    "approximated.",
    kPlanMotionService, kReferenceTopic);
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

  auto context = build_planner(*model_, message->data, settings_);
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

bool PlannerNode::read_start(crane_model::QA & q_a_start, std::string & why) const
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
    q_a_start[static_cast<Eigen::Index>(row)] = joint_states_->position[index];
  }
  return true;
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
  if (request.goal.header.frame_id != kPlanningFrame) {
    response.message = "the goal is in frame '" + request.goal.header.frame_id +
      "', and this planner plans in '" + std::string(kPlanningFrame) +
      "'. ROS 2 Interfaces 5 makes the assembly planner the one element that converts world into "
      + kPlanningFrame + "; nothing downstream of it converts, and this node does not either";
    RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
    return;
  }
  if (!ready()) {
    response.message = std::string("no usable robot description has arrived on ") +
      kRobotDescriptionTopic + " yet, so there is no model to plan against";
    RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
    return;
  }

  crane_model::QA q_a_start;
  std::string why;
  if (!read_start(q_a_start, why)) {
    response.message = "no start configuration: " + why;
    RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
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
    response.message = "the goal orientation is a zero quaternion";
    RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
    return;
  }
  motion.phi_z_d = phi_z_of(orientation);
  motion.q_a_start = q_a_start;
  motion.margin_factor = request.speed_scale;
  motion.avoid_collisions = request.avoid_collisions;

  // The payload as the equilibrium needs it. `wiki/robot_model.md` 5.3 is
  // explicit that for a gravity moment the payload is a point mass -- shape,
  // size and inertia do not enter the equilibrium condition at all -- and the
  // equilibrium and forward kinematics are the only calls this issue makes. So
  // the inertia goes over as zero, which crane_model accepts as a point mass.
  // The first issue that evaluates a *dynamics* call for a carried payload owes
  // a real tensor here; issue 043 is that issue.
  motion.payload.valid = true;
  motion.payload.inertia_k8_kg_m2.setZero();
  if (request.payload.shape == crane_msgs::msg::Payload::SHAPE_NONE) {
    motion.payload.mass_kg = 0.0;
    motion.payload.center_of_mass_k8_m.setZero();
  } else {
    if (!std::isfinite(request.payload.mass) || request.payload.mass <= 0.0) {
      response.message = "a payload shape was declared with a mass of " +
        std::to_string(request.payload.mass) +
        " kg; an unknown payload is not a zero-mass payload, so declare SHAPE_NONE for an empty "
        "gripper instead";
      RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
      return;
    }
    motion.payload.mass_kg = request.payload.mass;
    motion.payload.center_of_mass_k8_m = Eigen::Vector3d(
      request.payload.com.x, request.payload.com.y, request.payload.com.z);
  }

  auto solved = plan_motion(*model_, *context_, motion);
  if (!solved.ok()) {
    response.message = solved.status().message;
    RCLCPP_WARN(get_logger(), "planning refused: %s", response.message.c_str());
    return;
  }
  const MotionPlan & motion_plan = solved.value();

  // ROS 2 Interfaces 1: the stamp is the absolute time the first point is valid
  // for. The first point *is* the measured start configuration, so the stamp is
  // that measurement's own stamp and not the moment this reply was built.
  const rclcpp::Time origin(joint_states_->header.stamp);
  trajectory_msgs::msg::JointTrajectory & trajectory = response.trajectory;
  trajectory.header.stamp = origin;
  trajectory.header.frame_id = "";  // joint space has no geometric frame (1)
  const std::array<std::string, crane_model::kGeneralizedDof> & names =
    model_->urdf_joint_names();
  trajectory.joint_names.reserve(crane_model::kActuatedDof);
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    trajectory.joint_names.push_back(names[kActuatedRows[row]]);
  }
  trajectory.points.reserve(motion_plan.trajectory.time_from_start.size());
  for (std::size_t index = 0; index < motion_plan.trajectory.time_from_start.size(); ++index) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(crane_model::kActuatedDof);
    point.velocities.resize(crane_model::kActuatedDof);
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      point.positions[row] = motion_plan.trajectory.q_a_ref[index][axis];
      point.velocities[row] = motion_plan.trajectory.dq_a_ref[index][axis];
    }
    point.time_from_start =
      rclcpp::Duration::from_seconds(motion_plan.trajectory.time_from_start[index]);
    trajectory.points.push_back(std::move(point));
  }

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
    std::to_string(motion_plan.endpoint.d45) + " m after " +
    std::to_string(motion_plan.endpoint.fixed_point_iterations) +
    " wrist-offset iterations and " +
    std::to_string(motion_plan.endpoint.refinement_iterations) +
    " Jacobian refinements. The peak velocity is " +
    std::to_string(motion_plan.trajectory.limiting_fraction) +
    " of the scaled limit. No collision was checked (issue 041) and the timing is the ramp of "
    "this tracer, not the OCP of trajectory_planning 5.2 (issue 043)";
  RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
}

}  // namespace crane_planning
