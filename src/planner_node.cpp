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
  collision_scene_subscription_ = create_subscription<crane_msgs::msg::CollisionScene>(
    kCollisionSceneTopic, collision_scene_qos(),
    [this](crane_msgs::msg::CollisionScene::ConstSharedPtr message) {
      on_collision_scene(message);
    });
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
    "transient-local. This is the slice-5 tracer: a goal here is a placement goal, so the "
    "endpoint is the equilibrium-constrained IK of wiki/robot_model.md 2.2 -- the passive pair "
    "is a decision variable under g_u(q) = 0 rather than a pinning, so the tool arrives at rest "
    "-- the geometry is the structured lift/traverse/descend primitive of trajectory_planning "
    "4.4, built C2 in sigma with its transfer altitude derived from the endpoints and the tool's "
    "own reach rather than hard-coded, and the timing is one velocity-limited ramp run along it. "
    "The path is checked against %s -- the scene, the truck bed and the runges of "
    "trajectory_planning 4.2 keyed to the measured truck pose, and the crane against itself -- "
    "over the sway envelope of 4.3 at the q_sway_max of mpc 3 constraint 3. There is no sampling "
    "fallback when the primitive is blocked (issue 042) and no path-constrained OCP, so no force, "
    "flow or kappa margin (issue 043). Each of those is refused or named rather than "
    "approximated.",
    kPlanMotionService, kReferenceTopic, kCollisionSceneTopic);
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
    } else if (!primitive.dimensions_m.allFinite() ||
      (primitive.dimensions_m.array() <= 0.0).any())
    {
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

    // The other half of the same message: the shape and the extent, which the
    // equilibrium has no use for and the collision check cannot do without. It
    // becomes the scene primitive with the reserved id `payload`, placed at the
    // K8 pose of every configuration the path is checked at.
    motion.payload_shape.declared = true;
    motion.payload_shape.dimensions_m = Eigen::Vector3d(
      request.payload.dimensions.x, request.payload.dimensions.y, request.payload.dimensions.z);
    motion.payload_shape.center_k8_m = motion.payload.center_of_mass_k8_m;
    if (!shape_from_message(request.payload.shape, motion.payload_shape.shape)) {
      response.message = "the payload carries shape " +
        std::to_string(static_cast<int>(request.payload.shape)) +
        ", which is no shape this planner can place in the collision scene";
      RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
      return;
    }
    if (!motion.payload_shape.dimensions_m.allFinite() ||
      (motion.payload_shape.dimensions_m.array() <= 0.0).any())
    {
      response.message = "a payload shape was declared with no positive extent along every axis; "
        "dimensions are the extent per axis, so a box carries its three side lengths and a "
        "cylinder (2r, 2r, length)";
      RCLCPP_WARN(get_logger(), "%s", response.message.c_str());
      return;
    }
  }

  // The scene, if one has arrived. Whether its absence is fatal is the core's
  // decision and depends on `avoid_collisions`, so the note travels either way.
  motion.scene = scene_.has_value() ? &scene_.value() : nullptr;

  auto solved = plan_motion(*model_, *context_, motion);
  if (!solved.ok()) {
    response.message = solved.status().message;
    if (!scene_note_.empty()) {
      // What the planner knows about the world it just refused to plan in. A
      // refusal that says "blocked" without saying which scene it was blocked
      // against leaves an operator nothing to act on.
      response.message += ". As for the scene: " + scene_note_;
    }
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
    std::to_string(motion_plan.endpoint.d45) + " m in " +
    std::to_string(motion_plan.endpoint.nlp_iterations) +
    " equilibrium-constrained iterations and " + std::to_string(motion_plan.endpoint.elapsed_s) +
    " s, leaving the passive pair " +
    std::to_string(motion_plan.endpoint.residual_equilibrium) +
    " rad off Model::passive_equilibrium, so the tool arrives at rest. The peak velocity is " +
    std::to_string(motion_plan.trajectory.limiting_fraction) +
    " of the scaled limit. The geometry is the lift/traverse/descend primitive of "
    "trajectory_planning 4.4, built C2 in sigma over " +
    std::to_string(motion_plan.primitive.path.segment_count()) + " phases: " +
    describe(motion_plan.primitive.altitude) + ". " + motion_plan.primitive.check.note +
    ". The timing is the ramp of this tracer, not the OCP of trajectory_planning 5.2 (issue 043)";
  if (!scene_note_.empty()) {
    response.message += ". As for the scene: " + scene_note_;
  }
  RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
}

}  // namespace crane_planning
