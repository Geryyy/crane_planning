#include "crane_planning/a2b_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/planner_node.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace
{

using timber_crane_planning_interfaces::srv::CalcMovement;

/// Two `wood_log_msgs/LogShape` fields are the same number.
/**
 * `float32` on the wire, so the two are compared at single precision: a caller
 * that copied one shape into the other -- which is what the timber behaviour
 * tree's `on_tick` does -- gets bit-identical fields, and a caller that typed
 * both into a panel gets whatever the two strings parsed to.
 */
bool same_length(float left, float right)
{
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return false;
  }
  const float scale = std::max({1.0F, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= 1.0e-6F * scale;
}

/// The three components of a `geometry_msgs/Point` are finite.
bool finite(const geometry_msgs::msg::Point & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

/// The `.srv` default of `q0` and `q0_dot`: every element exactly zero.
/**
 * The legacy server reads that default as "use the measurement" with this very
 * predicate, so the two agree on where the boundary is rather than on a
 * tolerance one of them invented.
 */
template<std::size_t N>
bool all_zero(const std::array<double, N> & values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::abs(value) < std::numeric_limits<double>::epsilon();});
}

}  // namespace

namespace crane_planning
{

bool translate_a2b_start(
  const CalcMovement::Request & request, std::optional<MeasuredStart> & start, std::string & why)
{
  start.reset();
  if (all_zero(request.q0)) {
    if (!all_zero(request.q0_dot)) {
      why = "`q0_dot` is non-zero while `q0` is the all-zero .srv default. That default means "
        "use the measured start, and combining rates supplied for one state with positions "
        "measured from another would not describe one initial condition";
      return false;
    }
    return true;
  }

  if (!std::all_of(
      request.q0.begin(), request.q0.end(), [](double value) {
        return std::isfinite(value);
      }) ||
    !std::all_of(
      request.q0_dot.begin(), request.q0_dot.end(), [](double value) {
        return std::isfinite(value);
      }))
  {
    why = "`q0`/`q0_dot` do not contain sixteen finite numbers, so they are not an initial "
      "condition the native planner can use";
    return false;
  }

  MeasuredStart translated;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const std::size_t canonical = kActuatedRows[row];
    translated.q_a[static_cast<Eigen::Index>(row)] = request.q0[canonical];
    translated.dq_a[static_cast<Eigen::Index>(row)] = request.q0_dot[canonical];
  }
  translated.passive.measured = true;
  translated.passive.q_u = crane_model::QU(request.q0[4], request.q0[5]);
  translated.passive.dq_u = crane_model::DQU(request.q0_dot[4], request.q0_dot[5]);
  translated.passive.note =
    "the start is the retained CalcMovement request's explicit `q0`/`q0_dot`, mapped from its "
    "canonical eight rows; it is a supplied feasibility state, not a measurement from the live "
    "crane";
  start = std::move(translated);
  return true;
}

bool tip_to_tcp_offset(
  const crane_model::Model & model, const crane_model::Payload & payload, double phi_z, double q8,
  Eigen::Vector3d & offset_m, std::string & why)
{
  if (!std::isfinite(phi_z) || !std::isfinite(q8)) {
    why = "`phi_tool_n` or the measured tool coordinate q8 is not finite, so the hanging "
      "tip-to-tool offset cannot be evaluated";
    return false;
  }

  // First read the tool description's yaw convention at the canonical pose.
  // K_tcp has a fixed quarter-turn in today's descriptions, so q1 itself is not
  // phi_z; measuring the convention is the same rule as measuring the offset.
  crane_model::QA q_a = crane_model::QA::Zero();
  q_a[5] = q8;
  auto canonical_hanging = model.passive_equilibrium(q_a, payload);
  if (!canonical_hanging.ok()) {
    why = "the passive subsystem of this description has no hanging equilibrium for the "
      "request's payload, so the offset from the tip pivot to the tool cannot be read out: " +
      canonical_hanging.status().message;
    return false;
  }
  crane_model::Q canonical_q = crane_model::Q::Zero();
  canonical_q[7] = q8;
  canonical_q.segment<2>(4) = canonical_hanging.value();
  auto canonical_tcp =
    model.forward_kinematics(
    canonical_q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  if (!canonical_tcp.ok()) {
    why = "this description does not carry the tool centre point, so a CalcMovement goal "
      "cannot be placed against it: " + canonical_tcp.status().message;
    return false;
  }

  // The passive pair is a universal pendulum: once it is settled, changing the
  // boom or telescope pose does not change this world-expressed vector. q1
  // rotates the whole assembly about gravity, while q7 is held at zero. Rotate
  // q1 by the difference between the requested yaw and the measured canonical
  // yaw, then settle again so even that symmetry is checked through the model.
  const double canonical_phi_z = phi_z_of(canonical_tcp.value().orientation);
  q_a[0] = std::remainder(phi_z - canonical_phi_z, 2.0 * M_PI);
  auto hanging = model.passive_equilibrium(q_a, payload);
  if (!hanging.ok()) {
    why = "the passive subsystem of this description has no hanging equilibrium for the "
      "request's payload, so the offset from the tip pivot to the tool cannot be read out: " +
      hanging.status().message;
    return false;
  }

  crane_model::Q q = crane_model::Q::Zero();
  q[0] = q_a[0];
  q[7] = q8;
  q.segment<2>(4) = hanging.value();
  auto tip = model.forward_kinematics(q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
  auto tcp = model.forward_kinematics(q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  if (!tip.ok() || !tcp.ok()) {
    why = std::string(
      "this description does not carry both the tip pivot and the tool centre "
      "point, so a CalcMovement goal cannot be placed against it: ") +
      (tip.ok() ? tcp.status().message : tip.status().message);
    return false;
  }

  const double actual_phi_z = phi_z_of(tcp.value().orientation);
  const double yaw_error = std::remainder(actual_phi_z - phi_z, 2.0 * M_PI);
  if (!std::isfinite(actual_phi_z) || std::abs(yaw_error) > 1.0e-9) {
    why = "the canonical settled pose used to translate `y_n` has TCP yaw " +
      std::to_string(actual_phi_z) + " rad instead of `phi_tool_n` = " +
      std::to_string(phi_z) +
      " rad, so its tip-to-tool offset cannot be applied without approximation";
    return false;
  }
  offset_m = tcp.value().position_m - tip.value().position_m;
  if (!offset_m.allFinite() || !(offset_m.norm() > 0.0)) {
    why = "the description produced no finite non-zero offset from the tip pivot to the tool "
      "centre point, so `y_n` cannot be placed on the native tool goal";
    return false;
  }
  return true;
}

bool translate_a2b_payload(
  const CalcMovement::Request & request, crane_msgs::msg::Payload & payload, std::string & why)
{
  payload = crane_msgs::msg::Payload{};

  // An empty gripper, and every log field left unread. The timber trees send a
  // `logCarrying` shape on the *approach* leg too, describing the log they are
  // about to pick up; reading it there would hang a log in a gripper that is
  // open.
  if (!request.carries_log) {
    payload.shape = crane_msgs::msg::Payload::SHAPE_NONE;
    return true;
  }

  // `log_carrying` is the inertia shape and `coll_shape` is the collision shape,
  // and `crane_msgs/Payload` is one shape for both. Every caller that carries
  // sends them equal -- the behaviour tree assigns one to the other -- so a
  // request where they differ is asking for two bodies, which is refused rather
  // than resolved in favour of whichever was read first.
  if (!same_length(request.log_carrying.length, request.coll_shape.length) ||
    !same_length(request.log_carrying.radius_top, request.coll_shape.radius_top) ||
    !same_length(request.log_carrying.radius_bottom, request.coll_shape.radius_bottom))
  {
    why = "`log_carrying` and `coll_shape` describe two different bodies, and `crane_msgs/Payload` "
      "carries one shape for the mass and for the collision check alike. Send the same cylinder in "
      "both, or plan the two separately";
    return false;
  }

  const float radius =
    std::max(request.log_carrying.radius_top, request.log_carrying.radius_bottom);
  if (!std::isfinite(request.log_carrying.length) || !(request.log_carrying.length > 0.0F) ||
    !std::isfinite(radius) || !(radius > 0.0F))
  {
    why = "`carries_log` is true but `log_carrying` has no positive length and radius (length = " +
      std::to_string(request.log_carrying.length) + " m, radius_top = " +
      std::to_string(request.log_carrying.radius_top) + " m, radius_bottom = " +
      std::to_string(request.log_carrying.radius_bottom) +
      " m); an unknown payload is not a zero-sized one, so send `carries_log` false instead";
    return false;
  }
  if (!std::isfinite(request.m_log) || !(request.m_log > 0.0)) {
    why = "`carries_log` is true but `m_log` is " + std::to_string(request.m_log) +
      " kg; an unknown payload is not a massless one, so send `carries_log` false instead";
    return false;
  }

  // The two centres. `p_cyl_8` is the collision body's centre and every caller
  // fills all three of its components; `s_log_8` is the centre of mass and the
  // behaviour tree leaves its z at NaN on purpose, which is that caller saying
  // "not specified" rather than "at infinity". `crane_msgs/Payload` carries one
  // `com` for both, so the unspecified components come from `p_cyl_8` and the
  // specified ones have to agree with it: a homogeneous log's mass centre *is*
  // its geometric centre, and a request where the two disagree is describing two
  // bodies again.
  if (!finite(request.p_cyl_8)) {
    why = "`p_cyl_8` is not three finite numbers, and it is the centre `crane_msgs/Payload` "
      "carries; the collision body has to have somewhere to be";
    return false;
  }
  const std::array<double, 3U> centre_of_mass{
    request.s_log_8.x, request.s_log_8.y, request.s_log_8.z};
  const std::array<double, 3U> collision_centre{
    request.p_cyl_8.x, request.p_cyl_8.y, request.p_cyl_8.z};
  for (std::size_t axis = 0; axis < 3U; ++axis) {
    if (std::isfinite(centre_of_mass[axis]) &&
      std::abs(centre_of_mass[axis] - collision_centre[axis]) > 1.0e-9)
    {
      why = "`s_log_8` and `p_cyl_8` are two different points on axis " + std::to_string(axis) +
        " (" + std::to_string(centre_of_mass[axis]) + " m against " +
        std::to_string(collision_centre[axis]) +
        " m), and `crane_msgs/Payload` carries one `com` for the mass and for the collision body "
        "alike. A homogeneous log's centre of mass is its geometric centre; send one point";
      return false;
    }
  }

  // A cylinder stays a cylinder, and `dimensions` is the extent along each axis
  // of the primitive's own frame (ROS 2 Interfaces 6): (2r, 2r, length), never a
  // radius in the first entry. A tapered log is bounded by the *enclosing*
  // cylinder rather than by the legacy mean of the two radii, because the mean
  // leaves the wider end sticking out of its own collision body.
  payload.shape = crane_msgs::msg::Payload::SHAPE_CYLINDER;
  payload.dimensions.x = 2.0 * static_cast<double>(radius);
  payload.dimensions.y = 2.0 * static_cast<double>(radius);
  payload.dimensions.z = static_cast<double>(request.log_carrying.length);
  payload.mass = request.m_log;
  payload.com = request.p_cyl_8;
  return true;
}

bool translate_a2b_request(
  const CalcMovement::Request & request, const Eigen::Vector3d & tip_to_tcp_offset_m,
  crane_msgs::srv::PlanMotion::Request & plan, std::string & why)
{
  plan = crane_msgs::srv::PlanMotion::Request{};

  // The start mapping is carried beside this PlanMotion request by PlannerNode:
  // it is not a field the native .srv has, but it enters the same internal
  // planning call after the ROS boundary. Invoke the one mapping here too so a
  // caller using this translation function directly gets the same validation.
  std::optional<MeasuredStart> start;
  if (!translate_a2b_start(request, start, why)) {
    return false;
  }
  if (!std::isfinite(request.t_end) || request.t_end != 0.0) {
    why = "`t_end` asks for a fixed end time of " + std::to_string(request.t_end) +
      " s, and the timing of trajectory_planning 5.2 derives the duration from the path and the "
      "machine's own force and flow limits; `crane_msgs/PlanMotion` has no field to override it. "
      "Send `t_end` 0.0, which every caller does today";
    return false;
  }
  if (!std::all_of(
      request.v_d_tip.begin(), request.v_d_tip.end(),
      [](double value) {return std::isfinite(value) && value == 0.0;}))
  {
    why = "`v_d_tip` asks the tool to still be moving at the end of the trajectory, and 5.4's "
      "terminal condition brings it to rest hanging still. `crane_msgs/PlanMotion` has no field "
      "for a non-zero terminal velocity, so this is refused rather than dropped";
    return false;
  }

  // Obstacles. The world model publishes them on `/crane/collision_scene`
  // already converted into `K0_mounting_base` (ROS 2 Interfaces 4); a request
  // cannot carry its own, and dropping the ones this one carries would plan
  // through them.
  if (!request.logs_scene.empty()) {
    why = "`logs_scene` carries " + std::to_string(request.logs_scene.size()) +
      " obstacles, and this planner takes the scene from " + std::string(kCollisionSceneTopic) +
      " where the world model publishes it. ROS 2 Interfaces 9 does not carry log perception over, "
      "so there is nothing to translate these into -- and silently planning without them would be "
      "planning through them";
    return false;
  }

  // The two collision flags become the one `avoid_collisions` of
  // `crane_msgs/PlanMotion`, which covers the crane and the payload primitive
  // together. With an empty gripper there is no log to check, so
  // `check_log_collision` says nothing and the gripper's flag decides alone --
  // which is what `check_pose_feasibility.py` sends. With a log in the gripper
  // the two have to agree, because there is no way to check one and not the
  // other.
  if (request.carries_log && request.check_log_collision != request.check_gripper_collision) {
    why =
      "`check_log_collision` and `check_gripper_collision` disagree while a log is carried, and "
      "`crane_msgs/PlanMotion` has one `avoid_collisions` covering the crane and the payload it "
      "holds. Checking one and not the other is not something this planner can be asked for";
    return false;
  }
  // Which leaves the gripper's flag as the one that decides in both branches:
  // with a log it is the flag the log's own agrees with, and without one it is
  // the only flag that means anything.
  plan.avoid_collisions = request.check_gripper_collision;

  // kappa is the deployment's reservation and `speed_scale` is the caller's own
  // request (trajectory_planning 5.5), which is exactly what `slow_down` is: a
  // divider on the legacy limits. A divider below one asks to go *faster* than
  // the deployment allows, which is the one thing 5.5 exists to prevent.
  if (!std::isfinite(request.slow_down) || !(request.slow_down >= 1.0)) {
    why = "`slow_down` is " + std::to_string(request.slow_down) +
      " and it maps to `speed_scale` = 1 / slow_down, which trajectory_planning 5.5 bounds to "
      "(0, 1]. A divider below one asks the planner to exceed the deployment's own reservation";
    return false;
  }
  plan.speed_scale = 1.0 / request.slow_down;

  if (!translate_a2b_payload(request, plan.payload, why)) {
    return false;
  }

  // The goal. `CalcMovement` carries no frame, and all three callers convert
  // into `K0_mounting_base` before they call -- so this asserts that frame and
  // converts nothing, and ROS 2 Interfaces 4's two conversion sites stay two.
  if (!finite(request.y_n) || !std::isfinite(request.phi_tool_n)) {
    why = "`y_n` and `phi_tool_n` are not four finite numbers, so there is no goal to plan to";
    return false;
  }
  if (!tip_to_tcp_offset_m.allFinite() || !(tip_to_tcp_offset_m.norm() > 0.0)) {
    why = "the offset from the tip pivot to the tool centre point has not been read out of the "
      "description, so `y_n` -- which CalcMovement documents as the position of the tip, K5 -- "
      "cannot be placed on the tool pose `crane_msgs/PlanMotion` asks for";
    return false;
  }
  plan.goal.header.frame_id = kPlanningFrame;
  plan.goal.pose.position.x = request.y_n.x + tip_to_tcp_offset_m.x();
  plan.goal.pose.position.y = request.y_n.y + tip_to_tcp_offset_m.y();
  plan.goal.pose.position.z = request.y_n.z + tip_to_tcp_offset_m.z();
  // phi_z about K0's z (wiki/nomenclature.md 5), written scalar-last because ROS
  // messages are and Eigen is not.
  plan.goal.pose.orientation.w = std::cos(0.5 * request.phi_tool_n);
  plan.goal.pose.orientation.x = 0.0;
  plan.goal.pose.orientation.y = 0.0;
  plan.goal.pose.orientation.z = std::sin(0.5 * request.phi_tool_n);
  return true;
}

void translate_a2b_response(
  const crane_msgs::srv::PlanMotion::Response & plan, CalcMovement::Response & response)
{
  // bool to bool, from `PlanMotion`'s field and from nothing else. The `int64`
  // ROS 2 Interfaces 1's `[!warning]` records is `CalcGripMovement`'s, and this
  // adapter does not serve that service.
  response.success = plan.success;
  if (!plan.success) {
    // `/crane/plan_motion` hands back whatever is still standing on
    // `/crane/reference` and says so in `message`. `CalcMovement::Response` has
    // no `message`, so a caller could not tell that trajectory from a new one --
    // and the legacy server left the field empty on failure.
    response.trajectory = trajectory_msgs::msg::JointTrajectory{};
    response.tcp_path.clear();
    return;
  }
  response.trajectory = plan.trajectory;
  response.tcp_path = plan.tcp_path;
}

}  // namespace crane_planning
