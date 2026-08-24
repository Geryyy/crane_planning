#include "crane_planning/structured_primitive.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::Result;
using crane_model::Status;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

/// A refusal that names the phase, which every refusal in this file does.
Status refuse(PrimitivePhase phase, const std::string & why)
{
  return failure(
    ErrorCode::InvalidArgument,
    std::string("the ") + phase_name(phase) +
    " phase of the structured primitive (trajectory_planning 4.4) could not be built: " + why +
    ". The primitive is refused rather than deformed, and the sampling fallback that would take "
    "over is issue 042");
}

/// The path-space part of a canonical eight-vector.
PathVector path_of(const Q & q)
{
  PathVector q_a = PathVector::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }
  return q_a;
}

double tool_of(const Q & q)
{
  return q[static_cast<Eigen::Index>(kActuatedRows[kToolRow])];
}

/// Whether an endpoint is inside the range the description gives every axis.
Status inside_joint_limits(const Q & q, const JointLimits & limits, const char * which)
{
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (!limits.axis[row].bounded) {
      continue;
    }
    const double value = q[static_cast<Eigen::Index>(kActuatedRows[row])];
    if (value < limits.axis[row].lower || value > limits.axis[row].upper) {
      return failure(
        ErrorCode::InvalidArgument,
        std::string("the ") + which + " configuration puts actuated coordinate " +
        std::to_string(row) + " at " + std::to_string(value) + ", outside the [" +
        std::to_string(limits.axis[row].lower) + ", " + std::to_string(limits.axis[row].upper) +
        "] the description gives it");
    }
  }
  return Status{};
}

/// Where the endpoint's TCP is, on top of what `measure_tool_reach` returns.
struct EndpointGeometry
{
  Eigen::Vector3d p_tcp{Eigen::Vector3d::Zero()};
  ToolReach tool{};
};

Result<EndpointGeometry> probe_endpoint(
  const crane_model::Model & model, const Q & q, PrimitivePhase phase)
{
  auto tcp = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!tcp.ok()) {
    return Result<EndpointGeometry>::failure(refuse(phase, tcp.status().message));
  }
  auto reach = measure_tool_reach(model, q);
  if (!reach.ok()) {
    return Result<EndpointGeometry>::failure(refuse(phase, reach.status().message));
  }

  EndpointGeometry endpoint;
  endpoint.p_tcp = tcp.value().position_m;
  endpoint.tool = std::move(reach).value();
  return Result<EndpointGeometry>::success(endpoint);
}

}  // namespace

const char * phase_name(PrimitivePhase phase) noexcept
{
  switch (phase) {
    case PrimitivePhase::Lift:
      return "lift";
    case PrimitivePhase::Traverse:
      return "traverse";
    case PrimitivePhase::Descend:
      return "descend";
  }
  return "unknown";
}

const char * tool_reference_name(ToolReference reference) noexcept
{
  switch (reference) {
    case ToolReference::ToolContact:
      return "the gripper's own contact point";
    case ToolReference::ToolCentre:
      return "the tool centre point, which is as deep as this description goes";
  }
  return "an unknown tool frame";
}

crane_model::Result<ToolReach> measure_tool_reach(
  const crane_model::Model & model, const crane_model::Q & q)
{
  auto rotator = model.forward_kinematics(q, Frame::MountingBase, Frame::Rotator);
  if (!rotator.ok()) {
    return Result<ToolReach>::failure(rotator.status());
  }
  auto tcp = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!tcp.ok()) {
    return Result<ToolReach>::failure(tcp.status());
  }

  ToolReach measured;
  measured.z_tcp_m = tcp.value().position_m.z();
  measured.phi_z = phi_z_of(tcp.value().orientation);

  // The deepest tool frame the description carries. The model API contract 3 is
  // explicit that only the 7040's gripper defines a contact point and that the
  // PZS100 gets `FrameUnavailable` rather than a substituted pose, so this is a
  // difference between the two descriptions and not a failure of either.
  auto contact = model.forward_kinematics(q, Frame::MountingBase, Frame::ToolContact);
  const Eigen::Vector3d deepest =
    contact.ok() ? contact.value().position_m : tcp.value().position_m;
  measured.reference = contact.ok() ? ToolReference::ToolContact : ToolReference::ToolCentre;

  // The distance rather than its vertical component: at an equilibrium the tool
  // hangs, so the two agree, but the 7040's tool sits off the tilt axis and does
  // not hang exactly vertically -- and a clearance that shrinks as the tool tips
  // would be a clearance that fails exactly when it is needed.
  measured.reach_m = (deepest - rotator.value().position_m).norm();
  if (!(measured.reach_m > 0.0)) {
    return Result<ToolReach>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the description places the whole tool on the rotator bearing, so it has no reach for "
        "the transfer altitude to be derived from"));
  }
  return Result<ToolReach>::success(measured);
}

SceneExtent scene_without_obstacles()
{
  // The read that is not happening. Nothing in this package subscribes to
  // /crane/collision_scene, holds the truck bed or the runges of
  // trajectory_planning 4.2, or inflates the tool by the sway envelope of 4.3.
  // Returning `obstacles_known = false` with a NaN extent is the difference
  // between "no obstacle is known" and "no obstacle is there"; issue 041 is the
  // issue that makes them the same statement.
  return SceneExtent{};
}

crane_model::Result<TransferAltitude> derive_transfer_altitude(
  double z_tcp_start_m, double tool_reach_start_m, double z_tcp_goal_m, double tool_reach_goal_m,
  const SceneExtent & scene, const TransferAltitudeSettings & settings)
{
  if (!std::isfinite(z_tcp_start_m) || !std::isfinite(z_tcp_goal_m) ||
    !std::isfinite(tool_reach_start_m) || !std::isfinite(tool_reach_goal_m))
  {
    return Result<TransferAltitude>::failure(
      failure(ErrorCode::NonFiniteInput, "an endpoint altitude or tool reach is not finite"));
  }
  if (!(tool_reach_start_m > 0.0) || !(tool_reach_goal_m > 0.0)) {
    return Result<TransferAltitude>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "a tool reach of zero leaves the transfer altitude with nothing to clear the endpoints "
        "by, and a hard-coded clearance is what trajectory_planning 9 asks this issue to remove"));
  }

  TransferAltitude altitude;
  altitude.z_tcp_start_m = z_tcp_start_m;
  altitude.z_tcp_goal_m = z_tcp_goal_m;
  altitude.tool_reach_start_m = tool_reach_start_m;
  altitude.tool_reach_goal_m = tool_reach_goal_m;

  // The endpoints. Raising the TCP by the tool's own reach puts the tool's
  // lowest point at the altitude the TCP held at that endpoint, so the tool
  // clears whatever it was reaching into by its whole length before the traverse
  // starts. Both endpoints ask, and the taller ask wins.
  altitude.z_from_endpoints_m =
    std::max(z_tcp_start_m + tool_reach_start_m, z_tcp_goal_m + tool_reach_goal_m);
  altitude.z_m = altitude.z_from_endpoints_m;

  // The obstacles. Same statement about the tallest thing in the way, and it has
  // no input until issue 041 -- see `scene_without_obstacles()`.
  if (scene.obstacles_known) {
    if (!std::isfinite(scene.highest_obstacle_z_m)) {
      return Result<TransferAltitude>::failure(
        failure(
          ErrorCode::InvalidScene,
          "the scene claims to know its obstacle extent and does not carry a finite one"));
    }
    altitude.z_from_obstacles_m =
      scene.highest_obstacle_z_m + std::max(tool_reach_start_m, tool_reach_goal_m);
    altitude.obstacle_term_applied = true;
    altitude.z_m = std::max(altitude.z_m, altitude.z_from_obstacles_m);
  }

  // A configured floor may raise the derivation and a configured ceiling may
  // lower it. Neither may be it: with no endpoints there is no altitude here at
  // all, whatever a deployment configured.
  if (settings.floor_m.has_value()) {
    if (!std::isfinite(*settings.floor_m)) {
      return Result<TransferAltitude>::failure(
        failure(
          ErrorCode::InvalidArgument, "the configured transfer-altitude floor is not finite"));
    }
    if (*settings.floor_m > altitude.z_m) {
      altitude.z_m = *settings.floor_m;
      altitude.floor_binding = true;
    }
  }
  if (settings.ceiling_m.has_value()) {
    if (!std::isfinite(*settings.ceiling_m)) {
      return Result<TransferAltitude>::failure(
        failure(
          ErrorCode::InvalidArgument, "the configured transfer-altitude ceiling is not finite"));
    }
    if (*settings.ceiling_m < altitude.z_m) {
      altitude.z_m = *settings.ceiling_m;
      altitude.ceiling_binding = true;
    }
  }

  return Result<TransferAltitude>::success(altitude);
}

std::string describe(const TransferAltitude & altitude)
{
  std::string text = "the transfer altitude is " + std::to_string(altitude.z_m) +
    " m, derived from the endpoints at " + std::to_string(altitude.z_tcp_start_m) + " m and " +
    std::to_string(altitude.z_tcp_goal_m) + " m raised by the mounted tool's own reach of " +
    std::to_string(altitude.tool_reach_start_m) + " m and " +
    std::to_string(altitude.tool_reach_goal_m) + " m, measured to " +
    tool_reference_name(altitude.reference_goal);
  text += altitude.obstacle_term_applied ?
    ", and by an obstacle extent asking for " + std::to_string(altitude.z_from_obstacles_m) + " m" :
    ". The obstacle term of the derivation has no input: /crane/collision_scene arrives with "
    "issue 041, so this altitude clears the endpoints and nothing else";
  if (altitude.floor_binding) {
    text += ". A configured floor raised it";
  }
  if (altitude.ceiling_binding) {
    text += ". A configured ceiling lowered it";
  }
  return text;
}

PrimitiveCheck check_primitive(const GeometricPath & path)
{
  PrimitiveCheck check;
  check.clear = true;
  check.note = "The " + std::to_string(path.segment_count()) +
    " segments of the primitive were not checked against anything. trajectory_planning 4.4 is "
    "generate, check, accept, and this is the check: it passes unconditionally because there is "
    "no collision model here yet -- no /crane/collision_scene, no truck bed, no runges, and no "
    "tool inflated by the sway envelope of 4.3. This body is what issue 041 replaces, and this "
    "call site is already where it belongs";
  return check;
}

crane_model::Result<StructuredPrimitive> build_structured_primitive(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & ik, const PrimitiveSettings & settings, const PrimitiveRequest & request)
{
  if (!request.q_start.allFinite() || !request.q_goal.allFinite()) {
    return Result<StructuredPrimitive>::failure(
      refuse(PrimitivePhase::Lift, "an endpoint configuration is not finite"));
  }

  // The joint range, before anything is solved. The lift begins at the start and
  // the descend ends at the goal, so that is which phase each endpoint belongs
  // to and which phase its refusal names.
  {
    Status status = inside_joint_limits(request.q_start, limits, "start");
    if (!status.ok()) {
      return Result<StructuredPrimitive>::failure(
        refuse(PrimitivePhase::Lift, status.message));
    }
    status = inside_joint_limits(request.q_goal, limits, "goal");
    if (!status.ok()) {
      return Result<StructuredPrimitive>::failure(
        refuse(PrimitivePhase::Descend, status.message));
    }
  }

  auto start = probe_endpoint(model, request.q_start, PrimitivePhase::Lift);
  if (!start.ok()) {
    return Result<StructuredPrimitive>::failure(start.status());
  }
  auto goal = probe_endpoint(model, request.q_goal, PrimitivePhase::Descend);
  if (!goal.ok()) {
    return Result<StructuredPrimitive>::failure(goal.status());
  }

  auto derived = derive_transfer_altitude(
    start.value().p_tcp.z(), start.value().tool.reach_m, goal.value().p_tcp.z(),
    goal.value().tool.reach_m, request.scene, settings.altitude);
  if (!derived.ok()) {
    return Result<StructuredPrimitive>::failure(
      refuse(PrimitivePhase::Lift, derived.status().message));
  }

  StructuredPrimitive primitive;
  primitive.altitude = std::move(derived).value();
  primitive.altitude.reference_start = start.value().tool.reference;
  primitive.altitude.reference_goal = goal.value().tool.reference;

  // The one thing the derivation cannot be allowed to return. It only ever comes
  // from a configured ceiling, because every other term raises the altitude
  // above both endpoints by construction -- and a transfer altitude below the
  // endpoints is not a lower lift, it is no lift at all.
  const double highest_endpoint = std::max(start.value().p_tcp.z(), goal.value().p_tcp.z());
  if (primitive.altitude.z_m < highest_endpoint) {
    return Result<StructuredPrimitive>::failure(
      refuse(
        PrimitivePhase::Lift,
        "the transfer altitude came out at " + std::to_string(primitive.altitude.z_m) +
        " m, below the higher of the two endpoints at " + std::to_string(highest_endpoint) +
        " m, so there is nothing to lift to. " + describe(primitive.altitude)));
  }

  // The two interior waypoints: straight up from the start, straight down onto
  // the goal, both at the transfer altitude and both at their own end's yaw.
  const double q8_start = tool_of(request.q_start);
  const double q8_goal = tool_of(request.q_goal);

  IkRequest lift;
  lift.p_tcp_0 =
    Eigen::Vector3d(start.value().p_tcp.x(), start.value().p_tcp.y(), primitive.altitude.z_m);
  lift.phi_z_d = start.value().tool.phi_z;
  lift.q8 = q8_start;
  lift.payload = request.payload;
  auto lift_solution = solve_inverse_kinematics(model, geometry, limits, ik, lift);
  if (!lift_solution.ok()) {
    return Result<StructuredPrimitive>::failure(
      refuse(
        PrimitivePhase::Lift,
        "the arm does not reach straight up from the start to the transfer altitude: " +
        lift_solution.status().message + ". " + describe(primitive.altitude)));
  }
  primitive.lift_waypoint = std::move(lift_solution).value();

  IkRequest traverse;
  traverse.p_tcp_0 =
    Eigen::Vector3d(goal.value().p_tcp.x(), goal.value().p_tcp.y(), primitive.altitude.z_m);
  traverse.phi_z_d = goal.value().tool.phi_z;
  traverse.q8 = q8_goal;
  traverse.payload = request.payload;
  auto traverse_solution = solve_inverse_kinematics(model, geometry, limits, ik, traverse);
  if (!traverse_solution.ok()) {
    return Result<StructuredPrimitive>::failure(
      refuse(
        PrimitivePhase::Traverse,
        "the arm does not reach the point above the goal at the transfer altitude: " +
        traverse_solution.status().message + ". " + describe(primitive.altitude)));
  }
  primitive.traverse_waypoint = std::move(traverse_solution).value();

  // Generate. One fit over the four waypoints, C2 across both junctions because
  // the fit imposes the same velocity and the same zero acceleration from either
  // side of each. There is no corner-rounding pass after this and 4.5 is the
  // reason there must not be one.
  PathFitRequest fit;
  fit.waypoints = {
    path_of(request.q_start), path_of(primitive.lift_waypoint.q),
    path_of(primitive.traverse_waypoint.q), path_of(request.q_goal)};
  fit.segment_names = {
    phase_name(PrimitivePhase::Lift), phase_name(PrimitivePhase::Traverse),
    phase_name(PrimitivePhase::Descend)};
  fit.q8_start = q8_start;
  fit.q8_goal = q8_goal;

  auto fitted = fit_c2_path(fit, limits, settings.fit);
  if (!fitted.ok()) {
    // The fit was given the phase names, so its message already says which of
    // the three failed and this only says what it was doing.
    return Result<StructuredPrimitive>::failure(
      failure(
        fitted.status().code,
        "the structured primitive of trajectory_planning 4.4 could not be built: " +
        fitted.status().message +
        ". The primitive is refused rather than deformed, and the sampling fallback that would "
        "take over is issue 042"));
  }
  primitive.path = std::move(fitted).value();

  // Check, then accept. 4.4's order, as one step, so that issue 041 swaps a body
  // rather than a structure.
  primitive.check = check_primitive(primitive.path);
  if (!primitive.check.clear) {
    return Result<StructuredPrimitive>::failure(
      refuse(primitive.check.phase, primitive.check.note));
  }

  return Result<StructuredPrimitive>::success(std::move(primitive));
}

}  // namespace crane_planning
