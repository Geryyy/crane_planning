#include "crane_planning/planner_core.hpp"

#include <cstddef>
#include <string>
#include <utility>

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

}  // namespace

crane_model::Result<PlannerContext> build_planner(
  const crane_model::Model & model, const std::string & robot_description_xml,
  const PlannerSettings & settings)
{
  auto limits = read_joint_limits(robot_description_xml, model.urdf_joint_names());
  if (!limits.ok()) {
    return Result<PlannerContext>::failure(limits.status());
  }
  auto geometry = probe_arm_geometry(
    model, limits.value(), settings.geometry_samples, settings.ik.geometry);
  if (!geometry.ok()) {
    return Result<PlannerContext>::failure(geometry.status());
  }

  PlannerContext context;
  context.limits = std::move(limits).value();
  context.geometry = std::move(geometry).value();
  context.settings = settings;
  return Result<PlannerContext>::success(std::move(context));
}

crane_model::Result<MotionPlan> plan_motion(
  const crane_model::Model & model, const PlannerContext & context,
  const MotionRequest & request)
{
  // Refused, not stubbed. `crane_msgs/PlanMotion` defaults `avoid_collisions` to
  // true, so a caller that fills nothing in gets this message rather than a
  // trajectory that reads as collision-checked and is not.
  if (request.avoid_collisions) {
    return Result<MotionPlan>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "avoid_collisions=true is refused: this planner checks no collision at all -- it does not "
        "subscribe to /crane/collision_scene and it knows nothing about the truck bed or the "
        "runges. Collision, including the tool sway envelope of trajectory_planning 4.3, arrives "
        "with issue 041. Set avoid_collisions=false to ask for the collision-blind plan this can "
        "actually produce"));
  }
  {
    Status status = check_margin_factor(request.margin_factor);
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }
  if (!request.q_a_start.allFinite()) {
    return Result<MotionPlan>::failure(
      failure(ErrorCode::NonFiniteInput, "the start configuration is not finite"));
  }

  IkRequest goal;
  goal.p_tcp_0 = request.p_tcp_0;
  goal.phi_z_d = request.phi_z_d;
  // The tool coordinate is held where the machine has it. q8 is not a path
  // variable (trajectory_planning 4.1); what opens and closes the gripper is
  // `/crane/plan_grip`, which is issue 044.
  goal.q8 = request.q_a_start[static_cast<Eigen::Index>(kToolRow)];
  goal.payload = request.payload;

  // The route decision of robot_model 2.2's closing paragraph, made here rather
  // than asked of the caller: a `/crane/plan_motion` goal is a *placement* goal,
  // so its endpoint has to be a genuine steady state of the passive subsystem
  // and it is solved with the equilibrium-constrained NLP. The semi-analytic
  // route is 2.2's "where speed matters" one and it still runs -- inside this
  // call, as the initial guess.
  auto endpoint = solve_equilibrium_constrained_ik(
    model, context.geometry, context.limits, context.settings.ik,
    context.settings.equilibrium, goal);
  if (!endpoint.ok()) {
    return Result<MotionPlan>::failure(endpoint.status());
  }

  MotionPlan plan;
  plan.endpoint = std::move(endpoint).value();

  crane_model::QA q_a_goal;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a_goal[static_cast<Eigen::Index>(row)] =
      plan.endpoint.q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }

  auto trajectory = scaled_ramp(
    request.q_a_start, q_a_goal, context.limits, request.margin_factor, context.settings.ramp);
  if (!trajectory.ok()) {
    return Result<MotionPlan>::failure(trajectory.status());
  }
  plan.trajectory = std::move(trajectory).value();

  // The visualisation path, and it is only that: the row it fills in the
  // response is documented `for visualization only`, and a straight line in
  // joint space is not a straight line here.
  plan.tcp_path.reserve(plan.trajectory.q_a_ref.size());
  for (const crane_model::QA & q_a : plan.trajectory.q_a_ref) {
    auto settled = model.passive_equilibrium(q_a, request.payload);
    if (!settled.ok()) {
      return Result<MotionPlan>::failure(settled.status());
    }
    Q q = Q::Zero();
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
    }
    q[4] = settled.value()[0];
    q[5] = settled.value()[1];
    auto pose = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
    if (!pose.ok()) {
      return Result<MotionPlan>::failure(pose.status());
    }
    plan.tcp_path.push_back(std::move(pose).value());
  }

  return Result<MotionPlan>::success(std::move(plan));
}

}  // namespace crane_planning
