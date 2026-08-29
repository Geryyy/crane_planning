#include "crane_planning/planner_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "crane_planning/status.hpp"

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::Result;
using crane_model::Status;

}  // namespace

crane_model::Result<PlannerContext> build_planner(
  const crane_model::Model & model, const crane_model::ModelConfig & model_config,
  const PlannerSettings & settings)
{
  auto limits = read_joint_limits(model_config.robot_description_xml, model.urdf_joint_names());
  if (!limits.ok()) {
    return Result<PlannerContext>::failure(limits.status());
  }
  auto geometry = probe_arm_geometry(
    model, limits.value(), settings.geometry_samples, settings.ik.geometry);
  if (!geometry.ok()) {
    return Result<PlannerContext>::failure(geometry.status());
  }
  // The force limit of trajectory_planning 3, derived here rather than per
  // request: the chamber areas are the description's and do not move, and the
  // relief setting is a deployment constant. A description whose cylinders have
  // no usable area is refused here instead of at the first plan, so a machine
  // that cannot have a force limit never answers a request as though it had one.
  auto forces = derive_cylinder_force_limits(model, settings.system_pressure_pa);
  if (!forces.ok()) {
    return Result<PlannerContext>::failure(forces.status());
  }

  PlannerContext context;
  context.limits = std::move(limits).value();
  context.geometry = std::move(geometry).value();
  context.settings = settings;
  context.settings.timing.actuation.cylinder_force_max = forces.value();
  return Result<PlannerContext>::success(std::move(context));
}

crane_model::Result<MotionPlan> plan_motion(
  const crane_model::Model & model, const PlannerContext & context,
  const MotionRequest & request, LatencyLedger * ledger)
{
  // 7's latency bound, opened here so that everything below it is inside the
  // budget -- the endpoint IK, the geometry, the smoothing and the OCP. A caller
  // that brought its own reads which stage overran afterwards; one that did not
  // still gets the bound, because the bound is the planner's and not the
  // caller's.
  LatencyLedger owned(context.settings.latency);
  LatencyLedger & clock = (ledger == nullptr) ? owned : *ledger;

  // `crane_msgs/PlanMotion` defaults `avoid_collisions` to true, so a caller
  // that fills nothing in asks for a checked plan -- and gets a refusal rather
  // than a trajectory that reads as checked when no scene has arrived.
  if (request.avoid_collisions && request.scene == nullptr) {
    return Result<MotionPlan>::failure(
      failure(
        ErrorCode::NotReady,
        "avoid_collisions is set and nothing has been received on /crane/collision_scene, so "
        "there is no scene to check against. The truck bed and the runges of "
        "trajectory_planning 4.2 are keyed to the truck pose that scene carries, so they are not "
        "known either. Set avoid_collisions=false to ask for a plan that says in as many words "
        "that nothing was checked"));
  }
  {
    Status status = check_margin_factor(request.speed_scale);
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }
  if (!request.start.q_a.allFinite() || !request.start.dq_a.allFinite()) {
    return Result<MotionPlan>::failure(
      failure(ErrorCode::NonFiniteInput, "the start state is not finite"));
  }

  // 7's start, decided once and carried from here down. Three facts come out of
  // it and each of them is a *statement* rather than a default: whether the arm
  // is moving, whether the passive pair was measured, and what is reserved for
  // the sway when it was not.
  const StartStateSettings & at_rest = context.settings.start;
  const bool arm_is_moving = request.start.arm_is_moving(at_rest);
  const bool passive_known = request.start.passive.measured;
  const bool tool_is_swinging = request.start.tool_is_swinging(at_rest);

  // The one combination that has no honest answer. A moving arm re-planned with
  // an *assumed* still tool is precisely the mis-prediction 7's `[!warning]`
  // describes -- the first step's sway is predicted from a state nobody measured,
  // and a stall-recovery re-plan is the situation this occurs in. A machine
  // standing still is a different case: there the hanging pose is not an
  // assumption about the sway, it is where a settled tool is.
  if (arm_is_moving && !passive_known) {
    return Result<MotionPlan>::failure(
      failure(
        ErrorCode::NotReady,
        "this is a re-plan from a moving arm and the passive pair was not measured, so the sway "
        "the first step is predicted from would be assumed rather than known -- which is the "
        "mis-prediction trajectory_planning 7 names, and the situation a stall-recovery re-plan "
        "occurs in. " + request.start.passive.note));
  }
  if (!request.start.passive.q_u.allFinite() || !request.start.passive.dq_u.allFinite()) {
    return Result<MotionPlan>::failure(
      failure(ErrorCode::NonFiniteInput, "the measured passive start state is not finite"));
  }

  IkRequest goal;
  goal.p_tcp_0 = request.p_tcp_0;
  goal.phi_z_d = request.phi_z_d;
  // The tool coordinate is held where the machine has it. q8 is not a path
  // variable (trajectory_planning 4.1); what opens and closes the gripper is
  // `external grip service`.
  goal.q8 = request.start.q_a[static_cast<Eigen::Index>(kToolRow)];
  goal.payload = request.payload;
  // The clearance half of robot_model 2.2 step 3's redundancy score, which is a
  // preference between telescope extensions rather than a check. It is offered
  // the scene whenever there is one: `avoid_collisions` governs whether the path
  // is *checked*, and a well-cleared extension is not a worse answer to a caller
  // who asked for no check.
  goal.scene = request.scene;

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
  {
    Status status = clock.charge("the endpoint IK of robot_model 2.2");
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }

  MotionPlan plan;
  plan.endpoint = std::move(endpoint).value();

  // The start as all eight coordinates. The passive pair is what was measured
  // off `/joint_states`, or -- for a machine standing still with no usable
  // measurement -- where the tool hangs at the measured actuated configuration,
  // which is the stopped-start convention of 7 and is now the stated branch
  // rather than the only one.
  auto settled_start = model.passive_equilibrium(request.start.q_a, request.payload);
  if (!settled_start.ok()) {
    return Result<MotionPlan>::failure(settled_start.status());
  }
  const crane_model::QU q_u_start =
    passive_known ? request.start.passive.q_u : settled_start.value();

  OmplRequest geometry;
  geometry.q_start = expand(request.start.q_a, q_u_start);
  // The measured rate over the five path coordinates of 4.1, as the first
  // segment's own boundary condition. The tool row is not among them and is
  // checked separately below: q8 rides a quintic that is flat at both ends, so no
  // path this fit produces can carry a moving tool axis.
  for (std::size_t row = 0; row < kPathDof; ++row) {
    geometry.dq_start[static_cast<Eigen::Index>(row)] =
      request.start.dq_a[static_cast<Eigen::Index>(row)];
  }
  if (std::abs(request.start.dq_a[static_cast<Eigen::Index>(kToolRow)]) > at_rest.at_rest_dq_a) {
    return Result<MotionPlan>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the tool axis is measured moving at " +
        std::to_string(request.start.dq_a[static_cast<Eigen::Index>(kToolRow)]) +
        " rad/s or m/s. trajectory_planning 4.1 keeps q8 out of the path variables and it rides a "
        "shape that is flat at both ends, so no path this planner fits can leave its start with "
        "the tool moving -- and emitting one that reads as continuous while it is not is the "
        "defect 7 asks continuity to be a boundary condition against. Wait for the tool axis to "
        "stop before planning the arm motion"));
  }
  geometry.q_goal = plan.endpoint.q;
  geometry.payload = request.payload;
  geometry.payload_shape = request.payload_shape;
  geometry.collision_scene = request.avoid_collisions ? request.scene : nullptr;

  auto searched = plan_ompl_path(model, context.limits, context.settings.ompl, geometry);
  if (!searched.ok()) {
    return Result<MotionPlan>::failure(searched.status());
  }
  plan.geometry = std::move(searched).value();
  plan.path = plan.geometry.path;
  {
    Status status = clock.charge("OMPL RRT-Connect and C2 path fitting");
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }

  // 7's initial condition, posed against the path that was actually produced.
  // The measured actuated velocity reaches the OCP as a *path rate* -- stage 2
  // writes `dq_a = q_a'(sigma) sigma_dot`, so the one number that reproduces the
  // measurement through this path's own start slope is what pins `sigma_dot(0)`.
  // `start_path_rate` computes it from the fitted path rather than from what the
  // fit was asked for, and its residual is what makes the continuity of 7 a
  // boundary condition that was *checked* rather than one that was intended.
  TimingOcpRequest timing;
  timing.payload = request.payload;
  timing.speed_scale = request.speed_scale;
  timing.start.measured = passive_known;
  timing.start.q_u = q_u_start;
  timing.start.dq_u = passive_known ?
    request.start.passive.dq_u : crane_model::DQU::Zero();
  if (arm_is_moving) {
    PathVector measured = PathVector::Zero();
    for (std::size_t row = 0; row < kPathDof; ++row) {
      measured[static_cast<Eigen::Index>(row)] =
        request.start.dq_a[static_cast<Eigen::Index>(row)];
    }
    const StartPathRate rate = start_path_rate(plan.path, measured);
    if (!rate.defined) {
      return Result<MotionPlan>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the machine is moving and the path that was built meets its start at rest, so no path "
          "rate reproduces the measurement and the first emitted velocity would be a step down to "
          "zero. trajectory_planning 7 makes continuity a boundary condition rather than a "
          "tolerance, so this is refused rather than emitted"));
    }
    // The tolerance is the one the criterion states, and it is tight on purpose:
    // the fit was *given* this direction, so the two vectors are parallel by
    // construction and the residual is rounding. Anything larger means the path
    // leaves its start in a direction the machine is not travelling in, and a
    // trajectory whose first velocity is not the measured one is exactly the
    // seam 7 asks to be matched in position *and* velocity.
    constexpr double kVelocitySeamTolerance = 1.0e-9;
    if (!(rate.residual <= kVelocitySeamTolerance)) {
      return Result<MotionPlan>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the path leaves its start in a direction the machine is not travelling in: the best "
          "path rate still misses the measured joint velocity by " +
          std::to_string(rate.residual) + " rad/s against a seam tolerance of " +
          std::to_string(kVelocitySeamTolerance) +
          " rad/s. A concatenated segment has to match the previous one in position *and* "
          "velocity (trajectory_planning 7), so this is a refusal and not a rounding"));
    }
    timing.start.sigma_rate_pinned = true;
    timing.start.sigma_rate = rate.sigma_rate;
  }
  plan.start = timing.start;

  // The sway allowance held back when the passive pair was not measured. It is a
  // reservation and not an assumption: the plan is solved inside a *smaller* box
  // than mpc 3 constraints 3 and 4 permit, so the sway that may be there anyway
  // still fits inside what the controller allows and inside the envelope 4.3
  // cleared the path over. Zero reserve is a deployment that asked for the
  // request to be refused instead, and the node is where that decision is made.
  TimingOcpSettings timing_settings = context.settings.timing;
  if (!passive_known &&
    (request.start.passive.sway_reserve_rad > 0.0 ||
    request.start.passive.sway_rate_reserve > 0.0))
  {
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      timing_settings.q_u_max[row] -= request.start.passive.sway_reserve_rad;
      timing_settings.dq_u_max[row] -= request.start.passive.sway_rate_reserve;
      if (!(timing_settings.q_u_max[row] > 0.0) || !(timing_settings.dq_u_max[row] > 0.0)) {
        return Result<MotionPlan>::failure(
          failure(
            ErrorCode::InvalidArgument,
            "the conservative bound held back for the unmeasured sway is the whole of passive row "
            + std::to_string(row) +
            "'s allowance or more, so there is no box left to plan inside. Either the reserve is "
            "larger than mpc 3 constraint 3 permits at all, or this deployment should be refusing "
            "the request rather than bounding it"));
      }
    }
  }

  // trajectory_planning 5.2, and not 038's ramp: the timing that carries the
  // sway, holds the cylinder force and the pump flow inside what the hydraulics
  // can deliver, and leaves 5.5's kappa of the machine's authority unspent for
  // the MPC. A solve that does not converge is a refusal carrying acados' own
  // status word -- never a clipped trajectory (mpc.md 5.3 requirement 1).
  //
  // The solver's own wall clock is lowered to what 7's total has left, so the two
  // bounds cannot disagree about which of them fired: the OCP stops inside the
  // budget and the budget is what reports it.
  timing_settings.max_wall_clock =
    std::min(timing_settings.max_wall_clock, clock.remaining());
  auto solved = solve_timing_ocp(
    model, plan.path, context.limits, timing, timing_settings);
  if (!solved.ok()) {
    // The budget is charged first, so a solve that ran out of time is reported as
    // the budget firing rather than as acados' timeout -- 7 asks for the bound to
    // be the planner's, and a caller that reads `ACADOS_TIMEOUT` learns which
    // library gave up rather than that the plan exceeded its latency.
    Status overrun = clock.charge("the path-constrained OCP of trajectory_planning 5.2");
    if (!overrun.ok()) {
      return Result<MotionPlan>::failure(std::move(overrun));
    }
    return Result<MotionPlan>::failure(solved.status());
  }
  {
    Status status = clock.charge("the path-constrained OCP of trajectory_planning 5.2");
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }
  plan.timing = std::move(solved).value();
  plan.trajectory = plan.timing.trajectory;

  // The visualisation path, and it is only that: the row it fills in the
  // response is documented `for visualization only`. It is where the tool hangs
  // at each sampled configuration, so it shows the lift and the descend as the
  // operator will see them and it is not what any check ran against.
  plan.tcp_path.reserve(plan.trajectory.q_a_ref.size());
  for (const crane_model::QA & q_a : plan.trajectory.q_a_ref) {
    auto settled = model.passive_equilibrium(q_a, request.payload);
    if (!settled.ok()) {
      return Result<MotionPlan>::failure(settled.status());
    }
    const Q q = expand(q_a, settled.value());
    auto pose = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
    if (!pose.ok()) {
      return Result<MotionPlan>::failure(pose.status());
    }
    plan.tcp_path.push_back(std::move(pose).value());
  }
  {
    Status status = clock.charge("the visualisation path");
    if (!status.ok()) {
      return Result<MotionPlan>::failure(std::move(status));
    }
  }

  // What the start actually was, in as many words. `crane_msgs/PlanMotion` is
  // frozen (PRD 15) so `message` is the only surface this reaches a caller on,
  // and "planned from a measurement" against "planned from the hanging pose
  // because nothing measured it" is the difference 7 is about.
  plan.start_note = arm_is_moving ?
    "This is a re-plan from a moving machine: the start is (q, dq) as measured, so the path leaves "
    "its first waypoint at the measured joint velocity and the OCP's sigma_dot(0) = " +
    std::to_string(plan.start.sigma_rate) +
    " is the one path rate that reproduces it. trajectory_planning 7's stopped-start convention is "
    "not applied here" :
    "The machine is standing still to within " + std::to_string(at_rest.at_rest_dq_a) +
    " rad/s, so the start is at rest and the path meets it with q_a' = 0";
  plan.start_note += passive_known ?
    ". The passive pair is measured: q_u = (" + std::to_string(plan.start.q_u[0]) + ", " +
    std::to_string(plan.start.q_u[1]) + ") rad, dq_u = (" + std::to_string(plan.start.dq_u[0]) +
    ", " + std::to_string(plan.start.dq_u[1]) + ") rad/s, and the tool is " +
    (tool_is_swinging ? "swinging" : "hanging still") + " by it. " :
    ". The passive pair was **not** measured, so it is the hanging pose of the measured actuated "
    "configuration and not a statement about the sway. " + request.start.passive.note + " ";
  plan.stages = clock.stages();

  return Result<MotionPlan>::success(std::move(plan));
}

}  // namespace crane_planning
