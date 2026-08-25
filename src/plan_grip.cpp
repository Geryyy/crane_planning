#include "crane_planning/plan_grip.hpp"

#include <algorithm>
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

/// Every refusal this file makes names the phase it was answering.
Status refuse(GripPhase phase, ErrorCode code, std::string message)
{
  std::string text = std::string("the ") + grip_phase_name(phase) +
    " phase of /crane/plan_grip was refused: " + std::move(message);
  return Status{code, std::move(text)};
}

}  // namespace

const char * grip_phase_name(GripPhase phase) noexcept
{
  switch (phase) {
    case GripPhase::Descend: return "descend";
    case GripPhase::Close: return "close";
    case GripPhase::Open: return "open";
    case GripPhase::Lift: return "lift";
  }
  return "unknown";
}

bool grip_phase_from_message(std::uint8_t value, GripPhase & phase) noexcept
{
  switch (value) {
    case 1U: phase = GripPhase::Descend; return true;
    case 2U: phase = GripPhase::Close; return true;
    case 3U: phase = GripPhase::Open; return true;
    case 4U: phase = GripPhase::Lift; return true;
    default: return false;
  }
}

bool phase_moves_the_arm(GripPhase phase) noexcept
{
  return phase == GripPhase::Descend || phase == GripPhase::Lift;
}

crane_model::Result<GripPlan> plan_grip(
  const crane_model::Model & model, const PlannerContext & context, const GripRequest & request,
  LatencyLedger * ledger)
{
  // 7's latency bound. One phase is one plan, so one phase gets one budget --
  // the four of a grip are four separate requests from the task layer and
  // charging them against a shared total would make the last one refuse for what
  // the first three spent.
  LatencyLedger owned(context.settings.latency);
  LatencyLedger & clock = (ledger == nullptr) ? owned : *ledger;

  if (!request.start.q_a.allFinite() || !request.start.dq_a.allFinite()) {
    return Result<GripPlan>::failure(
      refuse(request.phase, ErrorCode::NonFiniteInput, "the start state is not finite"));
  }

  GripPlan plan;
  plan.phase = request.phase;
  plan.arm_phase = phase_moves_the_arm(request.phase);

  if (!plan.arm_phase) {
    // A tool phase emits the five path coordinates **held** (4.1's one clock), so
    // its first point commands the arm to a standstill. From a moving arm that is
    // a step, which is the sway excitation the whole architecture exists to
    // avoid, so it is refused rather than emitted.
    if (request.start.arm_is_moving(context.settings.start)) {
      return Result<GripPlan>::failure(
        refuse(
          request.phase, ErrorCode::NotReady,
          "the arm is still moving, and a tool phase emits the five path coordinates of 4.1 held: "
          "its first point would step the arm's velocity to zero, which is the excitation "
          "trajectory_planning 7 and mpc 6 both exist to avoid. Let the arm phase finish, or "
          "re-plan it, before closing or opening the gripper"));
    }
    // The tool, alone, on the arm's own clock. `tool_axis.hpp` is the whole of
    // it; what is decided here is only which end of the description's range this
    // phase drives to.
    auto target = tool_axis_target(
      context.limits, context.settings.tool_axis.closed_end, request.phase == GripPhase::Close);
    if (!target.ok()) {
      return Result<GripPlan>::failure(
        refuse(request.phase, target.status().code, target.status().message));
    }

    ToolAxisRequest tool;
    tool.q_a_start = request.start.q_a;
    tool.q8_goal = target.value();
    tool.payload = request.payload;
    tool.speed_scale = request.speed_scale;
    // The same kappa, the same hydraulic numbers and the same emitted grid the
    // arm phases are timed on. A tool phase held to a different margin from the
    // arm's would be a second reservation nobody declared.
    tool.kappa = context.settings.timing.kappa;
    tool.actuation = context.settings.timing.actuation;
    tool.sample_period = context.settings.timing.sample_period;
    tool.min_duration = context.settings.ramp.min_duration;

    auto driven = drive_tool_axis(model, context.limits, context.settings.tool_axis, tool);
    if (!driven.ok()) {
      return Result<GripPlan>::failure(
        refuse(request.phase, driven.status().code, driven.status().message));
    }
    plan.tool = std::move(driven).value();
    plan.trajectory = plan.tool.trajectory;
    Status charged = clock.charge("the tool-axis primitive of trajectory_planning 8");
    if (!charged.ok()) {
      return Result<GripPlan>::failure(
        refuse(request.phase, charged.code, std::move(charged.message)));
    }
    plan.stages = clock.stages();
    return Result<GripPlan>::success(std::move(plan));
  }

  // ------------------------------------------------------------------
  // The arm phases: `plan_motion` and nothing else, except that a **descend**
  // lowers the transfer altitude's ceiling to its own two endpoints.
  // ------------------------------------------------------------------
  auto settled_start = model.passive_equilibrium(request.start.q_a, request.payload);
  if (!settled_start.ok()) {
    return Result<GripPlan>::failure(
      refuse(request.phase, settled_start.status().code, settled_start.status().message));
  }
  Q q_start = Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_start[static_cast<Eigen::Index>(kActuatedRows[row])] =
      request.start.q_a[static_cast<Eigen::Index>(row)];
  }
  q_start.segment<2>(4) = settled_start.value();
  auto start_pose = model.forward_kinematics(q_start, Frame::MountingBase, Frame::Tcp);
  if (!start_pose.ok()) {
    return Result<GripPlan>::failure(
      refuse(request.phase, start_pose.status().code, start_pose.status().message));
  }

  // **The ceiling is a descend's and not a lift's**, and the asymmetry is the
  // primitive's own shape rather than a preference. `derive_transfer_altitude`
  // raises the higher endpoint by the mounted tool's reach, and the primitive is
  // always lift, traverse, descend. Capping the altitude at the higher endpoint
  // therefore collapses one of the three:
  //
  //   * on a **descend** the higher endpoint is the start, so what collapses is
  //     the *lift*. Its boundary condition is `sigma_dot(0)` free with the path
  //     meeting its own start at rest, which the OCP satisfies by construction,
  //     and the phase becomes across-and-down as its name says;
  //   * on a **lift** the higher endpoint is the goal, so what collapses is the
  //     final *descend* -- and 5.4's terminal condition `q_u(T) = q_eq,
  //     dq_u(T) = 0` is then imposed over shooting intervals in which the arm
  //     does not move, so nothing can steer the sway there. Measured: the same
  //     solve that converges in 83 SQP iterations at the derived altitude runs
  //     to `ACADOS_MAXITER` at the capped one, and it does so *erratically* --
  //     a ceiling 0.1 mm above the goal fails, 0.2 mm converges in 118, 0.5 mm
  //     fails again, 1 mm converges in 89. That is an ill-conditioned problem
  //     and not a margin to tune, so a lift keeps the derivation.
  //
  // A lift therefore rises past its goal by the tool's own reach, traverses and
  // sets down onto it. That is the derivation doing what it was written for --
  // clearing what the tool was among by the tool's whole length before the
  // traverse -- and a lift is the phase that starts among the runges holding a
  // block, so it is the phase that most wants it.
  PlannerContext phase_context = context;
  if (request.phase == GripPhase::Descend) {
    // The higher of the two endpoints, plus the residual the endpoint IK is
    // accepted at. Without that headroom a goal the solve closes a fraction of a
    // millimetre above where it was asked for would sit above its own transfer
    // altitude, and `build_structured_primitive` refuses that by name.
    const double endpoints_ceiling = std::max(
      start_pose.value().position_m.z(), request.p_tcp_0.z()) + context.settings.ik.eps_pos;
    plan.altitude_ceiling_applied = true;
    plan.altitude_ceiling_m = phase_context.settings.primitive.altitude.ceiling_m.has_value() ?
      std::min(*phase_context.settings.primitive.altitude.ceiling_m, endpoints_ceiling) :
      endpoints_ceiling;
    phase_context.settings.primitive.altitude.ceiling_m = plan.altitude_ceiling_m;
  }

  MotionRequest motion;
  motion.p_tcp_0 = request.p_tcp_0;
  motion.phi_z_d = request.phi_z_d;
  motion.start = request.start;
  motion.payload = request.payload;
  motion.payload_shape = request.payload_shape;
  motion.speed_scale = request.speed_scale;
  motion.avoid_collisions = request.avoid_collisions;
  motion.scene = request.scene;

  auto solved = plan_motion(model, phase_context, motion, &clock);
  if (!solved.ok()) {
    return Result<GripPlan>::failure(
      refuse(request.phase, solved.status().code, solved.status().message));
  }
  plan.motion = std::move(solved).value();
  plan.trajectory = plan.motion.trajectory;
  plan.stages = clock.stages();
  return Result<GripPlan>::success(std::move(plan));
}

std::string describe(const GripPlan & plan)
{
  std::string text = std::string("the ") + grip_phase_name(plan.phase) + " phase of ROS 2 "
    "Interfaces 5's /crane/plan_grip, answered over " +
    std::to_string(plan.trajectory.time_from_start.size()) + " points and " +
    std::to_string(plan.trajectory.duration) + " s. ";
  if (plan.arm_phase) {
    text += "It is an arm motion and it is the arm's own: the endpoint is the "
      "equilibrium-constrained IK of robot_model 2.2, the geometry came from " +
      std::string(mechanism_name(plan.motion.mechanism)) +
      " and the timing from the path-constrained OCP of trajectory_planning 5.2, so this phase "
      "carries the same collision check, the same sway envelope and the same kappa as "
      "/crane/plan_motion and is not a second path generator. ";
    text += plan.altitude_ceiling_applied ?
      "The transfer altitude's ceiling was lowered to " + std::to_string(plan.altitude_ceiling_m) +
      " m, the higher of this phase's own two endpoints, so the descend goes across and down "
      "rather than up, across and down. " :
      "The transfer altitude is the derivation's own: a lift starts among whatever the tool was "
      "reaching into and clearing that by the tool's whole length before the traverse is what the "
      "derivation was written for. ";
    text += (plan.motion.mechanism == PathMechanism::SamplingFallback) ?
      describe(plan.motion.sampled) :
      describe(plan.motion.primitive.altitude) + ". " + plan.motion.primitive.check.note;
  } else {
    text += describe(plan.tool);
    text += ". The five path coordinates of trajectory_planning 4.1 are held and emitted beside "
      "it, on the same sample period the arm phases are emitted on: geometrically the tool is "
      "decoupled from the arm, temporally it is not, and this trajectory is on the arm's clock";
  }
  return text;
}

}  // namespace crane_planning
