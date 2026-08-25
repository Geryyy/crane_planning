// `/crane/plan_grip`: descend, close, open, lift -- four phases, one clock.
//
// `wiki/implementation/ros2_interfaces.md` 5 gives `crane_planner` this service
// beside `/crane/plan_motion`, and 10 settles that the two stay separate while
// the tested task layer migrates. `crane_msgs/PlanGrip` is frozen: a phase, a
// goal pose, a payload and a `speed_scale` in, a `JointTrajectory` out.
//
// # Two of the four are arm motions, and they are the arm's own motions
//
// `PHASE_DESCEND` and `PHASE_LIFT` are moves of the five path coordinates, and
// this file does **not** generate them. It assembles a `MotionRequest` and calls
// `plan_motion`, so a descend gets the endpoint of issue 039, the structured
// primitive of 040, the collision check and sway envelope of 041, the sampling
// fallback of 042 and the path-constrained OCP of 043 -- the same check, the
// same envelope, the same kappa. There is no second, looser path generator here
// and there must not be one: a grip descend is the phase that puts a tool
// between the runges, which is the least forgiving move the machine makes.
//
// What a **descend** does change is one existing knob. `plan_motion`'s transfer
// altitude is derived by `derive_transfer_altitude` as the higher endpoint
// raised by the mounted tool's own reach, which is right for a place-to-place
// move and wrong for a descend: it would lift the tool by the length of the tool
// before bringing it back down onto the block it is already above. So a descend
// lowers the derivation's **ceiling** to the higher of its own two endpoints --
// the one thing `TransferAltitudeSettings::ceiling_m` exists to do, and the one
// direction it is allowed to move the altitude in. The phase is then
// across-and-down, still `C2`, still checked, still timed by the OCP.
//
// The ceiling carries `eps_pos` of headroom, because the altitude is compared
// against the *solved* endpoint's own TCP height and the solve is accepted at
// that residual: without it a goal the IK closes a tenth of a millimetre high
// would refuse its own phase for having no lift to make.
//
// **A lift keeps the derivation, and that asymmetry is measured rather than
// preferred.** Capping the altitude at the higher endpoint collapses one of the
// primitive's three segments, and which one depends on which endpoint is higher.
// On a descend it is the first, whose boundary condition the path already meets
// at rest. On a lift it is the last -- and 5.4's terminal condition on the sway
// is then imposed over intervals in which the arm does not move, so nothing can
// steer `dq_u` to zero there. `plan_grip.cpp` carries the numbers: the same
// solve converges in 83 SQP iterations at the derived altitude and runs to
// `ACADOS_MAXITER` at the capped one, erratically in the margin. A lift
// therefore rises past its goal by the tool's own reach and sets down onto it,
// which is the derivation doing what it was written for -- a lift is the phase
// that starts among the runges holding a block.
//
// # Two of the four are the tool, and they ride the arm's clock
//
// `PHASE_CLOSE` and `PHASE_OPEN` move q8 alone, on the retained cosine primitive
// of `wiki/trajectory_planning.md` 8 and inside the tool axis's own limits.
// `tool_axis.hpp` is that primitive and its header is where the reasoning lives.
// What matters here is 4.1's sentence: "geometrically the tool is decoupled from
// the arm; temporally it is not, and the two must share one clock". A tool phase
// therefore comes back over the same six actuated rows, on the same sample
// period, with the first point at `time_from_start = 0` -- the arm's time base
// and not a private schedule -- and the five path coordinates are emitted held
// rather than left out.
//
// # No memory, in either direction
//
// The service holds nothing between calls. Two identical requests return the
// same trajectory, and which phase ran last is not a fact this planner has: the
// start of every phase is the machine's own measured configuration, so the task
// layer sequences the four by moving the machine and re-measuring, and the
// planner answers each one on its own. That is what makes "no phase returns a
// trajectory whose first point is not at the previous phase's last" true by
// construction rather than by bookkeeping -- and it is why the phase enum below
// is a request field and never a member.

#ifndef CRANE_PLANNING__PLAN_GRIP_HPP_
#define CRANE_PLANNING__PLAN_GRIP_HPP_

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/collision.hpp"
#include "crane_planning/planner_core.hpp"
#include "crane_planning/tool_axis.hpp"

namespace crane_planning
{

/// The four phases of `crane_msgs/PlanGrip`, at the values the frozen .srv fixes.
enum class GripPhase : std::uint8_t
{
  Descend = 1,
  Close = 2,
  Open = 3,
  Lift = 4
};

[[nodiscard]] const char * grip_phase_name(GripPhase phase) noexcept;

/// One `PlanGrip.phase` byte as the enum, or false for a value the .srv does not define.
/**
 * Converted and never cast, on the same terms `planner_node.cpp` converts the
 * collision shapes: a fifth phase added to the message later is then a refusal
 * here instead of whatever the cast happened to land on.
 */
[[nodiscard]] bool grip_phase_from_message(std::uint8_t value, GripPhase & phase) noexcept;

/// Whether the phase moves the arm (descend, lift) or the tool (close, open).
[[nodiscard]] bool phase_moves_the_arm(GripPhase phase) noexcept;

/// One `/crane/plan_grip` request, with the message already off it.
struct GripRequest
{
  GripPhase phase{GripPhase::Descend};

  /// The goal pose of the two arm phases. **Not read by a tool phase.**
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};
  double phi_z_d{};

  /// Where the machine is and what it is doing -- the same start `plan_motion` takes.
  /**
   * An arm phase carries it straight into `MotionRequest::start`, so a descend or
   * a lift re-planned mid-motion is the re-plan of `wiki/trajectory_planning.md`
   * 7 and not a second, looser one. A **tool** phase emits the five path
   * coordinates held, so it is refused from a moving arm rather than answered
   * with a first point that steps the arm's velocity to zero.
   */
  MeasuredStart start{};

  crane_model::Payload payload{};
  PayloadShape payload_shape{};

  /// `crane_msgs/PlanGrip.speed_scale`, in (0, 1]. **Not** kappa, on either kind
  /// of phase -- the two multiply and no value of this raises the margin.
  double speed_scale{1.0};

  /// The arm phases carry it to `plan_motion`; a tool phase moves no link that
  /// was not already where it is, so nothing is swept and nothing is checked.
  bool avoid_collisions{true};

  const crane_model::CollisionScene * scene{nullptr};
};

/// One phase, answered.
struct GripPlan
{
  GripPhase phase{GripPhase::Descend};
  TimedTrajectory trajectory{};

  /// True when `motion` holds the answer, false when `tool` does.
  bool arm_phase{true};

  MotionPlan motion{};  ///< the arm phases', through `plan_motion` and nothing else
  ToolPhase tool{};     ///< the tool phases', through `drive_tool_axis`

  /// Whether this phase lowered the transfer altitude's ceiling, and to what, m.
  /**
   * Carried out because it is the one thing a grip's arm phase does differently
   * from a `/crane/plan_motion` for the same goal, and a caller that cannot see
   * it cannot tell a descend from a place. Only a descend does -- see the
   * header, and `plan_grip.cpp` for the measurement behind the asymmetry.
   */
  bool altitude_ceiling_applied{false};
  double altitude_ceiling_m{};

  /// What this phase cost, against 7's budget. See `replanning.hpp`.
  std::vector<StageTiming> stages{};
};

/// Plan one grip phase, or refuse it and say which phase and why.
/**
 * `ledger` is 7's latency bound, on the same terms `plan_motion` takes it: one
 * phase is one plan and gets one budget, and a caller that has to tell an overrun
 * from a refusal reads it back afterwards.
 */
[[nodiscard]] crane_model::Result<GripPlan> plan_grip(
  const crane_model::Model & model, const PlannerContext & context, const GripRequest & request,
  LatencyLedger * ledger = nullptr);

/// The phase in one sentence, for the service response.
[[nodiscard]] std::string describe(const GripPlan & plan);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__PLAN_GRIP_HPP_
