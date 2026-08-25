// Replanning from a moving, swinging state, and the latency bound of
// `wiki/trajectory_planning.md` 7 enforced **inside** the planner.
//
// # The stopped-start convention is the thing being removed
//
// 7's `[!warning]` names it: the legacy stack zeroes the initial velocity
// outright -- *"we deactivated qDot0, because we now always start in a stopped
// state"* -- so the whole pipeline only works as stop, measure, replan. A re-plan
// issued while the tool is still moving then mis-predicts the sway from the first
// step, which is exactly the situation a stall-recovery re-plan occurs in.
//
// `MeasuredStart` below is what replaces it. It is `(q, dq)` **as measured** over
// all eight coordinates: the six actuated positions and rates off `/joint_states`
// and the passive pair off `/crane/pendulum_state`. Nothing in it is defaulted
// silently -- `passive_measured` says whether the passive half is a measurement
// at all, and `passive_note` says why when it is not, because
// `wiki/control_architecture.md` 5.3's rule is that no input may stop arriving
// without a defined consequence and "never connected" and "died" are different
// things for an operator to chase.
//
// # The three consequences 7 draws, and where each of them lives
//
//   * **Continuity by construction.** A concatenated segment must match position
//     *and* velocity at the seam. That is a boundary condition on the geometry
//     (`PathFitRequest::start_rate`) and on the timing (`TimingOcpStart`), not a
//     tolerance checked afterwards.
//   * **Bounded latency**, enforced here rather than in a client timeout. That is
//     `LatencyLedger`: one budget over the endpoint IK, the geometric path, the
//     smoothing and the OCP, charged stage by stage, and a stage that spends the
//     rest of it stops the plan naming itself.
//   * **Fallback.** A plan that overruns keeps the previous trajectory and
//     reports it; it never emits a partial plan. The trajectory that is still
//     standing is the node's (`planner_node.hpp`), and this file is what tells it
//     that the budget rather than the machine is what refused.
//
// # Why the clock is a parameter
//
// A budget that can only be tested by sleeping is a budget whose test is a race.
// `PlanningClock` is a plain `double()` returning monotonic seconds, defaulted to
// `steady_planning_clock()` and replaced in the tests by a counter that advances a
// fixed amount per reading -- so "the budget fires on the stage that overran"
// is asserted deterministically and without a single wall-clock sleep, which is
// what the acceptance criteria ask for.

#ifndef CRANE_PLANNING__REPLANNING_HPP_
#define CRANE_PLANNING__REPLANNING_HPP_

#include <functional>
#include <string>
#include <vector>

#include "crane_model/model.hpp"

namespace crane_planning
{

/// Below which a measured rate is a standstill rather than a motion.
/**
 * Two numbers and not one, because the two halves of the state are measured by
 * different hardware. The actuated rates come off the encoders through
 * `/joint_states`; the passive pair is differenced from a gyro pair whose
 * quantiser is `2^-9` rad/s (measured from the recordings, see the workspace
 * `CLAUDE.md`), so a sway rate below that is quantisation and not sway.
 *
 * These decide one thing only: whether this plan is a re-plan from a *moving*
 * state at all. Below them the start is 7's stopped one and the plan is the plan
 * this package already built; above them the boundary conditions of
 * `TimingOcpStart` carry the measurement.
 */
struct StartStateSettings
{
  double at_rest_dq_a{1.0e-4};   ///< rad/s or m/s, per actuated coordinate
  double at_rest_dq_u{2.0e-3};   ///< rad/s, the gyro quantiser
};

/// The passive pair at the start, and whether it is a measurement.
/**
 * `/crane/pendulum_state` carries position, velocity, covariances and a `valid`
 * flag at 100 Hz, and issue 019 gave it staleness detection. Whether the estimate
 * that arrived is usable is decided at the node, against the topic's own 150 ms
 * deadline of `control_architecture` 5.3 and against the `valid` flag the
 * producer sets -- the three causes behind that flag are the producer's to judge
 * inside its own cycle, and a consumer carries its `status` string rather than
 * restating it.
 *
 * What reaches the algorithm is this: either a measurement, or a stated absence
 * with a bound reserved for the sway that may be there anyway. It is never a
 * silent zero.
 */
struct PassiveStart
{
  bool measured{false};
  crane_model::QU q_u{crane_model::QU::Zero()};
  crane_model::DQU dq_u{crane_model::DQU::Zero()};

  /// Where it came from, or why it is not a measurement. Never empty in the node.
  std::string note{};

  /// The sway allowance held back when the estimate is not a measurement, rad.
  /**
   * The conservative bound of the acceptance criteria, and it is a *reservation*
   * rather than an assumption: the plan is solved with `q_u_max` and `dq_u_max`
   * reduced by these, so whatever sway is actually there -- up to the bound --
   * still fits inside what `mpc.md` 3 constraint 3 permits and inside the
   * envelope `trajectory_planning` 4.3 cleared the path over. Zero means the
   * deployment asked for the request to be refused instead, which is the other
   * half of the same rule.
   */
  double sway_reserve_rad{0.0};
  double sway_rate_reserve{0.0};
};

/// Where the machine is **and what it is doing**, as 7 asks for it.
struct MeasuredStart
{
  crane_model::QA q_a{crane_model::QA::Zero()};
  crane_model::DQA dq_a{crane_model::DQA::Zero()};
  PassiveStart passive{};

  /// Whether any actuated coordinate is moving faster than `at_rest_dq_a`.
  [[nodiscard]] bool arm_is_moving(const StartStateSettings & settings) const;

  /// Whether the tool is swinging faster than `at_rest_dq_u`. False when the
  /// passive half is not a measurement -- an unknown sway is not a still one, and
  /// the caller of this is the branch that has already said which.
  [[nodiscard]] bool tool_is_swinging(const StartStateSettings & settings) const;
};

/// Monotonic seconds. A parameter so the budget is testable without sleeping.
using PlanningClock = std::function<double ()>;

/// `std::chrono::steady_clock`, as seconds since an arbitrary origin.
[[nodiscard]] PlanningClock steady_planning_clock();

/// 7's bounded latency, as one total over every stage of one plan.
struct LatencyBudgetSettings
{
  /// The whole of it, seconds, covering IK, path, smoothing and the OCP.
  /**
   * Sized against what a plan actually costs rather than round: one plan is some
   * 14 s here, of which the OCP's own `max_wall_clock` share is 12 s and the
   * solve itself is 4 s, so this is that with room for a slower machine. A bound
   * is only a bound if the ordinary case clears it somewhere else too.
   *
   * A value at or below zero is **no budget**, and it is spelled that way rather
   * than with a second flag: a deployment that has not chosen a latency bound
   * says so by not giving one, and every stage is then charged and reported
   * without any of them being able to refuse.
   */
  double total_s{45.0};

  /// Empty means `steady_planning_clock()`. Replaced in the tests, nowhere else.
  PlanningClock clock{};
};

/// One stage of one plan, and what it had left when it started.
struct StageTiming
{
  std::string stage;
  double elapsed_s{};
  double had_s{};  ///< the budget remaining when this stage opened; infinite if none
};

/// The latency bound of 7, charged stage by stage.
/**
 * Not thread safe and not meant to be: one ledger belongs to one plan, is created
 * where the request is and dies with the answer. `charge` closes the stage that
 * has been running since the previous `charge` (or since construction), records
 * it, and returns a failure naming that stage when the total is spent.
 *
 * **It is the planner's bound and not the caller's.** 7 is explicit that bounding
 * only how long the caller waits leaves a slow solve returning a failure to a
 * caller whose request is still running; the point of charging here is that the
 * plan stops, no partial answer is built, and the previous trajectory is still
 * the one standing.
 */
class LatencyLedger
{
public:
  /// Unbounded: every stage is charged and reported, and none can refuse.
  LatencyLedger();

  explicit LatencyLedger(const LatencyBudgetSettings & settings);

  /// Close the running stage and charge it. A failure names it and its budget.
  [[nodiscard]] crane_model::Status charge(std::string stage);

  /// What is left of the total, seconds. Infinite when there is no budget.
  [[nodiscard]] double remaining() const;

  [[nodiscard]] bool bounded() const noexcept {return bounded_;}
  [[nodiscard]] double total() const noexcept {return total_;}
  [[nodiscard]] double spent() const noexcept {return spent_;}
  [[nodiscard]] const std::vector<StageTiming> & stages() const noexcept {return stages_;}

  /// The stage that spent the budget, empty while none has.
  [[nodiscard]] const std::string & overrun() const noexcept {return overrun_;}

private:
  PlanningClock clock_;
  bool bounded_{false};
  double total_{};
  double origin_{};
  double mark_{};
  double spent_{};
  std::string overrun_{};
  std::vector<StageTiming> stages_{};
};

/// Every charged stage in one sentence, for the service response.
[[nodiscard]] std::string describe(const LatencyLedger & ledger);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__REPLANNING_HPP_
