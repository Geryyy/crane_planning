#include "crane_planning/replanning.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "crane_planning/status.hpp"

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Status;

/// Seconds, at whatever precision the platform's steady clock has.
double steady_seconds()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

bool MeasuredStart::arm_is_moving(const StartStateSettings & settings) const
{
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (std::abs(dq_a[static_cast<Eigen::Index>(row)]) > settings.at_rest_dq_a) {
      return true;
    }
  }
  return false;
}

bool MeasuredStart::tool_is_swinging(const StartStateSettings & settings) const
{
  if (!passive.measured) {
    return false;
  }
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    if (std::abs(passive.dq_u[static_cast<Eigen::Index>(row)]) > settings.at_rest_dq_u) {
      return true;
    }
  }
  return false;
}

PlanningClock steady_planning_clock()
{
  return PlanningClock(&steady_seconds);
}

LatencyLedger::LatencyLedger()
: LatencyLedger(LatencyBudgetSettings{-1.0, PlanningClock{}}) {}

LatencyLedger::LatencyLedger(const LatencyBudgetSettings & settings)
: clock_(settings.clock ? settings.clock : steady_planning_clock()),
  bounded_(std::isfinite(settings.total_s) && settings.total_s > 0.0),
  total_(bounded_ ? settings.total_s : std::numeric_limits<double>::infinity())
{
  origin_ = clock_();
  mark_ = origin_;
}

double LatencyLedger::remaining() const
{
  if (!bounded_) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, total_ - spent_);
}

crane_model::Status LatencyLedger::charge(std::string stage)
{
  // The budget this stage had when it opened, recorded before it is spent: a
  // refusal that says how long the stage took without saying how long it had
  // leaves the reader unable to tell a slow stage from a small budget.
  const double had = remaining();
  const double now = clock_();
  const double elapsed = now - mark_;
  mark_ = now;
  spent_ += elapsed;
  stages_.push_back(StageTiming{std::move(stage), elapsed, had});

  if (!bounded_ || spent_ <= total_) {
    return Status{};
  }
  if (overrun_.empty()) {
    overrun_ = stages_.back().stage;
  }
  return Status{
    ErrorCode::NotReady,
    "the latency bound of trajectory_planning 7 is enforced in the planner and not in the "
    "caller's timeout, and it fired: the " + stages_.back().stage + " stage took " +
    seconds(stages_.back().elapsed_s) + " of the " + seconds(had) +
    " it had left out of a total budget of " + seconds(total_) +
    ", so planning stopped there rather than running on under a caller that is still waiting. " +
    describe(*this) +
    ". No partial plan is emitted (7's fallback), so whatever trajectory was standing before this "
    "request is still the one standing"};
}

std::string describe(const LatencyLedger & ledger)
{
  if (ledger.stages().empty()) {
    return "No stage of this plan was charged against the latency budget";
  }
  std::string text = ledger.bounded() ?
    "Of a " + seconds(ledger.total()) + " total budget, " + seconds(ledger.spent()) +
    " was spent over " :
    "No latency budget is configured, so nothing could refuse on time; the stages were charged "
    "anyway over ";
  bool first = true;
  for (const StageTiming & stage : ledger.stages()) {
    text += first ? "" : ", ";
    text += stage.stage + " " + seconds(stage.elapsed_s);
    first = false;
  }
  return text;
}

}  // namespace crane_planning
