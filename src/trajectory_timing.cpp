#include "crane_planning/trajectory_timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Result;
using crane_model::Status;

/// The peak of s'(tau) for the cubic below, at tau = 1/2.
constexpr double kPeakRate = 1.5;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

}  // namespace

crane_model::Status check_margin_factor(double margin_factor)
{
  if (!(margin_factor > 0.0) || margin_factor > 1.0) {
    return failure(
      ErrorCode::InvalidArgument,
      "speed_scale = " + std::to_string(margin_factor) +
      " is outside (0, 1]; it is the kappa of trajectory_planning 5.5 and scales the velocity "
      "limits, so a value above one asks for more than the machine has and a value at or below "
      "zero asks for a trajectory that never arrives");
  }
  return Status{};
}

crane_model::Result<TimedTrajectory> scaled_ramp(
  const crane_model::QA & q_a_start, const crane_model::QA & q_a_goal, const JointLimits & limits,
  double margin_factor, const RampSettings & settings)
{
  if (!q_a_start.allFinite() || !q_a_goal.allFinite()) {
    return Result<TimedTrajectory>::failure(
      failure(ErrorCode::NonFiniteInput, "an endpoint of the ramp is not finite"));
  }
  {
    Status status = check_margin_factor(margin_factor);
    if (!status.ok()) {
      return Result<TimedTrajectory>::failure(std::move(status));
    }
  }
  if (!(settings.Ts > 0.0) || !(settings.min_duration > 0.0)) {
    return Result<TimedTrajectory>::failure(
      failure(ErrorCode::InvalidArgument, "Ts and min_duration must both be positive"));
  }

  const crane_model::QA travel = q_a_goal - q_a_start;

  // s(tau) = 3 tau^2 - 2 tau^3 is the lowest-order polynomial that is C1 at both
  // ends with s'(0) = s'(1) = 0, so the reference starts and ends at rest by
  // construction rather than by a clamp. Its rate peaks at 3/2, which is where
  // the duration below comes from.
  double duration = settings.min_duration;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const double distance = std::abs(travel[static_cast<Eigen::Index>(row)]);
    const double bound = margin_factor * limits.axis[row].dq_max;
    duration = std::max(duration, kPeakRate * distance / bound);
  }

  const std::size_t intervals =
    std::max<std::size_t>(1U, static_cast<std::size_t>(std::ceil(duration / settings.Ts)));
  TimedTrajectory trajectory;
  trajectory.duration = duration;
  trajectory.time_from_start.reserve(intervals + 1U);
  trajectory.q_a_ref.reserve(intervals + 1U);
  trajectory.dq_a_ref.reserve(intervals + 1U);
  for (std::size_t index = 0; index <= intervals; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(intervals);
    const double shape = fraction * fraction * (3.0 - 2.0 * fraction);
    const double rate = 6.0 * fraction * (1.0 - fraction) / duration;
    trajectory.time_from_start.push_back(fraction * duration);
    trajectory.q_a_ref.push_back(q_a_start + shape * travel);
    trajectory.dq_a_ref.push_back(rate * travel);
  }

  double limiting = 0.0;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const double peak =
      kPeakRate * std::abs(travel[static_cast<Eigen::Index>(row)]) / duration;
    limiting = std::max(limiting, peak / (margin_factor * limits.axis[row].dq_max));
  }
  trajectory.limiting_fraction = limiting;
  return Result<TimedTrajectory>::success(std::move(trajectory));
}

}  // namespace crane_planning
