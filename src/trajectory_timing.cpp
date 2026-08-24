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

crane_model::Result<TimedTrajectory> scaled_ramp_along_path(
  const GeometricPath & path, const JointLimits & limits, double margin_factor,
  const RampSettings & settings)
{
  if (path.segment_count() == 0U) {
    return Result<TimedTrajectory>::failure(
      failure(ErrorCode::InvalidArgument, "the path has no segments to run a ramp along"));
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

  // The peak of |q_a'| over the path, per actuated coordinate. It is probed
  // rather than known in closed form because the path is three ruckig profiles
  // and the peak may sit anywhere in any of them; the grid is fixed so the same
  // path always gets the same duration.
  crane_model::QA peak_rate = crane_model::QA::Zero();
  for (std::size_t index = 0; index < kPathRateSamples; ++index) {
    const double sigma =
      static_cast<double>(index) / static_cast<double>(kPathRateSamples - 1U);
    const PathSample sample = path.at(sigma);
    if (!sample.q_a.allFinite() || !sample.dq_a.allFinite() || !std::isfinite(sample.dq8)) {
      return Result<TimedTrajectory>::failure(
        failure(
          ErrorCode::NonFiniteInput,
          "the path is not finite at sigma = " + std::to_string(sigma)));
    }
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      peak_rate[axis] = std::max(peak_rate[axis], std::abs(sample.dq_a[axis]));
    }
    const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
    peak_rate[tool] = std::max(peak_rate[tool], std::abs(sample.dq8));
  }

  double duration = settings.min_duration;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const double bound = margin_factor * limits.axis[row].dq_max;
    duration =
      std::max(duration, kPeakRate * peak_rate[static_cast<Eigen::Index>(row)] / bound);
  }

  const std::size_t intervals =
    std::max<std::size_t>(1U, static_cast<std::size_t>(std::ceil(duration / settings.Ts)));
  TimedTrajectory trajectory;
  trajectory.duration = duration;
  trajectory.time_from_start.reserve(intervals + 1U);
  trajectory.q_a_ref.reserve(intervals + 1U);
  trajectory.dq_a_ref.reserve(intervals + 1U);
  double limiting = 0.0;
  for (std::size_t index = 0; index <= intervals; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(intervals);
    // The same s(tau) the chord ramp uses, so sigma_dot vanishes at both ends and
    // the reference is at rest there whatever the path does.
    const double sigma = fraction * fraction * (3.0 - 2.0 * fraction);
    const double sigma_rate = 6.0 * fraction * (1.0 - fraction) / duration;
    const PathSample sample = path.at(sigma);

    crane_model::QA q_a = crane_model::QA::Zero();
    crane_model::DQA dq_a = crane_model::DQA::Zero();
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      q_a[axis] = sample.q_a[axis];
      dq_a[axis] = sample.dq_a[axis] * sigma_rate;
    }
    const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
    q_a[tool] = sample.q8;
    dq_a[tool] = sample.dq8 * sigma_rate;

    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      const double bound = margin_factor * limits.axis[row].dq_max;
      limiting = std::max(limiting, std::abs(dq_a[static_cast<Eigen::Index>(row)]) / bound);
    }

    trajectory.time_from_start.push_back(fraction * duration);
    trajectory.q_a_ref.push_back(q_a);
    trajectory.dq_a_ref.push_back(dq_a);
  }
  trajectory.limiting_fraction = limiting;
  return Result<TimedTrajectory>::success(std::move(trajectory));
}

}  // namespace crane_planning
