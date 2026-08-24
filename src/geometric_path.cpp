#include "crane_planning/geometric_path.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <ruckig/ruckig.hpp>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Result;
using crane_model::Status;

using RuckigVector = std::array<double, kPathDof>;

/// Smallest rate ruckig may be handed as a bound. It refuses a non-positive one.
constexpr double kMinimumRate = 1.0e-9;

/// Relative room left above the prescribed boundary rates, so rounding cannot
/// push a target velocity a hair over its own bound and be refused for it.
constexpr double kRateHeadroom = 1.0 + 1.0e-6;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

/// The quintic q8 rides, on the path's own parameter.
/**
 * `wiki/trajectory_planning.md` 4.1 keeps the tool coordinate out of the path
 * variables and puts it on the same parameter anyway, so it is shaped here
 * rather than handed to the fit as a sixth degree of freedom -- a sixth degree
 * of freedom is exactly what 4.1 says it is not, and it would let q8 decide the
 * segment durations and therefore the geometry.
 *
 * The quintic is the lowest-order polynomial with `h = h' = h'' = 0` at both
 * ends, so the tool coordinate is C2 in sigma and at rest where the path is,
 * matching the five that ruckig produced without pretending to share their
 * profile. `/crane/plan_grip` (issue 044) is what actually opens and closes the
 * gripper; here q8 is normally held and this shape is then identically flat.
 */
double tool_shape(double sigma)
{
  return sigma * sigma * sigma * (10.0 + sigma * (-15.0 + 6.0 * sigma));
}

double tool_shape_rate(double sigma)
{
  const double rest = 1.0 - sigma;
  return 30.0 * sigma * sigma * rest * rest;
}

}  // namespace

PathSample GeometricPath::at(double sigma) const
{
  PathSample sample;
  sample.sigma = std::min(1.0, std::max(0.0, sigma));

  const double along = sample.sigma * span_;
  std::size_t segment = 0;
  while (segment + 1U < segments_.size() && along >= segment_starts_[segment + 1U]) {
    ++segment;
  }

  RuckigVector position{};
  RuckigVector rate{};
  RuckigVector curvature{};
  segments_[segment].at_time(along - segment_starts_[segment], position, rate, curvature);
  for (std::size_t row = 0; row < kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    sample.q_a[axis] = position[row];
    // ruckig integrated against the reference span; sigma is that span
    // normalised, so the chain rule is one constant factor per derivative.
    sample.dq_a[axis] = span_ * rate[row];
    sample.ddq_a[axis] = span_ * span_ * curvature[row];
  }

  const double travel = q8_goal_ - q8_start_;
  sample.q8 = q8_start_ + travel * tool_shape(sample.sigma);
  sample.dq8 = travel * tool_shape_rate(sample.sigma);
  return sample;
}

crane_model::Result<GeometricPath> fit_c2_path(
  const PathFitRequest & request, const JointLimits & limits, const PathFitSettings & settings)
{
  const std::size_t count = request.waypoints.size();
  if (count < 2U) {
    return Result<GeometricPath>::failure(
      failure(ErrorCode::InvalidArgument, "a path needs at least a start and an end waypoint"));
  }
  if (request.segment_names.size() + 1U != count) {
    return Result<GeometricPath>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the fit was given " + std::to_string(count) + " waypoints and " +
        std::to_string(request.segment_names.size()) +
        " segment names; a refusal has to be able to name the segment it came from, so there is "
        "one name per segment and no default"));
  }
  for (const PathVector & waypoint : request.waypoints) {
    if (!waypoint.allFinite()) {
      return Result<GeometricPath>::failure(
        failure(ErrorCode::NonFiniteInput, "a waypoint of the path is not finite"));
    }
  }
  if (!std::isfinite(request.q8_start) || !std::isfinite(request.q8_goal)) {
    return Result<GeometricPath>::failure(
      failure(ErrorCode::NonFiniteInput, "the carried tool coordinate is not finite"));
  }
  if (!(settings.rate_headroom > 1.0) || !(settings.acceleration_span > 0.0) ||
    !(settings.jerk_span > 0.0) || !(settings.min_span > 0.0) ||
    !(settings.overshoot_tolerance >= 0.0))
  {
    return Result<GeometricPath>::failure(
      failure(ErrorCode::InvalidArgument, "the path fit's shape settings are not usable"));
  }
  for (std::size_t row = 0; row < kPathDof; ++row) {
    if (!(limits.axis[row].dq_max > 0.0)) {
      return Result<GeometricPath>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "path coordinate " + std::to_string(row) +
          " has no positive velocity limit, so there is nothing to distribute sigma by"));
    }
  }

  const std::size_t segments = count - 1U;

  // Each segment's share of sigma, in the one unit that is a property of the
  // machine rather than of this file: how long the slowest of the five
  // coordinates would take over that chord at its own velocity limit. It is a
  // *ratio*, not a duration -- see the header -- and its only job is that a
  // segment which moves a slow axis a long way is given more sigma than one that
  // does not, so stage 2 (issue 043) has less to undo.
  std::vector<double> reference_span(segments, 0.0);
  std::vector<PathVector> chord(segments, PathVector::Zero());
  for (std::size_t segment = 0; segment < segments; ++segment) {
    chord[segment] = request.waypoints[segment + 1U] - request.waypoints[segment];
    double span = 0.0;
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const double travel = std::abs(chord[segment][static_cast<Eigen::Index>(row)]);
      span = std::max(span, travel / limits.axis[row].dq_max);
    }
    reference_span[segment] = std::max(span, settings.min_span);
  }

  // The boundary conditions, and the whole of the C2 claim. The two ends are at
  // rest; every interior waypoint gets a velocity and a zero acceleration that
  // *both* adjacent segments are solved against, so the junction agrees in
  // position, in q_a' and in q_a'' because the same three numbers were imposed
  // from either side.
  //
  // The interior velocity is the slower of the two chords the waypoint joins,
  // and zero on a coordinate whose two chords disagree in sign -- Fritsch and
  // Carlson's monotone-interpolation rule at its most conservative. Anything
  // faster than the slower chord obliges one side to overshoot the waypoint and
  // come back, which is the non-monotone path the acceptance criteria rule out.
  std::vector<PathVector> junction_rate(count, PathVector::Zero());
  for (std::size_t waypoint = 1U; waypoint + 1U < count; ++waypoint) {
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      const double before = chord[waypoint - 1U][axis] / reference_span[waypoint - 1U];
      const double after = chord[waypoint][axis] / reference_span[waypoint];
      if (before * after <= 0.0) {
        junction_rate[waypoint][axis] = 0.0;
        continue;
      }
      const double magnitude = std::min(std::abs(before), std::abs(after));
      junction_rate[waypoint][axis] = before > 0.0 ? magnitude : -magnitude;
    }
  }

  GeometricPath path;
  path.segment_names_ = request.segment_names;
  path.waypoints_ = request.waypoints;
  path.q8_start_ = request.q8_start;
  path.q8_goal_ = request.q8_goal;
  path.segments_.reserve(segments);
  path.segment_starts_.reserve(segments + 1U);

  ruckig::Ruckig<kPathDof> solver;
  double accumulated = 0.0;
  for (std::size_t segment = 0; segment < segments; ++segment) {
    const double span = reference_span[segment];
    ruckig::InputParameter<kPathDof> input;
    input.control_interface = ruckig::ControlInterface::Position;
    input.synchronization = ruckig::Synchronization::Time;
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      input.current_position[row] = request.waypoints[segment][axis];
      input.current_velocity[row] = junction_rate[segment][axis];
      input.current_acceleration[row] = 0.0;
      input.target_position[row] = request.waypoints[segment + 1U][axis];
      input.target_velocity[row] = junction_rate[segment + 1U][axis];
      input.target_acceleration[row] = 0.0;

      // The shape bounds. `rate_headroom` above the chord's mean rate is what
      // gives the profile a plateau instead of a triangle, and the two spans
      // put the acceleration and jerk bounds in the segment's own scale so the
      // profile looks the same on a long segment and a short one.
      const double mean = std::abs(chord[segment][axis]) / span;
      const double peak = std::max(
        {mean * settings.rate_headroom, std::abs(input.current_velocity[row]),
          std::abs(input.target_velocity[row])});
      input.max_velocity[row] = kRateHeadroom * peak + kMinimumRate;
      input.max_acceleration[row] = settings.acceleration_span * input.max_velocity[row] / span;
      input.max_jerk[row] = settings.jerk_span * input.max_acceleration[row] / span;
    }
    // The bounds above never bind, so this is what actually sets the segment's
    // share of sigma: ruckig stretches the profile to it.
    input.minimum_duration = span;

    ruckig::Trajectory<kPathDof> trajectory;
    const ruckig::Result outcome = solver.calculate(input, trajectory);
    if (outcome != ruckig::Result::Working) {
      return Result<GeometricPath>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the " + request.segment_names[segment] +
          " segment could not be interpolated: ruckig returned " + std::to_string(outcome) +
          " for the boundary-value problem it was given (libraries.md 1 lists ruckig for this "
          "solve, so a local spline basis is not the answer -- the boundary conditions are)"));
    }
    if (!(trajectory.get_duration() > 0.0) || !std::isfinite(trajectory.get_duration())) {
      return Result<GeometricPath>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the " + request.segment_names[segment] + " segment came back with no extent in sigma"));
    }

    // Monotone in sigma, and inside the joint range, checked on ruckig's own
    // exact extrema rather than on a sampling of the profile.
    const std::array<ruckig::PositionExtrema, kPathDof> extrema =
      trajectory.get_position_extrema();
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      const double from = request.waypoints[segment][axis];
      const double to = request.waypoints[segment + 1U][axis];
      const double lower = std::min(from, to) - settings.overshoot_tolerance;
      const double upper = std::max(from, to) + settings.overshoot_tolerance;
      if (extrema[row].min < lower || extrema[row].max > upper) {
        return Result<GeometricPath>::failure(
          failure(
            ErrorCode::InvalidArgument,
            "the " + request.segment_names[segment] + " segment is not monotone in sigma: path "
            "coordinate " + std::to_string(row) + " runs to [" + std::to_string(extrema[row].min) +
            ", " + std::to_string(extrema[row].max) + "] between waypoints at " +
            std::to_string(from) + " and " + std::to_string(to)));
      }
      if (limits.axis[row].bounded &&
        (extrema[row].min < limits.axis[row].lower || extrema[row].max > limits.axis[row].upper))
      {
        return Result<GeometricPath>::failure(
          failure(
            ErrorCode::InvalidArgument,
            "the " + request.segment_names[segment] + " segment leaves the joint range the "
            "description gives path coordinate " + std::to_string(row) + ": it reaches [" +
            std::to_string(extrema[row].min) + ", " + std::to_string(extrema[row].max) +
            "] against [" + std::to_string(limits.axis[row].lower) + ", " +
            std::to_string(limits.axis[row].upper) + "]"));
      }
    }

    path.segment_starts_.push_back(accumulated);
    accumulated += trajectory.get_duration();
    path.segments_.push_back(std::move(trajectory));
  }
  path.segment_starts_.push_back(accumulated);
  path.span_ = accumulated;

  path.junction_sigmas_.reserve(segments - 1U);
  for (std::size_t segment = 1U; segment < segments; ++segment) {
    path.junction_sigmas_.push_back(path.segment_starts_[segment] / path.span_);
  }

  return Result<GeometricPath>::success(std::move(path));
}

}  // namespace crane_planning
