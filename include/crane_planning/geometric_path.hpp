// The geometric path of `wiki/trajectory_planning.md` 4, as the object stage 2
// consumes: `q_a(sigma)` over the five path coordinates of 4.1, with
// `q_a'(sigma)` and `q_a''(sigma)` defined at **every** sigma in [0, 1].
//
// # Why the derivatives are the point, and not a convenience
//
// 4.5 is blunt about it. Stage 2 (5) writes
//
//   dq_a/dt = q_a' sigma_dot,   ddq_a/dt2 = q_a'' sigma_dot^2 + q_a' sigma_ddot
//
// so a path whose q_a' jumps has no q_a'' at the jump, and the acceleration
// limit there reads as `|q_a''| sigma_dot^2 <= ddq_a^max` with an unbounded
// `q_a''`: the path-velocity limit collapses to `sigma_dot = 0` and the machine
// **stops dead at every waypoint**. 4.5 calls that a behavioural failure rather
// than a numerical nuisance, and for a crane whose purpose is smooth, sway-free
// motion it is the worst available output.
//
// So this is built C2 rather than smoothed into C2 afterwards: the fit below
// solves each segment as a **boundary-value problem** whose end velocity and end
// acceleration are prescribed, and the next segment starts from exactly those
// numbers. Continuity is then a property of the construction and not of a
// tolerance. There is no corner-rounding pass anywhere in this package.
//
// # ruckig does the interpolation and the boundary-condition solve
//
// `wiki/implementation/libraries.md` 1 lists **ruckig** for exactly this --
// "jerk-limited interpolation and the clamped B-spline fit", against "spline
// bases, boundary-condition solves" -- so no spline basis is written here. A
// jerk-limited profile has continuous acceleration by definition, which is what
// makes `q_a''` defined *everywhere* rather than only away from the junctions.
//
// # sigma is not time, and this file must not be read as if it were
//
// ruckig's independent variable is called time and its bounds are called
// velocity, acceleration and jerk. Here they are none of those things: the
// independent variable is the path parameter, and the bounds are a **shape**
// choice that decides how sigma is distributed along the path. Nothing in this
// file establishes that the machine can traverse the result at any speed; the
// force, the pump flow, the sway and the kappa margin all arrive with the
// path-constrained OCP of 5.2, which is issue 043. The one machine number that
// does enter is the *ratio* of the joint velocity limits, and it enters as a
// parametrisation choice -- a slow axis is given more sigma than a fast one, so
// stage 2 has less to undo -- not as a feasibility claim.

#ifndef CRANE_PLANNING__GEOMETRIC_PATH_HPP_
#define CRANE_PLANNING__GEOMETRIC_PATH_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

#include <ruckig/trajectory.hpp>

#include "crane_model/model.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// The five path coordinates of `wiki/trajectory_planning.md` 4.1, as a vector.
/**
 * `q_a = (q1, q2, q3, q4, q7)`. q8 is not in here on purpose: 4.1 says in as
 * many words that the tool coordinate is not a path variable. It rides the same
 * sigma and is carried on `PathSample` beside these five.
 */
using PathVector = Eigen::Matrix<double, kPathDof, 1>;

/// The path and its first two derivatives at one sigma.
struct PathSample
{
  double sigma{};
  PathVector q_a{PathVector::Zero()};      ///< q_a(sigma)
  PathVector dq_a{PathVector::Zero()};     ///< q_a'(sigma), per unit sigma
  PathVector ddq_a{PathVector::Zero()};    ///< q_a''(sigma), per unit sigma squared
  double q8{};                             ///< carried, not a path variable
  double dq8{};                            ///< dq8/dsigma
  double ddq8{};                           ///< d2q8/dsigma2, for the same reason `ddq_a` is here
};

/// What the fit is asked to interpolate, and what its refusals may call things.
/**
 * `segment_names` is one name per segment, i.e. one fewer than `waypoints`. The
 * fit has no idea what a lift or a descend is; naming the segments here is what
 * lets a refusal say which of the primitive's three phases failed without this
 * file knowing about phases at all.
 */
struct PathFitRequest
{
  std::vector<PathVector> waypoints{};       ///< at least two, start first
  std::vector<std::string> segment_names{};  ///< one per segment
  double q8_start{};
  double q8_goal{};

  /// The **measured** actuated rate at the start, rad/s or m/s. Zero is at rest.
  /**
   * `wiki/trajectory_planning.md` 7 and the whole of issue 045: a path whose
   * start is met at rest can carry no start velocity at all, because stage 2
   * writes `dq_a = q_a'(sigma) sigma_dot` and a zero `q_a'(0)` makes that zero
   * whatever the path rate is. So a re-plan from a moving machine has to arrive
   * as a **boundary condition on the geometry** and not as a correction after it.
   *
   * It is passed in the machine's own units and used as ruckig's `current_velocity`
   * for the first segment, which is dimensionally the unit that segment's shape
   * bounds are already in -- the reference span is `max_i |chord_i| / dq_i^max`,
   * so the mean rate the bounds are built from is itself of the scale of
   * `dq^max`. What comes back is `q_a'(0) = reference_span() * start_rate`, and
   * the one path rate that reproduces the measurement is then
   * `sigma_dot(0) = 1 / reference_span()`. `start_path_rate` is that number,
   * computed from the fitted path rather than from this argument, so a caller
   * pins the OCP's initial condition on what the fit actually produced.
   *
   * Leaving it zero is the stopped start this package had before, and every path
   * built that way meets both ends with `q_a' = q_a'' = 0` exactly as before.
   */
  PathVector start_rate{PathVector::Zero()};
};

/// The path rate at which a fitted path's own start slope reproduces a measured rate.
/**
 * `dq_a = q_a'(0) sigma_dot`, solved for `sigma_dot` in the least-squares sense
 * over the five path coordinates. It is exact rather than approximate whenever
 * the fit was given `start_rate` -- the two vectors are then parallel by
 * construction -- and `residual` is what says so: it is the norm of
 * `q_a'(0) sigma_dot - dq_a`, and a caller that finds it above its own tolerance
 * has a path whose start direction does not carry the measurement, which is a
 * refusal and not a rounding.
 *
 * `defined` is false when the path meets its start at rest, i.e. when
 * `|q_a'(0)|` is below `floor`. There is then no rate that reproduces anything
 * but a zero velocity, which is the right answer for a machine standing still and
 * a refusal for one that is not.
 */
struct StartPathRate
{
  bool defined{false};
  double sigma_rate{};
  double residual{};
  double slope_norm{};
};

/// How sigma is distributed along the path, and how rounded the profile is.
/**
 * These are shape numbers, not machine limits — see the header comment. They are
 * expressed relative to each segment's own reference span so that the shape of
 * the profile is the same on a segment that travels a metre and on one that
 * travels a millimetre.
 */
struct PathFitSettings
{
  /// Peak `|q_a'|` a segment is allowed, as a multiple of its mean rate.
  /**
   * Two gives the profile a genuine plateau: the mean rate is the chord over the
   * span, so a peak of twice that is reached by accelerating over roughly a
   * quarter of the segment, cruising, and decelerating again.
   */
  double rate_headroom{2.0};

  /// Segment spans the acceleration bound is expressed in, i.e. `a = k v / T`.
  double acceleration_span{4.0};

  /// Segment spans the jerk bound is expressed in, i.e. `j = k a / T`.
  double jerk_span{4.0};

  /// Shortest reference span a segment may be given, in the same units as `T`.
  /**
   * A segment whose waypoints coincide has a zero chord and therefore a zero
   * reference span, which ruckig cannot be asked for. It is given this instead,
   * so a degenerate segment is a very short one rather than a refusal — the
   * primitive of `structured_primitive.hpp` produces one whenever the start is
   * already at the transfer altitude.
   */
  double min_span{1.0e-3};

  /// How far outside its two waypoints a coordinate may stray, m or rad.
  /**
   * The monotonicity test of the acceptance criteria, enforced rather than
   * asserted: ruckig reports each segment's true position extrema, so overshoot
   * is checked exactly and not sampled.
   */
  double overshoot_tolerance{1.0e-9};
};

/// A `C2` path in sigma, built segment by segment and continuous across all of them.
/**
 * Construct it with `fit_c2_path`. The object owns one ruckig trajectory per
 * segment and the sigma each segment starts at; `at()` is a lookup and an
 * evaluation, with no solve in it, so stage 2 may call it as often as it likes.
 */
class GeometricPath
{
public:
  /// Evaluate the path. `sigma` outside [0, 1] is clamped to the endpoints.
  [[nodiscard]] PathSample at(double sigma) const;

  [[nodiscard]] std::size_t segment_count() const noexcept {return segments_.size();}

  /// The sigmas the interior junctions sit at, in order — `segment_count() - 1` of them.
  [[nodiscard]] const std::vector<double> & junction_sigmas() const noexcept
  {
    return junction_sigmas_;
  }

  /// The names the fit was given, one per segment.
  [[nodiscard]] const std::vector<std::string> & segment_names() const noexcept
  {
    return segment_names_;
  }

  /// The waypoints the path interpolates, start first.
  [[nodiscard]] const std::vector<PathVector> & waypoints() const noexcept {return waypoints_;}

  /// The reference span the fit summed to normalise sigma. Not a duration.
  [[nodiscard]] double reference_span() const noexcept {return span_;}

private:
  friend crane_model::Result<GeometricPath> fit_c2_path(
    const PathFitRequest & request, const JointLimits & limits,
    const PathFitSettings & settings);

  std::vector<ruckig::Trajectory<kPathDof>> segments_{};
  std::vector<double> segment_starts_{};   ///< cumulative span at each segment's start
  std::vector<double> junction_sigmas_{};
  std::vector<std::string> segment_names_{};
  std::vector<PathVector> waypoints_{};
  double span_{};
  double q8_start_{};
  double q8_goal_{};
};

/// Interpolate the waypoints C2, or refuse and name the segment that failed.
/**
 * Every interior waypoint is met with a prescribed velocity and a **zero**
 * acceleration, so `q_a''` agrees across the junction by construction — both
 * sides are zero there — and the path has an inflection at the waypoint rather
 * than a kink. The two ends are met at rest, `q_a' = q_a'' = 0`, which is the
 * "start and end are at rest" of the acceptance criteria and is also what stage
 * 2 needs if it is to start and finish with `sigma_dot = 0`.
 *
 * The interior velocity is the slower of the two chords the waypoint joins, and
 * zero on any coordinate whose two chords disagree in sign. That is the
 * monotone-interpolation rule of Fritsch and Carlson in its most conservative
 * form, and the reason for it is that a waypoint velocity faster than either
 * chord forces one side or the other to overshoot and come back — which would
 * make the path non-monotone in sigma. Whether it worked is not assumed: ruckig
 * reports the exact position extrema of each segment and the fit refuses if any
 * coordinate leaves the box its two waypoints span.
 *
 * `limits` enters twice and both times as the description's own numbers: the
 * ratio of `dq_max` sets each segment's reference span, and the position bounds
 * are checked against the segment extrema so that a path that leaves the joint
 * range is refused rather than emitted.
 */
[[nodiscard]] crane_model::Result<GeometricPath> fit_c2_path(
  const PathFitRequest & request, const JointLimits & limits,
  const PathFitSettings & settings);

/// See `StartPathRate`. `floor` is the `|q_a'(0)|` below which the start is at rest.
[[nodiscard]] StartPathRate start_path_rate(
  const GeometricPath & path, const PathVector & dq_a_start, double floor = 1.0e-9);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__GEOMETRIC_PATH_HPP_
