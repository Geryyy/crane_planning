// The one degree of freedom `wiki/robot_model.md` 2.2 step 3 leaves over, and
// the single place the rule for spending it is written down.
//
// Step 3 is one sentence: three planar unknowns (q2, q3, q4) for two planar
// coordinates leaves one degree of freedom, "resolved by a scalar search over
// d45, scored by joint-range centring and collision clearance". Both endpoint
// routes in this package meet that same leftover freedom and they must spend it
// the same way -- the semi-analytic solve of `inverse_kinematics.hpp` evaluates
// this score over a grid of extensions and keeps the best, the
// equilibrium-constrained NLP of `equilibrium_ik.hpp` carries it as residual
// rows its Gauss-Newton step minimises inside the null space of the task. Two
// copies of a redundancy rule are two machines that drift apart, so there is
// one copy and it is here.
//
// # Clearance is the other half of that sentence, and it is here now
//
// The scene, the truck bed and the runges arrived with issue 041, so the second
// term of step 3's score is a real one: a candidate extension is scored by how
// close the whole crane comes to anything -- the scene, the truck model and
// itself -- at the configuration that extension closes on.
//
// **Which of the two routes evaluates it is not symmetric, and that is
// deliberate.** The scalar search of `inverse_kinematics.cpp` is where the
// redundancy is *resolved*, so that is where clearance votes: one
// `Model::collision_query` per candidate extension, once per solve. The
// equilibrium-constrained NLP of `equilibrium_ik.cpp` only *holds* the freedom
// the seed already resolved -- measured, it moves the centring score by less
// than a part in a million -- so a clearance row there would buy no decision and
// would cost a collision query per finite-difference probe, on a residual that
// is not smooth where a witness pair changes.

#ifndef CRANE_PLANNING__REDUNDANCY_HPP_
#define CRANE_PLANNING__REDUNDANCY_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "crane_model/model.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// The axes of `JointLimits` that 2.2 step 3's redundancy actually spans.
/**
 * The boom, the arm and the telescope -- q2, q3, q4, which are also rows 1, 2
 * and 3 of the canonical eight because `kActuatedRows` is the identity over its
 * first four entries.
 *
 * The slewing joint is deliberately not among them. 2.2 step 1 *answers* q1
 * from the goal's azimuth, so it is determined rather than free, and scoring it
 * would pull the slewing joint towards the middle of its own range for a reason
 * 2.2 does not give.
 */
inline constexpr std::array<std::size_t, 3> kRedundantAxes{{1, 2, 3}};

/// How the leftover freedom is scored -- one weight per term of 2.2 step 3.
struct RedundancyWeights
{
  double centring{1.0};   ///< joint-range centring
  double clearance{1.0};  ///< collision clearance, against `clearance_reference_m`

  /// The clearance above which an extension is as good as any other, m.
  /**
   * Not a safety margin -- the safety answer is `check_path`, which refuses. It
   * is where the *preference* stops: two extensions that both clear everything
   * by more than this are equally good on clearance and are then separated by
   * their centring alone, which is what keeps the score from steering the
   * telescope around obstacles it is nowhere near.
   */
  double clearance_reference_m{0.5};
};

/// Penalty on a candidate that comes closer than the reference clearance.
/**
 * Zero at and above the reference, rising to one at contact and past it inside,
 * so it is the same order as the centring term over the range where the two
 * trade against each other. A candidate that is actually **in** collision loses
 * to any candidate that is not, whatever its centring: that is what the constant
 * below buys, and it is what stops the search picking a beautifully centred
 * extension that puts the telescope through a runge.
 */
inline constexpr double kCollidingPenalty = 1.0e3;

[[nodiscard]] inline double clearance_penalty(double distance_m, double reference_m) noexcept
{
  if (!std::isfinite(distance_m) || !(reference_m > 0.0) || distance_m >= reference_m) {
    return 0.0;
  }
  const double shortfall = (reference_m - distance_m) / reference_m;
  const double penalty = shortfall * shortfall;
  return (distance_m < 0.0) ? kCollidingPenalty + penalty : penalty;
}

/// One axis' distance from the middle of its own range, in half-spans.
/**
 * Zero for an axis the description leaves unbounded -- the rotator on both
 * machines here -- because there is no range for it to be centred in, and a
 * made-up one would steer the redundancy without saying so.
 */
[[nodiscard]] inline double centring_offset(const AxisLimit & axis, double value) noexcept
{
  if (!axis.bounded) {
    return 0.0;
  }
  return (value - axis.centre()) / axis.half_span();
}

/// The joint-range centring of 2.2 step 3, as a sum of squared normalised offsets.
template<std::size_t N>
[[nodiscard]] double centring_score(
  const JointLimits & limits, const std::array<std::size_t, N> & axes,
  const std::array<double, N> & value) noexcept
{
  double score = 0.0;
  for (std::size_t index = 0; index < N; ++index) {
    const double offset = centring_offset(limits.axis[axes[index]], value[index]);
    score += offset * offset;
  }
  return score;
}

/// Refuse a score that cannot be evaluated as written.
[[nodiscard]] inline crane_model::Status check_redundancy_weights(
  const RedundancyWeights & weights)
{
  if (!(weights.centring >= 0.0)) {
    return crane_model::Status{
      crane_model::ErrorCode::InvalidArgument,
      "the joint-range centring weight of robot_model 2.2 step 3 must be finite and "
      "non-negative"};
  }
  if (!(weights.clearance >= 0.0)) {
    return crane_model::Status{
      crane_model::ErrorCode::InvalidArgument,
      "the collision-clearance weight of robot_model 2.2 step 3 must be finite and non-negative"};
  }
  if (weights.clearance > 0.0 && !(weights.clearance_reference_m > 0.0)) {
    return crane_model::Status{
      crane_model::ErrorCode::InvalidArgument,
      "a clearance weight needs a positive reference clearance to be measured against, or every "
      "candidate scores the same and the second half of 2.2 step 3's score votes for nothing"};
  }
  return crane_model::Status{};
}

}  // namespace crane_planning

#endif  // CRANE_PLANNING__REDUNDANCY_HPP_
