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
// # Clearance is the other half of that sentence, and it is not here
//
// This package checks no collision at all; the scene, the truck bed and the
// runges arrive with issue 041. Clearance still has a *weight* rather than no
// mention at all, because a score whose shape a caller cannot see is a score
// nobody can add a term to later. A non-zero clearance weight is **refused**, so
// that the day the scene arrives the refusal points at issue 041 instead of the
// weight being read and silently ignored -- which is the same rule the rest of
// this package follows for an absence.

#ifndef CRANE_PLANNING__REDUNDANCY_HPP_
#define CRANE_PLANNING__REDUNDANCY_HPP_

#include <array>
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
  double centring{1.0};   ///< joint-range centring; the only term there is today
  double clearance{0.0};  ///< collision clearance, issue 041. Non-zero is refused.
};

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

/// Refuse a score this package cannot actually evaluate.
[[nodiscard]] inline crane_model::Status check_redundancy_weights(
  const RedundancyWeights & weights)
{
  if (!(weights.centring >= 0.0)) {
    return crane_model::Status{
      crane_model::ErrorCode::InvalidArgument,
      "the joint-range centring weight of robot_model 2.2 step 3 must be finite and "
      "non-negative"};
  }
  if (weights.clearance != 0.0) {
    return crane_model::Status{
      crane_model::ErrorCode::InvalidArgument,
      "a non-zero collision-clearance weight is refused: robot_model 2.2 step 3 scores the "
      "telescope redundancy by joint-range centring *and* clearance, and this planner has no "
      "clearance to score -- it subscribes to no /crane/collision_scene and knows nothing about "
      "the truck bed or the runges. Clearance joins this score at issue 041. Leave the weight at "
      "zero to ask for the centring-only redundancy this can actually resolve"};
  }
  return crane_model::Status{};
}

}  // namespace crane_planning

#endif  // CRANE_PLANNING__REDUNDANCY_HPP_
