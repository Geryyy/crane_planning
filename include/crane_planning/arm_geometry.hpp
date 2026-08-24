// The planar two-link parameters the semi-analytic IK of `wiki/robot_model.md`
// 2.2 closes on, **probed out of the model** rather than written down.
//
// 2.2 names four numbers -- the boom length a2, the arm offset a3, the
// telescope-dependent forearm extension d45(q4), and the bearing gamma the
// closure is measured from. They are properties of the description every
// controller in the deployment is configured against, so a copy of them here
// would be a second machine to keep current. Instead they are recovered from
// `Model::forward_kinematics` at construction: the boom pivot is K1's own
// origin, K2's origin rides it at a constant radius, and the chain from there to
// K5 lies in one plane whatever q2, q3 and q4 are.
//
// That the structure holds is **checked, not assumed**. A description whose arm
// is not planar, or whose telescope does not extend affinely in q4, fails
// `probe_arm_geometry` with a message naming the residual, because a two-link
// closure run over a chain that is not a two-link chain answers confidently and
// wrongly.

#ifndef CRANE_PLANNING__ARM_GEOMETRY_HPP_
#define CRANE_PLANNING__ARM_GEOMETRY_HPP_

#include <cmath>
#include <cstddef>

#include "crane_model/model.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// How closely a description has to match the two-link structure of 2.2.
struct GeometryTolerance
{
  double planarity_m{1.0e-6};  ///< out-of-plane travel a probe may still show
  double structure_m{1.0e-6};  ///< residual of the affine d45(q4) fit
  double bearing_rad{1.0e-6};  ///< spread of the constant bearing offset
};

/// The arm as `wiki/robot_model.md` 2.2 poses it, in K1's own x-y plane.
/**
 * K1's x-y plane *is* the arm plane: the slewing joint turns about K0's z and
 * the description places K1 so that q2, q3 and q4 all move within z = 0 of it.
 * Angles below are bearings in that plane.
 */
struct ArmGeometry
{
  double a2{};       ///< boom length, m -- K1's origin to K2's origin
  double gamma0{};   ///< the boom's plane bearing at q2 = 0, rad
  double sign_q2{};  ///< +1 if a positive q2 advances that bearing
  double sign_q3{};  ///< +1 if a positive q3 advances the forearm's bearing
  double a3{};       ///< arm offset perpendicular to the telescope, m
  double d45_0{};    ///< d45 at q4 = 0, m
  double d45_gain{};  ///< dd45/dq4 -- 2 on this machine, both stages on one q4
  double psi0{};     ///< bearing the forearm's atan2(a3, d45) is measured from

  [[nodiscard]] double d45(double q4) const noexcept {return d45_0 + d45_gain * q4;}
  [[nodiscard]] double q4_of_d45(double d45_value) const noexcept
  {
    return (d45_value - d45_0) / d45_gain;
  }

  /// The effective forearm length sqrt(a3^2 + d45^2) of 2.2 step 2.
  [[nodiscard]] double forearm_length(double q4) const noexcept
  {
    return std::hypot(a3, d45(q4));
  }

  /// The forearm's bearing relative to the boom at q3 = 0, i.e. psi0 +
  /// atan2(a3, d45) -- the term 2.2 writes as `atan2(a3, d45) - pi/2`.
  [[nodiscard]] double forearm_bearing(double q4) const noexcept
  {
    return psi0 + std::atan2(a3, d45(q4));
  }
};

/// Recover `ArmGeometry` from the model, and refuse a description that is not one.
/**
 * `samples` is how many telescope extensions the affine fit and the structure
 * check are run over; it is not the resolution of the redundancy search, which
 * is its own parameter.
 */
[[nodiscard]] crane_model::Result<ArmGeometry> probe_arm_geometry(
  const crane_model::Model & model, const JointLimits & limits, std::size_t samples,
  const GeometryTolerance & tolerance);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__ARM_GEOMETRY_HPP_
