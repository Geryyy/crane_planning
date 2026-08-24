// The semi-analytic inverse kinematics of `wiki/robot_model.md` 2.2, over the
// five path coordinates q_a = (q1, q2, q3, q4, q7) of
// `wiki/trajectory_planning.md` 4.1.
//
// The five steps of 2.2, in the order it gives them:
//
//   1. slewing      q1 = atan2(y_d, x_d), which reduces the rest to the arm plane
//   2. two-link     q2 and q3 in closed form at a given d45, by the law of cosines
//   3. redundancy   a scalar search over d45, scored by joint-range centring
//   4. wrist offset the two-link chain reaches K5, not the tool, so the target is
//                   corrected by fixed-point iteration
//   5. rotator      q7 takes the residual yaw
//
// The passive pair is **pinned**, at `Model::passive_equilibrium` -- 2.2's own
// default for a placement goal, because pinning at a measured q_u solves for
// where the tool is now and a tool that is still swinging leaves a target that
// is right for an instant. Making the equilibrium a *constraint* rather than a
// pinning is the NLP of 2.2's second half and is issue 039.
//
// > "Analytic IK" is closed-form only in step 2. Steps 3 and 4 are a scalar
// > search and a fixed-point iteration. The solve is fast and deterministic, but
// > it is **not** a closed-form inverse and it can fail to converge. Every call
// > must be validated against forward kinematics before the result is used.
//
// That sentence is why `solve_inverse_kinematics` returns the residual it
// achieved and refuses outright when the residual is over `eps_pos`/`eps_yaw`.
// A configuration that does not put the tool where it was asked for is not a
// worse answer than none -- it is an answer that reads as success.

#ifndef CRANE_PLANNING__INVERSE_KINEMATICS_HPP_
#define CRANE_PLANNING__INVERSE_KINEMATICS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>

#include "crane_model/model.hpp"
#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// The yaw phi_z of `wiki/nomenclature.md` 5 out of a rotation.
/**
 * phi is XYZ fixed-axis, equivalently ZYX intrinsic, so phi_z is the first
 * rotation read off the matrix and is the yaw the assembly tasks use.
 */
[[nodiscard]] double phi_z_of(const Eigen::Quaterniond & orientation);

/// The tolerances and the iteration budgets 2.2's search and fixed point run to.
struct IkSettings
{
  double eps_pos{1.0e-4};              ///< m, the position half of 2.2's oracle
  double eps_yaw{1.0e-4};              ///< rad, the yaw half
  std::size_t d45_samples{41};         ///< resolution of the step-3 scalar search
  std::size_t redundancy_passes{2};    ///< how often step 3 is re-run on a corrected target
  std::size_t fixed_point_iterations{40};  ///< step 4's budget
  std::size_t refinement_iterations{20};   ///< the Jacobian polish's budget
  GeometryTolerance geometry{};
};

/// One goal, as 2.2 poses it: a position and a yaw, with the tool coordinate held.
struct IkRequest
{
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};  ///< K_tcp position in K0
  double phi_z_d{};                                  ///< desired yaw, rad
  double q8{};                                       ///< the tool coordinate, held
  crane_model::Payload payload{};                    ///< it moves q_eq (robot_model 5.1)
};

/// A solved configuration, with the residual it was accepted on.
struct IkSolution
{
  crane_model::Q q{crane_model::Q::Zero()};  ///< all eight; q5, q6 at the equilibrium
  double residual_p{};                        ///< ||p_fk - p_d||, m
  double residual_phi_z{};                    ///< |phi_z,fk - phi_z,d|, rad
  double d45{};                               ///< the extension step 3 resolved on
  std::size_t fixed_point_iterations{};
  std::size_t refinement_iterations{};
};

/// Solve 2.2 for one goal, and refuse anything the FK oracle does not accept.
[[nodiscard]] crane_model::Result<IkSolution> solve_inverse_kinematics(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & settings, const IkRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__INVERSE_KINEMATICS_HPP_
