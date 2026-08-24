// The semi-analytic inverse kinematics of `wiki/robot_model.md` 2.2, over the
// five path coordinates q_a = (q1, q2, q3, q4, q7) of
// `wiki/trajectory_planning.md` 4.1.
//
// The five steps of 2.2, in the order it gives them:
//
//   1. slewing      q1 = atan2(y_d, x_d), which reduces the rest to the arm plane
//   2. two-link     q2 and q3 in closed form at a given d45, by the law of cosines
//   3. redundancy   a scalar search over d45, scored by joint-range centring and
//                   by collision clearance -- see `redundancy.hpp`
//   4. wrist offset the two-link chain reaches K5, not the tool, so the target is
//                   corrected by fixed-point iteration
//   5. rotator      q7 takes the residual yaw
//
// The passive pair is **pinned**, at `Model::passive_equilibrium` -- 2.2's own
// default for a placement goal, because pinning at a measured q_u solves for
// where the tool is now and a tool that is still swinging leaves a target that
// is right for an instant. Making the equilibrium a *constraint* rather than a
// pinning is the NLP of 2.2's second half; it lives in `equilibrium_ik.hpp`,
// it is what a placement endpoint is solved with, and this route is 2.2's
// "where speed matters" one -- including as the initial guess that route starts
// from.
//
// # The pin is a function of the configuration, and is evaluated as one
//
// `q_u = q_eq(q_a)` is where the tool hangs at the configuration the solve is
// *currently testing*, so every residual this file measures re-solves it. That
// is the expensive choice -- `Model::passive_equilibrium` costs some 45 ms
// against 0.35 ms for one `forward_kinematics` call, measured on both machine
// descriptions in this workspace -- and it is not optional.
//
// The cheap alternative, holding q_u across a solve and re-solving it in an
// outer loop, does not converge on this machine, and it fails structurally
// rather than by bad luck. Measured on the PZS100, `dq_eq/dq_a` is
// `[0, -1, -1, 0, 0]` exactly: the passive tilt cancels q2 + q3, which is the
// statement that the tool hangs vertically whatever the arm does. Hold the pin
// at a tilt and the closure has to swing the arm until the *tilted* tool lands
// on the goal; re-solve the pin there and the tool swings back upright by
// exactly what the arm was moved by. The outer iteration therefore has a
// multiplier of magnitude one and lands in a period-two orbit -- observed, over
// eighty passes, alternating between two configurations a quarter of a radian
// apart and leaving 0.14 m of position residual that no budget closes.
//
// Evaluated as a function instead, the same structure is what makes the solve
// easy: the tool's offset from K5 at equilibrium is very nearly a constant
// `(0, 0, -L)` -- exactly constant on the PZS100 -- so the wrist target of step
// 4 is the goal raised by the tool's hanging length and the fixed point is a
// contraction that converges in two or three iterations.
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
#include <cstdint>

#include "crane_model/model.hpp"
#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/redundancy.hpp"

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
  double eps_pos{1.0e-4};             ///< m, the position half of 2.2's oracle
  double eps_yaw{1.0e-4};             ///< rad, the yaw half
  std::size_t d45_samples{41};        ///< resolution of the step-3 scalar search
  std::size_t fixed_point_iterations{4};  ///< step 4's budget
  std::size_t refinement_iterations{6};   ///< the Jacobian polish's budget
  GeometryTolerance geometry{};
  RedundancyWeights redundancy{};         ///< how step 3 spends the leftover freedom
};

/// One goal, as 2.2 poses it: a position and a yaw, with the tool coordinate held.
struct IkRequest
{
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};  ///< K_tcp position in K0
  double phi_z_d{};                                  ///< desired yaw, rad
  double q8{};                                       ///< the tool coordinate, held
  crane_model::Payload payload{};                    ///< it moves q_eq (robot_model 5.1)

  /// The scene step 3's clearance term is scored against. Null skips that term.
  /**
   * A *preference* and not the safety check: the answer is put through
   * `check_path` before it is flown, and a goal whose only reachable
   * configuration is in collision is refused there, not here.
   */
  const crane_model::CollisionScene * scene{nullptr};
};

/// Which of 2.2's two formulations produced an endpoint.
/**
 * 2.2 poses the same problem twice and its closing paragraph is the rule for
 * choosing: the semi-analytic route where speed matters, the
 * equilibrium-constrained NLP where the endpoint must be sway-free. Which one
 * ran is carried on the answer rather than asked of the caller.
 */
enum class EndpointRoute : std::uint8_t
{
  SemiAnalytic,            ///< 2.2 steps 1-5: the scalar search plus the fixed point
  EquilibriumConstrained   ///< 2.2's NLP, with g_u(q) = 0 as a constraint
};

/// A solved configuration, with the residual it was accepted on.
/**
 * One type for both routes, so that whatever produced an endpoint it travels
 * through `MotionPlan` and out of the service the same way. The diagnostic
 * counts are per-route and the ones the other route does not run stay zero:
 * `fixed_point_iterations`, `refinement_iterations` and `equilibrium_calls`
 * belong to the semi-analytic solve, `nlp_iterations`, `residual_g_u` and
 * `elapsed_s` to the constrained one. `route` says which to read.
 */
struct IkSolution
{
  crane_model::Q q{crane_model::Q::Zero()};  ///< all eight; q5, q6 at the equilibrium
  double residual_p{};                        ///< ||p_fk - p_d||, m
  double residual_phi_z{};                    ///< |phi_z,fk - phi_z,d|, rad
  double d45{};                               ///< the extension step 3 resolved on
  EndpointRoute route{EndpointRoute::SemiAnalytic};
  double residual_equilibrium{};              ///< ||q_u - passive_equilibrium(q_a)||, rad
  double residual_g_u{};                      ///< ||g_u(q)||, N m -- the constraint itself
  double elapsed_s{};
  std::size_t fixed_point_iterations{};
  std::size_t refinement_iterations{};
  std::size_t equilibrium_calls{};            ///< what the solve actually cost
  std::size_t nlp_iterations{};               ///< Gauss-Newton steps of the constrained route
};

/// Solve 2.2 for one goal, and refuse anything the FK oracle does not accept.
[[nodiscard]] crane_model::Result<IkSolution> solve_inverse_kinematics(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & settings, const IkRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__INVERSE_KINEMATICS_HPP_
