// The equilibrium-**constrained** inverse kinematics of `wiki/robot_model.md`
// 2.2, which is the endpoint `wiki/trajectory_planning.md` 6 asks a placement
// goal for:
//
//   q_e = argmin ||p(q) - p_d||^2 + ||R_d^T R(q) - I||_F^2   s.t.  g_u(q) = 0
//
// over **all eight** coordinates rather than over the actuated five with the
// passive pair pinned from outside. The answer is a genuine steady state of the
// passive subsystem, so the tool does not sway when it arrives --
// `trajectory_planning` 3 lists "ends at passive equilibrium, at rest" as a
// guarantee the planner owes, and a geometrically chosen endpoint carries no
// such guarantee.
//
// # g_u(q) is read off the frozen model, and it is not `passive_equilibrium`
//
// `wiki/implementation/model_api_contract.md` 7 states the invariant this whole
// file rests on: at a valid passive equilibrium with zero passive velocity and
// acceleration, the **passive rows of inverse dynamics are zero**. So
//
//   g_u(q) = inverse_dynamics(q, 0, 0, payload).segment<2>(4)
//
// is the constraint, evaluated through the frozen API with no second model and
// no new dependency. Measured on both descriptions here it is 1e-13 N m at an
// equilibrium and some 400 N m a tenth of a radian off one, and it costs
// **0.70 ms** against **33 ms** for one `Model::passive_equilibrium` -- a factor
// of 47, which is what makes an eight-coordinate NLP affordable at all.
//
// That `passive_equilibrium` is *not* the constraint is what gives the final
// assertion its teeth. `g_u(q) = 0` has more than one root: the tool hanging
// down, and the tool standing up. Both are equilibria and only the first is a
// steady state anything settles into. The solver can converge onto either; the
// model's own `passive_equilibrium` returns the settled one, so checking the
// answer against it is a **stability check** and not a restatement of the
// solver's own constraint residual.
//
// # The orientation residual: what R_d has to be, and why it is not R(phi_z,d)
//
// 2.2 writes the desired rotation as `R(phi_z,d)`, a pure yaw. Measured, that
// makes the Frobenius term a **constant** on this machine and therefore blind:
// at every equilibrium of the PZS100 the tool's attitude is exactly
// `R_z(phi_z) R_x(pi)` -- it hangs, so its own z axis points down -- and
//
//   ||R_z(phi_d)^T R_z(phi) R_x(pi) - I||_F^2 = 8
//
// identically, whatever the yaw error is. Minimising it would leave the rotator
// entirely unconstrained. The 7040 hangs nearly but not exactly so, and its rest
// attitude is not even constant: it wanders by 0.03 in Frobenius norm across
// configurations, because that tool's mass sits off the tilt axis.
//
// So the attitude the tool arrives at is *not* a goal -- gravity and the
// constraint set it -- and the only part of `R_d` that is a goal is the yaw.
// R_d is therefore built each iteration from the goal yaw and the attitude the
// current iterate actually hangs at,
//
//   R_d = R_z(phi_z,d - phi_z(q_k)) R(q_k)
//
// and is then held fixed while that iteration's residual and Jacobian are taken.
// At the iterate this evaluates to exactly `4(1 - cos(phi_z(q_k) - phi_z,d))`, a
// pure yaw error with no parametrisation and no singularity, which is the
// property 2.2 chose Frobenius for; away from it the nine entries still penalise
// the tool tipping out of the attitude the constraint is holding it in.
//
// # Bounded, because an unbounded NLP inside a service call is a hang
//
// Both an iteration cap and a wall-clock cap, both parameters. Hitting either is
// a refusal naming the cap that was hit, never the best answer so far dressed as
// a solution. The latency bound these feed is issue 045's; these exist here
// because `/crane/plan_motion` is a service and a caller cannot cancel one.

#ifndef CRANE_PLANNING__EQUILIBRIUM_IK_HPP_
#define CRANE_PLANNING__EQUILIBRIUM_IK_HPP_

#include <cstddef>

#include "crane_model/model.hpp"
#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/redundancy.hpp"

namespace crane_planning
{

/// The caps and tolerances the NLP of 2.2 runs to.
struct EquilibriumIkSettings
{
  /// rad, how far the answer's passive pair may sit from `passive_equilibrium`.
  /**
   * The assertion of the acceptance test, not the solver's own residual. It is
   * loose enough to absorb the model's own equilibrium solve converging to a
   * slightly different point than this one does, and tight enough that the
   * standing-up root of `g_u(q) = 0` -- which is radians away -- can never pass.
   */
  double eps_equilibrium{1.0e-4};

  /// Gauss-Newton steps. Hitting it is a refusal, not a best-effort answer.
  /**
   * Warm-started from the semi-analytic route this is one or two iterations on
   * almost every goal; the worst of fifty sampled goals across both descriptions
   * was 24. Sixty is that with room, so that reaching the cap means the solve was
   * genuinely still moving rather than that the budget was tight.
   */
  std::size_t max_iterations{60};
  double max_wall_clock_s{5.0};    ///< s, measured across the whole solve
  RedundancyWeights redundancy{};  ///< how 2.2 step 3's leftover freedom is spent
};

/// Solve the NLP of 2.2 for one goal, or refuse it and say what it missed by.
/**
 * `seed` is the semi-analytic settings the initial guess is taken with: 2.2's
 * closing paragraph is that the fast route is for where speed matters, and the
 * cheapest use of a fast route is to start the exact one near its answer. A goal
 * the semi-analytic route refuses is still attempted here, from the description's
 * own mid-range configuration, so a refusal below is this solver's own and
 * carries this solver's residual.
 *
 * The returned `IkSolution` carries `route == EquilibriumConstrained` together
 * with the two residuals of 2.2's acceptance test, the constraint residual in
 * N m and the equilibrium residual in rad that was asserted against
 * `Model::passive_equilibrium`.
 */
[[nodiscard]] crane_model::Result<IkSolution> solve_equilibrium_constrained_ik(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & seed, const EquilibriumIkSettings & settings, const IkRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__EQUILIBRIUM_IK_HPP_
