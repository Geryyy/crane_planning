// Stage 2 of `wiki/trajectory_planning.md`, in the one shape the slice-5 tracer
// is allowed to have: **a single scaled ramp between the endpoints, obeying the
// velocity limits and nothing else**.
//
// This is deliberately not the time parametrization the page specifies. §5.2's
// stage 2 is a path-constrained optimal control problem carrying the sway
// explicitly, with the force and pump-flow constraints and the kappa margin, and
// it arrives with **issue 043**. Nothing here reads a cylinder force, a pump
// flow or a sway state, and a reference produced here is feasible against the
// velocity bound alone.
//
// What it does keep from the page is the boundary behaviour, because a ramp that
// did not have it would be worse than useless downstream: the profile is C1 at
// both ends and starts and ends at rest, so the trajectory controller is handed
// a reference whose first and last velocity are zero rather than a step.

#ifndef CRANE_PLANNING__TRAJECTORY_TIMING_HPP_
#define CRANE_PLANNING__TRAJECTORY_TIMING_HPP_

#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// How the ramp is sampled and how short it may get.
struct RampSettings
{
  double Ts{0.04};           ///< sample period of the emitted points, s
  double min_duration{0.5};  ///< floor on the traversal time, s
};

/// The time-parametrized reference of `wiki/robot_model.md` 4.4.
/**
 * Positions and velocities only, over the six actuated coordinates in
 * ROS 2 Interfaces 3.1 order. Accelerations are left out on the same terms the
 * horizon leaves them out (ROS 2 Interfaces 10): the receiver owns derivatives.
 */
struct TimedTrajectory
{
  std::vector<double> time_from_start;         ///< s, strictly increasing, first is 0
  std::vector<crane_model::QA> q_a_ref;        ///< rad or m
  std::vector<crane_model::DQA> dq_a_ref;      ///< rad/s or m/s
  double duration{};                           ///< s
  double limiting_fraction{};                  ///< peak |dq| as a fraction of the scaled bound
};

/// Whether kappa is a scaling at all.
/**
 * Separate from `scaled_ramp` so the refusal can happen before the endpoint IK
 * rather than after it: a request that asks for `speed_scale = 2` is refused for
 * that reason, not for whatever the solver would have said next.
 */
[[nodiscard]] crane_model::Status check_margin_factor(double margin_factor);

/// The scaled ramp between two configurations.
/**
 * `margin_factor` is `crane_msgs/PlanMotion.speed_scale` and **not** kappa. Issue
 * 038 wrote them as one thing because the timing constrained nothing else; they
 * are two, they are owned by different parties, and `timing_ocp.hpp` is where
 * the distinction now lives. This scales the velocity bound and therefore the
 * duration; a value outside (0, 1] is refused rather than clamped, because a
 * clamp would answer a request nobody made.
 */
[[nodiscard]] crane_model::Result<TimedTrajectory> scaled_ramp(
  const crane_model::QA & q_a_start, const crane_model::QA & q_a_goal, const JointLimits & limits,
  double margin_factor, const RampSettings & settings);

/// How densely the path is probed for the peak of `q_a'` the duration follows from.
inline constexpr std::size_t kPathRateSamples = 401;

/// The same ramp, run along a geometric path instead of straight between two configurations.
/**
 * **This is the seam.** Issue 040 produces a `C2` path in sigma and issue 043
 * produces the timing; until 043 lands, something has to turn the one into the
 * other, and this is the smallest honest thing that does: sigma follows the same
 * cubic `s(tau)` the chord ramp uses, so `sigma_dot` is zero at both ends and the
 * reference still starts and ends at rest, and the traversal time is chosen so
 * that `|q_a'(sigma)| sigma_dot` clears the scaled velocity bound everywhere.
 *
 * The bound is taken conservatively -- the peak of `|q_a'|` over the whole path
 * against the peak of `sigma_dot`, rather than their true product -- so
 * `limiting_fraction` here reports the peak the emitted samples actually reach
 * and is normally below one. That is the honest report: this is not a
 * time-optimal parametrization and 5.2's OCP is what makes it one.
 *
 * Nothing here reads a cylinder force, a pump flow or a sway state, exactly as
 * `scaled_ramp` does not.
 */
[[nodiscard]] crane_model::Result<TimedTrajectory> scaled_ramp_along_path(
  const GeometricPath & path, const JointLimits & limits, double margin_factor,
  const RampSettings & settings);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__TRAJECTORY_TIMING_HPP_
