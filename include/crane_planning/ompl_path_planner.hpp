// The OMPL path planner of `wiki/trajectory_planning.md` 4.4 and the smoothing
// 4.5 says is not optional, over the five path coordinates `q_a` of 4.1.
//
// # Second, and only second
//
// 4.4 names two mechanisms and fixes their order: **generate the structured
// primitive, check it, accept it if clear, otherwise sample.** 4.4's
// `[!important]` is the reason the order is not a preference. A sampling planner
// is stochastic and its runtime is not bounded, which is a poor fit for the
// re-plan loop of 7; primitive-first buys deterministic latency in the common
// case and leaves completeness to the rare one, and it means the ordinary
// trajectory is smooth and predictable, which matters for an operator watching a
// large machine move. A fallback that runs anyway -- to compare, to warm a cache,
// to "check the primitive" -- destroys exactly that, so nothing in this file runs
// unless `plan_motion` has already generated a primitive and been refused one.
//
// What the fallback is for is what 4.4 lists: an obstacle above the transfer
// altitude, a structure that must be rounded, a partially built wall in the way.
//
// # RRT-Connect, and why it rather than something else
//
// 9's open items ask for the sampling planner and its smoothing stage to be
// **chosen**. This is the choice and this paragraph is the reason.
//
//   * **RRT-Connect**, `ompl::geometric::RRTConnect`.
//     `wiki/implementation/libraries.md` 1 lists OMPL for sampling-based path
//     planning against "RRT/PRM and their smoothing", so the library is not a
//     decision; which planner out of it is. 4.4 names "RRT-Connect or similar",
//     and the property that makes it the right one here is that it is
//     **feasibility-only and bidirectional**: it stops at the first solution
//     rather than improving one, so a solve ends when it has an answer instead of
//     spending its whole budget, which is what a per-call wall-clock cap needs.
//     Growing from both ends matters on this machine because the goal is a
//     placement -- the tool has to arrive down among the runges, in exactly the
//     narrow passage a single tree explores last.
//   * **Not RRT\*, BIT\* or any asymptotically optimal planner.** They are
//     anytime: they use the whole budget by construction and their answer is a
//     function of how long they were given, which is the opposite of the bounded,
//     repeatable latency 4.4 asks for. Path *quality* here is the shortcut and
//     the C2 fit's job, not the search's.
//   * **Not PRM.** A roadmap pays for itself when many queries share one static
//     scene. The scene here is `/crane/collision_scene`, which the world model
//     republishes as the site changes, so the roadmap would be rebuilt or
//     invalidated on the same cadence as the queries it was meant to amortise.
//   * **Not MoveIt's pipeline**, which is `libraries.md` 5's *deliberately
//     absent* row: "Pinocchio + Coal + OMPL directly". Its SRDF format is not
//     rejected and `crane_model` already reads one; its planning stack is.
//
// # Five coordinates, and never the passive pair
//
// 4.1's `[!warning]` is blunt: q5 and q6 are not commandable and are not IK
// outputs, they are dynamics outputs, and a sampling planner handed a joint group
// that includes them interpolates them as free variables and returns paths that
// satisfy every joint limit while being **dynamically impossible**. The legacy
// MoveIt SRDF declares exactly such a group. So the state space here is five
// dimensional and is built out of `JointLimits`, which is the description's own
// range; the passive pair never appears in it and is solved -- not sampled -- at
// every configuration the check looks at, through `Model::passive_equilibrium`.
//
// The tool coordinate q8 is not in the space either, for 4.1's other reason: it
// is not a path variable. It is held at the value the request carries and rides
// the fit's own parameter, exactly as it does for the primitive.
//
// The metric is not the raw Euclidean one over a vector that mixes radians with
// metres. Each axis is a subspace weighted by `1 / dq_max`, so a distance in this
// space is the sum of how long each axis would take at its own limit -- the same
// unit `geometric_path.hpp` distributes sigma by, and the only unit in which the
// five coordinates are comparable at all.
//
// # The check is issue 041's, not a second one
//
// `isValid` is `check_configuration` of `collision.hpp` -- the scene, the truck
// keyed to its measured pose, the crane against itself, and the sway envelope of
// 4.3 resolved at `q_sway_max`. Motion validation subdivides at the **same**
// stated resolution `check_path` samples at, measured with the same
// `watched_frame_travel_m`, so a sampled path is cleared on the same terms as a
// primitive and not on weaker ones. A motion checked only at its endpoints is not
// checked, and OMPL's default validator would have measured the subdivision in
// state-space distance rather than in how far the geometry moves.
//
// # Smoothing is mandatory, and so is re-checking it
//
// 4.5, in order and with no step skippable: a sampling planner returns a
// piecewise-linear C0 path; stage 2 needs `q_a'(sigma)` and `q_a''(sigma)`; at a
// kink the first is discontinuous and the second undefined, the path-velocity
// limit collapses to `sigma_dot = 0` and the machine **stops dead at every
// waypoint**. So every path this file returns has been through
//
//   1. `shortcut`   -- the corner-cutting pass, each replacement validated,
//   2. `fit_c2_path` -- the same C2 fit the primitive is built with,
//   3. `check_path`  -- the whole curve re-checked for collision,
//
// and a path that fails step 3 is refused. Step 3 is not belt and braces: steps 1
// and 2 both move the path off the polyline the search cleared. The shortcut
// replaces a detour by a chord, and the fit is monotone within each waypoint box
// but is not the chord between two waypoints -- so what was cleared is not what
// would be flown, and only step 3 answers for what would be.
//
// The primitive needs none of this. 4.5's last sentence says so and
// `geometric_path.hpp` is why: it is built C2 by construction rather than
// smoothed into C2 afterwards.
//
// # Bounded, and deterministic
//
// Two caps, both per call: a wall-clock budget and a cap on how many
// configurations the search may check. Exhausting either is a refusal naming it
// and never an approximate path -- a path that stops short of the goal is not a
// worse answer than none, it is an answer that reads as success. The replanning
// loop's own latency bound is issue 045's and is not this cap.
//
// The search samples from a `std::mt19937` seeded by `OmplSettings::seed`
// and from nothing else, so a fixed seed gives a fixed path: OMPL's global RNG
// is not what this planner draws from, because it can be seeded only once per
// process and a test that has to run second would then not be reproducible. The
// shortcut draws its pairs from the same generator, one stream on from the
// search. A stochastic planner in an automated suite is a flaky test, and the
// tests here fix the seed.

#ifndef CRANE_PLANNING__OMPL_PATH_PLANNER_HPP_
#define CRANE_PLANNING__OMPL_PATH_PLANNER_HPP_

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/collision.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning
{

/// Which of `wiki/trajectory_planning.md` 4.4's two mechanisms answered.
/**
 * It travels on the answer so that a caller can tell a lucky deterministic plan
 * from a sampled one. They are not interchangeable: the primitive's latency is
 * bounded by construction and its shape is the lift/traverse/descend an operator
 * expects, while a sampled path spent a budget to be found and rounds whatever
 * was in the way.
 */
/// What a deployment configures about the fallback.
struct OmplSettings
{
  PathFitSettings fit{};
  CollisionSettings collision{};
  /// The seed the search and the shortcut draw from. Fixed, never a clock.
  /**
   * A default that is a constant rather than a time is the point: two identical
   * requests against one scene get one answer, and a test gets the same answer
   * as the deployment. Change it to explore a scene the shipped seed happens to
   * do badly on; do not derive it from the clock.
   */
  std::uint32_t seed{20420042U};

  /// Wall-clock cap on one search, s. Exhausting it is a refusal naming it.
  double time_budget_s{5.0};

  /// Cap on the configurations one search may check, whatever the clock says.
  /**
   * The half of the bound that does not depend on how fast the machine running
   * the planner is, and therefore the half a test can rely on. Each check costs a
   * `Model::passive_equilibrium` plus the collision queries of its sway envelope,
   * so this is also the honest unit of what a search costs.
   */
  std::size_t max_validity_checks{4000};

  /// Longest motion the tree may extend by, in the state space's own metric.
  /**
   * Which is `sum over axes of |dq| / dq_max`, i.e. seconds at the axis's own
   * limit -- the unit `geometric_path.hpp` distributes sigma by. Zero asks OMPL
   * for its own default, a fraction of the space's extent, which knows nothing
   * about this machine.
   */
  double extension_span{2.0};

  /// How many replacements the shortcut pass of 4.5 tries.
  std::size_t shortcut_attempts{120};

  /// How far past its two endpoints an **unbounded** axis may be searched, rad.
  /**
   * The rotator is a `continuous` joint in both descriptions, so the description
   * gives it no range and a sampler needs one. Half a turn either side of the
   * interval the two endpoints span is every distinct yaw it has: q7 and
   * q7 + 2 pi are the same geometry, so a bound wider than this samples poses the
   * search has already been offered. Narrower would hide some.
   */
  double unbounded_margin_rad{3.14159265358979323846};
};

/// The bounds the five-dimensional state space is built on.
/**
 * `wiki/implementation/parameters.md` 3 is the source: position ranges come from
 * the URDF, and `joint_limits.hpp` is where this package reads them. A
 * `continuous` joint has no range there -- that is a statement about the rotator
 * and not an omission -- so its two entries are the endpoints' own interval
 * widened by `unbounded_margin_rad`, and `bounded_by_description` records which
 * of the two a coordinate got.
 */
struct StateSpaceBounds
{
  Eigen::Matrix<double, kPathDof, 1> lower{Eigen::Matrix<double, kPathDof, 1>::Zero()};
  Eigen::Matrix<double, kPathDof, 1> upper{Eigen::Matrix<double, kPathDof, 1>::Zero()};
  std::array<bool, kPathDof> bounded_by_description{};

  /// `1 / dq_max` per axis: what makes the metric a time rather than a mixture.
  Eigen::Matrix<double, kPathDof, 1> weight{Eigen::Matrix<double, kPathDof, 1>::Ones()};
};

/// The bounds the search would be given for one pair of endpoints.
/**
 * Exposed so that a test can assert the state space really is the five path
 * coordinates of 4.1 at the description's own ranges, rather than inferring it
 * from a path that happened to stay inside them.
 */
[[nodiscard]] crane_model::Result<StateSpaceBounds> state_space_bounds(
  const JointLimits & limits, const PathVector & q_a_start, const PathVector & q_a_goal,
  const OmplSettings & settings);

/// One move the fallback is asked to cover, with the primitive already refused.
struct OmplRequest
{
  crane_model::Q q_start{crane_model::Q::Zero()};  ///< all eight, passive pair as measured
  crane_model::Q q_goal{crane_model::Q::Zero()};   ///< the endpoint issue 039 solved

  /// The measured rate at the start, over the five path coordinates of 4.1.
  /**
   * The same field `PrimitiveRequest` carries and for the same reason: 4.5's
   * mandatory C2 refit is a `fit_c2_path` like any other, so a fallback answering
   * a re-plan from a moving machine owes its first segment the same boundary
   * condition the primitive's lift phase does. The *search* is unaffected -- a
   * sampled polyline is geometry and has no rates in it.
   */
  PathVector dq_start{PathVector::Zero()};

  crane_model::Payload payload{};
  PayloadShape payload_shape{};

  /// The scene to check, or null when scene collision checking is disabled.
  const crane_model::CollisionScene * collision_scene{nullptr};
};

/// The shortcut pass of 4.5, over a polyline.
/**
 * Repeatedly draws a pair of waypoints and, when `accept` takes the straight line
 * between them, drops everything between the two. Deterministic in `seed`, so
 * the same polyline shortcuts the same way every run.
 *
 * **`accept` is the whole of the collision content and none of the guarantee.**
 * The fallback passes a predicate that puts the chord through the same motion
 * validation the search used, so the polyline this returns is one that was
 * cleared -- and the path that is *flown* is the C2 fit of it, which is not that
 * polyline, which is why 4.5 mandates the re-check afterwards and why
 * `plan_ompl_path` runs one. Handed a predicate that accepts everything this
 * is the naive corner-cut 4.5 warns about, and the re-check is what has to catch
 * it; that is a case worth testing and this signature is what makes it reachable.
 */
[[nodiscard]] std::vector<PathVector> shortcut(
  const std::vector<PathVector> & waypoints, std::size_t attempts, std::uint32_t seed,
  const std::function<bool(const PathVector &, const PathVector &)> & accept);

/// The fallback's answer, with what it cost and what cleared it.
struct OmplPlan
{
  GeometricPath path{};        ///< shortcut, fitted C2, and re-checked -- in that order
  PathCheck recheck{};         ///< 4.5's re-check, on the curve that would be flown
  std::size_t search_states{};    ///< waypoints RRT-Connect returned, endpoints included
  std::size_t smoothed_states{};  ///< what the shortcut left of them
  std::size_t validity_checks{};  ///< configurations the search looked at
  std::size_t shortcut_checks{};  ///< configurations the shortcut pass looked at
  double search_s{};              ///< wall clock the search spent
  std::uint32_t seed{};           ///< the seed it was all drawn from
};

/// The fallback, as the sentence a response carries.
[[nodiscard]] std::string describe(const OmplPlan & plan);

/// Sample a path around what blocked the primitive, smooth it, and re-check it.
/**
 * 4.4's second mechanism and 4.5's three steps, as one call. Refuses rather than
 * returning anything partial: an exhausted budget, an approximate path that stops
 * short of the goal, a start or a goal the check already refuses, and a smoothed
 * path the re-check finds blocked are each a failure naming what happened.
 *
 * `limits` is the description's own range and is what the state space is built
 * on; `collision` is the same `CollisionSettings` the primitive was checked with,
 * so both mechanisms are cleared on one set of numbers; `fit` is the same C2 fit
 * the primitive is built with.
 */
[[nodiscard]] crane_model::Result<OmplPlan> plan_ompl_path(
  const crane_model::Model & model, const JointLimits & limits,
  const OmplSettings & settings, const OmplRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__OMPL_PATH_PLANNER_HPP_
