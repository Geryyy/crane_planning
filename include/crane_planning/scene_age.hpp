// How old `/crane/collision_scene` may be and still be planned against, and what
// happens when it is older than that.
//
// # Why the scene needs its own age at all
//
// The other three inputs the planner closes on -- `/joint_states`,
// `/crane/pendulum_state`, `/crane/payload_estimate` -- are all aged against a
// deadline in `PlannerNode`. The scene was not, and it is the one input whose
// subscription is **transient-local, depth 1**: a scene published once is latched
// by the middleware and handed to every planner that starts afterwards, for the
// rest of the session. An unaged latch does not read as "no scene"; it reads as a
// scene, and the planner plans against a world that has since been rebuilt.
//
// # Why it is not folded into `max_input_age`
//
// `max_input_age` is 0.5 s because `/joint_states` is a 100 Hz measurement of a
// machine that moves. The scene's natural rate is the world model's republish
// heartbeat, which is seconds and not milliseconds; one number cannot be both
// without either making the start state stale-blind or making every scene stale.
//
// # Why this file is ROS-free
//
// The decision is arithmetic on one age and one bound plus the request's own
// `avoid_collisions`, and `wiki/implementation/style_guide.md` 3 asks for that
// to be testable without a node, a clock or a graph. `PlannerNode` supplies the
// age; everything that decides what the age *means* is here.

#ifndef CRANE_PLANNING__SCENE_AGE_HPP_
#define CRANE_PLANNING__SCENE_AGE_HPP_

#include <string>

namespace crane_planning
{

/// The three states a subscribed collision scene can be in at request time.
enum class SceneFreshness
{
  NeverReceived,  ///< nothing has ever arrived on the topic, or nothing usable did
  Fresh,          ///< a scene arrived and its own stamp is inside `max_scene_age`
  Stale           ///< a scene arrived and its own stamp is outside `max_scene_age`
};

/// What the planner does with the scene it holds, and what it tells the caller.
struct SceneVerdict
{
  SceneFreshness freshness{SceneFreshness::NeverReceived};

  /// Whether the held scene may be handed to the core at all.
  /**
   * False for both absent and stale. The invariant this buys is the one the
   * goal of issue 096 asks for: **a scene the planner plans against is fresh**,
   * and a scene that is not is named in the answer instead of quietly used.
   */
  bool plan_against{false};

  /// Whether the request is refused outright rather than only annotated.
  /**
   * Set only when the scene is stale *and* the caller asked for collision
   * avoidance. See `judge_scene` for why that, and not "always refuse".
   */
  bool refuse{false};

  /// One sentence for the response `message`, or empty when there is nothing to add.
  std::string note;
};

/// Judge the held scene's age. ROS-free: the caller measures the age.
/**
 * `age_s` is read against the scene's own `header.stamp` and is ignored when
 * `have_scene` is false. A latched scene that is never republished simply keeps
 * getting older, which is exactly how the transient-local case is meant to
 * surface -- including the degenerate one where the publisher left the stamp at
 * zero, whose age is then the whole of ROS time and reads as stale rather than
 * as fresh.
 *
 * # Refuse or warn: the decision, and its reason
 *
 * Staleness **refuses when `avoid_collisions` is set and warns when it is not**,
 * which is the same rule the node already applies to the difference between an
 * input it closes on and an input that merely informs it:
 *
 * - With `avoid_collisions` set, the caller is asking for a trajectory certified
 *   clear of the world. A certificate issued against geometry that may have
 *   moved is a false statement, and `planner_core` already refuses the adjacent
 *   case -- `avoid_collisions` with no scene at all -- rather than returning a
 *   trajectory that reads as checked. A scene too old to trust is that case.
 * - With `avoid_collisions` clear, the scene was never going to bound the path.
 *   Refusing there would stop a stationary site from planning for the sake of an
 *   input the request declined to use, and the transient-local latch makes "the
 *   scene is not being republished" the common case rather than the exceptional
 *   one. So the request proceeds, without the stale geometry, and the answer
 *   says both.
 *
 * Neither branch is silent: `note` is non-empty for every outcome except "never
 * received", where `planner_core` already names the topic itself.
 */
[[nodiscard]] SceneVerdict judge_scene(
  bool have_scene, double age_s, double max_scene_age_s, bool avoid_collisions);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__SCENE_AGE_HPP_
