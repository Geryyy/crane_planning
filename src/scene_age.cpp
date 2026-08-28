#include "crane_planning/scene_age.hpp"

#include <cmath>
#include <string>

namespace crane_planning
{

SceneVerdict judge_scene(
  const bool have_scene, const double age_s, const double max_scene_age_s,
  const bool avoid_collisions)
{
  SceneVerdict verdict;
  if (!have_scene) {
    // Left silent on purpose: `planner_core` refuses this case by name when
    // `avoid_collisions` is set, and a second sentence saying the same thing in
    // other words is one more thing to keep in step with it.
    verdict.freshness = SceneFreshness::NeverReceived;
    return verdict;
  }

  // Written as `age <= bound` rather than as `age > bound` so that a NaN age --
  // which is what an unset or non-finite stamp arithmetic produces -- lands on
  // the stale side rather than passing every comparison.
  if (std::isfinite(age_s) && age_s <= max_scene_age_s) {
    verdict.freshness = SceneFreshness::Fresh;
    verdict.plan_against = true;
    verdict.note = "it is " + std::to_string(age_s) + " s old by its own header stamp, inside "
      "max_scene_age = " + std::to_string(max_scene_age_s) + " s, so it is the world as the "
      "world model last saw it";
    return verdict;
  }

  verdict.freshness = SceneFreshness::Stale;
  verdict.plan_against = false;
  verdict.note = "it is " + std::to_string(age_s) + " s old by its own header stamp, past "
    "max_scene_age = " + std::to_string(max_scene_age_s) +
    " s. The subscription is transient-local, so a scene published once is latched and handed "
    "over again at every start; this one has not been republished since, and a latch is not a "
    "measurement of the world now";
  if (avoid_collisions) {
    verdict.refuse = true;
    verdict.note += ". avoid_collisions is set, so the request is refused rather than answered "
      "with a trajectory that reads as checked against geometry that may have moved -- the same "
      "answer this planner gives when avoid_collisions is set and no scene has arrived at all. "
      "Republish the scene, or clear avoid_collisions to ask for a plan that says in as many "
      "words that nothing was checked";
  } else {
    verdict.note += ". avoid_collisions is clear, so the scene was not going to bound this path "
      "anyway: the request proceeds without it and this note is the whole of the consequence";
  }
  return verdict;
}

}  // namespace crane_planning
