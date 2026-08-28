// The freshness of `/crane/collision_scene`, offline.
//
// Three states the planner has to tell apart at every request -- never received,
// fresh, stale -- and, for the stale one, the two consequences that depend on
// what the caller asked for. Nothing here launches, subscribes or waits: the
// decision is arithmetic on one age against one bound, which is why it lives in
// `scene_age.hpp` rather than inside the node
// (`wiki/implementation/style_guide.md` 3).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "crane_planning/scene_age.hpp"

namespace
{

using crane_planning::SceneFreshness;
using crane_planning::SceneVerdict;
using crane_planning::judge_scene;

/// The deployed bound, `config/crane_planner.yaml`.
constexpr double kMaxSceneAge = 10.0;

/// `haystack` contains `needle`.
bool mentions(const std::string & haystack, const std::string & needle)
{
  return haystack.find(needle) != std::string::npos;
}

/// Both values of `avoid_collisions`, so a case can say "either way" once.
constexpr bool kBothWays[] = {false, true};

}  // namespace

TEST(SceneAge, NothingReceivedIsNotStaleAndSaysNothing)
{
  // "Never received" is the core's case and it already refuses it by name when
  // `avoid_collisions` is set. What this asserts is that the age check does not
  // add a second, differently worded sentence about the same absence -- and, in
  // particular, that it does not report an absent scene as *stale*, which would
  // send an operator looking for a world model that never published rather than
  // for one that never started.
  for (const bool avoid : kBothWays) {
    const SceneVerdict verdict = judge_scene(false, 0.0, kMaxSceneAge, avoid);
    EXPECT_EQ(verdict.freshness, SceneFreshness::NeverReceived);
    EXPECT_FALSE(verdict.plan_against);
    EXPECT_FALSE(verdict.refuse) << "the core owns this refusal, not the age check";
    EXPECT_TRUE(verdict.note.empty()) << verdict.note;
  }
}

TEST(SceneAge, AFreshSceneIsPlannedAgainstAndItsAgeTravelsWithTheAnswer)
{
  for (const bool avoid : kBothWays) {
    const SceneVerdict verdict = judge_scene(true, 1.5, kMaxSceneAge, avoid);
    EXPECT_EQ(verdict.freshness, SceneFreshness::Fresh);
    EXPECT_TRUE(verdict.plan_against);
    EXPECT_FALSE(verdict.refuse);
    // The age is evidence and not decoration: `crane_msgs/PlanMotion` is frozen
    // and carries no structured field for it, so the prose `message` is the only
    // surface a caller has to read it off.
    EXPECT_TRUE(mentions(verdict.note, "1.500000"));
    EXPECT_TRUE(mentions(verdict.note, "max_scene_age"));
  }
}

TEST(SceneAge, TheBoundItselfIsStillFresh)
{
  // `age <= bound`, on the same terms as `max_input_age` in the node: the bound
  // is the longest age that still counts, not the first that does not.
  EXPECT_EQ(judge_scene(true, kMaxSceneAge, kMaxSceneAge, true).freshness, SceneFreshness::Fresh);
  EXPECT_EQ(
    judge_scene(true, std::nextafter(kMaxSceneAge, 1.0e9), kMaxSceneAge, true).freshness,
    SceneFreshness::Stale);
}

TEST(SceneAge, AStaleSceneWithCollisionAvoidanceIsRefusedAndNamesTheLatch)
{
  const SceneVerdict verdict = judge_scene(true, 42.0, kMaxSceneAge, true);
  EXPECT_EQ(verdict.freshness, SceneFreshness::Stale);
  EXPECT_TRUE(verdict.refuse);
  // The invariant the goal of issue 096 asks for: a scene the planner plans
  // against is fresh. A stale one is never handed to the core, refusal or not.
  EXPECT_FALSE(verdict.plan_against);
  EXPECT_TRUE(mentions(verdict.note, "42.000000"));
  EXPECT_TRUE(mentions(verdict.note, "max_scene_age"));
  // Why it went stale, and not only that it did. The subscription is
  // transient-local depth 1, so "the world model stopped publishing" and "the
  // world model is publishing and you are reading a latch" look identical from
  // here, and an operator has to be told which one to go and check.
  EXPECT_TRUE(mentions(verdict.note, "transient-local"));
  EXPECT_TRUE(mentions(verdict.note, "avoid_collisions"));
}

TEST(SceneAge, AStaleSceneWithoutCollisionAvoidanceWarnsAndKeepsPlanning)
{
  // The decision this issue had to make, asserted rather than described: a
  // request that declined to use the scene is not stopped by the scene being
  // old. It is still not planned against, and the answer still says so.
  const SceneVerdict verdict = judge_scene(true, 42.0, kMaxSceneAge, false);
  EXPECT_EQ(verdict.freshness, SceneFreshness::Stale);
  EXPECT_FALSE(verdict.refuse);
  EXPECT_FALSE(verdict.plan_against);
  EXPECT_TRUE(mentions(verdict.note, "42.000000"));
  EXPECT_TRUE(mentions(verdict.note, "avoid_collisions is clear"));
}

TEST(SceneAge, ALatchedSceneWithNoStampReadsAsStaleRatherThanAsBrandNew)
{
  // The case the whole check exists for. A publisher that leaves `header.stamp`
  // at zero produces an age of "however long this ROS graph has been up", which
  // is large and positive and reads as stale on its own. A non-finite age --
  // which is what a clock that has not started yet leaves behind -- has to be
  // pushed onto the stale side deliberately, because every comparison against a
  // NaN is false and the naive `age > bound` test would have called it fresh.
  // (A *negative* age, a stamp from the future, reads as fresh here, on the same
  // terms as `max_input_age`: a producer whose clock is ahead is a clock problem
  // and not a staleness one, and this planner does not diagnose it.)
  EXPECT_EQ(judge_scene(true, 1.7e9, kMaxSceneAge, true).freshness, SceneFreshness::Stale);
  EXPECT_EQ(
    judge_scene(true, std::numeric_limits<double>::quiet_NaN(), kMaxSceneAge, true).freshness,
    SceneFreshness::Stale);
  EXPECT_EQ(
    judge_scene(true, std::numeric_limits<double>::infinity(), kMaxSceneAge, true).freshness,
    SceneFreshness::Stale);
}
