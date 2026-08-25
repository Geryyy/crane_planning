// The timing ramp, and the refusals the ROS-free core owes.
//
// The ramp is deliberately trivial -- one scaled interpolation between the two
// endpoints, obeying the velocity limits and nothing else. What it still has to
// be is C1 at both ends and at rest there, because a reference that starts with
// a velocity step is one the inner loop cannot follow whatever the planner does
// afterwards. The real time parametrization, with the sway carried explicitly
// and the force, flow and kappa constraints, is issue 043.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>

#include "crane_planning/joint_limits.hpp"
#include "crane_planning/planner_core.hpp"
#include "crane_planning/trajectory_timing.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning_test::Machine;

crane_planning::JointLimits machine_limits(const Machine & machine)
{
  const crane_model::Model model = crane_planning_test::build_model(machine);
  auto limits = crane_planning::read_joint_limits(
    crane_planning_test::description(machine), model.urdf_joint_names());
  if (!limits.ok()) {
    throw std::runtime_error(limits.status().message);
  }
  return std::move(limits).value();
}

crane_model::QA travel_goal(const crane_model::QA & start)
{
  crane_model::QA goal = start;
  goal[0] += 1.1;
  goal[1] += 0.4;
  goal[2] -= 0.7;
  goal[3] += 0.9;
  goal[4] += 2.0;
  goal[5] += 0.05;
  return goal;
}

}  // namespace

TEST(ScaledRamp, ItStartsAndEndsAtRestAndIsC1There)
{
  const crane_planning::JointLimits limits =
    machine_limits(crane_planning_test::machines().front());
  const crane_model::QA start = crane_model::QA::Zero();
  const crane_model::QA goal = travel_goal(start);

  crane_planning::RampSettings settings;
  auto ramp = crane_planning::scaled_ramp(start, goal, limits, 1.0, settings);
  ASSERT_TRUE(ramp.ok()) << ramp.status().message;
  const crane_planning::TimedTrajectory & trajectory = ramp.value();

  ASSERT_GE(trajectory.q_a_ref.size(), 2U);
  EXPECT_DOUBLE_EQ(trajectory.time_from_start.front(), 0.0);
  EXPECT_NEAR(trajectory.time_from_start.back(), trajectory.duration, 1.0e-12);
  EXPECT_LT((trajectory.q_a_ref.front() - start).norm(), 1.0e-12);
  EXPECT_LT((trajectory.q_a_ref.back() - goal).norm(), 1.0e-12);
  EXPECT_LT(trajectory.dq_a_ref.front().norm(), 1.0e-12);
  EXPECT_LT(trajectory.dq_a_ref.back().norm(), 1.0e-12);

  // C1: the emitted velocity is the derivative of the emitted position, so a
  // central difference of the samples has to land on it. That is the property
  // the receiving spline of ROS 2 Interfaces 7 depends on.
  for (std::size_t index = 1; index + 1 < trajectory.q_a_ref.size(); ++index) {
    const double span =
      trajectory.time_from_start[index + 1] - trajectory.time_from_start[index - 1];
    const crane_model::QA difference =
      (trajectory.q_a_ref[index + 1] - trajectory.q_a_ref[index - 1]) / span;
    EXPECT_LT((difference - trajectory.dq_a_ref[index]).norm(), 1.0e-3)
      << "point " << index;
  }
}

TEST(ScaledRamp, NoAxisExceedsTheScaledVelocityLimit)
{
  const crane_planning::JointLimits limits =
    machine_limits(crane_planning_test::machines().front());
  const crane_model::QA start = crane_model::QA::Zero();
  const crane_model::QA goal = travel_goal(start);

  for (const double margin_factor : {1.0, 0.5, 0.1}) {
    auto ramp = crane_planning::scaled_ramp(
      start, goal, limits, margin_factor, crane_planning::RampSettings{});
    ASSERT_TRUE(ramp.ok()) << ramp.status().message;
    for (const crane_model::DQA & dq_a : ramp.value().dq_a_ref) {
      for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
        EXPECT_LE(
          std::abs(dq_a[static_cast<Eigen::Index>(row)]),
          margin_factor * limits.axis[row].dq_max * (1.0 + 1.0e-9)) << "axis " << row;
      }
    }
    // At least one axis is actually at the bound, or the duration was not
    // decided by the limits at all.
    EXPECT_NEAR(ramp.value().limiting_fraction, 1.0, 1.0e-9) << margin_factor;
  }
}

TEST(ScaledRamp, HalvingTheScaleDoublesTheDuration)
{
  const crane_planning::JointLimits limits =
    machine_limits(crane_planning_test::machines().front());
  const crane_model::QA start = crane_model::QA::Zero();
  const crane_model::QA goal = travel_goal(start);

  auto full = crane_planning::scaled_ramp(
    start, goal, limits, 1.0, crane_planning::RampSettings{});
  auto half = crane_planning::scaled_ramp(
    start, goal, limits, 0.5, crane_planning::RampSettings{});
  ASSERT_TRUE(full.ok() && half.ok());
  EXPECT_NEAR(half.value().duration, 2.0 * full.value().duration, 1.0e-9);
}

TEST(ScaledRamp, AScaleOutsideTheUnitIntervalIsRefused)
{
  const crane_planning::JointLimits limits =
    machine_limits(crane_planning_test::machines().front());
  for (const double margin_factor : {0.0, -0.5, 1.0001, 2.0}) {
    auto ramp = crane_planning::scaled_ramp(
      crane_model::QA::Zero(), crane_model::QA::Ones(), limits, margin_factor,
      crane_planning::RampSettings{});
    EXPECT_FALSE(ramp.ok()) << margin_factor;
    EXPECT_NE(ramp.status().message.find("speed_scale"), std::string::npos) << margin_factor;
  }
  EXPECT_TRUE(crane_planning::check_margin_factor(1.0).ok());
}

TEST(PlanMotion, CollisionAvoidanceIsHonouredAndItsAbsenceIsSaidOutLoud)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);

  // Start and goal in the middle of the machine's own ranges. This test used to
  // start at every joint zero, where the PZS100's rail gripper lies against the
  // arm: with a real self-collision check that path is refused, correctly, and
  // this test is not about that.
  const crane_model::QA start = crane_planning_test::centred(context.limits, machine);
  const crane_model::Q goal =
    crane_planning_test::settled(model, crane_planning_test::moved(start, context.limits));
  auto goal_pose = model.forward_kinematics(
    goal, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  ASSERT_TRUE(goal_pose.ok()) << goal_pose.status().message;

  crane_planning::MotionRequest request;
  request.p_tcp_0 = goal_pose.value().position_m;
  request.q_a_start = start;
  request.payload = crane_planning_test::empty_gripper();
  request.avoid_collisions = true;  // which is what the .srv defaults to

  // No scene has been received, so there is nothing to check against and the
  // request is refused rather than answered with a trajectory that would read as
  // collision-checked.
  auto refused = crane_planning::plan_motion(model, context, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("/crane/collision_scene"), std::string::npos)
    << refused.status().message;

  // The same request against a scene is answerable, so the refusal is about the
  // missing scene and not about the goal.
  const crane_model::CollisionScene empty;
  request.scene = &empty;
  auto checked = crane_planning::plan_motion(model, context, request);
  ASSERT_TRUE(checked.ok()) << checked.status().message;
  EXPECT_TRUE(checked.value().primitive.check.checked);
  EXPECT_GT(checked.value().primitive.check.path.samples, 2U);

  // And the same request without the demand is answerable too, and says in as
  // many words that nothing was checked.
  request.scene = nullptr;
  request.avoid_collisions = false;
  auto planned = crane_planning::plan_motion(model, context, request);
  ASSERT_TRUE(planned.ok()) << planned.status().message;
  EXPECT_FALSE(planned.value().primitive.check.checked);
  EXPECT_NE(planned.value().primitive.check.note.find("Nothing was checked"), std::string::npos)
    << planned.value().primitive.check.note;
}

TEST(PlanMotion, TheScaleIsRefusedBeforeTheSolveRatherThanAfterIt)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);

  crane_planning::MotionRequest request;
  // A goal no arm reaches, so a refusal that mentions the scale can only have
  // come from before the inverse kinematics ran.
  request.p_tcp_0 = Eigen::Vector3d(500.0, 0.0, 0.0);
  request.q_a_start = crane_model::QA::Zero();
  request.payload = crane_planning_test::empty_gripper();
  request.avoid_collisions = false;
  request.speed_scale = 1.5;

  auto refused = crane_planning::plan_motion(model, context, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("speed_scale"), std::string::npos)
    << refused.status().message;
}

TEST(PlanMotion, AGoalOutsideTheWorkspaceIsRefusedRatherThanApproximated)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);

  crane_planning::MotionRequest request;
  request.p_tcp_0 = Eigen::Vector3d(500.0, 0.0, 0.0);
  request.q_a_start = crane_model::QA::Zero();
  request.payload = crane_planning_test::empty_gripper();
  request.avoid_collisions = false;

  auto refused = crane_planning::plan_motion(model, context, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_FALSE(refused.status().message.empty());
}
