// The `a2b_movement` adapter of `wiki/implementation/ros2_interfaces.md` 9, field
// by field and in both directions.
//
// Offline. Nothing here constructs a node, initialises rclcpp, opens a domain or
// plans: the two translations are message in, message out, and the far side of
// the boundary -- `PlannerNode::read_payload` -- is a static function. The one
// thing that is read off the machine is the settled offset from the tip pivot K5
// to the tool, and it comes out of the two checked-in descriptions. The served
// row is `test_plan_motion_service.cpp`.
//
// The requests below are not invented. `timber_request` is what
// `epsilon_crane_behavior_tree`'s `CalcA2BMovementService::on_tick` builds for
// the `load_log.xml` legs -- including the `s_log_8.z` it deliberately leaves at
// NaN and the `p_cyl_8.x` it copies out of `s_log_8.x` -- and
// `feasibility_request` is what `concrete_block_assembly_planning`'s
// `check_pose_feasibility.py` sends, `q0` and all.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "crane_planning/a2b_adapter.hpp"
#include "crane_planning/collision.hpp"
#include "crane_planning/planner_node.hpp"
#include "description_fixture.hpp"
#include "timber_crane_planning_interfaces/srv/calc_movement.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace
{

using crane_msgs::srv::PlanMotion;
using timber_crane_planning_interfaces::srv::CalcMovement;

/// An offset that is not any machine's, so a case that drops an axis is not silent.
const Eigen::Vector3d kOffset{0.20, -0.30, -1.25};

/// What `CalcA2BMovementService::on_tick` builds for a leg that carries nothing.
/**
 * `move_empty.xml` and `load_log.xml`'s "Calc approach log": the gripper is
 * open, `tEnd` is 0, `slowDown` comes off the blackboard, no `logsScene` port is
 * bound so the vector is empty, and `q0`/`q0_dot` are never touched at all.
 * `logCarrying` *is* filled -- it describes the log about to be picked up -- and
 * `on_tick` copies it into `coll_shape` and `s_log_8.x` into `p_cyl_8.x`
 * whatever `carriesLog` says.
 */
CalcMovement::Request timber_request()
{
  CalcMovement::Request request;
  request.y_n.x = 4.2;
  request.y_n.y = 1.1;
  request.y_n.z = 3.4;
  request.phi_tool_n = 0.7;
  request.t_end = 0.0;
  request.carries_log = false;
  request.slow_down = 1.0;
  request.log_carrying.length = 3.0F;
  request.log_carrying.radius_top = 0.15F;
  request.log_carrying.radius_bottom = 0.15F;
  request.m_log = 505.0 * 0.15 * 0.15 * 3.0 * M_PI;
  request.s_log_8.x = 0.0;
  request.s_log_8.y = 0.0;
  // The port's own default, and the reason the adapter reads a non-finite
  // component of `s_log_8` as "not specified" rather than refusing it.
  request.s_log_8.z = std::numeric_limits<double>::quiet_NaN();
  request.coll_shape = request.log_carrying;
  request.p_cyl_8.x = request.s_log_8.x;
  request.check_log_collision = true;
  request.check_gripper_collision = true;
  request.publish_path = true;
  return request;
}

/// The same caller's carrying leg: `load_log.xml`'s "Calc approach target".
CalcMovement::Request timber_carrying_request()
{
  CalcMovement::Request request = timber_request();
  request.carries_log = true;
  request.s_log_8.x = 0.35;
  request.p_cyl_8.x = request.s_log_8.x;
  return request;
}

/// What `check_pose_feasibility.py` sends, including the `q0` it pins.
CalcMovement::Request feasibility_request()
{
  CalcMovement::Request request;
  request.y_n.x = 2.0;
  request.y_n.y = -0.4;
  request.y_n.z = 2.0;
  request.phi_tool_n = 0.785;
  request.t_end = 0.0;
  request.carries_log = false;
  request.slow_down = 1.0;
  request.q0 = {0.785, 0.523599, 0.523602, 0.25, 0.546470, 1.570521, 0.0, 1.0};
  request.publish_path = false;
  request.check_log_collision = false;
  request.check_gripper_collision = true;
  return request;
}

/// One concrete block, declared the way a native caller declares it.
crane_msgs::msg::Payload concrete_block()
{
  crane_msgs::msg::Payload payload;
  payload.shape = crane_msgs::msg::Payload::SHAPE_BOX;
  payload.dimensions.x = 0.6;
  payload.dimensions.y = 0.4;
  payload.dimensions.z = 0.2;
  payload.mass = 120.0;
  payload.com.z = -0.2;
  return payload;
}

}  // namespace

TEST(A2bAdapter, TheServiceNameIsTheOneTheRetainedCallersResolve)
{
  // `mp_rviz_panel` and every timber behaviour tree ask for the relative
  // `a2b_movement` from a node in the root namespace, and
  // `check_pose_feasibility.py` asks for the absolute name outright. Renaming it
  // under ROS 2 Interfaces 1's `/crane/...` rule would be retiring the service
  // rather than keeping it compatible, and 1's rule is for *new* contracts.
  EXPECT_EQ(std::string(crane_planning::kA2bMovementService), "/a2b_movement");
}

TEST(A2bAdapter, TheSettledOffsetFromTheTipPivotToTheToolIsReadOutOfEachDescription)
{
  // `y_n` is the tip pivot K5 and `crane_msgs/PlanMotion::goal` is the tool, so
  // the adapter needs the vector between them. The 7040's asymmetric mass makes
  // that vector measurably horizontal even with an empty gripper, which is why
  // it is read in 3D rather than reduced to a vertical drop.
  for (const auto & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    for (const double phi_z : {0.0, 0.7, -1.1}) {
      Eigen::Vector3d offset;
      std::string why;
      ASSERT_TRUE(
        crane_planning::tip_to_tcp_offset(
          model, crane_planning_test::empty_gripper(), phi_z, machine.q8, offset, why))
        << machine.name << ": " << why;
      EXPECT_GT(offset.norm(), 0.0) << machine.name;
      EXPECT_LT(offset.norm(), 5.0) << machine.name << ": " << offset.norm()
                                    << " m is not a tool, it is an arm";

      crane_model::QA canonical_q_a = crane_model::QA::Zero();
      canonical_q_a[5] = machine.q8;
      auto canonical_equilibrium =
        model.passive_equilibrium(canonical_q_a, crane_planning_test::empty_gripper());
      ASSERT_TRUE(canonical_equilibrium.ok()) << machine.name;
      crane_model::Q canonical_q = crane_model::Q::Zero();
      canonical_q[7] = machine.q8;
      canonical_q.segment<2>(4) = canonical_equilibrium.value();
      auto canonical_tcp = model.forward_kinematics(
        canonical_q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
      ASSERT_TRUE(canonical_tcp.ok()) << machine.name;

      crane_model::QA q_a = crane_model::QA::Zero();
      q_a[5] = machine.q8;
      q_a[0] = std::remainder(
        phi_z - crane_planning::phi_z_of(canonical_tcp.value().orientation), 2.0 * M_PI);
      auto equilibrium = model.passive_equilibrium(q_a, crane_planning_test::empty_gripper());
      ASSERT_TRUE(equilibrium.ok()) << machine.name << ": " << equilibrium.status().message;
      crane_model::Q q = crane_model::Q::Zero();
      q[0] = q_a[0];
      q[7] = machine.q8;
      q.segment<2>(4) = equilibrium.value();
      auto tip =
        model.forward_kinematics(q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
      auto tcp =
        model.forward_kinematics(q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
      ASSERT_TRUE(tip.ok() && tcp.ok()) << machine.name;
      EXPECT_TRUE(offset.isApprox(tcp.value().position_m - tip.value().position_m, 1.0e-12))
        << machine.name << ": " << offset.transpose();
      EXPECT_NEAR(crane_planning::phi_z_of(tcp.value().orientation), phi_z, 1.0e-12)
        << machine.name;
    }
  }
}

TEST(A2bAdapter, TheTimberBehaviourTreesRequestTranslatesFieldForField)
{
  PlanMotion::Request plan;
  std::string why;
  ASSERT_TRUE(crane_planning::translate_a2b_request(timber_request(), kOffset, plan, why)) << why;

  // The frame `CalcMovement` has no field for, asserted and not converted:
  // ROS 2 Interfaces 4's two conversion sites stay two.
  EXPECT_EQ(plan.goal.header.frame_id, std::string(crane_planning::kPlanningFrame));

  // `y_n` is the tip pivot; the goal is the tool at the full settled 3D offset.
  EXPECT_DOUBLE_EQ(plan.goal.pose.position.x, 4.2 + kOffset.x());
  EXPECT_DOUBLE_EQ(plan.goal.pose.position.y, 1.1 + kOffset.y());
  EXPECT_DOUBLE_EQ(plan.goal.pose.position.z, 3.4 + kOffset.z());

  // phi_z about K0's z, scalar-last on the wire and scalar-first in Eigen.
  const Eigen::Quaterniond orientation(
    plan.goal.pose.orientation.w, plan.goal.pose.orientation.x, plan.goal.pose.orientation.y,
    plan.goal.pose.orientation.z);
  EXPECT_NEAR(crane_planning::phi_z_of(orientation), 0.7, 1.0e-12);

  // `slow_down` is a divider and `speed_scale` is a factor.
  EXPECT_DOUBLE_EQ(plan.speed_scale, 1.0);
  // Both flags are true, and there is no log, so the gripper's decides.
  EXPECT_TRUE(plan.avoid_collisions);
  // An open gripper is `SHAPE_NONE`, whatever `logCarrying` describes: on this
  // leg it is the log the tree is about to pick up, not one it is holding.
  EXPECT_EQ(plan.payload.shape, crane_msgs::msg::Payload::SHAPE_NONE);
}

TEST(A2bAdapter, TheDividerBecomesAFactorAndABelowOneDividerIsRefused)
{
  CalcMovement::Request request = timber_request();
  request.slow_down = 4.0;
  PlanMotion::Request plan;
  std::string why;
  ASSERT_TRUE(crane_planning::translate_a2b_request(request, kOffset, plan, why)) << why;
  EXPECT_DOUBLE_EQ(plan.speed_scale, 0.25);

  // kappa is the deployment's reservation (trajectory_planning 5.5) and a
  // divider below one asks to spend it.
  for (const double slow_down : {0.5, 0.0, -1.0}) {
    request.slow_down = slow_down;
    EXPECT_FALSE(crane_planning::translate_a2b_request(request, kOffset, plan, why)) << slow_down;
    EXPECT_NE(why.find("slow_down"), std::string::npos) << why;
  }
}

TEST(A2bAdapter, ACylinderOnTheAdapterPathStaysACylinderAndABlockOnTheNativePathStaysABox)
{
  // The whole point of the adapter: the fiction stops at its boundary. A
  // `wood_log_msgs/LogShape` becomes a `crane_msgs/Payload` cylinder here and
  // reaches the planner as a cylinder -- and a block declared natively is a box
  // all the way down, rather than the cylinder the legacy interface forced it to
  // be declared as.
  PlanMotion::Request plan;
  std::string why;
  ASSERT_TRUE(
    crane_planning::translate_a2b_request(timber_carrying_request(), kOffset, plan, why)) << why;
  EXPECT_EQ(plan.payload.shape, crane_msgs::msg::Payload::SHAPE_CYLINDER);
  // `dimensions` is the extent per axis of the primitive's own frame
  // (ROS 2 Interfaces 6): (2r, 2r, length), never a radius in the first entry.
  EXPECT_NEAR(plan.payload.dimensions.x, 0.3, 1.0e-6);
  EXPECT_NEAR(plan.payload.dimensions.y, 0.3, 1.0e-6);
  EXPECT_NEAR(plan.payload.dimensions.z, 3.0, 1.0e-6);
  EXPECT_NEAR(plan.payload.mass, 505.0 * 0.15 * 0.15 * 3.0 * M_PI, 1.0e-9);
  // One `com`, taken from `p_cyl_8` -- the component the tree left at NaN comes
  // from there rather than being invented or refused.
  EXPECT_DOUBLE_EQ(plan.payload.com.x, 0.35);
  EXPECT_DOUBLE_EQ(plan.payload.com.y, 0.0);
  EXPECT_DOUBLE_EQ(plan.payload.com.z, 0.0);

  crane_model::Payload payload;
  crane_planning::PayloadShape shape;
  ASSERT_TRUE(crane_planning::PlannerNode::read_payload(plan.payload, payload, shape, why)) << why;
  EXPECT_TRUE(shape.declared);
  EXPECT_EQ(shape.shape, crane_model::CollisionShape::Cylinder);
  EXPECT_NE(shape.shape, crane_model::CollisionShape::Box);

  // ...and the native path, on the same far side of the same boundary.
  crane_model::Payload block_payload;
  crane_planning::PayloadShape block_shape;
  ASSERT_TRUE(
    crane_planning::PlannerNode::read_payload(
      concrete_block(), block_payload, block_shape, why)) << why;
  EXPECT_EQ(block_shape.shape, crane_model::CollisionShape::Box);
  EXPECT_NE(block_shape.shape, crane_model::CollisionShape::Cylinder);
  EXPECT_DOUBLE_EQ(block_shape.dimensions_m.x(), 0.6);
  EXPECT_DOUBLE_EQ(block_shape.dimensions_m.y(), 0.4);
  EXPECT_DOUBLE_EQ(block_shape.dimensions_m.z(), 0.2);
  EXPECT_DOUBLE_EQ(block_payload.mass_kg, 120.0);
}

TEST(A2bAdapter, ATaperedLogBecomesTheEnclosingCylinderAndNotTheLegacyMean)
{
  // `wood_log_msgs/LogShape` is a truncated cone and `crane_msgs/Payload` carries
  // one radius. The legacy planner averaged the two, which leaves the wider end
  // outside its own collision body; the enclosing cylinder is the bound that does
  // not, and the mass stays the caller's own number rather than being recomputed
  // from a radius nobody sent.
  CalcMovement::Request request = timber_carrying_request();
  request.log_carrying.radius_top = 0.10F;
  request.log_carrying.radius_bottom = 0.20F;
  request.coll_shape = request.log_carrying;

  PlanMotion::Request plan;
  std::string why;
  ASSERT_TRUE(crane_planning::translate_a2b_request(request, kOffset, plan, why)) << why;
  EXPECT_NEAR(plan.payload.dimensions.x, 0.4, 1.0e-6);
  EXPECT_NEAR(plan.payload.dimensions.y, 0.4, 1.0e-6);
  EXPECT_NEAR(plan.payload.mass, timber_carrying_request().m_log, 1.0e-9);
}

TEST(A2bAdapter, TwoDifferentBodiesAreRefusedNamingBothFields)
{
  PlanMotion::Request plan;
  std::string why;

  // `log_carrying` is the inertia shape and `coll_shape` is the collision shape,
  // and there is one `crane_msgs/Payload` for both.
  CalcMovement::Request shapes = timber_carrying_request();
  shapes.coll_shape.length = 2.0F;
  EXPECT_FALSE(crane_planning::translate_a2b_request(shapes, kOffset, plan, why));
  EXPECT_NE(why.find("log_carrying"), std::string::npos) << why;
  EXPECT_NE(why.find("coll_shape"), std::string::npos) << why;

  // ...and one `com` for the centre of mass and the collision body's centre.
  CalcMovement::Request centres = timber_carrying_request();
  centres.p_cyl_8.x = centres.s_log_8.x + 0.2;
  EXPECT_FALSE(crane_planning::translate_a2b_request(centres, kOffset, plan, why));
  EXPECT_NE(why.find("s_log_8"), std::string::npos) << why;
  EXPECT_NE(why.find("p_cyl_8"), std::string::npos) << why;
}

TEST(A2bAdapter, ACarriedLogWithNoShapeOrNoMassIsRefusedRatherThanGuessed)
{
  PlanMotion::Request plan;
  std::string why;

  CalcMovement::Request massless = timber_carrying_request();
  massless.m_log = 0.0;
  EXPECT_FALSE(crane_planning::translate_a2b_request(massless, kOffset, plan, why));
  EXPECT_NE(why.find("m_log"), std::string::npos) << why;

  CalcMovement::Request shapeless = timber_carrying_request();
  shapeless.log_carrying.length = 0.0F;
  shapeless.coll_shape = shapeless.log_carrying;
  EXPECT_FALSE(crane_planning::translate_a2b_request(shapeless, kOffset, plan, why));
  EXPECT_NE(why.find("log_carrying"), std::string::npos) << why;
}

TEST(A2bAdapter, AFieldWithNoNativeEquivalentIsRefusedNamingIt)
{
  PlanMotion::Request plan;
  std::string why;

  // Obstacles arrive on `/crane/collision_scene`; a request cannot carry its
  // own, and dropping the ones it carries would plan through them.
  CalcMovement::Request scene = timber_request();
  scene.logs_scene.emplace_back();
  EXPECT_FALSE(crane_planning::translate_a2b_request(scene, kOffset, plan, why));
  EXPECT_NE(why.find("logs_scene"), std::string::npos) << why;

  // The OCP of trajectory_planning 5.2 derives the duration.
  CalcMovement::Request fixed_time = timber_request();
  fixed_time.t_end = 12.0;
  EXPECT_FALSE(crane_planning::translate_a2b_request(fixed_time, kOffset, plan, why));
  EXPECT_NE(why.find("t_end"), std::string::npos) << why;

  // 5.4's terminal condition brings the tool to rest hanging still.
  CalcMovement::Request moving_end = timber_request();
  moving_end.v_d_tip = {0.0, 0.0, 0.4};
  EXPECT_FALSE(crane_planning::translate_a2b_request(moving_end, kOffset, plan, why));
  EXPECT_NE(why.find("v_d_tip"), std::string::npos) << why;

  // Checking the gripper but not the log it holds is not a question this
  // planner's one `avoid_collisions` can be asked.
  CalcMovement::Request mixed = timber_carrying_request();
  mixed.check_log_collision = false;
  EXPECT_FALSE(crane_planning::translate_a2b_request(mixed, kOffset, plan, why));
  EXPECT_NE(why.find("check_log_collision"), std::string::npos) << why;
  EXPECT_NE(why.find("check_gripper_collision"), std::string::npos) << why;

  // And a goal that is not four finite numbers is not a goal.
  CalcMovement::Request nowhere = timber_request();
  nowhere.y_n.z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(crane_planning::translate_a2b_request(nowhere, kOffset, plan, why));
  EXPECT_NE(why.find("y_n"), std::string::npos) << why;
}

TEST(A2bAdapter, TheFeasibilityCheckersPinnedStartMapsAllEightCanonicalRows)
{
  // `check_pose_feasibility.py` probes reachability from a canned configuration
  // rather than from where the machine is. The retained field maps onto the same
  // internal MeasuredStart the native service fills from topics, so the checker
  // keeps working without weakening the native service's measured-start rule.
  const CalcMovement::Request pinned = feasibility_request();
  std::optional<crane_planning::MeasuredStart> start;
  PlanMotion::Request plan;
  std::string why;
  ASSERT_TRUE(crane_planning::translate_a2b_start(pinned, start, why)) << why;
  ASSERT_TRUE(start.has_value());
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    EXPECT_DOUBLE_EQ(
      start->q_a[static_cast<Eigen::Index>(row)], pinned.q0[crane_planning::kActuatedRows[row]]);
    EXPECT_DOUBLE_EQ(start->dq_a[static_cast<Eigen::Index>(row)], 0.0);
  }
  EXPECT_TRUE(start->passive.measured);
  EXPECT_DOUBLE_EQ(start->passive.q_u[0], pinned.q0[4]);
  EXPECT_DOUBLE_EQ(start->passive.q_u[1], pinned.q0[5]);
  ASSERT_TRUE(crane_planning::translate_a2b_request(pinned, kOffset, plan, why)) << why;

  // With the pin removed, the all-zero default selects the live measured start.
  // The mixed collision flags are not a refusal either: with an open gripper
  // there is no log to check, so the gripper's flag is the one that decides.
  CalcMovement::Request measured = feasibility_request();
  measured.q0 = {};
  ASSERT_TRUE(crane_planning::translate_a2b_start(measured, start, why)) << why;
  EXPECT_FALSE(start.has_value());
  ASSERT_TRUE(crane_planning::translate_a2b_request(measured, kOffset, plan, why)) << why;
  EXPECT_TRUE(plan.avoid_collisions);
  EXPECT_EQ(plan.payload.shape, crane_msgs::msg::Payload::SHAPE_NONE);

  CalcMovement::Request rates = feasibility_request();
  rates.q0 = {};
  rates.q0_dot[2] = 0.1;
  EXPECT_FALSE(crane_planning::translate_a2b_request(rates, kOffset, plan, why));
  EXPECT_NE(why.find("q0_dot"), std::string::npos) << why;
}

TEST(A2bAdapter, AnOffsetThatWasNeverReadIsARefusalAndNotAGoalOnThePivot)
{
  PlanMotion::Request plan;
  std::string why;
  EXPECT_FALSE(
    crane_planning::translate_a2b_request(
      timber_request(), Eigen::Vector3d::Zero(), plan, why));
  EXPECT_NE(why.find("y_n"), std::string::npos) << why;
}

TEST(A2bAdapter, TheAnswerComesBackWithSuccessOnBothSidesTheSameBool)
{
  // ROS 2 Interfaces 1's `[!warning]` records that two service types in the
  // retained stack disagree about the type of `success`. They are `CalcMovement`
  // (`bool`) and `CalcGripMovement` (`int64`), and 9 names only the first -- so
  // what this adapter reconciles is bool to bool with the same meaning, and the
  // int64 belongs to a service it does not serve.
  PlanMotion::Response planned;
  planned.success = true;
  planned.message = "planned";
  planned.trajectory.joint_names = {"theta1_slewing_joint"};
  planned.trajectory.points.resize(2U);
  planned.tcp_path.resize(2U);

  CalcMovement::Response response;
  crane_planning::translate_a2b_response(planned, response);
  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.trajectory.joint_names, planned.trajectory.joint_names);
  EXPECT_EQ(response.trajectory.points.size(), 2U);
  EXPECT_EQ(response.tcp_path.size(), 2U);
}

TEST(A2bAdapter, ARefusalCarriesNoTrajectoryBecauseThereIsNoMessageToExplainOne)
{
  // `/crane/plan_motion` hands a refused caller whatever is still standing on
  // `/crane/reference` and says so in `message`. `CalcMovement::Response` has no
  // `message`, so a caller could not tell that trajectory from a new one -- and
  // the legacy server left the field empty on failure.
  PlanMotion::Response planned;
  planned.success = false;
  planned.message = "no passive start state. The trajectory standing on /crane/reference is "
    "unchanged and is the one still being executed";
  planned.trajectory.joint_names = {"theta1_slewing_joint"};
  planned.trajectory.points.resize(4U);

  CalcMovement::Response response;
  response.tcp_path.resize(3U);
  crane_planning::translate_a2b_response(planned, response);
  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.trajectory.points.empty());
  EXPECT_TRUE(response.trajectory.joint_names.empty());
  EXPECT_TRUE(response.tcp_path.empty());
}
