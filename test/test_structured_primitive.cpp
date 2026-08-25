// The structured lift/traverse/descend primitive of
// `wiki/trajectory_planning.md` 4.4, and the two properties 4.5 says a geometric
// path owes stage 2: `q_a'(sigma)` continuous and `q_a''(sigma)` **defined**.
//
// 4.5's claim is behavioural, not numerical. At a point of discontinuous
// curvature the path-velocity limit collapses to `sigma_dot = 0`, so the machine
// stops dead at the waypoint -- the worst possible output for a crane whose
// purpose is smooth, sway-free motion. So the assertions here are on the
// derivatives at the junctions and not on how the path looks.
//
// Those assertions live in `c2_path_assertions.hpp` rather than in this file,
// because 4.4 has **two** producers and 4.5 holds both to the same requirement:
// the primitive is built C2 and the sampled path of `test_sampling_planner.cpp`
// is smoothed into it, and a second copy of the assertions here would be a
// second, weaker definition of what C2 means.
//
// Offline C++ throughout. Nothing here launches, links a simulator or reaches a
// ROS graph; the two machine descriptions are read where `crane_model` keeps
// them.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "c2_path_assertions.hpp"
#include "crane_planning/equilibrium_ik.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/planner_core.hpp"
#include "crane_planning/structured_primitive.hpp"
#include "crane_planning/trajectory_timing.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning::GeometricPath;
using crane_planning::JointLimits;
using crane_planning::PathSample;
using crane_planning::PathVector;
using crane_planning::PrimitivePhase;
using crane_planning::StructuredPrimitive;
using crane_planning_test::Machine;

/// The description's own limits, which is where every configuration below comes from.
JointLimits machine_limits(const crane_model::Model & model, const Machine & machine)
{
  auto limits = crane_planning::read_joint_limits(
    crane_planning_test::description(machine), model.urdf_joint_names());
  if (!limits.ok()) {
    throw std::runtime_error(limits.status().message);
  }
  return std::move(limits).value();
}

// The centred configuration, the move away from it and the hanging pose all live
// in `description_fixture.hpp`, because every offline suite starts from the same
// three and a second copy would be a second machine to keep current.
using crane_planning_test::centred;
using crane_planning_test::settled;

PathVector path_of(const crane_model::Q & q)
{
  PathVector q_a = PathVector::Zero();
  for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] =
      q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])];
  }
  return q_a;
}

using crane_planning_test::moved;

/// The primitive for one machine, from the centred configuration to `moved` of it.
StructuredPrimitive primitive_for(
  const crane_model::Model & model, const Machine & machine,
  const crane_planning::PlannerContext & context, const JointLimits & limits)
{
  const crane_model::QA start = centred(limits, machine);
  crane_planning::PrimitiveRequest request;
  request.q_start = settled(model, start);
  request.q_goal = settled(model, moved(start, limits));
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry

  auto built = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, context.settings.primitive, request);
  if (!built.ok()) {
    throw std::runtime_error(machine.name + std::string(": ") + built.status().message);
  }
  return std::move(built).value();
}

// The C2 assertions themselves are shared with the sampling fallback's suite --
// 4.5 holds both producers to one requirement. See `c2_path_assertions.hpp`.
using crane_planning_test::curvature_scale;
using crane_planning_test::expect_at_rest_and_monotone;
using crane_planning_test::expect_c2_everywhere;

}  // namespace

// --------------------------------------------------------------- the altitude

TEST(TransferAltitude, ItIsDerivedFromTheEndpointsAndTheToolRatherThanFromAConstant)
{
  // Two endpoints at different altitudes and a tool that reaches a metre below
  // the TCP: the derivation lifts the higher endpoint by the tool's own reach,
  // so the tool's lowest point clears where the TCP was.
  auto derived = crane_planning::derive_transfer_altitude(
    2.0, 1.0, 3.0, 1.0, crane_planning::scene_without_obstacles(),
    crane_planning::TransferAltitudeSettings{});
  ASSERT_TRUE(derived.ok()) << derived.status().message;
  EXPECT_DOUBLE_EQ(derived.value().z_m, 4.0);
  EXPECT_DOUBLE_EQ(derived.value().z_from_endpoints_m, 4.0);

  // A longer tool asks for more, which is the whole content of "the transfer
  // altitude comes from the scene": nothing here is a constant that a different
  // machine would still be handed.
  auto longer = crane_planning::derive_transfer_altitude(
    2.0, 1.0, 3.0, 2.5, crane_planning::scene_without_obstacles(),
    crane_planning::TransferAltitudeSettings{});
  ASSERT_TRUE(longer.ok()) << longer.status().message;
  EXPECT_DOUBLE_EQ(longer.value().z_m, 5.5);
  EXPECT_GT(longer.value().z_m, derived.value().z_m);

  // And it always clears both endpoints, which is the property the primitive
  // refuses on when a configured ceiling takes it away.
  EXPECT_GT(derived.value().z_m, 2.0);
  EXPECT_GT(derived.value().z_m, 3.0);
}

TEST(TransferAltitude, TheObstacleTermIsANamedGapAndNotAZero)
{
  const crane_planning::SceneExtent scene = crane_planning::scene_without_obstacles();
  // Not "there are no obstacles" -- "no obstacle extent is known". The two are
  // different facts, and only a scene that was actually received can make the
  // first one.
  EXPECT_FALSE(scene.obstacles_known);
  EXPECT_TRUE(std::isnan(scene.highest_obstacle_z_m));

  auto blind = crane_planning::derive_transfer_altitude(
    2.0, 1.0, 2.0, 1.0, scene, crane_planning::TransferAltitudeSettings{});
  ASSERT_TRUE(blind.ok()) << blind.status().message;
  EXPECT_FALSE(blind.value().obstacle_term_applied);
  EXPECT_TRUE(std::isnan(blind.value().z_from_obstacles_m));
  EXPECT_NE(describe(blind.value()).find("no scene carried an extent"), std::string::npos);

  // And with an extent the term applies, which is the read `scene_extent` makes
  // of a scene that did arrive.
  crane_planning::SceneExtent with_obstacle;
  with_obstacle.obstacles_known = true;
  with_obstacle.highest_obstacle_z_m = 6.0;
  auto seeing = crane_planning::derive_transfer_altitude(
    2.0, 1.0, 2.0, 1.0, with_obstacle, crane_planning::TransferAltitudeSettings{});
  ASSERT_TRUE(seeing.ok()) << seeing.status().message;
  EXPECT_TRUE(seeing.value().obstacle_term_applied);
  EXPECT_DOUBLE_EQ(seeing.value().z_m, 7.0);
}

TEST(TransferAltitude, AConfiguredValueBoundsTheDerivationAndIsNeverTheAnswer)
{
  const crane_planning::SceneExtent scene = crane_planning::scene_without_obstacles();

  // A floor under the derivation changes nothing.
  crane_planning::TransferAltitudeSettings low_floor;
  low_floor.floor_m = 1.0;
  auto ignored = crane_planning::derive_transfer_altitude(2.0, 1.0, 2.0, 1.0, scene, low_floor);
  ASSERT_TRUE(ignored.ok()) << ignored.status().message;
  EXPECT_DOUBLE_EQ(ignored.value().z_m, 3.0);
  EXPECT_FALSE(ignored.value().floor_binding);

  // A floor above it raises it, and says so.
  crane_planning::TransferAltitudeSettings high_floor;
  high_floor.floor_m = 5.0;
  auto raised = crane_planning::derive_transfer_altitude(2.0, 1.0, 2.0, 1.0, scene, high_floor);
  ASSERT_TRUE(raised.ok()) << raised.status().message;
  EXPECT_DOUBLE_EQ(raised.value().z_m, 5.0);
  EXPECT_TRUE(raised.value().floor_binding);
  // Even then the derivation is what was derived: the floor did not replace it.
  EXPECT_DOUBLE_EQ(raised.value().z_from_endpoints_m, 3.0);

  // A ceiling may only lower it.
  crane_planning::TransferAltitudeSettings ceiling;
  ceiling.ceiling_m = 2.5;
  auto lowered = crane_planning::derive_transfer_altitude(2.0, 1.0, 2.0, 1.0, scene, ceiling);
  ASSERT_TRUE(lowered.ok()) << lowered.status().message;
  EXPECT_DOUBLE_EQ(lowered.value().z_m, 2.5);
  EXPECT_TRUE(lowered.value().ceiling_binding);
}

// ------------------------------------------------------------------- the path

TEST(C2Path, EveryJunctionIsContinuousInBothDerivativesAndTheSecondIsDefinedEverywhere)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const JointLimits limits = machine_limits(model, machine);
    const StructuredPrimitive primitive = primitive_for(model, machine, context, limits);
    const GeometricPath & path = primitive.path;

    ASSERT_EQ(path.segment_count(), 3U) << machine.name;
    ASSERT_EQ(path.junction_sigmas().size(), 2U) << machine.name;

    // The requirement itself, from the one place it is written down. The
    // sampling fallback's own suite calls the same function on the path it
    // smoothed, which is what "the same C2 assertions, both producers" means.
    expect_c2_everywhere(path, machine.name);
  }
}

TEST(C2Path, ItStartsAndEndsAtRestAndIsMonotoneInSigma)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const JointLimits limits = machine_limits(model, machine);
    const StructuredPrimitive primitive = primitive_for(model, machine, context, limits);
    const GeometricPath & path = primitive.path;

    // Monotone in sigma, phase by phase: within a phase no coordinate leaves the
    // box its two waypoints span, so the path never runs past a waypoint and
    // comes back. Shared with the fallback's suite for the same reason the C2
    // assertions are.
    ASSERT_EQ(path.waypoints().size(), 4U) << machine.name;
    expect_at_rest_and_monotone(path, machine.name);
  }
}

TEST(C2Path, TheToolCoordinateRidesTheSameParameterAndIsNotAPathVariable)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const JointLimits limits = machine_limits(model, machine);

  // Five path coordinates, and q8 handed over separately. If q8 were a sixth
  // degree of freedom of the fit it would help decide the segment durations and
  // therefore the geometry, which is exactly what trajectory_planning 4.1 rules
  // out when it says the tool coordinate is not a path variable.
  crane_planning::PathFitRequest fit;
  fit.waypoints = {PathVector::Zero(), PathVector::Constant(0.2), PathVector::Constant(0.5)};
  fit.segment_names = {"first", "second"};
  fit.q8_start = 0.1;
  fit.q8_goal = 0.4;
  auto path = crane_planning::fit_c2_path(fit, limits, crane_planning::PathFitSettings{});
  ASSERT_TRUE(path.ok()) << path.status().message;

  EXPECT_DOUBLE_EQ(path.value().at(0.0).q8, 0.1);
  EXPECT_DOUBLE_EQ(path.value().at(1.0).q8, 0.4);
  EXPECT_DOUBLE_EQ(path.value().at(0.0).dq8, 0.0);
  EXPECT_DOUBLE_EQ(path.value().at(1.0).dq8, 0.0);
  double previous = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index <= 200U; ++index) {
    const double sigma = static_cast<double>(index) / 200.0;
    const double q8 = path.value().at(sigma).q8;
    EXPECT_GE(q8, previous - 1.0e-12) << "sigma " << sigma;
    previous = q8;
  }
}

// -------------------------------------------------------------- the refusals

TEST(StructuredPrimitive, AStartOutsideTheJointLimitsIsRefusedAndNamesTheLiftPhase)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);

  crane_planning::PrimitiveRequest request;
  request.q_start = settled(model, centred(limits, machine));
  request.q_goal = request.q_start;
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry
  // Past the boom's upper stop, which is where the lift would have started.
  request.q_start[1] = limits.axis[1].upper + 1.0;

  auto refused = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, context.settings.primitive, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("lift phase"), std::string::npos)
    << refused.status().message;
  // And the refusal says whose decision the fallback is, rather than taking it:
  // 4.4's order is enforced in plan_motion and this file only ever refuses.
  EXPECT_NE(refused.status().message.find("sampling fallback"), std::string::npos)
    << refused.status().message;
}

TEST(StructuredPrimitive, AGoalOutsideTheJointLimitsIsRefusedAndNamesTheDescendPhase)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);

  crane_planning::PrimitiveRequest request;
  request.q_start = settled(model, centred(limits, machine));
  request.q_goal = request.q_start;
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry
  request.q_goal[1] = limits.axis[1].lower - 1.0;

  auto refused = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, context.settings.primitive, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("descend phase"), std::string::npos)
    << refused.status().message;
}

TEST(StructuredPrimitive, ATransferAltitudeBelowBothEndpointsIsRefusedRatherThanFlattened)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);

  crane_planning::PrimitiveRequest request;
  const crane_model::QA start = centred(limits, machine);
  request.q_start = settled(model, start);
  request.q_goal = settled(model, moved(start, limits));
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry

  // A ceiling far below the machine's own mounting base: the derivation asks for
  // an altitude above both endpoints and the ceiling takes it away. A lift that
  // does not clear what it was derived to clear is not a shorter lift.
  crane_planning::PrimitiveSettings settings = context.settings.primitive;
  settings.altitude.ceiling_m = -50.0;

  auto refused = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, settings, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("lift phase"), std::string::npos)
    << refused.status().message;
  EXPECT_NE(refused.status().message.find("below the higher of the two endpoints"),
    std::string::npos) << refused.status().message;
}

TEST(StructuredPrimitive, AnUnreachableTransferAltitudeIsRefusedAndNamesTheLiftPhase)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);

  crane_planning::PrimitiveRequest request;
  const crane_model::QA start = centred(limits, machine);
  request.q_start = settled(model, start);
  request.q_goal = settled(model, moved(start, limits));
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry

  crane_planning::PrimitiveSettings settings = context.settings.primitive;
  settings.altitude.floor_m = 500.0;  // no crane on this site is 500 m tall

  auto refused = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, settings, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("lift phase"), std::string::npos)
    << refused.status().message;
}

TEST(StructuredPrimitive, AGoalColumnTheArmCannotReachIsRefusedAndNamesTheTraversePhase)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);

  // The lift and the traverse ask the arm for two *different* columns at the one
  // transfer altitude -- straight up from the start, straight down onto the goal
  // -- so the only way to refuse at the traverse rather than at the lift is a
  // goal whose column leaves the workspace while the start's does not. The
  // further out the tool stands, the lower the ceiling above it, so the goal is
  // left standing where the centred configuration puts it and the start is drawn
  // in underneath the machine: dropping the boom to its limit folds the arm in
  // rather than reaching it out, which is the opposite of what the joint name
  // suggests and is why this is measured below rather than assumed.
  const crane_model::QA standing_out = centred(limits, machine);
  crane_model::QA drawn_in = standing_out;
  const auto to_limit = [&limits](crane_model::QA & q_a, std::size_t row, bool upper) {
      if (!limits.axis[row].bounded) {
        return;
      }
      q_a[static_cast<Eigen::Index>(row)] =
        upper ? limits.axis[row].upper : limits.axis[row].lower;
    };
  to_limit(drawn_in, 1, false);  // boom to its limit, which brings the tool in and down
  to_limit(drawn_in, 3, true);   // telescope out, so this is not simply the shorter arm

  const crane_model::Q q_start = settled(model, drawn_in);
  const crane_model::Q q_goal = settled(model, standing_out);
  auto tcp_start =
    model.forward_kinematics(q_start, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  auto tcp_goal =
    model.forward_kinematics(q_goal, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  ASSERT_TRUE(tcp_start.ok()) << tcp_start.status().message;
  ASSERT_TRUE(tcp_goal.ok()) << tcp_goal.status().message;
  const Eigen::Vector3d p_start = tcp_start.value().position_m;
  const Eigen::Vector3d p_goal = tcp_goal.value().position_m;
  const double phi_start = crane_planning::phi_z_of(tcp_start.value().orientation);
  const double phi_goal = crane_planning::phi_z_of(tcp_goal.value().orientation);
  ASSERT_GT(p_goal.head<2>().norm(), p_start.head<2>().norm())
    << "the goal has to stand further out than the start for its column to be the "
    "lower one; this description does not put it there";

  // Whether a column is reachable at an altitude is a property of the description
  // and not a number this test may invent, so it is measured on the description's
  // own IK -- the same call `build_structured_primitive` makes for the waypoint,
  // with the same held tool coordinate and the same yaw.
  const auto reachable = [&](const Eigen::Vector3d & below, double altitude, double phi_z) {
      crane_planning::IkRequest probe;
      probe.p_tcp_0 = Eigen::Vector3d(below.x(), below.y(), altitude);
      probe.phi_z_d = phi_z;
      probe.q8 = machine.q8;
      probe.payload = crane_planning_test::empty_gripper();
      return crane_planning::solve_inverse_kinematics(
        model, context.geometry, limits, context.settings.ik, probe).ok();
    };

  // Bracket the goal column's ceiling. The bisection assumes reachability above a
  // standing tool is an interval and not a set of holes, which is what a boom
  // arm gives; nothing rests on the assumption, because the altitude it lands on
  // is then put through both columns directly below.
  double clears = p_goal.z();
  double blocked = p_goal.z() + 20.0;  // no crane on this site is twenty metres taller
  ASSERT_FALSE(reachable(p_goal, blocked, phi_goal));
  for (std::size_t step = 0; step < 20U; ++step) {
    const double middle = 0.5 * (clears + blocked);
    if (reachable(p_goal, middle, phi_goal)) {
      clears = middle;
    } else {
      blocked = middle;
    }
  }

  // The altitude the primitive will be made to run at: above the goal column's
  // ceiling, and above what the derivation asks for on its own so that the floor
  // binds rather than being ignored.
  auto reach_start = crane_planning::measure_tool_reach(model, q_start);
  auto reach_goal = crane_planning::measure_tool_reach(model, q_goal);
  ASSERT_TRUE(reach_start.ok()) << reach_start.status().message;
  ASSERT_TRUE(reach_goal.ok()) << reach_goal.status().message;
  auto derived = crane_planning::derive_transfer_altitude(
    p_start.z(), reach_start.value().reach_m, p_goal.z(), reach_goal.value().reach_m,
    crane_planning::scene_without_obstacles(), crane_planning::TransferAltitudeSettings{});
  ASSERT_TRUE(derived.ok()) << derived.status().message;
  const double altitude = std::max(blocked, derived.value().z_m + 0.05);

  // The premise, verified rather than assumed: at this one altitude the lift's
  // column is reachable and the traverse's is not.
  ASSERT_TRUE(reachable(p_start, altitude, phi_start))
    << "the start column is out of reach at " << altitude << " m too, so this would "
    "refuse at the lift and prove nothing about the traverse";
  ASSERT_FALSE(reachable(p_goal, altitude, phi_goal));

  crane_planning::PrimitiveRequest request;
  request.q_start = q_start;
  request.q_goal = q_goal;
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // collision is test_collision's; this is the geometry

  crane_planning::PrimitiveSettings settings = context.settings.primitive;
  settings.altitude.floor_m = altitude;

  auto refused = crane_planning::build_structured_primitive(
    model, context.geometry, limits, context.settings.ik, settings, request);
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("traverse phase"), std::string::npos)
    << refused.status().message;
  // And it is a refusal and not a deformation: no lower traverse was substituted
  // for the one that could not be built.
  EXPECT_NE(refused.status().message.find("refused rather than deformed"), std::string::npos)
    << refused.status().message;
}

TEST(C2Path, AFitRefusalNamesTheSegmentItCameFrom)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const JointLimits limits = machine_limits(model, machine);

  // The middle segment leaves the range the description gives the slewing joint;
  // the fit is given the primitive's own phase names, so its refusal is the one
  // a caller reads and it says `traverse`.
  PathVector inside = PathVector::Zero();
  PathVector outside = PathVector::Zero();
  ASSERT_TRUE(limits.axis[0].bounded);
  outside[0] = limits.axis[0].upper + 5.0;

  crane_planning::PathFitRequest fit;
  fit.waypoints = {inside, inside, outside, inside};
  fit.segment_names = {"lift", "traverse", "descend"};
  auto refused = crane_planning::fit_c2_path(fit, limits, crane_planning::PathFitSettings{});
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("traverse"), std::string::npos)
    << refused.status().message;
}

// ---------------------------------------------------- both tools, and the seam

TEST(StructuredPrimitive, TheDescendClearanceIsTheMountedToolsOwnAndDiffersBetweenTools)
{
  std::vector<double> reach;
  std::vector<crane_planning::ToolReference> reference;
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const JointLimits limits = machine_limits(model, machine);
    const StructuredPrimitive primitive = primitive_for(model, machine, context, limits);

    // The clearance the descend phase comes down through is the tool's own reach
    // from the TCP to the point that touches the load, read off the model.
    EXPECT_GT(primitive.altitude.tool_reach_start_m, 0.0) << machine.name;
    EXPECT_GT(primitive.altitude.tool_reach_goal_m, 0.0) << machine.name;
    EXPECT_GT(primitive.altitude.z_m, primitive.altitude.z_tcp_start_m) << machine.name;
    EXPECT_GT(primitive.altitude.z_m, primitive.altitude.z_tcp_goal_m) << machine.name;
    EXPECT_FALSE(primitive.altitude.obstacle_term_applied) << machine.name;
    EXPECT_FALSE(primitive.altitude.floor_binding) << machine.name;
    EXPECT_FALSE(primitive.altitude.ceiling_binding) << machine.name;
    reach.push_back(primitive.altitude.tool_reach_goal_m);
    reference.push_back(primitive.altitude.reference_goal);
  }

  ASSERT_EQ(reach.size(), 2U);
  // The rail gripper and the jaw are not the same length, so the two machines do
  // not lift to the same clearance -- which is what makes this a property of the
  // scene and the tool rather than a number in a file.
  EXPECT_GT(std::abs(reach[0] - reach[1]), 1.0e-3)
    << "pzs100 reaches " << reach[0] << " m, epsilon7040 " << reach[1] << " m";

  // And they are not even described to the same depth: only the 7040's jaw
  // carries a contact point, so the two clearances are measured to different
  // frames and the answer says which. A planner that assumed one frame would be
  // silently wrong on the other machine.
  EXPECT_EQ(reference[0], crane_planning::ToolReference::ToolCentre);
  EXPECT_EQ(reference[1], crane_planning::ToolReference::ToolContact);
}

TEST(StructuredPrimitive, TheCheckRunsBeforeAcceptanceAndSaysWhatItChecked)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const JointLimits limits = machine_limits(model, machine);
  const StructuredPrimitive primitive = primitive_for(model, machine, context, limits);

  // trajectory_planning 4.4 is generate, check, accept, and the accepted
  // primitive carries what the check said rather than the check being implied.
  // `primitive_for` asks for a collision-blind plan, so what it has to carry is
  // that nothing was checked -- the collision check itself is test_collision.
  EXPECT_TRUE(primitive.check.clear);
  EXPECT_FALSE(primitive.check.checked);
  EXPECT_NE(primitive.check.note.find("Nothing was checked"), std::string::npos)
    << primitive.check.note;

  crane_planning::PrimitiveRequest blind;
  blind.avoid_collisions = false;
  auto checked =
    crane_planning::check_primitive(model, primitive.path, blind, context.settings.primitive);
  ASSERT_TRUE(checked.ok()) << checked.status().message;
  EXPECT_TRUE(checked.value().clear);
}

TEST(StructuredPrimitive, TheEndpointsAreTheIkSolutionsAndTheRampRunsAlongThePath)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const JointLimits limits = machine_limits(model, machine);

    // The endpoint issue 039 produces: the equilibrium-constrained NLP of
    // robot_model 2.2, for a goal pose taken from a configuration the
    // description allows, so reachability is known by construction.
    const crane_model::QA start = centred(limits, machine);
    const crane_model::Q q_reference = settled(model, moved(start, limits));
    auto pose = model.forward_kinematics(
      q_reference, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    ASSERT_TRUE(pose.ok()) << pose.status().message;

    crane_planning::IkRequest goal;
    goal.p_tcp_0 = pose.value().position_m;
    goal.phi_z_d = crane_planning::phi_z_of(pose.value().orientation);
    goal.q8 = machine.q8;
    goal.payload = crane_planning_test::empty_gripper();
    auto endpoint = crane_planning::solve_equilibrium_constrained_ik(
      model, context.geometry, limits, context.settings.ik, context.settings.equilibrium, goal);
    ASSERT_TRUE(endpoint.ok()) << machine.name << ": " << endpoint.status().message;

    crane_planning::PrimitiveRequest request;
    request.q_start = settled(model, start);
    request.q_goal = endpoint.value().q;
    request.payload = crane_planning_test::empty_gripper();
    request.scene = crane_planning::scene_without_obstacles();
    request.avoid_collisions = false;
    auto built = crane_planning::build_structured_primitive(
      model, context.geometry, limits, context.settings.ik, context.settings.primitive, request);
    ASSERT_TRUE(built.ok()) << machine.name << ": " << built.status().message;

    // The path ends on the configuration the IK produced, not near it.
    EXPECT_LT((built.value().path.at(0.0).q_a - path_of(request.q_start)).norm(), 1.0e-9)
      << machine.name;
    EXPECT_LT((built.value().path.at(1.0).q_a - path_of(request.q_goal)).norm(), 1.0e-9)
      << machine.name;

    // And it ends on the *pose* to the tolerance 039 accepted the endpoint on,
    // which is the acceptance test of robot_model 2.2 re-run on this path's end.
    crane_model::Q q_end = request.q_goal;
    for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
      q_end[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
        built.value().path.at(1.0).q_a[static_cast<Eigen::Index>(row)];
    }
    auto reached = model.forward_kinematics(
      q_end, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    ASSERT_TRUE(reached.ok()) << reached.status().message;
    EXPECT_LT(
      (reached.value().position_m - goal.p_tcp_0).norm(), context.settings.ik.eps_pos)
      << machine.name;

    // The seam: the ramp of issue 038 run along this path rather than straight
    // between two configurations. Still at rest at both ends, still inside the
    // scaled velocity bound, and now following the lift and the descend.
    auto timed = crane_planning::scaled_ramp_along_path(
      built.value().path, limits, 1.0, context.settings.ramp);
    ASSERT_TRUE(timed.ok()) << machine.name << ": " << timed.status().message;
    const crane_planning::TimedTrajectory & trajectory = timed.value();
    ASSERT_GE(trajectory.q_a_ref.size(), 2U) << machine.name;
    EXPECT_LT(trajectory.dq_a_ref.front().norm(), 1.0e-9) << machine.name;
    EXPECT_LT(trajectory.dq_a_ref.back().norm(), 1.0e-9) << machine.name;
    for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      EXPECT_NEAR(trajectory.q_a_ref.front()[axis], path_of(request.q_start)[axis], 1.0e-9)
        << machine.name;
      EXPECT_NEAR(trajectory.q_a_ref.back()[axis], path_of(request.q_goal)[axis], 1.0e-9)
        << machine.name;
    }
    for (const crane_model::DQA & dq_a : trajectory.dq_a_ref) {
      for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
        EXPECT_LE(
          std::abs(dq_a[static_cast<Eigen::Index>(row)]),
          limits.axis[row].dq_max * (1.0 + 1.0e-9)) << machine.name << " axis " << row;
      }
    }
    EXPECT_LE(trajectory.limiting_fraction, 1.0 + 1.0e-9) << machine.name;
  }
}
