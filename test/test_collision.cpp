// Collision, the truck, and the sway envelope -- `wiki/trajectory_planning.md`
// 4.2 and 4.3, offline.
//
// Every scene here is **built in the test** rather than subscribed: nothing in
// this binary opens a socket, joins a graph or starts a simulator, and the only
// thing it links besides gtest is the planner core. The paths are the real
// structured primitive of 4.4 over the real machine descriptions, and the
// obstacles are placed against the geometry those paths actually have -- at the
// tool's own position at a junction, or at the swung position the sway envelope
// reaches -- so nothing here depends on a coordinate written down by hand.

#include <gtest/gtest.h>

#include <cstdio>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "crane_planning/collision.hpp"
#include "crane_planning/planner_core.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning::CollisionSettings;
using crane_planning::GeometricPath;
using crane_planning::JointLimits;
using crane_planning::PayloadShape;
using crane_planning::TruckModel;
using crane_planning_test::Machine;

/// The middle of every bounded actuated range, which both descriptions allow.
crane_model::QA centred(const JointLimits & limits, const Machine & machine)
{
  crane_model::QA q_a = crane_model::QA::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] =
      limits.axis[row].bounded ? limits.axis[row].centre() : 0.0;
  }
  q_a[static_cast<Eigen::Index>(crane_planning::kToolRow)] = machine.q8;
  return q_a;
}

crane_model::Q settled(const crane_model::Model & model, const crane_model::QA & q_a)
{
  auto equilibrium = model.passive_equilibrium(q_a, crane_planning_test::empty_gripper());
  if (!equilibrium.ok()) {
    throw std::runtime_error(equilibrium.status().message);
  }
  crane_model::Q q = crane_model::Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
      q_a[static_cast<Eigen::Index>(row)];
  }
  q.segment<2>(4) = equilibrium.value();
  return q;
}

/// A move worth checking: slew across, drop the boom, extend a little.
crane_model::QA moved(const crane_model::QA & start, const JointLimits & limits)
{
  crane_model::QA goal = start;
  const auto shift = [&limits](crane_model::QA & q_a, std::size_t row, double by) {
      const Eigen::Index axis = static_cast<Eigen::Index>(row);
      q_a[axis] += by;
      if (limits.axis[row].bounded) {
        q_a[axis] = std::min(limits.axis[row].upper, std::max(limits.axis[row].lower, q_a[axis]));
      }
    };
  shift(goal, 0, 0.7);
  shift(goal, 1, -0.15);
  shift(goal, 3, 0.2);
  return goal;
}

/// One machine, its context, and one collision-blind path over it.
/**
 * The path is built **once** per machine and then checked against many scenes,
 * which is the point of the split: `check_path` takes a path and a scene, so a
 * test can move the truck without re-solving any inverse kinematics.
 */
struct Scenario
{
  crane_model::Model model;
  crane_planning::PlannerContext context;
  crane_planning::StructuredPrimitive primitive;
};

Scenario build_scenario(const Machine & machine)
{
  crane_model::Model model = crane_planning_test::build_model(machine);
  crane_planning::PlannerContext context = crane_planning_test::build_context(model, machine);
  const crane_model::QA start = centred(context.limits, machine);

  crane_planning::PrimitiveRequest request;
  request.q_start = settled(model, start);
  request.q_goal = settled(model, moved(start, context.limits));
  request.payload = crane_planning_test::empty_gripper();
  request.scene = crane_planning::scene_without_obstacles();
  request.avoid_collisions = false;  // the path first, the scenes afterwards

  auto built = crane_planning::build_structured_primitive(
    model, context.geometry, context.limits, context.settings.ik, context.settings.primitive,
    request);
  if (!built.ok()) {
    throw std::runtime_error(machine.name + std::string(": ") + built.status().message);
  }
  return Scenario{std::move(model), std::move(context), std::move(built).value()};
}

/// Built once for the whole binary: a path costs two inverse-kinematics solves.
const Scenario & scenario(std::size_t index)
{
  static std::deque<Scenario> built;
  if (built.empty()) {
    for (const Machine & machine : crane_planning_test::machines()) {
      built.push_back(build_scenario(machine));
    }
  }
  return built[index];
}

/// The configuration one point of a path stands for, with the tool hanging.
crane_model::Q configuration_at(
  const crane_model::Model & model, const GeometricPath & path, double sigma)
{
  const crane_planning::PathSample sample = path.at(sigma);
  crane_model::QA q_a = crane_model::QA::Zero();
  for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = sample.q_a[static_cast<Eigen::Index>(row)];
  }
  q_a[static_cast<Eigen::Index>(crane_planning::kToolRow)] = sample.q8;
  return settled(model, q_a);
}

Eigen::Vector3d tcp_of(const crane_model::Model & model, const crane_model::Q & q)
{
  auto pose = model.forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  if (!pose.ok()) {
    throw std::runtime_error(pose.status().message);
  }
  return pose.value().position_m;
}

crane_model::CollisionPrimitive box_at(
  const std::string & id, const Eigen::Vector3d & centre, double side)
{
  crane_model::CollisionPrimitive box;
  box.id = id;
  box.shape = crane_model::CollisionShape::Box;
  box.pose_in_mounting_base = Eigen::Isometry3d::Identity();
  box.pose_in_mounting_base.translation() = centre;
  box.dimensions_m = Eigen::Vector3d::Constant(side);
  box.structural = false;
  return box;
}

/// A truck, as the world model measures it: one box, in K0, with the reserved id.
crane_model::CollisionPrimitive truck_at(const Eigen::Vector3d & centre, double yaw = 0.0)
{
  crane_model::CollisionPrimitive truck;
  truck.id = crane_planning::kTruckId;
  truck.shape = crane_model::CollisionShape::Box;
  truck.pose_in_mounting_base = Eigen::Isometry3d::Identity();
  truck.pose_in_mounting_base.linear() =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  truck.pose_in_mounting_base.translation() = centre;
  truck.dimensions_m = Eigen::Vector3d(6.5, 2.4, 1.2);
  truck.structural = true;
  return truck;
}

/// Coarse enough that a check costs a second, still finer than anything placed.
CollisionSettings quick_settings()
{
  CollisionSettings settings;
  settings.sway.q_sway_max = Eigen::Vector2d::Zero();
  settings.resolution_m = 0.5;
  return settings;
}

crane_model::CollisionScene scene_of(std::vector<crane_model::CollisionPrimitive> primitives)
{
  crane_model::CollisionScene scene;
  scene.primitives = std::move(primitives);
  return scene;
}

/// The scene a measured truck becomes: the bed and the runges of 4.2, and no
/// primitive standing for the vehicle the crane is bolted to.
crane_model::CollisionScene truck_scene(
  const crane_model::CollisionPrimitive & truck, const TruckModel & model)
{
  auto geometry = crane_planning::expand_truck(truck, model);
  if (!geometry.ok()) {
    throw std::runtime_error(geometry.status().message);
  }
  return scene_of(std::move(geometry).value());
}

}  // namespace

// ------------------------------------------------------------------ the truck

TEST(TruckModelGeometry, TheLegacyRungesArePreservedAndKeyedToTheMeasuredPose)
{
  // trajectory_planning 4.2: two rows of three, 0.28 x 0.31 x 2.12 m each,
  // preserved from the legacy motion-planning YAML and no longer written down in
  // K0 -- they are placed on the pose the scene measured.
  const TruckModel model;
  ASSERT_EQ(model.station_offsets_m.size(), 3U);

  auto geometry = crane_planning::expand_truck(truck_at(Eigen::Vector3d(4.0, 0.0, 0.0)), model);
  ASSERT_TRUE(geometry.ok()) << geometry.status().message;
  // One bed plus two rows of three.
  ASSERT_EQ(geometry.value().size(), 7U);

  std::size_t runges = 0;
  for (const crane_model::CollisionPrimitive & primitive : geometry.value()) {
    EXPECT_TRUE(primitive.structural) << primitive.id;
    if (primitive.id.rfind("truck_runge", 0) != 0) {
      continue;
    }
    ++runges;
    EXPECT_NEAR(primitive.dimensions_m.x(), 0.28, 1.0e-12) << primitive.id;
    EXPECT_NEAR(primitive.dimensions_m.y(), 0.31, 1.0e-12) << primitive.id;
    EXPECT_NEAR(primitive.dimensions_m.z(), 2.12, 1.0e-12) << primitive.id;
  }
  EXPECT_EQ(runges, 6U);
}

TEST(TruckModelGeometry, MovingTheTruckMovesTheRunges)
{
  const TruckModel model;
  const Eigen::Vector3d first(4.0, 0.0, 0.0);
  const Eigen::Vector3d shift(1.5, -0.75, 0.25);

  auto before = crane_planning::expand_truck(truck_at(first), model);
  auto after = crane_planning::expand_truck(truck_at(first + shift), model);
  ASSERT_TRUE(before.ok()) << before.status().message;
  ASSERT_TRUE(after.ok()) << after.status().message;
  ASSERT_EQ(before.value().size(), after.value().size());

  for (std::size_t index = 0; index < before.value().size(); ++index) {
    EXPECT_EQ(before.value()[index].id, after.value()[index].id);
    const Eigen::Vector3d moved_by =
      after.value()[index].pose_in_mounting_base.translation() -
      before.value()[index].pose_in_mounting_base.translation();
    EXPECT_LT((moved_by - shift).norm(), 1.0e-12) << before.value()[index].id;
  }
}

TEST(TruckModelGeometry, TurningTheTruckTurnsTheRungesWithIt)
{
  // The runges are the vehicle's, so a truck parked at an angle has them at that
  // angle: their offsets are in the bed's own axes and never in K0's.
  const TruckModel model;
  const Eigen::Vector3d centre(4.0, 0.0, 0.0);
  auto straight = crane_planning::expand_truck(truck_at(centre), model);
  auto turned = crane_planning::expand_truck(truck_at(centre, M_PI / 2.0), model);
  ASSERT_TRUE(straight.ok()) << straight.status().message;
  ASSERT_TRUE(turned.ok()) << turned.status().message;

  const Eigen::Matrix3d rotation =
    Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  for (std::size_t index = 0; index < straight.value().size(); ++index) {
    const Eigen::Vector3d expected =
      centre + rotation * (straight.value()[index].pose_in_mounting_base.translation() - centre);
    EXPECT_LT(
      (turned.value()[index].pose_in_mounting_base.translation() - expected).norm(), 1.0e-12)
      << straight.value()[index].id;
  }
}

TEST(TruckModelGeometry, ATruckModelThatDoesNotFitTheMeasuredTruckIsRefused)
{
  TruckModel model;
  model.station_offsets_m = {0.0, 12.0};  // off the end of a 6.5 m bed
  auto geometry = crane_planning::expand_truck(truck_at(Eigen::Vector3d(4.0, 0.0, 0.0)), model);
  ASSERT_FALSE(geometry.ok());
  EXPECT_NE(geometry.status().message.find("hang off the end"), std::string::npos)
    << geometry.status().message;
}

// ------------------------------------------------------- the path and a scene

TEST(PathCollision, AClearPathIsAccepted)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);

    // A truck, far enough away that nothing on this path comes near it.
    const crane_model::CollisionScene scene =
      truck_scene(truck_at(Eigen::Vector3d(20.0, 0.0, 0.0)), TruckModel{});
    auto checked = crane_planning::check_path(
      fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
      PayloadShape{}, quick_settings());
    ASSERT_TRUE(checked.ok()) << machine.name << ": " << checked.status().message;
    EXPECT_TRUE(checked.value().clear) << machine.name << ": " <<
      crane_planning::describe(checked.value());
    EXPECT_GT(checked.value().samples, 2U) << machine.name;
    EXPECT_EQ(checked.value().structural_primitives, 7U) << machine.name;
  }
}

TEST(PathCollision, ARungeInTheDescentCorridorIsRefusedAndTheTruckCanBeMovedOutOfIt)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);

    // Park the truck so that its middle left runge stands where this path's
    // descend phase ends. The station offsets are the vehicle's, so the pose is
    // the only thing this test chooses.
    const crane_model::Q goal = configuration_at(fixture.model, fixture.primitive.path, 1.0);
    const Eigen::Vector3d tcp = tcp_of(fixture.model, goal);
    TruckModel truck;
    truck.station_offsets_m = {0.0};
    const double y_offset = 0.5 * (2.4 - truck.runge_dimensions_m.y());
    const double z_offset = 0.5 * (1.2 + truck.runge_dimensions_m.z());
    const Eigen::Vector3d parked(tcp.x(), tcp.y() - y_offset, tcp.z() - z_offset);

    CollisionSettings settings = quick_settings();
    settings.truck = truck;

    auto blocked = crane_planning::check_path(
      fixture.model, fixture.primitive.path, truck_scene(truck_at(parked), truck),
      crane_planning_test::empty_gripper(), PayloadShape{}, settings);
    ASSERT_TRUE(blocked.ok()) << machine.name << ": " << blocked.status().message;
    EXPECT_FALSE(blocked.value().clear) << machine.name;

    // Which obstacle blocked, and where -- an operator is told what is in the way.
    const std::string note = crane_planning::describe(blocked.value());
    EXPECT_NE(blocked.value().blocked_at.blocker.other_id.find("truck_runge"), std::string::npos)
      << machine.name << ": " << note;
    EXPECT_TRUE(blocked.value().blocked_at.blocker.structural) << machine.name << ": " << note;
    EXPECT_NE(note.find("truck_runge"), std::string::npos) << note;
    EXPECT_NE(note.find("K0_mounting_base"), std::string::npos) << note;
    EXPECT_FALSE(blocked.value().blocked_at.blocker.witness_on_other_m.isZero()) << note;

    // The same path, once the truck moves. The runges are keyed to its pose, so
    // moving the vehicle moves them and nothing else changes.
    auto cleared = crane_planning::check_path(
      fixture.model, fixture.primitive.path,
      truck_scene(truck_at(parked + Eigen::Vector3d(0.0, -12.0, 0.0)), truck),
      crane_planning_test::empty_gripper(), PayloadShape{}, settings);
    ASSERT_TRUE(cleared.ok()) << machine.name << ": " << cleared.status().message;
    EXPECT_TRUE(cleared.value().clear) << machine.name << ": " <<
      crane_planning::describe(cleared.value());
  }
}

// --------------------------------------------------------- the sway envelope

TEST(SwayEnvelope, TheEnvelopeIsTheTravelOfThePendulumAtTheBoundOfMpcConstraintThree)
{
  crane_planning::SwayEnvelope envelope;
  envelope.q_sway_max = Eigen::Vector2d(0.2, 0.1);
  // Delta_sway = l_tool sin(max|q_u^+|), and the bound is the larger of the two
  // coordinates because either of them moves the same tool.
  EXPECT_NEAR(crane_planning::sway_clearance_m(2.5, envelope), 2.5 * std::sin(0.2), 1.0e-12);
  envelope.q_sway_max = Eigen::Vector2d::Zero();
  EXPECT_NEAR(crane_planning::sway_clearance_m(2.5, envelope), 0.0, 1.0e-12);
}

TEST(SwayEnvelope, APathClearForAStillToolIsRefusedForASwingingOne)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);

    // A junction is checked whatever the uniform grid lands on, so the obstacle
    // is placed at one: where the tool reaches when it swings to the corner of
    // the envelope, and nowhere near where it hangs.
    const double sigma = fixture.primitive.path.junction_sigmas().front();
    const crane_model::Q hanging = configuration_at(fixture.model, fixture.primitive.path, sigma);

    CollisionSettings swinging = quick_settings();
    swinging.sway.q_sway_max = Eigen::Vector2d(0.2, 0.2);

    crane_model::Q swung = hanging;
    swung.segment<2>(4) += swinging.sway.q_sway_max;
    const Eigen::Vector3d reached = tcp_of(fixture.model, swung);
    ASSERT_GT((reached - tcp_of(fixture.model, hanging)).norm(), 0.05) << machine.name;

    const crane_model::CollisionScene scene = scene_of({box_at("swing_target", reached, 0.20)});

    // Still: the tool hangs, the box is where the tool is not, and the path is
    // accepted.
    auto still = crane_planning::check_path(
      fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
      PayloadShape{}, quick_settings());
    ASSERT_TRUE(still.ok()) << machine.name << ": " << still.status().message;
    EXPECT_TRUE(still.value().clear) << machine.name << ": " <<
      crane_planning::describe(still.value());

    // Swinging, at the same q_u^+ the MPC is given: the same path is refused,
    // and the refusal says which sway pose found it.
    auto checked = crane_planning::check_path(
      fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
      PayloadShape{}, swinging);
    ASSERT_TRUE(checked.ok()) << machine.name << ": " << checked.status().message;
    ASSERT_FALSE(checked.value().clear) << machine.name << ": " <<
      crane_planning::describe(checked.value());
    EXPECT_EQ(checked.value().blocked_at.blocker.other_id, "swing_target") << machine.name;
    EXPECT_TRUE(checked.value().blocked_at.blocker.at_sway_bound) << machine.name;
    EXPECT_TRUE(checked.value().blocked_at.envelope_resolved) << machine.name;
    EXPECT_GT(checked.value().blocked_at.sway_clearance_m, 0.0) << machine.name;
    EXPECT_NE(
      crane_planning::describe(checked.value()).find("4.3"), std::string::npos);
  }
}

// -------------------------------------------------------- against the machine

TEST(SelfCollision, TheArmFoldedBackOverTheBoomIsRefusedAgainstAnEmptyScene)
{
  // trajectory_planning 4.2's [!todo]: nothing in the present system checks the
  // crane against itself. This does, against crane_model's derived
  // allowed-collision list -- and 034 measured that about a third of
  // configurations drawn from the joint limits self-collide, so it earns its
  // keep. The arm at its stop reaches the *boom*, not the column.
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);

    crane_model::QA folded = centred(fixture.context.limits, machine);
    ASSERT_TRUE(fixture.context.limits.axis[2].bounded) << machine.name;
    folded[2] = fixture.context.limits.axis[2].upper;
    const crane_model::Q q = settled(fixture.model, folded);

    auto checked = crane_planning::check_configuration(
      fixture.model, crane_model::CollisionScene{}, PayloadShape{}, quick_settings(), q, 0.1);
    ASSERT_TRUE(checked.ok()) << machine.name << ": " << checked.status().message;
    EXPECT_FALSE(checked.value().clear) << machine.name << ", at q3 = " << folded[2];
    EXPECT_TRUE(checked.value().blocker.self) << machine.name << ": " <<
      checked.value().blocker.other_id;
    // The self result names the closest checked link pair as `first|second`.
    EXPECT_NE(checked.value().blocker.other_id.find('|'), std::string::npos)
      << checked.value().blocker.other_id;
  }
}

TEST(SelfCollision, TheNeutralConfigurationIsNotRefusedByTheSwayEnvelope)
{
  // 034 measured 25 mm between the rail gripper and the inner telescope at the
  // neutral configuration. A check that demanded Delta_sway of the crane's own
  // link pairs would refuse that, which is why the envelope is resolved by
  // querying at the sway poses and not by a margin on the self distance.
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);
    const crane_model::Q q = settled(fixture.model, centred(fixture.context.limits, machine));

    CollisionSettings settings = quick_settings();
    settings.sway.q_sway_max = Eigen::Vector2d(0.2, 0.2);
    auto checked = crane_planning::check_configuration(
      fixture.model, crane_model::CollisionScene{}, PayloadShape{}, settings, q, 0.1);
    ASSERT_TRUE(checked.ok()) << machine.name << ": " << checked.status().message;
    EXPECT_TRUE(checked.value().clear) << machine.name << ": " <<
      checked.value().blocker.other_id << " at " <<
      checked.value().blocker.minimum_distance_m << " m";
  }
}

// ----------------------------------------------------------------- a payload

TEST(CarriedPayload, ItIsCheckedAsTheScenePrimitiveWithTheReservedId)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);
    const crane_model::Q q = settled(fixture.model, centred(fixture.context.limits, machine));

    // Two metres under K8 is below anything the description hangs there, so an
    // empty gripper clears a small box placed at it and a payload reaching that
    // far does not.
    PayloadShape shape;
    shape.declared = true;
    shape.shape = crane_model::CollisionShape::Box;
    shape.dimensions_m = Eigen::Vector3d(0.4, 0.4, 0.4);
    shape.center_k8_m = Eigen::Vector3d(0.0, 0.0, -2.0);

    auto placed = crane_planning::payload_primitive(fixture.model, q, shape);
    ASSERT_TRUE(placed.ok()) << machine.name << ": " << placed.status().message;
    EXPECT_EQ(placed.value().id, std::string(crane_planning::kPayloadId));

    const crane_model::CollisionScene scene = scene_of(
      {box_at("stack", placed.value().pose_in_mounting_base.translation(), 0.30)});

    auto empty = crane_planning::check_configuration(
      fixture.model, scene, PayloadShape{}, quick_settings(), q, 0.1);
    ASSERT_TRUE(empty.ok()) << machine.name << ": " << empty.status().message;
    EXPECT_TRUE(empty.value().clear) << machine.name << ": " << empty.value().blocker.other_id;

    auto carried = crane_planning::check_configuration(
      fixture.model, scene, shape, quick_settings(), q, 0.1);
    ASSERT_TRUE(carried.ok()) << machine.name << ": " << carried.status().message;
    EXPECT_FALSE(carried.value().clear) << machine.name;
    EXPECT_EQ(carried.value().blocker.other_id, "stack") << machine.name;

    // And the payload lengthens the pendulum, so the envelope it is cleared over
    // is its own rather than the empty tool's.
    auto with_payload = crane_planning::pendulum_length_m(fixture.model, q, shape);
    auto without = crane_planning::pendulum_length_m(fixture.model, q, PayloadShape{});
    ASSERT_TRUE(with_payload.ok() && without.ok());
    EXPECT_GT(with_payload.value(), without.value()) << machine.name;
  }
}

// --------------------------------------------------------------- resolution

TEST(CheckResolution, TheStepIsTightenedToHalfTheThinnestPrimitiveInTheScene)
{
  const Scenario & fixture = scenario(0);
  const crane_model::CollisionScene scene =
    scene_of({box_at("thin", Eigen::Vector3d(20.0, 0.0, 0.0), 0.28)});

  CollisionSettings settings = quick_settings();
  settings.resolution_m = 0.5;
  auto checked = crane_planning::check_path(
    fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
    PayloadShape{}, settings);
  ASSERT_TRUE(checked.ok()) << checked.status().message;
  EXPECT_TRUE(checked.value().resolution_tightened);
  EXPECT_NEAR(checked.value().step_m, 0.14, 1.0e-12);
  EXPECT_NEAR(checked.value().thinnest_primitive_m, 0.28, 1.0e-12);
  EXPECT_NE(
    crane_planning::describe(checked.value()).find("tightened"), std::string::npos);
}

TEST(CheckResolution, APrimitiveThinnerThanTheFloorIsRefusedRatherThanCheckedCoarsely)
{
  const Scenario & fixture = scenario(0);
  const crane_model::CollisionScene scene =
    scene_of({box_at("wire", Eigen::Vector3d(20.0, 0.0, 0.0), 0.004)});

  auto checked = crane_planning::check_path(
    fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
    PayloadShape{}, quick_settings());
  ASSERT_FALSE(checked.ok());
  EXPECT_NE(checked.status().message.find("not checked"), std::string::npos)
    << checked.status().message;
}

TEST(CheckResolution, APathIsCheckedAtMoreThanItsWaypoints)
{
  const Scenario & fixture = scenario(0);
  CollisionSettings settings = quick_settings();
  auto checked = crane_planning::check_path(
    fixture.model, fixture.primitive.path, crane_model::CollisionScene{},
    crane_planning_test::empty_gripper(), PayloadShape{}, settings);
  ASSERT_TRUE(checked.ok()) << checked.status().message;
  EXPECT_GT(checked.value().travel_m, 0.5);
  EXPECT_GE(
    static_cast<double>(checked.value().samples - 1U),
    checked.value().travel_m / checked.value().step_m);
  EXPECT_GT(checked.value().samples, fixture.primitive.path.segment_count() + 1U);
}

// ------------------------------------------------------ what the caller asked

TEST(AvoidCollisions, FalseStillPlansAndSaysNothingWasChecked)
{
  const Scenario & fixture = scenario(0);
  crane_planning::PrimitiveRequest request;
  request.avoid_collisions = false;

  auto checked = crane_planning::check_primitive(
    fixture.model, fixture.primitive.path, request, fixture.context.settings.primitive);
  ASSERT_TRUE(checked.ok()) << checked.status().message;
  EXPECT_TRUE(checked.value().clear);
  EXPECT_FALSE(checked.value().checked);
  EXPECT_NE(checked.value().note.find("Nothing was checked"), std::string::npos)
    << checked.value().note;
}

TEST(AvoidCollisions, TrueWithoutASceneIsARefusalRatherThanAnUncheckedPlan)
{
  const Scenario & fixture = scenario(0);
  crane_planning::PrimitiveRequest request;
  request.avoid_collisions = true;
  request.collision_scene = nullptr;

  auto checked = crane_planning::check_primitive(
    fixture.model, fixture.primitive.path, request, fixture.context.settings.primitive);
  ASSERT_FALSE(checked.ok());
  EXPECT_NE(checked.status().message.find("/crane/collision_scene"), std::string::npos)
    << checked.status().message;
}

TEST(AvoidCollisions, ASceneCarryingTheReservedPayloadIdIsRefused)
{
  const Scenario & fixture = scenario(0);
  const crane_model::CollisionScene scene =
    scene_of({box_at(crane_planning::kPayloadId, Eigen::Vector3d(20.0, 0.0, 0.0), 0.5)});
  auto checked = crane_planning::check_path(
    fixture.model, fixture.primitive.path, scene, crane_planning_test::empty_gripper(),
    PayloadShape{}, quick_settings());
  ASSERT_FALSE(checked.ok());
  EXPECT_NE(checked.status().message.find("payload"), std::string::npos)
    << checked.status().message;
}

// ------------------------------------------------------------ the extent read

TEST(SceneExtent, TheTallestPrimitiveIsWhatTheTransferAltitudeIsGivenToClear)
{
  const crane_model::CollisionScene empty;
  EXPECT_FALSE(crane_planning::scene_extent(empty).obstacles_known);
  EXPECT_FALSE(crane_planning::scene_without_obstacles().obstacles_known);

  const crane_planning::SceneExtent extent = crane_planning::scene_extent(
    truck_scene(truck_at(Eigen::Vector3d(4.0, 0.0, 0.0)), TruckModel{}));
  ASSERT_TRUE(extent.obstacles_known);
  // The top of a runge: the truck box is 1.2 m tall and centred at z = 0, so its
  // top face is at 0.6 m and 2.12 m of runge stands on it.
  EXPECT_NEAR(extent.highest_obstacle_z_m, 0.6 + 2.12, 1.0e-9);
}

TEST(Probe, DISABLED_PathClearance)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    crane_model::Model model = crane_planning_test::build_model(machine);
    crane_planning::PlannerContext context = crane_planning_test::build_context(model, machine);
    const JointLimits & limits = context.limits;
    std::printf("=== %s\n", machine.name);
    for (const double q3 : {0.3, 0.6, 0.93, 1.3}) {
      for (const double ceiling : {0.3, 0.8, 1.5}) {
        crane_model::QA start = centred(limits, machine);
        start[1] = 0.18;
        start[2] = q3;
        crane_planning::PrimitiveRequest request;
        request.q_start = settled(model, start);
        request.q_goal = settled(model, moved(start, limits));
        request.payload = crane_planning_test::empty_gripper();
        request.scene = crane_planning::scene_without_obstacles();
        request.avoid_collisions = false;
        crane_planning::PrimitiveSettings settings = context.settings.primitive;
        auto tcp = model.forward_kinematics(
          request.q_start, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
        settings.altitude.ceiling_m = tcp.value().position_m.z() + ceiling;
        auto built = crane_planning::build_structured_primitive(
          model, context.geometry, limits, context.settings.ik, settings, request);
        if (!built.ok()) {
          std::printf("q3=%.2f ceil=%.1f  no primitive: %s\n", q3, ceiling,
            built.status().message.c_str());
          continue;
        }
        auto checked = crane_planning::check_path(
          model, built.value().path, crane_model::CollisionScene{},
          crane_planning_test::empty_gripper(), PayloadShape{}, quick_settings());
        if (!checked.ok()) {
          std::printf("q3=%.2f ceil=%.1f  check failed\n", q3, ceiling);
          continue;
        }
        std::printf(
          "q3=%.2f ceil=%.1f  clear=%d worst=%.4f samples=%zu %s\n", q3, ceiling,
          checked.value().clear ? 1 : 0, checked.value().worst_clearance_m,
          checked.value().samples, checked.value().blocked_at.blocker.other_id.c_str());
      }
    }
  }
}

TEST(Probe, DISABLED_SelfClearanceOverTheArmPlane)
{
  for (std::size_t index = 0; index < crane_planning_test::machines().size(); ++index) {
    const Machine & machine = crane_planning_test::machines()[index];
    const Scenario & fixture = scenario(index);
    const JointLimits & limits = fixture.context.limits;
    std::printf("=== %s\n", machine.name);
    for (int i2 = 0; i2 <= 6; ++i2) {
      for (int i3 = 0; i3 <= 6; ++i3) {
        for (int i4 = 0; i4 <= 2; ++i4) {
          crane_model::QA q_a = centred(limits, machine);
          q_a[1] = limits.axis[1].lower + (limits.axis[1].upper - limits.axis[1].lower) * i2 / 6.0;
          q_a[2] = limits.axis[2].lower + (limits.axis[2].upper - limits.axis[2].lower) * i3 / 6.0;
          q_a[3] = limits.axis[3].lower + (limits.axis[3].upper - limits.axis[3].lower) * i4 / 2.0;
          auto eq = fixture.model.passive_equilibrium(q_a, crane_planning_test::empty_gripper());
          if (!eq.ok()) {
            std::printf("q2=%.2f q3=%.2f q4=%.2f  no equilibrium\n", q_a[1], q_a[2], q_a[3]);
            continue;
          }
          crane_model::Q q = crane_model::Q::Zero();
          for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
            q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
              q_a[static_cast<Eigen::Index>(row)];
          }
          q.segment<2>(4) = eq.value();
          auto answer = fixture.model.collision_query(q, crane_model::CollisionScene{});
          if (!answer.ok()) {
            std::printf("query failed: %s\n", answer.status().message.c_str());
            continue;
          }
          std::printf(
            "q2=%.2f q3=%.2f q4=%.2f  d=%.4f  %s\n", q_a[1], q_a[2], q_a[3],
            answer.value().minimum_distance_m, answer.value().other_id.c_str());
        }
      }
    }
  }
}
