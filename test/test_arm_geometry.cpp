// The two-link arm of `wiki/robot_model.md` 2.2, as probed out of each machine
// description -- and as checked against the forward kinematics it was probed
// from.
//
// The point of this suite is that nothing in `ArmGeometry` is written down. a2,
// a3, d45(q4) and the bearing they are measured from are recovered from the
// model, so what has to be asserted is not their values but that the *structure*
// they assume is the description's: one planar chain, a boom hinged at K1's own
// origin, and a forearm whose length follows sqrt(a3^2 + d45^2) with d45 affine
// in q4. If any of that stopped holding, the closed-form closure would keep
// answering and start being wrong.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/joint_limits.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning_test::Machine;

/// Machine precision is what both descriptions actually close to; this is the
/// slack a residual is allowed on top of it.
constexpr double kTolerance = 1.0e-9;

double uniform(std::mt19937_64 & source, double lower, double upper)
{
  return std::uniform_real_distribution<double>(lower, upper)(source);
}

}  // namespace

TEST(ArmGeometry, TheClosureParametersComeOutOfTheDescription)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    auto limits = crane_planning::read_joint_limits(
      crane_planning_test::description(machine), model.urdf_joint_names());
    ASSERT_TRUE(limits.ok()) << machine.name << ": " << limits.status().message;

    auto geometry = crane_planning::probe_arm_geometry(
      model, limits.value(), 9, crane_planning::GeometryTolerance{});
    ASSERT_TRUE(geometry.ok()) << machine.name << ": " << geometry.status().message;

    const crane_planning::ArmGeometry & arm = geometry.value();
    EXPECT_GT(arm.a2, 0.0) << machine.name;
    EXPECT_GT(arm.d45_gain, 0.0) << machine.name;
    // Both telescope stages advance by the same q4 (wiki/nomenclature.md 4 and
    // hydraulics 2.4), so the effective forearm grows at twice the commanded
    // extension. Asserted rather than assumed, because it is the one number in
    // the closure that is a property of the coupling and not of a link length.
    EXPECT_NEAR(arm.d45_gain, 2.0, 1.0e-6) << machine.name;
    EXPECT_EQ(std::abs(arm.sign_q2), 1.0) << machine.name;
    EXPECT_EQ(std::abs(arm.sign_q3), 1.0) << machine.name;
  }
}

TEST(ArmGeometry, ThePlanarClosurePredictsForwardKinematics)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    auto limits = crane_planning::read_joint_limits(
      crane_planning_test::description(machine), model.urdf_joint_names());
    ASSERT_TRUE(limits.ok()) << machine.name;
    auto probed = crane_planning::probe_arm_geometry(
      model, limits.value(), 9, crane_planning::GeometryTolerance{});
    ASSERT_TRUE(probed.ok()) << machine.name << ": " << probed.status().message;
    const crane_planning::ArmGeometry & arm = probed.value();
    const crane_planning::JointLimits & axis = limits.value();

    std::mt19937_64 source(20260824U);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
      crane_model::Q q = crane_model::Q::Zero();
      q[1] = uniform(source, axis.axis[1].lower, axis.axis[1].upper);
      q[2] = uniform(source, axis.axis[2].lower, axis.axis[2].upper);
      q[3] = uniform(source, axis.axis[3].lower, axis.axis[3].upper);

      auto boom = model.forward_kinematics(
        q, crane_model::Frame::SlewingColumn, crane_model::Frame::Boom);
      auto tip = model.forward_kinematics(
        q, crane_model::Frame::SlewingColumn, crane_model::Frame::Tip);
      ASSERT_TRUE(boom.ok() && tip.ok()) << machine.name;

      const double boom_bearing = arm.gamma0 + arm.sign_q2 * q[1];
      const Eigen::Vector2d predicted_boom =
        arm.a2 * Eigen::Vector2d(std::cos(boom_bearing), std::sin(boom_bearing));
      const double forearm_bearing =
        boom_bearing + arm.sign_q3 * q[2] + arm.forearm_bearing(q[3]);
      const Eigen::Vector2d predicted_tip =
        predicted_boom + arm.forearm_length(q[3]) *
        Eigen::Vector2d(std::cos(forearm_bearing), std::sin(forearm_bearing));

      worst = std::max(worst, (predicted_boom - boom.value().position_m.head<2>()).norm());
      worst = std::max(worst, (predicted_tip - tip.value().position_m.head<2>()).norm());
      EXPECT_LT(std::abs(boom.value().position_m.z()), kTolerance) << machine.name;
      EXPECT_LT(std::abs(tip.value().position_m.z()), kTolerance) << machine.name;
    }
    EXPECT_LT(worst, kTolerance) << machine.name << ": the two-link closure and the model's own "
      "forward kinematics disagree by " << worst << " m";
  }
}

TEST(ArmGeometry, ADescriptionThatIsNotThisArmIsRefused)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  auto limits = crane_planning::read_joint_limits(
    crane_planning_test::description(machine), model.urdf_joint_names());
  ASSERT_TRUE(limits.ok());

  // An affine fit over two points is exact whatever the data, so the check that
  // the telescope really does extend along a fixed direction needs a third.
  auto under_determined =
    crane_planning::probe_arm_geometry(model, limits.value(), 2, crane_planning::GeometryTolerance{});
  EXPECT_FALSE(under_determined.ok());

  // And the structure check is a check: squeeze the tolerance below the noise
  // floor of the model's own arithmetic and the same description stops passing.
  crane_planning::GeometryTolerance impossible;
  impossible.structure_m = 0.0;
  impossible.planarity_m = 0.0;
  impossible.bearing_rad = 0.0;
  auto refused = crane_planning::probe_arm_geometry(model, limits.value(), 9, impossible);
  EXPECT_FALSE(refused.ok());
  EXPECT_FALSE(refused.status().message.empty());
}

TEST(JointLimits, TheDescriptionStatesThemAndTheyAreNotInvented)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    auto limits = crane_planning::read_joint_limits(
      crane_planning_test::description(machine), model.urdf_joint_names());
    ASSERT_TRUE(limits.ok()) << machine.name << ": " << limits.status().message;

    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      const crane_planning::AxisLimit & axis = limits.value().axis[row];
      EXPECT_GT(axis.dq_max, 0.0) << machine.name << " axis " << row;
      EXPECT_TRUE(std::isfinite(axis.dq_max)) << machine.name << " axis " << row;
      if (axis.bounded) {
        EXPECT_LT(axis.lower, axis.upper) << machine.name << " axis " << row;
      }
    }
    // The rotator is `continuous` in both descriptions: it has a velocity limit
    // and no range at all, and carrying an infinite bound is what keeps it out
    // of the centring score instead of steering it with an invented one.
    EXPECT_FALSE(limits.value().axis[4].bounded) << machine.name;
  }

  // A description that states nothing is refused rather than defaulted: urdfdom
  // rejects the document itself here, and a description that parses but omits
  // one of the six is refused a line later with `MissingJoint`.
  const crane_model::Model model =
    crane_planning_test::build_model(crane_planning_test::machines().front());
  auto refused =
    crane_planning::read_joint_limits("<robot name=\"empty\"/>", model.urdf_joint_names());
  EXPECT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code, crane_model::ErrorCode::InvalidRobotDescription);

  std::array<std::string, crane_model::kGeneralizedDof> renamed = model.urdf_joint_names();
  renamed[crane_planning::kActuatedRows[3]] = "a_telescope_this_description_does_not_have";
  auto missing = crane_planning::read_joint_limits(
    crane_planning_test::description(crane_planning_test::machines().front()), renamed);
  EXPECT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code, crane_model::ErrorCode::MissingJoint);
}
