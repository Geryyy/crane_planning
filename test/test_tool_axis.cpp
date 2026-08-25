// The tool coordinate on the arm's clock: the retained cosine primitive of
// `wiki/trajectory_planning.md` 8, the end of the range a close and an open
// drive to, the limits the phase is held inside, and the transmission reversal
// issue 037's notes measured inside the 7040 jaw's own range.
//
// Offline C++ against both machine descriptions. Nothing here launches, links a
// simulator or reaches a ROS graph, and no phase in this file solves an OCP:
// a tool phase is a cosine and three scalar bounds, and that is the whole cost
// of it.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "crane_planning/tool_axis.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning::ToolEnd;
using crane_planning::kToolRow;
using crane_planning_test::Machine;

constexpr double kTwoPi = 6.283185307179586;

/// The settings a phase is asked for, with the machine's own numbers on it.
crane_planning::ToolAxisRequest tool_request(
  const crane_planning::PlannerContext & context, const crane_model::QA & q_a_start,
  double q8_goal)
{
  crane_planning::ToolAxisRequest request;
  request.q_a_start = q_a_start;
  request.q8_goal = q8_goal;
  request.payload = crane_planning_test::empty_gripper();
  request.speed_scale = 1.0;
  request.kappa = context.settings.timing.kappa;
  request.actuation = context.settings.timing.actuation;
  request.sample_period = context.settings.timing.sample_period;
  request.min_duration = context.settings.ramp.min_duration;
  return request;
}

/// A payload held **off** the tilt axis, which is what moves the equilibrium.
/**
 * `wiki/robot_model.md` 5.1: an offset grasp moves where the tool hangs. A point
 * mass on the pendulum's own axis does not, so a payload declared with a zero
 * centre of mass would leave the passive pair exactly where an empty gripper
 * leaves it and the assertion would be about nothing.
 */
crane_model::Payload offset_block()
{
  crane_model::Payload payload = crane_planning_test::empty_gripper();
  payload.mass_kg = 400.0;
  payload.center_of_mass_k8_m = Eigen::Vector3d(0.25, 0.0, -0.6);
  return payload;
}

}  // namespace

TEST(ToolAxis, TheRetainedCosineIsCtwoAtBothEndsAndMonotoneBetween)
{
  // 8 keeps the legacy grip cosine primitive; 4.1 puts it on the arm's clock
  // beside five path coordinates that meet their ends with q_a' = q_a'' = 0.
  // A raised cosine in *position* does not: its curvature at s = 0 is pi^2 / 2
  // and not zero. Carrying the cosine on the **rate** is the same primitive one
  // integral up and it does.
  EXPECT_NEAR(crane_planning::grip_cosine(0.0), 0.0, 1.0e-15);
  EXPECT_NEAR(crane_planning::grip_cosine(1.0), 1.0, 1.0e-15);
  EXPECT_NEAR(crane_planning::grip_cosine_rate(0.0), 0.0, 1.0e-15);
  EXPECT_NEAR(crane_planning::grip_cosine_rate(1.0), 0.0, 1.0e-15);
  EXPECT_NEAR(crane_planning::grip_cosine_curvature(0.0), 0.0, 1.0e-15);
  EXPECT_NEAR(crane_planning::grip_cosine_curvature(1.0), 0.0, 1.0e-14);

  // The cosine is still in there, and it is the rate: h'(s) = 1 - cos(2 pi s).
  // Monotone, so a close never opens on its way to closing.
  double peak_rate = 0.0;
  double peak_curvature = 0.0;
  double previous = crane_planning::grip_cosine(0.0);
  constexpr std::size_t kSamples = 4001;
  for (std::size_t index = 0; index < kSamples; ++index) {
    const double s = static_cast<double>(index) / static_cast<double>(kSamples - 1U);
    EXPECT_NEAR(crane_planning::grip_cosine_rate(s), 1.0 - std::cos(kTwoPi * s), 1.0e-15) << s;
    EXPECT_GE(crane_planning::grip_cosine_rate(s), -1.0e-15) << s;
    const double value = crane_planning::grip_cosine(s);
    EXPECT_GE(value, previous - 1.0e-15) << s;
    previous = value;
    peak_rate = std::max(peak_rate, crane_planning::grip_cosine_rate(s));
    peak_curvature = std::max(peak_curvature, std::abs(crane_planning::grip_cosine_curvature(s)));
  }
  // The two constants the timing is built on, asserted rather than trusted.
  EXPECT_NEAR(peak_rate, crane_planning::kGripCosinePeakRate, 1.0e-6);
  EXPECT_NEAR(peak_curvature, crane_planning::kGripCosinePeakCurvature, 1.0e-5);
}

TEST(ToolAxis, TheToolRowOfTheCylinderJacobianIsAFunctionOfTheToolCoordinateAlone)
{
  // `probe_tool_transmission` sweeps q8 across one `Q` and leaves the arm and
  // the passive pair where the caller put them. That is only sound because the
  // tool row of J_cyl depends on nothing else, and this is where that is
  // measured rather than assumed -- if a later description coupled the jaw to
  // the arm, the probe would be reading the wrong number and this fails first.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA first =
      crane_planning_test::working_centred(context.limits, machine);
    crane_model::QA second = first;
    second[0] += 0.7;   // slew
    second[1] += 0.2;   // boom
    second[2] -= 0.3;   // arm
    second[3] += 0.3;   // telescope

    crane_model::Q left = crane_planning_test::settled(model, first);
    crane_model::Q right = crane_planning_test::settled(model, second);
    // ...and a passive pair deliberately away from either equilibrium.
    right[4] += 0.15;
    right[5] -= 0.12;

    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];
    for (int step = 0; step <= 8; ++step) {
      const double q8 = axis.lower + (axis.upper - axis.lower) * step / 8.0;
      left[static_cast<Eigen::Index>(crane_planning::kActuatedRows[kToolRow])] = q8;
      right[static_cast<Eigen::Index>(crane_planning::kActuatedRows[kToolRow])] = q8;
      auto a = model.cylinder_jacobian(left);
      auto b = model.cylinder_jacobian(right);
      ASSERT_TRUE(a.ok() && b.ok()) << machine.name;
      const Eigen::Index row = static_cast<Eigen::Index>(kToolRow);
      EXPECT_NEAR(a.value()(row, row), b.value()(row, row), 1.0e-15)
        << machine.name << " at q8 = " << q8;
    }
  }
}

TEST(ToolAxis, TheJawReversesSignInsideItsOwnRangeAndTheRailDoesNot)
{
  // Issue 037's notes recorded this reversal on the 7040 and put it near
  // q8 = 0.29 rad -- which is the linear interpolation of the six-point table
  // they tabulate, between -0.016 m/rad at 0.20 and +0.057 m/rad at 0.50.
  // Bisected off the model the root is **0.2623 rad**, and the table itself is
  // reproduced exactly, so 0.29 is that table read straight rather than a
  // different measurement. What the planner refuses on is the bisected root, so
  // that is what is pinned here. The PZS100's rail cylinder is one-to-one with
  // q8 over the whole of 0.2 ... 0.7 m, so there is nothing to cross there.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
    const crane_model::Q q = crane_planning_test::settled(model, q_a);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];

    auto probed = crane_planning::probe_tool_transmission(
      model, q, axis.lower, axis.upper, context.settings.tool_axis);
    ASSERT_TRUE(probed.ok()) << machine.name << ": " << probed.status().message;

    if (machine.tool == crane_model::Tool::Epsilon7040) {
      EXPECT_TRUE(probed.value().crossed) << machine.name;
      EXPECT_NEAR(probed.value().q8, 0.2623, 1.0e-3)
        << "the jaw four-bar's own root, which issue 037's table brackets";
      // The table itself, at the two entries that bracket the root.
      auto below = crane_planning::probe_tool_transmission(
        model, q, 0.2, 0.2, context.settings.tool_axis);
      auto above = crane_planning::probe_tool_transmission(
        model, q, 0.5, 0.5, context.settings.tool_axis);
      ASSERT_TRUE(below.ok() && above.ok());
      EXPECT_NEAR(below.value().ratio_lower, -0.016, 1.0e-3);
      EXPECT_NEAR(above.value().ratio_lower, 0.057, 1.0e-3);
      // Negative below the crossing, positive above: the cylinder retracts as
      // the jaw opens on one side of it and extends on the other.
      EXPECT_LT(probed.value().ratio_lower, 0.0);
      EXPECT_GT(probed.value().ratio_upper, 0.0);
    } else {
      EXPECT_FALSE(probed.value().crossed) << machine.name;
      EXPECT_GT(probed.value().smallest_ratio, context.settings.tool_axis.transmission_floor);
    }
  }
}

TEST(ToolAxis, ATravelThatSpansTheReversalIsRefusedNamingItAndOneThatDoesNotIsPlanned)
{
  // The 7040 only. A close from an open jaw crosses the toggle, and at the
  // toggle the cylinder moves the jaw through no distance at all -- so the axis
  // is not controllable there and a velocity reference written through it is a
  // reference nothing can follow. Refused, naming the crossing; not clipped to
  // it, because a grip that stopped at the toggle and reported success would
  // read as a grip that closed.
  const Machine & machine = crane_planning_test::machines().at(1);
  ASSERT_EQ(machine.tool, crane_model::Tool::Epsilon7040);
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);

  crane_model::QA open_jaw = crane_planning_test::working_centred(context.limits, machine);
  open_jaw[static_cast<Eigen::Index>(kToolRow)] = 1.0;
  auto refused = crane_planning::drive_tool_axis(
    model, context.limits, context.settings.tool_axis,
    tool_request(context, open_jaw, context.limits.axis[kToolRow].lower));
  ASSERT_FALSE(refused.ok()) << "a close across the jaw's own transmission reversal was planned";
  EXPECT_NE(refused.status().message.find("reverses sign"), std::string::npos)
    << refused.status().message;
  EXPECT_NE(refused.status().message.find("0.262"), std::string::npos)
    << refused.status().message;

  // ...and the same axis, on one side of the crossing, is planned.
  const crane_model::QA closed_side =
    crane_planning_test::working_centred(context.limits, machine);
  ASSERT_LT(closed_side[static_cast<Eigen::Index>(kToolRow)], 0.2623);
  auto planned = crane_planning::drive_tool_axis(
    model, context.limits, context.settings.tool_axis,
    tool_request(context, closed_side, context.limits.axis[kToolRow].lower));
  ASSERT_TRUE(planned.ok()) << planned.status().message;
  EXPECT_NEAR(
    planned.value().trajectory.q_a_ref.back()[static_cast<Eigen::Index>(kToolRow)],
    context.limits.axis[kToolRow].lower, 1.0e-12);
}

TEST(ToolAxis, AClosePhaseAndAnOpenPhaseDriveToTheTwoEndsOfTheDescriptionsOwnRange)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];
    ASSERT_TRUE(axis.bounded) << machine.name;

    // The shipped reading: the lower end is the closed gripper.
    auto closing = crane_planning::tool_axis_target(context.limits, ToolEnd::Lower, true);
    auto opening = crane_planning::tool_axis_target(context.limits, ToolEnd::Lower, false);
    ASSERT_TRUE(closing.ok() && opening.ok()) << machine.name;
    EXPECT_DOUBLE_EQ(closing.value(), axis.lower);
    EXPECT_DOUBLE_EQ(opening.value(), axis.upper);

    // ...and a deployment that measures the other polarity gets the other end,
    // which is the whole reason it is a parameter and not a constant.
    auto flipped = crane_planning::tool_axis_target(context.limits, ToolEnd::Upper, true);
    ASSERT_TRUE(flipped.ok()) << machine.name;
    EXPECT_DOUBLE_EQ(flipped.value(), axis.upper);
  }
}

TEST(ToolAxis, ThePhaseRidesTheArmsSampleGridAndStaysInsideTheToolAxisOwnLimits)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];

    // Toward the closed end on both machines, which is the direction neither
    // tool's transmission reverses in from `working_centred`'s own q8.
    auto driven = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, tool_request(context, q_a, axis.lower));
    ASSERT_TRUE(driven.ok()) << machine.name << ": " << driven.status().message;
    const crane_planning::ToolPhase & phase = driven.value();
    const crane_planning::TimedTrajectory & trajectory = phase.trajectory;

    // The arm's own grid: first point at zero, every step the sample period, the
    // last point exactly at the duration.
    const double period = context.settings.timing.sample_period;
    ASSERT_GE(trajectory.time_from_start.size(), 2U) << machine.name;
    EXPECT_DOUBLE_EQ(trajectory.time_from_start.front(), 0.0);
    EXPECT_DOUBLE_EQ(trajectory.time_from_start.back(), trajectory.duration);
    for (std::size_t index = 1; index + 1U < trajectory.time_from_start.size(); ++index) {
      EXPECT_NEAR(
        trajectory.time_from_start[index] - trajectory.time_from_start[index - 1U], period,
        1.0e-12) << machine.name << " at " << index;
    }

    const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
    for (std::size_t index = 0; index < trajectory.q_a_ref.size(); ++index) {
      // The five path coordinates are held and *emitted*, not left out: the
      // reference is over all six actuated rows (trajectory_planning 4.1).
      for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
        const Eigen::Index at = static_cast<Eigen::Index>(row);
        EXPECT_NEAR(trajectory.q_a_ref[index][at], q_a[at], 1.0e-15) << machine.name;
        EXPECT_NEAR(trajectory.dq_a_ref[index][at], 0.0, 1.0e-15) << machine.name;
      }
      EXPECT_GE(trajectory.q_a_ref[index][tool], axis.lower - 1.0e-12) << machine.name;
      EXPECT_LE(trajectory.q_a_ref[index][tool], axis.upper + 1.0e-12) << machine.name;
    }
    // At rest at both ends, and landing exactly on the target.
    EXPECT_NEAR(trajectory.dq_a_ref.front()[tool], 0.0, 1.0e-12) << machine.name;
    EXPECT_NEAR(trajectory.dq_a_ref.back()[tool], 0.0, 1.0e-12) << machine.name;
    EXPECT_NEAR(trajectory.q_a_ref.front()[tool], q_a[tool], 1.0e-15) << machine.name;
    EXPECT_NEAR(trajectory.q_a_ref.back()[tool], axis.lower, 1.0e-12) << machine.name;

    // Inside the tool axis's own three limits, with kappa held back from each.
    const crane_planning::ToolAxisDemand & demand = phase.demand;
    EXPECT_LE(demand.velocity_fraction, context.settings.timing.kappa + 1.0e-9) << machine.name;
    EXPECT_LE(demand.acceleration_fraction, context.settings.timing.kappa + 1.0e-9)
      << machine.name;
    EXPECT_LE(
      demand.flow_fraction,
      context.settings.timing.kappa * context.settings.timing.actuation.pump_flow_planning_factor +
      1.0e-9) << machine.name;
    // The peaks the emitted samples really reach, measured off the trajectory
    // rather than off the planner's own report.
    double peak_rate = 0.0;
    for (const crane_model::DQA & velocity : trajectory.dq_a_ref) {
      peak_rate = std::max(peak_rate, std::abs(velocity[tool]));
    }
    EXPECT_LE(peak_rate, context.settings.timing.kappa * axis.dq_max + 1.0e-9) << machine.name;
    EXPECT_NEAR(peak_rate, demand.peak_rate, 1.0e-3 * demand.peak_rate) << machine.name;
  }
}

TEST(ToolAxis, SpeedScaleSlowsThePhaseAndCannotReachKappa)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    // The whole travel toward the closed end -- on the 7040 only as far as the
    // closed side of its own transmission reversal reaches.
    crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];
    q_a[static_cast<Eigen::Index>(kToolRow)] =
      (machine.tool == crane_model::Tool::Epsilon7040) ? 0.2 : axis.upper;

    crane_planning::ToolAxisRequest request = tool_request(context, q_a, axis.lower);
    auto full = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, request);
    ASSERT_TRUE(full.ok()) << machine.name << ": " << full.status().message;

    request.speed_scale = 0.5;
    auto halved = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, request);
    ASSERT_TRUE(halved.ok()) << machine.name << ": " << halved.status().message;
    EXPECT_GE(halved.value().demand.duration, full.value().demand.duration) << machine.name;
    EXPECT_LE(halved.value().demand.peak_rate, full.value().demand.peak_rate + 1.0e-12)
      << machine.name;
    // `speed_scale` multiplies the **velocity** bound alone, exactly as it does
    // in the OCP, so it lengthens a phase only where that bound is what chose
    // the duration. On the PZS100's rail it is; on the 7040's jaw the
    // acceleration limit binds over the whole of the joint's own range -- the
    // travel would have to be some 7.7 rad for velocity to win, and the range is
    // 3.15 -- so a jaw phase is not slowed and is right not to be.
    if (full.value().demand.velocity_binding) {
      EXPECT_GT(halved.value().demand.duration, full.value().demand.duration) << machine.name;
    } else {
      EXPECT_TRUE(full.value().demand.acceleration_binding || full.value().demand.flow_binding ||
        full.value().demand.duration_floor_binding) << machine.name;
    }

    // kappa is the deployment's reservation and no request reaches it: both of
    // these sit at or under it, and a speed_scale outside (0, 1] is refused
    // rather than clamped.
    EXPECT_LE(full.value().demand.velocity_fraction, context.settings.timing.kappa + 1.0e-9);
    EXPECT_LE(halved.value().demand.velocity_fraction, context.settings.timing.kappa + 1.0e-9);
    for (const double speed_scale : {0.0, 1.5}) {
      request.speed_scale = speed_scale;
      auto refused = crane_planning::drive_tool_axis(
        model, context.limits, context.settings.tool_axis, request);
      EXPECT_FALSE(refused.ok()) << machine.name << " at speed_scale = " << speed_scale;
      EXPECT_NE(refused.status().message.find("speed_scale"), std::string::npos)
        << refused.status().message;
    }
  }
}

TEST(ToolAxis, AClosePhaseAndAnOpenPhaseOverOneSpanDrawDifferentFlow)
{
  // `Q_i = A_i^{+-}(v_i) |v_i|` (crane_model's own transmission, and the
  // expression the OCP reads off the symbolic graph): the effective area is a
  // step at v = 0, so the pump draw per metre of piston travel is not the same
  // extending and retracting. A close and an open over the same span of q8 are
  // therefore not one phase reversed, and the flow limit binds them differently.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];

    const double from = axis.lower;
    // On the 7040, both ends on the closed side of the jaw's own reversal.
    const double to = (machine.tool == crane_model::Tool::Epsilon7040) ? 0.2 : axis.upper;
    crane_model::QA at_lower = crane_planning_test::working_centred(context.limits, machine);
    at_lower[static_cast<Eigen::Index>(kToolRow)] = from;
    crane_model::QA at_upper = at_lower;
    at_upper[static_cast<Eigen::Index>(kToolRow)] = to;

    auto outward = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, tool_request(context, at_lower, to));
    auto inward = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, tool_request(context, at_upper, from));
    ASSERT_TRUE(outward.ok()) << machine.name << ": " << outward.status().message;
    ASSERT_TRUE(inward.ok()) << machine.name << ": " << inward.status().message;
    EXPECT_NE(
      outward.value().demand.flow_per_unit_rate, inward.value().demand.flow_per_unit_rate)
      << machine.name
      << ": the two directions drew the same flow per unit rate, so the chamber switch of "
         "crane_model's transmission is not reaching this phase";
  }
}

TEST(ToolAxis, ThePayloadMovesTheEquilibriumThePhaseReports)
{
  // The emitted reference is six actuated rows and a payload does not move any
  // of them: what it moves is the **passive** pair (robot_model 5.1). So the
  // phase carries where the tool hangs at each of its two ends, and this is the
  // assertion that the payload reached the phase at all.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
    const crane_planning::AxisLimit & axis = context.limits.axis[kToolRow];

    crane_planning::ToolAxisRequest empty = tool_request(context, q_a, axis.lower);
    crane_planning::ToolAxisRequest carried = empty;
    carried.payload = offset_block();

    auto without = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, empty);
    auto with = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, carried);
    ASSERT_TRUE(without.ok()) << machine.name << ": " << without.status().message;
    ASSERT_TRUE(with.ok()) << machine.name << ": " << with.status().message;

    EXPECT_GT(
      (with.value().q_u_goal - without.value().q_u_goal).norm(), 1.0e-3)
      << machine.name << ": a block held 0.25 m off the tilt axis left the tool hanging where an "
                         "empty gripper leaves it";
    EXPECT_GT((with.value().q_u_start - without.value().q_u_start).norm(), 1.0e-3) << machine.name;
  }
}

TEST(ToolAxis, TwoIdenticalPhasesAreOneTrajectory)
{
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
    const crane_planning::ToolAxisRequest request =
      tool_request(context, q_a, context.limits.axis[kToolRow].lower);

    auto first = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, request);
    auto second = crane_planning::drive_tool_axis(
      model, context.limits, context.settings.tool_axis, request);
    ASSERT_TRUE(first.ok() && second.ok()) << machine.name;
    ASSERT_EQ(
      first.value().trajectory.time_from_start.size(),
      second.value().trajectory.time_from_start.size()) << machine.name;
    for (std::size_t index = 0; index < first.value().trajectory.q_a_ref.size(); ++index) {
      EXPECT_DOUBLE_EQ(
        first.value().trajectory.time_from_start[index],
        second.value().trajectory.time_from_start[index]) << machine.name;
      EXPECT_TRUE(
        first.value().trajectory.q_a_ref[index] == second.value().trajectory.q_a_ref[index])
        << machine.name << " at " << index;
      EXPECT_TRUE(
        first.value().trajectory.dq_a_ref[index] == second.value().trajectory.dq_a_ref[index])
        << machine.name << " at " << index;
    }
  }
}
