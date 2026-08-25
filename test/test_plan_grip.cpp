// The four phases of `crane_msgs/PlanGrip`, offline and per tool.
//
// What this binary is about is the three things the phases owe each other and
// that no single phase can be asked about on its own:
//
//   1. **the descend endpoint is an equilibrium solution** -- issue 039's route,
//      not the passive pair pinned where it happened to be measured -- and a
//      descend whose endpoint has no equilibrium is refused with the reason;
//   2. **one clock**: a close phase and an arm phase come back on the same time
//      base, and a phase starts where the machine is, so a sequence joins;
//   3. **no memory**: two identical requests are one trajectory and which phase
//      ran last is not a fact this planner has.
//
// The two arm phases are `plan_motion` itself -- same endpoint, same primitive,
// same collision check, same sway envelope, same kappa -- so this binary solves
// the OCP of `wiki/trajectory_planning.md` 5.2 once per arm phase and is priced
// accordingly. The two tool phases are the cosine of `tool_axis.hpp` and cost
// nothing; `test_tool_axis.cpp` is where that primitive is asserted in detail.
//
// Offline C++ against both machine descriptions. Every goal pose is produced by
// `crane_model::forward_kinematics` from a configuration the description allows,
// so what is asked for is a pose that certainly exists -- except the one case
// that is deliberately built the other way round. Nothing here launches, links a
// simulator or reaches a ROS graph.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "crane_planning/plan_grip.hpp"
#include "description_fixture.hpp"

namespace
{

using crane_planning::GripPhase;
using crane_planning::GripRequest;
using crane_planning::kToolRow;
using crane_planning_test::Machine;

/// How close to the slewing axis a goal may be, m -- `robot_model` 2.2 step 1
/// has no azimuth on the axis itself.
constexpr double kAxisExclusion = 2.0;

/// The canonical eight of one actuated configuration, with the tool hanging.
crane_model::Q settled_with(
  const crane_model::Model & model, const crane_model::QA & q_a,
  const crane_model::Payload & payload)
{
  auto equilibrium = model.passive_equilibrium(q_a, payload);
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

/// Where the tool hangs, in K0, at one actuated configuration.
crane_model::Pose tcp_of(
  const crane_model::Model & model, const crane_model::QA & q_a,
  const crane_model::Payload & payload)
{
  auto pose = model.forward_kinematics(
    settled_with(model, q_a, payload), crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  if (!pose.ok()) {
    throw std::runtime_error(pose.status().message);
  }
  return pose.value();
}

/// The actuated projection of a canonical eight-vector.
crane_model::QA actuated_of(const crane_model::Q & q)
{
  crane_model::QA q_a;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] =
      q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])];
  }
  return q_a;
}

/// One request, with the machine's own start on it.
GripRequest phase_request(GripPhase phase, const crane_model::QA & q_a_start)
{
  GripRequest request;
  request.phase = phase;
  request.q_a_start = q_a_start;
  request.payload = crane_planning_test::empty_gripper();
  request.speed_scale = 1.0;
  // The offline cases are about the phases and not about a scene; the served
  // contract is where `avoid_collisions` is exercised, because `/crane/plan_grip`
  // has no row for it and the node sets it.
  request.avoid_collisions = false;
  return request;
}

/// An arm phase's goal, taken off a configuration the description allows.
void aim_at(
  const crane_model::Model & model, const crane_model::QA & q_a_goal,
  const crane_model::Payload & payload, GripRequest & request)
{
  const crane_model::Pose pose = tcp_of(model, q_a_goal, payload);
  request.p_tcp_0 = pose.position_m;
  request.phi_z_d = crane_planning::phi_z_of(pose.orientation);
}

/// The same start every timing case in this package uses, with a stated tool coordinate.
crane_model::QA start_of(
  const crane_planning::PlannerContext & context, const Machine & machine, double q8)
{
  crane_model::QA q_a = crane_planning_test::working_centred(context.limits, machine);
  q_a[static_cast<Eigen::Index>(kToolRow)] = q8;
  return q_a;
}

/// A payload held off the tilt axis, which is what moves the equilibrium (5.1).
/**
 * The same block `test_equilibrium_ik.cpp` uses, and for the same reason: a
 * point mass on the pendulum's own axis leaves the tool hanging exactly where an
 * empty gripper does, so a payload with no lateral offset would make the
 * assertion below about nothing.
 */
crane_model::Payload offset_block()
{
  crane_model::Payload payload = crane_planning_test::empty_gripper();
  payload.mass_kg = 300.0;
  payload.center_of_mass_k8_m = Eigen::Vector3d(0.05, -0.02, 1.1);
  return payload;
}

/// Every point of an emitted trajectory is six rows of positions and velocities.
void expect_well_formed(
  const crane_planning::TimedTrajectory & trajectory, const crane_model::QA & q_a_start,
  double sample_period, const char * what)
{
  ASSERT_GE(trajectory.time_from_start.size(), 2U) << what;
  ASSERT_EQ(trajectory.q_a_ref.size(), trajectory.time_from_start.size()) << what;
  ASSERT_EQ(trajectory.dq_a_ref.size(), trajectory.time_from_start.size()) << what;
  EXPECT_DOUBLE_EQ(trajectory.time_from_start.front(), 0.0) << what;
  EXPECT_DOUBLE_EQ(trajectory.time_from_start.back(), trajectory.duration) << what;
  for (std::size_t index = 1; index < trajectory.time_from_start.size(); ++index) {
    EXPECT_GT(trajectory.time_from_start[index], trajectory.time_from_start[index - 1U]) << what;
    // The arm's own grid, and the same one for every phase: the sample period,
    // except for the last step, which is whatever is left over to the duration.
    const double step = trajectory.time_from_start[index] - trajectory.time_from_start[index - 1U];
    EXPECT_LE(step, sample_period + 1.0e-12) << what << " at " << index;
    if (index + 1U < trajectory.time_from_start.size()) {
      EXPECT_NEAR(step, sample_period, 1.0e-12) << what << " at " << index;
    }
  }
  // It starts where the machine is, and it ends at rest.
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(trajectory.q_a_ref.front()[axis], q_a_start[axis], 1.0e-9) << what << " row "
                                                                          << row;
    EXPECT_NEAR(trajectory.dq_a_ref.front()[axis], 0.0, 1.0e-9) << what << " row " << row;
    EXPECT_NEAR(trajectory.dq_a_ref.back()[axis], 0.0, 1.0e-9) << what << " row " << row;
  }
}

}  // namespace

TEST(PlanGrip, EveryPhaseAnswersForEveryTool)
{
  // "Offline, deterministic tests per phase per tool." All four, on both
  // machines, each coming back over the six actuated joints on one time base.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const double period = context.settings.timing.sample_period;
    const bool jaw = machine.tool == crane_model::Tool::Epsilon7040;

    // The 7040's jaw ratio reverses at q8 ~= 0.29 rad, so a phase that opens
    // has to start above the crossing and a phase that closes below it. The
    // PZS100's rail is one-to-one and starts wherever.
    const double closed_side = jaw ? 0.2 : machine.q8;
    const double open_side = jaw ? 0.5 : machine.q8;

    {
      const crane_model::QA q_a = start_of(context, machine, closed_side);
      crane_model::QA lower = q_a;
      lower[1] -= 0.15;  // theta2_boom_joint, which drops the tool
      ASSERT_LT(
        tcp_of(model, lower, crane_planning_test::empty_gripper()).position_m.z(),
        tcp_of(model, q_a, crane_planning_test::empty_gripper()).position_m.z()) << machine.name;

      GripRequest request = phase_request(GripPhase::Descend, q_a);
      aim_at(model, lower, request.payload, request);
      auto planned = crane_planning::plan_grip(model, context, request);
      ASSERT_TRUE(planned.ok()) << machine.name << ": " << planned.status().message;
      EXPECT_TRUE(planned.value().arm_phase);
      expect_well_formed(planned.value().trajectory, q_a, period, "descend");
    }
    {
      const crane_model::QA q_a = start_of(context, machine, closed_side);
      crane_model::QA higher = q_a;
      higher[1] += 0.15;
      ASSERT_GT(
        tcp_of(model, higher, crane_planning_test::empty_gripper()).position_m.z(),
        tcp_of(model, q_a, crane_planning_test::empty_gripper()).position_m.z()) << machine.name;

      GripRequest request = phase_request(GripPhase::Lift, q_a);
      aim_at(model, higher, request.payload, request);
      auto planned = crane_planning::plan_grip(model, context, request);
      ASSERT_TRUE(planned.ok()) << machine.name << ": " << planned.status().message;
      EXPECT_TRUE(planned.value().arm_phase);
      expect_well_formed(planned.value().trajectory, q_a, period, "lift");
    }
    {
      const crane_model::QA q_a = start_of(context, machine, closed_side);
      auto planned =
        crane_planning::plan_grip(model, context, phase_request(GripPhase::Close, q_a));
      ASSERT_TRUE(planned.ok()) << machine.name << ": " << planned.status().message;
      EXPECT_FALSE(planned.value().arm_phase);
      expect_well_formed(planned.value().trajectory, q_a, period, "close");
      EXPECT_NEAR(
        planned.value().trajectory.q_a_ref.back()[static_cast<Eigen::Index>(kToolRow)],
        context.limits.axis[kToolRow].lower, 1.0e-12) << machine.name;
    }
    {
      const crane_model::QA q_a = start_of(context, machine, open_side);
      auto planned = crane_planning::plan_grip(model, context, phase_request(GripPhase::Open, q_a));
      ASSERT_TRUE(planned.ok()) << machine.name << ": " << planned.status().message;
      EXPECT_FALSE(planned.value().arm_phase);
      expect_well_formed(planned.value().trajectory, q_a, period, "open");
      EXPECT_NEAR(
        planned.value().trajectory.q_a_ref.back()[static_cast<Eigen::Index>(kToolRow)],
        context.limits.axis[kToolRow].upper, 1.0e-12) << machine.name;
    }
  }
}

TEST(PlanGrip, TheJawIsRefusedByNameWhenAPhaseWouldSpanItsTransmissionReversal)
{
  // The tool differs and the planner knows it. On the 7040 an open from the
  // closed side of the jaw's own toggle spans it, and issue 037's notes are what
  // the refusal cites. On the PZS100 the same phase is planned, because that
  // tool's rail cylinder is one-to-one with q8 over the whole range.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::QA q_a = start_of(context, machine, context.limits.axis[kToolRow].lower);
    auto planned = crane_planning::plan_grip(model, context, phase_request(GripPhase::Open, q_a));

    if (machine.tool == crane_model::Tool::Epsilon7040) {
      ASSERT_FALSE(planned.ok()) << "an open across the jaw's transmission reversal was planned";
      EXPECT_NE(planned.status().message.find("reverses sign"), std::string::npos)
        << planned.status().message;
      EXPECT_NE(planned.status().message.find("open phase"), std::string::npos)
        << planned.status().message;
    } else {
      ASSERT_TRUE(planned.ok()) << machine.name << ": " << planned.status().message;
    }
  }
}

TEST(PlanGrip, TheDescendEndpointIsAnEquilibriumSolutionAndNotAPinning)
{
  // `trajectory_planning` 6: the endpoint is the equilibrium-constrained IK of
  // `robot_model` 2.2, so the tool arrives hanging still. Pinning the passive
  // joints at their measured values and descending anyway is the failure this
  // is here to prevent, and the two assertions are the two halves of 2.2's own
  // acceptance test -- which route ran, and that the answer really is a steady
  // state of the passive subsystem when put back through the frozen model API.
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const crane_model::QA q_a = start_of(context, machine, machine.q8);
  crane_model::QA lower = q_a;
  lower[1] -= 0.15;

  GripRequest request = phase_request(GripPhase::Descend, q_a);
  aim_at(model, lower, request.payload, request);
  auto planned = crane_planning::plan_grip(model, context, request);
  ASSERT_TRUE(planned.ok()) << planned.status().message;

  const crane_planning::IkSolution & endpoint = planned.value().motion.endpoint;
  EXPECT_EQ(endpoint.route, crane_planning::EndpointRoute::EquilibriumConstrained);
  EXPECT_LT(endpoint.residual_equilibrium, context.settings.equilibrium.eps_equilibrium);

  // ...and re-measured here rather than read off the solver's own report.
  auto settled = model.passive_equilibrium(actuated_of(endpoint.q), request.payload);
  ASSERT_TRUE(settled.ok()) << settled.status().message;
  EXPECT_LT(
    (endpoint.q.segment<2>(4) - settled.value()).norm(),
    context.settings.equilibrium.eps_equilibrium);
}

TEST(PlanGrip, ADescendWhoseEndpointHasNoEquilibriumIsRefusedWithTheReason)
{
  // The goal is the **wrist's** own highest reachable point beyond the slewing
  // exclusion, so a configuration that puts K5 exactly there exists by
  // construction and the point is inside the geometric workspace 2.2 steps 1-2
  // close over. The tool hangs below the wrist at every equilibrium, so no
  // equilibrium reaches it -- and the two measurements that argument needs are
  // taken here rather than assumed:
  //
  //     tcp_z  <=  tip_z - drop  <=  highest_tip_z - drop  <  goal.z()
  //
  // A descend answered here would be a descend planned with the passive joints
  // held somewhere they will not stay, which is exactly what this phase must not
  // do.
  for (const Machine & machine : crane_planning_test::machines()) {
    const crane_model::Model model = crane_planning_test::build_model(machine);
    const crane_planning::PlannerContext context =
      crane_planning_test::build_context(model, machine);
    const crane_model::Payload payload = crane_planning_test::empty_gripper();

    double highest = -std::numeric_limits<double>::infinity();
    double goal_z = -std::numeric_limits<double>::infinity();
    Eigen::Vector3d goal = Eigen::Vector3d::Zero();
    constexpr int kBoomSteps = 17;
    constexpr int kTelescopeSteps = 7;
    for (int i2 = 0; i2 < kBoomSteps; ++i2) {
      for (int i3 = 0; i3 < kBoomSteps; ++i3) {
        for (int i4 = 0; i4 < kTelescopeSteps; ++i4) {
          const std::array<int, 3> step{{i2, i3, i4}};
          const std::array<int, 3> steps{{kBoomSteps - 1, kBoomSteps - 1, kTelescopeSteps - 1}};
          crane_model::Q q = crane_model::Q::Zero();
          for (std::size_t row = 1; row < 4U; ++row) {
            const crane_planning::AxisLimit & axis = context.limits.axis[row];
            q[static_cast<Eigen::Index>(row)] = axis.lower +
              (axis.upper - axis.lower) * step[row - 1U] / steps[row - 1U];
          }
          auto pose = model.forward_kinematics(
            q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
          if (!pose.ok()) {
            continue;
          }
          const Eigen::Vector3d p = pose.value().position_m;
          highest = std::max(highest, p.z());
          if (p.head<2>().norm() >= kAxisExclusion && p.z() > goal_z) {
            goal_z = p.z();
            goal = p;
          }
        }
      }
    }
    ASSERT_GT(goal_z, -std::numeric_limits<double>::infinity()) << machine.name;

    std::mt19937_64 source(4404U);
    double drop = std::numeric_limits<double>::infinity();
    for (int index = 0; index < 40; ++index) {
      crane_model::QA q_a = crane_model::QA::Zero();
      for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
        const crane_planning::AxisLimit & axis = context.limits.axis[row];
        const double lower = axis.bounded ? axis.lower : -M_PI;
        const double upper = axis.bounded ? axis.upper : M_PI;
        q_a[static_cast<Eigen::Index>(row)] =
          std::uniform_real_distribution<double>(lower, upper)(source);
      }
      q_a[static_cast<Eigen::Index>(kToolRow)] = machine.q8;
      auto equilibrium = model.passive_equilibrium(q_a, payload);
      if (!equilibrium.ok()) {
        continue;
      }
      crane_model::Q q = crane_model::Q::Zero();
      for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
        q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
          q_a[static_cast<Eigen::Index>(row)];
      }
      q.segment<2>(4) = equilibrium.value();
      auto tip = model.forward_kinematics(
        q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
      auto tcp = model.forward_kinematics(
        q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
      ASSERT_TRUE(tip.ok() && tcp.ok()) << machine.name;
      drop = std::min(drop, tip.value().position_m.z() - tcp.value().position_m.z());
    }
    // The premise, asserted rather than assumed.
    ASSERT_GT(goal.z(), highest - drop) << machine.name
                                        << ": the hanging tool reaches the goal after all";

    GripRequest request =
      phase_request(GripPhase::Descend, start_of(context, machine, machine.q8));
    request.p_tcp_0 = goal;
    request.phi_z_d = 0.0;
    auto refused = crane_planning::plan_grip(model, context, request);
    ASSERT_FALSE(refused.ok()) << machine.name << ": a descend onto a goal "
                               << goal.z() - (highest - drop)
                               << " m above anything the hanging tool reaches was answered";
    EXPECT_NE(refused.status().message.find("no passive equilibrium"), std::string::npos)
      << refused.status().message;
    EXPECT_NE(refused.status().message.find("descend phase"), std::string::npos)
      << refused.status().message;
  }
}

TEST(PlanGrip, EveryPhaseIsOnTheArmsOneTimeBaseAndAPhaseStartsWhereTheLastOneEnded)
{
  // 4.1: "geometrically the tool is decoupled from the arm; temporally it is
  // not, and the two must share one clock." A close and a descend come back on
  // the same grid -- the same sample period, the first point at zero -- and
  // because a phase starts at the machine's measured configuration, sequencing
  // the four by moving the machine and re-measuring joins them exactly. That is
  // what makes "no phase returns a trajectory whose first point is not at the
  // previous phase's last" a property of the construction rather than of
  // bookkeeping the planner would have to keep.
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const double period = context.settings.timing.sample_period;

  const crane_model::QA q_a0 = start_of(context, machine, machine.q8);
  crane_model::QA lower = q_a0;
  lower[1] -= 0.15;
  GripRequest descend = phase_request(GripPhase::Descend, q_a0);
  aim_at(model, lower, descend.payload, descend);
  auto first = crane_planning::plan_grip(model, context, descend);
  ASSERT_TRUE(first.ok()) << first.status().message;

  const crane_model::QA q_a1 = first.value().trajectory.q_a_ref.back();
  auto second =
    crane_planning::plan_grip(model, context, phase_request(GripPhase::Close, q_a1));
  ASSERT_TRUE(second.ok()) << second.status().message;

  const crane_model::QA q_a2 = second.value().trajectory.q_a_ref.back();
  crane_model::QA higher = q_a2;
  higher[1] += 0.15;
  GripRequest lift = phase_request(GripPhase::Lift, q_a2);
  aim_at(model, higher, lift.payload, lift);
  auto third = crane_planning::plan_grip(model, context, lift);
  ASSERT_TRUE(third.ok()) << third.status().message;

  // One time base: the same period, both starting at zero.
  const std::array<const crane_planning::TimedTrajectory *, 3U> phases{
    {&first.value().trajectory, &second.value().trajectory, &third.value().trajectory}};
  for (const crane_planning::TimedTrajectory * trajectory : phases) {
    EXPECT_DOUBLE_EQ(trajectory->time_from_start.front(), 0.0);
    ASSERT_GE(trajectory->time_from_start.size(), 3U);
    EXPECT_NEAR(trajectory->time_from_start[1] - trajectory->time_from_start[0], period, 1.0e-12);
    EXPECT_NEAR(trajectory->time_from_start[2] - trajectory->time_from_start[1], period, 1.0e-12);
  }

  // ...and the joins, over all six rows including q8.
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(second.value().trajectory.q_a_ref.front()[axis], q_a1[axis], 1.0e-12) << row;
    EXPECT_NEAR(third.value().trajectory.q_a_ref.front()[axis], q_a2[axis], 1.0e-12) << row;
  }
  // The close really moved the tool and held the arm; the arm phases held the tool.
  const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
  EXPECT_NEAR(q_a1[tool], q_a0[tool], 1.0e-9) << "the descend moved the tool coordinate";
  EXPECT_NEAR(q_a2[tool], context.limits.axis[kToolRow].lower, 1.0e-12);
  for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(q_a2[axis], q_a1[axis], 1.0e-12) << "the close moved path coordinate " << row;
  }
}

TEST(PlanGrip, TwoIdenticalRequestsAreOneTrajectoryAndNoPhaseRemembersTheLast)
{
  // "Phase transitions carry no hidden state." The same request answered twice,
  // with another phase in between, is the same trajectory: the planner's answer
  // is a function of the request and the measured start alone.
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const crane_model::QA q_a = start_of(context, machine, machine.q8);

  const GripRequest close = phase_request(GripPhase::Close, q_a);
  auto before = crane_planning::plan_grip(model, context, close);
  ASSERT_TRUE(before.ok()) << before.status().message;

  crane_model::QA lower = q_a;
  lower[1] -= 0.15;
  GripRequest descend = phase_request(GripPhase::Descend, q_a);
  aim_at(model, lower, descend.payload, descend);
  auto between = crane_planning::plan_grip(model, context, descend);
  ASSERT_TRUE(between.ok()) << between.status().message;

  auto after = crane_planning::plan_grip(model, context, close);
  ASSERT_TRUE(after.ok()) << after.status().message;

  ASSERT_EQ(
    before.value().trajectory.time_from_start.size(),
    after.value().trajectory.time_from_start.size());
  for (std::size_t index = 0; index < before.value().trajectory.q_a_ref.size(); ++index) {
    EXPECT_DOUBLE_EQ(
      before.value().trajectory.time_from_start[index],
      after.value().trajectory.time_from_start[index]) << index;
    EXPECT_TRUE(
      before.value().trajectory.q_a_ref[index] == after.value().trajectory.q_a_ref[index])
      << index;
    EXPECT_TRUE(
      before.value().trajectory.dq_a_ref[index] == after.value().trajectory.dq_a_ref[index])
      << index;
  }
}

TEST(PlanGrip, ThePayloadIsHonouredInTheEquilibriumAndInTheFlowDemand)
{
  // Two halves, and they land on different phases because the model puts them
  // there. A payload held off the tilt axis moves where the tool hangs
  // (`robot_model` 5.1), which is what a **close** phase reports and no actuated
  // row of its trajectory shows. A payload carried through a **lift** also
  // changes what the move costs the hydraulics, because the OCP of 5.2 times the
  // path against the cylinder force and the pump flow of `mpc` 3 with that mass
  // in the dynamics.
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const crane_model::QA q_a = start_of(context, machine, machine.q8);

  GripRequest empty_close = phase_request(GripPhase::Close, q_a);
  GripRequest carried_close = empty_close;
  carried_close.payload = offset_block();
  auto without_close = crane_planning::plan_grip(model, context, empty_close);
  auto with_close = crane_planning::plan_grip(model, context, carried_close);
  ASSERT_TRUE(without_close.ok()) << without_close.status().message;
  ASSERT_TRUE(with_close.ok()) << with_close.status().message;
  EXPECT_GT(
    (with_close.value().tool.q_u_goal - without_close.value().tool.q_u_goal).norm(), 1.0e-3)
    << "the payload a close phase acquires did not move the equilibrium it ends at";

  crane_model::QA higher = q_a;
  higher[1] += 0.15;
  GripRequest empty_lift = phase_request(GripPhase::Lift, q_a);
  aim_at(model, higher, empty_lift.payload, empty_lift);
  GripRequest carried_lift = empty_lift;
  carried_lift.payload = offset_block();
  auto without_lift = crane_planning::plan_grip(model, context, empty_lift);
  auto with_lift = crane_planning::plan_grip(model, context, carried_lift);
  ASSERT_TRUE(without_lift.ok()) << without_lift.status().message;
  ASSERT_TRUE(with_lift.ok()) << with_lift.status().message;

  EXPECT_GT(
    (with_lift.value().motion.endpoint.q.segment<2>(4) -
    without_lift.value().motion.endpoint.q.segment<2>(4)).norm(), 1.0e-3)
    << "the payload a lift phase carries did not move the equilibrium it ends at";
  EXPECT_NE(
    with_lift.value().motion.timing.peak_demand.pump_flow,
    without_lift.value().motion.timing.peak_demand.pump_flow)
    << "the payload a lift phase carries did not change what the move costs the pump";
}

TEST(PlanGrip, ADescendGoesAcrossAndDownRatherThanUpAcrossAndDown)
{
  // `/crane/plan_motion`'s transfer altitude is the higher endpoint raised by the
  // mounted tool's own reach, which is right for a place-to-place move and wrong
  // for a descend: it would take the tool up by the length of the tool before
  // bringing it back down onto the block it is already above. A grip's arm phase
  // lowers the derivation's ceiling to its own two endpoints -- the one thing
  // `TransferAltitudeSettings::ceiling_m` exists to do -- so the tool never rises
  // above where it started.
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const crane_model::QA q_a = start_of(context, machine, machine.q8);
  crane_model::QA lower = q_a;
  lower[1] -= 0.15;

  GripRequest request = phase_request(GripPhase::Descend, q_a);
  aim_at(model, lower, request.payload, request);
  const double z_start = tcp_of(model, q_a, request.payload).position_m.z();
  auto planned = crane_planning::plan_grip(model, context, request);
  ASSERT_TRUE(planned.ok()) << planned.status().message;

  ASSERT_TRUE(planned.value().altitude_ceiling_applied);
  EXPECT_LE(planned.value().altitude_ceiling_m, z_start + context.settings.ik.eps_pos + 1.0e-12);
  EXPECT_NEAR(planned.value().motion.primitive.altitude.z_m, planned.value().altitude_ceiling_m,
    1.0e-9);
  EXPECT_TRUE(planned.value().motion.primitive.altitude.ceiling_binding);
  // What the derivation would have asked for on its own, and what the ceiling
  // therefore took off: the whole of the mounted tool's reach.
  EXPECT_GT(
    planned.value().motion.primitive.altitude.z_from_endpoints_m -
    planned.value().altitude_ceiling_m, 0.3);

  // ...and measured on the emitted trajectory, which is the claim that matters:
  // the tool goes across and down, never up first. What is left is millimetres,
  // and they are not a lift -- the collapsed first segment joins the measured
  // start to an IK solution for the start's own TCP pose, and a joint-space
  // interpolation between two configurations of equal TCP height bulges a little
  // in between. The bound is therefore stated against the lift the derivation
  // *would* have made rather than as an absolute: a hundredth of it.
  const double removed = planned.value().motion.primitive.altitude.z_from_endpoints_m -
    planned.value().altitude_ceiling_m;
  double rise = 0.0;
  for (const crane_model::QA & q : planned.value().trajectory.q_a_ref) {
    rise = std::max(rise, tcp_of(model, q, request.payload).position_m.z() - z_start);
  }
  EXPECT_LE(rise, 0.01 * removed) << "the tool rose " << rise << " m before descending, against "
                                  << removed << " m the derivation alone would have taken it up";
}

TEST(PlanGrip, SpeedScaleAppliesAsItDoesOnPlanMotionAndCannotReachKappa)
{
  const Machine & machine = crane_planning_test::machines().front();
  const crane_model::Model model = crane_planning_test::build_model(machine);
  const crane_planning::PlannerContext context =
    crane_planning_test::build_context(model, machine);
  const crane_model::QA q_a = start_of(context, machine, machine.q8);
  crane_model::QA lower = q_a;
  lower[1] -= 0.15;

  GripRequest request = phase_request(GripPhase::Descend, q_a);
  aim_at(model, lower, request.payload, request);
  auto full = crane_planning::plan_grip(model, context, request);
  ASSERT_TRUE(full.ok()) << full.status().message;

  request.speed_scale = 0.4;
  auto slowed = crane_planning::plan_grip(model, context, request);
  ASSERT_TRUE(slowed.ok()) << slowed.status().message;
  EXPECT_EQ(slowed.value().motion.timing.speed_scale, 0.4);

  // What `speed_scale` *is*, on `/crane/plan_motion` and here alike: it scales
  // the velocity bound and nothing else, so the peak velocity demand is held
  // under `kappa speed_scale` of the physical limit. It is not a duration
  // multiplier, and this descend is where the difference shows -- its timing is
  // chosen by the cylinder force and the pump flow of `mpc` 3, so the velocity
  // bound has nothing to bind and lowering it changes nothing. Both branches are
  // asserted, because which one holds is a property of the pose and not of the
  // contract.
  EXPECT_LE(
    slowed.value().motion.timing.peak_demand.joint_velocity,
    context.settings.timing.kappa * 0.4 + 1.0e-6);
  EXPECT_LE(
    full.value().motion.timing.peak_demand.joint_velocity,
    context.settings.timing.kappa + 1.0e-6);
  if (full.value().motion.timing.peak_demand.joint_velocity >
    context.settings.timing.kappa * 0.4 + 1.0e-6)
  {
    EXPECT_GT(slowed.value().trajectory.duration, full.value().trajectory.duration);
  } else {
    EXPECT_NEAR(
      slowed.value().trajectory.duration, full.value().trajectory.duration,
      0.05 * full.value().trajectory.duration);
  }

  // kappa is untouched by it, and every peak stays under kappa.
  EXPECT_EQ(slowed.value().motion.timing.kappa, context.settings.timing.kappa);
  EXPECT_LE(
    slowed.value().motion.timing.peak_demand.worst(), context.settings.timing.kappa + 1.0e-6);
  EXPECT_LE(
    full.value().motion.timing.peak_demand.worst(), context.settings.timing.kappa + 1.0e-6);

  for (const double speed_scale : {0.0, 1.5}) {
    request.speed_scale = speed_scale;
    auto refused = crane_planning::plan_grip(model, context, request);
    EXPECT_FALSE(refused.ok()) << speed_scale;
    EXPECT_NE(refused.status().message.find("speed_scale"), std::string::npos)
      << refused.status().message;
  }
}

