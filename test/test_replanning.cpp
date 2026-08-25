// Replanning from a moving, swinging state, and the latency bound of
// `wiki/trajectory_planning.md` 7 -- offline.
//
// Every case here builds its own path and solves in process. Nothing launches,
// nothing links a simulator and no ROS graph is reachable from this binary; the
// five cases the acceptance criteria name are:
//
//   * a re-plan from rest                `AReplanFromRestStillMeetsItsStartAtRest`
//   * a re-plan from a moving arm        `ContinuityIsABoundaryConditionAndNotAnAccident`
//   * a re-plan from a swinging tool     `TwoRequestsDifferingOnlyInTheSwayRateAnswerDifferently`
//   * a re-plan with no passive estimate `AnUnmeasuredSwayIsBoundedOrRefusedAndNeverAssumedZero`
//   * a budget overrun                   `TheBudgetFiresInThePlannerAndNamesTheStageThatOverran`
//
// **No wall-clock sleep is the mechanism under test anywhere.** The latency
// budget reads a `PlanningClock`, which is a plain `double()`; the cases below
// hand it a counter that advances a fixed amount per reading, so "the budget
// fires on the stage that overran" is asserted deterministically rather than
// raced against a real clock.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/planner_core.hpp"
#include "crane_planning/replanning.hpp"

#include "description_fixture.hpp"

namespace
{

using crane_planning_test::Machine;
using crane_planning_test::empty_gripper;
using crane_planning_test::machines;

/// The tolerance the seam is asserted to, rad/s or m/s.
/**
 * Stated rather than guessed, and it is the planner's own `at_rest_dq_a`: a
 * velocity mismatch below the threshold at which this planner calls the machine
 * stopped is not a discontinuity anybody could act on. The solver's feasibility
 * tolerance on the boxed initial state is `1e-6`, two decades under it.
 */
constexpr double kSeamTolerance = 1.0e-4;

/// One machine, its context, and a start and a goal it can actually hold.
/**
 * Built once and shared, because building it costs a URDF parse, a geometry
 * probe and a cylinder-force derivation, and none of those is what is under test.
 * Only the PZS100 is exercised: every case below solves the OCP of 5.2 at some
 * seconds a plan, the second machine differs in the tool's reach rather than in
 * anything this issue changed, and `test_plan_grip.cpp` and `test_timing_ocp.cpp`
 * already run both.
 */
struct Fixture
{
  crane_model::Model model;
  crane_planning::PlannerContext context;
  crane_model::QA start;
  Eigen::Vector3d goal_p;
};

const Fixture & fixture()
{
  static const Fixture built = [] {
      const Machine & machine = machines().front();
      crane_model::Model model = crane_planning_test::build_model(machine);
      crane_planning::PlannerContext context =
        crane_planning_test::build_context(model, machine);
      const crane_model::QA start =
        crane_planning_test::working_centred(context.limits, machine);
      const crane_model::Q goal = crane_planning_test::settled(
        model, crane_planning_test::moved(start, context.limits));
      auto pose = model.forward_kinematics(
        goal, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
      if (!pose.ok()) {
        throw std::runtime_error(pose.status().message);
      }
      return Fixture{std::move(model), std::move(context), start, pose.value().position_m};
    }();
  return built;
}

/// Where the tool hangs at one actuated configuration.
crane_model::QU hanging(const crane_model::QA & q_a)
{
  auto settled = fixture().model.passive_equilibrium(q_a, empty_gripper());
  EXPECT_TRUE(settled.ok()) << settled.status().message;
  return settled.ok() ? settled.value() : crane_model::QU::Zero();
}

/// A request that plans the fixture's move from a stated start state.
crane_planning::MotionRequest request_from(const crane_planning::MeasuredStart & start)
{
  crane_planning::MotionRequest request;
  request.p_tcp_0 = fixture().goal_p;
  request.start = start;
  request.payload = empty_gripper();
  // Collision-blind on purpose: every case here is about the *start state* and
  // the *budget*, and a scene would only add a check none of them is testing.
  request.avoid_collisions = false;
  return request;
}

/// The machine standing still, with the passive pair measured where it hangs.
crane_planning::MeasuredStart at_rest(const crane_model::QA & q_a)
{
  crane_planning::MeasuredStart start;
  start.q_a = q_a;
  start.dq_a = crane_model::DQA::Zero();
  start.passive.measured = true;
  start.passive.q_u = hanging(q_a);
  start.passive.dq_u = crane_model::DQU::Zero();
  start.passive.note = "the fixture measured it";
  return start;
}

/// The fixture's move, planned from rest. Solved once and shared.
const crane_planning::MotionPlan & from_rest()
{
  static const crane_planning::MotionPlan plan = [] {
      auto solved = crane_planning::plan_motion(
        fixture().model, fixture().context, request_from(at_rest(fixture().start)));
      if (!solved.ok()) {
        throw std::runtime_error(solved.status().message);
      }
      return std::move(solved).value();
    }();
  return plan;
}

/// A clock that advances a fixed amount per reading. No sleeping, no race.
crane_planning::PlanningClock stepping_clock(double step_s)
{
  auto reading = std::make_shared<double>(0.0);
  return [reading, step_s]() {
             const double now = *reading;
             *reading += step_s;
             return now;
           };
}

/// The fixture's settings with a latency budget on that clock.
crane_planning::PlannerContext budgeted(double total_s, double step_s)
{
  crane_planning::PlannerContext context = fixture().context;
  context.settings.latency.total_s = total_s;
  context.settings.latency.clock = stepping_clock(step_s);
  return context;
}

}  // namespace

// The cases below sit outside the anonymous namespace above, as every other test
// binary in this package does: cppcheck 2.7 cannot parse a `TEST(...)` macro that
// follows a declaration inside an anonymous namespace and reports a syntaxError
// on the first one, which `pre-commit run -a` fails CI on.

TEST(Replanning, AReplanFromRestStillMeetsItsStartAtRest)
{
  // The first of the five offline cases, and the one that says the change is a
  // *widening*: a machine that really is standing still still gets the plan this
  // package built before, with `q_a'(0) = 0` and both ends at rest.
  const crane_planning::MotionPlan & plan = from_rest();
  ASSERT_GE(plan.trajectory.q_a_ref.size(), 2U);
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(plan.trajectory.q_a_ref.front()[axis], fixture().start[axis], 1.0e-12) << row;
    EXPECT_NEAR(plan.trajectory.dq_a_ref.front()[axis], 0.0, kSeamTolerance) << row;
    EXPECT_NEAR(plan.trajectory.dq_a_ref.back()[axis], 0.0, kSeamTolerance) << row;
  }
  // And the initial condition says which branch it took, rather than leaving a
  // reader to infer it from a number that happens to be zero.
  EXPECT_TRUE(plan.start.measured);
  EXPECT_FALSE(plan.start.sigma_rate_pinned);
  EXPECT_NE(plan.start_note.find("standing still"), std::string::npos) << plan.start_note;

  // Every stage of the plan was charged, which is what makes the budget a bound
  // over the *whole* of a request rather than over the solve alone.
  ASSERT_FALSE(plan.stages.empty());
  const std::vector<std::string> expected{
    "the endpoint IK of robot_model 2.2",
    "the structured primitive of trajectory_planning 4.4",
    "the path-constrained OCP of trajectory_planning 5.2",
    "the visualisation path"};
  std::vector<std::string> charged;
  charged.reserve(plan.stages.size());
  for (const crane_planning::StageTiming & stage : plan.stages) {
    charged.push_back(stage.stage);
  }
  EXPECT_EQ(charged, expected);
}

TEST(Replanning, ContinuityIsABoundaryConditionAndNotAnAccident)
{
  // trajectory_planning 7's first consequence, asserted both ways round. A
  // trajectory that continues a previous one has to match it in position **and**
  // in velocity at the seam; the stopped-start convention 7 names matches the
  // position and drops the velocity to zero, and this is where that fails.
  const crane_planning::MotionPlan & first = from_rest();

  // The seam: a point early in the lift where the arm is genuinely moving, taken
  // off the first plan rather than invented. Early, because a re-plan issued
  // mid-lift rebuilds a lift from where the machine is and the two agree in
  // direction there -- which is the case a stall-recovery re-plan is.
  double peak = 0.0;
  for (const crane_model::DQA & rate : first.trajectory.dq_a_ref) {
    peak = std::max(peak, rate.cwiseAbs().maxCoeff());
  }
  ASSERT_GT(peak, 10.0 * kSeamTolerance) << "the first plan never moves, so there is no seam";
  std::size_t seam = 0;
  while (seam + 1U < first.trajectory.dq_a_ref.size() &&
    first.trajectory.dq_a_ref[seam].cwiseAbs().maxCoeff() < 0.2 * peak)
  {
    ++seam;
  }
  const crane_model::QA q_seam = first.trajectory.q_a_ref[seam];
  const crane_model::DQA dq_seam = first.trajectory.dq_a_ref[seam];
  ASSERT_GT(dq_seam.cwiseAbs().maxCoeff(), 10.0 * kSeamTolerance);

  crane_planning::MeasuredStart moving = at_rest(q_seam);
  moving.dq_a = dq_seam;

  auto continued = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(moving));
  ASSERT_TRUE(continued.ok()) << continued.status().message;
  const crane_planning::MotionPlan & second = continued.value();

  // Position **and** velocity, to the stated tolerance.
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(second.trajectory.q_a_ref.front()[axis], q_seam[axis], 1.0e-12) << row;
    EXPECT_NEAR(second.trajectory.dq_a_ref.front()[axis], dq_seam[axis], kSeamTolerance) << row;
  }
  EXPECT_TRUE(second.start.sigma_rate_pinned);
  EXPECT_GT(second.start.sigma_rate, 0.0);

  // ...and the position-only case fails it. This is the same request with the
  // measured velocity zeroed -- *"we deactivated qDot0, because we now always
  // start in a stopped state"* -- and it is a plan for a machine that is not
  // where this one is. It matches the position and misses the velocity by the
  // whole of what the machine is doing.
  auto stopped_start = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(at_rest(q_seam)));
  ASSERT_TRUE(stopped_start.ok()) << stopped_start.status().message;
  const crane_planning::MotionPlan & third = stopped_start.value();
  double worst = 0.0;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    EXPECT_NEAR(third.trajectory.q_a_ref.front()[axis], q_seam[axis], 1.0e-12) << row;
    worst = std::max(worst, std::abs(third.trajectory.dq_a_ref.front()[axis] - dq_seam[axis]));
  }
  EXPECT_GT(worst, kSeamTolerance)
    << "the stopped-start plan matched the seam's velocity, so this test asserts nothing";
  EXPECT_FALSE(third.start.sigma_rate_pinned);
}

TEST(Replanning, TwoRequestsDifferingOnlyInTheSwayRateAnswerDifferently)
{
  // The criterion's own test, and the one the zeroed-`qDot0` behaviour of 7
  // cannot pass: two requests that differ **only** in `dq_u` must come back with
  // different trajectories, because the sway is a state the OCP carries and a
  // tool that is already swinging is a different problem from one that is not.
  const crane_planning::MotionPlan & still = from_rest();

  crane_planning::MeasuredStart swinging = at_rest(fixture().start);
  // Well above the `2^-9` rad/s gyro quantiser and well inside mpc 3
  // constraint 4's `dq_u^+`, so this is sway and not noise and the box it must
  // be steered back inside is not the thing under test.
  swinging.passive.dq_u = crane_model::DQU(0.15, 0.0);

  auto solved = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(swinging));
  ASSERT_TRUE(solved.ok()) << solved.status().message;
  const crane_planning::MotionPlan & swung = solved.value();

  EXPECT_TRUE(swung.start.measured);
  EXPECT_NEAR(swung.start.dq_u[0], 0.15, 1.0e-12);
  EXPECT_NE(swung.start_note.find("swinging"), std::string::npos) << swung.start_note;

  // Different, and the OCP's own nodes are where it shows: 5.4 still pins the
  // *terminal* sway at zero, so what the swinging start bought is a different
  // path rate profile getting there.
  ASSERT_FALSE(swung.timing.nodes.empty());
  ASSERT_FALSE(still.timing.nodes.empty());
  EXPECT_NEAR(swung.timing.nodes.front().dq_u[0], 0.15, 1.0e-4);
  EXPECT_NEAR(still.timing.nodes.front().dq_u[0], 0.0, 1.0e-4);

  double difference = 0.0;
  const std::size_t common =
    std::min(swung.timing.nodes.size(), still.timing.nodes.size());
  for (std::size_t node = 0; node < common; ++node) {
    difference = std::max(
      difference, std::abs(swung.timing.nodes[node].sigma_rate -
      still.timing.nodes[node].sigma_rate));
  }
  EXPECT_GT(difference, 1.0e-6)
    << "the two solves came back with the same path rate profile, which is the zeroed-qDot0 "
    "behaviour trajectory_planning 7 warns against";

  // 5.4 is untouched by any of this: the tool still arrives hanging still.
  EXPECT_NEAR(swung.timing.nodes.back().dq_u[0], 0.0, 1.0e-4);
  EXPECT_NEAR(swung.timing.nodes.back().dq_u[1], 0.0, 1.0e-4);
}

TEST(Replanning, AnUnmeasuredSwayIsBoundedOrRefusedAndNeverAssumedZero)
{
  // control_architecture 5.3: no input may stop arriving without a defined
  // consequence. There are exactly two defined consequences here and this asserts
  // both -- the conservative bound, and the refusal for the case where no bound
  // can be honest.
  crane_planning::MeasuredStart unmeasured;
  unmeasured.q_a = fixture().start;
  unmeasured.dq_a = crane_model::DQA::Zero();
  unmeasured.passive.measured = false;
  unmeasured.passive.note = "nothing has ever been received on /crane/pendulum_state";
  unmeasured.passive.sway_reserve_rad = 0.05;
  unmeasured.passive.sway_rate_reserve = 0.10;

  auto bounded = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(unmeasured));
  ASSERT_TRUE(bounded.ok()) << bounded.status().message;
  EXPECT_FALSE(bounded.value().start.measured);
  // The answer says the passive pair was not measured, and carries the producer's
  // absence rather than restating it.
  EXPECT_NE(bounded.value().start_note.find("**not** measured"), std::string::npos)
    << bounded.value().start_note;
  EXPECT_NE(bounded.value().start_note.find("/crane/pendulum_state"), std::string::npos)
    << bounded.value().start_note;
  // The reservation is real and not decorative: every node stays inside the
  // *reduced* box, so sway up to the reserve can be present without leaving what
  // mpc 3 constraint 4 permits.
  const double reduced =
    fixture().context.settings.timing.dq_u_max[0] - unmeasured.passive.sway_rate_reserve;
  for (const crane_planning::OcpNode & node : bounded.value().timing.nodes) {
    EXPECT_LE(std::abs(node.dq_u[0]), reduced + 1.0e-6);
  }

  // ...and the one combination that has no honest answer: a moving arm whose
  // sway nobody measured. 7's `[!warning]` is exactly this case, so it is refused
  // rather than planned from an assumed still tool.
  crane_planning::MeasuredStart blind = unmeasured;
  blind.dq_a[0] = 0.05;
  auto refused = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(blind));
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("moving arm"), std::string::npos)
    << refused.status().message;
  EXPECT_NE(refused.status().message.find("/crane/pendulum_state"), std::string::npos)
    << refused.status().message;
}

TEST(Replanning, ASwayLargerThanTheBoxTheHorizonCarriesIsRefusedByName)
{
  // A measured start outside mpc 3's boxes is not a harder problem, it is an
  // infeasible one -- and a refusal that names the row and the bound is what a
  // caller can act on, where `ACADOS_QP_FAILURE` is not.
  crane_planning::MeasuredStart wild = at_rest(fixture().start);
  wild.passive.q_u[0] += 10.0 * fixture().context.settings.timing.q_u_max[0];
  auto refused = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(wild));
  ASSERT_FALSE(refused.ok());
  EXPECT_NE(refused.status().message.find("sway box"), std::string::npos)
    << refused.status().message;

  crane_planning::MeasuredStart fast = at_rest(fixture().start);
  fast.passive.dq_u[1] = 10.0 * fixture().context.settings.timing.dq_u_max[1];
  auto second = crane_planning::plan_motion(
    fixture().model, fixture().context, request_from(fast));
  ASSERT_FALSE(second.ok());
  EXPECT_NE(second.status().message.find("constraint 4"), std::string::npos)
    << second.status().message;
}

TEST(Replanning, TheBudgetFiresInThePlannerAndNamesTheStageThatOverran)
{
  // 7's second consequence: the bound is enforced **in the planner** and not in
  // the caller's timeout. The clock advances five seconds per reading, so the
  // stage that overruns is decided by arithmetic and not by how fast this machine
  // happens to be -- which is what "deterministic, no wall-clock sleeps as the
  // mechanism under test" asks for.
  //
  // Three seconds of budget against a five-second first stage: the endpoint IK
  // is what overruns, and nothing after it runs at all.
  crane_planning::LatencyLedger ledger(
    crane_planning::LatencyBudgetSettings{3.0, stepping_clock(5.0)});
  auto refused = crane_planning::plan_motion(
    fixture().model, budgeted(3.0, 5.0), request_from(at_rest(fixture().start)), &ledger);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(ledger.overrun(), "the endpoint IK of robot_model 2.2");
  EXPECT_NE(refused.status().message.find("endpoint IK"), std::string::npos)
    << refused.status().message;
  // Which stage, **and how long it had**: a refusal that says only how long a
  // stage took leaves a reader unable to tell a slow stage from a small budget.
  EXPECT_NE(refused.status().message.find("3.000000 s"), std::string::npos)
    << refused.status().message;
  EXPECT_NE(refused.status().message.find("caller's timeout"), std::string::npos)
    << refused.status().message;
  // 7's fallback: no partial plan, at all. A `Result` failure carries no value,
  // so this is structural rather than a promise -- and the node is what keeps the
  // previous trajectory standing.
  ASSERT_EQ(ledger.stages().size(), 1U);

  // The same budget one stage later. Eight seconds buys the five-second IK and
  // then runs out inside the geometry, which is the stage that is named.
  crane_planning::LatencyLedger second(
    crane_planning::LatencyBudgetSettings{8.0, stepping_clock(5.0)});
  auto later = crane_planning::plan_motion(
    fixture().model, budgeted(8.0, 5.0), request_from(at_rest(fixture().start)), &second);
  ASSERT_FALSE(later.ok());
  EXPECT_EQ(second.overrun(), "the structured primitive of trajectory_planning 4.4");
  EXPECT_NE(later.status().message.find("structured primitive"), std::string::npos)
    << later.status().message;
  ASSERT_EQ(second.stages().size(), 2U);
  EXPECT_NEAR(second.stages().front().had_s, 8.0, 1.0e-12);
  EXPECT_NEAR(second.stages().back().had_s, 3.0, 1.0e-12);
}

TEST(Replanning, ALedgerWithNoBudgetChargesEveryStageAndRefusesNothing)
{
  // A deployment that has not chosen a latency bound says so by not giving one,
  // and the stages are still charged and reported -- the report is what makes a
  // bound choosable later from measurements rather than from a guess.
  crane_planning::LatencyLedger ledger;
  EXPECT_FALSE(ledger.bounded());
  EXPECT_TRUE(std::isinf(ledger.remaining()));
  EXPECT_TRUE(ledger.charge("one").ok());
  EXPECT_TRUE(ledger.charge("two").ok());
  EXPECT_EQ(ledger.stages().size(), 2U);
  EXPECT_TRUE(ledger.overrun().empty());
  EXPECT_NE(crane_planning::describe(ledger).find("No latency budget"), std::string::npos);
}
