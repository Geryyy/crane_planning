// Stage 2 of `wiki/trajectory_planning.md` 5.2, offline.
//
// Every case here builds its own path and solves the OCP in process. Nothing
// launches, nothing links a simulator and no ROS graph is reachable from this
// binary -- the planner claims no interface and joins no cascade, so the timing
// is exactly as testable as the geometry was.
//
// The four assertions the acceptance criteria name, and where they live:
//
//   * the trajectory ends with the tool hanging still     `EndsWithTheToolStill`
//   * the flow expression is the graph's, not a copy      `FlowAgreesWithTheGraph`
//   * the peak demand sits at kappa and not at one        `KappaLeavesMargin`
//   * a binding force limit costs time                    `ForceLimitCostsTime`

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
// `FlowAgreesWithTheGraph` builds the graph a second time and drives it itself,
// which is the only way a test outside the library can settle that the planner's
// flow row *is* this one. It is why this binary links `crane_model::casadi_graph`
// while nothing else in the package's tests does.
#include "crane_model/symbolic/casadi_graph.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/planner_core.hpp"
#include "crane_planning/timing_ocp.hpp"

#include "description_fixture.hpp"

namespace
{

using crane_planning_test::Machine;
using crane_planning_test::build_model;

using crane_planning_test::description;
using crane_planning_test::empty_gripper;
using crane_planning_test::machines;


/// A relief pressure to run the tests at. Not a measurement -- see the config.
constexpr double kSystemPressure = 2.5e7;  // Pa

struct Fixture
{
  crane_model::Model model;
  crane_model::ModelConfig config;
  crane_planning::JointLimits limits;
  crane_planning::GeometricPath path;
  crane_planning::TimingOcpSettings settings;
};

/// A configuration inside the arm's **working** range, not the middle of the URDF's.
/**
 * `centred()` puts `theta3_arm_joint` at 1.85 rad, and the arm cylinder's
 * transmission Jacobian passes through zero at 1.85: the static cylinder force
 * there is 5e7 N, four hundred times any relief pressure can deliver, so the
 * force constraint of `mpc.md` 3 constraint 6 is infeasible before the solver
 * starts. `trajectory_planning` 4.2 says as much in passing -- "the arm joint's
 * range is far wider than the working range" -- and this is what that costs a
 * test that constrains force. Every case here therefore poses the machine where
 * the machine actually works.
 */
crane_model::QA working_pose(const Machine & machine, double q3, double q4)
{
  crane_model::QA q_a = crane_model::QA::Zero();
  q_a[0] = 0.0;    // slewing
  q_a[1] = -0.2;   // boom
  q_a[2] = q3;     // arm
  q_a[3] = q4;     // telescope
  q_a[4] = 0.0;    // rotator
  q_a[5] = machine.q8;
  return q_a;
}

crane_planning::PathVector path_vector(const crane_model::QA & q_a)
{
  crane_planning::PathVector value = crane_planning::PathVector::Zero();
  for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
    value[static_cast<Eigen::Index>(row)] = q_a[static_cast<Eigen::Index>(row)];
  }
  return value;
}

/// One machine, one straight two-waypoint path, and settings that can be solved.
Fixture make_fixture(const Machine & machine)
{
  Fixture fixture{build_model(machine), {}, {}, {}, {}};
  fixture.config.robot_description_xml = description(machine);
  fixture.config.tool = machine.tool;

  auto limits = crane_planning::read_joint_limits(
    description(machine), fixture.model.urdf_joint_names());
  EXPECT_TRUE(limits.ok()) << limits.status().message;
  fixture.limits = std::move(limits).value();

  const crane_model::QA start = working_pose(machine, 0.4, 1.0);
  const crane_model::QA goal = working_pose(machine, 0.9, 1.4);

  crane_planning::PathFitRequest request;
  request.waypoints = {path_vector(start), path_vector(goal)};
  request.segment_names = {"traverse"};
  request.q8_start = machine.q8;
  request.q8_goal = machine.q8;
  auto path = crane_planning::fit_c2_path(
    request, fixture.limits, crane_planning::PathFitSettings{});
  EXPECT_TRUE(path.ok()) << path.status().message;
  fixture.path = std::move(path).value();

  auto forces = crane_planning::derive_cylinder_force_limits(fixture.model, kSystemPressure);
  EXPECT_TRUE(forces.ok()) << forces.status().message;
  fixture.settings.actuation.cylinder_force_max = forces.value();
  return fixture;
}

/// Solve `make_fixture`'s straight move, or fail the calling test saying why.
crane_planning::TimingSolution solved(const Fixture & fixture)
{
  crane_planning::TimingOcpRequest request;
  request.payload = empty_gripper();
  auto solution = crane_planning::solve_timing_ocp(
    fixture.model, fixture.config, fixture.path, fixture.limits, request, fixture.settings);
  EXPECT_TRUE(solution.ok()) << solution.status().message;
  if (!solution.ok()) {
    return crane_planning::TimingSolution{};
  }
  return std::move(solution).value();
}

}  // namespace

// The cases below sit **outside** the anonymous namespace above, as every other
// test binary in this package does. That is not a style choice: cppcheck 2.7
// cannot parse a `TEST(...)` macro that follows a declaration inside an
// anonymous namespace, and reports it as a `syntaxError` on the first macro --
// a parser failure rather than a defect, but one `pre-commit run -a` fails CI
// on. Closing the namespace first is what the rest of the suite already does.

TEST(TimingOcp, SolvesTheStraightMove)
{
  for (const Machine & machine : machines()) {
    SCOPED_TRACE(machine.name);
    Fixture fixture = make_fixture(machine);
    crane_planning::TimingOcpRequest request;
    request.payload = empty_gripper();

    auto solution = crane_planning::solve_timing_ocp(
      fixture.model, fixture.config, fixture.path, fixture.limits, request, fixture.settings);
    ASSERT_TRUE(solution.ok()) << solution.status().message;
    EXPECT_GT(solution.value().trajectory.duration, 0.0);
    EXPECT_EQ(solution.value().solver_status, "ACADOS_SUCCESS");

    // The reference is what the acceptance criterion asks for: positions and
    // velocities on the six actuated joints, on one time base, at rest at both
    // ends because 5.4 pins `sigma_dot` there.
    const crane_planning::TimedTrajectory & timed = solution.value().trajectory;
    ASSERT_GT(timed.q_a_ref.size(), 2U);
    EXPECT_EQ(timed.q_a_ref.size(), timed.dq_a_ref.size());
    EXPECT_EQ(timed.q_a_ref.size(), timed.time_from_start.size());
    EXPECT_NEAR(timed.time_from_start.front(), 0.0, 1.0e-12);
    EXPECT_NEAR(timed.dq_a_ref.front().norm(), 0.0, 1.0e-6);
    EXPECT_NEAR(timed.dq_a_ref.back().norm(), 0.0, 1.0e-6);
    for (std::size_t index = 1; index < timed.time_from_start.size(); ++index) {
      EXPECT_GT(timed.time_from_start[index], timed.time_from_start[index - 1U]);
    }
  }
}

/// 5.4's terminal condition, which is the whole reason `q_u` is a state.
/**
 * Not "the reference ends where it was asked to" -- that is stage 1's business
 * and a ramp does it too. This is the thing no classical retiming can assert:
 * the *tool* is hanging still at the end, `dq_u(T) = 0` and `q_u(T) = q_u^eq`,
 * because those were imposed on a hidden state with memory rather than read off
 * the path.
 */
TEST(TimingOcp, EndsWithTheToolStill)
{
  for (const Machine & machine : machines()) {
    SCOPED_TRACE(machine.name);
    const Fixture fixture = make_fixture(machine);
    const crane_planning::TimingSolution solution = solved(fixture);
    ASSERT_FALSE(solution.nodes.empty());

    const crane_planning::OcpNode & last = solution.nodes.back();
    EXPECT_NEAR(last.sigma, 1.0, 1.0e-9);
    // Deliberately **not** `sigma_rate == 0`. 5.4 asks for `sigma_dot(T) = 0` and
    // this formulation cannot pin it, because the objective divides by it; it
    // does not need to, because the path meets its end with `q_a'(1) = 0` and the
    // joints are therefore at rest whatever the path rate reads. The assertion
    // below is on the machine, which is what 5.4 is about.
    EXPECT_NEAR(solution.trajectory.dq_a_ref.back().norm(), 0.0, 1.0e-6);
    // Hanging: at the equilibrium of the goal pose, not merely at some q_u.
    EXPECT_NEAR((last.q_u - last.q_u_equilibrium).norm(), 0.0, 1.0e-6)
      << "q_u(T) = " << last.q_u.transpose() << ", q_eq = " << last.q_u_equilibrium.transpose();
    // Still: and this is the one a ramp cannot deliver.
    EXPECT_NEAR(last.dq_u.norm(), 0.0, 1.0e-6) << "dq_u(T) = " << last.dq_u.transpose();

    // The sway did not simply stay at zero throughout, which would make the
    // assertion above vacuous -- the move excites the tool and the OCP damps it.
    double excursion = 0.0;
    for (const crane_planning::OcpNode & node : solution.nodes) {
      excursion = std::max(excursion, node.dq_u.norm());
    }
    EXPECT_GT(excursion, 1.0e-4) << "the path never moved the tool, so arriving still is no claim";
  }
}

/// 5.3 by construction: the planner's flow row *is* the graph's output map.
/**
 * The acceptance criterion asks for agreement "at sampled points rather than
 * merely resembling each other", and this is the only way to settle it from
 * outside the library: rebuild the graph the way `solve_timing_ocp` does, drive
 * it with the state and acceleration the solve reported at each node, sum
 * `mpc.md` 3 constraint 7's per-axis rows, and compare against the flow the
 * planner recorded there.
 *
 * The tolerance is `1e-15` relative, i.e. floating-point equality. Anything a
 * *second implementation* would produce -- a differently ordered sum, an
 * unsmoothed `sign(v)`, a different `eps` -- misses that by decades. Passing it
 * means one expression, evaluated twice.
 */
TEST(TimingOcp, FlowAgreesWithTheGraph)
{
  for (const Machine & machine : machines()) {
    SCOPED_TRACE(machine.name);
    const Fixture fixture = make_fixture(machine);
    const crane_planning::TimingSolution solution = solved(fixture);
    ASSERT_FALSE(solution.nodes.empty());

    crane_model::SymbolicGraphSpec spec;
    spec.sample_time_s = fixture.settings.sample_period;
    spec.include_output_map = true;
    auto graph = crane_model::symbolic::casadi_graph(fixture.config, spec, empty_gripper());
    ASSERT_TRUE(graph.ok()) << graph.status().message;

    double worst_flow = 0.0;
    double worst_force = 0.0;
    double largest = 0.0;
    for (const crane_planning::OcpNode & node : solution.nodes) {
      std::vector<double> state;
      state.reserve(16U);
      for (int row = 0; row < 6; ++row) {state.push_back(node.q_a[row]);}
      state.push_back(node.q_u[0]);
      state.push_back(node.q_u[1]);
      for (int row = 0; row < 6; ++row) {state.push_back(node.dq_a[row]);}
      state.push_back(node.dq_u[0]);
      state.push_back(node.dq_u[1]);
      std::vector<double> acceleration;
      acceleration.reserve(6U);
      for (int row = 0; row < 6; ++row) {acceleration.push_back(node.ddq_a[row]);}

      const std::vector<casadi::DM> outputs = graph.value()->z(
        std::vector<casadi::DM>{casadi::DM(state), casadi::DM(acceleration)});
      const std::vector<double> z = outputs[0].nonzeros();

      double flow = 0.0;
      for (int row = 0; row < 6; ++row) {
        flow += z[crane_model::symbolic::kAxisFlowOffset + static_cast<std::size_t>(row)];
        const double force =
          z[crane_model::symbolic::kCylinderForceOffset + static_cast<std::size_t>(row)];
        worst_force = std::max(worst_force, std::abs(force - node.cylinder_force[row]));
        largest = std::max(largest, std::abs(force));
      }
      worst_flow = std::max(worst_flow, std::abs(flow - node.pump_flow));
    }
    // Absolute, against the scale each quantity actually has here.
    EXPECT_LT(worst_flow, 1.0e-15 * fixture.settings.actuation.pump_flow_max)
      << "the planner's pump flow is not the graph's";
    EXPECT_LT(worst_force, 1.0e-12 * std::max(1.0, largest))
      << "the planner's cylinder force is not the graph's";
  }
}

/// 5.5: the answer sits at kappa of the machine, and the rest is the MPC's.
/**
 * `PeakDemand` is reported against the **physical** limit, so this reads
 * directly. The pair of solves is the point: at `kappa = 1` something has to
 * reach the limit or the duration was never decided by the limits at all, and at
 * the default the same move must be *slower* and must leave the margin unspent.
 */
TEST(TimingOcp, KappaLeavesMargin)
{
  const Machine & machine = machines().front();
  Fixture reserved = make_fixture(machine);
  Fixture saturated = make_fixture(machine);
  saturated.settings.kappa = 1.0;

  ASSERT_LT(reserved.settings.kappa, 1.0) << "the default kappa is the acceptance criterion";

  const crane_planning::TimingSolution with_margin = solved(reserved);
  const crane_planning::TimingSolution at_the_limit = solved(saturated);
  ASSERT_FALSE(with_margin.nodes.empty());
  ASSERT_FALSE(at_the_limit.nodes.empty());

  // At kappa = 1 the machine is worked to its limit: some constrained quantity
  // is at it. Otherwise the comparison below says nothing about kappa.
  EXPECT_GT(at_the_limit.peak_demand.worst(), 0.9)
    << "nothing bound at kappa = 1, so this move is not limit-decided";
  EXPECT_LE(at_the_limit.peak_demand.worst(), 1.0 + 1.0e-6);

  // And at the default it is not: the peak demand sits at kappa, not at one.
  EXPECT_LE(with_margin.peak_demand.worst(), reserved.settings.kappa + 1.0e-6)
    << "the reserved authority was spent: worst peak " << with_margin.peak_demand.worst()
    << " against kappa " << reserved.settings.kappa;

  // Which costs time, and that is what the margin is bought with.
  EXPECT_GT(with_margin.trajectory.duration, at_the_limit.trajectory.duration)
    << "reserving authority did not slow the move down, so it reserved nothing";

  // kappa is not speed_scale. Asking for full speed does not reach into the
  // margin: the same solve with speed_scale = 1 still sits at or below kappa.
  EXPECT_EQ(with_margin.kappa, reserved.settings.kappa);
  EXPECT_EQ(with_margin.speed_scale, 1.0);
}

/// 3's new constraint: the force limit binds, and a binding one costs time.
/**
 * `wiki/trajectory_planning.md` 3 records that no existing planner enforces the
 * cylinder force limit, so the thing to demonstrate is that enforcing it changes
 * the answer. Two solves over the same path, differing only in `F_i^max`: the
 * tighter one must take longer, and the force must be what decided it.
 */
TEST(TimingOcp, ForceLimitCostsTime)
{
  const Machine & machine = machines().front();
  const Fixture generous = make_fixture(machine);
  const crane_planning::TimingSolution loose = solved(generous);
  ASSERT_FALSE(loose.nodes.empty());

  // How far the limit can be squeezed is not a number to guess. A cylinder holds
  // the arm up at rest, so the force has a **static floor** that no timing can go
  // under: squeeze past it and the problem is genuinely infeasible, which is a
  // refusal and not a slower move. So walk the limit down from the loose solve's
  // own peak and take the first tightening that still solves.
  //
  // The walk is coarse where nothing can happen and fine where everything does.
  // On this fixture the loose solve peaks at some 0.64 of the physical limit and
  // the static floor sits at about 0.71x, so the whole window in which the force
  // both binds and is satisfiable is **0.80x to 0.71x** -- a tightening of 0.8x
  // leaves the peak at 99.5% of its allowance, close enough that which of the two
  // durations is larger is decided in the fifth decimal. A grid that steps
  // straight from 0.8 to 0.7 falls through that window on one side or the other.
  crane_planning::TimingSolution tight;
  double applied = 0.0;
  // What each tightening did, so that a failure here says which of the two ways
  // it can fail happened -- squeezed past the static floor and refused, or
  // solved and did not cost anything -- rather than only that neither worked.
  std::string walked;
  for (const double factor : {0.90, 0.85, 0.80, 0.78, 0.76, 0.74, 0.72}) {
    Fixture binding = make_fixture(machine);
    for (double & limit : binding.settings.actuation.cylinder_force_max) {
      limit *= factor;
    }
    crane_planning::TimingOcpRequest request;
    request.payload = empty_gripper();
    auto attempt = crane_planning::solve_timing_ocp(
      binding.model, binding.config, binding.path, binding.limits, request, binding.settings);
    walked += "\n  " + std::to_string(factor) + "x: " +
      (attempt.ok() ?
      "solved, duration " + std::to_string(attempt.value().trajectory.duration) +
      " s, peak force " + std::to_string(attempt.value().peak_demand.cylinder_force) :
      "refused -- " + attempt.status().message);
    if (attempt.ok() && attempt.value().trajectory.duration > loose.trajectory.duration) {
      tight = std::move(attempt).value();
      applied = factor;
      break;
    }
  }

  ASSERT_GT(applied, 0.0)
    << "no tightening of F_i^max between 0.90x and 0.72x made the move take longer, "
       "so the force limit is not deciding anything. The loose solve took " <<
    loose.trajectory.duration << " s at peak force " << loose.peak_demand.cylinder_force <<
    " of the physical limit, and the walk went:" << walked;
  EXPECT_GT(tight.trajectory.duration, loose.trajectory.duration);

  // And it was the force that decided it: the tight solve pushes the force to its
  // own scaled bound, which is kappa of the limit it was given. `PeakDemand` is a
  // fraction of the **configured** limit and the tightening went into that
  // configuration, so `factor` has already been divided out and does not appear
  // here -- multiplying by it again would ask the tight solve to sit at
  // `factor x kappa` of a limit that is itself `factor` smaller.
  const double bound = generous.settings.kappa;
  EXPECT_GT(tight.peak_demand.cylinder_force, 0.5 * bound)
    << "the force never approached the limit that was supposed to bind, at factor " << applied;
  EXPECT_LE(tight.peak_demand.cylinder_force, bound + 1.0e-6);
}

/// A flow-limited multi-axis move: several axes at once is what fills the pump.
/**
 * Pump flow is `mpc.md` 3 constraint 7 and it is the one constraint that is not
 * per-axis: `sum_i Q_i <= Q_P^max`, so a move that drives one axis hard never
 * shows it and a move that drives several at once does. Squeezing `Q_P^max`
 * alone must therefore slow this move down while every per-axis limit stays put.
 */
TEST(TimingOcp, AFlowLimitedMultiAxisMoveIsSlowerWhenThePumpIsSmaller)
{
  const Machine & machine = machines().front();
  const Fixture ample = make_fixture(machine);
  Fixture starved = make_fixture(machine);
  starved.settings.actuation.pump_flow_max *= 0.2;

  const crane_planning::TimingSolution fast = solved(ample);
  const crane_planning::TimingSolution slow = solved(starved);
  ASSERT_FALSE(fast.nodes.empty());
  ASSERT_FALSE(slow.nodes.empty());

  // The path of `make_fixture` moves the arm and the telescope together, so the
  // two contribute to the same sum.
  EXPECT_GT(slow.trajectory.duration, fast.trajectory.duration)
    << "a fifth of the pump cost nothing, so constraint 7 was not enforced";
  // The **duration** is what separates the two, and it is the only thing that
  // can. `PeakDemand` is a fraction of each solve's *own* configured limit, so
  // once both solves are pump-bound they both read `kappa` and the fraction says
  // nothing about which pump was smaller -- that is arithmetic, not a finding.
  // Issue 045's measured start widened stage 0 by `StartResolution`, which bought
  // the ample solve enough freedom to reach the bound as well; it read below
  // `kappa` before only because stage 0 was pinned to machine epsilon. What is
  // still worth asserting is that the starved solve is *on* its bound rather than
  // slowed down by something else that happened to move with the pump.
  EXPECT_NEAR(slow.peak_demand.pump_flow, ample.settings.kappa, 1.0e-3)
    << "the starved solve did not end up on its own pump bound, so the fifth of a pump is not "
    "what made it slower";
  EXPECT_GE(slow.peak_demand.pump_flow, fast.peak_demand.pump_flow - 1.0e-9);
}

/// The 0.95x of parameters.md 4, which is not kappa and is not optional.
TEST(TimingOcp, ThePumpPlanningFactorIsApplied)
{
  const Machine & machine = machines().front();
  const Fixture fixture = make_fixture(machine);
  EXPECT_DOUBLE_EQ(fixture.settings.actuation.pump_flow_max, 1.4e-3);
  EXPECT_DOUBLE_EQ(fixture.settings.actuation.pump_flow_planning_factor, 0.95);

  const crane_planning::TimingSolution solution = solved(fixture);
  ASSERT_FALSE(solution.nodes.empty());

  // The flow never exceeds kappa x 0.95 x Q_P^max, and `peak_demand.pump_flow`
  // is reported over `0.95 Q_P^max`, so the bound it must respect is kappa.
  EXPECT_LE(solution.peak_demand.pump_flow, fixture.settings.kappa + 1.0e-6);
  double worst = 0.0;
  for (const crane_planning::OcpNode & node : solution.nodes) {
    worst = std::max(worst, node.pump_flow);
  }
  EXPECT_LE(
    worst,
    fixture.settings.kappa * 0.95 * fixture.settings.actuation.pump_flow_max + 1.0e-12);
}

/// mpc.md 5.3 requirement 1: a solve that fails says so, and returns nothing.
/**
 * A backend that always reports success removes the only signal a supervisor
 * could act on, so the planner must refuse rather than clip. The wall-clock cap
 * of 7 is the cleanest way to force the refusal without inventing an infeasible
 * problem: set it below what any solve costs and the answer must be a failure
 * naming the solver's own status word.
 */
TEST(TimingOcp, ARefusalIsARefusalAndNotAClippedTrajectory)
{
  const Machine & machine = machines().front();
  Fixture fixture = make_fixture(machine);
  fixture.settings.max_wall_clock = 1.0e-6;

  crane_planning::TimingOcpRequest request;
  request.payload = empty_gripper();
  auto solution = crane_planning::solve_timing_ocp(
    fixture.model, fixture.config, fixture.path, fixture.limits, request, fixture.settings);

  ASSERT_FALSE(solution.ok()) << "a solve inside a microsecond budget reported success";
  // acados' own word for it, not a sentence this package made up.
  EXPECT_NE(solution.status().message.find("ACADOS_"), std::string::npos)
    << solution.status().message;
}
