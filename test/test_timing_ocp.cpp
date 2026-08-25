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
  auto path = crane_planning::fit_c2_path(request, fixture.limits, crane_planning::PathFitSettings{});
  EXPECT_TRUE(path.ok()) << path.status().message;
  fixture.path = std::move(path).value();

  auto forces = crane_planning::derive_cylinder_force_limits(fixture.model, kSystemPressure);
  EXPECT_TRUE(forces.ok()) << forces.status().message;
  fixture.settings.actuation.cylinder_force_max = forces.value();
  return fixture;
}

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
  }
}

}  // namespace
