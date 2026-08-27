// The two machine descriptions the offline tests are run against, and the model
// and planner context built from each.
//
// Read where they live -- `crane_model/test/description` -- rather than copied
// in. They are generated from `src/epsilon_crane_description` and are the same
// files `crane_model`'s own contract test holds its compiled geometry against,
// so a copy here would be a second machine to keep current.

#ifndef DESCRIPTION_FIXTURE_HPP_
#define DESCRIPTION_FIXTURE_HPP_

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>

#include "crane_model/model.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/planner_core.hpp"

namespace crane_planning_test
{

struct Machine
{
  const char * name;
  const char * file;
  crane_model::Tool tool;
  double q8;  ///< a tool coordinate inside the range that description gives it
};

/// One entry per tool, because the tool decides which joints are canonical.
inline const std::vector<Machine> & machines()
{
  static const std::vector<Machine> all{
    {"pzs100", "pzs100.urdf", crane_model::Tool::Pzs100, 0.3},
    {"epsilon7040", "epsilon_7040.urdf", crane_model::Tool::Epsilon7040, 0.2}};
  return all;
}

inline std::string description(const Machine & machine)
{
  const std::string path =
    std::string(CRANE_PLANNING_MACHINE_DESCRIPTION_DIR) + "/" + machine.file;
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

inline crane_model::Model build_model(const Machine & machine)
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description(machine);
  config.tool = machine.tool;
  auto model = crane_model::Model::create(config);
  if (!model.ok()) {
    throw std::runtime_error(machine.name + std::string(": ") + model.status().message);
  }
  return std::move(model).value();
}

/// An empty gripper, declared rather than defaulted.
/**
 * `crane_model::Payload` is a plain value type and Eigen does not zero a
 * default-constructed matrix, so `Payload{}.inertia_k8_kg_m2` holds whatever was
 * on the stack -- and the model rejects a payload whose inertia is not
 * symmetric. Every fixture that carries a payload therefore builds it here, and
 * an empty gripper is `mass_kg == 0` with `valid == true`, which is not the same
 * thing as the explicit unknown `valid == false` of the model API contract 4.
 */
inline crane_model::Payload empty_gripper()
{
  crane_model::Payload payload;
  payload.valid = true;
  payload.mass_kg = 0.0;
  payload.center_of_mass_k8_m.setZero();
  payload.inertia_k8_kg_m2.setZero();
  return payload;
}

/// The middle of every bounded actuated range, which both descriptions allow.
/**
 * The configuration offline tests start from, and deliberately **not** every
 * joint at zero. At zero the PZS100's rail gripper lies against the arm and the
 * inner telescope -- 034 measured 25 mm at the neutral pose and the primitives
 * are single enclosing boxes -- so a path from there is refused by a working
 * self-collision check. That refusal is the check doing its job; a test that is
 * about something else should not start where it fires.
 */
inline crane_model::QA centred(const crane_planning::JointLimits & limits, const Machine & machine)
{
  crane_model::QA q_a = crane_model::QA::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] =
      limits.axis[row].bounded ? limits.axis[row].centre() : 0.0;
  }
  q_a[static_cast<Eigen::Index>(crane_planning::kToolRow)] = machine.q8;
  return q_a;
}

/// `centred()`, pulled back into the range the arm actually **works** in.
/**
 * `centred()` centres every axis in the range the *description* gives it, and
 * `wiki/trajectory_planning.md` 4.2 says in passing that the arm joint's range
 * is far wider than its working range. That gap has a sharp edge: the arm
 * cylinder's transmission Jacobian passes through zero at `theta3 = 1.85 rad`,
 * which is almost exactly where the URDF's centre falls, and at a transmission
 * zero the cylinder produces no torque about the joint at all. The static
 * cylinder force there is some `5e7` N -- four hundred times what any relief
 * setting delivers -- so `mpc.md` 3 constraint 6 is infeasible at that pose
 * before a solver starts, and the machine cannot hold it either.
 *
 * That did not matter while the timing was issue 038's ramp, which constrained
 * nothing. It matters now that the timing is the OCP of 5.2, and it is the same
 * shape of correction as the one this fixture already carries for
 * the all-zero start: a pose the real check refuses, in a test that is not about
 * that refusal. Every case that plans a *timing* therefore starts here, and the
 * cases that only test geometry keep `centred()`.
 */
inline crane_model::QA working_centred(
  const crane_planning::JointLimits & limits, const Machine & machine)
{
  crane_model::QA q_a = centred(limits, machine);
  q_a[1] = -0.2;  // theta2_boom_joint
  q_a[2] = 0.6;   // theta3_arm_joint, well clear of the 1.85 transmission zero
  return q_a;
}

/// A move worth checking: slew across, drop the boom, extend a little.
inline crane_model::QA moved(
  const crane_model::QA & start, const crane_planning::JointLimits & limits)
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

/// The canonical eight of one actuated configuration, with the tool hanging.
inline crane_model::Q settled(const crane_model::Model & model, const crane_model::QA & q_a)
{
  auto equilibrium = model.passive_equilibrium(q_a, empty_gripper());
  if (!equilibrium.ok()) {
    throw std::runtime_error(equilibrium.status().message);
  }
  return crane_planning::expand(q_a, equilibrium.value());
}

inline crane_planning::PlannerContext build_context(
  const crane_model::Model & model, const Machine & machine,
  const crane_planning::PlannerSettings & settings = crane_planning::PlannerSettings{})
{
  crane_model::ModelConfig config;
  config.robot_description_xml = description(machine);
  config.tool = machine.tool;
  auto context = crane_planning::build_planner(model, config, settings);
  if (!context.ok()) {
    throw std::runtime_error(machine.name + std::string(": ") + context.status().message);
  }
  return std::move(context).value();
}

}  // namespace crane_planning_test

#endif  // DESCRIPTION_FIXTURE_HPP_
