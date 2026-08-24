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

#include "crane_model/model.hpp"
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

inline crane_planning::PlannerContext build_context(
  const crane_model::Model & model, const Machine & machine,
  const crane_planning::PlannerSettings & settings = crane_planning::PlannerSettings{})
{
  auto context = crane_planning::build_planner(model, description(machine), settings);
  if (!context.ok()) {
    throw std::runtime_error(machine.name + std::string(": ") + context.status().message);
  }
  return std::move(context).value();
}

}  // namespace crane_planning_test

#endif  // DESCRIPTION_FIXTURE_HPP_
