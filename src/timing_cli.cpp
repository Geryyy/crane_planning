// The deployed timing OCP of `wiki/trajectory_planning.md` 5.2, driven from a
// command line instead of from a service call.
//
// # Why this exists
//
// Tuning the OCP needs a fast loop: change a weight, look at the sway, change it
// again. Doing that against `crane_planner_node` means a graph, a description, a
// joint state and a scene; doing it in Python means a *second* implementation of
// everything `timing_ocp.cpp` does around `acados_solve` -- the scaled bounds,
// the per-node equilibrium, the stage parameter blocks and the bisected warm
// start -- and a tuning session that ends in a number the deployment never sees.
//
// So this binary is the same `solve_timing_ocp` the service calls, with a
// `main` in front of it. What is tuned here is what runs. There is no ROS in it
// and no second solver setup anywhere: `scripts/timing_a2b.py` builds an argv,
// runs this, and plots the CSV.
//
// # What it is not
//
// It is not a planner. It takes the path's waypoints and fits them with the same
// `fit_c2_path` the planner uses, because the OCP's answer depends on the shape
// of the path it is given and tuning against a hand-written quintic would tune
// against a path shape the deployment never produces. It does no IK, no
// collision checking and no search -- `plan_a2b.py` is where a whole request
// goes.

#include <crane_planning_timing_ocp_generated.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/timing_ocp.hpp"

namespace
{

using crane_planning::kPathDof;
using crane_planning::kToolRow;

/// `--flag value` and nothing else. Repeated flags collect, which `--waypoint` needs.
class Arguments
{
public:
  Arguments(int argc, char ** argv)
  {
    for (int index = 1; index < argc; ++index) {
      const std::string token(argv[index]);
      if (token.rfind("--", 0) != 0) {
        throw std::runtime_error(
                "unexpected argument '" + token + "'; every input is --flag value");
      }
      if (index + 1 >= argc) {
        throw std::runtime_error(token + " needs a value");
      }
      values_[token.substr(2)].push_back(argv[++index]);
    }
  }

  [[nodiscard]] bool has(const std::string & name) const {return values_.count(name) != 0U;}

  [[nodiscard]] const std::string & one(const std::string & name) const
  {
    const auto found = values_.find(name);
    if (found == values_.end()) {
      throw std::runtime_error("--" + name + " is required");
    }
    if (found->second.size() != 1U) {
      throw std::runtime_error("--" + name + " may be given only once");
    }
    return found->second.front();
  }

  [[nodiscard]] const std::vector<std::string> & all(const std::string & name) const
  {
    static const std::vector<std::string> empty;
    const auto found = values_.find(name);
    return found == values_.end() ? empty : found->second;
  }

  [[nodiscard]] double number(const std::string & name, double fallback) const
  {
    if (!has(name)) {
      return fallback;
    }
    const std::string & text = one(name);
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
      throw std::runtime_error("--" + name + " is not a finite number: '" + text + "'");
    }
    return value;
  }

  /// An integer inside `[low, high]`. The range is not optional, and that is
  /// deliberate: `--intervals -1` widens to `SIZE_MAX` on the way into
  /// `TimingOcpSettings` and sails past the solver's own `< 4` guard, so the
  /// refusal has to happen here or not at all.
  [[nodiscard]] int integer(const std::string & name, int fallback, int low, int high) const
  {
    if (!has(name)) {
      return fallback;
    }
    const std::string & text = one(name);
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      throw std::runtime_error("--" + name + " is not an integer: '" + text + "'");
    }
    if (value < low || value > high) {
      throw std::runtime_error(
              "--" + name + " must lie in [" + std::to_string(low) + ", " +
              std::to_string(high) + "], got " + std::to_string(value));
    }
    return value;
  }

  /// The names this run did not use, so a typo is a refusal and not a silent default.
  [[nodiscard]] std::vector<std::string> unknown(const std::vector<std::string> & known) const
  {
    std::vector<std::string> extra;
    for (const auto & entry : values_) {
      if (std::find(known.begin(), known.end(), entry.first) == known.end()) {
        extra.push_back(entry.first);
      }
    }
    return extra;
  }

private:
  std::map<std::string, std::vector<std::string>> values_;
};

std::vector<double> comma_separated(const std::string & text, const std::string & name)
{
  std::vector<double> values;
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, ',')) {
    std::size_t consumed = 0;
    const double value = std::stod(field, &consumed);
    if (consumed != field.size() || !std::isfinite(value)) {
      throw std::runtime_error("--" + name + " holds a non-numeric field: '" + field + "'");
    }
    values.push_back(value);
  }
  return values;
}

/// A fixed-width comma list, or a refusal naming the width it wanted.
template<std::size_t N>
std::array<double, N> fixed_list(
  const Arguments & arguments, const std::string & name, const std::array<double, N> & fallback)
{
  if (!arguments.has(name)) {
    return fallback;
  }
  const std::vector<double> values = comma_separated(arguments.one(name), name);
  if (values.size() != N) {
    throw std::runtime_error(
            "--" + name + " needs " + std::to_string(N) + " comma-separated values, got " +
            std::to_string(values.size()));
  }
  std::array<double, N> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

std::string read_file(const std::string & path)
{
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

crane_model::Tool tool_from_string(const std::string & name)
{
  if (name == "pzs100") {
    return crane_model::Tool::Pzs100;
  }
  if (name == "epsilon7040") {
    return crane_model::Tool::Epsilon7040;
  }
  throw std::runtime_error("--tool must be pzs100 or epsilon7040, got '" + name + "'");
}

/// Every flag this binary reads. `Arguments::unknown` refuses anything else.
const std::vector<std::string> & known_flags()
{
  static const std::vector<std::string> names{
    "description", "tool", "waypoint", "q8", "output",
    "payload-mass", "payload-com", "speed-scale", "system-pressure-pa",
    "kappa", "intervals", "sway-weight", "input-weight",
    "sigma-rate-min", "sigma-rate-max", "sigma-accel-max",
    "q-u-max", "dq-u-max", "ddq-a-max",
    "pump-flow-max", "pump-flow-planning-factor",
    "sample-period", "max-wall-clock", "max-iterations",
    "tolerance-stationarity", "tolerance-feasibility", "levenberg-marquardt",
    "start-resolution-q-u", "start-resolution-dq-u", "start-resolution-sigma-rate-fraction",
    "start-q-u", "start-dq-u", "start-sigma-rate",
    "path-rate-headroom", "path-acceleration-span", "path-jerk-span",
    "reference-output"};
  return names;
}

/// `sigma_ddot` on the node grid, differentiated from the solved rate.
/**
 * `OcpNode` carries the states and the constrained quantities, not the input, so
 * this is `sigma_dot d(sigma_dot)/d(sigma)` by central differences rather than
 * the `u` acados returned. It is a plotting convenience and nothing here is
 * constrained by it -- the accelerations that *are* constrained come off
 * `OcpNode::ddq_a`, which is the solver's own answer.
 */
std::vector<double> differentiated_sigma_accel(const std::vector<crane_planning::OcpNode> & nodes)
{
  const std::size_t count = nodes.size();
  std::vector<double> accel(count, 0.0);
  if (count < 2U) {
    return accel;
  }
  for (std::size_t at = 0; at < count; ++at) {
    const std::size_t left = (at == 0U) ? 0U : at - 1U;
    const std::size_t right = (at + 1U == count) ? at : at + 1U;
    const double d_sigma = nodes[right].sigma - nodes[left].sigma;
    if (std::abs(d_sigma) > 1.0e-12) {
      accel[at] = nodes[at].sigma_rate *
        (nodes[right].sigma_rate - nodes[left].sigma_rate) / d_sigma;
    }
  }
  return accel;
}

/// Traversal time at each node: `int dsigma / sigma_dot`, trapezoidal on the grid.
std::vector<double> node_times(const std::vector<crane_planning::OcpNode> & nodes)
{
  std::vector<double> time(nodes.size(), 0.0);
  for (std::size_t at = 1; at < nodes.size(); ++at) {
    const double rate_left = nodes[at - 1U].sigma_rate;
    const double rate_right = nodes[at].sigma_rate;
    if (!(rate_left > 0.0) || !(rate_right > 0.0)) {
      throw std::runtime_error("the solve returned a non-positive path rate");
    }
    time[at] = time[at - 1U] +
      0.5 * (nodes[at].sigma - nodes[at - 1U].sigma) * (1.0 / rate_left + 1.0 / rate_right);
  }
  return time;
}

const std::array<const char *, kPathDof> kAxisNames{
  {"slew", "boom", "arm", "telescope", "rotator"}};
const std::array<const char *, crane_model::kPassiveDof> kPassiveNames{{"sway_1", "sway_2"}};

/// All six actuated rows, tool included: a *reference* carries the tool row too.
const std::array<const char *, crane_model::kActuatedDof> kActuatedNames{
  {"slew", "boom", "arm", "telescope", "rotator", "tool"}};

/// The trajectory that is actually published, on the uniform time grid it is published on.
/**
 * `write_csv` above dumps `solution.nodes`, which is the sigma grid the
 * constraints were imposed on -- the right thing to look at when tuning a
 * weight. It is **not** what reaches `/crane/reference`: that is
 * `solution.trajectory`, resampled onto `sample_period`. The two are different
 * objects and a tool that only ever showed the first could not check the second,
 * so both are available and the caller says which it wants.
 */
void write_reference_csv(const std::string & path, const crane_planning::TimingSolution & solution)
{
  const crane_planning::TimedTrajectory & reference = solution.trajectory;
  if (reference.q_a_ref.size() != reference.time_from_start.size() ||
    reference.dq_a_ref.size() != reference.time_from_start.size())
  {
    throw std::runtime_error("the emitted reference is ragged");
  }
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  }
  stream << "time_s";
  for (const char * name : kActuatedNames) {stream << ",q_" << name;}
  for (const char * name : kActuatedNames) {stream << ",dq_" << name;}
  stream << '\n';
  stream << std::setprecision(12);
  for (std::size_t at = 0; at < reference.time_from_start.size(); ++at) {
    stream << reference.time_from_start[at];
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      stream << ',' << reference.q_a_ref[at][static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      stream << ',' << reference.dq_a_ref[at][static_cast<Eigen::Index>(row)];
    }
    stream << '\n';
  }
}

void write_csv(
  const std::string & path, const crane_planning::TimingSolution & solution,
  const std::array<double, crane_model::kActuatedDof> & force_limit, double flow_limit)
{
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  }
  stream << "time_s,sigma,sigma_rate,sigma_accel";
  for (const char * name : kAxisNames) {stream << ",q_" << name;}
  for (const char * name : kAxisNames) {stream << ",dq_" << name;}
  for (const char * name : kAxisNames) {stream << ",ddq_" << name;}
  for (const char * name : kPassiveNames) {stream << ",q_" << name;}
  for (const char * name : kPassiveNames) {stream << ",dq_" << name;}
  for (const char * name : kPassiveNames) {stream << ",q_eq_" << name;}
  for (const char * name : kAxisNames) {stream << ",cylinder_force_" << name << "_N";}
  for (const char * name : kAxisNames) {stream << ",force_use_" << name;}
  stream << ",pump_flow_m3_s,pump_use\n";

  const std::vector<double> time = node_times(solution.nodes);
  const std::vector<double> sigma_accel = differentiated_sigma_accel(solution.nodes);
  stream << std::setprecision(12);
  for (std::size_t at = 0; at < solution.nodes.size(); ++at) {
    const crane_planning::OcpNode & node = solution.nodes[at];
    stream << time[at] << ',' << node.sigma << ',' << node.sigma_rate << ',' << sigma_accel[at];
    for (std::size_t row = 0; row < kPathDof; ++row) {
      stream << ',' << node.q_a[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < kPathDof; ++row) {
      stream << ',' << node.dq_a[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < kPathDof; ++row) {
      stream << ',' << node.ddq_a[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      stream << ',' << node.q_u[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      stream << ',' << node.dq_u[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
      stream << ',' << node.q_u_equilibrium[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < kPathDof; ++row) {
      stream << ',' << node.cylinder_force[static_cast<Eigen::Index>(row)];
    }
    for (std::size_t row = 0; row < kPathDof; ++row) {
      stream << ',' <<
        std::abs(node.cylinder_force[static_cast<Eigen::Index>(row)]) / force_limit[row];
    }
    stream << ',' << node.pump_flow << ',' << node.pump_flow / flow_limit << '\n';
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Arguments arguments(argc, argv);
    const std::vector<std::string> extra = arguments.unknown(known_flags());
    if (!extra.empty()) {
      std::string message = "unknown flag(s):";
      for (const std::string & name : extra) {message += " --" + name;}
      throw std::runtime_error(message);
    }

    // ----------------------------------------------------------------
    // The machine
    // ----------------------------------------------------------------
    crane_model::ModelConfig config;
    config.robot_description_xml = read_file(arguments.one("description"));
    config.tool = tool_from_string(arguments.has("tool") ? arguments.one("tool") : "pzs100");
    auto model_result = crane_model::Model::create(config);
    if (!model_result.ok()) {
      throw std::runtime_error("model: " + model_result.status().message);
    }
    const crane_model::Model model = std::move(model_result).value();

    auto limits_result =
      crane_planning::read_joint_limits(config.robot_description_xml, model.urdf_joint_names());
    if (!limits_result.ok()) {
      throw std::runtime_error("joint limits: " + limits_result.status().message);
    }
    const crane_planning::JointLimits limits = std::move(limits_result).value();

    // ----------------------------------------------------------------
    // The path, fitted exactly as the planner fits it
    // ----------------------------------------------------------------
    const std::vector<std::string> & waypoint_text = arguments.all("waypoint");
    if (waypoint_text.size() < 2U) {
      throw std::runtime_error("--waypoint is needed at least twice (start first, goal last)");
    }
    crane_planning::PathFitRequest fit;
    for (const std::string & text : waypoint_text) {
      const std::vector<double> values = comma_separated(text, "waypoint");
      if (values.size() != kPathDof) {
        throw std::runtime_error(
                "--waypoint needs " + std::to_string(kPathDof) +
                " comma-separated coordinates (slew,boom,arm,telescope,rotator)");
      }
      crane_planning::PathVector point = crane_planning::PathVector::Zero();
      for (std::size_t row = 0; row < kPathDof; ++row) {
        point[static_cast<Eigen::Index>(row)] = values[row];
      }
      fit.waypoints.push_back(point);
    }
    fit.segment_names.assign(fit.waypoints.size() - 1U, "traverse");
    // Required, with no default. On the PZS100 `q8` is the rail opening in
    // **metres** and on the Epsilon 7040 it is the jaw angle in **radians**, so a
    // single literal here would be one number standing for two different physical
    // quantities. The caller knows which machine it asked for; this does not.
    fit.q8_start = arguments.number("q8", std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(fit.q8_start)) {
      throw std::runtime_error(
              "--q8 is required: it is the tool coordinate, in metres on the PZS100 rail "
              "gripper and in radians on the Epsilon 7040 jaw, and there is no default that "
              "is right for both");
    }
    fit.q8_goal = fit.q8_start;

    crane_planning::PathFitSettings fit_settings;
    fit_settings.rate_headroom = arguments.number("path-rate-headroom", fit_settings.rate_headroom);
    fit_settings.acceleration_span =
      arguments.number("path-acceleration-span", fit_settings.acceleration_span);
    fit_settings.jerk_span = arguments.number("path-jerk-span", fit_settings.jerk_span);

    auto path_result = crane_planning::fit_c2_path(fit, limits, fit_settings);
    if (!path_result.ok()) {
      throw std::runtime_error("path fit: " + path_result.status().message);
    }
    const crane_planning::GeometricPath path = std::move(path_result).value();

    // ----------------------------------------------------------------
    // The settings, defaulted from `TimingOcpSettings` and never restated
    // ----------------------------------------------------------------
    crane_planning::TimingOcpSettings settings;
    // Required for the same reason, and a sharper one: `config/hydraulic_limits.yaml`
    // is where this number is written down *with its evidence*, and a default here
    // would be the one copy of it that carries none.
    const double system_pressure =
      arguments.number("system-pressure-pa", std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(system_pressure)) {
      throw std::runtime_error(
              "--system-pressure-pa is required; config/hydraulic_limits.yaml is the file "
              "that carries the number and the evidence for it");
    }
    auto forces = crane_model::derive_cylinder_force_limits(model, system_pressure);
    if (!forces.ok()) {
      throw std::runtime_error("cylinder force limits: " + forces.status().message);
    }
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      settings.actuation.cylinder_force_max[row] = forces.value()[row].symmetric();
    }
    settings.actuation.ddq_a_max =
      fixed_list<crane_model::kActuatedDof>(arguments, "ddq-a-max", settings.actuation.ddq_a_max);
    settings.actuation.pump_flow_max =
      arguments.number("pump-flow-max", settings.actuation.pump_flow_max);
    settings.actuation.pump_flow_planning_factor =
      arguments.number("pump-flow-planning-factor", settings.actuation.pump_flow_planning_factor);

    settings.kappa = arguments.number("kappa", settings.kappa);
    settings.intervals = static_cast<std::size_t>(
      arguments.integer("intervals", static_cast<int>(settings.intervals), 4, 400));
    settings.sway_weight = arguments.number("sway-weight", settings.sway_weight);
    settings.input_weight = arguments.number("input-weight", settings.input_weight);
    settings.sigma_rate_min = arguments.number("sigma-rate-min", settings.sigma_rate_min);
    settings.sigma_rate_max = arguments.number("sigma-rate-max", settings.sigma_rate_max);
    settings.sigma_accel_max = arguments.number("sigma-accel-max", settings.sigma_accel_max);
    settings.q_u_max = fixed_list<crane_model::kPassiveDof>(arguments, "q-u-max", settings.q_u_max);
    settings.dq_u_max =
      fixed_list<crane_model::kPassiveDof>(arguments, "dq-u-max", settings.dq_u_max);
    settings.sample_period = arguments.number("sample-period", settings.sample_period);
    settings.max_wall_clock = arguments.number("max-wall-clock", settings.max_wall_clock);
    settings.max_iterations = arguments.integer(
      "max-iterations", settings.max_iterations, 1, CRANE_PLANNING_TIMING_MAX_ITERATIONS);
    settings.tolerance_stationarity =
      arguments.number("tolerance-stationarity", settings.tolerance_stationarity);
    settings.tolerance_feasibility =
      arguments.number("tolerance-feasibility", settings.tolerance_feasibility);
    settings.levenberg_marquardt =
      arguments.number("levenberg-marquardt", settings.levenberg_marquardt);
    settings.start_resolution.q_u =
      arguments.number("start-resolution-q-u", settings.start_resolution.q_u);
    settings.start_resolution.dq_u =
      arguments.number("start-resolution-dq-u", settings.start_resolution.dq_u);
    settings.start_resolution.sigma_rate_fraction = arguments.number(
      "start-resolution-sigma-rate-fraction", settings.start_resolution.sigma_rate_fraction);

    // ----------------------------------------------------------------
    // The request
    // ----------------------------------------------------------------
    crane_planning::TimingOcpRequest request;
    request.speed_scale = arguments.number("speed-scale", request.speed_scale);
    // Declared rather than defaulted: Eigen does not zero a default-constructed
    // matrix and the model rejects an asymmetric inertia, so an empty gripper is
    // built here exactly as `description_fixture.hpp` builds it for the tests.
    request.payload.valid = true;
    request.payload.mass_kg = arguments.number("payload-mass", 0.0);
    const std::array<double, 3> com =
      fixed_list<3>(arguments, "payload-com", std::array<double, 3>{{0.0, 0.0, 0.0}});
    request.payload.center_of_mass_k8_m = Eigen::Vector3d(com[0], com[1], com[2]);
    request.payload.inertia_k8_kg_m2.setZero();

    // 7's measured initial condition. Without these the CLI can only ever solve
    // the stopped start, which would leave the whole moving-re-plan path -- the
    // case 7 names as the one the legacy stack got wrong -- with no offline
    // harness, and would make `--start-resolution-sigma-rate-fraction` a flag
    // that cannot affect any run this binary is able to perform.
    if (arguments.has("start-q-u") || arguments.has("start-dq-u")) {
      const std::array<double, crane_model::kPassiveDof> zero{{0.0, 0.0}};
      const auto q_u = fixed_list<crane_model::kPassiveDof>(arguments, "start-q-u", zero);
      const auto dq_u = fixed_list<crane_model::kPassiveDof>(arguments, "start-dq-u", zero);
      request.start.measured = true;
      for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
        request.start.q_u[static_cast<Eigen::Index>(row)] = q_u[row];
        request.start.dq_u[static_cast<Eigen::Index>(row)] = dq_u[row];
      }
    }
    if (arguments.has("start-sigma-rate")) {
      request.start.sigma_rate_pinned = true;
      request.start.sigma_rate = arguments.number("start-sigma-rate", 0.0);
    }

    // ----------------------------------------------------------------
    // The solve. This is the deployment's own function, not a copy of it.
    // ----------------------------------------------------------------
    auto solution_result = crane_planning::solve_timing_ocp(model, path, limits, request, settings);
    if (!solution_result.ok()) {
      throw std::runtime_error("timing OCP: " + solution_result.status().message);
    }
    const crane_planning::TimingSolution solution = std::move(solution_result).value();

    std::array<double, crane_model::kActuatedDof> force_limit{};
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      force_limit[row] = settings.kappa * settings.actuation.cylinder_force_max[row];
    }
    const double flow_limit = settings.kappa *
      settings.actuation.pump_flow_planning_factor * settings.actuation.pump_flow_max;

    if (arguments.has("output")) {
      write_csv(arguments.one("output"), solution, force_limit, flow_limit);
    }
    if (arguments.has("reference-output")) {
      write_reference_csv(arguments.one("reference-output"), solution);
    }

    // `key=value`, one per line: the summary is read by `timing_a2b.py` and by a
    // person, and a format that needs a parser would be a third thing to keep
    // current.
    std::cout << std::setprecision(12);
    std::cout << "status=" << solution.solver_status << '\n';
    std::cout << "duration_s=" << solution.trajectory.duration << '\n';
    std::cout << "iterations=" << solution.iterations << '\n';
    std::cout << "solve_time_s=" << solution.solve_time << '\n';
    std::cout << "kappa=" << solution.kappa << '\n';
    std::cout << "speed_scale=" << solution.speed_scale << '\n';
    std::cout << "nodes=" << solution.nodes.size() << '\n';
    std::cout << "reference_samples=" << solution.trajectory.time_from_start.size() << '\n';
    std::cout << "limiting_fraction=" << solution.trajectory.limiting_fraction << '\n';
    std::cout << "peak_joint_velocity=" << solution.peak_demand.joint_velocity << '\n';
    std::cout << "peak_joint_acceleration=" << solution.peak_demand.joint_acceleration << '\n';
    std::cout << "peak_cylinder_force=" << solution.peak_demand.cylinder_force << '\n';
    std::cout << "peak_pump_flow=" << solution.peak_demand.pump_flow << '\n';
    std::cout << "cylinder_force_limit_N=";
    for (std::size_t row = 0; row < kPathDof; ++row) {
      std::cout << (row == 0U ? "" : ",") << force_limit[row];
    }
    std::cout << '\n';
    std::cout << "pump_flow_limit_m3_s=" << flow_limit << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "crane_planning_timing_cli: " << error.what() << '\n';
    return 2;
  }
}
