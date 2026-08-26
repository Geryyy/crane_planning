#include "crane_planning/timing_ocp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"

#include "acados_solver_crane_planning_timing_epsilon7040.h"  // NOLINT(build/include_subdir)
#include "acados_solver_crane_planning_timing_pzs100.h"  // NOLINT(build/include_subdir)

#include "crane_planning_timing_epsilon7040_output.h"  // NOLINT(build/include_subdir)
#include "crane_planning_timing_pzs100_output.h"  // NOLINT(build/include_subdir)
}

#include "crane_planning_timing_ocp_generated.h"  // NOLINT(build/include_subdir)

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Result;
using crane_model::Status;

// Every dimension below is read off `scripts/export_timing_ocp.py`'s own header
// rather than restated here. The problem is defined in Python now; what is left
// in this file is the binding that opens the shipped solver, writes what stays
// runtime-settable, builds the warm start acados does not provide, and reads
// acados' own answers back.
constexpr int kOcpStateDof = CRANE_PLANNING_TIMING_NX;   ///< (sigma, sigma_dot, q_u, dq_u)
constexpr int kOcpInputDof = CRANE_PLANNING_TIMING_NU;   ///< sigma_ddot
constexpr int kOcpConstraints = CRANE_PLANNING_TIMING_NH;
constexpr int kOcpParameters = CRANE_PLANNING_TIMING_NP;
constexpr int kResidualDof = CRANE_PLANNING_TIMING_NY;
constexpr int kTerminalResidualDof = CRANE_PLANNING_TIMING_NY_E;
constexpr int kPathBlock = CRANE_PLANNING_TIMING_PATH_BLOCK;
constexpr std::size_t kPayloadDof = CRANE_PLANNING_TIMING_PAYLOAD_DOF;

constexpr int kConstraintAcceleration = CRANE_PLANNING_TIMING_CONSTRAINT_ACCELERATION;
constexpr int kConstraintCylinderForce = CRANE_PLANNING_TIMING_CONSTRAINT_CYLINDER_FORCE;
constexpr int kConstraintPumpFlow = CRANE_PLANNING_TIMING_CONSTRAINT_PUMP_FLOW;

constexpr std::size_t kOutputCylinderForce = CRANE_PLANNING_TIMING_OUTPUT_CYLINDER_FORCE;
constexpr std::size_t kOutputAxisFlow = CRANE_PLANNING_TIMING_OUTPUT_AXIS_FLOW;
constexpr std::size_t kOutputDof = CRANE_PLANNING_TIMING_OUTPUT_DOF;

/// The generated tree's own conditioning divisors, read and never re-derived.
/**
 * Every row of `h` is a physical quantity divided by one of these, so the bound
 * that goes in is the machine's limit divided by the same number. The divisor is
 * conditioning and the bound is the limit -- see `export_timing_ocp.py`.
 */
constexpr std::array<double, static_cast<std::size_t>(kOcpConstraints)> kConstraintScale =
  CRANE_PLANNING_TIMING_CONSTRAINT_SCALE;

/// The `(row, column)` of the six independent entries of `Theta_L`, in `p`'s order.
constexpr std::array<std::array<int, 2>, 6> kInertiaEntries = {
  CRANE_PLANNING_TIMING_INERTIA_ENTRIES};

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

/// acados' status word, as the message a refusal carries.
std::string acados_status_word(int status)
{
  switch (status) {
    case ACADOS_SUCCESS: return "ACADOS_SUCCESS";
    case ACADOS_NAN_DETECTED: return "ACADOS_NAN_DETECTED";
    case ACADOS_MAXITER: return "ACADOS_MAXITER";
    case ACADOS_MINSTEP: return "ACADOS_MINSTEP";
    case ACADOS_QP_FAILURE: return "ACADOS_QP_FAILURE";
    case ACADOS_READY: return "ACADOS_READY";
    case ACADOS_TIMEOUT: return "ACADOS_TIMEOUT";
    default: return "acados status " + std::to_string(status);
  }
}

// The two shapes of generated function this file calls. acados' own casadi
// output takes `void * mem`; CasADi's takes an `int` memory token, and
// `<tool>_output` is CasADi's.
using CasadiFunction = int (*)(const double **, double **, int *, double *, int);

/// One generated timing solver, as the handful of entry points this file uses.
/**
 * Two artifacts, one shape. The capsule types differ per tool -- that is what
 * makes them two solvers rather than one parameterised one -- so the handle is
 * `void *` and the thunks below are where the cast lives, in exactly one place
 * per entry point per tool.
 */
struct Backend
{
  const char * name{nullptr};
  void * (*create_capsule)() = nullptr;
  int (*create_with_grid)(void *, int, double *) = nullptr;
  int (*solve)(void *) = nullptr;
  int (*destroy)(void *) = nullptr;
  int (*free_capsule)(void *) = nullptr;
  int (*update_params)(void *, int, double *, int) = nullptr;
  ocp_nlp_config * (*config)(void *) = nullptr;
  ocp_nlp_dims * (*dims)(void *) = nullptr;
  ocp_nlp_in * (*in)(void *) = nullptr;
  ocp_nlp_out * (*out)(void *) = nullptr;
  ocp_nlp_solver * (*solver)(void *) = nullptr;
  void * (*opts)(void *) = nullptr;

  CasadiFunction output{nullptr};
};

// One macro, two uses. The alternative is thirty lines of the same thunks with a
// different token pasted into every symbol, which is what a macro is for.
#define CRANE_PLANNING_TIMING_BACKEND(NAME)                                            \
  []() {                                                                               \
    using Capsule = NAME ## _solver_capsule;                                           \
    Backend backend;                                                                   \
    backend.name = #NAME;                                                              \
    backend.create_capsule = []() -> void * {return NAME ## _acados_create_capsule();}; \
    backend.create_with_grid = [](void * h, int n, double * steps) {                   \
        return NAME ## _acados_create_with_discretization(static_cast<Capsule *>(h), n, steps); \
      };                                                                               \
    backend.solve = [](void * h) {return NAME ## _acados_solve(static_cast<Capsule *>(h));};   \
    backend.destroy = [](void * h) {return NAME ## _acados_free(static_cast<Capsule *>(h));};  \
    backend.free_capsule =                                                             \
      [](void * h) {return NAME ## _acados_free_capsule(static_cast<Capsule *>(h));};  \
    backend.update_params = [](void * h, int stage, double * value, int np) {          \
        return NAME ## _acados_update_params(static_cast<Capsule *>(h), stage, value, np); \
      };                                                                               \
    backend.config =                                                                   \
      [](void * h) {return NAME ## _acados_get_nlp_config(static_cast<Capsule *>(h));}; \
    backend.dims =                                                                     \
      [](void * h) {return NAME ## _acados_get_nlp_dims(static_cast<Capsule *>(h));};  \
    backend.in = [](void * h) {return NAME ## _acados_get_nlp_in(static_cast<Capsule *>(h));}; \
    backend.out = [](void * h) {return NAME ## _acados_get_nlp_out(static_cast<Capsule *>(h));}; \
    backend.solver =                                                                   \
      [](void * h) {return NAME ## _acados_get_nlp_solver(static_cast<Capsule *>(h));}; \
    backend.opts =                                                                     \
      [](void * h) {return NAME ## _acados_get_nlp_opts(static_cast<Capsule *>(h));};  \
    backend.output = &NAME ## _output;                                                 \
    return backend;                                                                    \
  }()

Backend backend_for(crane_model::Tool tool)
{
  if (tool == crane_model::Tool::Pzs100) {
    return CRANE_PLANNING_TIMING_BACKEND(crane_planning_timing_pzs100);
  }
  return CRANE_PLANNING_TIMING_BACKEND(crane_planning_timing_epsilon7040);
}

#undef CRANE_PLANNING_TIMING_BACKEND

/// The full actuated row of a path sample: the five path coordinates, then q8.
/**
 * **The tool row is pinned, at every derivative above the zeroth.** `q8` is
 * carried, because the model has to be evaluated at the configuration the tool is
 * actually in -- its inertia is in `M(q)` and dropping it would time the arm
 * against a lighter machine. `dq8` and `ddq8` are **not**: the OCP holds the tool
 * still, so a rate or an acceleration on that row would be a motion this solve
 * neither planned nor bounded. Along an arm path they are zero anyway (`q8` is not
 * a path variable, 4.1), and zeroing them here is what makes that a property of
 * the problem rather than of the path that happened to arrive.
 */
crane_model::QA actuated(const PathSample & sample, int derivative)
{
  crane_model::QA value = crane_model::QA::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    value[axis] = derivative == 0 ? sample.q_a[axis] :
      (derivative == 1 ? sample.dq_a[axis] : sample.ddq_a[axis]);
  }
  const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
  value[tool] = derivative == 0 ? sample.q8 : 0.0;
  return value;
}

/// The payload half of `p`, packed as `export_timing_ocp.py` packs it.
std::array<double, kPayloadDof> payload_block(const crane_model::Payload & payload)
{
  std::array<double, kPayloadDof> block{};
  block[CRANE_PLANNING_TIMING_PAYLOAD_MASS] = payload.mass_kg;
  for (std::size_t row = 0; row < 3U; ++row) {
    block[CRANE_PLANNING_TIMING_PAYLOAD_COM + row] =
      payload.center_of_mass_k8_m[static_cast<Eigen::Index>(row)];
  }
  for (std::size_t entry = 0; entry < kInertiaEntries.size(); ++entry) {
    block[CRANE_PLANNING_TIMING_PAYLOAD_INERTIA + entry] = payload.inertia_k8_kg_m2(
      static_cast<Eigen::Index>(kInertiaEntries[entry][0]),
      static_cast<Eigen::Index>(kInertiaEntries[entry][1]));
  }
  return block;
}

/// One evaluation of the shipped output map `z`, in physical units.
/**
 * The generated solver reports the eleven rows of `h`, each already divided by
 * its conditioning constant. Three things want the physical quantity instead:
 * `OcpNode`'s record of what the answer demands of the machine, the static-force
 * refusal that names a path no timing exists for, and the bisected warm start,
 * which asks how far outside constraints 6 and 7 a node is at a candidate rate.
 *
 * `<tool>_output` is `crane_symbolic`'s own `z` over the shared module's own
 * `(x, u, p)` -- the same function `crane_mpc` ships. All six axes are returned,
 * because the tool cylinder is still in the model when its coordinate leaves the
 * OCP.
 */
bool evaluate_output(
  const Backend & backend, const crane_model::QA & q_a, const crane_model::QA & dq_a,
  const crane_model::QA & ddq_a, const crane_model::QU & q_u, const crane_model::DQU & dq_u,
  const std::array<double, kPayloadDof> & payload, std::vector<double> & z)
{
  const int planned = static_cast<int>(kPathDof);
  std::vector<double> state;
  state.reserve(2U * static_cast<std::size_t>(planned + crane_model::kPassiveDof));
  for (int row = 0; row < planned; ++row) {state.push_back(q_a[row]);}
  state.push_back(q_u[0]);
  state.push_back(q_u[1]);
  for (int row = 0; row < planned; ++row) {state.push_back(dq_a[row]);}
  state.push_back(dq_u[0]);
  state.push_back(dq_u[1]);

  std::vector<double> input(static_cast<std::size_t>(planned), 0.0);
  for (int row = 0; row < planned; ++row) {
    input[static_cast<std::size_t>(row)] = ddq_a[row];
  }

  std::vector<double> parameter;
  parameter.reserve(1U + kPayloadDof);
  parameter.push_back(q_a[static_cast<Eigen::Index>(kToolRow)]);
  for (const double value : payload) {parameter.push_back(value);}

  z.assign(kOutputDof, 0.0);
  const double * arguments[3] = {state.data(), input.data(), parameter.data()};
  double * results[1] = {z.data()};
  return backend.output(arguments, results, nullptr, nullptr, 0) == 0;
}

/// The scaled limits one solve is run against.
struct ScaledLimits
{
  crane_model::QA dq_a_max{crane_model::QA::Zero()};    ///< kappa * speed_scale * description
  crane_model::QA ddq_a_max{crane_model::QA::Zero()};   ///< kappa * parameters.md 3
  crane_model::QA force_max{crane_model::QA::Zero()};   ///< kappa * F^max
  double flow_max{};                                    ///< kappa * 0.95 * Q_P^max
};

ScaledLimits scale_limits(
  const JointLimits & limits, const TimingOcpSettings & settings, double speed_scale)
{
  ScaledLimits scaled;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    scaled.dq_a_max[axis] = settings.kappa * speed_scale * limits.axis[row].dq_max;
    scaled.ddq_a_max[axis] = settings.kappa * settings.actuation.ddq_a_max[row];
    scaled.force_max[axis] = settings.kappa * settings.actuation.cylinder_force_max[row];
  }
  scaled.flow_max = settings.kappa * settings.actuation.pump_flow_planning_factor *
    settings.actuation.pump_flow_max;
  return scaled;
}

/// Every acados object one solve owns, freed in the order acados wants.
class TimingSolver
{
public:
  explicit TimingSolver(Backend backend)
  : backend_(backend) {}

  ~TimingSolver()
  {
    if (capsule_ != nullptr) {
      backend_.destroy(capsule_);
      backend_.free_capsule(capsule_);
    }
  }

  TimingSolver(const TimingSolver &) = delete;
  TimingSolver & operator=(const TimingSolver &) = delete;
  TimingSolver(TimingSolver &&) = delete;
  TimingSolver & operator=(TimingSolver &&) = delete;

  /// Open the shipped solver on this solve's own grid.
  /**
   * `<name>_acados_create_with_discretization(capsule, N, steps)` is generated
   * beside the fixed-`N` entry point, so `TimingOcpSettings::intervals` is still
   * an argument and the artifact's own `DEFAULT_INTERVALS` is a default rather
   * than a contract. The steps carry `dsigma` into the ERK4 integrator, which is
   * what makes sigma and not time the independent variable.
   */
  bool open(int intervals, double d_sigma)
  {
    capsule_ = backend_.create_capsule();
    if (capsule_ == nullptr) {
      return false;
    }
    std::vector<double> steps(static_cast<std::size_t>(intervals), d_sigma);
    if (backend_.create_with_grid(capsule_, intervals, steps.data()) != 0) {
      return false;
    }
    config_ = backend_.config(capsule_);
    dims_ = backend_.dims(capsule_);
    in_ = backend_.in(capsule_);
    out_ = backend_.out(capsule_);
    solver_ = backend_.solver(capsule_);
    opts_ = backend_.opts(capsule_);
    return true;
  }

  const Backend & backend() const {return backend_;}
  void * capsule() const {return capsule_;}
  ocp_nlp_config * config() const {return config_;}
  ocp_nlp_dims * dims() const {return dims_;}
  ocp_nlp_in * in() const {return in_;}
  ocp_nlp_out * out() const {return out_;}
  ocp_nlp_solver * solver() const {return solver_;}
  void * opts() const {return opts_;}

private:
  Backend backend_;
  void * capsule_{nullptr};
  ocp_nlp_config * config_{nullptr};
  ocp_nlp_dims * dims_{nullptr};
  ocp_nlp_in * in_{nullptr};
  ocp_nlp_out * out_{nullptr};
  ocp_nlp_solver * solver_{nullptr};
  void * opts_{nullptr};
};

}  // namespace

crane_model::Result<std::array<double, crane_model::kActuatedDof>>
derive_cylinder_force_limits(const crane_model::Model & model, double system_pressure_pa)
{
  using Limits = std::array<double, crane_model::kActuatedDof>;
  if (!std::isfinite(system_pressure_pa) || !(system_pressure_pa > 0.0)) {
    return Result<Limits>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the hydraulic relief pressure must be finite and positive; it is the one number the "
        "cylinder force limit cannot be derived from the description and it is not measured "
        "(parameters.md 7)"));
  }

  crane_model::ChamberPressure pushing;
  pushing.p_a_pa.setConstant(system_pressure_pa);
  pushing.p_b_pa.setZero();
  crane_model::ChamberPressure pulling;
  pulling.p_a_pa.setZero();
  pulling.p_b_pa.setConstant(system_pressure_pa);

  auto push = model.cylinder_force(pushing);
  if (!push.ok()) {
    return Result<Limits>::failure(push.status());
  }
  auto pull = model.cylinder_force(pulling);
  if (!pull.ok()) {
    return Result<Limits>::failure(pull.status());
  }

  Limits limits{};
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const double extend = std::abs(push.value()[static_cast<Eigen::Index>(row)]);
    const double retract = std::abs(pull.value()[static_cast<Eigen::Index>(row)]);
    const double smaller = std::min(extend, retract);
    if (!std::isfinite(smaller) || !(smaller > 0.0)) {
      return Result<Limits>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "actuated row " + std::to_string(row) +
          " has no usable chamber area, so no cylinder force limit follows from the relief "
          "pressure"));
    }
    limits[row] = smaller;
  }
  return Result<Limits>::success(limits);
}

double PeakDemand::worst() const noexcept
{
  return std::max(
    std::max(joint_velocity, joint_acceleration), std::max(cylinder_force, pump_flow));
}

crane_model::Result<TimingSolution> solve_timing_ocp(
  const crane_model::Model & model, const crane_model::ModelConfig & model_config,
  const GeometricPath & path, const JointLimits & limits, const TimingOcpRequest & request,
  const TimingOcpSettings & settings)
{
  if (path.segment_count() == 0U) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::InvalidArgument, "the path has no segments to time"));
  }
  {
    const Status status = check_margin_factor(request.speed_scale);
    if (!status.ok()) {
      return Result<TimingSolution>::failure(status);
    }
  }
  if (!(settings.kappa > 0.0) || settings.kappa > 1.0) {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "kappa = " + std::to_string(settings.kappa) +
        " is outside (0, 1]; trajectory_planning 5.5 makes it the fraction of the machine's "
        "authority the planner may spend, and a value above one spends authority the machine "
        "does not have"));
  }
  if (settings.intervals < 4U) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::InvalidArgument, "the OCP needs at least four shooting intervals"));
  }
  if (!(settings.sigma_rate_min > 0.0) || settings.sigma_rate_min >= settings.sigma_rate_max) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::InvalidArgument, "sigma_rate_min must be positive and below its ceiling"));
  }
  // The one setting the generated artifact caps rather than carries. acados
  // sizes the SQP's own statistics array from `nlp_solver_max_iter` at code
  // generation, so a solve asking for more iterations than the shipped solver
  // allocated would write past it. Everything else in `TimingOcpSettings` --
  // kappa, the boxes, the weights, the grid, the tolerances, the wall clock --
  // is written onto the solver below and needs no re-export.
  if (settings.max_iterations > CRANE_PLANNING_TIMING_MAX_ITERATIONS) {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "this solve asks for " + std::to_string(settings.max_iterations) +
        " SQP iterations and the shipped solver allocated its statistics for " +
        std::to_string(CRANE_PLANNING_TIMING_MAX_ITERATIONS) +
        ". Raising the cap is a re-export (`./scripts/export_timing_ocp.py`), not a setting"));
  }
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (!(settings.actuation.ddq_a_max[row] > 0.0) ||
      !(settings.actuation.cylinder_force_max[row] > 0.0))
    {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "actuated row " + std::to_string(row) +
          " has no finite positive acceleration or cylinder force limit. The force limit is the "
          "one trajectory_planning 3 records as new, so there is no earlier planner to inherit a "
          "default from and inventing one here would be the silent stub the model API contract "
          "exists to prevent"));
    }
  }
  if (!(settings.actuation.pump_flow_max > 0.0) ||
    !(settings.actuation.pump_flow_planning_factor > 0.0))
  {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::InvalidArgument, "Q_P^max and its planning factor must be positive"));
  }

  const Backend backend = backend_for(model_config.tool);
  const std::array<double, kPayloadDof> payload = payload_block(request.payload);

  const int intervals = static_cast<int>(settings.intervals);
  const double d_sigma = 1.0 / static_cast<double>(intervals);
  const ScaledLimits scaled = scale_limits(limits, settings, request.speed_scale);
  const int dof = static_cast<int>(crane_model::kActuatedDof);
  const int planned = static_cast<int>(kPathDof);

  // ------------------------------------------------------------------
  // The path, sampled onto the grid
  // ------------------------------------------------------------------
  std::vector<PathSample> nodes;
  std::vector<PathSample> midpoints;
  nodes.reserve(static_cast<std::size_t>(intervals) + 1U);
  midpoints.reserve(static_cast<std::size_t>(intervals));
  for (int index = 0; index <= intervals; ++index) {
    nodes.push_back(path.at(static_cast<double>(index) * d_sigma));
    if (index < intervals) {
      midpoints.push_back(path.at((static_cast<double>(index) + 0.5) * d_sigma));
    }
  }
  for (const PathSample & sample : nodes) {
    if (!sample.q_a.allFinite() || !sample.dq_a.allFinite() || !sample.ddq_a.allFinite()) {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::NonFiniteInput,
          "the path is not finite at sigma = " + std::to_string(sample.sigma)));
    }
  }

  // The passive pair the machine hangs at, at **every** node and not only at the
  // two ends. 5.4's terminal condition is `q_u(T) = q_eq` and the start is the
  // settled pose the stopped-start convention gives, so both ends are pinned
  // here -- but `mpc.md` 3 constraint 3 writes the sway box around `q_u^eq` and
  // that equilibrium moves with the arm. Slewing the tool out along the path
  // carries the hanging pose with it by far more than `q_u_max`, so a box drawn
  // once around the *goal* equilibrium excludes the machine's own resting
  // configuration over most of the path and the QP is infeasible before the
  // first step. One `Model::passive_equilibrium` per node, none of it guessed.
  std::vector<crane_model::QU> q_u_equilibrium;
  q_u_equilibrium.reserve(nodes.size());
  for (const PathSample & node : nodes) {
    auto equilibrium = model.passive_equilibrium(actuated(node, 0), request.payload);
    if (!equilibrium.ok()) {
      return Result<TimingSolution>::failure(equilibrium.status());
    }
    q_u_equilibrium.push_back(equilibrium.value());
  }
  const crane_model::QU q_u_goal = q_u_equilibrium.back();

  // 7's initial condition. The stopped start is the passive pair hanging at the
  // start configuration's own equilibrium and not moving; a measured start
  // replaces both halves with what `/crane/pendulum_state` says. Neither is
  // *assumed*: which one this is arrived on the request.
  const crane_model::QU q_u_start =
    request.start.measured ? request.start.q_u : q_u_equilibrium.front();
  const crane_model::DQU dq_u_start =
    request.start.measured ? request.start.dq_u : crane_model::DQU::Zero();
  if (!q_u_start.allFinite() || !dq_u_start.allFinite()) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::NonFiniteInput, "the measured passive start state is not finite"));
  }

  // A start outside the box the rest of the horizon is solved in is not a harder
  // problem, it is an infeasible one: `mpc.md` 3 constraint 3 draws the sway box
  // around the node's own equilibrium and stage 1 is one interval away, so a
  // measurement further out than the whole allowance cannot be steered back
  // inside it. Said here, by name, rather than left to come back as a QP failure.
  for (std::size_t row = 0; row < crane_model::kPassiveDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    const double offset = std::abs(q_u_start[axis] - q_u_equilibrium.front()[axis]);
    if (offset > settings.q_u_max[row]) {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the tool is measured " + std::to_string(offset) +
          " rad off its hanging pose on passive row " + std::to_string(row) +
          ", past the " + std::to_string(settings.q_u_max[row]) +
          " rad sway box mpc 3 constraint 3 permits and trajectory_planning 4.3 cleared the path "
          "over. There is no timing of this path that starts there, so the re-plan is refused "
          "rather than solved from a state the rest of the horizon may not enter"));
    }
    if (std::abs(dq_u_start[axis]) > settings.dq_u_max[row]) {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the tool is measured swinging at " + std::to_string(dq_u_start[axis]) +
          " rad/s on passive row " + std::to_string(row) + ", past the " +
          std::to_string(settings.dq_u_max[row]) +
          " rad/s of mpc 3 constraint 4. A plan cannot begin at a sway rate the controller "
          "tracking it is not allowed to hold"));
    }
  }
  if (request.start.sigma_rate_pinned &&
    (!std::isfinite(request.start.sigma_rate) || !(request.start.sigma_rate > 0.0)))
  {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the path rate that reproduces the measured actuated velocity is " +
        std::to_string(request.start.sigma_rate) +
        ", which is not a positive number; the objective divides by sigma_dot, and a rate at or "
        "below zero is the path being run backwards"));
  }

  // The **static** force, before any timing is chosen, at every node.
  /*
   * A cylinder holds the arm up at rest, and that part of constraint 6 does not
   * fall when the move is slowed down: it is the floor the warm start below
   * cannot bisect under. If gravity alone is outside the scaled limit at any
   * node then no timing exists, and this is where that is said.
   *
   * It is said *here*, and not left to the solver, because the solver's answer
   * to it is `ACADOS_QP_FAILURE` -- a refusal that is correct under `mpc.md` 5.3
   * requirement 1 and useless to whoever has to act on it. The usual cause is a
   * pose near a transmission zero, where `J_c,ii -> 0` and the cylinder has no
   * moment arm about its joint at all, so `|tau_i| <= J_c,ii F_i^max` cannot
   * hold at any force. `trajectory_planning` 4.2 says in passing that the arm
   * joint's range is far wider than its working range; this is that sentence
   * with a number on it, and it names the node, the axis and the transmission
   * so a caller can tell an unreachable pose from an unavailable solver.
   */
  std::vector<double> z;
  for (std::size_t at = 0; at < nodes.size(); ++at) {
    const crane_model::QA position = actuated(nodes[at], 0);
    if (!evaluate_output(
        backend, position, crane_model::QA::Zero(), crane_model::QA::Zero(),
        q_u_equilibrium[at], crane_model::DQU::Zero(), payload, z))
    {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::SymbolicBackendFailure,
          std::string("the output map of mpc 3 could not be evaluated at rest at sigma = ") +
          std::to_string(nodes[at].sigma)));
    }

    // The path coordinates only: constraint 6 has no tool row, so a tool cylinder
    // holding a closed gripper at rest is not a reason to refuse a path.
    for (std::size_t row = 0; row < kPathDof; ++row) {
      const double force = z[kOutputCylinderForce + row];
      if (std::isfinite(force) && std::abs(force) <= scaled.force_max[static_cast<int>(row)]) {
        continue;
      }
      // The transmission at the same pose, so the message can say *why* rather
      // than only that. `J_c,ii` near zero is the usual cause and is the one a
      // reader can act on: it is a pose off the working range, not a load.
      crane_model::Q q = crane_model::Q::Zero();
      for (std::size_t axis = 0; axis < crane_model::kActuatedDof; ++axis) {
        q[static_cast<Eigen::Index>(kActuatedRows[axis])] =
          position[static_cast<Eigen::Index>(axis)];
      }
      q[4] = q_u_equilibrium[at][0];
      q[5] = q_u_equilibrium[at][1];
      crane_model::ChamberPressure quiescent;
      quiescent.p_a_pa.setZero();
      quiescent.p_b_pa.setZero();
      std::string transmission_note;
      auto ratios = model.transmission(q, crane_model::DQA::Zero(), quiescent);
      if (ratios.ok()) {
        transmission_note = ". J_c," + std::to_string(row) + std::to_string(row) + " = " +
          std::to_string(
          ratios.value().joint_to_cylinder(
            static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(row))) +
          " there, so a torque about the joint costs that much cylinder force";
      }

      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the path cannot be held at rest, so no timing of it exists: at sigma = " +
          std::to_string(nodes[at].sigma) + " the cylinder of actuated row " +
          std::to_string(row) + " must carry " + std::to_string(force) +
          " N to hold the machine still, against the " +
          std::to_string(scaled.force_max[static_cast<int>(row)]) +
          " N this solve allows it (kappa = " + std::to_string(settings.kappa) + " of " +
          std::to_string(settings.actuation.cylinder_force_max[row]) + " N)" +
          transmission_note +
          ". A static force does not fall when a move is slowed down, so this is a "
          "refusal about the path and not about the solver: the usual cause is a pose near "
          "that axis's transmission zero, where J_c,ii vanishes and the cylinder has no "
          "moment arm about the joint at any force (trajectory_planning 4.2 -- the arm "
          "joint's range is far wider than its working range)"));
    }
  }

  // ------------------------------------------------------------------
  // The shipped solver, opened on this solve's grid
  // ------------------------------------------------------------------
  TimingSolver ocp{backend};
  if (!ocp.open(intervals, d_sigma)) {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::BackendUnavailable,
        std::string("the generated timing solver ") + backend.name +
        " would not open on " + std::to_string(intervals) + " shooting intervals"));
  }

  // The cost. Every weight and the grid's own `dsigma` are in `W` rather than in
  // the shipped residual, which is what keeps `sway_weight`, `input_weight` and
  // `intervals` runtime settings: acados' nonlinear least squares cost is
  // `0.5 ||y||^2_W`, so `W_00 = 2 dsigma` on `y_0 = 1/sqrt(sigma_dot)` squares to
  // 5.2's `int dsigma / sigma_dot`, and the two sway rows to
  // `w ||dq_u||^2 dsigma / sigma_dot`, which is its `int ||dq_u||^2 dt` because
  // `dt = dsigma / sigma_dot`.
  std::vector<double> weight(
    static_cast<std::size_t>(kResidualDof * kResidualDof), 0.0);
  const auto diagonal = [](std::vector<double> & matrix, int size, int row, double value) {
      matrix[static_cast<std::size_t>(row * size + row)] = value;
    };
  diagonal(weight, kResidualDof, CRANE_PLANNING_TIMING_RESIDUAL_TIME, 2.0 * d_sigma);
  for (int row = 0; row < static_cast<int>(crane_model::kPassiveDof); ++row) {
    diagonal(
      weight, kResidualDof, CRANE_PLANNING_TIMING_RESIDUAL_SWAY + row,
      2.0 * settings.sway_weight * d_sigma);
  }
  diagonal(
    weight, kResidualDof, CRANE_PLANNING_TIMING_RESIDUAL_INPUT,
    2.0 * settings.input_weight * d_sigma);
  std::vector<double> terminal_weight(
    static_cast<std::size_t>(kTerminalResidualDof * kTerminalResidualDof), 0.0);
  for (int row = 0; row < kTerminalResidualDof; ++row) {
    diagonal(terminal_weight, kTerminalResidualDof, row, 2.0 * settings.sway_weight);
  }
  std::vector<double> reference(static_cast<std::size_t>(kResidualDof), 0.0);

  // The bounds, in the units the shipped rows are written in: each row of `h` is
  // a physical quantity divided by its own conditioning constant, so the limit
  // goes in divided by the same one. The flow row is one-sided because `Q` is a
  // draw and not a signed force.
  std::vector<double> lower_h(static_cast<std::size_t>(kOcpConstraints), 0.0);
  std::vector<double> upper_h(static_cast<std::size_t>(kOcpConstraints), 0.0);
  for (int row = 0; row < planned; ++row) {
    const std::size_t acceleration = static_cast<std::size_t>(kConstraintAcceleration + row);
    const std::size_t force = static_cast<std::size_t>(kConstraintCylinderForce + row);
    upper_h[acceleration] = scaled.ddq_a_max[row] / kConstraintScale[acceleration];
    lower_h[acceleration] = -upper_h[acceleration];
    upper_h[force] = scaled.force_max[row] / kConstraintScale[force];
    lower_h[force] = -upper_h[force];
  }
  upper_h[static_cast<std::size_t>(kConstraintPumpFlow)] =
    scaled.flow_max / kConstraintScale[static_cast<std::size_t>(kConstraintPumpFlow)];
  lower_h[static_cast<std::size_t>(kConstraintPumpFlow)] = 0.0;

  std::vector<double> input_lower{-settings.sigma_accel_max};
  std::vector<double> input_upper{settings.sigma_accel_max};

  std::vector<std::vector<double>> parameters;
  parameters.reserve(static_cast<std::size_t>(intervals) + 1U);

  for (int stage = 0; stage <= intervals; ++stage) {
    // Parameters: the node block, then the block the integrator holds fixed over
    // the interval, then the payload. The last stage has no interval, so it
    // repeats its own node.
    std::vector<double> stage_parameters(static_cast<std::size_t>(kOcpParameters), 0.0);
    const PathSample & node = nodes[static_cast<std::size_t>(stage)];
    const PathSample & mid =
      midpoints[static_cast<std::size_t>(std::min(stage, intervals - 1))];
    for (int derivative = 0; derivative < 3; ++derivative) {
      const crane_model::QA node_value = actuated(node, derivative);
      const crane_model::QA mid_value = actuated(mid, derivative);
      const int offset = derivative * planned;
      for (int row = 0; row < planned; ++row) {
        stage_parameters[static_cast<std::size_t>(offset + row)] = node_value[row];
        stage_parameters[static_cast<std::size_t>(kPathBlock + offset + row)] = mid_value[row];
      }
    }
    stage_parameters[CRANE_PLANNING_TIMING_BLOCK_TOOL] = node.q8;
    stage_parameters[kPathBlock + CRANE_PLANNING_TIMING_BLOCK_TOOL] = mid.q8;
    for (std::size_t entry = 0; entry < kPayloadDof; ++entry) {
      stage_parameters[CRANE_PLANNING_TIMING_PARAMETER_PAYLOAD + entry] = payload[entry];
    }
    parameters.push_back(std::move(stage_parameters));
    if (ocp.backend().update_params(
        ocp.capsule(), stage, parameters.back().data(), kOcpParameters) != 0)
    {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::BackendUnavailable,
          "the generated solver refused the stage parameters at stage " + std::to_string(stage)));
    }

    if (stage < intervals) {
      ocp_nlp_cost_model_set(ocp.config(), ocp.dims(), ocp.in(), stage, "W", weight.data());
      ocp_nlp_cost_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "yref", reference.data());
      ocp_nlp_constraints_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "lbu", input_lower.data());
      ocp_nlp_constraints_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "ubu", input_upper.data());
      ocp_nlp_constraints_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "lh", lower_h.data());
      ocp_nlp_constraints_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "uh", upper_h.data());
    } else {
      ocp_nlp_cost_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "W", terminal_weight.data());
      ocp_nlp_cost_model_set(
        ocp.config(), ocp.dims(), ocp.in(), stage, "yref", reference.data());
    }
  }

  // State boxes. The path rate carries the joint velocity limit of the 3 table:
  // `|q_a'(sigma_k)| sigma_dot <= kappa speed_scale dq_a^max` is one bound per
  // stage on one state, which is what 5.1's first row already says it is.
  std::vector<double> rate_ceilings(static_cast<std::size_t>(intervals) + 1U, 0.0);
  for (int stage = 0; stage <= intervals; ++stage) {
    double ceiling = settings.sigma_rate_max;
    const crane_model::QA rate_row = actuated(nodes[static_cast<std::size_t>(stage)], 1);
    for (int row = 0; row < dof; ++row) {
      const double slope = std::abs(rate_row[row]);
      if (slope > 1.0e-9) {
        ceiling = std::min(ceiling, scaled.dq_a_max[row] / slope);
      }
    }
    rate_ceilings[static_cast<std::size_t>(stage)] =
      std::max(ceiling, 2.0 * settings.sigma_rate_min);
  }

  // The measured start rate against the same ceiling, before the solver sees it.
  // Because `q_a'(0) sigma_dot(0)` **is** the measured joint velocity, this row
  // says one thing in the machine's own terms: the arm is already moving faster
  // than kappa and speed_scale leave a plan allowed to move it. No timing of this
  // path begins there, and a QP failure is a worse way to be told so.
  if (request.start.sigma_rate_pinned && request.start.sigma_rate > rate_ceilings.front()) {
    const crane_model::QA slope = actuated(nodes.front(), 1);
    std::string worst;
    for (int row = 0; row < dof; ++row) {
      const double speed = std::abs(slope[row]) * request.start.sigma_rate;
      if (speed > scaled.dq_a_max[row]) {
        worst += worst.empty() ? "" : ", ";
        worst += "actuated row " + std::to_string(row) + " at " + std::to_string(speed) +
          " against " + std::to_string(scaled.dq_a_max[row]);
      }
    }
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the machine is measured moving faster than this plan is allowed to move it, so there is "
        "no timing that starts where it is: " + worst +
        " rad/s or m/s, which is kappa = " + std::to_string(settings.kappa) +
        " times speed_scale = " + std::to_string(request.speed_scale) +
        " of the description's own limit. trajectory_planning 5.5 reserves that margin for the "
        "MPC and a plan may not spend it merely by having been asked for late"));
  }

  // **Every** state row is boxed at every stage, stage 0 included, and by
  // `lbx`/`ubx` **alone**: the export deliberately does not set `x0`, because
  // declaring the stage-0 rows to acados as equalities through `idxbxe_0` turns
  // every converging solve in this file into `ACADOS_QP_FAILURE` on the first QP
  // -- HPIPM status 3, a NaN in the solution at QP iteration 4. Removing the
  // declaration and changing nothing else makes all of them converge again.
  //
  // Two neighbouring explanations were measured and are wrong, so that the next
  // reader does not spend the time again: the terminal stage's five coincident
  // rows are **not** the trouble (widening them changes nothing), and neither is
  // boxing `dq_u[1]`, whose box can be opened to 1e3 with the failure unchanged.
  std::vector<std::vector<double>> state_lower;
  std::vector<std::vector<double>> state_upper;
  state_lower.reserve(static_cast<std::size_t>(intervals) + 1U);
  state_upper.reserve(static_cast<std::size_t>(intervals) + 1U);
  for (int stage = 0; stage <= intervals; ++stage) {
    const double rate_ceiling = rate_ceilings[static_cast<std::size_t>(stage)];

    std::vector<double> lower;
    std::vector<double> upper;
    if (stage == 0) {
      // The initial condition of 7, in full. The passive pair and its rate are
      // pinned at what was measured -- or, for a stopped start, at the hanging
      // pose and zero, which is the convention 7 names and this is where it is
      // now a *choice* rather than the only option.
      //
      // `sigma_dot(0)` carries the measurement only when the measurement asked
      // for it. A stopped start leaves it in the same box every running stage
      // carries, for the same reason the terminal stage does: the objective
      // divides by it, so a rate the QP may take through zero -- or negative,
      // which is the path run backwards -- makes `1/sigma_dot` singular on the
      // first step, and the machine still starts at rest whatever the box says
      // because the path meets its start with `q_a'(0) = 0`. A **measured** start
      // is the case where that last sentence is false: `q_a'(0)` is the measured
      // direction, so `dq_a = q_a'(0) sigma_dot` makes the rate the row that
      // carries the seam, and it is held to the measurement rather than left to
      // the box.
      //
      // **Every measured row stands off its measurement by `start_resolution`
      // rather than sitting on it**, and that width is a solver number before it
      // is a measurement one -- see `StartResolution`, which carries the sweep it
      // was measured by. A zero-width box is where an interior-point method is
      // singular, and pinning stage 0 exactly is what turns five of this
      // package's solves into `ACADOS_QP_FAILURE` on the first QP.
      const double rate_window = request.start.sigma_rate_pinned ?
        std::abs(request.start.sigma_rate) *
        settings.start_resolution.sigma_rate_fraction : 0.0;
      const double lower_rate = request.start.sigma_rate_pinned ?
        std::max(settings.sigma_rate_min, request.start.sigma_rate - rate_window) :
        settings.sigma_rate_min;
      const double upper_rate = request.start.sigma_rate_pinned ?
        std::min(rate_ceiling, request.start.sigma_rate + rate_window) : rate_ceiling;
      lower = {
        0.0, lower_rate,
        q_u_start[0] - settings.start_resolution.q_u,
        q_u_start[1] - settings.start_resolution.q_u,
        dq_u_start[0] - settings.start_resolution.dq_u,
        dq_u_start[1] - settings.start_resolution.dq_u};
      upper = {
        0.0, upper_rate,
        q_u_start[0] + settings.start_resolution.q_u,
        q_u_start[1] + settings.start_resolution.q_u,
        dq_u_start[0] + settings.start_resolution.dq_u,
        dq_u_start[1] + settings.start_resolution.dq_u};
    } else if (stage == intervals) {
      // 5.4, in full: the path is finished, the tool hangs at its equilibrium and
      // it is not moving. `sigma_dot(T)` is the one row 5.4 asks for that this
      // formulation cannot pin at zero -- the objective divides by it -- and it
      // does not need to be pinned, because the path meets its end with
      // `q_a'(1) = q_a''(1) = 0` and the joints are therefore at rest whatever
      // the path rate is. What the machine does is asserted, not the coordinate.
      lower = {1.0, settings.sigma_rate_min, q_u_goal[0], q_u_goal[1], 0.0, 0.0};
      upper = {1.0, rate_ceiling, q_u_goal[0], q_u_goal[1], 0.0, 0.0};
    } else {
      const crane_model::QU & rest = q_u_equilibrium[static_cast<std::size_t>(stage)];
      lower = {
        0.0, settings.sigma_rate_min, rest[0] - settings.q_u_max[0],
        rest[1] - settings.q_u_max[1], -settings.dq_u_max[0], -settings.dq_u_max[1]};
      upper = {
        1.0, rate_ceiling, rest[0] + settings.q_u_max[0],
        rest[1] + settings.q_u_max[1], settings.dq_u_max[0], settings.dq_u_max[1]};
    }
    state_lower.push_back(std::move(lower));
    state_upper.push_back(std::move(upper));
    ocp_nlp_constraints_model_set(
      ocp.config(), ocp.dims(), ocp.in(), stage, "lbx", state_lower.back().data());
    ocp_nlp_constraints_model_set(
      ocp.config(), ocp.dims(), ocp.in(), stage, "ubx", state_upper.back().data());
  }

  // The solver options the artifact ships as defaults and this solve owns.
  // acados re-reads every one of them while it iterates, so none of them is a
  // property of the generated code -- only `max_iter`'s ceiling is, and that was
  // checked above.
  int max_iterations = settings.max_iterations;
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "max_iter", &max_iterations);
  double budget = settings.max_wall_clock;
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "timeout_max_time", &budget);
  double regularisation = settings.levenberg_marquardt;
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "levenberg_marquardt", &regularisation);
  double tolerance_stationarity = settings.tolerance_stationarity;
  double tolerance_feasibility = settings.tolerance_feasibility;
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "tol_stat", &tolerance_stationarity);
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "tol_eq", &tolerance_feasibility);
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "tol_ineq", &tolerance_feasibility);
  ocp_nlp_solver_opts_set(ocp.config(), ocp.opts(), "tol_comp", &tolerance_feasibility);

  // A guess that already satisfies the boundary conditions and every box: the
  // path rate on the classical velocity-limit curve of 5.1 -- the largest rate at
  // which the path's own curvature alone, with `sigma_ddot = 0`, still respects
  // the acceleration limit -- and the sway at the node's own hanging pose, not
  // moving. Starting at the velocity limit instead would put the arm's
  // acceleration several times over its bound at the path's tightest bends, and
  // the first QP is then asked to repair a violation it has no feasible step for.
  std::vector<double> rate_guesses(static_cast<std::size_t>(intervals) + 1U, 0.0);
  for (int stage = 0; stage <= intervals; ++stage) {
    const crane_model::QA curvature_row = actuated(nodes[static_cast<std::size_t>(stage)], 2);
    double rate = rate_ceilings[static_cast<std::size_t>(stage)];
    for (int row = 0; row < dof; ++row) {
      const double curvature = std::abs(curvature_row[row]);
      if (curvature > 1.0e-9) {
        rate = std::min(rate, std::sqrt(scaled.ddq_a_max[row] / curvature));
      }
    }
    rate_guesses[static_cast<std::size_t>(stage)] =
      std::max(rate, 2.0 * settings.sigma_rate_min);
  }

  // ...and then pulled down until the *force* and *flow* rows are inside their
  // bounds too, which the curve above says nothing about.
  //
  // This is the difference between a first QP that solves and `ACADOS_QP_FAILURE`
  // on SQP iteration 1. The velocity-limit curve is feasible for constraints 1
  // and 2 of the 3 table by construction, but `F_cyl` and `Q` are evaluated
  // through the transmission and can be far outside constraints 6 and 7 at a
  // rate that respects the other two -- most sharply on the structured
  // primitive's lift phase, where the arm carries the load against gravity. The
  // QP is then handed a violation of a nonlinear row it has no feasible step
  // for, and it reports a solver failure rather than an infeasible problem.
  //
  // Both rows fall monotonically as the rate falls -- flow is linear in the
  // piston velocity and the dynamic part of the force is quadratic in it -- so
  // the largest feasible rate can be found by bisection and is worth finding.
  // What does *not* fall with the rate is the **static** force: at rest the
  // cylinder still holds the arm up. That case has already been refused above,
  // by name, so the floor here is known to be feasible and the bracket below is
  // known to close.
  //
  // **Bisecting rather than halving is what makes this converge in a service
  // call.** Halving stops at the first rate that fits, which is up to a factor
  // two under the largest one that does, and the guess then starts an order of
  // magnitude of traversal time away from the optimum in the worst nodes. The
  // cost is `1/sigma_dot`, whose Gauss-Newton curvature falls as the rate rises,
  // so climbing back up from a guess that is too slow is exactly the direction
  // the merit line search is slowest in: on the structured primitive that was
  // 117 SQP iterations and eight seconds. Twelve bisection steps place the guess
  // within 0.03% of the target below and cost twelve evaluations of an
  // expression the solver evaluates thousands of times.
  //
  // **The target is a fraction of the allowance and not the allowance itself.**
  // Bisecting against 1.0 places the guess *on* the boundary of constraints 6
  // and 7 -- and at almost every node at once, because the largest feasible rate
  // is what makes some row bind by definition. The QP underneath acados is an
  // interior-point method: a point with zero slack on a dozen rows per stage is
  // exactly where its barrier terms and its Riccati factorisation break down,
  // and what comes back is `ACADOS_QP_FAILURE` on the first QP with a NaN inside
  // it rather than an honest infeasibility. Backing the target off to 0.9 leaves
  // every row strictly interior at the cost of a tenth of the guess's speed,
  // which the SQP then recovers in a handful of iterations. This is what the
  // sampled fallback's path needs and the structured primitive did not: the
  // refit of 4.5 leaves curvatures an order of magnitude larger, so far more
  // nodes are pinned against a row at once.
  constexpr double kGuessInteriorTarget = 0.9;
  const double floor_rate = 2.0 * settings.sigma_rate_min;
  for (int stage = 0; stage <= intervals; ++stage) {
    const std::size_t at = static_cast<std::size_t>(stage);
    const crane_model::QA position = actuated(nodes[at], 0);
    const crane_model::QA slope = actuated(nodes[at], 1);
    const crane_model::QA curvature = actuated(nodes[at], 2);

    // How far outside constraints 6 and 7 this node is at one candidate rate,
    // as a fraction of the allowance. At or below one is feasible.
    const auto demand = [&](double rate) {
        crane_model::QA velocity = crane_model::QA::Zero();
        crane_model::QA acceleration = crane_model::QA::Zero();
        for (int row = 0; row < dof; ++row) {
          velocity[row] = slope[row] * rate;
          acceleration[row] = curvature[row] * rate * rate;
        }
        std::vector<double> outputs;
        if (!evaluate_output(
            backend, position, velocity, acceleration, q_u_equilibrium[at],
            crane_model::DQU::Zero(), payload, outputs))
        {
          return std::numeric_limits<double>::infinity();
        }
        double worst = 0.0;
        double flow = 0.0;
        // The rows the constraint vector carries, and no others: bisecting
        // against a tool row the solve cannot relieve would lower every guess
        // for a demand no timing can change.
        for (int row = 0; row < planned; ++row) {
          const double force = outputs[kOutputCylinderForce + static_cast<std::size_t>(row)];
          worst = std::max(
            worst, std::abs(force) / scaled.force_max[static_cast<std::size_t>(row)]);
          flow += outputs[kOutputAxisFlow + static_cast<std::size_t>(row)];
        }
        worst = std::max(worst, std::abs(flow) / scaled.flow_max);
        return worst;
      };

    double feasible = floor_rate;
    double infeasible = std::max(rate_guesses[at], floor_rate);
    if (!(demand(infeasible) > kGuessInteriorTarget)) {
      continue;  // the acceleration curve is already interior; nothing to do
    }
    for (int step = 0; step < 12; ++step) {
      const double middle = 0.5 * (feasible + infeasible);
      if (demand(middle) > kGuessInteriorTarget) {
        infeasible = middle;
      } else {
        feasible = middle;
      }
    }
    rate_guesses[at] = feasible;
  }

  // ...and then made *reachable*, which per-node feasibility does not make it.
  /*
   * Each rate above is the largest one node can be crossed at on its own. Where
   * the path is stationary -- the two endpoints, where `q_a'` and `q_a''` both
   * vanish -- nothing limits it at all and it comes back at `sigma_rate_max`,
   * beside neighbours an order of magnitude slower. A guess that steps by a
   * factor of twenty over one interval is not a trajectory of the dynamics it is
   * a guess for: the shooting gap at that stage is enormous, and the first QP
   * spends its step closing that rather than descending the objective.
   *
   * The classical forward-backward pass of 5.1 is what turns a per-node speed
   * limit into a profile the input can actually produce. In sigma it is one line
   * -- `d(sigma_dot^2)/dsigma = 2 sigma_ddot` -- so a sweep each way, taking the
   * smaller of what the node allows and what the neighbour can be accelerated
   * from, is enough. It only ever lowers a rate, so everything the bisection
   * above established stays true.
   */
  const double rate_step = 2.0 * settings.sigma_accel_max * d_sigma;

  // A measured start's `sigma_dot(0)` enters **before** the sweeps and not after
  // them, and that ordering is the whole of it. It is not a guess -- the box below
  // pins it -- so the forward sweep has to propagate it the way it propagates
  // every other node's limit. Setting it afterwards instead leaves stage 1 holding
  // whatever the *unpinned* stage 0 allowed, which on the structured primitive is
  // `sigma_rate_max`: the guess then opens with a step from 0.39 to 2.0 across one
  // interval that no bounded `sigma_ddot` can produce, the first shooting gap is
  // enormous, and the first QP comes back `ACADOS_QP_FAILURE` with a NaN in it --
  // exactly the failure the paragraph above says this pass exists to prevent, and
  // the reason a re-plan from a moving arm could not be solved at all.
  if (request.start.sigma_rate_pinned) {
    rate_guesses.front() = request.start.sigma_rate;
  }
  for (int stage = 1; stage <= intervals; ++stage) {
    const std::size_t at = static_cast<std::size_t>(stage);
    rate_guesses[at] = std::min(
      rate_guesses[at], std::sqrt(rate_guesses[at - 1U] * rate_guesses[at - 1U] + rate_step));
  }
  // The backward sweep stops one short of a pinned start. It asks whether stage 0
  // can decelerate into stage 1, and when the rate is measured that is a question
  // about the input rather than about the guess -- the solver's to answer.
  const int backward_last = request.start.sigma_rate_pinned ? 1 : 0;
  for (int stage = intervals - 1; stage >= backward_last; --stage) {
    const std::size_t at = static_cast<std::size_t>(stage);
    rate_guesses[at] = std::min(
      rate_guesses[at], std::sqrt(rate_guesses[at + 1U] * rate_guesses[at + 1U] + rate_step));
  }
  for (double & rate : rate_guesses) {
    rate = std::max(rate, floor_rate);
  }
  // Re-asserted after the floor, which is the last thing that could still move it:
  // a measured rate below `floor_rate` is a slow start, not a guess to be raised,
  // and a warm start that opens anywhere but on the pin opens outside its own box.
  if (request.start.sigma_rate_pinned) {
    rate_guesses.front() = request.start.sigma_rate;
  }
  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> guess(static_cast<std::size_t>(kOcpStateDof), 0.0);
    guess[0] = static_cast<double>(stage) * d_sigma;
    guess[1] = rate_guesses[static_cast<std::size_t>(stage)];
    guess[2] = q_u_equilibrium[static_cast<std::size_t>(stage)][0];
    guess[3] = q_u_equilibrium[static_cast<std::size_t>(stage)][1];
    if (stage == 0) {
      // ...and the passive half of the same boundary condition, so the first
      // shooting gap is the dynamics' and not the guess's.
      guess[2] = q_u_start[0];
      guess[3] = q_u_start[1];
      guess[4] = dq_u_start[0];
      guess[5] = dq_u_start[1];
    }
    ocp_nlp_out_set(ocp.config(), ocp.dims(), ocp.out(), stage, "x", guess.data());
    if (stage < intervals) {
      double input = 0.0;
      ocp_nlp_out_set(ocp.config(), ocp.dims(), ocp.out(), stage, "u", &input);
    }
    // acados allocates `nlp_out` with `malloc` and does not zero it, so the
    // multipliers start as whatever was on the heap -- and the first
    // stationarity residual is then a NaN before any step has been taken. The
    // examples get away with it because they leave far less of the vector
    // unused; a horizon with eleven nonlinear rows per stage does not.
    std::vector<double> zeros(64U, 0.0);
    ocp_nlp_out_set(ocp.config(), ocp.dims(), ocp.out(), stage, "lam", zeros.data());
    if (stage < intervals) {
      ocp_nlp_out_set(ocp.config(), ocp.dims(), ocp.out(), stage, "pi", zeros.data());
    }
  }

  const int status = ocp.backend().solve(ocp.capsule());

  int iterations = 0;
  double solve_time = 0.0;
  ocp_nlp_get(ocp.solver(), "sqp_iter", &iterations);
  ocp_nlp_get(ocp.solver(), "time_tot", &solve_time);

  if (status != ACADOS_SUCCESS) {
    // mpc.md 5.3 requirement 1, and 7's "do not emit a partial plan": what comes
    // back is the solver's own word and no trajectory at all.
    return Result<TimingSolution>::failure(
      failure(
        status == ACADOS_TIMEOUT ? ErrorCode::NotReady : ErrorCode::BackendUnavailable,
        "the path-constrained OCP of trajectory_planning 5.2 did not converge: " +
        acados_status_word(status) + " after " + std::to_string(iterations) +
        " SQP iterations and " + std::to_string(solve_time) + " s of a " +
        std::to_string(settings.max_wall_clock) + " s budget"));
  }

  // ------------------------------------------------------------------
  // The solution, back on a time base
  // ------------------------------------------------------------------
  std::vector<double> sigma_rate(static_cast<std::size_t>(intervals) + 1U, 0.0);
  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> state(static_cast<std::size_t>(kOcpStateDof), 0.0);
    ocp_nlp_out_get(ocp.config(), ocp.dims(), ocp.out(), stage, "x", state.data());
    if (!std::isfinite(state[1]) || !(state[1] > 0.0)) {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::NonFiniteInput,
          "acados returned a non-positive path rate at stage " + std::to_string(stage)));
    }
    sigma_rate[static_cast<std::size_t>(stage)] = state[1];
  }

  // `dt = dsigma / sigma_dot`, trapezoidal over the same grid the OCP used. The
  // arrival time is therefore the objective the solver minimised, evaluated the
  // same way.
  std::vector<double> node_time(static_cast<std::size_t>(intervals) + 1U, 0.0);
  for (int stage = 0; stage < intervals; ++stage) {
    const double left = 1.0 / sigma_rate[static_cast<std::size_t>(stage)];
    const double right = 1.0 / sigma_rate[static_cast<std::size_t>(stage) + 1U];
    node_time[static_cast<std::size_t>(stage) + 1U] =
      node_time[static_cast<std::size_t>(stage)] + 0.5 * d_sigma * (left + right);
  }
  const double duration = node_time.back();
  if (!std::isfinite(duration) || !(duration > 0.0)) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::NonFiniteInput, "the solved traversal time is not a positive number"));
  }

  TimingSolution solution;
  solution.kappa = settings.kappa;
  solution.speed_scale = request.speed_scale;
  solution.iterations = iterations;
  solution.solve_time = solve_time;
  solution.solver_status = acados_status_word(status);

  const std::size_t samples = std::max<std::size_t>(
    1U, static_cast<std::size_t>(std::ceil(duration / settings.sample_period)));
  TimedTrajectory & trajectory = solution.trajectory;
  trajectory.duration = duration;
  trajectory.time_from_start.reserve(samples + 1U);
  trajectory.q_a_ref.reserve(samples + 1U);
  trajectory.dq_a_ref.reserve(samples + 1U);

  double limiting = 0.0;
  std::size_t cursor = 0;
  for (std::size_t index = 0; index <= samples; ++index) {
    const double time =
      std::min(duration, static_cast<double>(index) * settings.sample_period);
    while (cursor + 1U < node_time.size() && node_time[cursor + 1U] < time) {
      ++cursor;
    }
    const double span = node_time[cursor + 1U] - node_time[cursor];
    const double fraction = span > 0.0 ? (time - node_time[cursor]) / span : 0.0;
    const double sigma =
      (static_cast<double>(cursor) + std::min(1.0, std::max(0.0, fraction))) * d_sigma;
    const double rate = sigma_rate[cursor] +
      (sigma_rate[cursor + 1U] - sigma_rate[cursor]) * std::min(1.0, std::max(0.0, fraction));

    const PathSample sample = path.at(std::min(1.0, sigma));
    const crane_model::QA position = actuated(sample, 0);
    const crane_model::QA slope = actuated(sample, 1);
    crane_model::DQA velocity = crane_model::DQA::Zero();
    for (int row = 0; row < dof; ++row) {
      velocity[row] = slope[row] * rate;
      const double bound = settings.kappa * request.speed_scale * limits.axis[
        static_cast<std::size_t>(row)].dq_max;
      limiting = std::max(limiting, std::abs(velocity[row]) / bound);
    }
    trajectory.time_from_start.push_back(time);
    trajectory.q_a_ref.push_back(position);
    trajectory.dq_a_ref.push_back(velocity);
  }
  trajectory.limiting_fraction = limiting;

  // ------------------------------------------------------------------
  // What the answer actually demands of the machine
  // ------------------------------------------------------------------
  solution.nodes.reserve(static_cast<std::size_t>(intervals) + 1U);
  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> state(static_cast<std::size_t>(kOcpStateDof), 0.0);
    ocp_nlp_out_get(ocp.config(), ocp.dims(), ocp.out(), stage, "x", state.data());
    double input = 0.0;
    if (stage < intervals) {
      ocp_nlp_out_get(ocp.config(), ocp.dims(), ocp.out(), stage, "u", &input);
    }
    const PathSample & node = nodes[static_cast<std::size_t>(stage)];
    const crane_model::QA position = actuated(node, 0);
    const crane_model::QA slope = actuated(node, 1);
    const crane_model::QA curvature = actuated(node, 2);

    // All six rows go into the model, because the tool link is pinned and not
    // removed; only the five the OCP constrained are reported as demand, because
    // `PeakDemand` is a fraction of a limit this solve was actually held to.
    crane_model::QA velocity = crane_model::QA::Zero();
    crane_model::QA acceleration = crane_model::QA::Zero();
    for (int row = 0; row < dof; ++row) {
      velocity[row] = slope[row] * state[1];
      acceleration[row] = curvature[row] * state[1] * state[1] + slope[row] * input;
      if (row >= planned) {
        continue;
      }
      solution.peak_demand.joint_acceleration = std::max(
        solution.peak_demand.joint_acceleration,
        std::abs(acceleration[row]) /
        settings.actuation.ddq_a_max[static_cast<std::size_t>(row)]);
      solution.peak_demand.joint_velocity = std::max(
        solution.peak_demand.joint_velocity,
        std::abs(velocity[row]) / limits.axis[static_cast<std::size_t>(row)].dq_max);
    }

    const crane_model::QU q_u(state[2], state[3]);
    const crane_model::DQU dq_u(state[4], state[5]);
    if (!evaluate_output(backend, position, velocity, acceleration, q_u, dq_u, payload, z)) {
      return Result<TimingSolution>::failure(
        failure(
          ErrorCode::SymbolicBackendFailure,
          "the output map of mpc 3 could not be evaluated at the solved node " +
          std::to_string(stage)));
    }

    OcpNode record;
    record.sigma = static_cast<double>(stage) * d_sigma;
    record.sigma_rate = state[1];
    record.q_a = position;
    record.q_u = q_u;
    record.dq_u = dq_u;
    record.q_u_equilibrium = q_u_equilibrium[static_cast<std::size_t>(stage)];
    double flow = 0.0;
    for (int row = 0; row < dof; ++row) {
      const double force = z[kOutputCylinderForce + static_cast<std::size_t>(row)];
      // The record carries all six -- the tool cylinder is still in the model and
      // what it holds is worth reading -- while the demand and constraint 7's sum
      // carry the five the constraint vector does.
      record.dq_a[row] = velocity[row];
      record.ddq_a[row] = acceleration[row];
      record.cylinder_force[row] = force;
      if (row >= planned) {
        continue;
      }
      solution.peak_demand.cylinder_force = std::max(
        solution.peak_demand.cylinder_force,
        std::abs(force) /
        settings.actuation.cylinder_force_max[static_cast<std::size_t>(row)]);
      flow += z[kOutputAxisFlow + static_cast<std::size_t>(row)];
    }
    record.pump_flow = flow;
    solution.nodes.push_back(record);
    solution.peak_demand.pump_flow = std::max(
      solution.peak_demand.pump_flow,
      flow / (settings.actuation.pump_flow_planning_factor * settings.actuation.pump_flow_max));
  }

  return Result<TimingSolution>::success(std::move(solution));
}

}  // namespace crane_planning
