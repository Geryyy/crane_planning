#include "crane_planning/timing_ocp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <casadi/casadi.hpp>

#include "crane_model/symbolic/casadi_graph.hpp"

#include "acados_casadi_bridge.hpp"

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
}

namespace crane_planning
{
namespace
{

using casadi::SX;
using crane_model::ErrorCode;
using crane_model::Result;
using crane_model::Status;

constexpr int kOcpStateDof = 6;   ///< (sigma, sigma_dot, q_u, dq_u), 5.2
constexpr int kOcpInputDof = 1;   ///< sigma_ddot
constexpr int kOcpConstraints = 13;  ///< 6 acceleration, 6 cylinder force, 1 pump flow

/// One block of path data: q_a(sigma), q_a'(sigma), q_a''(sigma), all six rows.
constexpr int kPathBlock = 3 * static_cast<int>(crane_model::kActuatedDof);

/// Two blocks: the node, which the cost and the constraints are written at, and
/// the interval midpoint, which the integrator holds the path fixed at.
constexpr int kOcpParameters = 2 * kPathBlock;

constexpr int kResidualDof = 4;      ///< traversal time, two sway rows, the input
constexpr int kTerminalResidualDof = 2;

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

/// The full actuated row of a path sample: the five path coordinates, then q8.
crane_model::QA actuated(const PathSample & sample, int derivative)
{
  crane_model::QA value = crane_model::QA::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    const Eigen::Index axis = static_cast<Eigen::Index>(row);
    value[axis] = derivative == 0 ? sample.q_a[axis] :
      (derivative == 1 ? sample.dq_a[axis] : sample.ddq_a[axis]);
  }
  const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
  value[tool] = derivative == 0 ? sample.q8 : (derivative == 1 ? sample.dq8 : sample.ddq8);
  return value;
}

/// The 16-state contract 2 orders as `[q_a, q_u, dq_a, dq_u]`, symbolically.
SX full_state(const SX & q_a, const SX & q_u, const SX & dq_a, const SX & dq_u)
{
  return SX::vertcat({q_a, q_u, dq_a, dq_u});
}

/// Everything the expressions below are written in terms of, unpacked once.
struct Symbols
{
  SX x;        ///< the OCP state, 6
  SX u;        ///< the OCP input, 1
  SX p;        ///< the parameter vector, kOcpParameters
  SX sigma_rate;
  SX q_u;
  SX dq_u;

  SX q_a_node;
  SX dq_a_node;    ///< q_a'(sigma_k)
  SX ddq_a_node;   ///< q_a''(sigma_k)
  SX q_a_mid;
  SX dq_a_mid;
  SX ddq_a_mid;
};

Symbols make_symbols()
{
  Symbols symbols;
  symbols.x = SX::sym("x", kOcpStateDof);
  symbols.u = SX::sym("u", kOcpInputDof);
  symbols.p = SX::sym("p", kOcpParameters);
  symbols.sigma_rate = symbols.x(1);
  symbols.q_u = symbols.x(casadi::Slice(2, 4));
  symbols.dq_u = symbols.x(casadi::Slice(4, 6));

  const int dof = static_cast<int>(crane_model::kActuatedDof);
  symbols.q_a_node = symbols.p(casadi::Slice(0, dof));
  symbols.dq_a_node = symbols.p(casadi::Slice(dof, 2 * dof));
  symbols.ddq_a_node = symbols.p(casadi::Slice(2 * dof, 3 * dof));
  symbols.q_a_mid = symbols.p(casadi::Slice(kPathBlock, kPathBlock + dof));
  symbols.dq_a_mid = symbols.p(casadi::Slice(kPathBlock + dof, kPathBlock + 2 * dof));
  symbols.ddq_a_mid = symbols.p(casadi::Slice(kPathBlock + 2 * dof, kPathBlock + 3 * dof));
  return symbols;
}

/// 5's chain rule: `ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot`.
SX joint_acceleration(const SX & dq_a_path, const SX & ddq_a_path, const SX & rate, const SX & input)
{
  return ddq_a_path * (rate * rate) + dq_a_path * input;
}

/// 5.2's right-hand side, divided through by `sigma_dot` because sigma is the
/// independent variable. The passive block is the graph's, not a second copy.
SX path_dynamics(
  const crane_model::symbolic::CasadiGraph & graph, const Symbols & symbols)
{
  const SX ddq_a = joint_acceleration(
    symbols.dq_a_mid, symbols.ddq_a_mid, symbols.sigma_rate, symbols.u);
  const SX state16 = full_state(
    symbols.q_a_mid, symbols.q_u, symbols.dq_a_mid * symbols.sigma_rate, symbols.dq_u);

  const std::vector<SX> rows = graph.passive_rows(std::vector<SX>{state16});
  const SX & mass_uu = rows[0];
  const SX & mass_ua = rows[1];
  const SX & bias_u = rows[2];

  // 2x2 by hand, exactly as `crane_model/src/symbolic_graph.cpp` does it: a
  // singular `M_uu` is then a non-finite expression rather than a branch the
  // solver would have to take.
  const SX right = SX::mtimes(mass_ua, ddq_a) + bias_u;
  const SX determinant = mass_uu(0, 0) * mass_uu(1, 1) - mass_uu(0, 1) * mass_uu(1, 0);
  const SX ddq_u = SX::vertcat(
    {-(mass_uu(1, 1) * right(0) - mass_uu(0, 1) * right(1)) / determinant,
      -(mass_uu(0, 0) * right(1) - mass_uu(1, 0) * right(0)) / determinant});

  // d/dsigma of 5.2's dx/dt, i.e. every time derivative over sigma_dot. The
  // first row is `dsigma/dsigma = 1` and it is what makes 5.4's `sigma(T) = 1`
  // a terminal condition on a state rather than a property of the grid.
  return SX::vertcat(
    {SX(1.0), symbols.u / symbols.sigma_rate, symbols.dq_u / symbols.sigma_rate,
      ddq_u / symbols.sigma_rate});
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

/// 5.3 and the 3 guarantee table, as one vector of nonlinear constraints.
/**
 * Joint *position* is absent on purpose and is not dropped: it is a property of
 * the geometry, which `fit_c2_path` has already refused a path for, and sigma is
 * pinned to the grid here, so nothing the OCP decides can move it. Joint
 * *velocity* is absent because it is `|q_a'(sigma_k)| sigma_dot` with a known
 * `q_a'`, i.e. a per-stage bound on one state and cheaper as a box than as a
 * nonlinear row.
 *
 * **Every row is divided by its own scaled limit**, so the vector is a set of
 * *fractions of the allowance* and the box is plain `[-1, 1]`. That is not
 * cosmetic. Written in physical units the thirteen rows span nine decades --
 * radians per second squared near one, newtons near `1e5`, cubic metres per
 * second near `1e-3` -- and the QP's constraint Jacobian inherits the spread.
 * HPIPM fails on it: `ACADOS_QP_FAILURE` on the first step, for a problem whose
 * functions are all finite and whose solution exists. Normalised, the same
 * problem converges.
 */
SX constraint_vector(
  const crane_model::symbolic::CasadiGraph & graph, const Symbols & symbols,
  const ScaledLimits & scaled)
{
  const SX ddq_a = joint_acceleration(
    symbols.dq_a_node, symbols.ddq_a_node, symbols.sigma_rate, symbols.u);
  const SX state16 = full_state(
    symbols.q_a_node, symbols.q_u, symbols.dq_a_node * symbols.sigma_rate, symbols.dq_u);

  // The output map of mpc.md 3, read rather than rebuilt: `F_cyl` is constraint
  // 6's left-hand side and the per-axis `Q` sums to constraint 7's, with 3.1's
  // smoothing already inside it.
  const std::vector<SX> outputs = graph.z(std::vector<SX>{state16, ddq_a});
  const SX & z = outputs[0];
  const int force = static_cast<int>(crane_model::symbolic::kCylinderForceOffset);
  const int flow = static_cast<int>(crane_model::symbolic::kAxisFlowOffset);
  const int dof = static_cast<int>(crane_model::kActuatedDof);

  const SX cylinder_force = z(casadi::Slice(force, force + dof));
  const SX pump_flow = SX::sum1(z(casadi::Slice(flow, flow + dof)));

  std::vector<SX> rows;
  rows.reserve(static_cast<std::size_t>(kOcpConstraints));
  for (int row = 0; row < dof; ++row) {
    rows.push_back(ddq_a(row) / scaled.ddq_a_max[row]);
  }
  for (int row = 0; row < dof; ++row) {
    rows.push_back(cylinder_force(row) / scaled.force_max[row]);
  }
  rows.push_back(pump_flow / scaled.flow_max);
  return SX::vertcat(rows);
}

/// The Gauss-Newton residual whose half-square-norm is 5.2's objective.
SX residual_vector(const Symbols & symbols, const TimingOcpSettings & settings, double d_sigma)
{
  // acados' nonlinear least squares cost is `0.5 ||y||^2_W`, so each residual
  // carries a `sqrt(2 dsigma)`: the first row squares to `dsigma / sigma_dot`,
  // which is 5.2's `int dsigma / sigma_dot`, and the two sway rows square to
  // `w ||dq_u||^2 dsigma / sigma_dot`, which is its `int ||dq_u||^2 dt` because
  // `dt = dsigma / sigma_dot`.
  const SX rate = symbols.sigma_rate;
  const SX inverse_root = 1.0 / sqrt(rate);
  return SX::vertcat(
    {std::sqrt(2.0 * d_sigma) * inverse_root,
      std::sqrt(2.0 * settings.sway_weight * d_sigma) * inverse_root * symbols.dq_u,
      std::sqrt(2.0 * settings.input_weight * d_sigma) * symbols.u});
}

/// Every acados object one solve owns, freed in the order acados wants.
class OcpSolver
{
public:
  explicit OcpSolver(int intervals)
  : intervals_(intervals) {}

  ~OcpSolver()
  {
    if (solver_ != nullptr) {ocp_nlp_solver_destroy(solver_);}
    if (out_ != nullptr) {ocp_nlp_out_destroy(out_);}
    if (in_ != nullptr) {ocp_nlp_in_destroy(in_);}
    if (opts_ != nullptr) {ocp_nlp_solver_opts_destroy(opts_);}
    if (dims_ != nullptr) {ocp_nlp_dims_destroy(dims_);}
    if (config_ != nullptr) {ocp_nlp_config_destroy(config_);}
    if (plan_ != nullptr) {ocp_nlp_plan_destroy(plan_);}
  }

  OcpSolver(const OcpSolver &) = delete;
  OcpSolver & operator=(const OcpSolver &) = delete;
  OcpSolver(OcpSolver &&) = delete;
  OcpSolver & operator=(OcpSolver &&) = delete;

  ocp_nlp_plan_t * plan_{nullptr};
  ocp_nlp_config * config_{nullptr};
  ocp_nlp_dims * dims_{nullptr};
  ocp_nlp_in * in_{nullptr};
  ocp_nlp_out * out_{nullptr};
  void * opts_{nullptr};
  ocp_nlp_solver * solver_{nullptr};
  int intervals_;

  /// The bound functions. One entry per distinct `casadi::Function`, each
  /// holding one acados struct per stage that registers it.
  std::vector<std::unique_ptr<AcadosCasadiFunction>> functions_{};

  AcadosCasadiFunction & bind(
    const casadi::Function & function, std::size_t instances, std::string & error)
  {
    functions_.push_back(std::make_unique<AcadosCasadiFunction>());
    std::string message = functions_.back()->bind(function, instances);
    if (!message.empty() && error.empty()) {
      error = std::move(message);
    }
    return *functions_.back();
  }
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

  // ------------------------------------------------------------------
  // The graph, and the expressions built out of it
  // ------------------------------------------------------------------
  crane_model::SymbolicGraphSpec spec;
  spec.sample_time_s = settings.sample_period;
  spec.include_output_map = true;
  auto graph_result = crane_model::symbolic::casadi_graph(model_config, spec, request.payload);
  if (!graph_result.ok()) {
    return Result<TimingSolution>::failure(graph_result.status());
  }
  const crane_model::symbolic::CasadiGraphHandle graph = std::move(graph_result).value();

  const int intervals = static_cast<int>(settings.intervals);
  const double d_sigma = 1.0 / static_cast<double>(intervals);
  const Symbols symbols = make_symbols();

  casadi::Function ode;
  casadi::Function vde;
  casadi::Function residual_jacobian;
  casadi::Function residual;
  casadi::Function terminal_residual_jacobian;
  casadi::Function terminal_residual;
  casadi::Function constraint_jacobian;
  casadi::Function constraint;
  try {
    const SX derivative = path_dynamics(*graph, symbols);
    const SX sensitivity_x = SX::sym("Sx", kOcpStateDof, kOcpStateDof);
    const SX sensitivity_u = SX::sym("Su", kOcpStateDof, kOcpInputDof);
    ode = casadi::Function(
      "expl_ode_fun", {symbols.x, symbols.u, symbols.p}, {densify(derivative)});
    vde = casadi::Function(
      "expl_vde_for", {symbols.x, sensitivity_x, sensitivity_u, symbols.u, symbols.p},
      {densify(derivative),
        densify(SX::jtimes(derivative, symbols.x, sensitivity_x)),
        densify(
          SX::jtimes(derivative, symbols.x, sensitivity_u) +
          SX::jacobian(derivative, symbols.u))});

    // acados asks for the *transposed* Jacobian with respect to `[u; x]`, which
    // is the order the QP carries its variables in.
    const SX variables = SX::vertcat({symbols.u, symbols.x});
    const SX algebraic = SX::sym("z", 0);
    const SX time = SX::sym("t");

    const SX cost_rows = residual_vector(symbols, settings, d_sigma);
    residual = casadi::Function(
      "nls_y_fun", {symbols.x, symbols.u, algebraic, time, symbols.p}, {densify(cost_rows)});
    residual_jacobian = casadi::Function(
      "nls_y_fun_jac", {symbols.x, symbols.u, algebraic, time, symbols.p},
      {densify(cost_rows), densify(SX::jacobian(cost_rows, variables).T()),
        SX::zeros(kResidualDof, 0)});

    // The terminal stage has no input, so `nu = 0` there: acados reads the
    // residual Jacobian as `(nu + nx) x ny`, which is `nx x ny` and not one row
    // taller, and it hands the function an empty `u`. A terminal function built
    // over the running stage's symbols is exactly one row wrong, which the
    // solver reports as a NaN rather than as a dimension error.
    const SX no_input = SX::sym("u_e", 0);
    const SX terminal_rows = std::sqrt(2.0 * settings.sway_weight) * symbols.dq_u;
    terminal_residual = casadi::Function(
      "nls_y_fun_e", {symbols.x, no_input, algebraic, time, symbols.p}, {densify(terminal_rows)});
    terminal_residual_jacobian = casadi::Function(
      "nls_y_fun_jac_e", {symbols.x, no_input, algebraic, time, symbols.p},
      {densify(terminal_rows), densify(SX::jacobian(terminal_rows, symbols.x).T()),
        SX::zeros(kTerminalResidualDof, 0)});

    const SX constraint_rows = constraint_vector(*graph, symbols, scaled);
    constraint = casadi::Function(
      "nl_constr_h_fun", {symbols.x, symbols.u, algebraic, symbols.p}, {densify(constraint_rows)});
    constraint_jacobian = casadi::Function(
      "nl_constr_h_fun_jac", {symbols.x, symbols.u, algebraic, symbols.p},
      {densify(constraint_rows), densify(SX::jacobian(constraint_rows, variables).T())});
  } catch (const std::exception & error) {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::SymbolicBackendFailure,
        std::string("the timing OCP could not be written over the symbolic graph: ") +
        error.what()));
  }

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

  const ScaledLimits scaled = scale_limits(limits, settings, request.speed_scale);

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
  const crane_model::QU q_u_start = q_u_equilibrium.front();
  const crane_model::QU q_u_goal = q_u_equilibrium.back();

  // ------------------------------------------------------------------
  // The acados problem
  // ------------------------------------------------------------------
  OcpSolver ocp(intervals);
  ocp.plan_ = ocp_nlp_plan_create(intervals);
  ocp.plan_->nlp_solver = SQP;
  ocp.plan_->regularization = NO_REGULARIZE;
  // A merit line search rather than a full step. The traversal-time cost is
  // `1/sigma_dot`, whose Gauss-Newton curvature *falls* as the rate rises, so a
  // full Newton step from a slow guess overshoots by orders of magnitude and the
  // QP is then asked about a state the model has no answer for.
  ocp.plan_->globalization = MERIT_BACKTRACKING;
  ocp.plan_->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
  for (int stage = 0; stage <= intervals; ++stage) {
    ocp.plan_->nlp_cost[stage] = NONLINEAR_LS;
    ocp.plan_->nlp_constraints[stage] = BGH;
    if (stage < intervals) {
      ocp.plan_->nlp_dynamics[stage] = CONTINUOUS_MODEL;
      ocp.plan_->sim_solver_plan[stage].sim_solver = ERK;
    }
  }
  ocp.config_ = ocp_nlp_config_create(*ocp.plan_);

  std::vector<int> nx(static_cast<std::size_t>(intervals) + 1U, kOcpStateDof);
  std::vector<int> nu(static_cast<std::size_t>(intervals) + 1U, kOcpInputDof);
  std::vector<int> nz(static_cast<std::size_t>(intervals) + 1U, 0);
  std::vector<int> ns(static_cast<std::size_t>(intervals) + 1U, 0);
  std::vector<int> np(static_cast<std::size_t>(intervals) + 1U, kOcpParameters);
  nu.back() = 0;

  ocp.dims_ = ocp_nlp_dims_create(ocp.config_);
  ocp_nlp_dims_set_opt_vars(ocp.config_, ocp.dims_, "nx", nx.data());
  ocp_nlp_dims_set_opt_vars(ocp.config_, ocp.dims_, "nu", nu.data());
  ocp_nlp_dims_set_opt_vars(ocp.config_, ocp.dims_, "nz", nz.data());
  ocp_nlp_dims_set_opt_vars(ocp.config_, ocp.dims_, "ns", ns.data());
  ocp_nlp_dims_set_opt_vars(ocp.config_, ocp.dims_, "np", np.data());

  // Stage 0 pins everything but the path rate: 5.4 fixes the *terminal* state,
  // and the start is where the machine already is. `sigma_dot(0)` is left free
  // because the path meets its own start with `q_a' = 0`, so no joint moves
  // there whatever the rate is.
  const int bounded_x0 = kOcpStateDof - 1;
  const int nonlinear_rows = kOcpConstraints;
  for (int stage = 0; stage <= intervals; ++stage) {
    const int constraints = stage < intervals ? nonlinear_rows : 0;
    const int residual_rows = stage < intervals ? kResidualDof : kTerminalResidualDof;
    const int bounded_x = stage == 0 ? bounded_x0 : kOcpStateDof;
    const int bounded_u = stage < intervals ? kOcpInputDof : 0;
    ocp_nlp_dims_set_cost(ocp.config_, ocp.dims_, stage, "ny", &residual_rows);
    ocp_nlp_dims_set_constraints(ocp.config_, ocp.dims_, stage, "nbx", &bounded_x);
    ocp_nlp_dims_set_constraints(ocp.config_, ocp.dims_, stage, "nbu", &bounded_u);
    ocp_nlp_dims_set_constraints(ocp.config_, ocp.dims_, stage, "nh", &constraints);
  }

  ocp.in_ = ocp_nlp_in_create(ocp.config_, ocp.dims_);
  ocp.out_ = ocp_nlp_out_create(ocp.config_, ocp.dims_);
  ocp.opts_ = ocp_nlp_solver_opts_create(ocp.config_, ocp.dims_);

  // Seven binds, and seven trampoline slots -- not seven per stage. Each carries
  // one acados struct per stage that registers it, because acados takes the
  // stage's parameter pointer at registration time.
  std::string bind_error;
  const std::size_t path_stages = static_cast<std::size_t>(intervals);
  AcadosCasadiFunction & bound_vde = ocp.bind(vde, path_stages, bind_error);
  AcadosCasadiFunction & bound_ode = ocp.bind(ode, path_stages, bind_error);
  AcadosCasadiFunction & bound_residual_jacobian =
    ocp.bind(residual_jacobian, path_stages, bind_error);
  AcadosCasadiFunction & bound_residual = ocp.bind(residual, path_stages, bind_error);
  AcadosCasadiFunction & bound_constraint_jacobian =
    ocp.bind(constraint_jacobian, path_stages, bind_error);
  AcadosCasadiFunction & bound_constraint = ocp.bind(constraint, path_stages, bind_error);
  AcadosCasadiFunction & bound_terminal_jacobian =
    ocp.bind(terminal_residual_jacobian, 1U, bind_error);
  AcadosCasadiFunction & bound_terminal = ocp.bind(terminal_residual, 1U, bind_error);
  if (!bind_error.empty()) {
    return Result<TimingSolution>::failure(
      failure(ErrorCode::BackendUnavailable, bind_error));
  }

  std::vector<double> weight(
    static_cast<std::size_t>(kResidualDof * kResidualDof), 0.0);
  for (int row = 0; row < kResidualDof; ++row) {
    weight[static_cast<std::size_t>(row * kResidualDof + row)] = 1.0;
  }
  std::vector<double> terminal_weight(
    static_cast<std::size_t>(kTerminalResidualDof * kTerminalResidualDof), 0.0);
  for (int row = 0; row < kTerminalResidualDof; ++row) {
    terminal_weight[static_cast<std::size_t>(row * kTerminalResidualDof + row)] = 1.0;
  }
  const std::vector<double> reference(static_cast<std::size_t>(kResidualDof), 0.0);

  // Every row of `constraint_vector` is already a fraction of its own scaled
  // limit, so the box is the unit one and carries no machine number at all. The
  // flow row is one-sided because `Q` is a draw and not a signed force.
  std::vector<double> lower_h(static_cast<std::size_t>(kOcpConstraints), -1.0);
  std::vector<double> upper_h(static_cast<std::size_t>(kOcpConstraints), 1.0);
  const int dof = static_cast<int>(crane_model::kActuatedDof);
  lower_h[static_cast<std::size_t>(2 * dof)] = 0.0;

  std::vector<int> input_index{0};
  std::vector<double> input_lower{-settings.sigma_accel_max};
  std::vector<double> input_upper{settings.sigma_accel_max};

  std::vector<std::vector<double>> parameters;
  parameters.reserve(static_cast<std::size_t>(intervals) + 1U);

  for (int stage = 0; stage <= intervals; ++stage) {
    // Parameters: the node block, then the block the integrator holds fixed over
    // the interval. The last stage has no interval, so it repeats its own node.
    std::vector<double> stage_parameters(static_cast<std::size_t>(kOcpParameters), 0.0);
    const PathSample & node = nodes[static_cast<std::size_t>(stage)];
    const PathSample & mid =
      midpoints[static_cast<std::size_t>(std::min(stage, intervals - 1))];
    for (int derivative = 0; derivative < 3; ++derivative) {
      const crane_model::QA node_value = actuated(node, derivative);
      const crane_model::QA mid_value = actuated(mid, derivative);
      for (int row = 0; row < dof; ++row) {
        stage_parameters[static_cast<std::size_t>(derivative * dof + row)] = node_value[row];
        stage_parameters[static_cast<std::size_t>(kPathBlock + derivative * dof + row)] =
          mid_value[row];
      }
    }
    parameters.push_back(std::move(stage_parameters));
    ocp_nlp_in_set(
      ocp.config_, ocp.dims_, ocp.in_, stage, "parameter_values",
      parameters.back().data());

    if (stage < intervals) {
      double step = d_sigma;
      ocp_nlp_in_set(ocp.config_, ocp.dims_, ocp.in_, stage, "Ts", &step);
      const std::size_t instance = static_cast<std::size_t>(stage);
      ocp_nlp_dynamics_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "expl_vde_for", bound_vde.handle(instance));
      ocp_nlp_dynamics_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "expl_ode_fun", bound_ode.handle(instance));
      ocp_nlp_cost_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "nls_y_fun_jac",
        bound_residual_jacobian.handle(instance));
      ocp_nlp_cost_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "nls_y_fun", bound_residual.handle(instance));
      if (nonlinear_rows > 0) {
        ocp_nlp_constraints_model_set_external_param_fun(
          ocp.config_, ocp.dims_, ocp.in_, stage, "nl_constr_h_fun_jac",
          bound_constraint_jacobian.handle(instance));
        ocp_nlp_constraints_model_set_external_param_fun(
          ocp.config_, ocp.dims_, ocp.in_, stage, "nl_constr_h_fun",
          bound_constraint.handle(instance));
      }
      ocp_nlp_cost_model_set(ocp.config_, ocp.dims_, ocp.in_, stage, "W", weight.data());
      ocp_nlp_cost_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "yref",
        const_cast<double *>(reference.data()));
      ocp_nlp_constraints_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "idxbu", input_index.data());
      ocp_nlp_constraints_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "lbu", input_lower.data());
      ocp_nlp_constraints_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "ubu", input_upper.data());
      if (nonlinear_rows > 0) {
        ocp_nlp_constraints_model_set(
          ocp.config_, ocp.dims_, ocp.in_, stage, "lh", lower_h.data());
        ocp_nlp_constraints_model_set(
          ocp.config_, ocp.dims_, ocp.in_, stage, "uh", upper_h.data());
      }
    } else {
      ocp_nlp_cost_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "nls_y_fun_jac",
        bound_terminal_jacobian.handle(0U));
      ocp_nlp_cost_model_set_external_param_fun(
        ocp.config_, ocp.dims_, ocp.in_, stage, "nls_y_fun", bound_terminal.handle(0U));
      ocp_nlp_cost_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "W", terminal_weight.data());
      ocp_nlp_cost_model_set(
        ocp.config_, ocp.dims_, ocp.in_, stage, "yref",
        const_cast<double *>(reference.data()));
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

  std::vector<std::vector<int>> state_index;
  std::vector<std::vector<double>> state_lower;
  std::vector<std::vector<double>> state_upper;
  state_index.reserve(static_cast<std::size_t>(intervals) + 1U);
  state_lower.reserve(static_cast<std::size_t>(intervals) + 1U);
  state_upper.reserve(static_cast<std::size_t>(intervals) + 1U);
  for (int stage = 0; stage <= intervals; ++stage) {
    const double rate_ceiling = rate_ceilings[static_cast<std::size_t>(stage)];

    std::vector<int> index;
    std::vector<double> lower;
    std::vector<double> upper;
    if (stage == 0) {
      index = {0, 2, 3, 4, 5};
      lower = {0.0, q_u_start[0], q_u_start[1], 0.0, 0.0};
      upper = lower;
    } else if (stage == intervals) {
      // 5.4, in full: the path is finished, the tool hangs at its equilibrium and
      // it is not moving. `sigma_dot(T)` is the one row 5.4 asks for that this
      // formulation cannot pin at zero -- the objective divides by it -- and it
      // does not need to be pinned, because the path meets its end with
      // `q_a'(1) = q_a''(1) = 0` and the joints are therefore at rest whatever
      // the path rate is. What the machine does is asserted, not the coordinate.
      index = {0, 1, 2, 3, 4, 5};
      lower = {1.0, settings.sigma_rate_min, q_u_goal[0], q_u_goal[1], 0.0, 0.0};
      upper = {1.0, rate_ceiling, q_u_goal[0], q_u_goal[1], 0.0, 0.0};
    } else {
      const crane_model::QU & rest = q_u_equilibrium[static_cast<std::size_t>(stage)];
      index = {0, 1, 2, 3, 4, 5};
      lower = {
        0.0, settings.sigma_rate_min, rest[0] - settings.q_u_max[0],
        rest[1] - settings.q_u_max[1], -settings.dq_u_max[0], -settings.dq_u_max[1]};
      upper = {
        1.0, rate_ceiling, rest[0] + settings.q_u_max[0],
        rest[1] + settings.q_u_max[1], settings.dq_u_max[0], settings.dq_u_max[1]};
    }
    state_index.push_back(std::move(index));
    state_lower.push_back(std::move(lower));
    state_upper.push_back(std::move(upper));
    ocp_nlp_constraints_model_set(
      ocp.config_, ocp.dims_, ocp.in_, stage, "idxbx", state_index.back().data());
    ocp_nlp_constraints_model_set(
      ocp.config_, ocp.dims_, ocp.in_, stage, "lbx", state_lower.back().data());
    ocp_nlp_constraints_model_set(
      ocp.config_, ocp.dims_, ocp.in_, stage, "ubx", state_upper.back().data());
  }

  int stages = 1;
  int steps = 1;
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "max_iter", const_cast<int *>(
      &settings.max_iterations));
  double budget = settings.max_wall_clock;
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "timeout_max_time", &budget);
  double regularisation = settings.levenberg_marquardt;
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "levenberg_marquardt", &regularisation);
  double tolerance_stationarity = settings.tolerance_stationarity;
  double tolerance_feasibility = settings.tolerance_feasibility;
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "tol_stat", &tolerance_stationarity);
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "tol_eq", &tolerance_feasibility);
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "tol_ineq", &tolerance_feasibility);
  ocp_nlp_solver_opts_set(ocp.config_, ocp.opts_, "tol_comp", &tolerance_feasibility);
  for (int stage = 0; stage < intervals; ++stage) {
    stages = 4;
    steps = 1;
    ocp_nlp_solver_opts_set_at_stage(ocp.config_, ocp.opts_, stage, "dynamics_ns", &stages);
    ocp_nlp_solver_opts_set_at_stage(
      ocp.config_, ocp.opts_, stage, "dynamics_num_steps", &steps);
  }

  ocp.solver_ = ocp_nlp_solver_create(ocp.config_, ocp.dims_, ocp.opts_, ocp.in_);

  const int precompute = ocp_nlp_precompute(ocp.solver_, ocp.in_, ocp.out_);
  if (precompute != ACADOS_SUCCESS) {
    return Result<TimingSolution>::failure(
      failure(
        ErrorCode::BackendUnavailable,
        "acados could not precompute the timing OCP: " + acados_status_word(precompute)));
  }

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
  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> guess(static_cast<std::size_t>(kOcpStateDof), 0.0);
    guess[0] = static_cast<double>(stage) * d_sigma;
    guess[1] = rate_guesses[static_cast<std::size_t>(stage)];
    guess[2] = q_u_equilibrium[static_cast<std::size_t>(stage)][0];
    guess[3] = q_u_equilibrium[static_cast<std::size_t>(stage)][1];
    ocp_nlp_out_set(ocp.config_, ocp.dims_, ocp.out_, stage, "x", guess.data());
    if (stage < intervals) {
      double input = 0.0;
      ocp_nlp_out_set(ocp.config_, ocp.dims_, ocp.out_, stage, "u", &input);
    }
    // acados allocates `nlp_out` with `malloc` and does not zero it, so the
    // multipliers start as whatever was on the heap -- and the first
    // stationarity residual is then a NaN before any step has been taken. The
    // examples get away with it because they leave far less of the vector
    // unused; a horizon with thirteen nonlinear rows per stage does not.
    std::vector<double> zeros(64U, 0.0);
    ocp_nlp_out_set(ocp.config_, ocp.dims_, ocp.out_, stage, "lam", zeros.data());
    if (stage < intervals) {
      ocp_nlp_out_set(ocp.config_, ocp.dims_, ocp.out_, stage, "pi", zeros.data());
    }
  }

  const int status = ocp_nlp_solve(ocp.solver_, ocp.in_, ocp.out_);

  int iterations = 0;
  double solve_time = 0.0;
  ocp_nlp_get(ocp.solver_, "sqp_iter", &iterations);
  ocp_nlp_get(ocp.solver_, "time_tot", &solve_time);

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
    ocp_nlp_out_get(ocp.config_, ocp.dims_, ocp.out_, stage, "x", state.data());
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
  for (int stage = 0; stage <= intervals; ++stage) {
    std::vector<double> state(static_cast<std::size_t>(kOcpStateDof), 0.0);
    ocp_nlp_out_get(ocp.config_, ocp.dims_, ocp.out_, stage, "x", state.data());
    double input = 0.0;
    if (stage < intervals) {
      ocp_nlp_out_get(ocp.config_, ocp.dims_, ocp.out_, stage, "u", &input);
    }
    const PathSample & node = nodes[static_cast<std::size_t>(stage)];
    const crane_model::QA slope = actuated(node, 1);
    const crane_model::QA curvature = actuated(node, 2);

    std::vector<double> acceleration(static_cast<std::size_t>(crane_model::kActuatedDof), 0.0);
    const crane_model::QA position = actuated(node, 0);
    for (int row = 0; row < dof; ++row) {
      acceleration[static_cast<std::size_t>(row)] =
        curvature[row] * state[1] * state[1] + slope[row] * input;
      solution.peak_demand.joint_acceleration = std::max(
        solution.peak_demand.joint_acceleration,
        std::abs(acceleration[static_cast<std::size_t>(row)]) /
        settings.actuation.ddq_a_max[static_cast<std::size_t>(row)]);
      solution.peak_demand.joint_velocity = std::max(
        solution.peak_demand.joint_velocity,
        std::abs(slope[row] * state[1]) /
        limits.axis[static_cast<std::size_t>(row)].dq_max);
    }

    std::vector<double> state16;
    state16.reserve(16U);
    for (int row = 0; row < dof; ++row) {state16.push_back(position[row]);}
    state16.push_back(state[2]);
    state16.push_back(state[3]);
    for (int row = 0; row < dof; ++row) {state16.push_back(slope[row] * state[1]);}
    state16.push_back(state[4]);
    state16.push_back(state[5]);

    const std::vector<casadi::DM> outputs = graph->z(
      std::vector<casadi::DM>{casadi::DM(state16), casadi::DM(acceleration)});
    const std::vector<double> z = outputs[0].nonzeros();
    double flow = 0.0;
    for (int row = 0; row < dof; ++row) {
      const double force =
        z[crane_model::symbolic::kCylinderForceOffset + static_cast<std::size_t>(row)];
      solution.peak_demand.cylinder_force = std::max(
        solution.peak_demand.cylinder_force,
        std::abs(force) /
        settings.actuation.cylinder_force_max[static_cast<std::size_t>(row)]);
      flow += z[crane_model::symbolic::kAxisFlowOffset + static_cast<std::size_t>(row)];
    }
    solution.peak_demand.pump_flow = std::max(
      solution.peak_demand.pump_flow,
      flow / (settings.actuation.pump_flow_planning_factor * settings.actuation.pump_flow_max));
  }

  return Result<TimingSolution>::success(std::move(solution));
}

}  // namespace crane_planning
