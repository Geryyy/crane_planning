#include "crane_planning/equilibrium_ik.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::QA;
using crane_model::QU;
using crane_model::Result;
using crane_model::Status;

/// The rows of the canonical eight the NLP decides.
/**
 * Seven of the eight: q1 to q4, the passive pair q5 and q6, and the rotator q7.
 * q8 is the tool coordinate and is held where the machine has it --
 * `wiki/trajectory_planning.md` 4.1 keeps it off the path, and what opens and
 * closes the gripper is `/crane/plan_grip`, which is issue 044.
 *
 * The passive rows are decision variables and not an inner solve. That is the
 * whole difference between this route and the semi-analytic one: `g_u(q) = 0`
 * is a constraint the step satisfies to first order, so q5 and q6 move with a
 * step of q2 and q3 rather than being re-solved after it.
 */
constexpr std::array<Eigen::Index, 7> kFreeRows{{0, 1, 2, 3, 4, 5, 6}};
constexpr Eigen::Index kFreeCount = 7;

/// The rows of the passive pair inside `kFreeRows`.
constexpr Eigen::Index kPassiveInFree = 4;

/// Residual rows: three of position, nine of Frobenius orientation, three of centring.
constexpr Eigen::Index kPositionRows = 3;
constexpr Eigen::Index kOrientationRows = 9;
constexpr Eigen::Index kCentringRows = 3;
constexpr Eigen::Index kResidualRows = kPositionRows + kOrientationRows + kCentringRows;

/// Step a numerical derivative is read at, rad.
/**
 * Forward differences. The residual and the constraint are both smooth here and
 * Gauss-Newton tolerates a Jacobian that is a little wrong -- what it does not
 * tolerate is a Jacobian swamped by cancellation, which is what a step much
 * below this gives on a constraint whose values run to hundreds of N m.
 */
constexpr double kProbeStep = 1.0e-6;

/// Largest step one Gauss-Newton iteration may take, in the norm of `kFreeRows`.
constexpr double kMaxStep = 0.5;

/// How often the line search halves a step that did not improve the residual.
/**
 * Deep, because the two halves of this objective live at very different scales.
 * The task rows are scaled by their own tolerances and the centring rows are
 * not, so a step that buys centring at the cost of a *second-order* position
 * error has to be shortened a long way before it pays -- measured, ten or so
 * halvings on a seed that is already inside the acceptance residual. Eight was
 * not enough: the line search failed outright on the first iteration and the
 * solve returned its seed unchanged.
 */
constexpr int kBacktrackingSteps = 14;

/// Relative objective improvement below which the iteration has stopped paying.
/**
 * The other half of the convergence test, and in practice the half that fires.
 * Flatness along 2.2 step 3's redundant direction is the normal case here rather
 * than a pathology: the semi-analytic seed resolves that freedom on a 41-point
 * grid and lands within 0.07 per cent of the continuous optimum, measured over
 * both descriptions. So there is almost nothing left along it, and a step-norm
 * test alone spends the whole iteration cap collecting the remainder -- measured
 * at 1e-4, two of fifty sampled goals ran out of iterations having already met
 * the acceptance residual, which is a refusal that says nothing true.
 *
 * A thousandth costs nothing that can be measured -- worst position residual
 * over the same fifty goals is 7.4e-8 m against 6.5e-8 m at 1e-4, both three
 * orders under `eps_pos` -- and brings the worst case back to 24 iterations.
 */
constexpr double kRelativeImprovement = 1.0e-3;

/// Budget for the feasibility restoration that follows every trial step.
constexpr int kRestorationSteps = 8;

/// How small ||g_u|| has to get, in N m, for the restoration to stop early.
/**
 * Not an acceptance tolerance -- the acceptance tolerance is `eps_equilibrium`,
 * in radians, and it is checked against `Model::passive_equilibrium` at the end.
 * This is only where the restoration Newton stops paying for itself: measured on
 * both descriptions the passive rows stiffen at some 4e3 N m per rad, so a
 * milli-newton-metre is well under a microradian of passive displacement.
 */
constexpr double kRestorationTolerance = 1.0e-3;

/// How heavily the joint-range centring of 2.2 step 3 counts against the task.
/**
 * The task rows are scaled by their own tolerances, so they are order one at the
 * acceptance boundary and go to zero inside it; the centring rows are order one
 * across the whole range. Any positive weight at all resolves the redundancy,
 * because along the one direction the task cannot see the task has no gradient
 * whatsoever and centring is the only thing voting. The size of the weight
 * therefore buys nothing on the redundancy and only sets how much of the *task*
 * is traded away second-order, so it is small: a thousandth leaves a position
 * error some three orders under `eps_pos`, where a weight of one spends 2.4e-6 m
 * of it to move the centring score by 0.07 per cent.
 *
 * What these rows do here is **hold** 2.2 step 3's redundancy rather than
 * re-resolve it. The seed arrives with the extension already picked off that
 * step's scalar search, and measured against it this objective then moves the
 * centring score by less than a part in a million. Their job is to keep the
 * redundant direction non-degenerate, so the Gauss-Newton step cannot wander
 * along the one direction the task cannot see.
 *
 * **Step 3's clearance term is therefore not a row here**, and that is a
 * decision rather than an omission: it votes where the redundancy is resolved,
 * which is the scalar search the seed comes off (`redundancy.hpp`). Adding it
 * here would cost a `Model::collision_query` per finite-difference probe, on a
 * residual that is not smooth where the closest pair changes, to move an answer
 * this objective is already measured not to move.
 */
constexpr double kCentringScale = 1.0e-3;

/// Step below which the constrained Gauss-Newton is a stationary point, rad.
/**
 * The convergence test of the NLP, and not the acceptance test -- the two are
 * different questions and conflating them is what makes a solver report success
 * for a minimum of the wrong thing. This says the iteration has stopped moving;
 * the forward-kinematics residual of 2.2 then says whether where it stopped is
 * the goal. Below roughly this a forward-difference Jacobian is reading its own
 * truncation error rather than the objective.
 */
constexpr double kStepTolerance = 1.0e-7;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

double wrap(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

/// The actuated projection of a canonical eight-vector.
QA actuated(const Q & q)
{
  QA q_a;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }
  return q_a;
}

/// Everything one configuration is judged by, and what it cost to find out.
struct Evaluation
{
  Eigen::Vector3d p_tcp{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  double phi_z{};
  QU g_u{QU::Zero()};  ///< the constraint, N m
};

/// The tool pose and the constraint at one configuration, through the frozen API.
/**
 * `g_u` is the passive rows of inverse dynamics at rest, which
 * `wiki/implementation/model_api_contract.md` 7 states are zero exactly at a
 * valid passive equilibrium. Two model calls, some 0.95 ms together.
 */
Status evaluate(
  const crane_model::Model & model, const crane_model::Payload & payload, const Q & q,
  Evaluation & out)
{
  auto pose = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!pose.ok()) {
    return pose.status();
  }
  auto tau = model.inverse_dynamics(q, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
  if (!tau.ok()) {
    return tau.status();
  }
  out.p_tcp = pose.value().position_m;
  out.rotation = pose.value().orientation.normalized().toRotationMatrix();
  out.phi_z = phi_z_of(pose.value().orientation);
  out.g_u = tau.value().segment<2>(4);
  return Status{};
}

/// The constraint alone -- what the restoration Newton iterates on.
Status constraint_only(
  const crane_model::Model & model, const crane_model::Payload & payload, const Q & q, QU & out)
{
  auto tau = model.inverse_dynamics(q, crane_model::DQ::Zero(), crane_model::DQ::Zero(), payload);
  if (!tau.ok()) {
    return tau.status();
  }
  out = tau.value().segment<2>(4);
  return Status{};
}

/// The desired rotation of 2.2, completed from the attitude this iterate hangs at.
/**
 * `R_d = R_z(phi_d - phi_k) R_k`, so that at the iterate the Frobenius residual
 * is exactly the yaw error and nothing else. See the header for the measurement
 * that rules out 2.2's literal `R(phi_z,d)`: against a pure yaw the term is the
 * constant 8 on this machine and carries no information at all.
 */
Eigen::Matrix3d desired_rotation(const Evaluation & evaluation, double phi_z_d)
{
  const Eigen::Matrix3d turn(
    Eigen::AngleAxisd(wrap(phi_z_d - evaluation.phi_z), Eigen::Vector3d::UnitZ()));
  return turn * evaluation.rotation;
}

/// The residual vector the Gauss-Newton step drives to zero.
/**
 * Scaled by the tolerances it is judged against, so that every row is order one
 * at the acceptance boundary and the position and the yaw halves of 2.2's oracle
 * trade against each other in the units they are actually stated in. The
 * orientation rows carry `1 / (sqrt(2) eps_yaw)` because a yaw error of d puts
 * `sqrt(4 (1 - cos d)) ~ sqrt(2) |d|` into their norm.
 */
Eigen::Matrix<double, kResidualRows, 1> residual_of(
  const JointLimits & limits, const IkSettings & seed, const RedundancyWeights & weights,
  const IkRequest & request, const Eigen::Matrix3d & desired, const Evaluation & evaluation,
  const Q & q)
{
  Eigen::Matrix<double, kResidualRows, 1> residual;
  residual.head<kPositionRows>() = (evaluation.p_tcp - request.p_tcp_0) / seed.eps_pos;

  const Eigen::Matrix3d error =
    desired.transpose() * evaluation.rotation - Eigen::Matrix3d::Identity();
  const double orientation_scale = 1.0 / (std::sqrt(2.0) * seed.eps_yaw);
  for (Eigen::Index column = 0; column < 3; ++column) {
    residual.segment<3>(kPositionRows + 3 * column) = orientation_scale * error.col(column);
  }

  for (Eigen::Index index = 0; index < kCentringRows; ++index) {
    const std::size_t axis = kRedundantAxes[static_cast<std::size_t>(index)];
    residual[kPositionRows + kOrientationRows + index] = kCentringScale * weights.centring *
      centring_offset(limits.axis[axis], q[static_cast<Eigen::Index>(axis)]);
  }
  return residual;
}

/// How far one evaluation is from being accepted, in units of its own tolerance.
double merit(const IkSettings & seed, const IkRequest & request, const Evaluation & evaluation)
{
  return std::max(
    (evaluation.p_tcp - request.p_tcp_0).norm() / seed.eps_pos,
    std::abs(wrap(evaluation.phi_z - request.phi_z_d)) / seed.eps_yaw);
}

bool inside(const AxisLimit & axis, double value)
{
  return !axis.bounded || (value >= axis.lower && value <= axis.upper);
}

/// Whether every actuated row of a configuration is inside the description's range.
/**
 * The passive pair is not checked here and has no row in `JointLimits`, which
 * reads the six actuated axes only. It does not need one: `g_u(q) = 0` is what
 * decides where the passive joints sit, and a range on a joint nothing actuates
 * is a statement about how far the tool may swing rather than a bound the
 * endpoint may be chosen inside.
 */
bool within_limits(const JointLimits & limits, const Q & q)
{
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    if (!inside(limits.axis[row], q[static_cast<Eigen::Index>(kActuatedRows[row])])) {
      return false;
    }
  }
  return true;
}

/// Put a trial configuration back onto `g_u(q) = 0` before it is judged.
/**
 * The Gauss-Newton step satisfies the constraint only to first order, so a trial
 * point is off the manifold by the curvature times the square of the step. Newton
 * on the passive pair alone puts it back, at the **frozen** passive block of the
 * constraint Jacobian the step was built with -- a chord iteration, so one model
 * call per step rather than three. It converges in a handful of steps because the
 * passive block is strongly diagonally dominant everywhere the tool hangs.
 *
 * Restoring rather than penalising is what keeps the line search honest: every
 * accepted iterate is feasible, so the merit is the task residual alone and there
 * is no penalty parameter to tune and no Maratos effect to work around.
 */
Status restore(
  const crane_model::Model & model, const crane_model::Payload & payload,
  const Eigen::Matrix2d & passive_block, Q & q, double & g_norm)
{
  QU g = QU::Zero();
  Status status = constraint_only(model, payload, q, g);
  if (!status.ok()) {
    return status;
  }
  const Eigen::PartialPivLU<Eigen::Matrix2d> solver(passive_block);
  for (int step = 0; step < kRestorationSteps && g.norm() > kRestorationTolerance; ++step) {
    const QU correction = solver.solve(QU(-g));
    if (!correction.allFinite()) {
      break;
    }
    const Q trial = [&q, &correction] {
        Q next = q;
        next.segment<2>(4) += correction;
        return next;
      }();
    QU next_g = QU::Zero();
    status = constraint_only(model, payload, trial, next_g);
    if (!status.ok()) {
      return status;
    }
    // A chord step that made the violation worse is a step the frozen Jacobian
    // no longer describes; stopping leaves the better point rather than walking
    // away from the manifold on a stale derivative.
    if (next_g.norm() >= g.norm()) {
      break;
    }
    q = trial;
    g = next_g;
  }
  g_norm = g.norm();
  return status;
}

/// The initial guess: the semi-analytic answer where there is one, mid-range otherwise.
/**
 * 2.2's closing paragraph says the semi-analytic route is for where speed
 * matters, and the cheapest thing a fast route can buy an exact one is a start
 * near the answer -- in particular the elbow branch and the telescope extension,
 * which are the two discrete choices this continuous solve cannot make for
 * itself.
 *
 * A goal that route refuses is still attempted, from the description's own
 * mid-range configuration settled at its equilibrium. So a refusal out of this
 * file is this solver's refusal, carrying this solver's residual, and not an
 * echo of the other one's.
 */
Status seed_configuration(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & seed, const IkRequest & request, Q & q, bool & from_semi_analytic)
{
  auto solved = solve_inverse_kinematics(model, geometry, limits, seed, request);
  if (solved.ok()) {
    q = solved.value().q;
    from_semi_analytic = true;
    return Status{};
  }

  from_semi_analytic = false;
  q = Q::Zero();
  for (std::size_t row = 0; row < 4U; ++row) {
    if (limits.axis[row].bounded) {
      q[static_cast<Eigen::Index>(row)] = limits.axis[row].centre();
    }
  }
  q[0] = std::atan2(request.p_tcp_0.y(), request.p_tcp_0.x());
  if (!inside(limits.axis[0], q[0])) {
    q[0] = limits.axis[0].centre();
  }
  q[7] = request.q8;
  auto settled = model.passive_equilibrium(actuated(q), request.payload);
  if (!settled.ok()) {
    return settled.status();
  }
  q.segment<2>(4) = settled.value();
  return Status{};
}

}  // namespace

crane_model::Result<IkSolution> solve_equilibrium_constrained_ik(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & seed, const EquilibriumIkSettings & settings, const IkRequest & request)
{
  const auto started = std::chrono::steady_clock::now();
  const auto elapsed = [&started]() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };

  if (!request.p_tcp_0.allFinite() || !std::isfinite(request.phi_z_d) ||
    !std::isfinite(request.q8))
  {
    return Result<IkSolution>::failure(
      failure(ErrorCode::NonFiniteInput, "the goal pose is not finite"));
  }
  if (!inside(limits.axis[5], request.q8)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the tool coordinate q8 = " + std::to_string(request.q8) +
        " is outside the range the description gives it"));
  }
  {
    Status status = check_redundancy_weights(settings.redundancy);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }
  }
  if (settings.max_iterations == 0U || !(settings.max_wall_clock_s > 0.0)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the equilibrium-constrained solve is bounded by an iteration cap and a wall-clock cap "
        "and both have to be positive; an unbounded NLP inside a service call is a hang"));
  }

  Q q = Q::Zero();
  bool from_semi_analytic = false;
  {
    Status status =
      seed_configuration(model, geometry, limits, seed, request, q, from_semi_analytic);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }
  }

  Evaluation evaluation;
  {
    Status status = evaluate(model, request.payload, q, evaluation);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }
  }
  double g_norm = evaluation.g_u.norm();
  std::size_t iterations = 0;
  bool converged = false;

  // The NLP is iterated to a **stationary point of its own objective**, not to
  // the first configuration that happens to pass 2.2's acceptance test. Those
  // are two different questions and a solver that answers the second while
  // claiming the first reports success for a minimum of the wrong thing. The
  // acceptance test is applied once, below, to the point this converges on.
  //
  // Warm-started from the semi-analytic answer this is usually one or two
  // iterations, because that answer is already feasible -- it carries
  // `q_u = q_eq(q_a)` exactly -- and already close. What the iteration adds over
  // it is measurable rather than decorative on the 7040, whose seed is the
  // weaker of the two: worst position residual over fifty sampled goals falls
  // from 1.8e-6 m to 7.4e-8 m.
  for (std::size_t iteration = 0; iteration < settings.max_iterations; ++iteration) {
    if (elapsed() > settings.max_wall_clock_s) {
      return Result<IkSolution>::failure(
        failure(
          ErrorCode::NotReady,
          "the equilibrium-constrained solve of robot_model 2.2 hit its wall-clock cap of " +
          std::to_string(settings.max_wall_clock_s) + " s after " + std::to_string(iteration) +
          " of its " + std::to_string(settings.max_iterations) +
          " permitted iterations, with " +
          std::to_string((evaluation.p_tcp - request.p_tcp_0).norm()) +
          " m of position residual still open. A bounded solve that ran out of time returns no "
          "endpoint rather than the best one it happened to have"));
    }

    // `desired` is rebuilt from this iterate's own hanging attitude and then held
    // fixed for the whole iteration -- residual, Jacobian and line search alike --
    // so what is linearised and what is line-searched are the same function.
    const Eigen::Matrix3d desired = desired_rotation(evaluation, request.phi_z_d);
    const Eigen::Matrix<double, kResidualRows, 1> residual =
      residual_of(limits, seed, settings.redundancy, request, desired, evaluation, q);
    double current = residual.squaredNorm();

    // The two Jacobians, forward differences over the seven free rows. Each probe
    // costs one forward-kinematics call and one inverse-dynamics call, so the
    // whole linearisation is some 6.7 ms -- a fifth of a single
    // `Model::passive_equilibrium`, which is the entire reason this route is
    // affordable over eight coordinates.
    Eigen::Matrix<double, kResidualRows, kFreeCount> task;
    Eigen::Matrix<double, 2, kFreeCount> constraint;
    for (Eigen::Index column = 0; column < kFreeCount; ++column) {
      Q probe = q;
      probe[kFreeRows[static_cast<std::size_t>(column)]] += kProbeStep;
      Evaluation probed;
      const Status status = evaluate(model, request.payload, probe, probed);
      if (!status.ok()) {
        return Result<IkSolution>::failure(Status(status));
      }
      task.col(column) =
        (residual_of(limits, seed, settings.redundancy, request, desired, probed, probe) -
        residual) / kProbeStep;
      constraint.col(column) = (probed.g_u - evaluation.g_u) / kProbeStep;
    }

    // One equality-constrained Gauss-Newton step:
    //     min ||task d + residual||^2   s.t.   constraint d = -g_u
    // taken in a null-space decomposition rather than as a penalty, so the
    // constraint is met to first order exactly and does not trade against the
    // task at any weight. `particular` restores feasibility in the minimum norm;
    // `basis` spans what is left, which is the five-dimensional freedom the two
    // constraints leave in the seven free rows, and the task -- four dimensions
    // of it, three of position and one of yaw -- is minimised inside that. The
    // one direction neither reaches is 2.2 step 3's telescope redundancy, and it
    // is the centring rows of the residual that pick a point along it.
    const Eigen::Matrix<double, kFreeCount, 1> particular =
      constraint.completeOrthogonalDecomposition().solve(
      Eigen::Matrix<double, 2, 1>(-evaluation.g_u));
    const Eigen::FullPivLU<Eigen::Matrix<double, 2, kFreeCount>> decomposition(constraint);
    const Eigen::MatrixXd basis = decomposition.kernel();
    Eigen::Matrix<double, kFreeCount, 1> step = particular;
    if (basis.cols() > 0) {
      const Eigen::MatrixXd reduced = task * basis;
      const Eigen::Matrix<double, kResidualRows, 1> target = -(task * particular + residual);
      const Eigen::VectorXd free_step = reduced.colPivHouseholderQr().solve(target);
      if (free_step.allFinite()) {
        step += basis * free_step;
      }
    }
    // A step the linear algebra could not produce leaves the last accepted
    // iterate standing, and the acceptance test below judges it on its residual
    // like any other. It is not an exhausted budget, so it is not reported as one.
    if (!step.allFinite()) {
      converged = true;
      break;
    }
    if (step.norm() > kMaxStep) {
      step *= kMaxStep / step.norm();
    }
    // A step this small is a stationary point of the constrained objective. Every
    // exit from this loop leaves `evaluation` at the best point found, and what
    // decides whether that point is returned is the acceptance test below, not
    // which of these exits was taken.
    if (step.norm() < kStepTolerance) {
      iterations = iteration + 1U;
      converged = true;
      break;
    }

    // The passive block of the constraint Jacobian, which the restoration below
    // reuses as a frozen chord rather than re-differencing per trial.
    const Eigen::Matrix2d passive_block = constraint.block<2, 2>(0, kPassiveInFree);

    bool improved = false;
    double scale = 1.0;
    for (int attempt = 0; attempt < kBacktrackingSteps; ++attempt) {
      Q trial = q;
      for (Eigen::Index column = 0; column < kFreeCount; ++column) {
        trial[kFreeRows[static_cast<std::size_t>(column)]] += scale * step[column];
      }
      if (within_limits(limits, trial)) {
        double trial_g = 0.0;
        Status status = restore(model, request.payload, passive_block, trial, trial_g);
        if (!status.ok()) {
          return Result<IkSolution>::failure(std::move(status));
        }
        Evaluation candidate;
        status = evaluate(model, request.payload, trial, candidate);
        if (!status.ok()) {
          return Result<IkSolution>::failure(std::move(status));
        }
        // Judged on the objective the step was computed from, at the same frozen
        // `desired`, and on the trial *after* it was restored onto the manifold --
        // so a step is only accepted if the point it actually leaves behind, a
        // feasible one, is better.
        const double value =
          residual_of(limits, seed, settings.redundancy, request, desired, candidate, trial)
          .squaredNorm();
        if (value < current) {
          q = trial;
          evaluation = candidate;
          g_norm = trial_g;
          improved = true;
          // The objective has stopped paying for the model calls another
          // iteration costs. This is the exit that normally fires: see
          // `kRelativeImprovement`.
          converged = (current - value) < kRelativeImprovement * current;
          break;
        }
      }
      scale *= 0.5;
    }
    iterations = iteration + 1U;
    if (converged) {
      break;
    }
    // A line search that could not improve at any depth is a stationary point as
    // far as a forward-difference Jacobian can resolve one, and is convergence
    // rather than an exhausted budget.
    if (!improved) {
      converged = true;
      break;
    }
  }

  // The other half of criterion the caps exist for. An iteration budget that runs
  // out is a solve that was still moving when it was stopped, so what it holds is
  // an iterate and not an answer -- even when that iterate happens to sit inside
  // the acceptance residual, because nothing establishes that the next step would
  // not have moved it back out.
  if (!converged) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::NotReady,
        "the equilibrium-constrained solve of robot_model 2.2 hit its iteration cap of " +
        std::to_string(settings.max_iterations) + " without converging, in " +
        std::to_string(elapsed()) + " s of its " + std::to_string(settings.max_wall_clock_s) +
        " s wall-clock cap, with " + std::to_string((evaluation.p_tcp - request.p_tcp_0).norm()) +
        " m of position residual open against eps_pos = " + std::to_string(seed.eps_pos) +
        " m. A bounded solve that ran out of iterations returns no endpoint rather than the "
        "best one it happened to have reached"));
  }

  IkSolution solution;
  solution.q = q;
  solution.route = EndpointRoute::EquilibriumConstrained;
  solution.residual_p = (evaluation.p_tcp - request.p_tcp_0).norm();
  solution.residual_phi_z = std::abs(wrap(evaluation.phi_z - request.phi_z_d));
  solution.residual_g_u = g_norm;
  solution.d45 = geometry.d45(q[3]);
  solution.nlp_iterations = iterations;

  // 2.2's own acceptance test, on this route too. A converged NLP with a residual
  // over tolerance is a refusal naming the residual, never a returned endpoint --
  // the solve minimises what it was given, and the only evidence that what it was
  // given was the goal is forward kinematics run on the answer.
  if (merit(seed, request, evaluation) > 1.0) {
    solution.elapsed_s = elapsed();
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the equilibrium-constrained solve of robot_model 2.2 did not converge onto the goal: "
        "after " + std::to_string(iterations) + " iterations it left " +
        std::to_string(solution.residual_p) + " m of position residual against eps_pos = " +
        std::to_string(seed.eps_pos) + " m and " + std::to_string(solution.residual_phi_z) +
        " rad of yaw residual against eps_yaw = " + std::to_string(seed.eps_yaw) +
        " rad. The constraint g_u(q) = 0 was met to " + std::to_string(g_norm) +
        " N m, so this is a goal no passive equilibrium of this machine reaches and not a "
        "constraint the solve failed to satisfy" +
        (from_semi_analytic ?
        std::string() :
        std::string(
          ". The semi-analytic route refused it as well, so the solve started from the "
          "description's mid-range configuration rather than near an answer"))));
  }

  // The assertion of the acceptance test, and the reason it is worth its 33 ms:
  // `g_u(q) = 0` has a second root with the tool standing up, which is an
  // equilibrium and is not a steady state anything settles into. The solver's own
  // constraint residual cannot tell the two apart. `Model::passive_equilibrium`
  // returns the settled one, so this is a stability check evaluated through the
  // frozen model API rather than a restatement of what the solver already
  // believes.
  auto settled = model.passive_equilibrium(actuated(q), request.payload);
  if (!settled.ok()) {
    return Result<IkSolution>::failure(settled.status());
  }
  solution.residual_equilibrium = (q.segment<2>(4) - settled.value()).norm();
  solution.elapsed_s = elapsed();
  if (!(solution.residual_equilibrium < settings.eps_equilibrium)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the equilibrium-constrained solve of robot_model 2.2 put the tool where it was asked "
        "for, but its passive pair sits " + std::to_string(solution.residual_equilibrium) +
        " rad from the equilibrium of its own actuated part, against eps_equilibrium = " +
        std::to_string(settings.eps_equilibrium) +
        " rad. g_u(q) = 0 was met to " + std::to_string(g_norm) +
        " N m, so the solve converged onto a root of the constraint that Model::passive_"
        "equilibrium does not settle into -- the tool standing up rather than hanging. "
        "trajectory_planning 6 asks for a genuine steady state, and this is not one"));
  }
  if (!within_limits(limits, q)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the solved configuration puts an actuated joint outside the range the description "
        "gives it"));
  }
  return Result<IkSolution>::success(std::move(solution));
}

}  // namespace crane_planning
