#include "crane_planning/inverse_kinematics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
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

/// Largest step the Jacobian polish may take in one iteration, rad.
constexpr double kMaxRefinementStep = 0.5;

/// How often the polish halves a step that did not improve the residual.
constexpr int kBacktrackingSteps = 8;

/// Fraction of a tolerance below which the wrist fixed point counts as stalled.
constexpr double kStagnation = 0.1;

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

/// One candidate of the step-3 search, before it has been checked against FK.
struct Closure
{
  double q1{};
  double q2{};
  double q3{};
  double q4{};
  double branch{};  ///< +1 or -1, the `q2 = gamma +- acos(.)` of 2.2 step 2
  double score{std::numeric_limits<double>::infinity()};
};

/// The joint-range centring of 2.2 step 3, as a sum of squared normalised offsets.
/**
 * An axis the description leaves unbounded -- the rotator -- contributes
 * nothing, because there is no range for it to be centred in and a made-up one
 * would silently steer the redundancy.
 */
double centring_score(const JointLimits & limits, const Closure & candidate)
{
  const std::array<double, 4> value{{candidate.q1, candidate.q2, candidate.q3, candidate.q4}};
  double score = 0.0;
  for (std::size_t row = 0; row < value.size(); ++row) {
    const AxisLimit & axis = limits.axis[row];
    if (!axis.bounded) {
      continue;
    }
    const double normalised = (value[row] - axis.centre()) / axis.half_span();
    score += normalised * normalised;
  }
  return score;
}

bool inside(const AxisLimit & axis, double value)
{
  return !axis.bounded || (value >= axis.lower && value <= axis.upper);
}

/// Steps 1 to 3: the azimuth, the planar closure, and the search that picks d45.
/**
 * `fixed` holds the extension and the elbow branch a previous pass resolved on.
 * Freezing them is what makes step 4 a fixed-point iteration at all: a search
 * re-run inside the loop can jump between two extensions that are equally well
 * centred, and the correction then oscillates between them for ever instead of
 * converging.
 */
bool solve_closure(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  std::size_t d45_samples, const Eigen::Vector3d & p_wrist_0, const Closure * fixed,
  Closure & out)
{
  const double q1 = std::atan2(p_wrist_0.y(), p_wrist_0.x());
  if (!inside(limits.axis[0], q1)) {
    return false;
  }
  Q q = Q::Zero();
  q[0] = q1;
  auto column = model.forward_kinematics(q, Frame::MountingBase, Frame::SlewingColumn);
  if (!column.ok()) {
    return false;
  }
  const Eigen::Vector3d wrist_k1 =
    column.value().orientation.conjugate() * (p_wrist_0 - column.value().position_m);
  const double reach = wrist_k1.head<2>().norm();
  const double gamma = std::atan2(wrist_k1.y(), wrist_k1.x());

  const AxisLimit & telescope = limits.axis[3];
  Closure best;
  const std::size_t samples = (fixed != nullptr) ? 1U : d45_samples;
  for (std::size_t index = 0; index < samples; ++index) {
    double q4 = 0.0;
    if (fixed != nullptr) {
      q4 = fixed->q4;
    } else {
      const double fraction =
        (samples == 1U) ? 0.0 : static_cast<double>(index) / static_cast<double>(samples - 1);
      q4 = telescope.lower + fraction * (telescope.upper - telescope.lower);
    }
    const double forearm = geometry.forearm_length(q4);
    // The law of cosines of 2.2 step 2 exists only inside the annulus the two
    // links span; outside it there is no closure at this extension at all.
    if (reach > geometry.a2 + forearm || reach < std::abs(geometry.a2 - forearm) ||
      reach <= 0.0)
    {
      continue;
    }
    const double cosine =
      std::min(1.0, std::max(-1.0, (reach * reach + geometry.a2 * geometry.a2 -
      forearm * forearm) / (2.0 * geometry.a2 * reach)));
    const double opening = std::acos(cosine);
    for (const double branch : {1.0, -1.0}) {
      if (fixed != nullptr && branch != fixed->branch) {
        continue;
      }
      const double boom_bearing = gamma + branch * opening;
      Closure candidate;
      candidate.q1 = q1;
      candidate.q2 = geometry.sign_q2 * wrap(boom_bearing - geometry.gamma0);
      const Eigen::Vector2d elbow =
        wrist_k1.head<2>() -
        geometry.a2 * Eigen::Vector2d(std::cos(boom_bearing), std::sin(boom_bearing));
      const double forearm_bearing = std::atan2(elbow.y(), elbow.x());
      candidate.q3 = geometry.sign_q3 *
        wrap(forearm_bearing - boom_bearing - geometry.forearm_bearing(q4));
      candidate.q4 = q4;
      candidate.branch = branch;
      if (!inside(limits.axis[1], candidate.q2) || !inside(limits.axis[2], candidate.q3)) {
        continue;
      }
      candidate.score = centring_score(limits, candidate);
      if (candidate.score < best.score) {
        best = candidate;
      }
    }
  }
  if (!(best.score < std::numeric_limits<double>::infinity())) {
    return false;
  }
  out = best;
  return true;
}

/// One residual evaluation: where the tool ends up, and where K5 ends up with it.
struct Evaluation
{
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};
  Eigen::Vector3d p_tip_0{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double residual_p{};
  double residual_phi_z{};
  double signed_phi_z_error{};
};

Status evaluate(
  const crane_model::Model & model, const IkRequest & request, Q & q, Evaluation & out)
{
  auto settled = model.passive_equilibrium(actuated(q), request.payload);
  if (!settled.ok()) {
    return settled.status();
  }
  const QU q_u = settled.value();
  q[4] = q_u[0];
  q[5] = q_u[1];

  auto tcp = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!tcp.ok()) {
    return tcp.status();
  }
  auto tip = model.forward_kinematics(q, Frame::MountingBase, Frame::Tip);
  if (!tip.ok()) {
    return tip.status();
  }
  out.p_tcp_0 = tcp.value().position_m;
  out.orientation = tcp.value().orientation;
  out.p_tip_0 = tip.value().position_m;
  out.residual_p = (out.p_tcp_0 - request.p_tcp_0).norm();
  out.signed_phi_z_error = wrap(phi_z_of(out.orientation) - request.phi_z_d);
  out.residual_phi_z = std::abs(out.signed_phi_z_error);
  return Status{};
}

/// How far one evaluation is from being accepted, in units of its own tolerance.
double merit(const IkSettings & settings, const Evaluation & evaluation)
{
  return std::max(
    evaluation.residual_p / settings.eps_pos, evaluation.residual_phi_z / settings.eps_yaw);
}

/// The mid-range configuration the first wrist correction is seeded from.
Q nominal_configuration(const JointLimits & limits, double q8)
{
  Q q = Q::Zero();
  for (std::size_t row = 0; row < 4U; ++row) {
    if (limits.axis[row].bounded) {
      q[static_cast<Eigen::Index>(row)] = limits.axis[row].centre();
    }
  }
  q[0] = 0.0;  // the azimuth is what step 1 answers; it is not a seed
  q[7] = q8;
  return q;
}

}  // namespace

double phi_z_of(const Eigen::Quaterniond & orientation)
{
  const Eigen::Matrix3d rotation = orientation.normalized().toRotationMatrix();
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

crane_model::Result<IkSolution> solve_inverse_kinematics(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  const IkSettings & settings, const IkRequest & request)
{
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

  // The wrist offset at a mid-range pose. It is only a seed -- step 4 replaces
  // it with the offset the solution actually has -- but seeding from the tool's
  // own length rather than from zero is what keeps a goal near the slewing axis
  // from starting on the far side of it.
  Q nominal = nominal_configuration(limits, request.q8);
  Evaluation nominal_evaluation;
  Status status = evaluate(model, request, nominal, nominal_evaluation);
  if (!status.ok()) {
    return Result<IkSolution>::failure(std::move(status));
  }
  const Eigen::Vector3d nominal_offset =
    nominal_evaluation.p_tcp_0 - nominal_evaluation.p_tip_0;

  IkSolution best;
  double best_merit = std::numeric_limits<double>::infinity();
  Eigen::Vector3d wrist = request.p_tcp_0 - nominal_offset;
  double q7 = 0.0;
  QU q_u = nominal.segment<2>(4);
  bool closed = false;

  const std::size_t passes = std::max<std::size_t>(1U, settings.redundancy_passes);
  for (std::size_t pass = 0; pass < passes; ++pass) {
    Closure resolved;
    bool resolved_valid = false;
    for (std::size_t iteration = 0; iteration < settings.fixed_point_iterations; ++iteration) {
      Closure candidate;
      if (!solve_closure(
          model, geometry, limits, settings.d45_samples, wrist,
          resolved_valid ? &resolved : nullptr, candidate))
      {
        break;
      }
      if (!resolved_valid) {
        resolved = candidate;
        resolved_valid = true;
      }
      closed = true;

      Q q = Q::Zero();
      q[0] = candidate.q1;
      q[1] = candidate.q2;
      q[2] = candidate.q3;
      q[3] = candidate.q4;
      q[4] = q_u[0];
      q[5] = q_u[1];
      q[6] = q7;
      q[7] = request.q8;
      Evaluation evaluation;
      status = evaluate(model, request, q, evaluation);
      if (!status.ok()) {
        return Result<IkSolution>::failure(std::move(status));
      }
      q_u = q.segment<2>(4);

      const double value = merit(settings, evaluation);
      if (value < best_merit) {
        best_merit = value;
        best.q = q;
        best.residual_p = evaluation.residual_p;
        best.residual_phi_z = evaluation.residual_phi_z;
        best.d45 = geometry.d45(candidate.q4);
        best.fixed_point_iterations = iteration + 1U;
      }
      // Step 4: the correction is the offset the *solution* has, not the one the
      // seed had. Step 5 rides with it -- the rotator takes whatever yaw is left.
      const Eigen::Vector3d corrected =
        request.p_tcp_0 - (evaluation.p_tcp_0 - evaluation.p_tip_0);
      const double movement = (corrected - wrist).norm();
      wrist = corrected;
      q7 -= evaluation.signed_phi_z_error;
      if (value <= 1.0) {
        break;
      }
      // A fixed point that has stopped moving has converged, whether or not it
      // converged onto the goal. Spending the rest of the budget on it would
      // only delay the residual being reported, and the Jacobian polish below is
      // what has a chance of closing what is left.
      if (movement < settings.eps_pos * kStagnation &&
        std::abs(evaluation.signed_phi_z_error) < settings.eps_yaw * kStagnation)
      {
        break;
      }
    }
    if (!closed) {
      return Result<IkSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the goal is outside the two-link workspace at every telescope extension the "
          "description allows, so robot_model 2.2 step 2 has no closure for it"));
    }
    if (best_merit <= 1.0) {
      break;
    }
    // Step 3 again, on the corrected target this time: the first pass resolved
    // the redundancy against a wrist point that was still a guess.
    Q q = best.q;
    Evaluation evaluation;
    status = evaluate(model, request, q, evaluation);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }
    wrist = request.p_tcp_0 - (evaluation.p_tcp_0 - evaluation.p_tip_0);
    q7 = q[6];
    q_u = q.segment<2>(4);
  }

  // 2.2's own admission is that steps 3 and 4 are a search and a fixed point,
  // and neither is guaranteed to converge. Where they leave a residual, the
  // Jacobian closes it: three columns of `Model::jacobian` against the three
  // position residuals, with the rotator and the equilibrium carried along and a
  // backtracking step so a direction that ignores dq_eq/dq_a cannot make things
  // worse. It runs zero times when the fixed point already met the tolerance.
  if (best_merit > 1.0 && closed) {
    Q q = best.q;
    Evaluation evaluation;
    status = evaluate(model, request, q, evaluation);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }
    double current = merit(settings, evaluation);
    for (std::size_t iteration = 0; iteration < settings.refinement_iterations; ++iteration) {
      if (current <= 1.0) {
        break;
      }
      auto jacobian = model.jacobian(q, Frame::Tcp);
      if (!jacobian.ok()) {
        break;
      }
      // `Model::jacobian` is expressed in the frame itself, so the linear block
      // is rotated into K0 before it is read as dp/dq.
      const Eigen::Matrix<double, 3, 8> linear =
        evaluation.orientation.toRotationMatrix() * jacobian.value().value.topRows<3>();
      Eigen::Matrix3d square;
      square.col(0) = linear.col(0);
      square.col(1) = linear.col(1);
      square.col(2) = linear.col(2);
      const Eigen::Vector3d residual = evaluation.p_tcp_0 - request.p_tcp_0;
      Eigen::Vector3d step =
        square.colPivHouseholderQr().solve(Eigen::Vector3d(-residual));
      if (!step.allFinite()) {
        break;
      }
      if (step.norm() > kMaxRefinementStep) {
        step *= kMaxRefinementStep / step.norm();
      }
      bool improved = false;
      double scale = 1.0;
      for (int attempt = 0; attempt < kBacktrackingSteps; ++attempt) {
        Q trial = q;
        trial[0] += scale * step[0];
        trial[1] += scale * step[1];
        trial[2] += scale * step[2];
        trial[6] -= evaluation.signed_phi_z_error;
        Evaluation candidate;
        const Status trial_status = evaluate(model, request, trial, candidate);
        if (trial_status.ok()) {
          const double value = merit(settings, candidate);
          if (value < current) {
            q = trial;
            evaluation = candidate;
            current = value;
            improved = true;
            break;
          }
        }
        scale *= 0.5;
      }
      if (!improved) {
        break;
      }
      best.refinement_iterations = iteration + 1U;
      if (current < best_merit) {
        best_merit = current;
        best.q = q;
        best.residual_p = evaluation.residual_p;
        best.residual_phi_z = evaluation.residual_phi_z;
      }
    }
  }

  if (best_merit > 1.0) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the semi-analytic solve of robot_model 2.2 did not converge onto the goal: it left " +
        std::to_string(best.residual_p) + " m of position residual against eps_pos = " +
        std::to_string(settings.eps_pos) + " m and " + std::to_string(best.residual_phi_z) +
        " rad of yaw residual against eps_yaw = " + std::to_string(settings.eps_yaw) + " rad"));
  }

  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const double value = best.q[static_cast<Eigen::Index>(kActuatedRows[row])];
    if (!inside(limits.axis[row], value)) {
      return Result<IkSolution>::failure(
        failure(
          ErrorCode::InvalidArgument,
          "the solved configuration puts an actuated joint outside the range the description "
          "gives it"));
    }
  }
  return Result<IkSolution>::success(std::move(best));
}

}  // namespace crane_planning
