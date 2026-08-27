#include "crane_planning/inverse_kinematics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "crane_planning/collision.hpp"
#include "crane_planning/redundancy.hpp"
#include "crane_planning/status.hpp"

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::QA;
using crane_model::Result;
using crane_model::Status;

/// The step a derivative is read at, rad -- the rotator's yaw sensitivity and
/// the polish's numerical Jacobian both use it.
constexpr double kProbeStep = 1.0e-4;

/// Below this the rotator does not turn the tool's yaw and step 5 has no answer.
constexpr double kMinYawSensitivity = 1.0e-3;

/// Largest step the Jacobian polish may take in one iteration, rad.
constexpr double kMaxRefinementStep = 0.5;

/// How often the polish halves a step that did not improve the residual.
constexpr int kBacktrackingSteps = 6;

/// Fraction of a tolerance below which the wrist fixed point counts as stalled.
constexpr double kStagnation = 0.1;

/// How heavily a closure that had to be clamped onto the annulus is penalised.
/**
 * The score of 2.2 step 3 is joint-range centring, which is order one. A metre
 * of unreachability has to outweigh any amount of it, or the search would pick a
 * well-centred arm that does not reach the goal over a reaching one.
 */
constexpr double kDeficitWeight = 1.0e4;

/// The rows of the canonical eight the Jacobian polish is free to move.
/**
 * q1, q2, q3 and the rotator: four unknowns against the three position residuals
 * and the yaw, which is square. q4 is deliberately not among them -- the
 * telescope is the redundancy, and 2.2 step 3 has already resolved it.
 */
constexpr std::array<Eigen::Index, 4> kRefinedRows{{0, 1, 2, 6}};

/// One candidate of the step-3 search, before it has been checked against FK.
struct Closure
{
  double q1{};
  double q2{};
  double q3{};
  double q4{};
  double branch{};   ///< +1 or -1, the `q2 = gamma +- acos(.)` of 2.2 step 2
  double deficit{};  ///< how far outside the two-link annulus the target was, m
  double score{std::numeric_limits<double>::infinity()};
};

/// The four axes the closure scores, which is 2.2 step 3's three plus the azimuth.
/**
 * q1 is in the sum here and is *not* in `kRedundantAxes`, and both are right.
 * The search this scores holds q1 fixed across every candidate -- step 1 has
 * already answered it -- so its term is a constant that shifts every score
 * equally and orders nothing. Dropping it would change the numbers this route
 * reports without changing any decision it makes, so it stays.
 */
constexpr std::array<std::size_t, 4> kClosureAxes{{0, 1, 2, 3}};

/// The joint-range centring of 2.2 step 3, from the one place it is written down.
double centring_score(const JointLimits & limits, const Closure & candidate)
{
  return crane_planning::centring_score(
    limits, kClosureAxes,
    std::array<double, 4>{{candidate.q1, candidate.q2, candidate.q3, candidate.q4}});
}

/// What the clearance half of 2.2 step 3's score is evaluated against.
/**
 * The passive pair is **pinned** here, at the equilibrium of the configuration
 * the solve started from, rather than re-solved per candidate: one
 * `Model::passive_equilibrium` costs some 45 ms against 0.35 ms for a forward
 * kinematics call, and forty-one of them per solve would cost more than the
 * whole endpoint NLP. The pin is honest because this score is a *preference*
 * between extensions and not the safety check -- `check_path` settles the pair
 * exactly at every configuration it checks, and it is what refuses a path.
 */
struct ClearanceContext
{
  const crane_model::CollisionScene * scene{nullptr};
  RedundancyWeights weights{};
  Eigen::Vector2d q_u{Eigen::Vector2d::Zero()};
  double q7{};
  double q8{};

  [[nodiscard]] bool votes() const noexcept
  {
    return scene != nullptr && weights.clearance > 0.0;
  }
};

/// The clearance term of the score, or zero when there is no scene to score on.
double clearance_score(
  const crane_model::Model & model, const ClearanceContext & clearance, const Closure & candidate)
{
  if (!clearance.votes()) {
    return 0.0;
  }
  Q q = Q::Zero();
  q[0] = candidate.q1;
  q[1] = candidate.q2;
  q[2] = candidate.q3;
  q[3] = candidate.q4;
  q.segment<2>(4) = clearance.q_u;
  q[6] = clearance.q7;
  q[7] = clearance.q8;

  auto distance = clearance_at(model, *clearance.scene, q);
  if (!distance.ok()) {
    // A description with no collision geometry is a model that answers every
    // other question perfectly well (crane_model's README), so the redundancy
    // falls back to centring alone rather than the whole solve failing on a
    // preference it could not evaluate.
    return 0.0;
  }
  return clearance.weights.clearance *
         clearance_penalty(distance.value(), clearance.weights.clearance_reference_m);
}
/// How many whole turns either way a representative is looked for.
constexpr int kTurns = 2;

/// The turn of an angle that the description's own range admits.
/**
 * `wrap` lands in (-pi, pi], and both machine descriptions in this workspace give
 * a revolute axis a range wider than that: theta3 runs [-0.91, 4.6] rad and
 * theta1 [-3.71, 3.71]. An angle is only defined up to a whole turn, so the
 * closure's wrapped answer and the value the joint can actually hold are
 * sometimes a turn apart -- and range-checking the wrapped one alone refuses
 * configurations the machine reaches. Measured over a sweep of 120 poses per
 * tool, that was three refusals on the 7040, every one of them a pose generated
 * by forward kinematics from a configuration the description allows.
 *
 * The turn nearest the range's centre wins where more than one is inside, which
 * is the same joint-range centring 2.2 step 3 scores the redundancy by. An angle
 * no turn of which is inside is returned wrapped, so the caller's own range check
 * refuses it and says so.
 */
double representative(const AxisLimit & axis, double angle)
{
  if (!axis.bounded) {
    return angle;
  }
  double best = angle;
  double best_distance = std::numeric_limits<double>::infinity();
  for (int turn = -kTurns; turn <= kTurns; ++turn) {
    const double candidate = angle + static_cast<double>(turn) * 2.0 * M_PI;
    if (!axis.contains(candidate)) {
      continue;
    }
    const double distance = std::abs(candidate - axis.centre());
    if (distance < best_distance) {
      best_distance = distance;
      best = candidate;
    }
  }
  return best;
}

/// Why steps 1 to 3 could not produce a candidate at all.
enum class ClosureFailure
{
  None,
  Azimuth,   ///< the goal's azimuth is outside the slewing range
  Degenerate,  ///< the goal sits on the slewing column, where step 1 has no bearing
  NoCandidate  ///< no extension puts q2 and q3 inside their ranges
};

/// Steps 1 to 3: the azimuth, the planar closure, and the search that picks d45.
/**
 * `fixed` holds the extension and the elbow branch a previous iteration resolved
 * on. Freezing them is what makes step 4 a fixed-point iteration at all: a search
 * re-run inside the loop can jump between two extensions that are equally well
 * centred, and the correction then oscillates between them for ever instead of
 * converging.
 *
 * A target outside the two-link annulus is **clamped onto it** rather than
 * skipped, and how far it had to be clamped is carried on the candidate. A
 * skipped extension leaves the fixed point with nothing to correct from, so a
 * goal whose first wrist estimate is a few centimetres out of reach would be
 * refused as unreachable when one iteration would have walked into range. The
 * clamp never hides anything: the deficit dominates the score, and the answer is
 * still put through the forward-kinematics oracle before it is returned.
 */
ClosureFailure solve_closure(
  const crane_model::Model & model, const ArmGeometry & geometry, const JointLimits & limits,
  std::size_t d45_samples, const Eigen::Vector3d & p_wrist_0, const Closure * fixed,
  const ClearanceContext & clearance, Closure & out)
{
  const double q1 = representative(limits.axis[0], std::atan2(p_wrist_0.y(), p_wrist_0.x()));
  if (!limits.axis[0].contains(q1)) {
    return ClosureFailure::Azimuth;
  }
  Q q = Q::Zero();
  q[0] = q1;
  auto column = model.forward_kinematics(q, Frame::MountingBase, Frame::SlewingColumn);
  if (!column.ok()) {
    return ClosureFailure::Degenerate;
  }
  const Eigen::Vector3d wrist_k1 =
    column.value().orientation.conjugate() * (p_wrist_0 - column.value().position_m);
  const double reach = wrist_k1.head<2>().norm();
  if (!(reach > 0.0)) {
    return ClosureFailure::Degenerate;
  }
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
    // links span. Outside it the closure is taken at the nearest point of the
    // annulus and the distance is carried, so the search prefers an extension
    // that reaches over one that does not.
    const double lower = std::abs(geometry.a2 - forearm);
    const double upper = geometry.a2 + forearm;
    const double closable = std::min(upper, std::max(lower, reach));
    if (!(closable > 0.0)) {
      continue;
    }
    const double cosine = std::min(
      1.0,
      std::max(
        -1.0,
        (closable * closable + geometry.a2 * geometry.a2 - forearm * forearm) /
        (2.0 * geometry.a2 * closable)));
    const double opening = std::acos(cosine);
    for (const double branch : {1.0, -1.0}) {
      if (fixed != nullptr && branch != fixed->branch) {
        continue;
      }
      const double boom_bearing = gamma + branch * opening;
      Closure candidate;
      candidate.q1 = q1;
      candidate.q2 =
        representative(limits.axis[1], geometry.sign_q2 * wrap(boom_bearing - geometry.gamma0));
      const Eigen::Vector2d elbow =
        closable * Eigen::Vector2d(std::cos(gamma), std::sin(gamma)) -
        geometry.a2 * Eigen::Vector2d(std::cos(boom_bearing), std::sin(boom_bearing));
      const double forearm_bearing = std::atan2(elbow.y(), elbow.x());
      candidate.q3 = representative(
        limits.axis[2],
        geometry.sign_q3 * wrap(forearm_bearing - boom_bearing - geometry.forearm_bearing(q4)));
      candidate.q4 = q4;
      candidate.branch = branch;
      candidate.deficit = std::abs(reach - closable);
      if (!limits.axis[1].contains(candidate.q2) || !limits.axis[2].contains(candidate.q3)) {
        continue;
      }
      // 2.2 step 3's score, both halves of it: joint-range centring and
      // collision clearance, with the reach deficit dominating either.
      candidate.score = centring_score(limits, candidate) +
        kDeficitWeight * candidate.deficit * candidate.deficit +
        clearance_score(model, clearance, candidate);
      if (candidate.score < best.score) {
        best = candidate;
      }
    }
  }
  if (!(best.score < std::numeric_limits<double>::infinity())) {
    return ClosureFailure::NoCandidate;
  }
  out = best;
  return ClosureFailure::None;
}

/// Where one configuration puts the tool, with the passive pair as it stands.
/**
 * Forward kinematics only. Whoever calls this owns the pin in `q[4]`, `q[5]`.
 */
struct Placement
{
  Eigen::Vector3d p_tcp{Eigen::Vector3d::Zero()};
  Eigen::Vector3d p_tip{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double phi_z{};
};

Status place(const crane_model::Model & model, const Q & q, Placement & out)
{
  auto tcp = model.forward_kinematics(q, Frame::MountingBase, Frame::Tcp);
  if (!tcp.ok()) {
    return tcp.status();
  }
  auto tip = model.forward_kinematics(q, Frame::MountingBase, Frame::Tip);
  if (!tip.ok()) {
    return tip.status();
  }
  out.p_tcp = tcp.value().position_m;
  out.p_tip = tip.value().position_m;
  out.orientation = tcp.value().orientation;
  out.phi_z = phi_z_of(out.orientation);
  return Status{};
}


/// Where one configuration puts the tool, with the pair at *its own* equilibrium.
/**
 * The pin of 2.2, evaluated as the function it is. This is the only expensive
 * call in a solve -- see the file header for why holding it instead does not
 * converge -- so everything below counts them and spends as few as it can.
 */
Status settle(
  const crane_model::Model & model, const crane_model::Payload & payload, Q & q,
  Placement & out, std::size_t & calls)
{
  ++calls;
  auto settled = model.passive_equilibrium(actuated(q), payload);
  if (!settled.ok()) {
    return settled.status();
  }
  q.segment<2>(4) = settled.value();
  return place(model, q, out);
}

/// How far one placement is from being accepted, in units of its own tolerance.
double merit(const IkSettings & settings, const IkRequest & request, const Placement & placement)
{
  return std::max(
    (placement.p_tcp - request.p_tcp_0).norm() / settings.eps_pos,
    std::abs(wrap(placement.phi_z - request.phi_z_d)) / settings.eps_yaw);
}

/// The four-vector the polish drives to zero: the position residual, then the yaw.
Eigen::Vector4d residual_of(const IkRequest & request, const Placement & placement)
{
  Eigen::Vector4d residual;
  residual.head<3>() = placement.p_tcp - request.p_tcp_0;
  residual[3] = wrap(placement.phi_z - request.phi_z_d);
  return residual;
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

/// The Jacobian polish, on the same settled map the acceptance test is measured on.
/**
 * 2.2's own admission is that steps 3 and 4 are a scalar search and a fixed
 * point, and that neither is guaranteed to converge. Where they leave a residual,
 * this closes it: a numerical Jacobian of the four residuals against the four
 * rows of `kRefinedRows`, a backtracking line search on the true residual so a
 * step can never make the answer worse, and the description's own joint ranges
 * enforced on every trial. It runs zero times when the fixed point already met
 * the tolerance, and how many times it did run travels with the answer.
 *
 * Every evaluation re-settles the pair, so the Jacobian is the derivative of the
 * *composed* map `q_a -> p_tcp(q_a, q_eq(q_a))`. A polish that held the pair
 * instead would converge onto a configuration that is only correct while the tool
 * hangs at an angle it will not hang at, which is the failure the file header
 * describes.
 *
 * It is numerical rather than a `Model::jacobian` call for two reasons. The
 * fourth residual is a yaw and not an angular velocity, so the analytic
 * Jacobian's omega would need the Euler-rate map, a second parametrisation to
 * keep true to `phi_z_of`; and the analytic Jacobian carries no `dq_eq/dq_a`, so
 * its actuated columns are the held-pin derivative and not this one.
 */
Status refine(
  const crane_model::Model & model, const JointLimits & limits, const IkSettings & settings,
  const IkRequest & request, Q & q, Placement & placement, std::size_t & iterations,
  std::size_t & calls)
{
  double current = merit(settings, request, placement);
  for (std::size_t iteration = 0; iteration < settings.refinement_iterations; ++iteration) {
    if (current <= 1.0) {
      return Status{};
    }
    const Eigen::Vector4d residual = residual_of(request, placement);
    Eigen::Matrix4d jacobian;
    for (std::size_t column = 0; column < kRefinedRows.size(); ++column) {
      Q probe = q;
      probe[kRefinedRows[column]] += kProbeStep;
      Placement probed;
      const Status status = settle(model, request.payload, probe, probed, calls);
      if (!status.ok()) {
        return status;
      }
      jacobian.col(static_cast<Eigen::Index>(column)) =
        (residual_of(request, probed) - residual) / kProbeStep;
    }
    Eigen::Vector4d step = jacobian.colPivHouseholderQr().solve(Eigen::Vector4d(-residual));
    if (!step.allFinite()) {
      return Status{};
    }
    if (step.norm() > kMaxRefinementStep) {
      step *= kMaxRefinementStep / step.norm();
    }

    bool improved = false;
    double scale = 1.0;
    for (int attempt = 0; attempt < kBacktrackingSteps; ++attempt) {
      Q trial = q;
      for (std::size_t column = 0; column < kRefinedRows.size(); ++column) {
        trial[kRefinedRows[column]] += scale * step[static_cast<Eigen::Index>(column)];
      }
      if (limits.contains(trial)) {
        Placement candidate;
        const Status status = settle(model, request.payload, trial, candidate, calls);
        if (!status.ok()) {
          return status;
        }
        const double value = merit(settings, request, candidate);
        if (value < current) {
          q = trial;
          placement = candidate;
          current = value;
          improved = true;
          break;
        }
      }
      scale *= 0.5;
    }
    if (!improved) {
      return Status{};
    }
    iterations = iteration + 1U;
  }
  return Status{};
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
  if (!limits.axis[5].contains(request.q8)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the tool coordinate q8 = " + std::to_string(request.q8) +
        " is outside the range the description gives it"));
  }

  std::size_t calls = 0;

  // The wrist offset at a mid-range pose, with the pair hanging where that pose
  // leaves it. It is only a seed -- step 4 replaces it with the offset the
  // solution actually has -- but seeding from the tool's own hanging length
  // rather than from zero is what keeps a goal near the slewing axis from
  // starting on the far side of it.
  {
    const Status weights = check_redundancy_weights(settings.redundancy);
    if (!weights.ok()) {
      return Result<IkSolution>::failure(weights);
    }
  }

  Q q = nominal_configuration(limits, request.q8);
  Placement placement;
  Status status = settle(model, request.payload, q, placement, calls);
  if (!status.ok()) {
    return Result<IkSolution>::failure(std::move(status));
  }
  Eigen::Vector3d wrist = request.p_tcp_0 - (placement.p_tcp - placement.p_tip);

  // The pin the clearance half of step 3's score is evaluated at: where the tool
  // hangs at the seed. See `ClearanceContext` for why it is not re-solved per
  // candidate.
  ClearanceContext clearance;
  clearance.scene = request.scene;
  clearance.weights = settings.redundancy;
  clearance.q_u = q.segment<2>(4);
  clearance.q8 = request.q8;
  double q7 = 0.0;
  double sensitivity = 0.0;
  bool sensitivity_known = false;

  IkSolution best;
  Q best_q = Q::Zero();
  Placement best_placement;
  double best_merit = std::numeric_limits<double>::infinity();
  bool closed = false;
  bool closed_exactly = false;
  ClosureFailure refusal = ClosureFailure::NoCandidate;

  // Step 3's answer, resolved once and then held. Re-running the scalar search on
  // every iteration is what makes the fixed point wander: two telescope
  // extensions can be equally well centred, the search picks first one and then
  // the other as the wrist target moves by a millimetre, and the arm jumps
  // between two quite different configurations. 2.2 resolves the redundancy once
  // and then *corrects*; so does this. It can be resolved on the first iteration
  // because the seed's wrist target is already the goal raised by the tool's
  // hanging length, which is within a centimetre of the final one.
  Closure resolved;
  bool resolved_valid = false;

  const std::size_t budget = std::max<std::size_t>(1U, settings.fixed_point_iterations);
  std::size_t fixed_point_iterations = 0;
  for (std::size_t iteration = 0; iteration < budget; ++iteration) {
    // ---- steps 1 to 3: the azimuth, the planar closure, the telescope.
    Closure candidate;
    clearance.q7 = q7;
    const ClosureFailure closure = solve_closure(
      model, geometry, limits, settings.d45_samples, wrist,
      resolved_valid ? &resolved : nullptr, clearance, candidate);
    if (closure != ClosureFailure::None) {
      if (!closed) {
        refusal = closure;
      }
      break;
    }
    if (!resolved_valid) {
      resolved = candidate;
      resolved_valid = true;
    }
    closed = true;
    closed_exactly = closed_exactly || candidate.deficit <= 0.0;
    fixed_point_iterations = iteration + 1U;

    q = Q::Zero();
    q[0] = candidate.q1;
    q[1] = candidate.q2;
    q[2] = candidate.q3;
    q[3] = candidate.q4;
    q[6] = q7;
    q[7] = request.q8;
    status = settle(model, request.payload, q, placement, calls);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }

    // ---- step 5, at the sensitivity the description actually has. Whether a
    // positive rotator advances or retards the tool's yaw, and by how much when
    // the pair hangs tilted, is a property of the frames and is read once per
    // solve -- assuming it is +1 turns the correction into a divergence on any
    // description that mounts the rotator the other way up.
    if (!sensitivity_known) {
      Q probe = q;
      probe[6] += kProbeStep;
      Placement probed;
      status = settle(model, request.payload, probe, probed, calls);
      if (!status.ok()) {
        return Result<IkSolution>::failure(std::move(status));
      }
      sensitivity = wrap(probed.phi_z - placement.phi_z) / kProbeStep;
      if (std::abs(sensitivity) < kMinYawSensitivity) {
        return Result<IkSolution>::failure(
          failure(
            ErrorCode::SingularConfiguration,
            "the rotator does not turn the tool's yaw at this configuration, so step 5 of "
            "robot_model 2.2 has no residual yaw to take"));
      }
      sensitivity_known = true;
    }

    const double yaw_error = wrap(placement.phi_z - request.phi_z_d);
    // Clamped, because a rotator range the description states is a range the
    // solution has to be inside anyway; letting the correction walk out of it and
    // refusing at the end would throw away the reachable part of the goal.
    q7 = limits.axis[4].clamp(q7 - yaw_error / sensitivity);
    q[6] = q7;
    status = settle(model, request.payload, q, placement, calls);
    if (!status.ok()) {
      return Result<IkSolution>::failure(std::move(status));
    }

    const double value = merit(settings, request, placement);
    if (value < best_merit) {
      best_merit = value;
      best_q = q;
      best_placement = placement;
      best.d45 = geometry.d45(q[3]);
      best.fixed_point_iterations = fixed_point_iterations;
    }

    // ---- step 4: the correction is the offset the *settled solution* has, not
    // the one the seed had.
    const Eigen::Vector3d corrected = request.p_tcp_0 - (placement.p_tcp - placement.p_tip);
    const double movement = (corrected - wrist).norm();
    wrist = corrected;
    // A fixed point that has stopped moving has converged, whether or not it
    // converged onto the goal. Spending the rest of the budget on it would only
    // delay the residual being reported, and the polish below is what has a
    // chance of closing what is left.
    if (movement < settings.eps_pos * kStagnation &&
      std::abs(yaw_error) < settings.eps_yaw * kStagnation)
    {
      break;
    }
  }

  if (!closed) {
    std::string why;
    switch (refusal) {
      case ClosureFailure::Azimuth:
        why = "its azimuth is outside the range the description gives the slewing joint";
        break;
      case ClosureFailure::Degenerate:
        why = "it sits on the slewing column, where step 1's azimuth is not a number";
        break;
      default:
        why = "no telescope extension the description allows puts q2 and q3 inside their own "
          "ranges, so step 2 has no closure for it";
        break;
    }
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the goal is outside the workspace of robot_model 2.2: " + why +
        ". No configuration was found at all, so there is no forward-kinematics residual to "
        "report against it"));
  }

  q = best_q;
  placement = best_placement;
  std::size_t refinement_iterations = 0;
  status = refine(
    model, limits, settings, request, q, placement, refinement_iterations, calls);
  if (!status.ok()) {
    return Result<IkSolution>::failure(std::move(status));
  }

  best.q = q;
  best.route = EndpointRoute::SemiAnalytic;
  best.residual_p = (placement.p_tcp - request.p_tcp_0).norm();
  best.residual_phi_z = std::abs(wrap(placement.phi_z - request.phi_z_d));
  best.d45 = geometry.d45(q[3]);
  best.refinement_iterations = refinement_iterations;
  best.equilibrium_calls = calls;

  if (merit(settings, request, placement) > 1.0) {
    // The two are separate sentences because they are separate answers: the first
    // is how far the solve missed by, the second is that it was never going to
    // arrive, and a caller deciding whether to retry with a different goal needs
    // to be able to tell them apart.
    const std::string unreachable = closed_exactly ?
      std::string() :
      std::string(
      ". No telescope extension the description allows closes on this goal at any elbow branch, "
      "so it is outside the two-link workspace of 2.2 step 2 and the configuration above is the "
      "nearest reach rather than a solution");
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the semi-analytic solve of robot_model 2.2 did not converge onto the goal: it left " +
        std::to_string(best.residual_p) + " m of position residual against eps_pos = " +
        std::to_string(settings.eps_pos) + " m and " + std::to_string(best.residual_phi_z) +
        " rad of yaw residual against eps_yaw = " + std::to_string(settings.eps_yaw) + " rad" +
        unreachable));
  }
  if (!limits.contains(best.q)) {
    return Result<IkSolution>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the solved configuration puts an actuated joint outside the range the description "
        "gives it"));
  }
  return Result<IkSolution>::success(std::move(best));
}

}  // namespace crane_planning
