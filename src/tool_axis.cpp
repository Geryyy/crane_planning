#include "crane_planning/tool_axis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Q;
using crane_model::QA;
using crane_model::Result;
using crane_model::Status;

constexpr double kTwoPi = 6.283185307179586;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

/// The canonical eight of one actuated configuration, with a stated passive pair.
Q canonical(const QA & q_a, const crane_model::QU & q_u)
{
  Q q = Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
  }
  q.segment<2>(4) = q_u;
  return q;
}

/// The tool row of `J_cyl` at one tool coordinate, everything else held.
Result<double> transmission_ratio(const crane_model::Model & model, const Q & q, double q8)
{
  Q probe = q;
  probe[static_cast<Eigen::Index>(kActuatedRows[kToolRow])] = q8;
  auto jacobian = model.cylinder_jacobian(probe);
  if (!jacobian.ok()) {
    return Result<double>::failure(jacobian.status());
  }
  const Eigen::Index row = static_cast<Eigen::Index>(kToolRow);
  return Result<double>::success(jacobian.value()(row, row));
}

}  // namespace

const char * tool_end_name(ToolEnd end) noexcept
{
  return end == ToolEnd::Lower ? "lower" : "upper";
}

double grip_cosine(double s) noexcept
{
  return s - std::sin(kTwoPi * s) / kTwoPi;
}

double grip_cosine_rate(double s) noexcept
{
  return 1.0 - std::cos(kTwoPi * s);
}

double grip_cosine_curvature(double s) noexcept
{
  return kTwoPi * std::sin(kTwoPi * s);
}

crane_model::Result<double> tool_axis_target(
  const JointLimits & limits, ToolEnd closed_end, bool closing)
{
  const AxisLimit & axis = limits.axis[kToolRow];
  if (!axis.bounded || !std::isfinite(axis.lower) || !std::isfinite(axis.upper) ||
    !(axis.upper > axis.lower))
  {
    return Result<double>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the description gives the tool axis no bounded range, so there is no end of it for a "
        "close or an open phase to drive to. Both machine descriptions in this workspace bound "
        "it -- 0.2 to 0.7 m on the PZS100's rail, 0 to 3.15 rad on the 7040's jaw -- and an "
        "invented end is not a worse answer than none, it is an answer that reads as a grip"));
  }
  // `closed_end` names the end that is a closed gripper; a close drives to it
  // and an open drives to the other one.
  const bool to_lower = (closed_end == ToolEnd::Lower) == closing;
  return Result<double>::success(to_lower ? axis.lower : axis.upper);
}

crane_model::Result<TransmissionCrossing> probe_tool_transmission(
  const crane_model::Model & model, const crane_model::Q & q, double q8_from, double q8_to,
  const ToolAxisSettings & settings)
{
  if (!std::isfinite(q8_from) || !std::isfinite(q8_to)) {
    return Result<TransmissionCrossing>::failure(
      failure(ErrorCode::NonFiniteInput, "the tool travel to probe is not finite"));
  }
  if (settings.transmission_samples < 2U || !(settings.transmission_floor >= 0.0)) {
    return Result<TransmissionCrossing>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the transmission probe needs at least two samples and a non-negative floor"));
  }

  const double lower = std::min(q8_from, q8_to);
  const double upper = std::max(q8_from, q8_to);
  const std::size_t samples = settings.transmission_samples;

  TransmissionCrossing crossing;
  crossing.smallest_ratio = std::numeric_limits<double>::infinity();
  double previous_q8 = lower;
  double previous = 0.0;
  for (std::size_t index = 0; index < samples; ++index) {
    const double fraction =
      static_cast<double>(index) / static_cast<double>(samples - 1U);
    const double q8 = lower + (upper - lower) * fraction;
    auto ratio = transmission_ratio(model, q, q8);
    if (!ratio.ok()) {
      return Result<TransmissionCrossing>::failure(ratio.status());
    }
    const double value = ratio.value();
    if (!std::isfinite(value)) {
      return Result<TransmissionCrossing>::failure(
        failure(
          ErrorCode::NonFiniteInput,
          "the tool axis's transmission ratio is not finite at q8 = " + std::to_string(q8)));
    }
    if (index == 0U) {
      crossing.ratio_lower = value;
    }
    crossing.ratio_upper = value;
    crossing.smallest_ratio = std::min(crossing.smallest_ratio, std::abs(value));

    const bool reversed = index > 0U && (value * previous < 0.0);
    const bool toggled = std::abs(value) <= settings.transmission_floor;
    if ((reversed || toggled) && !crossing.crossed) {
      crossing.crossed = true;
      crossing.q8 = q8;
      if (reversed) {
        // Bisect the bracket the grid just closed, so the answer is the axis's
        // own crossing and not the resolution this happened to be probed at.
        double left = previous_q8;
        double right = q8;
        double left_value = previous;
        for (int step = 0; step < 60; ++step) {
          const double middle = 0.5 * (left + right);
          auto probed = transmission_ratio(model, q, middle);
          if (!probed.ok()) {
            return Result<TransmissionCrossing>::failure(probed.status());
          }
          if (probed.value() * left_value < 0.0) {
            right = middle;
          } else {
            left = middle;
            left_value = probed.value();
          }
        }
        crossing.q8 = 0.5 * (left + right);
      }
    }
    previous = value;
    previous_q8 = q8;
  }
  return Result<TransmissionCrossing>::success(crossing);
}

crane_model::Result<ToolPhase> drive_tool_axis(
  const crane_model::Model & model, const JointLimits & limits,
  const ToolAxisSettings & settings, const ToolAxisRequest & request)
{
  {
    const Status status = check_margin_factor(request.speed_scale);
    if (!status.ok()) {
      return Result<ToolPhase>::failure(status);
    }
  }
  if (!(request.kappa > 0.0) || request.kappa > 1.0) {
    return Result<ToolPhase>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "kappa = " + std::to_string(request.kappa) +
        " is outside (0, 1]; trajectory_planning 5.5 makes it the fraction of the machine's "
        "authority the planner may spend, and the tool axis spends it on the same terms the arm "
        "does"));
  }
  if (!(request.sample_period > 0.0) || !(request.min_duration > 0.0)) {
    return Result<ToolPhase>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "a tool phase needs a positive sample period and a positive duration floor"));
  }
  if (!request.q_a_start.allFinite() || !std::isfinite(request.q8_goal)) {
    return Result<ToolPhase>::failure(
      failure(ErrorCode::NonFiniteInput, "the tool phase's start or target is not finite"));
  }

  const AxisLimit & axis = limits.axis[kToolRow];
  if (!(axis.dq_max > 0.0) || !std::isfinite(axis.dq_max)) {
    return Result<ToolPhase>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the description gives the tool axis no finite positive velocity limit, so there is "
        "nothing to time a grip against"));
  }
  const double ddq8_max = request.actuation.ddq_a_max[kToolRow];
  if (!(ddq8_max > 0.0) || !std::isfinite(ddq8_max)) {
    return Result<ToolPhase>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the tool axis has no finite positive acceleration limit (parameters.md 3)"));
  }
  if (!(request.actuation.pump_flow_max > 0.0) ||
    !(request.actuation.pump_flow_planning_factor > 0.0))
  {
    return Result<ToolPhase>::failure(
      failure(ErrorCode::InvalidArgument, "Q_P^max and its planning factor must be positive"));
  }

  const Eigen::Index tool = static_cast<Eigen::Index>(kToolRow);
  const double q8_start = request.q_a_start[tool];
  const double q8_goal = request.q8_goal;
  if (axis.bounded) {
    const std::array<std::pair<const char *, double>, 2U> ends{
      {{"start", q8_start}, {"goal", q8_goal}}};
    for (const std::pair<const char *, double> & named : ends) {
      if (named.second < axis.lower || named.second > axis.upper) {
        return Result<ToolPhase>::failure(
          failure(
            ErrorCode::InvalidArgument,
            std::string("the tool phase's ") + named.first + " puts q8 at " +
            std::to_string(named.second) + ", outside the [" + std::to_string(axis.lower) + ", " +
            std::to_string(axis.upper) + "] the description gives it"));
      }
    }
  }

  // Where the tool hangs at each end, with this phase's payload. Two calls, and
  // they are what makes "the payload is honoured" checkable: a payload held off
  // the tilt axis moves the passive pair (robot_model 5.1) and no actuated row
  // of the emitted reference shows that.
  auto settled_start = model.passive_equilibrium(request.q_a_start, request.payload);
  if (!settled_start.ok()) {
    return Result<ToolPhase>::failure(settled_start.status());
  }
  QA q_a_goal = request.q_a_start;
  q_a_goal[tool] = q8_goal;
  auto settled_goal = model.passive_equilibrium(q_a_goal, request.payload);
  if (!settled_goal.ok()) {
    return Result<ToolPhase>::failure(settled_goal.status());
  }

  const Q q_start = canonical(request.q_a_start, settled_start.value());
  const double travel = q8_goal - q8_start;
  const double direction = (travel > 0.0) ? 1.0 : ((travel < 0.0) ? -1.0 : 0.0);

  // The toggle of issue 037's notes, looked for before anything is timed. An
  // axis whose cylinder produces no joint motion is not an axis a velocity
  // reference can be written for, so a travel that spans the crossing is refused
  // rather than clipped to it.
  auto crossing = probe_tool_transmission(model, q_start, q8_start, q8_goal, settings);
  if (!crossing.ok()) {
    return Result<ToolPhase>::failure(crossing.status());
  }
  if (crossing.value().crossed) {
    const double reachable = crossing.value().q8;
    return Result<ToolPhase>::failure(
      failure(
        ErrorCode::SingularConfiguration,
        "the tool axis's transmission ratio reverses sign at q8 = " + std::to_string(reachable) +
        ", between this phase's start at " + std::to_string(q8_start) + " and its target at " +
        std::to_string(q8_goal) + " (the ratio runs from " +
        std::to_string(crossing.value().ratio_lower) + " to " +
        std::to_string(crossing.value().ratio_upper) +
        " across the travel, and its smallest magnitude on the way is " +
        std::to_string(crossing.value().smallest_ratio) +
        "). At the crossing J_c,GR = 0: the cylinder moves the jaw through no distance at all, so "
        "the axis is not controllable there and no cylinder velocity produces the commanded joint "
        "velocity. Issue 037's notes are where this reversal was first recorded, on the 7040 jaw, "
        "from a six-point table that puts it near q8 = 0.29 rad; the number above is the same root "
        "bisected off the model. Either way it is well inside the 0 to 3.15 rad the description "
        "gives that joint. The phase is refused and not clipped: the largest travel on this "
        "start's own side of the crossing ends at q8 = " +
        std::to_string(reachable) +
        ", and a grip that stopped there and reported success would read as a grip that closed"));
  }

  // The flow of mpc 3 constraint 7 for this one axis, at unit rate. `Q = A^{+-}(v)
  // |v|` is exactly linear in |v| for a fixed direction of travel, and the
  // direction is fixed because the primitive is monotone and the ratio does not
  // reverse -- so one evaluation per sample settles the whole phase, and a close
  // and an open over the same span differ because the two draw on different
  // chambers.
  double flow_per_unit_rate = 0.0;
  double smallest_ratio = crossing.value().smallest_ratio;
  if (direction != 0.0) {
    crane_model::DQA dq_a = crane_model::DQA::Zero();
    dq_a[tool] = direction;
    crane_model::ChamberPressure quiescent;
    quiescent.p_a_pa.setZero();
    quiescent.p_b_pa.setZero();

    const std::size_t samples = settings.transmission_samples;
    const double lower = std::min(q8_start, q8_goal);
    const double upper = std::max(q8_start, q8_goal);
    for (std::size_t index = 0; index < samples; ++index) {
      const double fraction =
        static_cast<double>(index) / static_cast<double>(samples - 1U);
      Q probe = q_start;
      probe[static_cast<Eigen::Index>(kActuatedRows[kToolRow])] =
        lower + (upper - lower) * fraction;
      auto transmission = model.transmission(probe, dq_a, quiescent);
      if (!transmission.ok()) {
        return Result<ToolPhase>::failure(transmission.status());
      }
      const double flow = transmission.value().pump_flow[tool];
      if (!std::isfinite(flow) || flow < 0.0) {
        return Result<ToolPhase>::failure(
          failure(
            ErrorCode::NonFiniteInput,
            "the tool axis's pump draw is not a finite non-negative number at q8 = " +
            std::to_string(probe[static_cast<Eigen::Index>(kActuatedRows[kToolRow])])));
      }
      flow_per_unit_rate = std::max(flow_per_unit_rate, flow);
    }
  }

  // The three limits, and the floor. kappa scales every physical one; speed_scale
  // multiplies the velocity bound alone, exactly as it does in the OCP, so no
  // value of it reaches into the margin.
  const double rate_max = request.kappa * request.speed_scale * axis.dq_max;
  const double acceleration_max = request.kappa * ddq8_max;
  const double flow_max = request.kappa * request.actuation.pump_flow_planning_factor *
    request.actuation.pump_flow_max;
  const double span = std::abs(travel);

  const double duration_velocity = span * kGripCosinePeakRate / rate_max;
  const double duration_acceleration =
    std::sqrt(span * kGripCosinePeakCurvature / acceleration_max);
  const double duration_flow = (flow_per_unit_rate > 0.0) ?
    (span * kGripCosinePeakRate * flow_per_unit_rate / flow_max) : 0.0;

  double duration = request.min_duration;
  ToolAxisDemand demand;
  demand.duration_floor_binding = true;
  const auto raise = [&duration, &demand](double candidate, bool ToolAxisDemand::* which) {
      if (candidate > duration) {
        duration = candidate;
        demand.velocity_binding = false;
        demand.acceleration_binding = false;
        demand.flow_binding = false;
        demand.duration_floor_binding = false;
        demand.*which = true;
      }
    };
  raise(duration_velocity, &ToolAxisDemand::velocity_binding);
  raise(duration_acceleration, &ToolAxisDemand::acceleration_binding);
  raise(duration_flow, &ToolAxisDemand::flow_binding);
  if (!std::isfinite(duration) || !(duration > 0.0)) {
    return Result<ToolPhase>::failure(
      failure(ErrorCode::NonFiniteInput, "the tool phase's duration is not a positive number"));
  }

  ToolPhase phase;
  phase.q8_start = q8_start;
  phase.q8_goal = q8_goal;
  phase.closed_end = settings.closed_end;
  phase.closing = (settings.closed_end == ToolEnd::Lower) ? (travel < 0.0) : (travel > 0.0);
  phase.q_u_start = settled_start.value();
  phase.q_u_goal = settled_goal.value();

  demand.travel = travel;
  demand.duration = duration;
  demand.peak_rate = span * kGripCosinePeakRate / duration;
  demand.peak_acceleration = span * kGripCosinePeakCurvature / (duration * duration);
  demand.flow_per_unit_rate = flow_per_unit_rate;
  demand.peak_flow = flow_per_unit_rate * demand.peak_rate;
  demand.smallest_ratio = smallest_ratio;
  demand.velocity_fraction = demand.peak_rate / axis.dq_max;
  demand.acceleration_fraction = demand.peak_acceleration / ddq8_max;
  demand.flow_fraction = demand.peak_flow / request.actuation.pump_flow_max;
  phase.demand = demand;

  // The emitted reference, on the arm's own grid: the same `sample_period` the
  // OCP resamples onto, the first point at zero and the last exactly at the
  // duration, so `h(1) = 1` lands the tool coordinate on its target and not near
  // it. The five path coordinates are held, because a grip does not move the arm
  // (trajectory_planning 4.1) -- and they are *emitted* rather than omitted,
  // because the reference is over all six actuated rows and a controller handed
  // five would have to invent the sixth.
  const std::size_t points = std::max<std::size_t>(
    1U, static_cast<std::size_t>(std::ceil(duration / request.sample_period)));
  TimedTrajectory & trajectory = phase.trajectory;
  trajectory.duration = duration;
  trajectory.time_from_start.reserve(points + 1U);
  trajectory.q_a_ref.reserve(points + 1U);
  trajectory.dq_a_ref.reserve(points + 1U);
  for (std::size_t index = 0; index <= points; ++index) {
    const double time =
      std::min(duration, static_cast<double>(index) * request.sample_period);
    const double s = time / duration;
    QA position = request.q_a_start;
    position[tool] = q8_start + travel * grip_cosine(s);
    crane_model::DQA velocity = crane_model::DQA::Zero();
    velocity[tool] = travel * grip_cosine_rate(s) / duration;
    trajectory.time_from_start.push_back(time);
    trajectory.q_a_ref.push_back(position);
    trajectory.dq_a_ref.push_back(velocity);
  }
  // The same convention `scaled_ramp` and the OCP report: the peak the emitted
  // samples reach, against the bound this phase was scaled to.
  trajectory.limiting_fraction = demand.peak_rate / rate_max;

  return Result<ToolPhase>::success(std::move(phase));
}

std::string describe(const ToolPhase & phase)
{
  std::string text = "the tool coordinate runs from " + std::to_string(phase.q8_start) + " to " +
    std::to_string(phase.q8_goal) + " -- the " + tool_end_name(phase.closed_end) +
    " end of the range the description gives it is the closed gripper, so this phase is " +
    (phase.closing ? "a close" : "an open") +
    " -- on the retained cosine primitive of trajectory_planning 8, carried on the rate so the "
    "tool coordinate is C2 at both ends like the five path coordinates beside it, over " +
    std::to_string(phase.trajectory.duration) + " s of the arm's own time base";
  text += ". Of the tool axis's own physical limits the peak demand is " +
    std::to_string(phase.demand.velocity_fraction) + " of joint velocity, " +
    std::to_string(phase.demand.acceleration_fraction) + " of joint acceleration and " +
    std::to_string(phase.demand.flow_fraction) + " of pump flow, at " +
    std::to_string(phase.demand.peak_flow) + " m^3/s through a transmission whose smallest ratio "
    "over the travel is " + std::to_string(phase.demand.smallest_ratio) + ". The duration was set "
    "by ";
  if (phase.demand.velocity_binding) {
    text += "the velocity limit";
  } else if (phase.demand.acceleration_binding) {
    text += "the acceleration limit";
  } else if (phase.demand.flow_binding) {
    text += "the pump flow";
  } else {
    text += "the duration floor, which none of the three limits reached";
  }
  return text;
}

}  // namespace crane_planning
