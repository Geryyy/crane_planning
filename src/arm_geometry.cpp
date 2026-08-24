#include "crane_planning/arm_geometry.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Frame;
using crane_model::Q;
using crane_model::Result;
using crane_model::Status;

/// How far a probe steps a joint to read the sign of its bearing derivative.
constexpr double kSignProbe = 1.0e-3;

Status failure(ErrorCode code, std::string message)
{
  return Status{code, std::move(message)};
}

double wrap(double angle)
{
  return std::remainder(angle, 2.0 * M_PI);
}

/// The configuration a probe runs at: every coordinate zero but the telescope.
/**
 * The passive pair, the rotator and the tool are all *downstream* of K5, so no
 * value of theirs moves any frame this probe reads. Zero is therefore not a
 * choice about them, it is the absence of one.
 */
Q probe_configuration(double q4)
{
  Q q = Q::Zero();
  q[3] = q4;
  return q;
}

/// The planar (x, y) of a frame's origin in K1, and its out-of-plane component.
Status planar_origin(
  const crane_model::Model & model, const Q & q, Frame frame, Eigen::Vector2d & planar,
  double & out_of_plane)
{
  auto pose = model.forward_kinematics(q, Frame::SlewingColumn, frame);
  if (!pose.ok()) {
    return pose.status();
  }
  const Eigen::Vector3d position = pose.value().position_m;
  planar = position.head<2>();
  out_of_plane = position.z();
  return Status{};
}

}  // namespace

crane_model::Result<ArmGeometry> probe_arm_geometry(
  const crane_model::Model & model, const JointLimits & limits, std::size_t samples,
  const GeometryTolerance & tolerance)
{
  if (samples < 3) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidArgument,
        "the affine telescope fit needs at least three extensions to be over-determined"));
  }
  const AxisLimit & telescope = limits.axis[3];
  if (!telescope.bounded) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the telescope has no range, so there is no interval to resolve the redundancy over"));
  }

  ArmGeometry geometry;
  Eigen::Vector2d boom;
  double out_of_plane = 0.0;
  Status status = planar_origin(model, probe_configuration(0.0), Frame::Boom, boom, out_of_plane);
  if (!status.ok()) {
    return Result<ArmGeometry>::failure(std::move(status));
  }
  if (std::abs(out_of_plane) > tolerance.planarity_m) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "K2 does not lie in K1's own x-y plane, so the arm of robot_model 2.2 is not planar in "
        "this description"));
  }
  geometry.a2 = boom.norm();
  geometry.gamma0 = std::atan2(boom.y(), boom.x());
  if (!(geometry.a2 > 0.0)) {
    return Result<ArmGeometry>::failure(
      failure(ErrorCode::InvalidRobotDescription, "the boom has no length"));
  }

  // The sign of q2, read rather than assumed: the description's own DH frames
  // decide whether a positive q2 advances or retards the boom's bearing.
  {
    Q q = probe_configuration(0.0);
    q[1] = kSignProbe;
    Eigen::Vector2d stepped;
    status = planar_origin(model, q, Frame::Boom, stepped, out_of_plane);
    if (!status.ok()) {
      return Result<ArmGeometry>::failure(std::move(status));
    }
    if (std::abs(stepped.norm() - geometry.a2) > tolerance.planarity_m) {
      return Result<ArmGeometry>::failure(
        failure(
          ErrorCode::InvalidRobotDescription,
          "q2 does not rotate K2 about K1's origin, so the boom is not the first link of a "
          "two-link chain hinged there"));
    }
    const double advance = wrap(std::atan2(stepped.y(), stepped.x()) - geometry.gamma0);
    geometry.sign_q2 = advance >= 0.0 ? 1.0 : -1.0;
  }

  // The forearm, once per telescope extension: its length and its bearing are
  // both functions of q4 alone, because q2 and q3 rotate the whole of it.
  std::vector<double> q4_grid(samples);
  std::vector<double> length(samples);
  std::vector<double> bearing(samples);
  for (std::size_t index = 0; index < samples; ++index) {
    const double fraction =
      static_cast<double>(index) / static_cast<double>(samples - 1);
    const double q4 = telescope.lower + fraction * (telescope.upper - telescope.lower);
    Eigen::Vector2d tip;
    status = planar_origin(model, probe_configuration(q4), Frame::Tip, tip, out_of_plane);
    if (!status.ok()) {
      return Result<ArmGeometry>::failure(std::move(status));
    }
    if (std::abs(out_of_plane) > tolerance.planarity_m) {
      return Result<ArmGeometry>::failure(
        failure(
          ErrorCode::InvalidRobotDescription,
          "K5 leaves K1's x-y plane as the telescope extends, so the arm of robot_model 2.2 is "
          "not planar in this description"));
    }
    const Eigen::Vector2d forearm = tip - boom;
    q4_grid[index] = q4;
    length[index] = forearm.norm();
    bearing[index] = std::atan2(forearm.y(), forearm.x());
    if (!(length[index] > 0.0)) {
      return Result<ArmGeometry>::failure(
        failure(ErrorCode::InvalidRobotDescription, "the forearm has no length"));
    }
  }

  {
    Q q = probe_configuration(telescope.lower);
    q[2] = kSignProbe;
    Eigen::Vector2d tip;
    status = planar_origin(model, q, Frame::Tip, tip, out_of_plane);
    if (!status.ok()) {
      return Result<ArmGeometry>::failure(std::move(status));
    }
    const Eigen::Vector2d forearm = tip - boom;
    const double advance =
      wrap(std::atan2(forearm.y(), forearm.x()) - bearing.front());
    geometry.sign_q3 = advance >= 0.0 ? 1.0 : -1.0;
  }

  // L(q4)^2 = a3^2 + (d45_0 + gain * q4)^2 is a quadratic in q4, so the three
  // numbers of 2.2 come out of one least-squares fit -- and its residual is the
  // statement that the telescope really does extend along a fixed direction.
  Eigen::MatrixXd basis(static_cast<Eigen::Index>(samples), 3);
  Eigen::VectorXd observed(static_cast<Eigen::Index>(samples));
  for (std::size_t index = 0; index < samples; ++index) {
    const Eigen::Index row = static_cast<Eigen::Index>(index);
    basis(row, 0) = 1.0;
    basis(row, 1) = q4_grid[index];
    basis(row, 2) = q4_grid[index] * q4_grid[index];
    observed[row] = length[index] * length[index];
  }
  const Eigen::Vector3d fit = basis.colPivHouseholderQr().solve(observed);
  if (!(fit[2] > 0.0)) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the forearm's length does not grow with the telescope extension"));
  }
  geometry.d45_gain = std::sqrt(fit[2]);
  geometry.d45_0 = fit[1] / (2.0 * geometry.d45_gain);
  const double offset_squared = fit[0] - geometry.d45_0 * geometry.d45_0;
  geometry.a3 = std::sqrt(std::max(0.0, offset_squared));

  // The residual is compared as a length, not as a length squared, so the
  // tolerance is in metres and means what it says.
  double structure_residual = 0.0;
  for (std::size_t index = 0; index < samples; ++index) {
    const double predicted = std::hypot(geometry.a3, geometry.d45(q4_grid[index]));
    structure_residual = std::max(structure_residual, std::abs(predicted - length[index]));
  }
  if (structure_residual > tolerance.structure_m) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the forearm does not follow sqrt(a3^2 + d45(q4)^2): the affine telescope fit leaves " +
        std::to_string(structure_residual) + " m, so this description is not the arm of "
        "robot_model 2.2"));
  }

  // a3 enters only through atan2(a3, d45), so its sign is the half the length
  // fit cannot see. It is the one that makes the bearing offset constant.
  const double offset_magnitude = geometry.a3;
  double best_spread = std::numeric_limits<double>::infinity();
  for (const double sign : {1.0, -1.0}) {
    const double offset = sign * offset_magnitude;
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();
    double first = 0.0;
    for (std::size_t index = 0; index < samples; ++index) {
      const double psi = bearing[index] - geometry.gamma0 -
        std::atan2(offset, geometry.d45_0 + geometry.d45_gain * q4_grid[index]);
      const double centred = (index == 0) ? psi : first + wrap(psi - first);
      if (index == 0) {
        first = psi;
      }
      lowest = std::min(lowest, centred);
      highest = std::max(highest, centred);
    }
    const double spread = highest - lowest;
    if (spread < best_spread) {
      best_spread = spread;
      geometry.a3 = offset;
      geometry.psi0 = wrap(0.5 * (lowest + highest));
    }
  }
  if (best_spread > tolerance.bearing_rad) {
    return Result<ArmGeometry>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        "the forearm's bearing offset is not constant over the telescope range: it moves by " +
        std::to_string(best_spread) + " rad"));
  }

  return Result<ArmGeometry>::success(std::move(geometry));
}

}  // namespace crane_planning
