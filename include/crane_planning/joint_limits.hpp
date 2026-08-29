// The position and velocity limits the robot description gives the six actuated
// joints, in `crane_model`'s canonical actuated order.
//
// # Why this is read here and not taken off the model
//
// `wiki/implementation/model_api_contract.md` freezes `crane_model`'s public
// surface and that surface carries no joint limits: `Model::create` validates
// them and then keeps them. The planner needs them for two things it cannot do
// without -- the joint-range centring that resolves the telescope redundancy
// (`wiki/robot_model.md` 2.2 step 3) and the velocity bound the timing obeys
// (`wiki/trajectory_planning.md` 3) -- and neither may be invented in a
// parameter file: a limit written down here is a limit that drifts away from the
// description every controller in the deployment was configured against.
//
// # The coordinate algebra lives here too
//
// `kActuatedRows` is the map between `crane_model`'s canonical eight and the six
// the description states a range for, so the conversions that use it -- and the
// range checks themselves -- are here rather than copied privately into each
// translation unit that needs one. They were copied, verbatim, into four of
// them; one row order written down twice is one that can be written down wrong.
//
// So they are read out of the same `robot_description` XML the model was built
// from, with urdfdom, which is Pinocchio's own URDF front end and the parser
// `crane_model/src/model.cpp` already uses for the `<mimic>` and `<dynamics>`
// entries Pinocchio drops. `wiki/implementation/libraries.md` lists Pinocchio for
// "URDF and SRDF parsing"; a second XML reader written here is exactly what its
// Style Guide 1.1 forbids.

#ifndef CRANE_PLANNING__JOINT_LIMITS_HPP_
#define CRANE_PLANNING__JOINT_LIMITS_HPP_

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "crane_model/model.hpp"

namespace crane_planning
{

/// The six actuated coordinates as rows of `crane_model`'s canonical eight.
/**
 * `wiki/implementation/model_api_contract.md` 2: the actuated projection is
 * `[0, 1, 2, 3, 6, 7]`, which is q1, q2, q3, q4, q7, q8 of
 * `wiki/nomenclature.md` 4 and the six joints of ROS 2 Interfaces 3.1 in the
 * order that section lists them.
 */
inline constexpr std::array<std::size_t, crane_model::kActuatedDof> kActuatedRows{
  {0, 1, 2, 3, 6, 7}};

/// The two passive coordinates as rows of the same canonical eight: tip, then tilt.
inline constexpr std::array<std::size_t, crane_model::kPassiveDof> kPassiveRows{{4, 5}};

/// The rows of `kActuatedRows` the geometric path runs over.
/**
 * `wiki/trajectory_planning.md` 4.1: planning happens over
 * `q_a = (q1, q2, q3, q4, q7)`. The sixth actuated coordinate, q8, is the tool
 * and is not a path variable; it rides the same time base and nothing else.
 */
inline constexpr std::size_t kPathDof = 5;
inline constexpr std::size_t kToolRow = 5;

/// The five path coordinates of `wiki/trajectory_planning.md` 4.1, as a vector.
/**
 * `q_a = (q1, q2, q3, q4, q7)`. q8 is not in here on purpose: 4.1 says in as
 * many words that the tool coordinate is not a path variable. It rides the same
 * sigma and is carried on `PathSample` beside these five.
 */
using PathVector = Eigen::Matrix<double, kPathDof, 1>;

/// One actuated axis, as the description states it.
struct AxisLimit
{
  double lower{};   ///< rad or m; -inf for a continuous joint
  double upper{};   ///< rad or m; +inf for a continuous joint
  double dq_max{};  ///< rad/s or m/s, strictly positive
  bool bounded{true};

  [[nodiscard]] double centre() const noexcept {return 0.5 * (lower + upper);}
  [[nodiscard]] double half_span() const noexcept {return 0.5 * (upper - lower);}

  /// Whether the description admits this value. An unbounded axis admits every one.
  [[nodiscard]] bool contains(double value) const noexcept
  {
    return !bounded || (value >= lower && value <= upper);
  }

  /// The nearest value the description admits, or the value itself where there is no range.
  [[nodiscard]] double clamp(double value) const noexcept
  {
    return bounded ? std::min(upper, std::max(lower, value)) : value;
  }
};

/// The six actuated axes, indexed the way `kActuatedRows` indexes them.
struct JointLimits
{
  std::array<AxisLimit, crane_model::kActuatedDof> axis{};

  /// Whether every actuated row of a configuration is inside the description's range.
  /**
   * The passive pair is not checked and has no row here: `JointLimits` reads the
   * six actuated axes only. It does not need one -- where the passive joints sit
   * is decided by the equilibrium and not by a bound an endpoint may be chosen
   * inside.
   */
  [[nodiscard]] bool contains(const crane_model::Q & q) const noexcept
  {
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      if (!axis[row].contains(q[static_cast<Eigen::Index>(kActuatedRows[row])])) {
        return false;
      }
    }
    return true;
  }
};

/// Read the six actuated axes out of a robot description.
/**
 * `urdf_joint_names` is `Model::urdf_joint_names()`, so which string the eighth
 * canonical coordinate is stays the model's answer and no per-tool table lives
 * here. A joint the description does not carry, a bounded joint with no usable
 * range, or an axis with no finite positive velocity limit is a failure and not
 * a default -- an invented bound is the silent-stub failure
 * `wiki/implementation/model_api_contract.md` 5 exists to prevent.
 */
[[nodiscard]] crane_model::Result<JointLimits> read_joint_limits(
  const std::string & robot_description_xml,
  const std::array<std::string, crane_model::kGeneralizedDof> & urdf_joint_names);

/// An angle folded into (-pi, pi].
[[nodiscard]] inline double wrap(double angle) noexcept
{
  return std::remainder(angle, 2.0 * M_PI);
}

/// The actuated projection of a canonical eight-vector.
[[nodiscard]] inline crane_model::QA actuated(const crane_model::Q & q) noexcept
{
  crane_model::QA q_a;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }
  return q_a;
}

/// The six actuated coordinates of a path vector, the tool held at q8.
[[nodiscard]] inline crane_model::QA actuated(const PathVector & q_a, double q8) noexcept
{
  crane_model::QA q_actuated = crane_model::QA::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q_actuated[static_cast<Eigen::Index>(row)] = q_a[static_cast<Eigen::Index>(row)];
  }
  q_actuated[static_cast<Eigen::Index>(kToolRow)] = q8;
  return q_actuated;
}

/// The five path coordinates of a canonical eight-vector. The tool row is not among them.
[[nodiscard]] inline PathVector path_of(const crane_model::Q & q) noexcept
{
  PathVector q_a = PathVector::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q_a[static_cast<Eigen::Index>(row)] = q[static_cast<Eigen::Index>(kActuatedRows[row])];
  }
  return q_a;
}

/// The canonical eight of six actuated coordinates, passive pair left at zero.
/**
 * Zero is a placeholder and not an equilibrium. Every caller that goes on to ask
 * the model where the tool is overwrites it -- with `expand(q_a, q_u)` at a
 * measured or settled pair -- so a configuration this built alone is one whose
 * passive rows nothing has read yet.
 */
[[nodiscard]] inline crane_model::Q expand(const crane_model::QA & q_a) noexcept
{
  crane_model::Q q = crane_model::Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
  }
  return q;
}

/// The canonical eight of six actuated coordinates at a given passive pair.
[[nodiscard]] inline crane_model::Q expand(
  const crane_model::QA & q_a, const crane_model::QU & q_u) noexcept
{
  crane_model::Q q = expand(q_a);
  q.segment<2>(4) = q_u;
  return q;
}

/// The canonical eight of a path vector, the tool held at q8, passive pair left at zero.
[[nodiscard]] inline crane_model::Q expand(const PathVector & q_a, double q8) noexcept
{
  crane_model::Q q = crane_model::Q::Zero();
  for (std::size_t row = 0; row < kPathDof; ++row) {
    q[static_cast<Eigen::Index>(kActuatedRows[row])] = q_a[static_cast<Eigen::Index>(row)];
  }
  q[static_cast<Eigen::Index>(kActuatedRows[kToolRow])] = q8;
  return q;
}

}  // namespace crane_planning

#endif  // CRANE_PLANNING__JOINT_LIMITS_HPP_
