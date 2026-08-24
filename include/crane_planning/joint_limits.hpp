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
// So they are read out of the same `robot_description` XML the model was built
// from, with urdfdom, which is Pinocchio's own URDF front end and the parser
// `crane_model/src/model.cpp` already uses for the `<mimic>` and `<dynamics>`
// entries Pinocchio drops. `wiki/implementation/libraries.md` lists Pinocchio for
// "URDF and SRDF parsing"; a second XML reader written here is exactly what its
// Style Guide 1.1 forbids.

#ifndef CRANE_PLANNING__JOINT_LIMITS_HPP_
#define CRANE_PLANNING__JOINT_LIMITS_HPP_

#include <array>
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

/// The rows of `kActuatedRows` the geometric path runs over.
/**
 * `wiki/trajectory_planning.md` 4.1: planning happens over
 * `q_a = (q1, q2, q3, q4, q7)`. The sixth actuated coordinate, q8, is the tool
 * and is not a path variable; it rides the same time base and nothing else.
 */
inline constexpr std::size_t kPathDof = 5;
inline constexpr std::size_t kToolRow = 5;

/// One actuated axis, as the description states it.
struct AxisLimit
{
  double lower{};   ///< rad or m; -inf for a continuous joint
  double upper{};   ///< rad or m; +inf for a continuous joint
  double dq_max{};  ///< rad/s or m/s, strictly positive
  bool bounded{true};

  [[nodiscard]] double centre() const noexcept {return 0.5 * (lower + upper);}
  [[nodiscard]] double half_span() const noexcept {return 0.5 * (upper - lower);}
};

/// The six actuated axes, indexed the way `kActuatedRows` indexes them.
struct JointLimits
{
  std::array<AxisLimit, crane_model::kActuatedDof> axis{};
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

}  // namespace crane_planning

#endif  // CRANE_PLANNING__JOINT_LIMITS_HPP_
