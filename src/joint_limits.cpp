#include "crane_planning/joint_limits.hpp"

#include <urdf_parser/urdf_parser.h>

#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include "crane_planning/status.hpp"

namespace crane_planning
{
namespace
{

using crane_model::ErrorCode;
using crane_model::Status;

}  // namespace

crane_model::Result<JointLimits> read_joint_limits(
  const std::string & robot_description_xml,
  const std::array<std::string, crane_model::kGeneralizedDof> & urdf_joint_names)
{
  ::urdf::ModelInterfaceSharedPtr tree;
  try {
    tree = ::urdf::parseURDF(robot_description_xml);
  } catch (const std::exception & error) {
    return crane_model::Result<JointLimits>::failure(
      failure(
        ErrorCode::InvalidRobotDescription,
        std::string("the robot description could not be parsed for joint limits: ") +
        error.what()));
  }
  if (!tree) {
    return crane_model::Result<JointLimits>::failure(
      failure(ErrorCode::InvalidRobotDescription, "the robot description is not valid URDF"));
  }

  JointLimits limits;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    const std::string & name = urdf_joint_names[kActuatedRows[row]];
    const ::urdf::JointConstSharedPtr joint = tree->getJoint(name);
    if (!joint) {
      return crane_model::Result<JointLimits>::failure(
        failure(ErrorCode::MissingJoint, "the robot description carries no joint " + name));
    }
    if (!joint->limits) {
      return crane_model::Result<JointLimits>::failure(
        failure(
          ErrorCode::InvalidRobotDescription,
          name + " declares no <limit>, so the description states neither how far nor how fast "
          "that axis may go and this planner will not invent either"));
    }

    AxisLimit & axis = limits.axis[row];
    // A continuous joint has a velocity limit and no range at all, which is a
    // statement and not an omission: the rotator turns without end
    // (wiki/nomenclature.md 4). Its range therefore contributes nothing to the
    // centring score of robot_model 2.2, and the solver says so by carrying the
    // infinite bound rather than by inventing a finite one.
    axis.bounded = joint->type != ::urdf::Joint::CONTINUOUS;
    if (axis.bounded) {
      axis.lower = joint->limits->lower;
      axis.upper = joint->limits->upper;
      if (!std::isfinite(axis.lower) || !std::isfinite(axis.upper) || !(axis.lower < axis.upper)) {
        return crane_model::Result<JointLimits>::failure(
          failure(
            ErrorCode::InvalidRobotDescription,
            name + " declares no usable position range"));
      }
    } else {
      axis.lower = -std::numeric_limits<double>::infinity();
      axis.upper = std::numeric_limits<double>::infinity();
    }

    axis.dq_max = joint->limits->velocity;
    if (!std::isfinite(axis.dq_max) || axis.dq_max <= 0.0) {
      return crane_model::Result<JointLimits>::failure(
        failure(
          ErrorCode::InvalidRobotDescription,
          name + " declares no finite positive velocity limit, and a trajectory timed against an "
          "absent bound is a trajectory timed against nothing"));
    }
  }
  return crane_model::Result<JointLimits>::success(std::move(limits));
}

}  // namespace crane_planning
