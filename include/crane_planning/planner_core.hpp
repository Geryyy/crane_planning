// One goal pose in, one timed joint trajectory out -- the whole of the slice-5
// tracer bullet, with no ROS in it.
//
// `wiki/trajectory_planning.md` 2 splits planning into a geometric stage and a
// timing stage. This carries the thinnest honest version of each: the geometric
// stage is the endpoint IK of `wiki/robot_model.md` 2.2 and a straight line
// between the two configurations, the timing stage is one scaled ramp. What is
// **absent** is named rather than approximated, and every absence is refused at
// the boundary instead of stubbed:
//
//   collision of any kind, and the sway envelope       issue 041
//   the equilibrium-constrained endpoint NLP           issue 039
//   the lift/traverse/descend primitive                issue 040
//   the sampling fallback and its smoothing            issue 042
//   the path-constrained OCP, force, flow, kappa       issue 043
//   replanning from a moving, swinging state           issue 045
//
// `wiki/implementation/style_guide.md` 3 is the reason this is a separate
// translation unit from the node: the algorithm takes plain types and returns
// plain types, and the node is the adapter that turns messages into them.

#ifndef CRANE_PLANNING__PLANNER_CORE_HPP_
#define CRANE_PLANNING__PLANNER_CORE_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/arm_geometry.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/trajectory_timing.hpp"

namespace crane_planning
{

/// Everything a deployment configures, in one place.
struct PlannerSettings
{
  IkSettings ik{};
  RampSettings ramp{};
  std::size_t geometry_samples{9};  ///< telescope extensions the structure check runs over
};

/// What is derived once from the description, and reused for every request.
struct PlannerContext
{
  ArmGeometry geometry{};
  JointLimits limits{};
  PlannerSettings settings{};
};

/// Derive the context from one model and the description it was built from.
[[nodiscard]] crane_model::Result<PlannerContext> build_planner(
  const crane_model::Model & model, const std::string & robot_description_xml,
  const PlannerSettings & settings);

/// One `/crane/plan_motion` request, with the message already off it.
struct MotionRequest
{
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};  ///< goal position in K0_mounting_base
  double phi_z_d{};                                   ///< goal yaw, rad
  crane_model::QA q_a_start{crane_model::QA::Zero()};  ///< where the machine is now
  crane_model::Payload payload{};                      ///< what is in the gripper
  double margin_factor{1.0};                           ///< kappa, i.e. `speed_scale`
  bool avoid_collisions{true};                         ///< refused here; issue 041 answers it
};

/// One answer.
struct MotionPlan
{
  TimedTrajectory trajectory{};
  std::vector<crane_model::Pose> tcp_path{};  ///< for visualization only
  IkSolution endpoint{};
};

/// Plan one move, or refuse it and say which of the six absences above applies.
[[nodiscard]] crane_model::Result<MotionPlan> plan_motion(
  const crane_model::Model & model, const PlannerContext & context,
  const MotionRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__PLANNER_CORE_HPP_
