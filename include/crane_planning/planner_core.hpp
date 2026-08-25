// One goal pose in, one timed joint trajectory out -- the whole of the slice-5
// tracer bullet, with no ROS in it.
//
// `wiki/trajectory_planning.md` 2 splits planning into a geometric stage and a
// timing stage. The geometric stage is the endpoint IK of
// `wiki/robot_model.md` 2.2 followed by the structured lift/traverse/descend
// primitive of `trajectory_planning` 4.4, built `C2` in sigma; the timing stage
// is still one scaled ramp, run along that path rather than straight between two
// configurations.
//
// **Which of 2.2's two formulations solves the endpoint is decided here and not
// by the caller.** 2.2's closing paragraph is the rule -- the semi-analytic
// route where speed matters, the equilibrium-constrained NLP where the endpoint
// must be sway-free -- and every goal this service is asked for is a placement
// goal, so every endpoint is solved with the constrained route of
// `equilibrium_ik.hpp`. The fast route still runs, as that solve's initial
// guess. There is no parameter for it, because a caller who picked the fast one
// would be choosing to have the tool arrive swinging without being told that is
// what the choice meant. What is
// **absent** is named rather than approximated, and every absence is refused at
// the boundary instead of stubbed:
//
//   the path-constrained OCP, force, flow, kappa       issue 043
//   replanning from a moving, swinging state           issue 045
//
// Collision is no longer among them: the scene of `/crane/collision_scene`, the
// truck model of `trajectory_planning` 4.2 and the sway envelope of 4.3 arrive
// through `collision.hpp`, and `avoid_collisions` is honoured rather than
// refused.
//
// # 4.4's two mechanisms, in 4.4's order, and the order is the point
//
// **Generate the structured primitive, check it, accept it if clear, otherwise
// sample.** This file is where that order is enforced, and 4.4's `[!important]`
// is why it is an order rather than a choice: a sampling planner is stochastic
// and its runtime is not bounded, so primitive-first is what gives deterministic
// latency in the common case and leaves completeness to the rare one. The
// fallback of `sampling_planner.hpp` therefore runs only after
// `build_structured_primitive` has been asked and refused -- never beside it, and
// never to compare the two.
//
// Which one answered travels out on `MotionPlan::mechanism`, because a caller
// cannot otherwise tell a deterministic plan from a sampled one and they are not
// the same product: one has bounded latency and the shape an operator expects,
// the other spent a budget and rounds whatever was in the way.
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
#include "crane_planning/equilibrium_ik.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/sampling_planner.hpp"
#include "crane_planning/structured_primitive.hpp"
#include "crane_planning/trajectory_timing.hpp"

namespace crane_planning
{

/// Everything a deployment configures, in one place.
struct PlannerSettings
{
  IkSettings ik{};
  EquilibriumIkSettings equilibrium{};
  PrimitiveSettings primitive{};
  SamplingSettings sampling{};      ///< the fallback of 4.4, reached only when the primitive is not
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
  PayloadShape payload_shape{};                        ///< and what shape it is
  double margin_factor{1.0};                           ///< kappa, i.e. `speed_scale`
  bool avoid_collisions{true};                         ///< honoured, per `crane_msgs/PlanMotion`

  /// The newest `/crane/collision_scene`, already expanded and in K0. May be null.
  const crane_model::CollisionScene * scene{nullptr};
};

/// One answer.
struct MotionPlan
{
  TimedTrajectory trajectory{};
  std::vector<crane_model::Pose> tcp_path{};  ///< for visualization only
  IkSolution endpoint{};

  /// The geometry the trajectory was timed along, whichever mechanism produced it.
  GeometricPath path{};

  /// Which of 4.4's two mechanisms that was. Never inferred, always carried.
  PathMechanism mechanism{PathMechanism::StructuredPrimitive};

  /// The primitive, when it was accepted. `mechanism` says whether it was.
  StructuredPrimitive primitive{};

  /// The sampled path, when the primitive was not. Same condition, other branch.
  SamplingPlan sampled{};

  /// Why the primitive was refused, when it was. Empty when it was accepted.
  /**
   * Kept rather than dropped because it is half of what a caller needs: a
   * sampled answer without the reason the deterministic one could not be given
   * leaves nobody able to tell an obstacle above the transfer altitude from an
   * arm that could not reach the column above the goal.
   */
  std::string primitive_refusal{};
};

/// Plan one move, or refuse it and say which of the six absences above applies.
[[nodiscard]] crane_model::Result<MotionPlan> plan_motion(
  const crane_model::Model & model, const PlannerContext & context,
  const MotionRequest & request);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__PLANNER_CORE_HPP_
