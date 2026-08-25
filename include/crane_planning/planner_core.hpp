// One goal pose in, one timed joint trajectory out -- the whole of the slice-5
// tracer bullet, with no ROS in it.
//
// `wiki/trajectory_planning.md` 2 splits planning into a geometric stage and a
// timing stage. The geometric stage is the endpoint IK of
// `wiki/robot_model.md` 2.2 followed by the structured lift/traverse/descend
// primitive of `trajectory_planning` 4.4, built `C2` in sigma; the timing stage
// is the path-constrained optimal control problem of 5.2, solved with acados
// over `crane_model`'s symbolic graph -- the sway carried as a state, the
// cylinder force and pump flow constrained out of the same expressions
// `crane_mpc` will use, and the kappa margin of 5.5 held back for the
// controller. The scaled ramp it replaced is still in the package, as the
// no-dynamics answer the OCP is measured against.
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
#include "crane_planning/timing_ocp.hpp"
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

  /// The path-constrained OCP of 5.2, which is what times every plan.
  /**
   * `RampSettings` above no longer times anything a caller receives. It is kept
   * because `scaled_ramp` and `scaled_ramp_along_path` are still the thing the
   * timing tests pose the OCP against -- a ramp is the answer the machine would
   * give with no dynamics in the loop, and the difference is what carrying the
   * sway bought.
   */
  TimingOcpSettings timing{};

  /// The hydraulic relief setting the cylinder force limit is derived from, Pa.
  /**
   * `config/hydraulic_limits.yaml` carries it and the evidence for it, which is
   * thin: `wiki/implementation/parameters.md` 7 lists the pressure constants
   * among its gaps, so this is the one number in the force limit that is neither
   * measured nor readable off the description. The chamber areas *are* the
   * model's, through `derive_cylinder_force_limits`.
   */
  double system_pressure_pa{2.5e7};

  std::size_t geometry_samples{9};  ///< telescope extensions the structure check runs over
};

/// What is derived once from the description, and reused for every request.
struct PlannerContext
{
  ArmGeometry geometry{};
  JointLimits limits{};
  PlannerSettings settings{};

  /// The config the model was built from, kept because the OCP needs it.
  /**
   * `crane_model::symbolic::casadi_graph` takes a `ModelConfig` and not a
   * `Model`: the model keeps its parse behind a private pimpl and its header is
   * frozen, so there is no route from a `const Model &` to what it parsed. Issue
   * 035's notes record that as structural rather than incidental. Holding the
   * config here is what lets `plan_motion` build the graph for the same machine
   * the `Model` describes instead of re-deriving one.
   */
  crane_model::ModelConfig model_config{};
};

/// Derive the context from one model and the config it was built from.
[[nodiscard]] crane_model::Result<PlannerContext> build_planner(
  const crane_model::Model & model, const crane_model::ModelConfig & model_config,
  const PlannerSettings & settings);

/// One `/crane/plan_motion` request, with the message already off it.
struct MotionRequest
{
  Eigen::Vector3d p_tcp_0{Eigen::Vector3d::Zero()};  ///< goal position in K0_mounting_base
  double phi_z_d{};                                   ///< goal yaw, rad
  crane_model::QA q_a_start{crane_model::QA::Zero()};  ///< where the machine is now
  crane_model::Payload payload{};                      ///< what is in the gripper
  PayloadShape payload_shape{};                        ///< and what shape it is
  /// `crane_msgs/PlanMotion.speed_scale`, in (0, 1]. **Not** kappa.
  /**
   * The caller's request to go slower, and nothing more. kappa of
   * `wiki/trajectory_planning.md` 5.5 is `PlannerSettings::timing.kappa`, it is
   * the authority the planner reserves for the MPC, and it is not reachable from
   * a request: the two multiply, so asking for more speed can only ever lower the
   * demand this places on the machine and never raise it above kappa.
   */
  double speed_scale{1.0};
  bool avoid_collisions{true};                         ///< honoured, per `crane_msgs/PlanMotion`

  /// The newest `/crane/collision_scene`, already expanded and in K0. May be null.
  const crane_model::CollisionScene * scene{nullptr};
};

/// One answer.
struct MotionPlan
{
  TimedTrajectory trajectory{};

  /// What the OCP spent, and how close to the machine's limits it came.
  /**
   * Carried out rather than kept inside, because `PeakDemand` is what makes
   * 5.5's margin checkable by a caller instead of trusted: every entry is a
   * fraction of the **physical** limit, so a plan built at `kappa = 0.8` reports
   * peaks at or below 0.8 and not at 1.0. `solver_status` is acados' own word,
   * and it says `ACADOS_SUCCESS` on every plan that exists.
   */
  TimingSolution timing{};
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
