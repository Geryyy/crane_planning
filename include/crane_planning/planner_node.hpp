// The `crane_planner` of `wiki/implementation/ros2_interfaces.md` 2: a node,
// beside the controller manager and not inside it, that answers
// `/crane/plan_motion` and publishes the reference it answered with.
//
// # It is not a second command producer
//
// The one thing this node must not become is a second writer of the machine.
// `crane_velocity_controller` is the sole claimant of the six `velocity` command
// interfaces (ROS 2 Interfaces 3.2) and which controller holds that claim is the
// supervisor's decision alone (5). So this node:
//
//   * publishes `/crane/reference`, which is a *reference* and not a command,
//     plus the visual-only TCP geometry of each adopted reference on
//     `/crane_planner/planned_path` -- ROS 2 Interfaces 4 gives the reference
//     row to the planner and it has had no producer until now;
//   * holds no `controller_manager` client of any kind, so it cannot move a
//     claim even indirectly;
//   * writes no command interface, because it is not a controller and no spawner
//     loads it.
//
// `crane_bringup`'s launch contract asserts all three by reading these sources,
// the same way it asserts them of `payload_estimator`.
//
// # What it reads, and why the description is the one name it is given a remap for
//
// Every name below is an absolute cross-node contract of ROS 2 Interfaces 1
// except `/robot_description`, which is a deployment's choice: the description
// composition publishes four descriptions and the profile decides which one this
// stack runs on. The profile gives this node the same remap it gives the
// `controller_manager` and the `payload_estimator` -- onto
// `robot_description_elastic_full` -- because a planner that solved the
// kinematics of a different description from the one the controllers were
// configured against would place the tool of a different crane.

#ifndef CRANE_PLANNING__PLANNER_NODE_HPP_
#define CRANE_PLANNING__PLANNER_NODE_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "crane_model/model.hpp"
#include "crane_msgs/msg/collision_scene.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/msg/payload_estimate.hpp"
#include "crane_msgs/msg/pendulum_state.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "crane_planning/planner_core.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "timber_crane_planning_interfaces/srv/calc_movement.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace crane_planning
{

/// The contract names of ROS 2 Interfaces 4 and 5, as constants.
/**
 * Constants and not parameters, on the terms 1 sets: a cross-node contract a
 * deployment can rename from a YAML file is not a contract.
 */
inline constexpr char kPlanMotionService[] = "/crane/plan_motion";

inline constexpr char kReferenceTopic[] = "/crane/reference";
/// The TCP geometry of the last adopted plan, for operator visualization only.
inline constexpr char kPlannedPathTopic[] = "/crane_planner/planned_path";
inline constexpr char kJointStatesTopic[] = "/joint_states";
inline constexpr char kCollisionSceneTopic[] = "/crane/collision_scene";

/// The passive half of the start state `wiki/trajectory_planning.md` 7 asks for.
inline constexpr char kPendulumStateTopic[] = "/crane/pendulum_state";

/// What is in the gripper, when the estimator says it knows (4).
inline constexpr char kPayloadEstimateTopic[] = "/crane/payload_estimate";

/// The one name that is not a contract, and that the profile remaps.
inline constexpr char kRobotDescriptionTopic[] = "/robot_description";

/// The frame planning geometry is in (ROS 2 Interfaces 1 and 4).
/**
 * The assembly planner converts `world` to this before it calls; nothing
 * downstream of that call converts, so a goal that arrives in any other frame is
 * refused rather than assumed.
 */
inline constexpr char kPlanningFrame[] = "K0_mounting_base";

/// Reliable, depth 1, **transient-local** -- the `/crane/reference` row of 4.
[[nodiscard]] rclcpp::QoS reference_qos();

/// Reliable, depth 1, volatile: the measured state this plans from.
[[nodiscard]] rclcpp::QoS input_qos();

/// Reliable, depth 1, **transient-local** -- the `/crane/collision_scene` row of 4.
/**
 * Transient-local because the scene is published *on change*: a planner that
 * started after the world model last spoke would otherwise have to refuse every
 * request until something moved.
 */
[[nodiscard]] rclcpp::QoS collision_scene_qos();

/// Reliable, depth 1, **transient-local** -- the `/crane/payload_estimate` row of 4.
/**
 * Transient-local because the estimate is published at 10 Hz but is only
 * *re-estimated* at rest, so a planner that started after the last estimate would
 * otherwise have to plan without one until the machine next stood still.
 */
[[nodiscard]] rclcpp::QoS payload_estimate_qos();

/// The description, latched by `robot_state_publisher`.
[[nodiscard]] rclcpp::QoS robot_description_qos();

/// What this deployment does when `/crane/pendulum_state` is not usable.
/**
 * The acceptance criterion offers exactly two, and `control_architecture` 5.3 is
 * why there is no third: an input that stops arriving must end in a **defined**
 * consequence, and silently reading the sway as zero is the stopped-start
 * convention `trajectory_planning` 7 exists to remove.
 */
enum class PassiveEstimatePolicy : std::uint8_t
{
  /// Refuse the request, naming which of "never connected", "died", "says it is
  /// not valid" it was. This is 5.3's row for a state the planner closes on.
  Refuse,

  /// Plan anyway, from the hanging pose, with a **stated** sway allowance held
  /// back so that the sway that may be there still fits what the MPC permits.
  Conservative
};

[[nodiscard]] const char * passive_policy_name(PassiveEstimatePolicy policy) noexcept;

/// The `crane_planner` node of ROS 2 Interfaces 2.
class PlannerNode : public rclcpp::Node
{
public:
  explicit PlannerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /// Whether a description has arrived, built a model and passed the structure
  /// check of `probe_arm_geometry`.
  [[nodiscard]] bool ready() const noexcept {return model_.has_value() && context_.has_value();}

  /// One request, answered. Public because the offline service test calls it
  /// directly as well as over the wire, so a refusal can be read without a
  /// round trip.
  void plan(
    const crane_msgs::srv::PlanMotion::Request & request,
    crane_msgs::srv::PlanMotion::Response & response);

  /// One `a2b_movement` request, answered through the native planning path.
  /**
   * The compatibility row of `wiki/implementation/ros2_interfaces.md` 9. It
   * translates, calls `plan_with_start` -- the private call
   * `/crane/plan_motion` reaches through `plan`, with the retained service's
   * optional explicit start -- and translates back. There is no second planner,
   * no second set of limits and no second collision configuration. What it maps
   * and what it refuses is `crane_planning/a2b_adapter.hpp`.
   *
   * Public for the same reason `plan` and `grip` are: the offline suite reads a
   * refusal without a round trip. And it has to log its own refusals, because
   * `CalcMovement::Response` carries no `message` field to put one on.
   */
  void a2b(
    const timber_crane_planning_interfaces::srv::CalcMovement::Request & request,
    timber_crane_planning_interfaces::srv::CalcMovement::Response & response);

  /// `crane_msgs/Payload` as the two halves the planner uses, or a reason it is not one.
  /**
   * The equilibrium takes a point mass (`wiki/robot_model.md` 5.3) and the
   * collision check takes a shape and an extent; they are different halves of
   * one message and both services read it the same way, so the conversion is
   * here and not written twice.
   *
   * Public, and static, because it is the far side of the adapter's boundary:
   * the offline suite reads what a translated `LogShape` and a natively declared
   * block each become here, without a node and without a graph.
   */
  [[nodiscard]] static bool read_payload(
    const crane_msgs::msg::Payload & message, crane_model::Payload & payload,
    PayloadShape & shape, std::string & why);

private:
  /// The native planning path with an optional retained-interface start override.
  void plan_with_start(
    const crane_msgs::srv::PlanMotion::Request & request,
    const MeasuredStart * start_override,
    crane_msgs::srv::PlanMotion::Response & response);

  void on_robot_description(std_msgs::msg::String::ConstSharedPtr message);

  /// The six actuated rows of one trajectory as `trajectory_msgs`, with 1's stamp.
  [[nodiscard]] trajectory_msgs::msg::JointTrajectory as_message(
    const TimedTrajectory & trajectory, const rclcpp::Time & origin) const;

  /// One `/crane/collision_scene`, converted and expanded, or refused with a reason.
  /**
   * Public through the service test rather than through the wire only, for the
   * same reason `plan` is: a refusal should be readable without a round trip.
   * The conversion is where the frame is checked, where the reserved ids of
   * `collision.hpp` are enforced, and where the primitive carrying the measured
   * truck pose becomes the bed and the six runges of `trajectory_planning` 4.2.
   */
  void on_collision_scene(crane_msgs::msg::CollisionScene::ConstSharedPtr message);

  /// The six actuated coordinates **and their rates**, out of the newest `/joint_states`.
  /**
   * By name and never by position: `sensor_msgs/JointState` fixes no order and
   * the eighth canonical joint is a different string per tool. A joint the
   * message does not carry refuses the whole read -- a start configuration with
   * an invented telescope extension is a plan for a crane that is somewhere
   * else.
   *
   * The `velocity` array is read on the same terms and its **absence is a
   * refusal**, not a zero. `wiki/trajectory_planning.md` 7 asks for `(q, dq)` as
   * measured, and a message that carries no rates is a measurement that does not
   * say whether the machine is moving; taking that as a standstill is precisely
   * the *"we deactivated qDot0, because we now always start in a stopped state"*
   * the page names. `joint_state_broadcaster` publishes the velocity interface.
   */
  [[nodiscard]] bool read_start(MeasuredStart & start, std::string & why) const;

  /// The passive half of the start, or the policy's answer to its absence.
  /**
   * Returns false only when the policy is `Refuse` and the estimate is not
   * usable; `why` then names which of the three absences it was. Under
   * `Conservative` it always returns true and fills `start.note` with what was
   * assumed and what was reserved for it.
   */
  [[nodiscard]] bool read_passive(PassiveStart & passive, std::string & why) const;

  /// `/crane/payload_estimate` where it is valid, and a stated absence where not.
  /**
   * The estimate carries `m_L`, `m_L r_x`, `m_L r_y` and an assumed `r_z`, which
   * is exactly the point mass `wiki/robot_model.md` 5.3 says a gravity moment
   * takes -- so where it is valid it *replaces* the mass and centre the caller
   * declared, because the estimator measured the machine and the caller declared
   * a model of it. Where it is not valid, or absent, the declaration stands and
   * `note` says so. Both CBS profiles publish `valid == false` today, so the
   * second branch is the one that runs.
   */
  void read_payload_estimate(crane_model::Payload & payload, std::string & note) const;

  /// The trajectory that is still standing on `/crane/reference`, in one sentence.
  [[nodiscard]] std::string describe_standing() const;

  crane_model::Tool tool_{crane_model::Tool::Pzs100};
  PlannerSettings settings_{};
  TruckModel truck_{};  ///< the vehicle's own geometry, keyed to the measured pose
  double max_input_age_{0.5};

  /// `/crane/pendulum_state`'s own freshness deadline, s -- 5.3's table gives 150 ms.
  double pendulum_deadline_{0.15};

  /// `/crane/payload_estimate`'s, s. Not in 5.3's table, because the supervisor
  /// does not watch this row; it is ten periods of a 10 Hz publication.
  double payload_deadline_{1.0};

  PassiveEstimatePolicy passive_policy_{PassiveEstimatePolicy::Refuse};
  double conservative_sway_rad_{0.05};
  double conservative_sway_rate_{0.10};

  std::optional<crane_model::Model> model_;
  std::optional<PlannerContext> context_;

  sensor_msgs::msg::JointState::ConstSharedPtr joint_states_;
  crane_msgs::msg::PendulumState::ConstSharedPtr pendulum_state_;
  crane_msgs::msg::PayloadEstimate::ConstSharedPtr payload_estimate_;

  /// The last trajectory this node actually adopted, and its stamp.
  /**
   * 7's fallback, held here because the node is what publishes: a plan that
   * overran its budget keeps the previous trajectory and reports it, and
   * `/crane/reference` is transient-local, so a refused re-plan that republished
   * would leave a late subscriber latching onto a plan nobody is executing.
   */
  std::optional<trajectory_msgs::msg::JointTrajectory> standing_;

  /// The newest usable scene, expanded. Absent until one arrives and converts.
  std::optional<crane_model::CollisionScene> scene_;
  std::string scene_note_;  ///< where it came from, or why the last one was refused

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_subscription_;
  rclcpp::Subscription<crane_msgs::msg::CollisionScene>::SharedPtr collision_scene_subscription_;
  rclcpp::Subscription<crane_msgs::msg::PendulumState>::SharedPtr pendulum_state_subscription_;
  rclcpp::Subscription<crane_msgs::msg::PayloadEstimate>::SharedPtr payload_estimate_subscription_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr reference_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_;
  rclcpp::Service<crane_msgs::srv::PlanMotion>::SharedPtr plan_motion_;
  /// The retained row of 9, served by the same node and answered by `plan`.
  rclcpp::Service<timber_crane_planning_interfaces::srv::CalcMovement>::SharedPtr a2b_movement_;
};

}  // namespace crane_planning

#endif  // CRANE_PLANNING__PLANNER_NODE_HPP_
