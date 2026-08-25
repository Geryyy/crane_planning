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
//   * publishes on exactly one topic, `/crane/reference`, which is a *reference*
//     and not a command -- ROS 2 Interfaces 4 gives that row to the planner and
//     it has had no producer until now;
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

#include <memory>
#include <optional>
#include <string>

#include "crane_model/model.hpp"
#include "crane_msgs/msg/collision_scene.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/srv/plan_grip.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "crane_planning/plan_grip.hpp"
#include "crane_planning/planner_core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace crane_planning
{

/// The contract names of ROS 2 Interfaces 4 and 5, as constants.
/**
 * Constants and not parameters, on the terms 1 sets: a cross-node contract a
 * deployment can rename from a YAML file is not a contract.
 */
inline constexpr char kPlanMotionService[] = "/crane/plan_motion";

/// The second service of 5, which 10 keeps separate while the task layer migrates.
inline constexpr char kPlanGripService[] = "/crane/plan_grip";
inline constexpr char kReferenceTopic[] = "/crane/reference";
inline constexpr char kJointStatesTopic[] = "/joint_states";
inline constexpr char kCollisionSceneTopic[] = "/crane/collision_scene";

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

/// The description, latched by `robot_state_publisher`.
[[nodiscard]] rclcpp::QoS robot_description_qos();

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

  /// One `/crane/plan_grip` phase, answered. Public for the same reason `plan` is.
  /**
   * It holds no state between calls and none about which phase ran last: the
   * start of every phase is the newest `/joint_states`, so the task layer
   * sequences the four by moving the machine and re-measuring.
   */
  void grip(
    const crane_msgs::srv::PlanGrip::Request & request,
    crane_msgs::srv::PlanGrip::Response & response);

private:
  void on_robot_description(std_msgs::msg::String::ConstSharedPtr message);

  /// `crane_msgs/Payload` as the two halves the planner uses, or a reason it is not one.
  /**
   * The equilibrium takes a point mass (`wiki/robot_model.md` 5.3) and the
   * collision check takes a shape and an extent; they are different halves of
   * one message and both services read it the same way, so the conversion is
   * here and not written twice.
   */
  [[nodiscard]] static bool read_payload(
    const crane_msgs::msg::Payload & message, crane_model::Payload & payload,
    PayloadShape & shape, std::string & why);

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

  /// The six actuated coordinates out of the newest `/joint_states`, by name.
  /**
   * By name and never by position: `sensor_msgs/JointState` fixes no order and
   * the eighth canonical joint is a different string per tool. A joint the
   * message does not carry refuses the whole read -- a start configuration with
   * an invented telescope extension is a plan for a crane that is somewhere
   * else.
   */
  [[nodiscard]] bool read_start(crane_model::QA & q_a_start, std::string & why) const;

  crane_model::Tool tool_{crane_model::Tool::Pzs100};
  PlannerSettings settings_{};
  TruckModel truck_{};  ///< the vehicle's own geometry, keyed to the measured pose
  double max_input_age_{0.5};

  std::optional<crane_model::Model> model_;
  std::optional<PlannerContext> context_;
  sensor_msgs::msg::JointState::ConstSharedPtr joint_states_;

  /// The newest usable scene, expanded. Absent until one arrives and converts.
  std::optional<crane_model::CollisionScene> scene_;
  std::string scene_note_;  ///< where it came from, or why the last one was refused

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_subscription_;
  rclcpp::Subscription<crane_msgs::msg::CollisionScene>::SharedPtr collision_scene_subscription_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr reference_;
  rclcpp::Service<crane_msgs::srv::PlanMotion>::SharedPtr plan_motion_;
  rclcpp::Service<crane_msgs::srv::PlanGrip>::SharedPtr plan_grip_;
};

}  // namespace crane_planning

#endif  // CRANE_PLANNING__PLANNER_NODE_HPP_
