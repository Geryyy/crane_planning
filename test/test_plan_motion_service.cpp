// The served contract: `/crane/plan_motion` answered by a real node, and the
// answer republished on `/crane/reference`.
//
// In process. The node object and a client are two nodes in one executor on the
// isolated domain `ralph/verify.sh` pins, with localhost-only transport: no
// launch, no controller manager, no simulator, no hardware. Nothing here sleeps
// -- every wait is a bounded spin on a predicate.
//
// The goal poses are not invented either: each one is produced by
// `crane_model::forward_kinematics` from a configuration the description allows,
// so what the service is asked for is a pose that certainly exists.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "crane_msgs/msg/collision_primitive.hpp"
#include "crane_msgs/msg/collision_scene.hpp"
#include "crane_msgs/msg/payload.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "crane_planning/planner_node.hpp"
#include "crane_planning/sampling_planner.hpp"
#include "description_fixture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace
{

using crane_msgs::srv::PlanMotion;
using sensor_msgs::msg::JointState;
using trajectory_msgs::msg::JointTrajectory;

/// How long any one wait may take before the test fails, s. Generous, because it
/// bounds a failure rather than a success.
constexpr double kBudget = 20.0;

/// The domain this binary runs on: one step off the one it was given, so that
/// the suites colcon runs beside it cannot see this fixture's `/joint_states`
/// and this fixture cannot see theirs. Same arithmetic as `crane_control`'s
/// stream fixture, for the same reason.
std::string stepped_domain()
{
  const char * const given = ::getenv("ROS_DOMAIN_ID");
  const int base = (given == nullptr) ? 0 : std::atoi(given);
  return std::to_string(1 + ((base % 101) + 101) % 101);
}

class RclcppEnvironment : public ::testing::Environment
{
public:
  void SetUp() override
  {
    // The isolation is the test's own, not the harness's. This workspace ships a
    // Cyclone configuration that pins a unicast peer at the real crane, and a
    // shell that has sourced it would put these nodes on the machine's live DDS
    // graph.
    ::unsetenv("CYCLONEDDS_URI");
    ::unsetenv("FASTRTPS_DEFAULT_PROFILES_FILE");
    ::setenv("ROS_LOCALHOST_ONLY", "1", 1);
    ::setenv("ROS_DOMAIN_ID", stepped_domain().c_str(), 1);
    rclcpp::init(0, nullptr);
  }
  void TearDown() override {rclcpp::shutdown();}
};

[[maybe_unused]] const ::testing::Environment * const kEnvironment =
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);

std::string fixture_description()
{
  const std::string path =
    std::string(CRANE_PLANNING_MACHINE_DESCRIPTION_DIR) + "/pzs100.urdf";
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

/// The configuration the fixture reports on `/joint_states`, and plans from.
const std::vector<double> & start_positions()
{
  static const std::vector<double> positions{0.0, 0.4, 1.2, 0.6, 0.0, 0.0, 0.0, 0.3};
  return positions;
}

class PlanMotionService : public ::testing::Test
{
protected:
  void SetUp() override
  {
    planner_ = std::make_shared<crane_planning::PlannerNode>();
    client_node_ = std::make_shared<rclcpp::Node>("crane_planning_plan_motion_client");
    client_ = client_node_->create_client<PlanMotion>(crane_planning::kPlanMotionService);
    reference_ = client_node_->create_subscription<JointTrajectory>(
      crane_planning::kReferenceTopic, crane_planning::reference_qos(),
      [this](JointTrajectory::ConstSharedPtr message) {published_.push_back(*message);});
    joint_states_ = client_node_->create_publisher<JointState>(
      crane_planning::kJointStatesTopic, crane_planning::input_qos());
    robot_description_ = client_node_->create_publisher<std_msgs::msg::String>(
      crane_planning::kRobotDescriptionTopic, crane_planning::robot_description_qos());
    collision_scene_ = client_node_->create_publisher<crane_msgs::msg::CollisionScene>(
      crane_planning::kCollisionSceneTopic, crane_planning::collision_scene_qos());

    executor_.add_node(planner_);
    executor_.add_node(client_node_);

    std_msgs::msg::String description;
    description.data = fixture_description();
    robot_description_->publish(description);
    ASSERT_TRUE(spin_until([this]() {return planner_->ready();}))
      << "the planner never built a model and a two-link closure from the fixture description";

    crane_model::ModelConfig config;
    config.robot_description_xml = description.data;
    config.tool = crane_model::Tool::Pzs100;
    auto model = crane_model::Model::create(config);
    ASSERT_TRUE(model.ok()) << model.status().message;
    model_.emplace(std::move(model).value());
  }

  void TearDown() override
  {
    executor_.remove_node(client_node_);
    executor_.remove_node(planner_);
  }

  bool spin_until(const std::function<bool()> & done)
  {
    const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(kBudget);
    while (std::chrono::steady_clock::now() < deadline) {
      if (done()) {
        return true;
      }
      executor_.spin_once(std::chrono::milliseconds(10));
    }
    return done();
  }

  /// Publish one `/joint_states` and let the planner take delivery of it.
  void publish_start() {publish_start(start_positions());}

  void publish_start(const std::vector<double> & positions)
  {
    JointState joints;
    joints.header.stamp = client_node_->now();
    joints.name.assign(
      model_->urdf_joint_names().begin(), model_->urdf_joint_names().end());
    joints.position = positions;
    start_stamp_ = joints.header.stamp;
    joint_states_->publish(joints);
  }

  /// The description's own limits, which the planner reads from the same XML.
  crane_planning::JointLimits fixture_limits() const
  {
    auto limits = crane_planning::read_joint_limits(
      fixture_description(), model_->urdf_joint_names());
    EXPECT_TRUE(limits.ok()) << limits.status().message;
    return std::move(limits).value();
  }

  /// One actuated configuration as the eight positions `/joint_states` carries.
  static std::vector<double> positions_of(const crane_model::QA & q_a)
  {
    std::vector<double> positions(
      crane_model::kActuatedDof + crane_model::kPassiveDof, 0.0);
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      positions[crane_planning::kActuatedRows[row]] = q_a[static_cast<Eigen::Index>(row)];
    }
    return positions;
  }

  /// A pose the arm certainly reaches, produced by forward kinematics.
  geometry_msgs::msg::PoseStamped reachable_goal(const crane_model::QA & q_a)
  {
    crane_model::Q q = crane_model::Q::Zero();
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
        q_a[static_cast<Eigen::Index>(row)];
    }
    auto settled =
      model_->passive_equilibrium(q_a, crane_planning_test::empty_gripper());
    EXPECT_TRUE(settled.ok()) << settled.status().message;
    q[4] = settled.value()[0];
    q[5] = settled.value()[1];
    auto pose = model_->forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    EXPECT_TRUE(pose.ok()) << pose.status().message;

    geometry_msgs::msg::PoseStamped goal;
    goal.header.frame_id = crane_planning::kPlanningFrame;
    goal.pose.position.x = pose.value().position_m.x();
    goal.pose.position.y = pose.value().position_m.y();
    goal.pose.position.z = pose.value().position_m.z();
    goal.pose.orientation.w = pose.value().orientation.w();
    goal.pose.orientation.x = pose.value().orientation.x();
    goal.pose.orientation.y = pose.value().orientation.y();
    goal.pose.orientation.z = pose.value().orientation.z();
    return goal;
  }

  static crane_model::QA goal_configuration()
  {
    crane_model::QA q_a;
    q_a << 0.9, 0.2, 1.6, 1.1, 0.4, 0.3;
    return q_a;
  }

  /// One request over the wire, answered inside the budget.
  PlanMotion::Response::SharedPtr call(const PlanMotion::Request::SharedPtr & request)
  {
    EXPECT_TRUE(spin_until([this]() {return client_->service_is_ready();}));
    auto future = client_->async_send_request(request);
    const bool answered = spin_until(
      [&future]() {
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      });
    EXPECT_TRUE(answered) << "the service never answered";
    return answered ? future.get() : nullptr;
  }

  PlanMotion::Request::SharedPtr collision_blind_request()
  {
    auto request = std::make_shared<PlanMotion::Request>();
    request->goal = reachable_goal(goal_configuration());
    request->payload.shape = crane_msgs::msg::Payload::SHAPE_NONE;
    request->speed_scale = 1.0;
    // The .srv defaults this to true and this planner refuses that; a caller
    // that wants the collision-blind plan has to say so.
    request->avoid_collisions = false;
    return request;
  }

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::shared_ptr<crane_planning::PlannerNode> planner_;
  std::shared_ptr<rclcpp::Node> client_node_;
  rclcpp::Client<PlanMotion>::SharedPtr client_;
  rclcpp::Subscription<JointTrajectory>::SharedPtr reference_;
  rclcpp::Publisher<JointState>::SharedPtr joint_states_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_description_;
  rclcpp::Publisher<crane_msgs::msg::CollisionScene>::SharedPtr collision_scene_;
  std::vector<JointTrajectory> published_;
  std::optional<crane_model::Model> model_;
  rclcpp::Time start_stamp_;
};

}  // namespace

TEST_F(PlanMotionService, TheNodeIsNamedForItsRowInTheInterfaceTable)
{
  EXPECT_EQ(std::string(planner_->get_name()), "crane_planner");
}

TEST_F(PlanMotionService, AGoalInAnyOtherFrameIsRefusedNamingBoth)
{
  publish_start();
  ASSERT_TRUE(spin_until([this]() {return true;}));

  auto request = collision_blind_request();
  request->goal.header.frame_id = "world";
  const auto response = call(request);
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find("world"), std::string::npos) << response->message;
  EXPECT_NE(response->message.find(crane_planning::kPlanningFrame), std::string::npos)
    << response->message;
  EXPECT_TRUE(response->trajectory.points.empty());
}

TEST_F(PlanMotionService, CollisionAvoidanceWithoutASceneIsRefusedNamingTheTopic)
{
  publish_start();
  auto request = collision_blind_request();
  request->avoid_collisions = true;
  const auto response = call(request);
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find(crane_planning::kCollisionSceneTopic), std::string::npos)
    << response->message;
  EXPECT_TRUE(response->trajectory.points.empty());
}

TEST_F(PlanMotionService, ASceneInAnyOtherFrameIsRefusedNamingBoth)
{
  // ROS 2 Interfaces 4 gives this row `K0_mounting_base` and names the world
  // model as the element that has already converted `world` into it. A scene in
  // another frame is a scene about somewhere else.
  crane_msgs::msg::CollisionScene scene;
  scene.header.frame_id = "world";
  scene.header.stamp = client_node_->now();
  collision_scene_->publish(scene);
  publish_start();

  auto request = collision_blind_request();
  request->avoid_collisions = true;
  const auto response = call(request);
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find("world"), std::string::npos) << response->message;
  EXPECT_NE(response->message.find(crane_planning::kPlanningFrame), std::string::npos)
    << response->message;
}

TEST_F(PlanMotionService, ASubscribedSceneBecomesTheTruckOfTrajectoryPlanningFourTwo)
{
  // One primitive with the reserved id `truck`, parked well clear of this goal,
  // becomes the bed and the six runges -- and the answer says so, which is the
  // whole of "keyed to the measured truck pose rather than hard-coded".
  crane_msgs::msg::CollisionScene scene;
  scene.header.frame_id = crane_planning::kPlanningFrame;
  scene.header.stamp = client_node_->now();
  crane_msgs::msg::CollisionPrimitive truck;
  truck.id = crane_planning::kTruckId;
  truck.shape = crane_msgs::msg::CollisionScene::SHAPE_BOX;
  truck.structural = true;
  truck.pose.position.x = 18.0;
  truck.pose.orientation.w = 1.0;
  truck.dimensions.x = 6.5;
  truck.dimensions.y = 2.4;
  truck.dimensions.z = 1.2;
  scene.primitives.push_back(truck);
  collision_scene_->publish(scene);

  // A start and a goal in the middle of the machine's own ranges. The
  // collision-blind fixture the other tests use starts with the boom low and the
  // arm folded well in, where the rail gripper is close enough to the inner
  // telescope that 0.2 rad of sway lays one against the other -- a refusal the
  // self check is right to make and this test is not about.
  const crane_planning::JointLimits limits = fixture_limits();
  const crane_model::QA start =
    crane_planning_test::centred(limits, crane_planning_test::machines().front());
  publish_start(positions_of(start));

  auto request = collision_blind_request();
  request->goal = reachable_goal(crane_planning_test::moved(start, limits));
  request->avoid_collisions = true;
  const auto response = call(request);
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success) << response->message;
  EXPECT_FALSE(response->trajectory.points.empty());
  EXPECT_NE(response->message.find("runges"), std::string::npos) << response->message;
  EXPECT_NE(response->message.find("structural"), std::string::npos) << response->message;
  EXPECT_NE(response->message.find("4.3"), std::string::npos) << response->message;
}

TEST_F(PlanMotionService, ASpeedScaleOutsideTheUnitIntervalIsRefused)
{
  publish_start();
  for (const double speed_scale : {0.0, 1.5}) {
    auto request = collision_blind_request();
    request->speed_scale = speed_scale;
    const auto response = call(request);
    ASSERT_NE(response, nullptr);
    EXPECT_FALSE(response->success) << speed_scale;
    EXPECT_NE(response->message.find("speed_scale"), std::string::npos) << response->message;
  }
}

TEST_F(PlanMotionService, WithoutAStartStateNothingIsPlanned)
{
  // No `/joint_states` has been published, so the planner does not know where
  // the machine is and says so rather than assuming the neutral pose.
  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find(crane_planning::kJointStatesTopic), std::string::npos)
    << response->message;
}

TEST_F(PlanMotionService, AReachableGoalComesBackAsATimedJointTrajectory)
{
  publish_start();
  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success) << response->message;

  const JointTrajectory & trajectory = response->trajectory;
  // The six actuated joints of ROS 2 Interfaces 3.1, in that order.
  const std::vector<std::string> expected{
    "theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope",
    "theta8_rotator_joint", "q9_left_rail_joint"};
  EXPECT_EQ(trajectory.joint_names, expected);

  // Joint space has no geometric frame, and the stamp is the absolute time the
  // first point is valid for -- which is the measurement the plan starts from.
  EXPECT_TRUE(trajectory.header.frame_id.empty());
  EXPECT_EQ(rclcpp::Time(trajectory.header.stamp), start_stamp_);

  ASSERT_GE(trajectory.points.size(), 2U);
  for (const auto & point : trajectory.points) {
    EXPECT_EQ(point.positions.size(), crane_model::kActuatedDof);
    EXPECT_EQ(point.velocities.size(), crane_model::kActuatedDof);
    // Positions and velocities only.
    EXPECT_TRUE(point.accelerations.empty());
    EXPECT_TRUE(point.effort.empty());
  }

  // It starts where the machine is and it ends at rest.
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    EXPECT_NEAR(
      trajectory.points.front().positions[row],
      start_positions()[crane_planning::kActuatedRows[row]], 1.0e-12) << row;
    EXPECT_NEAR(trajectory.points.front().velocities[row], 0.0, 1.0e-12) << row;
    EXPECT_NEAR(trajectory.points.back().velocities[row], 0.0, 1.0e-12) << row;
  }
  // q8 is not a path variable (trajectory_planning 4.1) and rides the same time
  // base: it is carried on every point, and it is held.
  const std::size_t tool = crane_planning::kToolRow;
  for (const auto & point : trajectory.points) {
    EXPECT_NEAR(point.positions[tool], start_positions()[crane_planning::kActuatedRows[tool]],
      1.0e-12);
    EXPECT_NEAR(point.velocities[tool], 0.0, 1.0e-12);
  }

  // Strictly increasing time, first point at zero.
  EXPECT_EQ(rclcpp::Duration(trajectory.points.front().time_from_start).seconds(), 0.0);
  for (std::size_t index = 1; index < trajectory.points.size(); ++index) {
    EXPECT_GT(
      rclcpp::Duration(trajectory.points[index].time_from_start).seconds(),
      rclcpp::Duration(trajectory.points[index - 1].time_from_start).seconds());
  }

  // The endpoint really is the goal, checked against the model rather than
  // against the planner's own report.
  crane_model::QA q_a_goal;
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q_a_goal[static_cast<Eigen::Index>(row)] = trajectory.points.back().positions[row];
  }
  auto settled =
    model_->passive_equilibrium(q_a_goal, crane_planning_test::empty_gripper());
  ASSERT_TRUE(settled.ok());
  crane_model::Q q = crane_model::Q::Zero();
  for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
    q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
      q_a_goal[static_cast<Eigen::Index>(row)];
  }
  q[4] = settled.value()[0];
  q[5] = settled.value()[1];
  auto pose = model_->forward_kinematics(
    q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
  ASSERT_TRUE(pose.ok());
  const auto & goal = collision_blind_request()->goal.pose.position;
  EXPECT_LT(
    (pose.value().position_m - Eigen::Vector3d(goal.x, goal.y, goal.z)).norm(), 1.0e-3);

  // And the visualisation path is one pose per point, in the planning frame.
  ASSERT_EQ(response->tcp_path.size(), trajectory.points.size());
  for (const auto & waypoint : response->tcp_path) {
    EXPECT_EQ(waypoint.header.frame_id, std::string(crane_planning::kPlanningFrame));
  }
}

TEST_F(PlanMotionService, TheSameTrajectoryArrivesOnTheReferenceTopic)
{
  publish_start();
  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success) << response->message;

  ASSERT_TRUE(spin_until([this]() {return !published_.empty();}))
    << "nothing was published on " << crane_planning::kReferenceTopic;
  ASSERT_EQ(published_.size(), 1U);
  EXPECT_EQ(published_.front().joint_names, response->trajectory.joint_names);
  ASSERT_EQ(published_.front().points.size(), response->trajectory.points.size());
  EXPECT_EQ(
    published_.front().points.back().positions, response->trajectory.points.back().positions);
  EXPECT_TRUE(published_.front().header.frame_id.empty());
}

TEST_F(PlanMotionService, TheAnswerNamesWhichOfTheTwoMechanismsProducedTheGeometry)
{
  // trajectory_planning 4.4's ordering argument is that its two mechanisms are
  // not one product: the primitive's latency is bounded by construction and its
  // shape is the lift/traverse/descend an operator expects, while a sampled path
  // spent a budget to be found and rounds whatever was in the way. A caller that
  // cannot tell which it got cannot act on that difference.
  //
  // The service response is the only surface an out-of-process caller has -- the
  // in-process `MotionPlan::mechanism` the sampling suite asserts on is not on
  // the wire, because `crane_msgs/PlanMotion` has no row for it -- so the
  // sentence carrying it is a contract and not a log line.
  publish_start();
  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success) << response->message;

  // Nothing is in the way here, so 4.4's first mechanism is what answered.
  EXPECT_NE(
    response->message.find(
      crane_planning::mechanism_name(crane_planning::PathMechanism::StructuredPrimitive)),
    std::string::npos) << response->message;

  // And the sentence has to discriminate: one that named both, or hedged, would
  // leave the caller exactly where it started.
  EXPECT_EQ(
    response->message.find(
      crane_planning::mechanism_name(crane_planning::PathMechanism::SamplingFallback)),
    std::string::npos) << response->message;
}
