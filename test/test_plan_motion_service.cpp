// The served contract: `/crane/plan_motion` and `/crane/plan_grip` answered by a
// real node, and the answer republished on `/crane/reference`.
//
// In process. The node object and a client are two nodes in one executor on the
// isolated domain `ralph/verify.sh` pins, with localhost-only transport: no
// launch, no controller manager, no simulator, no hardware. Nothing here sleeps
// -- every wait is a bounded spin on a predicate.
//
// The goal poses are not invented either: each one is produced by
// `crane_model::forward_kinematics` from a configuration the description allows,
// so what the service is asked for is a pose that certainly exists.
//
// The two services are one node and `wiki/implementation/ros2_interfaces.md` 10
// keeps them separate while the tested task layer migrates, so they are two
// fixtures here -- the second deriving from the first, because a grip is answered
// from the same model, the same `/joint_states` and the same scene.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "crane_msgs/msg/payload_estimate.hpp"
#include "crane_msgs/msg/pendulum_state.hpp"
#include "crane_msgs/srv/plan_grip.hpp"
#include "crane_msgs/srv/plan_motion.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "crane_planning/a2b_adapter.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/planner_node.hpp"
#include "crane_planning/sampling_planner.hpp"
#include "description_fixture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "timber_crane_planning_interfaces/srv/calc_movement.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace
{

using crane_msgs::srv::PlanGrip;
using crane_msgs::srv::PlanMotion;
using sensor_msgs::msg::JointState;
using timber_crane_planning_interfaces::srv::CalcMovement;
using trajectory_msgs::msg::JointTrajectory;

/// How long any one wait may take before the test fails, s. Generous, because it
/// bounds a failure rather than a success.
/**
 * It was 20 s while the timing was issue 038's ramp and one plan cost a few
 * seconds. The timing is now the OCP of `wiki/trajectory_planning.md` 5.2 and
 * one plan costs some 14 s -- of which the solve itself is 4 s and the rest is
 * the endpoint IK, the collision check and one `Model::passive_equilibrium` per
 * emitted point. Twenty seconds is then a coin toss rather than a bound, and a
 * flaky suite is worse than a slow one. **This is not a latency budget**: 7's
 * bounded latency is `PlannerSettings::latency` and is charged inside the planner
 * over every stage of one request, the OCP's own share of it is
 * `TimingOcpSettings::max_wall_clock`, and this number only decides how long a
 * *test* waits before calling a missing answer a failure.
 */
constexpr double kBudget = 60.0;

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
/**
 * Inside the arm's **working** range, which is narrower than the range the
 * description gives it. That did not matter while the timing was issue 038's
 * ramp, because a ramp constrains nothing; it matters now that the timing is the
 * OCP of `wiki/trajectory_planning.md` 5.2, which enforces `mpc.md` 3 constraint
 * 6 on the cylinder force. The boom-up, arm-out pose this used to hold --
 * `boom = 0.4, arm = 1.2` -- needs 0.91 of the arm cylinder's whole force just to
 * stand still, so there is nothing left under kappa = 0.8 for a move, and the
 * planner is right to refuse it. Here the same pose costs 0.49.
 */
const std::vector<double> & start_positions()
{
  static const std::vector<double> positions{0.0, -0.2, 0.6, 0.6, 0.0, 0.0, 0.0, 0.3};
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
    a2b_client_ = client_node_->create_client<CalcMovement>(crane_planning::kA2bMovementService);
    reference_ = client_node_->create_subscription<JointTrajectory>(
      crane_planning::kReferenceTopic, crane_planning::reference_qos(),
      [this](JointTrajectory::ConstSharedPtr message) {published_.push_back(*message);});
    joint_states_ = client_node_->create_publisher<JointState>(
      crane_planning::kJointStatesTopic, crane_planning::input_qos());
    robot_description_ = client_node_->create_publisher<std_msgs::msg::String>(
      crane_planning::kRobotDescriptionTopic, crane_planning::robot_description_qos());
    collision_scene_ = client_node_->create_publisher<crane_msgs::msg::CollisionScene>(
      crane_planning::kCollisionSceneTopic, crane_planning::collision_scene_qos());
    pendulum_state_ = client_node_->create_publisher<crane_msgs::msg::PendulumState>(
      crane_planning::kPendulumStateTopic, crane_planning::input_qos());
    payload_estimate_ = client_node_->create_publisher<crane_msgs::msg::PayloadEstimate>(
      crane_planning::kPayloadEstimateTopic, crane_planning::payload_estimate_qos());

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

  /// Publish one `/joint_states` and one `/crane/pendulum_state`, at rest.
  /**
   * Both, because both are the start state `wiki/trajectory_planning.md` 7 asks
   * for and the node refuses a request that is missing either. The velocities are
   * zero rather than absent: a message with no `velocity` array is a measurement
   * that does not say whether the machine is moving, and reading that as a
   * standstill is the stopped-start convention 7 removes -- so the node refuses
   * it, and `AJointStateWithoutVelocitiesIsRefusedRatherThanReadAsAStandstill`
   * asserts as much.
   */
  void publish_start() {publish_start(start_positions());}

  void publish_start(const std::vector<double> & positions)
  {
    publish_start(positions, std::vector<double>(positions.size(), 0.0));
  }

  void publish_start(
    const std::vector<double> & positions, const std::vector<double> & velocities)
  {
    publish_joints(positions, velocities);

    // The sway the tool really has at that configuration, measured off the model
    // rather than written as zero: the tool hangs where gravity puts it, and a
    // fixture that published `q_u = 0` would be posing a swinging start to every
    // case that meant to pose a settled one.
    crane_model::QA q_a = crane_model::QA::Zero();
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q_a[static_cast<Eigen::Index>(row)] = positions[crane_planning::kActuatedRows[row]];
    }
    auto settled = model_->passive_equilibrium(q_a, crane_planning_test::empty_gripper());
    EXPECT_TRUE(settled.ok()) << settled.status().message;
    publish_pendulum(settled.value(), crane_model::DQU::Zero());
  }

  /// `/joint_states` alone -- the half of the start state the encoders carry.
  void publish_joints(
    const std::vector<double> & positions, const std::vector<double> & velocities)
  {
    JointState joints;
    joints.header.stamp = client_node_->now();
    joints.name.assign(
      model_->urdf_joint_names().begin(), model_->urdf_joint_names().end());
    joints.position = positions;
    joints.velocity = velocities;
    start_stamp_ = joints.header.stamp;
    joint_states_->publish(joints);
  }

  /// One `/crane/pendulum_state`, at the sway and the rate the caller asks for.
  void publish_pendulum(
    const crane_model::QU & q_u, const crane_model::DQU & dq_u, bool valid = true,
    double age_s = 0.0)
  {
    crane_msgs::msg::PendulumState pendulum;
    pendulum.header.stamp = client_node_->now() - rclcpp::Duration::from_seconds(age_s);
    pendulum.position = {q_u[0], q_u[1]};
    pendulum.velocity = {dq_u[0], dq_u[1]};
    pendulum.valid = valid;
    pendulum.status = valid ? "all IMUs healthy and refreshing" : "IMU1x is degraded";
    pendulum_state_->publish(pendulum);
  }

  /// One `/crane/payload_estimate`, at the mass and validity the caller asks for.
  /**
   * `valid` on this message means "estimated at rest", and both CBS profiles
   * publish it `false` today (issue 033) -- so the branch that leaves the caller's
   * own declaration standing is the one that actually runs, and it is the one the
   * cases below spend most of their assertions on.
   */
  void publish_payload_estimate(double mass_kg, bool valid, double age_s = 0.0)
  {
    crane_msgs::msg::PayloadEstimate estimate;
    estimate.header.stamp = client_node_->now() - rclcpp::Duration::from_seconds(age_s);
    estimate.mass = mass_kg;
    // A first moment consistent with a centre 0.1 m out along K8's x, so the
    // substituted centre is a number a reader can check rather than a zero.
    estimate.m_r_x = 0.1 * mass_kg;
    estimate.m_r_y = 0.0;
    estimate.r_z = 0.0;
    estimate.valid = valid;
    payload_estimate_->publish(estimate);
  }

  /// The fixture's goal moved out of the machine's reach, so the endpoint refuses.
  /**
   * A refusal that comes from **inside** `plan_motion` rather than from the node's
   * own field checks, which is what the payload cases below need: the node reads
   * `/crane/payload_estimate` after the request's own payload and before it plans,
   * so the note saying what entered this plan only travels on an answer that got
   * that far.
   */
  PlanMotion::Request::SharedPtr out_of_reach_request()
  {
    auto request = collision_blind_request();
    request->goal.pose.position.x += 50.0;
    return request;
  }

  /// Spin a bounded number of times, to let anything that was going to arrive arrive.
  /**
   * Not a sleep and not a timeout: it is what makes "and nothing else was
   * published" a statement rather than a hope, in the one direction `spin_until`
   * cannot express.
   *
   * It is also what makes a *second* request in one case read the state published
   * just before it. `call` only spins while the service is not yet ready, so by
   * the second call it spins not at all before sending -- and a case that changes
   * `/crane/pendulum_state` between two requests would otherwise be racing its own
   * publication against its own request.
   */
  void settle()
  {
    for (int turn = 0; turn < 25; ++turn) {
      executor_.spin_once(std::chrono::milliseconds(10));
    }
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

  /// A goal that moves every axis worth moving, and that the machine can hold.
  /**
   * `arm = 1.6` used to stand here and it is 2.76 of the arm cylinder's force at
   * rest -- nearly three times what the hydraulics deliver, because 1.6 rad is
   * within a quarter radian of the transmission zero at 1.85 where `J_c,22`
   * vanishes and the cylinder has no moment arm about the joint at all. The
   * machine cannot hold that pose either, so it is not a goal to plan to. See
   * `start_positions()` above; this one costs 0.55.
   */
  static crane_model::QA goal_configuration()
  {
    crane_model::QA q_a;
    q_a << 0.9, -0.1, 0.8, 1.4, 0.4, 0.3;
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

  /// The retained request for the same reachable goal as `collision_blind_request`.
  CalcMovement::Request::SharedPtr a2b_collision_blind_request()
  {
    const crane_model::QA q_a = goal_configuration();
    crane_model::Q q = crane_model::Q::Zero();
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q[static_cast<Eigen::Index>(crane_planning::kActuatedRows[row])] =
        q_a[static_cast<Eigen::Index>(row)];
    }
    auto equilibrium =
      model_->passive_equilibrium(q_a, crane_planning_test::empty_gripper());
    EXPECT_TRUE(equilibrium.ok()) << equilibrium.status().message;
    q.segment<2>(4) = equilibrium.value();
    auto tip = model_->forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tip);
    auto tcp = model_->forward_kinematics(
      q, crane_model::Frame::MountingBase, crane_model::Frame::Tcp);
    EXPECT_TRUE(tip.ok() && tcp.ok());

    auto request = std::make_shared<CalcMovement::Request>();
    request->y_n.x = tip.value().position_m.x();
    request->y_n.y = tip.value().position_m.y();
    request->y_n.z = tip.value().position_m.z();
    request->phi_tool_n = crane_planning::phi_z_of(tcp.value().orientation);
    request->slow_down = 1.0;
    request->carries_log = false;
    request->check_log_collision = false;
    request->check_gripper_collision = false;
    request->publish_path = true;

    // The real concrete-block feasibility caller fills q0. Use a settled version
    // of this fixture's ordinary start so this request likewise needs no live
    // `/joint_states` or `/crane/pendulum_state` to describe its probe state.
    crane_model::QA q_a_start;
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      q_a_start[static_cast<Eigen::Index>(row)] =
        start_positions()[crane_planning::kActuatedRows[row]];
      request->q0[crane_planning::kActuatedRows[row]] =
        q_a_start[static_cast<Eigen::Index>(row)];
    }
    auto start_equilibrium =
      model_->passive_equilibrium(q_a_start, crane_planning_test::empty_gripper());
    EXPECT_TRUE(start_equilibrium.ok()) << start_equilibrium.status().message;
    request->q0[4] = start_equilibrium.value()[0];
    request->q0[5] = start_equilibrium.value()[1];
    return request;
  }

  CalcMovement::Response::SharedPtr call_a2b(const CalcMovement::Request::SharedPtr & request)
  {
    EXPECT_TRUE(spin_until([this]() {return a2b_client_->service_is_ready();}));
    auto future = a2b_client_->async_send_request(request);
    const bool answered = spin_until(
      [&future]() {
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      });
    EXPECT_TRUE(answered) << "the retained service never answered";
    return answered ? future.get() : nullptr;
  }

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::shared_ptr<crane_planning::PlannerNode> planner_;
  std::shared_ptr<rclcpp::Node> client_node_;
  rclcpp::Client<PlanMotion>::SharedPtr client_;
  rclcpp::Client<CalcMovement>::SharedPtr a2b_client_;
  rclcpp::Subscription<JointTrajectory>::SharedPtr reference_;
  rclcpp::Publisher<JointState>::SharedPtr joint_states_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_description_;
  rclcpp::Publisher<crane_msgs::msg::CollisionScene>::SharedPtr collision_scene_;
  rclcpp::Publisher<crane_msgs::msg::PendulumState>::SharedPtr pendulum_state_;
  rclcpp::Publisher<crane_msgs::msg::PayloadEstimate>::SharedPtr payload_estimate_;
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
    crane_planning_test::working_centred(limits, crane_planning_test::machines().front());
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

TEST_F(PlanMotionService, AJointStateWithoutVelocitiesIsRefusedRatherThanReadAsAStandstill)
{
  // `wiki/trajectory_planning.md` 7 asks for `(q, dq)` **as measured**, and a
  // `sensor_msgs/JointState` with no `velocity` array is a measurement that does
  // not say whether the machine is moving. Reading it as a standstill is exactly
  // the *"we deactivated qDot0, because we now always start in a stopped state"*
  // the page names as the defect not to inherit, so it is refused instead and the
  // refusal names the rate rather than the message.
  publish_joints(start_positions(), {});
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero());
  settle();

  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find("velocity"), std::string::npos) << response->message;
  EXPECT_NE(response->message.find(crane_planning::kJointStatesTopic), std::string::npos)
    << response->message;

  // And nothing was adopted, so nothing was published: the topic is
  // transient-local and a refusal that wrote to it would leave a late subscriber
  // latching onto a plan nobody is executing.
  settle();
  EXPECT_TRUE(published_.empty());
}

TEST_F(PlanMotionService, AnUnusablePendulumStateIsRefusedNamingWhichOfTheThreeAbsencesItWas)
{
  // `wiki/control_architecture.md` 5.3: no input may stop arriving without a
  // defined consequence, and "never connected" and "died" are different things
  // for an operator to chase. This deployment's `passive_estimate_policy` is
  // `refuse`, which is 5.3's row for a state the planner closes on -- and the
  // consequence that is ruled out in all three cases is the silent zero, because
  // an unknown sway is not a still one.
  //
  // The three arrive in the order they can: nothing has been received at all,
  // then a producer that says `valid == false` about itself, then one that
  // arrived and stopped.
  publish_joints(start_positions(), std::vector<double>(start_positions().size(), 0.0));
  settle();
  const auto never = call(collision_blind_request());
  ASSERT_NE(never, nullptr);
  EXPECT_FALSE(never->success);
  EXPECT_NE(never->message.find(crane_planning::kPendulumStateTopic), std::string::npos)
    << never->message;
  EXPECT_NE(never->message.find("never connected"), std::string::npos) << never->message;

  // The producer is running, is talking, and is telling the truth about itself.
  // Its own `status` is carried rather than restated: the three causes behind the
  // flag are the broadcaster's to judge inside its own cycle.
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero(), false);
  settle();
  const auto not_valid = call(collision_blind_request());
  ASSERT_NE(not_valid, nullptr);
  EXPECT_FALSE(not_valid->success);
  EXPECT_NE(not_valid->message.find("valid == false"), std::string::npos) << not_valid->message;
  EXPECT_NE(not_valid->message.find("IMU1x is degraded"), std::string::npos)
    << not_valid->message;

  // Past its own 150 ms deadline, which is the row `control_architecture` 5.3's
  // table gives this topic and not one age shared with every other input.
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero(), true, 1.0);
  settle();
  const auto stale = call(collision_blind_request());
  ASSERT_NE(stale, nullptr);
  EXPECT_FALSE(stale->success);
  EXPECT_NE(stale->message.find("died"), std::string::npos) << stale->message;
  EXPECT_NE(stale->message.find("deadline"), std::string::npos) << stale->message;

  settle();
  EXPECT_TRUE(published_.empty());
}

TEST_F(PlanMotionService, ThePayloadEstimateEntersWhereItIsValidAndIsStatedWhereItIsNot)
{
  // The estimate is handled on the pendulum's own terms -- stated, never assumed --
  // and the answer says which of the two happened. The goal is out of reach in
  // every case here, because what is under test is what the node put into the plan
  // and not whether the plan closed; the endpoint refuses and the note travels
  // with the refusal.
  publish_start();

  const auto absent = call(out_of_reach_request());
  ASSERT_NE(absent, nullptr);
  EXPECT_FALSE(absent->success);
  EXPECT_NE(absent->message.find(crane_planning::kPayloadEstimateTopic), std::string::npos)
    << absent->message;
  EXPECT_NE(absent->message.find("never connected"), std::string::npos) << absent->message;

  // The branch both CBS profiles actually run today (issue 033): the estimator is
  // there and says it has no estimate taken at rest, so the caller's declaration
  // stands and the answer says so rather than quietly substituting numbers the
  // estimator does not stand behind.
  publish_payload_estimate(137.0, false);
  // ...and the start state again with it. Each `out_of_reach_request()` above
  // spends some hundreds of milliseconds inside the equilibrium-constrained solve
  // before it refuses, so a `/joint_states` published once at the top of the case
  // is past `max_input_age` by the second call and the node refuses for *that*
  // reason instead -- a wall-clock race, and one this case is not about.
  publish_start();
  settle();
  const auto not_valid = call(out_of_reach_request());
  ASSERT_NE(not_valid, nullptr);
  EXPECT_FALSE(not_valid->success);
  EXPECT_NE(not_valid->message.find("valid == false"), std::string::npos) << not_valid->message;
  EXPECT_NE(
    not_valid->message.find("the payload is the one the request declared"), std::string::npos)
    << not_valid->message;
  // ...and it is not consumed: the mass it carries appears nowhere in what was planned.
  EXPECT_EQ(not_valid->message.find("137.000000 kg"), std::string::npos) << not_valid->message;

  // Where it *is* valid it replaces what the request declared, because the
  // estimator measured the machine and the request described it
  // (`wiki/robot_model.md` 5.3: for a gravity moment the payload is a point mass,
  // which is exactly what this message carries).
  publish_payload_estimate(137.0, true);
  publish_start();
  settle();
  const auto valid = call(out_of_reach_request());
  ASSERT_NE(valid, nullptr);
  EXPECT_FALSE(valid->success);
  EXPECT_NE(valid->message.find("137.000000 kg"), std::string::npos) << valid->message;
  EXPECT_NE(valid->message.find("It replaces what the request declared"), std::string::npos)
    << valid->message;

  settle();
  EXPECT_TRUE(published_.empty());
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
    EXPECT_NEAR(
      point.positions[tool], start_positions()[crane_planning::kActuatedRows[tool]],
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

TEST_F(PlanMotionService, ARefusedReplanLeavesTheStandingReferenceAloneAndReportsIt)
{
  // `wiki/trajectory_planning.md` 7's fallback, over the wire. A re-plan that is
  // refused keeps the previous trajectory and reports it; it never emits a partial
  // plan and it never republishes. `/crane/reference` is transient-local, so a
  // refusal that wrote to it would leave a late subscriber latching onto a plan
  // nobody is executing -- and a caller has to be able to tell "I kept planning"
  // from "here is something new".
  publish_start();
  const auto adopted = call(collision_blind_request());
  ASSERT_NE(adopted, nullptr);
  ASSERT_TRUE(adopted->success) << adopted->message;
  ASSERT_TRUE(spin_until([this]() {return !published_.empty();}))
    << "nothing was published on " << crane_planning::kReferenceTopic;
  ASSERT_EQ(published_.size(), 1U);

  // Now the passive estimate goes away mid-execution, which is the situation a
  // stall-recovery re-plan is issued in and the one 7's `[!warning]` is about.
  publish_start();
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero(), false);
  settle();
  const auto refused = call(collision_blind_request());
  ASSERT_NE(refused, nullptr);
  EXPECT_FALSE(refused->success);

  // Which trajectory is still standing is visible in the answer, and it is the
  // adopted one rather than an empty field: "the previous plan is still running"
  // and "I have nothing for you" are different answers to the same request.
  ASSERT_EQ(refused->trajectory.points.size(), adopted->trajectory.points.size());
  EXPECT_EQ(refused->trajectory.joint_names, adopted->trajectory.joint_names);
  EXPECT_EQ(
    refused->trajectory.points.back().positions, adopted->trajectory.points.back().positions);
  EXPECT_EQ(
    rclcpp::Time(refused->trajectory.header.stamp),
    rclcpp::Time(adopted->trajectory.header.stamp));
  EXPECT_NE(
    refused->message.find("is unchanged and is the one still being executed"),
    std::string::npos) << refused->message;
  // No partial plan travelled with it either: `success` is false, so the
  // trajectory on the response is the standing one and not something new.
  EXPECT_TRUE(refused->tcp_path.empty());

  // And the reference itself was left alone.
  settle();
  EXPECT_EQ(published_.size(), 1U);
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

TEST_F(PlanMotionService, TheRetainedA2bServiceUsesThisNodesNativePlanningPath)
{
  // There is one PlannerNode in this executor. Its retained service receives a
  // K5 goal built from a real forward-kinematics pose, translates it to K_tcp,
  // and the adopted answer appears on the same `/crane/reference` publisher the
  // native service uses. A separate adapter node or a second planner could not
  // satisfy that ownership assertion in this fixture.
  const auto response = call_a2b(a2b_collision_blind_request());
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success);
  ASSERT_FALSE(response->trajectory.points.empty());
  ASSERT_EQ(response->tcp_path.size(), response->trajectory.points.size());

  ASSERT_TRUE(spin_until([this]() {return !published_.empty();}))
    << "nothing was published on " << crane_planning::kReferenceTopic;
  ASSERT_EQ(published_.size(), 1U);
  EXPECT_EQ(published_.front().joint_names, response->trajectory.joint_names);
  EXPECT_EQ(
    published_.front().points.back().positions,
    response->trajectory.points.back().positions);
}

namespace
{

/// `/crane/plan_grip`, the same node's second row in ROS 2 Interfaces 5.
/**
 * What this fixture adds to the one above is a client and nothing else: a grip is
 * answered from the same model, the same `/joint_states` and the same subscribed
 * scene, because it is the same node object. That is the point of testing it
 * here rather than beside `test_plan_grip.cpp` -- the offline suite is where the
 * four phases are asserted, and this is where "the node serves it, and its answer
 * is `/crane/plan_motion`'s own answer shape" is.
 *
 * Two of the four phases solve the OCP of `wiki/trajectory_planning.md` 5.2,
 * because a grip's descend and lift are `plan_motion` itself. This fixture is
 * therefore priced like the motion cases above and not like a tool phase, which
 * is a cosine and costs nothing.
 */
class PlanGripService : public PlanMotionService
{
protected:
  void SetUp() override
  {
    PlanMotionService::SetUp();
    grip_client_ = client_node_->create_client<PlanGrip>(crane_planning::kPlanGripService);
  }

  /// A truck parked well clear of everything this fixture plans through.
  /**
   * `crane_msgs/PlanGrip` carries no `avoid_collisions` row and the .srv is
   * frozen, so the node reads the absent field as the checked plan -- a grip's
   * arm phase is the move that puts a tool between the runges. An arm phase with
   * nothing on the scene topic is therefore refused, which is asserted below and
   * is why every phase meant to succeed publishes this first.
   */
  void park_the_truck()
  {
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
  }

  /// One phase, with the fields the .srv gives every phase.
  /**
   * The goal is left in no frame at all, and the two tool phases are called with
   * it that way on purpose: a close reads no pose, so demanding a frame of a
   * field the phase does not use would refuse a well-formed request. The arm
   * phases fill it in below.
   */
  static PlanGrip::Request::SharedPtr grip_request(std::uint8_t phase)
  {
    auto request = std::make_shared<PlanGrip::Request>();
    request->phase = phase;
    request->payload.shape = crane_msgs::msg::Payload::SHAPE_NONE;
    request->speed_scale = 1.0;
    return request;
  }

  PlanGrip::Response::SharedPtr call_grip(const PlanGrip::Request::SharedPtr & request)
  {
    EXPECT_TRUE(spin_until([this]() {return grip_client_->service_is_ready();}));
    auto future = grip_client_->async_send_request(request);
    const bool answered = spin_until(
      [&future]() {
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      });
    EXPECT_TRUE(answered) << "the service never answered";
    return answered ? future.get() : nullptr;
  }

  /// `/crane/plan_motion`'s own answer shape: six rows, one absolute stamp, one grid.
  void expect_plan_motions_wire_form(const JointTrajectory & trajectory, const char * what)
  {
    // The six actuated joints of ROS 2 Interfaces 3.1, in that order -- including
    // q8, which trajectory_planning 4.1 keeps out of the path and on the clock.
    const std::vector<std::string> expected{
      "theta1_slewing_joint", "theta2_boom_joint", "theta3_arm_joint", "q4_big_telescope",
      "theta8_rotator_joint", "q9_left_rail_joint"};
    EXPECT_EQ(trajectory.joint_names, expected) << what;

    // Joint space has no geometric frame, and the stamp is the absolute time the
    // first point is valid for -- which is the measurement the phase started from.
    EXPECT_TRUE(trajectory.header.frame_id.empty()) << what;
    EXPECT_EQ(rclcpp::Time(trajectory.header.stamp), start_stamp_) << what;

    ASSERT_GE(trajectory.points.size(), 2U) << what;
    for (const auto & point : trajectory.points) {
      EXPECT_EQ(point.positions.size(), crane_model::kActuatedDof) << what;
      EXPECT_EQ(point.velocities.size(), crane_model::kActuatedDof) << what;
      EXPECT_TRUE(point.accelerations.empty()) << what;
      EXPECT_TRUE(point.effort.empty()) << what;
    }

    // One clock: the first point at zero, strictly increasing after it.
    EXPECT_EQ(rclcpp::Duration(trajectory.points.front().time_from_start).seconds(), 0.0) << what;
    for (std::size_t index = 1; index < trajectory.points.size(); ++index) {
      EXPECT_GT(
        rclcpp::Duration(trajectory.points[index].time_from_start).seconds(),
        rclcpp::Duration(trajectory.points[index - 1].time_from_start).seconds()) << what;
    }
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      EXPECT_NEAR(trajectory.points.front().velocities[row], 0.0, 1.0e-9) << what << " row " << row;
      EXPECT_NEAR(trajectory.points.back().velocities[row], 0.0, 1.0e-9) << what << " row " << row;
    }
  }

  rclcpp::Client<PlanGrip>::SharedPtr grip_client_;
};

}  // namespace

TEST_F(PlanGripService, AllFourPhasesComeBackOnTheSixActuatedJointsAndOnOneClock)
{
  // The served half of the issue's first criterion: all four phases, each
  // returning a `JointTrajectory` on the same six joints and the same stamp
  // semantics as `/crane/plan_motion`. Two of them are arm motions through the
  // OCP and two are the tool's own cosine, and 4.1's claim is that a caller
  // cannot tell which from the shape of the answer -- geometrically the tool is
  // decoupled from the arm, temporally it is not.
  const crane_planning::JointLimits limits = fixture_limits();
  const crane_model::QA start =
    crane_planning_test::working_centred(limits, crane_planning_test::machines().front());
  crane_model::QA lower = start;
  lower[1] -= 0.15;  // theta2_boom_joint, which drops the tool
  crane_model::QA higher = start;
  higher[1] += 0.15;

  park_the_truck();

  auto descend = grip_request(PlanGrip::Request::PHASE_DESCEND);
  descend->goal = reachable_goal(lower);
  auto lift = grip_request(PlanGrip::Request::PHASE_LIFT);
  lift->goal = reachable_goal(higher);

  const std::array<std::pair<const char *, PlanGrip::Request::SharedPtr>, 4U> phases{
    {{"descend", descend},
      {"close", grip_request(PlanGrip::Request::PHASE_CLOSE)},
      {"open", grip_request(PlanGrip::Request::PHASE_OPEN)},
      {"lift", lift}}};

  for (const auto & phase : phases) {
    // The machine is re-measured before every phase, and that is not test
    // scaffolding: a served descend costs some 7 s -- the endpoint IK, the
    // collision check and the OCP of 5.2 -- against a `max_input_age` of 0.5 s,
    // so four phases answered from one `/joint_states` would have three of them
    // refused for planning a crane that has moved. Sequencing the four by moving
    // the machine and re-measuring is what the task layer does anyway; it is the
    // same fact that makes the planner able to hold no state between phases.
    publish_start(positions_of(start));

    const auto response = call_grip(phase.second);
    ASSERT_NE(response, nullptr) << phase.first;
    ASSERT_TRUE(response->success) << phase.first << ": " << response->message;
    expect_plan_motions_wire_form(response->trajectory, phase.first);

    // Every phase starts at the measurement it was answered from, over all six
    // rows -- which is what makes a sequence join without the planner
    // remembering which phase ran last. The four are asked in one order here and
    // the planner is answering each from the same `/joint_states`, so if it kept
    // any phase-to-phase state at all this is where it would show.
    for (std::size_t row = 0; row < crane_model::kActuatedDof; ++row) {
      EXPECT_NEAR(
        response->trajectory.points.front().positions[row],
        start[static_cast<Eigen::Index>(row)], 1.0e-12) << phase.first << " row " << row;
    }
  }
}

TEST_F(PlanGripService, APhaseTheMessageDoesNotDefineIsRefusedRatherThanMappedOntoTheNearest)
{
  // `crane_msgs/PlanGrip` is frozen at four phases (PRD 15). A fifth value is a
  // caller talking about something this .srv does not define, and the refusal
  // names the four rather than picking whichever is closest.
  publish_start();
  auto request = grip_request(5U);
  request->goal.header.frame_id = crane_planning::kPlanningFrame;
  const auto response = call_grip(request);
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->success);
  EXPECT_NE(response->message.find("PHASE_DESCEND"), std::string::npos) << response->message;
  EXPECT_TRUE(response->trajectory.points.empty());
}

TEST_F(PlanGripService, AnArmPhaseIsCheckedAgainstTheSceneAndAToolPhaseNeedsNeitherSceneNorGoal)
{
  // The frozen .srv has no `avoid_collisions` row, so the node has to decide what
  // its absence means, and for a grip it means the checked plan. With nothing on
  // the scene topic a descend is refused naming it -- while a close moves no link
  // that was not already where it is, reads no goal, and is answered.
  const crane_planning::JointLimits limits = fixture_limits();
  const crane_model::QA start =
    crane_planning_test::working_centred(limits, crane_planning_test::machines().front());
  publish_start(positions_of(start));

  crane_model::QA lower = start;
  lower[1] -= 0.15;
  auto descend = grip_request(PlanGrip::Request::PHASE_DESCEND);
  descend->goal = reachable_goal(lower);
  const auto refused = call_grip(descend);
  ASSERT_NE(refused, nullptr);
  EXPECT_FALSE(refused->success);
  EXPECT_NE(refused->message.find(crane_planning::kCollisionSceneTopic), std::string::npos)
    << refused->message;
  // ...and it says which phase it was answering, because the task layer calls
  // this service four times for one grip.
  EXPECT_NE(refused->message.find("descend phase"), std::string::npos) << refused->message;
  EXPECT_TRUE(refused->trajectory.points.empty());

  const auto planned = call_grip(grip_request(PlanGrip::Request::PHASE_CLOSE));
  ASSERT_NE(planned, nullptr);
  ASSERT_TRUE(planned->success) << planned->message;
  EXPECT_FALSE(planned->trajectory.points.empty());
}

TEST_F(PlanGripService, AGripPhaseArrivesOnTheReferenceTopicToo)
{
  // ROS 2 Interfaces 4 gives `/crane/reference` all six actuated coordinates and
  // trajectory_planning 4.1 puts q8 among them, so `q8_ref` needs a producer
  // while a grip is running and this is it. A close is the cheapest phase to ask
  // that with: no OCP, no endpoint IK, just the cosine.
  publish_start();
  const auto response = call_grip(grip_request(PlanGrip::Request::PHASE_CLOSE));
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
