// The served contracts: `/crane/plan_motion` and `/a2b_movement` answered by a
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
#include "crane_msgs/srv/plan_motion.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "crane_planning/a2b_adapter.hpp"
#include "crane_planning/inverse_kinematics.hpp"
#include "crane_planning/planner_node.hpp"
#include "crane_planning/ompl_path_planner.hpp"
#include "description_fixture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "timber_crane_planning_interfaces/srv/calc_movement.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace
{

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

/// How many callbacks `drain()` lets through between the two halves of one `/joint_states` cycle.
constexpr int kDrainSpins = 20;

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
    // A second publisher for the passive half, because there are two broadcasters: one publisher
    // with a depth-1 history would drop whichever of the two partial messages it wrote first.
    passive_joint_states_ = client_node_->create_publisher<JointState>(
      crane_planning::kJointStatesTopic, crane_planning::input_qos());
    robot_description_ = client_node_->create_publisher<std_msgs::msg::String>(
      crane_planning::kRobotDescriptionTopic, crane_planning::robot_description_qos());
    collision_scene_ = client_node_->create_publisher<crane_msgs::msg::CollisionScene>(
      crane_planning::kCollisionSceneTopic, crane_planning::collision_scene_qos());
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

  /// Let every callback that is already queued run, without waiting on a predicate.
  /**
   * The two halves of `/joint_states` are two partial messages on one topic, and the input QoS is
   * `KeepLast(1)` on the *subscription* as well as on the publisher. Published back to back with
   * nothing spun in between, the second evicts the first and the planner is left holding one half.
   * The deployment never sees that because its executor is spinning between the two broadcasters'
   * 100 Hz writes; this stands in for that.
   */
  void drain()
  {
    for (int i = 0; i < kDrainSpins; ++i) {
      executor_.spin_once(std::chrono::milliseconds(1));
    }
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

  /// Publish both halves of `/joint_states` -- the actuated six and the passive pair -- at rest.
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
    drain();

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

  /// The actuated half alone, as `joint_state_broadcaster` publishes it.
  /**
   * `positions` and `velocities` are indexed over the canonical eight, as
   * `start_positions()` is; only the six actuated rows go out, because the two
   * halves of `/joint_states` are two separate partial messages and the passive
   * one comes from `publish_pendulum`. An empty `velocities` publishes no
   * `velocity` array at all.
   */
  void publish_joints(
    const std::vector<double> & positions, const std::vector<double> & velocities)
  {
    JointState joints;
    joints.header.stamp = client_node_->now();
    const std::array<std::string, crane_model::kGeneralizedDof> & names =
      model_->urdf_joint_names();
    for (const std::size_t row : crane_planning::kActuatedRows) {
      joints.name.push_back(names[row]);
      joints.position.push_back(positions[row]);
      if (!velocities.empty()) {
        joints.velocity.push_back(velocities[row]);
      }
    }
    start_stamp_ = joints.header.stamp;
    last_joints_ = joints;
    joint_states_->publish(joints);
  }

  /// The passive half alone, as `tip_tilt_state_broadcaster` publishes it.
  void publish_pendulum(
    const crane_model::QU & q_u, const crane_model::DQU & dq_u, double age_s = 0.0)
  {
    JointState passive;
    passive.header.stamp = client_node_->now() - rclcpp::Duration::from_seconds(age_s);
    const std::array<std::string, crane_model::kGeneralizedDof> & names =
      model_->urdf_joint_names();
    for (const std::size_t row : crane_planning::kPassiveRows) {
      passive.name.push_back(names[row]);
    }
    passive.position = {q_u[0], q_u[1]};
    passive.velocity = {dq_u[0], dq_u[1]};
    last_passive_ = passive;
    last_passive_age_s_ = age_s;
    passive_joint_states_->publish(passive);
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
   * the passive half between two requests would otherwise be racing its own
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

  /// One actuated configuration as the eight canonical positions the fixture indexes over.
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
    crane_model::Q q = crane_planning::expand(q_a);
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
  /// Both halves again, at this instant, as the two broadcasters would have them.
  /**
   * `settle()` spends up to a quarter of a second of wall clock, which is longer than the passive
   * pair's own 150 ms deadline. The deployment does not have that problem because both halves
   * stream at 100 Hz; a fixture that published each once would have every case refused for a
   * staleness it did not mean to pose. The deliberate age a case asked `publish_pendulum` for is
   * preserved, so a case that *does* mean to age the passive half still ages it.
   */
  void refresh_start()
  {
    if (last_joints_.has_value()) {
      last_joints_->header.stamp = client_node_->now();
      // The re-sent message *is* the start the answer will be stamped with, so the expectation
      // moves with it rather than staying on the copy this overwrote.
      start_stamp_ = last_joints_->header.stamp;
      joint_states_->publish(*last_joints_);
      drain();
    }
    if (last_passive_.has_value()) {
      last_passive_->header.stamp =
        client_node_->now() - rclcpp::Duration::from_seconds(last_passive_age_s_);
      passive_joint_states_->publish(*last_passive_);
      drain();
    }
  }

  PlanMotion::Response::SharedPtr call(const PlanMotion::Request::SharedPtr & request)
  {
    EXPECT_TRUE(spin_until([this]() {return client_->service_is_ready();}));
    refresh_start();
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
    crane_model::Q q = crane_planning::expand(q_a);
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
    // `/joint_states` to describe its probe state.
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
  rclcpp::Publisher<JointState>::SharedPtr passive_joint_states_;
  /// The newest of each half, kept so `refresh_start` can send them again unchanged.
  std::optional<JointState> last_joints_;
  std::optional<JointState> last_passive_;
  double last_passive_age_s_{0.0};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_description_;
  rclcpp::Publisher<crane_msgs::msg::CollisionScene>::SharedPtr collision_scene_;
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

TEST_F(PlanMotionService, ASceneOlderThanMaxSceneAgeIsRefusedRatherThanPlannedAgainst)
{
  // The whole of issue 096, end to end. This subscription is transient-local
  // depth 1, so the middleware hands a scene published once to every planner
  // that starts afterwards, for the rest of the session -- and before 096 the
  // node read `header.frame_id` and nothing else, so a latch of a world that had
  // since been rebuilt read exactly like the world. Here the scene is stamped
  // well past `max_scene_age` on arrival, which is the same thing the latch does
  // slowly, and the answer has to say so.
  crane_msgs::msg::CollisionScene scene;
  scene.header.frame_id = crane_planning::kPlanningFrame;
  scene.header.stamp = client_node_->now() - rclcpp::Duration::from_seconds(600.0);
  collision_scene_->publish(scene);
  publish_start();
  settle();

  auto request = collision_blind_request();
  request->avoid_collisions = true;
  const auto refused = call(request);
  ASSERT_NE(refused, nullptr);
  EXPECT_FALSE(refused->success) << refused->message;
  EXPECT_NE(refused->message.find("max_scene_age"), std::string::npos) << refused->message;
  EXPECT_NE(refused->message.find("transient-local"), std::string::npos) << refused->message;
  EXPECT_TRUE(refused->trajectory.points.empty());

  // ...and the other half of the decision: a caller that did not ask to avoid
  // collisions is not stopped by a scene it was not going to use. This goal is
  // out of reach, so the plan refuses for its own reason -- what is asserted is
  // that the reason is not the scene's age, and that the age is still stated.
  publish_start();
  settle();
  const auto warned = call(out_of_reach_request());
  ASSERT_NE(warned, nullptr);
  EXPECT_FALSE(warned->success);
  EXPECT_NE(warned->message.find("max_scene_age"), std::string::npos) << warned->message;
  EXPECT_NE(warned->message.find("avoid_collisions is clear"), std::string::npos)
    << warned->message;

  settle();
  EXPECT_TRUE(published_.empty());
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

TEST_F(PlanMotionService, AnUnusablePassiveStateIsRefusedNamingWhichOfTheTwoAbsencesItWas)
{
  // `wiki/control_architecture.md` 5.3: no input may stop arriving without a
  // defined consequence, and "never connected" and "died" are different things
  // for an operator to chase. This deployment's `passive_estimate_policy` is
  // `refuse`, which is 5.3's row for a state the planner closes on -- and the
  // consequence that is ruled out in both cases is the silent zero, because
  // an unknown sway is not a still one.
  //
  // There are two and not three: `sensor_msgs/JointState` carries no validity
  // flag, so a producer that has stopped publishing is the only failure this
  // source can report.
  publish_joints(start_positions(), std::vector<double>(start_positions().size(), 0.0));
  settle();
  const auto never = call(collision_blind_request());
  ASSERT_NE(never, nullptr);
  EXPECT_FALSE(never->success);
  EXPECT_NE(never->message.find(crane_planning::kJointStatesTopic), std::string::npos)
    << never->message;
  EXPECT_NE(never->message.find("never connected"), std::string::npos) << never->message;

  // Past its own 150 ms deadline, which is the row `control_architecture` 5.3's
  // table gives this input and not one age shared with every other one.
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero(), 1.0);
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
  const crane_model::Q q = crane_planning::expand(q_a_goal, settled.value());
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
  publish_pendulum(crane_model::QU::Zero(), crane_model::DQU::Zero(), 1.0);
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

TEST_F(PlanMotionService, TheAnswerNamesTheOmplGeometryPipeline)
{
  publish_start();
  const auto response = call(collision_blind_request());
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->success) << response->message;
  EXPECT_NE(response->message.find("OMPL RRT-Connect"), std::string::npos)
    << response->message;
  EXPECT_NE(response->message.find("C2"), std::string::npos) << response->message;
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
