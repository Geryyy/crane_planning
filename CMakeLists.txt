cmake_minimum_required(VERSION 3.8)
project(crane_planning)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(crane_model REQUIRED)
find_package(crane_msgs REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ruckig REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(urdfdom REQUIRED)

# The planner's algorithm core: the semi-analytic inverse kinematics of
# wiki/robot_model.md §2.2, the two-link geometry probed out of the model, the
# joint limits read off the description, and the timing ramp. ROS-free -- no
# node, no clock, no graph (wiki/implementation/style_guide.md §3), so the whole
# of it is tested offline against the machine's own descriptions.
#
# STATIC, for the reason `crane_control` records: crane_model installs a static
# archive, and the only consumer here is an executable rather than a pluginlib
# object, so no `--exclude-libs` workaround is needed.
add_library(planner_core STATIC
  src/arm_geometry.cpp
  src/equilibrium_ik.cpp
  src/geometric_path.cpp
  src/inverse_kinematics.cpp
  src/joint_limits.cpp
  src/planner_core.cpp
  src/structured_primitive.cpp
  src/trajectory_timing.cpp)
add_library(${PROJECT_NAME}::planner_core ALIAS planner_core)
set_target_properties(planner_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(planner_core PUBLIC cxx_std_17)
target_include_directories(planner_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${urdfdom_INCLUDE_DIRS})
# urdfdom is PRIVATE and is used for exactly one thing: the position and
# velocity limits the description gives the six actuated joints, which the frozen
# model API does not expose and this planner may not invent (see
# include/crane_planning/joint_limits.hpp).
# ruckig is PUBLIC because `geometric_path.hpp` holds one of its trajectories per
# path segment: `wiki/implementation/libraries.md` §1 lists it for "jerk-limited
# interpolation and the clamped B-spline fit" against "spline bases,
# boundary-condition solves", and the C² primitive of trajectory_planning §4.4 is
# exactly a sequence of boundary-condition solves. No spline basis is written here.
target_link_libraries(planner_core
  PUBLIC
    crane_model::crane_model
    Eigen3::Eigen
    ruckig::ruckig
  PRIVATE
    urdfdom::urdfdom_model)

generate_parameter_library(crane_planner_parameters
  src/crane_planner_parameters.yaml)

# The adapter, as a library rather than as part of the executable: the service
# test constructs this node object in process, so what is asserted is the node
# the deployment runs and not a second one written for the test.
add_library(planner_node STATIC src/planner_node.cpp)
add_library(${PROJECT_NAME}::planner_node ALIAS planner_node)
set_target_properties(planner_node PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(planner_node PUBLIC cxx_std_17)
target_include_directories(planner_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(planner_node
  PUBLIC
    planner_core
    rclcpp::rclcpp
    ${crane_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${trajectory_msgs_TARGETS}
  PRIVATE
    crane_planner_parameters)

# Beside the manager and not inside it (ROS 2 Interfaces §2), so it is an
# executable a launch profile composes rather than a plugin a spawner loads.
add_executable(crane_planner_node src/planner_main.cpp)
target_link_libraries(crane_planner_node PRIVATE planner_node)

install(
  DIRECTORY include/
  DESTINATION include)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(
  TARGETS planner_core planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  TARGETS crane_planner_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(planner_core planner_node)
ament_export_dependencies(
  crane_model
  crane_msgs
  Eigen3
  geometry_msgs
  rclcpp
  ruckig
  sensor_msgs
  std_msgs
  trajectory_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # The machine's own descriptions, read where they live rather than copied.
  # They are generated from src/epsilon_crane_description and checked in beside
  # crane_model's contract test, which is what holds crane_model's compiled
  # geometry in step with them -- so a copy here would be a second machine to
  # keep current. Read only, one file per tool.
  set(CRANE_PLANNING_MACHINE_DESCRIPTION_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../crane_model/test/description"
    CACHE PATH "Directory holding the machine descriptions the IK is exercised against")

  # S4: the two-link structure the IK closes on, probed out of both machine
  # descriptions and checked against forward kinematics. Offline C++ -- no ROS,
  # no DDS, no simulator is linked into this binary and none is reachable from
  # it.
  ament_add_gtest(test_arm_geometry test/test_arm_geometry.cpp)
  target_link_libraries(test_arm_geometry planner_core)
  target_compile_definitions(test_arm_geometry PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the semi-analytic IK of robot_model §2.2 over a sampled set of
  # reachable poses per tool, with the forward-kinematics residual as the oracle.
  # Every case is generated *by* forward kinematics from a configuration the
  # description allows, so reachability is known by construction rather than
  # assumed.
  ament_add_gtest(test_inverse_kinematics test/test_inverse_kinematics.cpp TIMEOUT 300)
  target_link_libraries(test_inverse_kinematics planner_core)
  target_compile_definitions(test_inverse_kinematics PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the equilibrium-constrained NLP of robot_model §2.2, over sampled
  # goal poses per tool with *two* oracles -- the forward-kinematics residual of
  # §2.2 and the equilibrium residual against Model::passive_equilibrium. Offline
  # C++, same descriptions, no ROS.
  ament_add_gtest(test_equilibrium_ik test/test_equilibrium_ik.cpp TIMEOUT 900)
  target_link_libraries(test_equilibrium_ik planner_core)
  target_compile_definitions(test_equilibrium_ik PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the timing ramp and the refusals the core owes. Plain data in,
  # plain data out.
  ament_add_gtest(test_trajectory_timing test/test_trajectory_timing.cpp TIMEOUT 600)
  target_link_libraries(test_trajectory_timing planner_core)
  target_compile_definitions(test_trajectory_timing PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the structured lift/traverse/descend primitive of
  # trajectory_planning §4.4, its C² junctions, the transfer altitude derived
  # from the scene, and the refusals it owes -- over both machine descriptions,
  # because the descend phase's clearance is the mounted tool's own reach and the
  # rail gripper and the jaw do not have the same one. Offline C++: the only
  # thing this binary links besides gtest is the planner core.
  ament_add_gtest(test_structured_primitive test/test_structured_primitive.cpp TIMEOUT 900)
  target_link_libraries(test_structured_primitive planner_core)
  target_compile_definitions(test_structured_primitive PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S5: the served contract, on the isolated domain `verify.sh` pins. Two nodes
  # in one in-process executor -- the deployment's own node object and a client
  # -- and no launch, no controller manager and no simulator, because the planner
  # claims no interface and joins no cascade.
  ament_add_gtest(test_plan_motion_service test/test_plan_motion_service.cpp TIMEOUT 300)
  target_link_libraries(test_plan_motion_service
    planner_node
    ${crane_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${trajectory_msgs_TARGETS})
  target_compile_definitions(test_plan_motion_service PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")
endif()

ament_package()
