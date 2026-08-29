cmake_minimum_required(VERSION 3.8)
project(crane_planning)

find_package(ament_cmake REQUIRED)
# acados, for the path-constrained OCP of trajectory_planning §5.2.
# `wiki/implementation/libraries.md` lists it for "the optimal control problems of
# `mpc` and `trajectory_planning`", so no NLP is written here.
#
# **How CI resolves it.** The image installs acados to `/usr/local` and its config
# lands at `/usr/local/cmake/acadosConfig.cmake` -- not under `lib/cmake` or
# `share/*/cmake`, where CMake's UNIX config search looks. It is nevertheless
# found without a hint, because `<prefix>/cmake/` is on the search list too and
# `/usr/local` is a default system prefix. The `ACADOS_SOURCE_DIR=/opt/acados`
# the image exports is the *source* tree and is not used here; only the install
# is. If a future image drops the config, `acados_DIR` is the one variable to set
# and this is the line that would need the hint.
find_package(acados REQUIRED)
# CasADi is an export-time dependency only.  The generated C below is the only
# CasADi output that reaches the planner binary; no live CasADi graph or bridge
# is linked at runtime.
find_package(Eigen3 REQUIRED)
find_package(crane_model REQUIRED)
find_package(crane_msgs REQUIRED)
# The acados layer, once: the capsule binding both this package and `crane_mpc`
# open their generated solvers through.
find_package(crane_ocp REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
# OMPL, the sole geometric path planner.
# `wiki/implementation/libraries.md` §1 lists it for sampling-based path planning
# against "RRT/PRM and their smoothing", so no RRT is written here.
#
# It needs no hint wherever it comes from: Debian's `libompl-dev` puts its config
# at /usr/share/ompl/cmake and `ros-humble-ompl` puts it at
# /opt/ros/humble/share/ompl/cmake, CMake's config-mode search covers
# `<prefix>/share/<name>*/cmake`, and both prefixes are already on the path. Only
# the OMPL_* variables are used below and never an `ompl::ompl` target: 1.5 exports
# no target and 1.7 does, and the variables are what both carry.
#
# `package.xml` declares `<depend>ompl</depend>`, which rosdep resolves to
# `ros-humble-ompl`. **The devcontainer image installs neither yet** -- see the
# build note at the top of README.md -- so a build in a fresh image fails here
# until that one line lands. REQUIRED is deliberate: a planner that quietly
# compiled without its fallback is worse than one that does not build.
find_package(ompl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ruckig REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# The retained `a2b_movement` contract of ROS 2 Interfaces §9 and the two message
# packages that carry it. They are **retained-tree interface packages**, not new
# third-party dependencies: `libraries.md` governs libraries, and these are the
# `.srv` and `.msg` definitions the timber stack already ships and that §9 keeps
# compatible rather than carrying over. Nothing in this package includes a
# `wood_log_msgs` header directly -- `CalcMovement` embeds `LogShape` and `Log[]`
# and pulls it in -- but it is named here anyway, because a transitive interface
# dependency that is not declared is one a manifest walk cannot see.
find_package(timber_crane_planning_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(urdfdom REQUIRED)
find_package(wood_log_msgs REQUIRED)

# The planner's algorithm core: the semi-analytic inverse kinematics of
# wiki/robot_model.md §2.2, the two-link geometry probed out of the model, the
# joint limits read off the description, and the timing ramp. ROS-free -- no
# node, no clock, no graph (wiki/implementation/style_guide.md §3), so the whole
# of it is tested offline against the machine's own descriptions.
#
# STATIC, for the reason `crane_control` records: crane_model installs a static
# archive, and the only consumer here is an executable rather than a pluginlib
# object, so no `--exclude-libs` workaround is needed.
set(CRANE_PLANNING_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
set(CRANE_PLANNING_SOLVERS
  crane_planning_timing_pzs100
  crane_planning_timing_epsilon7040)
set(CRANE_PLANNING_GENERATED_SOURCES "")
set(CRANE_PLANNING_GENERATED_INCLUDES "${CRANE_PLANNING_GENERATED_DIR}")
foreach(solver IN LISTS CRANE_PLANNING_SOLVERS)
  file(GLOB_RECURSE generated CONFIGURE_DEPENDS
    "${CRANE_PLANNING_GENERATED_DIR}/${solver}/*.c")
  if(NOT generated)
    message(FATAL_ERROR
      "`generated/${solver}/` is empty. It is checked in, not built: run "
      "`./scripts/export_timing_ocp.py` from ${CMAKE_CURRENT_SOURCE_DIR}.")
  endif()
  list(APPEND CRANE_PLANNING_GENERATED_SOURCES ${generated})
  list(APPEND CRANE_PLANNING_GENERATED_INCLUDES
    "${CRANE_PLANNING_GENERATED_DIR}/${solver}")
endforeach()
set_source_files_properties(${CRANE_PLANNING_GENERATED_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-w")

add_library(crane_planning_generated_ocp STATIC ${CRANE_PLANNING_GENERATED_SOURCES})
target_include_directories(crane_planning_generated_ocp PRIVATE ${CRANE_PLANNING_GENERATED_INCLUDES})
target_link_libraries(crane_planning_generated_ocp PRIVATE acados)

add_library(planner_core STATIC
  src/arm_geometry.cpp
  src/collision.cpp
  src/equilibrium_ik.cpp
  src/geometric_path.cpp
  src/inverse_kinematics.cpp
  src/joint_limits.cpp
  src/planner_core.cpp
  src/replanning.cpp
  src/scene_age.cpp
  src/ompl_path_planner.cpp
  src/timing_ocp.cpp)
add_library(${PROJECT_NAME}::planner_core ALIAS planner_core)
set_target_properties(planner_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(planner_core PUBLIC cxx_std_17)
target_include_directories(planner_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${urdfdom_INCLUDE_DIRS}
    ${CRANE_PLANNING_GENERATED_INCLUDES})
# urdfdom is PRIVATE and is used for exactly one thing: the position and
# velocity limits the description gives the six actuated joints, which the frozen
# model API does not expose and this planner may not invent (see
# include/crane_planning/joint_limits.hpp).
# ruckig is PUBLIC because `geometric_path.hpp` holds one of its trajectories per
# path segment: `wiki/implementation/libraries.md` §1 lists it for "jerk-limited
# interpolation and the clamped B-spline fit" against "spline bases,
# boundary-condition solves", and the C² primitive of trajectory_planning §4.4 is
# exactly a sequence of boundary-condition solves. No spline basis is written here.
# OMPL is PRIVATE and stays PRIVATE: `ompl_path_planner.hpp` names no OMPL type,
# so the choice of sampling planner is an implementation detail of this library
# and not something every consumer of the planner has to compile against.
# omplConfig.cmake's OMPL_INCLUDE_DIRS carries /usr/include and /usr/include/eigen3
# beside OMPL's own root; handing the compiler its own default system directory
# reorders the search path, so only the root goes on and Eigen comes from
# Eigen3::Eigen exactly as it does everywhere else here.
target_include_directories(planner_core SYSTEM PRIVATE ${OMPL_INCLUDE_DIR})
# acados is PRIVATE, and that is the same rule contract §9 applies to
# `crane_model`: `timing_ocp.hpp` names no acados type, so the generated OCP
# backend is an implementation detail of this library and a consumer of the
# planner compiles against neither acados nor the generated headers.
target_link_libraries(planner_core
  PUBLIC
    crane_model::crane_model
    Eigen3::Eigen
    ruckig::ruckig
  PRIVATE
    acados
    crane_ocp::crane_ocp
    ${OMPL_LIBRARIES}
    urdfdom::urdfdom_model
  PUBLIC
    crane_planning_generated_ocp)

generate_parameter_library(crane_planner_parameters
  src/crane_planner_parameters.yaml)

# The adapter, as a library rather than as part of the executable: the service
# test constructs this node object in process, so what is asserted is the node
# the deployment runs and not a second one written for the test.
#
# `src/a2b_adapter.cpp` is here and not in `planner_core` because it speaks ROS
# messages on both sides -- that is all it does -- while the core is ROS-free.
add_library(planner_node STATIC src/a2b_adapter.cpp src/planner_node.cpp)
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
    ${nav_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${timber_crane_planning_interfaces_TARGETS}
    ${trajectory_msgs_TARGETS}
    ${wood_log_msgs_TARGETS}
  PRIVATE
    crane_planner_parameters)

# Beside the manager and not inside it (ROS 2 Interfaces §2), so it is an
# executable a launch profile composes rather than a plugin a spawner loads.
add_executable(crane_planner_node src/planner_main.cpp)
target_link_libraries(crane_planner_node PRIVATE planner_node)

# The offline tuning entry point for the OCP of trajectory_planning §5.2. It
# links `planner_core` and nothing else -- no ROS, no node, no graph -- so what
# it solves is the deployment's own `solve_timing_ocp` and not a second setup of
# the same solver. `scripts/timing_a2b.py` drives it; see README.md.
add_executable(crane_planning_timing_cli src/timing_cli.cpp)
target_link_libraries(crane_planning_timing_cli PRIVATE planner_core)
# The generated header, for one thing only: the SQP iteration cap the shipped
# artifact allocated its statistics array for. `--max-iterations` is refused
# above it here rather than overrunning it inside acados, and the number is read
# off the artifact rather than restated. `test_timing_ocp` reaches it the same way.
target_include_directories(crane_planning_timing_cli PRIVATE
  "${CRANE_PLANNING_GENERATED_DIR}")

install(
  DIRECTORY include/
  DESTINATION include)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
# Installed without the `.py`, because `ros2 run <pkg> <name>` runs the file as
# it is installed and README.md documents `plan_a2b`.
install(
  PROGRAMS scripts/plan_a2b.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME plan_a2b)
# `scripts/timing_a2b.py` is deliberately **not** installed, for the same reason
# `scripts/export_timing_ocp.py` is not: it reads files that exist only in a
# source checkout -- crane_model's two checked-in machine descriptions, which are
# that package's test fixtures and are not installed either. Copying them here
# would be a second machine to keep current. It is run from the tree; what it
# *drives* is installed, which
# is the part that has to match the deployment.
install(
  TARGETS crane_planning_generated_ocp planner_core planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  TARGETS crane_planner_node crane_planning_timing_cli
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(crane_planning_generated_ocp planner_core planner_node)
ament_export_dependencies(
  crane_model
  crane_msgs
  crane_ocp
  Eigen3
  geometry_msgs
  rclcpp
  ruckig
  sensor_msgs
  std_msgs
  timber_crane_planning_interfaces
  trajectory_msgs
  nav_msgs
  wood_log_msgs)

# The checked-in solver is still what `scripts/export_timing_ocp.py` writes from
# `crane_model/scripts/crane_symbolic.py`. It runs at build time rather than as a
# test because a stale `generated/` tree is not a failing behaviour, it is a
# binary that solves a different problem than the source says -- and with the
# parity tests retired this is the only thing left that ties the two together.
# The check regenerates into a scratch tree and diffs; it costs a few seconds.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(crane_planning_generated_is_current ALL
  COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_timing_ocp.py" --check
  COMMENT "checking generated/ against scripts/export_timing_ocp.py"
  VERBATIM)
add_dependencies(crane_planning_generated_ocp crane_planning_generated_is_current)

ament_package()
