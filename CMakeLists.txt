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
# CasADi is reached through `crane_model::casadi_graph`, which carries it as an
# INTERFACE usage requirement; naming it here is what makes that requirement
# resolvable in the install tree, exactly as `crane_model` itself does.
find_package(casadi REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(crane_model REQUIRED)
find_package(crane_msgs REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(geometry_msgs REQUIRED)
# OMPL, for the sampling fallback of trajectory_planning §4.4.
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
add_library(planner_core STATIC
  src/acados_casadi_bridge.cpp
  src/arm_geometry.cpp
  src/collision.cpp
  src/equilibrium_ik.cpp
  src/geometric_path.cpp
  src/inverse_kinematics.cpp
  src/joint_limits.cpp
  src/plan_grip.cpp
  src/planner_core.cpp
  src/replanning.cpp
  src/sampling_planner.cpp
  src/structured_primitive.cpp
  src/timing_ocp.cpp
  src/tool_axis.cpp
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
# OMPL is PRIVATE and stays PRIVATE: `sampling_planner.hpp` names no OMPL type,
# so the choice of sampling planner is an implementation detail of this library
# and not something every consumer of the planner has to compile against.
# omplConfig.cmake's OMPL_INCLUDE_DIRS carries /usr/include and /usr/include/eigen3
# beside OMPL's own root; handing the compiler its own default system directory
# reorders the search path, so only the root goes on and Eigen comes from
# Eigen3::Eigen exactly as it does everywhere else here.
target_include_directories(planner_core SYSTEM PRIVATE ${OMPL_INCLUDE_DIR})
# acados and CasADi are both PRIVATE, and that is the same rule contract §9
# applies to `crane_model`: `timing_ocp.hpp` names neither an `acados` nor a
# `casadi` type, so the OCP backend is an implementation detail of this library
# and a consumer of the planner compiles against neither. `src/acados_casadi_bridge.hpp`
# is where they meet, and it is not installed.
target_link_libraries(planner_core
  PUBLIC
    crane_model::crane_model
    Eigen3::Eigen
    ruckig::ruckig
  PRIVATE
    acados
    crane_model::casadi_graph
    ${OMPL_LIBRARIES}
    urdfdom::urdfdom_model)

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
  timber_crane_planning_interfaces
  trajectory_msgs
  wood_log_msgs)

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

  # S4 as well: collision, the truck model keyed to a measured pose, the sway
  # envelope of trajectory_planning §4.3 and the self-collision of §4.2's
  # `[!todo]` — every scene built in the test rather than subscribed, so nothing
  # here needs a graph. Offline C++ against both machine descriptions.
  ament_add_gtest(test_collision test/test_collision.cpp TIMEOUT 900)
  target_link_libraries(test_collision planner_core)
  target_compile_definitions(test_collision PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the sampling fallback of trajectory_planning §4.4 and the
  # smoothing §4.5 makes mandatory — that the fallback runs *only* when the
  # primitive is blocked, that the state space is the five path coordinates of
  # §4.1 and never the passive pair, that shortcut → C² fit → re-check all three
  # run in that order, that the seed makes a run repeatable and the budget makes
  # it bounded. Offline C++ against both machine descriptions; every scene is
  # built in the test, and nothing here launches, links a simulator or reaches a
  # ROS graph.
  #
  # It is the slowest binary in this package and deliberately so: one state
  # validity check is one `Model::passive_equilibrium` plus the collision queries
  # of its sway envelope, and the fallback is judged on the same terms as the
  # primitive rather than on cheaper ones. The seeded budget is what bounds it.
  ament_add_gtest(test_sampling_planner test/test_sampling_planner.cpp TIMEOUT 1800)
  target_link_libraries(test_sampling_planner planner_core)
  target_compile_definitions(test_sampling_planner PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the path-constrained OCP of trajectory_planning §5.2 -- the sway
  # carried as a state, the force and pump-flow constraints read out of the same
  # symbolic graph `crane_mpc` will export from, and the κ margin of §5.5.
  # Offline C++ and deterministic: acados is a numerical solver in process, so no
  # graph, no launch and no simulator is reachable from this binary either.
  # This one binary links `crane_model::casadi_graph` on top of `planner_core`,
  # which keeps it PRIVATE: `FlowAgreesWithTheGraph` rebuilds the graph and
  # evaluates its output map itself, and a test that took the planner's word for
  # the flow would assert nothing about the two being one expression.
  ament_add_gtest(test_timing_ocp test/test_timing_ocp.cpp TIMEOUT 1200)
  target_link_libraries(test_timing_ocp planner_core crane_model::casadi_graph)
  target_compile_definitions(test_timing_ocp PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the tool coordinate of `/crane/plan_grip`'s close and open --
  # the retained cosine primitive of trajectory_planning §8 carried on the rate
  # so it is C² at both ends, the end of the description's own range a phase
  # drives to, the tool axis's own velocity, acceleration and pump-flow limits,
  # and the transmission reversal issue 037's notes measured inside the 7040
  # jaw's range. Offline C++ against both machine descriptions, and cheap: a tool
  # phase is a cosine and three scalar bounds, with no OCP in it.
  ament_add_gtest(test_tool_axis test/test_tool_axis.cpp TIMEOUT 600)
  target_link_libraries(test_tool_axis planner_core)
  target_compile_definitions(test_tool_axis PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: the four phases of `crane_msgs/PlanGrip`, per phase and per tool.
  # The two arm phases go through `plan_motion` itself, so this binary solves
  # OCPs and is priced accordingly; the two tool phases are the cosine above.
  # What it is really about is the three things the phases owe each other: the
  # descend endpoint is an equilibrium solution and a descend whose endpoint has
  # none is refused, every phase is on the arm's one time base, and no phase
  # remembers which one ran before it.
  ament_add_gtest(test_plan_grip test/test_plan_grip.cpp TIMEOUT 2400)
  target_link_libraries(test_plan_grip planner_core)
  target_compile_definitions(test_plan_grip PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S4 as well: replanning from a moving, swinging state and the latency bound of
  # trajectory_planning §7 — the measured start as a boundary condition on the
  # geometry and on the OCP, continuity at the seam in position *and* velocity,
  # the unmeasured sway bounded or refused but never assumed zero, and the budget
  # firing in the planner rather than in a caller's timeout. Offline C++ and
  # deterministic: the budget reads an injected clock that steps a fixed amount
  # per reading, so no case here sleeps and none races a real clock.
  #
  # It solves five OCPs of §5.2 at some seconds each and is priced accordingly.
  # One machine only, and the header says why.
  ament_add_gtest(test_replanning test/test_replanning.cpp TIMEOUT 1200)
  target_link_libraries(test_replanning planner_core)
  target_compile_definitions(test_replanning PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S5: the served contract of both rows the planner has in ROS 2 Interfaces §5 —
  # `/crane/plan_motion` and `/crane/plan_grip` — on the isolated domain
  # `verify.sh` pins. Two nodes in one in-process executor: the deployment's own
  # node object and a client, and no launch, no controller manager and no
  # simulator, because the planner claims no interface and joins no cascade.
  #
  # The timeout was 300 s while `/crane/plan_motion` was the only row. A grip's
  # descend and lift are `plan_motion` itself, so serving the four phases adds
  # two more OCP solves of §5.2 at some 14 s each; 900 s is that with room, and
  # it bounds a hang rather than budgeting a latency (§7's latency is issue 045's).
  # The timeout was 900 s while `/crane/plan_motion` and `/crane/plan_grip` were
  # the only rows. `a2b_movement` (ROS 2 Interfaces §9) is served here too and one
  # of its cases plans, which is one more OCP solve of §5.2 at some 14 s; the rest
  # of them refuse, and a refusal costs nothing. 1200 s is that with room, and it
  # bounds a hang rather than budgeting a latency (§7's latency is issue 045's).
  ament_add_gtest(test_plan_motion_service test/test_plan_motion_service.cpp TIMEOUT 1200)
  target_link_libraries(test_plan_motion_service
    planner_node
    ${crane_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${timber_crane_planning_interfaces_TARGETS}
    ${trajectory_msgs_TARGETS})
  target_compile_definitions(test_plan_motion_service PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")

  # S5 as well, and offline: the `a2b_movement` adapter of ROS 2 Interfaces §9,
  # field by field in both directions. No node, no clock, no graph -- the two
  # translations are message in, message out, and the one thing they read off the
  # machine is the settled 3D offset from the tip pivot to the tool, which comes
  # out of the two checked-in descriptions. Cheap: nothing here plans, because
  # what is under test is what the planner is *asked*, and
  # `test_plan_motion_service.cpp` is where the served row is.
  ament_add_gtest(test_a2b_adapter test/test_a2b_adapter.cpp TIMEOUT 300)
  target_link_libraries(test_a2b_adapter
    planner_node
    ${crane_msgs_TARGETS}
    ${timber_crane_planning_interfaces_TARGETS}
    ${wood_log_msgs_TARGETS})
  target_compile_definitions(test_a2b_adapter PRIVATE
    CRANE_PLANNING_MACHINE_DESCRIPTION_DIR="${CRANE_PLANNING_MACHINE_DESCRIPTION_DIR}")
endif()

ament_package()
