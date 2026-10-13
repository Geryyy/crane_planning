cmake_minimum_required(VERSION 3.16)
project(vpsto_crane LANGUAGES C CXX)
# needs the workspace sourced (ros2_vpsto, ros2_ccmaes built) and gen/ from export_model.py
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(coal REQUIRED)
# ament export targets do not resolve outside colcon, and the installed libros2_vpsto.a is not
# PIC: build Marc's vpsto.cpp (unmodified) in here, take headers/ccmaes from the install prefixes
find_package(ros2_vpsto REQUIRED)
find_package(ros2_ccmaes REQUIRED)
get_filename_component(VPSTO_PREFIX ${ros2_vpsto_DIR}/../../.. ABSOLUTE)
get_filename_component(CCMAES_PREFIX ${ros2_ccmaes_DIR}/../../.. ABSOLUTE)
set(VPSTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../ros2_vpsto/src/vpsto.cpp)
add_library(vpsto_crane SHARED vpsto_crane.cpp ${VPSTO_SRC} gen/vpsto_crane_gen.c)
target_include_directories(vpsto_crane PRIVATE . ${VPSTO_PREFIX}/include ${CCMAES_PREFIX}/include)
target_link_libraries(vpsto_crane PRIVATE ${CCMAES_PREFIX}/lib/libros2_ccmaes.so coal::coal Eigen3::Eigen OpenMP::OpenMP_CXX)
