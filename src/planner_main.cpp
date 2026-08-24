// The `crane_planner_node` executable of `wiki/implementation/ros2_interfaces.md`
// 8: one node per package entry point, named for the node it runs.

#include <memory>

#include "crane_planning/planner_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<crane_planning::PlannerNode>());
  rclcpp::shutdown();
  return 0;
}
