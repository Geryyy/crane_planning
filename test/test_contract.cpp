#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>

#include "crane_model/testing/mock_model.hpp"

namespace crane_planning
{
inline constexpr std::size_t kPlanningJointCount = 6;
using JointVector = std::array<double, kPlanningJointCount>;

struct PlannerRequest
{
  JointVector start{};
  JointVector goal{};
  double speed_scale{1.0};
  bool avoid_collisions{true};
};

struct PlannerResult
{
  JointVector first_point{};
  JointVector final_point{};
  bool success{false};
  std::string message{};
};

class PlannerContract
{
public:
  virtual ~PlannerContract() = default;
  virtual PlannerResult plan(const PlannerRequest & request) const = 0;
};
}  // namespace crane_planning

namespace
{
class PlannerDouble final : public crane_planning::PlannerContract
{
public:
  crane_planning::PlannerResult plan(
    const crane_planning::PlannerRequest & request) const override
  {
    return {request.start, request.goal, true, "fixture"};
  }
};
}  // namespace

TEST(CranePlanningContract, FixturePreservesEndpoints)
{
  crane_planning::PlannerRequest request;
  request.start[0] = 2.0;
  request.goal[5] = -3.0;
  const auto result = PlannerDouble().plan(request);
  EXPECT_TRUE(result.success);
  EXPECT_DOUBLE_EQ(result.first_point[0], 2.0);
  EXPECT_DOUBLE_EQ(result.final_point[5], -3.0);
}

TEST(CranePlanningContract, W05UsesInstalledModelCollisionContract)
{
  const auto model = crane_model::testing::MockModel::create(crane_model::Tool::Pzs100);
  ASSERT_TRUE(model.ok());
  const auto result = model.value().collision_query(
    crane_model::Q::Zero(), crane_model::CollisionScene{});
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value().collision);
}
