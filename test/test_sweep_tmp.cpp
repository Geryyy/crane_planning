#include <gtest/gtest.h>
#include <cstdio>
#include "crane_model/model.hpp"
#include "crane_planning/joint_limits.hpp"
#include "description_fixture.hpp"
namespace {
TEST(Sweep, ArmForce)
{
  for (const auto & machine : crane_planning_test::machines()) {
    auto model = crane_planning_test::build_model(machine);
    auto limits = crane_planning::read_joint_limits(
      crane_planning_test::description(machine), model.urdf_joint_names());
    ASSERT_TRUE(limits.ok());
    for (std::size_t row = 0; row < 6; ++row) {
      std::printf("%s axis %zu: [%g, %g] dq %g\n", machine.name, row,
        limits.value().axis[row].lower, limits.value().axis[row].upper,
        limits.value().axis[row].dq_max);
    }
    crane_model::QA qa = crane_planning_test::centred(limits.value(), machine);
    for (double q3 = -0.4; q3 < 2.6; q3 += 0.4) {
      for (double q2 = -1.0; q2 < 0.6; q2 += 0.4) {
        crane_model::QA test = qa; test[1] = q2; test[2] = q3;
        auto eq = model.passive_equilibrium(test, crane_planning_test::empty_gripper());
        if (!eq.ok()) {std::printf("q2=%5.2f q3=%5.2f no equilibrium\n", q2, q3); continue;}
        crane_model::Q q = crane_model::Q::Zero();
        const int prows[6] = {0,1,2,3,6,7};
        for (int i=0;i<6;i++) q[prows[i]] = test[i];
        q[4]=eq.value()[0]; q[5]=eq.value()[1];
        auto jc = model.cylinder_jacobian(q);
        crane_model::DQ dq = crane_model::DQ::Zero();
        crane_model::DQ ddq = crane_model::DQ::Zero();
        auto tau = model.inverse_dynamics(q, dq, ddq, crane_planning_test::empty_gripper());
        if (!jc.ok() || !tau.ok()) {std::printf("q2=%.2f q3=%.2f FAIL\n", q2, q3); continue;}
        const int rows[6] = {0,1,2,3,6,7};
        std::printf("q2=%5.2f q3=%5.2f F:", q2, q3);
        for (int i = 0; i < 6; ++i) {
          std::printf(" %10.4g", tau.value()[rows[i]] / jc.value()(i, i));
        }
        std::printf("  Jc:");
        for (int i = 0; i < 6; ++i) {std::printf(" %8.3g", jc.value()(i, i));}
        std::printf("\n");
      }
    }
    break;
  }
}
}
