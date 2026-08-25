// The two properties `wiki/trajectory_planning.md` 4.5 says a geometric path
// owes stage 2, written once and asserted of **both** producers.
//
// 4.5's claim is behavioural rather than numerical: at a point of discontinuous
// curvature the path-velocity limit collapses to `sigma_dot = 0`, so the machine
// stops dead at the waypoint -- the worst possible output for a crane whose
// purpose is smooth, sway-free motion. The structured primitive of 4.4 is built
// C2 and needs no smoothing stage; a sampled path is piecewise-linear C0 and is
// only allowed out after the shortcut and the C2 refit 4.5 makes mandatory. That
// is two very different routes to one requirement, which is exactly why the
// requirement is spelled out in one place and both are put through it: a second
// copy of these assertions would be a second, weaker definition of C2.

#ifndef C2_PATH_ASSERTIONS_HPP_
#define C2_PATH_ASSERTIONS_HPP_

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"

namespace crane_planning_test
{

/// The largest |q_a''| the path reaches, which the C2 tolerances are stated against.
inline double curvature_scale(const crane_planning::GeometricPath & path)
{
  double scale = 1.0;
  for (std::size_t index = 0; index <= 200U; ++index) {
    const double sigma = static_cast<double>(index) / 200.0;
    scale = std::max(scale, path.at(sigma).ddq_a.norm());
  }
  return scale;
}

/// `q_a'` continuous and `q_a''` defined at every sigma, junctions included.
/**
 * The tolerance is stated as two things rather than one number, because one
 * number cannot tell a continuous derivative from a discontinuous one. A gap
 * measured across a junction with a finite difference of step h is the finite
 * difference's own truncation error plus whatever the jump really is: on a C2
 * junction it is the first alone and shrinks with h, on a kink the second
 * dominates and does not. So both are asserted -- the gap at the fine step is
 * under a thousandth of the largest curvature the path itself reaches, and
 * shrinking the step by a hundred shrinks the gap by at least fifty.
 */
inline void expect_c2_everywhere(
  const crane_planning::GeometricPath & path, const std::string & label)
{
  const double scale = curvature_scale(path);
  const double coarse = 1.0e-5;
  const double step = coarse / 100.0;

  for (const double sigma : path.junction_sigmas()) {
    const crane_planning::PathSample before = path.at(sigma - step);
    const crane_planning::PathSample after = path.at(sigma + step);
    const crane_planning::PathSample well_before = path.at(sigma - coarse);
    const crane_planning::PathSample well_after = path.at(sigma + coarse);

    const double position_gap = (after.q_a - before.q_a).norm();
    const double rate_gap = (after.dq_a - before.dq_a).norm();
    const double curvature_gap = (after.ddq_a - before.ddq_a).norm();
    EXPECT_LT(position_gap, 1.0e-3 * scale) << label << " at junction " << sigma;
    EXPECT_LT(rate_gap, 1.0e-3 * scale) << label << " q_a' across junction " << sigma;
    EXPECT_LT(curvature_gap, 1.0e-3 * scale) << label << " q_a'' across junction " << sigma;

    const double coarse_rate_gap = (well_after.dq_a - well_before.dq_a).norm();
    const double coarse_curvature_gap = (well_after.ddq_a - well_before.ddq_a).norm();
    EXPECT_LT(50.0 * rate_gap, coarse_rate_gap + 1.0e-12)
      << label << " q_a' is not closing across junction " << sigma;
    EXPECT_LT(50.0 * curvature_gap, coarse_curvature_gap + 1.0e-12)
      << label << " q_a'' is not closing across junction " << sigma;
  }

  // Defined everywhere, which is the property 4.5 says a kinked path lacks:
  // there is no sigma at which the second derivative fails to exist, and no
  // sigma at which it jumps.
  double worst_jump = 0.0;
  for (std::size_t index = 0; index <= 2000U; ++index) {
    const double sigma = static_cast<double>(index) / 2000.0;
    const crane_planning::PathSample sample = path.at(sigma);
    ASSERT_TRUE(sample.q_a.allFinite()) << label << " at sigma " << sigma;
    ASSERT_TRUE(sample.dq_a.allFinite()) << label << " at sigma " << sigma;
    ASSERT_TRUE(sample.ddq_a.allFinite()) << label << " at sigma " << sigma;
    if (sigma <= coarse || sigma >= 1.0 - coarse) {
      continue;
    }
    worst_jump = std::max(
      worst_jump, (path.at(sigma + step).ddq_a - path.at(sigma - step).ddq_a).norm());
  }
  EXPECT_LT(worst_jump, 1.0e-3 * scale) << label;

  // And the first derivative really is the derivative of the path, so the
  // numbers stage 2 would put into q_a' sigma_dot are the ones the path moves at
  // rather than a second, independently produced set.
  for (std::size_t index = 1; index < 200U; ++index) {
    const double sigma = static_cast<double>(index) / 200.0;
    const double span = 1.0e-6;
    const crane_planning::PathVector difference =
      (path.at(sigma + span).q_a - path.at(sigma - span).q_a) / (2.0 * span);
    EXPECT_LT((difference - path.at(sigma).dq_a).norm(), 1.0e-4 * (1.0 + scale))
      << label << " at sigma " << sigma;
  }
}

/// At rest at both ends, and monotone in sigma inside every segment.
/**
 * The second half is what keeps a path from running past a waypoint and coming
 * back: inside a segment no coordinate leaves the box its two waypoints span.
 */
inline void expect_at_rest_and_monotone(
  const crane_planning::GeometricPath & path, const std::string & label)
{
  const double scale = curvature_scale(path);
  EXPECT_LT(path.at(0.0).dq_a.norm(), 1.0e-9 * (1.0 + scale)) << label;
  EXPECT_LT(path.at(1.0).dq_a.norm(), 1.0e-9 * (1.0 + scale)) << label;
  EXPECT_LT(path.at(0.0).ddq_a.norm(), 1.0e-9 * (1.0 + scale)) << label;
  EXPECT_LT(path.at(1.0).ddq_a.norm(), 1.0e-9 * (1.0 + scale)) << label;
  EXPECT_DOUBLE_EQ(path.at(0.0).dq8, 0.0) << label;
  EXPECT_DOUBLE_EQ(path.at(1.0).dq8, 0.0) << label;

  const std::vector<crane_planning::PathVector> & waypoints = path.waypoints();
  ASSERT_EQ(waypoints.size(), path.segment_count() + 1U) << label;
  std::vector<double> edges{0.0};
  edges.insert(edges.end(), path.junction_sigmas().begin(), path.junction_sigmas().end());
  edges.push_back(1.0);
  for (std::size_t segment = 0; segment + 1U < edges.size(); ++segment) {
    for (std::size_t step = 0; step <= 100U; ++step) {
      const double blend = static_cast<double>(step) / 100.0;
      const double sigma = edges[segment] + blend * (edges[segment + 1U] - edges[segment]);
      const crane_planning::PathVector q_a = path.at(sigma).q_a;
      for (std::size_t row = 0; row < crane_planning::kPathDof; ++row) {
        const Eigen::Index axis = static_cast<Eigen::Index>(row);
        const double from = waypoints[segment][axis];
        const double to = waypoints[segment + 1U][axis];
        EXPECT_GE(q_a[axis], std::min(from, to) - 1.0e-6)
          << label << " segment " << segment << " row " << row;
        EXPECT_LE(q_a[axis], std::max(from, to) + 1.0e-6)
          << label << " segment " << segment << " row " << row;
      }
    }
  }
}

}  // namespace crane_planning_test

#endif  // C2_PATH_ASSERTIONS_HPP_
