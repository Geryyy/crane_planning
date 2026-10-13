// VP-STO cost for the PZS100 crane as in EcB25 (T bound + W-SDF collision + joint limits),
// plus a plain C ABI for ctypes. Const methods only touch per-call locals: OpenMP-safe.
#include <coal/distance.h>
#include <coal/shape/geometric_shapes.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <ros2_vpsto/vpsto.hpp>

#include "gen/vpsto_crane_gen.h"

namespace
{
constexpr int NDOF = 5;  // slew, boom, arm, telescope, rotator
using Geometry = std::shared_ptr<coal::CollisionGeometry>;

struct Body
{
  Geometry g;
  double weight;
};

coal::Transform3s pose_of(const double * p)  // R row-major, t
{
  coal::Matrix3s R;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {R(i, j) = p[3 * i + j];}
  }
  return coal::Transform3s(R, coal::Vec3s(p[9], p[10], p[11]));
}

// 0 box, 1 capsule(r, len), 2 cylinder(2r,.,len), 3 sphere(2r)
Geometry make(int kind, const double * d)
{
  Geometry g;
  if (kind == 0) {
    g = std::make_shared<coal::Box>(d[0], d[1], d[2]);
  } else if (kind == 1) {
    g = std::make_shared<coal::Capsule>(d[0], d[1]);
  } else if (kind == 2) {
    g = std::make_shared<coal::Cylinder>(0.5 * d[0], d[2]);
  } else {
    g = std::make_shared<coal::Sphere>(0.5 * d[0]);
  }
  g->computeLocalAABB();
  return g;
}

// Signed distance, negative inside; +inf when the bounding spheres already separate them.
double signed_distance(
  const Geometry & a, const coal::Transform3s & ta, const Geometry & b,
  const coal::Transform3s & tb)
{
  const double gap = (ta.transform(a->aabb_center) - tb.transform(b->aabb_center)).norm() -
    a->aabb_radius - b->aabb_radius;
  if (gap > 0.) {return gap;}
  coal::DistanceRequest request;
  request.enable_signed_distance = true;
  coal::DistanceResult result;
  return coal::distance(a.get(), ta, b.get(), tb, request, result);
}

class CraneVpstoModel : public vpsto::VpstoModelBase
{
public:
  double dq_max[NDOF], ddq_max[NDOF], lower[NDOF], upper[NDOF], q_pump_max = 1., q_tool = 0.;
  std::vector<Body> links;
  std::vector<Geometry> obstacles;
  std::vector<coal::Transform3s> obstacle_pose;
  Geometry payload;
  coal::Transform3s payload_offset;

  double compute_final_time(
    const Eigen::VectorXd & q, const Eigen::VectorXd & qp,
    const Eigen::VectorXd & qpp) const override
  {
    double T = 0.;
    const int n = q.size() / NDOF;
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < NDOF; ++i) {
        T = std::max(T, std::abs(qp[k * NDOF + i]) / dq_max[i]);
        T = std::max(T, std::sqrt(std::abs(qpp[k * NDOF + i]) / ddq_max[i]));
      }
      // flow is positively homogeneous in dq (A(sign v)|v|), so Q(q, q') is the flow at T = 1
      const double * in[2] = {&q[k * NDOF], &qp[k * NDOF]};
      double flow = 0.;
      double * out[1] = {&flow};
      vc_pump(in, out, nullptr, nullptr, 0);
      T = std::max(T, std::abs(flow) / q_pump_max);
    }
    return T;
  }

  double compute_trajectory_cost(
    const Eigen::VectorXd & q, const Eigen::VectorXd & qp,
    const Eigen::VectorXd & qpp) const override
  {
    double cost = compute_final_time(q, qp, qpp);
    const int n = q.size() / NDOF;
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < NDOF; ++i) {
        const double v = q[k * NDOF + i];
        if (v < lower[i]) {
          cost += 1e2 * (1. + lower[i] - v);
        } else if (v > upper[i]) {
          cost += 1e2 * (1. + v - upper[i]);
        }
      }
      cost += collision(&q[k * NDOF]);
    }
    return cost;
  }

  // W-SDF at the passive equilibrium: weight * (1 - sd) per penetrating link-obstacle pair
  double collision(const double * q) const
  {
    if (obstacles.empty()) {return 0.;}
    std::vector<double> frames(12 * (links.size() + 1));
    const double * in[2] = {q, &q_tool};
    double * out[1] = {frames.data()};
    vc_frames(in, out, nullptr, nullptr, 0);
    double cost = 0.;
    for (size_t l = 0; l < links.size(); ++l) {
      const auto pose = pose_of(&frames[12 * l]);
      for (size_t o = 0; o < obstacles.size(); ++o) {
        const double sd = signed_distance(links[l].g, pose, obstacles[o], obstacle_pose[o]);
        if (sd < 0.) {cost += links[l].weight * (1. - sd);}
      }
    }
    if (payload) {
      const auto pose = pose_of(&frames[12 * links.size()]) * payload_offset;
      for (size_t o = 0; o < obstacles.size(); ++o) {
        const double sd = signed_distance(payload, pose, obstacles[o], obstacle_pose[o]);
        if (sd < 0.) {cost += 1e2 * (1. - sd);}
      }
    }
    return cost;
  }
};

using Handle = std::shared_ptr<CraneVpstoModel>;
Eigen::Map<const Eigen::VectorXd> vec(const double * p, int n) {return {p, n};}
CraneVpstoModel & model(void * h) {return **static_cast<Handle *>(h);}

// Knot vector [q0; via; qd; 0; 0] of the cubic spline, as Vpsto's sampleTrajectory builds it
Eigen::VectorXd knots(const double * q0, const double * qd, const double * via, int nVia)
{
  Eigen::VectorXd w = Eigen::VectorXd::Zero(NDOF * (nVia + 2));
  w.head(NDOF) = vec(q0, NDOF);
  w.segment(NDOF, NDOF * (nVia - 2)) = vec(via, NDOF * (nVia - 2));
  w.segment(NDOF * (nVia - 1), NDOF) = vec(qd, NDOF);
  return w;
}
}  // namespace

extern "C" {

void * vc_create(
  const double * dq_max, const double * ddq_max, const double * lower, const double * upper,
  double q_pump_max, double q_tool)
{
  auto m = std::make_shared<CraneVpstoModel>();
  std::copy(dq_max, dq_max + NDOF, m->dq_max);
  std::copy(ddq_max, ddq_max + NDOF, m->ddq_max);
  std::copy(lower, lower + NDOF, m->lower);
  std::copy(upper, upper + NDOF, m->upper);
  m->q_pump_max = q_pump_max;
  m->q_tool = q_tool;
  return new Handle(m);
}

void vc_destroy(void * h) {delete static_cast<Handle *>(h);}

// order = rows of link_table.json, then nothing else
void vc_add_link(void * h, int kind, const double * dims, double weight)
{
  model(h).links.push_back({make(kind, dims), weight});
}

void vc_add_obstacle(void * h, int kind, const double * dims, const double * pose12)
{
  model(h).obstacles.push_back(make(kind, dims));
  model(h).obstacle_pose.push_back(pose_of(pose12));
}

// payload body at K8, offset = centre in K8 (no rotation)
void vc_set_payload(void * h, int kind, const double * dims, const double * offset3)
{
  auto & m = model(h);
  m.payload = make(kind, dims);
  m.payload_offset = coal::Transform3s(coal::Vec3s(offset3[0], offset3[1], offset3[2]));
}

// out = T, cost, solve ms; via_out (nVia-2)*NDOF; eval_out 3*nEval*NDOF (q, q', q'' at eval)
void vc_solve(
  void * h, const double * q0, const double * qd, int nVia, int nEval, int nSamples, int threads,
  double * via_out, double * out, double * eval_out)
{
  vpsto::Vpsto solver(*static_cast<Handle *>(h), NDOF, nVia, nEval, nSamples, threads);
  Eigen::VectorXd via, q, qp, qpp;
  double T = 0.;
  int ms = 0;
  solver.solve(vec(q0, NDOF), vec(qd, NDOF), via, T, ms);
  solver.getPath(via, q, qp, qpp);
  Eigen::Map<Eigen::VectorXd>(via_out, via.size()) = via;
  out[0] = T;
  out[1] = model(h).compute_trajectory_cost(q, qp, qpp);
  out[2] = ms;
  const int n = NDOF * nEval;
  Eigen::Map<Eigen::VectorXd>(eval_out, n) = q;
  Eigen::Map<Eigen::VectorXd>(eval_out + n, n) = qp;
  Eigen::Map<Eigen::VectorXd>(eval_out + 2 * n, n) = qpp;
}

// (M+1) uniform samples over [0, T], exact T (sampleTrajectory truncates it to int)
void vc_sample(
  const double * q0, const double * qd, const double * via, int nVia, double T, int M,
  double * q, double * dq, double * ddq)
{
  Eigen::MatrixXd Phi, dPhi, ddPhi;
  vpsto::Vpsto::get_basis_matrices(NDOF, nVia, M + 1, Phi, dPhi, ddPhi);
  const auto w = knots(q0, qd, via, nVia);
  Eigen::Map<Eigen::VectorXd>(q, NDOF * (M + 1)) = Phi * w;
  Eigen::Map<Eigen::VectorXd>(dq, NDOF * (M + 1)) = dPhi * w / T;
  Eigen::Map<Eigen::VectorXd>(ddq, NDOF * (M + 1)) = ddPhi * w / (T * T);
}

// cost of a given path on the nEval grid (T = 1 derivatives), to cross-check solve
double vc_cost(
  void * h, const double * q0, const double * qd, const double * via, int nVia, int nEval)
{
  Eigen::MatrixXd Phi, dPhi, ddPhi;
  vpsto::Vpsto::get_basis_matrices(NDOF, nVia, nEval, Phi, dPhi, ddPhi);
  const auto w = knots(q0, qd, via, nVia);
  return model(h).compute_trajectory_cost(Phi * w, dPhi * w, ddPhi * w);
}

double vc_collision(void * h, const double * q5) {return model(h).collision(q5);}

}  // extern "C"
