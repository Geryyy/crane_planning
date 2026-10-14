// VP-STO cost for the PZS100 crane as in EcB25 (T bound + W-SDF collision + joint limits),
// optionally EcB25a's sway terms (13)-(15) on the passive dynamics integrated along the path,
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
  double w_sway[3] = {0., 0., 0.};  // EcB25a w_vI, w_qT, w_vT; all 0 = EcB25
  double qu0[2] = {0., 0.}, qu_end[2] = {0., 0.};  // start pose, equilibrium at the goal
  std::vector<double> p;  // crane_model params, as replay.py

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
    const double T = compute_final_time(q, qp, qpp);
    double cost = T;
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
    if (w_sway[0] + w_sway[1] + w_sway[2] > 0.) {cost += sway(q, qp, qpp, T);}
    return cost;
  }

  // EcB25a (13)-(15): RK4 of the passive pair between eval points, actuated state lerped
  double sway(
    const Eigen::VectorXd & q, const Eigen::VectorXd & qp, const Eigen::VectorXd & qpp,
    double T) const
  {
    const int n = q.size() / NDOF;
    const double h = T / (n - 1);
    auto rhs = [&](int k, double a, const double * s, double * ds) {
        double x[14], u[NDOF], ddq[2];
        for (int i = 0; i < NDOF; ++i) {
          const int j = k * NDOF + i, m = j + NDOF;
          x[i] = q[j] + a * (q[m] - q[j]);
          x[7 + i] = (qp[j] + a * (qp[m] - qp[j])) / T;
          u[i] = (qpp[j] + a * (qpp[m] - qpp[j])) / (T * T);
        }
        x[5] = s[0]; x[6] = s[1]; x[12] = s[2]; x[13] = s[3];
        const double * in[3] = {x, u, p.data()};
        double * out[1] = {ddq};
        vc_ddqu(in, out, nullptr, nullptr, 0);
        ds[0] = s[2]; ds[1] = s[3]; ds[2] = ddq[0]; ds[3] = ddq[1];
      };
    double s[4] = {qu0[0], qu0[1], 0., 0.}, k1[4], k2[4], k3[4], k4[4], t[4], cost = 0.;
    for (int k = 0; k + 1 < n; ++k) {
      rhs(k, 0., s, k1);
      for (int i = 0; i < 4; ++i) {t[i] = s[i] + 0.5 * h * k1[i];}
      rhs(k, 0.5, t, k2);
      for (int i = 0; i < 4; ++i) {t[i] = s[i] + 0.5 * h * k2[i];}
      rhs(k, 0.5, t, k3);
      for (int i = 0; i < 4; ++i) {t[i] = s[i] + h * k3[i];}
      rhs(k, 1., t, k4);
      for (int i = 0; i < 4; ++i) {s[i] += h / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);}
      cost += w_sway[0] * (s[2] * s[2] + s[3] * s[3]) * T / n;
    }
    const double e0 = s[0] - qu_end[0], e1 = s[1] - qu_end[1];
    cost += w_sway[1] * (e0 * e0 + e1 * e1) + w_sway[2] * (s[2] * s[2] + s[3] * s[3]);
    return std::isfinite(cost) ? cost : 1e6;
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

void vc_set_sway(
  void * h, const double * w3, const double * qu0, const double * qu_end, const double * p,
  int np)
{
  auto & m = model(h);
  std::copy(w3, w3 + 3, m.w_sway);
  std::copy(qu0, qu0 + 2, m.qu0);
  std::copy(qu_end, qu_end + 2, m.qu_end);
  m.p.assign(p, p + np);
}

}  // extern "C"
