// Stage 2 of `wiki/trajectory_planning.md` as 5.2 specifies it: the
// path-constrained optimal control problem that carries the sway, solved with
// acados over `crane_model`'s CasADi graph.
//
// # Why an OCP and not retiming
//
// 5.1 is the argument and it is short. Every actuation limit *is* expressible in
// the classical path-parametrisation form -- joint velocity is a speed limit,
// acceleration is linear in `(sigma_dot^2, sigma_ddot)`, pump flow reduces to a
// scalar speed limit and cylinder force is affine. What is not expressible is the
// **sway**: `q_u` is a dynamic state with memory, not a function of
// `(sigma, sigma_dot, sigma_ddot)`, so "arrive with zero sway" is a boundary
// condition on a hidden state and no classical method can impose it. An
// off-the-shelf time-optimal parametrisation would also silently drop the
// pump-flow constraint that even the legacy retiming enforces.
//
// # The problem
//
//     x_sigma = (sigma, sigma_dot, q_u, dq_u) in R^6,   u_sigma = sigma_ddot
//
// with the passive rows of 5.2,
// `ddq_u = -M_uu^-1 (M_ua ddq_a(sigma, sigma_dot, u) + h_u)`, taken from
// `crane_model::symbolic::CasadiGraph::passive_rows` -- the same graph
// `crane_mpc` exports its own model from, so there is one dynamics
// implementation in the system and not two.
//
// **The independent variable is sigma, not time.** 5.2 writes the traversal-time
// objective as `int_0^1 dsigma / sigma_dot`, which is already an integral over
// the path parameter, and discretising along sigma is what makes
// `q_a'(sigma)` and `q_a''(sigma)` *known numbers* at every shooting node: the
// grid is fixed, so the path derivatives are stage parameters rather than
// something that would have to be re-expressed symbolically in a state. The
// dynamics is 5.2's, divided through by `sigma_dot`; `sigma` is carried as a
// state anyway so that 5.4's `sigma(T) = 1` is an imposed terminal condition
// rather than an artefact of the grid.
//
// # What is shared with the MPC, and why that is the point
//
// The force and flow expressions are **not restated here**. They are read out of
// the graph's output map `z`, which is where `wiki/mpc.md` 3 constraints 6 and 7
// live, including the smoothing 3.1 requires -- `A^{+-}(v) sqrt(v^2 + eps^2)`
// with a `tanh` of width `eps_v`, already compiled into the graph. 5.3's
// **"a reference the MPC would reject is a planner bug"** then holds by
// construction rather than by agreement between two implementations.
//
// # kappa is not speed_scale
//
// `kappa` (5.5) is the fraction of the machine's authority the planner is allowed
// to spend, and the rest is what the MPC has left to correct with. `speed_scale`
// (`crane_msgs/PlanMotion`) is a caller asking to go slower. They multiply and
// they are not interchangeable: `speed_scale` touches the joint velocity bound
// only, `kappa` scales every physical limit, and no value of `speed_scale`
// raises `kappa`.

#ifndef CRANE_PLANNING__TIMING_OCP_HPP_
#define CRANE_PLANNING__TIMING_OCP_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "crane_model/model.hpp"
#include "crane_planning/geometric_path.hpp"
#include "crane_planning/joint_limits.hpp"
#include "crane_planning/trajectory_timing.hpp"

namespace crane_planning
{

/// The machine's actuation limits, unscaled. `kappa` is applied to these.
/**
 * Position and velocity limits are *not* here: they come off the description
 * through `JointLimits` and may not be written down beside a node. What is here
 * is what the description does not carry -- `wiki/implementation/parameters.md`
 * 3 and 4, and the cylinder force limit that no existing planner enforces
 * (`wiki/trajectory_planning.md` 3).
 */
struct ActuationLimits
{
  /// `ddq^max` of parameters.md 3, over the six actuated coordinates.
  std::array<double, crane_model::kActuatedDof> ddq_a_max{
    {0.5, 0.7, 0.5, 1.0, 6.0, 2.0}};

  /// `F_i^max`, newtons, the right-hand side of mpc.md 3 constraint 6.
  /**
   * The constraint the graph's output map states is `|F_cyl,i| <= F_i^max`, with
   * `F_cyl = tau_a,i / J_c,ii` -- so the configuration dependence of
   * `|tau_i| <= J_c,ii F_i^max` is carried by the transmission and the number
   * here is a plain cylinder force.
   */
  std::array<double, crane_model::kActuatedDof> cylinder_force_max{};

  /// `Q_P^max` of parameters.md 4, before the planning factor below.
  double pump_flow_max{1.4e-3};

  /// The `0.95x` parameters.md 4 prescribes for planning. Not a second kappa.
  double pump_flow_planning_factor{0.95};
};

/// How precisely the measured initial passive state is known.
/**
 * The window the OCP holds `q_u(0)` and `dq_u(0)` inside, and it is a
 * **measurement** number rather than a solver tuning one: the sway angle comes
 * off a complementary filter and the rate off a differenced gyro pair, so
 * pinning either to machine epsilon claims a precision the sensor does not have.
 *
 * It also happens to be what makes the problem solvable. Pinning all four passive
 * rows exactly at stage 0 turns converging solves into `ACADOS_QP_FAILURE` on the
 * first QP -- HPIPM is an interior-point method and a bound with zero slack is
 * where its barrier breaks down, which this file already documents for the warm
 * start. Declaring them equalities through acados' `idxbxe` does not help, so the
 * trouble is the step the QP has left and not how the bound is written. A window
 * the width of the estimate's own noise gives that step back and gives up nothing
 * the estimate actually said.
 */
struct StartResolution
{
  /// Sway angle, rad. **A design value** -- no recording measures the filter's
  /// angle resolution, and `PendulumState::position_covariance` is the per-request
  /// number a later issue could read instead of this constant.
  double q_u{1.0e-3};

  /// Sway rate, rad/s. **Measured**: the gyro quantiser is `2^-9` rad/s, from the
  /// 7040 recordings, and a differenced pair cannot resolve below it.
  double dq_u{1.953125e-3};
};

/// Everything about the solve that a deployment chooses.
struct TimingOcpSettings
{
  /// kappa of `wiki/trajectory_planning.md` 5.5. Must lie in (0, 1].
  /**
   * The default is below one and that is the whole point of the parameter:
   * retiming exactly at the limits saturates the actuators by construction and
   * leaves the MPC's corrective acceleration nowhere to go.
   */
  double kappa{0.8};

  /// Shooting intervals over sigma in [0, 1].
  std::size_t intervals{40};

  /// `Q_dot_u` of 5.2's sway-energy regulariser, per passive coordinate.
  double sway_weight{2.0};

  /// Regularisation on `sigma_ddot`. Not a limit; the limit is `sigma_accel_max`.
  double input_weight{1.0e-3};

  /// Box on the path rate. Both ends are numerical, not machine numbers.
  /**
   * `sigma_dot` has no physical meaning of its own -- what the machine feels is
   * `q_a'(sigma) sigma_dot`. The floor keeps `1/sigma_dot` finite in the
   * objective; the ceiling bounds `sigma_dot` where `q_a'` and `q_a''` both
   * vanish, which is exactly at the two endpoints, where the path is stationary
   * and no joint limit constrains the rate at all.
   */
  double sigma_rate_min{0.02};
  double sigma_rate_max{2.0};

  /// Box on `sigma_ddot`, the input.
  double sigma_accel_max{20.0};

  /// mpc.md 3 constraint 3: the sway box, around `q_u^eq` rather than around zero.
  /**
   * The same number `wiki/trajectory_planning.md` 4.3 clears the path for, so
   * "any sway the MPC permits is sway the path was cleared for" stays true of
   * the timing as well as of the geometry.
   */
  std::array<double, crane_model::kPassiveDof> q_u_max{{0.2, 0.2}};

  /// mpc.md 3 constraint 4: `|dq_u| <= dq_u^+`, parameters.md 3.
  std::array<double, crane_model::kPassiveDof> dq_u_max{{1.0, 0.5}};

  /// The physical limits kappa is applied to.
  ActuationLimits actuation{};

  /// How precisely `q_u(0)` and `dq_u(0)` are held to the measurement.
  StartResolution start_resolution{};

  /// Sample period of the emitted reference, seconds.
  double sample_period{0.04};

  /// The bound of 7, enforced inside the solver rather than in a caller's timeout.
  /**
   * Sized against what the solve actually costs rather than round: the
   * structured lift/traverse/descend primitive converges in some 44 SQP
   * iterations and 4.1 s here, and a bound is only a bound if the ordinary case
   * clears it on a slower machine too. This is the piece of 7's latency that
   * lives in the OCP; the end-to-end budget is `LatencyBudgetSettings::total_s`
   * in `replanning.hpp`, and it is the larger number -- one plan is 14 s of which
   * the solve is under a third. `plan_motion` lowers this to whatever the total
   * has left when the solve opens, so the two bounds cannot disagree about which
   * of them fired.
   */
  double max_wall_clock{12.0};

  /// SQP iteration cap. Hitting it is a refusal, never a clipped trajectory.
  int max_iterations{250};

  /// When the SQP step is small enough to stop, in the two units it has.
  /**
   * Separate on purpose. The feasibility residuals are in the units of the
   * constraints -- radians per second squared, newtons, cubic metres per second
   * -- and `1e-6` there is far below anything the machine can be commanded at.
   * The stationarity residual is in the units of the *objective's gradient*,
   * which carries `1/sigma_dot` and its `sqrt`, so it is a couple of decades
   * larger for the same quality of answer; asking it for `1e-6` spends the whole
   * iteration budget polishing a trajectory that stopped changing long before.
   */
  double tolerance_stationarity{1.0e-4};
  double tolerance_feasibility{1.0e-6};

  /// Levenberg-Marquardt term on the Gauss-Newton Hessian.
  /**
   * Not a taste: the traversal-time term is `1/sigma_dot`, whose curvature falls
   * as the rate rises, so the Hessian is nearly flat exactly where the optimum
   * is and an unregularised step leaves the region the model is valid on.
   */
  double levenberg_marquardt{1.0e-2};
};

/// The peak of each constrained quantity, as a fraction of the **physical** limit.
/**
 * Reported against the unscaled limit rather than against `kappa x limit`, so
 * that "the peak demand sits at kappa and not at one" is something a caller can
 * read rather than something it has to trust.
 */
struct PeakDemand
{
  double joint_velocity{};
  double joint_acceleration{};
  double cylinder_force{};
  double pump_flow{};

  [[nodiscard]] double worst() const noexcept;
};

/// One shooting node of the solved OCP, in physical units.
/**
 * The trajectory is resampled onto a uniform time grid and carries the six
 * actuated rows only, which is what a reference is. This is the solve itself:
 * the sway states 5.2 carries and the two constrained quantities read off the
 * graph's output map, at the nodes the constraints were actually imposed at.
 *
 * It exists so that the two things this issue's acceptance turns on are
 * *checkable from outside* rather than only believed. 5.4's terminal condition
 * is a statement about `q_u` and `dq_u`, which a six-row reference cannot show;
 * and "the planner's flow is the graph's flow, not a second copy of it" is a
 * claim a test can only settle by re-evaluating the graph at the same points and
 * comparing. Both are what `test_timing_ocp.cpp` does with this.
 */
struct OcpNode
{
  double sigma{};
  double sigma_rate{};                ///< `sigma_dot`, per second
  crane_model::QA q_a{};
  crane_model::DQA dq_a{};            ///< `q_a'(sigma) sigma_dot`, so a real velocity
  crane_model::QA ddq_a{};            ///< 5.2's `q_a'' sigma_dot^2 + q_a' sigma_ddot`
  crane_model::QU q_u{};
  crane_model::DQU dq_u{};
  crane_model::QU q_u_equilibrium{};  ///< where the tool hangs at this `q_a`
  crane_model::QA cylinder_force{};   ///< N, mpc.md 3 constraint 6's left-hand side
  double pump_flow{};                 ///< m^3/s, constraint 7's, summed over the axes
};

/// One solved timing.
struct TimingSolution
{
  TimedTrajectory trajectory{};

  /// The solve on its own sigma grid. See `OcpNode`.
  std::vector<OcpNode> nodes;

  PeakDemand peak_demand{};
  double kappa{};             ///< the margin actually applied
  double speed_scale{};       ///< the caller's request, kept separate from kappa
  int iterations{};
  double solve_time{};        ///< seconds of wall clock inside acados
  std::string solver_status;  ///< acados' own status word, success or not
};

/// 7's initial condition: where the machine is, and what it is already doing.
/**
 * Absent -- `measured == false` -- is the **stopped start** of 7's `[!warning]`:
 * the passive pair pinned at the equilibrium of the start configuration and not
 * moving, and the path rate left free because the path meets its own start with
 * `q_a' = 0`. That is the convention 7 calls a defect, and it is kept as the
 * answer for a machine that really is standing still.
 *
 * Present, it is the measurement: `q_u` and `dq_u` off `/crane/pendulum_state`
 * and a `sigma_rate` that reproduces the measured actuated velocity through the
 * path's own start slope (`start_path_rate` in `geometric_path.hpp`). The three
 * together are what make a plan issued mid-motion start where the machine is
 * rather than where it would be at rest -- and 5.4's terminal conditions are
 * untouched, because the end of a placement move is still a tool hanging still.
 */
struct TimingOcpStart
{
  bool measured{false};
  crane_model::QU q_u{crane_model::QU::Zero()};
  crane_model::DQU dq_u{crane_model::DQU::Zero()};

  /// Whether `sigma_rate` pins `sigma_dot(0)` rather than leaving it to the box.
  /**
   * Separate from `measured` because the two are separately absent: a machine
   * whose tool is swinging while its arm stands still has a measured passive pair
   * and no path rate to pin, and the sway still has to be carried.
   */
  bool sigma_rate_pinned{false};
  double sigma_rate{};
};

/// What one solve is asked about, beyond the path itself.
struct TimingOcpRequest
{
  crane_model::Payload payload{};

  /// `crane_msgs/PlanMotion.speed_scale`, in (0, 1]. Scales the velocity bound only.
  double speed_scale{1.0};

  /// 7's measured initial state. Default-constructed is the stopped start.
  TimingOcpStart start{};
};

/// `F_i^max` from the model's own chamber areas and the relief pressure.
/**
 * `wiki/trajectory_planning.md` 3 records that **no existing planner enforces
 * the force limit**, so there is no earlier number to inherit -- and
 * `wiki/implementation/parameters.md` carries the chamber areas but no relief
 * setting at all; 7 lists the pressure constants among its gaps. What is
 * available is `hydraulics.md` 4's `F_i = A_A p_A - A_B p_B`, which
 * `Model::cylinder_force` already implements, so the areas are asked of the
 * model rather than copied into a parameter file where they would drift.
 *
 * The limit returned is the **smaller** of the two chamber areas times
 * `system_pressure_pa`, because mpc.md 3 constraint 6 is written symmetrically
 * in `|tau_i|` and the retracting side of a differential cylinder is the weaker
 * one. `system_pressure_pa` is the one number this cannot derive and it is not
 * measured -- see `config/hydraulic_limits.yaml`.
 */
[[nodiscard]] crane_model::Result<std::array<double, crane_model::kActuatedDof>>
derive_cylinder_force_limits(const crane_model::Model & model, double system_pressure_pa);

/// Solve 5.2's OCP along `path`, or refuse and say what the solver said.
/**
 * `model_config` is the one `model` was built from: the CasADi graph is built by
 * `crane_model::symbolic::casadi_graph`, which takes a `ModelConfig` and not a
 * `Model` because the model keeps its parse behind a private pimpl and its
 * header is frozen.
 *
 * A solve that does not converge is a failure carrying acados' own status word,
 * never a partial or clipped trajectory -- `wiki/mpc.md` 5.3 requirement 1 binds
 * the planner exactly as it binds the controller.
 */
[[nodiscard]] crane_model::Result<TimingSolution> solve_timing_ocp(
  const crane_model::Model & model, const crane_model::ModelConfig & model_config,
  const GeometricPath & path, const JointLimits & limits, const TimingOcpRequest & request,
  const TimingOcpSettings & settings);

}  // namespace crane_planning

#endif  // CRANE_PLANNING__TIMING_OCP_HPP_
