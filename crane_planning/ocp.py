"""
The trajectory OCP: its definition, and the solver that answers with it.

`x = [q_a, q_u, dq_a, dq_u, theta]`, `u = ddq_a`, on normalised time over [0, 1]
with `theta = T / ocp_horizon` a state whose derivative is zero -- so one solve is
time-optimal and there is no outer search over the duration.

The definition lives here rather than beside the exporter because this package's
consumer is Python: `scripts/export_timing_ocp.py` is a command-line front end on
`build_ocp`, and `TrajectoryOcp` is the same problem with a solver attached.

Every residual row is divided by the limit it is measured against, so a row reads
1.0 at its bound and `W` is preference alone -- see `weights`. Gauss-Newton builds
its Hessian from `J' W J`, so one row left in physical units sets the conditioning
of everything: `tau_a` in newton metres stalled this solve at a stationarity
residual of 1e6, and the horizon in seconds cost 40-60x on the same measure.

**There are no obstacle rows.** The geometric stage still certifies a path and
this solver is handed its endpoints, but it is free to move between them however
the dynamics prefer -- which is where its speed comes from and what voids the
clearance proof. The corridor is `q_a`'s box, narrowed, and is not wired yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crane_model import symbolic as cs

from . import weights as w
from .config import PlanningError, passive_equilibrium

#: Where the generated solver is written and cached between runs.
GENERATED = Path(__file__).resolve().parent.parent / "generated"

TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"
SOLVER_NAME = f"crane_planning_ocp_{TOOL}"
GENERATED_HEADER = "crane_planning_ocp_generated.h"

#: The two hinges the hanging pose is a closed form in. `crane_model.symbolic`
#: numbers the planned axes but does not name them; `planner.py` carries the same.
BOOM_AXIS = 1
ARM_AXIS = 2

#: `x` gains the horizon as its last component, with `dT/ds = 0`.
X_HORIZON = cs.NX
NX = cs.NX + 1

#: Row order of `h`, which the caller reads back and must not re-derive.
H_SWAY, H_FLOW, NH = 0, 2, 3
#: Row order of `h_e`: the settled box the caller accepts.
HE_SWAY, HE_SWAY_RATE, NH_E = 0, 2, 4


def equilibrium(q_a):
    """Return where the tool hangs: a two-hinge pendulum hangs straight down."""
    return ca.vertcat(0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi)


def description_limits(model) -> tuple:
    """Return `(dq_max, tau_max)` per planned joint, off the parsed description."""
    inner = model.description.model
    rows = [model.description.joints[row].velocity_index for row in cs.K_PLANNED_ROWS]
    return (
        np.array([float(inner.velocityLimit[row]) for row in rows]),
        np.array([float(inner.effortLimit[row]) for row in rows]),
    )


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict):
    """Assemble the OCP. Returns `(ocp, scale, model)` as the sibling exporter does."""
    model = cs.CraneSymbolicModel(description_xml, TOOL)
    dq_max, tau_max = description_limits(model)
    q_sway_max = np.asarray(parameters["q_sway_max"], dtype=float)
    dq_sway_max = np.asarray(parameters["dq_sway_max"], dtype=float)
    ddq_a_max = np.asarray(parameters["ddq_a_max"], dtype=float)
    nominal = float(parameters["ocp_horizon"])

    # ---------------------------------------------------------------- the model

    x, u, p = model.x, model.u, model.p
    q_a = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
    q_u = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    dq_a = x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF]
    dq_u = x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF]
    sway = q_u - equilibrium(q_a)

    # `theta = T / T_nominal`: one decision variable shared by every node. acados
    # fixes `p` for the duration of a solve, so a horizon the solver may choose
    # cannot live there.
    theta = ca.SX.sym("theta")

    acados_model = AcadosModel()
    acados_model.name = SOLVER_NAME
    acados_model.x = ca.vertcat(x, theta)
    acados_model.u = u
    acados_model.p = p
    acados_model.f_expl_expr = ca.vertcat(nominal * theta * model.xdot, 0.0)

    # ----------------------------------------------------------------- the cost

    # `tau_a` minus its value at rest on the same configuration. 91% of the raw
    # effort integral is gravity plus the held load, which no timing choice can
    # change, so a residual against zero prices holding the block and is blind to
    # the part that shakes the boom. Substituting the hanging pendulum, zero rates
    # and zero input into the same graph makes this the dynamic part exactly.
    tau_a = model.tau_a[: cs.K_PLANNED_DOF]
    at_rest = ca.vertcat(
        q_a,
        equilibrium(q_a),
        ca.SX.zeros(cs.K_PLANNED_DOF),
        ca.SX.zeros(cs.K_PASSIVE_DOF),
    )
    tau_static = ca.substitute(
        tau_a, ca.vertcat(x, u), ca.vertcat(at_rest, ca.SX.zeros(cs.NU))
    )

    # Each row over its own limit, so 1.0 means "at the bound" on every row alike.
    # No `q_a` row: the goal is a terminal box, and tracking it as well only adds
    # a stiff direction to a Hessian that has little else in it.
    residual = ca.vertcat(
        dq_a / dq_max,
        sway / q_sway_max,
        dq_u / dq_sway_max,
        (tau_a - tau_static) / tau_max,
        u / ddq_a_max,
    )
    terminal_residual = ca.vertcat(
        dq_a / dq_max, sway / q_sway_max, dq_u / dq_sway_max, theta
    )
    acados_model.cost_y_expr_0 = residual
    acados_model.cost_y_expr = residual
    acados_model.cost_y_expr_e = terminal_residual

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.parameter_values = np.zeros(cs.NP)

    # The yaml's weights, baked as this solver's defaults. They stay
    # runtime-settable -- the node writes its own onto the built solver -- so a
    # weight change never needs a re-export.
    stage_w, terminal_w = w.matrices(parameters["weights"])
    if (residual.shape[0], terminal_residual.shape[0]) != (w.NY, w.NY_E):
        raise ValueError(
            f"the residual is {residual.shape[0]}/{terminal_residual.shape[0]} rows "
            f"but `weights` describes {w.NY}/{w.NY_E}"
        )
    for stage, matrix in (("_0", stage_w), ("", stage_w), ("_e", terminal_w)):
        setattr(ocp.cost, f"cost_type{stage}", "NONLINEAR_LS")
        setattr(ocp.cost, f"W{stage}", matrix)
        setattr(ocp.cost, f"yref{stage}", np.zeros(matrix.shape[0]))

    # ---------------------------------------------------------- the constraints

    # Two nonlinear rows and no third: there is deliberately no cylinder-force
    # constraint in this package, because it was the smaller chamber area times a
    # relief pressure nothing in this workspace has measured.
    flow = ca.sum1(
        model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
    )
    pump_max = float(hydraulics["pump_flow_max"])
    scale = np.concatenate([q_sway_max, [pump_max]])
    scale_e = np.concatenate([q_sway_max, dq_sway_max])
    acados_model.con_h_expr_0 = ca.vertcat(sway / q_sway_max, flow / pump_max)
    acados_model.con_h_expr = acados_model.con_h_expr_0
    acados_model.con_h_expr_e = ca.vertcat(sway, dq_u) / scale_e

    # Every bound below is a placeholder the caller overwrites; what is baked is
    # which rows exist and which are soft. Stage 0's box is the measured state --
    # over `cs.NX`, not `NX`, because the horizon is the one component of `x` the
    # solver may pick. `constraints.x0` would pin it and freeze the objective.
    ocp.constraints.idxbx_0 = np.arange(cs.NX)
    ocp.constraints.lbx_0 = np.zeros(cs.NX)
    ocp.constraints.ubx_0 = np.zeros(cs.NX)
    for stage in ("", "_e"):
        setattr(ocp.constraints, f"idxbx{stage}", np.arange(NX))
        setattr(ocp.constraints, f"lbx{stage}", -np.ones(NX))
        setattr(ocp.constraints, f"ubx{stage}", np.ones(NX))
    ocp.constraints.idxbu = np.arange(cs.NU)
    ocp.constraints.lbu = -np.ones(cs.NU)
    ocp.constraints.ubu = np.ones(cs.NU)
    for stage in ("_0", ""):
        setattr(ocp.constraints, f"lh{stage}", np.concatenate([-np.ones(2), [0.0]]))
        setattr(ocp.constraints, f"uh{stage}", np.ones(NH))
    ocp.constraints.lh_e = -np.ones(NH_E)
    ocp.constraints.uh_e = np.ones(NH_E)

    # `L1` and never quadratic: a quadratic price is cheap near the boundary and
    # leaks a little violation everywhere, a linear one with a large enough
    # coefficient is exact. Soft rows are the ones a caller would rather have
    # answered late than refused -- sway, the pump, the settled box. The input,
    # the joint range and the goal stay hard: violating those is not a slower
    # plan, it is a wrong one.
    ocp.constraints.idxsh_0 = np.arange(NH)
    ocp.constraints.idxsh = np.arange(NH)
    ocp.constraints.idxsh_e = np.arange(NH_E)
    for stage, rows in (("_0", NH), ("", NH), ("_e", NH_E)):
        setattr(ocp.cost, f"Zl{stage}", np.zeros(rows))
        setattr(ocp.cost, f"Zu{stage}", np.zeros(rows))
        setattr(ocp.cost, f"zl{stage}", np.ones(rows))
        setattr(ocp.cost, f"zu{stage}", np.ones(rows))

    # -------------------------------------------------------------- the backend

    # Normalised time: the grid is [0, 1] and the horizon is `theta * T_nominal`.
    ocp.solver_options.N_horizon = int(parameters["ocp_intervals"])
    ocp.solver_options.tf = 1.0
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = int(parameters["ocp_max_iterations"])
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.levenberg_marquardt = float(parameters["levenberg_marquardt"])
    # Set, not left at acados' 1e-6 default, which is a control-loop number: a
    # plan is resampled onto a 25 Hz reference and tracked by a controller that
    # closes the loop on it, so the last two decades buy it nothing and cost it
    # plenty -- a solve landing at 1.69e-6 spent sixty iterations and eleven
    # seconds failing to halve it, where one landing at 5.39e-7 finished in six.
    tolerance = float(parameters["ocp_tolerance"])
    for condition in ("stat", "eq", "ineq", "comp"):
        setattr(ocp.solver_options, f"nlp_solver_tol_{condition}", tolerance)

    return ocp, np.concatenate([scale, scale_e]), model


@dataclass
class Trajectory:
    """What the solve decided, on its own uniform grid over `[0, T]`."""

    time: np.ndarray  # (N+1,)
    q_a: np.ndarray  # (N+1, 5)
    dq_a: np.ndarray
    ddq_a: np.ndarray  # the input, and what the acceleration box bounds
    q_u: np.ndarray  # (N+1, 2)
    dq_u: np.ndarray
    q_u_eq: np.ndarray
    pump_flow: np.ndarray  # (N+1,) as a fraction of the physical pump
    slack: float
    iterations: int
    solve_time_s: float

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def terminal_sway(self) -> float:
        """How far the tool still hangs off its rest pose at the goal, in radians."""
        return float(np.max(np.abs(self.q_u[-1] - self.q_u_eq[-1])))

    @property
    def terminal_sway_rate(self) -> float:
        """Largest passive rate remaining at the goal, in radians per second."""
        return float(np.max(np.abs(self.dq_u[-1])))


class TrajectoryOcp:
    """
    The generated solver, with the bounds one request needs written onto it.

    Built once. acados regenerates and recompiles into `generated/`, which costs
    seconds on a warm tree and minutes on a cold one, so this is construction-time
    work and not per-request work.
    """

    def __init__(self, description_xml: str, limits, config, weights: dict):
        self.limits, self.config = limits, config
        ocp, self.scale, model = build_ocp(
            description_xml,
            {
                "ocp_intervals": config.ocp_intervals,
                "ocp_horizon": config.ocp_horizon,
                "ocp_max_iterations": config.ocp_max_iterations,
                "ocp_tolerance": config.ocp_tolerance,
                "levenberg_marquardt": config.levenberg_marquardt,
                "q_sway_max": config.q_sway_max,
                "dq_sway_max": config.dq_sway_max,
                "ddq_a_max": config.ddq_a_max,
                "weights": weights,
            },
            {"pump_flow_max": config.pump_flow_max},
        )
        # The pump row in physical units, off the same graph the solver constrains,
        # so what is reported and what is bounded cannot drift apart.
        flow = ca.sum1(
            model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
        )
        self._flow = ca.Function("flow", [model.x, model.u, model.p], [flow])
        self.N = ocp.solver_options.N_horizon
        self.nominal = float(config.ocp_horizon)
        ocp.code_export_directory = str(GENERATED / SOLVER_NAME)
        self.solver = AcadosOcpSolver(
            ocp, json_file=str(GENERATED / f"{SOLVER_NAME}.json"), verbose=False
        )

    def solve(
        self,
        q_a_start: np.ndarray,
        q_a_goal: np.ndarray,
        q_u_start: np.ndarray,
        dq_a_start: np.ndarray,
        dq_u_start: np.ndarray,
        payload: np.ndarray,
        q_tool: float,
        speed_scale: float,
    ) -> Trajectory:
        """Solve one request, or refuse with what the solver said."""
        cfg, lim, N = self.config, self.limits, self.N
        solver = self.solver
        big = 1.0e6
        dq_max = cfg.kappa * speed_scale * lim.dq_max
        ddq_max = cfg.kappa * speed_scale**2 * cfg.ddq_a_max
        flow_hi = cfg.kappa * speed_scale * lim.flow_max / cfg.pump_flow_max
        theta = np.array([cfg.ocp_duration_min, cfg.ocp_duration_max]) / self.nominal

        lower = np.where(lim.bounded, lim.lower, -big)
        upper = np.where(lim.bounded, lim.upper, big)
        lbx = np.concatenate(
            [lower, [-big, -big], -dq_max, -cfg.dq_sway_max, theta[:1]]
        )
        ubx = np.concatenate([upper, [big, big], dq_max, cfg.dq_sway_max, theta[1:]])
        x0 = np.concatenate([q_a_start, q_u_start, dq_a_start, dq_u_start])
        lh = np.concatenate([-np.ones(2), [0.0]])
        uh = np.concatenate([np.ones(2), [flow_hi]])
        settled = np.concatenate(
            [
                cfg.terminal_q_sway_max / cfg.q_sway_max,
                cfg.terminal_dq_sway_max / cfg.dq_sway_max,
            ]
        )
        parameters = np.concatenate([[q_tool], payload])
        price = float(cfg.ocp_slack_price)

        # A straight line in joint space with the pendulum hanging: no rollout and
        # nothing that knows the answer, which is all this problem has needed.
        solver.reset()
        for node in range(N + 1):
            ramp = node / N
            guess = q_a_start + ramp * (q_a_goal - q_a_start)
            solver.set(node, "p", parameters)
            solver.set(
                node,
                "x",
                np.concatenate([guess, passive_equilibrium(guess), np.zeros(7), [1.0]]),
            )
            if node < N:
                solver.set(node, "u", np.zeros(cs.NU))
                solver.constraints_set(node, "lbu", -ddq_max)
                solver.constraints_set(node, "ubu", ddq_max)
                solver.constraints_set(node, "lh", lh)
                solver.constraints_set(node, "uh", uh)
                solver.cost_set(node, "zl", np.full(NH, price))
                solver.cost_set(node, "zu", np.full(NH, price))
            if 0 < node < N:
                solver.constraints_set(node, "lbx", lbx)
                solver.constraints_set(node, "ubx", ubx)
        solver.constraints_set(0, "lbx", x0)
        solver.constraints_set(0, "ubx", x0)
        # The goal is a hard terminal box, and arriving stopped with it. The tool
        # may hang where it likes; how still it has to be is `h_e`, softly.
        solver.constraints_set(
            N, "lbx", np.concatenate([q_a_goal, [-big, -big], np.zeros(7), theta[:1]])
        )
        solver.constraints_set(
            N, "ubx", np.concatenate([q_a_goal, [big, big], np.zeros(7), theta[1:]])
        )
        solver.constraints_set(N, "lh", -settled)
        solver.constraints_set(N, "uh", settled)
        solver.cost_set(N, "zl", np.full(NH_E, price))
        solver.cost_set(N, "zu", np.full(NH_E, price))

        started = time.monotonic()
        status = solver.solve()
        elapsed = time.monotonic() - started

        state = np.array([solver.get(node, "x") for node in range(N + 1)])
        duration = float(state[0, X_HORIZON]) * self.nominal
        q_a = state[:, cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
        q_u = state[:, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
        slack = max(
            float(np.max(solver.get(node, "sl"), initial=0.0)) for node in range(N + 1)
        )
        control = np.array([solver.get(node, "u") for node in range(N)])
        flow = np.array(
            [
                float(
                    self._flow(
                        state[node, : cs.NX], control[min(node, N - 1)], parameters
                    )
                )
                / cfg.pump_flow_max
                for node in range(N + 1)
            ]
        )
        if status != 0:
            raise PlanningError(
                f"the trajectory OCP did not converge: acados status {status} after "
                f"{solver.get_stats('sqp_iter')} iterations, {elapsed:.2f} s"
            )
        return Trajectory(
            time=np.linspace(0.0, duration, N + 1),
            q_a=q_a,
            dq_a=state[
                :, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF
            ],
            ddq_a=np.vstack([control, control[-1]]),
            q_u=q_u,
            dq_u=state[
                :, cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF
            ],
            q_u_eq=np.array([passive_equilibrium(row) for row in q_a]),
            pump_flow=flow,
            slack=slack,
            iterations=int(solver.get_stats("sqp_iter")),
            solve_time_s=elapsed,
        )
