#!/usr/bin/env python3
"""
Write the planner's trajectory OCP as a generated acados solver.

The problem is `crane_mpc`'s -- same state, same input, same dynamics graph, same
output map -- turned from a tracker into a planner:

1. converged `SQP`, not `SQP_RTI`: a plan is solved once and then followed.
2. no stage reference. A planner has a goal, not a trajectory to track, so the
   stage rows price only rates, sway, boom excitation and input, all against zero.
3. sway is `q_u - q_eq(q_a)`. The MPC can box `q_u` because its reference supplies
   `q_eq` per stage; here `q_a` is a decision variable, so the hanging pose moves
   with the answer and the equilibrium has to be inside the expression.
4. the settled box is a constraint with `L1` slack, so a hard move comes back late
   instead of `Infeasible_Problem_Detected`.
5. **minimum time in one solve.** Normalised time over `[0, 1]`, with the horizon
   carried as a state whose derivative is zero and priced by one terminal row.

    ./scripts/export_timing_ocp.py            # rewrite `generated/`
    ./scripts/export_timing_ocp.py --check    # regenerate into a scratch tree and diff

## Every residual row is dimensionless

Each row is divided by the limit it is measured against, so a row reads 1.0 when
it sits at its bound and `W` is a pure statement of preference. Gauss-Newton
builds its Hessian out of `J' W J`, so one row left in physical units sets the
scaling of the whole problem: `tau_a` in newton metres is O(1e5) beside
neighbours that are O(1), and the solve stalls at a stationarity residual of 1e6
having spent every iteration shaving torque while the settled box blows out.
Divided by rated effort, the same solve converges in three.

The horizon is scaled the same way and for the same reason -- `T / T_nominal`,
not seconds, which bought 40-60x on stationarity.

## `W_T` is a ceiling, not a preference

With every row dimensionless, `W = I` is the honest default -- one bound violated
costs the same wherever it happens -- and the only weight that has to be chosen
is the terminal time row. On the swinging-start case, `W = I` throughout:

    W_T    T*        iters  status   pump / reservation   peak sway
    0.01   11.58 s      48     ok           0.44             0.10
    0.03    9.36 s      18     ok           0.57             0.14
    0.1     7.49 s      67     ok           0.73             0.21
    0.3     6.17 s      17     ok           0.91             0.31
    1.0     5.19 s     200   max it         1.00             0.44

0.3 is the operating point: 6.17 s with the pump at 0.91 of its reservation and
every other row well inside. Past it the time row asks for more flow than the
pump can deliver and the solve stops converging -- the regime whose honest answer
is "no such plan". **Read the pump row before believing a `T*` from a
`status != 0` solve.**

Normalising the rows is what bought the headroom: before it, the same sweep lost
convergence above 0.03. The physical answer did not move -- 6.17 s here against
6.15 s there -- so the reweighting bought conditioning, not a different plan.

The bilinear `T f(x, u)` coupling puts no local minima in this problem: `T*`
agrees to two decimals from seeds spanning 5 to 12 s, and seeding from a
converged fixed-horizon solve lands within 0.04 s of a cold start.

## What is not here: collision

No obstacle rows, deliberately. Clearance in this package is *proved* -- a margin
paired with a bound on how far the machine moves between two checked
configurations -- and a nonconvex obstacle row in a local solve would trade that
proof for a local minimum. The geometric stage hands this solver a corridor
instead: box rows on `q_a`, narrowed from the joint range, runtime-settable, which
is why nothing about them is baked here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp

PACKAGE = Path(__file__).resolve().parent.parent

# Neither `crane_ocp`'s nor `crane_model`'s `scripts/` is installed, so both
# shared modules are imported by path out of the source tree.
sys.path.insert(0, str(PACKAGE.parent / "crane_ocp" / "scripts"))

import crane_ocp_export as ox  # noqa: E402

sys.path.insert(0, str(PACKAGE))

from crane_planning import weights as w  # noqa: E402

cs = ox.import_crane_symbolic(PACKAGE)

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


# ------------------------------------------------------------------ generation


README = """\
# generated

`scripts/export_timing_ocp.py` writes this tree. Do not edit it; re-run the
script. `--check` regenerates into a scratch tree and diffs, which is what CI
runs.

The horizon, the weights `W`, every box and every `L1` slack price are set on the
generated solver at run time, so moving any of them needs no re-export. What is
baked is the **structure**: the residual rows and their scaling, the row order of
`h`, and the description the dynamics were built from.
"""


def write_header(output: Path, parameters: dict, scale: np.ndarray) -> Path:
    """Write the row offsets and divisors the caller must agree with, not invent."""
    path = output / GENERATED_HEADER
    guard = "CRANE_PLANNING_OCP_GENERATED_H_"
    divisors = ", ".join(f"{value!r}" for value in scale.tolist())
    path.write_text(
        "\n".join(
            [
                "// Generated by `scripts/export_timing_ocp.py`. Do not edit.",
                f"#ifndef {guard}",
                f"#define {guard}",
                "",
                f"#define CRANE_PLANNING_OCP_INTERVALS {int(parameters['ocp_intervals'])}",
                "// T = x[CRANE_PLANNING_OCP_X_HORIZON] * CRANE_PLANNING_OCP_HORIZON_S.",
                f"#define CRANE_PLANNING_OCP_X_HORIZON {X_HORIZON}",
                f"#define CRANE_PLANNING_OCP_HORIZON_S {float(parameters['ocp_horizon'])!r}",
                "",
                f"#define CRANE_PLANNING_OCP_H_SWAY {H_SWAY}",
                f"#define CRANE_PLANNING_OCP_H_FLOW {H_FLOW}",
                f"#define CRANE_PLANNING_OCP_NH {NH}",
                f"#define CRANE_PLANNING_OCP_HE_SWAY {HE_SWAY}",
                f"#define CRANE_PLANNING_OCP_HE_SWAY_RATE {HE_SWAY_RATE}",
                f"#define CRANE_PLANNING_OCP_NH_E {NH_E}",
                "",
                f"#define CRANE_PLANNING_OCP_SCALE {{{divisors}}}",
                "",
                f"#endif  // {guard}",
                "",
            ]
        )
    )
    return path


def generate(
    output: Path, descriptions: Path, parameters: dict, hydraulics: dict
) -> None:
    """Write the whole tree, from an empty directory."""
    output.mkdir(parents=True, exist_ok=True)
    ocp, scale, model = build_ocp(
        (descriptions / DESCRIPTION).read_text(), parameters, hydraulics
    )
    tree = ox.generate_solver(ocp, output)
    ox.write_output_map(model, ocp.model.name, tree)
    write_header(output, parameters, scale)
    ox.finalise(output, README)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE)
    arguments = parser.parse_args()

    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_planner.yaml", "crane_planner"
    )
    return ox.run(
        arguments.output,
        arguments.check,
        lambda output: generate(output, arguments.descriptions, parameters, parameters),
        "export_timing_ocp.py",
    )


if __name__ == "__main__":
    sys.exit(main())
