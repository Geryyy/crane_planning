#!/usr/bin/env python3
"""
Write the path-constrained OCP of `wiki/trajectory_planning.md` §5.2 as a generated solver.

This is `docs/features/cbs-ocp-python/grill.md` D1 and D8 for `crane_planning`, and
it is the second and last consumer of the C++ symbolic graph: the problem is
*defined* here, in Python, over `crane_model/scripts/crane_symbolic.py`, and
*shipped* as generated C under `generated/`.

    ./scripts/export_timing_ocp.py              # rewrite `generated/`
    ./scripts/export_timing_ocp.py --check      # regenerate into a scratch tree and diff

## It is not the MPC's problem and cannot share its solver

`crane_mpc/scripts/export_ocp.py` writes a tracking problem over time. This one is

    x_sigma = (sigma, sigma_dot, q_u, dq_u) in R^6,   u_sigma = sigma_ddot in R^1

with **sigma as the independent variable**, the actuated coordinates eliminated as
known functions of sigma, a minimum-traversal-time objective `int dsigma/sigma_dot`
rather than a tracking least-squares, and a full `SQP` solved to convergence rather
than `SQP_RTI`.

## What it *does* share is the model, and that sharing is the point

The force and flow expressions are **not restated here**. `crane_symbolic` carries
the dynamics, the transmission and the output map `z`; this file takes the same
`K_CYLINDER_FORCE_OFFSET` and `K_AXIS_FLOW_OFFSET` slices of that map over the same
`K_PLANNED_AXES` that `export_ocp.py` takes, so `trajectory_planning` §5.3's *"a
reference the MPC would reject is a planner bug"* holds by construction rather than
by agreement between two implementations. `test_timing_ocp.cpp` asserts it from
outside, on the shipped artifacts.

The passive rows are the module's `ddq_u` -- `wiki/robot_model.md` §3.1's Schur
complement -- with the actuated position, velocity and acceleration substituted by
the path's own `q_a(sigma)`, `q_a'(sigma) sigma_dot` and
`q_a'' sigma_dot^2 + q_a' sigma_ddot`. Nothing here writes an equation of motion.

## What is baked and what stays runtime-settable

Baked, because acados fixes them at code generation: the row order of `h`, each
row's conditioning divisor, the dynamics and the description each solver was built
from, and `nlp_solver_max_iter` -- which sizes the SQP's own statistics array, so a
caller asking for more than the artifact allocated is refused rather than silently
overrunning it.

Runtime-settable, and set by `src/timing_ocp.cpp` on every solve: `W` and `yref`
(which is where `sway_weight`, `input_weight` and the grid's `dsigma` live -- see
`residual_vector` below), every state and input box, `lh`/`uh` (which is where
kappa, `speed_scale` and the actuation limits live), the Levenberg-Marquardt term,
the four tolerances, the wall-clock budget, **and `N`**: acados writes
`<name>_acados_create_with_discretization(capsule, N, steps)` beside the fixed-`N`
entry point, so `TimingOcpSettings::intervals` is still an argument.

## The generated tree is normalised on write

`ament_cpplint` and `ament_cppcheck` match `.c` and `.h`, and machine output fails
them by thousands. What this tree can be is a **fixed point of the hooks that
rewrite files**: every generated file has its trailing whitespace stripped and ends
in exactly one newline. Do not pass `generated/` to `pre-commit run --files`.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np
import yaml
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

PACKAGE = Path(__file__).resolve().parent.parent

# `crane_model`'s `scripts/` is not installed (issue 070's notes), so the shared
# model module is imported by path out of the source tree, exactly as
# `crane_mpc/scripts/export_ocp.py` imports it.
MODEL_PACKAGE = PACKAGE.parent / "crane_model"
sys.path.insert(0, str(MODEL_PACKAGE / "scripts"))

import crane_symbolic as cs  # noqa: E402

DEFAULT_DESCRIPTIONS = MODEL_PACKAGE / "test" / "description"

# One solver per tool, for the reason `crane_mpc` ships two: the description is
# baked into the dynamics, and the two descriptions are different machines.
DESCRIPTIONS = (
    ("pzs100", "pzs100.urdf"),
    ("epsilon7040", "epsilon_7040.urdf"),
)

SOLVER_PREFIX = "crane_planning_timing"
GENERATED_HEADER = "crane_planning_timing_ocp_generated.h"

UNSHIPPED = ("Makefile", "acados_solver.pxd")
UNSHIPPED_PREFIXES = ("main_", "acados_sim_solver_")
UNSHIPPED_SUFFIXES = ("_hess.c",)

# --- the problem's own dimensions ---------------------------------------------

NX = 6  # (sigma, sigma_dot, q_u, dq_u)
NU = 1  # sigma_ddot

X_SIGMA = 0
X_SIGMA_RATE = 1
X_PASSIVE_POSITION = 2
X_PASSIVE_VELOCITY = 4

# One block of path data at one value of sigma: q_a(sigma), q_a'(sigma),
# q_a''(sigma) over the five planned coordinates, then the tool coordinate the
# model is pinned at there. The tool's rate and acceleration are not carried
# because `crane_symbolic` pins them at zero -- the gripper is driven by the
# low-level controller, so no timing this OCP chooses can move it.
BLOCK_POSITION = 0
BLOCK_SLOPE = cs.K_PLANNED_DOF
BLOCK_CURVATURE = 2 * cs.K_PLANNED_DOF
BLOCK_TOOL = 3 * cs.K_PLANNED_DOF
PATH_BLOCK = 3 * cs.K_PLANNED_DOF + 1

# `p`: the node block, the block the integrator holds the path fixed at over the
# interval, then the payload body. Two blocks because the cost and the
# constraints are written at the node while the dynamics is integrated across the
# interval, and one path sample cannot be both.
PARAMETER_NODE = 0
PARAMETER_MID = PATH_BLOCK
PARAMETER_PAYLOAD = 2 * PATH_BLOCK
PAYLOAD_DOF = cs.NP - cs.P_PAYLOAD_MASS
NP = 2 * PATH_BLOCK + PAYLOAD_DOF

# `h`: five acceleration rows, five cylinder force rows, one pump flow row. The
# tool has a row in none of the three -- `timing_ocp.hpp` gives the reason.
CONSTRAINT_ACCELERATION = 0
CONSTRAINT_CYLINDER_FORCE = cs.K_PLANNED_DOF
CONSTRAINT_PUMP_FLOW = 2 * cs.K_PLANNED_DOF
NH = 2 * cs.K_PLANNED_DOF + 1

# `y`: traversal time, the two sway rows, the input.
RESIDUAL_TIME = 0
RESIDUAL_SWAY = 1
RESIDUAL_INPUT = 1 + cs.K_PASSIVE_DOF
NY = 2 + cs.K_PASSIVE_DOF
NY_E = cs.K_PASSIVE_DOF

# The artifact's defaults. `TimingOcpSettings` carries the same numbers and is
# what a deployment sets; only `MAX_ITERATIONS` is a ceiling rather than a
# default, because it sizes the SQP's statistics array at code generation.
DEFAULT_INTERVALS = 40
MAX_ITERATIONS = 250
DEFAULT_LEVENBERG_MARQUARDT = 1.0e-2


# ------------------------------------------------------------------ the numbers


def read_parameters(path: Path) -> dict:
    """Read this package's hydraulic limits down to their `ros__parameters`."""
    with open(path) as stream:
        root = yaml.safe_load(stream)
    node = root["crane_planner"]["ros__parameters"]
    if not isinstance(node, dict):
        raise ValueError(f"{path}: crane_planner.ros__parameters is not a mapping")
    return node


def constraint_scale(model: cs.CraneSymbolicModel, hydraulics: dict) -> np.ndarray:
    """
    Return the divisor of each row of `h`, in that row's own physical unit.

    **Conditioning and not a bound.** Written in physical units the eleven rows
    span nine decades -- radians per second squared near one, newtons near `1e5`,
    cubic metres per second near `1e-3` -- and HPIPM fails on the constraint
    Jacobian that produces: `ACADOS_QP_FAILURE` on the first step, for a problem
    whose functions are all finite and whose solution exists.

    The divisors are baked and the limits are not: kappa, `speed_scale` and every
    entry of `ActuationLimits` reach the solver through `lh`/`uh`, which
    `src/timing_ocp.cpp` divides by exactly these numbers.

    The acceleration rows are divided by one, and that is a number rather than an
    omission: `parameters.md` §3's `ddq_a^max` is already of order one on every
    planned axis, so a divisor would only move the row away from the scale the
    other ten are being brought to.
    """
    relief = float(hydraulics["system_pressure_pa"])
    pressure = np.full(cs.K_ACTUATED_DOF, relief)
    zero = np.zeros(cs.K_ACTUATED_DOF)
    # `wiki/hydraulics.md` §4's F_i = A_A p_A - A_B p_B, asked of the shared model
    # rather than restated: the two chambers pressurised one at a time. The
    # **larger** of the two, exactly as `export_ocp.py`'s own divisor, so a row
    # this package conditions and a row the MPC conditions are the same number.
    extend = np.array(ca.evalf(model.chamber_force(pressure, zero))).ravel()
    retract = np.array(ca.evalf(model.chamber_force(zero, pressure))).ravel()

    scale = np.ones(NH)
    for axis in cs.K_PLANNED_AXES:
        scale[CONSTRAINT_CYLINDER_FORCE + axis] = max(
            abs(extend[axis]), abs(retract[axis])
        )
    scale[CONSTRAINT_PUMP_FLOW] = float(
        hydraulics["pump_flow_planning_factor"]
    ) * float(hydraulics["pump_flow_max"])
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            "every row of h needs a finite positive divisor; a zero one is a "
            "constraint with no right-hand side"
        )
    return scale


# ------------------------------------------------------------------- assembly


def path_point(block, sigma_rate, sigma_accel, q_u, dq_u, payload):
    """
    Return `crane_symbolic`'s own `(x, u, p)` at one sampled point of the path.

    This is the whole of the reduction §5.2 performs: the actuated coordinates are
    **eliminated** as known functions of sigma, so what the shared model is asked
    about is a state assembled out of the path block and the four sway rows that
    remain free.

        q_a  = q_a(sigma_k)                       (a stage parameter)
        dq_a = q_a'(sigma_k) sigma_dot            (§5's chain rule)
        ddq_a = q_a''(sigma_k) sigma_dot^2 + q_a'(sigma_k) sigma_ddot
    """
    q_a = block[BLOCK_POSITION : BLOCK_POSITION + cs.K_PLANNED_DOF]
    slope = block[BLOCK_SLOPE : BLOCK_SLOPE + cs.K_PLANNED_DOF]
    curvature = block[BLOCK_CURVATURE : BLOCK_CURVATURE + cs.K_PLANNED_DOF]
    dq_a = slope * sigma_rate
    ddq_a = curvature * (sigma_rate * sigma_rate) + slope * sigma_accel
    return (
        ca.vertcat(q_a, q_u, dq_a, dq_u),
        ddq_a,
        ca.vertcat(block[BLOCK_TOOL], payload),
    )


def at_path_point(model: cs.CraneSymbolicModel, expression, point):
    """Evaluate one of the shared model's expressions at `path_point`'s state."""
    return ca.substitute(
        expression,
        ca.vertcat(model.x, model.u, model.p),
        ca.vertcat(*point),
    )


def build_ocp(description_xml: str, tool: str, hydraulics: dict) -> tuple:
    """Assemble `wiki/trajectory_planning.md` §5.2 for one description."""
    model = cs.CraneSymbolicModel(description_xml, tool)
    scale = constraint_scale(model, hydraulics)

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)

    sigma_rate = x[X_SIGMA_RATE]
    q_u = x[X_PASSIVE_POSITION : X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    dq_u = x[X_PASSIVE_VELOCITY : X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF]
    payload = p[PARAMETER_PAYLOAD:NP]

    node = p[PARAMETER_NODE : PARAMETER_NODE + PATH_BLOCK]
    mid = p[PARAMETER_MID : PARAMETER_MID + PATH_BLOCK]

    acados_model = AcadosModel()
    acados_model.name = f"{SOLVER_PREFIX}_{tool}"
    acados_model.x = x
    acados_model.u = u
    acados_model.p = p

    # --- §5.2's dynamics, divided through by sigma_dot ------------------------
    #
    # The passive block is the shared module's `ddq_u`, which is
    # `wiki/robot_model.md` §3.1's `-M_uu^-1 (M_ua ddq_a + h_u)`, and it is read
    # rather than restated. Every time derivative is then divided by `sigma_dot`,
    # because sigma and not time is the independent variable. The first row is
    # `dsigma/dsigma = 1`, and it is what makes §5.4's `sigma(T) = 1` a terminal
    # condition on a state rather than a property of the grid.
    #
    # The **midpoint** block, because this is what the integrator crosses the
    # interval with; the cost and the constraints below use the node's.
    ddq_u = at_path_point(
        model, model.ddq_u, path_point(mid, sigma_rate, u, q_u, dq_u, payload)
    )
    acados_model.f_expl_expr = ca.vertcat(
        ca.SX(1.0), u / sigma_rate, dq_u / sigma_rate, ddq_u / sigma_rate
    )

    # --- §5.2's objective, as a Gauss-Newton residual -------------------------
    #
    # acados' nonlinear least squares cost is `0.5 ||y||^2_W`, so the first row
    # squares to `dsigma / sigma_dot` under `W_00 = 2 dsigma` -- which is §5.2's
    # `int dsigma / sigma_dot` -- and the two sway rows to
    # `w ||dq_u||^2 dsigma / sigma_dot`, which is its `int ||dq_u||^2 dt` because
    # `dt = dsigma / sigma_dot`.
    #
    # **Every weight and the grid spacing live in `W` and not in the expression.**
    # That is what keeps `sway_weight`, `input_weight` and `intervals` runtime
    # settings rather than regenerate-only ones.
    residual = ca.vertcat(1.0 / ca.sqrt(sigma_rate), dq_u / ca.sqrt(sigma_rate), u)
    # The terminal stage has no input and no traversal time left to spend; what
    # §5.4 asks of it is that the tool is not moving, and `q_u(T) = q_u^eq` is a
    # box rather than a cost.
    terminal_residual = dq_u
    acados_model.cost_y_expr_0 = residual
    acados_model.cost_y_expr = residual
    acados_model.cost_y_expr_e = terminal_residual

    # --- §5.3 and the §3 guarantee table, as one vector -----------------------
    #
    # Joint *position* is absent on purpose: it is a property of the geometry,
    # which `fit_c2_path` has already refused a path for, and sigma is pinned to
    # the grid, so nothing the OCP decides can move it. Joint *velocity* is
    # absent because it is `|q_a'(sigma_k)| sigma_dot` with a known `q_a'`, i.e. a
    # per-stage bound on one state and cheaper as a box than as a nonlinear row.
    #
    # The force and flow rows are the same slices of the same output map
    # `export_ocp.py` takes, over the same planned axes.
    point = path_point(node, sigma_rate, u, q_u, dq_u, payload)
    ddq_a_node = point[1]
    z = at_path_point(model, model.z, point)
    force = z[
        cs.K_CYLINDER_FORCE_OFFSET : cs.K_CYLINDER_FORCE_OFFSET + cs.K_ACTUATED_DOF
    ]
    flow = z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_ACTUATED_DOF]

    rows = [
        ddq_a_node[axis] / scale[CONSTRAINT_ACCELERATION + axis]
        for axis in cs.K_PLANNED_AXES
    ]
    rows += [
        force[axis] / scale[CONSTRAINT_CYLINDER_FORCE + axis]
        for axis in cs.K_PLANNED_AXES
    ]
    rows.append(
        ca.sum1(flow[: cs.K_PLANNED_DOF]) / scale[CONSTRAINT_PUMP_FLOW]
    )
    constraint = ca.vertcat(*rows)
    acados_model.con_h_expr_0 = constraint
    acados_model.con_h_expr = constraint
    # No terminal row: the terminal stage has no input, so `ddq_a` and with it
    # `tau_a` are undefined there. `crane_mpc` drops its own at the same node for
    # the same reason.

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.parameter_values = np.zeros(NP)

    # sigma runs over [0, 1], so the horizon is one and the step is `dsigma`.
    # Both are arguments again at runtime -- see the module docstring.
    ocp.solver_options.N_horizon = DEFAULT_INTERVALS
    ocp.solver_options.tf = 1.0

    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_0 = np.eye(NY)
    ocp.cost.W = np.eye(NY)
    ocp.cost.W_e = np.eye(NY_E)
    ocp.cost.yref_0 = np.zeros(NY)
    ocp.cost.yref = np.zeros(NY)
    ocp.cost.yref_e = np.zeros(NY_E)

    # Every state row is boxed at every stage, stage zero included, and the
    # *values* arrive at runtime because all three kinds of stage are different:
    # §7's measured initial condition at stage 0, §3 constraint 3's sway box
    # around each node's own equilibrium in between, and §5.4's terminal
    # condition at the end.
    #
    # **`x0` is deliberately not set.** It would declare the stage-0 rows to
    # acados as equalities through `idxbxe_0`, and `timing_ocp.hpp` records what
    # that costs in this acados build: every converging solve turns into
    # `ACADOS_QP_FAILURE` on the first QP, HPIPM status 3, a NaN at QP iteration
    # 4. The rows are held by `lbx_0`/`ubx_0` alone.
    ocp.constraints.idxbx_0 = np.arange(NX)
    ocp.constraints.lbx_0 = -np.ones(NX)
    ocp.constraints.ubx_0 = np.ones(NX)
    ocp.constraints.idxbx = np.arange(NX)
    ocp.constraints.lbx = -np.ones(NX)
    ocp.constraints.ubx = np.ones(NX)
    ocp.constraints.idxbx_e = np.arange(NX)
    ocp.constraints.lbx_e = -np.ones(NX)
    ocp.constraints.ubx_e = np.ones(NX)
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = -np.ones(NU)
    ocp.constraints.ubu = np.ones(NU)

    # The flow row is one-sided because `Q` is a pump draw and not a signed
    # force. Both ends of every row are rewritten at runtime.
    ocp.constraints.lh_0 = np.concatenate([-np.ones(NH - 1), [0.0]])
    ocp.constraints.uh_0 = np.ones(NH)
    ocp.constraints.lh = np.concatenate([-np.ones(NH - 1), [0.0]])
    ocp.constraints.uh = np.ones(NH)

    # A full `SQP` solved to convergence, not `SQP_RTI`: a plan is computed once
    # and then tracked, so there is no cycle to be deterministic inside of, and
    # §7's latency is bounded by a wall clock rather than by an iteration count.
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    # A merit line search rather than a full step. The traversal-time cost is
    # `1/sigma_dot`, whose Gauss-Newton curvature *falls* as the rate rises, so a
    # full Newton step from a slow guess overshoots by orders of magnitude and
    # the QP is then asked about a state the model has no answer for.
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.levenberg_marquardt = DEFAULT_LEVENBERG_MARQUARDT
    # The one option that is a **ceiling** and not a default: acados sizes the
    # SQP's statistics array from it at code generation, so `timing_ocp.cpp`
    # refuses a `max_iterations` above this rather than overrunning the array.
    ocp.solver_options.nlp_solver_max_iter = MAX_ITERATIONS
    ocp.solver_options.timeout_max_time = 0.0

    return ocp, scale, model


# ------------------------------------------------------------------ generation


def write_output_map(model: cs.CraneSymbolicModel, name: str, tree: Path) -> None:
    """
    Code-generate the shared output map beside the solver, in physical units.

    acados generates only what it solves, and what it solves is eleven rows each
    already divided by its conditioning constant. Three things need the physical
    quantity instead: `OcpNode`'s record of what the answer demands of the
    machine, the static-force refusal that names a path no timing exists for, and
    the bisected warm start, which asks how far outside constraints 6 and 7 a
    node is at a candidate rate.

    It is `crane_symbolic`'s own `z` over the module's own `(x, u, p)` --
    **exactly the function `crane_mpc` ships**, up to the symbol prefix, which is
    what `test_timing_ocp.cpp` compares the two trees on.
    """
    function = ca.Function(
        f"{name}_output",
        [model.x, model.u, model.p],
        [ca.densify(model.z)],
        ["x", "u", "p"],
        ["z"],
    )
    generator = ca.CodeGenerator(
        f"{name}_output.c",
        {
            "mex": False,
            "casadi_int": "int",
            "casadi_real": "double",
            "with_header": True,
        },
    )
    generator.add(function)
    generator.generate(str(tree) + "/")


def normalise(path: Path) -> None:
    """Strip trailing whitespace and leave exactly one final newline."""
    text = path.read_text()
    body = "\n".join(line.rstrip() for line in text.splitlines())
    path.write_text(body.rstrip("\n") + "\n")


def prune(tree: Path) -> None:
    """Delete what acados generates and this package does not ship."""
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        if (
            path.name in UNSHIPPED
            or path.name.startswith(UNSHIPPED_PREFIXES)
            or path.name.endswith(UNSHIPPED_SUFFIXES)
        ):
            path.unlink()


def write_header(output: Path, scale: np.ndarray) -> Path:
    """Write the numbers `src/timing_ocp.cpp` would otherwise derive a second time."""
    path = output / GENERATED_HEADER
    guard = "CRANE_PLANNING_TIMING_OCP_GENERATED_H_"
    scales = ", ".join(f"{value!r}" for value in scale.tolist())
    entries = ", ".join(f"{{{row}, {column}}}" for row, column in cs.INERTIA_ENTRIES)
    lines = [
        "// Generated by `scripts/export_timing_ocp.py`. Do not edit; re-run the script.",
        "//",
        "// The OCP of `wiki/trajectory_planning.md` §5.2 as it was written, in the",
        "// numbers the C++ side must agree with rather than re-derive. Dimensions",
        "// acados carries itself are in `acados_solver_crane_planning_timing_<tool>.h`",
        "// and are deliberately not repeated.",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "// x_sigma = (sigma, sigma_dot, q_u, dq_u) and u_sigma = sigma_ddot.",
        f"#define CRANE_PLANNING_TIMING_NX {NX}",
        f"#define CRANE_PLANNING_TIMING_NU {NU}",
        f"#define CRANE_PLANNING_TIMING_STATE_SIGMA {X_SIGMA}",
        f"#define CRANE_PLANNING_TIMING_STATE_SIGMA_RATE {X_SIGMA_RATE}",
        f"#define CRANE_PLANNING_TIMING_STATE_PASSIVE_POSITION {X_PASSIVE_POSITION}",
        f"#define CRANE_PLANNING_TIMING_STATE_PASSIVE_VELOCITY {X_PASSIVE_VELOCITY}",
        "",
        "// `p`: the node's path block, the block the integrator holds the path fixed",
        "// at over the interval, then the payload body -- mass, centre of mass in K8",
        "// and the six independent entries of Theta_L. One block is",
        "// `q_a(sigma), q_a'(sigma), q_a''(sigma)` over the five planned coordinates",
        "// and then the tool coordinate the model is pinned at; the tool's rate and",
        "// acceleration are not carried, because the shared model pins them at zero.",
        f"#define CRANE_PLANNING_TIMING_NP {NP}",
        f"#define CRANE_PLANNING_TIMING_PLANNED_DOF {cs.K_PLANNED_DOF}",
        f"#define CRANE_PLANNING_TIMING_PATH_BLOCK {PATH_BLOCK}",
        f"#define CRANE_PLANNING_TIMING_PARAMETER_NODE {PARAMETER_NODE}",
        f"#define CRANE_PLANNING_TIMING_PARAMETER_MID {PARAMETER_MID}",
        f"#define CRANE_PLANNING_TIMING_PARAMETER_PAYLOAD {PARAMETER_PAYLOAD}",
        f"#define CRANE_PLANNING_TIMING_BLOCK_POSITION {BLOCK_POSITION}",
        f"#define CRANE_PLANNING_TIMING_BLOCK_SLOPE {BLOCK_SLOPE}",
        f"#define CRANE_PLANNING_TIMING_BLOCK_CURVATURE {BLOCK_CURVATURE}",
        f"#define CRANE_PLANNING_TIMING_BLOCK_TOOL {BLOCK_TOOL}",
        "",
        "// The payload half of `p`, and the (row, column) of Theta_L's six",
        "// independent entries in the order they are packed. Theta_L is symmetric",
        "// and about the payload's own centre of mass with the axes of K8, which is",
        "// the URDF `<inertial>` convention.",
        f"#define CRANE_PLANNING_TIMING_PAYLOAD_DOF {PAYLOAD_DOF}",
        f"#define CRANE_PLANNING_TIMING_PAYLOAD_MASS {cs.P_PAYLOAD_MASS - cs.P_PAYLOAD_MASS}",
        f"#define CRANE_PLANNING_TIMING_PAYLOAD_COM {cs.P_PAYLOAD_COM - cs.P_PAYLOAD_MASS}",
        "#define CRANE_PLANNING_TIMING_PAYLOAD_INERTIA "
        f"{cs.P_PAYLOAD_INERTIA - cs.P_PAYLOAD_MASS}",
        f"#define CRANE_PLANNING_TIMING_INERTIA_ENTRIES {{{entries}}}",
        "",
        "// The rows of `h`: five accelerations, five cylinder forces, one pump flow.",
        f"#define CRANE_PLANNING_TIMING_NH {NH}",
        f"#define CRANE_PLANNING_TIMING_CONSTRAINT_ACCELERATION {CONSTRAINT_ACCELERATION}",
        "#define CRANE_PLANNING_TIMING_CONSTRAINT_CYLINDER_FORCE "
        f"{CONSTRAINT_CYLINDER_FORCE}",
        f"#define CRANE_PLANNING_TIMING_CONSTRAINT_PUMP_FLOW {CONSTRAINT_PUMP_FLOW}",
        "",
        "// What each row of `h` was divided by, in that row's own physical unit:",
        "// one on the acceleration rows, newtons on the force rows and m^3/s on the",
        "// pump row. Conditioning and **not** a bound -- `lh`/`uh` carry kappa,",
        "// `speed_scale` and the actuation limits, divided by exactly these.",
        f"#define CRANE_PLANNING_TIMING_CONSTRAINT_SCALE {{{scales}}}",
        "",
        "// The rows of the stage residual `y = [1/sqrt(sigma_dot),",
        "// dq_u/sqrt(sigma_dot), sigma_ddot]`. The weights and the grid's dsigma are",
        "// in `W`, which is why they are still runtime settings.",
        f"#define CRANE_PLANNING_TIMING_NY {NY}",
        f"#define CRANE_PLANNING_TIMING_NY_E {NY_E}",
        f"#define CRANE_PLANNING_TIMING_RESIDUAL_TIME {RESIDUAL_TIME}",
        f"#define CRANE_PLANNING_TIMING_RESIDUAL_SWAY {RESIDUAL_SWAY}",
        f"#define CRANE_PLANNING_TIMING_RESIDUAL_INPUT {RESIDUAL_INPUT}",
        "",
        "// The four blocks of the shipped output map `z`, six axes each. These are",
        "// `crane_symbolic`'s own offsets, and they are the offsets",
        "// `crane_mpc/scripts/export_ocp.py` slices constraints 6 and 7 out of --",
        "// which is what makes trajectory_planning §5.3 structural.",
        f"#define CRANE_PLANNING_TIMING_OUTPUT_DOF {cs.K_OUTPUT_DOF}",
        "#define CRANE_PLANNING_TIMING_OUTPUT_ACTUATED_FORCE "
        f"{cs.K_ACTUATED_FORCE_OFFSET}",
        "#define CRANE_PLANNING_TIMING_OUTPUT_CYLINDER_FORCE "
        f"{cs.K_CYLINDER_FORCE_OFFSET}",
        "#define CRANE_PLANNING_TIMING_OUTPUT_PISTON_VELOCITY "
        f"{cs.K_PISTON_VELOCITY_OFFSET}",
        f"#define CRANE_PLANNING_TIMING_OUTPUT_AXIS_FLOW {cs.K_AXIS_FLOW_OFFSET}",
        "",
        "// The grid the artifact was generated on, which is a **default**: acados'",
        "// `_acados_create_with_discretization` keeps N an argument. The iteration",
        "// cap is not a default -- it sizes the SQP's statistics array, so a solve",
        "// asking for more is refused.",
        f"#define CRANE_PLANNING_TIMING_DEFAULT_INTERVALS {DEFAULT_INTERVALS}",
        f"#define CRANE_PLANNING_TIMING_MAX_ITERATIONS {MAX_ITERATIONS}",
        "",
        f"#endif  // {guard}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


README = """\
# `crane_planning/generated/` -- the shipped timing solver

Machine output. **Do not edit any file here**: change
`scripts/export_timing_ocp.py`, `crane_model/scripts/crane_symbolic.py` or one of
the config files below, re-run

    ./scripts/export_timing_ocp.py

and commit what changes. `./scripts/export_timing_ocp.py --check` regenerates into
a scratch tree and diffs, which is what says the tree still matches its inputs.

## What is in here

    crane_planning_timing_ocp_generated.h    the row offsets, the parameter
                                             packing and the conditioning
                                             divisors -- written by the export,
                                             read by `src/timing_ocp.cpp`
    crane_planning_timing_pzs100/            the PZS100 timing solver
    crane_planning_timing_epsilon7040/       the 7040 timing solver

Two solvers because the description is *baked in*: a generated solver is one
machine's, and `test_timing_ocp.cpp` solves on both.

Inside each solver directory, `crane_planning_timing_<tool>_output.{c,h}` is not
acados' -- it is `crane_symbolic`'s output map `z` code-generated beside the
solver, all six axes of `tau_a`, `F_cyl`, `v` and `Q` in physical units. acados
generates only what it solves, which is eleven rows already divided by their
conditioning constants, and three things need the physical quantity: `OcpNode`,
the static-force refusal and the bisected warm start.

It is the **same function `crane_mpc` ships**, up to the symbol prefix, because
both are `crane_symbolic`'s `z` over the module's own `(x, u, p)`.
`test_timing_ocp.cpp` compares the two trees on exactly that.

## There is no staleness guard, and that is a decision

The conditioning divisors come from `config/hydraulic_limits.yaml`, the smoothing
and the hydraulic constants from `crane_model/config/hydraulics.yaml`, and the
dynamics from the two descriptions under `crane_model/test/description/`.
**Nothing in the build or the test suite fails when one of those files and this
tree disagree.**

`docs/features/cbs-ocp-python/grill.md` §4 records the alternative that was
considered and rejected -- hashing the inputs into the generated code and
comparing in a test -- and records that this is a deliberate divergence from both
existing generator scripts in this repository.

What is **not** in the staleness surface is `config/crane_planner.yaml` and the
whole of `TimingOcpSettings`: kappa, the actuation limits, the boxes, the weights,
`intervals`, the tolerances and the wall-clock budget are every one of them set on
the generated solver before each solve. Only `max_iterations` has a ceiling, and
asking for more than it is refused rather than ignored.
"""


def generate(output: Path, descriptions: Path, hydraulics: dict) -> None:
    """Write the whole tree, from an empty directory."""
    output.mkdir(parents=True, exist_ok=True)
    scale = None
    for tool, description in DESCRIPTIONS:
        ocp, scale, model = build_ocp(
            (descriptions / description).read_text(), tool, hydraulics
        )
        name = ocp.model.name
        tree = output / name
        ocp.code_export_directory = str(tree)
        # The JSON is the only artifact carrying the absolute path of the tree it
        # was written into, so it goes to a scratch file: a checked-in copy would
        # differ between a devcontainer and a native checkout for a reason that
        # has nothing to do with the OCP.
        with tempfile.TemporaryDirectory() as scratch:
            AcadosOcpSolver.generate(ocp, json_file=str(Path(scratch) / f"{name}.json"))
        prune(tree)
        write_output_map(model, name, tree)

    write_header(output, scale)
    (output / "README.md").write_text(README)
    for path in sorted(output.rglob("*")):
        if path.is_file():
            normalise(path)


def compare(left: Path, right: Path) -> list:
    """Return every path under `left` that `right` does not match byte for byte."""
    differences = []
    names = {path.relative_to(left) for path in left.rglob("*") if path.is_file()}
    names |= {path.relative_to(right) for path in right.rglob("*") if path.is_file()}
    for name in sorted(names):
        one = left / name
        other = right / name
        if not one.is_file() or not other.is_file():
            differences.append(name)
        elif not filecmp.cmp(one, other, shallow=False):
            differences.append(name)
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_DESCRIPTIONS,
        help="where the two expanded machine descriptions are",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "generated",
        help="where the generated solvers go",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a scratch tree and diff instead of rewriting",
    )
    arguments = parser.parse_args()

    hydraulics = read_parameters(PACKAGE / "config" / "hydraulic_limits.yaml")

    if not arguments.check:
        if arguments.output.exists():
            shutil.rmtree(arguments.output)
        generate(arguments.output, arguments.descriptions, hydraulics)
        print(f"wrote {arguments.output}")
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch) / "generated"
        generate(fresh, arguments.descriptions, hydraulics)
        differences = compare(arguments.output, fresh)
    if differences:
        print(
            "the checked-in solver is not what `export_timing_ocp.py` writes "
            "today. Re-run the script and commit the result; if only the CasADi "
            "or acados banner moved, the toolchain changed rather than the model."
        )
        for name in differences:
            print(f"  {name}")
        return 1
    print("the checked-in solver is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
