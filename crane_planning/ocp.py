"""
The trajectory OCP: its definition, and the solver that answers with it.

`x = [sigma, q_u, v, dq_u, theta]`, `u = a`, on normalised time over [0, 1] with
`theta = T / ocp_horizon` a state whose derivative is zero -- so one solve is
time-optimal and there is no outer search over the duration.

**The machine stays on the certified curve.** `q_a` is not a decision variable:
the geometric stage hands over `c(sigma)`, the curve `Geometry.check_path`
certified, and this solver decides only how fast to move along it.

    q_a   = c(sigma)
    dq_a  = c'(sigma) v
    ddq_a = c''(sigma) v^2 + c'(sigma) a         with  v = d sigma/dt,  a = dv/dt

So the executed curve *is* the certified curve and the clearance proof holds
verbatim -- no obstacle rows, no corridor, nothing to re-prove. The stage this
replaced planned `q_a` freely between two endpoints and left that curve by
0.84-1.29 m of machine displacement against 0.05 m of spare clearance, which
voided the proof; measured against it, staying on the curve costs 0.4-4.4% of
duration and half the SQP iterations.

What it costs is lateral authority: sway can now only be damped by *timing*.
That is the whole price, and the numbers above are what it comes to.

The definition lives here, not beside the exporter, because this package's
consumer is Python: `scripts/export_timing_ocp.py` is a CLI front end on
`build_ocp`, and `TrajectoryOcp` is the same problem with a solver attached.

Every residual row is divided by the limit it is measured against, so 1.0 is the
bound and `W` is preference alone (see `weights`). Gauss-Newton builds its
Hessian from `J' W J`, so one row left in physical units sets the conditioning of
everything: `tau_a` in Nm stalled this solve at a stationarity residual of 1e6,
and the horizon in seconds cost 40-60x on the same measure.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crane_model import symbolic as cs

from . import weights as w
from .config import PlanningError, passive_equilibrium

#: What `scripts/export_timing_ocp.py` writes: source only, pruned, and what
#: `--check` reviews. Nothing at run time writes here.
GENERATED = Path(__file__).resolve().parent.parent / "generated"

#: Where the solver is compiled, deliberately **outside** the source tree. The
#: shared exporter normalises every file it ships and cannot walk a `.so`, so a
#: built tree and a shippable one are not the same tree; keeping the build here
#: is what stops a run overwriting the artifact under review.
CACHE = (
    Path(os.environ.get("CRANE_PLANNING_OCP_CACHE", tempfile.gettempdir()))
    / "crane_planning_ocp"
)

TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"
SOLVER_NAME = f"crane_planning_ocp_{TOOL}"
GENERATED_HEADER = "crane_planning_ocp_generated.h"

#: The two hinges the hanging pose is a closed form in. `crane_model.symbolic`
#: numbers the planned axes but does not name them; `planner.py` carries the same.
BOOM_AXIS = 1
ARM_AXIS = 2

#: Row order of `x`. `theta` is last, as the one component the solver picks.
X_SIGMA, X_PASSIVE, X_SPEED, X_PASSIVE_RATE, X_HORIZON = 0, 1, 3, 4, 6
NX, NU = 7, 1

#: Row order of `h`, which the caller reads back and must not re-derive. The
#: rate and input blocks were boxes on `x` and `u` when `q_a` was planned; they
#: are nonlinear rows now because what they bound is an expression in `sigma`.
H_SWAY, H_FLOW, H_RATE, H_INPUT = 0, 2, 3, 3 + cs.K_PLANNED_DOF
NH = 3 + 2 * cs.K_PLANNED_DOF
#: Row order of `h_e`: the settled box the caller accepts.
HE_SWAY, HE_SWAY_RATE, NH_E = 0, 2, 4

#: Cubic path, so four power-basis coefficients per segment.
ORDER = 4

#: Box on `a`, a numerical guard and not a physical limit -- `sigma''` is bounded
#: only through the `ddq_a` rows, and where the tangent is short that leaves its
#: column nearly free. Measured at 0.01-0.02 of this across the shipped defaults
#: and `speed_scale` 0.9; `solve` refuses rather than answer if it ever binds,
#: because a duration this constant decided is not the machine's.
SIGMA_INPUT_MAX = 20.0

#: What the compiled solver's structure depends on. `weights` is deliberately
#: absent: `W` is a runtime cost field the node writes onto the built solver, so
#: hashing it would rebuild for a number that changes nothing.
BAKED = (
    "ocp_intervals",
    "ocp_horizon",
    "ocp_integrator",
    "ocp_max_iterations",
    "ocp_tolerance",
    "levenberg_marquardt",
    "q_sway_max",
    "dq_sway_max",
    "ddq_a_max",
    "path_segments",
)

#: The integrator, and the stage count that makes each one order 4: ERK wants 4
#: stages, Gauss-Legendre IRK 2. ERK4 is far cheaper per node but its stability
#: region is finite and `theta` scales the step, so the choice is not uniform
#: over the duration box: at `ocp_duration_max` the pendulum (w = 3.6 rad/s)
#: sits at w*dt = 1.8, where RK4's per-step amplification is 0.854 -- 500x of
#: fabricated damping over 40 intervals, pricing the sway rows as if the swing
#: settled on its own. Gauss-Legendre is symplectic, |R(iy)| = 1 at any step, so
#: it cannot invent that. At the ~7 s answer ERK4 is within 1.6% over the whole
#: horizon, which is why this is a knob and not a correction.
INTEGRATORS = {"ERK": 4, "IRK": 2}


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


def cache_tree(parameters: dict, hydraulics: dict) -> Path:
    """
    Where this configuration's solver is built, named so a stale one cannot load.

    A cached `.so` is loaded, not compared. Every value that enters the
    *expressions* or the dimensions therefore has to move the directory, or a
    changed `ocp_integrator` -- or a changed `path_segments`, which changes the
    parameter vector itself -- silently answers with the previous build.
    """
    baked = {}
    for key in BAKED:
        value = parameters[key]
        baked[key] = value.tolist() if hasattr(value, "tolist") else value
    baked["pump_flow_max"] = float(hydraulics["pump_flow_max"])
    digest = hashlib.sha1(json.dumps(baked, sort_keys=True).encode()).hexdigest()
    return CACHE / f"{SOLVER_NAME}_{digest[:10]}"


# -------------------------------------------------------------------- the path


def power_coefficients(path, segments: int) -> np.ndarray:
    """
    `(segments, 4, 5)`: `c(sigma) = sum_m a[k, m] (sigma - k/S)^m` on segment k.

    A power basis on uniform breakpoints, not the B-spline basis, because the
    solver has to select a segment symbolically and uniform breakpoints make that
    a `floor` instead of a knot search. `geometry.fit` places its interior knots
    on `linspace(0, 1, path_segments + 1)` and always uses all of them, so nothing
    is approximated here -- the conversion is exact, and it is checked.
    """
    spline = path.spline
    breaks = np.unique(spline.t)
    if len(breaks) - 1 != segments or not np.allclose(
        breaks, np.linspace(0.0, 1.0, segments + 1)
    ):
        raise PlanningError(
            f"the fitted path carries {len(breaks) - 1} segments, but the solver "
            f"was generated for {segments}; its parameter vector is that shape"
        )
    rows, derivative, factorial = [], spline, 1.0
    for order in range(ORDER):
        if order:
            factorial *= order
            derivative = derivative.derivative()
        rows.append(np.atleast_2d(derivative(breaks[:-1])) / factorial)
    return np.stack(rows, axis=1)


def evaluate(coefficients: np.ndarray, sigma, order: int = 0) -> np.ndarray:
    """
    `c(sigma)` and its derivatives, from the coefficients the solver is handed.

    Deliberately a mirror of `path_expression` and not a call into the spline:
    what is reported back has to be what was constrained, and the two would drift
    the moment the segment lookup differed.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    segments = coefficients.shape[0]
    index = np.clip(np.floor(sigma * segments).astype(int), 0, segments - 1)
    local = sigma - index / segments
    out = np.zeros((sigma.size, coefficients.shape[2]))
    for m in range(order, ORDER):
        scale = float(np.prod([m - k for k in range(order)])) if order else 1.0
        out += scale * (local[:, None] ** (m - order)) * coefficients[index, m, :]
    return out


def path_expression(sigma, coefficients, segments: int) -> list:
    """Return `[c, c', c'']` as `casadi.SX`, coefficients left symbolic."""
    index = ca.fmin(ca.fmax(ca.floor(sigma * segments), 0.0), segments - 1)
    local = sigma - index / segments
    value = [ca.SX.zeros(cs.K_PLANNED_DOF) for _ in range(3)]
    for segment in range(segments):
        # A comparison carries no derivative in casadi, so `d/dsigma` of the sum
        # is the selected polynomial's own derivative -- which is what is wanted,
        # and correct at a breakpoint because the curve is C2 across it.
        on = index == segment
        for m in range(ORDER):
            first = (segment * ORDER + m) * cs.K_PLANNED_DOF
            block = on * coefficients[first : first + cs.K_PLANNED_DOF]
            value[0] += block if m == 0 else block * local**m
            if m >= 1:
                value[1] += m * block if m == 1 else m * block * local ** (m - 1)
            if m >= 2:
                value[2] += (
                    m * (m - 1) * block
                    if m == 2
                    else m * (m - 1) * block * local ** (m - 2)
                )
    return value


# --------------------------------------------------------------------- the OCP


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict):
    """Assemble the OCP. Returns `(ocp, scale, model)` as the sibling exporter does."""
    model = cs.CraneSymbolicModel(description_xml, TOOL)
    dq_max, tau_max = description_limits(model)
    q_sway_max = np.asarray(parameters["q_sway_max"], dtype=float)
    dq_sway_max = np.asarray(parameters["dq_sway_max"], dtype=float)
    ddq_a_max = np.asarray(parameters["ddq_a_max"], dtype=float)
    nominal = float(parameters["ocp_horizon"])
    segments = int(parameters["path_segments"])
    integrator = str(parameters["ocp_integrator"]).upper()
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"ocp_integrator is {integrator!r}, not one of {sorted(INTEGRATORS)}"
        )

    # ---------------------------------------------------------------- the model

    sigma = ca.SX.sym("sigma")
    q_u = ca.SX.sym("q_u", cs.K_PASSIVE_DOF)
    speed = ca.SX.sym("v")
    dq_u = ca.SX.sym("dq_u", cs.K_PASSIVE_DOF)
    # `theta = T / T_nominal`, one decision variable shared by every node. acados
    # fixes `p` for a solve, so a horizon the solver picks cannot live there.
    theta = ca.SX.sym("theta")
    acceleration = ca.SX.sym("a")
    #: The path, as `p`: `path_segments` fixes the knot vector, so only these
    #: numbers change between requests and the solver is generated once.
    coefficients = ca.SX.sym("c", segments * ORDER * cs.K_PLANNED_DOF)

    centre, tangent, curvature = path_expression(sigma, coefficients, segments)
    q_a = centre
    dq_a = tangent * speed
    ddq_a = curvature * speed * speed + tangent * acceleration
    sway = q_u - equilibrium(q_a)

    # Everything about the machine is `crane_model`'s, reached by substituting the
    # eliminated coordinates into its graph rather than by rebuilding it.
    planned = ca.vertcat(model.x, model.u)
    moving = ca.vertcat(q_a, q_u, dq_a, dq_u, ddq_a)
    ddq_u = ca.substitute(model.ddq_u, planned, moving)
    tau_a = ca.substitute(model.tau_a[: cs.K_PLANNED_DOF], planned, moving)
    flow = ca.substitute(
        ca.sum1(
            model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
        ),
        planned,
        moving,
    )

    acados_model = AcadosModel()
    acados_model.name = SOLVER_NAME
    acados_model.x = ca.vertcat(sigma, q_u, speed, dq_u, theta)
    acados_model.u = acceleration
    acados_model.p = ca.vertcat(model.p, coefficients)
    acados_model.f_expl_expr = ca.vertcat(
        nominal * theta * ca.vertcat(speed, dq_u, acceleration, ddq_u), 0.0
    )
    # acados does not derive one form from the other: ERK reads `f_expl_expr`,
    # IRK reads `f_impl_expr` and errors on an empty one. Both are set here
    # unconditionally so switching integrator is a solver option, not a re-model.
    acados_model.xdot = ca.SX.sym("xdot", NX)
    acados_model.f_impl_expr = acados_model.xdot - acados_model.f_expl_expr

    # ----------------------------------------------------------------- the cost

    # `tau_a` minus its value at rest on the same configuration. 91% of the raw
    # effort integral is gravity plus held load, which no timing choice changes,
    # so a residual against zero prices holding the block and is blind to what
    # shakes the boom. Substituting hanging pendulum, zero rates and zero input
    # into the same graph isolates the dynamic part exactly.
    at_rest = ca.vertcat(
        q_a,
        equilibrium(q_a),
        ca.SX.zeros(cs.K_PLANNED_DOF),
        ca.SX.zeros(cs.K_PASSIVE_DOF),
        ca.SX.zeros(cs.NU),
    )
    tau_static = ca.substitute(model.tau_a[: cs.K_PLANNED_DOF], planned, at_rest)

    # Each row over its own limit, so 1.0 means "at the bound" on every row alike.
    # No `q_a` row: the goal is the end of the curve, and tracking a position the
    # machine is on by construction only adds a stiff direction to the Hessian.
    residual = ca.vertcat(
        dq_a / dq_max,
        sway / q_sway_max,
        dq_u / dq_sway_max,
        (tau_a - tau_static) / tau_max,
        ddq_a / ddq_a_max,
    )
    terminal_residual = ca.vertcat(
        dq_a / dq_max, sway / q_sway_max, dq_u / dq_sway_max, theta
    )
    acados_model.cost_y_expr_0 = residual
    acados_model.cost_y_expr = residual
    acados_model.cost_y_expr_e = terminal_residual

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.parameter_values = np.zeros(cs.NP + segments * ORDER * cs.K_PLANNED_DOF)

    # The yaml's weights baked as defaults. Still runtime-settable -- the node
    # writes its own onto the built solver -- so a weight change needs no
    # re-export.
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

    # No cylinder-force row: it was the smaller chamber area times a relief
    # pressure nothing in this workspace has measured. No `q_a` range row either
    # -- `q_a` is on the certified curve for every `sigma` in [0, 1], and `fit`
    # already refuses a curve that leaves joint range.
    pump_max = float(hydraulics["pump_flow_max"])
    scale = np.concatenate([q_sway_max, [pump_max], dq_max, ddq_a_max])
    scale_e = np.concatenate([q_sway_max, dq_sway_max])
    acados_model.con_h_expr_0 = ca.vertcat(
        sway / q_sway_max, flow / pump_max, dq_a / dq_max, ddq_a / ddq_a_max
    )
    acados_model.con_h_expr = acados_model.con_h_expr_0
    acados_model.con_h_expr_e = ca.vertcat(sway, dq_u) / scale_e

    # Bounds below are placeholders the caller overwrites; what is baked is which
    # rows exist and which are soft. Stage 0's box is the measured state, over
    # `NX - 1` because the horizon is the one component of `x` the solver picks.
    # `constraints.x0` would pin it and freeze the objective.
    ocp.constraints.idxbx_0 = np.arange(NX - 1)
    ocp.constraints.lbx_0 = np.zeros(NX - 1)
    ocp.constraints.ubx_0 = np.zeros(NX - 1)
    for stage in ("", "_e"):
        setattr(ocp.constraints, f"idxbx{stage}", np.arange(NX))
        setattr(ocp.constraints, f"lbx{stage}", -np.ones(NX))
        setattr(ocp.constraints, f"ubx{stage}", np.ones(NX))
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = -np.full(NU, SIGMA_INPUT_MAX)
    ocp.constraints.ubu = np.full(NU, SIGMA_INPUT_MAX)
    for stage in ("_0", ""):
        setattr(ocp.constraints, f"lh{stage}", -np.ones(NH))
        setattr(ocp.constraints, f"uh{stage}", np.ones(NH))
    ocp.constraints.lh_e = -np.ones(NH_E)
    ocp.constraints.uh_e = np.ones(NH_E)

    # L1, never quadratic: a quadratic price is cheap near the boundary and leaks
    # violation everywhere, a linear one with a big enough coefficient is exact.
    # Soft rows are the ones a caller would rather have late than refused -- sway,
    # pump, settled box. Rate, input and progress stay hard: violating those is
    # not a slower plan, it is a wrong one.
    soft = np.arange(H_RATE)
    ocp.constraints.idxsh_0 = soft
    ocp.constraints.idxsh = soft
    ocp.constraints.idxsh_e = np.arange(NH_E)
    for stage, rows in (("_0", soft.size), ("", soft.size), ("_e", NH_E)):
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
    ocp.solver_options.integrator_type = integrator
    # Read for IRK only. Radau IIA is L-stable and damps the swing harder than
    # ERK4 already does, which is the wrong direction for an anti-sway problem.
    ocp.solver_options.collocation_type = "GAUSS_LEGENDRE"
    ocp.solver_options.sim_method_num_stages = INTEGRATORS[integrator]
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.levenberg_marquardt = float(parameters["levenberg_marquardt"])
    # Set, not acados' 1e-6 default, which is a control-loop number. A plan is
    # resampled onto a 25 Hz reference and tracked by a controller closing the
    # loop on it, so the last two decades buy nothing and cost plenty: a solve
    # landing at 1.69e-6 spent sixty iterations and eleven seconds failing to
    # halve it, one landing at 5.39e-7 finished in six.
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
    ddq_a: np.ndarray  # what the acceleration rows bound
    #: The path state itself, and the coefficients it indexes. `q_a` is a
    #: function of these and not an independent answer, so a consumer that has to
    #: resample must resample `sigma` and evaluate the curve -- interpolating
    #: `q_a` leaves the certified curve for the chord between two of its points.
    #: `acceleration` is the solver's own input, repeated at `N` so every array
    #: here is `N+1` long; acados holds it constant across an interval, which
    #: makes `sigma` exactly quadratic in time there.
    sigma: np.ndarray  # (N+1,)
    speed: np.ndarray  # (N+1,)
    acceleration: np.ndarray  # (N+1,)
    coefficients: np.ndarray  # (segments, ORDER, 5)
    q_u: np.ndarray  # (N+1, 2)
    dq_u: np.ndarray
    q_u_eq: np.ndarray
    pump_flow: np.ndarray  # (N+1,) as a fraction of the physical pump
    slack: float
    iterations: int
    solve_time_s: float  # wall clock around the call, so it carries Python and load
    #: acados' `time_tot` and the four KKT residuals it stopped on. Wall clock is
    #: load-bound -- the same solve measured 0.53 s idle, 4.97 s under a running
    #: Gazebo -- so read a regression off the iteration count and these residuals
    #: against `ocp_tolerance`, never off the time.
    acados_time_s: float
    residuals: np.ndarray  # (4,) stationarity, equality, inequality, complementarity

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

    Built once. acados regenerates and recompiles into `cache_tree`, outside the
    source tree -- seconds on a warm tree, minutes on a cold one. Construction-time
    work, not per-request work.
    """

    def __init__(self, description_xml: str, limits, config, weights: dict):
        self.limits, self.config = limits, config
        self.segments = int(config.path_segments)
        baked = {
            "ocp_intervals": config.ocp_intervals,
            "ocp_horizon": config.ocp_horizon,
            "ocp_integrator": config.ocp_integrator,
            "ocp_max_iterations": config.ocp_max_iterations,
            "ocp_tolerance": config.ocp_tolerance,
            "levenberg_marquardt": config.levenberg_marquardt,
            "q_sway_max": config.q_sway_max,
            "dq_sway_max": config.dq_sway_max,
            "ddq_a_max": config.ddq_a_max,
            "path_segments": config.path_segments,
        }
        hydraulics = {"pump_flow_max": config.pump_flow_max}
        ocp, self.scale, model = build_ocp(
            description_xml, {**baked, "weights": weights}, hydraulics
        )
        # The pump row in the *unreduced* coordinates, off the graph the solver
        # constrains, so reported and bounded cannot drift apart.
        flow = ca.sum1(
            model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
        )
        self._flow = ca.Function("flow", [model.x, model.u, model.p], [flow])
        self.N = ocp.solver_options.N_horizon
        self.nominal = float(config.ocp_horizon)
        # **Generated by `scripts/export_timing_ocp.py`, not here.** Loads a built
        # tree; builds one on first construction, costing minutes. Regenerating
        # unconditionally would overwrite the exporter's output and make `--check`
        # meaningless.
        tree = cache_tree(baked, hydraulics)
        library = tree / f"libacados_ocp_solver_{SOLVER_NAME}.so"
        fresh = not library.is_file()
        CACHE.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(tree)
        self.solver = AcadosOcpSolver(
            ocp,
            json_file=str(tree.with_suffix(".json")),
            generate=fresh,
            build=fresh,
            verbose=False,
        )
        # `W` reaches the generated code only at export time, and a cached `.so`
        # is *loaded*, not regenerated -- so without this the weights a caller
        # passes are inert on every run but the one that built the tree, and
        # `BAKED` excludes them so nothing invalidates the cache either. Swept
        # `weights.time` over 0.3 to 10 on a warm cache and got durations equal
        # to the centisecond; rebuilt cold at 10, two moves that had refused
        # converged and durations fell 17-18%. Writing it here is what the
        # `BAKED` docstring already claims happens.
        stage_w, terminal_w = w.matrices(weights)
        for node in range(self.N):
            self.solver.cost_set(node, "W", stage_w)
        self.solver.cost_set(self.N, "W", terminal_w)

    def start_speed(self, coefficients: np.ndarray, dq_a_start) -> tuple:
        """
        Return `v(0)` from the measured start velocity, and how much is lost.

        A motion confined to the path can only leave along its tangent, so a
        start velocity off it is not representable at all. The projection is the
        best available answer and the residual says how wrong it is; `solve`
        refuses rather than start from a state the machine is not in.
        """
        tangent = evaluate(coefficients, 0.0, order=1)[0]
        dq_a_start = np.asarray(dq_a_start, dtype=float)
        squared = float(tangent @ tangent)
        if squared <= 0.0:
            return 0.0, float(np.max(np.abs(dq_a_start)))
        speed = float(tangent @ dq_a_start) / squared
        return speed, float(np.max(np.abs(tangent * speed - dq_a_start)))

    def solve(
        self,
        coefficients: np.ndarray,
        q_u_start: np.ndarray,
        dq_a_start: np.ndarray,
        dq_u_start: np.ndarray,
        payload: np.ndarray,
        q_tool: float,
        speed_scale: float,
    ) -> Trajectory:
        """Solve one request, or refuse with what the solver said."""
        cfg, lim, N = self.config, self.limits, self.N
        solver, big = self.solver, 1.0e6
        rate_hi = cfg.kappa * speed_scale
        input_hi = cfg.kappa * speed_scale**2
        flow_hi = cfg.kappa * speed_scale * lim.flow_max / cfg.pump_flow_max
        theta = np.array([cfg.ocp_duration_min, cfg.ocp_duration_max]) / self.nominal

        speed_start, unrepresentable = self.start_speed(coefficients, dq_a_start)
        # 1e-6 rad/s is measurement noise; above it the path's tangent is not the
        # direction the machine is already going, and projecting silently would
        # put the solve on a state it is not in.
        if unrepresentable > 1.0e-6:
            raise PlanningError(
                f"the measured start velocity is {unrepresentable:.3e} rad/s off "
                "the path tangent, and a motion confined to the path cannot leave "
                "in any other direction"
            )

        # `sigma` in [0, 1] and `v >= 0`: the machine may not run backwards along
        # its own curve, which is the one thing the certificate says nothing
        # about.
        lbx = np.concatenate([[0.0], [-big, -big], [0.0], -cfg.dq_sway_max, theta[:1]])
        ubx = np.concatenate([[1.0], [big, big], [big], cfg.dq_sway_max, theta[1:]])
        x0 = np.concatenate([[0.0], q_u_start, [speed_start], dq_u_start])
        planned = cs.K_PLANNED_DOF
        lh = np.concatenate(
            [
                -np.ones(2),
                [0.0],
                np.full(planned, -rate_hi),
                np.full(planned, -input_hi),
            ]
        )
        uh = np.concatenate(
            [
                np.ones(2),
                [flow_hi],
                np.full(planned, rate_hi),
                np.full(planned, input_hi),
            ]
        )
        settled = np.concatenate(
            [
                cfg.terminal_q_sway_max / cfg.q_sway_max,
                cfg.terminal_dq_sway_max / cfg.dq_sway_max,
            ]
        )
        parameters = np.concatenate(
            [[q_tool], payload, np.asarray(coefficients, dtype=float).reshape(-1)]
        )
        price = float(cfg.ocp_slack_price)
        soft = H_RATE

        # Uniform progress with the pendulum hanging: no rollout and nothing that
        # knows the answer, which is all this problem has needed.
        solver.reset()
        for node in range(N + 1):
            sigma = node / N
            guess = evaluate(coefficients, sigma)[0]
            solver.set(node, "p", parameters)
            solver.set(
                node,
                "x",
                np.concatenate(
                    [
                        [sigma],
                        passive_equilibrium(guess),
                        [1.0 / self.nominal],
                        np.zeros(cs.K_PASSIVE_DOF),
                        [1.0],
                    ]
                ),
            )
            if node < N:
                solver.set(node, "u", np.zeros(NU))
                solver.constraints_set(node, "lh", lh)
                solver.constraints_set(node, "uh", uh)
                solver.cost_set(node, "zl", np.full(soft, price))
                solver.cost_set(node, "zu", np.full(soft, price))
            if 0 < node < N:
                solver.constraints_set(node, "lbx", lbx)
                solver.constraints_set(node, "ubx", ubx)
        solver.constraints_set(0, "lbx", x0)
        solver.constraints_set(0, "ubx", x0)
        # The goal is the end of the curve, arrived at stopped. The tool may hang
        # where it likes; how still it has to be is `h_e`, softly.
        solver.constraints_set(
            N,
            "lbx",
            np.concatenate([[1.0], [-big, -big], [0.0], np.zeros(2), theta[:1]]),
        )
        solver.constraints_set(
            N,
            "ubx",
            np.concatenate([[1.0], [big, big], [0.0], np.zeros(2), theta[1:]]),
        )
        solver.constraints_set(N, "lh", -settled)
        solver.constraints_set(N, "uh", settled)
        solver.cost_set(N, "zl", np.full(NH_E, price))
        solver.cost_set(N, "zu", np.full(NH_E, price))

        started = time.monotonic()
        status = solver.solve()
        elapsed = time.monotonic() - started

        state = np.array([solver.get(node, "x") for node in range(N + 1)])
        control = np.array([solver.get(node, "u") for node in range(N)])
        control = np.vstack([control, control[-1]])
        guard = float(np.max(np.abs(control))) / SIGMA_INPUT_MAX
        duration = float(state[0, X_HORIZON]) * self.nominal
        sigma = state[:, X_SIGMA]
        speed = state[:, X_SPEED]
        q_a = evaluate(coefficients, sigma)
        tangent = evaluate(coefficients, sigma, order=1)
        curvature = evaluate(coefficients, sigma, order=2)
        dq_a = tangent * speed[:, None]
        ddq_a = curvature * speed[:, None] ** 2 + tangent * control
        q_u = state[:, X_PASSIVE : X_PASSIVE + cs.K_PASSIVE_DOF]
        dq_u = state[:, X_PASSIVE_RATE : X_PASSIVE_RATE + cs.K_PASSIVE_DOF]
        slack = max(
            float(np.max(solver.get(node, "sl"), initial=0.0)) for node in range(N + 1)
        )
        flow = np.array(
            [
                float(
                    self._flow(
                        np.concatenate([q_a[node], q_u[node], dq_a[node], dq_u[node]]),
                        ddq_a[node],
                        parameters[: cs.NP],
                    )
                )
                / cfg.pump_flow_max
                for node in range(N + 1)
            ]
        )
        if guard > 0.99:
            raise PlanningError(
                f"the sigma'' guard bound at {guard:.2f} of {SIGMA_INPUT_MAX:g}: the "
                "duration this answer carries is partly that constant's and not the "
                "machine's"
            )
        if status != 0:
            # The residuals are the whole diagnosis of a non-convergence -- which
            # of the four stalled says whether it is the dynamics, the goal box or
            # a soft row -- and this is the one path where no `Trajectory` is
            # built to carry them, so they go in the message instead.
            stat, eq, ineq, comp = np.asarray(
                solver.get_stats("residuals"), dtype=float
            )
            raise PlanningError(
                f"the trajectory OCP did not converge: acados status {status} after "
                f"{solver.get_stats('sqp_iter')} iterations, {elapsed:.2f} s, on "
                f"residuals stat {stat:.1e} eq {eq:.1e} ineq {ineq:.1e} comp {comp:.1e}"
            )
        return Trajectory(
            time=np.linspace(0.0, duration, N + 1),
            q_a=q_a,
            dq_a=dq_a,
            ddq_a=ddq_a,
            sigma=sigma,
            speed=speed,
            acceleration=control[:, 0],
            coefficients=coefficients,
            q_u=q_u,
            dq_u=dq_u,
            q_u_eq=np.array([passive_equilibrium(row) for row in q_a]),
            pump_flow=flow,
            slack=slack,
            iterations=int(solver.get_stats("sqp_iter")),
            solve_time_s=elapsed,
            acados_time_s=float(solver.get_stats("time_tot")),
            residuals=np.asarray(solver.get_stats("residuals"), dtype=float),
        )
