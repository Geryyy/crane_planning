"""
Trajectory OCP: definition + solver.

`x = [sigma, v, a, j, q_u, dq_u, theta]`, `u = s`, time normalised to [0, 1].
`theta = T / ocp_horizon`: one solve is time-optimal. `q_a` not decided here:
geometric stage hands over certified curve `c(sigma)`, this only picks speed:
    q_a   = c(sigma)
    dq_a  = c'(sigma) v
    ddq_a = c''(sigma) v^2 + c'(sigma) a
`sigma` integrated 4x (input snap): `q_a(t)` is C4, what C3's flat inversion
needs. Executed curve is the certified curve: clearance proof holds verbatim.
Each residual row / its limit, 1.0 is the bound, `W` preference alone -- one row
in physical units wrecks Gauss-Newton conditioning (stalled at stationarity 1e6;
horizon in seconds cost 40-60x).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
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

#: What `scripts/export_timing_ocp.py` writes and `--check` reviews.
GENERATED = Path(__file__).resolve().parent.parent / "generated"

#: Build dir, outside source tree: stops a run overwriting the artifact under review.
CACHE = (
    Path(os.environ.get("CRANE_PLANNING_OCP_CACHE", tempfile.gettempdir()))
    / "crane_planning_ocp"
)

TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"
SOLVER_NAME = f"crane_planning_ocp_{TOOL}"
GENERATED_HEADER = "crane_planning_ocp_generated.h"

#: Built solver's `cache_key`, written beside it; names what diverged on a miss.
MANIFEST = "cache_key.json"

#: Two hinges hanging pose is closed form in; `crane_model.symbolic` numbers, doesn't name.
BOOM_AXIS = 1
ARM_AXIS = 2

#: Row order of `x`: `sigma` chain, passive pair, then `theta`.
X_SIGMA, X_SPEED, X_ACCEL, X_JERK = 0, 1, 2, 3
X_PASSIVE, X_PASSIVE_RATE, X_HORIZON = 4, 6, 8
NX, NU = 9, 1

#: Row order of `h`; caller reads it back, must not re-derive.
H_SWAY, H_FLOW, H_RATE = 0, 2, 3
NH_SWAY = 2  # rows sway box occupies, stage and terminal alike
H_ACCEL = 3 + cs.K_PLANNED_DOF
#: C3 command, `u = dq_a + tau_dot_a / k`; not `dddq_a` (jerk alone priced a sum, not the worst case
# the machine pays).
H_COMMAND = 3 + 2 * cs.K_PLANNED_DOF
NH = 3 + 3 * cs.K_PLANNED_DOF
#: Row order of `h_e`: settled box caller accepts.
HE_SWAY, HE_SWAY_RATE, NH_E = 0, 2, 4

#: Quintic path: degree 5, simple interior knots -> `C4` in `sigma` by construction.
ORDER = 6

#: Derivative levels built off path: `c` through its fourth, what `q_a(t) = c(sigma(t))` needs.
PATH_DERIVATIVES = 5

#: Numerical guard on input, not physical; `solve` refuses if it binds. Swept 20/200/2000/20000 at
# `speed_scale` 0.9: binding stops at 200, above it agrees to the hundredth (`max|s|`
# 24.5/131.8/51.3/41.2, demand ~132). `H_COMMAND` carries the physical row, this bounds only snap,
# loosely.
SIGMA_INPUT_MAX = 2000.0

#: What the compiled solver's structure depends on; `weights` absent (runtime field).
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
    "dddq_a_max",
    "command_k",
    "command_u_min",
    "command_u_max",
    "path_segments",
)

#: acados settings being tuned. Every one is compiled into the `.so`, so they go through
#: `cache_key` -- without that a variant loads its predecessor's solver and reads as a null result.
#: Separate from `BAKED` (config keys, the problem) because these are the solver, not the problem:
#: none of them changes what an answer means, only whether it is found and how fast.
#: Sweep without editing this file:
#:     CRANE_PLANNING_OCP_OPTIONS='{"hpipm_mode": "SPEED"}' ./scripts/bench_ocp.py
SOLVER_TUNING = {
    # Not plain SQP: 8 of the shipped solver's 11 bench refusals died in the *first* QP, 10 ms in,
    # stationarity still at its initial 1e3 -- an infeasible QP, not a hard problem. Byrd-Omojokun
    # solves a relaxed feasibility QP there instead of giving up. 11 refusals -> 3 on its own.
    "nlp_solver_type": "SQP_WITH_FEASIBLE_QP",
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hpipm_mode": "BALANCE",
    "qp_solver_cond_N": None,  # acados' own default is `N_horizon`, i.e. no partial condensing
    "qp_solver_iter_max": 50,
    "qp_solver_warm_start": 0,
    "qp_solver_ric_alg": 1,
    "globalization": "MERIT_BACKTRACKING",
    "globalization_use_SOC": 0,
    "regularize_method": "NO_REGULARIZE",
    # `levenberg_marquardt` alone is 1e-6, i.e. none; this starts at 1e-3 and relaxes, so the first
    # QP is regularised where it needs to be and later ones do not pay for it. Worth a third of the
    # iterations by itself, and the last refusal the feasibility QP does not reach.
    "with_adaptive_levenberg_marquardt": True,
    "nlp_solver_warm_start_first_qp": False,
    "search_direction_mode": "BYRD_OMOJOKUN",
    "qpscaling_scale_constraints": "NO_CONSTRAINT_SCALING",
    "qpscaling_scale_objective": "NO_OBJECTIVE_SCALING",
    "nlp_qp_tol_strategy": "FIXED_QP_TOL",
}


def solver_tuning() -> dict:
    """`SOLVER_TUNING` with `CRANE_PLANNING_OCP_OPTIONS` applied. Unknown key is an error, not a typo silently ignored -- a sweep that misspells a knob would otherwise re-measure the baseline."""
    override = json.loads(os.environ.get("CRANE_PLANNING_OCP_OPTIONS") or "{}")
    unknown = set(override) - set(SOLVER_TUNING)
    if unknown:
        raise ValueError(
            f"CRANE_PLANNING_OCP_OPTIONS names {sorted(unknown)}, which "
            f"`SOLVER_TUNING` does not carry; known keys are {sorted(SOLVER_TUNING)}"
        )
    return {**SOLVER_TUNING, **override}


#: ERK order 4, Gauss-Legendre IRK order 2. ERK4 cheaper but finite stability region, and `theta`
# scales the step: at `ocp_duration_max` pendulum (w=3.6 rad/s) sits at w*dt=1.8, RK4 amplifies
# 0.854/step -- 500x fabricated damping over 40 intervals. Gauss-Legendre symplectic, cannot invent
# that; `SIM_SUBSTEPS` keeps ERK honest.
INTEGRATORS = {"ERK": 4, "IRK": 2}

#: Substeps/interval, not a knob: one RK4 step at w*dt=1.43 amplified 0.955 -- 0.16 over 40
# intervals, 84% of an early swing looked vanished by the goal. Three substeps: 16s at 0.991, 20s at
# 0.965, quartic reconstruction stays exact. Over 21 bench moves no duration moved >0.25%.
SIM_SUBSTEPS = 3

#: Max ratio of interior speed to fitted start tangent. Fit leaves along measured start velocity, so
# noise-floor velocity pins `c'(0)` at noise and tangent grows two orders inside one knot span --
# first QP dies (acados status 4, iter 1). Measured live: 179x/228x on failed noise starts,
# 0.9x/1.2x from rest; 10 clears both with an order to spare.
TANGENT_STEP_MAX = 10.0


def equilibrium(q_a):
    return ca.vertcat(0.5 * np.pi - q_a[BOOM_AXIS] - q_a[ARM_AXIS], 0.5 * np.pi)


def description_limits(model) -> tuple:
    """`(dq_max, tau_max)`."""
    inner = model.description.model
    rows = [model.description.joints[row].velocity_index for row in cs.K_PLANNED_ROWS]
    return (
        np.array([float(inner.velocityLimit[row]) for row in rows]),
        np.array([float(inner.effortLimit[row]) for row in rows]),
    )


def baked_parameters(config) -> dict:
    return {name: getattr(config, name) for name in BAKED}


def cache_key(parameters: dict, hydraulics: dict, description: str) -> dict:
    """Solver validity key, tree named after it; `.so` loaded not compared, so anything entering expressions/dimensions must move the directory (sim uses a different xacro at `sim_hydraulics:=false`, so description is hashed here, not assumed)."""
    baked = {}
    for key in BAKED:
        value = parameters[key]
        baked[key] = value.tolist() if hasattr(value, "tolist") else value
    baked["pump_flow_max"] = float(hydraulics["pump_flow_max"])
    baked["path_order"] = ORDER  # sets parameter vector width
    baked["nx"] = NX  # solver shape, not a parameter
    baked["sim_substeps"] = SIM_SUBSTEPS  # compiled into the integrator
    # `INTEGRATORS` decides the order; editing the map alone would load the old one.
    baked["sim_stages"] = INTEGRATORS[parameters["ocp_integrator"]]
    baked["path_derivatives"] = PATH_DERIVATIVES
    # Solver settings are generated into the C, so a tuning change must move the tree too.
    baked["tuning"] = solver_tuning()
    baked["description"] = hashlib.sha1(description.encode()).hexdigest()
    return baked


def tree_of(key: dict) -> Path:
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return CACHE / f"{SOLVER_NAME}_{digest[:10]}"


def cache_tree(parameters: dict, hydraulics: dict, description: str) -> Path:
    return tree_of(cache_key(parameters, hydraulics, description))


def write_manifest(tree: Path, key: dict) -> None:
    (tree / MANIFEST).write_text(json.dumps(key, sort_keys=True, indent=1))


class SolverNotExported(RuntimeError):
    pass


#: Four KKT residuals acados stops on, in the order it returns them.
RESIDUALS = ("stationarity", "equality", "inequality", "complementarity")

#: Above this, solve paid L1 price instead of soft bound. 1e-6 not 1e-9: two converged bench solves
# report 1.3e-9/1.6e-8 slack with every soft bound met -- QP's own floor, 3-5 orders under
# `ocp_tolerance`.
SLACK_SPENT = 1e-6

#: Answer's cost vs how hard it was to find; `nan` when no `Trajectory` was built, so a consumer
# reads one key set regardless of outcome.
PLAN_ROWS = (
    "slack",
    "sway_slack",
    "pump_flow_peak",
    "terminal_sway",
    "terminal_sway_rate",
    "duration_s",
)


def solver_stats(solver, status: int, elapsed: float) -> dict:
    """One solve as flat named numbers for `~/solver_stats`, built once before the outcome branches (regress on iteration counts/residuals, never time -- load-bound); QP status is its own key since acados reports a QP that merely hit HPIPM's limit as success."""
    qp_iterations = np.asarray(solver.get_stats("qp_iter"), dtype=float)
    qp_status = np.asarray(solver.get_stats("qp_stat"), dtype=float)
    stats = {
        "acados_status": float(status),
        "sqp_iterations": float(solver.get_stats("sqp_iter")),
        "qp_iterations": float(qp_iterations.sum()),
        "qp_status_worst": float(qp_status.max()) if qp_status.size else 0.0,
        "solve_time_s": float(elapsed),
        "acados_time_s": float(solver.get_stats("time_tot")),
        "qp_time_s": float(solver.get_stats("time_qp")),
    }
    stats.update(
        {
            f"residual_{name}": float(value)
            for name, value in zip(RESIDUALS, solver.get_stats("residuals"))
        }
    )
    stats.update({name: float("nan") for name in PLAN_ROWS})
    return stats


def slack_spent(solver, N: int) -> tuple:
    """Worst soft-row slack, and sway's alone; both sides checked (`sl` alone misses upper-bound overshoot). Excludes node `N`, whose sway rows carry the settled box, not the envelope."""
    slack, sway_slack = 0.0, 0.0
    for node in range(N + 1):
        for side in ("sl", "su"):
            value = np.asarray(solver.get(node, side), dtype=float).ravel()
            slack = max(slack, float(np.max(value, initial=0.0)))
            if node < N:
                sway_slack = max(
                    sway_slack, float(np.max(value[:NH_SWAY], initial=0.0))
                )
    return slack, sway_slack


def divergence(key: dict) -> list:
    """Which baked names separate `key` from each cached solver: hash says a prebake was missed, this says what -- "recompiling, three minutes" vs "planning for an unexported machine"."""
    lines = []
    for manifest in sorted(CACHE.glob(f"{SOLVER_NAME}_*/{MANIFEST}")):
        try:
            other = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        differ = sorted(
            name for name in set(key) | set(other) if key.get(name) != other.get(name)
        )
        lines.append(f"  {manifest.parent.name}: differs in {', '.join(differ)}")
    return lines


def power_coefficients(path, segments: int) -> np.ndarray:
    """`(segments, ORDER, 5)`: `c(sigma) = sum_m a[k, m] (sigma - k/S)^m`. Power basis on uniform breakpoints, not B-spline, so the solver's segment lookup is a `floor`, not a knot search."""
    spline = path.spline
    breaks = np.unique(spline.t)
    if len(breaks) - 1 != segments or not np.allclose(
        breaks, np.linspace(0.0, 1.0, segments + 1)
    ):
        raise PlanningError(
            f"the fitted path carries {len(breaks) - 1} segments, but the solver "
            f"was generated for {segments}; its parameter vector is that shape"
        )
    # degree must agree with `ORDER`; a short power basis truncates silently, doesn't refuse.
    if spline.k != ORDER - 1:
        raise PlanningError(
            f"the fitted path is degree {spline.k}, but the solver carries "
            f"{ORDER} coefficients per segment, which is degree {ORDER - 1}"
        )
    rows, derivative, factorial = [], spline, 1.0
    for order in range(ORDER):
        if order:
            factorial *= order
            derivative = derivative.derivative()
        rows.append(np.atleast_2d(derivative(breaks[:-1])) / factorial)
    return np.stack(rows, axis=1)


def evaluate(coefficients: np.ndarray, sigma, order: int = 0) -> np.ndarray:
    """`c(sigma)` and derivatives, from the coefficients the solver holds; mirrors `path_expression` (not the spline) so reported equals constrained."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    segments = coefficients.shape[0]
    index = np.clip(np.floor(sigma * segments).astype(int), 0, segments - 1)
    local = sigma - index / segments
    out = np.zeros((sigma.size, coefficients.shape[2]))
    for m in range(order, ORDER):
        scale = float(np.prod([m - k for k in range(order)])) if order else 1.0
        out += scale * (local[:, None] ** (m - order)) * coefficients[index, m, :]
    return out


#: Initial guess for the `sigma` chain. "CONSTANT" is the shipped one: uniform progress at
#: `ocp_horizon / speed_scale`, i.e. the same 7 s whatever was asked. "PATH_PROFILE" guesses the
#: fastest profile the rate and acceleration rows alone allow. Read here, not compiled in, so it
#: stays out of `cache_key` -- a variant of it reuses the solver.
INITIAL_GUESS = os.environ.get("CRANE_PLANNING_OCP_INIT", "CONSTANT")

#: Tangent below which a row constrains `s = v^2` directly instead of its slope (units of the
#: normalised curve, so this is "that joint does not move here").
TANGENT_EPS = 1.0e-9


def path_profile(coefficients, dq_max, ddq_a_max, speed_start, samples: int):
    """Fastest `(sigma, v)` the rate and acceleration rows alone allow on this curve: TOPP's velocity limit curve, then a forward and a backward pass in `s = v^2`. Sway, pump and the command row are not in it, so the duration this implies is a floor and the profile is a shape to start from -- never an answer."""
    sigma = np.linspace(0.0, 1.0, samples)
    step = 1.0 / (samples - 1)
    tangent = evaluate(coefficients, sigma, order=1)
    curvature = evaluate(coefficients, sigma, order=2)
    moving = np.abs(tangent) > TANGENT_EPS

    # Rate rows cap `v` outright. Where a joint is still, its acceleration row caps `s` instead --
    # `ddq_a = c'' v^2 + c' a` has no `a` in it there.
    ceiling = np.where(
        moving, dq_max / np.maximum(np.abs(tangent), TANGENT_EPS), np.inf
    )
    still = ~moving & (np.abs(curvature) > TANGENT_EPS)
    s_cap = np.where(
        still, ddq_a_max / np.maximum(np.abs(curvature), TANGENT_EPS), np.inf
    )
    limit = np.minimum(np.min(ceiling, axis=1) ** 2, np.min(s_cap, axis=1))

    def slopes(node: int, s: float) -> tuple:
        """Admissible `ds/dsigma` at `node`, from the acceleration rows at speed `s`."""
        rows = moving[node]
        if not rows.any():
            return -np.inf, np.inf
        c1, c2 = tangent[node][rows], curvature[node][rows]
        edge = 2.0 * (np.stack([-ddq_a_max[rows], ddq_a_max[rows]]) - c2 * s) / c1
        return float(np.max(np.min(edge, axis=0))), float(np.min(np.max(edge, axis=0)))

    s = np.minimum(limit, np.inf)
    s[0] = min(limit[0], speed_start**2)
    for node in range(samples - 1):  # forward: accelerate as hard as the rows allow
        s[node + 1] = min(s[node + 1], s[node] + step * slopes(node, s[node])[1])
    s[-1] = 0.0  # arrives stopped, which is what the terminal box asks for
    for node in range(samples - 2, -1, -1):  # backward: and brake in time
        s[node] = min(s[node], s[node + 1] - step * slopes(node + 1, s[node + 1])[0])
    s = np.maximum(s, 0.0)

    speed = np.sqrt(s)
    pair = speed[:-1] + speed[1:]
    duration = float(
        np.sum(
            np.where(
                pair > TANGENT_EPS, 2.0 * step / np.maximum(pair, TANGENT_EPS), 0.0
            )
        )
    )
    return sigma, speed, duration


#: What the profile's duration is multiplied by to guess the answer's. The profile prices rate and
#: acceleration and nothing else, so the answer is always slower; set from the measured ratio.
PROFILE_SLACK = float(os.environ.get("CRANE_PLANNING_OCP_INIT_SLACK", "1.0"))


def profile_guess(
    coefficients, dq_max, ddq_a_max, speed_start, nodes: int, samples: int = 201
):
    """`(sigma, v, a, duration)` on the solver's grid, which is uniform in *time*, not in `sigma`. The shipped guess is uniform in `sigma`, which on a curve that accelerates and brakes is wrong at both ends. `None` when the profile says the curve cannot be traversed at all."""
    sigma_of, speed, duration = path_profile(
        coefficients, dq_max, ddq_a_max, speed_start, samples
    )
    if not np.isfinite(duration) or duration <= 0.0:
        return None
    step = 1.0 / (samples - 1)
    pair = speed[:-1] + speed[1:]
    elapsed = np.concatenate(
        [
            [0.0],
            np.cumsum(
                np.where(
                    pair > TANGENT_EPS, 2.0 * step / np.maximum(pair, TANGENT_EPS), 0.0
                )
            ),
        ]
    )
    # Stretching time by `PROFILE_SLACK` leaves the shape alone: `v` scales by its inverse, `a` by
    # its square, and the normalised grid does not move.
    duration = duration * PROFILE_SLACK
    times = np.linspace(0.0, elapsed[-1], nodes)
    sigma = np.interp(times, elapsed, sigma_of)
    v = np.interp(times, elapsed, speed) / PROFILE_SLACK
    a = np.gradient(v, times / PROFILE_SLACK, edge_order=1)
    return sigma, v, a, duration


def path_expression(sigma, coefficients, segments: int) -> list:
    """`PATH_DERIVATIVES` levels of the path as `casadi.SX`; symbolic mirror of `evaluate`, must agree at every level."""
    index = ca.fmin(ca.fmax(ca.floor(sigma * segments), 0.0), segments - 1)
    local = sigma - index / segments
    value = [ca.SX.zeros(cs.K_PLANNED_DOF) for _ in range(PATH_DERIVATIVES)]
    for segment in range(segments):
        # comparison carries no derivative in casadi, so d/dsigma is the selected polynomial's own
        # -- correct at a breakpoint since C4 there.
        on = index == segment
        for m in range(ORDER):
            first = (segment * ORDER + m) * cs.K_PLANNED_DOF
            block = on * coefficients[first : first + cs.K_PLANNED_DOF]
            for order in range(min(PATH_DERIVATIVES, m + 1)):
                scale = float(np.prod([m - k for k in range(order)]))
                power = m - order
                value[order] += (
                    scale * block if power == 0 else scale * block * local**power
                )
    return value


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict):
    """Return `(ocp, scale, model)`, as the sibling exporter does."""
    model = cs.CraneSymbolicModel(description_xml, TOOL)
    dq_max, tau_max = description_limits(model)
    q_sway_max = np.asarray(parameters["q_sway_max"], dtype=float)
    dq_sway_max = np.asarray(parameters["dq_sway_max"], dtype=float)
    ddq_a_max = np.asarray(parameters["ddq_a_max"], dtype=float)
    dddq_a_max = np.asarray(parameters["dddq_a_max"], dtype=float)
    command_k = np.asarray(parameters["command_k"], dtype=float)
    command_u_min = np.asarray(parameters["command_u_min"], dtype=float)
    command_u_max = np.asarray(parameters["command_u_max"], dtype=float)
    # 1.0 is the wider side of an asymmetric domain.
    command_ref = np.maximum(np.abs(command_u_min), command_u_max)
    nominal = float(parameters["ocp_horizon"])
    segments = int(parameters["path_segments"])
    integrator = str(parameters["ocp_integrator"]).upper()
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"ocp_integrator is {integrator!r}, not one of {sorted(INTEGRATORS)}"
        )

    sigma = ca.SX.sym("sigma")
    q_u = ca.SX.sym("q_u", cs.K_PASSIVE_DOF)
    speed = ca.SX.sym("v")
    dq_u = ca.SX.sym("dq_u", cs.K_PASSIVE_DOF)
    # `theta = T / T_nominal`: acados fixes `p` for a solve, horizon the solver picks cannot live
    # there.
    theta = ca.SX.sym("theta")
    # Chain: `a`, `j` states, `s` input; holding snap constant leaves C4.
    acceleration = ca.SX.sym("a")
    jerk = ca.SX.sym("j")
    snap = ca.SX.sym("s")
    #: `path_segments` fixes the knot vector: only these numbers change between requests, solver
    # generated once.
    coefficients = ca.SX.sym("c", segments * ORDER * cs.K_PLANNED_DOF)

    q_a, tangent, curvature, third = path_expression(sigma, coefficients, segments)[:4]
    # Faa di Bruno on `q_a(t) = c(sigma(t))`, physical seconds since `v` = d sigma/dt. Snap not
    # built: jerk bound + input box already bounds `q_a''''`.
    dq_a = tangent * speed
    ddq_a = curvature * speed * speed + tangent * acceleration
    dddq_a = third * speed**3 + 3.0 * curvature * speed * acceleration + tangent * jerk
    sway = q_u - equilibrium(q_a)

    # Substitute the eliminated coordinates into `crane_model`'s graph rather than rebuild it.
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
    acados_model.x = ca.vertcat(sigma, speed, acceleration, jerk, q_u, dq_u, theta)
    acados_model.u = snap
    acados_model.p = ca.vertcat(model.p, coefficients)
    acados_model.f_expl_expr = ca.vertcat(
        nominal * theta * ca.vertcat(speed, acceleration, jerk, snap, dq_u, ddq_u),
        0.0,
    )
    # acados derives neither form from the other: ERK reads `f_expl_expr`, IRK reads `f_impl_expr`
    # (errors if empty); both set so switching integrator is a solver option, not a re-model.
    acados_model.xdot = ca.SX.sym("xdot", NX)
    acados_model.f_impl_expr = acados_model.xdot - acados_model.f_expl_expr

    # `u_d = qdot_d + tau_dot_d / k_i`: `jtimes` gives exact total derivative incl. passive
    # coupling. `command_k` identified against plain CRBA vs `tau_a`'s mimic+transmission graph --
    # differ 1.09x-3.40x, pose-dependent, so dividing this `tau` by this `k` matches refused to what
    # the machine sees. `f_expl_expr` carries `nominal * theta`; divide it back out or the row
    # tightens as the answer lengthens.
    tau_dot_a = ca.jtimes(tau_a, acados_model.x, acados_model.f_expl_expr) / (
        nominal * theta
    )
    command = dq_a + tau_dot_a / command_k
    model.command_function = ca.Function(
        "command", [acados_model.x, acados_model.p], [command]
    )

    # `tau_a` minus value at rest: 91% of raw effort is gravity+held load; residual against zero
    # isolates the dynamic part (hanging, zero rate/input).
    at_rest = ca.vertcat(
        q_a,
        equilibrium(q_a),
        ca.SX.zeros(cs.K_PLANNED_DOF),
        ca.SX.zeros(cs.K_PASSIVE_DOF),
        ca.SX.zeros(cs.NU),
    )
    tau_static = ca.substitute(model.tau_a[: cs.K_PLANNED_DOF], planned, at_rest)

    # No `q_a` row: tracking a position the machine is on by construction only adds a stiff Hessian
    # direction.
    residual = ca.vertcat(
        dq_a / dq_max,
        sway / q_sway_max,
        dq_u / dq_sway_max,
        (tau_a - tau_static) / tau_max,
        ddq_a / ddq_a_max,
        dddq_a / dddq_a_max,
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

    # Yaml weights baked as defaults, still runtime-settable so a weight change needs no re-export.
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

    # No cylinder-force row: relief pressure unmeasured. No `q_a` range row: on the certified curve,
    # `fit` already refuses a curve leaving joint range.
    pump_max = float(hydraulics["pump_flow_max"])
    scale = np.concatenate([q_sway_max, [pump_max], dq_max, ddq_a_max, command_ref])
    scale_e = np.concatenate([q_sway_max, dq_sway_max])
    acados_model.con_h_expr_0 = ca.vertcat(
        sway / q_sway_max,
        flow / pump_max,
        dq_a / dq_max,
        ddq_a / ddq_a_max,
        command / command_ref,
    )
    acados_model.con_h_expr = acados_model.con_h_expr_0
    acados_model.con_h_expr_e = ca.vertcat(sway, dq_u) / scale_e

    # Placeholders the caller overwrites; baked is which rows exist and are soft. Stage 0's box
    # excludes theta (`NX - 1`); `constraints.x0` would pin it and freeze the objective.
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

    # L1 never quadratic: quadratic leaks violation near the boundary, linear with a big coefficient
    # is exact. Soft: sway, pump, settled box. Hard: rate/input/progress -- violating those is a
    # wrong plan, not a slower one.
    soft = np.arange(H_RATE)
    ocp.constraints.idxsh_0 = soft
    ocp.constraints.idxsh = soft
    ocp.constraints.idxsh_e = np.arange(NH_E)
    for stage, rows in (("_0", soft.size), ("", soft.size), ("_e", NH_E)):
        setattr(ocp.cost, f"Zl{stage}", np.zeros(rows))
        setattr(ocp.cost, f"Zu{stage}", np.zeros(rows))
        setattr(ocp.cost, f"zl{stage}", np.ones(rows))
        setattr(ocp.cost, f"zu{stage}", np.ones(rows))

    # Normalised time: grid is [0, 1], horizon is `theta * T_nominal`.
    ocp.solver_options.N_horizon = int(parameters["ocp_intervals"])
    ocp.solver_options.tf = 1.0
    ocp.solver_options.nlp_solver_max_iter = int(parameters["ocp_max_iterations"])
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = integrator
    # Radau IIA (IRK-only default) is L-stable, damps the swing harder than ERK4 -- wrong direction
    # for anti-sway.
    ocp.solver_options.collocation_type = "GAUSS_LEGENDRE"
    ocp.solver_options.sim_method_num_stages = INTEGRATORS[integrator]
    ocp.solver_options.sim_method_num_steps = SIM_SUBSTEPS
    ocp.solver_options.levenberg_marquardt = float(parameters["levenberg_marquardt"])
    # Last, so a tuning key beats anything set above it and the tree hash tells the whole story.
    # `None` is "leave acados' own default", which some keys (`qp_solver_cond_N`) refuse to be set to.
    for name, value in solver_tuning().items():
        if value is not None:
            setattr(ocp.solver_options, name, value)
    # Not acados' 1e-6 default: plan is resampled onto a 25 Hz reference and tracked closed-loop, so
    # last two decades buy nothing -- a solve at 1.69e-6 spent 60 iter/11s failing to halve it, one
    # at 5.39e-7 finished in six.
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
    dddq_a: np.ndarray  # what the C3 inversion pays for, third term
    #: `u_d = dq_a + tau_dot_a / k` per axis, what `AddC3Feedforward` will command.
    command: np.ndarray
    #: `q_a` is a function of these coefficients: resample `sigma`, don't interpolate `q_a` (that
    # leaves the curve for a chord).
    sigma: np.ndarray  # (N+1,)
    speed: np.ndarray  # (N+1,)
    acceleration: np.ndarray  # (N+1,)
    jerk: np.ndarray  # (N+1,)
    snap: np.ndarray  # (N+1,)
    coefficients: np.ndarray  # (segments, ORDER, 5)
    q_u: np.ndarray  # (N+1, 2)
    dq_u: np.ndarray
    q_u_eq: np.ndarray
    pump_flow: np.ndarray  # (N+1,) as a fraction of the physical pump
    #: What the flow row was actually bounded by, same units. Carried rather than re-derived: the
    #: caller refuses a plan that bought this row, and two derivations would be two answers.
    pump_flow_bound: float
    slack: float
    #: Slack on the sway box alone (units of `q_sway_max`): sway bought with slack leaves the
    # envelope `Geometry` builds from this bound.
    sway_slack: float
    iterations: int
    solve_time_s: float  # wall clock around the call, so it carries Python and load
    #: acados' `time_tot`; wall clock is load-bound (0.53s idle, 4.97s under Gazebo, same solve) --
    # regress on iteration count instead.
    acados_time_s: float
    residuals: np.ndarray  # (4,) stationarity, equality, inequality, complementarity
    stats: dict  # same solve as flat named numbers, for `~/solver_stats`

    def report(self) -> dict:
        return self.stats | {
            "slack": float(self.slack),
            "sway_slack": float(self.sway_slack),
            "pump_flow_peak": float(np.max(self.pump_flow)),
            "terminal_sway": self.terminal_sway,
            "terminal_sway_rate": self.terminal_sway_rate,
            "duration_s": self.duration,
        }

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def terminal_sway(self) -> float:
        """Tool's rest-pose offset at the goal, radians."""
        return float(np.max(np.abs(self.q_u[-1] - self.q_u_eq[-1])))

    @property
    def terminal_sway_rate(self) -> float:
        """Largest passive rate remaining at the goal, rad/s."""
        return float(np.max(np.abs(self.dq_u[-1])))


class TrajectoryOcp:
    """Generated solver, bounds for one request written onto it; built once at construction (acados regen/recompile into `cache_tree` -- seconds warm, minutes cold)."""

    def __init__(
        self,
        description_xml: str,
        limits,
        config,
        weights: dict,
        build_missing: bool = True,
    ):
        self.limits, self.config = limits, config
        self.segments = int(config.path_segments)
        baked = baked_parameters(config)
        hydraulics = {"pump_flow_max": config.pump_flow_max}
        # Loads a built tree; builds one on first construction (minutes). Decided before `build_ocp`
        # so a refusal costs nothing.
        key = cache_key(baked, hydraulics, description_xml)
        self.tree = tree = tree_of(
            key
        )  # which built solver answered, reported by the node
        fresh = not (tree / f"libacados_ocp_solver_{SOLVER_NAME}.so").is_file()
        if fresh and not build_missing:
            # Node's path: a miss means live config wasn't exported for; compiling here would answer
            # minutes later for an unreviewed machine.
            raise SolverNotExported(
                "\n".join(
                    [
                        f"no exported solver for this configuration ({tree.name});"
                        " export one and restart.",
                        f"  description sha1 {key['description'][:10]}",
                        *(divergence(key) or ["  the cache holds no other solver"]),
                        "  scripts/dump_robot_description.py live.urdf",
                        "  scripts/export_timing_ocp.py --description live.urdf"
                        " --compile-only",
                    ]
                )
            )
        ocp, self.scale, model = build_ocp(
            description_xml, {**baked, "weights": weights}, hydraulics
        )
        # Pump row in unreduced coordinates so reported/bounded cannot drift apart.
        flow = ca.sum1(
            model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
        )
        self._flow = ca.Function("flow", [model.x, model.u, model.p], [flow])
        self._command = model.command_function
        self.N = ocp.solver_options.N_horizon
        self.nominal = float(config.ocp_horizon)
        CACHE.mkdir(parents=True, exist_ok=True)
        if fresh:
            # Loud: usual cause is "exported for another machine/setting"; a silent recompile hides
            # that behind three quiet minutes.
            print(
                "\n".join(
                    [
                        f"no exported solver for this configuration ({tree.name});"
                        " compiling, which takes minutes.",
                        f"  description sha1 {key['description'][:10]}",
                        *(divergence(key) or ["  the cache holds no other solver"]),
                        "  export one with scripts/export_timing_ocp.py"
                        " --description <urdf> --compile-only",
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )
        ocp.code_export_directory = str(tree)
        self.solver = AcadosOcpSolver(
            ocp,
            json_file=str(tree.with_suffix(".json")),
            generate=fresh,
            build=fresh,
            verbose=False,
        )
        if fresh:
            write_manifest(tree, key)
        # `.so` loaded not regenerated: without this, weights are inert on every run but the one
        # that built the tree. Swept warm (equal durations) vs cold rebuild: two refused moves
        # converged, durations fell 17-18%.
        stage_w, terminal_w = w.matrices(weights)
        for node in range(self.N):
            self.solver.cost_set(node, "W", stage_w)
        self.solver.cost_set(self.N, "W", terminal_w)

    def start_speed(self, coefficients: np.ndarray, dq_a_start) -> tuple:
        """`v(0)` from measured start velocity, and how much is lost; motion confined to the path can only leave along its tangent, so the projection is the best available answer."""
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
        # `CalcMovement.slow_down` divides qDotMax/qDDotMax/qDDDotMax alike; s**2 on acceleration
        # would remove the authority to cancel sway at the (fixed) pendulum frequency, so rate/accel
        # scale linearly instead.
        rate_hi = accel_hi = cfg.kappa * speed_scale
        # Command row not scaled by `speed_scale`: slowing down doesn't shrink the valve's domain.
        # `kappa` still applies.
        reference = np.maximum(np.abs(cfg.command_u_min), cfg.command_u_max)
        command_lo = cfg.kappa * cfg.command_u_min / reference
        command_hi = cfg.kappa * cfg.command_u_max / reference
        flow_hi = cfg.kappa * speed_scale * lim.flow_max / cfg.pump_flow_max
        theta = np.array([cfg.ocp_duration_min, cfg.ocp_duration_max]) / self.nominal

        speed_start, unrepresentable = self.start_speed(coefficients, dq_a_start)
        # 1e-6 rad/s is measurement noise; above it, projecting silently would put the solve on a
        # state it is not in.
        if unrepresentable > 1.0e-6:
            raise PlanningError(
                f"the measured start velocity is {unrepresentable:.3e} rad/s off "
                "the path tangent, and a motion confined to the path cannot leave "
                "in any other direction"
            )
        # Noise fitted as direction of travel, caught via `TANGENT_STEP_MAX`; only when start moves
        # (zero tangent under nonzero velocity already refused above).
        tangent_start = float(np.linalg.norm(evaluate(coefficients, 0.0, order=1)[0]))
        interior = float(
            np.median(
                [
                    np.linalg.norm(evaluate(coefficients, sigma, order=1)[0])
                    for sigma in np.linspace(0.0, 1.0, 21)
                ]
            )
        )
        if np.any(dq_a_start) and tangent_start * TANGENT_STEP_MAX < interior:
            raise PlanningError(
                f"the fitted path leaves at {interior / tangent_start:.0f}x under "
                f"its own interior speed: the measured start velocity peaks at "
                f"{np.max(np.abs(dq_a_start)):.2e} rad/s, which this fit took for "
                "the direction of travel and the timing solve cannot recover from. "
                "That rate is the encoder noise floor, not motion -- raise "
                "`REST_VELOCITY` (planner.py) above this machine's standing noise, "
                "or plan from a machine that has settled"
            )

        # `sigma` in [0, 1], `v >= 0`: machine may not run backwards along its own curve.
        free = np.full(4, big)
        lbx = np.concatenate([[0.0, 0.0], -free, -cfg.dq_sway_max, theta[:1]])
        ubx = np.concatenate([[1.0, big], free, cfg.dq_sway_max, theta[1:]])
        # No measured accel/jerk: chain starts flat above the measured velocity.
        x0 = np.concatenate([[0.0, speed_start, 0.0, 0.0], q_u_start, dq_u_start])
        planned = cs.K_PLANNED_DOF
        lh = np.concatenate(
            [
                -np.ones(2),
                [0.0],
                np.full(planned, -rate_hi),
                np.full(planned, -accel_hi),
                command_lo,
            ]
        )
        uh = np.concatenate(
            [
                np.ones(2),
                [flow_hi],
                np.full(planned, rate_hi),
                np.full(planned, accel_hi),
                command_hi,
            ]
        )
        settled = np.concatenate(
            [
                cfg.terminal_q_sway_max / cfg.q_sway_max,
                cfg.terminal_dq_sway_max / cfg.dq_sway_max,
            ]
        )
        # Terminal node is what `sway_slack` doesn't watch; relaxing `terminal_q_sway_max` past
        # `q_sway_max` would let the goal pose leave the clearance envelope with neither check
        # firing -- worst place for it.
        if np.any(cfg.terminal_q_sway_max > cfg.q_sway_max):
            raise PlanningError(
                f"terminal_q_sway_max {np.asarray(cfg.terminal_q_sway_max).tolist()} "
                f"is looser than q_sway_max {np.asarray(cfg.q_sway_max).tolist()} on "
                "some hinge, which would let the goal pose leave the clearance "
                "envelope unwatched: the settled box is a tightening of the "
                "envelope, never a relaxation of it"
            )
        parameters = np.concatenate(
            [[q_tool], payload, np.asarray(coefficients, dtype=float).reshape(-1)]
        )
        price = float(cfg.ocp_slack_price)
        soft = H_RATE

        # Uniform progress, pendulum hanging: no rollout. `1/nominal` alone guesses `ocp_horizon`
        # not the request: first QP fails outright when far off (needs 16s vs nominal 7, dies at SQP
        # iter 1), since what sets 16s is sway rows, not a kinematic ceiling. `speed_scale` known:
        # half speed roughly doubles duration, so the guess carries it.
        duration_guess = float(
            np.clip(
                self.nominal / speed_scale, cfg.ocp_duration_min, cfg.ocp_duration_max
            )
        )
        chain = None
        if INITIAL_GUESS == "PATH_PROFILE":
            chain = profile_guess(
                coefficients,
                rate_hi * lim.dq_max,
                accel_hi * np.asarray(cfg.ddq_a_max, dtype=float),
                speed_start,
                N + 1,
            )
        if chain is not None:
            sigma_guess, speed_guess, accel_guess, duration_guess = chain
            duration_guess = float(
                np.clip(duration_guess, cfg.ocp_duration_min, cfg.ocp_duration_max)
            )
        else:
            sigma_guess = np.linspace(0.0, 1.0, N + 1)
            speed_guess = np.full(N + 1, 1.0 / duration_guess)
            accel_guess = np.zeros(N + 1)
        solver.reset()
        for node in range(N + 1):
            sigma = float(sigma_guess[node])
            guess = evaluate(coefficients, sigma)[0]
            solver.set(node, "p", parameters)
            solver.set(
                node,
                "x",
                np.concatenate(
                    [
                        [sigma, speed_guess[node], accel_guess[node], 0.0],
                        passive_equilibrium(guess),
                        np.zeros(cs.K_PASSIVE_DOF),
                        [duration_guess / self.nominal],
                    ]
                ),
            )
            if node < N:
                solver.set(node, "u", np.zeros(NU))
                # `SIGMA_INPUT_MAX` is baked into the `.so`, not in the cache hash: changing it on a
                # warm tree would be inert, same failure as weights.
                solver.constraints_set(node, "lbu", -np.full(NU, SIGMA_INPUT_MAX))
                solver.constraints_set(node, "ubu", np.full(NU, SIGMA_INPUT_MAX))
                solver.constraints_set(node, "lh", lh)
                solver.constraints_set(node, "uh", uh)
                solver.cost_set(node, "zl", np.full(soft, price))
                solver.cost_set(node, "zu", np.full(soft, price))
            if 0 < node < N:
                solver.constraints_set(node, "lbx", lbx)
                solver.constraints_set(node, "ubx", ubx)
        solver.constraints_set(0, "lbx", x0)
        solver.constraints_set(0, "ubx", x0)
        # Goal: end of curve, arrived stopped. Tool may hang anywhere; `h_e` bounds how still,
        # softly.
        solver.constraints_set(
            N,
            "lbx",
            np.concatenate(
                [[1.0, 0.0, 0.0, 0.0], [-big, -big], np.zeros(2), theta[:1]]
            ),
        )
        solver.constraints_set(
            N,
            "ubx",
            np.concatenate([[1.0, 0.0, 0.0, 0.0], [big, big], np.zeros(2), theta[1:]]),
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
        accel = state[:, X_ACCEL]
        q_a = evaluate(coefficients, sigma)
        tangent = evaluate(coefficients, sigma, order=1)
        curvature = evaluate(coefficients, sigma, order=2)
        third = evaluate(coefficients, sigma, order=3)
        dq_a = tangent * speed[:, None]
        ddq_a = curvature * speed[:, None] ** 2 + tangent * accel[:, None]
        jerk = state[:, X_JERK]
        dddq_a = (
            third * speed[:, None] ** 3
            + 3.0 * curvature * speed[:, None] * accel[:, None]
            + tangent * jerk[:, None]
        )
        q_u = state[:, X_PASSIVE : X_PASSIVE + cs.K_PASSIVE_DOF]
        dq_u = state[:, X_PASSIVE_RATE : X_PASSIVE_RATE + cs.K_PASSIVE_DOF]
        slack, sway_slack = slack_spent(solver, N)
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
        command = np.array(
            [
                np.asarray(self._command(state[node], parameters)).ravel()
                for node in range(N + 1)
            ]
        )
        stats = solver_stats(solver, status, elapsed)
        # Status read before guard: a diverged iterate is likelier to saturate the snap chain, so
        # guard-first would misreport a non-convergence.
        if status != 0:
            # Which of the four residuals stalled says dynamics vs goal box vs soft row; no
            # `Trajectory` carries them, so they ride the message.
            stat, eq, ineq, comp = np.asarray(
                solver.get_stats("residuals"), dtype=float
            )
            raise PlanningError(
                f"the trajectory OCP did not converge: acados status {status} after "
                f"{int(stats['sqp_iterations'])} iterations, {elapsed:.2f} s, on "
                f"residuals stat {stat:.1e} eq {eq:.1e} ineq {ineq:.1e} comp {comp:.1e}",
                stats=stats,
            )
        if guard > 0.99:
            raise PlanningError(
                f"the snap guard bound at {guard:.2f} of {SIGMA_INPUT_MAX:g}: the "
                "duration this answer carries is partly that constant's and not the "
                "machine's",
                stats=stats,
            )
        return Trajectory(
            time=np.linspace(0.0, duration, N + 1),
            q_a=q_a,
            dq_a=dq_a,
            ddq_a=ddq_a,
            dddq_a=dddq_a,
            command=command,
            sigma=sigma,
            speed=speed,
            acceleration=accel,
            jerk=jerk,
            snap=control[:, 0],
            coefficients=coefficients,
            q_u=q_u,
            dq_u=dq_u,
            q_u_eq=np.array([passive_equilibrium(row) for row in q_a]),
            pump_flow=flow,
            pump_flow_bound=flow_hi,
            slack=slack,
            sway_slack=sway_slack,
            iterations=int(solver.get_stats("sqp_iter")),
            solve_time_s=elapsed,
            acados_time_s=float(solver.get_stats("time_tot")),
            residuals=np.asarray(solver.get_stats("residuals"), dtype=float),
            stats=stats,
        )
