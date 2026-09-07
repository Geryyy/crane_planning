"""
The trajectory OCP: its definition, and the solver that answers with it.

`x = [sigma, v, a, j, q_u, dq_u, theta]`, `u = s`, on normalised time over [0, 1]
with
`theta = T / ocp_horizon` a state whose derivative is zero -- so one solve is
time-optimal and there is no outer search over the duration.

**The machine stays on the certified curve.** `q_a` is not a decision variable:
the geometric stage hands over `c(sigma)`, the curve `Geometry.check_path`
certified, and this solver decides only how fast to move along it.

    q_a   = c(sigma)
    dq_a  = c'(sigma) v
    ddq_a = c''(sigma) v^2 + c'(sigma) a

with `sigma` integrated four times, so the input is the snap and `q_a(t)` is C4 --
the derivative count C3's flat inversion consumes. Holding the *acceleration*
constant per interval, as an input is held, left `q_a(t)` C1 whatever the curve
was.

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

#: Written beside each built solver: the `cache_key` it was built from. The tree
#: name is a digest and says nothing about what it holds; this is what lets a
#: missed prebake name the value that diverged instead of just recompiling.
MANIFEST = "cache_key.json"

#: The two hinges the hanging pose is a closed form in. `crane_model.symbolic`
#: numbers the planned axes but does not name them; `planner.py` carries the same.
BOOM_AXIS = 1
ARM_AXIS = 2

#: Row order of `x`. The `sigma` chain is contiguous and leads, the passive pair
#: follows, `theta` is last as the one component the solver picks.
X_SIGMA, X_SPEED, X_ACCEL, X_JERK = 0, 1, 2, 3
X_PASSIVE, X_PASSIVE_RATE, X_HORIZON = 4, 6, 8
NX, NU = 9, 1

#: Row order of `h`, which the caller reads back and must not re-derive. The
#: rate and input blocks were boxes on `x` and `u` when `q_a` was planned; they
#: are nonlinear rows now because what they bound is an expression in `sigma`.
H_SWAY, H_FLOW, H_RATE = 0, 2, 3
H_ACCEL = 3 + cs.K_PLANNED_DOF
#: The C3 command itself, `u = dq_a + tau_dot_a / k`, against the domain the
#: compensator was identified over. It replaced a `dddq_a` row: bounding the
#: third derivative was only ever a proxy for this, and a decomposed proxy --
#: reserve the rate, reserve the acceleration, give jerk the remainder -- prices
#: a sum of worst cases where the machine pays the sum at each instant.
H_COMMAND = 3 + 2 * cs.K_PLANNED_DOF
NH = 3 + 3 * cs.K_PLANNED_DOF
#: Row order of `h_e`: the settled box the caller accepts.
HE_SWAY, HE_SWAY_RATE, NH_E = 0, 2, 4

#: Quintic path, so six power-basis coefficients per segment. Degree 5 with
#: simple interior knots is `C4` in `sigma`, which is what C3's flat inversion
#: needs -- continuity becomes a property of the parameterization rather than
#: something a constraint has to enforce.
ORDER = 6

#: How many derivative levels the solver builds off the path: `c` through its
#: fourth, because `q_a(t) = c(sigma(t))` differentiated four times reaches it.
PATH_DERIVATIVES = 5

#: Box on the input, a numerical guard and not a physical limit. `solve` refuses
#: rather than answer if it binds, because a duration this constant decided is
#: not the machine's.
#:
#: It bounded `a` while `a` was the input, at 20.0, where `a` reached 0.01-0.02
#: of it. It bounds the **snap** now and the old value is two derivatives too
#: small: at 20.0 four of five bench moves refuse with the guard at exactly 1.00.
#: Swept 20 / 200 / 2000 / 20000 on the shipped defaults at `speed_scale` 0.9:
#: the box stops binding at 200, and above it the answer is identical to the
#: hundredth -- same durations, same `max|s|` of 24.5 / 131.8 / 51.3 / 41.2. So
#: the solve's own demand is ~132 and 2000 is ~15x headroom, which is what a
#: guard should be.
#:
#: The **command** carries the physical row now (`H_COMMAND`), and it contains
#: `dddq_a` through `M_ii/k`, so this bounds only the snap and deliberately
#: loosely: with the command bounded, `q_a''''` is finite whatever this is, and
#: finite is what a PT1 inversion would need if one were ever done.
SIGMA_INPUT_MAX = 2000.0

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
    "dddq_a_max",
    "command_k",
    "command_u_min",
    "command_u_max",
    "path_segments",
)

#: The integrator, and the stage count that makes each one order 4: ERK wants 4
#: stages, Gauss-Legendre IRK 2. ERK4 is far cheaper per node but its stability
#: region is finite and `theta` scales the step, so the choice is not uniform
#: over the duration box: at `ocp_duration_max` the pendulum (w = 3.6 rad/s)
#: sits at w*dt = 1.8, where RK4's per-step amplification is 0.854 -- 500x of
#: fabricated damping over 40 intervals, pricing the sway rows as if the swing
#: settled on its own. Gauss-Legendre is symplectic, |R(iy)| = 1 at any step, so
#: it cannot invent that. `SIM_SUBSTEPS` is what keeps ERK honest instead --
#: see there; the choice stays a knob.
INTEGRATORS = {"ERK": 4, "IRK": 2}

#: Substeps per shooting interval. Not a knob: answers land at 16 s, not at the
#: ~7 s the stability argument above was first written for, and one RK4 step per
#: interval at w*dt = 1.43 amplifies by 0.955 -- 0.16 over 40 intervals, so the
#: solver believes 84 % of a swing excited early has vanished by the goal. The
#: sway rows, the settled box and the terminal pin would then be met by the
#: integrator rather than by the timing. Three substeps put 16 s at 0.991 and
#: 20 s at 0.965; the quartic reconstruction stays exact because each substep is
#: still exact on the nilpotent chain.
#:
#: Measured over the 21 bench moves: no duration moves by more than 0.25 % and no
#: refusal changes, so the fabricated damping was never what the answers rested
#: on. This is a correctness fix, not a retiming, and it is free.
SIM_SUBSTEPS = 3


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


def baked_parameters(config) -> dict:
    """
    Read the `BAKED` names off a config. One place, because three callers agree.

    Building this dict by hand is how a new `BAKED` entry goes missing: the tuple
    is what `cache_key` iterates, so a name added there and not here is a
    `KeyError` at best and a wrong hash at worst.
    """
    return {name: getattr(config, name) for name in BAKED}


def cache_key(parameters: dict, hydraulics: dict, description: str) -> dict:
    """
    Everything a built solver is only valid for. The tree is named after it.

    A cached `.so` is loaded, not compared. Every value that enters the
    *expressions* or the dimensions therefore has to move the directory, or a
    changed `ocp_integrator` -- or a changed `path_segments`, which changes the
    parameter vector itself -- silently answers with the previous build.

    The **description** is one of those values and the least visible: the
    dynamics are built from it, the node builds its planner from whatever
    `/robot_description` carries, and the exporter bakes from a file on disk.
    Those two are not the same machine unless someone checked -- the sim's
    `/robot_description` is a different xacro at `sim_hydraulics:=false` -- so it
    is hashed here and reported by `divergence` when a prebaked tree is missed.
    """
    baked = {}
    for key in BAKED:
        value = parameters[key]
        baked[key] = value.tolist() if hasattr(value, "tolist") else value
    baked["pump_flow_max"] = float(hydraulics["pump_flow_max"])
    # Not a parameter, so it cannot come from `BAKED`, and it sets the parameter
    # vector's width: a warm cache built at a different `ORDER` would be handed a
    # coefficient block of the wrong shape.
    baked["path_order"] = ORDER
    # Same reasoning for the state count: it is the solver's shape, not a
    # parameter, and a cached `.so` is loaded rather than compared.
    baked["nx"] = NX
    # And for the substep count: it is compiled into the integrator, so a cache
    # built at one value would keep answering with that integrator's damping.
    baked["sim_substeps"] = SIM_SUBSTEPS
    # `ocp_integrator` names the method, `INTEGRATORS` decides its order, and only
    # the name is a config value -- editing the map alone would load the old order.
    baked["sim_stages"] = INTEGRATORS[parameters["ocp_integrator"]]
    # How many levels `path_expression` builds, i.e. how long the chain is.
    baked["path_derivatives"] = PATH_DERIVATIVES
    # The machine itself: masses, inertias, limits and the linkage the dynamics
    # are generated from. Hashed, not stored -- a URDF is megabytes.
    baked["description"] = hashlib.sha1(description.encode()).hexdigest()
    return baked


def tree_of(key: dict) -> Path:
    """Where the solver for `key` is built, named so a stale one cannot load."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return CACHE / f"{SOLVER_NAME}_{digest[:10]}"


def cache_tree(parameters: dict, hydraulics: dict, description: str) -> Path:
    """`tree_of` for callers that hold the inputs rather than the key."""
    return tree_of(cache_key(parameters, hydraulics, description))


def write_manifest(tree: Path, key: dict) -> None:
    """Record what a freshly built tree was built from, for `divergence` to read."""
    (tree / MANIFEST).write_text(json.dumps(key, sort_keys=True, indent=1))


class SolverNotExported(RuntimeError):
    """No compiled solver for the description and settings the node was handed."""


def divergence(key: dict) -> list:
    """
    Say which baked names separate `key` from each solver already in the cache.

    The hash decides *that* a prebaked tree was missed; this says *what* missed
    it, which is the whole difference between "recompiling, three minutes" and
    "you are about to plan for a machine nobody exported for". `description`
    appearing here means the model diverged, not a setting.
    """
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


# -------------------------------------------------------------------- the path


def power_coefficients(path, segments: int) -> np.ndarray:
    """
    `(segments, ORDER, 5)`: `c(sigma) = sum_m a[k, m] (sigma - k/S)^m` on segment k.

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
    # `geometry.fit`'s degree and `ORDER` are one number written in two places,
    # and the failure if they drift is silent: a shorter power basis truncates
    # the curve instead of refusing it.
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
    """
    `PATH_DERIVATIVES` levels of the path as `casadi.SX`, coefficients symbolic.

    One loop over derivative order rather than a branch per level, so raising
    `ORDER` is a constant change and not an edit here. It is the symbolic mirror
    of `evaluate` and the two must agree at every level.
    """
    index = ca.fmin(ca.fmax(ca.floor(sigma * segments), 0.0), segments - 1)
    local = sigma - index / segments
    value = [ca.SX.zeros(cs.K_PLANNED_DOF) for _ in range(PATH_DERIVATIVES)]
    for segment in range(segments):
        # A comparison carries no derivative in casadi, so `d/dsigma` of the sum
        # is the selected polynomial's own derivative -- which is what is wanted,
        # and correct at a breakpoint because the curve is C4 across it.
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


# --------------------------------------------------------------------- the OCP


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict):
    """Assemble the OCP. Returns `(ocp, scale, model)` as the sibling exporter does."""
    model = cs.CraneSymbolicModel(description_xml, TOOL)
    dq_max, tau_max = description_limits(model)
    q_sway_max = np.asarray(parameters["q_sway_max"], dtype=float)
    dq_sway_max = np.asarray(parameters["dq_sway_max"], dtype=float)
    ddq_a_max = np.asarray(parameters["ddq_a_max"], dtype=float)
    dddq_a_max = np.asarray(parameters["dddq_a_max"], dtype=float)
    command_k = np.asarray(parameters["command_k"], dtype=float)
    command_u_min = np.asarray(parameters["command_u_min"], dtype=float)
    command_u_max = np.asarray(parameters["command_u_max"], dtype=float)
    # One scale for the row, so 1.0 is the wider side of an asymmetric domain.
    command_ref = np.maximum(np.abs(command_u_min), command_u_max)
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
    # The chain: `a` and `j` are states now, `s` is the input. Holding the
    # *acceleration* constant per interval, as acados does with an input, made
    # `ddq_a` discontinuous and `q_a(t)` C1 whatever the curve was; holding the
    # snap constant leaves `q_a(t)` C4, which is what C3's inversion consumes.
    acceleration = ca.SX.sym("a")
    jerk = ca.SX.sym("j")
    snap = ca.SX.sym("s")
    #: The path, as `p`: `path_segments` fixes the knot vector, so only these
    #: numbers change between requests and the solver is generated once.
    coefficients = ca.SX.sym("c", segments * ORDER * cs.K_PLANNED_DOF)

    centre, tangent, curvature, third = path_expression(sigma, coefficients, segments)[
        :4
    ]
    q_a = centre
    # Faa di Bruno on `q_a(t) = c(sigma(t))`. No `theta` here: `f_expl_expr`
    # carries `nominal * theta`, so `v` is d sigma/dt in physical seconds
    # already and these are the physical derivatives the limits are written
    # against. The snap level is deliberately not built: bounding the jerk plus
    # the input box already leaves `q_a''''` finite, which is all the command
    # PT1 inversion needs, and a fourth row costs solve time on a problem that
    # is already the expensive half of the plan.
    dq_a = tangent * speed
    ddq_a = curvature * speed * speed + tangent * acceleration
    dddq_a = third * speed**3 + 3.0 * curvature * speed * acceleration + tangent * jerk
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
    acados_model.x = ca.vertcat(sigma, speed, acceleration, jerk, q_u, dq_u, theta)
    acados_model.u = snap
    acados_model.p = ca.vertcat(model.p, coefficients)
    acados_model.f_expl_expr = ca.vertcat(
        nominal * theta * ca.vertcat(speed, acceleration, jerk, snap, dq_u, ddq_u),
        0.0,
    )
    # acados does not derive one form from the other: ERK reads `f_expl_expr`,
    # IRK reads `f_impl_expr` and errors on an empty one. Both are set here
    # unconditionally so switching integrator is a solver option, not a re-model.
    acados_model.xdot = ca.SX.sym("xdot", NX)
    acados_model.f_impl_expr = acados_model.xdot - acados_model.f_expl_expr

    # What `AddC3Feedforward` will actually command for this plan, in its own
    # arithmetic: `u_d = qdot_d + tau_dot_d / k_i` (`c3_feedforward_math.hpp`),
    # with `tau` the same RNEA over all eight coordinates the node evaluates.
    # `jtimes` against the state derivative is the exact total derivative --
    # d tau/dq qdot + d tau/dqdot qddot + M qdddot, passive coupling included --
    # and it is the expression issue 114 needs, so it is built once, here.
    #
    # Convention: `tau_a` is `crane_model`'s, which carries the mimic coupling and
    # the transmission projection, while `command_k` was identified against a
    # plain CRBA -- the two differ by 1.09x to 3.40x and the factor is
    # pose-dependent, so it cannot be absorbed into a constant. The node divides
    # exactly this `tau` by exactly this `k`, so the planner refuses what the
    # machine will actually be asked for; correcting the convention corrects both
    # at once. See `issues/open/113`'s notes.
    #
    # `f_expl_expr` is `dx/dtau` on acados' own `[0, 1]` grid -- it carries the
    # `nominal * theta` scaling -- so `jtimes` against it gives `d tau_a/dtau`,
    # which is `T` times the physical derivative. Dividing it back out is what
    # makes this row a command and not a command times the answer's own duration:
    # left in, the row's acceleration coefficient was `T D_ii/k` against a
    # physical `d_i/k`, so it tightened as the answer lengthened and the horizon
    # state appeared in `con_h` where no other row uses it.
    tau_dot_a = ca.jtimes(tau_a, acados_model.x, acados_model.f_expl_expr) / (
        nominal * theta
    )
    command = dq_a + tau_dot_a / command_k
    model.command_function = ca.Function(
        "command", [acados_model.x, acados_model.p], [command]
    )

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
    ocp.solver_options.sim_method_num_steps = SIM_SUBSTEPS
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
    dddq_a: np.ndarray  # what the C3 inversion pays for, third term
    #: `u_d = dq_a + tau_dot_a / k` per axis, what `AddC3Feedforward` will
    #: command. This is the row `H_COMMAND` bounds.
    command: np.ndarray
    #: The path state itself, and the coefficients it indexes. `q_a` is a
    #: function of these and not an independent answer, so a consumer that has to
    #: resample must resample `sigma` and evaluate the curve -- interpolating
    #: `q_a` leaves the certified curve for the chord between two of its points.
    #: `snap` is the solver's own input, repeated at `N` so every array here is
    #: `N+1` long; acados holds it constant across an interval, which makes
    #: `sigma` exactly quartic in time there.
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
        # **Generated by `scripts/export_timing_ocp.py`, not here.** Loads a built
        # tree; builds one on first construction, costing minutes. Regenerating
        # unconditionally would overwrite the exporter's output and make `--check`
        # meaningless. Decided before `build_ocp`, so a refusal costs nothing.
        key = cache_key(baked, hydraulics, description_xml)
        #: Which built solver answered. Reported by the node, so a run says on
        #: disk what it planned with.
        self.tree = tree = tree_of(key)
        library = tree / f"libacados_ocp_solver_{SOLVER_NAME}.so"
        fresh = not library.is_file()
        if fresh and not build_missing:
            # The node takes this path: a miss means the live description or a
            # setting is not the one anything was exported for, and compiling it
            # here would answer minutes later for a machine nobody reviewed.
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
        # The pump row in the *unreduced* coordinates, off the graph the solver
        # constrains, so reported and bounded cannot drift apart.
        flow = ca.sum1(
            model.z[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
        )
        self._flow = ca.Function("flow", [model.x, model.u, model.p], [flow])
        self._command = model.command_function
        self.N = ocp.solver_options.N_horizon
        self.nominal = float(config.ocp_horizon)
        CACHE.mkdir(parents=True, exist_ok=True)
        if fresh:
            # Loud, because the honest cause is usually not "nothing is exported"
            # but "what is exported is for another machine or another setting",
            # and the recompile hides that behind three quiet minutes.
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
        # `CalcMovement.slow_down` is a "divider to reduce max speed/acceleration",
        # and the reference server divides qDotMax, qDDotMax and qDDDotMax by it
        # alike (`mp_crane_lib_helpers.hpp` `apply_slow_down`). Uniform time
        # scaling would give s**k on the k-th derivative, but it is not a symmetry
        # of this problem -- the pendulum period is fixed, so an s**2 acceleration
        # ceiling takes away the authority a solve needs to cancel sway at the
        # swing frequency and it refuses instead of slowing down. The jerk ceiling
        # is a valve budget, not a comfort choice, and scaling it at all is
        # unnecessary; it is scaled only to stay one rule with the other two.
        rate_hi = accel_hi = cfg.kappa * speed_scale
        # The command row is not scaled by `speed_scale`: `slow_down` is a request
        # to move more gently, not a claim that the valve's domain shrank. `kappa`
        # still applies -- it is the deployment reservation on every limit.
        reference = np.maximum(np.abs(cfg.command_u_min), cfg.command_u_max)
        command_lo = cfg.kappa * cfg.command_u_min / reference
        command_hi = cfg.kappa * cfg.command_u_max / reference
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
        free = np.full(4, big)
        lbx = np.concatenate([[0.0, 0.0], -free, -cfg.dq_sway_max, theta[:1]])
        ubx = np.concatenate([[1.0, big], free, cfg.dq_sway_max, theta[1:]])
        # No measured acceleration or jerk: the machine reports a velocity, so
        # the chain starts flat above it rather than from an invented number.
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
        parameters = np.concatenate(
            [[q_tool], payload, np.asarray(coefficients, dtype=float).reshape(-1)]
        )
        price = float(cfg.ocp_slack_price)
        soft = H_RATE

        # Uniform progress with the pendulum hanging: no rollout and nothing that
        # knows the answer.
        #
        # `1 / nominal` is a guess about `ocp_horizon`, not about the request, and
        # when the answer is far from nominal the first QP fails outright --
        # `across` needs 16 s against a nominal of 7 and dies at SQP iteration 1
        # with a stationarity residual of 1e3, as does `pair_tuck`. Seeding from a
        # rate- and acceleration-limited duration estimate does not fix it: both
        # estimates come out under 7 s on every bench move including `across`,
        # because what sets that 16 s is the sway rows and not a kinematic ceiling.
        # A guess that helps has to know the pendulum, and that is not this change.
        #
        # What *is* known is `speed_scale`: a request to move at half speed is a
        # request for an answer about twice as long, so the guess carries it. Left
        # alone the `stow` move at `speed_scale` 0.5 answers at 11.5 s against a
        # guess of 7 and fails the same way.
        duration_guess = float(
            np.clip(
                self.nominal / speed_scale, cfg.ocp_duration_min, cfg.ocp_duration_max
            )
        )
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
                        [sigma, 1.0 / duration_guess, 0.0, 0.0],
                        passive_equilibrium(guess),
                        np.zeros(cs.K_PASSIVE_DOF),
                        [duration_guess / self.nominal],
                    ]
                ),
            )
            if node < N:
                solver.set(node, "u", np.zeros(NU))
                # The input box is written here, not left to the export. Set
                # only on the `AcadosOcp` it is baked into the `.so`, and it is
                # not in the cache hash, so changing the constant on a warm tree
                # would be inert -- the same failure the weights had.
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
        # The goal is the end of the curve, arrived at stopped. The tool may hang
        # where it likes; how still it has to be is `h_e`, softly.
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
        command = np.array(
            [
                np.asarray(self._command(state[node], parameters)).ravel()
                for node in range(N + 1)
            ]
        )
        if guard > 0.99:
            raise PlanningError(
                f"the snap guard bound at {guard:.2f} of {SIGMA_INPUT_MAX:g}: the "
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
            slack=slack,
            iterations=int(solver.get_stats("sqp_iter")),
            solve_time_s=elapsed,
            acados_time_s=float(solver.get_stats("time_tot")),
            residuals=np.asarray(solver.get_stats("residuals"), dtype=float),
        )
