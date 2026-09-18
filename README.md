# crane_planning

The `crane_planner` node. Two stages, one dependency that matters.

```
plan(goal position + yaw in K0_mounting_base)
  -> deterministic corridors   direct first, then lift/traverse/descend + lateral
  -> time-optimal OCP          acados, horizon as a state, along that curve
  -> JointTrajectory on /crane/reference
```

No stochastic sampler. Everything the machine can do -- reach, hang, collide --
is asked of `crane_model`; nothing about the machine is written down here. The
EOM comes from `crane_model.symbolic`, and the pump limit the description lacks
is `crane_model`'s too, read off `crane_model/config/hydraulics.yaml` and
overridable -- not copied -- in `config/crane_planner.yaml`.

Joint ranges and velocity limits come from the `robot_description` the profile
remaps this node onto, **intersected with
`crane_model/config/control_safe_limits.yaml`** -- narrower than the description
on the boom, the arm and four velocity rows, and the same box `crane_mpc`
enforces, so this planner cannot certify a pose or a speed the MPC refuses. That
box may never exclude the measured pose: the machine parks with the boom folded
below its 0.02 rad, so `Limits.relaxed_to` widens the row to wherever the
machine is, exactly as the MPC's `position_box` does. The start state is still
checked against the *description*'s stops, which is a different question -- can
this measurement be real.

## The tool corridors

Each candidate is a TCP polyline in `K0_mounting_base`; yaw follows the shortest
arc over its Cartesian length. q8 is not planned -- the low-level controller
holds it at its start value.

Direct line first. Then the **joint-space line** to the cold goal solve -- no IK
along it, and the line the reference iLQR planner effectively moves on; a
close-in goal the tool chord cannot reach without folding the arm through a limit
is answered here. Then transfer planes, starting above the highest scene
primitive plus the full clearance requirement and rising by
`corridor_height_step`; each plane gets a straight traverse plus configured
left/right offsets. A plane corner the tool cannot be placed at refuses every
corridor through it before lifting (one IK per corner, not hundreds per
corridor). First candidate surviving lifting, fitting and the certificate wins. A
fit failure advances to the next candidate rather than ending the call.

**IK is the lift, not the check.** Each polyline sample is lifted into the five
planned coordinates by continuation IK seeded from its predecessor; the passive
pair is re-settled at every configuration, so what is solved is where the tool
actually hangs. Telescope redundancy is taken up by a weak pull towards the seed.
Only the *goal* solve is cold, spreading `ik_restarts` over the telescope range
-- the coordinate the residual is flat in.

Sample spacing is decided by the certificate, not a fixed count: the step halves
until the machine's own motion fits inside `margin_interp`, and grows back by a
quarter over a clear run. A small step for the tool is not a small step for the
arm -- near a singularity the tool barely moves while the boom swings through it.

## The collision certificate

Not to be weakened by accident.

Sampling proves nothing alone. What makes finitely many checks a statement about
a continuous motion is a **margin paired with a Lipschitz bound**. Rotating
planned joint `j` by `dq_j` displaces any point below it by at most
`radii[j]*|dq_j|`, so between two configurations nothing moves further than

    step_bound = sum_j radii[j] * |dq_j|

*If every checked configuration clears the scene by more than `required`, and
`step_bound <= margin_interp` for every consecutive pair, then nothing touches
anything in between.* Both halves needed. Without the margin the bound bounds
nothing; without the bound the samples are just samples and a thin obstacle
tunnels.

    required = margin_safety + margin_interp + envelope

| | |
|---|---|
| `margin_safety` | clearance the answer actually carries |
| `margin_interp` | pays for the gap between two checked configurations |
| `envelope` | `l_tool*sin(q_sway_max)` -- how far the tool swings inside the admissible box |

The envelope is **computed, not configured** -- pendulum length read from the
description. One distance test at the hanging pose answers the whole box for any
configuration clearing more than `required`: no sway state can then reach
anything.

That test is sufficient, not necessary, and the envelope is owed by what hangs on
the hinges and by nothing else. Column, boom and arm do not swing, so a
configuration inside `required` at the hanging pose is not refused on it. The
machine is split at the upper passive hinge (`collision_queries(..., swinging=)`
in `crane_model`) and each half held to what it owes:

    swinging half:  clearance > margin_safety + margin_interp + envelope
    rigid half:     clearance > margin_safety + margin_interp

Two more queries, only where the first could not decide (`Geometry.margin`).
Self-collision is a property of the whole machine and is tested on the first
query only.

`radii` is over-estimated on purpose. Measured at full telescope extension where
every radius is largest; outermost point taken as TCP + `tool_radius` + the
carried body's reach; distance taken from each joint's *origin*, not its axis.
The prismatic telescope carries radius 1 -- it displaces what it holds by exactly
its extension. Too large costs samples, too small is unsound.

Self-collision tests at zero, not against `required`: the links are near each
other by design.

The certificate runs **once, on the fitted spline** (`Geometry.check_path`). The
lifted polyline is not checked -- it exists to establish a continuous IK branch
and the interpolation bound. The spline is a different curve, and it is the curve
that executes.

## Why the lift marches instead of bisecting

Five planned coordinates against a four-DOF tool task leaves one redundant DOF,
so every pose is reached by a one-parameter family and the IK picks one member by
continuity from its seed.

So the **goal configuration is an output of the walk, not an input**. Solving the
goal independently picks its own member, and between two members there is a
finite jump no refinement closes. What is specified is the goal *pose*; the cold
multi-restart solve at the top of `plan` only establishes reachability and its
answer is discarded.

An IK branch change cannot slip through: it is a large joint step for an
arbitrarily small tool step, so `step_bound` refuses it and the step halves until
`MIN_LIFT_STEP`, where the plan is refused by name. Resolution test and
continuity guard are one test.

## The hanging pose is closed form

    q_eq = (pi/2 - q_boom - q_arm, pi/2)

A two-hinge pendulum hangs straight down: the tip joint takes up whatever the
boom four-bar accumulated, the tilt joint does not move. Measured against the
general Newton-plus-grid solve it replaces at 1e-4 rad across the workspace, and
independent of slew, telescope, rotator, tool and payload -- against a sway box
half-width of 0.2 rad.

22.6 ms versus two subtractions. That is why the pendulum can be settled at
*every* configuration the lift checks.

## The certified curve

Least-squares **quintic** B-spline through the lifted configurations, `sigma`
distributed by how long each chord takes at its slowest axis's limit. Degree 5,
not 3: simple interior knots make a degree-`d` B-spline C(d-1), so a quintic is
C4 in sigma -- the derivative count C3's flat inversion consumes, out of the
parameterisation rather than enforced. A kink is not a curve the machine can
follow either way. `geometry.fit`'s docstring carries the rest; `ocp.ORDER` must
stay `degree + 1`.

It approximates rather than interpolates: the certificate wants samples dense
and the curve wants long end intervals, and interpolating a dense path spikes
`q_a''` at the endpoint, 361 against 0.4 in the interior. A fixed
`path_segments` breaks the tie. It is structural besides -- the OCP is
code-generated against a parameter vector that many polynomials wide, so a short
lift is resampled along its own chords rather than fitted with fewer pieces.

Endpoints are imposed exactly by writing the two outer coefficients. The start
slope is written too, but **only while the machine is moving**: the OCP leaves
along the tangent, so the tangent has to be the measured direction. From rest
there is no direction to honour, and writing one anyway sets `q_a'(0) = 0`,
where `ddq_a = q_a'' v^2 + q_a' a` contains no `a` -- a singular input, not a
slow one. The goal end is never clamped at all: arriving stopped is `v = 0`, a
condition on the timing, and asking the geometry for it too pays twice.

## The timing OCP

`ocp.py` builds one acados OCP on normalised time with the horizon a
zero-derivative state: a single solve is time-optimal, no outer duration search.
Every residual row is divided by the limit it is measured against, so 1.0 is the
bound on all of them alike -- and it has to be, because Gauss-Newton builds its
Hessian from `J' W J`, so one row left in physical units sets the conditioning of
everything.

**`q_a` is not a decision variable.** It is handed the certified curve and moves
*along* it, deciding only how fast:

    q_a = c(sigma)   dq_a = c'(sigma) v   ddq_a = c''(sigma) v^2 + c'(sigma) a

`sigma` is integrated **four** times -- `x = [sigma, v, a, j, q_u, dq_u, theta]`
and the input is snap -- so `q_a(t)` is C4, which is what C3's flat inversion
consumes. Holding acceleration constant per interval left it C1. The jerk the C3
feedforward can afford is a hard row per axis; `c3_feedforward:=false` at launch
is for a controller that does not consume the effort field at all.

The executed curve is the certified curve, so the clearance proof holds verbatim
-- no obstacle rows, no corridor, nothing to re-prove. Planning `q_a` freely
between the curve's endpoints instead took the machine 0.84-1.29 m off it against
0.05 m of spare clearance, voiding the proof; staying on it costs 0.4-4.4% of
duration and buys back half the SQP iterations. The price is lateral authority:
sway is damped by timing alone.

Joint velocity and acceleration, reserved pump flow and the sway box are
constraints in the solve; `sigma` is boxed to [0, 1] and `v >= 0`, because the
machine may not run backwards along its own curve. `q_a` needs no range row -- it
is on the curve, and `fit` already refused a curve leaving joint range. Terminal
sway offset and rate are checked after the solve against
`terminal_q_sway_max`/`terminal_dq_sway_max`; a near miss is refused, not
published.

## Cost

PZS100, `scripts/plan_example.py` in a clear scene at shipped defaults, on this
development image, one run:

| | |
|---|---|
| lifted configurations | 177 |
| OCP solve | 1.95 s, 20 SQP iterations |
| planner call | 2.32 s |
| trajectory duration | 6.43 s |
| terminal sway | 0.39 deg, 0.000 rad/s |
| peak pump draw | 0.76 of the limit at kappa = 0.8 |

The solve is now most of it. Two more integrator states and the C3 command row
cost roughly an order of magnitude over the acceleration-input stage these
numbers replaced. `margin_interp` is the knob on the geometric half -- one IK
solve plus one collision query per lifted configuration, and halving it roughly
doubles the configurations.

## Tuning

`plan_example.py` runs the deployment's own `Planner` against the checked-in
description -- no ROS, no graph, no clock.

```bash
./scripts/plan_example.py --show     # --help for the rest
```

Every constrained row is plotted over its own bound, so which row decided the
duration is one glance. Read a regression off iterations and residuals, **never off solve time**: the
same problem measures 0.27 s idle and 4.97 s inside a running Gazebo.

It exercises every stage, so its refusal is the node's refusal.

`plan_goals.py` is the same planner against a set of goals, all from
`crane_model.presets.OUTSIDE`, each rolled on the MuJoCo plant
(`crane_model.mujoco_plant`) with its TCP path drawn into the scene.

```bash
./scripts/plan_goals.py                    # every goal, viewer on
./scripts/plan_goals.py --goals out across --headless
```

`bench_ocp.py` is the OCP's own bench: 100 paths sampled inside the
control-safe box, start and goal both, each planned and logged.

```bash
./scripts/bench_ocp.py                          # baseline -> build/bench_ocp.json
./scripts/bench_ocp.py --out build/after.json   # then diff the two logs
```

It sweeps nothing -- changing the OCP means editing it and re-exporting, and
every knob that matters (`ocp_intervals`, `ocp_horizon`, `ocp_integrator`,
`ocp_max_iterations`, `ocp_tolerance`, `levenberg_marquardt`) is in `BAKED`, so
it moves the solver tree. The log records those, so a run cannot be attributed
to the wrong solver. Read the cost block (SQP and QP iterations) against the
quality block below it: iterations fall for free if the answers may get worse.
On the shipped OCP, 63 of 100 paths solve at 28 SQP iterations median, 97 worst
against a cap of 100; 11 are refused by the solver and 26 by geometry before it.

Its summary puts the OCP's own peak sway beside MuJoCo's for the same motion:
an independent integrator on the same URDF, so a shaping change that only the
OCP believes shows up as the two columns parting. `tune_planner.py` is one goal
of this in detail, with the sway traces plotted; both share the rollout.

## The node

| | |
|---|---|
| services | `/a2b_movement` (`timber_crane_planning_interfaces/CalcMovement`) |
| publishes | `/crane/reference` (`trajectory_msgs/JointTrajectory`, transient-local) |
| | `/crane_planner/planned_path` (`nav_msgs/Path`, visualization only) |
| | `tcp_path` (`nav_msgs/Path`) -- the legacy A2B server's name, for RViz |
| | `/crane_planner/markers` (`visualization_msgs/MarkerArray`, transient-local) |
| | `~/solver_stats` (`diagnostic_msgs/DiagnosticArray`, transient-local) |
| subscribes | `/joint_states`, `/robot_description`, `/crane/collision_scene`, `/crane/payload_estimate` |
| frame | `K0_mounting_base` throughout; nothing is converted |

**Not a second writer of the machine**: `/crane/reference` is a reference, not a
command; `crane_velocity_controller` stays sole claimant of the six velocity
command interfaces; this node holds no `controller_manager` client.

The PZS100 Gazebo actuator reports q9 as total opening with an EPSCOPE
`state_factor` of 2, while URDF and collision model use one rail's coordinate.
Launch with `joint_states_topic:=/joint_states_rviz` -- that bringup already
publishes the conversion, turning a reported 1.261858 into the actual 0.420929 m.
Either coordinate outside the URDF range is refused, never projected.

`/joint_states` comes from `joint_state_broadcaster` at depth 10: two producers
publish partial messages back to back and a depth-1 queue can drop one for good.
Cached and read by joint name, never index. The passive pair has a freshness
deadline -- `JointState` has no `valid` flag, so age is the whole test, and a
stale or absent sway measurement is a refusal, not a zero.

Every refusal names itself and leaves the standing reference alone: unreachable
goal, no corridor surviving its certificate, and untimeable path are three
different answers.

## What the tool carries

A payload reaches the collision check as a body marked `attached_to_tool`. Two
rules that only work together: it is **not** checked against the links holding it
(a gripped block sits inside the gripper -- checking there refuses every pose),
and it **is** checked against the rest of the scene, which a scene body otherwise
never is (what a carried body is for is hitting the world).

Which links count as holding it is read off the description: a body is carrying
if the joint placing it is at or below the joint a payload mounts on. On the
PZS100 that is the rotator lower part, the TCP and the two rails.

Placed at **every configuration checked**, not once at the start -- otherwise a
ghost stands in the start pose while the real payload rides the tool unchecked.
Its reach also enters `radii`, so the step bound covers what is in the gripper.

## Seeing the plan

`/crane_planner/markers` draws four things: the **path** as one line; the **tool**
swept along it (tip hinge to TCP at samples down the trajectory -- that segment is
the pendulum, so its lean is the predicted sway); the **goal** pose and facing;
and **the scene the planner actually checked**, which is not the scene anyone
published -- by then `truck` has become a bed, six runges and a headboard, and the
payload is drawn where it will be. Structural, perceived and payload are three
colours, because "why was this refused" is usually answered by which kind it hit.

Goal and geometry go out **before** the solve, so a refusal leaves them on screen.

Display: `cbs.rviz`, under *Planning and Control*.

## The truck

The scene carries one primitive with the reserved id `truck`; `expand_truck`
turns it into what the tool can hit -- bed top face, six runges, headboard closing
the cab end. The crane is bolted to the vehicle, so the box itself is not an
obstacle. Dimensions are configured, position arrives with the primitive, so
moving the truck moves all of it.

A runge sits flush against the bed edge rather than centred: the description puts
the real post's outer face there, and the configured section is an inflation of
that post, so it grows inboard. The headboard sits flush against the box's +x
face, the end the outermost runge station is at.

## The `a2b_movement` compatibility service

Drop-in replacement for the legacy `a2b_ilqr_server`, and an **adapter, not a
second planner**: `a2b.py` maps `CalcMovement` onto the native `Planner.plan`
call, `node.py` runs it. One start state, one scene, one kappa, one
`/crane/reference` publication.

| `CalcMovement` | native | rule |
|---|---|---|
| `y_n` | goal position | the **tip pivot K5**, plus the settled tip-to-tool offset from the model |
| (no field) | goal frame | `K0_mounting_base`, asserted and never converted |
| `phi_tool_n` | goal yaw | about K0's z |
| `slow_down` | speed scale | `1/slow_down`; a divider below 1 is refused |
| `carries_log`, `log_carrying`, `m_log`, `s_log_8`, `coll_shape`, `p_cyl_8` | payload | a tapered log becomes its *enclosing* cylinder, not the legacy mean radius |
| `check_log_collision`, `check_gripper_collision` | one collision flag | must agree while a log is carried |
| `q0`, `q0_dot` | explicit start | all eight canonical rows; all-zero means "use the measurement" |
| `publish_path` | -- | gates the `tcp_path` topic; the answer carries `tcp_path` either way |
| `t_end`, `v_d_tip`, `logs_scene` | -- | refused unless empty, each by name |

`tip_to_tcp_offset` is read from the model at the hanging equilibrium, never
written down. Since the hanging pose is closed form in boom and arm angles alone
it no longer depends on payload; the argument is kept so the call site stays
honest, and the result is *checked* to carry the requested yaw.

**The answer carries the canonical eight**, not the actuated six:
`trajectory_controller_a2b` lists the passive tip and tilt joints among its
`joints` -- it commands six and tracks eight -- so a six-wide goal is rejected.
The passive columns carry the sway the OCP predicts, which is what the trajectory
claims: on the way to the goal the tool is swinging.

`CalcMovement.Response` has no message field, so a refusal cannot say why. It
goes to the log and the trajectory is left empty rather than carrying whatever is
still standing -- a caller could not tell those apart, and the legacy server left
it empty too.

Three things the legacy server published that this does not: the `path_marker` /
`target_marker` / `collision_marker_array` overlays, the `plot_subplot`
(`plotter_msgs`) trace, and the latched `joint_trajectory` copy. Nothing
subscribes to any of them. `joint_trajectory` is omitted on purpose: it would
give this node a second `JointTrajectory` publisher, and that count is what the
launch contract reads as evidence the planner cannot be a second command
producer.

## Build and run

```bash
./ralph/verify.sh crane_model crane_planning
ros2 launch crane_planning crane_planner.launch.py
```
