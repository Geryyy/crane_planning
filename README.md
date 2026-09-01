# crane_planning

The `crane_planner` node. Two stages, one dependency that matters.

```
plan(goal position + yaw in K0_mounting_base)
  -> deterministic corridors   direct first, then lift/traverse/descend + lateral
  -> time-optimal OCP          acados, horizon as a state, endpoints only
  -> JointTrajectory on /crane/reference
```

No stochastic sampler. Everything the machine can do -- reach, hang, collide --
is asked of `crane_model`; nothing about the machine is written down here. Joint
ranges and velocity limits come from the `robot_description` the profile remaps
this node onto, the EOM from `crane_model.symbolic`. The one number the
description lacks (pump limit) is in `config/crane_planner.yaml` with its
evidence.

## The tool corridors

Each candidate is a TCP polyline in `K0_mounting_base`; yaw follows the shortest
arc over its Cartesian length. q8 is not planned -- the low-level controller
holds it at its start value.

Direct line first. Then transfer planes, starting above the highest scene
primitive plus the full clearance requirement and rising by
`corridor_height_step`; each plane gets a straight traverse plus configured
left/right offsets. First candidate that survives lifting, fitting and the
certificate wins. A fit failure advances to the next candidate rather than
ending the call.

**IK is the lift, not the check.** Each polyline sample is lifted into the five
planned coordinates by continuation IK seeded from its predecessor; the passive
pair is re-settled at every configuration, so what is solved is where the tool
actually hangs. The telescope's redundancy is taken up by a weak pull towards
the seed. Only the *goal* solve is cold, spreading `ik_restarts` over the
telescope range -- the coordinate the residual is flat in.

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
description. That retired the nine-corner gridding of the
sway box: one distance test answers the whole box, because the box is inside the
margin.

`radii` is over-estimated on purpose. Measured at full telescope extension where
every radius is largest; outermost point taken as TCP + `tool_radius` + the
carried body's reach; distance taken from each joint's *origin*, not its axis.
The prismatic telescope carries radius 1 -- it displaces what it holds by exactly
its extension. Too large costs samples, too small is unsound.

Self-collision tests at zero, not against `required`: the links are near each
other by design.

The certificate runs **once, on the fitted spline** (`Geometry.check_path`). The
lifted polyline is not checked -- it exists to establish a continuous IK branch
and the interpolation bound. The spline is a different curve and it is the curve
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

## The C2 fit

Least-squares cubic B-spline through the lifted configurations, `sigma`
distributed by how long each chord takes at its slowest axis's limit. Twice
differentiable because a kink is not a curve the machine can follow: `q_a''`
is unbounded there, so the step bound refuses it.

It approximates rather than interpolates. Interpolation ties segment count to
sample count, and those want opposite things -- the certificate wants samples
dense, the curve wants long end intervals for the clamp. A fixed
`path_segments` breaks the tie and stops the fit chasing IK noise.

Endpoints and end slopes are imposed exactly by writing four coefficients; the
rest are left as fitted. The chord total `L` scales the start slope --
`q_a'(0) = dq_a*L` is the same motion measured in sigma, so a plan starting from
a moving machine leaves along the direction it is already going.

`Path` carries `position(sigma)` and nothing else. No consumer wants a
derivative.

## The timing OCP

`ocp.py` builds one acados OCP on normalised time with the horizon `T` a state
whose derivative is zero: a single solve is time-optimal, no outer duration
search. Every residual row is divided by the limit it is measured against, so
1.0 is the bound on all of them alike.

It is handed the geometric stage's two endpoints and nothing else.

**No obstacle rows.** It moves between those endpoints however the dynamics
prefer -- where its speed comes from, and what voids the clearance proof the
stage above just produced: the certified curve is not the published curve.
Wiring the corridor is `docs/features/corridor-mpc/brief.md`, and it is not done.

Joint velocity/acceleration, reserved pump flow and the sway box are constraints
in the solve. Terminal sway offset and rate are checked after it against
`terminal_q_sway_max`/`terminal_dq_sway_max`; a near miss is refused, not
published.

## Cost

PZS100, `scripts/plan_example.py` in a clear scene at shipped defaults, on this
development image:

| | |
|---|---|
| lifted configurations | 177 |
| OCP solve | 0.18 s, 22 SQP iterations |
| planner call | 1.1-1.2 s |
| trajectory duration | 6.19 s |
| terminal sway | 0.29 deg, 0.000 rad/s |
| peak pump draw | 0.72 of the limit at kappa = 0.8 |

The geometric stage is nearly all of it: one IK solve plus one collision query
per lifted configuration. `margin_interp` is the knob -- halving it roughly
doubles the configurations.

## Tuning

`plan_example.py` runs the deployment's own `Planner` against the checked-in
description -- no ROS, no graph, no clock.

```bash
./scripts/plan_example.py --show
./scripts/plan_example.py --payload-mass 400 --csv plan.csv
./scripts/plan_example.py --margin-interp 0.02
```

Two views of one answer. Physical panels: joint positions, velocities,
`u = ddq_a`, sway offset and rate, pump draw, each beside its limit. Normalised
panel: every constrained row over its bound, so which row decided the duration is
one glance. Beside it, what the solve did -- iterations, four KKT residuals
against `ocp_tolerance`, chosen horizon, slack paid.

Read a regression off iterations and residuals, **never off solve time**: the
same problem measures 0.27 s idle and 4.97 s inside a running Gazebo.

It exercises every stage, so its refusal is the node's refusal. `--goal` takes a
joint configuration and puts its tool pose through the real IK; `--goal-pose`
takes a Cartesian pose, which may be unreachable.

## The node

| | |
|---|---|
| services | `/a2b_movement` (`timber_crane_planning_interfaces/CalcMovement`) |
| publishes | `/crane/reference` (`trajectory_msgs/JointTrajectory`, transient-local) |
| | `/crane_planner/planned_path` (`nav_msgs/Path`, visualization only) |
| | `tcp_path` (`nav_msgs/Path`) -- the legacy A2B server's name, for RViz |
| | `/crane_planner/markers` (`visualization_msgs/MarkerArray`, transient-local) |
| subscribes | `/joint_states`, `/robot_description`, `/crane/collision_scene`, `/crane/payload_estimate` |
| frame | `K0_mounting_base` throughout; nothing is converted |

**Not a second writer of the machine**: `/crane/reference` is a reference, not a
command; `crane_velocity_controller` stays the sole claimant of the six velocity
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

Display: `concrete_block_behavior_tree/rviz/cbs.rviz`, under *Planning and
Control*.

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
