# crane_planning

The `crane_planner` node. Two stages and one dependency that matters.

```
plan(goal position and yaw in K0_mounting_base)
  -> deterministic corridors  direct first, then lift/traverse/descend and
                               lateral alternatives, lifted and certified C2
  -> CasADi/IPOPT timing OCP   how fast that curve may be traversed
  -> JointTrajectory on /crane/reference
```

**There is no stochastic sampling planner.** The tool centre point first tries a
straight line. A blocked line triggers a bounded family of crane-specific
alternatives: lift to a transfer plane, traverse, descend, with symmetric lateral
corridors and successively higher planes. The common case stays deterministic and
cheap, while a runge in the direct corridor is no longer an automatic refusal.

Everything the machine can do -- reach, hang, collide -- is asked of
`crane_model`. Nothing about the machine is written down here: joint ranges and
velocity limits come from the `robot_description` the profile remaps this node
onto, the equations of motion come from `crane_model.symbolic`, and the one
number the description does not carry (the pump limit) is in
`config/crane_planner.yaml` with its evidence beside it.

## The deterministic tool corridors

Every candidate is a TCP polyline in `K0_mounting_base`; yaw follows the shortest
arc over its Cartesian length. Every sample is lifted into the five planned
coordinates by one least-squares inverse kinematics **seeded from its
predecessor**, and the whole machine is collision-checked there. The tool axis q8
is not planned; it is held by the low-level controller and rides the path at the
value it started at.

The direct candidate is first. Transfer planes begin above the highest scene
primitive plus the complete clearance requirement, then rise by
`corridor_height_step`. Each plane gets a straight traverse and configured
left/right offsets. The first candidate whose lifted polyline **and fitted C2
curve** pass the certificate is timed. A smoothing failure advances to the next
candidate instead of discarding a collision-free polyline and ending the call.

The inverse kinematics is not a closed form: the passive pair is re-settled at
every configuration tested, so what is solved is where the tool actually ends up
rather than where it would be if it did not hang. The redundancy the telescope
leaves is taken up by a weak pull towards the seed. Only the **goal** solve is
cold, and it spreads its `ik_restarts` over the telescope range because that is
the coordinate the residual is flat in.

Where the samples land is decided by the certificate below and not by a fixed
count: the step halves until the machine's own motion fits inside
`margin_interp`, and grows back by a quarter over a clear run so one tight corner
does not hold the rest of the walk at the step it needed. A small step for the
tool is not a small step for the arm -- near a singularity, or across the null
space the redundant telescope leaves, the tool can barely move while the boom
swings through it.

## The collision certificate

This is the part not to weaken by accident.

Sampling proves nothing on its own. What makes a finite set of checks a statement
about a continuous motion is a **margin paired with a Lipschitz bound**.
Rotating planned joint `j` by `dq_j` displaces any point below it by at most
`radii[j] * |dq_j|`, so between two configurations no point of the machine moves
further than

    step_bound = sum_j radii[j] * |dq_j|

Therefore: *if every checked configuration clears the scene by more than
`required`, and `step_bound <= margin_interp` for every consecutive pair, then no
point of the machine touches anything anywhere in between.* Both halves are
needed and neither is decoration. Drop the margin and the bound bounds nothing;
drop the bound and the samples are just samples, and a thin obstacle can be
tunnelled.

The margin is spent three ways, and each part is spent once:

    required = margin_safety + margin_interp + envelope

| | |
|---|---|
| `margin_safety` | what is actually left over as clearance in the answer |
| `margin_interp` | what pays for the gap between two checked configurations |
| `envelope` | `l_tool * sin(q_sway_max)` -- how far the tool can swing inside the admissible box |

The envelope is **computed, not configured**: the pendulum length is read out of
the description at the tool the profile selected. That is what retired the
nine-corner gridding of the sway box that used to run at every pose -- one
distance test now answers the whole box, because the box is inside the margin.

`radii` is over-estimated on purpose, in the safe direction. It is measured at
full telescope extension, where every radius is largest; the outermost point of
the machine is taken to be the tool centre point plus `tool_radius` plus the
carried body's own reach; and the distance is taken from each joint's *origin*
rather than its axis. The telescope is prismatic and carries a radius of one --
it displaces what it holds by exactly its own extension. Too large costs samples;
too small is unsound, which is the asymmetry that decides every one of those
choices.

Self-collision keeps its own test at zero rather than against `required`. The
machine's links are near each other by design, and holding them a margin apart
would refuse poses it is built to reach.

The certificate is run **twice**: once on the lifted polyline as it is walked,
and again by `Geometry.check_path` on the fitted spline. The polyline satisfies
it by construction; the spline through it is a different curve, and it is the
curve that executes.

## Why the lift marches instead of bisecting

Five planned coordinates against a four-DOF tool task leaves one redundant
degree of freedom, so every pose on the line is reached by a one-parameter family
of configurations, and the inverse kinematics picks one of them by continuity
from its seed.

That makes the **goal configuration an output of the walk, not an input to it**.
Solving the goal independently picks its own point on that family, and between
two points of the family there is a finite jump no refinement can close: bisect
towards it forever and the gap stays. So what is specified here is the goal
*pose*, the end configuration is whatever marching to it produces, and the cold
multi-restart solve at the top of `plan` is used only to establish that the pose
is reachable at all -- its answer is discarded.

A genuine inverse-kinematics branch change still cannot pass unnoticed. It is a
large joint step for an arbitrarily small tool step, so `step_bound` refuses it,
the step halves, and it keeps halving until it hits `MIN_LIFT_STEP` and the plan
is refused by name. The resolution test and the continuity guard are one test.

## The hanging pose is a closed form

    q_eq = (pi/2 - q_boom - q_arm, pi/2)

A two-hinge pendulum under gravity hangs straight down, so the tip joint takes up
whatever the boom four-bar accumulated and the tilt joint does not move at all.
`crane_mpc/src/mpc_node.cpp` measured this against the general Newton-plus-grid
solve it replaces and found it good to 1e-4 rad across the workspace, and
independent of slew, telescope, rotator, tool and payload -- against a sway box
half-width of 0.2 rad.

The general solve costs 22.6 ms and this costs two subtractions. That difference
is the whole reason the pendulum can be settled at *every* configuration the lift
checks, which is what makes the geometry above answer where the tool actually
hangs rather than where it would be if it did not.

## The C2 fit

A clamped cubic spline through the lifted configurations, with `sigma`
distributed by how long each chord takes at its slowest axis's own limit. Twice
differentiable, and that is the entire point: the timing stage writes
`ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot`, so at a kink `q_a''` is unbounded,
the admissible path rate collapses to zero and the machine stops dead at every
waypoint.

The chord total is a duration, and it also pins the start rate. Choosing
`q_a'(0) = dq_a * L` makes `dq_a = q_a'(0) sigma_dot(0)` hold exactly at
`sigma_dot(0) = 1/L`, so a plan that starts from a moving machine has a boundary
condition rather than a residual to accept or refuse.

## The timing OCP

One NLP over `sigma`, solved with IPOPT. The independent variable is the path
parameter and not time; minimising traversal time is minimising the integral of
`dsigma/sigma_dot`, so the end time is **free** -- that integral *is* the end
time.

| | |
|---|---|
| state | `(sigma_dot, q_u, dq_u)` -- the path rate and the two sway coordinates |
| control | `sigma_ddot` |
| dynamics | `ddq_u = -M_uu^-1 (M_ua ddq_a + h_u)`, every derivative divided by `sigma_dot` |
| cost | `int dsigma/sigma_dot` + `w_sway int ||dq_u||^2 dt` + `w_tau int ||tau_a/tau_hold||^2 dt` + `w_u int sigma_ddot^2` + terminal sway + flow slack |
| hard | `ddq_a`, `q_u` inside the sway box, `dq_u` bounded, `sigma_dot` under the per-node velocity ceiling |
| hard with slack | summed pump flow, at **every** node |
| boundary | `q_u` and `dq_u` measured at the start; the goal's rest condition is a cost |

The sway is a *state* and not an afterthought, which is what makes the tool
arrive still. Integration is RK4 in `sigma`, so the path is evaluated at interval
midpoints as well as at nodes. The rows are evaluated at every node **including
the last**: the terminal node carries no input of its own but does not need one,
because the path arrives at rest in `q_a` and `q_a'(1) sigma_ddot` vanishes.
Leaving it out is how a plan ends with nine times the admissible deceleration.

### Terminal sway is a cost

One control -- the rate along a path the OCP may not leave -- cannot in general
drive four terminal quantities to zero. Asking for it as a hard equality does not
produce a slower trajectory; it produces `Infeasible_Problem_Detected`. So it is
priced, normalised by the admissible box, which makes `terminal_sway_weight` a
multiple of "the whole allowance" and lets the answer report the residual in
degrees. The converged result is still refused unless it finishes inside the
configured 0.02 rad displacement and 0.04 rad/s settled-rate bounds.

### Effort is priced, not bounded

`tau_a` is the actuated generalized force of `wiki/robot_model.md` 3.3. It enters
the cost normalised per request by what the *same path costs to hold* -- gravity
on these configurations at this payload, read off the model -- so `tau_weight` is
dimensionless and O(1), and what it prices is the dynamic effort a fast traversal
adds on top of the pose's own gravity load. That is how the machine's inertia
gets into the answer: the optimiser feels what it costs to accelerate a loaded
telescope.

It replaced a hard cylinder-force row whose limit was the smaller chamber area
times a relief pressure **nothing in this workspace has ever measured**.

**The consequence is real and is the trade: this planner no longer refuses a move
it cannot lift. It plans it, and the machine stalls on it.** The pump is now the
only actuator limit enforced, which is what raises the value of actually
measuring it. Restoring a real force bound needs a reading off the power pack or
the identification campaign of `hydraulics.md` 5.8 -- both human-only.

### The pump carries L1 slack

Hard, but with a penalised slack variable per node. A move that needs a few
percent more flow than the reservation allows should come back slower and say so
in its message, not come back as a refusal. `flow_slack_weight` is large enough
that the slack stays at zero whenever zero is attainable; the message reports the
overrun when it is not.

### The initial guess

Three passes, and all three earn their place. The velocity ceiling says nothing
about acceleration, so the classical `sqrt(ddq_max / |q_a''|)` curve goes on top
of it. Neither says anything about pump flow, which is what actually binds on a
lift, so the rate is bisected down until the worst node sits at 90% of its
allowance -- 90% and not 100% because a barrier method started with a dozen rows
at zero slack is a barrier method that does not start. And per-node feasibility
is not reachability, so a forward and a backward sweep under
`d(sigma_dot^2)/dsigma = 2 sigma_ddot` connect the nodes to each other.

`speed_scale` takes every *rate* budget with it -- joint velocity, the path rate
and the pump -- and acceleration goes with its square, because along a fixed path
`ddq_a` is quadratic in the rate. Scaling the pump is what makes a divider mean
anything on this machine: these moves are flow-limited long before they are
velocity-limited, so a scale that only touched joint speeds would hand a caller
asking for half speed the very same trajectory back.

## What it costs

PZS100, the checked-in offline example in a clear scene at the shipped defaults
(`intervals = 40`, `margin_interp = 0.10`), measured on this development image:

| | |
|---|---|
| lifted configurations | 177 |
| IPOPT | 4.9 s over 41 iterations |
| planner call | 8.3 s |
| trajectory duration | 5.44 s |
| terminal sway | 0.02 deg off rest |
| peak pump draw | 0.80 of the physical limit |

Halving `margin_interp` roughly doubles the lifted configurations and so the
geometric stage. Coarsening `intervals` does **not** buy time the way it looks
like it should: at `intervals = 25` the solve hits the CPU-time cap and returns
nothing, because a coarser RK4 on the pendulum makes the NLP harder, not smaller.
40 is a solved point, not a round number.

## Tuning it

`scripts/plan_example.py` runs the deployment's own `Planner` against the
checked-in machine descriptions -- no ROS, no graph, no clock -- and plots what
the answer asks of the machine: joint positions, velocities and accelerations,
the path rate, the sway offset and rate, the actuated force and the pump flow,
each against its own limit.

```bash
./scripts/plan_example.py --show
./scripts/plan_example.py --sway-weight 8 --intervals 60 --output slow.png
./scripts/plan_example.py --tool epsilon7040 --payload-mass 400 --csv plan.csv
./scripts/plan_example.py --margin-interp 0.02 --terminal-sway-weight 500
```

It exercises every stage, so a refusal from it is the refusal the node would
give. `--goal` takes a joint configuration and puts its tool pose through the
real inverse kinematics; `--goal-pose` takes a Cartesian pose instead, which may
turn out to be unreachable.

## The node

| | |
|---|---|
| services | `/a2b_movement` (`timber_crane_planning_interfaces/CalcMovement`) |
| publishes | `/crane/reference` (`trajectory_msgs/JointTrajectory`, transient-local) |
| | `/crane_planner/planned_path` (`nav_msgs/Path`, visualization only) |
| | `tcp_path` (`nav_msgs/Path`) -- the legacy A2B server's own name, for RViz |
| | `/crane_planner/markers` (`visualization_msgs/MarkerArray`, transient-local) |
| subscribes | `/joint_states`, `/robot_description`, `/crane/collision_scene`, `/crane/payload_estimate` |
| frame | `K0_mounting_base` for every goal and every published pose; nothing is converted |

The PZS100 Gazebo actuator reports q9 as total opening with an EPSCOPE
`state_factor` of two, while the URDF and collision model use one rail's joint
coordinate. Its bringup already publishes the canonical conversion on
`/joint_states_rviz`; launch this planner with
`joint_states_topic:=/joint_states_rviz`. A raw reported value such as 1.261858
then becomes the actual 0.420929 m rail coordinate. The planner refuses either
coordinate outside the URDF range and never projects it.

`/a2b_movement` is the only service. The native `/crane/plan_motion` it used to
sit beside had no caller anywhere in the workspace and was removed; the reference
it published is unchanged, so anything reading `/crane/reference` reads the same
topic it always did.

It is **not a second writer of the machine**: `/crane/reference` is a reference
and not a command, `crane_velocity_controller` remains the sole claimant of the
six velocity command interfaces, and this node holds no `controller_manager`
client of any kind.

`/joint_states` comes from `joint_state_broadcaster`, actuated and passive joints
alike, at depth 10 because two producers publish partial messages back to back
and a depth-1 queue can drop one of them for good. Messages are cached and read
by joint name, never by index. The passive pair has its own freshness deadline:
`sensor_msgs/JointState` has no `valid` flag, so age is the whole test, and a
stale or absent sway measurement is a refusal rather than a zero.

Every refusal names itself and leaves the standing reference alone: a goal that
cannot be reached, no deterministic corridor surviving its fitted-path
certificate, and a path that cannot be timed are three different answers.

## What the tool is carrying

A payload is handed to the collision check as a body marked `attached_to_tool`,
and that flag is two rules that only work together. It is **not** checked against
the links that hold it -- a gripped block sits inside the gripper, which is what
gripping is, so checking it there refuses every pose the machine can reach. And
it **is** checked against the rest of the scene, which a scene body otherwise
never is -- because what a carried body is for is hitting the world, and a
payload checked only against the machine carrying it has not been checked at all.

Which links count as holding it is read off the description, not written down: a
body is carrying if the joint that places it is at or below the joint a payload
mounts on. On the PZS100 that is the rotator lower part, the tool centre point
and the two rails, and nothing else.

The body is placed **at every configuration checked**, not once at the start. A
payload pinned to the pose the machine set off from is a ghost standing in the
start pose while the real one rides the tool through the scene unchecked. Its
own reach also enters `radii`, so the step bound covers what is in the gripper
and not just the gripper.

## Seeing the plan

`/crane_planner/markers` draws four things, and each answers a question the
numbers do not. The **path** the tool takes, as one line. The **tool** swept
along it -- the segment from the tip hinge to the tool centre at samples down the
trajectory, which is the pendulum, so its lean is the sway the OCP planned and a
plan that swings looks like it swings. The **goal**, where the request asked for
the tool and which way round. And the **scene the planner actually checked
against**, which is not the scene anyone published: by that point the reserved
`truck` primitive has become a bed, six runges and a headboard, and what is in
the gripper is drawn where it will actually be. Structural bodies, perceived ones
and the payload are three colours, because "why was this refused" is usually
answered by which kind it hit.

The goal and the geometry go out **before** the solve, so a refusal leaves them
on screen: "it said no" and "it said no, and here is the runge it would have hit"
are very different messages.

`concrete_block_behavior_tree/rviz/cbs.rviz` carries the display, under
*Planning and Control*, beside the plain `nav_msgs/Path` traces.

## The truck

The scene carries one primitive with the reserved id `truck`, and `expand_truck`
turns it into what the tool can actually hit: the bed's top face, six runges
standing on it and the headboard closing its cab end. The crane is bolted to the
vehicle, so the box itself is not an obstacle. The dimensions are the vehicle's
and are configured; the position is measured and arrives with the primitive, so
moving the truck moves all of it.

A runge is placed flush against the bed edge rather than centred on it, which is
where the description puts the real post's outer face -- the configured section
is an inflation of that post, so it has to grow inboard. The headboard sits flush
against the box's +x face the same way, which is the end the outermost runge
station is at.

## The `a2b_movement` compatibility service

`/a2b_movement` is served as a drop-in replacement for the legacy
`a2b_ilqr_server`. It is an **adapter, not a second planner**: `a2b.py` maps the
`CalcMovement` request onto the native `Planner.plan` call and `node.py` runs it,
so there is one start state, one scene, one kappa and one `/crane/reference`
publication.

| `CalcMovement` | native | rule |
|---|---|---|
| `y_n` | goal position | the **tip pivot K5**, plus the settled tip-to-tool offset read out of the model |
| (no field) | goal frame | `K0_mounting_base`, asserted and never converted |
| `phi_tool_n` | goal yaw | about K0's z |
| `slow_down` | speed scale | `1 / slow_down`; a divider below 1 is refused |
| `carries_log`, `log_carrying`, `m_log`, `s_log_8`, `coll_shape`, `p_cyl_8` | payload | a tapered log becomes its *enclosing* cylinder, not the legacy mean radius |
| `check_log_collision`, `check_gripper_collision` | one collision flag | they must agree while a log is carried |
| `q0`, `q0_dot` | explicit start | all eight canonical rows; the all-zero default means "use the measurement" |
| `publish_path` | -- | gates the `tcp_path` topic; the answer carries `tcp_path` either way |
| `t_end`, `v_d_tip`, `logs_scene` | -- | refused unless empty, each by name |

`tip_to_tcp_offset` is read out of the model at the hanging equilibrium and never
written down: the PZS100's rail gripper and the 7040's jaw do not hang at the
same offset. Because the hanging pose is a closed form in the boom and arm angles
alone, the offset no longer depends on the payload -- the argument is kept so the
call site stays honest about what it is asking for, and the result is *checked*
to carry the requested yaw rather than assumed to.

**The answer carries the canonical eight**, not the actuated six: the
`trajectory_controller_a2b` this trajectory is fed to is configured with the
passive tip and tilt joints among its `joints` -- it commands six and tracks
eight -- so a goal naming only six is a goal it rejects. The passive columns
carry the sway the OCP *planned*, which costs nothing because the OCP solves for
it anyway, and is what the trajectory actually claims: on the way to the goal the
tool is swinging.

`CalcMovement.Response` has no message field, so a refusal cannot say why in the
answer. It goes to the log, and the trajectory is left empty rather than carrying
whatever is still standing -- a caller could not tell those two apart, and the
legacy server left it empty too.

Three things the legacy server published that this does not: the `path_marker` /
`target_marker` / `collision_marker_array` RViz overlays, the `plot_subplot`
(`plotter_msgs`) trace, and the latched `joint_trajectory` copy. Nothing in the
workspace subscribes to any of them. `joint_trajectory` is left out on purpose
rather than merely unimplemented: it would give this node a second
`JointTrajectory` publisher, and that count is what the launch contract reads as
evidence that the planner cannot be a second command producer.

## Build and run

```bash
./ralph/verify.sh crane_model crane_planning
ros2 launch crane_planning crane_planner.launch.py
```
