# crane_planning

The `crane_planner` node of `wiki/implementation/ros2_interfaces.md` §2: it
serves `/crane/plan_motion` and `/crane/plan_grip` and publishes the trajectory
it answered with on `/crane/reference`. §10 keeps the two services separate while
the tested task layer migrates, and they are one node, one model and one clock.

It grew out of the **slice-5 tracer bullet** of `docs/features/cbs-arch/prd.md`
§2 — one goal pose in, one timed joint trajectory out, through a real node and a
real service. Both stages of `wiki/trajectory_planning.md` are now there in full —
stage 1's geometry and stage 2's path-constrained OCP — and the point of this
package is that every remaining absence is *refused or named*, never approximated.

> **Build note.** This package needs **OMPL** (`libraries.md` §1) for §4.4's
> sampling fallback, and the devcontainer image does not carry it yet: add
> `ros-humble-ompl` — what `rosdep resolve ompl` returns for this package's
> `<depend>ompl</depend>` — to `.devcontainer/Dockerfile.vscode`'s apt block.
> `find_package(ompl REQUIRED)` is deliberate: a planner that quietly compiled
> without its fallback is worse than one that does not build.

## What it does

| Step | Where |
|---|---|
| endpoint from a goal pose | the equilibrium-**constrained** NLP of `wiki/robot_model.md` §2.2, over all eight coordinates |
| passive pair | a decision variable under `g_u(q) = 0`, so the endpoint is a genuine steady state |
| initial guess | the semi-analytic IK of §2.2 steps 1–5, with the passive pair pinned |
| redundancy | the scalar search over `d45` of §2.2 step 3, scored by joint-range centring |
| acceptance | the forward-kinematics residual of §2.2, on **every** call, before the result is used |
| assertion | the answer's passive pair against `Model::passive_equilibrium`, through the frozen API |
| geometric path | the structured lift/traverse/descend primitive of `trajectory_planning` §4.4, built C² |
| transfer altitude | derived from the endpoints and the mounted tool's own reach, never configured |
| collision check | the middle step of §4.4's generate–check–accept, against the scene, the truck and the crane itself |
| sway | §4.3's envelope, at the `q_sway_max` of `mpc` §3 constraint 3, resolved rather than inflated |
| fallback | §4.4's second mechanism, RRT-Connect over `q_a`, reached **only** when the primitive is blocked |
| smoothing | §4.5's mandatory shortcut → C² fit → re-check, on every sampled path and on no primitive |
| timing | the path-constrained OCP of §5.2, solved with acados over `crane_model`'s CasADi graph |
| sway in the timing | `q_u` is a **state** of that OCP, so §5.4's "arrive hanging still" is an imposed terminal condition |
| force and flow | the graph's own output-map rows — the expressions `mpc` §3 constraints 6 and 7 are written from |
| margin | §5.5's κ, applied to every physical limit, reserved for the MPC and not spendable by a caller |
| a grip's descend and lift | every row above, unchanged, except that a descend lowers the transfer altitude's ceiling to its own endpoints |
| a grip's close and open | §8's **retained** cosine primitive on `q8` alone, C² at both ends, on the arm's own time base (§4.1) |
| the start state | `(q, dq)` as **measured** over all eight coordinates — §7, and not the stopped start it calls a defect |
| a moving start | a boundary condition on the geometry and on the OCP, so a seam matches in position *and* velocity |
| an unmeasured sway | `passive_estimate_policy`: refused by name, or planned inside a reduced box with the reserve stated |
| latency | §7's bound, `latency_budget`, charged stage by stage **inside** the planner and not in a caller's timeout |
| an overrun | the previous trajectory keeps standing on `/crane/reference`, and the answer says which one it is |

**Which of §2.2's two formulations runs is this planner's decision, not a
parameter.** §2.2's closing paragraph is the rule — the semi-analytic route where
speed matters, the equilibrium-constrained one where the endpoint must be
sway-free — and every `/crane/plan_motion` goal is a *placement* goal, so every
endpoint takes the constrained route and the fast route runs inside it as the
initial guess. There is no switch for it, because a caller who picked the fast
one would be choosing to have the tool arrive swinging without being told that is
what the choice meant.

The constraint is `g_u(q) = inverse_dynamics(q, 0, 0, payload)` on the passive
rows, which the model API contract §7 states is zero exactly at a valid passive
equilibrium — so no second model and no new dependency. It is *not*
`Model::passive_equilibrium`, and that is what gives the final assertion its
teeth: `g_u(q) = 0` also has the tool **standing up** as a root, which is an
equilibrium and is not a steady state anything settles into, and the solver's own
constraint residual cannot tell the two apart. `passive_equilibrium` returns the
settled root, so checking against it is a stability check rather than a
restatement of what the solver already believes.

A solve costs what `Model::passive_equilibrium` costs — some 33–45 ms against
0.35 ms for a forward-kinematics call — so the constraint being cheaper than the
equilibrium is the whole reason an eight-coordinate NLP is affordable here. One
endpoint takes 0.25 s on the PZS100 and 0.73 s at its worst on the Epsilon 7040.
Both an iteration cap (`nlp_max_iterations`) and a wall-clock cap
(`nlp_max_wall_clock`) bound it, and hitting either is a refusal naming the cap,
never the best iterate dressed as an answer — an unbounded NLP inside a service
call is a hang, and a caller cannot cancel one. These are the numbers the
per-request latency bound below was sized against, and the IK is the first stage
it charges.

The seed matters and is measurable: on the PZS100 `dq_eq/dq_a` is
`[0, -1, -1, 0, 0]` exactly — the tool hangs vertically whatever the arm does —
so the semi-analytic answer is already feasible and the NLP converges in a single
iteration. The 7040's tool sits off the tilt axis, its seed is only close, and
the worst of twenty five sampled goals took 24 iterations.

## The structured primitive, and where its transfer altitude comes from

Most crane moves are *lift, traverse, descend*, and `trajectory_planning` §4.4
makes that the first mechanism tried: it is cheap, deterministic, and smooth by
construction, where a sampling planner is stochastic with an unbounded runtime.
§4.4's order is **generate, check, accept** and all three happen inside one call,
`build_structured_primitive`. A blocked primitive is a refusal naming which of the
three phases failed, and `plan_motion` is where that refusal turns into §4.4's
second mechanism — see *The fallback* below.

The path is built **C² by construction and never smoothed into C² afterwards**.
Each phase is a boundary-value problem solved with `ruckig`
(`libraries.md` §1: "jerk-limited interpolation … boundary-condition solves"),
and the next phase starts from exactly the velocity and acceleration the previous
one ended at, so continuity is a property of the construction rather than of a
tolerance. §4.5 is why that matters and the reason is behavioural, not numerical:
at a point of discontinuous curvature the path-velocity limit collapses to
`σ̇ = 0`, so the machine **stops dead at every waypoint** — the worst possible
output for a crane whose purpose is smooth, sway-free motion. A jerk-limited
profile has continuous acceleration by definition, which is what makes `q_a''(σ)`
*defined* everywhere rather than only away from the junctions.

**The transfer altitude is derived, not configured.** §9's open items ask for
exactly that, and it is the one thing the legacy planner hard-codes. The
derivation lives in `derive_transfer_altitude` and nowhere else:

    z = max over endpoints of ( z_tcp + | p_tool − p_rotator | )

so at the transfer altitude everything hanging below the rotator bearing sits at
or above the altitude the TCP held at the higher endpoint. The tool's reach is
read off the model at that endpoint's own configuration, which makes it a
property of the mounted tool: **0.7705 m on the PZS100's rail gripper and a
different number on the Epsilon 7040's jaw.** The two are not even described to
the same depth — only the 7040 carries a `tool_contact_point` link, and
`Frame::ToolContact` is `FrameUnavailable` on the PZS100 by the model API
contract's own design — so the answer records *which* frame the clearance was
measured to instead of leaving a caller to assume.

Two bounds a deployment may configure, both empty by default:
`transfer_altitude_floor` may only raise the derivation and
`transfer_altitude_ceiling` may only lower it. Neither can *be* the answer: with
no endpoints there is no altitude at all, whatever is configured. A ceiling that
takes the altitude below both endpoints does not shorten the lift, it refuses it.

The obstacle term of the derivation is **a named gap when there is no scene, not
a zero**. `scene_extent` reads the tallest primitive in the subscribed scene —
including the runges the truck pose was expanded into, which is exactly what a
transfer altitude has to clear — and `scene_without_obstacles()` is what a
collision-blind request gets: `obstacles_known == false` with a NaN extent, which
is not the same claim as "there are no obstacles". An *empty* scene is still
`obstacles_known == false`, so an altitude is never raised by a claim nobody
made.

The four numbers the closure needs — `a2`, `a3`, `d45(q4)` and the bearing they
are measured from — are **probed out of the model** at startup rather than
written down, and the probe is also a check: a description whose arm is not
planar, or whose telescope does not extend affinely in `q4`, fails to start with
the residual in the message. The joint position and velocity limits come out of
the same `robot_description` XML, because the frozen model API does not expose
them and a limit restated beside a node is a limit that drifts away from the
description the controllers were configured against.

## The fallback, and why it is second

`trajectory_planning` §4.4 names two mechanisms and fixes their order: generate
the primitive, check it, accept it if clear, **otherwise** sample. `plan_motion`
is where that order lives, and §4.4's `[!important]` is why it is an order and not
a preference — a sampling planner is stochastic and its runtime is not bounded, so
primitive-first is what buys deterministic latency in the common case and leaves
completeness to the rare one. Nothing runs the fallback beside the primitive, to
compare with it or to "check" it; a fallback that runs anyway spends exactly what
the ordering was for, and there is a test that asserts a clear scene comes back
with the primitive's own path and **zero** configurations sampled. The other half
of the order is asserted through the same `plan_motion` call and the same request,
with only the scene moved onto the path: the primitive is refused, the search runs,
and what comes back is the sampled path itself — the geometry the trajectory was
timed along, with the primitive's refusal kept beside it.

Which of the two answered is on `MotionPlan::mechanism` and in the service
`message`, because a caller cannot otherwise tell a lucky deterministic plan from
a sampled one, and they are not the same product.

**RRT-Connect, and the reason, which closes §9's open item.** `libraries.md` §1
lists OMPL for sampling-based path planning against "RRT/PRM and their smoothing",
so the library was never a decision; which planner out of it is. It is
feasibility-only and bidirectional: it stops at the first solution rather than
improving one, so a solve ends when it has an answer instead of spending its whole
budget — which is what a per-call wall-clock cap needs — and growing from both ends
matters here because the goal is a *placement*, down among the runges, in exactly
the narrow passage a single tree explores last. Not RRT\*, BIT\* or any
asymptotically optimal planner: they are anytime, they use the whole budget by
construction, and their answer is a function of how long they were given. Not PRM:
a roadmap pays for itself over many queries against a static scene, and
`/crane/collision_scene` is republished as the site changes. Not MoveIt's pipeline,
which is `libraries.md` §5's *deliberately absent* row against "Pinocchio + Coal +
OMPL directly" — its SRDF format is not rejected and `crane_model` reads one.

**Five coordinates, never the passive pair.** §4.1's `[!warning]` is that a
sampling planner handed a joint group with `q5`/`q6` in it interpolates them as
free variables and returns paths that satisfy every joint limit while being
dynamically impossible, and that the legacy MoveIt SRDF declares exactly such a
group. So the state space is five dimensional, built from `JointLimits` — the
description's own range — and the passive pair is *solved* at every checked
configuration through `Model::passive_equilibrium` rather than sampled. The tool
coordinate `q8` is not in it either, for §4.1's other reason. The metric is not a
norm over a vector that mixes radians with metres: each axis is a subspace weighted
by `1/dq_max`, so a distance is a sum of times at each axis's own limit — the unit
`geometric_path.hpp` already distributes σ by. The rotator is a `continuous` joint
in both descriptions and therefore has no range to take; it is searched over what
the two endpoints ask for widened by half a turn either way, which is every
distinct pose it has because `q7` and `q7 + 2π` are the same geometry.

**The check is issue 041's.** `isValid` is `check_configuration`: the scene, the
truck keyed to its measured pose, the crane against itself, and §4.3's sway
envelope. Motion validation subdivides at the **same** stated resolution
`check_path` samples at, measured with the same `watched_frame_travel_m` — OMPL's
own validator would have subdivided in state-space distance, which says nothing
about how far the geometry moved, and half a radian of slewing moves the tool much
further with the boom out than with it in.

**Smoothing is mandatory and so is re-checking it.** §4.5, all three steps and in
order: shortcut, then the same `fit_c2_path` the primitive is built with, then
`check_path` over the whole curve — and a path that fails the last one is refused.
The third step is not a formality. The first two both move the path off the
polyline the search cleared: the shortcut replaces a detour by its chord, and the
fit is monotone inside each waypoint box but is not the chord between two
waypoints. So what was cleared is not what would be flown, and only the re-check
answers for what would be. The primitive needs none of this and §4.5's last
sentence says so.

**Bounded, and deterministic.** Two per-call caps — `sampling_time_budget` and
`sampling_max_validity_checks`, the second being the half that does not depend on
how fast the machine running the planner is. Exhausting either is a refusal naming
it, and an *approximate* path that stops short of the goal is refused with the
rest: it is not a worse answer than none, it is an answer that reads as success.
`latency_budget` is the bound over the *whole* request and is not this cap; this
one is the search's own share of it, and the two are charged separately. The
search draws from this package's own `std::mt19937` seeded by `sampling_seed` and
not from OMPL's global generator, which can be seeded once per process and would
leave whichever test ran second irreproducible; the nearest-neighbour structure is
linear for the same reason — OMPL's default GNAT picks pivots from that same global
generator, and the tree is small enough that an exact linear search is free beside
one 39 ms validity check.

## Collision, the truck, and the sway envelope

`avoid_collisions=true` — what `crane_msgs/PlanMotion` defaults to — is honoured:
the path is checked against the scene on `/crane/collision_scene` (reliable,
transient-local, `K0_mounting_base`, and a scene in any other frame is refused
naming both), against the truck, and against the crane itself on `crane_model`'s
derived allowed-collision list. `avoid_collisions=false` still plans and says in
`message` that nothing was checked. Every distance is a `Model::collision_query`,
so there is no second collision model here to drift from the one the kinematics
came out of, and `collision_queries` is not a real-time call — the planner is an
on-demand node and nothing here puts it on a control loop.

**The truck is a pose, not three numbers in this YAML.** The world model
publishes one primitive with the reserved id `truck` carrying the measured
vehicle box; `expand_truck` turns it into the bed slab and the six runges of
`trajectory_planning` §4.2 at the legacy dimensions (0.28 × 0.31 × 2.12 m, two
rows of three), every piece `structural = true`, which is that field's first
consumer. Move the truck and the runges move with it. What *is* configured is a
property of the vehicle — how big a runge is and where along the bed the three
stations are — and never a position in K0. The bed enters as a slab at the top
face of the vehicle box rather than as the box, because the crane is bolted to
the front of that vehicle and checking the mounting base against the whole of it
would report the crane colliding with itself.

**The sway envelope is resolved, not inflated.** §4.3 asked for the tool geometry
inflated by `Δ_sway = l_tool·sin(max|q_u^+|)`; the frozen model API has no
inflation parameter, and it does not need one. At each checked configuration the
hanging pose is queried first, and if the smallest distance to the *scene* then
exceeds `Δ_sway` no admissible sway can reach it — that is §4.3's test used as a
*sufficient* condition and it settles the free-space majority in one query. Only
where it fails is the envelope gridded and queried exactly, offset from
`Model::passive_equilibrium` because `mpc` §3 constraint 3 is a box around the
equilibrium and not around zero.

**The envelope covers the scene; the crane covers itself at the hanging pose.**
Self-collision is checked at every configuration and not re-checked at each sway
pose, and the reason is the machine: the PZS100's rail gripper sits 25 mm from
the inner telescope at rest and 0.2 rad of sway closes that gap at every
configuration with the boom up. Refusing on it refuses every path — §4.3's own
warning — on the error of a single enclosing box rather than on anything
measured, and it says nothing a caller can act on, because how close the tool
comes to the crane when it swings depends on `q_a` and the bound alone. What that
25 mm constrains is `q_sway_max` itself, which is `mpc` §3's to impose and the
virtual working cell's to bound.

**`q_sway_max` lives here and nowhere else.** It is this package's read-only
parameter, `[tip, tilt]`, and it is the same `q_u^+` `mpc` §3 constraint 3
imposes as a state box — so "any sway the MPC permits is sway the path was
cleared for" is true by construction. `crane_supervisor`'s virtual working cell
consumes the same number and, by `control_architecture` §5.1's own rule, derives
its envelope independently of this planner's data.

**The resolution is stated and related to the geometry.** `check_resolution` is
the largest distance a watched frame may move between two checked
configurations, and it is tightened to at most half the thinnest extent in the
scene — a step larger than the thinnest primitive can cross it between samples,
and a two-dimensional sway grid can do it diagonally. A scene needing a step
finer than `min_check_resolution` is refused rather than checked coarsely. It is
a sampling rule, not a proof: the model answers for a configuration, not for the
sweep between two.

**A refusal says what is in the way.** `CollisionResult::other_id` and both
witness points reach `message`, with the sway pose it was found at when the
envelope had to be resolved, so an operator is told which runge or which link
pair blocked and where.

## What it does not do, and what does

- **Payload against obstacle is not checked, and that is the model's boundary,
  not an omission here.** The payload enters as a scene primitive with the
  reserved id `payload` at the K8 pose read out of `forward_kinematics`, and
  `Model::collision_queries` checks every scene primitive against the crane's
  links — so the payload is checked against the crane, and the crane against the
  obstacles, but *primitive against primitive* is not a query the frozen API
  offers. A payload set down beside a stack is therefore cleared on the arm's
  geometry and not on the load's. Closing it means either a scene-primitive pair
  query in `crane_model` or a second geometry path here, and the first is the
  right one.
- **Issue 038's velocity-limited ramp no longer answers anything.** The
  trajectory `/crane/plan_motion` returns and `/crane/reference` republishes is
  §5.2's, on every call and with no path back to the ramp — a timing that could
  silently degrade to one that ignores force and flow would put exactly the
  reference §5.3 calls a planner bug on the wire. `scaled_ramp_along_path` stays
  in `trajectory_timing.hpp` because `TimedTrajectory` and its sampling are what
  the OCP's grid is resampled onto, and it is still covered by its own tests;
  nothing in the planning path calls it.
- **Clearance is half the redundancy score, and only where the redundancy is
  resolved.** `robot_model` §2.2 step 3 scores the telescope's one leftover
  degree of freedom by joint-range centring *and* collision clearance, and both
  rows live in `include/crane_planning/redundancy.hpp` so the two endpoint routes
  spend that freedom the same way. With no scene the clearance row does not vote,
  which is not the same as scoring it zero.
- **The transfer altitude's clearance is a lower bound on the tool's true
  envelope.** The deepest tool frame either description carries is the contact
  point or the TCP; the rail's and the jaw's tips live in links the frozen
  `Frame` enum does not name (the PZS100's rails run a further metre from `K9`,
  the 7040's jaws 0.81 m), so the derivation cannot see them and must not invent
  them. The *collision* check does see them — its geometry is the fitted link
  primitives, not the frame tree — so an altitude that looks generous and a path
  that is refused are consistent, not contradictory. A deployment that knows its
  site needs more says so with `transfer_altitude_floor`.
- **A smoothed path the re-check refuses is a refusal, not a retry.** The
  fallback does not back the smoothing off segment by segment and try again: it
  says the smoothed curve is blocked where the sampled one was clear and stops.
  That is the honest answer for a per-call planner and it is also the one place
  the fallback can fail after having found a way round. A graduated retry belongs
  with whatever decides *when* to re-plan, which is the supervisor's and the task
  layer's; this node answers requests and says why it could not.
- **Executing a grip is not this node's.** `/crane/plan_grip` answers one phase
  with a trajectory; sequencing the four, deciding when the sway has settled
  (`crane_msgs/SwaySettled` is the supervisor's, and acting on it the task
  layer's — ROS 2 Interfaces §4) and running them is above this node.
- **Deciding *when* to re-plan is not this node's.** Replanning *from* a moving
  and swinging start is, and it is below; the trigger — a stall, a tracking
  fault, a new goal — belongs to `crane_supervisor` and the task layer, and this
  service answers requests. The `a2b_movement` adapter is **issue 046**.

## κ is not `speed_scale`

They are both numbers below one that make the machine go slower, they multiply,
and confusing them would quietly give away the thing §5.5 exists to keep.

| | κ | `speed_scale` |
|---|---|---|
| whose | the **deployment's** reservation | the **caller's** request |
| where | `timing.kappa`, a node parameter | a field of `crane_msgs/PlanMotion` |
| applies to | *every* physical limit — velocity, acceleration, cylinder force, pump flow | the joint **velocity** bound alone |
| default | `0.8` | `1.0` |
| means | "the MPC gets the other 20 %" | "I want this move taken gently" |

> [!IMPORTANT]
> **κ defaults below one on purpose, and raising it to one is not a speed
> setting — it is a decision to hand the MPC a reference it cannot correct.**
> §5.5 is the argument: a timing computed exactly at the actuation limits
> saturates the actuators *by construction*, at every instant the limit binds. The
> MPC's job is to add a corrective acceleration on top of the reference, and at
> saturation there is none left to add — the controller's authority is zero
> precisely where the trajectory is hardest to track. The legacy planner had the
> same insight and expressed it as a refusal to iterate its time-scale factor to
> unity; making it a named parameter is the change, not the idea.

**A caller cannot raise κ by asking for more speed.** `speed_scale` enters
`scale_limits()` on the velocity row only and multiplies κ there rather than
replacing it, so `speed_scale > 1` — which the service validates away in any case
— could at most undo a caller's own earlier slow-down. The reserved authority is
not reachable from the request message. That is why `TimingSolution` carries
`kappa` and `speed_scale` as two fields and why `PeakDemand` reports each peak as
a fraction of the **physical** limit rather than of the scaled one: "the peak
demand sat at κ and not at one" is then something the caller reads, not something
it has to trust.

`Q_P^max`'s own `0.95×` is a third factor and is neither of these. It is
`parameters.md` §4's discount on one weakly-evidenced number, it applies to the
pump flow alone, and `config/hydraulic_limits.yaml` carries it with the "measured
2023, pre-retrofit, not re-verified" provenance that is the reason for it.

## What the OCP is, and what it refuses

State `x_σ = (σ, σ̇, q_u, dq_u) ∈ ℝ⁶`, input `u_σ = σ̈`, discretised over **σ and
not over time** — §5.2 already writes the objective as `∫₀¹ dσ/σ̇`, and on a fixed
σ-grid `q_a'(σ)` and `q_a''(σ)` are known numbers at every shooting node instead
of something that would have to be carried symbolically. `σ` is a state anyway so
that §5.4's `σ(T) = 1` is imposed rather than an artefact of the grid.

The passive rows come from `crane_model::symbolic::CasadiGraph`, and the force and
flow rows are read out of that graph's **output map** — the same expressions
`mpc` §3 constraints 6 and 7 are built from, including §3.1's smoothing, which is
compiled into the graph and must not be applied a second time. §5.3's *"a
reference the MPC would reject is a planner bug"* therefore holds by construction
and not by two implementations agreeing; `test_timing_ocp.cpp` asserts the
planner's flow row and the graph's output map agree pointwise rather than
resembling each other.

Three things it will not do:

- **A solve that does not converge is `success=false` carrying acados' own status
  word**, never a partial or clipped trajectory. `mpc` §5.3 requirement 1 — a
  backend that always reports success removes the only signal a supervisor could
  act on — binds the planner exactly as it binds the controller.
- **The solve is bounded in wall clock here**, by `timing.max_wall_clock`, and not
  in a caller's timeout. §7's bounded-latency requirement as a whole is
  `latency_budget`, charged over every stage of one request; this cap is the piece
  of it that lives inside the OCP and refuses in the solver. The measured cost
  of the structured primitive is some 44 SQP iterations and 4 s, and the cap is
  three times that, so an ordinary plan clears it on a slower machine too.
- **A path the machine cannot hold at rest is refused before the solver runs,**
  by name. A cylinder holds the arm up at rest and that part of constraint 6 does
  not fall when the move is slowed down, so if gravity alone is outside the scaled
  limit at any node no timing exists. Left to acados that comes back as
  `ACADOS_QP_FAILURE`, which is a correct refusal and a useless diagnostic; here
  it names the node, the axis, the force it would take and the transmission
  `J_c,ii` at that pose. The usual cause is a pose near a transmission zero, where
  `J_c,ii` vanishes and the cylinder has no moment arm about the joint at any
  force — the sharp edge on §4.2's remark that the arm joint's range is far wider
  than its working range. On the PZS100 the arm's zero is at `θ₃ = 1.85 rad`,
  which is almost exactly where the URDF's own mid-range falls.

**The OCP starts where the machine is**, and it is worth being exact about which
coordinate says so. `q_u(0)` and `dq_u(0)` are the measured passive pair; the
*joints* leave their start at the measured velocity because the path is fitted to
meet it with a `q_a'(0)` in that direction and `σ̇(0)` is pinned to the one path
rate that reproduces it. `σ̇` is otherwise not pinned at either end — it is a path
rate, the objective divides by it, and pinning it at both would make the first and
last stage singular. What is asserted is what the machine does, not what the
coordinate reads. §5.4's terminal conditions are untouched by any of this: the
tool still arrives hanging still.

A machine that really is standing still still gets the plan this package built
before — `q_a'(0) = 0`, the passive pair at the settled `q_u^eq` — because that is
the right answer for a machine at rest. What changed is that it is no longer the
*only* answer.

## Replanning from a moving, swinging state, and the latency bound

`wiki/trajectory_planning.md` §7's `[!warning]` names the convention this package
had inherited: the legacy stack zeroes the initial velocity outright — *"we
deactivated qDot0, because we now always start in a stopped state"* — so the whole
pipeline only works as stop, measure, replan, and a re-plan issued while the tool
is still moving mis-predicts the sway from the first step. That is exactly the
situation a stall-recovery re-plan occurs in. §7 draws three consequences and each
of them is a mechanism here rather than a promise.

**Continuity is a boundary condition.** `MotionRequest::start` is `(q, dq)` as
measured over all eight coordinates — the six actuated positions and rates off
`/joint_states`, the passive pair off `/crane/pendulum_state`. A moving start is
fitted into the *geometry* (`PathFitRequest::start_rate`) and pinned into the
*timing* (`TimingOcpStart::sigma_rate`), so a concatenated segment matches the
previous one in position **and** velocity at the seam. When no path rate
reproduces the measurement — the fitted path leaves its start in a direction the
machine is not travelling in — that is a refusal naming the residual, never a
first emitted velocity that steps down to zero.

> [!WARNING]
> **A re-plan issued at speed can be refused, and that refusal is the honest
> answer rather than a gap to paper over.** `σ̇(0)` on a re-fitted path is set by
> the path's own extent — `q_a'(0)` is the measured *direction* scaled by the
> fit's reference span — so a re-plan issued mid-lift, where the remaining path is
> short, needs a high path rate to carry the measurement. §5.2's force row carries
> `q_a''(0) σ̇²`, quadratic in it, so past some fraction of the first plan's peak
> velocity **no timing of the new path starts where the machine is** and the solve
> refuses. Slowing before re-planning is what makes one exist. Bounding the fitted
> path's start curvature so that the feasible seam speed is a stated number rather
> than a discovered one is not done here.

**The latency bound is the planner's, not the caller's.** `latency_budget` is one
total over the endpoint IK, the geometric path, the smoothing and the OCP, charged
stage by stage by `LatencyLedger`; the stage that spends it stops the plan naming
itself and saying how long it had. §7 is explicit about why a client timeout will
not do: it bounds how long the caller *waits* and leaves a slow solve running
under a request nobody is waiting on any more. The OCP's own `max_wall_clock` is
lowered to whatever the total has left, so the two bounds cannot disagree about
which of them fired. The ledger reads a `PlanningClock` — a plain `double()` — so
the tests drive it with a counter and no wall-clock sleep is the mechanism under
test.

**On overrun the previous trajectory keeps standing.** No partial plan is emitted;
`/crane/reference` is written in exactly one place, reached only by a plan that
was adopted. A refusal carries the standing trajectory back on the response and
says it is unchanged, because a caller has to be able to tell "I kept planning"
from "here is something new" — and the topic is transient-local, so a refused
re-plan that republished would leave a late subscriber latching onto a plan nobody
is executing.

**An absent passive estimate is stated, never read as zero sway.**
`wiki/control_architecture.md` §5.3 asks every input that stops arriving to end in
a defined consequence, and "never connected", "died" and "the producer says
`valid == false`" are three different things for an operator to chase — the answer
names which. What happens next is the deployment's `passive_estimate_policy` and
there are exactly two values:

| | `refuse` (default) | `conservative` |
|---|---|---|
| answer | `success=false` naming which absence it was | a plan, with what was assumed stated in the answer |
| sway | not assumed at all | the hanging pose of the measured actuated configuration |
| box | — | `mpc` §3 constraints 3 and 4 reduced by `conservative_sway`, so sway up to the reserve still fits |

There is no third value, because reading an absent estimate as zero is the
stopped-start convention §7 removes. One combination is refused under **either**
policy: a moving arm whose sway nobody measured, which is §7's `[!warning]` in
full — a machine standing still is a different case, where the hanging pose is
where a settled tool is rather than an assumption about the sway.

**`/crane/payload_estimate` is read on the same terms.** Where it is valid it
*replaces* the mass and centre the caller declared, because the estimator measured
the machine and the caller described it; where it is absent, stale or
`valid == false` the declaration stands and the answer says so. Both CBS profiles
publish `valid == false` today (issue 033), so the second branch is the one that
actually runs and it is the one the tests spend their assertions on.

> [!NOTE]
> **This is not the MPC's warm start.** `crane_mpc`'s own warm start and fallback
> are issue 052's, they run at the controller's rate against the reference this
> node published, and the two mechanisms answer different questions: one is what
> to do when a *plan* could not be produced in time, the other is what to do when
> a *control step* could not.

**Deciding when to re-plan is not here.** The trigger belongs to
`crane_supervisor` — which owns `FAULT_TRACKING` since issue 024 — and to the task
layer. This node answers requests.

## `/crane/plan_grip`: four phases, one clock

`crane_msgs/PlanGrip` is frozen at four phases — `PHASE_DESCEND`, `PHASE_CLOSE`,
`PHASE_OPEN`, `PHASE_LIFT` — and they split two and two.

**Descend and lift are arm motions, and they are `plan_motion` itself.**
`plan_grip.cpp` assembles a `MotionRequest` and calls it, so a grip's arm phase
gets the equilibrium-constrained endpoint of `robot_model` §2.2, the structured
primitive of §4.4, the same collision check, the same §4.3 sway envelope and the
same κ. There is no second, looser path generator here and there must not be one:
a grip descend is the phase that puts a tool between the runges.

The one knob a **descend** changes is the transfer altitude's *ceiling*, lowered
to the higher of its own two endpoints so the phase goes across-and-down instead
of up, across and down onto the block it is already above. A **lift** keeps the
derivation, and that asymmetry is measured rather than preferred: capping the
altitude collapses the primitive's final descend segment, over which §5.4's
terminal condition on the sway then has no arm motion to steer `dq_u` with. The
same solve converges in 83 SQP iterations at the derived altitude and runs to
`ACADOS_MAXITER` at the capped one, erratically in the margin — an
ill-conditioned problem, not a margin to tune.

**Close and open are the tool alone, on the arm's own clock.**
`wiki/trajectory_planning.md` §8 *retains* the legacy grip cosine primitive, so
what changes is not its shape but its clock and its limits. §4.1: "geometrically
the tool is decoupled from the arm; temporally it is not, and the two must share
one clock." A tool phase therefore comes back over the same six actuated rows, on
the same sample period, first point at `time_from_start = 0`, with the five path
coordinates held and emitted beside `q8` rather than left out. The cosine is
carried on the **rate**, `h'(s) = 1 - cos 2πs`, because a raised cosine in
position is only C¹ at its ends and everything else this planner emits is C².

**The tool differs and the planner knows it.** `q9_left_rail_joint` on the
PZS100, `theta10_outer_jaw_joint` on the 7040 (§3.1). The 7040's jaw
transmission **reverses sign inside its own range** — issue 037's notes measured
the crossing at `q8 ≈ 0.29 rad` — and at the crossing the cylinder moves the jaw
through no distance at all. A phase whose travel spans it is refused naming it,
never clipped and never planned through. Which end of the range closes the
gripper is not in the description, so it is the `gripper_closed_end` parameter:
measured on the 7040, assumed on the PZS100, where
`commissioning_prerequisites.md` item 4 already owns that axis as human work.

**No memory, in either direction.** The service holds nothing between calls and
which phase ran last is not a fact it has: every phase starts at the machine's
own measured configuration, so the task layer sequences the four by moving the
machine and re-measuring. That is what makes "no phase returns a trajectory whose
first point is not at the previous phase's last" true by construction.

`crane_msgs/PlanGrip` has no `avoid_collisions` row and the .srv is frozen
(PRD §15), so the node reads its absence as the **checked** plan — an arm phase
with nothing on `/crane/collision_scene` is refused naming the topic. A tool
phase moves no link that was not already where it is, sweeps nothing, and reads
no goal pose at all, so it demands neither a scene nor a frame.

## It is not a second command producer

`crane_velocity_controller` is the sole claimant of the six `velocity` command
interfaces and which controller holds that claim is the supervisor's decision
alone (ROS 2 Interfaces §3.2 and §5). So this node publishes one topic and it is
a *reference*, holds no `controller_manager` client, claims no interface and is
loaded by no spawner. `crane_bringup`'s launch contract asserts all of that by
reading these sources.

## One thing a later issue must not inherit

- **The payload goes to the model with a zero inertia tensor.** That is exact for
  what this issue uses it for and only for that: `wiki/robot_model.md` §5.3 says
  in as many words that for a gravity moment the payload is a point mass, and the
  equilibrium and forward kinematics are the only model calls made here. The
  first issue that evaluates a *dynamics* call for a carried payload owes a real
  tensor at that boundary.

## Layout

    include/crane_planning/joint_limits.hpp        the description's own limits
    include/crane_planning/arm_geometry.hpp        the probed two-link closure
    include/crane_planning/inverse_kinematics.hpp  robot_model §2.2, semi-analytic
    include/crane_planning/equilibrium_ik.hpp      robot_model §2.2, constrained
    include/crane_planning/redundancy.hpp          §2.2 step 3's leftover freedom
    include/crane_planning/geometric_path.hpp      q_a(σ) and its two derivatives, C² by ruckig
    include/crane_planning/structured_primitive.hpp  §4.4 lift/traverse/descend, and the altitude
    include/crane_planning/collision.hpp           §4.2's scene and truck, §4.3's sway envelope
    include/crane_planning/sampling_planner.hpp    §4.4's fallback and §4.5's mandatory smoothing
    include/crane_planning/trajectory_timing.hpp   the ramp, and the seam it meets the path at
    include/crane_planning/timing_ocp.hpp          §5.2's OCP, §5.3's constraints and §5.5's κ
    include/crane_planning/tool_axis.hpp           §8's retained grip cosine, on the arm's clock
    include/crane_planning/plan_grip.hpp           the four phases of crane_msgs/PlanGrip
    include/crane_planning/replanning.hpp          §7's measured start, and its latency ledger
    src/acados_casadi_bridge.hpp                   where acados meets CasADi; not installed
    config/hydraulic_limits.yaml                   Q_P^max and the relief setting, with their evidence
    include/crane_planning/planner_core.hpp        goal in, trajectory out, ROS-free
    include/crane_planning/planner_node.hpp        the adapter
    config/crane_planner.yaml                      what a deployment configures

Everything above `planner_node.hpp` is ROS-free and is tested offline against the
machine's own descriptions, which are read where they live in
`crane_model/test/description` rather than copied in.
