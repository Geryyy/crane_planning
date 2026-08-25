# crane_planning

The `crane_planner` node of `wiki/implementation/ros2_interfaces.md` §2: it
serves `/crane/plan_motion` and publishes the trajectory it answered with on
`/crane/reference`.

This is the **slice-5 tracer bullet** of `docs/features/cbs-arch/prd.md` §2 — one
goal pose in, one timed joint trajectory out, through a real node and a real
service. Almost everything `wiki/trajectory_planning.md` specifies is still
absent, and the point of this package as it stands is that every absence is
*refused or named*, never approximated.

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
| timing | one scaled ramp obeying the velocity limits, run along that path |

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
call is a hang, and a caller cannot cancel one. Issue 045 owns the latency bound
and inherits these numbers.

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
`build_structured_primitive`. The check passes unconditionally today and says so;
issue 041 replaces its body rather than the structure around it. A blocked
primitive is a **refusal** naming which of the three phases failed — the sampling
fallback that would take over is issue 042.

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
- **The timing is deliberately trivial.** One velocity-limited ramp, C¹ at both
  ends and at rest there. The real stage 2 — the path-constrained OCP of
  `trajectory_planning` §5.2, carrying the sway explicitly, with the cylinder
  force and pump-flow constraints and the κ margin that leaves the MPC authority
  — arrives with **issue 043**. Nothing here reads `Q_P^max` or a cylinder force.
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
- The sampling fallback and its mandatory smoothing is **issue 042**,
  `/crane/plan_grip` **issue 044**, replanning from a moving and swinging start
  **issue 045**, and the `a2b_movement` adapter **issue 046**.

## It is not a second command producer

`crane_velocity_controller` is the sole claimant of the six `velocity` command
interfaces and which controller holds that claim is the supervisor's decision
alone (ROS 2 Interfaces §3.2 and §5). So this node publishes one topic and it is
a *reference*, holds no `controller_manager` client, claims no interface and is
loaded by no spawner. `crane_bringup`'s launch contract asserts all of that by
reading these sources.

## Two things a later issue must not inherit

- **The payload goes to the model with a zero inertia tensor.** That is exact for
  what this issue uses it for and only for that: `wiki/robot_model.md` §5.3 says
  in as many words that for a gravity moment the payload is a point mass, and the
  equilibrium and forward kinematics are the only model calls made here. The
  first issue that evaluates a *dynamics* call for a carried payload owes a real
  tensor at that boundary.
- **The start state is assumed to be at rest.** The plan begins at the measured
  configuration on `/joint_states` and its `header.stamp` is that measurement's
  stamp, but the measured *velocity* is ignored and the ramp starts from zero.
  `trajectory_planning` §7 is explicit that inheriting the stopped-start
  convention is a defect; lifting it is issue 045, and until then a request
  issued while the machine is moving is refused only by the freshness deadline,
  not by the velocity.

## Layout

    include/crane_planning/joint_limits.hpp        the description's own limits
    include/crane_planning/arm_geometry.hpp        the probed two-link closure
    include/crane_planning/inverse_kinematics.hpp  robot_model §2.2, semi-analytic
    include/crane_planning/equilibrium_ik.hpp      robot_model §2.2, constrained
    include/crane_planning/redundancy.hpp          §2.2 step 3's leftover freedom
    include/crane_planning/geometric_path.hpp      q_a(σ) and its two derivatives, C² by ruckig
    include/crane_planning/structured_primitive.hpp  §4.4 lift/traverse/descend, and the altitude
    include/crane_planning/collision.hpp           §4.2's scene and truck, §4.3's sway envelope
    include/crane_planning/trajectory_timing.hpp   the ramp, and the seam it meets the path at
    include/crane_planning/planner_core.hpp        goal in, trajectory out, ROS-free
    include/crane_planning/planner_node.hpp        the adapter
    config/crane_planner.yaml                      what a deployment configures

Everything above `planner_node.hpp` is ROS-free and is tested offline against the
machine's own descriptions, which are read where they live in
`crane_model/test/description` rather than copied in.
