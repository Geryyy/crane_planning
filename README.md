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
| geometric path | a straight line in joint space between the start and the endpoint |
| timing | one scaled ramp obeying the velocity limits |

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

The four numbers the closure needs — `a2`, `a3`, `d45(q4)` and the bearing they
are measured from — are **probed out of the model** at startup rather than
written down, and the probe is also a check: a description whose arm is not
planar, or whose telescope does not extend affinely in `q4`, fails to start with
the residual in the message. The joint position and velocity limits come out of
the same `robot_description` XML, because the frozen model API does not expose
them and a limit restated beside a node is a limit that drifts away from the
description the controllers were configured against.

## What it does not do, and what does

- **No collision of any kind.** `/crane/collision_scene` is not subscribed, the
  truck bed and the runges are not known, and the tool sway envelope of
  `trajectory_planning` §4.3 is not inflated. `avoid_collisions=true` — which is
  what `crane_msgs/PlanMotion` *defaults* to — is **refused**, with a message
  naming **issue 041**. A caller that wants the collision-blind plan has to ask
  for it explicitly.
- **The timing is deliberately trivial.** One velocity-limited ramp, C¹ at both
  ends and at rest there. The real stage 2 — the path-constrained OCP of
  `trajectory_planning` §5.2, carrying the sway explicitly, with the cylinder
  force and pump-flow constraints and the κ margin that leaves the MPC authority
  — arrives with **issue 043**. Nothing here reads `Q_P^max` or a cylinder force.
- **Clearance does not enter the redundancy score.** `robot_model` §2.2 step 3
  scores the telescope's one leftover degree of freedom by joint-range centring
  *and* collision clearance; there is no scene here to score against, so a
  non-zero clearance weight is **refused** rather than read and silently ignored.
  It joins the score at **issue 041**, and
  `include/crane_planning/redundancy.hpp` is the one place the rule is written.
- The structured lift/traverse/descend primitive is **issue 040**, the sampling
  fallback and its mandatory smoothing **issue 042**, `/crane/plan_grip` **issue
  044**, replanning from a moving and swinging start **issue 045**, and the
  `a2b_movement` adapter **issue 046**.

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
    include/crane_planning/trajectory_timing.hpp   the ramp
    include/crane_planning/planner_core.hpp        goal in, trajectory out, ROS-free
    include/crane_planning/planner_node.hpp        the adapter
    config/crane_planner.yaml                      what a deployment configures

Everything above `planner_node.hpp` is ROS-free and is tested offline against the
machine's own descriptions, which are read where they live in
`crane_model/test/description` rather than copied in.
