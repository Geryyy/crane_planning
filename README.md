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
| endpoint from a goal pose | the semi-analytic IK of `wiki/robot_model.md` §2.2, over `q_a = (q1, q2, q3, q4, q7)` |
| passive pair | **pinned** at `Model::passive_equilibrium`, §2.2's default for a placement goal |
| redundancy | the scalar search over `d45` of §2.2 step 3, scored by joint-range centring |
| acceptance | the forward-kinematics residual of §2.2, on **every** call, before the result is used |
| geometric path | a straight line in joint space between the start and the endpoint |
| timing | one scaled ramp obeying the velocity limits |

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
- The equilibrium-constrained endpoint NLP of `robot_model` §2.2 is **issue
  039**; this pins `q_u` instead of constraining it.
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
    include/crane_planning/inverse_kinematics.hpp  robot_model §2.2
    include/crane_planning/trajectory_timing.hpp   the ramp
    include/crane_planning/planner_core.hpp        goal in, trajectory out, ROS-free
    include/crane_planning/planner_node.hpp        the adapter
    config/crane_planner.yaml                      what a deployment configures

Everything above `planner_node.hpp` is ROS-free and is tested offline against the
machine's own descriptions, which are read where they live in
`crane_model/test/description` rather than copied in.
