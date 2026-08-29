# crane_planning

The `crane_planner` node. Three stages and one dependency that matters.

```
PlanMotion(goal in K0_mounting_base)
  -> inverse kinematics        the joint configuration that puts the tool there, hanging
  -> OMPL RRT-Connect          a collision-free polyline in the five planned coordinates
  -> C2 fit                    q_a(sigma) with q_a' and q_a'' defined everywhere
  -> CasADi/IPOPT timing OCP   how fast that path may be traversed
  -> JointTrajectory on /crane/reference
```

Everything the machine can do -- reach, hang, collide, lift -- is asked of
`crane_model`. Nothing about the machine is written down here: joint ranges and
velocity limits come from the `robot_description` the profile remaps this node
onto, the cylinder areas and the equations of motion come from
`crane_model.symbolic`, and the two numbers the description does not carry (the
pump limit and the relief setting) are in `config/crane_planner.yaml` with their
evidence beside them.

## The three stages, and why each one is there

**Inverse kinematics.** The request is Cartesian and the search is not, so
something has to turn `(position, yaw)` into a joint configuration. It is one
least-squares problem over the five planned coordinates with the passive pair
re-settled at every configuration tested -- so what is solved is where the tool
actually ends up, not where it would be if it did not hang. The telescope
redundancy is taken up by a weak pull towards the start, and the restarts spread
over the telescope range because that is the direction the residual is flat in.

**The geometric search.** RRT-Connect over the five planned coordinates, in
coordinates scaled by each axis's own velocity limit -- so the metric OMPL
extends and interpolates in is *seconds at full speed* and not a sum of radians
and metres. The passive pair is never sampled: it hangs, so it is solved
wherever geometry is checked. Only exact solutions are accepted; an approximate
path is a refusal.

**The C2 fit.** A clamped cubic spline through the shortcut polyline, with
`sigma` distributed by how long each chord takes at its slowest axis's limit. It
is twice differentiable, and that is the entire point: the timing stage writes
`ddq_a = q_a'' sigma_dot^2 + q_a' sigma_ddot`, so at a kink in the polyline
`q_a''` is unbounded, the admissible path rate collapses to zero, and the
machine stops dead at every waypoint.

**The timing OCP.** One NLP over `sigma`, solved with IPOPT:

| | |
|---|---|
| state | `(sigma_dot, q_u, dq_u)` -- the path rate and the two sway coordinates |
| control | `sigma_ddot` |
| dynamics | `ddq_u = -M_uu^-1 (M_ua ddq_a + h_u)`, every derivative divided by `sigma_dot` |
| cost | `int dsigma/sigma_dot` + `w_sway int ||dq_u||^2 dt` + `w_u int sigma_ddot^2` |
| constraints | `ddq_a`, cylinder force and summed pump flow at **every** node, `q_u` inside the sway box, `dq_u` bounded, `sigma_dot` under the per-node velocity ceiling |
| boundary | `q_u` and `dq_u` measured at the start, at the hanging pose and at rest at the goal |

The sway is a state and not an afterthought, which is what makes the tool arrive
still. Integration is RK4 in `sigma`, so the path is evaluated at interval
midpoints as well as at nodes.

The initial guess earns its three passes: the velocity ceiling says nothing
about acceleration, so the classical `sqrt(ddq_max / |q_a''|)` curve goes on top
of it; neither says anything about cylinder force or pump flow, which are what
actually bind on a lift, so the rate is bisected down until the worst row sits
at 90% of its allowance; and per-node feasibility is not reachability, so a
forward and a backward sweep under `d(sigma_dot^2)/dsigma = 2 sigma_ddot`
connect the nodes to each other.

## Tuning it

`scripts/plan_example.py` runs the deployment's own `Planner` against the
checked-in machine descriptions -- no ROS, no graph, no clock -- and plots what
the answer asks of the machine: joint positions, velocities and accelerations,
the path rate, the sway offset and rate, cylinder force and pump flow, each
against its own limit.

```bash
./scripts/plan_example.py --show
./scripts/plan_example.py --sway-weight 8 --intervals 60 --output slow.png
./scripts/plan_example.py --tool epsilon7040 --payload-mass 400 --csv plan.csv
```

It exercises every stage, so a refusal from it is the refusal the node would
give. `--goal` takes a joint configuration and puts its tool pose through the
real inverse kinematics; `--goal-pose` takes a Cartesian pose instead, which may
turn out to be unreachable.

## The node

| | |
|---|---|
| service | `/crane/plan_motion` (`crane_msgs/PlanMotion`) |
| publishes | `/crane/reference` (`trajectory_msgs/JointTrajectory`, transient-local) |
| | `/crane_planner/planned_path` (`nav_msgs/Path`, visualization only) |
| subscribes | `/joint_states`, `/robot_description`, `/crane/collision_scene`, `/crane/payload_estimate` |
| frame | `K0_mounting_base` for every goal and every published pose; nothing is converted |

It is **not a second writer of the machine**: `/crane/reference` is a reference
and not a command, `crane_velocity_controller` remains the sole claimant of the
six velocity command interfaces, and this node holds no `controller_manager`
client of any kind.

`/joint_states` carries partial messages from two producers --
`joint_state_broadcaster` for the actuated six and `tip_tilt_state_broadcaster`
for the passive pair -- so they are cached in two slots and read by joint name,
never by index.

Every refusal names itself and leaves the standing reference alone: a goal that
cannot be reached, a path that cannot be found, a path that cannot be smoothed
inside the joint ranges and a path that cannot be timed are four different
answers.

## Build and run

```bash
./ralph/verify.sh crane_model crane_planning
ros2 launch crane_planning crane_planner.launch.py
```
