# crane_planning

## Offline-style A-to-B visualization

After starting `crane_planner_node` with the normal robot description and state
publishers, a single planner response can be plotted without writing a second
planner:

```bash
ros2 run crane_planning plan_a2b --goal 1.5 0.0 1.2 --yaw 0.0 --show
```

The goal is expressed in `K0_mounting_base`. The script saves `plan_a2b.png`
and the returned joint/TCP samples as `plan_a2b.csv`; use `--output` to choose
another basename. Collision checking is enabled by default and can be disabled
with `--no-collision-check`.

## Standalone timing-OCP tuning

Tuning the time parametrization of §5.2 does not need a graph, and it does not
need a second implementation of the problem either. `crane_planning_timing_cli`
is the deployed `solve_timing_ocp` with a `main` in front of it: no ROS, no
node, one path, one solve, one CSV. `scripts/timing_a2b.py` builds its argument
list and plots what comes back. The binary is installed; the script is run from
the tree, because it reads `crane_model`'s checked-in machine descriptions and
those are test fixtures rather than installed files. Source the workspace first
so it can find the binary, or point `--binary` at it.

```bash
python3 src/concrete_block_stack/crane_planning/scripts/timing_a2b.py --show
```

The default output is `timing_a2b.png` in the working directory with the
matching `.csv`, one row per shooting node — the grid the constraints are
actually imposed on. Use `--a` and `--b` to change the five planned joint
coordinates, `--via` to add interior waypoints, or `--tool epsilon7040` for the
other machine:

```bash
python3 .../scripts/timing_a2b.py --sway-weight 5.0 --kappa 0.7 --intervals 60 --show
```

**Every tuning option left unset is the deployment's own default.** The script
passes through only what it is given, and the CLI fills the rest in from
`TimingOcpSettings`, which is where `crane_planner.yaml`'s defaults come from
too — so a weight tuned here is a weight the node runs once it is written into
that file, and no number lives in two places. `--print-command` shows the
invocation. Run `--help` for all OCP, path-shape, hydraulic and payload options.

The CLI can also be driven directly, which is what a test or a sweep script
should do:

```bash
ros2 run crane_planning crane_planning_timing_cli \
  --description "$(ros2 pkg prefix crane_model)/../../src/.../pzs100.urdf" \
  --waypoint 0,-0.2,0.4,1.0,0 --waypoint 0,-0.2,0.9,1.4,0 \
  --sway-weight 5.0 --output sweep.csv
```

It prints a `key=value` summary on stdout — status, duration, iterations, the
peak demand of each constrained quantity against its **unscaled** limit — and
refuses on stderr with the solver's own status word.

The path is fitted with the planner's own `fit_c2_path`, so the OCP is tuned
against the path shape the deployment produces and not against a polynomial
written for the occasion.

## What it does

`crane_planner_node` exposes the two planning contracts:

* `/crane/plan_motion` is the native crane planning service.
* `/a2b_movement` is a compatibility adapter for the timber stack and delegates
  to the same native planning pipeline.

Successful requests follow one bounded pipeline:

1. Convert the request and measured state, including payload and passive sway.
2. Solve equilibrium-constrained endpoint IK and validate the forward-kinematics
   residual.
3. Search the five actuated path coordinates with deterministic OMPL
   RRT-Connect. The passive pair is recovered from the model equilibrium at
   each checked configuration; the tool axis remains fixed.
4. Shortcut the resulting polyline, fit the required C² path with Ruckig, and
   re-check the fitted curve against the scene, crane self-collision, and sway
   envelope. Unchecked requests skip scene checks but still enforce model and
   joint validity.
5. Time the accepted geometric path with the generated acados path-constrained
   OCP, including sway, force, flow, and terminal-rest constraints.

Collision-checked requests require a current scene in the expected frame.
Every stage has bounded work and contributes to `latency_budget`; refusals name
the stage or constraint that prevented a plan. A moving measured start is
preserved in both path and timing boundary conditions, so replanning does not
silently reset velocity to zero. On refusal, the last adopted trajectory
remains published on `/crane/reference`.

## Interfaces and ownership

`/crane/plan_grip` is not provided by this package. Grip sequencing and tool
actuation belong to
`concrete_block_motion_planning/grip_traj_movement`; `crane_msgs/PlanGrip`
remains only as a compatibility schema for clients that still build that
request.

The package intentionally has no structured lift/traverse/descend planner,
sampling fallback, grip trajectory generator, tool-axis mechanism, or legacy
velocity-ramp timing. OMPL is the sole geometric search; C² fitting is its
required preprocessing for the OCP, not a second search mechanism.

## Configuration and generated timing code

`config/crane_planner.yaml` contains solve settings: endpoint IK tolerances,
OMPL budgets and seed, shortcut/C²-fit settings, collision and sway sampling,
input freshness, replanning latency, and OCP limits. Joint limits and machine
geometry come from `robot_description` and `crane_model`.

The checked-in acados sources for the PZS100 and Epsilon 7040 are generated
artifacts. `export_timing_ocp.py` is the source of truth; run its `--check`
mode when changing the timing model or validating a checkout. The generated
solver is kept separate from the handwritten planner implementation at build
time.

`export_timing_ocp.py` owns the *formulation* -- the symbolic model, the
constraint rows, the parameter layout -- and `src/timing_ocp.cpp` owns
everything that is written onto the shipped solver at runtime: the scaled
bounds, the per-node passive equilibrium, the stage parameter blocks and the
bisected warm start. That line is the whole reason `timing_a2b.py` shells out to
a binary rather than driving acados itself; the alternative was a second copy of
those four things in Python, agreeing with the C++ by hand.

## Dependencies

The planner uses OMPL, Ruckig, Eigen, `crane_model`, and the generated acados
timing solver. ROS interface dependencies provide `PlanMotion` and the retained
`CalcMovement` adapter contract. MoveIt is not required.

The two installed scripts are developer tools and are not part of the planner.
`plan_a2b` is installed and needs `rclpy`, numpy and matplotlib. `timing_a2b`
needs numpy, matplotlib and pyyaml, joins no graph, and is run from the tree
beside `export_timing_ocp.py`. Neither needs casadi or acados --
`export_timing_ocp.py` does, and it is an export-time script.
