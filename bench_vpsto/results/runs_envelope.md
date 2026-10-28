# CBS vs VP-STO (generated 2026-10-28)

CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, nproc 8, cpus 0-7; VP-STO OpenMP threads [1], nVia 8, nEval 80, population 50, 20 runs/move/setting; CBS acados OpenMP threads [1] (set explicitly before every plan), 20 repeats/move. Linux-7.0.0-38-generic-x86_64-with-glibc2.39

## Aggregate (without the excluded moves)

| planner | runs | settled at T | median residual sway deg | median duration s | median plan s |
|---|---|---|---|---|---|
| cbs_a1 | 60 | 60/60 | 0.39 | 7.20 | 2.29 |
| cbs_min0.5_a1 | 60 | 60/60 | 0.39 | 7.20 | 2.02 |
| vpsto_t1 | 60 | 0/60 | 21.40 | 5.61 | 0.73 |

## Per move

| move | planner | n | success | plan s med / p95 | duration s | peak sway | residual sway med / max | settled at T | settle s | acc0 / accT | over limit | min clearance eq / sway | load flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bench_carry | cbs_a1 | 20 | 20/20 | 11.52 / 12.60 | 11.81 | 7.53 | 0.67 / 0.67 | 20/20 | 0.00 | 0.00 / 0.00 | 20 | 0.54 / 0.53 | 0 |
| bench_carry | cbs_min0.5_a1 | 20 | 20/20 | 11.72 / 12.07 | 11.81 | 7.53 | 0.67 / 0.67 | 20/20 | 0.00 | 0.00 / 0.00 | 20 | 0.54 / 0.53 | 0 |
| bench_carry | vpsto_t1 | 20 | 8/20 | 2.86 / 4.24 | 2.95 | 63.07 | 62.69 / 67.91 | 0/20 | >20 | 0.96 / 1.00 | 4 | 0.01 / 0.12 | 0 |
| example_wall_nobody | cbs_a1 | 20 | 20/20 | 0.43 / 0.47 | 7.20 | 2.92 | 0.38 / 0.38 | 20/20 | 0.00 | 0.00 / 0.00 | 0 | 2.12 / 2.13 | 0 |
| example_wall_nobody | cbs_min0.5_a1 | 20 | 20/20 | 0.89 / 1.46 | 7.20 | 2.92 | 0.38 / 0.38 | 20/20 | 0.00 | 0.00 / 0.00 | 0 | 2.12 / 2.13 | 0 |
| example_wall_nobody | vpsto_t1 | 20 | 20/20 | 0.61 / 0.81 | 5.62 | 15.29 | 18.14 / 34.81 | 0/20 | >20 | 0.76 / 0.73 | 0 | 1.47 / 1.42 | 0 |
| example_wall_body | cbs_a1 | 20 | 20/20 | 2.29 / 3.12 | 7.18 | 3.32 | 0.39 / 0.39 | 20/20 | 0.00 | 0.00 / 0.00 | 0 | 1.13 / 1.13 | 0 |
| example_wall_body | cbs_min0.5_a1 | 20 | 20/20 | 2.02 / 2.08 | 7.17 | 3.32 | 0.39 / 0.39 | 20/20 | 0.00 | 0.00 / 0.00 | 0 | 1.13 / 1.13 | 0 |
| example_wall_body | vpsto_t1 | 20 | 20/20 | 0.67 / 0.85 | 5.61 | 12.80 | 13.36 / 38.28 | 0/20 | >20 | 0.79 / 0.72 | 0 | 1.13 / 1.13 | 0 |

## CBS time split

| move | planner | acados threads | ocp s | geom s | rest s | iterations | ms / iteration |
|---|---|---|---|---|---|---|---|
| bench_carry | cbs_a1 | 1 | 0.21 | 11.25 | 0.07 | 21.00 | 6.23 |
| bench_carry | cbs_min0.5_a1 | 1 | 0.40 | 11.24 | 0.08 | 46.00 | 7.01 |
| example_wall_nobody | cbs_a1 | 1 | 0.23 | 0.18 | 0.01 | 21.00 | 7.22 |
| example_wall_nobody | cbs_min0.5_a1 | 1 | 0.64 | 0.21 | 0.01 | 55.00 | 9.84 |
| example_wall_body | cbs_a1 | 1 | 0.46 | 1.77 | 0.02 | 40.00 | 9.45 |
| example_wall_body | cbs_min0.5_a1 | 1 | 0.27 | 1.72 | 0.02 | 24.00 | 7.75 |

## What each planner enforces

| | CBS (`crane_planning`) | VP-STO (EcB25 cost, ros2_vpsto) |
|---|---|---|
| rate / accel | kappa-scaled dq, ddq rows in the OCP | same bounds, at 80 eval points only |
| pump | flow row, soft, refused if bought | Q(q,q')/Q_max time bound at eval points |
| command | u_c3 domain (Psi range), jerk-bounded C4 command | none |
| sway | 0.2 rad box at every node; settled terminal box 0.02 rad / 0.04 rad/s | none (free end accelerations) |
| collision | fixed TCP corridor, margins 0.05 + 0.10 + swing envelope, certificate, self-collision | W-SDF (sd < 0 only) vs scene, boom down, at passive equilibrium, 80 eval points; no self-collision |
| joint box | planner box | penalty 1e2 (1 + violation) |
| time | min duration 2 s, weighted LSQ + time cost | pure min time |
| goal | TCP pose, IK inside plan time | goal joints given (no IK in its time) |

VP-STO settings: ros2_vpsto as is (full-covariance ccmaes, sigma0 0.5 vs paper 1, stopTolFunHist 1e-2
vs old sepaCMA ftol/xtol 1e-2, time-based seed: passed as 0, not settable without editing Marc's
code); nVia and threads on the command line (default 8 = 6 free via points; old server nVia 8).

Columns: plan s = wall of the whole call (CBS Planner.plan; VP-STO vc_solve incl. basis setup;
one-off warm-up plan not counted). CBS split: ocp = solver phase, geom = scene + geometry + IK +
corridor/certificate, rest = remainder. Sway/settle from the shared replay (start at the settled
hanging pose, actuators held after T): peak = max during the move, residual = max over
[T, T+20 s], settle = time after T to enter and stay in the 0.02 rad / 0.04 rad/s box (">20" never).
acc0/accT = max |ddq| / bound at t = 0 / T. Success = (VP-STO: solver cost has no penalty and) the
1 kHz check finds no penetration and no joint-box violation; util > 1 (+1e-3) is listed apart
(over-limit column). Clearance: 100 Hz grid, scene bodies only (the 21 bench moves have none).
`example_wall_body` carries the 0.9x0.6x0.6 block (CoM as bench_carry) for CBS, VP-STO and the
dynamics; `example_wall_nobody` is carry_move as is (no carried body collision-checked).
`cbs_min0.5` = CBS with ocp_duration_min 0.5 s (separates the 2 s floor from sway effects).
Thread counts are per row (`planner` suffix _aN = acados threads, `vpsto_tN` = VP-STO threads).
Rows with load flag (1-min load > 1.5 at move start) have unreliable timings.
lim_boom_high / lim_arm_out: CBS refuses (self-collision); VP-STO has no self-collision term;
rows kept, excluded from the aggregate.

## What this supports

Supported: open loop, a min-time spline with free end accelerations leaves large residual sway;
a timing OCP with sway as state ends settled without a downstream damper.
Not supported: system-level sway of EcB25 (VP-STO + iLQR), "CBS is X times slower than EcB25".
