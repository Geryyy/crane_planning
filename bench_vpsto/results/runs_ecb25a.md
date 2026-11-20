# CBS vs VP-STO (generated 2026-11-20, geometric_stage ces, run in per-move chunks)

CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, nproc 8; EcB25a sway weights 1,10,1; VP-STO threads [1]; CBS 1 repeat/move; unpinned; nVia 8, nEval 80, population 50

## Aggregate (without the excluded moves)

| planner | runs | settled at T | median residual sway deg | median duration s | median plan s |
|---|---|---|---|---|---|
| cbs_a1 | 22 | 22.0/22 | 0.42 | 5.95 | 0.41 |
| ecb25a_t1 | 440 | 49.0/440 | 2.63 | 3.67 | 1.48 |

## Per move

| move | planner | n | success | plan s med / p95 | duration s | peak sway | residual sway med / max | settled at T | settle s | acc0 / accT | over limit | min clearance eq / sway | load flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| across | cbs_a1 | 1 | 1/1 | 0.40 / 0.40 | 10.74 | 6.74 | 0.61 / 0.61 | 1.0/1 | 0.00 | 0.00 / 0.00 | 1.0 | - | 0.0 |
| across | ecb25a_t1 | 20 | 20/20 | 2.14 / 3.03 | 9.19 | 5.99 | 1.89 / 2.64 | 0.0/20 | 4.59 | 0.41 / 0.35 | 0.0 | - | 0.0 |
| ax_arm | cbs_a1 | 1 | 1/1 | 0.39 / 0.39 | 4.34 | 2.15 | 0.20 / 0.20 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| ax_arm | ecb25a_t1 | 20 | 20/20 | 1.37 / 1.60 | 3.01 | 3.81 | 1.84 / 2.42 | 0.0/20 | 5.37 | 0.93 / 0.90 | 0.0 | - | 0.0 |
| ax_boom | cbs_a1 | 1 | 1/1 | 0.31 / 0.31 | 6.27 | 3.67 | 0.36 / 0.36 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| ax_boom | ecb25a_t1 | 20 | 20/20 | 1.46 / 1.92 | 5.08 | 3.56 | 2.64 / 3.57 | 0.0/20 | 8.63 | 0.63 / 0.76 | 0.0 | - | 0.0 |
| ax_rot | cbs_a1 | 1 | 1/1 | 1.02 / 1.02 | 2.75 | 0.00 | 0.00 / 0.00 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| ax_rot | ecb25a_t1 | 20 | 20/20 | 1.02 / 1.14 | 1.27 | 0.38 | 0.20 / 0.62 | 20.0/20 | 0.00 | 1.00 / 0.99 | 0.0 | - | 0.0 |
| ax_slew | cbs_a1 | 1 | 1/1 | 0.57 / 0.57 | 10.64 | 7.26 | 0.56 / 0.56 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| ax_slew | ecb25a_t1 | 20 | 20/20 | 1.27 / 1.49 | 6.23 | 21.54 | 6.83 / 7.35 | 0.0/20 | 18.34 | 0.60 / 0.81 | 0.0 | - | 0.0 |
| ax_tele | cbs_a1 | 1 | 1/1 | 0.37 / 0.37 | 5.27 | 2.98 | 0.36 / 0.36 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| ax_tele | ecb25a_t1 | 20 | 20/20 | 1.32 / 1.48 | 3.57 | 5.97 | 1.31 / 1.86 | 6.0/20 | 1.91 | 0.90 / 0.96 | 0.0 | - | 0.0 |
| bench_carry | cbs_a1 | 1 | 1/1 | 5.96 / 5.96 | 9.34 | 3.63 | 0.70 / 0.70 | 1.0/1 | 0.00 | 0.00 / 0.00 | 1.0 | 0.52 / 0.53 | 0.0 |
| bench_carry | ecb25a_t1 | 20 | 3/20 | 4.53 / 5.25 | 3.78 | 17.60 | 5.15 / 5.81 | 0.0/20 | >20 | 1.00 / 1.00 | 0.0 | 0.00 / 0.00 | 0.0 |
| example_wall_nobody | cbs_a1 | 1 | 1/1 | 0.45 / 0.45 | 6.86 | 2.86 | 0.44 / 0.44 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | 2.12 / 2.12 | 0.0 |
| example_wall_nobody | ecb25a_t1 | 20 | 20/20 | 2.19 / 2.80 | 5.26 | 5.05 | 1.84 / 2.35 | 1.0/20 | 13.06 | 0.93 / 0.84 | 1.0 | 2.01 / 1.98 | 0.0 |
| example_wall_body | cbs_a1 | 1 | 1/1 | 0.55 / 0.55 | 6.97 | 3.05 | 0.66 / 0.66 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | 1.18 / 1.18 | 0.0 |
| example_wall_body | ecb25a_t1 | 20 | 20/20 | 2.13 / 2.50 | 5.25 | 4.56 | 2.12 / 2.55 | 0.0/20 | >20 | 0.75 / 0.72 | 0.0 | 1.18 / 1.18 | 0.0 |
| lift | cbs_a1 | 1 | 1/1 | 0.43 / 0.43 | 7.07 | 4.27 | 0.54 / 0.54 | 1.0/1 | 0.00 | 0.00 / 0.00 | 1.0 | - | 0.0 |
| lift | ecb25a_t1 | 20 | 20/20 | 1.56 / 1.74 | 4.76 | 8.58 | 1.73 / 2.29 | 0.0/20 | 3.79 | 0.61 / 0.66 | 0.0 | - | 0.0 |
| lim_arm_out (excluded) | cbs_a1 | 1 | 0/1 | - | - | - | - / - | 0/0 | - | - / - | 0 | - | 0.0 |
| lim_arm_out (excluded) | ecb25a_t1 | 20 | 20/20 | 2.26 / 2.52 | 8.48 | 3.77 | 3.03 / 3.94 | 0.0/20 | 10.51 | 0.50 / 0.86 | 3.0 | - | 0.0 |
| lim_boom_high (excluded) | cbs_a1 | 1 | 0/1 | - | - | - | - / - | 0/0 | - | - / - | 0 | - | 0.0 |
| lim_boom_high (excluded) | ecb25a_t1 | 20 | 20/20 | 1.51 / 1.75 | 5.94 | 5.50 | 8.91 / 9.63 | 0.0/20 | >20 | 0.73 / 0.83 | 1.0 | - | 0.0 |
| lim_boom_low | cbs_a1 | 1 | 1/1 | 0.51 / 0.51 | 4.32 | 2.49 | 0.39 / 0.39 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| lim_boom_low | ecb25a_t1 | 20 | 20/20 | 1.17 / 1.33 | 2.03 | 8.35 | 5.85 / 6.33 | 0.0/20 | >20 | 1.00 / 1.00 | 3.0 | - | 0.0 |
| lim_tele_out | cbs_a1 | 1 | 1/1 | 0.28 / 0.28 | 5.64 | 3.20 | 0.37 / 0.37 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| lim_tele_out | ecb25a_t1 | 20 | 20/20 | 1.27 / 1.46 | 4.01 | 5.41 | 0.74 / 1.07 | 20.0/20 | 0.00 | 0.88 / 0.86 | 0.0 | - | 0.0 |
| pair_reach | cbs_a1 | 1 | 1/1 | 0.48 / 0.48 | 5.32 | 2.91 | 0.38 / 0.38 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| pair_reach | ecb25a_t1 | 20 | 20/20 | 1.37 / 1.57 | 3.03 | 4.69 | 2.45 / 2.77 | 0.0/20 | 7.75 | 1.00 / 1.00 | 0.0 | - | 0.0 |
| pair_slew_lift | cbs_a1 | 1 | 1/1 | 0.36 / 0.36 | 8.49 | 5.19 | 0.48 / 0.48 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| pair_slew_lift | ecb25a_t1 | 20 | 20/20 | 1.77 / 2.14 | 6.27 | 9.55 | 2.96 / 4.80 | 0.0/20 | 10.27 | 0.52 / 0.81 | 0.0 | - | 0.0 |
| pair_tuck | cbs_a1 | 1 | 1/1 | 0.42 / 0.42 | 9.68 | 4.94 | 0.52 / 0.52 | 1.0/1 | 0.00 | 0.00 / 0.00 | 1.0 | - | 0.0 |
| pair_tuck | ecb25a_t1 | 20 | 20/20 | 1.73 / 1.96 | 7.23 | 7.85 | 7.01 / 7.34 | 0.0/20 | >20 | 1.00 / 0.65 | 0.0 | - | 0.0 |
| sh_boom | cbs_a1 | 1 | 1/1 | 0.26 / 0.26 | 2.05 | 0.77 | 0.77 / 0.77 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| sh_boom | ecb25a_t1 | 20 | 20/20 | 1.69 / 1.93 | 0.55 | 1.18 | 1.18 / 1.23 | 2.0/20 | 1.74 | 0.98 / 0.99 | 14.0 | - | 0.0 |
| sh_multi | cbs_a1 | 1 | 1/1 | 0.61 / 0.61 | 3.92 | 2.02 | 0.46 / 0.46 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| sh_multi | ecb25a_t1 | 20 | 20/20 | 1.28 / 1.49 | 1.73 | 8.66 | 5.29 / 5.94 | 0.0/20 | 14.57 | 1.00 / 1.00 | 6.0 | - | 0.0 |
| sh_slew | cbs_a1 | 1 | 1/1 | 0.29 / 0.29 | 3.67 | 2.14 | 0.37 / 0.37 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| sh_slew | ecb25a_t1 | 20 | 20/20 | 1.35 / 1.52 | 1.66 | 9.01 | 6.13 / 6.79 | 0.0/20 | 16.59 | 0.99 / 1.00 | 17.0 | - | 0.0 |
| sh_tele | cbs_a1 | 1 | 1/1 | 0.33 / 0.33 | 2.57 | 1.26 | 0.38 / 0.38 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| sh_tele | ecb25a_t1 | 20 | 20/20 | 1.49 / 1.87 | 0.64 | 4.95 | 6.35 / 6.47 | 0.0/20 | >20 | 0.98 / 1.00 | 8.0 | - | 0.0 |
| slew | cbs_a1 | 1 | 1/1 | 0.52 / 0.52 | 7.10 | 4.79 | 0.55 / 0.55 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| slew | ecb25a_t1 | 20 | 20/20 | 1.72 / 1.92 | 3.51 | 17.73 | 4.73 / 5.41 | 0.0/20 | 12.18 | 1.00 / 1.00 | 9.0 | - | 0.0 |
| stow | cbs_a1 | 1 | 1/1 | 0.34 / 0.34 | 7.41 | 4.21 | 0.41 / 0.41 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| stow | ecb25a_t1 | 20 | 20/20 | 1.57 / 1.80 | 4.47 | 9.56 | 5.03 / 5.53 | 0.0/20 | 19.18 | 0.99 / 0.76 | 0.0 | - | 0.0 |
| telescope | cbs_a1 | 1 | 1/1 | 0.38 / 0.38 | 4.76 | 2.63 | 0.35 / 0.35 | 1.0/1 | 0.00 | 0.00 / 0.00 | 0.0 | - | 0.0 |
| telescope | ecb25a_t1 | 20 | 20/20 | 1.35 / 1.57 | 2.96 | 6.10 | 2.62 / 2.90 | 0.0/20 | 8.09 | 0.99 / 1.00 | 0.0 | - | 0.0 |

## CBS time split

| move | planner | acados threads | ocp s | geom s | rest s | iterations | ms / iteration |
|---|---|---|---|---|---|---|---|
| across | cbs_a1 | 1.0 | 0.22 | 0.18 | 0.00 | 13.00 | 15.48 |
| ax_arm | cbs_a1 | 1.0 | 0.59 | 0.06 | -0.26 | 29.00 | 8.51 |
| ax_boom | cbs_a1 | 1.0 | 0.24 | 0.07 | 0.00 | 16.00 | 13.89 |
| ax_rot | cbs_a1 | 1.0 | 0.98 | 0.04 | 0.00 | 61.00 | 15.80 |
| ax_slew | cbs_a1 | 1.0 | 0.41 | 0.15 | 0.00 | 33.00 | 12.07 |
| ax_tele | cbs_a1 | 1.0 | 0.31 | 0.06 | 0.00 | 23.00 | 13.00 |
| bench_carry | cbs_a1 | 1.0 | 0.89 | 5.03 | 0.04 | 60.00 | 13.67 |
| example_wall_nobody | cbs_a1 | 1.0 | 0.31 | 0.14 | 0.01 | 25.00 | 9.41 |
| example_wall_body | cbs_a1 | 1.0 | 0.29 | 0.26 | 0.01 | 16.00 | 13.70 |
| lift | cbs_a1 | 1.0 | 0.32 | 0.11 | 0.00 | 21.00 | 14.50 |
| lim_boom_low | cbs_a1 | 1.0 | 0.88 | 0.03 | -0.40 | 43.00 | 9.06 |
| lim_tele_out | cbs_a1 | 1.0 | 0.22 | 0.06 | 0.00 | 16.00 | 12.85 |
| pair_reach | cbs_a1 | 1.0 | 0.43 | 0.04 | 0.00 | 44.00 | 9.52 |
| pair_slew_lift | cbs_a1 | 1.0 | 0.22 | 0.14 | 0.00 | 18.00 | 11.42 |
| pair_tuck | cbs_a1 | 1.0 | 0.24 | 0.18 | 0.00 | 22.00 | 10.38 |
| sh_boom | cbs_a1 | 1.0 | 0.39 | 0.04 | -0.17 | 15.00 | 10.71 |
| sh_multi | cbs_a1 | 1.0 | 0.56 | 0.06 | 0.00 | 36.00 | 14.77 |
| sh_slew | cbs_a1 | 1.0 | 0.26 | 0.03 | 0.00 | 25.00 | 9.63 |
| sh_tele | cbs_a1 | 1.0 | 0.53 | 0.03 | -0.23 | 21.00 | 10.38 |
| slew | cbs_a1 | 1.0 | 0.45 | 0.07 | 0.00 | 40.00 | 10.80 |
| stow | cbs_a1 | 1.0 | 0.23 | 0.11 | 0.00 | 19.00 | 11.41 |
| telescope | cbs_a1 | 1.0 | 0.34 | 0.04 | 0.00 | 36.00 | 9.16 |

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
