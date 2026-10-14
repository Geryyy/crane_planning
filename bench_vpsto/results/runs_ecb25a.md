# CBS vs VP-STO (generated 2026-10-14)

CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, nproc 8, cpus unpinned; VP-STO OpenMP threads [1], nVia 8, nEval 80, population 50, 20 runs/move/setting; CBS acados OpenMP threads [1] (set explicitly before every plan), 1 repeats/move. Linux-7.0.0-38-generic-x86_64-with-glibc2.39

## Aggregate (without the excluded moves)

| planner | runs | settled at T | median residual sway deg | median duration s | median plan s |
|---|---|---|---|---|---|
| cbs_a1 | 22 | 22/22 | 0.40 | 6.05 | 0.75 |
| ecb25a_t1 | 440 | 74/440 | 2.53 | 3.79 | 2.80 |

## Per move

| move | planner | n | success | plan s med / p95 | duration s | peak sway | residual sway med / max | settled at T | settle s | acc0 / accT | over limit | min clearance eq / sway | load flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slew | cbs_a1 | 1 | 1/1 | 2.26 / 2.26 | 7.15 | 3.94 | 0.57 / 0.57 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| slew | ecb25a_t1 | 20 | 20/20 | 3.34 / 3.77 | 3.51 | 17.68 | 4.70 / 4.97 | 0/20 | 12.16 | 1.00 / 1.00 | 10 | - | 20 |
| telescope | cbs_a1 | 1 | 1/1 | 0.31 / 0.31 | 4.80 | 2.60 | 0.36 / 0.36 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| telescope | ecb25a_t1 | 20 | 20/20 | 2.59 / 3.02 | 3.11 | 5.50 | 1.87 / 2.20 | 0/20 | 4.48 | 0.98 / 0.99 | 0 | - | 20 |
| lift | cbs_a1 | 1 | 1/1 | 1.70 / 1.70 | 7.23 | 4.34 | 0.48 / 0.48 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| lift | ecb25a_t1 | 20 | 20/20 | 2.85 / 3.29 | 4.73 | 8.68 | 1.71 / 2.57 | 0/20 | 2.91 | 0.57 / 0.64 | 0 | - | 20 |
| stow | cbs_a1 | 1 | 1/1 | 0.55 / 0.55 | 7.23 | 3.98 | 0.47 / 0.47 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| stow | ecb25a_t1 | 20 | 20/20 | 2.90 / 3.56 | 4.88 | 7.46 | 2.95 / 3.83 | 0/20 | 10.02 | 0.84 / 0.91 | 0 | - | 20 |
| across | cbs_a1 | 1 | 1/1 | 2.42 / 2.42 | 16.04 | 4.78 | 0.88 / 0.88 | 1/1 | 0.00 | 0.00 / 0.00 | 1 | - | 1 |
| across | ecb25a_t1 | 20 | 20/20 | 4.33 / 5.65 | 9.17 | 6.05 | 1.64 / 2.83 | 1/20 | 3.52 | 0.45 / 0.36 | 1 | - | 20 |
| ax_slew | cbs_a1 | 1 | 1/1 | 1.42 / 1.42 | 10.64 | 7.26 | 0.56 / 0.56 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| ax_slew | ecb25a_t1 | 20 | 20/20 | 2.37 / 3.04 | 6.23 | 21.54 | 6.87 / 7.49 | 0/20 | 18.35 | 0.59 / 0.81 | 0 | - | 20 |
| ax_boom | cbs_a1 | 1 | 1/1 | 0.36 / 0.36 | 6.27 | 2.01 | 0.40 / 0.40 | 1/1 | 0.00 | 0.00 / 0.00 | 1 | - | 1 |
| ax_boom | ecb25a_t1 | 20 | 20/20 | 2.93 / 3.48 | 5.04 | 3.56 | 2.91 / 3.88 | 0/20 | 9.82 | 0.62 / 0.72 | 0 | - | 20 |
| ax_arm | cbs_a1 | 1 | 1/1 | 0.48 / 0.48 | 4.54 | 1.67 | 0.38 / 0.38 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| ax_arm | ecb25a_t1 | 20 | 20/20 | 2.35 / 2.81 | 3.01 | 3.87 | 2.00 / 2.40 | 0/20 | 5.54 | 0.88 / 0.93 | 0 | - | 20 |
| ax_tele | cbs_a1 | 1 | 1/1 | 0.52 / 0.52 | 5.42 | 2.95 | 0.37 / 0.37 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| ax_tele | ecb25a_t1 | 20 | 20/20 | 2.31 / 2.71 | 3.81 | 5.19 | 0.73 / 1.15 | 20/20 | 0.00 | 0.88 / 0.92 | 0 | - | 20 |
| ax_rot | cbs_a1 | 1 | 1/1 | 1.52 / 1.52 | 2.75 | 0.00 | 0.00 / 0.00 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| ax_rot | ecb25a_t1 | 20 | 20/20 | 1.90 / 2.38 | 1.27 | 0.34 | 0.19 / 0.56 | 20/20 | 0.00 | 1.00 / 1.00 | 0 | - | 20 |
| sh_slew | cbs_a1 | 1 | 1/1 | 0.53 / 0.53 | 3.71 | 2.04 | 0.35 / 0.35 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| sh_slew | ecb25a_t1 | 20 | 20/20 | 2.60 / 3.43 | 1.66 | 9.07 | 6.10 / 7.06 | 0/20 | 16.55 | 1.00 / 0.99 | 15 | - | 20 |
| sh_boom | cbs_a1 | 1 | 1/1 | 0.37 / 0.37 | 2.08 | 0.71 | 0.71 / 0.71 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| sh_boom | ecb25a_t1 | 20 | 20/20 | 3.11 / 3.62 | 0.54 | 1.15 | 1.19 / 1.24 | 3/20 | 1.75 | 0.98 / 0.99 | 13 | - | 20 |
| sh_tele | cbs_a1 | 1 | 1/1 | 0.54 / 0.54 | 2.57 | 1.26 | 0.38 / 0.38 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| sh_tele | ecb25a_t1 | 20 | 20/20 | 3.01 / 3.46 | 0.64 | 4.93 | 6.34 / 6.39 | 0/20 | >20 | 0.98 / 0.99 | 14 | - | 20 |
| sh_multi | cbs_a1 | 1 | 1/1 | 0.63 / 0.63 | 3.96 | 2.12 | 0.44 / 0.44 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| sh_multi | ecb25a_t1 | 20 | 20/20 | 2.40 / 2.75 | 1.73 | 8.75 | 5.48 / 5.87 | 0/20 | 15.51 | 1.00 / 1.00 | 5 | - | 20 |
| lim_boom_low | cbs_a1 | 1 | 1/1 | 0.73 / 0.73 | 4.29 | 2.20 | 0.37 / 0.37 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| lim_boom_low | ecb25a_t1 | 20 | 20/20 | 2.16 / 2.60 | 2.03 | 8.19 | 5.75 / 6.42 | 0/20 | >20 | 0.99 / 1.00 | 1 | - | 20 |
| lim_boom_high (excluded) | cbs_a1 | 1 | 0/1 | - | - | - | - / - | 0/0 | - | - / - | 0 | - | 1 |
| lim_boom_high (excluded) | ecb25a_t1 | 20 | 20/20 | 2.67 / 3.47 | 5.94 | 5.48 | 9.00 / 9.34 | 0/20 | >20 | 0.72 / 0.84 | 1 | - | 20 |
| lim_tele_out | cbs_a1 | 1 | 1/1 | 0.41 / 0.41 | 5.83 | 3.18 | 0.38 / 0.38 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| lim_tele_out | ecb25a_t1 | 20 | 20/20 | 2.38 / 2.88 | 4.32 | 4.88 | 0.63 / 1.03 | 20/20 | 0.00 | 0.82 / 0.85 | 0 | - | 20 |
| lim_arm_out (excluded) | cbs_a1 | 1 | 0/1 | - | - | - | - / - | 0/0 | - | - / - | 0 | - | 1 |
| lim_arm_out (excluded) | ecb25a_t1 | 20 | 20/20 | 3.94 / 4.30 | 8.48 | 4.02 | 3.37 / 4.49 | 0/20 | 11.72 | 0.50 / 0.87 | 3 | - | 20 |
| pair_slew_lift | cbs_a1 | 1 | 1/1 | 1.09 / 1.09 | 11.36 | 5.35 | 1.07 / 1.07 | 1/1 | 0.75 | 0.00 / 0.00 | 1 | - | 1 |
| pair_slew_lift | ecb25a_t1 | 20 | 20/20 | 2.95 / 3.73 | 6.25 | 9.75 | 2.81 / 4.42 | 0/20 | 9.66 | 0.51 / 0.69 | 2 | - | 20 |
| pair_reach | cbs_a1 | 1 | 1/1 | 0.78 / 0.78 | 5.32 | 2.91 | 0.38 / 0.38 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | - | 1 |
| pair_reach | ecb25a_t1 | 20 | 20/20 | 2.80 / 3.35 | 3.02 | 4.78 | 2.44 / 2.70 | 0/20 | 8.19 | 1.00 / 1.00 | 0 | - | 20 |
| pair_tuck | cbs_a1 | 1 | 1/1 | 2.62 / 2.62 | 12.83 | 3.66 | 0.78 / 0.78 | 1/1 | 0.00 | 0.00 / 0.00 | 1 | - | 1 |
| pair_tuck | ecb25a_t1 | 20 | 20/20 | 3.57 / 4.65 | 8.21 | 5.86 | 2.82 / 3.67 | 0/20 | 10.17 | 0.71 / 0.77 | 0 | - | 20 |
| bench_carry | cbs_a1 | 1 | 1/1 | 4.25 / 4.25 | 11.57 | 7.38 | 0.71 / 0.71 | 1/1 | 0.00 | 0.00 / 0.00 | 1 | 0.54 / 0.53 | 1 |
| bench_carry | ecb25a_t1 | 20 | 3/20 | 8.53 / 11.33 | 3.75 | 17.80 | 4.90 / 5.37 | 0/20 | >20 | 1.00 / 1.00 | 0 | 0.00 / 0.01 | 20 |
| example_wall_nobody | cbs_a1 | 1 | 1/1 | 0.96 / 0.96 | 7.20 | 2.92 | 0.38 / 0.38 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | 2.12 / 2.13 | 1 |
| example_wall_nobody | ecb25a_t1 | 20 | 20/20 | 4.74 / 7.76 | 5.62 | 4.64 | 1.49 / 2.93 | 8/20 | 9.13 | 0.93 / 0.78 | 1 | 1.81 / 1.80 | 20 |
| example_wall_body | cbs_a1 | 1 | 1/1 | 1.77 / 1.77 | 7.18 | 3.32 | 0.39 / 0.39 | 1/1 | 0.00 | 0.00 / 0.00 | 0 | 1.13 / 1.13 | 1 |
| example_wall_body | ecb25a_t1 | 20 | 20/20 | 4.48 / 8.01 | 5.61 | 4.44 | 1.66 / 2.56 | 2/20 | 18.78 | 0.77 / 0.72 | 0 | 1.13 / 1.13 | 20 |

## CBS time split

| move | planner | acados threads | ocp s | geom s | rest s | iterations | ms / iteration |
|---|---|---|---|---|---|---|---|
| slew | cbs_a1 | 1 | 2.02 | 0.23 | 0.01 | 67.00 | 29.59 |
| telescope | cbs_a1 | 1 | 0.22 | 0.08 | 0.00 | 12.00 | 16.38 |
| lift | cbs_a1 | 1 | 1.48 | 0.22 | 0.00 | 68.00 | 21.33 |
| stow | cbs_a1 | 1 | 0.30 | 0.25 | 0.00 | 11.00 | 24.57 |
| across | cbs_a1 | 1 | 1.79 | 0.62 | 0.00 | 70.00 | 25.20 |
| ax_slew | cbs_a1 | 1 | 0.66 | 0.76 | 0.00 | 34.00 | 18.38 |
| ax_boom | cbs_a1 | 1 | 0.20 | 0.15 | 0.00 | 9.00 | 19.07 |
| ax_arm | cbs_a1 | 1 | 0.36 | 0.12 | 0.00 | 15.00 | 22.25 |
| ax_tele | cbs_a1 | 1 | 0.39 | 0.12 | 0.00 | 16.00 | 22.76 |
| ax_rot | cbs_a1 | 1 | 1.46 | 0.06 | 0.00 | 58.00 | 24.61 |
| sh_slew | cbs_a1 | 1 | 0.45 | 0.08 | 0.00 | 22.00 | 19.06 |
| sh_boom | cbs_a1 | 1 | 0.31 | 0.06 | 0.00 | 14.00 | 19.91 |
| sh_tele | cbs_a1 | 1 | 0.46 | 0.08 | 0.00 | 22.00 | 19.72 |
| sh_multi | cbs_a1 | 1 | 0.55 | 0.08 | 0.00 | 27.00 | 19.25 |
| lim_boom_low | cbs_a1 | 1 | 0.64 | 0.08 | 0.00 | 35.00 | 17.59 |
| lim_tele_out | cbs_a1 | 1 | 0.26 | 0.15 | 0.00 | 11.00 | 20.75 |
| pair_slew_lift | cbs_a1 | 1 | 0.69 | 0.40 | 0.00 | 29.00 | 22.87 |
| pair_reach | cbs_a1 | 1 | 0.60 | 0.17 | 0.00 | 34.00 | 17.00 |
| pair_tuck | cbs_a1 | 1 | 2.11 | 0.51 | 0.00 | 100.00 | 20.72 |
| bench_carry | cbs_a1 | 1 | 0.37 | 3.79 | 0.09 | 14.00 | 17.22 |
| example_wall_nobody | cbs_a1 | 1 | 0.60 | 0.34 | 0.01 | 21.00 | 22.58 |
| example_wall_body | cbs_a1 | 1 | 1.25 | 0.50 | 0.02 | 40.00 | 27.52 |

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
