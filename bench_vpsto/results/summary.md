# CBS vs VP-STO (generated 2026-10-13)

CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, nproc 8; VP-STO 8 OpenMP threads, nVia 6, nEval 80, population 50, 20 runs/move; CBS acados single thread, 3 repeats/move. Linux-7.0.0-38-generic-x86_64-with-glibc2.39

| move | planner | n | success | plan s med / p95 | duration s | term sway deg med / max | peak sway deg | settled at T | settle s | min clearance eq / sway m |
|---|---|---|---|---|---|---|---|---|---|---|
| slew | cbs | 3 | 3/3 | 0.98 / 1.08 | 7.15 | 0.57 / 0.57 | 3.94 | 3/3 | 0.00 | - |
| slew | vpsto | 20 | 20/20 | 0.09 / 1.17 | 3.02 | 15.83 / 22.14 | 49.72 | 0/20 | >20 | - |
| telescope | cbs | 3 | 3/3 | 0.19 / 0.27 | 4.80 | 0.36 / 0.36 | 2.60 | 3/3 | 0.00 | - |
| telescope | vpsto | 20 | 20/20 | 0.08 / 0.18 | 3.24 | 4.16 / 9.96 | 7.81 | 1/20 | >20 | - |
| lift | cbs | 3 | 3/3 | 1.90 / 5.34 | 7.23 | 0.48 / 0.48 | 4.34 | 3/3 | 0.00 | - |
| lift | vpsto | 20 | 20/20 | 0.12 / 2.56 | 4.94 | 9.23 / 37.78 | 18.03 | 0/20 | >20 | - |
| stow | cbs | 3 | 3/3 | 0.46 / 0.72 | 7.23 | 0.47 / 0.47 | 3.98 | 3/3 | 0.00 | - |
| stow | vpsto | 20 | 20/20 | 0.10 / 0.25 | 5.08 | 12.85 / 21.59 | 21.72 | 0/20 | >20 | - |
| across | cbs | 3 | 3/3 | 2.13 / 2.13 | 16.04 | 0.88 / 0.88 | 4.78 | 3/3 | 0.00 | - |
| across | vpsto | 20 | 20/20 | 0.16 / 0.50 | 9.52 | 8.21 / 22.20 | 9.76 | 0/20 | >20 | - |
| ax_slew | cbs | 3 | 3/3 | 1.61 / 2.17 | 10.64 | 0.56 / 0.56 | 7.26 | 3/3 | 0.00 | - |
| ax_slew | vpsto | 20 | 20/20 | 0.12 / 0.77 | 5.78 | 26.26 / 32.75 | 39.70 | 0/20 | >20 | - |
| ax_boom | cbs | 3 | 3/3 | 0.19 / 0.19 | 6.27 | 0.40 / 0.40 | 2.01 | 3/3 | 0.00 | - |
| ax_boom | vpsto | 20 | 20/20 | 0.05 / 0.15 | 5.25 | 6.10 / 11.86 | 6.33 | 0/20 | >20 | - |
| ax_arm | cbs | 3 | 3/3 | 0.22 / 0.24 | 4.54 | 0.38 / 0.38 | 1.67 | 3/3 | 0.00 | - |
| ax_arm | vpsto | 20 | 20/20 | 0.05 / 0.22 | 3.12 | 3.69 / 9.54 | 7.73 | 1/20 | >20 | - |
| ax_tele | cbs | 3 | 3/3 | 0.27 / 1.07 | 5.42 | 0.37 / 0.37 | 2.95 | 3/3 | 0.00 | - |
| ax_tele | vpsto | 20 | 20/20 | 0.05 / 0.14 | 3.97 | 5.58 / 13.24 | 8.13 | 0/20 | >20 | - |
| ax_rot | cbs | 3 | 3/3 | 0.73 / 0.95 | 2.75 | 0.00 / 0.00 | 0.00 | 3/3 | 0.00 | - |
| ax_rot | vpsto | 20 | 20/20 | 0.03 / 0.61 | 1.26 | 2.37 / 6.04 | 2.69 | 1/20 | 6.31 | - |
| sh_slew | cbs | 3 | 3/3 | 0.28 / 2.06 | 3.71 | 0.35 / 0.35 | 2.04 | 3/3 | 0.00 | - |
| sh_slew | vpsto | 20 | 20/20 | 0.05 / 0.18 | 1.01 | 2.38 / 5.88 | 19.77 | 0/20 | >20 | - |
| sh_boom | cbs | 3 | 3/3 | 0.19 / 0.26 | 2.08 | 0.71 / 0.71 | 0.71 | 3/3 | 0.00 | - |
| sh_boom | vpsto | 20 | 20/20 | 0.06 / 1.01 | 0.54 | 1.12 / 1.23 | 1.27 | 6/20 | 1.78 | - |
| sh_tele | cbs | 3 | 3/3 | 0.97 / 2.13 | 2.57 | 0.38 / 0.38 | 1.26 | 3/3 | 0.00 | - |
| sh_tele | vpsto | 20 | 20/20 | 0.06 / 3.92 | 0.64 | 4.35 / 4.50 | 5.22 | 0/20 | >20 | - |
| sh_multi | cbs | 3 | 3/3 | 4.81 / 5.65 | 3.96 | 0.44 / 0.44 | 2.12 | 3/3 | 0.00 | - |
| sh_multi | vpsto | 20 | 20/20 | 0.32 / 5.09 | 1.01 | 2.82 / 4.06 | 19.55 | 0/20 | >20 | - |
| lim_boom_low | cbs | 3 | 3/3 | 2.00 / 2.47 | 4.29 | 0.37 / 0.37 | 2.20 | 3/3 | 0.00 | - |
| lim_boom_low | vpsto | 20 | 20/20 | 0.04 / 0.12 | 2.10 | 9.42 / 11.82 | 9.58 | 0/20 | >20 | - |
| lim_boom_high | cbs | 3 | 0/3 | 0.03 / 0.03 | - | - | - | - | - | - |
| lim_boom_high | vpsto | 20 | 20/20 | 0.06 / 0.10 | 6.04 | 11.12 / 11.68 | 11.12 | 0/20 | >20 | - |
| lim_tele_out | cbs | 3 | 3/3 | 0.20 / 0.20 | 5.83 | 0.38 / 0.38 | 3.18 | 3/3 | 0.00 | - |
| lim_tele_out | vpsto | 20 | 20/20 | 0.05 / 0.16 | 4.49 | 7.76 / 23.04 | 9.35 | 0/20 | >20 | - |
| lim_arm_out | cbs | 3 | 0/3 | 0.01 / 0.01 | - | - | - | - | - | - |
| lim_arm_out | vpsto | 20 | 20/20 | 0.05 / 0.11 | 8.82 | 14.59 / 19.23 | 14.59 | 0/20 | >20 | - |
| pair_slew_lift | cbs | 3 | 3/3 | 0.52 / 0.52 | 11.36 | 1.07 / 1.07 | 5.35 | 3/3 | 0.75 | - |
| pair_slew_lift | vpsto | 20 | 20/20 | 0.05 / 0.13 | 6.51 | 8.02 / 16.57 | 12.88 | 0/20 | >20 | - |
| pair_reach | cbs | 3 | 3/3 | 0.36 / 0.46 | 5.32 | 0.38 / 0.38 | 2.91 | 3/3 | 0.00 | - |
| pair_reach | vpsto | 20 | 20/20 | 0.04 / 0.08 | 3.00 | 3.54 / 7.46 | 5.01 | 0/20 | >20 | - |
| pair_tuck | cbs | 3 | 3/3 | 1.53 / 1.56 | 12.83 | 0.78 / 0.78 | 3.66 | 3/3 | 0.00 | - |
| pair_tuck | vpsto | 20 | 20/20 | 0.07 / 0.11 | 8.53 | 10.76 / 16.83 | 14.56 | 0/20 | >20 | - |
| bench_carry | cbs | 3 | 3/3 | 2.67 / 2.77 | 11.46 | 0.63 / 0.63 | 7.09 | 3/3 | 0.00 | 0.54 / 0.53 |
| bench_carry | vpsto | 20 | 20/20 | 0.99 / 1.33 | 3.00 | 61.54 / 64.79 | 63.00 | 0/20 | >20 | 0.01 / 0.08 |
| example_wall | cbs | 3 | 3/3 | 0.52 / 0.53 | 7.23 | 0.39 / 0.39 | 3.05 | 3/3 | 0.00 | 2.12 / 2.13 |
| example_wall | vpsto | 20 | 20/20 | 0.16 / 0.24 | 5.84 | 11.72 / 23.98 | 13.02 | 0/20 | >20 | 1.73 / 1.68 |

Columns: plan s = wall of the whole call (CBS Planner.plan incl. geometry; VP-STO vc_solve incl.
basis setup); duration = plan T; sway/settle from the shared passive-dynamics replay (start at
rest, q_u0 = the planner's start; settle = time after T to enter and stay in the 0.02 rad /
0.04 rad/s box with actuators held, ">20" = not within 20 s); clearance = worst run, 80 samples
along the path, scene bodies only (empty for the 21 bench moves). VP-STO success = final cost
has no penalty. Both planners stay inside their limits (see runs.csv util_*, fractions of the
kappa-scaled bound). lim_boom_high / lim_arm_out: CBS refuses (goal folds the machine into
itself); VP-STO has no self-collision term and was given the bench goal joints. Plan times are
load-bound; an unrelated Gazebo session ran during a first pass, which was discarded.
