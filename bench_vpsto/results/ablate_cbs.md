# CBS duration ablation (ablate_cbs.py)

| variant | median T s | planned | settled at T | median residual sway deg |
|---|---|---|---|---|
| shipped | 6.05 | 22/22 | 22/22 | 0.40 |
| settled box removed | 6.05 | 22/22 | 22/22 | 0.40 |
| jerk preference off | 6.61 | 20/22 | 20/22 | 0.37 |
| settled box removed, jerk off | 6.13 | 21/22 | 21/22 | 0.42 |
| sway + damping cost off | 5.09 | 21/22 | 3/22 | 5.80 |
| ... + sway bound in transit 0.6 rad | 5.02 | 22/22 | 3/22 | 5.44 |

## Duration per move, s

| move | shipped | settled box removed | jerk preference off | settled box removed, jerk off | sway + damping cost off | ... + sway bound in transit 0.6 rad |
|---|---|---|---|---|---|---|
| slew | 7.15 | 7.15 | 7.09 | 7.09 | 5.09 | 5.06 |
| telescope | 4.80 | 4.81 | 4.70 | 4.71 | 4.39 | 4.38 |
| lift | 7.23 | 7.23 | 7.17 | 7.17 | 5.86 | 5.86 |
| stow | 7.23 | 7.23 | 7.16 | 7.16 | 6.07 | 6.07 |
| across | 16.04 | 16.04 | 16.03 | 16.03 | 15.29 | 15.29 |
| ax_slew | 10.64 | 10.64 | 10.64 | 10.64 | 7.57 | 6.59 |
| ax_boom | 6.27 | 6.27 | 6.13 | 6.13 | 6.14 | 6.14 |
| ax_arm | 4.54 | 4.54 | 4.40 | 4.40 | 4.24 | 4.24 |
| ax_tele | 5.42 | 5.42 | 5.30 | 5.31 | 4.71 | 4.72 |
| ax_rot | 2.75 | 2.75 | 2.70 | 2.66 | 2.75 | 2.75 |
| sh_slew | 3.71 | 3.71 | 3.70 | 3.70 | 2.62 | 2.62 |
| sh_boom | 2.08 | 2.08 | nan | 2.04 | 1.96 | 1.96 |
| sh_tele | 2.57 | 2.57 | nan | nan | 2.07 | 2.07 |
| sh_multi | 3.96 | 3.96 | 3.91 | 3.91 | 2.59 | 2.59 |
| lim_boom_low | 4.29 | 4.29 | 4.14 | 4.14 | 3.60 | 3.59 |
| lim_tele_out | 5.83 | 5.83 | 5.71 | 5.71 | 4.96 | 4.97 |
| pair_slew_lift | 11.36 | 11.36 | 11.31 | 11.31 | 10.69 | 10.69 |
| pair_reach | 5.32 | 5.32 | 5.18 | 5.18 | nan | 4.67 |
| pair_tuck | 12.83 | 12.83 | 12.71 | 12.71 | 12.47 | 12.47 |
| bench_carry | 11.57 | 11.57 | 11.51 | 11.51 | 10.13 | 12.22 |
| example_wall_nobody | 7.20 | 7.20 | 7.12 | 7.12 | 6.24 | 6.24 |
| example_wall_body | 7.17 | 7.18 | 7.09 | 7.09 | 6.63 | 6.63 |
