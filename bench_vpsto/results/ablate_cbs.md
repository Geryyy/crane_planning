# CBS duration ablation (ablate_cbs.py)

| variant | median T s | planned | settled at T | median residual sway deg |
|---|---|---|---|---|
| shipped | 5.95 | 22/22 | 22/22 | 0.42 |
| settled box removed | 5.95 | 22/22 | 22/22 | 0.42 |
| jerk preference off | 6.78 | 19/22 | 19/22 | 0.41 |
| settled box removed, jerk off | 6.07 | 21/22 | 21/22 | 0.41 |
| sway + damping cost off | 4.76 | 22/22 | 3/22 | 8.56 |
| ... + sway bound in transit 0.6 rad | 4.77 | 22/22 | 3/22 | 8.58 |

## Duration per move, s

| move | shipped | settled box removed | jerk preference off | settled box removed, jerk off | sway + damping cost off | ... + sway bound in transit 0.6 rad |
|---|---|---|---|---|---|---|
| slew | 7.10 | 7.10 | 7.09 | 7.09 | 4.61 | 4.48 |
| telescope | 4.76 | 4.76 | 4.67 | 4.67 | 4.33 | 4.32 |
| lift | 7.07 | 7.07 | 7.02 | 7.03 | 5.69 | 5.65 |
| stow | 7.41 | 7.41 | 7.37 | 7.37 | 5.62 | 5.56 |
| across | 10.74 | 10.74 | 10.72 | 10.72 | 9.83 | 9.63 |
| ax_slew | 10.64 | 10.64 | 10.64 | 10.64 | 7.57 | 6.59 |
| ax_boom | 6.26 | 6.27 | 6.07 | 6.07 | 5.66 | 5.66 |
| ax_arm | 4.34 | 4.35 | 4.24 | 4.25 | 4.00 | 4.04 |
| ax_tele | 5.27 | 5.27 | 5.17 | 5.17 | 4.66 | 4.66 |
| ax_rot | 2.75 | 2.74 | nan | 2.69 | 2.74 | 2.74 |
| sh_slew | 3.67 | 3.67 | 3.66 | 3.66 | 2.97 | 2.97 |
| sh_boom | 2.04 | 2.04 | nan | 1.98 | 1.92 | 1.92 |
| sh_tele | 2.57 | 2.57 | nan | nan | 2.07 | 2.07 |
| sh_multi | 3.92 | 3.92 | 3.88 | 3.88 | 2.39 | 2.39 |
| lim_boom_low | 4.32 | 4.32 | 4.16 | 4.16 | 3.60 | 3.60 |
| lim_tele_out | 5.63 | 5.64 | 5.53 | 5.53 | 4.86 | 4.87 |
| pair_slew_lift | 8.50 | 8.49 | 8.46 | 8.46 | 7.21 | 6.54 |
| pair_reach | 5.32 | 5.32 | 5.18 | 5.19 | 4.66 | 4.67 |
| pair_tuck | 9.68 | 9.68 | 9.65 | 9.65 | 8.29 | 8.08 |
| bench_carry | 9.34 | 9.34 | 9.22 | 9.22 | 7.91 | 7.91 |
| example_wall_nobody | 6.86 | 6.86 | 6.78 | 6.78 | 5.99 | 5.99 |
| example_wall_body | 6.97 | 6.98 | 6.89 | 6.89 | 6.39 | 6.39 |
