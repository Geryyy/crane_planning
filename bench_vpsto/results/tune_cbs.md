# CBS duration by cost weights (tune_cbs.py)

## Trials

| trial | time | dq_a | sway | dq_u | dynamic_tau | ddq_a | dddq_a | median T s | failed | max it |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 15.7 | 0.877 | 26.2 | 0.0255 | 0.241 | 0.0254 | 1.86 | 5.34 | slew,lift,stow,across,ax_boom,ax_arm,ax_tele,sh_slew,sh_multi,lim_tele_out,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 27 | 2.82 | 0.794 | 0.571 | 5.75 | 3.16 | 1.22 | 0.654 | 5.41 | stow,across,ax_slew,ax_boom,sh_tele,sh_multi,bench_carry,example_wall_nobody | 100 |
| 3 | 10 | 1 | 4 | 4 | 1 | 1 | 1 | 5.56 | slew,lift,stow,across,ax_boom,ax_arm,ax_tele,sh_multi,lim_tele_out,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 46 | 1.1 | 0.272 | 0.765 | 2.22 | 3.51 | 0.347 | 0.478 | 5.57 | across,pair_slew_lift,bench_carry | 100 |
| 2 | 3 | 1 | 4 | 4 | 1 | 1 | 1 | 5.59 | lift,stow,across,ax_boom | 100 |
| 39 | 0.609 | 0.619 | 0.237 | 1.04 | 2.34 | 0.24 | 0.616 | 5.65 | lift,across,pair_slew_lift,bench_carry | 100 |
| 11 | 1.26 | 0.0151 | 0.341 | 0.0127 | 0.388 | 1.81 | 0.093 | 5.70 | telescope,lift,across,ax_slew,ax_boom,ax_rot,sh_tele,pair_tuck,bench_carry | 100 |
| 21 | 0.914 | 0.153 | 0.472 | 6.06 | 0.167 | 0.574 | 1.74 | 5.83 | across,ax_slew,example_wall_body | 100 |
| 33 | 1.07 | 0.737 | 1.73 | 2.29 | 2.83 | 0.43 | 1.07 | 5.85 | across,pair_slew_lift | 100 |
| 43 | 0.701 | 0.221 | 0.932 | 3.3 | 2 | 0.133 | 1.09 | 5.87 | across | 100 |
| 31 | 1.64 | 2.21 | 1.86 | 12.8 | 0.482 | 0.311 | 0.468 | 5.88 | across,ax_slew | 100 |
| 9 | 5.35 | 3.2 | 0.01 | 0.113 | 0.0324 | 0.0209 | 0.0444 | 5.89 | slew,telescope,lift,stow,across,ax_slew,ax_boom,ax_arm,ax_tele,sh_slew,sh_boom,sh_tele,sh_multi,lim_tele_out,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 8 | 13.3 | 3.07 | 1.25 | 0.785 | 0.297 | 1.76 | 0.332 | 5.93 | slew,telescope,lift,stow,across,ax_slew,ax_boom,ax_arm,ax_tele,sh_boom,sh_tele,sh_multi,lim_tele_out,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 4 | 30 | 1 | 4 | 4 | 1 | 1 | 1 | 5.93 | slew,telescope,lift,stow,across,ax_slew,ax_boom,ax_arm,ax_tele,sh_boom,sh_tele,sh_multi,lim_boom_low,lim_tele_out,pair_slew_lift,pair_reach,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 44 | 0.964 | 1.45 | 0.356 | 9.84 | 3.65 | 0.186 | 0.362 | 6.01 | ax_slew | 100 |
| 1 | 1 | 1 | 4 | 4 | 1 | 1 | 1 | 6.05 | - | 100 |
| 13 | 4 | 0.171 | 3.63 | 16.4 | 23.1 | 2.57 | 0.132 | 6.05 | lift,across,ax_rot,example_wall_nobody | 100 |
| 6 | 10.4 | 0.507 | 0.0232 | 0.0364 | 0.829 | 0.0514 | 0.0112 | 6.12 | slew,telescope,lift,stow,across,ax_slew,ax_boom,ax_arm,ax_tele,ax_rot,sh_slew,sh_boom,sh_tele,lim_boom_low,lim_tele_out,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 5 | 100 | 1 | 4 | 4 | 1 | 1 | 1 | 6.13 | slew,telescope,lift,stow,ax_slew,ax_boom,ax_arm,ax_tele,sh_slew,sh_boom,sh_tele,sh_multi,lim_boom_low,lim_tele_out,pair_slew_lift,pair_reach,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 42 | 0.634 | 0.415 | 0.341 | 3.32 | 3.17 | 0.251 | 0.311 | 6.13 | lift,across,ax_rot,pair_tuck | 100 |
| 36 | 0.408 | 0.166 | 0.695 | 5.39 | 3.96 | 0.902 | 0.258 | 6.18 | - | 100 |
| 19 | 0.471 | 0.845 | 0.397 | 3.11 | 2.72 | 0.33 | 0.624 | 6.22 | - | 100 |
| 15 | 1.2 | 1.03 | 1.79 | 2.14 | 1.28 | 1.95 | 13.3 | 6.32 | ax_rot | 100 |
| 51 | 0.371 | 0.362 | 2.23 | 0.93 | 3.34 | 0.492 | 0.459 | 6.33 | - | 100 |
| 40 | 0.512 | 1.03 | 2.37 | 1.97 | 2.67 | 0.501 | 0.398 | 6.40 | - | 100 |
| 23 | 0.392 | 1.39 | 0.127 | 0.865 | 0.714 | 0.346 | 1.03 | 6.48 | across,ax_rot | 100 |
| 38 | 0.326 | 0.968 | 0.428 | 1.3 | 1.1 | 0.559 | 0.262 | 6.49 | across | 100 |
| 47 | 0.311 | 0.441 | 0.741 | 2.45 | 4.56 | 0.201 | 0.638 | 6.55 | - | 100 |
| 50 | 0.341 | 0.215 | 2.91 | 7.06 | 3.73 | 0.792 | 0.519 | 6.60 | - | 100 |
| 7 | 13.5 | 2.9 | 0.103 | 0.597 | 12.7 | 13.1 | 0.0273 | 6.60 | slew,lift,stow,across,ax_slew,ax_tele,ax_rot,sh_slew,sh_boom,sh_tele,sh_multi,pair_slew_lift,pair_tuck,bench_carry,example_wall_nobody | 100 |
| 22 | 1.36 | 0.236 | 3.75 | 24.2 | 3.02 | 0.466 | 0.109 | 6.64 | across,ax_boom,sh_multi | 100 |
| 52 | 0.409 | 0.237 | 2.25 | 2.74 | 9.86 | 0.104 | 0.429 | 6.68 | - | 100 |
| 35 | 0.424 | 1.21 | 0.72 | 2.67 | 3.67 | 0.801 | 0.402 | 6.71 | - | 100 |
| 48 | 0.614 | 0.753 | 5.15 | 1.64 | 1.58 | 0.713 | 0.565 | 6.89 | ax_rot | 100 |
| 34 | 0.849 | 3.48 | 0.121 | 1.3 | 8.36 | 0.324 | 1.52 | 6.97 | lift,across,ax_arm,ax_rot | 100 |
| 12 | 142 | 22.4 | 0.215 | 5.66 | 0.69 | 0.944 | 16.5 | 6.98 | slew,lift,stow,ax_slew,ax_boom,ax_arm,ax_rot,sh_slew,sh_tele,lim_boom_low,lim_tele_out,pair_reach,pair_tuck,bench_carry,example_wall_nobody,example_wall_body | 100 |
| 37 | 0.329 | 0.269 | 0.838 | 7.18 | 1.33 | 0.462 | 0.745 | 7.04 | across,ax_rot | 100 |
| 0 | 0.3 | 1 | 4 | 4 | 1 | 1 | 1 | 7.15 | - | 100 |
| 45 | 0.439 | 0.76 | 0.759 | 6.66 | 7.25 | 0.416 | 1.46 | 7.20 | across,ax_rot | 100 |
| 41 | 0.421 | 0.677 | 0.423 | 1.88 | 10 | 1.58 | 0.701 | 7.28 | ax_rot | 100 |
| 32 | 0.534 | 2.58 | 3.53 | 4.21 | 0.837 | 1.01 | 0.275 | 7.56 | ax_rot | 100 |
| 28 | 0.774 | 5.1 | 0.646 | 5.07 | 6.28 | 0.541 | 1.36 | 7.66 | ax_slew,ax_rot | 100 |
| 16 | 0.369 | 0.0816 | 1.9 | 24.9 | 0.675 | 0.47 | 2.11 | 7.67 | ax_rot | 100 |
| 25 | 0.687 | 5.09 | 1.46 | 0.799 | 5.82 | 0.992 | 1.58 | 7.83 | ax_rot | 100 |
| 29 | 0.397 | 2.8 | 0.492 | 2.64 | 9.17 | 0.158 | 0.521 | 7.84 | - | 100 |
| 49 | 0.31 | 2.3 | 0.456 | 2.26 | 3.2 | 1.5 | 0.535 | 7.94 | - | 100 |
| 17 | 0.34 | 0.426 | 8.63 | 6.01 | 0.131 | 1.51 | 0.759 | 8.04 | ax_tele,sh_slew | 100 |
| 18 | 1.97 | 20.4 | 2.27 | 2.83 | 6.42 | 1.25 | 0.292 | 8.07 | across,ax_rot | 100 |
| 30 | 0.368 | 1.58 | 8.12 | 2.6 | 3.99 | 0.348 | 1.26 | 8.09 | ax_rot | 100 |
| 20 | 0.855 | 8.82 | 7.09 | 3.25 | 3.57 | 5.14 | 4.7 | 8.45 | - | 100 |
| 14 | 0.402 | 1.97 | 24.8 | 2.1 | 0.708 | 0.244 | 1.4 | 9.94 | telescope,ax_tele,ax_rot,lim_tele_out | 100 |
| 24 | 0.386 | 5.18 | 27.9 | 15.7 | 3.29 | 0.948 | 0.185 | 10.57 | ax_rot,lim_tele_out,bench_carry | 100 |

## Best feasible: trial 1

| move | ok | duration s | residual sway deg | settled at T | iterations | util pump |
|---|---|---|---|---|---|---|
| slew | 1 | 7.15 | 0.57 | 1 | 55 | 0.79 |
| telescope | 1 | 4.80 | 0.36 | 1 | 13 | 1.00 |
| lift | 1 | 7.23 | 0.48 | 1 | 76 | 1.00 |
| stow | 1 | 7.23 | 0.47 | 1 | 11 | 1.00 |
| across | 1 | 16.04 | 0.88 | 1 | 70 | 1.04 |
| ax_slew | 1 | 10.64 | 0.56 | 1 | 37 | 0.42 |
| ax_boom | 1 | 6.27 | 0.40 | 1 | 9 | 1.00 |
| ax_arm | 1 | 4.54 | 0.38 | 1 | 12 | 1.00 |
| ax_tele | 1 | 5.42 | 0.37 | 1 | 13 | 1.00 |
| ax_rot | 1 | 2.75 | 0.00 | 1 | 21 | 0.11 |
| sh_slew | 1 | 3.71 | 0.35 | 1 | 100 | 0.06 |
| sh_boom | 1 | 2.08 | 0.71 | 1 | 17 | 0.27 |
| sh_tele | 1 | 2.57 | 0.38 | 1 | 26 | 0.20 |
| sh_multi | 1 | 3.96 | 0.44 | 1 | 29 | 0.32 |
| lim_boom_low | 1 | 4.29 | 0.37 | 1 | 38 | 0.63 |
| lim_tele_out | 1 | 5.83 | 0.38 | 1 | 9 | 1.00 |
| pair_slew_lift | 1 | 11.36 | 1.07 | 1 | 33 | 1.01 |
| pair_reach | 1 | 5.32 | 0.38 | 1 | 16 | 0.75 |
| pair_tuck | 1 | 12.83 | 0.78 | 1 | 90 | 1.03 |
| bench_carry | 1 | 11.57 | 0.71 | 1 | 15 | 1.06 |
| example_wall_nobody | 1 | 7.20 | 0.38 | 1 | 55 | 1.00 |
| example_wall_body | 1 | 7.17 | 0.39 | 1 | 24 | 1.00 |
