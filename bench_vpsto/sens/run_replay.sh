#!/bin/sh
V=ideal,ideal_d0.5,ideal_d1.5,ideal_f0,ideal_f2,ideal_inertia,lp2Hz,lp4Hz
for c in example_wall_body example_wall_nobody bench_carry slew lift ax_slew; do
  nice python3 sens/sens.py --only $c --seeds 5 --variants $V --out /workspaces/ros2_baustelle_ws/src/concrete_block_stack/crane_planning/bench_vpsto/sens/out/replay_$c.json
done
