#!/bin/sh
for c in example_wall_body example_wall_nobody bench_carry slew lift ax_slew; do
  nice python3 sens/mj.py --only $c --seeds 5 --out /workspaces/ros2_baustelle_ws/src/concrete_block_stack/crane_planning/bench_vpsto/sens/out/mj_$c.json
done
