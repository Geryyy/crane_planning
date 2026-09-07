#!/usr/bin/env bash
# Compile the trajectory solver for the machine a deployment actually publishes.
#
#   ./scripts/prepare_solver.sh                       # the CBS pzs100 sim
#   ./scripts/prepare_solver.sh initial_pose:=2       # extra launch arguments
#
# Brings the deployment up headless, dumps `/robot_description` byte for byte,
# kills it, and compiles a solver named after those bytes. Doing it from the
# running system rather than from a second xacro invocation is the whole point:
# the description depends on launch arguments nobody wants to restate, and
# restating them is how the prepared solver ends up being for another machine.
# `fake_model.launch.py` is not a substitute -- same links and masses, 20 kB more
# XML from `fake_hardware:=true`, and therefore a different hash.
#
# Its own ROS_DOMAIN_ID, so bringing a second crane up next to a running one
# neither joins that graph nor is heard by it.
set -euo pipefail

PACKAGE=${PREPARE_PACKAGE:-concrete_block_behavior_tree}
LAUNCH=${PREPARE_LAUNCH:-pzs100_bringup.launch.py}
OUTPUT=${PREPARE_OUTPUT:-${TMPDIR:-/tmp}/crane_planning_prepare/robot_description.urdf}
export ROS_DOMAIN_ID=${PREPARE_DOMAIN:-77}
# Long, because it covers Gazebo starting, not just the topic being answered.
TIMEOUT=${PREPARE_TIMEOUT:-180}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG=$(dirname "$OUTPUT")/bringup.log
mkdir -p "$(dirname "$OUTPUT")"

# Own process group: `ros2 launch` leaves gzserver and a dozen nodes behind if
# only its own pid is signalled.
# `gui` is read by a PythonExpression, so it takes False and not false.
setsid ros2 launch "$PACKAGE" "$LAUNCH" gui:=False "$@" >"$LOG" 2>&1 &
GROUP=$!
cleanup() {
  kill -INT -- "-$GROUP" 2>/dev/null || true
  # SIGINT is what shuts a launch down cleanly; the wait bounds how long that
  # politeness is worth before the group is taken down.
  for _ in $(seq 20); do kill -0 -- "-$GROUP" 2>/dev/null || return 0; sleep 0.5; done
  kill -KILL -- "-$GROUP" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for /robot_description from $PACKAGE $LAUNCH (log: $LOG)"
python3 "$HERE/dump_robot_description.py" "$OUTPUT" --timeout "$TIMEOUT"
cleanup
trap - EXIT

python3 "$HERE/export_timing_ocp.py" --description "$OUTPUT" --compile-only
