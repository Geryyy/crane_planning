#!/usr/bin/env bash
# codegen -> libvpsto_crane.so -> sanity checks. Needs ROS + workspace sourced,
# `colcon build --packages-up-to ros2_vpsto` done.
set -euo pipefail
cd "$(dirname "$0")"
python3 export_model.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j8
python3 check_model.py
