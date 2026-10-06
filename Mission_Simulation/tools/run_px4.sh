#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash tools/run_px4.sh /path/to/PX4-Autopilot" >&2
  exit 2
fi
addc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
px4_root="$(cd -- "$1" && pwd)"
if ! ign topic -l | rg -q '^/world/addc/clock$'; then
  echo "Start the ADDC Fortress world first with tools/run_gazebo.sh" >&2
  exit 1
fi
if [[ ! -x "$px4_root/build/px4_sitl_default/bin/px4" ]]; then
  echo "Build PX4 first (see ROS2_README.md)" >&2
  exit 1
fi
export IGN_GAZEBO_RESOURCE_PATH="$addc_root/sim/models:${IGN_GAZEBO_RESOURCE_PATH:-}"
export PX4_SYS_AUTOSTART=4009
export PX4_SIM_MODEL=gz_addc_quad
unset PX4_GZ_MODEL PX4_GZ_MODEL_POSE
export PX4_GZ_MODEL_NAME=addc_quad_0
# Separate rootfs so prior SITL parameters cannot override this airframe's defaults.
mkdir -p "$addc_root/.runtime/px4"
cd "$addc_root/.runtime/px4"
exec "$px4_root/build/px4_sitl_default/bin/px4" "$px4_root/build/px4_sitl_default/etc"
