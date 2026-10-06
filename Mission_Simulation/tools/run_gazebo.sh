#!/usr/bin/env bash
set -euo pipefail
addc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scenario="${1:-$addc_root/sim}"
export IGN_GAZEBO_RESOURCE_PATH="$scenario/models:$addc_root/sim/models:${IGN_GAZEBO_RESOURCE_PATH:-}"
exec ign gazebo -r "$scenario/worlds/addc.sdf"
