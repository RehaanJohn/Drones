#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash tools/prepare_px4.sh /path/to/pristine/PX4-Autopilot-v1.14.4" >&2
  exit 2
fi
addc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
px4_root="$(cd -- "$1" && pwd)"
python3 "$addc_root/tools/patch_px4_fortress.py" "$px4_root"
destination="$px4_root/ROMFS/px4fmu_common/init.d-posix/airframes/4009_gz_addc_quad"
if [[ -e "$destination" ]] && ! cmp -s "$destination" "$addc_root/sim/airframes/4009_gz_addc_quad"; then
  echo "Refusing to overwrite a different existing airframe: $destination" >&2
  exit 1
fi
cp "$addc_root/sim/airframes/4009_gz_addc_quad" "$destination"
# ROMFS airframes are enumerated explicitly in this release.
python3 - "$px4_root/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
if '4009_gz_addc_quad' not in text:
    if '4001_gz_x500' not in text:
        raise SystemExit('Unsupported airframe list; 4001_gz_x500 missing')
    p.write_text(text.replace('4001_gz_x500', '4001_gz_x500\n\t4009_gz_addc_quad', 1))
PY
echo "Prepared PX4 v1.14.4 for ADDC Fortress. Build: make -C '$px4_root' px4_sitl_default"
