# ADDC two-camera prototype — Ubuntu 22.04 / Humble / Fortress

This ROS 2 implementation lives in `ros2_ws`; the original ROS 1 code is retained.
It discovers an orange test cache with an oblique RGB camera, estimates its location
from capture-time MAVROS pose, approaches without completing a field sweep, hands
off to a downward RGB camera, confirms two QR digits, publishes them, then returns
to its recorded home position and requests autonomous landing.

**Implementation status:** offline geometry, mission, image decoding, and generated
asset checks are tested. PX4 compilation, ROS message transport, Gazebo rendering,
and a complete SITL flight must still be validated on Ubuntu. This repository was
implemented on macOS without ROS or a running Linux container.

## Version choice and Fortress adaptation

Use Ubuntu **22.04**, ROS **Humble**, Gazebo **Fortress / ignition-gazebo6**, and
PX4 **v1.14.4**. Do not use Gazebo Classic plugins or launch commands in this path.

Stock PX4 v1.14.4 expects Garden (`gz-transport12`, `gz sim`), not Fortress. The
included `tools/patch_px4_fortress.py` adapts its bridge to Fortress's transport11,
msgs8, math6, and `ign gazebo` CLI. It checks exact upstream SHA-256 hashes before
changing files, preserves originals, refuses unknown local modifications, and can
be reapplied. This is a project compatibility port, not an upstream-supported
PX4/Fortress pairing. Its compilation and live behaviour remain an Ubuntu test gate.

References:
- [PX4 v1.14.4 bridge dependency source](https://github.com/PX4/PX4-Autopilot/blob/v1.14.4/src/modules/simulation/gz_bridge/CMakeLists.txt)
- [Fortress installation](https://gazebosim.org/docs/fortress/install_ubuntu/)
- [ROS 2 Humble installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [Fortress ROS integration](https://gazebosim.org/docs/fortress/ros2_integration/)
- [MAVROS ROS 2 installation](https://github.com/mavlink/mavros/blob/ros2/docs/installation.md)

## Prepare Ubuntu

Install ROS Humble and Fortress using the official instructions above. Their apt
repositories must be configured first. Then install the application dependencies:

```bash
sudo apt update
sudo apt install ros-humble-mavros ros-humble-mavros-extras ros-humble-ros-gz \
  ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-launch-xml \
  python3-colcon-common-extensions python3-numpy python3-opencv python3-yaml \
  python3-venv python3-pip build-essential cmake ninja-build git ripgrep \
  libignition-transport11-dev libignition-msgs8-dev libignition-math6-dev
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
source /opt/ros/humble/setup.bash
```

Some MAVROS distributions put the dataset installer under
`/opt/ros/humble/share/mavros`; locate the installed script if the above path differs.

Clone a dedicated PX4 checkout; do not apply this port to your hardware firmware checkout:

```bash
git clone --branch v1.14.4 --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-ADDC-Fortress
```

From this project's `Drones/KamiKaze_Drone` directory:

```bash
bash tools/prepare_px4.sh ~/PX4-ADDC-Fortress
mkdir -p .runtime
python3 -m venv .runtime/px4-build-venv
source .runtime/px4-build-venv/bin/activate
printf 'numpy<2\nempy==3.3.4\n' > .runtime/px4-constraints.txt
python -m pip install -c .runtime/px4-constraints.txt -r ~/PX4-ADDC-Fortress/Tools/setup/requirements.txt
make -C ~/PX4-ADDC-Fortress px4_sitl_default
deactivate
```

Do not run PX4's default simulator installer: that installs a different Gazebo
generation. If compilation needs another build dependency, use the pinned PX4
release's build documentation while retaining Fortress.

## Build the ROS workspace

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
python3 tools/preflight.py
python3 tools/check_scenario.py
```

An additional image-driven closed-loop smoke check uses synthetic camera rendering
and an idealised motion integrator, with no target coordinates supplied to the
mission. It is useful offline but does not replace the Ubuntu SITL gate:

```bash
python3 tools/offline_smoke.py
```

The package installs a copy of the existing `../QR_Scanning/scanner.py` as a decoding
library. Its original terminal program is unchanged. By default the adapter uses
OpenCV QRCodeDetector and reuses the existing scanner's cropped fallbacks; it uses
WeChat when compatible OpenCV-contrib and all four weights are present. Mission
startup never downloads models. Provision weights once if using WeChat:

```bash
python3 tools/provision_qr_models.py --directory /tmp/addc-qr-models
```

For the existing scanner's OpenCV-contrib backend, use a companion virtualenv
with system ROS packages visible and build from that interpreter:

```bash
python3 -m venv --system-site-packages .runtime/companion-venv
source .runtime/companion-venv/bin/activate
python -m pip install 'numpy==1.26.4' 'opencv-contrib-python-headless==4.10.0.84'
cd ros2_ws
python -m colcon build --symlink-install
source install/setup.bash
cd ..
```

Use the same interpreter environment when launching. Avoid OpenCV pip installations
into the system ROS Python environment.

## Run the simulation

Start these in three terminals from `Drones/KamiKaze_Drone`:

**1 — Fortress world**
```bash
bash tools/run_gazebo.sh
```

**2 — PX4 SITL, attached to the existing `addc_quad_0`**
```bash
bash tools/run_px4.sh ~/PX4-ADDC-Fortress
```

**3 — ROS bridge, MAVROS, perception, localisation, and mission**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch addc_mission mission.launch.py
```

Before starting, verify `/mavros/state` reports connected and disarmed, aircraft
pose is available, and both image streams and calibration messages arrive:

```bash
ros2 topic echo /mavros/state
ros2 topic hz /camera/discovery/image_raw
ros2 topic hz /camera/down/image_raw
ros2 topic echo /camera/down/camera_info --once
```

The `frame_id` values must be `discovery_camera_optical` and `down_camera_optical`.
These refer to ROS optical axes, not Gazebo's forward-X camera axes. If camera
messages have another frame ID, fix the sensor / bridge configuration rather than
silently accepting the incorrect frame. All flight and perception nodes use `/clock`.
Check that MAVROS pose timestamps follow the same simulation clock.

Start only after the official start event:

```bash
ros2 service call /addc/start std_srvs/srv/Trigger '{}'
```

Observe:
```bash
ros2 topic echo /addc/status
ros2 topic echo /addc/result --qos-durability transient_local
```

Expected default outcome: digits `07`, then home return and landing. Result JSON and
QR evidence images are saved in `/tmp/addc-evidence`. The `/addc/result` publisher
retains the result for a reconnecting subscriber; its confirmation is independent
of the original terminal scanner's two-second logging cooldown.

To abort the prototype:
```bash
ros2 service call /addc/abort std_srvs/srv/Trigger '{}'
```

The controller never automatically re-enters OFFBOARD or re-arms after an external
mode change, disconnection, or abort. Start a new mission process for a new attempt.
Return uses a bounded local path and `AUTO.LAND`, rather than an unconfigured RTL
climb. PX4's offboard-loss landing fallback is configured in the simulation airframe.

## Configuration and new placements

`ros2_ws/src/addc_mission/config/mission.yaml` is the source for camera mounting,
resolution, and flight limits. If camera parameters change, regenerate the vehicle
and restart Gazebo; the launch publishes matching static optical transforms:

```bash
python3 tools/generate_vehicle.py --output sim
python3 tools/check_scenario.py
```

Generate an alternate world without putting its truth into mission parameters:

```bash
python3 tools/generate_scenario.py --output .runtime/scenario-2 \
  --cache 35 -10 1.2 --payload 42 --qr-size 0.25 --seed 2 --distractors 4
bash tools/run_gazebo.sh .runtime/scenario-2
```

Stop all three terminals before each new scenario. The runner adds both alternate
and base model directories to Fortress's resource path. `scenario_truth.json` is
for evaluation only; no mission node reads it. The A3 paper is 420 × 297 mm; the QR
square is independently configurable. Cache top height defaults to 0.6 m.

`ground_z` is in MAVROS's local frame, not Gazebo world coordinates. Its default
is -0.24 m because the local origin is established at the resting vehicle above
the ground. Verify this offset after estimator startup; adjust it if spawn height
or estimator origin changes. Capture-time pose and camera calibration must share
the simulation clock.

The default field boundary is x [-3, 50], y [-18, 18] m, height <= 12 m. Viewpoints
are a fallback grid, not a demonstrated coverage guarantee. Set spacing using
measured detection range. The 180-second drone search budget reserves team mission
time but does not simulate the operative or establish whole-team success.

## Checks and staged acceptance

Offline tests require only Python, NumPy, OpenCV, and PyYAML:

```bash
PYTHONPATH=ros2_ws/src/addc_mission python3 -m unittest discover -s tests -v
python3 tools/check_scenario.py
```

Ubuntu acceptance gates, in order:
1. Compile the Fortress-adapted PX4 bridge; build ROS interfaces and nodes.
2. Verify images, optical frames, MAVROS timestamps, motors, takeoff and hold.
3. Default scenario: initial-view discovery -> approach -> overhead acquisition ->
   correct digits -> autonomous home landing.
4. Repeat cache locations/orientations; record time to digits and handoff failures.
5. Test absent cache, distractors, dropped images, rejected mode/arming requests,
   stale pose, external takeover, and decode timeout. Inspect status and PX4 logs.
6. Replay images on Pi 4; measure inference latency and CPU load before hardware use.

The orange-cache detector is deliberately an integration fixture. A representative
detector, field appearance, real camera blur/exposure, terrain variation, navigation
error, and actual communication link performance remain later work. The controller
assumes a flat field, known cache-top height, and a MAVROS local frame aligned to
the field at launch. Ground-plane localisation uncertainty is a heuristic, not
a calibrated covariance. Simulator truth goes only to PX4's simulated sensors and
evaluation, never directly to cache perception or mission planning.
