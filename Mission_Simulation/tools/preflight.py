#!/usr/bin/env python3
"""Read-only Ubuntu runtime dependency check; does not arm or start flight."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    errors = []
    if sys.platform != 'linux':
        errors.append('Live SITL requires the Ubuntu 22.04 environment; current host is '+sys.platform)
    for command in ('ros2', 'ign', 'colcon'):
        if not shutil.which(command):
            errors.append(f'Missing command: {command}')
    for module in ('rclpy', 'cv_bridge', 'cv2', 'numpy', 'addc_interfaces', 'addc_mission'):
        if importlib.util.find_spec(module) is None:
            errors.append(f'Missing Python module: {module}; source ROS and the built workspace')
    if shutil.which('ign'):
        result = subprocess.run(['ign', 'gazebo', '--versions'], capture_output=True, text=True)
        if result.returncode or not any(line.strip().startswith('6.') for line in result.stdout.splitlines()):
            errors.append('Expected Fortress / ignition-gazebo6')
    if shutil.which('ros2'):
        result = subprocess.run(['ros2', 'pkg', 'prefix', 'mavros'], capture_output=True, text=True)
        if result.returncode:
            errors.append('MAVROS package missing')
    print('\n'.join(errors) if errors else 'Runtime dependencies found. Check telemetry and images before /addc/start.')
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
