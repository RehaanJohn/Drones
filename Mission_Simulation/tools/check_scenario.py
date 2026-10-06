#!/usr/bin/env python3
"""Validate generated SDF structure, QR texture, and projection configuration offline."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import yaml
import math


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, default=root/'sim')
    parser.add_argument('--config', type=Path, default=root/'ros2_ws/src/addc_mission/config/mission.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())['/**']['ros__parameters']
    model_file = args.scenario/'models/addc_quad/model.sdf'
    if not model_file.exists():
        model_file = root/'sim/models/addc_quad/model.sdf'
    model = ET.parse(model_file)
    errors = []
    for camera in ('discovery', 'down'):
        sensor = model.find(f'.//sensor[@name="{camera}_camera"]')
        pose = list(map(float, sensor.find('pose').text.split()))
        expected_tilt = math.radians(config['discovery_tilt_deg']) if camera == 'discovery' else math.pi/2
        if any(abs(a-b) > 1e-8 for a, b in zip(pose[:3], config[f'{camera}_mount'])) or abs(pose[4]-expected_tilt) > 1e-8:
            errors.append(f'{camera}: model mounting differs from localisation configuration')
        if sensor.find('camera/optical_frame_id').text != f'{camera}_camera_optical':
            errors.append(f'{camera}: optical frame differs')
    texture = cv2.imread(str(args.scenario/'models/intelligence_cache/materials/textures/qr.png'))
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(texture)
    if not text or len(text) != 2:
        errors.append('QR texture does not decode to two digits')
    ET.parse(args.scenario/'worlds/addc.sdf')
    print('\n'.join(errors) if errors else f'Assets consistent; texture decodes to {text!r}. No Gazebo flight has been run.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
