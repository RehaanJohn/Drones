import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from addc_mission.vision import discover, QRBackend

ROOT = Path(__file__).resolve().parents[1]


class VisionTests(unittest.TestCase):
    def test_discovery_uses_bottom_of_image_box(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(frame, (200, 100), (300, 200), (10, 65, 255), -1)
        u, v, _ = discover(frame)
        self.assertAlmostEqual(u, 250.5)
        self.assertAlmostEqual(v, 201)
        self.assertIsNone(discover(np.zeros_like(frame)))

    def test_real_qr_and_original_scanner_adapter_preserve_zero(self):
        texture = cv2.imread(str(ROOT/'sim/models/intelligence_cache/materials/textures/qr.png'))
        frame = np.full((1080, 1920, 3), 210, dtype=np.uint8)
        frame[350:750, 700:1100] = cv2.resize(texture, (400, 400), interpolation=cv2.INTER_NEAREST)
        scanner = ROOT.parent/'QR_Scanning'/'scanner.py'
        backend = QRBackend(str(scanner), '/nonexistent-addc-test-weights')
        results = backend.detect(frame)
        self.assertIn('07', [r[2] for r in results])

    def test_world_truth_is_not_a_ros_parameter(self):
        config = (ROOT/'ros2_ws/src/addc_mission/config/mission.yaml').read_text()
        self.assertNotIn('payload:', config)
        self.assertNotIn('cache_position:', config)
        world = ET.parse(ROOT/'sim/worlds/addc.sdf')
        self.assertEqual(world.find('.//world').attrib['name'], 'addc')
        model = ET.parse(ROOT/'sim/models/addc_quad/model.sdf')
        cameras = model.findall('.//sensor[@type="camera"]')
        self.assertEqual(len(cameras), 2)
        self.assertEqual({s.find('camera/optical_frame_id').text for s in cameras},
                         {'discovery_camera_optical', 'down_camera_optical'})
        self.assertEqual(len(model.findall('.//plugin')), 4)

    def test_saved_and_generated_vehicle_inertias_are_physical(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT/'tools/generate_vehicle.py'),
                            '--output', directory], check=True)
            for path in (ROOT/'sim/models/addc_quad/model.sdf',
                         Path(directory)/'models/addc_quad/model.sdf'):
                for link in ET.parse(path).findall('.//link'):
                    with self.subTest(model=str(path), link=link.get('name')):
                        inertial = link.find('inertial')
                        self.assertGreater(float(inertial.findtext('mass')), 0)
                        inertia = inertial.find('inertia')
                        values = {tag: float(inertia.findtext(tag, '0'))
                                  for tag in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
                        matrix = np.array([
                            [values['ixx'], values['ixy'], values['ixz']],
                            [values['ixy'], values['iyy'], values['iyz']],
                            [values['ixz'], values['iyz'], values['izz']]])
                        moments = np.linalg.eigvalsh(matrix)
                        self.assertTrue(np.all(np.isfinite(moments)))
                        self.assertGreater(moments[0], 0)
                        self.assertLessEqual(moments[2], moments[0] + moments[1])

    def test_saved_and_generated_barometer_has_pressure_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT/'tools/generate_vehicle.py'),
                            '--output', directory], check=True)
            for path in (ROOT/'sim/models/addc_quad/model.sdf',
                         Path(directory)/'models/addc_quad/model.sdf'):
                with self.subTest(model=str(path)):
                    sensor = ET.parse(path).find('.//sensor[@name="air_pressure_sensor"]')
                    noise = sensor.find('air_pressure/pressure/noise')
                    self.assertIsNotNone(noise)
                    self.assertEqual(noise.get('type'), 'gaussian')
                    self.assertEqual(float(noise.findtext('mean')), 0)
                    self.assertGreater(float(noise.findtext('stddev')), 0)
