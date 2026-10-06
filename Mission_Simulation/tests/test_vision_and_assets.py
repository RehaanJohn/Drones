import importlib.util
from pathlib import Path
import tempfile
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
