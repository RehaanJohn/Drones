from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_devices
import cv2


class CameraDiscoveryTests(unittest.TestCase):
    def test_mac_listing_uses_avfoundation(self):
        device = SimpleNamespace(index=1, name='MacBook Air Camera', backend=cv2.CAP_AVFOUNDATION)
        enumerate_cameras = Mock(return_value=[device])
        module = SimpleNamespace(enumerate_cameras=enumerate_cameras)
        with patch.object(camera_devices.sys, 'platform', 'darwin'), patch.dict(sys.modules, {'cv2_enumerate_cameras': module}):
            self.assertEqual(camera_devices.list_cameras('avfoundation'), [device])
        enumerate_cameras.assert_called_once_with(cv2.CAP_AVFOUNDATION)

    def test_mac_default_prefers_builtin_over_continuity_camera(self):
        phone = SimpleNamespace(index=0, name='Phone Camera', backend=cv2.CAP_AVFOUNDATION)
        builtin = SimpleNamespace(index=1, name='MacBook Air Camera', backend=cv2.CAP_AVFOUNDATION)
        with patch.object(camera_devices.sys, 'platform', 'darwin'), patch.object(camera_devices, 'list_cameras', return_value=[phone, builtin]), patch.object(camera_devices, 'probe_camera', return_value=True) as probe:
            self.assertIs(camera_devices.find_working_camera('avfoundation'), builtin)
        probe.assert_called_once_with(1, cv2.CAP_AVFOUNDATION)

    def test_skips_non_delivering_device_and_ir_camera(self):
        cameras = [SimpleNamespace(index=2, name='Lenovo IR Camera', backend=cv2.CAP_MSMF),
                   SimpleNamespace(index=0, name='Lenovo RGB Camera', backend=cv2.CAP_MSMF),
                   SimpleNamespace(index=1, name='Lenovo RGB Camera', backend=cv2.CAP_MSMF)]
        with patch.object(camera_devices, 'list_cameras', return_value=cameras), patch.object(camera_devices, 'probe_camera', side_effect=[False, True]) as probe:
            self.assertIs(camera_devices.find_working_camera(), cameras[2])
        self.assertEqual([c.args for c in probe.call_args_list], [(0, cv2.CAP_MSMF), (1, cv2.CAP_MSMF)])

    def test_probe_timeout_does_not_hang_discovery(self):
        with patch.object(camera_devices.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 5)):
            self.assertFalse(camera_devices.probe_camera(0, cv2.CAP_DSHOW))

    def test_probe_requires_frame_not_just_successful_open(self):
        with patch.object(camera_devices.subprocess, 'run', return_value=Mock(returncode=0, stdout='')):
            self.assertFalse(camera_devices.probe_camera(0, cv2.CAP_DSHOW))

    def test_name_selection_does_not_fall_back_to_an_unrelated_camera(self):
        cameras = [SimpleNamespace(index=0, name='Other Camera', backend=cv2.CAP_MSMF)]
        with patch.object(camera_devices, 'list_cameras', return_value=cameras), patch.object(camera_devices, 'probe_camera') as probe:
            with self.assertRaisesRegex(RuntimeError, 'no matching'):
                camera_devices.find_working_camera(name='Lenovo')
        probe.assert_not_called()
