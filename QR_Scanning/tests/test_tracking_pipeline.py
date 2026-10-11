import datetime
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import linux_camera
import pipeline
import qualification
import scanner
import tracking


CONTROLS = '''
 auto_exposure 0x009a0901 (menu) : min=0 max=3 default=3 value=3
        1: Manual Mode
        3: Aperture Priority Mode
 exposure_time_absolute 0x009a0902 (int) : min=1 max=10000 step=1 default=156 value=100
 focus_absolute 0x009a090a (int) : min=0 max=255 step=5 default=0 value=20 flags=inactive
'''
FORMATS = '''
 [0]: 'MJPG' (Motion-JPEG)
    Size: Discrete 1920x1080
        Interval: Discrete 0.033s (30.000 fps)
 [1]: 'YUYV' (YUYV 4:2:2)
    Size: Discrete 1920x1080
        Interval: Discrete 0.200s (5.000 fps)
    Size: Discrete 640x480
        Interval: Discrete 0.033s (30.000 fps)
'''


def qr_frame(payload='https://qr-codes.io/zuabFE', x=100):
    code = cv2.QRCodeEncoder_create().encode(payload)
    code = cv2.copyMakeBorder(code, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    code = cv2.resize(code, (180, 180), interpolation=cv2.INTER_NEAREST)
    frame = np.full((400, 640, 3), 160, np.uint8)
    frame[100:280, x:x + 180] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    return frame


class LinuxCameraTests(unittest.TestCase):
    def test_realistic_modes_keep_resolution_and_format(self):
        profile = {'modes': linux_camera.parse_modes(FORMATS)}
        self.assertEqual(linux_camera.choose_mode(profile, 1920, 1080, 30)['fourcc'], 'MJPG')
        self.assertEqual(linux_camera.choose_mode(profile, 1920, 1080, 30, 'YUYV')['fps'], 5)
        with self.assertRaisesRegex(RuntimeError, 'silently'):
            linux_camera.choose_mode(profile, 1280, 720, 30)

    def test_linux_control_aliases_units_and_readback(self):
        controls = linux_camera.parse_controls(CONTROLS)
        self.assertEqual(controls['exposure_auto']['menu'][1], 'Manual Mode')
        with patch.object(linux_camera, 'ctl', side_effect=['', CONTROLS]) as ctl:
            result = linux_camera.set_control('/dev/video2', 'exposure_absolute', 100, controls)
        self.assertTrue(result['validated'])
        self.assertIn('exposure_time_absolute=100', ctl.call_args_list[0].args[1])

    def test_unsupported_inactive_and_invalid_controls_never_written(self):
        controls = linux_camera.parse_controls(CONTROLS)
        with patch.object(linux_camera, 'ctl') as ctl:
            self.assertFalse(linux_camera.set_control('camera', 'gain', 1, controls)['supported'])
            self.assertFalse(linux_camera.set_control('camera', 'focus_absolute', 20, controls)['supported'])
            with self.assertRaises(RuntimeError):
                linux_camera.set_control('camera', 'exposure_absolute', 10001, controls)
        ctl.assert_not_called()

    def test_failed_manual_mode_does_not_set_exposure(self):
        profile = {'device': '/dev/video1', 'controls': linux_camera.parse_controls(CONTROLS),
                   'modes': linux_camera.parse_modes(FORMATS)}
        options = dict(width=1920, height=1080, fps=30, exposure_us=1000)
        with patch.object(linux_camera, 'set_control', return_value={'validated': False}) as setter:
            linux_camera.configure(Mock(get=Mock(return_value=30)), profile, options)
        self.assertEqual(setter.call_count, 1)

    def test_exposure_feedback_deadband_ceiling_and_steps(self):
        control = {'min': 1, 'max': 10000, 'step': 1}
        self.assertEqual(linux_camera.exposure_step(control, 10, 120), 10)
        self.assertLess(linux_camera.exposure_step(control, 10, 220), 10)
        self.assertGreater(linux_camera.exposure_step(control, 10, 40), 10)
        self.assertEqual(linux_camera.exposure_step(control, 20, 40), 20)

    def test_camera_profiles_are_keyed_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = linux_camera.save_profile(directory, {'key': 'usb123', 'actual_size': [1920, 1080]})
            self.assertEqual(path.name, 'usb123.json')
            self.assertEqual(json.loads(path.read_text())['actual_size'], [1920, 1080])
            self.assertFalse(path.with_suffix('.tmp').exists())


class TrackingTests(unittest.TestCase):
    def test_duplicate_frame_cannot_confirm_but_second_capture_does(self):
        engine = scanner.QRScanner(tiles_per_frame=0, budget_ms=300)
        frame = qr_frame()
        first = engine.observe(frame, 1, 10)
        self.assertTrue(any(o['payload'] for o in first))
        self.assertFalse(any(o['confirmed'] for o in first))
        duplicate = engine.observe(frame, 1, 10)
        self.assertFalse(any(o['confirmed'] for o in duplicate))
        second = engine.observe(frame, 2, 10.1)
        self.assertTrue(any(o['confirmed'] and o['payload'] == 'https://qr-codes.io/zuabFE' for o in second))

    def test_tracking_handles_translation_without_mutating_input(self):
        engine = scanner.QRScanner(tiles_per_frame=0, budget_ms=300)
        first, second = qr_frame(), qr_frame(x=110)
        saved = second.copy()
        engine.observe(first, 1, 10)
        result = engine.observe(second, 2, 10.1)
        self.assertTrue(any(o['confirmed'] for o in result))
        np.testing.assert_array_equal(second, saved)

    def test_expired_track_does_not_confirm_new_encounter(self):
        engine = scanner.QRScanner(tiles_per_frame=0, budget_ms=300)
        engine.observe(qr_frame(), 1, 10)
        observations = engine.observe(qr_frame(), 2, 13)
        self.assertFalse(any(o['confirmed'] for o in observations))

    def test_historical_detection_does_not_rewind_geometry(self):
        engine = scanner.QRScanner(tiles_per_frame=0, budget_ms=300)
        engine.observe(qr_frame(), 2, 10.1)
        geometry = engine.tracker.tracks[0].corners.copy()
        engine.observe(qr_frame(x=95), 1, 10)
        np.testing.assert_array_equal(engine.tracker.tracks[0].corners, geometry)
        self.assertEqual(engine.tracker.tracks[0].seen, 10.1)

    def test_independent_candidates_are_tracked_without_decoding(self):
        detector = Mock()
        detector.detectAndDecode.return_value = ([], [])
        engine = scanner.QRScanner(detector, use_zxing=False, lighting=False, budget_ms=300)
        points = np.float32([[[10, 10], [40, 10], [40, 40], [10, 40]],
                            [[70, 70], [100, 70], [100, 100], [70, 100]]])
        engine.qr = Mock()
        engine.qr.detectMulti.return_value = (True, points)
        engine.qr.detectAndDecode.return_value = ('', None, None)
        result = engine.observe(np.zeros((128, 128, 3), np.uint8), 1, 10)
        self.assertEqual(len(engine.tracker.tracks), 2)
        self.assertEqual(len({o['track_id'] for o in result}), 2)

    def test_variants_bounded_and_rectification_rejects_degenerate_corners(self):
        image = qr_frame()
        self.assertLessEqual(len(tracking.variants(image, tracking.quality(image))), 2)
        self.assertIsNone(tracking.rectified(image, np.zeros((4, 2), np.float32)))


class WorkerTests(unittest.TestCase):
    def test_decoder_worker_confirms_distinct_frames_and_stops(self):
        history = pipeline.FrameHistory()
        detector = scanner.QRScanner(tiles_per_frame=0, budget_ms=300)
        stamp = datetime.datetime.now().astimezone()
        with pipeline.DecoderWorker(detector, interval=0) as worker:
            history.add(qr_frame(), stamp)
            worker.submit(history.snapshot())
            first, _ = worker.results.get(timeout=3)
            self.assertFalse(any(o['confirmed'] for o in first))
            history.add(qr_frame(x=105), stamp)
            worker.submit(history.snapshot())
            second, _ = worker.results.get(timeout=3)
            self.assertTrue(any(o['confirmed'] for o in second))
        self.assertIsNone(worker.error)
        self.assertFalse(worker.thread.is_alive())

    def test_history_enforces_age_and_byte_cap(self):
        history = pipeline.FrameHistory(byte_limit=200)
        image = np.zeros((10, 10), np.uint8)
        for index in range(3):
            history.add(image, None, index * .1)
        self.assertEqual(history.bytes, 200)
        self.assertEqual(len(history.snapshot()), 2)
        history.add(image, None, 1)
        self.assertEqual(len(history.snapshot()), 1)

    def test_queue_replaces_stale_work(self):
        channel = queue.Queue(1)
        pipeline.replace_latest(channel, 1)
        pipeline.replace_latest(channel, 2)
        self.assertEqual(channel.get_nowait(), 2)

    def test_process_camera_timeout_has_bounded_restarts(self):
        capture = pipeline.ProcessCapture({}, retries=2)
        channel = Mock()
        channel.get.side_effect = queue.Empty
        capture.channel = channel
        with patch.object(capture, '_close'), patch.object(capture, '_start'), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'after 2 restarts'):
                capture.read()
        self.assertEqual(capture.restart_count, 2)

    def test_output_emits_only_confirmed_payloads_and_drains(self):
        stream = io.StringIO()
        stamp = datetime.datetime.now().astimezone()
        observations = [{'payload': 'provisional', 'confirmed': False, 'captured_at': stamp},
                        {'payload': 'confirmed', 'confirmed': True, 'captured_at': stamp}]
        with patch('builtins.print'), pipeline.OutputWorker(scanner.ScanWriter(stream)) as output:
            output.submit(observations)
        self.assertEqual(json.loads(stream.getvalue())['data'], 'confirmed')


class QualificationTests(unittest.TestCase):
    def test_optics_distinguishes_resolution_limit_from_decode_claim(self):
        report = qualification.optical_report(1920, 29, 297, 10, hfov=70)
        self.assertLess(report['native_pixels_per_module'], 1.2)
        self.assertFalse(report['sampling_gate_pass'])
        self.assertLess(report['four_pixel_distance_estimate_m'], 3)

    def test_qualification_confidence_requires_enough_trials(self):
        self.assertLess(qualification.binomial_lower(299, 300), .99)
        self.assertGreater(qualification.binomial_lower(300, 300), .99)
        row = {'condition': 'sun', 'expected': ['url'], 'success': True, 'incorrect_payloads': [],
               'distance_m': 10, 'stable_hover': True, 'obliquity_deg': 0}
        self.assertFalse(qualification.summarize([row])['sun']['qualification_pass'])
        self.assertTrue(qualification.summarize([row] * 300)['sun']['qualification_pass'])
        self.assertFalse(qualification.summarize([dict(row, distance_m=3)] * 300)['sun']['qualification_pass'])

    def test_video_replay_confirms_real_url_without_truth_in_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 400))
            if not writer.isOpened():
                self.skipTest('MJPEG video writer unavailable')
            for _ in range(20):
                writer.write(qr_frame())
            writer.release()
            trial = {'id': 'real-url', 'condition': 'sun', 'start_s': 0, 'end_s': 1.9,
                     'expected': ['https://qr-codes.io/zuabFE']}
            result = qualification.replay_trial(path, trial)
            self.assertTrue(result['success'], result)
            self.assertGreaterEqual(result['processed_frames'], 2)


if __name__ == '__main__':
    unittest.main()
