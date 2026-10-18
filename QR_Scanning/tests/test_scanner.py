import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner


class ScannerTests(unittest.TestCase):
    def test_main_thread_capture_reads_and_releases_on_calling_thread(self):
        owner = threading.get_ident()
        frame = np.full((24, 32, 3), 100, np.uint8)
        cap = Mock()
        def read():
            self.assertEqual(threading.get_ident(), owner)
            return True, frame
        cap.read.side_effect = read
        with scanner.MainThreadCapture(cap) as capture:
            self.assertIs(capture.read()[0], frame)
        cap.release.assert_called_once()

    def test_camera_test_bypasses_decoder_and_saves_unmodified_frame(self):
        frame = np.full((24, 32, 3), 123, np.uint8)
        cap = Mock()
        cap.read.return_value = (True, frame)
        preview = Mock()
        preview.show.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            args = scanner.make_parser().parse_args(['--camera-test', '--save-frames', directory])
            with patch.object(scanner, 'open_camera', return_value=cap), patch.object(scanner, 'configure_camera'), patch.object(scanner, 'Preview', return_value=preview), patch.object(scanner.time, 'monotonic', side_effect=[0, 3]), patch.object(scanner, 'QRScanner') as decoder, patch('builtins.print'):
                self.assertEqual(scanner.camera_test(args), 0)
            saved = cv2.imread(str(Path(directory) / 'raw_camera.png'))
            np.testing.assert_array_equal(saved, frame)
            decoder.assert_not_called()
        cap.release.assert_called_once()
        preview.close.assert_called_once()

    def test_frame_metrics_distinguish_black_from_visible_image(self):
        black = scanner.frame_metrics(np.zeros((20, 20, 3), np.uint8))
        visible = scanner.frame_metrics(np.full((20, 20, 3), 180, np.uint8))
        self.assertEqual(black['image_max'], 0)
        self.assertEqual(visible['image_mean'], 180)

    def test_mac_accepts_negotiated_resolution_and_uses_main_thread_capture(self):
        frame = np.full((360, 640, 3), 120, np.uint8)
        capture = Mock()
        capture.__enter__ = Mock(return_value=capture)
        capture.__exit__ = Mock(return_value=False)
        capture.read.return_value = (frame, 0, scanner.datetime.datetime.now().astimezone())
        preview = Mock()
        preview.show.side_effect = [True, False]
        detector = Mock(observations=[])
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner.sys, 'platform', 'darwin'), patch.object(scanner, 'QRScanner', return_value=detector), patch.object(scanner, 'open_camera'), patch.object(scanner, 'configure_camera'), patch.object(scanner, 'MainThreadCapture', return_value=capture) as main_capture, patch.object(scanner, 'LatestFrameCapture') as threaded, patch.object(scanner, 'Preview', return_value=preview), patch('builtins.print') as log:
            self.assertEqual(scanner.main(['--width', '640', '--height', '480', '--no-resolve-links', '--output', str(Path(directory) / 'scans.jsonl')]), 0)
            self.assertTrue(any('continuing at its actual resolution' in str(call) for call in log.call_args_list))
        main_capture.assert_called_once()
        threaded.assert_not_called()

    def test_records_preserve_payload_and_cooldown_per_code(self):
        output = io.StringIO()
        writer = scanner.ScanWriter(output)
        payload = 'hello\n世界 "QR"'
        with patch.object(scanner.time, 'monotonic', side_effect=[10, 10.1, 11, 12]), patch('builtins.print'):
            self.assertTrue(writer.record(payload))
            self.assertTrue(writer.record('second'))
            self.assertFalse(writer.record(payload))
            self.assertTrue(writer.record(payload))
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['data'] for r in records], [payload, 'second', payload])

    def test_real_wechat_detection_of_generated_qr(self):
        qr = cv2.QRCodeEncoder_create().encode('terminal-test-123')
        qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        frame = cv2.cvtColor(cv2.resize(qr, (400, 400), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        detector = scanner.QRScanner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.jsonl'
            with path.open('a', encoding='utf-8') as output, patch('builtins.print'):
                scanner.scan_frame(detector, frame, scanner.ScanWriter(output))
                self.assertEqual(json.loads(path.read_text())['data'], 'terminal-test-123')

    def test_fallback_uses_independent_candidates_when_wechat_returns_nothing(self):
        region = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float32)
        detector = Mock()
        detector.detectAndDecode.return_value = ([], [])
        engine = scanner.LegacyQRScanner(detector, use_zxing=False, lighting=False)
        engine.qr = Mock()
        engine.qr.detectMulti.return_value = (True, np.array([region, region]))
        with patch.object(scanner, 'fallback_decode', side_effect=['recovered-one', 'recovered-two']) as fallback:
            results = engine.decode(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(results, ['recovered-one', 'recovered-two'])

    def test_mac_backend_and_release_before_fallback(self):
        failed = Mock()
        failed.isOpened.return_value = False
        opened = Mock()
        opened.isOpened.return_value = True
        with patch.object(scanner.sys, 'platform', 'darwin'), patch.object(scanner.cv2, 'VideoCapture', side_effect=[failed, opened]) as capture:
            self.assertIs(scanner.open_camera(1), opened)
        self.assertEqual(capture.call_args_list[0].args, (1, cv2.CAP_AVFOUNDATION))
        failed.release.assert_called_once()

    def test_windows_backend_fallback_releases_failed_devices(self):
        failed_msmf, failed_dshow, opened = Mock(), Mock(), Mock()
        failed_msmf.isOpened.return_value = False
        failed_dshow.isOpened.return_value = False
        opened.isOpened.return_value = True
        with patch.object(scanner.sys, 'platform', 'win32'), patch.object(cv2, 'VideoCapture', side_effect=[failed_msmf, failed_dshow, opened]) as capture:
            self.assertIs(scanner.open_camera(0), opened)
        self.assertEqual([c.args for c in capture.call_args_list], [(0, cv2.CAP_MSMF), (0, cv2.CAP_DSHOW), (0,)])
        failed_msmf.release.assert_called_once()
        failed_dshow.release.assert_called_once()
        opened.release.assert_not_called()

    def test_explicit_backend_does_not_silently_switch(self):
        failed = Mock()
        failed.isOpened.return_value = False
        with patch.object(scanner.sys, 'platform', 'win32'), patch.object(cv2, 'VideoCapture', return_value=failed) as capture:
            with self.assertRaisesRegex(RuntimeError, '--list-cameras'):
                scanner.open_camera(1, 'msmf')
        capture.assert_called_once_with(1, cv2.CAP_MSMF)
        failed.release.assert_called_once()

    def test_windows_default_uses_discovered_device_not_fixed_zero(self):
        device = Mock(index=1, backend=cv2.CAP_DSHOW)
        cap = Mock()
        cap.isOpened.return_value = True
        with patch.object(scanner.sys, 'platform', 'win32'), patch.object(scanner, 'find_working_camera', return_value=device) as discover, patch.object(cv2, 'VideoCapture', return_value=cap) as capture:
            self.assertIs(scanner.open_camera(camera_name='Lenovo Performance RGB'), cap)
        discover.assert_called_once_with('auto', 'Lenovo Performance RGB', False)
        capture.assert_called_once_with(1, cv2.CAP_DSHOW)

    def test_interrupt_and_read_failure_release_camera(self):
        for interrupted in (True, False):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                cap = Mock()
                capture = Mock()
                capture.__enter__ = Mock(return_value=capture)
                capture.__exit__ = Mock(return_value=False)
                capture.read.side_effect = KeyboardInterrupt if interrupted else RuntimeError('no frames')
                path = Path(directory) / 'scans.jsonl'
                path.write_text('existing\n')
                with patch.object(scanner, 'QRScanner'), patch.object(scanner, 'configure_camera'), patch.object(scanner, 'open_camera', return_value=cap), patch.object(scanner, 'LatestFrameCapture', return_value=capture), patch.object(scanner, 'MainThreadCapture', return_value=capture), patch('builtins.print'):
                    code = scanner.main(['--output', str(path), '--no-preview', '--no-resolve-links'])
                self.assertEqual(code, 0 if interrupted else 1)
                self.assertEqual(path.read_text(), 'existing\n')
                capture.__exit__.assert_called_once()

    def test_capture_config_failure_releases_camera(self):
        cap = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner, 'QRScanner'), patch.object(scanner, 'open_camera', return_value=cap), patch.object(scanner, 'configure_camera', side_effect=RuntimeError('driver')), patch('builtins.print'):
            self.assertEqual(scanner.main(['--output', str(Path(directory) / 'scans.jsonl')]), 1)
        cap.release.assert_called_once()

    def test_tile_coverage_and_overlap_at_edges(self):
        regions = scanner.tile_regions((1080, 1920))
        coverage = np.zeros((1080, 1920), dtype=np.uint8)
        for x1, y1, x2, y2 in regions:
            coverage[y1:y2, x1:x2] += 1
        self.assertGreaterEqual(coverage.min(), 1)
        # The code and its quiet zone fit in a crop even across this seam.
        self.assertTrue(any(x1 <= 610 and x2 >= 700 and y1 <= 480 and y2 >= 570
                            for x1, y1, x2, y2 in regions))

    def test_tiles_advance_even_when_full_frame_decodes(self):
        detector = Mock()
        detector.detectAndDecode.return_value = (['easy'], [])
        engine = scanner.LegacyQRScanner(detector, use_zxing=False, tiles_per_frame=1)
        frame = np.zeros((720, 1280), np.uint8)
        for _ in range(8):
            self.assertEqual(engine.decode(frame), ['easy'])
        self.assertGreaterEqual(engine.cursor, 4)
        # Every second call is a native-resolution crop, without 0.5 downscaling.
        self.assertEqual(detector.setScaleFactor.call_args_list[1].args, (1.0,))

    def test_crop_handles_both_corner_shapes_and_clamps_padding(self):
        frame = np.zeros((60, 80, 3), np.uint8)
        points = np.array([[0, 0], [79, 0], [79, 59], [0, 59]], np.float32)
        self.assertEqual(scanner.crop_qr_region(frame, points).shape, frame.shape)
        self.assertEqual(scanner.crop_qr_region(frame, [points]).shape, frame.shape)
        self.assertIsNone(scanner.crop_qr_region(frame, [[np.nan, 1]]))

    def test_raw_image_is_not_mutated_by_lighting_variants(self):
        engine = scanner.QRScanner(Mock())
        frame = np.random.default_rng(2).integers(0, 256, (128, 160, 3), dtype=np.uint8)
        original = frame.copy()
        for mode in engine.LIGHTING_MODES:
            result = engine.correct_lighting(frame, mode)
            self.assertEqual(result.shape, frame.shape[:2])
            self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(frame, original)

    def test_missing_zxing_uses_wechat(self):
        detector = Mock()
        detector.detectAndDecode.return_value = (['opencv'], [np.float32([[10,10],[60,10],[60,60],[10,60]])])
        with patch.object(scanner, 'zxingcpp', None):
            engine = scanner.QRScanner(detector, tiles_per_frame=0)
            self.assertFalse(engine.use_zxing)
            self.assertEqual(engine.decode(np.zeros((128, 128), np.uint8)), ['opencv'])

    @unittest.skipIf(scanner.zxingcpp is None, 'ZXing not installed')
    def test_real_small_qr_on_1080p_frame_and_unicode_payload(self):
        payload = 'line1\n世界 "QR"'
        qr = cv2.QRCodeEncoder_create().encode(payload)
        qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, (110, 110), interpolation=cv2.INTER_AREA)
        frame = np.full((1080, 1920, 3), 180, np.uint8)
        frame[470:580, 600:710] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        engine = scanner.QRScanner()
        self.assertIn(payload, engine.decode(frame))

    def test_real_opencv_path_without_zxing(self):
        qr = cv2.QRCodeEncoder_create().encode('opencv-only')
        qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, (200, 200), interpolation=cv2.INTER_NEAREST)
        self.assertIn('opencv-only', scanner.QRScanner(use_zxing=False).decode(qr))

    def test_latest_frame_capture_drops_old_frames_and_releases(self):
        cap = Mock()
        capture = scanner.LatestFrameCapture(cap)
        produced = threading.Event()
        calls = 0

        def read():
            nonlocal calls
            calls += 1
            if calls <= 3:
                return True, np.full((8, 8), calls, np.uint8)
            produced.set()
            capture.stop.wait(2)
            return False, None

        cap.read.side_effect = read
        with capture:
            self.assertTrue(produced.wait(2))
            frame, age, captured_at = capture.read()
            self.assertTrue(np.all(frame == 3))
            self.assertGreaterEqual(age, 0)
            self.assertIsNotNone(captured_at.tzinfo)
        cap.release.assert_called_once()
        self.assertFalse(capture.thread.is_alive())

    def test_capture_read_exception_is_propagated_and_released(self):
        cap = Mock()
        cap.read.side_effect = OSError('disconnected')
        with scanner.LatestFrameCapture(cap) as capture:
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                capture.read()
        cap.release.assert_called_once()

    def test_thirty_failed_camera_reads_are_reported(self):
        cap = Mock()
        cap.read.return_value = (False, None)
        capture = scanner.LatestFrameCapture(cap)
        capture.stop = Mock()
        capture.stop.is_set.return_value = False
        capture._capture()
        with self.assertRaisesRegex(RuntimeError, '30 attempts'):
            capture.read()
        self.assertEqual(cap.read.call_count, 30)
        cap.release.assert_called_once()

    def test_real_shadowed_small_qr_recovers(self):
        from benchmark import synthetic_cases
        _, frame, expected, _ = next(case for case in synthetic_cases() if case[0] == '41px_shadow')
        engine = scanner.QRScanner()
        results = set()
        for _ in range(4):
            results.update(engine.decode(frame))
        self.assertEqual(results, set(expected))

    @unittest.skipIf(scanner.zxingcpp is None, 'ZXing not installed')
    def test_multiple_real_codes_are_preserved(self):
        frame = np.full((720, 1280, 3), 190, np.uint8)
        for payload, x in [('left-code', 100), ('right-code', 900)]:
            qr = cv2.QRCodeEncoder_create().encode(payload)
            qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
            qr = cv2.resize(qr, (170, 170), interpolation=cv2.INTER_NEAREST)
            frame[270:440, x:x + 170] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        self.assertEqual(set(scanner.QRScanner().decode(frame)), {'left-code', 'right-code'})

    def test_resolution_is_requested_and_exposure_left_unchanged_by_default(self):
        cap = Mock()
        with patch('builtins.print'):
            scanner.configure_camera(cap, scanner.make_parser().parse_args([]))
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        properties = [call.args[0] for call in cap.set.call_args_list]
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, properties)
        self.assertNotIn(cv2.CAP_PROP_AUTO_EXPOSURE, properties)

    def test_readable_logs_keep_capture_time_and_original_payload(self):
        stamp = scanner.datetime.datetime.fromisoformat('2026-10-11T14:23:05.123+05:30')
        output = io.StringIO()
        writer = scanner.ScanWriter(output)
        with patch('builtins.print') as log:
            writer.record('hello\nworld', stamp)
        log.assert_called_once_with('[14:23:05] QR: hello\\nworld', flush=True)
        record = json.loads(output.getvalue())
        self.assertEqual(record['data'], 'hello\nworld')
        self.assertEqual(record['time'], '2026-10-11T14:23:05.123+05:30')

    def test_url_and_content_are_logged_as_distinct_events(self):
        stamp = scanner.datetime.datetime.now().astimezone()
        url = 'https://qr-codes.io/zuabFE'
        resolver = Mock()
        resolver.submit.return_value = []
        resolver.poll.return_value = [(url, stamp, {'url': url, 'content': '69'})]
        output = io.StringIO()
        writer = scanner.ScanWriter(output, resolver=resolver)
        with patch('builtins.print') as log:
            writer.record(url, stamp)
            writer.poll_links()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records[0]['data'], url)
        self.assertEqual(records[1]['type'], 'link_content')
        self.assertEqual(records[1]['link']['content'], '69')
        self.assertEqual(records[0]['time'], records[1]['time'])
        self.assertIn('69', log.call_args.args[0])

    def test_non_url_payload_is_not_fetched(self):
        resolver = Mock()
        writer = scanner.ScanWriter(io.StringIO(), resolver=resolver)
        with patch('builtins.print'):
            writer.record('69')
            writer.record('file:///private.txt')
        resolver.submit.assert_not_called()

    def test_camera_configuration_is_quiet_by_default(self):
        with patch('builtins.print') as log:
            scanner.configure_camera(Mock(), scanner.make_parser().parse_args([]))
        log.assert_not_called()

    def test_preview_passes_original_frame_and_quits_on_q(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), patch.object(cv2, 'imshow') as show, patch.object(cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyWindow') as destroy:
            preview = scanner.Preview()
            self.assertFalse(preview.show(frame))
            self.assertIs(show.call_args.args[1], frame)
            preview.close()
        destroy.assert_called_once()

    def test_preview_exit_releases_camera_and_resolver(self):
        stamp = scanner.datetime.datetime.now().astimezone()
        capture = Mock()
        capture.__enter__ = Mock(return_value=capture)
        capture.__exit__ = Mock(return_value=False)
        capture.read.return_value = (np.zeros((128, 128, 3), np.uint8), 0, stamp)
        preview = Mock()
        preview.show.return_value = False
        resolver = Mock()
        resolver.poll.return_value = []
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner, 'QRScanner'), patch.object(scanner, 'open_camera'), patch.object(scanner, 'configure_camera'), patch.object(scanner, 'LatestFrameCapture', return_value=capture), patch.object(scanner, 'MainThreadCapture', return_value=capture), patch.object(scanner, 'Preview', return_value=preview), patch.object(scanner, 'LinkResolver', return_value=resolver):
            self.assertEqual(scanner.main(['--output', str(Path(directory) / 'scans.jsonl')]), 0)
        capture.__exit__.assert_called_once()
        preview.close.assert_called_once()
        resolver.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
