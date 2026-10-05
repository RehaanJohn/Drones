import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner


class ScannerTests(unittest.TestCase):
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
        detector = scanner.create_detector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.jsonl'
            with path.open('a', encoding='utf-8') as output, patch('builtins.print'):
                scanner.scan_frame(detector, frame, scanner.ScanWriter(output))
                self.assertEqual(json.loads(path.read_text())['data'], 'terminal-test-123')

    def test_fallback_runs_for_each_undecoded_region(self):
        region = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.float32)
        detector = Mock()
        detector.detectAndDecode.return_value = (['decoded', '', ''], [region, region, region])
        writer = Mock()
        with patch.object(scanner, 'fallback_decode', side_effect=['recovered-one', 'recovered-two']) as fallback:
            scanner.scan_frame(detector, np.zeros((20, 20, 3), dtype=np.uint8), writer)
        self.assertEqual(fallback.call_count, 2)
        writer.record.assert_any_call('recovered-one')
        writer.record.assert_any_call('recovered-two')

    def test_mac_backend_and_release_before_fallback(self):
        failed = Mock()
        failed.isOpened.return_value = False
        opened = Mock()
        opened.isOpened.return_value = True
        with patch.object(scanner.sys, 'platform', 'darwin'), patch.object(scanner.cv2, 'VideoCapture', side_effect=[failed, opened]) as capture:
            self.assertIs(scanner.open_camera(1), opened)
        self.assertEqual(capture.call_args_list[0].args, (1, cv2.CAP_AVFOUNDATION))
        failed.release.assert_called_once()

    def test_interrupt_and_read_failure_release_camera(self):
        for interrupted in (True, False):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                cap = Mock()
                if interrupted:
                    cap.read.side_effect = KeyboardInterrupt
                else:
                    cap.read.return_value = (False, None)
                path = Path(directory) / 'scans.jsonl'
                path.write_text('existing\n')
                with patch.object(scanner, 'create_detector'), patch.object(scanner, 'open_camera', return_value=cap), patch.object(scanner.time, 'sleep'), patch('builtins.print'):
                    code = scanner.main(['--output', str(path)])
                self.assertEqual(code, 0 if interrupted else 1)
                self.assertEqual(path.read_text(), 'existing\n')
                cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
