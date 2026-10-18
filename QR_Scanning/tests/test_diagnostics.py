import contextlib
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
import diagnostics
import pipeline
import scanner


def observation(frame_id, stamp, payload='target', confirmed=True):
    return dict(frame_id=frame_id, capture_monotonic=stamp, track_id=1,
                payload=payload, confirmed=confirmed, method='crop',
                corners=[[20,20],[78,20],[78,78],[20,78]],
                quality=dict(module_px_estimate=2, contrast=180, sharpness=80),
                captured_at=None)


class DiagnosticsTests(unittest.TestCase):
    def test_simulation_identity_and_stress_preserve_source(self):
        frame = np.full((100, 160, 3), 150, np.uint8)
        self.assertIs(diagnostics.simulate(frame, diagnostics.DEFAULT_CONTROLS), frame)
        before = frame.copy()
        dark = diagnostics.simulate(frame, dict(brightness=50, blur=0, shadow=0, glare=0))
        self.assertAlmostEqual(float(dark.mean()), 75)
        stressed = diagnostics.simulate(frame, dict(brightness=100, blur=2, shadow=60, glare=70))
        self.assertEqual(stressed.shape, frame.shape)
        self.assertGreater(float(stressed.std()), 0)
        np.testing.assert_array_equal(frame, before)

    def test_trials_count_failures_wrong_payloads_and_capture_timestamps(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory, expected='target', distance=3, seconds=2)
            frame = np.full((100,160,3), 150, np.uint8)
            session.prepare(frame, 10, diagnostics.DEFAULT_CONTROLS)
            for i, payload in enumerate(['target', '', 'wrong']):
                stamp = 10.5 + i*.4
                obs = [] if not payload else [observation(i, stamp, payload)]
                session.consume(obs, {'processed_frames': [dict(frame_id=i, capture_monotonic=stamp,
                                                                latency_ms=100, decode_ms=30)]})
            # Late result captured before trial start must not count as success.
            session.consume([observation(9, 9)], {'processed_frames':[dict(frame_id=9, capture_monotonic=9,
                                                                         latency_ms=100, decode_ms=30)]})
            session.prepare(frame, 15.1, diagnostics.DEFAULT_CONTROLS)
            result = session.last_trial
            self.assertEqual(result['processed_frames'], 3)
            self.assertAlmostEqual(result['decoded_rate'], 1/3)
            self.assertEqual(result['incorrect_payloads'], ['wrong'])
            self.assertEqual(result['minimum_module_px'], 2)
            self.assertEqual(session.farthest, 3)
            self.assertAlmostEqual(result['first_confirmation_s'], .6)
            self.assertEqual(len((Path(directory)/'samples.jsonl').read_text().splitlines()), 3)
            session.close()
            self.assertTrue((Path(directory)/'trials.csv').exists())

    def test_slow_confirmation_does_not_pass_two_second_requirement(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory, expected='target')
            session.start_trial(10)
            session.consume([observation(1,10.1)], {'processed_frames': [dict(
                frame_id=1, capture_monotonic=10.1, result_monotonic=13.2,
                latency_ms=3100, decode_ms=3000)]})
            session.finish('complete')
            self.assertFalse(session.last_trial['confirmation_within_2s'])
            self.assertAlmostEqual(session.last_trial['first_confirmation_s'],3.2)
            session.close()

    def test_unknown_truth_and_empty_trial_do_not_claim_accuracy(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory)
            session.start_trial(10)
            session.finish('complete')
            self.assertFalse(session.last_trial['ground_truth_known'])
            self.assertIsNone(session.last_trial['decoded_rate'])
            self.assertFalse(session.last_trial['confirmation_within_2s'])
            session.close()

    def test_control_changes_invalidate_active_trial(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory)
            frame = np.full((100,160,3), 150, np.uint8)
            session.prepare(frame, 10, diagnostics.DEFAULT_CONTROLS)
            session.prepare(frame, 11, dict(brightness=50, blur=0, shadow=0, glare=0))
            self.assertIsNone(session.trial)
            self.assertEqual(session.last_trial['status'], 'controls_changed')
            session.close()

    def test_overlay_expires_stale_boxes_and_preserves_frame(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory)
            frame = np.full((100,160,3), 150, np.uint8)
            session.prepare(frame, 10, diagnostics.DEFAULT_CONTROLS)
            session.observations = [observation(1, 10)]
            with patch.object(diagnostics, 'overlay', wraps=pipeline.overlay) as render:
                display = session.render(frame, {}, 10.1)
                self.assertEqual(len(render.call_args.args[1]), 1)
                session.render(frame, {}, 11)
                self.assertEqual(render.call_args.args[1], [])
            self.assertEqual(display.shape, (820,620,3))
            np.testing.assert_array_equal(frame, np.full_like(frame,150))
            session.close()

    def test_report_groups_recovery_and_conditions_and_excludes_aborted(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            session = diagnostics.DiagnosticSession(directory, expected='target')
            session.start_trial(10)
            session.consume([observation(1,10.5)], {'processed_frames':[dict(frame_id=1,capture_monotonic=10.5,
                                                                          latency_ms=100,decode_ms=30)]})
            session.finish('complete')
            session.start_trial(20)
            session.finish('controls_changed')
            session.close()
            result = diagnostics.report(directory)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['trials'],1)
            self.assertEqual(result[0]['correct_decode_rate'],1)
            self.assertLess(result[0]['within_2s_lower_95'], .99)

    def test_competition_diagnostics_stays_headless_without_experiments(self):
        capture = Mock()
        capture.__enter__ = Mock(return_value=capture)
        capture.__exit__ = Mock(return_value=False)
        capture.read.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory, patch.object(scanner, 'QRScanner'), patch.object(scanner, 'open_camera'), patch.object(scanner, 'configure_camera'), patch.object(scanner, 'MainThreadCapture', return_value=capture), patch.object(scanner.sys, 'platform', 'darwin'), patch.object(scanner, 'Preview') as preview, patch.object(diagnostics, 'DiagnosticSession') as experiment:
            self.assertEqual(scanner.main(['--camera','1','--competition','--diagnostics','--no-resolve-links',
                                           '--output',str(Path(directory)/'scans.jsonl')]),0)
        preview.assert_not_called()
        experiment.assert_not_called()


if __name__ == '__main__':
    unittest.main()
