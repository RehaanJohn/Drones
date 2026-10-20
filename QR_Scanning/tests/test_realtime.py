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
import scanner
import pipeline
import realtime
import qualification
import link_resolver


def frame(payload='https://example.com/code', width=640, height=400):
    qr = cv2.QRCodeEncoder_create().encode(payload)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    image = np.full((height, width), 180, np.uint8)
    image[100:280, 120:300] = cv2.resize(qr, (180, 180), interpolation=cv2.INTER_NEAREST)
    return image


class FastAndRecoveryTests(unittest.TestCase):
    def test_fast_decode_never_calls_wechat_and_uses_actual_payload(self):
        engine = realtime.FastScanner({})
        with patch.object(engine.engine.detector, 'detectAndDecode', side_effect=AssertionError('Heavy path in fast worker')):
            observations = engine.observe(frame(), 1, 1)
        self.assertTrue(any(o['payload'] == 'https://example.com/code' and o['confirmed'] for o in observations))

    def test_tracked_region_is_decoded_between_localization_frames(self):
        engine = realtime.FastScanner({})
        image = frame()
        engine.observe(image, 1, 1)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (640, 400), borderValue=180)
        with patch.object(engine.tracker.finder, 'find', side_effect=AssertionError('Localizer should not be needed')):
            observations = engine.observe(shifted, 2, 1.03)
        self.assertTrue(any(o['confirmed'] for o in observations))

    def test_unrelated_frame_does_not_inherit_valid_tracking(self):
        image = frame()
        corners = np.float32([[120, 100], [300, 100], [300, 280], [120, 280]])
        self.assertIsNone(realtime.follow_region(image, np.full_like(image, 180), corners))

    def test_every_fourth_job_advances_independent_multiscale_coverage(self):
        engine = realtime.RecoveryScanner({})
        image = np.zeros((1080, 1920), np.uint8)
        candidates = [(np.float32([[50,50],[100,50],[100,100],[50,100]]), 100)]
        visited = []
        for job in range(1, 33):
            view, kind, _ = engine.choose_view(image, candidates, job*.01)
            if job % 4 == 0:
                self.assertEqual(kind, 'tile')
                visited.append(view)
        self.assertEqual(len(set(visited)), 8)
        self.assertTrue(any(v[2]-v[0] == 192 for v in visited))
        self.assertTrue(any(v[2]-v[0] == 384 for v in visited))

    def test_recovery_raw_precedes_recipe_and_retry_keeps_original_frame(self):
        engine = realtime.RecoveryScanner({})
        image = np.zeros((400,640),np.uint8)
        engine.job = 1  # choose a raw tile before scheduling its retry
        with patch.object(engine.tracker, '_read', return_value=[]), patch.object(engine.tracker, '_run', return_value=[]) as heavy:
            engine.observe(image, 7, 10)
            heavy.assert_not_called()
            engine.observe(np.full_like(image, 200), 8, 10.05)
        heavy.assert_called_once()
        self.assertEqual(engine.last_source[1], 7)
        self.assertEqual(int(heavy.call_args.args[2].max()), 0)

    def test_slow_raw_failure_still_gets_recovery_on_original_pixels(self):
        engine = realtime.RecoveryScanner({})
        engine.job = 1
        image = np.zeros((400,640),np.uint8)
        with patch.object(engine.tracker, '_read', return_value=[]), patch.object(engine.tracker, '_run', return_value=[]) as heavy:
            engine.observe(image, 1, 10)
            engine.observe(np.full_like(image,200), 2, 10.5)
        heavy.assert_called_once()
        self.assertEqual(engine.last_source[1],1)

    def test_ncnn_proposal_decodes_native_crop_and_box_is_not_success(self):
        engine = realtime.RecoveryScanner({})
        engine.model = Mock()
        engine.model.infer.return_value = [{'box': [120,100,300,280], 'confidence': .8}]
        # Make the overview fail to exercise proposal -> original crop decoding.
        original_read = engine.tracker._read
        def read(image, *args, **kwargs):
            return [] if image.shape == (400,640) else original_read(image, *args, **kwargs)
        with patch.object(engine.tracker, '_read', side_effect=read):
            observations = engine.observe(frame(), 1, 10)
        self.assertTrue(any(o['confirmed'] and o['method'] == 'recovery/ncnn/zxing' for o in observations))
        engine.reset()
        with patch.object(engine.tracker, '_read', return_value=[]):
            observations = engine.observe(np.zeros((400,640),np.uint8), 2, 11)
        self.assertFalse(any(o['confirmed'] for o in observations))

    def test_two_spawned_workers_return_real_decode_and_close(self):
        import importlib.util
        modes = ['heuristic'] + (['hybrid'] if importlib.util.find_spec('ncnn') else [])
        for mode in modes:
            with self.subTest(mode=mode):
                options = dict(width=640, height=400, localizer=mode)
                worker = realtime.ProcessDecoder(options)
                with worker:
                    worker.wait_ready(timeout=15)
                    self.assertEqual(len({p.pid for p in worker.processes}), 2)
                    started = time.monotonic()
                    image = frame()
                    stamp = datetime.datetime.now().astimezone()
                    worker.submit(((1, started, stamp, image),))
                    result = worker.confirmed.get(timeout=10)
                    self.assertEqual(result['payload'], 'https://example.com/code')
                    self.assertEqual(result['frame_id'], 1)
                    self.assertGreaterEqual(result['result_monotonic'], started)
                self.assertIsNone(worker.error)

    def test_paced_replay_uses_real_workers_without_ground_truth_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.avi'
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,400))
            if not writer.isOpened():
                self.skipTest('Video writer unavailable')
            for _ in range(8):
                writer.write(cv2.cvtColor(frame(),cv2.COLOR_GRAY2BGR))
            writer.release()
            trial = dict(id='paced',condition='shade',start_s=0,end_s=.7,
                         expected=['deliberately-wrong-ground-truth'])
            report = qualification.replay_realtime(path,trial)
            self.assertFalse(report['success'])
            self.assertEqual(report['incorrect_payloads'],['https://example.com/code'])
            self.assertIn('https://example.com/code',report['confirmed_latency_s'])

    def test_current_camera_pixels_are_not_painted_with_old_boxes(self):
        current = np.full((300,500,3), 90, np.uint8)
        source = np.full_like(current, 180)
        observation = dict(corners=[[20,100],[80,100],[80,160],[20,160]], confirmed=True,
                           payload='old', quality={'module_px_estimate':3}, track_id=1, method='test')
        rendered = pipeline.live_overlay(current, (source, [observation], time.monotonic()-.5))
        np.testing.assert_array_equal(rendered[100,20], current[100,20])
        np.testing.assert_array_equal(rendered[-20,-20], source[-20,-20])


class DeliveryAndTimingTests(unittest.TestCase):
    def test_url_final_event_contains_content_not_url_and_provenance(self):
        resolver = Mock()
        resolver.submit.return_value = []
        output = io.StringIO()
        writer = scanner.ScanWriter(output, resolver=resolver)
        now = time.monotonic()
        captured = datetime.datetime.now().astimezone()
        observation = dict(frame_id=42, capture_monotonic=now-.2, result_monotonic=now-.1, decode_ms=90)
        with patch('builtins.print') as log:
            writer.record('https://example.com', captured, observation)
            resolver.poll.return_value = [('https://example.com', captured, dict(url='https://example.com/final',content='69',fetch_ms=20))]
            writer.poll_links()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        final = records[-1]
        self.assertEqual((final['type'], final['schema_version'], final['status'], final['data']), ('result',2,'ok','69'))
        self.assertEqual(final['original_payload'],'https://example.com')
        self.assertEqual(final['frame_id'],42)
        self.assertEqual(final['final_url'],'https://example.com/final')
        self.assertGreaterEqual(final['timings']['read_complete_to_content_ms'],199.9)
        self.assertIn('Result: 69',log.call_args.args[0])

    def test_failed_fetch_never_becomes_a_url_result(self):
        output = io.StringIO()
        writer = scanner.ScanWriter(output)
        with patch('builtins.print'):
            writer._record_link('https://example.com',datetime.datetime.now(),dict(url='https://example.com',error='offline'))
        final = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(final['status'],'resolution_failed')
        self.assertIsNone(final['data'])

    def test_confirmed_output_survives_full_diagnostic_queue(self):
        output = io.StringIO()
        worker = pipeline.OutputWorker(scanner.ScanWriter(output))
        for _ in range(16):
            worker.queue.put(([],None))
        with patch('builtins.print'):
            worker.submit([dict(confirmed=True,payload='important',captured_at=None)])
            with worker:
                pass
        self.assertEqual(json.loads(output.getvalue())['data'],'important')

    def test_capture_drains_to_latest_and_keeps_full_profile(self):
        capture = pipeline.ProcessCapture(dict(width=20,height=10))
        capture.channel = queue.Queue()
        capture.received_since_restart = 0
        old = np.zeros((10,20),np.uint8)
        new = np.ones_like(old)
        now = time.monotonic()
        capture.channel.put(('frame',old,now,None,{'selected_mode':{'fps':30}}, {'capture_count':1}))
        capture.channel.put(('frame',new,now,None,None,{'capture_count':2}))
        image, _, _ = capture.read()
        self.assertIs(image,new)
        self.assertEqual(capture.profile['selected_mode']['fps'],30)
        self.assertIn('queue_residence_ms',capture.profile)

    def test_competition_rejects_disabled_link_resolution(self):
        with patch('sys.stderr',new=io.StringIO()), self.assertRaises(SystemExit):
            scanner.main(['--competition','--no-resolve-links'])

    def test_grayscale_diagnostics_reach_linux_capture_benchmark(self):
        with patch.object(scanner.sys,'platform','linux'), patch('capture_benchmark.run',return_value=0) as run:
            self.assertEqual(scanner.main(['--camera-test','--gray-capture','--diagnostics','--no-preview']),0)
        self.assertTrue(run.call_args.args[0]['gray_capture'])

    def test_qualification_reports_empirical_target_separately_from_confidence(self):
        row = dict(condition='shade',expected=['x'],success=True,incorrect_payloads=[],distance_m=10,stable_hover=True,obliquity_deg=0)
        rows = [row]*95+[dict(row,success=False)]*5
        result = qualification.summarize(rows)['shade']
        self.assertTrue(result['qualification_pass'])
        self.assertFalse(result['statistically_supported_95pct'])
        self.assertFalse(qualification.summarize(rows[:-1])['shade']['qualification_pass'])


if __name__ == '__main__':
    unittest.main()
