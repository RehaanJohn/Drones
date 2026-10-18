"""Regressions for small symbols, recovery scheduling, fusion and capture integrity."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner
import tracking
import pipeline
from benchmark import synthetic_cases


class RangeRegressionTests(unittest.TestCase):
    def test_five_real_codes_do_not_starve_fifth_detection(self):
        frame = np.full((1080, 1920), 180, np.uint8)
        expected = {f'CODE-{i}' for i in range(5)}
        for i, payload in enumerate(sorted(expected)):
            qr = cv2.resize(cv2.QRCodeEncoder_create().encode(payload), (180, 180),
                            interpolation=cv2.INTER_NEAREST)
            x = 60 + 360 * i
            frame[200:380, x:x + 180] = qr
        engine = scanner.QRScanner()
        found = set()
        for i in range(9):
            found.update(engine.decode(frame))
        self.assertEqual(found, expected)
        self.assertGreater(engine.cursor, 0)

    def test_cheap_failed_full_search_still_visits_tiles(self):
        engine = scanner.QRScanner(Mock(), budget_ms=300)
        points = np.float32([[30,30],[100,30],[100,100],[30,100]])
        with patch.object(engine.tracker.finder, 'find', return_value=[(points, 100)]), \
                patch.object(engine.tracker, '_run', return_value=[]), \
                patch.object(engine.tracker, '_read', return_value=[]), \
                patch.object(engine.tracker, '_read_wechat', return_value=[]):
            for i in range(10):
                engine.observe(np.full((300,500),180,np.uint8), i, 10 + i * .1)
        self.assertGreater(engine.cursor, 0)

    def test_identical_crops_fuse_without_darkening_or_input_mutation(self):
        roi = np.random.default_rng(7).integers(20,235,(60,60),dtype=np.uint8)
        track = tracking.Track(1, np.float32([[0,0],[60,0],[60,60],[0,60]]), 0)
        track.crops.extend([roi.copy(), roi.copy(), roi.copy()])
        engine = scanner.QRScanner(Mock())
        fused, _ = engine.tracker._fuse(track, roi, 2)
        expected = cv2.resize(roi, (120,120), interpolation=cv2.INTER_CUBIC)
        self.assertLess(np.abs(fused.astype(float) - expected).mean(), 1)
        for crop in track.crops:
            np.testing.assert_array_equal(crop, roi)

    def test_small_isolated_codes_are_localized_without_global_percentile_gate(self):
        finder = tracking.CandidateFinder()
        for size in (40, 60, 80, 100):
            with self.subTest(size=size):
                qr = cv2.resize(cv2.QRCodeEncoder_create().encode('DRONE-01'), (size,size),
                                interpolation=cv2.INTER_NEAREST)
                frame = np.full((1080,1920),180,np.uint8)
                frame[475:475+size,610:610+size] = qr
                candidates = finder.find(frame)
                self.assertTrue(candidates)
                center = np.float32([610+size/2,475+size/2])
                self.assertTrue(any(np.linalg.norm(p.mean(axis=0)-center) < size/2 for p, _ in candidates))

    def test_lighting_disabled_excludes_all_crop_and_fusion_corrections(self):
        engine = scanner.QRScanner(Mock(), lighting=False, budget_ms=500)
        points = np.float32([[30,30],[100,30],[100,100],[30,100]])
        with patch.object(engine.tracker.finder, 'find', return_value=[(points, 100)]), \
                patch.object(engine.tracker, '_run', return_value=[]) as run, \
                patch.object(engine.tracker, '_discover', return_value=False):
            for i in range(8):
                engine.observe(np.full((300,500),180,np.uint8), i, 10 + i*.1)
        self.assertFalse(tracking.LIGHTING_RECIPES.intersection(c.args[0] for c in run.call_args_list))
        roi = np.random.default_rng(3).integers(0,255,(30,30),dtype=np.uint8)
        track = tracking.Track(1, points, 0)
        track.crops.extend([roi,roi])
        with patch.object(engine.tracker, '_decode_roi', return_value=[]) as decode:
            engine.tracker._run('fuse',track,roi,(0,0),2)
        self.assertEqual(decode.call_count, 1)

    def test_successful_upscale_retried_on_next_capture_for_confirmation(self):
        engine = scanner.QRScanner(Mock(), confirm_frames=2, budget_ms=500)
        points = np.float32([[30,30],[100,30],[100,100],[30,100]])
        with patch.object(engine.tracker.finder, 'find', return_value=[(points, 100)]), \
                patch.object(engine.tracker, '_run', side_effect=lambda name,*a: [('tiny',points)] if name=='crop-up' else []), \
                patch.object(engine.tracker, '_discover', return_value=False):
            first = engine.observe(np.full((300,500),180,np.uint8), 1, 10)
            second = engine.observe(np.full((300,500),180,np.uint8), 2, 10.1)
        self.assertFalse(first[0]['confirmed'])
        self.assertTrue(second[0]['confirmed'])

    def test_real_shadow_wechat_recovery_confirms_on_next_frame(self):
        _, frame, _, _ = next(c for c in synthetic_cases() if c[0] == '90px_shadow')
        engine = scanner.QRScanner(confirm_frames=2, budget_ms=500)
        first = engine.observe(frame, 1, 10)
        self.assertTrue(any(o['payload'] == 'DRONE-01' for o in first))
        second = engine.observe(frame, 2, 10.1)
        self.assertTrue(any(o['confirmed'] and o['payload'] == 'DRONE-01' for o in second))

    def test_blank_image_does_not_create_candidates_or_payloads(self):
        engine = scanner.QRScanner()
        for brightness in (0, 80, 255):
            frame = np.full((300,500),brightness,np.uint8)
            self.assertEqual(engine.decode(frame), [])
            self.assertEqual(engine.last_stats['candidates'], 0)


class SharedFrameTests(unittest.TestCase):
    def make_capture(self):
        capture = pipeline.ProcessCapture({})
        capture.slot_bytes = 16
        capture.shm = SimpleNamespace(buf=bytearray(range(16)))
        capture.slot_locks = [threading.Lock()]
        capture.slot_versions = [2]
        return capture

    def test_overwritten_slot_is_rejected_not_relabelled_with_old_timestamp(self):
        capture = self.make_capture()
        self.assertIsNone(capture._copy_shared_frame((0,(4,4),1)))
        frame = capture._copy_shared_frame((0,(4,4),2))
        np.testing.assert_array_equal(frame,np.arange(16,dtype=np.uint8).reshape(4,4))
        capture.shm.buf[:] = bytes(16)
        self.assertEqual(int(frame.sum()),120)  # Parent owns a stable copy.

    def test_reader_waits_for_writer_then_rechecks_generation(self):
        capture = self.make_capture()
        lock = capture.slot_locks[0]
        lock.acquire()
        results = []
        thread = threading.Thread(target=lambda: results.append(capture._copy_shared_frame((0,(4,4),2))))
        thread.start()
        capture.slot_versions[0] = 3
        lock.release()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results,[None])


if __name__ == '__main__':
    unittest.main()
