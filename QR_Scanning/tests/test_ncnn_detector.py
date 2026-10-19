import importlib.util
import queue
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ncnn_detector as detector


class GeometryTests(unittest.TestCase):
    def test_nms_merges_overlapping_and_contained_tile_fragments(self):
        boxes = [{'box': [10,10,110,110], 'confidence': .99},
                 {'box': [12,12,112,112], 'confidence': .8},
                 {'box': [20,20,60,60], 'confidence': .7},
                 {'box': [200,200,230,230], 'confidence': .6}]
        self.assertEqual(detector.suppress_overlaps(boxes), [boxes[0],boxes[3]])

    def test_corrupt_or_missing_models_fail_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                detector.model_files(directory)
            (Path(directory)/'detect.param').write_text('bad')
            with self.assertRaisesRegex(RuntimeError, 'Checksum'):
                detector.model_files(directory)

    def test_draw_preserves_input(self):
        frame = np.full((200,300,3),150,np.uint8)
        original = frame.copy()
        detector.draw(frame,[{'box':[40,40,120,120],'confidence':.8}],{})
        np.testing.assert_array_equal(frame,original)


class LivePreviewTests(unittest.TestCase):
    def capture(self, frames):
        capture = MagicMock()
        capture.__enter__.return_value = capture
        capture.read.side_effect = [(frame, 0, None) for frame in frames]
        return capture

    def test_camera_window_opens_before_ncnn_model_initialization(self):
        capture = self.capture([np.full((120,160,3),70,np.uint8)])
        with patch.object(detector.sys,'platform','linux'), \
                patch.object(detector,'ProcessCapture',return_value=capture), \
                patch.object(detector,'NCNNQRDetector') as model, \
                patch.object(cv2,'namedWindow'), patch.object(cv2,'resizeWindow'), \
                patch.object(cv2,'imshow') as show, patch.object(cv2,'waitKey',return_value=ord('q')), \
                patch.object(cv2,'destroyWindow'), patch('builtins.print'):
            self.assertEqual(detector.main(['--device','/dev/video0']),0)
        show.assert_called_once()
        model.assert_not_called()
        capture.__exit__.assert_called_once()

    def test_live_pixels_keep_changing_without_new_inference_results(self):
        frames = [np.full((120,160,3),value,np.uint8) for value in (40,80,120)]
        capture = self.capture(frames)
        worker = MagicMock()
        worker.__enter__.return_value = worker
        worker.error, worker.status = None, 'Inference ready'
        worker.results = queue.Queue()
        worker.results.put((np.full_like(frames[0],200),0,[],{}))
        shown = []
        with patch.object(detector.sys,'platform','linux'), \
                patch.object(detector,'ProcessCapture',return_value=capture), \
                patch.object(detector,'DetectionWorker',return_value=worker), \
                patch.object(cv2,'namedWindow'), patch.object(cv2,'resizeWindow'), \
                patch.object(cv2,'imshow',side_effect=lambda name,image: shown.append(image.copy())), \
                patch.object(cv2,'waitKey',side_effect=[-1,-1,-1,ord('q')]), \
                patch.object(cv2,'getWindowProperty',return_value=1), \
                patch.object(cv2,'destroyWindow'), patch('builtins.print'):
            self.assertEqual(detector.main(['--device','/dev/video0']),0)
        self.assertEqual([int(frame[50,30,0]) for frame in shown],[40,40,80,120])
        self.assertEqual(worker.submit.call_count,3)
        # The analyzed image belongs only in the inset, never over the live view.
        self.assertEqual(int(shown[-1][-10,-10,0]),200)

    def test_live_view_has_no_stale_boxes_on_current_frame(self):
        current = np.full((400,640,3),80,np.uint8)
        old = np.full_like(current,180)
        result = (old,10,[{'box':[20,80,100,160],'confidence':.9}],{})
        image = detector.live_view(current,result,12,'Inference ready')
        np.testing.assert_array_equal(image[80:160,20:100],current[80:160,20:100])


@unittest.skipUnless(importlib.util.find_spec('ncnn') and (detector.MODEL_DIR/'detect.bin').is_file(),
                     'Optional NCNN runtime and pinned models required')
class RealNCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = detector.NCNNQRDetector(threads=1, tiles_per_frame=2)

    def test_blank_output_is_safe_and_no_decode_api_is_used(self):
        with patch.object(cv2, 'QRCodeDetector', side_effect=AssertionError('Must not decode')):
            self.assertEqual(self.engine.detect(np.full((300,500),128,np.uint8)), [])

    def test_real_small_qr_detects_with_original_coordinates_and_preserves_input(self):
        qr = cv2.resize(cv2.QRCodeEncoder_create().encode('NCNN-only'), (60,60),
                        interpolation=cv2.INTER_NEAREST)
        frame = np.full((1080,1920),170,np.uint8)
        frame[700:760,1300:1360] = qr
        original = frame.copy()
        found = self.engine.detect(frame)
        self.assertTrue(any(d['box'][0] <= 1330 <= d['box'][2] and
                            d['box'][1] <= 730 <= d['box'][3] for d in found), found)
        np.testing.assert_array_equal(frame,original)

    def test_crop_offset_and_non_square_letterbox_coordinates(self):
        qr = cv2.resize(cv2.QRCodeEncoder_create().encode('offset'),(180,180),interpolation=cv2.INTER_NEAREST)
        frame = np.full((240,480),180,np.uint8)
        frame[30:210,150:330] = qr
        boxes = self.engine.infer(frame, (1000,500))
        self.assertTrue(any(d['box'][0] <= 1240 <= d['box'][2] and
                            d['box'][1] <= 620 <= d['box'][3] for d in boxes), boxes)
        for item in boxes:
            x1,y1,x2,y2 = item['box']
            self.assertTrue(1000 <= x1 < x2 <= 1480)
            self.assertTrue(500 <= y1 < y2 <= 740)

    def test_each_tile_gets_every_lighting_recipe_across_sweeps(self):
        engine = detector.NCNNQRDetector(threads=1, tile_sizes=(128,), tiles_per_frame=1,
                                          max_candidates=0)
        # Three tiles: a frame-based modulo-three rotation would revisit each
        # tile with the same recipe forever. Rotation must use the tile sweep.
        frame = np.full((100,300),128,np.uint8)
        with patch.object(engine,'infer',return_value=[]) as infer:
            for _ in range(9):
                engine.detect(frame)
        methods = {call.args[2] for call in infer.call_args_list
                   if call.args[1] == (0,0) and call.args[2].startswith('tile/')}
        self.assertEqual(methods,{'tile/raw','tile/clahe','tile/shadow','tile/gamma'})

    def test_still_preview_calls_imshow_and_closes_window(self):
        image = detector.MODEL_DIR.parent/'diagnostic_target.png'
        with patch.object(cv2,'namedWindow'), patch.object(cv2,'resizeWindow'), \
                patch.object(cv2,'imshow') as show, patch.object(cv2,'waitKey',return_value=ord('q')), \
                patch.object(cv2,'destroyWindow') as close, patch('builtins.print'):
            self.assertEqual(detector.main(['--image',str(image),'--threads','1']),0)
        show.assert_called_once()
        close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
