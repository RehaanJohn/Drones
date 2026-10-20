"""Detection-only NCNN QR preview. Boxes are hypotheses, never decoded/validated payloads.

Uses the WeChat QR SSD converted by EdVince/QRCode-NCNN. Model assets are pinned
and verified; normal startup is offline. See NCNN_DETECTOR.md for setup and limits.
"""
import argparse
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

if __name__ == '__main__':
    print('[startup] Importing OpenCV and NumPy...', file=sys.stderr, flush=True)

import cv2
import numpy as np

if __package__:
    from .scanner import open_camera, configure_camera, LatestFrameCapture, MainThreadCapture
    from .pipeline import ProcessCapture, replace_latest
else:
    from scanner import open_camera, configure_camera, LatestFrameCapture, MainThreadCapture
    from pipeline import ProcessCapture, replace_latest

if __package__:
    from .ncnn_model import NCNNQRDetector, MODEL_DIR, MODEL_HASHES, model_files, suppress_overlaps
else:
    from ncnn_model import NCNNQRDetector, MODEL_DIR, MODEL_HASHES, model_files, suppress_overlaps
WINDOW = 'NCNN QR regions (no decoding) - Q / Esc to quit'


class DetectionWorker:
    """Latest frame wins; inference stays off the preview/capture thread."""
    def __init__(self, detector=None, factory=None):
        self.detector = detector
        self.factory = factory
        self.status = 'Loading NCNN model' if detector is None else 'Waiting for first inference'
        self.input, self.results = queue.Queue(1), queue.Queue(1)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='ncnn-qr')

    def __enter__(self):
        self.thread.start()
        return self

    def submit(self, frame, stamp):
        replace_latest(self.input, (frame, stamp))

    def _progress(self, message):
        self.status = message
        print('[NCNN] ' + message, file=sys.stderr, flush=True)

    def _run(self):
        try:
            if self.detector is None:
                self.detector = self.factory(self._progress)
            first = True
            while not self.stop.is_set():
                try:
                    frame, stamp = self.input.get(timeout=.1)
                except queue.Empty:
                    continue
                if first:
                    self._progress('Running first inference...')
                boxes = self.detector.detect(frame)
                replace_latest(self.results, (frame, stamp, boxes, dict(self.detector.last_stats)))
                if first:
                    self._progress('Inference ready')
                    first = False
        except Exception as exc:
            self.error = exc

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(2)


def draw(frame, boxes, stats):
    image = frame.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in boxes:
        x1, y1, x2, y2 = (int(v) for v in item['box'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(image, f"QR-like {item['confidence']:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
    label = (f"QR-like regions: {len(boxes)} (not decoded) | {stats.get('inference_ms', 0):.0f} ms | "
             f"tiles {stats.get('tiles_visited', 0)}/{stats.get('tiles_in_sweep', 0)} | "
             f"age {stats.get('result_age_ms', 0):.0f} ms")
    label = stats.get('status', label)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return image


def live_view(frame, result, now, status):
    """Always show the current capture; old geometry stays in a labelled inset."""
    image = draw(frame, [], {'status': 'LIVE CAMERA | ' + status + ' | Q / Esc to quit'})
    if result is None:
        return image
    analyzed, stamp, boxes, stats = result
    age_ms = max(0, (now - stamp) * 1000)
    thumbnail = draw(analyzed, boxes, dict(stats, result_age_ms=age_ms))
    height, width = image.shape[:2]
    scale = min(.42 * width / thumbnail.shape[1], .42 * height / thumbnail.shape[0])
    tw, th = max(1, round(thumbnail.shape[1] * scale)), max(1, round(thumbnail.shape[0] * scale))
    thumbnail = cv2.resize(thumbnail, (tw, th), interpolation=cv2.INTER_AREA)
    x, y = width - tw, height - th
    image[y:, x:] = thumbnail
    cv2.rectangle(image, (x, y), (width - 1, height - 1), (0, 220, 255), 2)
    cv2.putText(image, f'Last analysis: {len(boxes)} regions, {age_ms:.0f} ms ago',
                (x, max(48, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
    return image


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--camera', type=int)
    sources.add_argument('--camera-name')
    sources.add_argument('--device', help='Linux V4L2 device path')
    sources.add_argument('--image', type=Path, help='Analyze a still image; --exhaustive searches all tiles/lighting')
    sources.add_argument('--video', type=Path, help='Analyze every frame of a recording')
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--download-models', action='store_true', help='Provision/check pinned weights and exit')
    parser.add_argument('--self-test', action='store_true', help='Test NCNN on a blank image, without camera or window')
    parser.add_argument('--confidence', type=float, default=.6,
                        help='Model score threshold; lower toward 0.25 for more candidates and more false positives')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tile-sizes', nargs='+', type=int, default=[384, 192])
    parser.add_argument('--tiles-per-frame', type=int, default=4)
    parser.add_argument('--max-candidates', type=int, default=2)
    parser.add_argument('--exhaustive', action='store_true', help='All tiles and lighting recipes on each frame; slow')
    parser.add_argument('--no-lighting', action='store_true')
    parser.add_argument('--no-preview', action='store_true', help='For headless image/video checks')
    parser.add_argument('--diagnostics', action='store_true', help='Print requested and actual camera settings')
    parser.add_argument('--output', type=Path, help='Save annotated image when --image is used')
    parser.add_argument('--backend', choices=('auto', 'msmf', 'dshow', 'avfoundation', 'v4l2', 'any'), default='auto')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--fourcc')
    parser.add_argument('--exposure-us', type=int, help='Linux manual exposure, microseconds')
    parser.add_argument('--auto-exposure', type=float, help='Desktop backend-specific value')
    parser.add_argument('--exposure', type=float, help='Desktop backend-specific value')
    parser.add_argument('--autofocus', type=int, choices=(0, 1))
    parser.add_argument('--focus', type=float)
    parser.add_argument('--gain', type=float)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('Width, height and FPS must be positive')
    if args.fourcc and len(args.fourcc) != 4:
        parser.error('--fourcc must have four characters')
    for name in ('auto_exposure', 'exposure', 'focus', 'gain'):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f'--{name.replace("_", "-")} must be finite')
    if args.exposure_us is not None and args.exposure_us <= 0:
        parser.error('--exposure-us must be positive')
    if args.output and not args.image:
        parser.error('--output requires --image')
    linux = sys.platform.startswith('linux')
    if not linux and (args.device or args.exposure_us):
        parser.error('--device and --exposure-us require Linux')
    if linux and (args.camera_name or args.backend not in ('auto', 'v4l2') or
                  args.exposure is not None or args.auto_exposure is not None):
        parser.error('Linux uses --device/--camera, V4L2 and --exposure-us')
    opened = False
    cap = None
    try:
        if args.download_models:
            model_files(args.model_dir, download=True)
            print(f'NCNN models verified: {args.model_dir.resolve()}')
            return 0
        def report(message):
            print('[startup] ' + message, file=sys.stderr, flush=True)

        def load_detector(progress=report):
            return NCNNQRDetector(args.model_dir, args.confidence, args.threads, args.tile_sizes,
                                   args.tiles_per_frame, not args.no_lighting, args.max_candidates,
                                   args.exhaustive, progress=progress)

        def show(frame, boxes, stats, wait=1, prepared=False):
            nonlocal opened
            image = frame if prepared else draw(frame, boxes, stats)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.output), image):
                    raise OSError(f'Could not write {args.output}')
            if args.no_preview:
                return True
            if not opened:
                report('Opening live preview window...')
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, min(1280, image.shape[1]),
                                 round(image.shape[0] * min(1, 1280 / image.shape[1])))
                opened = True
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                return False
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1

        if args.self_test:
            detector = load_detector()
            report('Testing NCNN inference without the camera...')
            detector.infer(np.full((384, 384), 128, np.uint8))
            report('NCNN self-test passed')
            return 0
        if args.image or args.video:
            detector = load_detector()
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise RuntimeError(f'Cannot read {args.image}')
            boxes = detector.detect(frame)
            print(json.dumps({'detections': boxes, 'stats': detector.last_stats}))
            show(frame, boxes, detector.last_stats, 0)
            return 0
        if args.video:
            cap = cv2.VideoCapture(str(args.video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {args.video}')
            while True:
                ok, frame = cap.read()
                if not ok:
                    return 0
                boxes = detector.detect(frame)
                if args.no_preview:
                    print(json.dumps({'detections': boxes, 'stats': detector.last_stats}))
                if not show(frame, boxes, detector.last_stats):
                    return 0
        report(f'Opening camera {args.device or args.camera_name or args.camera or "(auto)"}; '
               f'requested {args.width}x{args.height} at {args.fps:g} FPS...')
        if linux:
            source = ProcessCapture(vars(args))
        else:
            cap = open_camera(args.camera, args.backend, False, args.camera_name)
            configure_camera(cap, args)
            source = MainThreadCapture(cap) if sys.platform == 'darwin' else LatestFrameCapture(cap)
            cap = None
        next_report = 0
        with source as capture:
            report('Waiting for first camera frame...')
            frame, age, _ = capture.read()
            stamp = time.monotonic() - age / 1000
            report(f'Camera live: {frame.shape[1]}x{frame.shape[0]}')
            # Display the camera BEFORE starting model loading or native inference.
            if not args.no_preview and not show(live_view(frame, None, time.monotonic(), 'Starting NCNN'),
                                                 [], {}, prepared=True):
                return 0
            last_result = None
            with DetectionWorker(factory=load_detector) as worker:
                while True:
                    now = time.monotonic()
                    worker.submit(frame, stamp)
                    if worker.error:
                        raise RuntimeError(f'NCNN inference failed: {worker.error}') from worker.error
                    try:
                        last_result = worker.results.get_nowait()
                    except queue.Empty:
                        pass
                    if now >= next_report:
                        if last_result is None:
                            report('Camera live; ' + worker.status)
                        else:
                            _, result_stamp, boxes, stats = last_result
                            stats = dict(stats, result_age_ms=round((now - result_stamp) * 1000, 1))
                            print(json.dumps({'detections': boxes, 'stats': stats}), flush=True)
                        next_report = now + 2
                    # imshow and waitKey execute for every capture, including when
                    # the result queue is empty or a native inference is slow.
                    if not args.no_preview and not show(live_view(frame, last_result, now, worker.status),
                                                         [], {}, prepared=True):
                        return 0
                    frame, age, _ = capture.read()
                    stamp = time.monotonic() - age / 1000
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f'NCNN detector: {exc}', file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if opened:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass


if __name__ == '__main__':
    sys.exit(main())
