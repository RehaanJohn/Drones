"""Shared pinned NCNN inference; no camera or GUI ownership."""
import hashlib
import math
from pathlib import Path
import time
import urllib.request
import cv2
import numpy as np
if __package__:
    from .scanner import tile_regions
    from .tracking import CandidateFinder, roi_rect
else:
    from scanner import tile_regions
    from tracking import CandidateFinder, roi_rect

MODEL_DIR = Path(__file__).resolve().parent / 'ncnn_models'
REVISION = '0e006ebf71ad977af7c6e0e86da038b242792c4e'
MODEL_URL = ('https://raw.githubusercontent.com/EdVince/QRCode-NCNN/' + REVISION +
             '/Project_android/ncnn-android-nanodet/app/src/main/assets/')
MODEL_HASHES = {
    'detect.param': 'f136b8beb1d03505735e527a2ba6ff9247cd9a0b70d15f3ba9cabfaf227c32e3',
    'detect.bin': '11dc5df644206da6ccc0d34aa761523255fa441cd163fd4e70e12ad033d352a5',
}



def model_files(directory, download=False):
    directory = Path(directory)
    for name, expected in MODEL_HASHES.items():
        path = directory / name
        if not path.is_file() and download:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                with urllib.request.urlopen(MODEL_URL + name, timeout=30) as response:
                    data = response.read(2 * 1024 * 1024)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise RuntimeError(f'Checksum mismatch for downloaded {name}')
                temporary.write_bytes(data)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file():
            raise RuntimeError(f'Missing {path}. Run this script with --download-models first.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch for {path}; restore the pinned model before running.')
    return directory / 'detect.param', directory / 'detect.bin'


def suppress_overlaps(detections, threshold=.35):
    """Cross-view NMS in original-frame coordinates; retain strongest model evidence."""
    kept = []
    for item in sorted(detections, key=lambda d: -d['confidence']):
        x1, y1, x2, y2 = item['box']
        area = (x2 - x1) * (y2 - y1)
        duplicate = False
        for other in kept:
            a, b, c, d = other['box']
            intersection = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            union = area + (c - a) * (d - b) - intersection
            smaller = min(area, (c - a) * (d - b))
            if union > 0 and (intersection / union > threshold or
                              (smaller > 0 and intersection / smaller > .85)):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


class NCNNQRDetector:
    """Native crops preserve tiny targets that disappear in a downscaled overview.

    The inexpensive edge localizer only proposes crops: every returned box must
    come from NCNN inference. No QR decoding API is called by this class.
    """
    def __init__(self, model_dir=MODEL_DIR, confidence=.6, threads=2, tile_sizes=(384, 192),
                 tiles_per_frame=4, lighting=True, max_candidates=2, exhaustive=False, progress=None):
        report = progress or (lambda message: None)
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError('Confidence must be between zero and one')
        if threads < 1 or tiles_per_frame < 0 or max_candidates < 0 or not tile_sizes or min(tile_sizes) < 64:
            raise ValueError('Invalid thread count, tile sizes, tile count or candidate count')
        try:
            report('Importing NCNN runtime...')
            import ncnn
        except ImportError as exc:
            raise RuntimeError('Install NCNN with: python -m pip install --no-deps ncnn==1.0.20260526') from exc
        report('Verifying model files...')
        param, weights = model_files(model_dir)
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        # The upstream graph filters at 0.2. Set its threshold to the requested
        # value too, otherwise --confidence below 0.2 would have no effect.
        lines = param.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            if line.startswith('DetectionOutput'):
                tokens = line.split()
                lines[i] = ' '.join(f'4={confidence}' if t.startswith('4=') else t for t in tokens)
        report('Loading NCNN model...')
        if self.net.load_param_mem('\n'.join(lines) + '\n') != 0 or self.net.load_model(str(weights)) != 0:
            raise RuntimeError('NCNN could not load the QR detector')
        self.confidence = confidence
        self.tile_sizes = tuple(dict.fromkeys(tile_sizes))
        self.tiles_per_frame, self.lighting = tiles_per_frame, lighting
        self.exhaustive = exhaustive
        self.finder = CandidateFinder(max_candidates=max_candidates, min_score=5)
        self.clahe = cv2.createCLAHE(2, (8, 8))
        self.cursor = self.sequence = 0
        self.shape = None
        self.last_stats = {}
        report('NCNN model ready')

    def infer(self, gray, offset=(0, 0), method='raw'):
        """Letterbox to 384 square; undo padding and scale before adding crop offset."""
        h, w = gray.shape
        scale = 384 / max(h, w)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_CUBIC)
        left, top = (384 - rw) // 2, (384 - rh) // 2
        padded = np.full((384, 384), 127, np.uint8)
        padded[top:top + rh, left:left + rw] = resized
        mat = self.ncnn.Mat.from_pixels(padded, self.ncnn.Mat.PixelType.PIXEL_GRAY, 384, 384)
        mat.substract_mean_normalize([], [1 / 255])
        with self.net.create_extractor() as extractor:
            if extractor.input('data', mat) != 0:
                raise RuntimeError('NCNN input failed')
            status, output = extractor.extract('detection_output')
            if status != 0:
                raise RuntimeError('NCNN output failed')
            # Empty NCNN Mats have no buffer; NumPy conversion can hang/crash
            # in native bindings. A negative detection is an ordinary result.
            if output.empty():
                return []
            rows = np.asarray(output).copy()
        if not rows.size:
            return []
        if rows.size % 6:
            raise RuntimeError(f'Unexpected NCNN output shape: {rows.shape}')
        found = []
        for label, confidence, x1, y1, x2, y2 in rows.reshape(-1, 6):
            if not np.isfinite([label, confidence, x1, y1, x2, y2]).all():
                continue
            if label != 1 or confidence < self.confidence:
                continue
            # Separate scale factors account for integer resize rounding.
            x1, x2 = np.clip((np.array([x1, x2]) * 384 - left) * w / rw, 0, w)
            y1, y2 = np.clip((np.array([y1, y2]) * 384 - top) * h / rh, 0, h)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            found.append({'box': [float(x1 + offset[0]), float(y1 + offset[1]),
                                  float(x2 + offset[0]), float(y2 + offset[1])],
                          'confidence': float(confidence), 'method': method})
        return found

    def _lighting(self, gray, mode):
        if mode == 'clahe':
            return self.clahe.apply(gray)
        if mode == 'shadow':
            background = cv2.GaussianBlur(gray, (0, 0), max(3, min(gray.shape) / 12))
            return self.clahe.apply(cv2.divide(gray, np.maximum(background, 1), scale=128))
        # Lift dark regions without altering the untouched inference view.
        lut = np.clip((np.arange(256) / 255.) ** .5 * 255, 0, 255).astype(np.uint8)
        return cv2.LUT(gray, lut)

    def detect(self, frame):
        if frame is None or not frame.size:
            return []
        start = time.perf_counter()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.shape:
            self.shape, self.cursor = gray.shape, 0
        self.sequence += 1
        h, w = gray.shape
        # Interleave scales so the smaller native crops do not wait an entire
        # large-tile sweep. Every edge is covered; crops overlap by 25%.
        grids = [tile_regions(gray.shape, size) for size in self.tile_sizes]
        regions = list(dict.fromkeys(region for i in range(max(map(len, grids)))
                                     for grid in grids for region in grid[i:i + 1]))
        views = [(0, 0, w, h, 'overview', self.sequence - 1)]
        for points, _ in self.finder.find(gray):
            rect = roi_rect(gray.shape, points, .35)
            if rect:
                views.append((*rect, 'candidate', self.sequence - 1))
        count = len(regions) if self.exhaustive else min(self.tiles_per_frame, len(regions))
        for _ in range(count):
            sweep, index = divmod(self.cursor, len(regions))
            views.append((*regions[index], 'tile', sweep))
            self.cursor += 1
        found, passes = [], 0
        visited = set()
        for x1, y1, x2, y2, kind, phase in views:
            if (x1, y1, x2, y2) in visited:
                continue
            visited.add((x1, y1, x2, y2))
            roi = gray[y1:y2, x1:x2]
            found.extend(self.infer(roi, (x1, y1), kind + '/raw'))
            passes += 1
            if self.lighting:
                modes = ('clahe', 'shadow', 'gamma')
                selected = modes if self.exhaustive else (modes[phase % len(modes)],)
                for mode in selected:
                    found.extend(self.infer(self._lighting(roi, mode), (x1, y1), kind + '/' + mode))
                    passes += 1
        result = suppress_overlaps(found)
        self.last_stats = {'inference_ms': round((time.perf_counter() - start) * 1000, 1),
                           'passes': passes, 'tiles_visited': count, 'tiles_in_sweep': len(regions),
                           'tile_cursor': self.cursor, 'regions': len(result)}
        return result

