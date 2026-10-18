"""ROI-first QR pipeline: localize candidates, decode only their crops, escalate enhancement.

Per frame:
  1. CandidateFinder: cheap native-resolution localizer (dense two-direction edge energy).
  2. Each candidate ROI is decoded on its own crop. On failure the crop is enhanced with a
     rotating ladder of recipes (upscale, WeChat CNN+SR, multi-frame fusion, lighting fixes...).
  3. Periodic full-frame / rolling-tile discovery preserves coverage even after a successful decode.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

TRACK_TTL = 0.75      # seconds a candidate survives without being localized or decoded
MAX_TRACKS = 8
ROI_PAD = 0.30        # padding around a candidate; doubles as the quiet zone for the decoder
HEAVY = ('wechat', 'fuse', 'clahe', 'shadow', 'sharpen', 'rectify', 'adaptive')
LIGHTING_RECIPES = frozenset(('clahe', 'shadow', 'sharpen', 'adaptive'))


def quality(image, modules=29):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (5, 95))
    means = [float(part.mean()) for row in np.array_split(gray, 2) for part in np.array_split(row, 2, axis=1)]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(high - low)
    return {'sharpness': round(sharpness, 2), 'contrast': round(contrast, 2),
            'bright_fraction': float(np.mean(gray >= 250)), 'dark_fraction': float(np.mean(gray <= 5)),
            'mean': float(gray.mean()), 'illumination_range': max(means) - min(means),
            'module_px_estimate': min(gray.shape) / (modules + 8),
            # White/black QR modules naturally reach endpoints; clipping alone is NOT glare proof.
            'score': float(np.log1p(sharpness) * contrast / 255)}


def variants(image, metrics, modules=29):
    """Kept for diagnostics compatibility; the live path uses the recipe ladder below."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ppm = max(1, min(gray.shape) / (modules + 8))
    clahe = cv2.createCLAHE(2, (8, 8))
    choices = []
    if metrics['illumination_range'] > 35:
        background = cv2.GaussianBlur(gray, (0, 0), max(2, ppm * 5))
        choices.append(('shadow', clahe.apply(cv2.divide(gray, np.maximum(background, 1), scale=128))))
    if metrics['contrast'] < 100 or not choices:
        choices.append(('clahe', clahe.apply(gray)))
    block = max(3, int(round(ppm * 9)) | 1)
    if metrics['illumination_range'] > 20:
        choices.append(('adaptive', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, block, 5)))
    else:
        blurred = cv2.GaussianBlur(gray, (0, 0), max(.5, min(1, ppm / 4)))
        choices.append(('sharpen', cv2.addWeighted(gray, 1.4, blurred, -.4, 0)))
    return choices[:2]


def rectified(frame, corners):
    points = np.asarray(corners, np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or not cv2.isContourConvex(points):
        return None
    lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    if min(lengths) < 8 or max(lengths) / min(lengths) > 5:
        return None
    edge = min(1024, max(16, int(max(lengths))))
    border = max(4, round(edge * .18))
    target = np.float32([[border, border], [border + edge - 1, border],
                         [border + edge - 1, border + edge - 1], [border, border + edge - 1]])
    fill = int(np.percentile(frame, 90))
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(points, target),
                               (edge + border * 2,) * 2, borderValue=fill)


def roi_rect(shape, corners, pad=ROI_PAD):
    """Padded integer ROI (x1, y1, x2, y2) around corner points, clamped to the image."""
    pts = np.asarray(corners, np.float32).reshape(-1, 2)
    if not np.isfinite(pts).all():
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    x1, x2 = max(0, int(math.floor(x1 - w * pad))), min(shape[1], int(math.ceil(x2 + w * pad)))
    y1, y2 = max(0, int(math.floor(y1 - h * pad))), min(shape[0], int(math.ceil(y2 + h * pad)))
    return (x1, y1, x2, y2) if x2 - x1 > 4 and y2 - y1 > 4 else None


class CandidateFinder:
    """Localize QR-like regions without decoding anything.

    QR symbols and some background textures have dense edges in BOTH directions. We take Sobel
    magnitude at native resolution (so a tiny distant code keeps its energy), block-average it,
    keep min(gx, gy) so 1-D barcodes, lines and single edges are suppressed, then threshold and
    group blocks. This costs a few ms and only has to be roughly right: a false candidate costs
    one cheap crop decode, a true one gets all the decoding effort.
    """

    def __init__(self, block=8, max_candidates=4, min_px=24, min_score=8.0):
        self.block, self.max_candidates = block, max_candidates
        self.min_px, self.min_score = min_px, min_score
        self.kernel = np.ones((3, 3), np.uint8)

    def find(self, gray):
        h, w = gray.shape[:2]
        bw, bh = max(1, w // self.block), max(1, h // self.block)
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        ex = cv2.resize(cv2.convertScaleAbs(gx, alpha=.25), (bw, bh), interpolation=cv2.INTER_AREA)
        ey = cv2.resize(cv2.convertScaleAbs(gy, alpha=.25), (bw, bh), interpolation=cv2.INTER_AREA)
        score = np.minimum(ex, ey)
        # A distant symbol may occupy far less than 0.3% of the frame. Estimate
        # contrast from active cells, not a percentile dominated by empty sky/paper.
        active = score[score >= self.min_score]
        if not active.size:
            return []
        top = float(np.percentile(active, 90))
        threshold = max(self.min_score, .4 * top)
        for _ in range(3):
            mask = (score >= threshold).astype(np.uint8)
            if mask.mean() <= .12:
                break
            threshold *= 1.5          # busy texture everywhere: demand more energy
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sx, sy = w / bw, h / bh
        min_blocks = max(2, self.min_px // self.block)
        limit = .9 * min(h, w)
        found = []
        for i in sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:16]:
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if cw < min_blocks or ch < min_blocks or not .4 <= cw / ch <= 2.5 or area < .3 * cw * ch:
                continue
            if cw * sx > limit or ch * sy > limit:
                continue
            mean = cv2.mean(score, mask=(labels == i).astype(np.uint8))[0]
            cx, cy = (x + cw / 2) * sx, (y + ch / 2) * sy
            half = max(cw * sx, ch * sy) / 2
            x1, y1 = max(0., cx - half), max(0., cy - half)
            x2, y2 = min(w - 1., cx + half), min(h - 1., cy + half)
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            found.append((corners, float(mean * math.sqrt(area))))
        found.sort(key=lambda item: -item[1])
        return [(self.refine(gray, corners), score) for corners, score in found[:self.max_candidates]]

    def refine(self, gray, corners):
        """Tighten a coarse block-level box using native-resolution edge energy.

        Block boxes run oversize (blur, block quantization, neighbouring clutter). A tight box
        matters because it sets the module-size estimate that drives the upscale factor.
        """
        rect = roi_rect(gray.shape, corners, .15)
        if rect is None:
            return corners
        x1, y1, x2, y2 = rect
        roi = gray[y1:y2, x1:x2]
        ex = cv2.blur(cv2.convertScaleAbs(cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3), alpha=.25), (5, 5))
        ey = cv2.blur(cv2.convertScaleAbs(cv2.Sobel(roi, cv2.CV_16S, 0, 1, ksize=3), alpha=.25), (5, 5))
        energy = np.minimum(ex, ey)
        peak = float(np.percentile(energy, 97))
        if peak < self.min_score * .5:
            return corners
        mask = energy >= .35 * peak
        columns, rows = mask.mean(axis=0), mask.mean(axis=1)
        xs = np.flatnonzero(columns >= .25 * columns.max())
        ys = np.flatnonzero(rows >= .25 * rows.max())
        if not len(xs) or not len(ys):
            return corners
        left, right, top, bottom = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
        half = max(right - left, bottom - top) / 2
        cx, cy = x1 + (left + right) / 2, y1 + (top + bottom) / 2
        if half < self.min_px / 2:
            return corners
        height, width = gray.shape[:2]
        left, top = max(0., cx - half), max(0., cy - half)
        right, bottom = min(width - 1., cx + half), min(height - 1., cy + half)
        return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


@dataclass
class Track:
    id: int
    corners: np.ndarray
    seen: float
    score: float = 0.0
    fails: int = 0
    rot: int = 0                          # position in the HEAVY recipe rotation
    last_decode: float = float('-inf')
    last_method: str = ''
    last_service: int = -1
    payload_frames: dict = field(default_factory=dict)
    crops: deque = field(default_factory=lambda: deque(maxlen=4))


class TrackingDecoder:
    def __init__(self, owner, zxing, crop, tiles, budget_ms=150, modules=29, confirm_frames=1,
                 invert=False, max_candidates=4, discovery_every=3, target_ppm=5.0, heavy_per_visit=2):
        self.owner, self.zxing, self.crop, self.tiles = owner, zxing, crop, tiles
        self.budget = budget_ms / 1000
        self.modules = modules
        self.confirm_frames = max(1, confirm_frames)
        self.invert = invert
        self.discovery_every = max(1, discovery_every)
        self.target_ppm = target_ppm
        self.heavy_per_visit = heavy_per_visit
        self.finder = CandidateFinder(max_candidates=max_candidates)
        self.binarizers = [None]
        if zxing is not None and hasattr(zxing, 'Binarizer'):
            self.binarizers.append(zxing.Binarizer.GlobalHistogram)
        self.clahe = cv2.createCLAHE(2, (8, 8))
        self.tracks = []
        self.next_id = 1
        self.sequence = 0
        self.idle = 0
        self.attempted = 0
        self.observations = []
        self.full_ms = 0.0     # moving average cost of one native full-frame ZXing pass
        self._ppm = 5.0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.sequence = 0
        self.idle = 0
        self.observations = []
        self.full_ms = 0.0

    # ------------------------------------------------------------------ tracks
    def _associate(self, points, stamp, refresh=True, score=0.0):
        points = np.asarray(points, np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        size = max(8., float(np.linalg.norm(points[0] - points[2])))
        best, best_distance = None, float('inf')
        for track in self.tracks:
            distance = float(np.linalg.norm(track.corners.mean(axis=0) - center))
            reach = .6 * max(size, float(np.linalg.norm(track.corners[0] - track.corners[2])))
            if distance < reach and distance < best_distance:
                best, best_distance = track, distance
        if best is None:
            if len(self.tracks) >= MAX_TRACKS:
                self.tracks.remove(min(self.tracks, key=lambda t: t.seen))
            best = Track(self.next_id, points, stamp)
            self.next_id += 1
            self.tracks.append(best)
        if refresh and stamp >= best.seen:
            best.corners, best.seen = points, stamp
        if score:
            best.score = score
        return best

    def _record(self, text, points, method, gray, frame_id, stamp, captured_at):
        points = None if points is None else np.asarray(points, np.float32).reshape(4, 2)
        track = self._associate(points, stamp, refresh=bool(text)) if points is not None else None
        crop = self.crop(gray, points) if points is not None else gray
        if crop is None or not crop.size:
            return
        confirmed = False
        if track and text:
            track.payload_frames.setdefault(text, {})[frame_id] = stamp
            for payload in list(track.payload_frames):
                track.payload_frames[payload] = {k: t for k, t in track.payload_frames[payload].items()
                                                 if abs(stamp - t) <= 2}
                if not track.payload_frames[payload]:
                    del track.payload_frames[payload]
            confirmed = len(track.payload_frames.get(text, {})) >= self.confirm_frames
        metrics = quality(crop, self.modules)
        if points is not None:
            edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            metrics['module_px_estimate'] = float(min(edges) / self.modules)
        metrics['assumed_symbol_modules'] = self.modules
        observation = {'payload': text, 'corners': points.tolist() if points is not None else None,
                       'track_id': track.id if track else None, 'frame_id': frame_id,
                       'capture_monotonic': stamp, 'captured_at': captured_at,
                       'method': method, 'quality': metrics, 'confirmed': confirmed}
        if not any(o['payload'] == text and o['track_id'] == observation['track_id'] for o in self.observations):
            self.observations.append(observation)

    # ---------------------------------------------------------------- decoding
    def _read(self, image, binarizer=None, downscale=False):
        """Decode one image. Returns [(text, 4x2 corners in image coordinates)]."""
        if self.owner.use_zxing:
            options = dict(formats=self.zxing.BarcodeFormat.QRCode, try_rotate=False,
                           try_downscale=downscale, try_invert=self.invert,
                           text_mode=self.zxing.TextMode.Plain)
            if binarizer is not None:
                options['binarizer'] = binarizer
            found = []
            for result in self.zxing.read_barcodes(np.ascontiguousarray(image), **options):
                if not result.valid or not result.text:
                    continue
                p = result.position
                found.append((result.text, np.float32([[q.x, q.y] for q in
                                                       (p.top_left, p.top_right, p.bottom_right, p.bottom_left)])))
            return found
        text, points, _ = self.owner.qr.detectAndDecode(image)
        if text and points is not None:
            return [(text, points.reshape(-1, 2)[:4].astype(np.float32))]
        return []

    def _read_wechat(self, image, scale=1.0):
        self.owner.detector.setScaleFactor(scale)
        texts, corners = self.owner.detector.detectAndDecode(image)
        return [(text, np.asarray(corners[i], np.float32).reshape(4, 2))
                for i, text in enumerate(texts) if text and i < len(corners)]

    def _prepare(self, roi, scale, prescaled=False):
        """Upscale (cubic) and surround with a bright quiet zone. Returns (image, scale, border)."""
        if prescaled:
            image, scale = roi, scale
        elif scale > 1.01:
            image = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            image, scale = roi, 1.0
        border = int(max(image.shape[:2]) * .06) + 4
        image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT,
                                   value=int(np.percentile(roi, 90)))
        return image, scale, border

    @staticmethod
    def _map(found, offset, scale, border):
        return [(text, (pts - border) / scale + offset) for text, pts in found]

    def _decode_roi(self, roi, offset, scale, pre=None, binarizers=(None,), prescaled=False):
        image, scale, border = self._prepare(roi, scale, prescaled)
        if pre is not None:
            image = pre(image, scale)
        for binarizer in binarizers:
            found = self._read(image, binarizer)
            if found:
                return self._map(found, offset, scale, border)
        if pre is not None:
            return self._map(self._read_wechat(image), offset, scale, border)
        return []

    # Lighting recipes operate on the upscaled, bordered image.
    def _clahe(self, image, scale):
        return self.clahe.apply(image)

    def _shadow(self, image, scale):
        background = cv2.GaussianBlur(image, (0, 0), max(4., self._ppm * scale * 4))
        return self.clahe.apply(cv2.divide(image, np.maximum(background, 1), scale=128))

    def _sharpen(self, image, scale):
        blurred = cv2.GaussianBlur(image, (0, 0), max(.8, self._ppm * scale * .25))
        return cv2.addWeighted(image, 1.8, blurred, -.8, 0)

    def _adaptive(self, image, scale):
        block = max(15, int(round(self._ppm * scale * 8)) | 1)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 7)

    def _fuse(self, track, roi, scale):
        """Average several recent crops after translation alignment: suppresses sensor noise."""
        if len(track.crops) < 2:
            return None
        height, width = (max(16, int(v * scale)) for v in roi.shape)
        if max(height, width) > 640:
            factor = 640 / max(height, width)
            height, width = max(16, int(height * factor)), max(16, int(width * factor))
        reference = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        total, used = reference.copy(), 1
        for old in list(track.crops)[:-1]:
            image = cv2.resize(old, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            # OpenCV can apply the Hanning window in place to floating inputs.
            # Preserve unwindowed pixels for averaging and subsequent alignment.
            (dx, dy), response = cv2.phaseCorrelate(reference.copy(), image.copy(), window)
            if not all(math.isfinite(v) for v in (dx, dy, response)) or response < .1 or math.hypot(dx, dy) > .25 * min(width, height):
                continue
            total += cv2.warpAffine(image, np.float32([[1, 0, -dx], [0, 1, -dy]]), (width, height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            used += 1
        if used < 2:
            return None
        return np.clip(total / used, 0, 255).astype(np.uint8), width / roi.shape[1]

    def _run(self, name, track, roi, offset, scale):
        if name == 'crop':
            return self._decode_roi(roi, offset, 1.0)
        if name == 'crop-up':
            return self._decode_roi(roi, offset, max(scale, 1.5), binarizers=self.binarizers)
        if name == 'wechat':
            image, s, border = self._prepare(roi, 1.0)   # WeChat has its own super-resolution stage
            return self._map(self._read_wechat(image), offset, s, border)
        if name == 'fuse':
            fused = self._fuse(track, roi, max(scale, 2.0))
            if fused is None:
                return []
            image, s = fused
            found = self._decode_roi(image, offset, s, prescaled=True, binarizers=self.binarizers)
            if not found and self.owner.lighting:
                found = self._decode_roi(image, offset, s, self._clahe, prescaled=True)
            return found
        if name == 'rectify':
            image, s, border = self._prepare(roi, max(scale, 2.0))
            ok, points = self.owner.qr.detect(image)
            if not ok or points is None:
                return []
            corners = points.reshape(-1, 2)[:4].astype(np.float32)
            warped = rectified(image, corners)
            if warped is None:
                return []
            for text, _ in self._read(warped):
                return [(text, (corners - border) / s + offset)]
            return []
        pre = {'clahe': self._clahe, 'shadow': self._shadow, 'sharpen': self._sharpen,
               'adaptive': self._adaptive}[name]
        return self._decode_roi(roi, offset, max(scale, 2.0), pre, self.binarizers if name == 'sharpen' else (None,))

    def _service(self, track, gray, frame_id, stamp, captured_at, deadline):
        rect = roi_rect(gray.shape, track.corners)
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
        track.crops.append(roi)
        edge = float(np.mean(np.linalg.norm(track.corners - np.roll(track.corners, -1, axis=0), axis=1)))
        self._ppm = max(.5, edge / self.modules)
        scale = min(float(np.clip(self.target_ppm / self._ppm, 1.0, 4.0)), 1000 / max(roi.shape))
        track.last_service = self.sequence
        heavy = [name for name in HEAVY if self.owner.lighting or name not in LIGHTING_RECIPES]
        names = ['crop']
        if track.last_method in ('crop-up', *heavy):
            names.append(track.last_method)
        names += ['crop-up'] + [heavy[(track.rot + i) % len(heavy)] for i in range(self.heavy_per_visit)]
        for index, name in enumerate(dict.fromkeys(names)):
            if index and time.perf_counter() >= deadline:
                break
            found = self._run(name, track, roi, (x1, y1), scale)
            if found:
                for text, points in found:
                    self._record(text, points, name, gray, frame_id, stamp, captured_at)
                track.fails, track.last_decode, track.last_method = 0, stamp, name
                return True
        track.fails += 1
        track.rot += self.heavy_per_visit
        return False

    def _discover(self, frame, frame_id, stamp, captured_at, deadline):
        """Alternate global acquisition and native tiles; neither path can starve the other."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.idle += 1
        tiles = self.owner.tiles_per_frame
        if self.idle % 2 or not tiles:
            began = time.perf_counter()
            found = self._read(gray, downscale=True)
            self.full_ms = .7 * self.full_ms + .3 * (time.perf_counter() - began) * 1000
            method = 'full'
            if not found and (not self.owner.use_zxing or time.perf_counter() < deadline):
                method = 'full/wechat'
                found = self._read_wechat(gray, min(1., self.owner.tile_size / max(gray.shape)))
            for text, points in found:
                self._record(text, points, method, gray, frame_id, stamp, captured_at)
            return bool(found)
        regions = self.tiles(gray.shape, self.owner.tile_size)
        decoded = False
        for _ in range(tiles):
            if self.attempted and time.perf_counter() >= deadline:
                break
            sweep, index = divmod(self.owner.cursor, len(regions))
            self.owner.cursor += 1
            self.attempted += 1
            x1, y1, x2, y2 = regions[index]
            tile = np.ascontiguousarray(frame[y1:y2, x1:x2])
            modes = (*self.owner.LIGHTING_MODES, 'raw') if self.owner.lighting else ('raw',)
            mode = modes[sweep % len(modes)]
            view = tile if mode == 'raw' else self.owner.correct_lighting(tile, mode)
            method, found = 'tile/' + mode, self._read(view)
            # One tile is a complete decoder pair, even if the first native call
            # overruns. Otherwise expensive raw calls can starve recovery forever.
            if not found:
                method, found = method + '/wechat', self._read_wechat(view)
            for text, points in found:
                self._record(text, points + (x1, y1), method, gray, frame_id, stamp, captured_at)
            decoded |= bool(found)
        return decoded

    # -------------------------------------------------------------------- main
    def decode(self, frame, frame_id=None, timestamp=None, captured_at=None):
        if frame is None or not frame.size:
            self.observations = []
            return []
        self.sequence += 1
        frame_id = self.sequence if frame_id is None else frame_id
        stamp = time.monotonic() if timestamp is None else timestamp
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        deadline = started + self.budget
        self.observations = []
        self.attempted = 0
        self.tracks = [t for t in self.tracks if stamp - t.seen < TRACK_TTL]

        candidates = self.finder.find(gray)
        localize_ms = (time.perf_counter() - started) * 1000
        for corners, score in candidates:
            self._associate(corners, stamp, True, score)

        decoded = False
        # Reserve a turn before crop work on periodic frames. A busy scene or a
        # native call overrunning the budget cannot postpone discovery forever.
        discovery_due = self.sequence % self.discovery_every == 0
        if discovery_due:
            decoded |= self._discover(frame, frame_id, stamp, captured_at, started + self.budget * .4)
        live = [t for t in self.tracks if t.seen == stamp]
        # Oldest serviced first, then promising candidates. Large easy codes must
        # not monopolize the budget while a smaller code waits for recovery.
        live.sort(key=lambda t: (t.last_service, -t.score))
        service_deadline = started + self.budget * .8
        serviced = 0
        for track in live:
            if serviced and time.perf_counter() >= service_deadline:
                break
            if track.fails >= 8 and self.sequence % 3:
                continue
            serviced += 1
            decoded |= self._service(track, gray, frame_id, stamp, captured_at, service_deadline)

        # Full-frame ZXing binarizes differently from a crop, so it catches frames the ROI path
        # misses. Run it every frame while it is cheap; throttle it on slow hardware.
        cheap = self.full_ms < self.budget * 500 and self.owner.use_zxing
        if not discovery_due and not decoded and (not self.tracks or cheap):
            decoded |= self._discover(frame, frame_id, stamp, captured_at, deadline)

        for track in live:   # localization is useful diagnostic evidence even when decoding failed
            if not any(o['track_id'] == track.id for o in self.observations):
                self._record('', track.corners, 'candidate', gray, frame_id, stamp, captured_at)

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.owner.last_stats = {'decode_ms': round(elapsed_ms, 2),
                                 'budget_overrun_ms': round(max(0, elapsed_ms - self.budget * 1000), 2),
                                 'tracks': len(self.tracks), 'candidates': len(candidates),
                                 'localize_ms': round(localize_ms, 2),
                                 'methods': sorted({o['method'] for o in self.observations}),
                                 'tiles': self.attempted, 'budget_ms': self.budget * 1000}
        return list(dict.fromkeys(o['payload'] for o in self.observations if o['payload']))
