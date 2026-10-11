"""Native-pixel candidate tracking, quality-directed recovery and confirmation."""
from dataclasses import dataclass, field
import time

import cv2
import numpy as np


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
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ppm = max(1, min(gray.shape) / (modules + 8))
    clahe = cv2.createCLAHE(2, (8, 8))
    choices = []
    if metrics['illumination_range'] > 35:
        background = cv2.GaussianBlur(gray, (0, 0), max(2, ppm * 5))
        choices.append(('shadow', clahe.apply(cv2.divide(gray, np.maximum(background, 1), scale=128))))
    if image.ndim == 3:
        spreads = np.percentile(image, 95, axis=(0, 1)) - np.percentile(image, 5, axis=(0, 1))
        channel = int(np.argmax(spreads))
        if spreads[channel] > metrics['contrast'] * 1.2:
            choices.append(('channel', clahe.apply(image[:, :, channel])))
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
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(points, target),
                               (edge + border * 2,) * 2, borderValue=(255, 255, 255))


@dataclass
class Track:
    id: int
    corners: np.ndarray
    seen: float
    attempted: float = 0
    payload_frames: dict = field(default_factory=dict)


class TrackingDecoder:
    def __init__(self, owner, zxing, crop, tiles, budget_ms=150, modules=29):
        self.owner, self.zxing, self.crop, self.tiles = owner, zxing, crop, tiles
        self.budget = budget_ms / 1000
        self.modules = modules
        self.tracks = []
        self.next_id = 1
        self.sequence = 0
        self.previous_gray = None
        self.previous_time = None
        self.discovery_credit = 0.0
        self.observations = []

    def reset(self):
        self.tracks.clear()
        self.previous_gray = None
        self.previous_time = None
        self.discovery_credit = 0
        self.next_id = 1
        self.sequence = 0
        self.observations = []

    def _associate(self, points, timestamp, refresh=True):
        points = np.asarray(points, np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        size = max(8, float(np.linalg.norm(points[0] - points[2])))
        matches = [t for t in self.tracks if np.linalg.norm(t.corners.mean(axis=0) - center) < size * .3]
        track = min(matches, key=lambda t: np.linalg.norm(t.corners.mean(axis=0) - center)) if matches else None
        if track is None:
            if len(self.tracks) >= 64:
                self.tracks.remove(min(self.tracks, key=lambda t: t.seen))
            track = Track(self.next_id, points, timestamp)
            self.next_id += 1
            self.tracks.append(track)
        if refresh and timestamp >= track.seen:
            track.corners, track.seen = points, timestamp
        return track

    def _record(self, text, points, method, frame, frame_id, stamp, captured_at):
        track = self._associate(points, stamp, refresh=bool(text)) if points is not None else None
        crop = self.crop(frame, points) if points is not None else frame
        if crop is None or not crop.size:
            return
        confirmed = False
        if track and text:
            evidence = track.payload_frames.setdefault(text, {})
            evidence[frame_id] = stamp
            for payload in list(track.payload_frames):
                track.payload_frames[payload] = {key: when for key, when in track.payload_frames[payload].items()
                                                 if abs(max(stamp, track.seen) - when) <= 2}
                if not track.payload_frames[payload]:
                    del track.payload_frames[payload]
            confirmed = len(track.payload_frames.get(text, {})) >= 2
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
        return track

    def _decode_view(self, view, offset, scale, method, frame, frame_id, stamp, captured_at,
                     known_points=None):
        decoded = False
        def record(text, points):
            nonlocal decoded
            if not text:
                return
            decoded = True
            corners = known_points if known_points is not None else points
            if corners is not None and known_points is None:
                corners = np.asarray(corners, np.float32).reshape(4, 2) + offset
            self._record(text, corners, method, frame, frame_id, stamp, captured_at)
        if self.owner.use_zxing:
            for result in self.zxing.read_barcodes(view, formats=self.zxing.BarcodeFormat.QRCode,
                                                  try_rotate=True, try_downscale=True, try_invert=True,
                                                  text_mode=self.zxing.TextMode.Plain):
                position = result.position
                points = np.float32([[p.x, p.y] for p in
                                    (position.top_left, position.top_right, position.bottom_right, position.bottom_left)])
                record(result.text, points)
        if not decoded:
            self.owner.detector.setScaleFactor(scale)
            texts, corners = self.owner.detector.detectAndDecode(view)
            for i, text in enumerate(texts):
                record(text, corners[i] if i < len(corners) else None)
        if not decoded and known_points is not None:
            text, _, _ = self.owner.qr.detectAndDecode(view)
            record(text, known_points)
        return decoded

    def _move_tracks(self, frame, stamp):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Backward history retries do not rewind track geometry or create optical-flow jumps.
        if self.previous_time is not None and stamp <= self.previous_time:
            return False
        self.tracks = [t for t in self.tracks if stamp - t.seen < .75]
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape and self.tracks:
            points = np.concatenate([t.corners for t in self.tracks]).astype(np.float32).reshape(-1, 1, 2)
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None)
            if moved is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None)
                for index, track in enumerate(self.tracks):
                    sl = slice(index * 4, index * 4 + 4)
                    if back is not None and np.all(status[sl]) and np.all(back_status[sl]) and np.max(np.linalg.norm(back[sl] - points[sl], axis=2)) < 2:
                        track.corners = moved[sl].reshape(4, 2)
        self.previous_gray, self.previous_time = gray, stamp
        return True

    def decode(self, frame, frame_id=None, timestamp=None, captured_at=None):
        if frame is None or not frame.size:
            return []
        self.sequence += 1
        frame_id = self.sequence if frame_id is None else frame_id
        stamp = time.monotonic() if timestamp is None else timestamp
        started = time.perf_counter()
        self.observations = []
        attempted = 0
        forward = self._move_tracks(frame, stamp)
        if not forward:
            # Re-localize historical frames; never use current corners to crop an old image.
            self._decode_view(frame, (0, 0), min(1, 640 / max(frame.shape[:2])), 'history',
                              frame, frame_id, stamp, captured_at)
        else:
            # Existing candidates get the first 75% of the scheduling window.
            self._recover(frame, frame_id, stamp, captured_at, started + self.budget * .75)
            # Guaranteed discovery reservation; debt carries over when a native call overruns.
            self.discovery_credit = min(self.budget, self.discovery_credit + self.budget * .25)
            discovery_start = time.perf_counter()
            discover = self.discovery_credit > 0 or not self.tracks
            if not self.tracks:
                self._decode_view(frame, (0, 0), min(1, 640 / max(frame.shape[:2])), 'full',
                                  frame, frame_id, stamp, captured_at)
            regions = self.tiles(frame.shape, self.owner.tile_size)
            attempted = 0
            while discover and attempted < self.owner.tiles_per_frame and (attempted == 0 or time.perf_counter() - discovery_start < self.discovery_credit):
                x1, y1, x2, y2 = regions[self.owner.cursor % len(regions)]
                self.owner.cursor += 1
                attempted += 1
                tile = frame[y1:y2, x1:x2]
                self._decode_view(tile, (x1, y1), 1, 'tile', frame, frame_id, stamp, captured_at)
                gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                found, corners = self.owner.qr.detectMulti(gray)
                if found and corners is not None:
                    for points in corners:
                        self._associate(points + (x1, y1), stamp)
                # A localizer can fail under shadow too. One measured tile variant is a discovery fallback.
                if self.owner.lighting and not found and time.perf_counter() - started < self.budget:
                    mode, corrected = variants(tile, quality(tile, self.modules), self.modules)[0]
                    self._decode_view(corrected, (x1, y1), 1, 'search/' + mode, frame, frame_id, stamp, captured_at)
            if discover:
                self.discovery_credit -= time.perf_counter() - discovery_start
            self._recover(frame, frame_id, stamp, captured_at, started + self.budget)
        self.owner.last_stats = {'decode_ms': round((time.perf_counter() - started) * 1000, 2),
                                 'tracks': len(self.tracks), 'methods': sorted({o['method'] for o in self.observations}),
                                 'tiles': attempted,
                                 'budget_ms': self.budget * 1000}
        return list(dict.fromkeys(o['payload'] for o in self.observations if o['payload']))

    def _recover(self, frame, frame_id, stamp, captured_at, deadline):
        # Oldest unattempted track first, not detector output ordering.
        for track in sorted(self.tracks, key=lambda t: t.attempted):
            if time.perf_counter() >= deadline:
                break
            if any(o['track_id'] == track.id for o in self.observations):
                continue
            track.attempted = time.perf_counter()
            crop = self.crop(frame, track.corners)
            if crop is None or not crop.size:
                continue
            metrics = quality(crop, self.modules)
            self._record('', track.corners, 'candidate', frame, frame_id, stamp, captured_at)
            views = [('crop', crop)]
            warped = rectified(frame, track.corners)
            if warped is not None:
                views.append(('rectified', warped))
            if self.owner.lighting:
                views.extend(variants(warped if warped is not None else crop, metrics, self.modules))
            for name, view in views:
                if time.perf_counter() >= deadline:
                    break
                if self._decode_view(view, (0, 0), 1, name, frame, frame_id, stamp, captured_at, track.corners):
                    break
