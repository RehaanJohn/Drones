"""Independent latest-frame fast decoding and scheduled discovery/recovery.

Only frame notifications and small results use queues. Native grayscale pixels
cross processes in a generation-checked shared ring. Ground truth is never input.
"""
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import queue
import signal
import sys
import time

import cv2
import numpy as np

if __package__:
    from .scanner import QRScanner, create_detector, tile_regions
    from .tracking import TRACK_TTL, roi_rect
    from .pipeline import replace_latest, _attach_shm
else:
    from scanner import QRScanner, create_detector, tile_regions
    from tracking import TRACK_TTL, roi_rect
    from pipeline import replace_latest, _attach_shm


class LazyWeChat:
    def __init__(self):
        self.instance = None
        self.scale = 1.

    def setScaleFactor(self, scale):
        self.scale = scale

    def detectAndDecode(self, image):
        if self.instance is None:
            self.instance = create_detector()
        self.instance.setScaleFactor(self.scale)
        return self.instance.detectAndDecode(image)


def scanner_options(options, fast=False):
    return dict(detector=LazyWeChat(), budget_ms=options.get('fast_budget_ms', 33) if fast else options.get('budget_ms', 150),
                modules=options.get('qr_modules', 29), confirm_frames=options.get('confirm_frames', 1),
                use_zxing=not options.get('no_zxing', False), lighting=not options.get('no_lighting', False),
                max_candidates=options.get('max_candidates', 4), target_ppm=options.get('target_ppm', 5),
                invert=options.get('try_invert', False))


def follow_region(previous, current, corners):
    """Forward/backward checked local translation; reject uncertain correspondences."""
    rect = roi_rect(previous.shape, corners, .1)
    if rect is None or previous.shape != current.shape:
        return None
    x1, y1, x2, y2 = rect
    points = cv2.goodFeaturesToTrack(previous[y1:y2, x1:x2], 16, .05, 3)
    if points is None or len(points) < 4:
        return None
    points += np.float32([x1, y1])
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(15, 15), maxLevel=2)
    if moved is None:
        return None
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(15, 15), maxLevel=2)
    if back is None:
        return None
    good = (status.ravel() != 0) & (status_back.ravel() != 0) & (np.linalg.norm(back-points, axis=2).ravel() < 1.5)
    if np.count_nonzero(good) < 4:
        return None
    offsets = (moved-points).reshape(-1, 2)[good]
    shift = np.median(offsets, axis=0)
    if not np.isfinite(shift).all() or np.median(np.linalg.norm(offsets-shift, axis=1)) > 2:
        return None
    mapped = np.asarray(corners, np.float32) + shift
    if mapped.min() < 0 or mapped[:, 0].max() >= current.shape[1] or mapped[:, 1].max() >= current.shape[0]:
        return None
    return mapped


class FastScanner:
    def __init__(self, options):
        self.engine = QRScanner(**scanner_options(options, fast=True))
        self.tracker = self.engine.tracker
        self.previous = None
        self.previous_stamp = None
        self.last_stats = {}

    def reset(self):
        self.engine.reset()
        self.previous = None

    def observe(self, gray, frame_id, stamp, captured_at=None, proposals=()):
        started = time.perf_counter()
        tracker = self.tracker
        tracker.sequence += 1
        tracker.observations = []
        tracker.tracks = [t for t in tracker.tracks if 0 <= stamp-t.seen < TRACK_TTL]
        usable = set()
        if self.previous is not None and 0 <= stamp-self.previous_stamp < TRACK_TTL:
            for track in tracker.tracks:
                mapped = follow_region(self.previous, gray, track.corners)
                if mapped is not None:
                    track.corners = mapped
                    usable.add(track.id)
        for points in proposals:
            usable.add(tracker._associate(points, stamp).id)
        # Retained tracks get service even between independent localizer runs.
        candidates = tracker.finder.find(gray) if not usable or tracker.sequence % 3 == 1 else []
        for points, score in candidates:
            usable.add(tracker._associate(points, stamp, score=score).id)
        localized = time.perf_counter()
        method_ms = {}
        live = sorted((t for t in tracker.tracks if t.id in usable), key=lambda t: (t.last_service, -t.score))
        for index, track in enumerate(live):
            if index and time.perf_counter()-started >= tracker.budget:
                break
            rect = roi_rect(gray.shape, track.corners)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
            track.last_service = tracker.sequence
            names = ['crop', 'crop-up'] if track.last_method != 'crop-up' else ['crop-up', 'crop']
            found = []
            for name in names:
                if name != names[0] and time.perf_counter()-started >= tracker.budget:
                    break
                before = time.perf_counter()
                edge = max(1., float(np.min(np.linalg.norm(track.corners-np.roll(track.corners, -1, axis=0), axis=1))))
                scale = min(4., max(1., tracker.target_ppm*tracker.modules/edge))
                found = tracker._run(name, track, roi, (x1, y1), scale)
                method_ms[name] = method_ms.get(name, 0) + (time.perf_counter()-before)*1000
                if found:
                    track.last_method = name
                    for text, points in found:
                        tracker._record(text, points, 'fast/'+name, gray, frame_id, stamp, captured_at)
                    break
            if not found:
                tracker._record('', track.corners, 'fast/tracked', gray, frame_id, stamp, captured_at)
        self.previous, self.previous_stamp = gray, stamp
        elapsed = (time.perf_counter()-started)*1000
        self.last_stats = dict(decode_ms=elapsed, localize_ms=(localized-started)*1000,
                               budget_overrun_ms=max(0, elapsed-tracker.budget*1000),
                               tracks=len(tracker.tracks), candidates=len(candidates), method_ms=method_ms,
                               methods=sorted({o['method'] for o in tracker.observations}))
        return tracker.observations


class RecoveryScanner:
    """One scheduled view per job; fourth jobs always advance independent tiles."""
    def __init__(self, options):
        self.options = options
        self.engine = QRScanner(**scanner_options(options))
        self.tracker = self.engine.tracker
        self.model = None
        if options.get('localizer', 'heuristic') == 'hybrid':
            if __package__:
                from .ncnn_model import NCNNQRDetector
            else:
                from ncnn_model import NCNNQRDetector
            self.model = NCNNQRDetector(confidence=options.get('ncnn_confidence', .6),
                                         threads=options.get('ncnn_threads', 1), lighting=False,
                                         tiles_per_frame=0, max_candidates=0)
        self.job = self.cursor = self.candidate_cursor = 0
        self.shape = None
        self.tiles = []
        self.sweep_start = None
        self.sweep_ms = None
        self.retry = None
        self.last_stats = {}

    def reset(self):
        self.engine.reset()
        self.job = self.cursor = self.candidate_cursor = 0
        self.shape = None
        self.retry = None
        self.sweep_start = self.sweep_ms = None

    def choose_view(self, gray, candidates, stamp):
        self.job += 1
        if self.shape != gray.shape:
            self.shape, self.cursor = gray.shape, 0
            grids = [tile_regions(gray.shape, edge) for edge in (384, 192)]
            self.tiles = list(dict.fromkeys(region for i in range(max(map(len, grids)))
                                           for grid in grids for region in grid[i:i+1]))
            self.sweep_start = stamp
        if self.job % 4 == 0 or (not candidates and self.retry is None and self.job % 8 != 1):
            return self.next_tile(stamp), 'tile', None
        if self.job % 8 == 1:
            return (0, 0, gray.shape[1], gray.shape[0]), 'overview', None
        if self.retry is not None:
            view, source_stamp, recipe = self.retry
            self.retry = None
            if 0 <= stamp-source_stamp < .75:
                # Retried on the original crop by observe(), never stale coordinates on this frame.
                return view, 'retry', recipe
        if candidates:
            points, _ = candidates[self.candidate_cursor % len(candidates)]
            self.candidate_cursor += 1
            return roi_rect(gray.shape, points), 'candidate', None
        return self.next_tile(stamp), 'tile', None

    def next_tile(self, stamp):
        view = self.tiles[self.cursor % len(self.tiles)]
        self.cursor += 1
        if self.cursor % len(self.tiles) == 0:
            self.sweep_ms = (stamp-self.sweep_start)*1000
            self.sweep_start = stamp
        return view

    def observe(self, gray, frame_id, stamp, captured_at=None):
        began = time.perf_counter()
        tracker = self.tracker
        tracker.sequence += 1
        tracker.observations = []
        tracker.tracks = [t for t in tracker.tracks if 0 <= stamp-t.seen < TRACK_TTL]
        candidates = tracker.finder.find(gray)
        for points, score in candidates:
            tracker._associate(points, stamp, score=score)
        localize_ms = (time.perf_counter()-began)*1000
        view, kind, recipe = self.choose_view(gray, candidates, stamp)
        source = (gray, frame_id, stamp, captured_at)
        if kind == 'retry':
            source = self.retry_source
            gray, frame_id, stamp, captured_at = source
        self.last_source = source
        if view is None:
            view = (0, 0, gray.shape[1], gray.shape[0])
        x1, y1, x2, y2 = view
        roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
        corners = np.float32([[x1, y1], [x2-1, y1], [x2-1, y2-1], [x1, y2-1]])
        method_ms = {}
        before = time.perf_counter()
        if recipe:
            track = tracker._associate(corners, stamp, refresh=False)
            # Fusion only uses recent independently located crops, not arbitrary background tiles.
            found = tracker._run(recipe, track, roi, (x1, y1), 2.)
        else:
            found = [(text, points+(x1, y1)) for text, points in tracker._read(roi, downscale=kind == 'overview')]
        method_ms[recipe or 'zxing'] = (time.perf_counter()-before)*1000
        for text, points in found:
            tracker._record(text, points, 'recovery/'+kind+'/'+(recipe or 'raw'), gray, frame_id, stamp, captured_at)
        if not found and self.model is not None and recipe is None:
            before = time.perf_counter()
            boxes = self.model.infer(roi, (x1, y1), kind+'/raw')
            method_ms['ncnn'] = (time.perf_counter()-before)*1000
            for box in boxes[:4]:
                a, b, c, d = box['box']
                points = np.float32([[a, b], [c, b], [c, d], [a, d]])
                rect = roi_rect(gray.shape, points)
                if rect is None:
                    continue
                track = tracker._associate(points, stamp)
                u, v, w, h = rect
                crop = np.ascontiguousarray(gray[v:h, u:w])
                before = time.perf_counter()
                decoded = tracker._decode_roi(crop, (u, v), 1.)
                method_ms['proposal_decode'] = method_ms.get('proposal_decode', 0)+(time.perf_counter()-before)*1000
                if decoded:
                    found.extend(decoded)
                    for text, quad in decoded:
                        tracker._record(text, quad, 'recovery/ncnn/zxing', gray, frame_id, stamp, captured_at)
                else:
                    tracker._record('', points, 'recovery/ncnn', gray, frame_id, stamp, captured_at)
                track.crops.append(crop)
                if time.perf_counter()-began >= tracker.budget:
                    break
        if not found and recipe is None and kind != 'overview':
            track = tracker._associate(corners, stamp)
            if kind == 'candidate':
                track.crops.append(roi)
            recipes = ['wechat', 'crop-up', 'rectify']
            if self.engine.lighting:
                recipes += ['clahe', 'shadow', 'sharpen', 'adaptive']
            if kind == 'candidate' and len(track.crops) >= 2:
                recipes.append('fuse')
            self.retry = (view, stamp, recipes[track.rot % len(recipes)])
            self.retry_source = source
            track.rot += 1
        # Independent proposals, even when no payload was decoded.
        for points, _ in candidates:
            if kind != 'retry':
                tracker._record('', points, 'recovery/candidate', gray, frame_id, stamp, captured_at)
        elapsed = (time.perf_counter()-began)*1000
        self.last_stats = dict(decode_ms=elapsed, localize_ms=localize_ms, method_ms=method_ms,
                               methods=sorted({o['method'] for o in tracker.observations}),
                               discovery_job=kind, tiles_visited=self.cursor, tiles_in_sweep=len(self.tiles),
                               sweep_ms=self.sweep_ms, sweep_elapsed_ms=max(0, (stamp-self.sweep_start)*1000),
                               budget_overrun_ms=max(0, elapsed-tracker.budget*1000))
        return tracker.observations


def read_slot(shm, locks, versions, slot_bytes, message):
    sequence, stamp, captured, shape, slot = message
    if not locks[slot].acquire(timeout=.5):
        raise RuntimeError('Analysis shared-memory read stalled')
    try:
        if versions[slot] != sequence:
            return None
        return np.ndarray(shape, np.uint8, buffer=shm.buf, offset=slot*slot_bytes).copy()
    finally:
        locks[slot].release()


def _analysis_process(kind, options, name, slot_bytes, locks, versions, inputs,
                      updates, confirmed, errors, proposals, controls, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
    cv2.setNumThreads(1)
    shm = _attach_shm(name)
    try:
        engine = FastScanner(options) if kind == 'fast' else RecoveryScanner(options)
        updates.put(([], {'worker': kind, 'ready': True}))
        minimum_stamp = None
        started, count = time.monotonic(), 0
        cpu_started = time.process_time()
        sent = {}
        durations = deque(maxlen=300)
        ages = deque(maxlen=300)
        while not stop.is_set():
            try:
                message = inputs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                lighting, minimum_stamp = controls.get_nowait()
            except queue.Empty:
                pass
            else:
                engine.engine.lighting = lighting
                engine.reset()
            sequence, stamp, captured, shape, slot = message
            if minimum_stamp is not None and stamp < minimum_stamp:
                continue
            copy_started = time.monotonic()
            frame = read_slot(shm, locks, versions, slot_bytes, message)
            if frame is None:
                continue
            begin = time.monotonic()
            mapped = []
            if kind == 'fast':
                try:
                    old_message, points = proposals.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if 0 <= stamp-old_message[1] < .25:
                        old = read_slot(shm, locks, versions, slot_bytes, old_message)
                        if old is not None:
                            for corners in points:
                                region = follow_region(old, frame, corners)
                                if region is not None:
                                    mapped.append(region)
                observations = engine.observe(frame, sequence, stamp, captured, mapped)
            else:
                observations = engine.observe(frame, sequence, stamp, captured)
                points = [o['corners'] for o in observations if o['corners'] is not None and o['frame_id'] == sequence]
                replace_latest(proposals, (message, points))
                _, sequence, stamp, captured = engine.last_source
            completed = time.monotonic()
            durations.append((completed-begin)*1000)
            ages.append((completed-stamp)*1000)
            count += 1
            for observation in observations:
                observation.update(result_monotonic=completed, decode_ms=engine.last_stats['decode_ms'], worker=kind)
                if observation['confirmed'] and observation['payload']:
                    key = observation['payload']
                    sent = {k: t for k, t in sent.items() if completed-t < 2}
                    if key not in sent:
                        # Payloads must not share replaceable frame/preview queues.
                        while True:
                            try:
                                confirmed.put(observation, timeout=.1)
                                sent[key] = completed
                                break
                            except queue.Full:
                                if stop.is_set():
                                    raise RuntimeError('Shutdown with confirmed queue saturated; output did not drain')
                                continue
            stats = dict(engine.last_stats, worker=kind, worker_ms=(completed-begin)*1000,
                         analysis_shared_memory_read_ms=(begin-copy_started)*1000,
                         analysis_queue_ms=max(0,(copy_started-message[1])*1000),
                         cpu_percent=100*(time.process_time()-cpu_started)/max(.001,completed-started),
                         worker_p50_ms=float(np.median(durations)), worker_p95_ms=float(np.percentile(durations,95)),
                         read_complete_to_result_p95_ms=float(np.percentile(ages,95)),
                         processed_fps=count/max(.001, completed-started),
                         frame_age_ms=(completed-stamp)*1000, frame_id=sequence,
                         completed_monotonic=completed, processed_frames=[dict(frame_id=sequence,
                         capture_monotonic=stamp, result_monotonic=completed,
                         latency_ms=(completed-stamp)*1000, decode_ms=engine.last_stats['decode_ms'])])
            replace_latest(updates, (observations, stats))
            stop.wait(max(0, options.get('interval', 0)-(time.monotonic()-begin)))
    except Exception as exc:
        errors.put((kind, str(exc)))
    finally:
        shm.close()


class ProcessDecoder:
    """Two independent consumers of one bounded shared grayscale ring."""
    def __init__(self, options):
        self.options = dict(options)
        self.context = mp.get_context('spawn')
        self.stop = self.context.Event()
        self.slot_bytes = options['width']*options['height']
        self.slots = 8
        self.shm = None
        self.processes = []
        self.channels = []
        self.observations, self.stats = [], {}
        self.results = queue.Queue(2)
        self.error = None
        self.dropped_results = 0
        self.workers_stats = {}
        self.ready = set()
        self.result_sink = None

    def _queue(self, size):
        channel = self.context.Queue(size)
        self.channels.append(channel)
        return channel

    def __enter__(self):
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.slot_bytes*self.slots)
            self.locks = [self.context.Lock() for _ in range(self.slots)]
            self.versions = self.context.Array('Q', self.slots, lock=False)
            self.confirmed, self.errors, self.proposals = self._queue(128), self._queue(4), self._queue(1)
            self.inputs, self.updates, self.controls = [], [], []
            for kind in ('fast', 'recovery'):
                inp, update, control = self._queue(1), self._queue(1), self._queue(1)
                self.inputs.append(inp)
                self.updates.append(update)
                self.controls.append(control)
                process = self.context.Process(target=_analysis_process, args=(kind, self.options,
                    self.shm.name, self.slot_bytes, self.locks, self.versions, inp, update,
                    self.confirmed, self.errors, self.proposals, control, self.stop), daemon=True)
                process.start()
                self.processes.append(process)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def set_lighting(self, enabled, since=None):
        for channel in self.controls:
            replace_latest(channel, (enabled, since))

    def submit(self, history):
        if not history:
            return
        sequence, stamp, captured, image = history[-1]
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.nbytes > self.slot_bytes:
            raise RuntimeError('Frame exceeds configured shared-memory dimensions')
        slot = sequence % self.slots
        began = time.monotonic()
        if not self.locks[slot].acquire(timeout=.5):
            raise RuntimeError('Analysis shared-memory write stalled')
        try:
            np.ndarray(gray.shape, np.uint8, buffer=self.shm.buf, offset=slot*self.slot_bytes)[...] = gray
            self.versions[slot] = sequence
        finally:
            self.locks[slot].release()
        for channel in self.inputs:
            replace_latest(channel, (sequence, stamp, captured, gray.shape, slot))
        self.stats['analysis_shared_memory_write_ms'] = (time.monotonic()-began)*1000
        self.poll()

    def poll(self):
        try:
            kind, message = self.errors.get_nowait()
            self.error = RuntimeError(f'{kind}: {message}')
        except queue.Empty:
            pass
        for process in self.processes:
            if process.exitcode is not None and not self.stop.is_set():
                self.error = self.error or RuntimeError(f'Analysis worker exited: {process.exitcode}')
        for channel in self.updates:
            try:
                observations, stats = channel.get_nowait()
            except queue.Empty:
                continue
            self.workers_stats[stats['worker']] = stats
            if stats.get('ready'):
                self.ready.add(stats['worker'])
                continue
            self.stats.update(stats)
            self.stats['workers'] = dict(self.workers_stats)
            self.observations = observations
            replace_latest(self.results, (observations, stats))

    def wait_ready(self, timeout=30):
        deadline = time.monotonic()+timeout
        while len(self.ready) < 2:
            self.poll()
            if self.error:
                raise self.error
            if time.monotonic() >= deadline:
                raise RuntimeError('Analysis workers did not initialize within timeout')
            time.sleep(.01)

    def __exit__(self, *exc):
        self.stop.set()
        for process in self.processes:
            process.join(1)
            if process.is_alive():
                print('Stopping analysis worker still finishing a native call; unfinished work has no result.', file=sys.stderr, flush=True)
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        for channel in self.channels:
            if channel is getattr(self, 'confirmed', None) and self.result_sink:
                while True:
                    try:
                        self.result_sink([channel.get_nowait()])
                    except queue.Empty:
                        break
            channel.cancel_join_thread()
            channel.close()
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
