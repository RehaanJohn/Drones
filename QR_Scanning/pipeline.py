"""Bounded camera, decoding and output workers; HighGUI stays on the main thread."""
from collections import deque
import datetime
import json
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


def replace_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


class FrameHistory:
    def __init__(self, seconds=.5, byte_limit=64 * 1024 * 1024):
        self.seconds, self.byte_limit = seconds, byte_limit
        self.frames, self.bytes = deque(), 0
        self.sequence = 0

    def add(self, frame, captured_at, timestamp=None):
        timestamp = time.monotonic() if timestamp is None else timestamp
        self.sequence += 1
        item = (self.sequence, timestamp, captured_at, frame)
        self.frames.append(item)
        self.bytes += frame.nbytes
        while self.frames and (self.bytes > self.byte_limit or timestamp - self.frames[0][1] > self.seconds):
            self.bytes -= self.frames.popleft()[3].nbytes
        return item

    def snapshot(self):
        return tuple(self.frames)


def _linux_capture(options, channel, commands):
    # Imported only inside the spawned child; the parent can terminate any stalled native call.
    if __package__:
        from . import linux_camera
    else:
        import linux_camera
    cap = None
    try:
        device = options.get('device')
        if device is None and options.get('camera') is not None:
            device = '/dev/video' + str(options['camera'])
        candidates = [linux_camera.inspect(device)] if device else linux_camera.list_devices()
        if not candidates:
            raise RuntimeError('No RGB capture modes found. Install v4l-utils and run --list-cameras.')
        # The parent probes one candidate per child, so a hung first device cannot hide a working second one.
        profile = candidates[0]
        cap = cv2.VideoCapture(profile['device'], cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError('Could not open ' + profile['device'])
        profile = linux_camera.configure(cap, profile, options)
        started, count, failures = time.monotonic(), 0, 0
        cpu_started = time.process_time()
        next_control = 0
        while True:
            begin = time.monotonic()
            ok, frame = cap.read()
            end = time.monotonic()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 10:
                    raise RuntimeError('USB camera stopped delivering frames')
                time.sleep(.05)
                continue
            failures = 0
            height, width = frame.shape[:2]
            if (width, height) != (options['width'], options['height']):
                raise RuntimeError(f'Camera delivered {width}x{height}, requested {options["width"]}x{options["height"]}; '
                                   'select an advertised mode explicitly')
            count += 1
            profile['actual_size'] = [width, height]
            profile['measured_fps'] = count / max(.001, end - started)
            profile['read_and_decompress_ms'] = round((end - begin) * 1000, 2)
            profile['capture_count'] = count
            profile['capture_cpu_percent'] = 100 * (time.process_time() - cpu_started) / max(.001, end - started)
            if options.get('adaptive_exposure') and end >= next_control:
                try:
                    metric_time, mean = commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    applied = profile['applied_controls'].get('exposure_absolute', {})
                    if end - metric_time < .5 and applied.get('validated'):
                        controls = profile['controls']
                        value = linux_camera.exposure_step(controls['exposure_absolute'], applied['actual'], mean,
                                                           options.get('max_exposure_us', 2000))
                        if value != applied['actual']:
                            profile['applied_controls']['exposure_absolute'] = linux_camera.set_control(
                                profile['device'], 'exposure_absolute', value, controls)
                            profile['last_exposure_change_monotonic'] = end
                            next_control = end + .5
            replace_latest(channel, ('frame', frame, end, datetime.datetime.now().astimezone(), profile))
    except Exception as exc:
        # Errors must survive queue pressure. The parent continues consuming while this bounded put waits.
        try:
            channel.put(('error', str(exc)), timeout=1)
        except queue.Full:
            pass
    finally:
        if cap is not None:
            cap.release()


class ProcessCapture:
    """Linux driver ownership in a killable process, with at most three restarts."""
    def __init__(self, options, timeout=5, retries=3):
        self.options = dict(options)
        self.timeout, self.retries = timeout, retries
        self.process = self.channel = None
        self.commands = None
        self.profile = None
        self.restart_count = 0
        self.captured_monotonic = 0
        self.context = mp.get_context('spawn')
        self.candidates = []

    def _start(self):
        self.received_since_restart = 0
        self.channel = self.context.Queue(maxsize=2)
        self.commands = self.context.Queue(maxsize=1)
        self.process = self.context.Process(target=_linux_capture, args=(self.options, self.channel, self.commands), daemon=True)
        self.process.start()

    def _close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(1)
            self.process.close()
        if self.channel is not None:
            self.channel.cancel_join_thread()
            self.channel.close()
        if self.commands is not None:
            self.commands.cancel_join_thread()
            self.commands.close()
        self.process = self.channel = None
        self.commands = None

    def meter(self, observation):
        if self.commands is not None:
            replace_latest(self.commands, (observation['capture_monotonic'], observation['quality']['mean']))

    def __enter__(self):
        if not self.options.get('device') and self.options.get('camera') is None:
            if __package__:
                from .linux_camera import list_devices
            else:
                from linux_camera import list_devices
            self.candidates = [p['device'] for p in list_devices()]
            if not self.candidates:
                raise RuntimeError('No RGB USB webcam found; run --list-cameras')
            self.options['device'] = self.candidates.pop(0)
        self._start()
        return self

    def read(self, timeout=None):
        while True:
            try:
                message = self.channel.get(timeout=self.timeout if timeout is None else timeout)
                if message[0] == 'error':
                    raise RuntimeError(message[1])
                _, frame, stamp, captured, self.profile = message
                self.received_since_restart += 1
                self.profile['capture_queue_drops'] = max(0, self.profile['capture_count'] - self.received_since_restart)
                self.captured_monotonic = stamp
                self.candidates.clear()  # Once selected, reconnect only to this camera identity.
                return frame, (time.monotonic() - stamp) * 1000, captured
            except (queue.Empty, RuntimeError) as exc:
                self._close()
                if self.restart_count >= self.retries:
                    raise RuntimeError(f'Camera unavailable after {self.retries} restarts: {exc or "frame timeout"}') from exc
                self.restart_count += 1
                if self.candidates:
                    self.options['device'] = self.candidates.pop(0)
                print(f'Camera retry {self.restart_count}/{self.retries}: {exc or "frame timeout"}', file=sys.stderr, flush=True)
                self._start()

    def __exit__(self, *exc):
        self._close()


class DecoderWorker:
    def __init__(self, detector, interval=.1, diagnostics=False):
        self.detector, self.interval = detector, interval
        self.diagnostics = diagnostics
        self.input, self.results = queue.Queue(1), queue.Queue(2)
        self.stop = threading.Event()
        self.error = None
        self.observations = []
        self.stats = {}
        self.dropped_results = 0
        self.thread = threading.Thread(target=self._run, name='qr-decode', daemon=True)
        self.seen = deque(maxlen=128)
        self.settings = queue.Queue(1)
        self.minimum_stamp = None

    def set_lighting(self, enabled, since=None):
        replace_latest(self.settings, (bool(enabled), since))

    def __enter__(self):
        self.thread.start()
        return self

    def submit(self, history):
        replace_latest(self.input, history)

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    history = self.input.get(timeout=.1)
                except queue.Empty:
                    continue
                if not history:
                    continue
                try:
                    lighting, self.minimum_stamp = self.settings.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.detector.lighting = lighting
                    self.detector.reset()
                    self.seen.clear()
                if self.minimum_stamp is not None:
                    history = tuple(item for item in history if item[1] >= self.minimum_stamp)
                    if not history:
                        continue
                started = time.monotonic()
                latest = history[-1]
                candidates = [latest]
                # Score history using tracked ROIs where available. Re-localize the chosen old frame.
                def score(item):
                    image = item[3]
                    regions = [self.detector.tracker.crop(image, t.corners) for t in self.detector.tracker.tracks]
                    regions = [r for r in regions if r is not None and r.size]
                    if not regions:
                        regions = [image[::4, ::4]]
                    scores = []
                    for region in regions:
                        step = max(1, int(np.ceil(max(region.shape[:2]) / 160)))
                        region = region[::step, ::step]
                        gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                        low, high = np.percentile(gray, (5, 95))
                        scores.append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) * (high - low)))
                    return max(scores)
                # Score a bounded handful of old frames, after the current frame gets its turn.
                older = [item for item in history[:-1] if item[0] not in self.seen]
                all_observations = []
                processed = []
                for sequence, stamp, captured, frame in candidates:
                    if sequence in self.seen or self.stop.is_set():
                        continue
                    if processed and time.monotonic() - started >= self.detector.tracker.budget:
                        break
                    self.seen.append(sequence)
                    all_observations.extend(self.detector.observe(frame, sequence, stamp, captured))
                    processed.append({'frame_id': sequence, 'capture_monotonic': stamp,
                                      'latency_ms': (time.monotonic() - stamp) * 1000,
                                      'decode_ms': self.detector.last_stats.get('decode_ms', 0)})
                    if self.diagnostics:
                        if __package__:
                            from .tracking import quality
                        else:
                            from tracking import quality
                        step = max(1, int(np.ceil(max(frame.shape[:2]) / 320)))
                        processed[-1]['image_quality'] = quality(frame[::step, ::step])
                    if len(processed) == 1 and older and time.monotonic() - started < self.detector.tracker.budget:
                        # Downsample ROI scoring as well as full-frame scoring.
                        candidates.append(max(older[-3:], key=score))
                self.observations = all_observations
                completed = time.monotonic()
                for item in processed:
                    item['result_monotonic'] = completed
                    item['latency_ms'] = (completed - item['capture_monotonic']) * 1000
                self.stats = dict(self.detector.last_stats, worker_ms=(completed - started) * 1000,
                                  frame_age_ms=(completed - latest[1]) * 1000,
                                  processed_frames=processed, completed_monotonic=completed)
                self.stats['worker_budget_overrun_ms'] = max(0, self.stats['worker_ms'] - self.detector.tracker.budget * 1000)
                # Bound output and make drops observable, without stalling capture or preview.
                try:
                    self.results.put_nowait((all_observations, self.stats))
                except queue.Full:
                    self.dropped_results += 1
                self.stop.wait(max(0, self.interval - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(2)
        if self.thread.is_alive():
            print('Decoder is finishing a native call; pending results will be discarded.', file=sys.stderr)


class OutputWorker:
    def __init__(self, writer, save_frames=None, observations=False):
        self.writer, self.save_frames, self.log_observations = writer, save_frames, observations
        self.queue = queue.Queue(16)
        self.stop = threading.Event()
        self.error = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name='qr-output')

    def __enter__(self):
        self.thread.start()
        return self

    def submit(self, observations, frame=None):
        try:
            self.queue.put_nowait((observations, frame))
        except queue.Full:
            self.dropped += 1
            print('Output queue full; result batch dropped', file=sys.stderr, flush=True)

    def _run(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    observations, saved = self.queue.get(timeout=.1)
                except queue.Empty:
                    self.writer.poll_links()
                    continue
                for observation in observations:
                    if self.log_observations:
                        record = dict(observation, type='observation')
                        if record['captured_at'] is not None:
                            record['captured_at'] = record['captured_at'].isoformat()
                        self.writer._write(record)
                    if observation['confirmed'] and observation['payload']:
                        self.writer.record(observation['payload'], observation['captured_at'])
                self.writer.poll_links()
                if saved is not None and self.save_frames:
                    stamp, image = saved
                    path = Path(self.save_frames) / (stamp.strftime('%Y%m%d_%H%M%S_%f') + '.png')
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f'Could not save {path}')
        except Exception as exc:
            self.error = exc

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError('Output worker did not finish within three seconds')
        if self.error:
            raise RuntimeError(f'Output worker failed: {self.error}') from self.error


def overlay(frame, observations):
    image = frame.copy()
    for observation in observations:
        if observation['corners'] is None:
            continue
        points = np.array(observation['corners'], np.int32)
        color = (0, 200, 0) if observation['confirmed'] else ((255, 200, 0) if observation['payload'] else (0, 180, 255))
        cv2.polylines(image, [points], True, color, 2)
        q = observation['quality']
        status = 'confirmed' if observation['confirmed'] else ('decoded' if observation['payload'] else 'candidate')
        label = f"#{observation['track_id']} {status} contrast={q['contrast']:.0f} px/module~{q['module_px_estimate']:.1f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        x = min(max(0, int(points[:, 0].min())), max(0, image.shape[1] - label_size[0] - 4))
        y = min(max(label_size[1] + 4, int(points[:, 1].min()) - 5), image.shape[0] - baseline - 1)
        cv2.rectangle(image, (x, y - label_size[1] - 3), (x + label_size[0] + 3, y + baseline), (20,20,20), -1)
        cv2.putText(image, label, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return image


def system_metrics():
    result = {}
    try:
        result['temperature_c'] = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    except (OSError, ValueError):
        pass
    if sys.platform.startswith('linux'):
        try:
            result['throttled'] = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                                  text=True, timeout=1).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            import resource
            result['max_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            pass
    return result
