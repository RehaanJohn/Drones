"""Live diagnostic rendering and labelled, ground-truth-aware field trials.

Measurements describe observed samples, never a guaranteed optical range.
Expected payloads are consumed here only, never by the decoder.
"""
from collections import deque
import csv
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .pipeline import overlay
    from .tracking import quality, variants
else:
    from pipeline import overlay
    from tracking import quality, variants


DEFAULT_CONTROLS = dict(brightness=100, blur=0, shadow=0, glare=0)


def thumbnail(image, width=200, height=112):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1,round(image.shape[1]*scale)), max(1,round(image.shape[0]*scale))))
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    target = np.full((height,width,3),22,np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    target[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
    return target


def simulate(frame, controls):
    """Optional synthetic stress; identity preserves native input and avoids a copy."""
    if controls == DEFAULT_CONTROLS:
        return frame
    image = frame.astype(np.float32)
    image *= controls['brightness'] / 100
    if controls['shadow']:
        ramp = np.linspace(1 - controls['shadow'] / 100, 1, frame.shape[1], dtype=np.float32)
        image *= ramp[None, :, None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if controls['blur']:
        image = cv2.GaussianBlur(image, (0, 0), controls['blur'] / 2)
    if controls['glare']:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.float32)
        cv2.circle(mask, (width // 2, height // 2), max(4, min(height, width) // 8),
                   controls['glare'] / 100, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, min(height, width) / 40))[:, :, None]
        image = np.clip(image * (1 - mask) + 255 * mask, 0, 255).astype(np.uint8)
    return image


class DiagnosticSession:
    def __init__(self, directory, expected=None, condition='baseline', distance=None,
                 seconds=10, modules=29, lighting=True, options=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / 'samples.jsonl').open('a', encoding='utf-8')
        self.expected, self.condition, self.distance = expected, condition, distance
        self.seconds, self.modules, self.lighting = seconds, modules, lighting
        option_values = options or {}
        self.configuration = {key: option_values.get(key) for key in
                              ('camera', 'camera_name', 'device', 'backend', 'width', 'height', 'fps',
                               'fourcc', 'exposure_us', 'adaptive_exposure', 'auto_exposure', 'exposure',
                               'gain', 'autofocus', 'focus', 'budget_ms', 'tile_size', 'tiles_per_frame',
                               'interval', 'no_zxing')}
        self.configuration['qr_modules'] = modules
        self.controls = dict(DEFAULT_CONTROLS)
        self.trial = None
        self.last_trial = None
        self.results = []
        self.recent = deque(maxlen=100)
        self.started = False
        self.raw = self.frame = None
        self.metrics = {}
        self.observations = []
        self.minimum_ppm = None
        self.minimum_edge = None
        self.farthest = None
        self.last_metrics = 0
        self.corrected_thumb = None
        self.thumb_method = ''
        self.last_thumb = 0
        self.notice = 'First trial starts automatically; change conditions then press T.'
        metadata = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
                        expected_payload=expected, condition=condition, distance_m=distance,
                        trial_seconds=seconds, qr_modules=modules, options=options or {},
                        note='Distance is operator measured. Rates use distinct processed frames, not all camera frames. '
                             'Synthetic effects are stress tests, not a calibrated model of real lighting.')
        (self.directory / 'session.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def start_trial(self, now):
        if self.trial is not None:
            self.finish('restarted')
        self.trial = dict(start=now, end=now + self.seconds, rows=[], captured=0,
                          controls=dict(self.controls), lighting=self.lighting,
                          configuration=dict(self.configuration, actual_size=list(self.raw.shape[1::-1])
                                             if self.raw is not None else None))
        self.notice = 'Trial running: keep distance/conditions constant.'

    def prepare(self, frame, now, controls):
        self.raw = frame
        if controls != self.controls:
            if self.trial is not None:
                self.finish('controls_changed')
            self.controls = dict(controls)
            self.notice = 'Controls changed. Press T to measure this condition.'
        if not self.started:
            self.started = True
            self.start_trial(now)
        self.frame = simulate(frame, self.controls)
        if now - self.last_metrics >= .25:
            # HUD quality is measured on a bounded view to avoid starving capture.
            step = max(1, int(np.ceil(max(frame.shape[:2]) / 320)))
            self.metrics = quality(self.frame[::step, ::step], self.modules)
            self.last_metrics = now
        if self.trial is not None:
            if now <= self.trial['end']:
                self.trial['captured'] += 1
            elif now > self.trial['end'] + 3:
                self.finish('complete')
        return self.frame

    def consume(self, observations, stats):
        self.observations = observations
        for item in stats.get('processed_frames', []):
            selected = [o for o in observations if o['frame_id'] == item['frame_id']]
            payloads = sorted({o['payload'] for o in selected if o['payload']})
            confirmed = sorted({o['payload'] for o in selected if o['payload'] and o['confirmed']})
            matched = bool(payloads) if self.expected is None else self.expected in payloads
            correct = bool(confirmed) if self.expected is None else self.expected in confirmed
            incorrect = [] if self.expected is None else [p for p in payloads if p != self.expected]
            candidates = [o for o in selected if o['corners'] is not None]
            successful = [o for o in candidates if o['payload'] and
                          (self.expected is None or o['payload'] == self.expected)]
            ppm = min((o['quality']['module_px_estimate'] for o in successful), default=None)
            edges = [float(min(np.linalg.norm(np.asarray(o['corners']) -
                                              np.roll(np.asarray(o['corners']), -1, axis=0), axis=1)))
                     for o in successful]
            row = dict(item, candidate_count=len({o['track_id'] for o in candidates}), payloads=payloads,
                       confirmed_payloads=confirmed, decoded=matched, confirmed=correct,
                       incorrect_payloads=incorrect, minimum_module_px=ppm,
                       minimum_symbol_edge_px=min(edges, default=None),
                       qr_quality=[o['quality'] for o in candidates])
            self.recent.append(row)
            if ppm is not None and (self.minimum_ppm is None or ppm < self.minimum_ppm):
                self.minimum_ppm, self.minimum_edge = ppm, min(edges)
            trial = self.trial
            if trial and trial['start'] <= item['capture_monotonic'] <= trial['end']:
                trial['rows'].append(row)
                record = dict(row, condition=self.condition, distance_m=self.distance,
                              controls=trial['controls'], lighting=trial['lighting'],
                              trial_id=len(self.results) + 1)
                self.stream.write(json.dumps(record) + '\n')
                self.stream.flush()

    def finish(self, status):
        trial = self.trial
        if trial is None:
            return
        rows = trial['rows']
        total = len(rows)
        latencies = [r['latency_ms'] for r in rows]
        # Confirmation is available after decoding, not when its source frame was captured.
        times = [r.get('result_monotonic', r['capture_monotonic'] + r['latency_ms'] / 1000) - trial['start']
                 for r in rows if r['confirmed']]
        result = dict(trial_id=len(self.results) + 1, condition=self.condition, distance_m=self.distance,
                      status=status, ground_truth_known=self.expected is not None,
                      expected_payload=self.expected, duration_s=self.seconds, controls=trial['controls'],
                      lighting=trial['lighting'], configuration=trial['configuration'],
                      captured_frames=trial['captured'], processed_frames=total,
                      decoded_frames=sum(r['decoded'] for r in rows),
                      confirmed_frames=sum(r['confirmed'] for r in rows),
                      decoded_rate=sum(r['decoded'] for r in rows) / total if total else None,
                      confirmed_rate=sum(r['confirmed'] for r in rows) / total if total else None,
                      incorrect_payloads=sorted({p for r in rows for p in r['incorrect_payloads']}),
                      first_confirmation_s=min(times, default=None),
                      confirmation_within_2s=bool(times and min(times) <= 2),
                      latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
                      minimum_module_px=min((r['minimum_module_px'] for r in rows
                                             if r['minimum_module_px'] is not None), default=None))
        self.results.append(result)
        if status == 'complete' and times and self.distance is not None:
            self.farthest = max(self.farthest or 0, self.distance)
        self.last_trial, self.trial = result, None
        with (self.directory / 'trials.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result) + '\n')
        fields = [k for k in result if k not in ('controls', 'incorrect_payloads', 'configuration')]
        fields += ['brightness', 'blur', 'shadow', 'glare']
        with (self.directory / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.results:
                writer.writerow({k: row.get(k, row['controls'].get(k)) for k in fields})
        summary = dict(trials=len(self.results), minimum_decoded_module_px=self.minimum_ppm,
                       minimum_decoded_edge_px=self.minimum_edge, farthest_confirmed_distance_m=self.farthest,
                       note='Observed limits in this session; not a guaranteed maximum range.')
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        self.notice = f'Trial {result["trial_id"]}: {status}. Press T for another trial.'
        print('Diagnostic trial ' + json.dumps(result), flush=True)

    def key(self, key, decoder, now):
        if key in (ord('t'), ord('T')):
            self.start_trial(now)
            decoder.set_lighting(self.lighting, since=now)
        elif key in (ord('l'), ord('L')):
            self.finish('lighting_changed')
            self.lighting = not self.lighting
            decoder.set_lighting(self.lighting, since=now)
            self.notice = 'Lighting recovery changed. Press T for a comparison trial.'
        elif key in (ord('s'), ord('S')):
            stem = datetime.datetime.now().strftime('%H%M%S_%f')
            for name, image in [('raw', self.raw), ('scan', self.frame)]:
                if image is not None and not cv2.imwrite(str(self.directory / f'{stem}_{name}.png'), image):
                    raise OSError('Could not save diagnostic snapshot')
            self.notice = 'Raw and simulated input snapshots saved.'

    def render(self, frame, stats, now):
        observations = [o for o in self.observations if 0 <= now - o['capture_monotonic'] <= .75]
        # One box per track; a decoded observation takes precedence over its candidate entry.
        by_track = {}
        for o in observations:
            previous = by_track.get(o['track_id'])
            rank = lambda item: (item['capture_monotonic'], bool(item['payload']), item['confirmed'])
            if previous is None or rank(o) > rank(previous):
                by_track[o['track_id']] = o
        image = overlay(frame, list(by_track.values()))
        scale = min(1, 960 / image.shape[1])
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        panel_width, height = 460, max(820, image.shape[0])
        canvas = np.full((height, image.shape[1] + panel_width, 3), 22, np.uint8)
        canvas[:image.shape[0], :image.shape[1]] = image
        x, y = image.shape[1] + 14, 25
        def line(text, color=(225, 225, 225)):
            nonlocal y
            cv2.putText(canvas, text[:61], (x, y), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
            y += 23
        line('DIAGNOSTIC FIELD TEST', (255, 210, 70))
        line(f'{frame.shape[1]}x{frame.shape[0]} | camera {stats.get("delivered_fps", 0):.1f} fps')
        age = stats.get('frame_age_ms', 0) + max(0, now - stats.get('completed_monotonic', now))*1000
        line(f'Worker {stats.get("worker_ms", 0):.0f} ms | result age {age:.0f} ms')
        line(f'Decode {stats.get("decode_ms", 0):.0f} / {stats.get("budget_ms", 0):.0f} ms (soft budget)')
        if stats.get('worker_budget_overrun_ms', stats.get('budget_overrun_ms', 0)) > 0:
            line('Processing exceeded budget; frame results may lag.', (0, 170, 255))
        q = self.metrics
        line(f'Brightness {q.get("mean", 0):.0f}/255 | contrast {q.get("contrast", 0):.0f}')
        line(f'Sharpness {q.get("sharpness", 0):.0f} | light spread {q.get("illumination_range", 0):.0f}')
        line(f'Clipped bright {100*q.get("bright_fraction", 0):.1f}% / dark {100*q.get("dark_fraction", 0):.1f}%')
        line('Sampled quality; clipped QR black/white is not glare proof.')
        line(f'Lighting recovery: {"ON" if self.lighting else "OFF"} (L toggles)')
        synthetic = self.controls != DEFAULT_CONTROLS
        line('SIMULATED INPUT' if synthetic else 'REAL CAMERA INPUT', (0, 170, 255) if synthetic else (80, 230, 80))
        line(f'Condition: {self.condition} | distance: {self.distance if self.distance else "unset"} m')
        line(f'Module count: {self.modules} (set actual --qr-modules)')
        line('Green=confirmed | cyan=decoded | yellow=candidate')
        line('Boxes expire after 0.75 s; localization is not decoding.')
        best = next((o for o in by_track.values() if o['corners'] is not None), None)
        if best:
            line(f'QR #{best["track_id"]}: {best["quality"]["module_px_estimate"]:.2f} native px/module')
            line(f'QR contrast {best["quality"]["contrast"]:.0f} | sharpness {best["quality"]["sharpness"]:.0f}')
        else:
            line('No fresh localized QR; move closer or improve light.')
        line(f'Smallest decoded: {self.minimum_ppm:.2f} px/module' if self.minimum_ppm is not None else 'Smallest decoded: no successful samples yet')
        line('Observed pixel limit; maximum metres is not inferred.')
        line(f'Farthest confirmed trial: {self.farthest:g} m' if self.farthest else 'Farthest measured confirmation: no completed trial')
        trial = self.trial
        rows = trial['rows'] if trial else self.last_trial or {}
        if trial:
            line(f'Trial remaining {max(0, trial["end"]-now):.1f}s | samples {len(rows)}')
            rate = sum(r['decoded'] for r in rows) / len(rows) if rows else None
        else:
            rate = rows.get('decoded_rate') if isinstance(rows, dict) else None
        label = 'Correct decode' if self.expected is not None else 'Decode presence (truth unset)'
        line(f'{label}: {100*rate:.1f}%' if rate is not None else f'{label}: no samples')
        line('T=new timed trial | L=recovery | S=snapshot | Q=quit')
        line(self.notice, (255, 210, 70))
        # Show the untouched camera feed alongside a suggested candidate correction.
        thumb_width = 200
        raw = thumbnail(self.raw)
        canvas[y+20:y+132, x:x+thumb_width] = raw
        cv2.putText(canvas, 'Untouched camera', (x, y+14), cv2.FONT_HERSHEY_SIMPLEX, .4, (225,225,225), 1)
        if best:
            corners = np.asarray(best['corners'])
            left, top = np.maximum(corners.min(axis=0).astype(int), 0)
            right, bottom = np.minimum(corners.max(axis=0).astype(int)+1, [frame.shape[1], frame.shape[0]])
            crop = frame[top:bottom, left:right]
            if right > left and bottom > top and min(crop.shape[:2]) >= 2:
                if now - self.last_thumb >= .5 or self.corrected_thumb is None:
                    scale = min(1, 320 / max(crop.shape[:2]))
                    crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                    self.thumb_method, corrected = variants(crop, quality(crop, self.modules), self.modules)[0]
                    self.corrected_thumb = thumbnail(corrected)
                    self.last_thumb = now
                canvas[y+20:y+132, x+220:x+420] = self.corrected_thumb
                cv2.putText(canvas, 'Suggested: '+self.thumb_method, (x+220,y+14), cv2.FONT_HERSHEY_SIMPLEX,.4,(225,225,225),1)
        return canvas

    def close(self):
        self.finish('interrupted')
        self.stream.close()


def report(directory):
    """Compare completed field trials across runs, grouping identical test settings."""
    from collections import defaultdict
    groups = defaultdict(list)
    for path in Path(directory).rglob('trials.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['status'] != 'complete':
                continue
            key = (row['condition'], row['distance_m'], row['lighting'],
                   json.dumps(row['controls'], sort_keys=True), row['expected_payload'],
                   json.dumps(row.get('configuration', {}), sort_keys=True), row['duration_s'])
            groups[key].append(row)
    rows = []
    for (condition, distance, lighting, controls, expected, configuration, seconds), trials in groups.items():
        samples = sum(t['processed_frames'] for t in trials)
        decoded = sum(t['decoded_frames'] for t in trials)
        confirmed = sum(t['confirmed_frames'] for t in trials)
        success = sum(t['confirmation_within_2s'] for t in trials)
        if __package__:
            from .qualification import binomial_lower
        else:
            from qualification import binomial_lower
        rows.append(dict(condition=condition, distance_m=distance, lighting=lighting,
                         controls=json.loads(controls), expected_payload=expected,
                         configuration=json.loads(configuration), trial_seconds=seconds,
                         trials=len(trials), samples=samples,
                         correct_decode_rate=decoded / samples if samples and expected is not None else None,
                         confirmed_frame_rate=confirmed / samples if samples else None,
                         within_2s_trials=success, within_2s_trial_rate=success / len(trials),
                         within_2s_lower_95=binomial_lower(success, len(trials)) if expected is not None else None,
                         incorrect_payloads=sorted({p for t in trials for p in t['incorrect_payloads']})))
    target = Path(directory) / 'comparison.json'
    target.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))
    print(f'Comparison saved to {target.resolve()}')
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Diagnostic run directory (searched recursively)')
    report(parser.parse_args().report)
