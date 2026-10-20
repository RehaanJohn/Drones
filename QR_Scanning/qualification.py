"""Optical sizing, lossless field recording and timestamp-aware QR qualification.

Ground truth is consumed ONLY by this evaluator, never by the scanner.
"""
import argparse
from collections import defaultdict
import datetime
import json
import math
from pathlib import Path
import platform
import queue
import time

import cv2
import numpy as np

if __package__:
    from .scanner import QRScanner, LegacyQRScanner
    from .pipeline import ProcessCapture, system_metrics
else:
    from scanner import QRScanner, LegacyQRScanner
    from pipeline import ProcessCapture, system_metrics


def optical_report(width_px, symbol_modules, qr_width_mm, distance_m, hfov=None,
                   outer_pixels=None, tilt_deg=0):
    if width_px <= 0 or qr_width_mm <= 0 or distance_m <= 0 or symbol_modules < 21 or symbol_modules > 177 or (symbol_modules - 21) % 4:
        raise ValueError('Positive dimensions/distance and a valid QR module count are required')
    if not 0 <= tilt_deg < 90 or (hfov is not None and not 0 < hfov < 180):
        raise ValueError('Invalid target tilt or horizontal field of view')
    if outer_pixels is None:
        if hfov is None:
            raise ValueError('Provide measured outer pixels or measured horizontal field of view')
        outer_pixels = width_px * qr_width_mm / 1000 / (2 * distance_m * math.tan(math.radians(hfov / 2)))
        outer_pixels *= math.cos(math.radians(tilt_deg))
    if outer_pixels <= 0:
        raise ValueError('Measured outer pixels must be positive')
    ppm = outer_pixels / (symbol_modules + 8)
    return {'distance_m': distance_m, 'compressed_outer_pixels': outer_pixels,
            'native_pixels_per_module': ppm, 'sampling_gate_pass': ppm >= 4,
            'four_pixel_distance_estimate_m': distance_m * ppm / 4,
            'note': 'QR width includes four-module quiet zone on each side. Measured pixels must use the compressed axis. '
                    'Distance estimate assumes constant geometry; it is not an empirically validated decode range.'}


def binomial_lower(successes, total, alpha=.05):
    """One-sided exact Clopper-Pearson bound, without a SciPy dependency."""
    if not total or not successes:
        return 0.0
    if successes == total:
        return alpha ** (1 / total)
    def tail(p):
        return sum(math.exp(math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
                            + k * math.log(p) + (total - k) * math.log1p(-p))
                   for k in range(successes, total + 1))
    low, high = 0., 1.
    for _ in range(60):
        mid = (low + high) / 2
        if tail(mid) < alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def summarize(trials):
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial['condition']].append(trial)
    summary = {}
    for condition, rows in grouped.items():
        positive = [row for row in rows if row['expected']]
        eligible = [row for row in positive if row.get('distance_m') == 10
                    and row.get('stable_hover') is True and 0 <= row.get('obliquity_deg', 91) <= 30]
        successes = sum(row['success'] for row in positive)
        wrong = sum(bool(row['incorrect_payloads']) for row in rows)
        lower = binomial_lower(successes, len(positive))
        eligible_successes = sum(row['success'] for row in eligible)
        eligible_rate = eligible_successes/len(eligible) if eligible else 0
        summary[condition] = {'positive_attempts': len(positive), 'successes': successes,
                              'negative_attempts': len(rows) - len(positive), 'incorrect_attempts': wrong,
                              'one_sided_95pct_lower_bound': lower,
                              'eligible_10m_attempts': len(eligible),
                              'eligible_10m_success_rate': eligible_rate,
                              'eligible_one_sided_95pct_lower_bound': binomial_lower(eligible_successes, len(eligible)),
                              'qualification_pass': len(eligible) >= 100 and eligible_rate >= .95 and wrong == 0,
                              'statistically_supported_95pct': len(eligible) >= 100
                              and binomial_lower(eligible_successes, len(eligible)) >= .95 and wrong == 0}
    return summary


def replay_trial(path, trial, legacy=False, budget_ms=150, interval=0, deadline_s=1):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError('Video has no valid frame rate; supply a usable recording')
    timestamps = trial.get('timestamps')
    start, end = float(trial['start_s']), float(trial.get('end_s', trial['start_s'] + 2))
    if start < 0 or end <= start:
        cap.release()
        raise ValueError('Trial end must follow a nonnegative start')
    engine = LegacyQRScanner() if legacy else QRScanner(budget_ms=budget_ms, modules=trial.get('modules', 29))
    confirmed, evidence, methods = {}, {}, set()
    index, available, processed, dropped = 0, start, 0, 0
    candidate_seen = False
    times = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    previous_timestamp = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if timestamps is not None and index >= len(timestamps):
                raise ValueError('Timestamp list is shorter than the recording')
            stamp = float(timestamps[index]) if timestamps is not None else index / fps
            if not math.isfinite(stamp) or stamp <= previous_timestamp:
                raise ValueError('Capture timestamps must be finite and strictly increasing')
            previous_timestamp = stamp
            index += 1
            if stamp < start:
                continue
            if stamp > end:
                break
            if stamp < available:
                dropped += 1
                continue
            begin = time.perf_counter()
            if legacy:
                payloads = engine.decode(frame)
                for text in payloads:
                    evidence.setdefault(text, set()).add(index)
                observations = [{'payload': text, 'confirmed': len(evidence[text]) >= 2, 'corners': None}
                                for text in payloads]
            else:
                observations = engine.observe(frame, index, stamp)
            elapsed = time.perf_counter() - begin
            available = stamp + max(interval, elapsed)
            processed += 1
            times.append(elapsed * 1000)
            candidate_seen |= any(o['corners'] is not None for o in observations)
            methods.update(engine.last_stats['methods'])
            for observation in observations:
                if observation['confirmed'] and observation['payload']:
                    confirmed.setdefault(observation['payload'], stamp + elapsed - start)
    finally:
        cap.release()
    expected = set(trial['expected'])
    wrong = sorted(set(confirmed) - expected)
    latencies = [confirmed.get(payload, math.inf) for payload in expected]
    success = processed > 0 and all(latency <= deadline_s for latency in latencies) and not wrong
    failure = None if success else ('incorrect_payload' if wrong else
               'no_processed_frames' if not processed else 'decode_or_confirmation' if candidate_seen else 'localization_or_image_quality')
    return {'id': trial['id'], 'condition': trial['condition'], 'expected': sorted(expected),
            'distance_m': trial.get('distance_m'), 'stable_hover': trial.get('stable_hover'),
            'obliquity_deg': trial.get('obliquity_deg'), 'confirmed_latency_s': confirmed,
            'success': success, 'incorrect_payloads': wrong, 'failure_stage': failure,
            'processed_frames': processed, 'skipped_for_compute': dropped,
            'decode_p50_ms': float(np.median(times)) if times else None,
            'decode_p95_ms': float(np.percentile(times, 95)) if times else None,
            'cpu_seconds': time.process_time() - cpu_start, 'evaluation_seconds': time.perf_counter() - wall_start,
            'methods': sorted(methods), 'system': system_metrics()}


def replay_realtime(path, trial, localizer='heuristic', confidence=.6, threads=1,
                    fast_budget_ms=33, budget_ms=150, deadline_s=1):
    """Paced replay through the actual two analysis processes, including contention.

    Model construction finishes before target entry. Lazy recovery initialization,
    IPC and scheduling are included. USB and HTTP are deliberately not simulated.
    """
    if __package__:
        from .realtime import ProcessDecoder
    else:
        from realtime import ProcessDecoder
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Recording must have a positive frame rate')
    start, end = float(trial['start_s']), float(trial.get('end_s', trial['start_s']+2))
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        cap.release()
        raise ValueError('Invalid trial interval')
    options = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   localizer=localizer, ncnn_confidence=confidence, ncnn_threads=threads,
                   qr_modules=trial.get('modules', 29), fast_budget_ms=fast_budget_ms, budget_ms=budget_ms)
    confirmed, methods, times, worker_ids = {}, set(), [], set()
    submitted = 0
    candidate_seen = False
    worker_stats = {}
    try:
        with ProcessDecoder(options) as worker:
            worker.wait_ready()
            base = time.monotonic()
            def consume():
                nonlocal candidate_seen
                worker.poll()
                if worker.error:
                    raise worker.error
                while True:
                    try:
                        observation = worker.confirmed.get_nowait()
                    except queue.Empty:
                        break
                    value = observation['result_monotonic']-base
                    confirmed[observation['payload']] = min(value, confirmed.get(observation['payload'], math.inf))
                while True:
                    try:
                        observations, stats = worker.results.get_nowait()
                    except queue.Empty:
                        break
                    worker_ids.add((stats['worker'], stats['frame_id']))
                    times.append(stats['decode_ms'])
                    methods.update(stats['methods'])
                    candidate_seen |= any(o['corners'] is not None for o in observations)
            previous, index = -1, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                stamps = trial.get('timestamps')
                if stamps is not None and index >= len(stamps):
                    raise ValueError('Timestamp list is shorter than recording')
                stamp = float(stamps[index]) if stamps is not None else index/fps
                index += 1
                if not math.isfinite(stamp) or stamp <= previous:
                    raise ValueError('Capture timestamps must be finite and increasing')
                previous = stamp
                if stamp < start:
                    continue
                if stamp > end:
                    break
                target = base+stamp-start
                while time.monotonic() < target:
                    consume()
                    time.sleep(min(.005, max(0, target-time.monotonic())))
                worker.submit(((index, target, None, frame),))
                submitted += 1
                consume()
            # Include completion of in-flight jobs; late results remain late.
            finish = max(base+end-start, time.monotonic())+.25
            while time.monotonic() < finish:
                consume()
                time.sleep(.005)
            worker_stats = dict(worker.workers_stats)
    finally:
        cap.release()
    expected = set(trial['expected'])
    wrong = sorted(set(confirmed)-expected)
    success = submitted > 0 and all(confirmed.get(p, math.inf) <= deadline_s for p in expected) and not wrong
    return dict(id=trial['id'], condition=trial['condition'], expected=sorted(expected),
                distance_m=trial.get('distance_m'), stable_hover=trial.get('stable_hover'),
                obliquity_deg=trial.get('obliquity_deg'), confirmed_latency_s=confirmed,
                success=success, incorrect_payloads=wrong, engine=localizer, workers=worker_stats,
                submitted_frames=submitted, observed_worker_updates=len(worker_ids),
                failure_stage=None if success else 'incorrect_payload' if wrong else
                              'decode_or_confirmation' if candidate_seen else 'localization_or_image_quality',
                decode_p50_ms=float(np.median(times)) if times else None,
                decode_p95_ms=float(np.percentile(times, 95)) if times else None,
                methods=sorted(methods), system=system_metrics())


def record(args):
    args.output.mkdir(parents=True, exist_ok=False)
    options = {'device': args.device, 'camera': None, 'width': args.width, 'height': args.height,
               'fps': args.fps, 'fourcc': args.fourcc, 'exposure_us': args.exposure_us}
    writer, timestamps, profile = None, [], None
    try:
        with ProcessCapture(options) as capture:
            first = None
            while True:
                frame, age, captured = capture.read()
                stamp = capture.captured_monotonic
                if first is None:
                    first = stamp
                    writer = cv2.VideoWriter(str(args.output / 'capture.mkv'), cv2.VideoWriter_fourcc(*'FFV1'),
                                             args.fps, (args.width, args.height))
                    if not writer.isOpened():
                        raise RuntimeError('Lossless FFV1 recording unavailable in this OpenCV build')
                if stamp - first > args.seconds:
                    break
                writer.write(frame)
                timestamps.append(stamp - first)
                profile = capture.profile
    finally:
        if writer is not None:
            writer.release()
        (args.output / 'timestamps.json').write_text(json.dumps(timestamps), encoding='utf-8')
        (args.output / 'camera.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
    print(f'Recorded {len(timestamps)} frames. Add independently labelled trials to a manifest; timestamps are read-completion times.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    optical = sub.add_parser('optical')
    optical.add_argument('--width-px', type=int, default=1920)
    optical.add_argument('--modules', type=int, required=True)
    optical.add_argument('--qr-width-mm', type=float, required=True)
    optical.add_argument('--distance-m', type=float, default=10)
    optical.add_argument('--tilt-deg', type=float, default=0)
    geometry = optical.add_mutually_exclusive_group(required=True)
    geometry.add_argument('--hfov', type=float)
    geometry.add_argument('--outer-pixels', type=float)
    replay = sub.add_parser('replay')
    replay.add_argument('manifest', type=Path)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--compare-legacy', action='store_true')
    replay.add_argument('--budget-ms', type=float, default=150)
    replay.add_argument('--engine', choices=('current', 'heuristic', 'hybrid'), default='heuristic')
    replay.add_argument('--compare-all', action='store_true', help='Paced heuristic/hybrid workers plus current baseline')
    replay.add_argument('--interval', type=float, default=.1, help='Current baseline scan interval only')
    replay.add_argument('--deadline-s', type=float, default=1)
    replay.add_argument('--fast-budget-ms', type=float, default=33)
    replay.add_argument('--ncnn-confidence', type=float, default=.6)
    replay.add_argument('--ncnn-threads', type=int, default=1)
    recording = sub.add_parser('record')
    recording.add_argument('--device')
    recording.add_argument('--output', type=Path, required=True)
    recording.add_argument('--seconds', type=float, default=10)
    recording.add_argument('--width', type=int, default=1920)
    recording.add_argument('--height', type=int, default=1080)
    recording.add_argument('--fps', type=float, default=30)
    recording.add_argument('--fourcc', default='MJPG')
    recording.add_argument('--exposure-us', type=int)
    args = parser.parse_args(argv)
    if args.command == 'optical':
        values = vars(args).copy()
        values.pop('command')
        values['symbol_modules'] = values.pop('modules')
        print(json.dumps(optical_report(**values), indent=2))
    elif args.command == 'record':
        if not math.isfinite(args.seconds) or args.seconds <= 0 or args.fps <= 0:
            parser.error('Recording duration and FPS must be positive')
        record(args)
    else:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        rows, old = [], []
        comparisons = {name: [] for name in ('current', 'heuristic', 'hybrid')}
        if (not all(math.isfinite(v) and v > 0 for v in (args.deadline_s, args.budget_ms, args.fast_budget_ms))
                or not math.isfinite(args.interval) or args.interval < 0 or not 0 < args.ncnn_confidence < 1 or args.ncnn_threads < 1):
            parser.error('Invalid replay timing or NCNN settings')
        ids = set()
        for trial in manifest['trials']:
            if trial['id'] in ids:
                raise ValueError('Each independently labelled trial needs a unique ID')
            ids.add(trial['id'])
            trial = dict(trial)
            if 'timestamps_file' in trial:
                trial['timestamps'] = json.loads((args.manifest.parent / trial['timestamps_file']).read_text())
            path = args.manifest.parent / trial['video']
            for name in comparisons if args.compare_all else (args.engine,):
                if name == 'current':
                    row = replay_trial(path, trial, budget_ms=args.budget_ms, interval=args.interval, deadline_s=args.deadline_s)
                else:
                    row = replay_realtime(path, trial, name, args.ncnn_confidence, args.ncnn_threads,
                                          args.fast_budget_ms, args.budget_ms, args.deadline_s)
                comparisons[name].append(row)
            rows.append(comparisons[args.engine][-1])
            if args.compare_legacy:
                old.append(replay_trial(path, trial, legacy=True, interval=args.interval, deadline_s=args.deadline_s))
            print(f'{trial["id"]}: success={rows[-1]["success"]}, latency={rows[-1]["confirmed_latency_s"]}', flush=True)
        report = {'platform': platform.platform(), 'opencv': cv2.__version__, 'results': rows,
                  'summary': summarize(rows), 'legacy_results': old, 'legacy_summary': summarize(old),
                  'comparisons': {name: dict(results=items, summary=summarize(items)) for name, items in comparisons.items() if items},
                  'target': dict(distance_m=10, correct_fraction=.95, deadline_s=args.deadline_s, trials_per_condition=100),
                  'replay_gate_pass': all(summarize(rows).get(category, {}).get('qualification_pass', False)
                                            for category in ('sun', 'shade', 'backlight', 'mixed_shadows', 'low_light')),
                  'field_qualified': False,
                  'limitations': 'Replay measures decoder compute and skips unavailable frames. It excludes live USB/preview/output latency '
                                 'and does not prove independent trials, capture timing accuracy, or 10m geometry. '
                                 'Missing candidates cannot alone distinguish poor optics from localization failure.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
