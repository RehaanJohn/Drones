"""Finite capture-only measurements on the same Linux path used by the scanner."""
import json
from pathlib import Path
import time
import cv2
import numpy as np

if __package__:
    from .pipeline import ProcessCapture, live_overlay, runtime_inventory, system_metrics
else:
    from pipeline import ProcessCapture, live_overlay, runtime_inventory, system_metrics


def run(options, seconds=30, output=None, preview=False):
    samples = []
    profile = None
    first = None
    shown = 0
    try:
        with ProcessCapture(options) as capture:
            while True:
                frame, age, captured = capture.read()
                now = time.monotonic()
                if first is None:
                    first = now
                profile = dict(capture.profile or {})
                samples.append({key: profile.get(key) for key in (
                    'capture_count', 'acquisition_ms', 'jpeg_decode_ms', 'read_and_decompress_ms',
                    'shared_memory_write_ms', 'shared_memory_read_ms', 'queue_residence_ms')})
                if preview:
                    cv2.imshow('QR capture benchmark - Q / Esc', live_overlay(frame))
                    shown += 1
                    if cv2.waitKey(1) & 0xff in (27, ord('q')):
                        break
                if now-first >= seconds:
                    break
    finally:
        if preview:
            cv2.destroyAllWindows()
    elapsed = max(.001, time.monotonic()-(first or time.monotonic()))
    timings = {}
    for key in samples[0] if samples else ():
        if key.endswith('_ms'):
            values = [s[key] for s in samples if s[key] is not None]
            timings[key] = dict(p50=float(np.median(values)), p95=float(np.percentile(values, 95))) if values else None
    report = dict(type='capture_benchmark', seconds=elapsed, consumed_frames=len(samples),
                  delivered_fps=max(0, len(samples)-1)/elapsed, preview_fps=shown/elapsed,
                  timings=timings, camera=profile, runtime=runtime_inventory(), system=system_metrics(),
                  limitations='Samples cover consumed frames. Acquisition includes driver wait/USB. '
                              'Color capture does not expose separate JPEG timing. Driver exposure timestamps are unavailable through this OpenCV path.')
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main(argv=None):
    import argparse
    if __package__:
        from . import linux_camera
    else:
        import linux_camera
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for mode/exposure comparison reports')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--rates', nargs='+', type=float, default=[30, 60])
    parser.add_argument('--exposures-us', nargs='+', type=int, default=[1500, 3000, 6000, 10000])
    args = parser.parse_args(argv)
    if (not np.isfinite(args.seconds) or args.seconds <= 0 or min(args.width,args.height) <= 0
            or any(not np.isfinite(rate) or rate <= 0 for rate in args.rates) or min(args.exposures_us) <= 0):
        parser.error('Dimensions, duration, rates and exposures must be positive and finite')
    profile = linux_camera.inspect(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'inventory.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    results = []
    try:
        for rate in args.rates:
            advertised = any(m['fourcc']=='MJPG' and m['width']==args.width and m['height']==args.height
                             and abs(m['fps']-rate)<.1 for m in profile['modes'])
            if not advertised:
                results.append(dict(fps=rate,status='skipped',reason='Exact MJPG mode not advertised'))
                continue
            for exposure in args.exposures_us:
                name = f'{rate:g}fps_{exposure}us.json'
                options = dict(device=args.device,width=args.width,height=args.height,fps=rate,
                               fourcc='MJPG',gray_capture=True,exposure_us=exposure)
                try:
                    run(options,args.seconds,args.output/name)
                    results.append(dict(fps=rate,exposure_us=exposure,status='recorded',report=name))
                except RuntimeError as exc:
                    results.append(dict(fps=rate,exposure_us=exposure,status='failed',error=str(exc)))
    finally:
        (args.output/'matrix.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
