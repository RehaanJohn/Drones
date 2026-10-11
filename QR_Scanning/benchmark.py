"""Compare the former 0.5-scale WeChat path with the new scanner, without a camera.

Synthetic frames probe pixel/lighting limits; they do NOT establish outdoor range.
Use --input-dir with saved native camera frames for a meaningful field comparison.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from scanner import QRScanner, create_detector


def synthetic_cases():
    payload = "DRONE-01"
    qr = cv2.QRCodeEncoder_create().encode(payload)
    ys, xs = np.where(qr == 0)
    # OpenCV's encoder includes its own border; normalize to exactly 4 modules.
    symbol = qr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    qr = cv2.copyMakeBorder(symbol, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    source = cv2.resize(qr, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
    rng = np.random.default_rng(42)
    base = np.full((1080, 1920, 3), 165, np.uint8)
    for _ in range(90):
        x, y = rng.integers([0, 0], [1920, 1080])
        color = int(rng.integers(45, 210))
        cv2.rectangle(base, (int(x), int(y)), (int(x + 40), int(y + 25)), (color,) * 3, -1)
    for size in (41, 60, 90, 120):
        patch = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        for condition in ("clean", "shadow", "low_contrast", "color_cast", "defocus", "motion", "glare", "perspective"):
            data = patch.copy()
            if condition == "shadow":
                illumination = np.linspace(0.18, 1.0, size, dtype=np.float32)[None, :]
                illumination[:, :size // 2] *= 0.45
                data *= illumination
            elif condition == "low_contrast":
                data = 150 + data * 0.12
            elif condition == "defocus":
                data = cv2.GaussianBlur(data, (0, 0), 0.9)
            elif condition == "motion":
                kernel = np.zeros((5, 5), np.float32)
                kernel[2, :] = 0.2
                data = cv2.filter2D(data, -1, kernel)
            elif condition == "glare":
                yy, xx = np.mgrid[:size, :size]
                glow = np.exp(-((xx - size * 0.28) ** 2 + (yy - size * 0.35) ** 2) / (size * 0.20) ** 2)
                data += glow * 350  # Deliberately loses information to clipping.
            elif condition == "perspective":
                src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
                dst = np.float32([[size * 0.18, size * 0.08], [size * 0.86, 0], [size - 1, size - 1], [0, size * 0.85]])
                data = cv2.warpPerspective(data, cv2.getPerspectiveTransform(src, dst),
                                           (size, size), borderValue=255)
            data += rng.normal(0, 1.5, data.shape).astype(np.float32)
            data = cv2.cvtColor(np.clip(data, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if condition == "color_cast":
                data = np.clip(data.astype(np.float32) * [0.3, 0.65, 1.0] + [10, 5, 0], 0, 255).astype(np.uint8)
            frame = base.copy()
            # Deliberately cross a 640px tile boundary, away from frame center.
            frame[475:475 + size, 610:610 + size] = data
            yield f"{size}px_{condition}", frame, [payload], {
                "outer_qr_pixels": size, "modules": int(symbol.shape[0]),
                "pixels_per_module": round(size / qr.shape[0], 2), "condition": condition,
            }
    yield "no_qr_clutter", base, [], {"condition": "negative_control"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Directory of raw PNG/JPG frames; otherwise use synthetic cases")
    parser.add_argument("--expected", help="Expected payload for every input frame (omit for manual inspection)")
    parser.add_argument("--passes", type=int, default=8, help="Maximum rolling scans per still image")
    parser.add_argument("--no-zxing", action="store_true")
    parser.add_argument("--no-lighting", action="store_true")
    parser.add_argument("--save-images", action="store_true", help="Save synthetic test images alongside results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "benchmark_output/results.json")
    args = parser.parse_args(argv)
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.input_dir:
        paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not paths:
            parser.error("No PNG/JPG images found")
        cases = ((p.name, cv2.imread(str(p)), [args.expected] if args.expected is not None else None,
                  {"condition": "camera_frame"}) for p in paths)
    else:
        cases = synthetic_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = create_detector()
    baseline.setScaleFactor(0.5)
    improved = QRScanner(use_zxing=not args.no_zxing, lighting=not args.no_lighting)
    # Warm up native libraries/models before recording timings.
    blank = np.full((256, 256, 3), 180, np.uint8)
    baseline.detectAndDecode(blank)
    improved.decode(blank)
    records = []
    for name, frame, expected, details in cases:
        if frame is None:
            raise RuntimeError(f"Cannot read image {name}")
        if args.save_images:
            if not cv2.imwrite(str(args.output.parent / (name + ".png")), frame):
                raise RuntimeError(f"Cannot save image {name}")
        started = time.perf_counter()
        old_payloads, _ = baseline.detectAndDecode(frame)
        old_ms = (time.perf_counter() - started) * 1000
        improved.reset()
        payloads = set()
        scan_times = []
        methods = []
        started = time.perf_counter()
        for _ in range(args.passes):
            payloads.update(improved.decode(frame))
            scan_times.append(improved.last_stats["decode_ms"])
            methods.extend(improved.last_stats["methods"])
            if (expected and set(expected).issubset(payloads)) or (expected is None and payloads):
                break
        record = dict(details, name=name, expected=expected,
                      baseline_payloads=list(old_payloads), improved_payloads=sorted(payloads),
                      baseline_ms=round(old_ms, 1), improved_total_ms=round((time.perf_counter() - started) * 1000, 1),
                      scan_ms=scan_times, passes=len(scan_times), methods=sorted(set(methods)))
        if expected is not None:
            record["baseline_ok"] = set(old_payloads) == set(expected)
            record["improved_ok"] = payloads == set(expected)
        records.append(record)
        print(f"{name}: old={record.get('baseline_ok', list(old_payloads))}, "
              f"new={record.get('improved_ok', sorted(payloads))}, "
              f"old_ms={old_ms:.1f}, new_total_ms={record['improved_total_ms']}, passes={len(scan_times)}", flush=True)
    report = {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "opencv": cv2.__version__, "zxing": version('zxing-cpp') if improved.use_zxing else None},
        "settings": {"max_passes": args.passes, "tiles_per_frame": improved.tiles_per_frame,
                     "tile_size": improved.tile_size, "lighting": improved.lighting},
        "note": "Synthetic success is not a distance guarantee. Baseline gets one deterministic still-image pass; "
                "new scanner gets up to max_passes for its rolling tile/lighting schedule. "
                "Timings exclude capture, model loading, scan interval waits, and file I/O.",
        "results": records,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    measured = [r for r in records if r['expected']]
    if measured:
        print(f"Positive frames: old {sum(r['baseline_ok'] for r in measured)}/{len(measured)}, "
              f"new {sum(r['improved_ok'] for r in measured)}/{len(measured)}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
