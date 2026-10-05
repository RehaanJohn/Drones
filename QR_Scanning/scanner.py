"""Terminal QR scanner: print scans and append UTF-8 JSON records to a file."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import urllib.request

import cv2

SCAN_INTERVAL = 0.10
DETECT_SCALE = 0.50
ROI_PADDING = 0.25
UPSCALE_FACTOR = 2.0
COOLDOWN_SECONDS = 2.0
MODEL_DIR = Path(__file__).resolve().parent / "wechat_models"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "qr_scans.jsonl"

MODEL_FILES = {
    "detect.prototxt":
        "https://raw.githubusercontent.com/"
        "WeChatCV/opencv_3rdparty/"
        "a8b69ccc738421293254aec5ddb38bd523503252/"
        "detect.prototxt",

    "detect.caffemodel":
        "https://raw.githubusercontent.com/"
        "WeChatCV/opencv_3rdparty/"
        "a8b69ccc738421293254aec5ddb38bd523503252/"
        "detect.caffemodel",

    "sr.prototxt":
        "https://raw.githubusercontent.com/"
        "WeChatCV/opencv_3rdparty/"
        "a8b69ccc738421293254aec5ddb38bd523503252/"
        "sr.prototxt",

    "sr.caffemodel":
        "https://raw.githubusercontent.com/"
        "WeChatCV/opencv_3rdparty/"
        "a8b69ccc738421293254aec5ddb38bd523503252/"
        "sr.caffemodel",
}

def ensure_models_exist():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        filepath = MODEL_DIR / filename
        if filepath.exists():
            continue
        print(f"Downloading {filename}...", flush=True)
        temporary = filepath.with_suffix(filepath.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(filepath)
        finally:
            temporary.unlink(missing_ok=True)


def create_detector():
    ensure_models_exist()
    detector = cv2.wechat_qrcode.WeChatQRCode(
        *(str(MODEL_DIR / name) for name in MODEL_FILES)
    )
    detector.setScaleFactor(DETECT_SCALE)
    return detector


def crop_qr_region(frame, points):

    if points is None or len(points) == 0:
        return None

    pts = points[0]

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x1 = int(max(0, x_coords.min()))
    y1 = int(max(0, y_coords.min()))

    x2 = int(min(frame.shape[1], x_coords.max()))
    y2 = int(min(frame.shape[0], y_coords.max()))

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    # Add padding
    pad_x = int(width * ROI_PADDING)
    pad_y = int(height * ROI_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    return frame[y1:y2, x1:x2]


def preprocess_for_fallback(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def fallback_decode(crop):

    if crop is None:
        return None

    # --------------------------------------------------------
    # Attempt 1: normal QR detector
    # --------------------------------------------------------

    qr_detector = cv2.QRCodeDetector()

    data, points, _ = qr_detector.detectAndDecode(crop)

    if data:
        return data

    # --------------------------------------------------------
    # Attempt 2: enlarged crop
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    data, points, _ = qr_detector.detectAndDecode(enlarged)

    if data:
        return data

    # --------------------------------------------------------
    # Attempt 3: CLAHE grayscale
    # --------------------------------------------------------

    enhanced = preprocess_for_fallback(enlarged)

    data, points, _ = qr_detector.detectAndDecode(enhanced)

    if data:
        return data

    return None


class ScanWriter:
    """Keep payloads intact, flush each scan, and suppress repeats per payload."""
    def __init__(self, stream, cooldown=COOLDOWN_SECONDS):
        self.stream = stream
        self.cooldown = cooldown
        self.last_seen = {}

    def record(self, payload):
        if not payload:
            return False
        now = time.monotonic()
        self.last_seen = {
            text: seen for text, seen in self.last_seen.items()
            if now - seen < self.cooldown
        }
        if payload in self.last_seen:
            return False
        record = {
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "data": payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)
        self.last_seen[payload] = now
        return True


def scan_frame(detector, frame, writer):
    results, points = detector.detectAndDecode(frame)
    for payload in results:
        writer.record(payload)
    # Retry each region that the detector found but could not decode.
    if points is not None:
        for index, region in enumerate(points):
            if index < len(results) and results[index]:
                continue
            crop = crop_qr_region(frame, [region])
            payload = fallback_decode(crop)
            writer.record(payload)


def open_camera(index):
    backend = {
        "darwin": cv2.CAP_AVFOUNDATION,
        "win32": cv2.CAP_DSHOW,
    }.get(sys.platform, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {index}. On macOS, allow camera access "
            "for your terminal in System Settings > Privacy & Security > Camera."
        )
    return cap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Append-only JSON Lines scan file")
    args = parser.parse_args(argv)
    cap = None
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            detector = create_detector()
            cap = open_camera(args.camera)
            writer = ScanWriter(stream)
            print(f"Scanning camera {args.camera}. Results: {args.output.resolve()}", flush=True)
            print("Press Ctrl+C to stop.", flush=True)
            failures = 0
            next_scan = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("Camera returned no frames after 30 attempts.")
                    time.sleep(SCAN_INTERVAL)
                    continue
                failures = 0
                now = time.monotonic()
                if now < next_scan:
                    continue
                next_scan = now + SCAN_INTERVAL
                scan_frame(detector, frame, writer)
    except KeyboardInterrupt:
        print("\nScanner stopped.", flush=True)
        return 0
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"Scanner error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    sys.exit(main())
