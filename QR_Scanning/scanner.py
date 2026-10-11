"""QR scanner with a live preview, readable scan logs, and background link lookup."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2
import numpy as np

if __package__:
    from .link_resolver import LinkResolver, is_web_url
    from .camera_devices import find_working_camera, list_cameras
else:
    from link_resolver import LinkResolver, is_web_url
    from camera_devices import find_working_camera, list_cameras

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

SCAN_INTERVAL = 0.10
DETECT_SCALE = 1.0
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

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(pts).all():
        return None

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x1 = int(max(0, x_coords.min()))
    y1 = int(max(0, y_coords.min()))

    x2 = int(min(frame.shape[1], math.ceil(x_coords.max()) + 1))
    y2 = int(min(frame.shape[0], math.ceil(y_coords.max()) + 1))

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


def grayscale(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_for_fallback(image, clahe=None):
    clahe = clahe if clahe is not None else cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(grayscale(image))


def read_zxing(image):
    if zxingcpp is None:
        return []
    # Plain text preserves newlines/Unicode instead of HRI display formatting.
    return [result.text for result in zxingcpp.read_barcodes(
        image, formats=zxingcpp.BarcodeFormat.QRCode,
        try_rotate=True, try_downscale=True, try_invert=True,
        text_mode=zxingcpp.TextMode.Plain,
    ) if result.valid and result.text]


def fallback_decode(crop, qr_detector=None, clahe=None, use_zxing=True):
    """Recover an independently located candidate; never enlarge the whole frame."""
    if crop is None or crop.size == 0:
        return None
    qr_detector = qr_detector if qr_detector is not None else cv2.QRCodeDetector()
    gray = grayscale(crop)

    def views():
        yield gray
        scale = min(UPSCALE_FACTOR, 1024 / max(gray.shape))
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield enlarged
        yield preprocess_for_fallback(enlarged, clahe)

    for image in views():
        if use_zxing:
            decoded = read_zxing(image)
            if decoded:
                return decoded[0]
        data, _, _ = qr_detector.detectAndDecode(image)
        if data:
            return data
    return None


def tile_regions(shape, tile_size=640, overlap=0.25):
    """Cover every edge, retaining overlap for small codes crossing tile seams."""
    height, width = shape[:2]

    def starts(length):
        last = max(0, length - tile_size)
        step = max(1, round(tile_size * (1 - overlap)))
        return sorted(set(range(0, last + 1, step)) | {last})

    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in starts(height) for x in starts(width)]


class QRScanner:
    """Native-resolution decoding plus a rolling search of overlapping tiles.

    Work is bounded by tile count, not a fictitious hard time limit: a native
    decoder call cannot be interrupted. Tiles advance even when another code
    succeeds, retaining spatial coverage rather than repeatedly visiting one area.
    """

    LIGHTING_MODES = ("clahe", "shadow", "sharpen", "adaptive", "channel")

    def __init__(self, detector=None, tile_size=640, tiles_per_frame=2,
                 use_zxing=True, lighting=True):
        if tile_size < 128 or tiles_per_frame < 0:
            raise ValueError("tile_size must be >= 128; tiles_per_frame must be >= 0")
        self.detector = detector if detector is not None else create_detector()
        self.qr = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.tile_size = tile_size
        self.tiles_per_frame = tiles_per_frame
        self.use_zxing = use_zxing and zxingcpp is not None
        self.lighting = lighting
        self.cursor = 0
        self.shape = None
        self.last_stats = {}

    def correct_lighting(self, image, mode):
        gray = grayscale(image)
        if mode == "clahe":
            return self.clahe.apply(gray)
        if mode == "shadow":
            # Divide out slowly varying illumination, then stretch surviving
            # contrast. This cannot recover clipped highlights or lost modules.
            background = cv2.GaussianBlur(gray, (0, 0), 21)
            corrected = cv2.divide(gray, np.maximum(background, 1), scale=128)
            return self.clahe.apply(corrected)
        if mode == "sharpen":
            blurred = cv2.GaussianBlur(gray, (0, 0), 0.8)
            return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        if mode == "adaptive":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 5)
        if mode == "channel" and image.ndim == 3:
            # A channel can retain contrast that a color cast hides in luma.
            # Use robust spread; this is a heuristic, not glare removal.
            sample = image[::4, ::4]
            spread = np.percentile(sample, 95, axis=(0, 1)) - np.percentile(sample, 5, axis=(0, 1))
            return self.clahe.apply(image[:, :, int(np.argmax(spread))])
        return gray

    def decode(self, frame):
        if frame is None or frame.size == 0:
            return []
        started = time.perf_counter()
        decoded = []
        methods = []

        def remember(payloads, method):
            for payload in payloads:
                if payload and payload not in decoded:
                    decoded.append(payload)
                    methods.append(method)

        def decode_view(view, method, scale=1.0):
            payloads = read_zxing(view) if self.use_zxing else []
            if payloads:
                remember(payloads, method + "/zxing")
                return True
            self.detector.setScaleFactor(scale)
            payloads, _ = self.detector.detectAndDecode(view)
            remember(payloads, method + "/wechat")
            return any(payloads)

        # ZXing sees every source pixel. The CNN overview is only the fast
        # large-code path; native-resolution tiles below handle distant codes.
        decode_view(frame, "full", min(1.0, self.tile_size / max(frame.shape[:2])))
        if frame.shape[:2] != self.shape:
            self.shape = frame.shape[:2]
            self.cursor = 0
        regions = tile_regions(frame.shape, self.tile_size)
        attempted = min(self.tiles_per_frame, len(regions))
        for _ in range(attempted):
            sweep, index = divmod(self.cursor, len(regions))
            self.cursor += 1
            x1, y1, x2, y2 = regions[index]
            tile = frame[y1:y2, x1:x2]
            if decode_view(tile, "tile"):
                continue
            if self.lighting:
                mode = self.LIGHTING_MODES[sweep % len(self.LIGHTING_MODES)]
                if decode_view(self.correct_lighting(tile, mode), mode):
                    continue
            # WeChat exports only DECODED corners. Detect candidates separately
            # so the crop/upscale recovery path actually runs on failures.
            found, points = self.qr.detectMulti(grayscale(tile))
            if found and points is not None:
                for region in points[:2]:
                    payload = fallback_decode(crop_qr_region(tile, region), self.qr,
                                              self.clahe, self.use_zxing)
                    remember([payload], "candidate")
        self.last_stats = {
            "decode_ms": round((time.perf_counter() - started) * 1000, 1),
            "tiles": attempted, "tiles_in_sweep": len(regions),
            "methods": methods,
        }
        return decoded


class ScanWriter:
    """Keep payloads intact, flush each scan, and suppress repeats per payload."""
    def __init__(self, stream, cooldown=COOLDOWN_SECONDS, resolver=None):
        self.stream = stream
        self.cooldown = cooldown
        self.last_seen = {}
        self.resolver = resolver

    def _write(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()

    @staticmethod
    def _print(captured_at, label, value):
        # Escape newlines/control characters so one capture stays on one line.
        readable = json.dumps(value, ensure_ascii=False)[1:-1]
        print(f"[{captured_at:%H:%M:%S}] {label}: {readable}", flush=True)

    def record(self, payload, captured_at=None):
        if not payload:
            return False
        now = time.monotonic()
        self.last_seen = {
            text: seen for text, seen in self.last_seen.items()
            if now - seen < self.cooldown
        }
        if payload in self.last_seen:
            return False
        captured_at = captured_at or datetime.datetime.now().astimezone()
        record = {
            "time": captured_at.isoformat(timespec="milliseconds"),
            "data": payload,
        }
        self._write(record)
        self._print(captured_at, "QR", payload)
        self.last_seen[payload] = now
        if self.resolver is not None and is_web_url(payload):
            for result in self.resolver.submit(payload, captured_at):
                self._record_link(*result)
        return True

    def _record_link(self, payload, captured_at, result):
        self._write({"type": "link_content", "time": captured_at.isoformat(timespec="milliseconds"),
                     "data": payload, "link": result})
        content = result.get("content")
        if content is not None:
            display = content[:500] + ("..." if len(content) > 500 or result.get("truncated") else "")
            self._print(captured_at, f"Link content ({payload})", display)
        else:
            self._print(captured_at, f"Link content ({payload})", "unavailable - " + result["error"])

    def poll_links(self):
        if self.resolver is not None:
            for result in self.resolver.poll():
                self._record_link(*result)


def scan_frame(detector, frame, writer, captured_at=None):
    for payload in detector.decode(frame):
        writer.record(payload, captured_at)
    writer.poll_links()


def open_camera(index=None, backend="auto", diagnostics=False, camera_name=None):
    if sys.platform == 'win32' and (index is None or camera_name):
        device = find_working_camera(backend, camera_name, diagnostics)
        cap = cv2.VideoCapture(device.index, device.backend)
        if cap.isOpened():
            if diagnostics:
                print(f'Using {device.name}: index {device.index}, {cap.getBackendName()}',
                      file=sys.stderr, flush=True)
            return cap
        cap.release()
        raise RuntimeError(f'{device.name} delivered a test frame but could not be reopened. '
                           'The device may have disconnected; run --list-cameras.')
    if camera_name:
        raise RuntimeError('--camera-name is currently supported on Windows; use --camera on this platform')
    index = 0 if index is None else index
    backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW,
                "avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2,
                "any": cv2.CAP_ANY}
    if backend == "auto":
        choices = {"win32": ["msmf", "dshow", "any"],
                   "darwin": ["avfoundation", "any"]}.get(sys.platform, ["v4l2", "any"])
    else:
        choices = [backend]
    for name in choices:
        cap = cv2.VideoCapture(index) if name == "any" else cv2.VideoCapture(index, backends[name])
        if cap.isOpened():
            if diagnostics:
                print(f"Camera {index}: opened with {cap.getBackendName()}", file=sys.stderr, flush=True)
            return cap
        cap.release()
    hint = {
        "win32": "The requested device/backend did not open. Run --list-cameras to see current indices, "
                 "or omit --camera to discover a camera that delivers frames. "
                 "This error alone does not establish a permissions or camera-in-use problem.",
        "darwin": "Allow terminal camera access in System Settings > Privacy & Security > Camera.",
    }.get(sys.platform, "Check the camera connection, /dev/video device permissions, and camera index.")
    raise RuntimeError(f"Could not open camera {index}; tried {', '.join(choices)}. {hint}")


def configure_camera(cap, args):
    """Request capture properties; backend success does not guarantee hardware support."""
    settings = []
    if args.fourcc:
        settings.append(("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc)))
    settings.extend([
        ("width", cv2.CAP_PROP_FRAME_WIDTH, args.width),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT, args.height),
        ("fps", cv2.CAP_PROP_FPS, args.fps),
    ])
    # Do not assume universal auto-exposure values or exposure units.
    # DirectShow and V4L2 use different conventions; accept explicit raw values.
    for name, prop in (("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                       ("exposure", cv2.CAP_PROP_EXPOSURE),
                       ("autofocus", cv2.CAP_PROP_AUTOFOCUS),
                       ("focus", cv2.CAP_PROP_FOCUS),
                       ("gain", cv2.CAP_PROP_GAIN)):
        value = getattr(args, name)
        if value is not None:
            settings.append((name, prop, value))
    for name, prop, value in settings:
        accepted = cap.set(prop, value)
        actual = cap.get(prop)
        if args.diagnostics:
            print(f"Camera {name}: requested={value}, reported={actual}, accepted={accepted}",
                  file=sys.stderr, flush=True)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; many backends ignore this.


class LatestFrameCapture:
    """Drain the camera continuously, replacing one frame instead of queuing stale ones.

    The capture thread owns read/release; decoder objects stay on the main thread.
    No frame is edited in place after being published.
    """

    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.captured_at = 0.0
        self.captured_time = None
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True, name="qr-camera")

    def _capture(self):
        failures = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("Camera returned no frames after 30 attempts.")
                    self.stop.wait(SCAN_INTERVAL)
                    continue
                failures = 0
                with self.condition:
                    self.frame = frame
                    self.captured_at = time.monotonic()
                    self.captured_time = datetime.datetime.now().astimezone()
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            self.cap.release()

    def __enter__(self):
        self.thread.start()
        return self

    def read(self, timeout=5.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence != self.consumed or self.error is not None,
                timeout=timeout,
            )
            if self.error is not None:
                raise RuntimeError(f"Camera capture failed: {self.error}") from self.error
            if not ready:
                raise RuntimeError(f"No fresh camera frame for {timeout:g} seconds.")
            self.consumed = self.sequence
            return self.frame, (time.monotonic() - self.captured_at) * 1000, self.captured_time

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            print("Camera driver is still blocking in read(); it will be released when read returns.",
                  file=sys.stderr, flush=True)


class Preview:
    """Keep HighGUI on the main thread; resize the window, not decoder input."""

    NAME = "QR scanner - Q or Esc to quit"

    def __init__(self):
        self.opened = False

    def show(self, frame):
        if not self.opened:
            try:
                cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError("Preview needs GUI-enabled opencv-contrib-python. "
                                   "Install QR_Scanning/requirements.txt or use --no-preview.") from exc
            self.opened = True
            height, width = frame.shape[:2]
            cv2.resizeWindow(self.NAME, min(width, 1280), round(height * min(1, 1280 / width)))
        cv2.imshow(self.NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return False
        try:
            return cv2.getWindowProperty(self.NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False  # Some backends throw after the window's X is clicked.

    def close(self):
        if self.opened:
            try:
                cv2.destroyWindow(self.NAME)
            except cv2.error:
                pass  # It may already have been closed with the window's X.
            self.opened = False


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Explicit camera index; default discovers a working RGB camera on Windows")
    source.add_argument("--camera-name", help="Select a Windows camera by name, e.g. 'Lenovo Performance RGB Camera'")
    parser.add_argument("--list-cameras", action="store_true", help="List current Windows camera names, indices, and backends")
    parser.add_argument("--backend", choices=("auto", "msmf", "dshow", "avfoundation", "v4l2", "any"),
                        default="auto", help="Capture backend; auto tries MSMF first on Windows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Append-only JSON Lines scan file")
    parser.add_argument("--width", type=int, default=1920, help="Requested native capture width")
    parser.add_argument("--height", type=int, default=1080, help="Requested native capture height")
    parser.add_argument("--fps", type=float, default=30, help="Requested camera frame rate")
    parser.add_argument("--fourcc", help="Optional camera format, e.g. MJPG; default leaves driver selection")
    parser.add_argument("--auto-exposure", type=float, help="Raw backend-specific auto-exposure mode")
    parser.add_argument("--exposure", type=float, help="Raw backend-specific exposure; disable auto first")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), help="Enable/disable autofocus if supported")
    parser.add_argument("--focus", type=float, help="Raw manual focus value; use --autofocus 0")
    parser.add_argument("--gain", type=float, help="Raw camera gain value")
    parser.add_argument("--tile-size", type=int, default=640, help="Native-resolution tile edge (>=128)")
    parser.add_argument("--tiles-per-frame", type=int, default=2, help="Rolling tiles per scan; 0 disables recovery")
    parser.add_argument("--interval", type=float, default=SCAN_INTERVAL, help="Minimum seconds between scan starts")
    parser.add_argument("--no-zxing", action="store_true", help="Use OpenCV decoders only")
    parser.add_argument("--no-lighting", action="store_true", help="Disable lighting variants for comparisons")
    parser.add_argument("--no-preview", action="store_true", help="Disable the default OpenCV camera window")
    parser.add_argument("--no-resolve-links", action="store_true", help="Log QR URLs without fetching their text")
    parser.add_argument("--diagnostics", action="store_true", help="Print timing and image statistics every 2 seconds")
    parser.add_argument("--save-frames", type=Path, help="Save raw PNGs here every 2 seconds for offline evaluation")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if (args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps) or args.fps <= 0
            or not math.isfinite(args.interval) or args.interval < 0
            or args.tile_size < 128 or args.tiles_per_frame < 0):
        parser.error("Positive width/height/fps, tile-size >=128, and nonnegative interval/tiles required")
    if args.fourcc and len(args.fourcc) != 4:
        parser.error("--fourcc must be exactly four characters")
    for name in ("auto_exposure", "exposure", "focus", "gain"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    cap = None
    preview = Preview() if not args.no_preview else None
    resolver = None
    try:
        if args.list_cameras:
            if sys.platform != 'win32':
                raise RuntimeError('--list-cameras is currently supported on Windows')
            devices = list_cameras(args.backend)
            for device in devices:
                print(f'{device.name} | --camera {device.index} --backend '
                      f'{cv2.videoio_registry.getBackendName(device.backend).lower()}', flush=True)
            if not devices:
                print('No video cameras are currently exposed by Windows.', flush=True)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_frames:
            args.save_frames.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            detector = QRScanner(tile_size=args.tile_size, tiles_per_frame=args.tiles_per_frame,
                                 use_zxing=not args.no_zxing, lighting=not args.no_lighting)
            if args.diagnostics and not args.no_zxing and zxingcpp is None:
                print("ZXing unavailable; using OpenCV only. Install QR_Scanning/requirements.txt.",
                      file=sys.stderr, flush=True)
            cap = open_camera(args.camera, args.backend, args.diagnostics, args.camera_name)
            configure_camera(cap, args)
            resolver = LinkResolver() if not args.no_resolve_links else None
            writer = ScanWriter(stream, resolver=resolver)
            if args.diagnostics:
                print(f"Scanning camera {args.camera}. Results: {args.output.resolve()}", file=sys.stderr, flush=True)
            next_scan = 0.0
            next_report = 0.0
            reported_shape = None
            with LatestFrameCapture(cap) as capture:
                cap = None  # Capture thread now owns release, including on errors.
                while True:
                    delay = next_scan - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    frame, age_ms, captured_at = capture.read()
                    writer.poll_links()
                    if preview is not None and not preview.show(frame):
                        return 0
                    now = time.monotonic()
                    next_scan = now + args.interval
                    if frame.shape[:2] != reported_shape:
                        reported_shape = frame.shape[:2]
                        height, width = reported_shape
                        if args.diagnostics:
                            print(f"Actual frames: {width}x{height}; ZXing={detector.use_zxing}",
                                  file=sys.stderr, flush=True)
                            if (width, height) != (args.width, args.height):
                                print("Camera did not deliver the requested resolution. Check its supported modes/format.",
                                      file=sys.stderr, flush=True)
                    scan_frame(detector, frame, writer, captured_at)
                    if now >= next_report:
                        next_report = now + 2
                        if args.diagnostics:
                            gray = grayscale(frame)
                            stats = dict(detector.last_stats, frame_age_ms=round(age_ms, 1),
                                         dark_pct=round(float(np.mean(gray <= 5)) * 100, 1),
                                         bright_pct=round(float(np.mean(gray >= 250)) * 100, 1))
                            print("Diagnostics " + json.dumps(stats), file=sys.stderr, flush=True)
                        if args.save_frames:
                            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f.png")
                            path = args.save_frames / name
                            if not cv2.imwrite(str(path), frame):
                                raise OSError(f"Could not save frame: {path}")
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, cv2.error) as exc:
        print(f"Scanner error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if preview is not None:
            preview.close()
        if resolver is not None:
            resolver.close()


if __name__ == "__main__":
    sys.exit(main())
