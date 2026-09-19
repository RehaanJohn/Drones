import cv2
import threading
import queue
import time
import datetime
import os
import urllib.request
from flask import Flask, jsonify
from pyngrok import ngrok
import logging

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAMERA_FPS = 30

# QR detection frequency
SCAN_INTERVAL = 0.10       # ~10 QR searches/sec

# WeChat detector scaling
# 1.0 = maximum, 0.5 = half-resolution
DETECT_SCALE = 0.50

# Extra area around detected QR
ROI_PADDING = 0.25

# Upscale factor for fallback decoding
UPSCALE_FACTOR = 2.0

COOLDOWN_SECONDS = 2.0
MAX_HISTORY = 20

MODEL_DIR = "wechat_models"

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

# ============================================================
# GLOBAL STATE
# ============================================================

frame_queue = queue.Queue(maxsize=1)

running = True

scan_history = []
history_lock = threading.Lock()

last_scanned_text = ""
last_scan_time = 0

last_qr_points = None
points_lock = threading.Lock()


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_models_exist():

    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, url in MODEL_FILES.items():

        filepath = os.path.join(MODEL_DIR, filename)

        if os.path.exists(filepath):
            continue

        print(f"Downloading {filename}...")

        urllib.request.urlretrieve(
            url,
            filepath
        )

        print(f"Downloaded {filename}")


# ============================================================
# QR DECODER INITIALIZATION
# ============================================================

def create_detector():

    detector = cv2.wechat_qrcode.WeChatQRCode(
        os.path.join(MODEL_DIR, "detect.prototxt"),
        os.path.join(MODEL_DIR, "detect.caffemodel"),
        os.path.join(MODEL_DIR, "sr.prototxt"),
        os.path.join(MODEL_DIR, "sr.caffemodel"),
    )

    # Critical for distant QR detection.
    detector.setScaleFactor(DETECT_SCALE)

    return detector


# ============================================================
# IMAGE HELPERS
# ============================================================

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


def fallback_decode(detector, crop):

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


# ============================================================
# QR WORKER
# ============================================================

def decode_worker():

    global running
    global last_scanned_text
    global last_scan_time
    global last_qr_points

    ensure_models_exist()

    print("Loading WeChat QR detector...")

    detector = create_detector()

    print("QR detector ready.")

    next_scan_time = 0
    last_payload = None

    while running:

        try:

            frame = frame_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            continue

        now = time.monotonic()

        # Limit detector frequency
        if now < next_scan_time:
            continue

        next_scan_time = now + SCAN_INTERVAL

        # ----------------------------------------------------
        # SEARCH STAGE
        # ----------------------------------------------------

        search_frame = frame

        try:

            results, points = detector.detectAndDecode(
                search_frame
            )

        except Exception as e:

            print("QR detector error:", e)

            continue

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if results:

            for i, payload in enumerate(results):

                if not payload:
                    continue

                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Avoid duplicate logging
                if (
                    payload == last_payload
                    and time.time() - last_scan_time
                    < COOLDOWN_SECONDS
                ):
                    continue

                print()
                print("======================================")
                print("QR CODE DETECTED")
                print("TIME:", timestamp)
                print("DATA:", payload)
                print("======================================")

                last_scanned_text = payload
                last_scan_time = time.time()

                with history_lock:

                    scan_history.append({
                        "time": timestamp,
                        "data": payload
                    })

                    if len(scan_history) > MAX_HISTORY:
                        scan_history.pop(0)

                last_payload = payload

                # Save points for visualization
                with points_lock:

                    if points is not None and len(points) > 0:
                        last_qr_points = points[0]

                continue

        # ----------------------------------------------------
        # FALLBACK: detector found QR geometry but decode failed
        # ----------------------------------------------------

        if points is not None and len(points) > 0:

            crop = crop_qr_region(
                frame,
                points
            )

            payload = fallback_decode(
                detector,
                crop
            )

            if payload:

                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                print()
                print("======================================")
                print("QR CODE RECOVERED")
                print("TIME:", timestamp)
                print("DATA:", payload)
                print("======================================")

                last_scanned_text = payload
                last_scan_time = time.time()

                with history_lock:

                    scan_history.append({
                        "time": timestamp,
                        "data": payload
                    })

                    if len(scan_history) > MAX_HISTORY:
                        scan_history.pop(0)

                last_payload = payload


# ============================================================
# CAMERA
# ============================================================

def configure_camera(cap):

    # MJPEG helps USB cameras avoid excessive raw USB bandwidth.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    # Request minimum buffering.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Attempt to enable autofocus.
    cap.set(
        cv2.CAP_PROP_AUTOFOCUS,
        1
    )


# ============================================================
# WEB SERVER
# ============================================================

app = Flask(__name__)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@app.route("/")
def dashboard():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Drone QR Telemetry</title>

        <style>

            body {
                font-family: Arial;
                padding: 20px;
                background: #121212;
                color: white;
            }

            .scan-item {
                background: #1e1e1e;
                padding: 15px;
                margin: 10px 0;
                border-left: 5px solid #00ff00;
                border-radius: 4px;
            }

            .timestamp {
                color: #888;
                font-size: 0.8em;
            }

            .payload {
                font-size: 1.2em;
                font-weight: bold;
                margin-top: 5px;
                word-break: break-all;
            }

        </style>

        <script>

            async function fetchScans() {

                try {

                    const res =
                        await fetch('/api/scans');

                    const data =
                        await res.json();

                    const container =
                        document.getElementById('scans');

                    container.innerHTML =
                        data.map(scan => `

                            <div class="scan-item">

                                <div class="timestamp">
                                    ${scan.time}
                                </div>

                                <div class="payload">
                                    ${scan.data}
                                </div>

                            </div>

                        `).join('');

                }

                catch (e) {

                    console.error(e);

                }

            }

            setInterval(fetchScans, 1000);

            window.onload = fetchScans;

        </script>

    </head>

    <body>

        <h2>Drone QR Telemetry Feed</h2>

        <div id="scans"></div>

    </body>

    </html>
    """


@app.route("/api/scans")
def get_scans():

    with history_lock:

        return jsonify(
            list(reversed(scan_history))
        )


def start_web_server():

    port = 5000

    public_url = ngrok.connect(
        port
    ).public_url

    print(
        f"\nNGROK: {public_url}\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    web_thread = threading.Thread(
        target=start_web_server,
        daemon=True
    )

    web_thread.start()

    decoder_thread = threading.Thread(
        target=decode_worker,
        daemon=True
    )

    decoder_thread.start()

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera"
        )

    configure_camera(cap)

    actual_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Camera: {actual_w}x{actual_h} @ {actual_fps} FPS"
    )

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            # ------------------------------------------------
            # Latest-frame queue
            # ------------------------------------------------

            if frame_queue.full():

                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put_nowait(frame)

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            display = frame.copy()

            if last_scanned_text:

                age = time.time() - last_scan_time

                if age < 2:

                    cv2.rectangle(
                        display,
                        (0, 0),
                        (display.shape[1], 80),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        display,
                        f"QR: {last_scanned_text}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3
                    )

            else:

                cv2.putText(
                    display,
                    "SEARCHING FOR QR...",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Show QR bounding box
            # ------------------------------------------------

            with points_lock:

                points = last_qr_points

            if points is not None:

                pts = points.astype(int)

                cv2.polylines(
                    display,
                    [pts],
                    True,
                    (0, 255, 0),
                    3
                )

            cv2.imshow(
                "Drone QR Scanner",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:

        pass

    finally:

        print("\nShutting down...")

        running = False

        cap.release()

        cv2.destroyAllWindows()

        try:
            ngrok.kill()
        except:
            pass


if __name__ == "__main__":

    main()