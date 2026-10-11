# Outdoor QR scanner

Requests 1920x1080 capture, decodes native pixels with ZXing, and searches
overlapping crops with WeChat and lighting corrections. Prints payloads and
appends UTF-8 JSON Lines (`time`, `data`) to `QR_Scanning/qr_scans.jsonl`.
Payloads, including newlines and Unicode, remain intact; repeats are suppressed
per payload for two seconds. A live `cv2.imshow` preview opens by default.
Press **Q**, **Esc**, close the window, or use Ctrl+C to stop. No server is needed.

**A3 at 10 m requires checking the lens.** A 29.7 cm square including its border
occupies only about 41 pixels at 10 m with 1080p and a 70-degree horizontal FOV.
Read [the research, range calculations, and measured comparison](OUTDOOR_RESEARCH.md).

## Run

From the repository root on Windows:

```powershell
.\venv\Scripts\python.exe -m pip install -r QR_Scanning/requirements.txt
.\venv\Scripts\python.exe QR_Scanning/scanner.py
```

On Linux/macOS use your environment's `python` in place of
`.\venv\Scripts\python.exe`. On Windows, startup discovers an RGB camera and
verifies frame delivery in a child process with a five-second timeout per candidate.
Stalled candidates are skipped; IR face-authentication devices are excluded by default.
Use `--output path/to/scans.jsonl` to change the output. Models resolve relative to the script and download
if missing. ZXing is recommended; an OpenCV-only fallback remains available.

Select the Lenovo RGB camera by name instead of an index that may change when USB
devices are reconnected:

```powershell
.\venv\Scripts\python.exe QR_Scanning/scanner.py --camera-name "Lenovo Performance RGB Camera"
```

Use `--list-cameras` to list current indices per backend without opening the preview.
`--camera INDEX` remains an explicit override and bypasses discovery. Do not reuse
an old index after reconnecting: during troubleshooting the working Lenovo device
changed from index 1 to index 0. Named discovery and 1920x1080 frame delivery were
verified with the actual Lenovo camera. Linux/macOS retain index 0 as the default.

The preview requires **opencv-contrib-python**, not a headless wheel. The current
project environment has been migrated. If upgrading another environment that
already has multiple OpenCV wheels, remove them first and reinstall requirements
so the shared `cv2` files are not left conflicting:

```powershell
.\venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
.\venv\Scripts\python.exe -m pip install -r QR_Scanning/requirements.txt
```

The preview displays the raw camera frame in a resizable window, initially at
most 1280 pixels wide. Decoder input stays at full capture resolution. Preview
refresh follows scan speed. Use `--no-preview` when running without a display.

## Terminal output and QR links

Normal output contains the frame acquisition time and decoded content:

```text
[14:23:05] QR: DRONE-01
[14:23:08] QR: https://qr-codes.io/zuabFE
[14:23:08] Link content (https://qr-codes.io/zuabFE): 69
```

The timestamp is when the camera read completed, before decoding or HTTP lookup.
Newlines/control characters in payloads are escaped on screen; JSON retains the
original content. Camera property logs and timings are opt-in with `--diagnostics`.

HTTP(S) payloads are fetched in background workers. HTML main content is extracted
without scripts, navigation, or footers; plain text and JSON are supported too.
Your example link was verified to return `69`. Repeated lookups share a bounded
queue and cache successful content for 60 seconds (errors for five seconds).
Internet access is required. `--no-resolve-links` disables fetching.

Lookups use a five-second network timeout, limited redirects, and a 1 MB response
limit. Private/local destinations and links containing credentials are rejected.
JavaScript-only pages, login pages, and PDFs/images are not rendered or OCR'd;
failures are logged as `Link content (...): unavailable - ...`. HTML extraction is
a best-effort text view, not a general browser. Terminal content is limited to
500 characters and stored link text to 4000. Unfinished lookups are abandoned on
exit so they do not hold up camera shutdown.

The JSONL file keeps the original `time`/`data` scan record and appends a separate
`type: "link_content"` event when lookup completes. That event retains the scan's
timestamp and QR payload, with `link.url` (final destination), `link.content` or
`link.error`, and `link.truncated` when applicable. Consumers counting physical
scans should exclude these link events.

Use `--diagnostics` to check the **Actual frames** startup message. Some USB cameras need MJPEG to
deliver 1080p at a useful rate; try `--fourcc MJPG` if that is a supported mode.
It is optional because driver support and compression vary. Requesting a higher
resolution than the camera natively supports does not create detail.

macOS uses AVFoundation (allow terminal camera access in System Settings), Windows
tries MSMF, then DirectShow, then automatic selection; Linux tries V4L2 and
automatic selection. Use `--backend msmf` or `--backend dshow` to test a specific
Windows backend without switching. `--diagnostics` reports the opened backend.
MSMF hardware transforms default to disabled to avoid GPU initialization delays;
an explicitly set `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS` environment variable
overrides this. The index mismatch, rather than a demonstrated privacy restriction,
was the cause of the latest failed `--camera 1` attempt.
A DSHOW "can't be used to capture by index" warning means that attempt failed;
if a preview appears afterward, a fallback succeeded. If nothing opens, check the
camera index, USB connection, other apps using it, and Windows camera permissions
including desktop app access. A Pi CSI/libcamera camera may need a separate capture adapter; this script
uses OpenCV `VideoCapture`.

## Controls

| Option | Purpose |
| --- | --- |
| `--width 1920 --height 1080 --fps 30` | Default requested camera mode; actual delivery is reported. |
| `--autofocus 1` | Request autofocus when supported; inspect whether it focuses on the target. |
| `--autofocus 0 --focus VALUE` | Lock a tested focus position; units depend on the camera. |
| `--auto-exposure VALUE --exposure VALUE` | Set raw backend-specific mode/exposure values. |
| `--gain VALUE` | Request camera gain. |
| `--tiles-per-frame 1` | Less work per scan; slower spatial coverage. Default: 2. |
| `--tiles-per-frame 4` | Faster spatial coverage, more work per scan. |
| `--tile-size 640` | Default native tile edge with 25% overlap. |
| `--interval 0.10` | Minimum interval between scan starts, not a guarantee of 10 scans/s. |
| `--no-lighting` | Disable tile lighting variants for comparison; candidate recovery still uses CLAHE. |
| `--no-zxing` | Use OpenCV decoders only. |
| `--tiles-per-frame 0` | Disable tile/candidate recovery, keeping the full-frame path. |
| `--no-preview` | Disable the default `cv2.imshow` window. |
| `--no-resolve-links` | Log QR URLs without fetching linked text. |
| `--diagnostics` | Show camera configuration, actual resolution, and processing statistics. |

Exposure/focus/gain stay at existing camera settings unless requested. Diagnostics
print property acceptance and readback. There is no universal numeric outdoor exposure preset:
[OpenCV properties depend on hardware and backend](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).

The raw image is tried first. Failed tiles receive one retry variant: CLAHE,
shadow normalization, mild sharpening, adaptive thresholding, or channel contrast.
The variant changes after each spatial sweep. Independent OpenCV candidates enable
padded crop/upscale recovery even when WeChat returns no corners.

At 1080p, eight default tiles take four scans per sweep and 20 scans to schedule
all five variants everywhere when raw decoding fails. Rapid motion can outrun
this rolling search. More tiles per scan shorten that delay at greater CPU cost.

The capture thread replaces one latest frame continuously. It avoids an application
backlog, but cannot eliminate buffering inside a camera driver. Diagnostics include
decode time, age since frame read completed, sweep size, successful methods, and
nearly black/white pixel fractions. Those fractions cover the whole scene and
are not standalone exposure scores.

## Field evidence and comparisons

Save untouched native PNG frames about every two seconds while measuring distance:

```powershell
.\venv\Scripts\python.exe QR_Scanning/scanner.py --diagnostics --save-frames QR_Scanning/captures
.\venv\Scripts\python.exe QR_Scanning/benchmark.py --input-dir QR_Scanning/captures --expected DRONE-01 --passes 20
```

Substitute the actual payload for `DRONE-01`, or omit `--expected` for manual
inspection. All scored input frames should have that same expected content.
Frames keep accumulating until scanning stops; monitor disk usage. Capture and
benchmark output directories are ignored by Git.

Reproduce the synthetic comparison and camera-free regression tests:

```powershell
.\venv\Scripts\python.exe QR_Scanning/benchmark.py --save-images
.\venv\Scripts\python.exe -m unittest discover -s QR_Scanning/tests -v
```

The benchmark writes `QR_Scanning/benchmark_output/results.json`, including decoded
strings, methods, and timing. It advances through up to `--passes` rolling scans
of each still image, stopping when expected content is found. This does not
simulate independent live frames or prove outdoor range. See
[OUTDOOR_RESEARCH.md](OUTDOOR_RESEARCH.md) for the validation procedure.
