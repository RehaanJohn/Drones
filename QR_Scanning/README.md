# Outdoor USB webcam QR scanner

Targets a 1080p USB webcam on Raspberry Pi 4, with Windows and macOS preview support.
Uses native-resolution decoding, candidate tracking, quality-directed recovery
and two-frame confirmation. **10 m / 99% within two seconds remains a field
qualification target, not a measured capability.**

## Run

Windows development (preview enabled by default):

```powershell
.\venv\Scripts\python.exe QR_Scanning/scanner.py --camera-name "Lenovo Performance RGB Camera"
```

Q, Esc or closing the window stops scanning. Orange outlines are provisional;
green means the payload matches in two distinct captured frames. Preview is
independent of decoding. Input frames retain native resolution. Windows camera
discovery remains available through `--list-cameras`; indices can change after
USB reconnection.

macOS development (from inside `QR_Scanning`):

```bash
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation --diagnostics
```

macOS reads the camera on the main thread and uses the actual delivered resolution
when AVFoundation chooses a different mode. Diagnostics include image brightness
and pixel range; receiving frames alone does not establish a usable camera image.
For a black preview, isolate capture from all QR processing:

```bash
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation --camera-test --save-frames camera_debug
```

This displays raw frames, prints brightness statistics, and saves
`camera_debug/raw_camera.png` after two seconds. A normal saved image with a black
window points to preview rendering; a black saved image means the captured pixels
are black too. Q or Esc exits the test.

Install the updated requirements to enable macOS camera listing and name selection:

```bash
python3 -m pip install -r requirements.txt
python3 scanner.py --list-cameras --backend avfoundation
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation --diagnostics
```

Without an explicit camera, macOS discovery prefers a built-in MacBook/FaceTime
camera. Index 0 can be an iPhone Continuity Camera, and indices may change when
devices connect or disconnect; use the listed name for stable selection.

## Diagnostic field testing

Run these commands **from inside `QR_Scanning`** with the virtual environment active.
On the Mac used during development, camera 0 was the iPhone Continuity Camera and
camera 1 was the MacBook Air camera. Name selection avoids relying on that ordering.

```bash
python3 -m pip install -r requirements.txt
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation --diagnostics
```

The display draws yellow outlines around localized QR candidates, cyan around
decoded payloads awaiting confirmation, and green around payloads confirmed in
two distinct frames. Localization can continue beyond the distance at which a
payload is readable, but arbitrary distant pixels cannot reliably be identified
as a QR. Boxes expire after 0.75 seconds and carry an estimated native pixels/module
reading based on `--qr-modules`. The panel shows brightness, contrast, sampled
sharpness, clipping, uneven illumination, decoding time and result age. QR black
and white areas naturally clip; clipping percentages alone do not prove glare.

For repeatable tests, display or print [diagnostic_target.png](diagnostic_target.png).
It encodes `diagnostic-test-123` and has 29 modules per side, excluding its quiet
zone. Place it at a **measured** distance and start a labelled session:

```bash
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation \
  --diagnostics --expected-payload diagnostic-test-123 --qr-modules 29 \
  --condition indoor-baseline --distance-m 1 --trial-seconds 10 \
  --diagnostics-dir diagnostic_runs --no-resolve-links
```

Use the actual payload/module count when testing your own QR. Expected text is
used only by the evaluator, never to help the decoder recognize a target. Without
expected text, the panel reports decode presence rather than correctness.

With the **preview window focused**:

| Control | Action |
| --- | --- |
| `T` | Begin a new timed trial, clearing prior decoder confirmation state/history eligibility. |
| `L` | Toggle lighting recovery for comparison, then press `T` for a fresh trial. |
| `S` | Save untouched camera and simulated scanner-input PNGs. |
| `Q` / Esc | Quit. In the terminal use Ctrl+C. |
| Brightness slider | 100 is unchanged; 50 halves image brightness. |
| Blur slider | 0 is unchanged; higher values add Gaussian blur. |
| Shadow / glare sliders | 0 is unchanged; add a left-to-right shadow or central glare patch. |

The first trial starts automatically. Hold the target and conditions constant
for the trial duration, then allow up to three seconds for in-flight results.
Changing a simulation slider or lighting-recovery mode aborts the active trial;
press `T` to test the new setting. Synthetic effects change the scanner input and
are explicitly labelled; they are not calibrated reproductions of physical
lighting, movement or camera exposure. The suggested correction thumbnail is
illustrative; the reported decoder method identifies the actual successful path.

Test real sunlight, shade, backlight, changing light, motion, oblique angles and
progressively greater measured distances. Restart with a new `--condition` and
`--distance-m` for each physical setup. Repeat trials with recovery on/off under
the same conditions. For labelled negative tests without a target, use the replay
workflow below with `expected: []`. Use fresh target presentations for independent trials; adjacent frames
or repeated trials of a continuously stationary target are correlated.

Each timestamped subdirectory contains `session.json`, `samples.jsonl`,
`trials.jsonl`, `trials.csv`, and `summary.json`. Trials include correct decode and
confirmation rates per distinct **processed** frame, incorrect payloads, captured
and processed frame counts, first confirmation completion time, latency p95 and
minimum successfully decoded pixels/module. A frame captured early but decoded
after two seconds does not pass the two-second timing check. Results arriving
after the three-second drain allowance are excluded. Inspect processed/captured
counts and queue-drop diagnostics when interpreting rates.

The smallest decoded pixel size and farthest measured confirmation are observed
results, not a guaranteed maximum range. Compare completed trials across runs:

```bash
python3 diagnostics.py --report diagnostic_runs
```

`comparison.json` groups condition, distance, simulation settings, recovery,
expected payload, trial duration and camera/decoder configuration. Confidence
bounds assume independent trials; a handful of successful trials cannot establish
99% reliability. The diagnostic display and simulations add overhead, so repeat
latency measurements in competition mode for the final operating conditions.

Competition mode uses the same improved decoder and lighting recovery, with
preview, simulations and diagnostic trial recording disabled:

```bash
python3 scanner.py --camera-name "MacBook Air Camera" --backend avfoundation --competition
```

Add `--diagnostics` to print performance statistics while remaining headless.
An individual native decoder call may exceed the scheduling budget. The worker
now prioritizes the latest frame, skips historical retries after overruns, and
builds recovery variants only after a direct crop fails; full-frame overruns do
not permanently starve native-tile discovery.

Raspberry Pi setup and headless competition mode:

```bash
sudo apt install v4l-utils
python -m pip install -r QR_Scanning/requirements.txt
python QR_Scanning/scanner.py --list-cameras
python QR_Scanning/scanner.py --inspect-camera --device /dev/video0
python QR_Scanning/scanner.py --competition --profile-dir QR_Scanning/camera_profiles
```

Linux discovery prefers stable `/dev/v4l/by-id/` paths. Pin the actual RGB endpoint
reported by discovery with `--device PATH`. An explicit `--camera 0` selects
`/dev/video0`; raw node numbers may change after reconnection.

V4L2 runs in a separate process: blocked open/read calls can be terminated after
five seconds, with at most three restarts. Discovery can skip failed candidates;
after selection, reconnects stay on that device. There is no CSI/Picamera2 dependency.

Default request: 1920x1080 at 30 fps, preferring advertised MJPEG when suitable.
Resolution is never silently reduced. A lower advertised frame rate is reported.
Compare `--fourcc YUYV` if available: MJPEG saves USB bandwidth but costs decoding
CPU and may damage small modules. Actual frame rate must be measured on the Pi.

Profiles are identity-keyed evidence snapshots, not automatically reapplied
configuration. They record advertised modes/controls, actual dimensions, reported
format, validated settings and measured frame rate. Each startup negotiates again.
Provision the WeChat model files before going offline; missing files download on
first use.

## Camera controls

Default exposure, gain and focus remain under the camera's existing settings.
Linux disables dynamic exposure frame-rate reduction and resets absolute zoom to
its advertised minimum where controls exist. Proprietary auto-framing is not
assumed controllable through standard V4L2.

```bash
# Starting experiment, not a universal daylight preset:
python QR_Scanning/scanner.py --exposure-us 1000 --diagnostics

# Opt-in feedback after validating the motion/exposure ceiling:
python QR_Scanning/scanner.py --adaptive-exposure --max-exposure-us 2000 --diagnostics
```

Linux uses discovered manual exposure modes and 100-microsecond exposure units,
with range/step checking and readback. Unsupported controls are reported. Manual
exposure is not written if the mode change fails. Windows/macOS retain raw
`--auto-exposure` / `--exposure` options. `--autofocus 0 --focus VALUE` and
`--gain VALUE` require camera-specific calibration; fixed-focus cameras cannot
be focused through software.

Adaptive exposure starts at 1000 us unless overridden. It uses recent candidate
brightness, a deadband, changes of at most 20%, and at least 0.5 seconds between
changes. It does not automatically increase gain or claim measured sensor
settling latency. Validate settling using recordings and control-change timestamps.

## Decoder and output

Every `--discovery-every 3` processed frames reserves discovery before crop work,
even while other codes decode. Global and native tiled discovery alternate so
neither can starve. Additional discovery runs after unsuccessful crop work when
affordable. Native decoder calls cannot be interrupted: `--budget-ms 150` is a
scheduling budget, not a hard deadline. Candidates are serviced oldest-first.

Recovery tries untouched crops, the last successful recipe, upscaling, and a
rotating ladder of WeChat, aligned frame fusion, CLAHE, shadow normalization,
sharpening, perspective correction and adaptive thresholding. Successful recovery
is retried on the next capture, allowing prompt multi-frame confirmation.
Lighting-corrected crops and tiles retain WeChat fallback. Native tile sweeps
also rotate through raw and lighting variants, including color-channel extraction.
`--no-lighting` disables corrections in both crop and fusion paths as well as tiles.

The recent-frame cache retains at most 0.5 seconds and 64 MiB. Best-history
selection uses tracked regions when available, followed by re-localization.
Linux shared-memory slots use locks and generation checks: overwritten messages
are discarded, and a frame copy cannot overlap a writer. Active frames, worker queues and native libraries use additional memory; this
is not a total-process memory limit. Separate capture, decode and output workers
have bounded queues and report dropped results.

| Option | Purpose |
| --- | --- |
| `--qr-modules 29` | Actual symbol modules per side, excluding border; default is an estimate. |
| `--budget-ms 150` | Scheduling budget before native-call overruns. |
| `--tile-size 640 --tiles-per-frame 2` | Native discovery tiles with 25% overlap; count is an upper bound. |
| `--tiles-per-frame 0` | Disable tiled discovery, retaining initial full-frame acquisition and tracking. |
| `--interval 0.10` | Minimum interval between decode batches. |
| `--no-lighting` / `--no-zxing` | Ablation controls for comparisons. |
| `--diagnostics` | Timings, observed FPS, queue drops, CPU and Pi temperature/throttling when available. |
| `--log-observations` | Add structured provisional/confirmed observations to JSONL. |
| `--save-frames DIRECTORY` | Asynchronously save a native PNG about every two seconds. |
| `--no-preview` | Disable preview; `--competition` also disables PNG saving. |
| `--no-resolve-links` | Disable HTTP lookup. |

```text
[14:23:05] QR: https://qr-codes.io/zuabFE
[14:23:05] Link content (https://qr-codes.io/zuabFE): 69
```

Times mean completion of the confirming camera read, not exposure start or HTTP
completion. Payloads are preserved and terminal control characters escaped.
Original `time`/`data` JSONL records remain compatible; optional `observation`
records add geometry, frame/track IDs, methods, quality and confirmation.
`link_content` events remain separate asynchronous results. Scan counters must
exclude those additional event types.

HTTP resolution retains bounded workers, caching, public-URL validation, response
limits and a five-second network timeout. It handles static HTML/text/JSON, not
logged-in or JavaScript-rendered pages. Network failure does not invalidate QR
decoding.

`read_and_decompress_ms` combines camera read and decompression: this OpenCV path
cannot separately time sensor exposure, USB transfer and JPEG decoding. Timestamps
are read-completion times. The preview's `ppm~` estimate depends on the supplied
module count and must not be mistaken for a calibrated range guarantee.

Library compatibility: `QRScanner.decode(frame)` returns provisional payload
strings for still-image callers. `observe(frame, frame_id, timestamp, captured_at)`
returns structured observations; distinct frame IDs are necessary for confirmation.
Use `reset()` between independent sessions. Existing `create_detector`,
`crop_qr_region`, and `fallback_decode` interfaces remain for mission integration.
`LegacyQRScanner` retains the preceding scanner for comparison.

## Separate NCNN presence detector

[`ncnn_detector.py`](ncnn_detector.py) runs a QR-specific NCNN model without decoding,
with an OpenCV preview, small native crops and lighting variants. See
[`NCNN_DETECTOR.md`](NCNN_DETECTOR.md) for installation, camera/image/video commands,
the 280 mm A3 example, and the distinction between detection and readable range.

For the decoder's outdoor tests, keep lighting recovery enabled and start with:

```bash
python QR_Scanning/scanner.py --width 1920 --height 1080 --budget-ms 300 --diagnostics --no-resolve-links
```

The larger scheduling budget gives difficult crops more recovery opportunities,
at the cost of processing fewer frames. Compare recorded flight frames before
choosing it over the 150 ms default. For longer range, native optical detail and
camera focus/exposure matter more than arbitrary digital enlargement. No maximum
distance is claimed without a measured camera, lens and printed target.

## Optical qualification

Measure the QR square including its four-module border, actual module count, and
compressed-axis image width at 10 m:

```bash
python QR_Scanning/qualification.py optical --modules 29 --qr-width-mm 280 --distance-m 10 --outer-pixels 45
```

Alternatively supply measured `--hfov` and `--tilt-deg`. The report estimates
native pixels/module and the distance for a four-pixel design margin. It is **not**
a measured decode range. Upscaling cannot recover information lost to insufficient
sampling, clipping, blur or obstruction.

## Record and replay field sequences

Record lossless FFV1 video plus read-completion timestamps on the Pi:

```bash
python QR_Scanning/qualification.py record --seconds 10 --output QR_Scanning/captures/sun-01
```

The output directory must not already exist. Optional `--device`, `--fourcc`,
`--fps`, `--width`, `--height`, and `--exposure-us` compare supported modes.
Recording itself can reduce throughput; saved timestamps and camera counts expose
that gap. FFV1 adds no lossy encoding stage but cannot undo webcam MJPEG artifacts.

Create an independently labelled manifest (paths relative to the manifest).
Start timing when the **complete target enters the qualified view during stable
hover**, not when the decoder notices it. Include at least two seconds afterward:

```json
{
  "trials": [{
    "id": "sun-session01-attempt01",
    "condition": "sun",
    "video": "sun-01/capture.mkv",
    "timestamps_file": "sun-01/timestamps.json",
    "start_s": 1.0,
    "end_s": 3.5,
    "expected": ["https://qr-codes.io/zuabFE"],
    "distance_m": 10,
    "obliquity_deg": 0,
    "stable_hover": true,
    "modules": 29
  }]
}
```

Use `expected: []` for negative trials. Trial independence must be established
in the field; relabelling adjacent frames does not create independent attempts.

```bash
python QR_Scanning/qualification.py replay QR_Scanning/captures/trials.json --compare-legacy --output QR_Scanning/benchmark_output/field.json
python QR_Scanning/benchmark.py --output QR_Scanning/benchmark_output/stills.json
python -m unittest discover -s QR_Scanning/tests -v
```

Replay skips frames unavailable under measured decoder compute, and applies
two-frame confirmation to both implementations. It reports latency, wrong payloads,
negative controls, coarse failure stage, and exact one-sided binomial bounds.
It does not model the full asynchronous history scheduler or live USB/output
latency, and cannot distinguish optics from localization solely from a missed QR.

The replay gate requires at least 300 eligible 10 m trials per daylight category
(`sun`, `shade`, `overcast`, `backlight`, `transition`), a 95% lower success
bound of at least 99%, and no wrong confirmed payloads. Zero failures in 300
independent attempts satisfies that bound. Final field acceptance additionally
requires live end-to-end latency and sustained Pi thermal/motion testing.

Historical optical research and earlier benchmark results are in
[OUTDOOR_RESEARCH.md](OUTDOOR_RESEARCH.md).
