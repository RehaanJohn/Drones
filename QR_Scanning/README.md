# Outdoor USB webcam QR scanner

Targets a 1080p USB webcam on Raspberry Pi 4, with Windows preview support.
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

Tracked regions get the first 75% of the scheduling window. Discovery receives
25% credit per scan, with debt carried forward after expensive calls. Native
decoder calls cannot be interrupted: `--budget-ms 150` is a scheduling budget,
not a hard deadline. Candidates are processed oldest-attempt-first.

Recovery tries untouched crops, valid perspective rectification, then at most two
quality-selected variants: shadow normalization, CLAHE, channel extraction,
adaptive thresholding or mild sharpening. Windows scale with the configured QR
module count. A tile-level lighting fallback also helps when localization fails.

The recent-frame cache retains at most 0.5 seconds and 64 MiB. Best-history
selection uses tracked regions when available, followed by re-localization.
Active frames, worker queues and native libraries use additional memory; this
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
