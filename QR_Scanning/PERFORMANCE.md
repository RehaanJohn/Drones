# Pi 4B latency, content and range qualification

All commands below run **inside `QR_Scanning`**, with the virtual environment active.
From the repository root, prefix script and requirements paths with `QR_Scanning/`.

## Run with page content and a live camera

```bash
python -m pip install -r requirements.txt
python -u scanner.py --device /dev/video0 --competition --preview --diagnostics --exposure-us 1500
```

This uses the realtime heuristic pipeline, native 1920x1080 grayscale capture,
requested 30 FPS, one valid decode, asynchronous page lookup and `cv.imshow`.
Remove `--preview` for headless competition use. `--no-resolve-links` is deliberately
invalid with competition mode. Exposure 1500 us is an experiment, not a universal
low-light setting: check applied controls and brightness in the reports.

The main view always contains current camera pixels. Last analysis is shown on its
source frame in a labelled inset; a slow model cannot paint old geometry on a new frame.
Q/Esc exits. Grayscale diagnostics report timing without running visual stress simulations.
The existing desktop diagnostic panel remains available with the current pipeline.

Opt-in hybrid run:

```bash
python -m pip install --no-deps ncnn==1.0.20260526
python ncnn_detector.py --self-test
python -u scanner.py --device /dev/video0 --competition --preview --diagnostics --localizer hybrid --ncnn-confidence 0.6 --ncnn-threads 1
```

The standalone `ncnn_detector.py` still only detects QR-like regions and keeps its
live preview. Both entrypoints use `ncnn_model.py` and the same pinned weights.
Do not install NCNN's OpenCV dependency over `opencv-contrib-python`.

## Architecture and measurement boundaries

The Linux default consists of the main display/coordinator and three child processes:

1. Capture: V4L2 camera ownership, optional direct grayscale JPEG decoding, bounded
   shared memory, generation checks and camera recovery.
2. Fast decoding: newest-frame crops, ZXing, bounded enlargement and locally checked
   forward/backward optical flow. The initial scheduling target is 33 ms. No WeChat
   or lighting-recovery call runs here.
3. Discovery/recovery: one selected view per job, native crops, periodic full-frame
   ZXing, and alternating 384/192-pixel tiles with 25% overlap. Every fourth job
   must advance independent spatial coverage. Hybrid mode adds one NCNN inference
   on a failed raw view, followed by immediate decoding of original-resolution crops.

Raw views come first. Later jobs may retry the original source crop with WeChat,
enlargement, rectification or lighting corrections. Candidate fusion rejects large
brightness/aspect changes and uncertain translation alignment. This is noise
reduction, not a claim of true multi-frame super-resolution. Tracking/proposals
carry source frame identity; old proposals require checked optical-flow mapping.

Native calls can exceed budgets. Process separation protects the fast path from
those calls, but CPU/memory contention still matters. OpenCV and NCNN initially use
one compute thread; OpenMP waits are passive. Compare two NCNN threads on the Pi.
Frame/diagnostic queues are replaceable; confirmed results use separate bounded
delivery. Output saturation is an explicit failure, not a silently lost scan.

`--pipeline current --interval 0.1 --color-capture` reproduces the preceding live
scheduler/capture settings for comparisons (code fixes still apply). The current
pipeline retains its old tile/discovery controls. Hybrid requires realtime.

Telemetry separates capture FPS, preview FPS, each worker's processed FPS,
localization/recipe times, worker p50/p95, tile sweep time and output timings.
`frame_age_ms` starts after acquisition/decompression. It does not measure exposure
or hidden driver buffering. OpenCV does not expose reliable V4L2 exposure timestamps
on this path; use an external visible-timer test for true scene-to-result latency.
Camera queue drops are estimated, and replaced diagnostic batches are not a count
of lost successful decodes.

## Isolate the camera bottleneck

```bash
python scanner.py --inspect-camera --device /dev/video0
python scanner.py --camera-test --device /dev/video0 --gray-capture --no-preview --test-seconds 30 --benchmark-output capture-30.json --fps 30 --fourcc MJPG
```

Reports contain modes, USB identity/speed (when sysfs exposes it), readbacks,
OpenCV JPEG/SIMD build facts, temperature/throttling, acquisition, JPEG decode,
shared-memory copy and queue-residence measurements. Color capture cannot split
acquisition and JPEG decode; compare it using `--color-capture`.

For an automatic stationary exposure/FPS matrix:

```bash
python capture_benchmark.py --device /dev/video0 --output capture-matrix --seconds 30
```

The output directory must be new. This tries exactly advertised 1080p MJPG 30/60
FPS modes and 1500/3000/6000/10000 us exposures. Unsupported FPS modes are recorded
as skipped. Examine each report's exposure validation; a requested value is not
proof it applied. The matrix measures capture, not QR accuracy. Use the same scenes
and exposure settings in the decode trials before selecting a mode. Check actual
USB speed, cooling and throttling rather than assuming the camera's advertised
60 FPS or a different USB port guarantees delivery.

Inspect JPEG/SIMD build facts before replacing libraries. Keep native resolution
and select 30/60 FPS by correct-result latency and range, not reported camera FPS.

## Final-result contract

Consumer code should select JSONL events with `type == "result"` and
`schema_version == 2`. Fields:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `resolution_failed`, or explicit experiment `decoded_only` |
| `data`, `content` | Main page text, direct QR text, or null on lookup failure |
| `original_payload`, `final_url` | Original QR text and redirect destination |
| `frame_id` | Source capture identity, when available |
| `time`, `completed_at` | Source capture wall time and publication wall time |
| `timings` | Decode, read-completion-to-decode, publication, HTTP queue/fetch/extraction/resolve, total after read completion |
| `cached`, `truncated`, `error` | Lookup provenance and failure details |

`decoded`/`resolving` and legacy `link_content` events are intermediate/provenance
events. URL lookup failures never produce the URL as successful page content.
Cache hits have zero HTTP timing for that observation. Socket timeout is not a
guaranteed whole-request wall-clock deadline. A stopped pending request produces
an explicit failure. HTML extraction handles static main content; no browser is
launched for JS-only/login pages. The raw `QRScanner.decode()`/`observe()` APIs
remain compatible for the mission adapter and never fetch URLs themselves.

## Field qualification

Use the existing URL-bearing print. Measure the actual square including quiet zone,
module count, focus and shorter projected dimension. A3 does not imply the QR fills
the paper. At 10 m a 280 mm square spans about 38 pixels at 70-degree HFOV, versus
152 at 20 degrees with a 1920-wide image. Four native pixels/module is a design
margin, not a guaranteed decoder threshold. Check optics before adding filters.

```bash
python qualification.py optical --modules 29 --qr-width-mm 280 --distance-m 10 --hfov 70
python qualification.py record --device /dev/video0 --output captures/run1 --seconds 30 --exposure-us 1500
python qualification.py replay captures/trials.json --compare-all --output comparisons.json
```

Use the manifest structure in README.md. Record independent target entries at
2/4/6/8/10/12 m in sun, shade, backlight, mixed shadows and measured low light.
Include centre/edges/corners/tile seams, empty clutter, multiple codes and a tiny
code beside an easy large code. Record actual lighting, exposure, pose, pixel size,
FPS, printed module count, and stationary/moving status. Do not use adjacent frames
as separate independent trials. Ground truth goes only to the evaluator.

Paced heuristic/hybrid replay runs the actual analysis processes and includes IPC
and CPU contention. Current baseline replay measures compute and skips unavailable
frames using `--interval` (default 0.1 for the baseline). Trial workers start fresh.
Startup/model construction completes before timed target entry. Lazy recovery
initialization and all subsequent work remain timed. Compare NCNN confidence
0.6/0.4/0.25 and one/two threads with `--ncnn-confidence`/`--ncnn-threads`.

Initial acceptance is **at least 95/100 correct within one second per condition**
at 10 m, stationary, with zero incorrect payloads. Report the exact one-sided 95%
confidence bound separately: empirical acceptance is not proof of 95% reliability.
Slower recovery continues and is reported. No pooling across lighting conditions.
Compare first-result latency and recall, not detection-box count. Promote hybrid
only when difficult cases improve without regressing already-qualified conditions.

Replay excludes USB, live preview, HTTP and physical target-entry timing. Repeat
the selected settings live, measure decoded-URL and final-content latency separately,
and perform sustained thermal tests before moving trials. `field_qualified` remains
false in automated replay. No 10 m/low-light or Pi throughput claim follows from
local synthetic tests. Optics changes, custom training, quantization, GPU acceleration
and true multi-frame super-resolution are later decisions based on measured failures.

Research background: [SAHI](https://arxiv.org/abs/2202.06934),
[NCNN threading](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/openmp-best-practice.html),
[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo),
[Google multi-frame SR](https://research.google/pubs/handheld-multi-frame-super-resolution/).
