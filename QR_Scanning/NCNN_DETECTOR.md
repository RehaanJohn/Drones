# NCNN detection-only preview

`ncnn_detector.py` locates QR-looking regions, draws boxes with model scores using
`cv2.imshow`, and prints JSON detections. It never attempts to decode the QR.
Scores are model outputs, not calibrated probabilities. A detected box is not
proof that a valid QR payload exists or is readable.

## Install and run

Use the same Python environment as the scanner:

```bash
python -m pip install -r QR_Scanning/requirements.txt
python -m pip install --no-deps ncnn==1.0.20260526
python QR_Scanning/ncnn_detector.py --download-models
python QR_Scanning/ncnn_detector.py --camera 0 --width 1920 --height 1080
```

`--no-deps` is intentional: NCNN's package dependencies include base
`opencv-python`, which can overwrite the scanner's GUI-enabled contrib OpenCV.
The NCNN core binding used here only needs the already-installed NumPy package.
The tested version is pinned; use an NCNN wheel/build for your OS, architecture
and Python version. See [NCNN's installation instructions](https://github.com/Tencent/ncnn/tree/master/python).

The two small model files are included, so provisioning normally just checks
their hashes. If missing, `--download-models` retrieves the pinned copies and
exits. Normal camera startup is offline. Provenance and notices are in
[`ncnn_models/README.md`](ncnn_models/README.md).

Omit `--camera` for desktop camera discovery, or use `--camera-name "NAME"`.
Linux uses `--device /dev/video0` with process-isolated V4L2 capture. Q, Escape,
or closing the window exits. The main `cv2.imshow` view displays every current
camera frame, including before the first inference result. Model loading and
inference run in a latest-frame worker after the camera preview opens. A labelled
inset shows the last analyzed frame, its boxes and age; old boxes are not painted
over the live camera frame. Startup stages print immediately to the terminal.

If startup appears stuck, run from inside `QR_Scanning`:

```bash
python -u ncnn_detector.py --device /dev/video0 --diagnostics
# Isolate model loading/inference from the camera and desktop display:
python -u ncnn_detector.py --self-test --threads 1
```

The preview requires a graphical desktop/display connection. The self-test does
not open a camera or window. Report the last printed startup stage if either
command stops making progress.

```bash
# Small/distant target search; slower and more permissive than defaults.
python QR_Scanning/ncnn_detector.py --camera 0 --tile-sizes 384 192 --tiles-per-frame 8 --confidence 0.25

# Every tile and lighting variant on one recorded image; save an annotated result.
python QR_Scanning/ncnn_detector.py --image frame.png --exhaustive --no-preview --output annotated.png

# Preview every frame in a video; processing can be slower than real time.
python QR_Scanning/ncnn_detector.py --video recording.mkv
```

## What improves distant and poorly lit targets

- A whole-frame overview finds large nearby codes.
- Native overlapping crops of 384 and 192 pixels retain small targets that
  disappear when the entire 1080p frame is reduced to the model's input size.
  Both tile grids cover the frame edges. Scales are interleaved and their cursor
  advances regardless of successful detections.
- An independent edge localizer proposes additional small crops. It cannot emit
  a detection by itself: NCNN must find a QR-looking region in each crop.
- Every visited crop gets an untouched inference pass. Additional CLAHE, shadow
  normalization, and gamma variants rotate between processed frames. With
  `--exhaustive`, all three run on every crop. `--no-lighting` disables them.
- Cross-view suppression merges duplicate and contained crop detections.

Default confidence is 0.6. Lower thresholds can expose weaker candidates and
also increase false positives (the synthetic clutter control produced one at
0.25). Inspect actual outdoor backgrounds when choosing a threshold.
`--tiles-per-frame` controls throughput versus sweep time; the status line shows
how many tiles were visited out of the total. A blank result from a partial sweep
does not prove the entire scene is QR-free. `--exhaustive` searches the whole
current image but can be too slow for a moving drone.

## A3 and range

The working example is a 280 mm QR square **including the quiet zone** on an A3
sheet, captured at 1920×1080. The actual QR size, module count, lens field of view,
focus, motion and illumination determine range. A3 paper alone does not specify
the number of pixels available for the symbol.

For a frontal target, estimate the entire QR width in pixels using:

```text
outer_pixels = image_width_px * qr_width_m / (2 * distance_m * tan(horizontal_fov / 2))
pixels_per_module = outer_pixels / (symbol_modules + 8)
```

The extra eight modules account for the required four-module quiet zone on
each side ([DENSO WAVE guidance](https://www.qrcode.com/en/howto/code.html)).
Keep the payload short to allow a lower module count. Use the actual printed
version when setting `scanner.py --qr-modules`; the default 29 is an estimate.

```bash
python QR_Scanning/qualification.py optical --modules 29 --qr-width-mm 280 --distance-m 10 --outer-pixels 45
```

Use native camera resolution, focus at the target distance, and compare exposure
settings on recorded flight frames. Longer exposure helps brightness but can
destroy modules through motion blur. Avoid reflections from glossy covers.
Use supported camera controls (`--exposure-us` on Linux, backend-specific
`--exposure` on desktop) after measuring their effect, rather than assuming one
setting handles sun, shade and motion.

Upscaling and contrast enhancement cannot restore detail lost to undersampling,
clipped glare, darkness or severe blur. Detection can continue after decoding
fails; neither script promises a maximum distance or success in all lighting.
Qualify the system with printed A3 targets, independent payload labels, negative
backgrounds, several distances, and sun/shade/backlight/motion recordings using
the existing qualification tools.

## Validation

`tests/test_ncnn_detector.py` checks real inference, safe empty outputs, coordinate
mapping, input preservation, NMS, model integrity and the imshow path. Real NCNN
tests skip when its optional runtime/models are unavailable.
`validation/synthetic_recovery.json` records before/after decoder checks and
NCNN target-center detection on synthetic images. These are software regression
checks, not an outdoor range measurement.
