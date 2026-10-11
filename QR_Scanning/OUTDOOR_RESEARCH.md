# Outdoor QR detection and the A3 / 10 m target

Research and local evaluation: 2026-10-11. Known setup: 1080p camera, QR printed
on A3, desired reading distance at least 10 m. Camera model, horizontal field of
view, printed QR width, payload/version, motion, and outdoor frames have not been
supplied. The range estimates below are conditional, not measured performance.

## Problems in the original scanner

1. **No capture resolution was requested or checked.** A 1080p-capable camera can
   open at a lower driver default. For example, 640-wide capture provides one
   third of the samples across compared with 1920-wide capture at the same FOV.
2. **The neural detector scaled both dimensions by 0.5.** It localized codes with
   a quarter of the pixels. Decoding still used source pixels, but only after
   localization succeeded. [OpenCV scale-factor API](https://docs.opencv.org/4.x/d5/d04/classcv_1_1wechat__qrcode_1_1WeChatQRCode.html)
3. **The fallback expected undecoded corners from WeChat.** The implementation
   exports points from successful decodes, so the old loop normally never ran.
   The replacement independently calls `QRCodeDetector.detectMulti` to locate
   candidates for recovery. [OpenCV source](https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/src/wechat_qrcode.cpp)
4. **Acquisition paused during decoding.** Extra recovery work could leave stale
   frames in the driver's queue. The new capture thread keeps only the latest
   frame. Decoders and the CLAHE object are reused instead of reconstructed.

## The pixel budget at 10 m

Let `W` be the square's physical width **including its quiet zone**, `D` the
distance, `H` the frame width in pixels, and `theta` the horizontal angular FOV.
For a front-facing code near the center of a rectilinear image:

```text
scene width       = 2 * D * tan(theta / 2)
QR pixels across  = H * W / scene width
modules per side  = 17 + 4 * QR version
pixels per module = QR pixels across / (modules per side + 8)
```

The first two expressions follow from
[Edmund Optics' field-of-view geometry](https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/).
DENSO WAVE documents the [version/module counts](https://www.qrcode.com/en/about/version.html)
and [four-module quiet zone on each side](https://www.qrcode.com/en/howto/code.html).

Assuming the entire 297 mm short dimension of A3 is used for the square including
its border, with 1920-wide capture at 10 m:

| Horizontal FOV | Entire QR width, pixels | Version 1: pixels/module | Version 3: pixels/module |
| --- | ---: | ---: | ---: |
| 90 degrees | 28.5 | 0.98 | 0.77 |
| 70 degrees | 40.7 | 1.40 | 1.10 |
| 60 degrees | 49.4 | 1.70 | 1.33 |
| 45 degrees | 68.8 | 2.37 | 1.86 |
| 30 degrees | 106.4 | 3.67 | 2.88 |
| 25 degrees | 128.6 | 4.43 | 3.48 |

Version 1 has 21 modules across the symbol; Version 3 has 29. The table includes
eight extra modules for borders. Printer margins, a smaller QR on the page,
oblique viewing, blur, compression, and movement reduce usable detail further.
Do not substitute a manufacturer's diagonal FOV for horizontal FOV.

For this project, **3–4 captured pixels/module is a practical starting design
target**, with more margin for motion. It is neither a standard-defined minimum
nor a decoder guarantee. Clean synthetic codes can decode below it. DENSO's
recommendation of four **printer dots** per module is a separate printing rule;
their guide also notes that lens/scanner resolution limits readable module size.
[DENSO module-size guidance](https://www.qrcode.com/en/howto/cell.html)

Calculated consequences:

- For 4 pixels/module at 10 m, Version 1 on A3 needs approximately **27.6-degree
  HFOV or narrower** at 1080p; Version 3 needs about **21.8 degrees or narrower**.
- At 70-degree HFOV, a Version 1 print would need to be about **0.85 m wide**
  including its border for that same sampling target.
- True 3840-wide capture doubles sampling at unchanged FOV. At 70 degrees it
  still gives only about **2.8 pixels/module** for Version 1 on A3.
- Optical zoom changes target sampling. Digital enlargement cannot add missing
  sensor detail. Lens choice requires the actual sensor dimensions, mount, and
  ability to focus at the working distance.

The unknown camera HFOV is therefore central to the 10 m requirement. A correctly
focused narrow lens may provide a larger improvement than further color processing.

## Techniques evaluated

| Technique | Decision and limitation |
| --- | --- |
| Native pixels, overlapping tiles | Implemented: ZXing sees every pixel, the WeChat overview stays small, and 640px overlapping tiles use scale 1.0. Protects distant codes from global localization downsampling. |
| Multiple decoders | Implemented: ZXing with rotation/inversion/multiscale search, WeChat when a view fails, and independent OpenCV candidate recovery. Different decoders have different failure modes. |
| CLAHE | Implemented as a retry for local contrast; may also amplify noise. |
| Shadow normalization | Implemented as a retry: divide by a blurred illumination estimate, then CLAHE. A heuristic for slowly changing light; hard shadow edges remain difficult. |
| Adaptive thresholding | Implemented as a retry for varying illumination. Its 31px window is a starting value; it is not universally optimal across module sizes. |
| Mild sharpening | Implemented as a retry for soft but still resolved edges. Cannot reliably reverse severe defocus/motion and can amplify artifacts. |
| Color channels | Implemented as a retry: enhance the channel with greatest robust intensity spread. May help color casts; background texture can affect this heuristic. |
| Crop, pad, and 2x upscale | Implemented on independently located candidates, with bounded output size. Can suit a decoder's scale; cannot create optical detail. |
| Perspective correction | Existing decoders handle QR geometry. Another full-frame warp is avoided; severe foreshortening still destroys sampling on one axis. |
| Gamma, global equalization, white balance | Not stacked on every frame. Remapping does not restore clipping, and global contrast can hurt mixed lighting. Prefer the raw view followed by targeted local retries. |
| Denoising and morphology | Not enabled globally. Blur, erosion, and dilation can merge or erase tiny modules. Tune only for demonstrated defects. |
| Learned super-resolution | WeChat already has a QR-oriented SR model. Generic SR cannot replace enough native detail and validated decoding. |
| Multi-frame SR/deblurring | Deferred: requires accurate registration/exposure consistency. Tiny features, motion, and rolling shutter complicate alignment. Requires real sequences and known payloads for evaluation. |
| HDR/exposure bracketing | Deferred: another exposure must actually capture the missing information, and moving exposures require alignment. No correction can reconstruct clipped contrast from a single image. |
| Custom learned detector | Consider if field images have clearly resolved modules but localization still fails. Requires representative training data; cannot fix unreadable sampling. |

Primary references for these choices:
[ZXing Python reader options](https://github.com/zxing-cpp/zxing-cpp/blob/master/wrappers/python/zxing.cpp),
[OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html),
[adaptive thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html),
and [WeChat SR implementation](https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/src/scale/super_scale.cpp).
The implementation decisions are engineering judgments for this scanner; they
are not claims that every technique has improved this particular outdoor camera.

## Acquisition and printing priorities

**Focus/exposure:** inspect an unscaled crop at the target distance. If adjacent
modules smear together, correct focus or exposure first. Autofocus can select the
background; lock a verified focus if working distance permits. Shorter exposure
reduces motion blur, while excessive gain increases noise. Basler describes these
tradeoffs, but its numeric controls are not interchangeable with another camera.
[Acquisition guidance](https://docs.baslerweb.com/optimizing-image-quality)

An engineering estimate is `blur_px = image_speed_px_per_second * exposure_seconds`.
At 600 pixels/second, 1/1000 s produces approximately 0.6px blur; 1/60 s produces
10px. Select exposure from actual movement and image brightness. Frame rate and
exposure duration are different controls. The CLI intentionally accepts explicit
raw exposure values and reports readbacks rather than choosing a universal preset.
[OpenCV backend-dependent camera properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)

**Glare:** use flat matte material, avoid shiny lamination where practical, and
adjust illumination/viewing angle without extreme foreshortening. A lens polarizer
can help some reflections, depending on angle/material, at a light-loss cost.
Cross-polarization works best with controlled illumination; sunlight cannot be
treated as your own polarized light source. Test this setup before buying a filter.
[Polarization guidance](https://www.edmundoptics.com/knowledge-center/application-notes/illumination/successful-light-polarization-techniques/)

**QR design:** use as much of A3's short side as practical, black on white, without
logos, with the full quiet zone. Encode a short ID such as `DRONE-01` with a lookup
when possible. Higher versions make cells smaller at fixed physical width.
Stronger error correction can require a denser version for a fixed payload. M is
a reasonable starting trial for clean prints; evaluate higher redundancy when
damage is expected. Redundancy cannot make unresolved finder patterns readable.
[DENSO error correction](https://www.qrcode.com/en/about/error_correction.html)

## Measured comparison

`benchmark.py` generates reproducible 1920x1080 scenes with clutter, `DRONE-01`,
exactly four quiet-zone modules, area resampling to 41/60/90/120px, and noise seed
42. Eight conditions cover clean, shadow, low contrast, color cast, defocus,
motion, clipped glare, and perspective. Local versions: Python 3.13.0, OpenCV
4.14.0, ZXing-C++ 3.1.1, Windows 11.

| QR width including border | Original: decoded / 8 | Updated: decoded / 8 |
| --- | ---: | ---: |
| 41px | 0 | 4 |
| 60px | 5 | 5 |
| 90px | 4 | 7 |
| 120px | 6 | 7 |
| Total | **15 / 32** | **23 / 32** |

The additional clutter-only control produced no payload. One control is not an
estimate of the real-world false-positive rate. Both algorithms received identical
full-resolution frames, so this does not measure gains from correcting an actual
low-resolution camera default.

A separate run with tile lighting variants disabled (`--no-lighting`) decoded
**21/32**, versus **23/32** with them enabled. CLAHE recovered the additional 41px
and 60px shadow cases. This ablation still leaves CLAHE in candidate crop recovery;
it isolates the scheduled tile variants, not every use of contrast enhancement.
The other variants are available recovery options, not independently demonstrated
improvements in these default eight-pass results.

The original baseline uses 0.5-scale WeChat; its unreachable fallback adds nothing
on these images. The new scanner gets up to eight rolling scans of each still,
versus one deterministic original scan. All successful new cases decoded on scan
one. Default eight-pass failures cover CLAHE/shadow sweeps; `--passes 20` schedules
all five variants across a full 1080p frame. Codes are near an early tile seam,
so this does not measure worst-position latency or varied subpixel phases.

Initial processing times were about 17–36ms per original scan, 56–239ms per
successful updated scan, and 1.6–1.8s for eight-pass updated failures. These exclude
capture, initialization, scan-interval waits, and file output. The implementation
trades additional computation for recovery; it is not a blanket speed improvement.
The deployment computer may be slower.

The clean 41px recovery is encouraging, but 41px blur, motion, glare, and perspective
still failed. All deliberately severe glare cases failed. **This does not validate
reliable 10 m outdoor operation.** Regression tests additionally cover real
decoders, Unicode, small codes, independent candidates, tile coverage, frame
immutability, capture lifecycle, resolution settings, and logging cooldown.

## Field validation

1. Record camera model/native mode/HFOV, actual QR width, payload/version/error
   correction, focus, and exposure.
2. Capture native frames at 2/4/6/8/10/12 m in sun, shade, backlight, and mixed
   shadow. Begin stationary, then test expected movement and viewing angles.
3. Measure QR pixel width and inspect adjacent modules at 10 m. Compare with the
   range calculation before spending time on more filters.
4. Benchmark those exact frames with their known payload, then compare
   `--no-lighting` and `--no-zxing`. Inspect unexpected strings as well as misses.
5. Measure live time to first correct decode and repeated-trial success across
   frame positions. JSONL counts are not per-frame detection rates because of
   the two-second duplicate cooldown.
6. Set an acceptance criterion, e.g. 95% correct within one second at 10 m under
   specified conditions. Change optics/resolution/print size when detail is
   insufficient; tune detection using failed frames when detail is present.
