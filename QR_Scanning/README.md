# Terminal QR scanner

Run from the repository root using the same virtual environment as the MCP server:

```bash
.venv/bin/python -m pip install -r QR_Scanning/requirements.txt
.venv/bin/python QR_Scanning/scanner.py
```

The scanner opens camera 0, prints decoded QR payloads with timestamps to the
terminal, and appends each scan to `QR_Scanning/qr_scans.jsonl`. It uses no Flask,
preview windows, web streaming, or `imshow`. Stop with Ctrl+C; the camera and
output file are closed. Existing scans are preserved across runs.

Choose a camera or output file:

```bash
.venv/bin/python QR_Scanning/scanner.py --camera 1 --output /tmp/qr_results.jsonl
```

Each UTF-8 JSON Lines record contains `time` and `data`. JSON escaping preserves
multiline QR payloads as one record per line. Results are flushed immediately.
Repeated payloads are suppressed for two seconds, including when multiple codes
are visible. The same payload can be recorded again after that interval.

Model files resolve relative to the scanner script, so launching from another
directory works. Missing WeChat QR model files are downloaded on first use.
Detection retains the cropped, upscaled, and contrast-enhanced fallback decoder.

On macOS, enable camera access for your terminal in System Settings > Privacy &
Security > Camera. The scanner uses OpenCV's
[AVFoundation backend](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).
It falls back to automatic backend selection if opening the camera fails.

Run tests without a camera:

```bash
.venv/bin/python -m unittest discover -s QR_Scanning/tests -v
```
