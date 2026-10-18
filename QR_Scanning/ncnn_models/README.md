# Model provenance

These are the WeChat QR-specific SSD detection weights converted to NCNN by
[EdVince/QRCode-NCNN](https://github.com/EdVince/QRCode-NCNN). No super-resolution
or decoder weights are used by `ncnn_detector.py`. These weights have not been
retrained or certified for this project's camera, A3 print, or outdoor conditions.

Pinned revision: `0e006ebf71ad977af7c6e0e86da038b242792c4e`.
Source directory: `Project_android/ncnn-android-nanodet/app/src/main/assets/`.

| File | SHA-256 |
| --- | --- |
| detect.param | f136b8beb1d03505735e527a2ba6ff9247cd9a0b70d15f3ba9cabfaf227c32e3 |
| detect.bin | 11dc5df644206da6ccc0d34aa761523255fa441cd163fd4e70e12ad033d352a5 |

Runtime checks these hashes before loading. The input is one grayscale channel
normalized to 0–1, named `data`; the `detection_output` rows contain class, score,
and normalized bounding coordinates. The graph confidence filter is adjusted
in memory to match `--confidence`; the files remain unchanged.

Original WeChat work is copyright Tencent. The conversion repository refers
users to the original license. `LICENSE-WeChat.txt` is the notice from the
[OpenCV WeChat QR module](https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/LICENSE),
including its upstream third-party notices. NCNN runtime itself uses BSD-3-Clause.

Provisioning is explicit: `python QR_Scanning/ncnn_detector.py --download-models`.
No network request is made during normal inference startup.
