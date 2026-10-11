"""Discover Windows cameras and check frame delivery with a bounded probe."""
import os
from pathlib import Path
import subprocess
import sys

# Avoid expensive GPU transforms during MSMF initialization; allow explicit override.
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')
import cv2


def list_cameras(backend='auto'):
    from cv2_enumerate_cameras import enumerate_cameras
    choices = {'auto': (cv2.CAP_MSMF, cv2.CAP_DSHOW),
               'any': (cv2.CAP_MSMF, cv2.CAP_DSHOW),
               'msmf': (cv2.CAP_MSMF,), 'dshow': (cv2.CAP_DSHOW,)}
    if backend not in choices:
        raise RuntimeError('Windows camera discovery supports auto, any, msmf, or dshow')
    return [camera for api in choices[backend] for camera in enumerate_cameras(api)]


def probe_camera(index, backend, timeout=5):
    # Native open/read calls may hang. A child process can be terminated safely
    # without leaving a capture thread holding a failed device in this process.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--probe', str(index), str(backend)],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and result.stdout.strip() == 'FRAME_OK'
    except subprocess.TimeoutExpired:
        return False


def find_working_camera(backend='auto', name=None, diagnostics=False):
    cameras = list_cameras(backend)
    if name:
        cameras = [c for c in cameras if name.casefold() in c.name.casefold()]
    else:
        # Face-authentication IR cameras are not the intended QR video source.
        cameras = [c for c in cameras if 'ir' not in c.name.lower().split()
                   and 'infrared' not in c.name.casefold()]
        cameras.sort(key=lambda c: 'rgb' not in c.name.casefold())
    for camera in cameras:
        if diagnostics:
            print(f'Testing {camera.name}: index {camera.index}, {cv2.videoio_registry.getBackendName(camera.backend)}',
                  file=sys.stderr, flush=True)
        if probe_camera(camera.index, camera.backend):
            return camera
    description = ', '.join(f'{c.name} (index {c.index}, {cv2.videoio_registry.getBackendName(c.backend)})'
                            for c in cameras) or 'no matching video devices'
    raise RuntimeError(f'No camera delivered a frame. Discovered: {description}. '
                       'Run --list-cameras to see current device indices.')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--probe':
        sys.exit(2)
    cap = cv2.VideoCapture(int(sys.argv[2]), int(sys.argv[3]))
    success = False
    try:
        if cap.isOpened():
            ok, frame = cap.read()
            success = bool(ok and frame is not None and frame.size)
    finally:
        cap.release()
    if success:
        print('FRAME_OK', flush=True)
    sys.exit(0 if success else 1)
