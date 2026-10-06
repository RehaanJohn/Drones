"""Image-only prototype discovery and adapter for the existing QR scanner."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np


def discover(frame, mode='color', min_area=80):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if mode == 'color':
        # ONLY the orange simulator cache. Replace with representative trained detector.
        mask = cv2.inRange(hsv, np.array([3, 120, 65]), np.array([25, 255, 255]))
    elif mode == 'sheet':
        mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 65, 255]))
    else:
        raise ValueError('Discovery mode must be color or sheet')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5 or not 0.2 < w/h < 8 or area/(w*h) < 0.3:
            continue
        boxes.append((area, x, y, w, h))
    if not boxes:
        return None
    _, x, y, w, h = max(boxes)
    # Cache body bottom touches ground; white sheet instead projects onto cache top.
    u, v = x+w/2, y+h if mode == 'color' else y+h/2
    return float(u), float(v), [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]


class QRBackend:
    def __init__(self, scanner_path='', model_dir=''):
        self.scanner = None
        self.detector = cv2.QRCodeDetector()
        self.name = 'OpenCV QRCodeDetector'
        if scanner_path:
            spec = importlib.util.spec_from_file_location('addc_existing_qr_scanner', scanner_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.scanner = module
            if model_dir:
                module.MODEL_DIR = Path(model_dir).expanduser()
            # Flight startup never downloads files; explicitly provision them beforehand.
            if (hasattr(cv2, 'wechat_qrcode') and
                    all((module.MODEL_DIR/name).is_file() for name in module.MODEL_FILES)):
                self.detector = module.create_detector()
                self.name = 'Existing scanner WeChat + cropped fallbacks'

    def detect(self, frame):
        if self.name.startswith('Existing'):
            texts, points = self.detector.detectAndDecode(frame)
        else:
            _, texts, points, _ = self.detector.detectAndDecodeMulti(frame)
        observations = []
        if points is not None:
            for i, region in enumerate(points):
                payload = texts[i] if i < len(texts) else ''
                if not payload and self.scanner:
                    crop = self.scanner.crop_qr_region(frame, [region])
                    payload = self.scanner.fallback_decode(crop) or ''
                points2 = np.asarray(region).reshape(-1, 2)
                observations.append((points2.mean(axis=0), points2.tolist(), payload))
        if not observations:
            # Can center on paper before the QR pattern is sufficiently resolved.
            sheet = discover(frame, 'sheet', min_area=150)
            if sheet:
                u, v, corners = sheet
                observations.append(((u, v), corners, ''))
        return observations
