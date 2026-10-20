"""UVC discovery and capability-aware V4L2 configuration (no shell commands)."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import cv2


def ctl(device, *arguments):
    try:
        result = subprocess.run(['v4l2-ctl', '-d', str(device), *arguments],
                                capture_output=True, text=True, timeout=3)
    except FileNotFoundError as exc:
        raise RuntimeError('Install v4l-utils to inspect/configure USB webcams: sudo apt install v4l-utils') from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'V4L2 control timed out for {device}') from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f'V4L2 failed for {device}')
    return result.stdout


def parse_controls(text):
    controls = {}
    current = None
    for line in text.splitlines():
        match = re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\(([^)]+)\)\s*:\s*(.*)', line)
        if match:
            name, kind, fields = match.groups()
            canonical = {'auto_exposure': 'exposure_auto', 'exposure_time_absolute': 'exposure_absolute',
                         'exposure_dynamic_framerate': 'exposure_auto_priority'}.get(name, name)
            current = controls[canonical] = {'name': name, 'type': kind, 'menu': {}, 'inactive': 'inactive' in fields}
            current.update({key: int(value) for key, value in
                            re.findall(r'(min|max|step|default|value)=(-?\d+)', fields)})
        elif current is not None:
            item = re.match(r'\s*(-?\d+):\s*(.*)', line)
            if item:
                current['menu'][int(item[1])] = item[2]
    return controls


def parse_modes(text):
    modes, fourcc, size = [], None, None
    for line in text.splitlines():
        match = re.search(r"\[\d+\]:\s*'(.{4})'", line)
        if match:
            fourcc = match[1]
        match = re.search(r'Size: Discrete (\d+)x(\d+)', line)
        if match:
            size = tuple(map(int, match.groups()))
        match = re.search(r'\(([\d.]+) fps\)', line)
        if match and fourcc and size:
            modes.append({'fourcc': fourcc, 'width': size[0], 'height': size[1], 'fps': float(match[1])})
    return modes


def inspect(device):
    info = ctl(device, '--all')
    formats = ctl(device, '--list-formats-ext')
    controls = ctl(device, '--list-ctrls-menus')
    # Device-node numbers are not an identity. Prefer persistent USB ID; otherwise bus identity.
    bus = re.search(r'Bus info\s*:\s*(.+)', info)
    identity = str(device) if '/by-id/' in str(device) else (bus[1].strip() if bus else str(Path(device).resolve()))
    usb = {}
    node = Path('/sys/class/video4linux') / Path(device).resolve().name / 'device'
    for parent in (node.resolve(), *node.resolve().parents):
        if (parent / 'idVendor').is_file():
            for field in ('idVendor', 'idProduct', 'product', 'manufacturer', 'speed'):
                try:
                    usb[field] = (parent / field).read_text().strip()
                except OSError:
                    pass
            break
    return {'device': str(device), 'identity': identity, 'usb': usb,
            'key': hashlib.sha256(identity.encode()).hexdigest()[:16],
            'info': info, 'formats_raw': formats, 'modes': parse_modes(formats),
            'controls': parse_controls(controls)}


def list_devices():
    paths = sorted(Path('/dev/v4l/by-id').glob('*')) + sorted(Path('/dev').glob('video*'))
    seen, devices = set(), []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            entry = inspect(path)
        except RuntimeError:
            continue
        if not entry['modes'] or re.search(r'\b(infrared|IR)\b', entry['info']):
            continue
        devices.append(entry)
    return devices


def choose_mode(profile, width, height, fps, fourcc=None):
    modes = [m for m in profile['modes'] if m['width'] == width and m['height'] == height
             and (not fourcc or m['fourcc'] == fourcc)]
    if not modes:
        raise RuntimeError(f'No advertised {width}x{height} {fourcc or ""} mode; use --inspect-camera. '
                           'Resolution will not be silently reduced.')
    # Prefer a mode satisfying requested FPS; use MJPEG as the initial bandwidth-friendly format.
    return min(modes, key=lambda m: (m['fps'] + .1 < fps, m['fourcc'] != 'MJPG', abs(m['fps'] - fps)))


def set_control(device, name, value, controls):
    control = controls.get(name)
    if control is None or control.get('inactive'):
        return {'supported': False, 'requested': value}
    value = int(round(value))
    if not control.get('min', value) <= value <= control.get('max', value):
        raise RuntimeError(f'{name}={value} outside supported range')
    step = control.get('step', 1)
    if step and (value - control.get('min', 0)) % step:
        raise RuntimeError(f'{name}={value} does not match supported step {step}')
    ctl(device, f'--set-ctrl={control.get("name", name)}={value}')
    actual = parse_controls(ctl(device, '--list-ctrls-menus')).get(name, {}).get('value')
    return {'supported': True, 'requested': value, 'actual': actual, 'validated': actual == value}


def configure(cap, profile, options):
    device = profile['device']
    mode = choose_mode(profile, options['width'], options['height'], options['fps'], options.get('fourcc'))
    for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode['fourcc'])),
                        (cv2.CAP_PROP_FRAME_WIDTH, mode['width']), (cv2.CAP_PROP_FRAME_HEIGHT, mode['height']),
                        (cv2.CAP_PROP_FPS, min(mode['fps'], options['fps']))):
        cap.set(prop, value)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    controls = profile['controls']
    applied = {}
    exposure = options.get('exposure_us')
    if options.get('adaptive_exposure') and exposure is None:
        exposure = 1000
    if exposure is not None:
        manual = next((key for key, label in controls.get('exposure_auto', {}).get('menu', {}).items()
                       if 'manual' in label.lower()), None)
        if manual is not None and 'exposure_absolute' in controls:
            previous_mode = controls['exposure_auto'].get('value')
            applied['exposure_auto'] = set_control(device, 'exposure_auto', manual, controls)
            if applied['exposure_auto'].get('validated'):
                controls = parse_controls(ctl(device, '--list-ctrls-menus'))
                try:
                    applied['exposure_absolute'] = set_control(device, 'exposure_absolute', exposure / 100, controls)
                except RuntimeError as exc:
                    applied['exposure_absolute'] = {'supported': True, 'validated': False, 'error': str(exc)}
                if not applied['exposure_absolute'].get('validated') and previous_mode is not None:
                    applied['exposure_auto_restored'] = set_control(device, 'exposure_auto', previous_mode, controls)
        else:
            applied['exposure_absolute'] = {'supported': False, 'requested_us': exposure}
    for option, name in (('autofocus', 'focus_automatic_continuous'), ('focus', 'focus_absolute'), ('gain', 'gain')):
        if option == 'autofocus' and name not in controls:
            name = 'focus_auto'
        if options.get(option) is not None:
            applied[name] = set_control(device, name, options[option], controls)
            controls = parse_controls(ctl(device, '--list-ctrls-menus'))
    # Disable frame-rate-changing auto exposure and relative digital zoom where exposed.
    for name, value in [('exposure_auto_priority', 0), ('zoom_continuous', 0)]:
        if name in controls:
            applied[name] = set_control(device, name, value, controls)
    if 'zoom_absolute' in controls and 'min' in controls['zoom_absolute']:
        applied['zoom_absolute'] = set_control(device, 'zoom_absolute', controls['zoom_absolute']['min'], controls)
    profile['unavailable_controls'] = [name for name in ('exposure_absolute', 'focus_absolute', 'gain', 'zoom_absolute')
                                       if name not in controls]
    profile['controls'] = controls
    profile.update(selected_mode=mode, applied_controls=applied,
                   reported_fps=cap.get(cv2.CAP_PROP_FPS),
                   reported_fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)))
    return profile


def exposure_step(control, current, mean, max_us=2000):
    """Conservative QR-region adjustment, bounded by the calibrated motion ceiling."""
    factor = 1.2 if mean < 80 else .8 if mean > 180 else 1
    step = max(1, control.get('step', 1))
    minimum = control.get('min', 1)
    maximum = min(control.get('max', current), max_us // 100)
    if maximum < minimum:
        return current
    value = int(round((current * factor - minimum) / step)) * step + minimum
    maximum = minimum + ((maximum - minimum) // step) * step
    return max(minimum, min(maximum, value))


def save_profile(directory, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (profile['key'] + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path
