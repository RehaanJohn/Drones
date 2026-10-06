#!/usr/bin/env python3
"""Synthetic-camera closed-loop smoke check, NOT PX4/Gazebo flight validation.

Scene truth is used only to render images. The mission receives image-derived
observations and aircraft telemetry. Plant motion is an idealised bounded integrator.
"""
import math
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros2_ws/src/addc_mission'))
from addc_mission.geometry import Pose, rotation, optical_to_body, project_pixel
from addc_mission.mission import Mission, Estimate, DownObservation
from addc_mission.vision import discover, QRBackend


def image_points(points, pose, mount, tilt, k):
    r = rotation(pose.quaternion)
    origin = np.array(pose.position)+r@np.array(mount)
    coordinates = (optical_to_body(tilt).T@r.T@(np.array(points)-origin).T).T
    if np.any(coordinates[:, 2] <= 0.05):
        return None
    pixels = (k@coordinates.T).T
    return (pixels[:, :2]/pixels[:, 2:3]).astype(np.float32)


def render(pose, camera, k, texture, center=(12, 3)):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = (45, 80, 50)
    x, y = center
    mount = (0.20, 0, -0.05) if camera == 'discovery' else (0, 0, -0.10)
    tilt = math.radians(25) if camera == 'discovery' else math.pi/2
    # Cuboid faces; colour detector sees ground-contact edge, not provided location.
    for face in [
        [(x-.325, y-.25, 0), (x-.325, y+.25, 0), (x-.325, y+.25, .6), (x-.325, y-.25, .6)],
        [(x-.325, y+.25, 0), (x+.325, y+.25, 0), (x+.325, y+.25, .6), (x-.325, y+.25, .6)],
        [(x-.325, y-.25, .6), (x+.325, y-.25, .6), (x+.325, y+.25, .6), (x-.325, y+.25, .6)],
    ]:
        pixels = image_points(face, pose, mount, tilt, k)
        if pixels is not None:
            cv2.fillConvexPoly(frame, pixels.astype(np.int32), (10, 65, 255))
    paper = [(x+.21, y+.1485, .604), (x+.21, y-.1485, .604),
             (x-.21, y-.1485, .604), (x-.21, y+.1485, .604)]
    pixels = image_points(paper, pose, mount, tilt, k)
    if pixels is not None:
        cv2.fillConvexPoly(frame, pixels.astype(np.int32), (255, 255, 255))
    code = [(x+.125, y+.125, .605), (x+.125, y-.125, .605),
            (x-.125, y-.125, .605), (x-.125, y+.125, .605)]
    pixels = image_points(code, pose, mount, tilt, k)
    if pixels is not None:
        source = np.float32([[0, 0], [1023, 0], [1023, 1023], [0, 1023]])
        h = cv2.getPerspectiveTransform(source, pixels)
        warped = cv2.warpPerspective(texture, h, (1920, 1080), flags=cv2.INTER_NEAREST)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, pixels.astype(np.int32), 255)
        frame[mask > 0] = warped[mask > 0]
    return frame, mount, tilt


def main():
    mission = Mission()
    position = np.array([0.0, 0.0, .24])
    yaw, armed, mode = 0.0, False, 'POSCTL'
    focal = 1920/(2*math.tan(math.radians(60)/2))
    k = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1]])
    texture = cv2.imread(str(ROOT/'sim/models/intelligence_cache/materials/textures/qr.png'))
    scanner = ROOT.parent/'QR_Scanning/scanner.py'
    backend = QRBackend(str(scanner), '/nonexistent-addc-smoke-weights')
    mission.start(0, position, True, False)
    previous = mission.state
    for index in range(1, 2401):
        t = index*.1
        pose = Pose(t, tuple(position), (0, 0, math.sin(yaw/2), math.cos(yaw/2)))
        if index % 2 == 0 and mission.state in ('DISCOVER', 'VIEWPOINT', 'APPROACH', 'ACQUIRE_DOWN', 'SCAN'):
            frame, mount, tilt = render(pose, 'discovery', k, texture)
            detected = discover(frame)
            if detected:
                try:
                    point, uncertainty = project_pixel(detected[0], detected[1], k, pose, mount, tilt)
                    mission.estimate(Estimate(t, point, uncertainty), t)
                except ValueError:
                    pass
            if mission.state in ('APPROACH', 'ACQUIRE_DOWN', 'SCAN'):
                frame, mount, tilt = render(pose, 'down', k, texture)
                results = backend.detect(frame)
                results.sort(key=lambda r: not bool(r[2]))
                if results:
                    center, _, payload = results[0]
                    try:
                        point, _ = project_pixel(*center, k, pose, mount, tilt, plane_z=.6)
                        mission.downward(DownObservation(t, point, payload), t)
                    except ValueError:
                        pass
        command = mission.tick(t, tuple(position), yaw, True, armed, mode)
        if command.mode:
            mode = command.mode
        if command.arm:
            armed = True
        if armed and command.publish and command.position:
            goal = np.array(command.position)
            delta = goal[:2]-position[:2]
            length = np.linalg.norm(delta)
            position[:2] += delta*min(1, .3/length) if length else 0
            position[2] += np.clip(goal[2]-position[2], -.05, .1)
            error = (command.yaw-yaw+math.pi) % (2*math.pi)-math.pi
            yaw += np.clip(error, -.07, .07)
        if mode == 'AUTO.LAND':
            position[2] = max(.24, position[2]-.05)
            if position[2] <= .24:
                armed = False
        if mission.state != previous:
            print(f'{t:5.1f}s {previous} -> {mission.state}: {mission.reason}', flush=True)
            previous = mission.state
        if mission.state in ('DONE', 'ABORTED'):
            break
    success = mission.state == 'DONE' and mission.result == '07' and mission.view_index == 0
    print(f'Synthetic smoke {"PASS" if success else "FAIL"}: digits={mission.result!r}, '
          f'fallback viewpoints={mission.view_index}; no ROS, PX4, or Gazebo was run.')
    return int(not success)


if __name__ == '__main__':
    raise SystemExit(main())
