"""Camera-to-body FLU-to-local ENU projection. All angles are radians."""
from bisect import bisect_left
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Pose:
    stamp: float
    position: tuple
    quaternion: tuple  # x, y, z, w; body FLU -> local ENU


def rotation(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError('Invalid quaternion')
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def optical_to_body(tilt):
    # Camera optical: x right, y down, z forward. Body: x forward, y left, z up.
    c, s = math.cos(tilt), math.sin(tilt)
    return np.array([[0, -s, c], [-1, 0, 0], [0, -c, -s]])


def project_pixel(u, v, k, pose, mount, tilt, plane_z=0.0,
                  minimum_depression_deg=8.0, max_range=80.0):
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    ray = np.linalg.solve(k, np.array([u, v, 1.0]))
    r = rotation(pose.quaternion)
    ray = r @ optical_to_body(tilt) @ ray
    ray /= np.linalg.norm(ray)
    origin = np.asarray(pose.position, dtype=float) + r @ np.asarray(mount)
    if not np.isfinite(origin).all() or not np.isfinite(ray).all():
        raise ValueError('Nonfinite projection')
    if ray[2] >= -math.sin(math.radians(minimum_depression_deg)):
        raise ValueError('Ray too close to or above horizon')
    distance = (plane_z-origin[2])/ray[2]
    if distance <= 0 or distance > max_range:
        raise ValueError('Intersection outside reliable range')
    point = origin + ray * distance
    # Conservative heuristic, not a calibrated covariance: 1 degree angular error.
    uncertainty = max(0.15, distance * math.radians(1.0) / abs(ray[2]))
    return tuple(point), uncertainty


class PoseBuffer:
    """Interpolate capture-time pose; reject extrapolation beyond tolerance."""
    def __init__(self, duration=5.0, tolerance=0.15):
        self.duration, self.tolerance = duration, tolerance
        self.poses = []

    def append(self, pose):
        if self.poses and pose.stamp < self.poses[-1].stamp:
            self.poses.clear()  # Simulator reset / clock jump.
        self.poses.append(pose)
        self.poses = [p for p in self.poses if p.stamp >= pose.stamp-self.duration]

    def at(self, stamp):
        if not self.poses:
            raise ValueError('No aircraft poses')
        index = bisect_left([p.stamp for p in self.poses], stamp)
        if index == 0 or index == len(self.poses):
            p = self.poses[min(index, len(self.poses)-1)]
            if abs(p.stamp-stamp) > self.tolerance:
                raise ValueError('Image has no matching aircraft pose')
            return Pose(stamp, p.position, p.quaternion)
        a, b = self.poses[index-1:index+1]
        if b.stamp-a.stamp > 2*self.tolerance:
            raise ValueError('Gap in aircraft pose history')
        t = (stamp-a.stamp)/(b.stamp-a.stamp)
        q0, q1 = np.array(a.quaternion), np.array(b.quaternion)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1-t)*q0 + t*q1
        q /= np.linalg.norm(q)
        pos = (1-t)*np.array(a.position)+t*np.array(b.position)
        return Pose(stamp, tuple(pos), tuple(q))


def inside(point, bounds, margin=0.0):
    x, y = point[:2]
    return (math.isfinite(x) and math.isfinite(y) and
            bounds[0]+margin <= x <= bounds[1]-margin and
            bounds[2]+margin <= y <= bounds[3]-margin)


def yaw_of(q):
    r = rotation(q)
    return math.atan2(r[1, 0], r[0, 0])
