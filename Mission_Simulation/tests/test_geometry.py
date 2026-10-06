import math
import unittest
import numpy as np
from addc_mission.geometry import Pose, PoseBuffer, project_pixel, optical_to_body, rotation


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.k = [1000, 0, 960, 0, 1000, 540, 0, 0, 1]
        self.pose = Pose(1, (3, 2, 6), (0, 0, 0, 1))

    def test_downward_center_hits_under_camera(self):
        point, _ = project_pixel(960, 540, self.k, self.pose, (0, 0, -0.1), math.pi/2, plane_z=0.6)
        np.testing.assert_allclose(point, (3, 2, 0.6), atol=1e-8)

    def test_oblique_center_range_and_yaw(self):
        point, _ = project_pixel(960, 540, self.k, self.pose, (0, 0, 0), math.radians(25))
        self.assertAlmostEqual(point[0], 3+6/math.tan(math.radians(25)))
        yaw_pose = Pose(1, (3, 2, 6), (0, 0, math.sin(math.pi/4), math.cos(math.pi/4)))
        point, _ = project_pixel(960, 540, self.k, yaw_pose, (0, 0, 0), math.radians(25))
        self.assertAlmostEqual(point[1], 2+6/math.tan(math.radians(25)))

    def test_reject_horizon_and_bad_intrinsics(self):
        with self.assertRaises(ValueError):
            project_pixel(960, 540, self.k, self.pose, (0, 0, 0), 0)
        with self.assertRaises(ValueError):
            project_pixel(960, 540, [0]*9, self.pose, (0, 0, 0), math.pi/2)

    def test_capture_time_interpolation_not_latest_pose(self):
        history = PoseBuffer()
        history.append(Pose(1, (0, 0, 6), (0, 0, 0, 1)))
        history.append(Pose(1.1, (2, 0, 6), (0, 0, 0, -1)))
        self.assertAlmostEqual(history.at(1.05).position[0], 1)
        np.testing.assert_allclose(rotation(history.at(1.05).quaternion), np.eye(3))
        with self.assertRaises(ValueError):
            history.at(0.5)

    def test_optical_transform_matches_camera_pitch(self):
        for tilt in (0, math.radians(25), math.pi/2):
            c, s = math.cos(tilt/2), math.sin(tilt/2)
            q = (-0.5*(c+s), 0.5*(c+s), 0.5*(s-c), 0.5*(c-s))
            np.testing.assert_allclose(rotation(q), optical_to_body(tilt), atol=1e-8)
