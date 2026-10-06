import unittest
from addc_mission.mission import Mission, Estimate, DownObservation, Config


class MissionTests(unittest.TestCase):
    def ready(self):
        mission = Mission()
        self.assertTrue(mission.start(0, (0, 0, 0.24), True, False))
        mission.tick(2.1, (0, 0, 0.24), 0, True, False, '')
        mission.tick(2.2, (0, 0, 0.24), 0, True, True, 'OFFBOARD')
        mission.tick(3, (0, 0, 6), 0, True, True, 'OFFBOARD')
        mission.tick(4.2, (0, 0, 6), 0, True, True, 'OFFBOARD')
        self.assertEqual(mission.state, 'DISCOVER')
        return mission

    def approach(self):
        m = self.ready()
        for t in (4.3, 4.5, 4.7):
            m.estimate(Estimate(t, (12, 3, 0), 1), t)
        self.assertEqual(m.state, 'APPROACH')
        self.assertEqual(m.view_index, 0)  # Found immediately; no coverage route required.
        return m

    def test_start_is_explicit_and_disarmed(self):
        m = Mission()
        self.assertFalse(m.tick(1, (0, 0, 0), 0, True, False, '').publish)
        self.assertFalse(m.start(1, (0, 0, 0), True, True))

    def test_stale_duplicate_and_outside_observations_do_not_confirm(self):
        m = self.ready()
        for _ in range(5):
            m.estimate(Estimate(4.3, (12, 3, 0), 1), 4.4)
        self.assertEqual(m.candidate_hits, 1)
        m.estimate(Estimate(4.5, (100, 3, 0), 1), 4.5)
        m.estimate(Estimate(4.6, (12, 3, 0), 1), 10)
        self.assertEqual(m.state, 'DISCOVER')

    def test_direct_approach_handoff_decode_and_land(self):
        m = self.approach()
        cmd = m.tick(4.8, (0, 0, 6), 0, True, True, 'OFFBOARD')
        self.assertLessEqual((cmd.position[0]**2+cmd.position[1]**2)**0.5, 1.01)
        m.tick(10, (12, 3, 6), 0, True, True, 'OFFBOARD')
        self.assertEqual(m.state, 'ACQUIRE_DOWN')
        m.downward(DownObservation(10.1, (12, 3, 0.6)), 10.1)
        m.tick(10.2, (12, 3, 6), 0, True, True, 'OFFBOARD')
        self.assertEqual(m.state, 'SCAN')
        for t in (10.3, 10.5, 10.7):
            m.downward(DownObservation(t, (12, 3, 0.6), '07'), t)
        self.assertEqual(m.result, '07')
        self.assertEqual(m.state, 'RETURN')
        cmd = m.tick(11, (0, 0, 6), 0, True, True, 'OFFBOARD')
        self.assertEqual(cmd.mode, 'AUTO.LAND')
        self.assertFalse(cmd.publish)
        cmd = m.tick(15, (0, 0, 0.24), 0, True, False, 'AUTO.LAND')
        self.assertEqual(m.state, 'DONE')
        self.assertFalse(cmd.arm)

    def test_lost_downward_image_prevents_descent(self):
        m = self.approach()
        m.tick(10, (12, 3, 6), 0, True, True, 'OFFBOARD')
        m.downward(DownObservation(10.1, (12, 3, 0.6)), 10.1)
        m.tick(10.2, (12, 3, 6), 0, True, True, 'OFFBOARD')
        cmd = m.tick(14, (12, 3, 4), 0, True, True, 'OFFBOARD')
        self.assertGreaterEqual(cmd.position[2], 4)
        self.assertEqual(m.state, 'ACQUIRE_DOWN')

    def test_manual_mode_change_never_rearms_or_reenters_offboard(self):
        m = self.approach()
        cmd = m.tick(5, (0, 0, 6), 0, True, True, 'POSCTL')
        self.assertEqual(m.state, 'ABORTED')
        self.assertFalse(cmd.publish)
        self.assertFalse(cmd.arm)
        self.assertEqual(cmd.mode, '')
        self.assertFalse(m.tick(6, (0, 0, 6), 0, True, False, 'POSCTL').arm)

    def test_handoff_timeout_rejects_candidate(self):
        m = self.approach()
        m.tick(10, (12, 3, 6), 0, True, True, 'OFFBOARD')
        m.tick(31, (12, 3, 6), 0, True, True, 'OFFBOARD')
        self.assertEqual(m.state, 'DISCOVER')
        self.assertEqual(len(m.rejected), 1)

    def test_timeout_returns_and_stale_telemetry_relinquishes(self):
        m = self.ready()
        m.tick(181, (0, 0, 6), 0, True, True, 'OFFBOARD')
        self.assertIn(m.state, ('RETURN', 'LAND'))
        m = self.ready()
        cmd = m.tick(5, (0, 0, 6), 0, True, True, 'OFFBOARD', telemetry_age=5)
        self.assertEqual(m.state, 'ABORTED')
        self.assertFalse(cmd.publish)

    def test_wrong_digits_cannot_complete(self):
        m = self.approach()
        for t in (5, 5.2, 5.4):
            m.downward(DownObservation(t, (12, 3, 0.6), '007'), t)
        self.assertEqual(m.result, '')
