#!/usr/bin/env python3
"""Estimate this run's ground plane from fresh, disarmed MAVROS pose samples.

Run with the vehicle resting on level ground and the default ROS launch active.
The 20 cm spread / 10 cm half-window drift limits are prototype calibration
heuristics, not PX4 arming checks or guarantees of flight performance.
"""
import argparse
import math
from pathlib import Path
import statistics
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def summarize(values):
    if len(values) < 30 or not all(math.isfinite(z) for z in values):
        raise ValueError('Need at least 30 finite altitude readings')
    midpoint = len(values)//2
    span = max(values)-min(values)
    drift = abs(statistics.median(values[:midpoint])-statistics.median(values[midpoint:]))
    if span > 0.20 or drift > 0.10:
        raise ValueError(f'Altitude has not settled: spread={span:.3f} m, drift={drift:.3f} m')
    return statistics.median(values), span, drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'ros2_ws/src/addc_mission/config/mission.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'.runtime/mission-calibrated.yaml')
    args = parser.parse_args()
    # Import ROS only for live capture, so summary checks can run offline.
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped
    from mavros_msgs.msg import State, ExtendedState

    rclpy.init()
    node = rclpy.create_node('addc_ground_calibration')
    states = {}
    samples = []

    def state_cb(msg):
        states['state'] = (msg, time.monotonic())

    def landed_cb(msg):
        states['landed'] = (msg, time.monotonic())

    def ready(now):
        if 'state' not in states or 'landed' not in states:
            return False
        state, state_time = states['state']
        landed, landed_time = states['landed']
        return (state.connected and not state.armed and now-state_time < 3
                and now-landed_time < 3
                and landed.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def pose_cb(msg):
        now = time.monotonic()
        if not ready(now):
            samples.clear()
            return
        z = msg.pose.position.z
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if not math.isfinite(z) or msg.header.frame_id != 'map':
            samples.clear()
            return
        if samples and stamp <= samples[-1][1]:
            if stamp < samples[-1][1]:
                samples.clear()
            return
        samples.append((now, stamp, z))
        while samples and now-samples[0][0] > 8.5:
            samples.pop(0)

    node.create_subscription(State, '/mavros/state', state_cb, qos_profile_sensor_data)
    node.create_subscription(ExtendedState, '/mavros/extended_state', landed_cb, qos_profile_sensor_data)
    node.create_subscription(PoseStamped, '/mavros/local_position/pose', pose_cb, qos_profile_sensor_data)
    print('Sampling the connected, disarmed vehicle on the ground (up to 45 seconds)...', flush=True)
    last_error = 'No fresh connected/disarmed/on-ground telemetry; start the default ROS launch'
    started = time.monotonic()
    try:
        while time.monotonic()-started < 45:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if not ready(now) or not samples or now-samples[-1][0] > 1:
                samples.clear()
                continue
            if samples[-1][0]-samples[0][0] < 8:
                continue
            try:
                z, span, drift = summarize([sample[2] for sample in samples])
            except ValueError as exc:
                last_error = str(exc)
                continue
            config = yaml.safe_load(args.config.read_text())
            # Model base_link is approximately 0.24 m above a level floor at rest.
            config['/**']['ros__parameters']['ground_z'] = z-0.24
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(yaml.safe_dump(config))
            print(f'PASS: resting z={z:.3f} m; spread={span:.3f} m; drift={drift:.3f} m')
            print(f'Ground z={z-0.24:.3f} m. Saved {args.output.resolve()}')
            return 0
        print(f'Calibration not saved: {last_error}. Do not launch a calibrated config from this attempt.')
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
