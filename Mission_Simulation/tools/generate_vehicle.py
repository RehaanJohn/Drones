#!/usr/bin/env python3
"""Generate a primitive quadrotor with Fortress motors and two RGB cameras.

Motor ordering/constants follow PX4 v1.14.4's x500 simulation. No downloaded meshes.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from generate_scenario import child, box, write_xml


def generate(output, width=1920, height=1080, tilt_deg=25.0, hfov_deg=60.0, fps=15.0,
             discovery_mount=(0.20, 0, -0.05), down_mount=(0, 0, -0.10)):
    output = Path(output)
    root = ET.Element('sdf', version='1.8')
    model = child(root, 'model', name='addc_quad')
    child(model, 'pose', '0 0 0.24 0 0 0')
    child(model, 'self_collide', 'false')
    base = child(model, 'link', name='base_link')
    inertial = child(base, 'inertial')
    child(inertial, 'mass', '1.7')
    inertia = child(inertial, 'inertia')
    for tag, value in [('ixx', 0.0217), ('iyy', 0.0217), ('izz', 0.04), ('ixy', 0), ('ixz', 0), ('iyz', 0)]:
        child(inertia, tag, value)
    box(base, 'body', '0.25 0.18 0.08', '0 0 0 0 0 0', '0.1 0.15 0.3 1')
    for i, y in enumerate((-0.13, 0.13)):
        box(base, f'leg_{i}', '0.3 0.02 0.02', f'0 {y} -0.22 0 0 0', '0.1 0.1 0.1 1')
        box(base, f'strut_{i}', '0.02 0.02 0.22', f'0 {y} -0.11 0 0 0', '0.1 0.1 0.1 1')
    for name, kind, rate in [('imu_sensor', 'imu', 250), ('air_pressure_sensor', 'air_pressure', 50)]:
        sensor = child(base, 'sensor', name=name, type=kind)
        child(sensor, 'always_on', 'true')
        child(sensor, 'update_rate', rate)
        if kind == 'air_pressure':
            # Match PX4 v1.14.4 x500: constant noiseless pressure at rest
            # triggers PX4's repeated-value (STALE) sensor check.
            pressure = child(child(sensor, 'air_pressure'), 'pressure')
            noise = child(pressure, 'noise', type='gaussian')
            child(noise, 'mean', 0)
            child(noise, 'stddev', 0.01)
    for name, mount, tilt in [('discovery', discovery_mount, math.radians(tilt_deg)),
                              ('down', down_mount, math.pi/2)]:
        sensor = child(base, 'sensor', name=f'{name}_camera', type='camera')
        child(sensor, 'pose', f'{mount[0]} {mount[1]} {mount[2]} 0 {tilt} 0')
        child(sensor, 'always_on', 'true')
        child(sensor, 'update_rate', fps)
        child(sensor, 'topic', f'/camera/{name}/image')
        camera = child(sensor, 'camera', name=name)
        child(camera, 'horizontal_fov', math.radians(hfov_deg))
        child(camera, 'optical_frame_id', f'{name}_camera_optical')
        child(camera, 'camera_info_topic', f'/camera/{name}/camera_info')
        image = child(camera, 'image')
        child(image, 'width', width)
        child(image, 'height', height)
        child(image, 'format', 'R8G8B8')
        clip = child(camera, 'clip')
        child(clip, 'near', '0.05')
        child(clip, 'far', '100')
    for i, (x, y, direction) in enumerate([(0.174, -0.174, 'ccw'), (-0.174, 0.174, 'ccw'),
                                         (0.174, 0.174, 'cw'), (-0.174, -0.174, 'cw')]):
        rotor = child(model, 'link', name=f'rotor_{i}')
        child(rotor, 'pose', f'{x} {y} 0.06 0 0 0')
        inertial = child(rotor, 'inertial')
        child(inertial, 'mass', '0.016')
        inertia = child(inertial, 'inertia')
        # Principal moments must satisfy Izz <= Ixx + Iyy.
        # The previous rounded Izz (2.65e-5) exceeded that sum.
        for tag, value in [('ixx', 3.85e-7), ('iyy', 2.61e-5), ('izz', 2.64e-5)]:
            child(inertia, tag, value)
        box(rotor, 'propeller', '0.28 0.015 0.003', '0 0 0 0 0 0', '0.1 0.1 0.1 1', False)
        joint = child(model, 'joint', name=f'rotor_{i}_joint', type='revolute')
        child(joint, 'parent', 'base_link')
        child(joint, 'child', f'rotor_{i}')
        axis = child(joint, 'axis')
        child(axis, 'xyz', '0 0 1')
        limit = child(axis, 'limit')
        child(limit, 'lower', '-1e16')
        child(limit, 'upper', '1e16')
        plugin = child(model, 'plugin', filename='ignition-gazebo-multicopter-motor-model-system',
                       name='ignition::gazebo::systems::MulticopterMotorModel')
        for tag, value in [('robotNamespace', 'addc_quad_0'), ('jointName', f'rotor_{i}_joint'),
                           ('linkName', f'rotor_{i}'), ('turningDirection', direction),
                           ('timeConstantUp', 0.0125), ('timeConstantDown', 0.025),
                           ('maxRotVelocity', 1000), ('motorConstant', 8.54858e-6),
                           ('momentConstant', 0.016), ('commandSubTopic', 'command/motor_speed'),
                           ('motorNumber', i), ('rotorDragCoefficient', 8.06428e-5),
                           ('rollingMomentCoefficient', 1e-6), ('rotorVelocitySlowdownSim', 10),
                           ('motorType', 'velocity')]:
            child(plugin, tag, value)
    write_xml(root, output/'models'/'addc_quad'/'model.sdf')
    config = ET.Element('model')
    child(config, 'name', 'addc_quad')
    child(config, 'version', '1.0')
    child(config, 'sdf', 'model.sdf', version='1.8')
    write_xml(config, output/'models'/'addc_quad'/'model.config')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'ros2_ws/src/addc_mission/config/mission.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())['/**']['ros__parameters']
    generate(args.output, config['camera_width'], config['camera_height'],
             config['discovery_tilt_deg'], config['camera_hfov_deg'], config['camera_fps'],
             config['discovery_mount'], config['down_mount'])
