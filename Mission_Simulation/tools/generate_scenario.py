#!/usr/bin/env python3
"""Generate a Fortress world and an A3 QR cache, never passed to perception."""
import argparse
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def child(parent, tag, text=None, **attributes):
    item = ET.SubElement(parent, tag, attributes)
    if text is not None:
        item.text = str(text)
    return item


def write_xml(root, path):
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def box(link, name, size, pose, color, collision=True):
    visual = child(link, 'visual', name=name)
    child(visual, 'pose', pose)
    child(child(child(visual, 'geometry'), 'box'), 'size', size)
    material = child(visual, 'material')
    child(material, 'ambient', color)
    child(material, 'diffuse', color)
    if collision:
        hit = child(link, 'collision', name=name+'_collision')
        child(hit, 'pose', pose)
        child(child(child(hit, 'geometry'), 'box'), 'size', size)
    return visual


def generate(output, cache=(12.0, 3.0, 0.0), payload='07', qr_size=0.25,
             seed=1, distractors=0):
    if len(payload) != 2 or not all('0' <= c <= '9' for c in payload):
        raise ValueError('Payload must contain exactly two ASCII digits')
    if not 0.05 <= qr_size <= 0.297:
        raise ValueError('QR square must fit within the 297 mm sheet width')
    if not (-2 < cache[0] < 49 and -17 < cache[1] < 17):
        raise ValueError('Cache must be inside configured arena')
    output = Path(output)
    cache_dir = output/'models'/'intelligence_cache'
    textures = cache_dir/'materials'/'textures'
    textures.mkdir(parents=True, exist_ok=True)
    # Real machine-readable QR; nearest-neighbour scaling preserves modules.
    qr = cv2.QRCodeEncoder_create().encode(payload)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(textures/'qr.png'), qr)
    config = ET.Element('model')
    child(config, 'name', 'intelligence_cache')
    child(config, 'version', '1.0')
    child(config, 'sdf', 'model.sdf', version='1.8')
    write_xml(config, cache_dir/'model.config')
    root = ET.Element('sdf', version='1.8')
    model = child(root, 'model', name='intelligence_cache')
    child(model, 'static', 'true')
    link = child(model, 'link', name='cache')
    # Orange is an integration fixture, explicitly NOT a general cache detector.
    box(link, 'container', '0.65 0.5 0.6', '0 0 0.3 0 0 0', '1 0.25 0.04 1')
    box(link, 'a3_sheet', '0.420 0.297 0.002', '0 0 0.602 0 0 0', '1 1 1 1', False)
    visual = child(link, 'visual', name='qr')
    child(visual, 'pose', '0 0 0.604 0 0 0')
    child(visual, 'cast_shadows', 'false')
    plane = child(child(visual, 'geometry'), 'plane')
    child(plane, 'normal', '0 0 1')
    child(plane, 'size', f'{qr_size} {qr_size}')
    material = child(visual, 'material')
    child(material, 'ambient', '1 1 1 1')
    child(material, 'diffuse', '1 1 1 1')
    metal = child(child(material, 'pbr'), 'metal')
    child(metal, 'albedo_map', 'model://intelligence_cache/materials/textures/qr.png')
    child(metal, 'metalness', '0')
    child(metal, 'roughness', '1')
    write_xml(root, cache_dir/'model.sdf')

    root = ET.Element('sdf', version='1.8')
    world = child(root, 'world', name='addc')
    physics = child(world, 'physics', name='default', type='ignored')
    child(physics, 'max_step_size', '0.004')
    child(physics, 'real_time_factor', '1.0')
    child(world, 'gravity', '0 0 -9.80665')
    for system in ('Physics', 'UserCommands', 'SceneBroadcaster', 'Imu', 'AirPressure', 'Sensors'):
        kebab = {'UserCommands': 'user-commands', 'SceneBroadcaster': 'scene-broadcaster',
                 'AirPressure': 'air-pressure'}.get(system, system.lower())
        plugin = child(world, 'plugin', filename=f'ignition-gazebo-{kebab}-system',
                       name=f'ignition::gazebo::systems::{system}')
        if system == 'Sensors':
            child(plugin, 'render_engine', 'ogre2')
    light = child(world, 'light', name='sun', type='directional')
    child(light, 'pose', '0 0 10 0 0 0')
    child(light, 'diffuse', '0.9 0.9 0.9 1')
    child(light, 'specular', '0.2 0.2 0.2 1')
    child(light, 'direction', '-0.5 0.1 -0.9')
    child(light, 'cast_shadows', 'true')
    attenuation = child(light, 'attenuation')
    child(attenuation, 'range', '1000')
    child(attenuation, 'constant', '0.9')
    child(attenuation, 'linear', '0.01')
    ground = child(world, 'model', name='ground')
    child(ground, 'static', 'true')
    box(child(ground, 'link', name='ground'), 'field', '120 100 0.1', '20 0 -0.05 0 0 0', '0.22 0.34 0.18 1')
    home = child(world, 'model', name='home_zone')
    child(home, 'static', 'true')
    box(child(home, 'link', name='marker'), 'home', '2 2 0.005', '0 0 0.003 0 0 0', '0.15 0.3 0.7 1', False)
    include = child(world, 'include')
    child(include, 'uri', 'model://intelligence_cache')
    child(include, 'pose', f'{cache[0]} {cache[1]} 0 0 0 {cache[2]}')
    vehicle = child(world, 'include')
    child(vehicle, 'uri', 'model://addc_quad')
    child(vehicle, 'name', 'addc_quad_0')
    child(vehicle, 'pose', '0 0 0.24 0 0 0')
    rng = random.Random(seed)
    for i in range(distractors):
        x, y = rng.uniform(3, 45), rng.uniform(-15, 15)
        model = child(world, 'model', name=f'distractor_{i}')
        child(model, 'static', 'true')
        child(model, 'pose', f'{x} {y} 0 0 0 {rng.uniform(0, math.pi)}')
        box(child(model, 'link', name='body'), 'box', '0.6 0.4 0.4', '0 0 0.2 0 0 0',
            '0.4 0.4 0.4 1' if i % 2 else '1 1 1 1')
    write_xml(root, output/'worlds'/'addc.sdf')
    (output/'scenario_truth.json').write_text(json.dumps(
        {'cache': cache, 'payload': payload, 'qr_size_m': qr_size, 'seed': seed}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=float, nargs=3, metavar=('X', 'Y', 'YAW'), default=(12, 3, 0))
    parser.add_argument('--payload', default='07')
    parser.add_argument('--qr-size', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--distractors', type=int, default=0)
    args = parser.parse_args()
    generate(args.output, args.cache, args.payload, args.qr_size, args.seed, args.distractors)
    print(args.output/'worlds'/'addc.sdf')


if __name__ == '__main__':
    main()
