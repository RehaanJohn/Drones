#!/usr/bin/env python3
"""Small, hash-guarded PX4 v1.14.4 Garden -> Fortress compatibility port.

Only source names / dependency lookup / CLI selection change. Rotor dynamics and
PX4 control remain PX4's implementation. Build and SITL validation require Ubuntu.
"""
import argparse
import hashlib
from pathlib import Path
import re

HASHES = {
    'CMakeLists.txt': '1531626036c7a33ff35e1d490fefaf14a750f1e4e83d594273f90d8654b6a5c4',
    'GZBridge.cpp': 'ff92ea982ae9cb6953fbc1b6b00fd1442c9eb3b30a3a772a35e71873cab331d5',
    'GZBridge.hpp': '085bf2932f6367af21a77e0e23cb7a77a49965f3f5459ace599b7eca18513526',
    'GZMixingInterfaceESC.cpp': '04f29b568bc33593eb3af423db7ff8b1fc6ee12f2dbcf2e135f96071b8460fae',
    'GZMixingInterfaceESC.hpp': 'd1ff853691f1c19d54fa7f605e031cbc5cd342ca92797db65e86932cdfbed16c',
    'GZMixingInterfaceServo.cpp': '3d48d2bf9629c4b498fcec3f2f9c38cdb9e94544161f7fd23a07651584685c8e',
    'GZMixingInterfaceServo.hpp': 'e9832d9fa1dbf61bd12042bb76f400f507e454ade0dcaab135d13438853bc27c',
    'px4-rc.simulator': '7d26a8ddd0be60e40726d2a991a73b556515acaaaeff8541200f6efca52eabcd',
}


def transform(name, text):
    if name.endswith(('.cpp', '.hpp')):
        return text.replace('<gz/', '<ignition/').replace('gz::', 'ignition::')
    if name == 'CMakeLists.txt':
        start = text.index('# Find the gz_Transport library')
        end = text.index('\n\tpx4_add_module(', start)
        # Explicitly name the versions installed by ignition-fortress on Jammy.
        replacement = '''# ADDC: Fortress transport11, msgs8, math6.
find_package(ignition-transport11 REQUIRED COMPONENTS core)
find_package(ignition-msgs8 REQUIRED)
find_package(ignition-math6 REQUIRED)
if(ignition-transport11_FOUND)
    add_compile_options(-frtti -fexceptions)
    set(GZ_TRANSPORT_LIB ignition-transport11::core ignition-msgs8::ignition-msgs8 ignition-math6::ignition-math6)
'''
        return text[:start]+replacement+text[end:]
    if name == 'px4-rc.simulator':
        start = text.index('\t# "gz sim" only')
        end = text.index('\t# look for running', start)
        replacement = '''\t# ADDC: use Fortress CLI; the world is launched separately.
    if ign gazebo --versions >/dev/null 2>&1; then
        gz_command="ign"
        gz_sub_command="gazebo"
    else
        echo "ERROR [init] ADDC requires Gazebo Fortress (ign gazebo)"
        exit 1
    fi

'''
        return text[:start]+replacement+text[end:]
    raise ValueError(name)


def patch(root, check=False):
    changes = []
    for name, expected in HASHES.items():
        relative = ('ROMFS/px4fmu_common/init.d-posix/'+name if name == 'px4-rc.simulator'
                    else 'src/modules/simulation/gz_bridge/'+name)
        path = root/relative
        backup = path.with_name(path.name+'.addc-original')
        original = backup.read_bytes() if backup.exists() else path.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(f'{relative}: source is not the supported pristine PX4 v1.14.4 file')
        modified = transform(name, original.decode()).encode()
        current = path.read_bytes()
        if current not in (original, modified):
            raise ValueError(f'{relative}: local modifications would be overwritten')
        changes.append((path, backup, original, modified))
    # Validate every file before touching any of them. Idempotent, backups retained.
    if not check:
        for path, backup, original, modified in changes:
            if not backup.exists():
                backup.write_bytes(original)
            path.write_bytes(modified)
    return len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('px4_dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        count = patch(args.px4_dir.resolve(), args.check)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc)+'\n')
    print(f'{count} PX4 source files validated'+('' if args.check else ' and adapted for Fortress'))


if __name__ == '__main__':
    main()
