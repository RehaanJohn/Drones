from glob import glob
from os.path import relpath
from pathlib import Path
from setuptools import setup

setup(
    name='addc_mission', version='0.1.0', packages=['addc_mission'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/addc_mission']),
        ('share/addc_mission', ['package.xml']),
        ('share/addc_mission/launch', glob('launch/*.launch.py')),
        ('share/addc_mission/config', glob('config/*.yaml')),
        ('share/addc_mission/qr', [relpath(Path(__file__).resolve().parents[4]/'QR_Scanning'/'scanner.py',
                                             Path(__file__).resolve().parent)]),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ADDC team', maintainer_email='team@example.com',
    description='Two-camera ADDC simulation', license='Apache-2.0',
    entry_points={'console_scripts': [
        'perception = addc_mission.perception_node:main',
        'localizer = addc_mission.localizer_node:main',
        'mission = addc_mission.mission_node:main',
    ]},
)
