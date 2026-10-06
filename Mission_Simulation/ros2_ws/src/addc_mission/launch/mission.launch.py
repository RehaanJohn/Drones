from pathlib import Path
import math
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def camera_transforms(context):
    config_path = LaunchConfiguration('config').perform(context)
    config = yaml.safe_load(Path(config_path).read_text())['/**']['ros__parameters']
    transforms = []
    # Optical frame rotation is Ry(tilt) * the standard forward camera optical rotation.
    for camera, mount, tilt in [('discovery', config['discovery_mount'], math.radians(config['discovery_tilt_deg'])),
                                ('down', config['down_mount'], math.pi/2)]:
        c, s = math.cos(tilt/2), math.sin(tilt/2)
        quaternion = (-0.5*(c+s), 0.5*(c+s), 0.5*(s-c), 0.5*(c-s))
        values = dict(zip(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], (*mount, *quaternion)))
        arguments = [part for key, value in values.items() for part in (f'--{key}', str(value))]
        arguments += ['--frame-id', 'base_link', '--child-frame-id', f'{camera}_camera_optical']
        transforms.append(Node(package='tf2_ros', executable='static_transform_publisher',
                               name=f'{camera}_camera_tf', arguments=arguments,
                               parameters=[{'use_sim_time': True}]))
    return transforms


def generate_launch_description():
    share = Path(get_package_share_directory('addc_mission'))
    mavros = Path(get_package_share_directory('mavros'))
    parameters = LaunchConfiguration('config')
    scanner = share/'qr'/'scanner.py'
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        DeclareLaunchArgument('config', default_value=str(share/'config'/'mission.yaml')),
        DeclareLaunchArgument('fcu_url', default_value='udp://:14540@127.0.0.1:14557'),
        DeclareLaunchArgument('qr_model_dir', default_value='/tmp/addc-qr-models'),
        OpaqueFunction(function=camera_transforms),
        IncludeLaunchDescription(AnyLaunchDescriptionSource(str(mavros/'launch'/'px4.launch')),
                                 launch_arguments={'fcu_url': LaunchConfiguration('fcu_url'),
                                                   'namespace': 'mavros'}.items()),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='addc_bridge',
             parameters=[{'config_file': str(share/'config'/'bridge.yaml'), 'use_sim_time': True}], output='screen'),
        Node(package='addc_mission', executable='perception', name='addc_perception',
             parameters=[parameters, {'scanner_path': str(scanner),
                                      'qr_model_dir': LaunchConfiguration('qr_model_dir')}], output='screen'),
        Node(package='addc_mission', executable='localizer', name='addc_localizer',
             parameters=[parameters], output='screen'),
        Node(package='addc_mission', executable='mission', name='addc_mission',
             parameters=[parameters], output='screen'),
    ])
