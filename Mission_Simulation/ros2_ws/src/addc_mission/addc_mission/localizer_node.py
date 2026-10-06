import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from addc_interfaces.msg import TargetObservation, TargetEstimate
from .geometry import PoseBuffer, inside, project_pixel
from .ros_utils import parameter, pose_value, seconds


class Localizer(Node):
    def __init__(self):
        super().__init__('addc_localizer')
        self.poses = PoseBuffer()
        self.intrinsics = {}
        self.pending = []
        self.bounds = parameter(self, 'bounds', [-3.0, 50.0, -18.0, 18.0])
        self.max_range = parameter(self, 'maximum_projection_range', 80.0)
        self.max_age = parameter(self, 'maximum_observation_age', 0.8)
        self.ground_z = parameter(self, 'ground_z', -0.24)
        self.top_height = parameter(self, 'cache_top_height', 0.6)
        self.mounts = {c: parameter(self, f'{c}_mount', [0.15, 0.0, -0.05])
                       for c in ('discovery', 'down')}
        self.tilts = {
            'discovery': math.radians(parameter(self, 'discovery_tilt_deg', 25.0)),
            'down': math.pi/2,
        }
        self.frames = {c: f'{c}_camera_optical' for c in self.mounts}
        self.publisher = self.create_publisher(TargetEstimate, '/addc/target_estimates', 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(TargetObservation, '/addc/observations', self.observe, 10)
        self.info_subs = [self.create_subscription(
            CameraInfo, f'/camera/{c}/camera_info', lambda msg, camera=c: self.info(msg, camera),
            qos_profile_sensor_data) for c in self.mounts]
        self.create_timer(0.05, self.flush)

    def info(self, msg, camera):
        self.intrinsics[camera] = msg.k

    def pose_cb(self, msg):
        self.poses.append(pose_value(msg))

    def observe(self, msg):
        if msg.camera in self.mounts:
            self.pending.append(msg)
            self.pending = self.pending[-20:]

    def flush(self):
        now = self.get_clock().now().nanoseconds*1e-9
        retained = []
        for msg in self.pending:
            stamp = seconds(msg.header.stamp)
            if not 0 <= now-stamp <= self.max_age:
                continue
            if msg.header.frame_id != self.frames[msg.camera]:
                continue
            try:
                pose = self.poses.at(stamp)
                k = self.intrinsics[msg.camera]
            except (ValueError, KeyError):
                retained.append(msg)
                continue
            try:
                plane = self.ground_z if msg.kind == 'cache_base' else self.ground_z+self.top_height
                point, uncertainty = project_pixel(
                    msg.u, msg.v, k, pose, self.mounts[msg.camera], self.tilts[msg.camera],
                    plane_z=plane, max_range=self.max_range)
                if not inside(point, self.bounds, 0.5):
                    continue
                estimate = TargetEstimate()
                estimate.header.stamp = msg.header.stamp
                estimate.header.frame_id = 'map'
                estimate.camera = msg.camera
                estimate.position.x, estimate.position.y, estimate.position.z = point
                estimate.uncertainty_m = uncertainty
                self.publisher.publish(estimate)
            except ValueError:
                pass
        self.pending = retained


def main(args=None):
    rclpy.init(args=args)
    node = Localizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
