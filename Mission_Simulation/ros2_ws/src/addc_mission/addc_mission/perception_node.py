import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
from addc_interfaces.msg import TargetObservation
from .ros_utils import parameter
from .vision import QRBackend, discover


class Perception(Node):
    def __init__(self):
        super().__init__('addc_perception')
        self.bridge = CvBridge()
        self.mode = parameter(self, 'discovery_mode', 'color')
        self.period = 1.0/parameter(self, 'max_processing_hz', 5.0)
        self.max_age = parameter(self, 'maximum_image_age', 0.8)
        self.min_area = parameter(self, 'minimum_discovery_area', 80.0)
        self.frames = {}
        self.processed = {}
        self.last_work = {}
        self.backend = QRBackend(parameter(self, 'scanner_path', ''), parameter(self, 'qr_model_dir', ''))
        self.evidence_dir = Path(parameter(self, 'evidence_directory', '/tmp/addc-evidence')).expanduser()
        self.saved_payloads = set()
        self.publisher = self.create_publisher(TargetObservation, '/addc/observations', 10)
        self.debug = {c: self.create_publisher(Image, f'/addc/{c}/annotated', 2)
                      for c in ('discovery', 'down')}
        self.subscribers = [self.create_subscription(
            Image, f'/camera/{c}/image_raw', lambda msg, camera=c: self.frames.update({camera: msg}),
            qos_profile_sensor_data) for c in ('discovery', 'down')]
        self.create_timer(0.05, self.process)
        self.get_logger().info(f'QR backend: {self.backend.name}; discovery: {self.mode} (prototype)')

    def process(self):
        # Independent process from mission control; newest frame wins over backlog.
        for camera, msg in list(self.frames.items()):
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            now = self.get_clock().now().nanoseconds*1e-9
            age = now-(key[0]+key[1]*1e-9)
            if (self.processed.get(camera) == key or not 0 <= age <= self.max_age or
                    time.monotonic()-self.last_work.get(camera, 0) < self.period):
                continue
            self.processed[camera] = key
            self.last_work[camera] = time.monotonic()
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                if camera == 'discovery':
                    found = discover(frame, self.mode, self.min_area)
                    observations = [((found[0], found[1]), found[2], '')] if found else []
                else:
                    observations = self.backend.detect(frame)
                    # One observation per camera frame keeps payload/localisation association exact.
                    observations = sorted(observations, key=lambda o: (
                        not bool(o[2]), (o[0][0]-frame.shape[1]/2)**2+(o[0][1]-frame.shape[0]/2)**2))[:1]
                annotated = frame.copy()
                for center, corners, payload in observations:
                    observation = TargetObservation()
                    observation.header = msg.header
                    observation.camera = camera
                    observation.kind = ('cache_base' if self.mode == 'color' else 'sheet') if camera == 'discovery' else 'qr_or_sheet'
                    observation.u, observation.v = float(center[0]), float(center[1])
                    observation.confidence = 0.8
                    observation.payload = payload
                    observation.corners = [Point32(x=float(x), y=float(y), z=0.0) for x, y in corners]
                    self.publisher.publish(observation)
                    cv2.polylines(annotated, [np.array(corners, dtype='int32')], True, (0, 255, 0), 2)
                    if payload and payload not in self.saved_payloads:
                        self.evidence_dir.mkdir(parents=True, exist_ok=True)
                        # Timestamp rather than payload in filename: arbitrary QR content is untrusted.
                        path = self.evidence_dir/f'qr-{key[0]}-{key[1]}.png'
                        cv2.imwrite(str(path), frame)
                        self.saved_payloads.add(payload)
                debug = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                debug.header = msg.header
                self.debug[camera].publish(debug)
            except (cv2.error, ValueError, RuntimeError) as exc:
                self.get_logger().error(f'{camera}: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = Perception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
