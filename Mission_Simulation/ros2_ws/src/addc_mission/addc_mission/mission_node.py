import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
from std_srvs.srv import Trigger
from addc_interfaces.msg import TargetObservation, TargetEstimate
from .geometry import yaw_of
from .mission import Config, Mission, Estimate, DownObservation
from .ros_utils import parameter, pose_value, seconds


class MissionNode(Node):
    def __init__(self):
        super().__init__('addc_mission')
        defaults = Config()
        values = {name: parameter(self, name, list(value) if isinstance(value, tuple) else value)
                  for name, value in vars(defaults).items()}
        self.mission = Mission(Config(**values))
        self.pose = None
        self.state = State()
        self.pose_received = self.state_received = -math.inf
        self.observations = {}
        self.result_written = False
        self.last_status = -math.inf
        self.last_request = {'mode': -math.inf, 'arm': -math.inf}
        self.pending = {}
        self.result_file = Path(parameter(self, 'result_file', '/tmp/addc-evidence/results.jsonl')).expanduser()
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result_pub = self.create_publisher(String, '/addc/result', qos)
        self.status_pub = self.create_publisher(String, '/addc/status', 10)
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile_sensor_data)
        self.create_subscription(TargetObservation, '/addc/observations', self.observation_cb, 10)
        self.create_subscription(TargetEstimate, '/addc/target_estimates', self.estimate_cb, 10)
        self.create_service(Trigger, '/addc/start', self.start_cb)
        self.create_service(Trigger, '/addc/abort', self.abort_cb)
        # Runs in a separate process from inference; service calls are asynchronous.
        self.create_timer(0.05, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def pose_cb(self, msg):
        self.pose = pose_value(msg)
        self.pose_received = time.monotonic()

    def state_cb(self, msg):
        self.state = msg
        self.state_received = time.monotonic()

    def observation_cb(self, msg):
        if msg.camera == 'down':
            self.observations[seconds(msg.header.stamp)] = msg
            self.observations = {s: o for s, o in self.observations.items()
                                 if 0 <= self.now()-s < self.mission.c.target_max_age}

    def estimate_cb(self, msg):
        p = msg.position
        e = Estimate(seconds(msg.header.stamp), (p.x, p.y, p.z), msg.uncertainty_m, msg.camera)
        if msg.header.frame_id != 'map':
            return
        if msg.camera == 'discovery':
            self.mission.estimate(e, self.now())
        elif msg.camera == 'down':
            observation = self.observations.pop(e.stamp, None)
            if observation:
                self.mission.downward(DownObservation(e.stamp, e.position, observation.payload), self.now())

    def telemetry_age(self):
        return time.monotonic()-min(self.pose_received, self.state_received)

    def start_cb(self, request, response):
        response.success = bool(self.pose and self.telemetry_age() < self.mission.c.telemetry_max_age
                                and self.mission.start(self.now(), self.pose.position,
                                                       self.state.connected, self.state.armed))
        response.message = 'Mission started' if response.success else 'Start rejected: need fresh connected telemetry, disarmed aircraft at home, and WAIT_START state'
        return response

    def abort_cb(self, request, response):
        # Relinquish OFFBOARD and request landing; never re-arm after abort.
        self.mission.transition('ABORTED', self.now(), 'Explicit abort requested')
        if self.state.connected and self.state.armed:
            self.request('mode', 'AUTO.LAND')
        response.success, response.message = True, 'Mission aborted; landing requested'
        return response

    def request(self, kind, value):
        previous = self.pending.get(kind)
        if previous and not previous.done():
            return
        now = time.monotonic()
        if now-self.last_request[kind] < 1.0:
            return
        client = self.mode_client if kind == 'mode' else self.arm_client
        if not client.service_is_ready():
            return
        self.last_request[kind] = now
        request = SetMode.Request(custom_mode=value) if kind == 'mode' else CommandBool.Request(value=True)
        future = client.call_async(request)
        self.pending[kind] = future
        future.add_done_callback(lambda f: self.report_service(kind, f))

    def report_service(self, kind, future):
        try:
            response = future.result()
            accepted = response.mode_sent if kind == 'mode' else response.success
            if not accepted:
                self.get_logger().warning(f'{kind} request rejected; waiting for aircraft state')
        except Exception as exc:
            self.get_logger().error(f'{kind} service: {exc}')

    def tick(self):
        if not self.pose:
            return
        now = self.now()
        old_state = self.mission.state
        command = self.mission.tick(now, self.pose.position, yaw_of(self.pose.quaternion),
                                    self.state.connected, self.state.armed, self.state.mode,
                                    self.telemetry_age())
        if command.publish and command.position is not None:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = command.position
            msg.pose.orientation.z = math.sin(command.yaw/2)
            msg.pose.orientation.w = math.cos(command.yaw/2)
            self.setpoint_pub.publish(msg)
        if command.mode:
            self.request('mode', command.mode)
        if command.arm:
            self.request('arm', True)
        if old_state != self.mission.state:
            self.get_logger().info(f'{old_state} -> {self.mission.state}: {self.mission.reason}')
        if self.mission.result and not self.result_written:
            record = {'digits': self.mission.result, 'sim_time': now,
                      'elapsed_s': now-self.mission.started, 'cache_enu': self.mission.target}
            encoded = json.dumps(record)
            self.result_pub.publish(String(data=encoded))
            try:
                self.result_file.parent.mkdir(parents=True, exist_ok=True)
                with self.result_file.open('a') as stream:
                    stream.write(encoded+'\n')
            except OSError as exc:
                self.get_logger().error(f'Result file: {exc}')
            self.result_written = True
        if now-self.last_status >= 1.0:
            self.last_status = now
            status = {'state': self.mission.state, 'reason': self.mission.reason,
                      'digits': self.mission.result, 'target': self.mission.target,
                      'elapsed_s': now-self.mission.started if self.mission.started is not None else 0.0}
            self.status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
