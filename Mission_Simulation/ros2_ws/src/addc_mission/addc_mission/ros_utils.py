from .geometry import Pose


def seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def pose_value(msg):
    p, q = msg.pose.position, msg.pose.orientation
    return Pose(seconds(msg.header.stamp), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def parameter(node, name, default):
    return node.declare_parameter(name, default).value
