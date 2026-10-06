"""Deterministic mission decisions, independent of ROS and Gazebo truth."""
from dataclasses import dataclass, field
import math
from .geometry import inside


@dataclass
class Config:
    bounds: tuple = (-3.0, 50.0, -18.0, 18.0)
    search_height: float = 6.0
    minimum_height: float = 2.0
    maximum_height: float = 12.0
    scan_rate: float = 0.5
    approach_lookahead: float = 1.0
    position_tolerance: float = 0.5
    confirmation_count: int = 3
    target_max_age: float = 2.0
    telemetry_max_age: float = 1.0
    candidate_radius: float = 3.0
    maximum_uncertainty: float = 5.0
    handoff_timeout: float = 20.0
    approach_timeout: float = 60.0
    scan_timeout: float = 30.0
    mission_timeout: float = 180.0  # Reserve time for the ground operative.
    prime_duration: float = 2.0
    arm_timeout: float = 15.0
    landing_timeout: float = 45.0
    descent_rate: float = 0.3
    viewpoint_spacing: float = 12.0
    decode_confirmation_count: int = 3
    payload_length: int = 2


@dataclass(frozen=True)
class Estimate:
    stamp: float
    position: tuple
    uncertainty: float
    camera: str = 'discovery'


@dataclass(frozen=True)
class DownObservation:
    stamp: float
    position: tuple  # Projected QR/sheet point on cache top plane.
    payload: str = ''


@dataclass
class Command:
    position: tuple | None = None
    yaw: float = 0.0
    mode: str = ''
    arm: bool = False
    publish: bool = True


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def step_toward(a, b, limit):
    d = distance(a, b)
    fraction = min(1.0, limit/d) if d else 1.0
    return (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]), b[2])


def viewpoints(bounds, spacing, z):
    if spacing <= 0:
        raise ValueError('Viewpoint spacing must be positive')
    xmin, xmax, ymin, ymax = bounds
    xs = [xmin+2+i*spacing for i in range(max(1, math.ceil((xmax-xmin-4)/spacing)))]
    ys = [ymin+2+i*spacing for i in range(max(1, math.ceil((ymax-ymin-4)/spacing)))]
    return [(x, y, z) for j, y in enumerate(ys)
            for x in (xs if j % 2 == 0 else list(reversed(xs)))
            if inside((x, y), bounds, 1.0)]


class Mission:
    def __init__(self, config=None):
        self.c = config or Config()
        if not 0 < self.c.minimum_height <= self.c.search_height <= self.c.maximum_height:
            raise ValueError('Invalid altitude limits')
        self.state = 'WAIT_START'
        self.reason = 'Waiting for /addc/start'
        self.home = None
        self.hold = None
        self.target = None
        self.candidate = None
        self.candidate_hits = 0
        self.last_estimate_stamp = -math.inf
        self.down = None
        self.payload_candidate = ''
        self.payload_hits = 0
        self.payload_stamp = -math.inf
        self.result = ''
        self.started = None
        self.entered = 0.0
        self.last_tick = None
        self.scan_yaw = 0.0
        self.viewpoints = viewpoints(self.c.bounds, self.c.viewpoint_spacing, self.c.search_height)
        self.view_index = 0
        self.rejected = []
        self.scan_z = self.c.search_height
        self.settle_since = None
        self.ever_offboard = False

    def transition(self, state, now, reason=''):
        self.state, self.entered = state, now
        self.reason = reason or state
        self.settle_since = None

    def start(self, now, position, connected, armed):
        if self.state != 'WAIT_START' or not connected or armed:
            return False
        if not inside(position, self.c.bounds) or position[2] > 1.0:
            return False
        self.home, self.hold = tuple(position), tuple(position)
        self.started = now
        self.transition('PRIME', now)
        return True

    def estimate(self, e, now):
        if self.state not in ('DISCOVER', 'VIEWPOINT', 'APPROACH'):
            return
        if (e.stamp <= self.last_estimate_stamp or not 0 <= now-e.stamp <= self.c.target_max_age
                or not math.isfinite(e.uncertainty) or e.uncertainty > self.c.maximum_uncertainty
                or not inside(e.position, self.c.bounds, 0.5)):
            return
        self.last_estimate_stamp = e.stamp
        if any(distance(e.position, p) < self.c.candidate_radius for p in self.rejected):
            return
        if self.state == 'APPROACH':
            if distance(e.position, self.target) < self.c.candidate_radius:
                self.target = tuple(0.7*a+0.3*b for a, b in zip(self.target, e.position))
            return
        if self.candidate is None or distance(e.position, self.candidate) > self.c.candidate_radius:
            self.candidate, self.candidate_hits = e.position, 1
        else:
            self.candidate = tuple(0.5*a+0.5*b for a, b in zip(self.candidate, e.position))
            self.candidate_hits += 1
        if self.candidate_hits >= self.c.confirmation_count:
            self.target = self.candidate
            self.transition('APPROACH', now, 'Confirmed discovery; interrupting search')

    def downward(self, observation, now):
        if self.state not in ('APPROACH', 'ACQUIRE_DOWN', 'SCAN'):
            return
        if (not 0 <= now-observation.stamp <= self.c.target_max_age
                or (self.down and observation.stamp <= self.down.stamp)
                or not inside(observation.position, self.c.bounds, 0.5)
                or self.target is None
                or distance(observation.position, self.target) > self.c.candidate_radius):
            return
        self.down = observation
        payload = observation.payload
        if (payload and len(payload) == self.c.payload_length and
                all('0' <= char <= '9' for char in payload)):
            if payload != self.payload_candidate or observation.stamp-self.payload_stamp > self.c.target_max_age:
                self.payload_candidate, self.payload_hits = payload, 1
            else:
                self.payload_hits += 1
            self.payload_stamp = observation.stamp
            if self.payload_hits >= self.c.decode_confirmation_count:
                self.result = payload
                self.transition('RETURN', now, 'QR digits confirmed')
        else:
            self.payload_hits = 0

    def reject(self, now, position):
        if self.target:
            self.rejected.append(self.target)
        self.target = self.candidate = self.down = None
        self.candidate_hits = self.payload_hits = 0
        self.hold = (position[0], position[1], self.c.search_height)
        self.transition('DISCOVER', now, 'Candidate timed out; resuming discovery')

    def tick(self, now, position, yaw, connected, armed, mode, telemetry_age=0.0):
        dt = min(0.1, max(0.0, now-self.last_tick)) if self.last_tick is not None else 0.05
        if self.last_tick is not None and now < self.last_tick:
            self.transition('ABORTED', now, 'Simulation clock reset; restart mission process')
        self.last_tick = now
        if self.state in ('WAIT_START', 'ABORTED', 'DONE'):
            return Command(publish=False)
        if not connected or telemetry_age > self.c.telemetry_max_age:
            self.transition('ABORTED', now, 'Telemetry unavailable; defer to PX4 failsafe')
            return Command(publish=False)
        if not inside(position, self.c.bounds) or position[2] > self.c.maximum_height+0.5:
            self.transition('ABORTED', now, 'Outside flight envelope; request landing')
            return Command(mode='AUTO.LAND', publish=False)
        if self.state not in ('PRIME', 'ARMING', 'LAND') and not armed:
            self.transition('ABORTED', now, 'Aircraft unexpectedly disarmed')
            return Command(publish=False)
        if self.ever_offboard and mode != 'OFFBOARD' and self.state != 'LAND':
            self.transition('ABORTED', now, 'External mode change; relinquishing control')
            return Command(publish=False)
        if self.state not in ('RETURN', 'LAND') and now-self.started > self.c.mission_timeout:
            self.transition('RETURN', now, 'Mission time budget exhausted')
        cmd = Command(self.hold, yaw)
        if self.state == 'PRIME':
            if now-self.entered >= self.c.prime_duration:
                self.transition('ARMING', now)
        elif self.state == 'ARMING':
            cmd.mode = 'OFFBOARD' if mode != 'OFFBOARD' else ''
            cmd.arm = mode == 'OFFBOARD' and not armed
            if mode == 'OFFBOARD' and armed:
                self.ever_offboard = True
                self.hold = (self.home[0], self.home[1], self.c.search_height)
                self.transition('TAKEOFF', now)
            elif now-self.entered > self.c.arm_timeout:
                self.transition('ABORTED', now, 'OFFBOARD / arming timed out')
                cmd.publish = False
        elif self.state == 'TAKEOFF':
            cmd.position = self.hold
            if abs(position[2]-self.c.search_height) < self.c.position_tolerance:
                self.settle_since = self.settle_since or now
                if now-self.settle_since > 1.0:
                    self.scan_yaw = yaw
                    self.transition('DISCOVER', now)
            else:
                self.settle_since = None
            if now-self.entered > 30.0:
                self.transition('RETURN', now, 'Takeoff timed out')
        elif self.state == 'DISCOVER':
            cmd.position = self.hold
            self.scan_yaw += self.c.scan_rate*dt
            cmd.yaw = self.scan_yaw
            if now-self.last_estimate_stamp > self.c.target_max_age:
                self.candidate, self.candidate_hits = None, 0
            if now-self.entered >= 2*math.pi/max(self.c.scan_rate, 0.01):
                if self.view_index < len(self.viewpoints):
                    self.hold = self.viewpoints[self.view_index]
                    self.view_index += 1
                    self.transition('VIEWPOINT', now)
                else:
                    self.transition('RETURN', now, 'Discovery viewpoints exhausted')
        elif self.state == 'VIEWPOINT':
            cmd.position = step_toward(position, self.hold, self.c.approach_lookahead)
            cmd.yaw = math.atan2(self.hold[1]-position[1], self.hold[0]-position[0])
            if distance(position, self.hold) < self.c.position_tolerance:
                self.scan_yaw = yaw
                self.transition('DISCOVER', now)
            elif now-self.entered > self.c.approach_timeout:
                self.transition('RETURN', now, 'Viewpoint transit timed out')
        elif self.state == 'APPROACH':
            goal = (*self.target[:2], self.c.search_height)
            cmd.position = step_toward(position, goal, self.c.approach_lookahead)
            cmd.yaw = math.atan2(goal[1]-position[1], goal[0]-position[0])
            # Last estimate bridges the deliberate gap between the two camera views.
            if distance(position, goal) < self.c.position_tolerance:
                self.hold = goal
                self.transition('ACQUIRE_DOWN', now)
            elif now-self.entered > self.c.approach_timeout:
                self.reject(now, position)
        elif self.state == 'ACQUIRE_DOWN':
            if self.down and 0 <= now-self.down.stamp < self.c.target_max_age:
                self.scan_z = position[2]
                self.transition('SCAN', now)
            elif now-self.entered > self.c.handoff_timeout:
                self.reject(now, position)
            else:
                # Bounded square around last location; no blind descent.
                offsets = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
                dx, dy = offsets[min(int((now-self.entered)/3), len(offsets)-1)]
                p = (self.target[0]+dx, self.target[1]+dy, self.c.search_height)
                cmd.position = p if inside(p, self.c.bounds, 0.5) else self.hold
        elif self.state == 'SCAN':
            if self.down and 0 <= now-self.down.stamp < self.c.target_max_age:
                goal = (*self.down.position[:2], self.scan_z)
                cmd.position = step_toward(position, goal, 0.4)
                if distance(position, goal) < 0.25:
                    self.settle_since = self.settle_since or now
                    if now-self.settle_since > 0.75:
                        self.scan_z = max(self.c.minimum_height, self.scan_z-self.c.descent_rate*dt)
                        cmd.position = (*cmd.position[:2], self.scan_z)
                else:
                    self.settle_since = None
                self.hold = cmd.position
            else:
                # Freeze vertical position when imagery is stale; climb to reacquire.
                self.hold = (position[0], position[1], self.c.search_height)
                self.down = None
                self.transition('ACQUIRE_DOWN', now, 'Downward target lost')
                cmd.position = self.hold
            if now-self.entered > self.c.scan_timeout:
                self.reject(now, position)
        elif self.state == 'RETURN':
            goal = (self.home[0], self.home[1], self.c.search_height)
            cmd.position = step_toward(position, goal, self.c.approach_lookahead)
            if distance(position, goal) < self.c.position_tolerance and abs(position[2]-goal[2]) < 0.5:
                self.transition('LAND', now)
                cmd.mode, cmd.publish = 'AUTO.LAND', False
            elif now-self.entered > 60.0:
                self.transition('LAND', now, 'Return timed out; landing at current position')
                cmd.mode, cmd.publish = 'AUTO.LAND', False
        elif self.state == 'LAND':
            cmd.mode = 'AUTO.LAND' if mode == 'OFFBOARD' else ''
            cmd.publish = False
            if not armed:
                self.transition('DONE', now, 'Aircraft landed and disarmed')
            elif now-self.entered > self.c.landing_timeout:
                self.transition('ABORTED', now, 'Landing did not finish; inspect PX4 status')
        return cmd
