"""
DroneController — thin wrapper around pymavlink for use by the MCP tool layer.

Handles: connecting, arming/disarming, mode changes, guided-mode waypoint
navigation, RTL, landing, and telemetry polling.

This talks directly to the Pixhawk over MAVLink (USB or TELEM2 serial).
Run it on the Raspberry Pi, alongside MAVProxy if you want Mission Planner
connected at the same time (see README for the MAVProxy routing setup).
"""
import time

from pymavlink import mavutil


class DroneController:
    def __init__(self, connection_string: str = "/dev/ttyACM0", baud: int = 115200):
        self.connection_string = connection_string
        self.baud = baud
        self.master = None

    def connect(self, timeout: int = 30, retries: int = 3, retry_delay: float = 3.0) -> dict:
        """Open the MAVLink connection and wait for the first heartbeat.

        Retries on failure — Pixhawks over USB enumerate twice on power-up
        (bootloader, then firmware, ~5s apart), so a connection attempt
        made during that window can grab a stale file descriptor. A short
        retry with a fresh connection object rides past that."""
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                if self.master:
                    self.master.close()
                self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud)
                self.master.wait_heartbeat(timeout=timeout)
                break
            except Exception as exc:
                last_error = exc
                if self.master:
                    self.master.close()
                    self.master = None
                if attempt < retries:
                    time.sleep(retry_delay)
        else:
            raise ConnectionError(
                f"Failed to connect to {self.connection_string} after {retries} attempts: {last_error}"
            )

        return {
            "status": "connected",
            "system_id": self.master.target_system,
            "component_id": self.master.target_component,
        }

    def _ensure_connected(self):
        if self.master is None:
            raise RuntimeError("Not connected. Call connect() first.")

    def set_mode(self, mode: str) -> dict:
        self._ensure_connected()
        mode_id = self.master.mode_mapping().get(mode.upper())
        if mode_id is None:
            available = list(self.master.mode_mapping().keys())
            raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        return {"status": "mode_set_requested", "mode": mode.upper()}

    def arm(self, force: bool = False) -> dict:
        self._ensure_connected()
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,                       # 1 = arm
            21196 if force else 0,   # magic number bypasses pre-arm checks
            0, 0, 0, 0, 0,
        )
        return self._wait_ack("ARM")

    def disarm(self) -> dict:
        self._ensure_connected()
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        return self._wait_ack("DISARM")

    def takeoff(self, altitude_m: float) -> dict:
        self._ensure_connected()
        self.set_mode("GUIDED")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )
        return self._wait_ack("TAKEOFF")

    def goto(self, lat: float, lon: float, alt_m: float) -> dict:
        """Send the vehicle to a lat/lon at a relative altitude. Requires GUIDED mode."""
        self._ensure_connected()
        self.set_mode("GUIDED")
        self.master.mav.mission_item_send(
            self.master.target_system, self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2, 0,           # current=2 (guided-mode single item), autocontinue=0
            0, 0, 0, 0,     # param1-4 unused for simple waypoint
            lat, lon, alt_m,
        )
        return {"status": "waypoint_sent", "lat": lat, "lon": lon, "alt_m": alt_m}

    def rtl(self) -> dict:
        self._ensure_connected()
        return self.set_mode("RTL")

    def land(self) -> dict:
        self._ensure_connected()
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        return self._wait_ack("LAND")

    def get_telemetry(self) -> dict:
        self._ensure_connected()
        # Drain a few messages so cached values are fresh
        for _ in range(5):
            self.master.recv_match(blocking=True, timeout=1)

        pos = self.master.messages.get("GLOBAL_POSITION_INT")
        hud = self.master.messages.get("VFR_HUD")
        hb = self.master.messages.get("HEARTBEAT")

        return {
            "lat": pos.lat / 1e7 if pos else None,
            "lon": pos.lon / 1e7 if pos else None,
            "relative_alt_m": pos.relative_alt / 1000 if pos else None,
            "groundspeed_mps": hud.groundspeed if hud else None,
            "heading_deg": hud.heading if hud else None,
            "armed": bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) if hb else None,
            "mode": self.master.flightmode if hb else None,
        }

    def _wait_ack(self, label: str, timeout: int = 5) -> dict:
        ack = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
        if ack is None:
            return {"status": "no_ack", "command": label}
        return {
            "status": "ok" if ack.result == 0 else "failed",
            "command": label,
            "result_code": ack.result,
        }

    def close(self):
        if self.master:
            self.master.close()
            self.master = None