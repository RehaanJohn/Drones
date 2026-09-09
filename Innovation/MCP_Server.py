"""
MCP server exposing drone control as chat-callable tools.

Run on the Raspberry Pi:
    python mcp_server.py

Override the connection with env vars:
    DRONE_CONN=/dev/ttyACM0   (or a MAVProxy UDP output, e.g. udp:127.0.0.1:14550)
    DRONE_BAUD=115200

Point your MCP client (Claude Desktop / Claude Code config, or any MCP client)
at this script as a stdio server. See README for the config snippet.

SAFETY NOTE (proof-of-concept only): there is no geofence or bounds check
here. Before flying for real, add altitude/distance limits and a
confirmation step in goto_location / arm_drone before this touches a
vehicle with props on.
"""
import os
from mcp.server.fastmcp import FastMCP
from drone_controller import DroneController

CONN = os.environ.get("DRONE_CONN", "/dev/ttyACM0")
BAUD = int(os.environ.get("DRONE_BAUD", "115200"))

mcp = FastMCP("Drone Control")
drone = DroneController(connection_string=CONN, baud=BAUD)


@mcp.tool()
def connect_drone() -> dict:
    """Connect to the Pixhawk over MAVLink. Call this before any other tool."""
    return drone.connect()


@mcp.tool()
def arm_drone(force: bool = False) -> dict:
    """Arm the drone's motors. The vehicle must be in a mode that allows arming
    (e.g. GUIDED or STABILIZE) and pass pre-arm checks unless force=True."""
    return drone.arm(force=force)


@mcp.tool()
def disarm_drone() -> dict:
    """Disarm the drone's motors immediately."""
    return drone.disarm()


@mcp.tool()
def takeoff(altitude_m: float) -> dict:
    """Switch to GUIDED mode and take off to the given altitude in meters."""
    return drone.takeoff(altitude_m)


@mcp.tool()
def goto_location(lat: float, lon: float, alt_m: float) -> dict:
    """Send the drone to a GPS location (latitude, longitude) at the given
    relative altitude in meters. Switches to GUIDED mode first."""
    return drone.goto(lat, lon, alt_m)


@mcp.tool()
def return_to_launch() -> dict:
    """Command the drone to RTL — return to the home position and land."""
    return drone.rtl()


@mcp.tool()
def land() -> dict:
    """Command the drone to land at its current position."""
    return drone.land()


@mcp.tool()
def set_flight_mode(mode: str) -> dict:
    """Set the flight mode directly, e.g. GUIDED, LOITER, RTL, LAND, STABILIZE."""
    return drone.set_mode(mode)


@mcp.tool()
def get_telemetry() -> dict:
    """Get current position, groundspeed, heading, armed state, and flight mode."""
    return drone.get_telemetry()


if __name__ == "__main__":
    mcp.run(transport="stdio")