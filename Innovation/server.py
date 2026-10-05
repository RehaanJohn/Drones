"""Pixhawk tools for Claude Desktop and other stdio MCP clients."""
import os
from contextlib import asynccontextmanager
from functools import wraps
from threading import RLock

from fastmcp import FastMCP
from Drone_Controller import DroneController

CONN = os.environ.get("DRONE_CONN", "/dev/ttyACM0")
BAUD = int(os.environ.get("DRONE_BAUD", "115200"))
drone = DroneController(connection_string=CONN, baud=BAUD)
_lock = RLock()


def serialized(function):
    """Serialize access to the shared MAVLink connection and ACK stream."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _lock:
            return function(*args, **kwargs)
    return wrapped


@asynccontextmanager
async def lifespan(server):
    # Hardware is opened only when connect_drone is called.
    try:
        yield
    finally:
        with _lock:
            drone.close()


mcp = FastMCP(
    "Pixhawk",
    instructions=(
        "Call connect_drone before using other tools. Check returned status: "
        "a sent command does not confirm the vehicle reached its target. "
        "Ask the user to confirm arming, takeoff, or waypoint movement before "
        "calling those tools."
    ),
    lifespan=lifespan,
)


@mcp.tool()
@serialized
def connect_drone() -> dict:
    """Connect to the Pixhawk over MAVLink. Call this before any other tool."""
    return drone.connect()


@mcp.tool()
@serialized
def arm_drone(force: bool = False) -> dict:
    """Arm the drone's motors. The vehicle must be in a mode that allows arming
    (e.g. GUIDED or STABILIZE) and pass pre-arm checks unless force=True."""
    return drone.arm(force=force)


@mcp.tool()
@serialized
def disarm_drone() -> dict:
    """Disarm the drone's motors immediately."""
    return drone.disarm()


@mcp.tool()
@serialized
def takeoff(altitude_m: float) -> dict:
    """Switch to GUIDED mode and take off to the given altitude in meters."""
    return drone.takeoff(altitude_m)


@mcp.tool()
@serialized
def goto_location(lat: float, lon: float, alt_m: float) -> dict:
    """Send the drone to a GPS location (latitude, longitude) at the given
    relative altitude in meters. Switches to GUIDED mode first."""
    return drone.goto(lat, lon, alt_m)


@mcp.tool()
@serialized
def return_to_launch() -> dict:
    """Command the drone to RTL — return to the home position and land."""
    return drone.rtl()


@mcp.tool()
@serialized
def land() -> dict:
    """Command the drone to land at its current position."""
    return drone.land()


@mcp.tool()
@serialized
def set_flight_mode(mode: str) -> dict:
    """Set the flight mode directly, e.g. GUIDED, LOITER, RTL, LAND, STABILIZE."""
    return drone.set_mode(mode)


@mcp.tool()
@serialized
def get_telemetry() -> dict:
    """Get current position, groundspeed, heading, armed state, and flight mode."""
    return drone.get_telemetry()


if __name__ == "__main__":
    mcp.run(transport="stdio")
