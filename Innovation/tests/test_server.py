"""Exercise the actual MCP boundary without flight hardware."""
import asyncio
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from fastmcp import Client

INNOVATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INNOVATION))
import server


TOOLS = {
    "connect_drone": ("connect", {}, (), {}),
    "arm_drone": ("arm", {"force": True}, (), {"force": True}),
    "disarm_drone": ("disarm", {}, (), {}),
    "takeoff": ("takeoff", {"altitude_m": 3.0}, (3.0,), {}),
    "goto_location": ("goto", {"lat": 12.0, "lon": 77.0, "alt_m": 3.0}, (12.0, 77.0, 3.0), {}),
    "return_to_launch": ("rtl", {}, (), {}),
    "land": ("land", {}, (), {}),
    "set_flight_mode": ("set_mode", {"mode": "LOITER"}, ("LOITER",), {}),
    "get_telemetry": ("get_telemetry", {}, (), {}),
}


class ServerTests(unittest.TestCase):
    def test_tool_discovery_forwarding_and_cleanup(self):
        fake = Mock()
        for method, *_ in TOOLS.values():
            getattr(fake, method).return_value = {"status": "test"}

        async def exercise():
            async with Client(server.mcp) as client:
                tools = {tool.name: tool for tool in await client.list_tools()}
                self.assertEqual(set(tools), set(TOOLS))
                self.assertEqual(tools["arm_drone"].inputSchema["properties"]["force"]["default"], False)
                fake.connect.assert_not_called()
                for name, (method, arguments, args, kwargs) in TOOLS.items():
                    result = await client.call_tool(name, arguments)
                    self.assertEqual(result.data, {"status": "test"})
                    getattr(fake, method).assert_called_once_with(*args, **kwargs)
                invalid = await client.call_tool("takeoff", {}, raise_on_error=False)
                self.assertTrue(invalid.is_error)
                fake.takeoff.assert_called_once()

        with patch.object(server, "drone", fake):
            asyncio.run(exercise())
            fake.close.assert_called_once()

    def test_stdio_launch_from_unrelated_directory(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def exercise(script):
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(INNOVATION / script)],
                cwd=str(INNOVATION.parent.parent),
                env={**os.environ, "DRONE_CONN": "udpin:127.0.0.1:14599", "DRONE_BAUD": "57600"},
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    self.assertEqual({t.name for t in (await client.list_tools()).tools}, set(TOOLS))
                    result = await client.call_tool("get_telemetry", {})
                    self.assertTrue(result.isError)
                    self.assertIn("Not connected", result.content[0].text)

        for script in ("server.py", "MCP_Server.py"):
            with self.subTest(script=script):
                asyncio.run(asyncio.wait_for(exercise(script), timeout=30))


if __name__ == "__main__":
    unittest.main()
