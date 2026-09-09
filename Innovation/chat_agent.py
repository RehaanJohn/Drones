"""
DroneChatAgent — connects to the local MCP server as a client, and drives
the Anthropic tool-use loop so a plain chat message can trigger drone tools.

Runs entirely on the Pi. No SSH, no separate machine required.
"""
import asyncio
from contextlib import AsyncExitStack

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are a drone control assistant. You have tools to connect to, arm, "
    "fly, and land a real (or bench-test) drone over MAVLink. Always call "
    "connect_drone first if you haven't already this session. Confirm "
    "risky actions (arm, goto, takeoff) in your reply before assuming they "
    "succeeded — check the tool result status."
)


class DroneChatAgent:
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self.anthropic = Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools = []
        self.history = []

    async def connect(self):
        params = StdioServerParameters(command="python3", args=[self.server_script])
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        listed = await self.session.list_tools()
        self.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in listed.tools
        ]

    async def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        reply_parts = []

        while True:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.history,
            )

            assistant_content = []
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    reply_parts.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_calls.append(block)

            self.history.append({"role": "assistant", "content": assistant_content})

            if not tool_calls:
                break

            tool_results = []
            for call in tool_calls:
                result = await self.session.call_tool(call.name, call.input)
                text = "\n".join(
                    part.text for part in result.content if hasattr(part, "text")
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": text,
                })
            self.history.append({"role": "user", "content": tool_results})

        return "\n\n".join(reply_parts).strip() or "(no text response)"

    async def close(self):
        await self.exit_stack.aclose()