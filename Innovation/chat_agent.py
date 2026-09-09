"""
DroneChatAgent — connects to the local MCP server as a client, and drives
the Google Gemini tool-use loop so a plain chat message can trigger drone tools.

Runs entirely on the companion computer (e.g. Raspberry Pi). No SSH or separate machine required.
"""
import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Support google-genai (unified Google GenAI SDK) with fallback to google-generativeai
try:
    from google import genai
    from google.genai import types
    SDK_FLAVOR = "google-genai"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        from google.generativeai import types as genai_legacy_types
        SDK_FLAVOR = "google-generativeai"
    except ImportError:
        SDK_FLAVOR = None

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a drone control assistant. You have tools to connect to, arm, "
    "fly, and land a real (or bench-test) drone over MAVLink. Always call "
    "connect_drone first if you haven't already this session. Confirm "
    "risky actions (arm, goto, takeoff) in your reply before assuming they "
    "succeeded — check the tool result status."
)


def _clean_schema(schema: dict) -> dict:
    """Recursively cleans and formats an OpenAPI/JSON Schema for Gemini function declarations."""
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    allowed_keys = {"type", "properties", "required", "description", "items", "enum"}

    for k, v in schema.items():
        if k in allowed_keys:
            if k == "properties" and isinstance(v, dict):
                cleaned[k] = {
                    prop_name: _clean_schema(prop_val)
                    for prop_name, prop_val in v.items()
                }
            elif k == "items" and isinstance(v, dict):
                cleaned[k] = _clean_schema(v)
            else:
                cleaned[k] = v

    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"
    elif not cleaned:
        cleaned = {"type": "object", "properties": {}}

    return cleaned


class DroneChatAgent:
    def __init__(self, server_script: str = "mcp_server.py"):
        # Resolve script path across working directory, file location, and casing variations
        candidate_paths = [
            server_script,
            "MCP_Server.py" if server_script == "mcp_server.py" else "mcp_server.py",
            os.path.join(os.path.dirname(__file__), server_script),
            os.path.join(os.path.dirname(__file__), "MCP_Server.py" if server_script == "mcp_server.py" else "mcp_server.py"),
        ]
        resolved = next((p for p in candidate_paths if os.path.exists(p)), server_script)
        self.server_script = resolved

        self.model = MODEL
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools = []
        self.history: List[Any] = []
        self.client = None
        self.gemini_declarations: List[Any] = []
        self._legacy_chat = None

    async def connect(self):
        """Connect to MCP server and initialize Gemini client."""
        if SDK_FLAVOR is None:
            raise RuntimeError(
                "Neither 'google-genai' nor 'google-generativeai' is installed. "
                "Please run: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Obtain a key from https://aistudio.google.com and set GEMINI_API_KEY."
            )

        # Launch MCP server subprocess via stdio transport
        params = StdioServerParameters(command="python3", args=[self.server_script])
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        # Discover tools from MCP server
        listed = await self.session.list_tools()
        self.tools = listed.tools

        # Setup Gemini tool declarations and client
        if SDK_FLAVOR == "google-genai":
            self.client = genai.Client(api_key=api_key)
            self.gemini_declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters_json_schema=_clean_schema(t.inputSchema),
                )
                for t in self.tools
            ]
        elif SDK_FLAVOR == "google-generativeai":
            genai_legacy.configure(api_key=api_key)
            legacy_tools = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": _clean_schema(t.inputSchema),
                        }
                        for t in self.tools
                    ]
                }
            ]
            legacy_model = genai_legacy.GenerativeModel(
                model_name=self.model,
                system_instruction=SYSTEM_PROMPT,
                tools=legacy_tools,
            )
            self._legacy_chat = legacy_model.start_chat(history=[])

    async def _chat_genai(self, user_message: str) -> str:
        """Process chat message using google-genai SDK."""
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
        )
        reply_parts: List[str] = []

        while True:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[types.Tool(function_declarations=self.gemini_declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.2,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self.history,
                config=config,
            )

            if not response.candidates:
                break

            candidate = response.candidates[0]
            if candidate.content:
                self.history.append(candidate.content)

            # Collect any text response parts
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        reply_parts.append(part.text)

            # Detect function calls
            function_calls = getattr(response, "function_calls", None)
            if not function_calls and candidate.content and candidate.content.parts:
                function_calls = [
                    p.function_call for p in candidate.content.parts
                    if getattr(p, "function_call", None) is not None
                ]

            if not function_calls:
                break

            # Execute tool calls via MCP session
            tool_response_parts = []
            for call in function_calls:
                call_name = call.name
                call_args = call.args if isinstance(call.args, dict) else (dict(call.args) if call.args else {})
                try:
                    result = await self.session.call_tool(call_name, call_args)
                    text_result = "\n".join(
                        part.text for part in result.content if hasattr(part, "text")
                    )
                except Exception as err:
                    text_result = f"Error executing tool '{call_name}': {err}"

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=call_name,
                        response={"result": text_result},
                    )
                )

            self.history.append(
                types.Content(
                    role="tool",
                    parts=tool_response_parts,
                )
            )

        return "\n\n".join(reply_parts).strip() or "(no text response)"

    async def _chat_legacy(self, user_message: str) -> str:
        """Process chat message using google-generativeai legacy SDK."""
        reply_parts: List[str] = []
        response = await self._legacy_chat.send_message_async(user_message)

        while True:
            # Check text
            try:
                if response.text:
                    reply_parts.append(response.text)
            except Exception:
                pass

            # Detect function calls
            function_calls = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "function_call", None):
                        function_calls.append(part.function_call)

            if not function_calls:
                break

            # Execute each function call and send response back
            for call in function_calls:
                call_name = call.name
                call_args = dict(call.args) if call.args else {}
                try:
                    result = await self.session.call_tool(call_name, call_args)
                    text_result = "\n".join(
                        part.text for part in result.content if hasattr(part, "text")
                    )
                except Exception as err:
                    text_result = f"Error executing tool '{call_name}': {err}"

                response = await self._legacy_chat.send_message_async(
                    genai_legacy_types.Part.from_function_response(
                        name=call_name,
                        response={"result": text_result},
                    )
                )

        return "\n\n".join(reply_parts).strip() or "(no text response)"

    async def chat(self, user_message: str) -> str:
        """Send a user message to the Gemini agent and execute any requested drone tools."""
        if SDK_FLAVOR == "google-genai":
            return await self._chat_genai(user_message)
        elif SDK_FLAVOR == "google-generativeai":
            return await self._chat_legacy(user_message)
        else:
            raise RuntimeError("Neither 'google-genai' nor 'google-generativeai' is available.")

    async def close(self):
        """Close MCP session and subprocess."""
        await self.exit_stack.aclose()