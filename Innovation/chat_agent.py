"""
DroneChatAgent — connects to the local MCP server as a client, and drives
the AI tool-use loop (Groq or Gemini) so a plain chat message can trigger drone tools.
"""
import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Support Groq
try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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

SYSTEM_PROMPT = (
    "You are a drone control assistant. You have tools to connect to, arm, "
    "fly, and land a real (or bench-test) drone over MAVLink. Always call "
    "connect_drone first if you haven't already this session. Confirm "
    "risky actions (arm, goto, takeoff) in your reply before assuming they "
    "succeeded — check the tool result status."
)


def _clean_schema(schema: dict) -> dict:
    """Recursively cleans and formats an OpenAPI/JSON Schema for Gemini/Groq function declarations."""
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
        candidate_paths = [
            server_script,
            "MCP_Server.py" if server_script == "mcp_server.py" else "mcp_server.py",
            os.path.join(os.path.dirname(__file__), server_script),
            os.path.join(os.path.dirname(__file__), "MCP_Server.py" if server_script == "mcp_server.py" else "mcp_server.py"),
        ]
        resolved = next((p for p in candidate_paths if os.path.exists(p)), server_script)
        self.server_script = resolved

        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools = []
        
        # State for Gemini
        self.gemini_history: List[Any] = []
        self.gemini_client = None
        self.gemini_declarations: List[Any] = []
        self._legacy_chat = None
        
        # State for Groq
        self.groq_history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.groq_client = None
        self.groq_tools = []
        
        self._connected = False

    async def connect(self):
        """Connect to MCP server. Initialize AI clients dynamically upon chat if needed."""
        if self._connected:
            return

        # Launch MCP server subprocess via stdio transport
        params = StdioServerParameters(command=sys.executable, args=[self.server_script])
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        # Discover tools from MCP server
        listed = await self.session.list_tools()
        self.tools = listed.tools
        self._connected = True

    def _init_clients(self):
        """Initialize Groq or Gemini clients based on available API keys."""
        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        if not groq_key and not gemini_key:
            raise RuntimeError("No API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY.")

        # Setup Groq
        if groq_key and GROQ_AVAILABLE and not self.groq_client:
            self.groq_client = AsyncGroq(api_key=groq_key)
            self.groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": _clean_schema(
                            getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
                        ),
                    }
                }
                for t in self.tools
            ]

        # Setup Gemini
        if gemini_key and SDK_FLAVOR and not self.gemini_client and not self._legacy_chat:
            if SDK_FLAVOR == "google-genai":
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.gemini_declarations = [
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description or "",
                        parameters_json_schema=_clean_schema(
                            getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
                        ),
                    )
                    for t in self.tools
                ]
            elif SDK_FLAVOR == "google-generativeai":
                genai_legacy.configure(api_key=gemini_key)
                legacy_tools = [
                    {
                        "function_declarations": [
                            {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": _clean_schema(
                                    getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
                                ),
                            }
                            for t in self.tools
                        ]
                    }
                ]
                legacy_model = genai_legacy.GenerativeModel(
                    model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                    system_instruction=SYSTEM_PROMPT,
                    tools=legacy_tools,
                )
                self._legacy_chat = legacy_model.start_chat(history=[])

    async def _chat_groq(self, user_message: str) -> str:
        """Process chat message using Groq."""
        self.groq_history.append({"role": "user", "content": user_message})
        
        while True:
            response = await self.groq_client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=self.groq_history,
                tools=self.groq_tools,
                tool_choice="auto",
                temperature=0.2,
            )
            
            response_message = response.choices[0].message
            
            # Groq returns tool_calls or text content
            if response_message.tool_calls:
                # Add assistant message with tool calls to history
                self.groq_history.append(response_message.model_dump())
                
                for tool_call in response_message.tool_calls:
                    call_name = tool_call.function.name
                    try:
                        call_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        call_args = {}
                    
                    try:
                        try:
                            result = await self.session.call_tool(call_name, arguments=call_args)
                        except TypeError:
                            result = await self.session.call_tool(call_name, call_args)

                        text_result = "\n".join(
                            part.text for part in result.content if hasattr(part, "text")
                        )
                    except Exception as err:
                        text_result = f"Error executing tool '{call_name}': {err}"

                    # Add tool response to history
                    self.groq_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": call_name,
                        "content": text_result
                    })
            else:
                self.groq_history.append({"role": "assistant", "content": response_message.content})
                return response_message.content or "(no text response)"

    async def _chat_genai(self, user_message: str) -> str:
        """Process chat message using google-genai SDK."""
        self.gemini_history.append(
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

            response = await self.gemini_client.aio.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=self.gemini_history,
                config=config,
            )

            if not response.candidates:
                break

            candidate = response.candidates[0]
            if candidate.content:
                self.gemini_history.append(candidate.content)

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        reply_parts.append(part.text)

            function_calls = getattr(response, "function_calls", None)
            if not function_calls and candidate.content and candidate.content.parts:
                function_calls = [
                    p.function_call for p in candidate.content.parts
                    if getattr(p, "function_call", None) is not None
                ]

            if not function_calls:
                break

            tool_response_parts = []
            for call in function_calls:
                call_name = call.name
                call_args = call.args if isinstance(call.args, dict) else (dict(call.args) if call.args else {})
                try:
                    try:
                        result = await self.session.call_tool(call_name, arguments=call_args)
                    except TypeError:
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

            self.gemini_history.append(
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
            try:
                if response.text:
                    reply_parts.append(response.text)
            except Exception:
                pass

            function_calls = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "function_call", None):
                        function_calls.append(part.function_call)

            if not function_calls:
                break

            for call in function_calls:
                call_name = call.name
                call_args = dict(call.args) if call.args else {}
                try:
                    try:
                        result = await self.session.call_tool(call_name, arguments=call_args)
                    except TypeError:
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
        """Send a user message and execute drone tools. Tries Groq first, falls back to Gemini."""
        if not self._connected:
            await self.connect()
            
        self._init_clients()
        
        # Try Groq if configured
        if self.groq_client:
            try:
                return await self._chat_groq(user_message)
            except Exception as e:
                # If Groq fails and Gemini is not configured, re-raise
                if not (self.gemini_client or self._legacy_chat):
                    raise
                print(f"Groq API failed: {e}. Falling back to Gemini...")
                
        # Fallback to Gemini
        if self.gemini_client:
            return await self._chat_genai(user_message)
        elif self._legacy_chat:
            return await self._chat_legacy(user_message)
            
        raise RuntimeError("No AI client could handle the request.")

    async def close(self):
        """Close MCP session and subprocess."""
        await self.exit_stack.aclose()