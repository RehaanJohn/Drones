"""Compatibility entry point. Prefer running server.py for new configurations."""
from server import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio")
