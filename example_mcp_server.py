"""
Example FastMCP server for testing the streaming HTTP proxy.

This server implements various MCP features to thoroughly test proxy functionality:
- Tools (sync and async)
- Resources (static and dynamic)
- Prompts
- Streaming responses
- Large payloads
- Error handling
- Progress notifications

Usage:
    python example_mcp_server.py [port]

Default port is 9000.
"""

import asyncio
import sys
from datetime import datetime
from typing import AsyncIterator

from fastmcp import FastMCP, Context
from fastmcp.resources import FunctionResource


# Create the MCP server
mcp = FastMCP(name="TestProxyServer")


# ============== TOOLS ==============

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def concat(strings: list[str], separator: str = " ") -> str:
    """Concatenate a list of strings with a separator."""
    return separator.join(strings)


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the provided message."""
    return f"Echo: {message}"


@mcp.tool()
def get_timestamp() -> str:
    """Get the current server timestamp."""
    return datetime.now().isoformat()


@mcp.tool()
async def async_delay(seconds: float, message: str) -> str:
    """Wait for specified seconds then return the message."""
    await asyncio.sleep(seconds)
    return f"After {seconds}s: {message}"


@mcp.tool()
def generate_large_response(size_kb: int = 100) -> str:
    """Generate a large response of approximately the specified size in KB."""
    # Generate repeating pattern to reach target size
    pattern = "ABCDEFGHIJ" * 100  # 1000 chars
    repeats = (size_kb * 1024) // len(pattern) + 1
    return (pattern * repeats)[:size_kb * 1024]


@mcp.tool()
def process_binary_data(data: str) -> dict:
    """Process base64-encoded binary data and return stats."""
    import base64
    try:
        decoded = base64.b64decode(data)
        return {
            "original_length": len(data),
            "decoded_length": len(decoded),
            "first_bytes": list(decoded[:10]),
            "last_bytes": list(decoded[-10:]) if len(decoded) >= 10 else list(decoded),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def error_tool(error_type: str = "value") -> str:
    """A tool that intentionally raises errors for testing error handling."""
    if error_type == "value":
        raise ValueError("Intentional ValueError for testing")
    elif error_type == "type":
        raise TypeError("Intentional TypeError for testing")
    elif error_type == "runtime":
        raise RuntimeError("Intentional RuntimeError for testing")
    elif error_type == "key":
        d = {}
        return d["nonexistent"]  # type: ignore
    else:
        return f"Unknown error type: {error_type}"


def _build_nested(depth: int) -> dict:
    """Helper function to build nested dictionary."""
    if depth <= 0:
        return {"value": "leaf", "timestamp": datetime.now().isoformat()}

    return {
        "level": depth,
        "nested": _build_nested(depth - 1),
        "array": [{"index": i, "data": f"item_{i}"} for i in range(3)],
        "metadata": {
            "created": datetime.now().isoformat(),
            "depth_remaining": depth,
        }
    }


@mcp.tool()
def nested_data(depth: int = 3) -> dict:
    """Generate nested dictionary data structure."""
    return _build_nested(depth)


@mcp.tool()
async def stream_numbers(count: int = 10, delay_ms: int = 100) -> list[int]:
    """
    Generate a sequence of numbers with delays between each.
    Useful for testing streaming behavior.
    """
    results = []
    for i in range(count):
        await asyncio.sleep(delay_ms / 1000.0)
        results.append(i)
    return results


@mcp.tool()
def unicode_test(text: str = "Hello 世界 🌍 مرحبا") -> dict:
    """Test Unicode handling through the proxy."""
    return {
        "original": text,
        "length": len(text),
        "encoded_length": len(text.encode('utf-8')),
        "reversed": text[::-1],
        "upper": text.upper(),
    }


@mcp.tool()
def headers_info() -> dict:
    """Return information about the request (what the server sees)."""
    return {
        "server_name": "TestProxyServer",
        "server_time": datetime.now().isoformat(),
        "note": "This tool returns static server info since we're stateless",
    }


# ============== RESOURCES ==============

@mcp.resource("config://settings")
def get_settings() -> str:
    """Server configuration settings."""
    import json
    return json.dumps({
        "server_name": "TestProxyServer",
        "version": "1.0.0",
        "features": ["tools", "resources", "prompts"],
        "max_request_size": "10MB",
        "timeout": 30,
    }, indent=2)


@mcp.resource("data://sample.json")
def get_sample_data() -> str:
    """Sample JSON data for testing."""
    import json
    return json.dumps({
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
            {"id": 3, "name": "Charlie", "role": "user"},
        ],
        "metadata": {
            "total": 3,
            "page": 1,
            "generated": datetime.now().isoformat(),
        }
    }, indent=2)


@mcp.resource("text://readme")
def get_readme() -> str:
    """A text readme file."""
    return """# Test MCP Server

This is a test MCP server for validating the streaming HTTP proxy.

## Features
- Multiple tool types (sync, async, error-generating)
- Resources (static and dynamic)
- Prompts for common operations
- Large payload handling
- Unicode support
- Nested data structures

## Usage
Connect via the MCP client using streamable-http transport.
"""


@mcp.resource("data://large.txt")
def get_large_resource() -> str:
    """A large text resource for testing streaming."""
    lines = []
    for i in range(1000):
        lines.append(f"Line {i:04d}: " + "x" * 100)
    return "\n".join(lines)


@mcp.resource("data://timestamp")
def get_dynamic_timestamp() -> str:
    """A dynamic resource that returns current timestamp."""
    return datetime.now().isoformat()


# ============== PROMPTS ==============

@mcp.prompt()
def analyze_data(data_description: str) -> str:
    """Create a prompt for analyzing data."""
    return f"""Please analyze the following data:

{data_description}

Provide:
1. A summary of the data
2. Key insights
3. Any anomalies or issues
4. Recommendations for next steps
"""


@mcp.prompt()
def code_review(language: str, code_snippet: str) -> str:
    """Create a prompt for code review."""
    return f"""Please review the following {language} code:

```{language}
{code_snippet}
```

Consider:
- Code quality and style
- Potential bugs or issues
- Performance implications
- Security concerns
- Suggestions for improvement
"""


@mcp.prompt()
def explain_concept(concept: str, audience: str = "beginner") -> str:
    """Create a prompt for explaining a concept."""
    return f"""Explain the concept of "{concept}" for a {audience} audience.

Include:
- A clear definition
- Real-world examples
- Common misconceptions
- Related concepts
"""


@mcp.prompt()
def debug_error(error_message: str, context: str = "") -> str:
    """Create a prompt for debugging an error."""
    ctx = f"\nContext: {context}" if context else ""
    return f"""Help debug the following error:

Error: {error_message}{ctx}

Please:
1. Explain what this error means
2. Identify likely causes
3. Suggest debugging steps
4. Provide potential solutions
"""


# ============== MAIN ==============

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

    print(f"Starting MCP server on port {port}...")
    print(f"MCP endpoint: http://127.0.0.1:{port}/mcp")
    print("\nAvailable tools:")
    print("  - add, multiply, concat, echo")
    print("  - get_timestamp, async_delay")
    print("  - generate_large_response, process_binary_data")
    print("  - error_tool, nested_data, stream_numbers")
    print("  - unicode_test, headers_info")
    print("\nAvailable resources:")
    print("  - config://settings")
    print("  - data://sample.json, data://large.txt, data://timestamp")
    print("  - text://readme")
    print("\nAvailable prompts:")
    print("  - analyze_data, code_review, explain_concept, debug_error")

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=port,
        path="/mcp",
        stateless_http=True,  # Recommended for production/proxy scenarios
        json_response=True,   # Use JSON responses for better compatibility
    )
