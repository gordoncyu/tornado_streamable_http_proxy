"""
Comprehensive tests for the streaming HTTP proxy using FastMCP.

These tests verify that the proxy correctly handles all aspects of
MCP's streamable-http transport protocol by running an actual FastMCP
server and client through the proxy.

Requirements:
    pip install fastmcp pytest pytest-asyncio

Usage:
    pytest test_mcp_proxy.py -v

The tests will:
1. Start an example FastMCP server
2. Start the Tornado streaming proxy
3. Connect an MCP client through the proxy
4. Test all MCP features (tools, resources, prompts)
5. Verify streaming, large payloads, errors, etc.
"""

import asyncio
import base64
import json
import subprocess
import sys
import time
from typing import Generator

import pytest

# Check for required packages
try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

try:
    import tornado.httpserver
    import tornado.ioloop
    import tornado.web
    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not FASTMCP_AVAILABLE or not TORNADO_AVAILABLE,
    reason="fastmcp and tornado packages required. Install with: pip install fastmcp"
)


# Port configuration
MCP_SERVER_PORT = 9200
PROXY_PORT = 9201


class ProxyServer:
    """Manages the Tornado proxy server in a background thread."""

    def __init__(self, target_host: str, port: int):
        self.target_host = target_host
        self.port = port
        self.server = None
        self.io_loop = None
        self._thread = None

    def start(self):
        """Start the proxy server in a background thread."""
        import threading

        def run_server():
            # Import here to avoid issues with event loop
            import asyncio
            import tornado.ioloop
            from streaming_proxy import StreamingProxyHandler

            # Create a new event loop for this thread
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.io_loop = tornado.ioloop.IOLoop.current()

            app = tornado.web.Application([
                (r".*", StreamingProxyHandler, {"target_host": self.target_host}),
            ])

            self.server = tornado.httpserver.HTTPServer(app)
            self.server.listen(self.port)

            self.io_loop.start()

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()

        # Give the server time to start
        time.sleep(0.5)

    def stop(self):
        """Stop the proxy server."""
        if self.io_loop:
            self.io_loop.add_callback(self.io_loop.stop)
        if self._thread:
            self._thread.join(timeout=2)


class MCPServerProcess:
    """Manages the FastMCP server as a subprocess."""

    def __init__(self, port: int):
        self.port = port
        self.process = None

    def start(self):
        """Start the MCP server subprocess."""
        self.process = subprocess.Popen(
            [sys.executable, "example_mcp_server.py", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/home/gordo/work/streamable-http-proxy",
        )

        # Wait for server to be ready
        time.sleep(2)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"MCP server failed to start:\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

    def stop(self):
        """Stop the MCP server subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


@pytest.fixture(scope="module")
def mcp_server() -> Generator[MCPServerProcess, None, None]:
    """Fixture to start/stop the MCP server."""
    server = MCPServerProcess(MCP_SERVER_PORT)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def proxy_server(mcp_server: MCPServerProcess) -> Generator[ProxyServer, None, None]:
    """Fixture to start/stop the proxy server."""
    target = f"http://127.0.0.1:{MCP_SERVER_PORT}"
    proxy = ProxyServer(target, PROXY_PORT)
    proxy.start()
    yield proxy
    proxy.stop()


@pytest.fixture
def proxy_url(proxy_server: ProxyServer) -> str:
    """Get the proxy URL for MCP client."""
    return f"http://127.0.0.1:{PROXY_PORT}/mcp"


@pytest.fixture
def direct_url(mcp_server: MCPServerProcess) -> str:
    """Get the direct MCP server URL (for comparison tests)."""
    return f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp"


# ============== BASIC CONNECTIVITY TESTS ==============

class TestBasicConnectivity:
    """Test basic proxy connectivity and MCP initialization."""

    @pytest.mark.asyncio
    async def test_client_connects_through_proxy(self, proxy_url: str):
        """Test that MCP client can connect through the proxy."""
        client = Client(proxy_url)
        async with client:
            # Connection successful if we get here
            assert client.is_connected()

    @pytest.mark.asyncio
    async def test_ping_through_proxy(self, proxy_url: str):
        """Test ping operation through proxy."""
        client = Client(proxy_url)
        async with client:
            # ping() should succeed without error
            await client.ping()

    @pytest.mark.asyncio
    async def test_list_tools_through_proxy(self, proxy_url: str):
        """Test listing available tools through proxy."""
        client = Client(proxy_url)
        async with client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            # Verify expected tools are present
            assert "add" in tool_names
            assert "multiply" in tool_names
            assert "echo" in tool_names
            assert "generate_large_response" in tool_names

    @pytest.mark.asyncio
    async def test_list_resources_through_proxy(self, proxy_url: str):
        """Test listing available resources through proxy."""
        client = Client(proxy_url)
        async with client:
            resources = await client.list_resources()
            resource_uris = [str(r.uri) for r in resources]

            assert any("settings" in uri for uri in resource_uris)
            assert any("sample.json" in uri for uri in resource_uris)

    @pytest.mark.asyncio
    async def test_list_prompts_through_proxy(self, proxy_url: str):
        """Test listing available prompts through proxy."""
        client = Client(proxy_url)
        async with client:
            prompts = await client.list_prompts()
            prompt_names = [p.name for p in prompts]

            assert "analyze_data" in prompt_names
            assert "code_review" in prompt_names


# ============== TOOL TESTS ==============

class TestTools:
    """Test MCP tool calls through the proxy."""

    @pytest.mark.asyncio
    async def test_add_tool(self, proxy_url: str):
        """Test simple addition tool."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("add", {"a": 5, "b": 3})
            assert len(result.content) > 0
            assert "8" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_multiply_tool(self, proxy_url: str):
        """Test multiplication tool."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("multiply", {"a": 2.5, "b": 4.0})
            assert len(result.content) > 0
            assert "10" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_concat_tool(self, proxy_url: str):
        """Test string concatenation tool."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool(
                "concat",
                {"strings": ["Hello", "World", "Test"], "separator": "-"}
            )
            assert "Hello-World-Test" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_echo_tool(self, proxy_url: str):
        """Test echo tool."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("echo", {"message": "Proxy test message"})
            assert "Proxy test message" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_timestamp_tool(self, proxy_url: str):
        """Test timestamp tool returns valid timestamp."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("get_timestamp", {})
            # Should be an ISO format timestamp
            text = str(result.content[0].text)
            assert "T" in text  # ISO format contains T
            assert "-" in text  # Date separator

    @pytest.mark.asyncio
    async def test_async_delay_tool(self, proxy_url: str):
        """Test async tool with delay."""
        client = Client(proxy_url)
        async with client:
            start = time.time()
            result = await client.call_tool(
                "async_delay",
                {"seconds": 0.5, "message": "delayed response"}
            )
            elapsed = time.time() - start

            assert "delayed response" in str(result.content[0].text)
            assert elapsed >= 0.4  # Should have some delay

    @pytest.mark.asyncio
    async def test_nested_data_tool(self, proxy_url: str):
        """Test tool returning nested JSON structure."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("nested_data", {"depth": 3})
            text = str(result.content[0].text)

            # Parse the JSON response
            data = json.loads(text)
            assert data["level"] == 3
            assert "nested" in data
            assert data["nested"]["level"] == 2

    @pytest.mark.asyncio
    async def test_unicode_tool(self, proxy_url: str):
        """Test Unicode handling through proxy."""
        client = Client(proxy_url)
        async with client:
            test_text = "Hello 世界 🌍 مرحبا Привет"
            result = await client.call_tool("unicode_test", {"text": test_text})
            text = str(result.content[0].text)

            data = json.loads(text)
            assert data["original"] == test_text
            assert data["reversed"] == test_text[::-1]


# ============== LARGE PAYLOAD TESTS ==============

class TestLargePayloads:
    """Test large payload handling through the proxy."""

    @pytest.mark.asyncio
    async def test_large_response_10kb(self, proxy_url: str):
        """Test 10KB response through proxy."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool(
                "generate_large_response",
                {"size_kb": 10}
            )
            text = str(result.content[0].text)
            assert len(text) >= 10 * 1024

    @pytest.mark.asyncio
    async def test_large_response_100kb(self, proxy_url: str):
        """Test 100KB response through proxy."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool(
                "generate_large_response",
                {"size_kb": 100}
            )
            text = str(result.content[0].text)
            assert len(text) >= 100 * 1024

    @pytest.mark.asyncio
    async def test_large_response_500kb(self, proxy_url: str):
        """Test 500KB response through proxy."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool(
                "generate_large_response",
                {"size_kb": 500}
            )
            text = str(result.content[0].text)
            assert len(text) >= 500 * 1024

    @pytest.mark.asyncio
    async def test_binary_data_through_proxy(self, proxy_url: str):
        """Test binary data encoded as base64 through proxy."""
        client = Client(proxy_url)
        async with client:
            # Create binary data
            binary_data = bytes(range(256)) * 10  # 2560 bytes
            encoded = base64.b64encode(binary_data).decode('utf-8')

            result = await client.call_tool(
                "process_binary_data",
                {"data": encoded}
            )
            text = str(result.content[0].text)
            data = json.loads(text)

            assert data["decoded_length"] == 2560
            assert data["first_bytes"] == list(range(10))


# ============== RESOURCE TESTS ==============

class TestResources:
    """Test MCP resource access through the proxy."""

    @pytest.mark.asyncio
    async def test_read_settings_resource(self, proxy_url: str):
        """Test reading config://settings resource."""
        client = Client(proxy_url)
        async with client:
            content = await client.read_resource("config://settings")
            text = str(content[0].text)

            data = json.loads(text)
            assert data["server_name"] == "TestProxyServer"
            assert data["version"] == "1.0.0"
            assert "tools" in data["features"]

    @pytest.mark.asyncio
    async def test_read_sample_json_resource(self, proxy_url: str):
        """Test reading data://sample.json resource."""
        client = Client(proxy_url)
        async with client:
            content = await client.read_resource("data://sample.json")
            text = str(content[0].text)

            data = json.loads(text)
            assert len(data["users"]) == 3
            assert data["users"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_read_text_resource(self, proxy_url: str):
        """Test reading text://readme resource."""
        client = Client(proxy_url)
        async with client:
            content = await client.read_resource("text://readme")
            text = str(content[0].text)

            assert "# Test MCP Server" in text
            assert "streaming HTTP proxy" in text

    @pytest.mark.asyncio
    async def test_read_large_resource(self, proxy_url: str):
        """Test reading large data://large.txt resource."""
        client = Client(proxy_url)
        async with client:
            content = await client.read_resource("data://large.txt")
            text = str(content[0].text)

            # Should have 1000 lines
            lines = text.strip().split("\n")
            assert len(lines) == 1000
            assert "Line 0000:" in lines[0]
            assert "Line 0999:" in lines[-1]

    @pytest.mark.asyncio
    async def test_read_dynamic_resource(self, proxy_url: str):
        """Test reading dynamic data://timestamp resource."""
        client = Client(proxy_url)
        async with client:
            content1 = await client.read_resource("data://timestamp")
            await asyncio.sleep(0.1)
            content2 = await client.read_resource("data://timestamp")

            # Timestamps should be different (dynamic)
            text1 = str(content1[0].text)
            text2 = str(content2[0].text)
            # They might be the same if very fast, but format should be valid
            assert "T" in text1  # ISO format


# ============== PROMPT TESTS ==============

class TestPrompts:
    """Test MCP prompt rendering through the proxy."""

    @pytest.mark.asyncio
    async def test_analyze_data_prompt(self, proxy_url: str):
        """Test analyze_data prompt."""
        client = Client(proxy_url)
        async with client:
            messages = await client.get_prompt(
                "analyze_data",
                {"data_description": "Sales data for Q4 2024"}
            )

            # Should have at least one message
            assert len(messages.messages) > 0
            text = str(messages.messages[0].content)
            assert "Sales data for Q4 2024" in text
            assert "summary" in text.lower()

    @pytest.mark.asyncio
    async def test_code_review_prompt(self, proxy_url: str):
        """Test code_review prompt."""
        client = Client(proxy_url)
        async with client:
            code = "def hello(): print('world')"
            messages = await client.get_prompt(
                "code_review",
                {"language": "python", "code_snippet": code}
            )

            text = str(messages.messages[0].content)
            assert "python" in text.lower()
            assert code in text

    @pytest.mark.asyncio
    async def test_explain_concept_prompt(self, proxy_url: str):
        """Test explain_concept prompt."""
        client = Client(proxy_url)
        async with client:
            messages = await client.get_prompt(
                "explain_concept",
                {"concept": "recursion", "audience": "intermediate"}
            )

            text = str(messages.messages[0].content)
            assert "recursion" in text
            assert "intermediate" in text

    @pytest.mark.asyncio
    async def test_debug_error_prompt(self, proxy_url: str):
        """Test debug_error prompt."""
        client = Client(proxy_url)
        async with client:
            messages = await client.get_prompt(
                "debug_error",
                {
                    "error_message": "IndexError: list index out of range",
                    "context": "Iterating over user list"
                }
            )

            text = str(messages.messages[0].content)
            assert "IndexError" in text
            assert "user list" in text


# ============== ERROR HANDLING TESTS ==============

class TestErrorHandling:
    """Test error handling through the proxy."""

    @pytest.mark.asyncio
    async def test_tool_value_error(self, proxy_url: str):
        """Test that ValueError from tool is properly propagated."""
        client = Client(proxy_url)
        async with client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool("error_tool", {"error_type": "value"})

            assert "ValueError" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_tool_type_error(self, proxy_url: str):
        """Test that TypeError from tool is properly propagated."""
        client = Client(proxy_url)
        async with client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool("error_tool", {"error_type": "type"})

            assert "TypeError" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_nonexistent_tool(self, proxy_url: str):
        """Test calling a tool that doesn't exist."""
        client = Client(proxy_url)
        async with client:
            with pytest.raises(Exception):
                await client.call_tool("nonexistent_tool_xyz", {})

    @pytest.mark.asyncio
    async def test_invalid_tool_arguments(self, proxy_url: str):
        """Test calling a tool with invalid arguments."""
        client = Client(proxy_url)
        async with client:
            with pytest.raises(Exception):
                # add() requires 'a' and 'b', we're passing wrong params
                await client.call_tool("add", {"x": 1, "y": 2})


# ============== CONCURRENT REQUEST TESTS ==============

class TestConcurrency:
    """Test concurrent requests through the proxy."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, proxy_url: str):
        """Test multiple concurrent tool calls."""
        client = Client(proxy_url)
        async with client:
            # Fire off multiple requests concurrently
            tasks = [
                client.call_tool("add", {"a": i, "b": i * 2})
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            # Verify all results
            for i, result in enumerate(results):
                expected = i + i * 2
                assert str(expected) in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, proxy_url: str):
        """Test concurrent tool calls and resource reads."""
        client = Client(proxy_url)
        async with client:
            tasks = [
                client.call_tool("echo", {"message": "msg1"}),
                client.read_resource("config://settings"),
                client.call_tool("add", {"a": 1, "b": 2}),
                client.read_resource("data://sample.json"),
                client.call_tool("get_timestamp", {}),
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            # All should complete successfully

    @pytest.mark.asyncio
    async def test_concurrent_with_delays(self, proxy_url: str):
        """Test concurrent async tools with delays."""
        client = Client(proxy_url)
        async with client:
            start = time.time()

            # All should run concurrently, total time ~0.3s not 0.9s
            tasks = [
                client.call_tool("async_delay", {"seconds": 0.3, "message": "a"}),
                client.call_tool("async_delay", {"seconds": 0.3, "message": "b"}),
                client.call_tool("async_delay", {"seconds": 0.3, "message": "c"}),
            ]
            results = await asyncio.gather(*tasks)

            elapsed = time.time() - start

            # Should complete in roughly 0.3-0.5s, not 0.9s
            assert elapsed < 0.8
            assert len(results) == 3


# ============== STREAMING BEHAVIOR TESTS ==============

class TestStreaming:
    """Test streaming-specific behavior through the proxy."""

    @pytest.mark.asyncio
    async def test_streaming_numbers(self, proxy_url: str):
        """Test tool that simulates streaming behavior."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool(
                "stream_numbers",
                {"count": 5, "delay_ms": 50}
            )
            text = str(result.content[0].text)

            # Should return list [0, 1, 2, 3, 4]
            data = json.loads(text)
            assert data == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self, proxy_url: str):
        """Test many sequential requests (proxy connection reuse)."""
        client = Client(proxy_url)
        async with client:
            for i in range(20):
                result = await client.call_tool("add", {"a": i, "b": 1})
                assert str(i + 1) in str(result.content[0].text)


# ============== COMPARISON TESTS (PROXY VS DIRECT) ==============

class TestProxyVsDirect:
    """Compare proxy results with direct server results."""

    @pytest.mark.asyncio
    async def test_same_tool_result(self, proxy_url: str, direct_url: str):
        """Verify proxy returns same result as direct connection."""
        # Connect through proxy
        proxy_client = Client(proxy_url)
        async with proxy_client:
            proxy_result = await proxy_client.call_tool(
                "concat",
                {"strings": ["a", "b", "c"], "separator": ","}
            )

        # Connect directly
        direct_client = Client(direct_url)
        async with direct_client:
            direct_result = await direct_client.call_tool(
                "concat",
                {"strings": ["a", "b", "c"], "separator": ","}
            )

        assert str(proxy_result.content[0].text) == str(direct_result.content[0].text)

    @pytest.mark.asyncio
    async def test_same_resource_content(self, proxy_url: str, direct_url: str):
        """Verify proxy returns same resource as direct connection."""
        proxy_client = Client(proxy_url)
        async with proxy_client:
            proxy_content = await proxy_client.read_resource("config://settings")

        direct_client = Client(direct_url)
        async with direct_client:
            direct_content = await direct_client.read_resource("config://settings")

        # Parse and compare (timestamps might differ slightly)
        proxy_data = json.loads(str(proxy_content[0].text))
        direct_data = json.loads(str(direct_content[0].text))

        assert proxy_data["server_name"] == direct_data["server_name"]
        assert proxy_data["version"] == direct_data["version"]

    @pytest.mark.asyncio
    async def test_same_prompt_structure(self, proxy_url: str, direct_url: str):
        """Verify proxy returns same prompt as direct connection."""
        proxy_client = Client(proxy_url)
        async with proxy_client:
            proxy_messages = await proxy_client.get_prompt(
                "explain_concept",
                {"concept": "testing"}
            )

        direct_client = Client(direct_url)
        async with direct_client:
            direct_messages = await direct_client.get_prompt(
                "explain_concept",
                {"concept": "testing"}
            )

        assert len(proxy_messages.messages) == len(direct_messages.messages)


# ============== EDGE CASES ==============

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_string_argument(self, proxy_url: str):
        """Test tool with empty string argument."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("echo", {"message": ""})
            assert "Echo:" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_very_long_string_argument(self, proxy_url: str):
        """Test tool with very long string argument."""
        client = Client(proxy_url)
        async with client:
            long_message = "x" * 10000
            result = await client.call_tool("echo", {"message": long_message})
            assert long_message in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_special_characters_in_argument(self, proxy_url: str):
        """Test tool with special characters."""
        client = Client(proxy_url)
        async with client:
            special = 'Test with "quotes", <tags>, and \n newlines & ampersands'
            result = await client.call_tool("echo", {"message": special})
            # Just verify it doesn't crash; exact handling may vary
            assert "Test" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_zero_values(self, proxy_url: str):
        """Test tool with zero values."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("add", {"a": 0, "b": 0})
            assert "0" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_negative_numbers(self, proxy_url: str):
        """Test tool with negative numbers."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("add", {"a": -5, "b": -3})
            assert "-8" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_float_precision(self, proxy_url: str):
        """Test float precision through proxy."""
        client = Client(proxy_url)
        async with client:
            result = await client.call_tool("multiply", {"a": 0.1, "b": 0.2})
            text = str(result.content[0].text)
            # 0.1 * 0.2 = 0.02 (with potential float precision issues)
            assert "0.02" in text or "0.0200" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
