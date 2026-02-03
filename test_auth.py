"""
Authentication tests for the streaming HTTP proxy.

These tests verify that authentication is properly enforced across all
proxy entrypoints when an auth function is configured.

The tests focus on:
- All HTTP methods require authentication
- Missing token returns 401
- Invalid token returns 401
- Valid token allows request to proceed
- Custom auth header name works
- MCP operations through authenticated proxy require auth
"""

import asyncio
import subprocess
import sys
import threading
import time
from typing import Generator

import pytest
import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.web

from thing import StreamingProxyHandler, AdvancedStreamingProxyHandler


# Test tokens
VALID_TOKEN = "valid-secret-token-12345"
INVALID_TOKEN = "wrong-token"


async def simple_auth_fn(token: str) -> bool:
    """Simple async auth function that checks against a known token."""
    return token == VALID_TOKEN


async def raising_auth_fn(token: str) -> bool:
    """Async auth function that raises an exception for certain tokens."""
    if token == "raise-error":
        raise ValueError("Token validation failed")
    return token == VALID_TOKEN


# Port configuration
AUTH_PROXY_PORT = 9300
AUTH_UPSTREAM_PORT = 9301
CUSTOM_HEADER_PROXY_PORT = 9302
MCP_SERVER_PORT = 9303
MCP_AUTH_PROXY_PORT = 9304


class MockUpstreamHandler(tornado.web.RequestHandler):
    """Simple upstream server that returns request info."""

    def get(self):
        self.write({"method": "GET", "path": self.request.path})

    def post(self):
        self.write({"method": "POST", "path": self.request.path})

    def put(self):
        self.write({"method": "PUT", "path": self.request.path})

    def delete(self):
        self.write({"method": "DELETE", "path": self.request.path})

    def head(self):
        self.set_header("X-Method", "HEAD")

    def patch(self):
        self.write({"method": "PATCH", "path": self.request.path})

    def options(self):
        self.set_header("Allow", "GET, POST, PUT, DELETE, HEAD, PATCH, OPTIONS")
        self.write({"method": "OPTIONS", "path": self.request.path})


class ServerManager:
    """Manages a Tornado server in a background thread."""

    def __init__(self, app: tornado.web.Application, port: int):
        self.app = app
        self.port = port
        self.server = None
        self.io_loop = None
        self._thread = None

    def start(self):
        def run():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.io_loop = tornado.ioloop.IOLoop.current()
            self.server = tornado.httpserver.HTTPServer(self.app)
            self.server.listen(self.port)
            self.io_loop.start()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def stop(self):
        if self.io_loop:
            self.io_loop.add_callback(self.io_loop.stop)
        if self._thread:
            self._thread.join(timeout=2)


@pytest.fixture(scope="module")
def upstream_server() -> Generator[ServerManager, None, None]:
    """Start the mock upstream server."""
    app = tornado.web.Application([
        (r".*", MockUpstreamHandler),
    ])
    server = ServerManager(app, AUTH_UPSTREAM_PORT)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def auth_proxy_server(upstream_server: ServerManager) -> Generator[ServerManager, None, None]:
    """Start the proxy server with authentication enabled."""
    app = tornado.web.Application([
        (r".*", StreamingProxyHandler, {
            "target_host": f"http://127.0.0.1:{AUTH_UPSTREAM_PORT}",
            "auth_fn": simple_auth_fn,
        }),
    ])
    server = ServerManager(app, AUTH_PROXY_PORT)
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def custom_header_proxy_server(upstream_server: ServerManager) -> Generator[ServerManager, None, None]:
    """Start the proxy server with custom auth header."""
    app = tornado.web.Application([
        (r".*", StreamingProxyHandler, {
            "target_host": f"http://127.0.0.1:{AUTH_UPSTREAM_PORT}",
            "auth_fn": simple_auth_fn,
            "auth_header": "X-API-Key",
        }),
    ])
    server = ServerManager(app, CUSTOM_HEADER_PROXY_PORT)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def http_client() -> tornado.httpclient.HTTPClient:
    """Create a synchronous HTTP client."""
    return tornado.httpclient.HTTPClient()


# ============== HTTP METHOD AUTH TESTS ==============

class TestHttpMethodsRequireAuth:
    """Test that all HTTP methods require authentication."""

    def test_get_without_token_returns_401(self, auth_proxy_server, http_client):
        """GET without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(f"http://127.0.0.1:{AUTH_PROXY_PORT}/test")
        assert exc_info.value.code == 401

    def test_post_without_token_returns_401(self, auth_proxy_server, http_client):
        """POST without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="POST",
                body=b"data"
            )
        assert exc_info.value.code == 401

    def test_put_without_token_returns_401(self, auth_proxy_server, http_client):
        """PUT without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="PUT",
                body=b"data"
            )
        assert exc_info.value.code == 401

    def test_delete_without_token_returns_401(self, auth_proxy_server, http_client):
        """DELETE without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="DELETE"
            )
        assert exc_info.value.code == 401

    def test_head_without_token_returns_401(self, auth_proxy_server, http_client):
        """HEAD without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="HEAD"
            )
        assert exc_info.value.code == 401

    def test_patch_without_token_returns_401(self, auth_proxy_server, http_client):
        """PATCH without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="PATCH",
                body=b"data"
            )
        assert exc_info.value.code == 401

    def test_options_without_token_returns_401(self, auth_proxy_server, http_client):
        """OPTIONS without token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="OPTIONS"
            )
        assert exc_info.value.code == 401


class TestInvalidTokenRejected:
    """Test that invalid tokens are rejected for all methods."""

    def _make_headers(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def test_get_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """GET with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_post_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """POST with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="POST",
                body=b"data",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_put_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """PUT with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="PUT",
                body=b"data",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_delete_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """DELETE with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="DELETE",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_head_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """HEAD with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="HEAD",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_patch_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """PATCH with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="PATCH",
                body=b"data",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401

    def test_options_with_invalid_token_returns_401(self, auth_proxy_server, http_client):
        """OPTIONS with invalid token should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
                method="OPTIONS",
                headers=self._make_headers(INVALID_TOKEN)
            )
        assert exc_info.value.code == 401


class TestValidTokenAccepted:
    """Test that valid tokens allow requests through."""

    def _make_headers(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def test_get_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """GET with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_post_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """POST with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="POST",
            body=b"data",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_put_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """PUT with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="PUT",
            body=b"data",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_delete_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """DELETE with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="DELETE",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_head_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """HEAD with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="HEAD",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_patch_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """PATCH with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="PATCH",
            body=b"data",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200

    def test_options_with_valid_token_succeeds(self, auth_proxy_server, http_client):
        """OPTIONS with valid token should succeed."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            method="OPTIONS",
            headers=self._make_headers(VALID_TOKEN)
        )
        assert response.code == 200


class TestTokenFormats:
    """Test different token format handling."""

    def test_bearer_prefix_is_stripped(self, auth_proxy_server, http_client):
        """Token with 'Bearer ' prefix should work."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"}
        )
        assert response.code == 200

    def test_token_without_bearer_prefix_works(self, auth_proxy_server, http_client):
        """Token without 'Bearer ' prefix should also work."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            headers={"Authorization": VALID_TOKEN}
        )
        assert response.code == 200

    def test_bearer_case_insensitive(self, auth_proxy_server, http_client):
        """'bearer' prefix should be case-insensitive."""
        response = http_client.fetch(
            f"http://127.0.0.1:{AUTH_PROXY_PORT}/test",
            headers={"Authorization": f"BEARER {VALID_TOKEN}"}
        )
        assert response.code == 200


class TestCustomAuthHeader:
    """Test custom authentication header configuration."""

    def test_default_header_ignored_when_custom_set(self, custom_header_proxy_server, http_client):
        """Authorization header should be ignored when custom header is configured."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                f"http://127.0.0.1:{CUSTOM_HEADER_PROXY_PORT}/test",
                headers={"Authorization": f"Bearer {VALID_TOKEN}"}
            )
        assert exc_info.value.code == 401

    def test_custom_header_works(self, custom_header_proxy_server, http_client):
        """Custom auth header should be used for authentication."""
        response = http_client.fetch(
            f"http://127.0.0.1:{CUSTOM_HEADER_PROXY_PORT}/test",
            headers={"X-API-Key": VALID_TOKEN}
        )
        assert response.code == 200

    def test_custom_header_with_bearer_prefix(self, custom_header_proxy_server, http_client):
        """Custom header with Bearer prefix should also work."""
        response = http_client.fetch(
            f"http://127.0.0.1:{CUSTOM_HEADER_PROXY_PORT}/test",
            headers={"X-API-Key": f"Bearer {VALID_TOKEN}"}
        )
        assert response.code == 200


class TestAuthFunctionBehavior:
    """Test auth function edge cases."""

    @pytest.fixture(scope="class")
    def raising_auth_proxy(self, upstream_server) -> Generator[ServerManager, None, None]:
        """Proxy with auth function that can raise exceptions."""
        app = tornado.web.Application([
            (r".*", StreamingProxyHandler, {
                "target_host": f"http://127.0.0.1:{AUTH_UPSTREAM_PORT}",
                "auth_fn": raising_auth_fn,
            }),
        ])
        server = ServerManager(app, 9310)
        server.start()
        yield server
        server.stop()

    def test_auth_fn_exception_returns_401(self, raising_auth_proxy, http_client):
        """Auth function raising exception should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                "http://127.0.0.1:9310/test",
                headers={"Authorization": "Bearer raise-error"}
            )
        assert exc_info.value.code == 401

    def test_auth_fn_returning_false_returns_401(self, raising_auth_proxy, http_client):
        """Auth function returning False should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                "http://127.0.0.1:9310/test",
                headers={"Authorization": f"Bearer {INVALID_TOKEN}"}
            )
        assert exc_info.value.code == 401


class TestWwwAuthenticateHeader:
    """Test WWW-Authenticate header is returned on 401."""

    def test_401_includes_www_authenticate_header(self, auth_proxy_server, http_client):
        """401 response should include WWW-Authenticate header."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(f"http://127.0.0.1:{AUTH_PROXY_PORT}/test")

        response = exc_info.value.response
        assert response.headers.get("WWW-Authenticate") == "Bearer"


# ============== MCP AUTH TESTS ==============

class MCPServerProcess:
    """Manages the FastMCP server as a subprocess."""

    def __init__(self, port: int):
        self.port = port
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            [sys.executable, "example_mcp_server.py", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/home/gordo/work/streamable-http-proxy",
        )
        time.sleep(2)
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"MCP server failed: {stderr.decode()}")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


# Check for FastMCP
try:
    from fastmcp import Client
    import httpx

    class BearerAuth(httpx.Auth):
        """Custom httpx Auth for Bearer tokens."""
        def __init__(self, token: str):
            self.token = token

        def auth_flow(self, request):
            request.headers["Authorization"] = f"Bearer {self.token}"
            yield request

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False


@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp not installed")
class TestMCPAuthRequired:
    """Test that MCP operations require authentication through proxy."""

    @pytest.fixture(scope="class")
    def mcp_server(self) -> Generator[MCPServerProcess, None, None]:
        server = MCPServerProcess(MCP_SERVER_PORT)
        server.start()
        yield server
        server.stop()

    @pytest.fixture(scope="class")
    def mcp_auth_proxy(self, mcp_server) -> Generator[ServerManager, None, None]:
        """Proxy with auth for MCP server."""
        app = tornado.web.Application([
            (r".*", StreamingProxyHandler, {
                "target_host": f"http://127.0.0.1:{MCP_SERVER_PORT}",
                "auth_fn": simple_auth_fn,
            }),
        ])
        server = ServerManager(app, MCP_AUTH_PROXY_PORT)
        server.start()
        yield server
        server.stop()

    @pytest.mark.asyncio
    async def test_mcp_connect_without_auth_fails(self, mcp_auth_proxy):
        """MCP client connection without auth should fail."""
        client = Client(f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp")
        with pytest.raises(Exception):
            async with client:
                pass

    @pytest.mark.asyncio
    async def test_mcp_connect_with_invalid_auth_fails(self, mcp_auth_proxy):
        """MCP client connection with invalid auth should fail."""
        client = Client(
            f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp",
            auth=BearerAuth(INVALID_TOKEN)
        )
        with pytest.raises(Exception):
            async with client:
                pass

    @pytest.mark.asyncio
    async def test_mcp_connect_with_valid_auth_succeeds(self, mcp_auth_proxy):
        """MCP client connection with valid auth should succeed."""
        client = Client(
            f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp",
            auth=BearerAuth(VALID_TOKEN)
        )
        async with client:
            assert client.is_connected()

    @pytest.mark.asyncio
    async def test_mcp_list_tools_requires_auth(self, mcp_auth_proxy):
        """MCP list_tools requires authentication."""
        # Without auth
        client_no_auth = Client(f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp")
        with pytest.raises(Exception):
            async with client_no_auth:
                await client_no_auth.list_tools()

    @pytest.mark.asyncio
    async def test_mcp_call_tool_requires_auth(self, mcp_auth_proxy):
        """MCP call_tool requires authentication."""
        # With valid auth - should work
        client = Client(
            f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp",
            auth=BearerAuth(VALID_TOKEN)
        )
        async with client:
            result = await client.call_tool("add", {"a": 1, "b": 2})
            assert "3" in str(result.content[0].text)

    @pytest.mark.asyncio
    async def test_mcp_read_resource_requires_auth(self, mcp_auth_proxy):
        """MCP read_resource requires authentication."""
        # With valid auth - should work
        client = Client(
            f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp",
            auth=BearerAuth(VALID_TOKEN)
        )
        async with client:
            content = await client.read_resource("config://settings")
            assert "TestProxyServer" in str(content[0].text)

    @pytest.mark.asyncio
    async def test_mcp_get_prompt_requires_auth(self, mcp_auth_proxy):
        """MCP get_prompt requires authentication."""
        # With valid auth - should work
        client = Client(
            f"http://127.0.0.1:{MCP_AUTH_PROXY_PORT}/mcp",
            auth=BearerAuth(VALID_TOKEN)
        )
        async with client:
            messages = await client.get_prompt("explain_concept", {"concept": "testing"})
            assert len(messages.messages) > 0


# ============== ADVANCED HANDLER AUTH TESTS ==============

class TestForwardAuth:
    """Test auth header forwarding behavior."""

    @pytest.fixture(scope="class")
    def forward_auth_upstream(self) -> Generator[ServerManager, None, None]:
        """Upstream that echoes back received headers."""
        class HeaderEchoHandler(tornado.web.RequestHandler):
            def get(self):
                # Return all received headers as JSON
                headers = dict(self.request.headers)
                self.write({"headers": headers})

        app = tornado.web.Application([(r".*", HeaderEchoHandler)])
        server = ServerManager(app, 9330)
        server.start()
        yield server
        server.stop()

    @pytest.fixture(scope="class")
    def strip_auth_proxy(self, forward_auth_upstream) -> Generator[ServerManager, None, None]:
        """Proxy that strips auth header (default behavior)."""
        app = tornado.web.Application([
            (r".*", StreamingProxyHandler, {
                "target_host": "http://127.0.0.1:9330",
                "auth_fn": simple_auth_fn,
                "forward_auth": False,
            }),
        ])
        server = ServerManager(app, 9331)
        server.start()
        yield server
        server.stop()

    @pytest.fixture(scope="class")
    def forward_auth_proxy(self, forward_auth_upstream) -> Generator[ServerManager, None, None]:
        """Proxy that forwards auth header."""
        app = tornado.web.Application([
            (r".*", StreamingProxyHandler, {
                "target_host": "http://127.0.0.1:9330",
                "auth_fn": simple_auth_fn,
                "forward_auth": True,
            }),
        ])
        server = ServerManager(app, 9332)
        server.start()
        yield server
        server.stop()

    def test_auth_header_stripped_when_disabled(self, strip_auth_proxy, http_client):
        """Auth header should be stripped when forward_auth=False."""
        import json
        response = http_client.fetch(
            "http://127.0.0.1:9331/test",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"}
        )
        data = json.loads(response.body)
        # Auth header should NOT be in the headers received by upstream
        header_names = [k.lower() for k in data["headers"].keys()]
        assert "authorization" not in header_names

    def test_auth_header_forwarded_when_enabled(self, forward_auth_proxy, http_client):
        """Auth header should be forwarded when forward_auth=True."""
        import json
        response = http_client.fetch(
            "http://127.0.0.1:9332/test",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"}
        )
        data = json.loads(response.body)
        # Auth header SHOULD be in the headers received by upstream
        assert "Authorization" in data["headers"]
        assert VALID_TOKEN in data["headers"]["Authorization"]

    def test_custom_auth_header_stripped(self, forward_auth_upstream, http_client):
        """Custom auth header should be stripped when forward_auth=False."""
        import json

        # Create a proxy with custom header that strips auth
        app = tornado.web.Application([
            (r".*", StreamingProxyHandler, {
                "target_host": "http://127.0.0.1:9330",
                "auth_fn": simple_auth_fn,
                "auth_header": "X-API-Key",
                "forward_auth": False,
            }),
        ])
        server = ServerManager(app, 9333)
        server.start()
        try:
            response = http_client.fetch(
                "http://127.0.0.1:9333/test",
                headers={"X-API-Key": VALID_TOKEN}
            )
            data = json.loads(response.body)
            header_names = [k.lower() for k in data["headers"].keys()]
            assert "x-api-key" not in header_names
        finally:
            server.stop()


class TestAdvancedHandlerAuth:
    """Test authentication for AdvancedStreamingProxyHandler."""

    @pytest.fixture(scope="class")
    def advanced_auth_proxy(self, upstream_server) -> Generator[ServerManager, None, None]:
        """Proxy using AdvancedStreamingProxyHandler with auth."""
        app = tornado.web.Application([
            (r".*", AdvancedStreamingProxyHandler, {
                "target_host": f"http://127.0.0.1:{AUTH_UPSTREAM_PORT}",
                "auth_fn": simple_auth_fn,
            }),
        ])
        server = ServerManager(app, 9320)
        server.start()
        yield server
        server.stop()

    def test_advanced_post_without_auth_returns_401(self, advanced_auth_proxy, http_client):
        """Advanced handler POST without auth should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                "http://127.0.0.1:9320/test",
                method="POST",
                body=b"test data"
            )
        assert exc_info.value.code == 401

    def test_advanced_post_with_invalid_auth_returns_401(self, advanced_auth_proxy, http_client):
        """Advanced handler POST with invalid auth should return 401."""
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                "http://127.0.0.1:9320/test",
                method="POST",
                body=b"test data",
                headers={"Authorization": f"Bearer {INVALID_TOKEN}"}
            )
        assert exc_info.value.code == 401

    def test_advanced_post_with_valid_auth_succeeds(self, advanced_auth_proxy, http_client):
        """Advanced handler POST with valid auth should succeed."""
        response = http_client.fetch(
            "http://127.0.0.1:9320/test",
            method="POST",
            body=b"test data",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"}
        )
        assert response.code == 200

    def test_advanced_large_body_with_invalid_auth_rejected(self, advanced_auth_proxy, http_client):
        """Large request body with invalid auth should be rejected early."""
        large_body = b"x" * 10000
        with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
            http_client.fetch(
                "http://127.0.0.1:9320/test",
                method="POST",
                body=large_body,
                headers={"Authorization": f"Bearer {INVALID_TOKEN}"}
            )
        assert exc_info.value.code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
