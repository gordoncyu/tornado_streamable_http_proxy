from typing import Awaitable, Callable, cast
from typing_extensions import override
import tornado.ioloop
import tornado.web
import tornado.httpclient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type alias for async authentication function
AuthFn = Callable[[str], Awaitable[bool]]


class StreamingProxyHandler(tornado.web.RequestHandler):
    """
    A streaming HTTP proxy that forwards requests and responses chunk-by-chunk
    without buffering the entire response in memory.

    Supports optional authentication via a configurable header and auth function.
    """

    target_host: str  # pyright: ignore[reportUninitializedInstanceVariable]
    status_line_received: bool  # pyright: ignore[reportUninitializedInstanceVariable]
    auth_fn: AuthFn | None  # pyright: ignore[reportUninitializedInstanceVariable]
    auth_header: str  # pyright: ignore[reportUninitializedInstanceVariable]
    forward_auth: bool  # pyright: ignore[reportUninitializedInstanceVariable]
    lock_target_path: bool  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def initialize(
        self,
        target_host: str,
        auth_fn: AuthFn | None = None,
        auth_header: str = "Authorization",
        forward_auth: bool = True,
        lock_target_path: bool = False,
    ):
        """
        Initialize with target host and optional authentication.

        Args:
            target_host: The upstream server URL to proxy requests to.
            auth_fn: Optional authentication function that takes a token string
                     and returns True if valid, False otherwise. Can also raise
                     an exception for invalid tokens.
            auth_header: The header name to read the bearer token from.
                         Defaults to "Authorization".
            forward_auth: Whether to forward the auth header to upstream.
                          Defaults to True (forward auth header to upstream).
            lock_target_path: When True, all requests go to target_host exactly,
                              ignoring the request path. Defaults to False.
        """
        self.target_host = target_host.rstrip('/') if not lock_target_path else target_host
        self.status_line_received = False
        self.auth_fn = auth_fn
        self.auth_header = auth_header
        self.forward_auth = forward_auth
        self.lock_target_path = lock_target_path

    async def _check_auth(self) -> bool:
        """
        Check authentication if auth_fn is configured.

        Returns True if auth passes or no auth is configured.
        Returns False and sets 401 response if auth fails.
        """
        if self.auth_fn is None:
            return True

        # Get the auth header value
        auth_value = self.request.headers.get(self.auth_header)
        if not auth_value:
            self.set_status(401)
            self.set_header("WWW-Authenticate", "Bearer")
            self.finish("Authentication required")
            return False

        # Extract bearer token
        token = auth_value
        if auth_value.lower().startswith("bearer "):
            token = auth_value[7:]  # Remove "Bearer " prefix

        # Call the async auth function
        try:
            if not await self.auth_fn(token):
                self.set_status(401)
                self.set_header("WWW-Authenticate", "Bearer")
                self.finish("Invalid token")
                return False
        except Exception as e:
            logger.warning(f"Auth function raised exception: {e}")
            self.set_status(401)
            self.set_header("WWW-Authenticate", "Bearer")
            self.finish("Authentication failed")
            return False

        return True

    @override
    async def get(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def post(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def put(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def delete(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def head(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def patch(self):
        if await self._check_auth():
            await self._proxy_request()

    @override
    async def options(self):
        if await self._check_auth():
            await self._proxy_request()

    async def _proxy_request(self):
        """Main proxy logic with streaming support"""
        target_url = self._get_target_url()
        headers: dict[str, str] = self._prepare_headers()
        http_client = tornado.httpclient.AsyncHTTPClient()

        try:
            request = tornado.httpclient.HTTPRequest(
                url=target_url,
                method=self.request.method,
                headers=headers,
                body=self.request.body if self.request.body else None,
                streaming_callback=self._on_upstream_chunk,
                header_callback=self._on_upstream_header,
                follow_redirects=False,
                allow_nonstandard_methods=True,
                request_timeout=0,
                decompress_response=False
            )

            await http_client.fetch(request)
            self.finish()

        except tornado.httpclient.HTTPClientError as e:
            logger.error(f"Proxy HTTP error: {e}")
            if e.response:
                self._handle_error_response(e.response)
            else:
                self.set_status(502)
                self.finish("Bad Gateway")
        except (ConnectionRefusedError, OSError) as e:
            logger.error(f"Proxy connection error: {e}")
            self.set_status(502)
            self.finish("Bad Gateway: upstream connection failed")

    def _get_target_url(self):
        """Construct the target URL from configured host and request path/query"""
        if self.lock_target_path:
            # Lock path but preserve query params
            if self.request.query:
                return f"{self.target_host}?{self.request.query}"
            return self.target_host
        # request.uri includes path and query string
        return f"{self.target_host}{self.request.uri}"

    def _prepare_headers(self):
        """
        Prepare headers for upstream request.
        Filters out hop-by-hop headers that shouldn't be forwarded.
        Optionally strips auth header based on forward_auth setting.
        """
        hop_by_hop = {
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers',
            'transfer-encoding', 'upgrade', 'host'
        }

        headers = {}
        for name, value in self.request.headers.items():
            if name.lower() in hop_by_hop:
                continue
            # Strip auth header if not forwarding
            if not self.forward_auth and name.lower() == self.auth_header.lower():
                continue
            headers[name] = value

        return headers

    def _on_upstream_header(self, header_line):
        """
        Called for each header line from upstream server.
        Forwards headers to the client before body chunks arrive.
        """
        header_line = header_line.strip()

        if not header_line:
            return

        if header_line.startswith('HTTP/'):
            # Status line (e.g., "HTTP/1.1 200 OK")
            # Handle multiple status lines (e.g., 100 Continue followed by 200 OK)
            parts = header_line.split(' ', 2)
            if len(parts) >= 2:
                status_code = int(parts[1])
                # For 1xx informational responses, don't mark as final status
                if status_code >= 200:
                    self.status_line_received = True
                self.set_status(status_code)
        else:
            # Regular header
            if ':' in header_line:
                name, value = header_line.split(':', 1)
                name = name.strip()
                value = value.strip()

                # Filter out hop-by-hop response headers
                if name.lower() in {'connection', 'transfer-encoding', 'keep-alive'}:
                    return

                # Headers that can appear multiple times need add_header
                if name.lower() in {'set-cookie', 'www-authenticate', 'proxy-authenticate'}:
                    self.add_header(name, value)
                else:
                    self.set_header(name, value)

    def _on_upstream_chunk(self, chunk):
        """
        Called for each chunk of data from upstream.
        Writes and flushes immediately to enable true streaming.

        Note: This callback is synchronous, so flush() cannot be awaited.
        For slow clients with fast upstreams, chunks may buffer in memory.
        """
        self.write(chunk)
        # flush() returns a Future but we can't await in a sync callback.
        # Tornado will handle the write, but backpressure is limited.
        self.flush()

    def _handle_error_response(self, response):
        """Handle error responses from upstream"""
        self.set_status(response.code)

        for header in ('Content-Type', 'Content-Length'):
            value = response.headers.get(header)
            if value:
                self.set_header(header, value)

        if response.body:
            self.write(response.body)

        self.finish()


@tornado.web.stream_request_body
class AdvancedStreamingProxyHandler(StreamingProxyHandler):
    """
    Advanced version with request body streaming support.
    Uses @stream_request_body to receive upload data incrementally.
    """

    request_body_chunks: list[bytes]  # pyright: ignore[reportUninitializedInstanceVariable]
    _auth_checked: bool  # pyright: ignore[reportUninitializedInstanceVariable]
    _auth_passed: bool  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def initialize(
        self,
        target_host: str,
        auth_fn: AuthFn | None = None,
        auth_header: str = "Authorization",
        forward_auth: bool = True,
        lock_target_path: bool = False,
    ):
        super().initialize(target_host, auth_fn, auth_header, forward_auth, lock_target_path)
        self.request_body_chunks = []
        self._auth_checked = False
        self._auth_passed = False

    @override
    async def prepare(self):
        """Called after headers are read but before body. Check auth early."""
        logger.info(f"Preparing to proxy {self.request.method} {self.request.uri}")
        # Check auth in prepare() so we reject early before receiving body
        self._auth_checked = True
        self._auth_passed = await super()._check_auth()

    @override
    async def _check_auth(self) -> bool:
        """Use cached auth result from prepare() to avoid double-checking."""
        if self._auth_checked:
            return self._auth_passed
        # Fallback if somehow called before prepare
        return await super()._check_auth()

    @override
    def data_received(self, chunk):
        """
        Called as request body chunks arrive from client.
        Accumulates chunks for forwarding to upstream.
        """
        # Only accumulate if auth passed
        if self._auth_passed:
            self.request_body_chunks.append(chunk)

    async def _proxy_request(self):
        """Override to use accumulated body chunks"""
        target_url = self._get_target_url()
        headers = self._prepare_headers()

        # Use accumulated chunks, or fall back to request.body for non-streaming
        if self.request_body_chunks:
            body = b''.join(self.request_body_chunks)
        elif self.request.body:
            body = self.request.body
        else:
            body = None

        http_client = tornado.httpclient.AsyncHTTPClient()

        try:
            request = tornado.httpclient.HTTPRequest(
                url=target_url,
                method=self.request.method,
                headers=headers,
                body=body,
                streaming_callback=self._on_upstream_chunk,
                header_callback=self._on_upstream_header,
                follow_redirects=False,
                allow_nonstandard_methods=True,
                request_timeout=0,
                decompress_response=False
            )

            await http_client.fetch(request)
            self.finish()

        except tornado.httpclient.HTTPClientError as e:
            logger.error(f"Proxy HTTP error: {e}")
            if e.response:
                self._handle_error_response(e.response)
            else:
                self.set_status(502)
                self.finish("Bad Gateway")
        except (ConnectionRefusedError, OSError) as e:
            logger.error(f"Proxy connection error: {e}")
            self.set_status(502)
            self.finish("Bad Gateway: upstream connection failed")


def make_app(
    target_host: str = "http://httpbin.org",
    auth_fn: AuthFn | None = None,
    auth_header: str = "Authorization",
    forward_auth: bool = True,
    lock_target_path: bool = False,
):
    """
    Create the proxy application.

    Args:
        target_host: The upstream server to proxy requests to.
        auth_fn: Optional authentication function that takes a token string
                 and returns True if valid, False otherwise.
        auth_header: The header name to read the bearer token from.
        forward_auth: Whether to forward the auth header to upstream.
        lock_target_path: When True, ignore request path and always use target_host exactly.
    """
    handler_args = {
        "target_host": target_host,
        "auth_fn": auth_fn,
        "auth_header": auth_header,
        "forward_auth": forward_auth,
        "lock_target_path": lock_target_path,
    }
    return tornado.web.Application([
        (r"/proxy.*", StreamingProxyHandler, handler_args),
        (r"/advanced-proxy.*", AdvancedStreamingProxyHandler, handler_args),
    ], debug=True)


if __name__ == "__main__":
    import sys

    # Allow target to be specified via command line
    target = sys.argv[1] if len(sys.argv) > 1 else "http://httpbin.org"

    app = make_app(target_host=target)
    port = 8888
    app.listen(port)
    logger.info(f"Streaming proxy server running on http://localhost:{port}")
    logger.info(f"Proxying to: {target}")
    logger.info(f"Example: curl http://localhost:{port}/proxy/get")
    tornado.ioloop.IOLoop.current().start()
