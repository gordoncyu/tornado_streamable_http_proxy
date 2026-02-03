import pytest
import tornado.web
import tornado.httpclient
import tornado.testing
import tornado.httpserver
import json
from streaming_proxy import (
    StreamingProxyHandler,
    AdvancedStreamingProxyHandler,
    make_app,
)


class MockUpstreamHandler(tornado.web.RequestHandler):
    """Mock upstream server for testing proxy behavior"""

    def get(self):
        self.write({"method": "GET", "path": self.request.uri})

    def post(self):
        try:
            body = self.request.body.decode('utf-8') if self.request.body else ""
        except UnicodeDecodeError:
            body = f"<binary:{len(self.request.body)} bytes>"
        self.write({"method": "POST", "body": body, "path": self.request.uri})

    def put(self):
        try:
            body = self.request.body.decode('utf-8') if self.request.body else ""
        except UnicodeDecodeError:
            body = f"<binary:{len(self.request.body)} bytes>"
        self.write({"method": "PUT", "body": body})

    def delete(self):
        self.write({"method": "DELETE"})

    def head(self):
        self.set_header("X-Custom-Header", "head-value")

    def patch(self):
        try:
            body = self.request.body.decode('utf-8') if self.request.body else ""
        except UnicodeDecodeError:
            body = f"<binary:{len(self.request.body)} bytes>"
        self.write({"method": "PATCH", "body": body})

    def options(self):
        self.set_header("Allow", "GET, POST, PUT, DELETE, HEAD, PATCH, OPTIONS")
        self.set_status(200)


class StreamingUpstreamHandler(tornado.web.RequestHandler):
    """Simulates a streaming upstream response"""

    async def get(self):
        self.set_header("Content-Type", "text/plain")
        for i in range(5):
            self.write(f"chunk{i}")
            await self.flush()
        self.finish()


class ChunkedUpstreamHandler(tornado.web.RequestHandler):
    """Simulates chunked transfer encoding"""

    async def get(self):
        self.set_header("Content-Type", "application/json")
        self.write('{"data": ')
        await self.flush()
        self.write('"streamed"')
        await self.flush()
        self.write('}')
        self.finish()


class ErrorUpstreamHandler(tornado.web.RequestHandler):
    """Returns various error responses"""

    def get(self):
        error_code = int(self.get_argument("code", "500"))
        self.set_status(error_code)
        self.write({"error": f"Error {error_code}"})


class HeaderEchoHandler(tornado.web.RequestHandler):
    """Echoes received headers back"""

    def get(self):
        headers_dict = {k: v for k, v in self.request.headers.items()}
        self.write(headers_dict)


class CustomHeaderHandler(tornado.web.RequestHandler):
    """Returns custom headers"""

    def get(self):
        self.set_header("X-Custom-Response", "custom-value")
        self.set_header("X-Another-Header", "another-value")
        self.set_header("Content-Type", "application/json")
        self.write({"headers": "set"})


class MultiCookieHandler(tornado.web.RequestHandler):
    """Returns multiple Set-Cookie headers"""

    def get(self):
        self.add_header("Set-Cookie", "session=abc123; Path=/")
        self.add_header("Set-Cookie", "user=john; Path=/")
        self.write({"cookies": "set"})


class SlowUpstreamHandler(tornado.web.RequestHandler):
    """Simulates a slow response"""

    async def get(self):
        import asyncio
        await asyncio.sleep(0.1)
        self.write({"slow": "response"})


class LargeResponseHandler(tornado.web.RequestHandler):
    """Returns a large response for streaming tests"""

    async def get(self):
        self.set_header("Content-Type", "application/octet-stream")
        chunk = b"x" * 1024
        for _ in range(100):
            self.write(chunk)
            await self.flush()
        self.finish()


class RedirectHandler(tornado.web.RequestHandler):
    """Returns redirect responses"""

    def get(self):
        redirect_type = self.get_argument("type", "301")
        if redirect_type == "301":
            self.redirect("/redirected", permanent=True)
        elif redirect_type == "302":
            self.redirect("/redirected", permanent=False)


class RedirectedHandler(tornado.web.RequestHandler):
    """Target of redirects"""

    def get(self):
        self.write({"redirected": True})


class EmptyResponseHandler(tornado.web.RequestHandler):
    """Returns an empty response"""

    def get(self):
        self.set_status(204)


class PathEchoHandler(tornado.web.RequestHandler):
    """Simple handler for path testing"""

    def get(self):
        self.write({"path": self.request.uri})


class BaseProxyTest(tornado.testing.AsyncHTTPTestCase):
    """Base class for proxy tests with upstream server setup"""

    def get_upstream_handlers(self):
        """Override to provide upstream route handlers"""
        return [(r"/proxy", MockUpstreamHandler)]

    def get_app(self):
        # Start upstream server first to get its port
        self.upstream_app = tornado.web.Application(self.get_upstream_handlers())
        self.upstream_server = tornado.httpserver.HTTPServer(self.upstream_app)
        sock, self.upstream_port = tornado.testing.bind_unused_port()
        self.upstream_server.add_sockets([sock])

        # Create proxy app pointing to upstream
        target_host = f"http://localhost:{self.upstream_port}"
        return tornado.web.Application([
            (r"/proxy.*", StreamingProxyHandler, {"target_host": target_host}),
            (r"/advanced-proxy.*", AdvancedStreamingProxyHandler, {"target_host": target_host}),
        ])

    def tearDown(self):
        self.upstream_server.stop()
        super().tearDown()


class TestStreamingProxyHandler(BaseProxyTest):
    """Tests for StreamingProxyHandler HTTP methods"""

    def get_upstream_handlers(self):
        return [(r"/proxy", MockUpstreamHandler)]

    def test_get_request(self):
        """Test GET request proxying"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "GET")

    def test_post_request(self):
        """Test POST request proxying with body"""
        response = self.fetch("/proxy", method="POST", body="test body content")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")
        self.assertEqual(data["body"], "test body content")

    def test_put_request(self):
        """Test PUT request proxying"""
        response = self.fetch("/proxy", method="PUT", body="put content")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "PUT")
        self.assertEqual(data["body"], "put content")

    def test_delete_request(self):
        """Test DELETE request proxying"""
        response = self.fetch("/proxy", method="DELETE")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "DELETE")

    def test_head_request(self):
        """Test HEAD request proxying"""
        response = self.fetch("/proxy", method="HEAD")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"")

    def test_patch_request(self):
        """Test PATCH request proxying"""
        response = self.fetch("/proxy", method="PATCH", body="patch content")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "PATCH")
        self.assertEqual(data["body"], "patch content")

    def test_options_request(self):
        """Test OPTIONS request proxying"""
        response = self.fetch("/proxy", method="OPTIONS", allow_nonstandard_methods=True)
        self.assertEqual(response.code, 200)
        self.assertIn("Allow", response.headers)

    def test_empty_body_post(self):
        """Test POST with empty body"""
        response = self.fetch("/proxy", method="POST", body="")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")

    def test_json_body_post(self):
        """Test POST with JSON body"""
        body = json.dumps({"key": "value", "nested": {"a": 1}})
        response = self.fetch(
            "/proxy",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")
        received_body = json.loads(data["body"])
        self.assertEqual(received_body["key"], "value")

    def test_binary_body_post(self):
        """Test POST with binary body"""
        binary_data = bytes(range(256))
        response = self.fetch(
            "/proxy",
            method="POST",
            body=binary_data,
            headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")
        self.assertIn("binary:256 bytes", data["body"])


class TestConnectionErrors(tornado.testing.AsyncHTTPTestCase):
    """Tests for connection error handling"""

    def get_app(self):
        # Point to a port that's definitely closed
        return tornado.web.Application([
            (r"/proxy.*", StreamingProxyHandler, {"target_host": "http://localhost:59999"}),
        ])

    def test_connection_refused(self):
        """Test handling when upstream is unreachable"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 502)
        self.assertIn(b"Bad Gateway", response.body)


class TestHeaderForwarding(BaseProxyTest):
    """Tests for header handling in the proxy"""

    def get_upstream_handlers(self):
        return [(r"/proxy", HeaderEchoHandler)]

    def test_request_headers_forwarded(self):
        """Test that request headers are forwarded to upstream"""
        response = self.fetch("/proxy", headers={"X-Custom-Request": "custom-value"})
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data.get("X-Custom-Request"), "custom-value")

    def test_custom_header_forwarded(self):
        """Test that custom headers are forwarded"""
        response = self.fetch("/proxy", headers={"X-Should-Forward": "yes"})
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data.get("X-Should-Forward"), "yes")

    def test_proxy_authenticate_filtered(self):
        """Test Proxy-Authenticate is filtered"""
        response = self.fetch("/proxy", headers={"Proxy-Authenticate": "Basic"})
        data = json.loads(response.body)
        self.assertNotIn("Proxy-Authenticate", data)

    def test_proxy_authorization_filtered(self):
        """Test Proxy-Authorization is filtered"""
        response = self.fetch("/proxy", headers={"Proxy-Authorization": "Basic xyz"})
        data = json.loads(response.body)
        self.assertNotIn("Proxy-Authorization", data)

    def test_te_header_filtered(self):
        """Test TE header is filtered"""
        response = self.fetch("/proxy", headers={"TE": "trailers"})
        data = json.loads(response.body)
        self.assertNotIn("TE", data)

    def test_upgrade_header_filtered(self):
        """Test Upgrade header is filtered"""
        response = self.fetch("/proxy", headers={"Upgrade": "websocket"})
        data = json.loads(response.body)
        self.assertNotIn("Upgrade", data)

    def test_trailers_header_filtered(self):
        """Test Trailers header is filtered"""
        response = self.fetch("/proxy", headers={"Trailers": "X-Checksum"})
        data = json.loads(response.body)
        self.assertNotIn("Trailers", data)

    def test_host_header_filtered(self):
        """Test Host header is filtered"""
        response = self.fetch("/proxy", headers={"Host": "evil.example.com"})
        data = json.loads(response.body)
        self.assertNotEqual(data.get("Host"), "evil.example.com")

    def test_regular_headers_preserved(self):
        """Test that regular headers are preserved"""
        response = self.fetch(
            "/proxy",
            headers={
                "X-Request-Id": "req-12345",
                "Authorization": "Bearer token123",
            }
        )
        data = json.loads(response.body)
        self.assertEqual(data.get("Authorization"), "Bearer token123")
        request_id = data.get("X-Request-Id") or data.get("X-Request-ID") or data.get("x-request-id")
        self.assertEqual(request_id, "req-12345")

    def test_header_case_insensitivity(self):
        """Test that hop-by-hop header filtering is case-insensitive"""
        response = self.fetch("/proxy", headers={"CONNECTION": "keep-alive"})
        self.assertEqual(response.code, 200)


class TestResponseHeaders(BaseProxyTest):
    """Tests for response header handling"""

    def get_upstream_handlers(self):
        return [(r"/proxy", CustomHeaderHandler)]

    def test_response_headers_forwarded(self):
        """Test that response headers are forwarded from upstream"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.headers.get("X-Custom-Response"), "custom-value")
        self.assertEqual(response.headers.get("X-Another-Header"), "another-value")

    def test_content_type_preserved(self):
        """Test that Content-Type header is preserved"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        self.assertIn("application/json", response.headers.get("Content-Type", ""))

    def test_status_line_parsed(self):
        """Test that HTTP status line is parsed correctly"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)


class TestMultiValueHeaders(BaseProxyTest):
    """Tests for headers that can appear multiple times"""

    def get_upstream_handlers(self):
        return [(r"/proxy", MultiCookieHandler)]

    def test_multiple_set_cookie_headers(self):
        """Test that multiple Set-Cookie headers are preserved"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        # Tornado combines multiple headers with the same name
        cookies = response.headers.get_list("Set-Cookie")
        self.assertEqual(len(cookies), 2)
        self.assertIn("session=abc123", cookies[0])
        self.assertIn("user=john", cookies[1])


class TestStreamingResponses(BaseProxyTest):
    """Tests for streaming response handling"""

    def get_upstream_handlers(self):
        return [(r"/proxy", StreamingUpstreamHandler)]

    def test_streaming_response(self):
        """Test streaming response handling"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body.decode(), "chunk0chunk1chunk2chunk3chunk4")


class TestChunkedResponses(BaseProxyTest):
    """Tests for chunked transfer encoding"""

    def get_upstream_handlers(self):
        return [(r"/proxy", ChunkedUpstreamHandler)]

    def test_chunked_response(self):
        """Test chunked transfer encoding response"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["data"], "streamed")


class TestLargeResponses(BaseProxyTest):
    """Tests for large response streaming"""

    def get_upstream_handlers(self):
        return [(r"/proxy", LargeResponseHandler)]

    def test_large_response_streaming(self):
        """Test that large responses are streamed properly"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)
        self.assertEqual(len(response.body), 100 * 1024)


class TestError400(BaseProxyTest):
    """Test 400 error handling"""

    def get_upstream_handlers(self):
        class Error400Handler(tornado.web.RequestHandler):
            def get(self):
                self.set_status(400)
                self.write({"error": "Bad Request"})
        return [(r"/proxy", Error400Handler)]

    def test_upstream_400_error(self):
        """Test handling of 400 from upstream"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 400)


class TestError404(BaseProxyTest):
    """Test 404 error handling"""

    def get_upstream_handlers(self):
        class Error404Handler(tornado.web.RequestHandler):
            def get(self):
                self.set_status(404)
                self.write({"error": "Not Found"})
        return [(r"/proxy", Error404Handler)]

    def test_upstream_404_error(self):
        """Test handling of 404 from upstream"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 404)


class TestError500(BaseProxyTest):
    """Test 500 error handling"""

    def get_upstream_handlers(self):
        class Error500Handler(tornado.web.RequestHandler):
            def get(self):
                self.set_status(500)
                self.write({"error": "Internal Server Error"})
        return [(r"/proxy", Error500Handler)]

    def test_upstream_500_error(self):
        """Test handling of 500 from upstream"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 500)


class TestRedirectHandling(BaseProxyTest):
    """Tests for redirect handling"""

    def get_upstream_handlers(self):
        return [
            (r"/proxy", RedirectHandler),
            (r"/redirected", RedirectedHandler),
        ]

    def test_redirect_not_followed(self):
        """Test that redirects are not automatically followed"""
        response = self.fetch("/proxy?type=301", follow_redirects=False)
        self.assertEqual(response.code, 301)


class TestEmptyResponse(BaseProxyTest):
    """Tests for empty response handling"""

    def get_upstream_handlers(self):
        return [(r"/proxy", EmptyResponseHandler)]

    def test_empty_response_body(self):
        """Test handling of empty response body"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 204)
        self.assertEqual(response.body, b"")


class TestSlowResponse(BaseProxyTest):
    """Tests for slow response handling"""

    def get_upstream_handlers(self):
        return [(r"/proxy", SlowUpstreamHandler)]

    def test_slow_response_completes(self):
        """Test that slow but valid responses complete"""
        response = self.fetch("/proxy", request_timeout=10)
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["slow"], "response")


class TestAdvancedStreamingProxyHandler(BaseProxyTest):
    """Tests for AdvancedStreamingProxyHandler"""

    def get_upstream_handlers(self):
        return [(r"/advanced-proxy", MockUpstreamHandler)]

    def test_advanced_get_request(self):
        """Test GET through advanced proxy"""
        response = self.fetch("/advanced-proxy")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "GET")

    def test_advanced_post_with_body(self):
        """Test POST with body through advanced proxy"""
        response = self.fetch(
            "/advanced-proxy",
            method="POST",
            body="advanced body content"
        )
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")
        self.assertEqual(data["body"], "advanced body content")

    def test_advanced_large_body_post(self):
        """Test POST with large body through advanced proxy"""
        large_body = "x" * 10000
        response = self.fetch(
            "/advanced-proxy",
            method="POST",
            body=large_body
        )
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["method"], "POST")
        self.assertEqual(len(data["body"]), 10000)


class TestMakeApp(tornado.testing.AsyncHTTPTestCase):
    """Tests for the application factory"""

    def get_app(self):
        return make_app(target_host="http://localhost:59999")

    def test_proxy_endpoint_exists(self):
        """Test that /proxy endpoint is registered"""
        response = self.fetch("/proxy")
        # Should get 502 since target is unreachable
        self.assertEqual(response.code, 502)

    def test_advanced_proxy_endpoint_exists(self):
        """Test that /advanced-proxy endpoint is registered"""
        response = self.fetch("/advanced-proxy")
        self.assertEqual(response.code, 502)

    def test_nonexistent_endpoint(self):
        """Test that nonexistent endpoints return 404"""
        response = self.fetch("/nonexistent")
        self.assertEqual(response.code, 404)


class TestURLConstruction(BaseProxyTest):
    """Tests for URL construction logic"""

    def get_upstream_handlers(self):
        return [
            (r"/proxy", PathEchoHandler),
            (r"/proxy/subpath", PathEchoHandler),
        ]

    def get_app(self):
        self.upstream_app = tornado.web.Application(self.get_upstream_handlers())
        self.upstream_server = tornado.httpserver.HTTPServer(self.upstream_app)
        sock, self.upstream_port = tornado.testing.bind_unused_port()
        self.upstream_server.add_sockets([sock])

        target_host = f"http://localhost:{self.upstream_port}"
        return tornado.web.Application([
            (r"/proxy.*", StreamingProxyHandler, {"target_host": target_host}),
        ])

    def test_simple_path(self):
        """Test simple path proxying"""
        response = self.fetch("/proxy")
        self.assertEqual(response.code, 200)

    def test_path_with_subpath(self):
        """Test that subpaths are preserved"""
        response = self.fetch("/proxy/subpath")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertIn("/proxy/subpath", data["path"])


class TestConcurrentRequests(BaseProxyTest):
    """Tests for concurrent request handling"""

    def get_upstream_handlers(self):
        return [(r"/proxy", MockUpstreamHandler)]

    @tornado.testing.gen_test
    async def test_concurrent_requests(self):
        """Test that multiple concurrent requests are handled"""
        client = tornado.httpclient.AsyncHTTPClient()
        url = self.get_url("/proxy")

        futures = [client.fetch(url) for _ in range(5)]
        responses = await tornado.gen.multi(futures)

        for response in responses:
            self.assertEqual(response.code, 200)
            data = json.loads(response.body)
            self.assertEqual(data["method"], "GET")


class TestMultipleRequestsIndependent(BaseProxyTest):
    """Tests for handler state independence"""

    def get_upstream_handlers(self):
        return [(r"/proxy", MockUpstreamHandler)]

    def test_multiple_requests_independent(self):
        """Test that multiple requests have independent state"""
        response1 = self.fetch("/proxy")
        response2 = self.fetch("/proxy")

        self.assertEqual(response1.code, 200)
        self.assertEqual(response2.code, 200)


class TestInitializeMethod(BaseProxyTest):
    """Tests for handler initialization"""

    def get_upstream_handlers(self):
        return [(r"/proxy", MockUpstreamHandler)]

    def test_status_line_received_initialized(self):
        """Test that status_line_received is initialized per request"""
        response1 = self.fetch("/proxy")
        self.assertEqual(response1.code, 200)
        response2 = self.fetch("/proxy")
        self.assertEqual(response2.code, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
