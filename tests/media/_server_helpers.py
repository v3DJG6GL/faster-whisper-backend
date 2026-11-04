"""Loopback HTTP server helpers shared by the media test modules (not a conftest: no fixtures)."""

import contextlib
import http.server


@contextlib.contextmanager
def _stopping(srv):
    """Stop a loopback HTTPServer (its serve_forever thread, then its socket) on exit."""
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    """A loopback request handler that logs nothing."""

    def _ok_headers(self, content_type, length):
        """The 200 status line and headers of a `length`-byte body."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def log_message(self, *a):
        pass
