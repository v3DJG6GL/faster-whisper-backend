"""Loopback HTTP server helpers shared by the media test modules (not a conftest: no fixtures)."""

import contextlib


@contextlib.contextmanager
def _stopping(srv):
    """Stop a loopback HTTPServer (its serve_forever thread, then its socket) on exit."""
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
