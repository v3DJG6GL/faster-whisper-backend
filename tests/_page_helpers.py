"""Helpers shared by the page-script test modules (slicing inline JS, running
it under node).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._page_helpers import js_body``.
"""

import json
import os
import shutil
import subprocess

import pytest

from faster_whisper_backend import paths

# Column-0 closing brace of a top-level page-script function.
TOP_CLOSE = "\n}\n"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def js_upto(html: str, start: str, end: str) -> str:
    """Page-script source from ``start`` up to (not including) the first
    ``end`` after it."""
    body = html[html.index(start):]
    return body[:body.index(end)]


def js_through(html: str, start: str, end: str = "\n  }\n") -> str:
    """Page-script source from ``start`` up to and including the first
    ``end`` after it."""
    return js_upto(html, start, end) + end


def js_body(html: str, start: str) -> str:
    """Page-script source from ``start`` up to (not including) the first
    two-space-indented closing brace: a top-level script function's body."""
    return js_upto(html, start, "\n  }\n")


def js_fn(html: str, start: str) -> str:
    """Page-script source from ``start`` through its two-space-indented
    closing brace (the brace included, the newline after it not)."""
    return js_body(html, start) + "\n  }"


def settings_template_src() -> str:
    """admin/templates/settings.html as written, before render_page() fills it."""
    with open(os.path.join(paths.REPO_ROOT, "faster_whisper_backend", "admin",
                           "templates", "settings.html"), encoding="utf-8") as fh:
        return fh.read()


def script_src(asset: str) -> str:
    """The bare JS of a web_common ``<script>...</script>`` asset."""
    return asset.strip()[len("<script>"):-len("</script>")]


def node_stdout(script: str, timeout: float = 30) -> str:
    """Run ``script`` under ``node -`` over stdin; its stdout. A non-zero exit
    fails with node's stderr as the message."""
    p = subprocess.run(["node", "-"], input=script, capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout)
    assert p.returncode == 0, p.stderr
    return p.stdout


def node_json(script: str, timeout: float = 30):
    """Run ``script`` under ``node -`` over stdin; its whole stdout parsed as
    JSON."""
    return json.loads(node_stdout(script, timeout))
