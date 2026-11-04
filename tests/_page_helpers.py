"""Helpers shared by the page-script test modules (slicing inline JS, running
it under node).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._page_helpers import js_body``.
"""

import json
import subprocess

# Column-0 closing brace of a top-level page-script function.
TOP_CLOSE = "\n}\n"


def js_upto(html: str, start: str, end: str) -> str:
    """Page-script source from ``start`` up to (not including) the first
    ``end`` after it."""
    body = html[html.index(start):]
    return body[:body.index(end)]


def js_body(html: str, start: str) -> str:
    """Page-script source from ``start`` up to (not including) the first
    two-space-indented closing brace: a top-level script function's body."""
    return js_upto(html, start, "\n  }\n")


def js_fn(html: str, start: str) -> str:
    """Page-script source from ``start`` through its two-space-indented
    closing brace (the brace included, the newline after it not)."""
    return js_body(html, start) + "\n  }"


def node_stdout(script: str, timeout: float = 30) -> str:
    """Run ``script`` under ``node -`` over stdin; its stdout. A non-zero exit
    raises."""
    return subprocess.run(["node", "-"], input=script, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout, check=True).stdout


def node_json(script: str, timeout: float = 30):
    """Run ``script`` under ``node -`` over stdin; its whole stdout parsed as
    JSON."""
    return json.loads(node_stdout(script, timeout))
