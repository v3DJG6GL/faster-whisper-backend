"""Helpers shared by the page-script test modules (slicing inline JS).

Kept out of conftest.py so the fixture file stays about fixtures; import as
``from tests._page_helpers import js_body``.
"""


def js_body(html: str, start: str) -> str:
    """Page-script source from ``start`` up to (not including) the first
    two-space-indented closing brace: a top-level script function's body."""
    body = html[html.index(start):]
    return body[:body.index("\n  }\n")]
