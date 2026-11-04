"""The no-access landing's "Pages you can access" list adds /dictate to the
per-scope pages: every signed-in identity can reach it (an ungated navlink,
tier "any" on the hub), so a key with every page scope at 'none' is offered
/dictate instead of "no other pages are available either" while the nav bar
on the same page links it.

The landing script runs under node against a minimal DOM stub; the test
reads back the action buttons it rendered into <main>."""

import json
import re
import shutil

import pytest

from faster_whisper_backend.core import web_common
from tests._page_helpers import node_stdout


_DOM_STUB = r"""
global.window = globalThis;
const main = { innerHTML: '' };
global.document = {
  body: { classList: { remove() {} } },
  getElementsByTagName(tag) { return tag === 'main' ? [main] : []; },
};
"""


def _landing(whoami: dict, page: str) -> str:
    js = (f"{_DOM_STUB}window.__whoami = {json.dumps(whoami)};\n"
          f"{web_common.NOT_ADMIN_LANDING_JS}\n"
          f"_renderNoAccessLanding({{ page: {json.dumps(page)} }});\n"
          "console.log(main.innerHTML);\n")
    return node_stdout(js)


def _buttons(html: str) -> list[str]:
    return re.findall(r'<a href="([^"]+)" class="landing-btn">', html)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_key_with_no_page_scope_is_offered_dictate():
    pages = {p: "none" for p in
             ("quick_config", "captures", "reports", "stats", "logs")}
    html = _landing({"permissions": {"pages": pages}}, "stats")
    assert _buttons(html) == ["/dictate"]
    assert "no other pages are available" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dictate_joins_the_scoped_pages_a_key_can_reach():
    pages = {"quick_config": "none", "captures": "none", "reports": "read",
             "stats": "none", "logs": "read"}
    html = _landing({"permissions": {"pages": pages}}, "stats")
    assert _buttons(html) == ["/reports", "/logs", "/dictate"]
