"""A whoami retry whose fetch was rejected (server restarting) returned at
once when another whoami on the page had filled window.__whoami meanwhile:
the chrome stayed cleared (no admin chrome, no nav), the "retrying…" notice
stayed up and nothing retried again. A rejected fetch now falls back to the
cached whoami exactly like a 5xx does.

Reuses the node harness of test_whoami_unavailable_retry; the fetch stub is
wrapped so the page's own whoami "succeeds" right before the retry fires."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common
from tests.core.test_whoami_unavailable_retry import (
    _CHROME_DRIVE, _CHROME_STUB, _CLOCK,
)


_FILL_CACHE_ON_RETRY = r"""
const stubFetch = global.fetch;
global.fetch = (url) => {
  if (url === '/auth/whoami' && calls === 1) {
    window.__whoami = { open_mode: false, is_admin: true };
  }
  return stubFetch(url);
};
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rejected_retry_reapplies_a_cache_filled_meanwhile():
    js = web_common.OPEN_MODE_BANNER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    given = f"const STATUSES = {json.dumps([503, 'reject', 200])};\n"
    out = subprocess.run(
        ["node", "-"],
        input=given + _CHROME_STUB + _CLOCK + _FILL_CACHE_ON_RETRY + js
        + _CHROME_DRIVE,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    res = json.loads(out.stdout)
    assert (res["admin"], res["calls"], res["note_left"], res["gate"]) == (
        True, 2, False, 0)
