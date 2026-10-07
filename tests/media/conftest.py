"""URL-domain test fixtures.

Every test here runs with the process temp dir pointed at its own tmp_path.
The media-package tests prove cleanup by listing the temp dir for `pkg-` /
`urlmedia-` entries before and after a call; on the single-host Windows
runner two CI jobs share C:\\Windows\\Temp, so a concurrent job's live workdir
showed up in the "after" set and failed a test that had cleaned up fine
(runs 971/972, 2026-09-15). `tempfile.tempdir` feeds both `gettempdir()` and
a bare `mkdtemp(prefix=...)`, so the scan and the code under test move
together and the assertion only ever sees this test's own entries.
"""

import tempfile
import urllib.error
from types import SimpleNamespace

import pytest

from faster_whisper_backend.media import download as udl


@pytest.fixture(autouse=True)
def _private_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield


@pytest.fixture
def fake_capped_get(monkeypatch):
    """A fake guarded GET (download._capped_get): `table` maps url → body
    (bytes, or an int HTTP status); honours Range, max_bytes and `accept`;
    records (url, headers) per request in `calls` and each max_bytes in
    `caps`; `on_get()` runs first, `ctype` is the answer's content type;
    `redirects` maps a requested url to the one that answers (the body is
    looked up under the target, which `want_url` reports)."""
    fake = SimpleNamespace(table={}, calls=[], caps=[], on_get=lambda: None,
                           ctype="application/octet-stream", redirects={})

    def _get(url, *, max_bytes, timeout, accept=None, headers=None,
             want_url=False):
        fake.calls.append((url, dict(headers or {})))
        fake.caps.append(max_bytes)
        fake.on_get()
        final = fake.redirects.get(url, url)
        body = fake.table[final]
        if isinstance(body, int):
            raise urllib.error.HTTPError(url, body, "x", {}, None)
        rng = (headers or {}).get("Range")
        if rng:
            a, b = map(int, rng.removeprefix("bytes=").split("-"))
            body = body[a:b + 1]
        if accept is not None and not accept(fake.ctype):
            raise udl.UrlDownloadError("the site answered with an unexpected file type")
        if len(body) > max_bytes:
            raise udl.UrlTooLargeError("the file is over the server's size limit")
        return (fake.ctype, body, final) if want_url else (fake.ctype, body)
    monkeypatch.setattr(udl, "_capped_get", _get)
    return fake
