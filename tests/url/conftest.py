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

import pytest


@pytest.fixture(autouse=True)
def _private_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield
