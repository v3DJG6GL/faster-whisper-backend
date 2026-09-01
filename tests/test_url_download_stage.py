"""transcribe-from-URL records a `downloading` stage timing so the receipt's
Pipeline table (and the response's `stages`) account for the fetch."""

from __future__ import annotations

import os

import pytest

import url_download
import url_media_store
from url_download import UrlMediaInfo

_URL = "https://www.youtube.com/watch?v=abc123xyz"


@pytest.fixture
def url_enabled(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "URL_DOWNLOAD_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "URL_MEDIA_DIR",
                        str(tmp_path / "url_media"), raising=False)
    url_media_store.startup_reset()

    async def _probe(url, *, timeout):
        return UrlMediaInfo(
            url=url, extractor_key="Youtube", title="A talk", duration=90.0,
            uploader="chan", filesize_approx=4096, is_live=False,
            thumbnail_url=None, ext="m4a", abr=128.0)

    async def _download(url, *, dest_dir, max_bytes=None, timeout=None,
                        progress_cb=None, cancel_check=None):
        path = os.path.join(dest_dir, "media.m4a")
        with open(path, "wb") as f:
            f.write(b"m4a-bytes" * 8)
        return path

    monkeypatch.setattr(url_download, "probe", _probe)
    monkeypatch.setattr(url_download, "download", _download)
    return app_module


def test_url_download_appears_in_response_stages(client, url_enabled):
    r = client.post("/v1/audio/transcriptions",
                    data={"model": "whisper-1", "source_url": _URL,
                          "response_format": "verbose_json"})
    assert r.status_code == 200
    stages = r.json().get("stages") or []
    dl = [s for s in stages if s.get("name") == "downloading"]
    assert len(dl) == 1
    assert dl[0]["detail"] == "Youtube"
    assert isinstance(dl[0]["secs"], float)
    assert stages[0]["name"] == "downloading"
