"""Route-level tests for transcribe-from-URL: the `source_url` branch of
POST /v1/audio/transcriptions, POST /v1/audio/url-preview, and
GET /v1/audio/url-media/{id}. url_download's network/subprocess halves are
stubbed at the module boundary; everything from the handler down runs real.
"""

from __future__ import annotations

import os

import pytest

import url_download
import url_media_store
from url_download import UrlDownloadError, UrlMediaInfo

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
_PID = "beef" * 8
_URL = "https://www.youtube.com/watch?v=abc123xyz"


def _post_url(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("source_url", _URL)
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", data=data)


def _info(**kw):
    base = dict(url=_URL, extractor_key="Youtube", title="A talk",
                duration=90.0, uploader="chan", filesize_approx=4096,
                is_live=False, thumbnail_url=None, ext="m4a", abr=128.0)
    base.update(kw)
    return UrlMediaInfo(**base)


@pytest.fixture
def url_enabled(app_module, tmp_path, monkeypatch):
    """Feature on + retention rooted in a temp dir + happy-path stubs."""
    monkeypatch.setattr(app_module.cfg, "URL_DOWNLOAD_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "URL_MEDIA_DIR",
                        str(tmp_path / "url_media"), raising=False)
    url_media_store.startup_reset()

    async def _probe(url, *, timeout):
        return _info(url=url)

    async def _download(url, *, dest_dir, max_bytes=None, timeout=None,
                        progress_cb=None, cancel_check=None):
        if progress_cb is not None:
            progress_cb(0.5, 4096)
        path = os.path.join(dest_dir, "media.m4a")
        with open(path, "wb") as f:
            f.write(b"m4a-bytes" * 8)
        return path

    monkeypatch.setattr(url_download, "probe", _probe)
    monkeypatch.setattr(url_download, "download", _download)
    return app_module


# --- feature flag off (the default) -----------------------------------------

def test_source_url_403_when_disabled(client):
    r = _post_url(client)
    assert r.status_code == 403
    assert "not enabled" in r.json()["detail"]


def test_preview_403_when_disabled(client):
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 403


def test_media_403_when_disabled(client):
    r = client.get(f"/v1/audio/url-media/{'a' * 32}")
    assert r.status_code == 403


def test_me_reports_disabled(client):
    body = client.get("/v1/me").json()
    assert body["url_download_enabled"] is False
    assert "yt_dlp_version" not in body


# --- source arg validation ---------------------------------------------------

def test_both_file_and_url_is_422(client, url_enabled):
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "source_url": _URL})
    assert r.status_code == 422
    assert "not both" in r.json()["detail"]


def test_neither_file_nor_url_is_422(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 422


def test_file_upload_regression_unchanged(client, url_enabled):
    # The classic upload path must be byte-identical with the feature on.
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 200
    assert r.json() == {"text": "hallo welt"}


# --- happy path --------------------------------------------------------------

def test_url_verbose_json_carries_media_id(client, url_enabled):
    r = _post_url(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hallo welt"
    mid = body["source_media_id"]
    assert isinstance(mid, str) and len(mid) == 32
    assert body["source_media_expires_at"] > 0

    # the retained audio is fetchable, with Range support, then gone at TTL 0
    m = client.get(f"/v1/audio/url-media/{mid}")
    assert m.status_code == 200
    assert m.headers["cache-control"] == "no-store"
    assert m.content == b"m4a-bytes" * 8
    ranged = client.get(f"/v1/audio/url-media/{mid}",
                        headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206
    assert ranged.content == b"m4a-"


def test_url_plain_json_carries_media_id(client, url_enabled):
    r = _post_url(client, response_format="json")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hallo welt"
    assert len(body["source_media_id"]) == 32


def test_media_expired_is_404(client, url_enabled, monkeypatch):
    mid = _post_url(client).json()["source_media_id"]
    monkeypatch.setattr(url_enabled.cfg, "URL_MEDIA_TTL_SEC", 0, raising=False)
    assert client.get(f"/v1/audio/url-media/{mid}").status_code == 404


def test_media_unknown_and_malformed_ids(client, url_enabled):
    assert client.get(f"/v1/audio/url-media/{'f' * 32}").status_code == 404
    assert client.get("/v1/audio/url-media/NOPE").status_code == 422


def test_progress_sees_downloading_stage(client, url_enabled):
    seen = {}
    orig_download = url_download.download

    async def _spying_download(url, **kw):
        # capture the live progress entry the moment the stub reports bytes
        result = await orig_download(url, **kw)
        seen.update(url_enabled._BATCH_PROGRESS.get(_PID) or {})
        return result

    url_download.download = _spying_download
    try:
        r = _post_url(client, progress_id=_PID)
    finally:
        url_download.download = orig_download
    assert r.status_code == 200
    assert seen.get("stage") == "downloading"
    assert seen.get("progress") == 0.5
    assert seen.get("total_bytes") == 4096
    # finished request popped its entry
    assert client.get(
        f"/v1/audio/transcriptions/progress/{_PID}").json()["stage"] == "unknown"


def test_me_reports_enabled_with_version(client, url_enabled):
    body = client.get("/v1/me").json()
    assert body["url_download_enabled"] is True
    assert "yt_dlp_version" in body  # value may be None when not installed


# --- error mapping -----------------------------------------------------------

def test_policy_reject_is_client_safe_400(client, url_enabled, monkeypatch):
    async def _reject(url, *, timeout):
        raise UrlDownloadError("this site isn't on the server's allowed list")
    monkeypatch.setattr(url_download, "probe", _reject)
    r = _post_url(client)
    assert r.status_code == 400
    assert r.json()["detail"] == "this site isn't on the server's allowed list"


def test_download_error_is_400_not_500(client, url_enabled, monkeypatch):
    async def _boom(url, **kw):
        raise UrlDownloadError("this video is private")
    monkeypatch.setattr(url_download, "download", _boom)
    r = _post_url(client)
    assert r.status_code == 400
    assert r.json()["detail"] == "this video is private"


def test_precancelled_url_request_is_499(client, url_enabled):
    url_enabled._BATCH_CANCELLED.add(_PID)
    try:
        r = _post_url(client, progress_id=_PID)
        assert r.status_code == 499, r.text
        assert _PID not in url_enabled._BATCH_CANCELLED
    finally:
        url_enabled._BATCH_CANCELLED.discard(_PID)


def test_malformed_url_is_400(client, url_enabled):
    r = _post_url(client, source_url="notaurl")
    assert r.status_code == 400


def test_translations_twin_accepts_source_url(client, url_enabled):
    r = client.post("/v1/audio/translations",
                    data={"model": "whisper-1", "source_url": _URL})
    assert r.status_code == 200
    assert r.json()["text"] == "hallo welt"


# --- preview endpoint --------------------------------------------------------

def test_preview_happy_path(client, url_enabled, monkeypatch):
    async def _thumb(url, **kw):
        return None
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri", _thumb)
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "title": "A talk", "duration": 90.0, "uploader": "chan",
        "extractor": "Youtube", "estimated_bytes": 4096, "thumbnail": None,
        "ext": "m4a", "abr": 128.0,
    }


def test_preview_policy_reject_400(client, url_enabled, monkeypatch):
    async def _reject(url, *, timeout):
        raise UrlDownloadError("live streams aren't supported")
    monkeypatch.setattr(url_download, "probe", _reject)
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 400
    assert "live" in r.json()["detail"]


def test_preview_validation(client, url_enabled):
    assert client.post("/v1/audio/url-preview", json={}).status_code == 422
    assert client.post("/v1/audio/url-preview",
                       content=b"not json").status_code == 422


def test_preview_rate_limited(client, url_enabled, monkeypatch):
    async def _thumb(url, **kw):
        return None
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri", _thumb)
    for _ in range(url_enabled._URL_PREVIEW_RATE_MAX):
        assert client.post("/v1/audio/url-preview",
                           json={"url": _URL}).status_code == 200
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 429
