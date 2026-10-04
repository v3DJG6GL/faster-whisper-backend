"""Route-level tests for transcribe-from-URL: the `source_url` branch of
POST /v1/audio/transcriptions, POST /v1/audio/url-preview, and
GET /v1/audio/url-media/{id}. url_download's network/subprocess halves are
stubbed at the module boundary; everything from the handler down runs real.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from faster_whisper_backend.url import download as url_download
from faster_whisper_backend.url import media_store as url_media_store
from faster_whisper_backend.url.download import UrlDownloadError, UrlMediaInfo

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


def test_lifespan_reset_is_temp_rooted(client, app_module, tmp_path):
    # The lifespan's startup_reset() rmtree's URL_MEDIA_DIR unconditionally;
    # conftest must have pointed it under tmp_path before the app started.
    assert app_module.cfg.URL_MEDIA_DIR.startswith(str(tmp_path))
    assert os.path.isdir(app_module.cfg.URL_MEDIA_DIR)


# --- feature flag off (the default) -----------------------------------------

def test_source_url_403_when_disabled(client):
    r = _post_url(client)
    assert r.status_code == 403
    assert "not enabled" in r.json()["detail"]


def test_preview_403_when_disabled(client):
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 403


def test_media_403_when_disabled(client, app_module, monkeypatch):
    # The fetch serves both producers' ids (link runs AND packaging uploads,
    # the latter on by default): 403 only when neither feature is on.
    monkeypatch.setattr(app_module.cfg, "MEDIA_PACKAGE_ENABLED", False,
                        raising=False)
    r = client.get(f"/v1/audio/url-media/{'a' * 32}")
    assert r.status_code == 403


def test_media_is_served_while_packaging_is_enabled(client, app_module):
    # URL download off (the default), packaging on (the default): an unknown
    # id is a plain 404, not "URL download is not enabled".
    assert app_module.cfg.URL_DOWNLOAD_ENABLED is False
    assert getattr(app_module.cfg, "MEDIA_PACKAGE_ENABLED", True) is True
    r = client.get(f"/v1/audio/url-media/{'a' * 32}")
    assert r.status_code == 404


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
    monkeypatch.setattr(url_enabled.cfg, "URL_MEDIA_TTL_S", 0, raising=False)
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
        "video_ladder": [], "media_max_bytes": url_enabled.cfg.MEDIA_MAX_BYTES,
        "language": None, "subtitle_tracks": [],
    }


def test_preview_lists_tracks_never_their_urls(client, url_enabled, monkeypatch):
    async def _thumb(url, **kw):
        return None

    async def _probe(url, *, timeout):
        return _info(url=url, language="de", subtitle_tracks=[
            {"id": "m-de-CH", "lang": "de-CH", "name": "German", "kind": "manual",
             "ext": "vtt", "hoh": False}],
            subtitle_sources={"m-de-CH": {"url": "https://x.test/s?pot=SECRET",
                                          "ext": "vtt"}})
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri", _thumb)
    monkeypatch.setattr(url_download, "probe", _probe)
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    body = r.json()
    assert body["language"] == "de"
    assert [t["id"] for t in body["subtitle_tracks"]] == ["m-de-CH"]
    assert "SECRET" not in r.text


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
    limit = int(url_enabled.cfg.URL_PREVIEW_RATE_PER_MIN)
    for _ in range(limit):
        assert client.post("/v1/audio/url-preview",
                           json={"url": _URL}).status_code == 200
    r = client.post("/v1/audio/url-preview", json={"url": _URL})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    body = r.json()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["error"]["param"] == "URL_PREVIEW_RATE_PER_MIN"
    # The in-repo toast handlers read j.detail — it mirrors error.message.
    assert body["detail"] == body["error"]["message"]


def test_preview_rate_limit_is_per_user(client, url_enabled, make_user_key,
                                        monkeypatch):
    """Two identities must not share a bucket. The loopback `client` fixture
    runs in OPEN mode as one synthetic admin, so a user-keyed limit would
    degrade to a single shared bucket there — real keys are needed (creating
    the first admin key also flips the app to locked-down)."""
    from tests.conftest import bearer

    async def _thumb(url, **kw):
        return None
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri", _thumb)
    _uid_a, key_a = make_user_key("alice", is_admin=True)
    _uid_b, key_b = make_user_key("bob", is_admin=True)

    limit = int(url_enabled.cfg.URL_PREVIEW_RATE_PER_MIN)
    for _ in range(limit):
        assert client.post("/v1/audio/url-preview", json={"url": _URL},
                           headers=bearer(key_a)).status_code == 200
    assert client.post("/v1/audio/url-preview", json={"url": _URL},
                       headers=bearer(key_a)).status_code == 429
    # bob's budget is untouched by alice spending hers.
    assert client.post("/v1/audio/url-preview", json={"url": _URL},
                       headers=bearer(key_b)).status_code == 200


# --- hard-restart TMPDIR sweep ----------------------------------------------

def test_reclaim_hard_restart_orphans(tmp_path, monkeypatch):
    """The startup sweep reclaims what an admin restart (os.execv/os._exit,
    no ASGI shutdown) orphaned — urldl- job dirs, sepsrc-/vocals- WAVs —
    while leaving fresh entries (a just-overlapping process) and unrelated
    names alone. (No app_module fixture: that fixture stubs the sweep out
    so TestClient startups never touch the real tempdir.)"""
    import time as _time

    from faster_whisper_backend import main as app_module

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(app_module.tempfile, "gettempdir",
                        lambda: str(fake_tmp))

    old = _time.time() - 3600
    old_dir = fake_tmp / "urldl-dead"
    old_dir.mkdir()
    (old_dir / "media.m4a.part").write_bytes(b"x")
    os.utime(old_dir, (old, old))
    for name in ("sepsrc-dead.wav", "vocals-dead.wav"):
        p = fake_tmp / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    fresh_dir = fake_tmp / "urldl-live"
    fresh_dir.mkdir()
    (fake_tmp / "sepsrc-live.wav").write_bytes(b"x")
    (fake_tmp / "unrelated.txt").write_bytes(b"x")

    app_module._reclaim_hard_restart_orphans()

    assert not old_dir.exists()
    assert not (fake_tmp / "sepsrc-dead.wav").exists()
    assert not (fake_tmp / "vocals-dead.wav").exists()
    assert fresh_dir.exists()
    assert (fake_tmp / "sepsrc-live.wav").exists()
    assert (fake_tmp / "unrelated.txt").exists()


def test_reclaim_hard_restart_orphans_covers_pipeline_copies_and_uploads(
        tmp_path, monkeypatch):
    """The sweep also reclaims the `urlmedia-` pipeline copies
    (url_media_store.make_pipeline_copy) and the `whisperup-` batch upload
    spools — neither carried a matchable name before, so no sweep could
    ever see them — under the same 60 s age guard."""
    import time as _time

    from faster_whisper_backend import main as app_module

    fake_tmp = tmp_path / "faketmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(app_module.tempfile, "gettempdir",
                        lambda: str(fake_tmp))
    old = _time.time() - 3600
    for name in ("urlmedia-dead.m4a", "whisperup-dead.wav"):
        p = fake_tmp / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    (fake_tmp / "urlmedia-live.m4a").write_bytes(b"x")
    (fake_tmp / "whisperup-live.wav").write_bytes(b"x")
    (fake_tmp / "tmpabc123.wav").write_bytes(b"x")   # someone else's tempfile
    os.utime(fake_tmp / "tmpabc123.wav", (old, old))

    app_module._reclaim_hard_restart_orphans()

    assert not (fake_tmp / "urlmedia-dead.m4a").exists()
    assert not (fake_tmp / "whisperup-dead.wav").exists()
    assert (fake_tmp / "urlmedia-live.m4a").exists()
    assert (fake_tmp / "whisperup-live.wav").exists()
    assert (fake_tmp / "tmpabc123.wav").exists()


# --- keep_video: the run-time video fetch --------------------------------------

_LADDER = [
    {"kind": "video", "height": 1080, "fps": 30, "hdr": False, "vcodec": "vp09",
     "acodec": "opus", "container": "mkv", "approx_bytes": 5000, "over_cap": False,
     "label": "1080p"},
    {"kind": "video", "height": 720, "fps": 30, "hdr": False, "vcodec": "avc1",
     "acodec": "mp4a", "container": "mp4", "approx_bytes": 3000, "over_cap": False,
     "label": "720p"},
    {"kind": "audio", "height": None, "ext": "m4a", "abr": 128.0,
     "approx_bytes": 4096, "over_cap": False, "label": "audio only · m4a · 128 kbps"},
]


@pytest.fixture
def video_enabled(url_enabled, monkeypatch):
    """url_enabled + video on, a two-rung ladder, and a download_video stub
    that writes `media.<container>` after one progress tick."""
    import asyncio

    app_module = url_enabled
    monkeypatch.setattr(app_module.cfg, "URL_VIDEO_ENABLED", True, raising=False)
    import threading

    calls: list = []
    # A threading.Event (not an asyncio one): the test sets it from the
    # TestClient's caller thread while the app loop runs the task.
    gate: dict = {"release": None}
    gate["make"] = threading.Event

    async def _probe(url, *, timeout):
        return _info(url=url, video_ladder=[dict(r) for r in _LADDER])

    async def _download_video(url, *, dest_dir, max_bytes=None, max_height=None,
                              container="mkv", expected_total=None, timeout=None,
                              progress_cb=None, cancel_check=None, **kw):
        calls.append({"max_height": max_height, "container": container,
                      "expected_total": expected_total, **kw})
        if progress_cb is not None:
            progress_cb(0.4, expected_total or 5000, 2000)
        if gate["release"] is not None:
            while not gate["release"].is_set():
                if cancel_check is not None and cancel_check():
                    raise url_download.UrlCancelled()
                await asyncio.sleep(0.01)
        path = os.path.join(dest_dir, f"media.{container}")
        with open(path, "wb") as f:
            f.write(b"video-bytes" * 8)
        return path

    monkeypatch.setattr(url_download, "probe", _probe)
    monkeypatch.setattr(url_download, "download_video", _download_video)
    app_module._video_calls = calls
    app_module._video_gate = gate
    return app_module


def test_preview_carries_the_ladder(client, video_enabled, monkeypatch):
    async def _thumb(url, **kw):
        return None
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri", _thumb)
    body = client.post("/v1/audio/url-preview", json={"url": _URL}).json()
    assert [r["height"] for r in body["video_ladder"]] == [1080, 720, None]
    assert body["media_max_bytes"] == video_enabled.cfg.MEDIA_MAX_BYTES


def test_keep_video_response_carries_video_id_when_the_task_finishes(
        client, video_enabled):
    r = _post_url(client, keep_video="true", video_max_height="720",
                  progress_id=_PID)
    assert r.status_code == 200, r.text
    body = r.json()
    vid = body.get("source_video_media_id")
    # The stub finishes instantly, but the task still runs on the loop: the
    # response either carries the id or says pending — never an error.
    if vid is None:
        assert body.get("source_video_pending") is True
        # ...and the task then retains it: wait for the registry row instead
        # of returning early, so the assertions below run on every schedule.
        import time as _time
        for _ in range(300):
            ids = [m for m, e in url_media_store._REG.items()
                   if e.get("kind") == "video"]
            if ids:
                break
            _time.sleep(0.01)
        assert len(ids) == 1, "the pending video task never retained its file"
        vid = ids[0]
        entry = url_media_store._REG[vid]
        assert entry["ext"] == "mp4"
        assert entry["size"] == len(b"video-bytes" * 8)
    else:
        assert body["source_video_height"] == 720
        assert body["source_video_container"] == "mp4"
        assert body["source_video_bytes"] == len(b"video-bytes" * 8)
    assert len(vid) == 32 and vid != body["source_media_id"]
    assert video_enabled._video_calls[0] == {
        "max_height": 720, "container": "mp4", "expected_total": 3000,
        "format_ids": None, "leg_estimates": None}
    r = client.get(f"/v1/audio/url-media/{vid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/mp4")


def test_keep_video_pending_then_progress_reports_done(client, video_enabled):
    import time as _time

    release = video_enabled._video_gate["make"]()
    video_enabled._video_gate["release"] = release
    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200, r.text
    assert r.json().get("source_video_pending") is True
    entry = video_enabled._BATCH_PROGRESS.get(_PID)
    assert entry is not None, "the handler must leave the entry to the video task"
    assert entry["video"]["state"] == "downloading"
    assert entry["video"]["progress"] == 0.4
    assert _PID in video_enabled._VIDEO_TASKS
    prog = client.get(f"/v1/audio/transcriptions/progress/{_PID}").json()
    assert prog["video"]["state"] == "downloading"
    # Release the download: the task registers the file and pops the entry.
    release.set()
    for _ in range(300):
        prog = client.get(f"/v1/audio/transcriptions/progress/{_PID}").json()
        if prog.get("stage") == "unknown":
            break
        _time.sleep(0.01)
    assert prog.get("stage") == "unknown"
    assert _PID not in video_enabled._VIDEO_TASKS
    # The retained video is fetchable with a video mime.
    ids = [m for m, e in url_media_store._REG.items() if e.get("kind") == "video"]
    assert len(ids) == 1
    r = client.get(f"/v1/audio/url-media/{ids[0]}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/x-matroska")


def test_keep_video_finishing_late_lands_in_the_stored_job_result(
        client, video_enabled):
    import time as _time

    release = video_enabled._video_gate["make"]()
    video_enabled._video_gate["release"] = release
    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200 and r.json().get("source_video_pending") is True
    release.set()
    # Re-attaching after the fetch ended: the stored result names the video
    # (the progress entry that reported it is gone by then).
    body: dict = {}
    for _ in range(300):
        body = client.get(f"/v1/jobs/{_PID}/result").json()
        if body.get("source_video_media_id"):
            break
        _time.sleep(0.01)
    ids = [m for m, e in url_media_store._REG.items() if e.get("kind") == "video"]
    assert body.get("source_video_media_id") == ids[0]
    assert body["source_video_container"] == "mkv"
    assert "source_video_pending" not in body


def test_keep_video_without_progress_id_is_rejected(client, video_enabled):
    # The fetch outlives the response and reports through the progress entry
    # alone: without an id its media id would be unreachable by anyone.
    r = _post_url(client, keep_video="true")
    assert r.status_code == 422
    assert "progress_id" in r.json()["detail"]
    assert video_enabled._video_calls == []
    assert not [m for m, e in url_media_store._REG.items()
                if e.get("kind") == "video"]


def test_keep_video_with_an_upload_is_422(client, video_enabled):
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "keep_video": "true"})
    assert r.status_code == 422
    assert "link" in r.json()["detail"]


def test_keep_video_403_when_video_disabled(client, url_enabled, monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "URL_VIDEO_ENABLED", False, raising=False)
    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 403
    assert "video" in r.json()["detail"]


def test_keep_video_rate_limited(client, video_enabled, monkeypatch):
    monkeypatch.setattr(video_enabled.cfg, "URL_VIDEO_RATE_PER_MIN", 1, raising=False)
    assert _post_url(client, keep_video="true",
                     progress_id=_PID).status_code == 200
    r = _post_url(client, keep_video="true", progress_id="cafe" * 8)
    assert r.status_code == 429
    assert "video" in r.json()["detail"]


def test_keep_video_no_video_track_is_a_soft_error(client, url_enabled, monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "URL_VIDEO_ENABLED", True, raising=False)
    # url_enabled's probe returns an empty ladder: the transcript still
    # succeeds and the response says why there is no video.
    r = _post_url(client, keep_video="true", progress_id=_PID)
    assert r.status_code == 200, r.text
    assert r.json()["source_video_error"] == "this link has no video track"
    assert "source_video_media_id" not in r.json()


def test_on_demand_video_route(client, video_enabled):
    r = client.post("/v1/audio/url-media/video",
                    json={"url": _URL, "max_height": 720, "progress_id": _PID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["media_id"]) == 32
    assert body["height"] == 720 and body["container"] == "mp4"
    assert body["expires_at"] > 0 and body["bytes"] == len(b"video-bytes" * 8)
    assert _PID not in video_enabled._BATCH_PROGRESS
    assert client.get(f"/v1/audio/url-media/{body['media_id']}").status_code == 200
    # Validation and gates.
    assert client.post("/v1/audio/url-media/video", json={}).status_code == 422
    assert client.post("/v1/audio/url-media/video",
                       content=b"nope").status_code == 422


def test_video_reports_the_container_that_landed(client, video_enabled, monkeypatch):
    # No merge happened: the site's own pre-muxed webm sits behind a rung
    # that predicted "mp4". The client names its export after this field.
    async def _single(url, *, dest_dir, container="mkv", **kw):
        path = os.path.join(dest_dir, "media.webm")
        with open(path, "wb") as f:
            f.write(b"webm-bytes")
        return path
    monkeypatch.setattr(url_download, "download_video", _single)
    body = client.post("/v1/audio/url-media/video",
                       json={"url": _URL, "max_height": 720}).json()
    assert body["container"] == "webm"
    assert url_media_store._REG[body["media_id"]]["ext"] == "webm"


def test_video_ratio_is_learned_against_the_unscaled_estimate(
        client, video_enabled, monkeypatch):
    # approx_bytes is ALREADY scaled by the learned ratio: measuring against
    # it would settle the EWMA on sqrt(true ratio).
    from faster_whisper_backend.runtime import stage_rates

    async def _probe(url, *, timeout):
        rung = dict(_LADDER[1], bytes_approx=True, approx_bytes=44,
                    raw_approx_bytes=176, extractor="Youtube", protocol="m3u8")
        return _info(url=url, video_ladder=[rung, dict(_LADDER[2])])
    monkeypatch.setattr(url_download, "probe", _probe)
    seen: list = []
    monkeypatch.setattr(stage_rates, "record", lambda *a: seen.append(a))
    r = client.post("/v1/audio/url-media/video", json={"url": _URL})
    assert r.status_code == 200, r.text
    assert len(seen) == 1 and seen[0][0] == url_download.RATIO_STAGE
    assert seen[0][-1] == pytest.approx(len(b"video-bytes" * 8) / 176)


def test_on_demand_audio_route(client, url_enabled):
    r = client.post("/v1/audio/url-media/audio", json={"url": _URL, "progress_id": _PID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["media_id"]) == 32 and body["ext"] == "m4a"
    assert body["bytes"] == len(b"m4a-bytes" * 8) and body["expires_at"] > 0
    entry = url_media_store.resolve_entry(body["media_id"], user_id=None)
    assert entry["kind"] == "audio" and entry["source_url"] == _URL
    assert _PID not in url_enabled._BATCH_PROGRESS
    assert client.post("/v1/audio/url-media/audio", json={}).status_code == 422


def test_on_demand_audio_errors(client, url_enabled, monkeypatch):
    async def _fail(url, **kw):
        raise UrlDownloadError("this media is unavailable or has been removed")
    monkeypatch.setattr(url_download, "download", _fail)
    r = client.post("/v1/audio/url-media/audio", json={"url": _URL})
    assert r.status_code == 400 and "unavailable" in r.json()["detail"]

    async def _cancelled(url, **kw):
        raise url_download.UrlCancelled()
    monkeypatch.setattr(url_download, "download", _cancelled)
    assert client.post("/v1/audio/url-media/audio",
                       json={"url": _URL}).status_code == 499


def test_on_demand_audio_403_when_url_download_is_off(client):
    assert client.post("/v1/audio/url-media/audio",
                       json={"url": _URL}).status_code == 403


def test_on_demand_video_over_cap_is_400(client, video_enabled, monkeypatch):
    async def _probe(url, *, timeout):
        rungs = [dict(r, over_cap=True) for r in _LADDER]
        return _info(url=url, video_ladder=rungs)
    monkeypatch.setattr(url_download, "probe", _probe)
    r = client.post("/v1/audio/url-media/video", json={"url": _URL})
    assert r.status_code == 400
    assert "size limit" in r.json()["detail"]


def test_on_demand_video_403_when_disabled(client, url_enabled, monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "URL_VIDEO_ENABLED", False, raising=False)
    r = client.post("/v1/audio/url-media/video", json={"url": _URL})
    assert r.status_code == 403


def test_me_reports_video_caps(client, url_enabled, monkeypatch):
    body = client.get("/v1/me").json()
    assert body["url_video_enabled"] is True
    assert body["url_video_default_max_height"] is None
    assert body["media_max_bytes"] == url_enabled.cfg.MEDIA_MAX_BYTES
    monkeypatch.setattr(url_enabled.cfg, "URL_VIDEO_ENABLED", False, raising=False)
    body = client.get("/v1/me").json()
    assert body["url_video_enabled"] is False
    assert "url_video_default_max_height" not in body


def test_me_reports_video_off_when_url_download_is_off(client):
    body = client.get("/v1/me").json()
    assert body["url_video_enabled"] is False
    assert "media_max_bytes" not in body


def test_url_media_mime_by_kind(client, url_enabled, tmp_path):
    a = tmp_path / "a.mkv"
    a.write_bytes(b"a")
    v = tmp_path / "v.mp4"
    v.write_bytes(b"v")
    aid = url_media_store.register(str(a), user_id=None, kind="audio")
    vid = url_media_store.register(str(v), user_id=None, kind="video")
    assert client.get(f"/v1/audio/url-media/{aid}").headers["content-type"] \
        .startswith("audio/x-matroska")
    assert client.get(f"/v1/audio/url-media/{vid}").headers["content-type"] \
        .startswith("video/mp4")


# --- POST /v1/audio/url-subtitles ---------------------------------------------

_SUB_TRACKS = [{"id": "m-de", "lang": "de", "name": None, "kind": "manual",
                "ext": "vtt", "hoh": False}]
_SUB_SOURCES = {"m-de": {"url": "https://subs.test/de.vtt?pot=SECRET", "ext": "vtt"}}


@pytest.fixture
def subs_enabled(url_enabled, monkeypatch):
    async def _probe(url, *, timeout):
        return _info(url=url, subtitle_tracks=list(_SUB_TRACKS),
                     subtitle_sources=dict(_SUB_SOURCES))
    monkeypatch.setattr(url_download, "probe", _probe)
    monkeypatch.setattr(url_download, "_capped_get", lambda url, **kw: (
        "text/vtt", b"WEBVTT\n\n00:00.000 --> 00:01.000\nHallo\n"))
    return url_enabled


def test_subtitles_403_when_url_download_is_off(client):
    r = client.post("/v1/audio/url-subtitles", json={"url": _URL, "tracks": ["m-de"]})
    assert r.status_code == 403


def test_subtitles_403_when_its_switch_is_off(client, subs_enabled, monkeypatch):
    monkeypatch.setattr(subs_enabled.cfg, "URL_SUBTITLES_ENABLED", False, raising=False)
    r = client.post("/v1/audio/url-subtitles", json={"url": _URL, "tracks": ["m-de"]})
    assert r.status_code == 403
    assert "subtitle download" in r.json()["detail"]
    # …and the preview stops listing tracks.
    monkeypatch.setattr(url_download, "fetch_thumbnail_data_uri",
                        lambda *a, **k: asyncio.sleep(0))
    assert client.post("/v1/audio/url-preview",
                       json={"url": _URL}).json()["subtitle_tracks"] == []


@pytest.mark.parametrize("tracks", [[], ["m-de"] * 9, ["../x"], [7], "m-de", None])
def test_subtitles_422_on_bad_ids(client, subs_enabled, tracks):
    r = client.post("/v1/audio/url-subtitles", json={"url": _URL, "tracks": tracks})
    assert r.status_code == 422


def test_subtitles_happy_path_never_leaks_the_source(client, subs_enabled, caplog):
    r = client.post("/v1/audio/url-subtitles",
                    json={"url": _URL, "tracks": ["m-de", "m-de", "a-fr"]})
    assert r.status_code == 200
    body = r.json()
    assert [(t["id"], t["lang"], t["kind"], t["ext"]) for t in body["tracks"]] == [
        ("m-de", "de", "manual", "vtt")]
    assert body["tracks"][0]["text"].startswith("WEBVTT")
    assert [f["id"] for f in body["failed"]] == ["a-fr"]
    assert "SECRET" not in r.text and "SECRET" not in caplog.text


def test_subtitles_probe_rejection_is_400(client, subs_enabled, monkeypatch):
    async def _reject(url, *, timeout):
        raise UrlDownloadError("this video is private")
    monkeypatch.setattr(url_download, "probe", _reject)
    r = client.post("/v1/audio/url-subtitles", json={"url": _URL, "tracks": ["m-de"]})
    assert r.status_code == 400 and "private" in r.json()["detail"]


def test_subtitles_rate_limited(client, subs_enabled, monkeypatch):
    monkeypatch.setattr(subs_enabled.cfg, "URL_SUBTITLES_RATE_PER_MIN", 2, raising=False)
    for _ in range(2):
        assert client.post("/v1/audio/url-subtitles",
                           json={"url": _URL, "tracks": ["m-de"]}).status_code == 200
    r = client.post("/v1/audio/url-subtitles", json={"url": _URL, "tracks": ["m-de"]})
    assert r.status_code == 429
    assert r.json()["error"]["param"] == "URL_SUBTITLES_RATE_PER_MIN"


def test_me_reports_subtitle_caps(client, url_enabled, monkeypatch):
    assert client.get("/v1/me").json()["url_subtitles_enabled"] is True
    monkeypatch.setattr(url_enabled.cfg, "URL_SUBTITLES_ENABLED", False, raising=False)
    assert client.get("/v1/me").json()["url_subtitles_enabled"] is False


# --- POST /v1/audio/url-language ----------------------------------------------

@pytest.fixture
def lang_check(url_enabled, monkeypatch, fake_model):
    """The check with its decode stubbed (the url_enabled download writes no
    real audio) and every lease release recorded."""
    import numpy as np
    from faster_whisper_backend.audio import transcode
    monkeypatch.setattr(transcode, "decode_pieces_16k", lambda path, starts, s: [
        np.zeros(16000, dtype=np.float32) for _ in starts])
    released: list = []
    monkeypatch.setattr(url_enabled, "_release_model_lease", released.append)
    fake_model.released = released
    return fake_model


def test_language_403_when_off(client, url_enabled, monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "URL_LANGUAGE_CHECK_ENABLED", False, raising=False)
    r = client.post("/v1/audio/url-language", json={"url": _URL})
    assert r.status_code == 403 and "language check" in r.json()["detail"]
    # URL download off entirely: same 403, whatever the check's own switch.
    monkeypatch.setattr(url_enabled.cfg, "URL_DOWNLOAD_ENABLED", False, raising=False)
    monkeypatch.setattr(url_enabled.cfg, "URL_LANGUAGE_CHECK_ENABLED", True, raising=False)
    assert client.post("/v1/audio/url-language", json={"url": _URL}).status_code == 403


def test_language_check_undecodable_audio_is_400(client, url_enabled):
    # url_enabled's download writes bytes no demuxer reads.
    r = client.post("/v1/audio/url-language", json={"url": _URL})
    assert r.status_code == 400 and "decoded" in r.json()["detail"]


def test_language_check_votes_and_keeps_the_audio(client, url_enabled, lang_check,
                                                  monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "INFERENCE_CONCURRENCY", 1, raising=False)
    held: list = []
    real = type(lang_check).detect_language

    def _detect(self, audio=None, **kw):
        held.append((url_enabled.get_inference_semaphore().locked(), kw))
        return real(self, audio=audio, **kw)
    monkeypatch.setattr(type(lang_check), "detect_language", _detect)
    lang_check.heard = [("de", 0.9), ("de", 0.8), ("en", 0.88)]
    r = client.post("/v1/audio/url-language",
                    json={"url": _URL, "model": "whisper-1", "progress_id": _PID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["language"], body["verdict"], body["also"]) == ("de", "mixed", ["en"])
    assert [p["at"] for p in body["pieces"]] == [18.0, 45.0, 70.0]   # 90 s media
    assert [p["language"] for p in body["pieces"]] == ["de", "de", "en"]
    assert held == [(True, {"vad_filter": True})] * 3      # under the GPU gate
    assert lang_check.released == [url_enabled.cfg.DEFAULT_MODEL]
    entry = url_media_store.resolve_entry(body["media_id"], user_id=None)
    assert entry["kind"] == "audio" and entry["source_url"] == _URL
    assert body["media_expires_at"] > 0
    assert _PID not in url_enabled._BATCH_PROGRESS


def test_language_check_cancel_between_pieces(client, url_enabled, lang_check,
                                              monkeypatch):
    real = type(lang_check).detect_language

    def _detect(self, audio=None, **kw):
        url_enabled._BATCH_CANCELLED.add(_PID)       # the cancel route's flag
        return real(self, audio=audio, **kw)
    monkeypatch.setattr(type(lang_check), "detect_language", _detect)
    r = client.post("/v1/audio/url-language", json={"url": _URL, "progress_id": _PID})
    assert r.status_code == 499
    assert lang_check.detect_calls == 1 and len(lang_check.released) == 1


def test_language_check_no_speech_is_unknown(client, url_enabled, lang_check, monkeypatch):
    lang_check.heard = [(None, 0.0)]
    body = client.post("/v1/audio/url-language", json={"url": _URL}).json()
    assert body["verdict"] == "unknown" and body["language"] is None


def test_language_check_rate_limited(client, url_enabled, lang_check, monkeypatch):
    monkeypatch.setattr(url_enabled.cfg, "URL_LANGUAGE_RATE_PER_MIN", 1, raising=False)
    assert client.post("/v1/audio/url-language", json={"url": _URL}).status_code == 200
    r = client.post("/v1/audio/url-language", json={"url": _URL})
    assert r.status_code == 429
    assert r.json()["error"]["param"] == "URL_LANGUAGE_RATE_PER_MIN"


def test_me_reports_language_check_caps(client, url_enabled, monkeypatch):
    assert client.get("/v1/me").json()["url_language_check_enabled"] is True
    monkeypatch.setattr(url_enabled.cfg, "URL_LANGUAGE_CHECK_ENABLED", False, raising=False)
    assert client.get("/v1/me").json()["url_language_check_enabled"] is False

