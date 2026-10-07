"""Route-level tests for the media export: POST /v1/audio/media (raw-body
upload), GET /v1/audio/media/{id}/streams, POST /v1/audio/media/{id}/package.
ffmpeg and PyAV are stubbed at the package-module boundary; the media store,
the middleware and the handlers run real."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from faster_whisper_backend.media import media_store as ums
from faster_whisper_backend.media import subtitle_mux as pk

_ID = "a" * 32
_REAL_PROBE = pk.probe_streams   # before any fixture stubs it
_REAL_BUILD_ARGV = pk.build_package_argv


def _streams(**kw):
    base = dict(video_codec="h264", audio_codec="aac", width=1920, height=1080,
                duration=12.5, mp4_ok=True, mp4_reason=None)
    base.update(kw)
    return pk.MediaStreams(**base)


@pytest.fixture
def package_enabled(app_module, tmp_path, monkeypatch):
    """Packaging on, the store under tmp, ffmpeg + PyAV stubbed: the argv
    becomes a python script that copies the source and appends a marker."""
    monkeypatch.setattr(app_module.cfg, "MEDIA_PACKAGE_ENABLED", True, raising=False)
    # URL download stays OFF (the shipped default): packaging ids must be
    # fetchable through /v1/audio/url-media without the link feature.
    monkeypatch.setattr(app_module.cfg, "URL_DOWNLOAD_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module.cfg, "URL_MEDIA_DIR", str(tmp_path / "url_media"),
                        raising=False)
    ums.startup_reset()
    pk._reset_for_tests()
    monkeypatch.setattr(pk, "ffmpeg_capabilities",
                        lambda: pk.FfmpegCaps(True, True, True, None, "7.0.2-test"))
    probe_calls: list = []

    def _probe(path):
        probe_calls.append(path)
        return _streams()
    monkeypatch.setattr(pk, "probe_streams", _probe)

    # The empty-caption scan would run the host's real `ffmpeg -bsfs` (and
    # an ffmpeg demux of the fake source) for every h264 package.
    async def _no_strip(*a, **k):
        return False
    monkeypatch.setattr(pk, "_empty_captions_to_strip", _no_strip)
    latch = {"path": None}

    def _argv(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        # "<latch>.started" says the job got this far — past the in-flight
        # slot it holds — before it blocks on the latch.
        # The wait has its own deadline: a regressed guard or a failed
        # assert before the unlink must fail the test, not hang the suite.
        script = f"""
import os, shutil, time
if {latch['path']!r}:
    open({latch['path']!r} + ".started", "w").close()
deadline = time.time() + 20
while ({latch['path']!r} and os.path.exists({latch['path']!r})
       and time.time() < deadline):
    time.sleep(0.02)
shutil.copyfile({src!r}, {out_path!r})
with open({out_path!r}, "ab") as f:
    f.write(b"|muxed:" + {container!r}.encode() + b":" + str({len(srt_paths)}).encode())
"""
        return [sys.executable, "-c", script]
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    app_module._probe_calls = probe_calls
    app_module._package_latch = latch
    return app_module


def _upload(client, data=b"video-bytes" * 16, ext="mp4", headers=None):
    return client.post(f"/v1/audio/media?ext={ext}", content=data,
                       headers={"Content-Type": "application/octet-stream",
                                **(headers or {})})


def _tracks():
    return [{"lang": "en", "label": "English · original",
             "srt": "1\n00:00:00,000 --> 00:00:01,000\nHi\n"},
            {"lang": "de", "srt": "1\n00:00:00,000 --> 00:00:01,000\nHallo\n"}]


# --- upload -------------------------------------------------------------------

def test_upload_streams_to_the_store_and_is_fetchable(client, package_enabled):
    r = _upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["media_id"]) == 32 and body["bytes"] == len(b"video-bytes" * 16)
    assert body["expires_at"] > 0
    e = ums.resolve_entry(body["media_id"], user_id=None)
    assert e["kind"] == "video" and e["ext"] == "mp4"
    r = client.get(f"/v1/audio/url-media/{body['media_id']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/mp4")
    assert r.content == b"video-bytes" * 16
    # No .part left in staging.
    assert not any(n.endswith(".part") for n in os.listdir(ums.staging_dir()))


def test_upload_over_cap_413_leaves_no_file(client, package_enabled, monkeypatch):
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_MAX_BYTES", 100, raising=False)

    def _gen():
        for _ in range(3):
            yield b"x" * 100
    r = client.post("/v1/audio/media?ext=mp4", content=_gen(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 413
    assert not any(n.startswith("upload-") for n in os.listdir(ums.staging_dir()))
    assert not [m for m, e in ums._REG.items() if e.get("kind") == "video"]


def test_upload_declared_length_over_cap_is_an_early_413(client, package_enabled,
                                                          monkeypatch):
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_MAX_BYTES", 100, raising=False)
    r = client.post("/v1/audio/media?ext=mp4", content=b"x",
                    headers={"Content-Type": "application/octet-stream",
                             "Content-Length": "5000"})
    assert r.status_code == 413


def test_upload_over_the_retention_cap_is_an_early_413_naming_it(
        client, package_enabled, monkeypatch):
    """register() drops a file over RETAINED_MEDIA_MAX_BYTES on arrival: an
    upload between that cap and MEDIA_MAX_BYTES is refused up front with
    the cap that binds, not with a 507 "store is full" after the transfer."""
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_MAX_BYTES", 10_000, raising=False)
    monkeypatch.setattr(package_enabled.cfg, "RETAINED_MEDIA_MAX_BYTES", 1000,
                        raising=False)
    r = _upload(client, data=b"x" * 5000)
    assert r.status_code == 413
    assert "RETAINED_MEDIA_MAX_BYTES" in r.json()["detail"]

    def _gen():
        for _ in range(5):
            yield b"x" * 1000
    r = client.post("/v1/audio/media?ext=mp4", content=_gen(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 413
    assert not any(n.startswith("upload-") for n in os.listdir(ums.staging_dir()))


def test_upload_without_disk_space_is_507_and_leaves_no_file(
        client, package_enabled, monkeypatch):
    from faster_whisper_backend.media import routes as media_routes
    real = shutil.disk_usage

    def _full(path):
        return real(path)._replace(free=1024)
    monkeypatch.setattr(media_routes.shutil, "disk_usage", _full)
    r = _upload(client)
    assert r.status_code == 507
    assert "disk space" in r.json()["detail"]
    assert not any(n.startswith("upload-") for n in os.listdir(ums.staging_dir()))


def test_upload_hitting_a_full_disk_mid_stream_is_507(client, package_enabled,
                                                      monkeypatch):
    import errno
    import io

    from faster_whisper_backend.media import routes as media_routes
    real_fdopen = os.fdopen

    class _Full(io.RawIOBase):
        def __init__(self, fd):
            self._f = real_fdopen(fd, "wb")

        def writable(self):
            return True

        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):
            self._f.close()
            super().close()
    monkeypatch.setattr(media_routes.os, "fdopen", lambda fd, mode: _Full(fd))
    r = _upload(client)
    assert r.status_code == 507
    assert not any(n.startswith("upload-") for n in os.listdir(ums.staging_dir()))


def test_upload_validation_and_gates(client, package_enabled, monkeypatch):
    assert _upload(client, ext="").status_code == 422
    assert _upload(client, ext="MP4!").status_code == 422
    assert _upload(client, data=b"").status_code == 422
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_PACKAGE_ENABLED", False, raising=False)
    assert _upload(client).status_code == 403


def test_upload_rate_limited(client, package_enabled, monkeypatch):
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_UPLOAD_RATE_PER_MIN", 1, raising=False)
    assert _upload(client).status_code == 200
    r = _upload(client)
    assert r.status_code == 429
    assert "upload" in r.json()["detail"]


def test_upload_is_exempt_from_the_request_body_cap(client, package_enabled, monkeypatch):
    # The service-wide multipart/other cap is tiny; the media route counts
    # its own bytes against MEDIA_MAX_BYTES and must still accept 500 B.
    monkeypatch.setattr(package_enabled.cfg, "MAX_REQUEST_BYTES", 10, raising=False)
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_MAX_BYTES", 1000, raising=False)
    assert _upload(client, data=b"x" * 500).status_code == 200


# --- streams ------------------------------------------------------------------

def test_streams_reports_the_probe_and_caches_it(client, package_enabled):
    mid = _upload(client).json()["media_id"]
    r = client.get(f"/v1/audio/media/{mid}/streams")
    assert r.status_code == 200
    assert r.json() == _streams().as_dict()
    client.get(f"/v1/audio/media/{mid}/streams")
    assert len(package_enabled._probe_calls) == 1
    assert client.get(f"/v1/audio/media/{'f' * 32}/streams").status_code == 404
    assert client.get("/v1/audio/media/NOPE/streams").status_code == 422


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a system ffmpeg")
def test_ffconcat_upload_naming_a_retained_video_is_unreadable(
        client, package_enabled, monkeypatch, tmp_path):
    """A ~100-byte ffconcat upload listing another retained file N times
    would probe as THAT video (the whitelisted file protocol admits a bare
    same-directory name) and stream-copy N x its size past MEDIA_MAX_BYTES
    and the free-space check: the probe refuses the demuxer instead."""
    pytest.importorskip("av")
    monkeypatch.setattr(pk, "probe_streams", _REAL_PROBE)
    src = str(tmp_path / "a.mkv")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
                    "-c:v", "mpeg4", src], check=True, timeout=60)
    with open(src, "rb") as f:
        id_a = _upload(client, data=f.read(), ext="mkv").json()["media_id"]
    assert client.get(f"/v1/audio/media/{id_a}/streams").json()["video_codec"]
    playlist = ("ffconcat version 1.0\n" + f"file '{id_a}.mkv'\n" * 50).encode()
    id_b = _upload(client, data=playlist, ext="mp4").json()["media_id"]
    r = client.get(f"/v1/audio/media/{id_b}/streams")
    assert r.status_code == 200 and r.json()["unreadable"] is True, r.text
    r = client.post(f"/v1/audio/media/{id_b}/package",
                    json={"container": "mkv", "subtitles": _tracks()})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unreadable"


# --- package ------------------------------------------------------------------

def test_package_forwards_original_track_and_audio_language(client, package_enabled, monkeypatch):
    seen = {}

    def _argv(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        seen.update(kw, default_track=default_track)
        return [sys.executable, "-c", f"open({out_path!r}, 'wb').write(b'video-bytes')"]
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mkv", "subtitles": _tracks(), "default_track": 0,
                          "original_track": 0, "audio_lang": " de ", "audio_label": "Ger\x01man"})
    assert r.status_code == 200, r.text
    assert seen == {"default_track": 0, "original_track": 0,
                    "audio_lang": "de", "audio_label": "German"}
    # Out-of-range / malformed values are refused, not silently dropped.
    for bad in ({"original_track": 5}, {"original_track": True}, {"audio_lang": "German"}):
        r = client.post(f"/v1/audio/media/{mid}/package",
                        json={"container": "mkv", "subtitles": _tracks(), **bad})
        assert r.status_code == 422, (bad, r.text)


def test_package_forwards_per_track_flags(client, package_enabled, monkeypatch):
    seen = {}

    def _argv(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        seen["flags"] = [(t.default, t.original, t.hearing_impaired) for t in tracks]
        seen["indices"] = (default_track, kw.get("original_track"))
        return [sys.executable, "-c", f"open({out_path!r}, 'wb').write(b'video-bytes')"]
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    mid = _upload(client).json()["media_id"]
    subs = _tracks()
    subs[0].update(default=True, original=True)
    subs[1].update(hearing_impaired=True)
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mkv", "subtitles": subs, "original_track": 1})
    assert r.status_code == 200, r.text
    # Absent booleans are off; the legacy index travels alongside untouched.
    assert seen == {"flags": [(True, True, False), (False, False, True)],
                    "indices": (None, 1)}
    for k in ("default", "original", "hearing_impaired"):
        for bad in (1, "true", None):
            subs = _tracks()
            subs[0][k] = bad
            r = client.post(f"/v1/audio/media/{mid}/package",
                            json={"container": "mkv", "subtitles": subs})
            assert r.status_code == 422 and f"subtitles[0].{k}" in r.text, (k, bad, r.text)


def test_package_mkv_happy_path_streams_the_file_and_cleans_up(client, package_enabled):
    mid = _upload(client).json()["media_id"]
    before = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mkv", "subtitles": _tracks(),
                          "default_track": 0, "filename": "My talk: final?"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("video/x-matroska")
    assert "My%20talk%20final.mkv" in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "no-store"
    assert r.content.endswith(b"|muxed:mkv:2")
    assert r.content.startswith(b"video-bytes")
    after = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
    assert after <= before


@pytest.mark.parametrize("rng,status", [("bytes=999999999999-", 416), ("bytes=x-y", 400)])
def test_package_range_refusal_still_removes_the_workdir(client, package_enabled,
                                                        rng, status):
    """Starlette's FileResponse answers a bad Range (honoured on POST too)
    without running `background`: the workdir — a full copy of the video —
    must go anyway."""
    mid = _upload(client).json()["media_id"]
    before = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
    r = client.post(f"/v1/audio/media/{mid}/package", headers={"Range": rng},
                    json={"container": "mkv", "subtitles": _tracks()})
    assert r.status_code == status, r.text
    after = {n for n in os.listdir(tempfile.gettempdir()) if n.startswith("fwb-pkg-")}
    assert after <= before


def test_package_mp4_when_the_streams_fit_else_422_with_code(client, package_enabled,
                                                            monkeypatch):
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mp4", "subtitles": _tracks()[:1]})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("video/mp4")
    assert r.content.endswith(b"|muxed:mp4:1")
    # A VP9 source: MP4 refused with the reason, MKV still fine.
    monkeypatch.setattr(pk, "probe_streams", lambda p: _streams(
        video_codec="vp9", audio_codec="opus", mp4_ok=False,
        mp4_reason="MP4 can't carry VP9 video without re-encoding — choose MKV"))
    mid2 = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid2}/package",
                    json={"container": "mp4", "subtitles": []})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "mp4_incompatible"
    assert "VP9" in r.json()["detail"]["message"]
    assert client.post(f"/v1/audio/media/{mid2}/package",
                       json={"container": "mkv"}).status_code == 200


def test_package_no_video_stream_422(client, package_enabled, monkeypatch):
    monkeypatch.setattr(pk, "probe_streams", lambda p: _streams(video_codec=None))
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "no_video"


def test_package_unreadable_source_is_not_reported_as_no_video(
        client, package_enabled, monkeypatch):
    def _boom(path):
        raise ValueError("moov atom not found")
    monkeypatch.setattr(pk, "probe_streams", _boom)
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
    assert r.status_code == 422
    assert r.json()["detail"] == {"code": "unreadable",
                                  "message": pk.UNREADABLE_REASON}
    assert "moov" not in r.text


def test_package_cover_art_only_says_so(client, package_enabled, monkeypatch):
    monkeypatch.setattr(pk, "probe_streams", lambda p: _streams(
        video_codec=None, cover_art_only=True))
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "no_video"
    assert "cover art" in r.json()["detail"]["message"]


def test_package_maps_the_probed_video_index(client, package_enabled, monkeypatch):
    seen: list = []
    inner = pk.build_package_argv

    def _argv(*a, **kw):
        seen.append(kw.get("video_index"))
        return inner(*a, **kw)
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    monkeypatch.setattr(pk, "probe_streams", lambda p: _streams(video_index=1))
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
    assert r.status_code == 200, r.text
    assert seen == [1]


def test_package_body_cap_covers_every_limit_compliant_request(
        client, package_enabled, monkeypatch):
    # MAX_TRACKS tracks of MAX_SRT_BYTES each (24 MiB, past the old 16 MiB
    # cap) must reach the route, not the middleware's bare 413.
    def _argv(src, srt_paths, tracks, *, container, out_path, **kw):
        return [sys.executable, "-c", f"open({out_path!r}, 'wb').write(b'video-bytes')"]
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    mid = _upload(client).json()["media_id"]
    cue = "1\n00:00:00,000 --> 00:00:01,000\n"
    srt = cue + "x" * (pk.MAX_SRT_BYTES - len(cue) - 1) + "\n"
    subs = [{"lang": "en", "srt": srt} for _ in range(pk.MAX_TRACKS)]
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mkv", "subtitles": subs})
    assert r.status_code == 200, r.text[:200]


def test_upload_whose_spool_vanished_is_a_deliberate_500(
        client, package_enabled, monkeypatch):
    from faster_whisper_backend.media import routes as media_routes

    def _gone(src, dst):
        raise FileNotFoundError(src)
    monkeypatch.setattr(media_routes.os, "replace", _gone)
    r = _upload(client)
    assert r.status_code == 500
    assert r.json()["detail"] == "upload staging file vanished"
    # The handler's finally still drops the .part.
    assert not any(n.startswith("upload-") for n in os.listdir(ums.staging_dir()))


@pytest.mark.parametrize("body", [
    {"container": "webm"},
    {"container": "mkv", "subtitles": [{"lang": "EN", "srt": "1\n0 --> 1\nx"}] },
    {"container": "mkv", "subtitles": [{"lang": "en", "srt": "no cues here"}]},
    # Valid SRT, so only the MAX_TRACKS cap (12) can refuse it.
    {"container": "mkv",
     "subtitles": [{"lang": "en", "srt": "1\n00:00:00,000 --> 00:00:01,000\nx\n"}] * 13},
    {"container": "mkv", "subtitles": [{"lang": "en", "srt": "1\n0 --> 1\nx"}],
     "default_track": 5},
    {"container": "mkv", "subtitles": "nope"},
])
def test_package_validation_422(client, package_enabled, body):
    mid = _upload(client).json()["media_id"]
    r = client.post(f"/v1/audio/media/{mid}/package", json=body)
    assert r.status_code == 422, r.text


def test_package_lone_surrogates_are_422_or_stripped_never_500(
        client, package_enabled, monkeypatch):
    """A JSON body may carry a lone UTF-16 surrogate: in an SRT it is not
    SRT (it cannot be encoded), in a label it is dropped — it would
    otherwise reach ffmpeg's argv, where os.fsencode raises."""
    import json

    def _post(body):
        # ASCII-escaped on the wire ("\\ud800"), as a client would send it:
        # httpx's json= cannot encode a lone surrogate at all.
        return client.post(f"/v1/audio/media/{mid}/package",
                           content=json.dumps(body),
                           headers={"Content-Type": "application/json"})
    mid = _upload(client).json()["media_id"]
    r = _post({"container": "mkv", "subtitles": [
        {"lang": "en", "srt": "1\n00:00:00,000 --> 00:00:01,000\n\ud800\n"}]})
    assert r.status_code == 422
    assert "not SRT" in r.json()["detail"]
    seen = {}

    def _argv(src, srt_paths, tracks, *, container, out_path, default_track, **kw):
        argv = _REAL_BUILD_ARGV(
            src, srt_paths, tracks, container=container, out_path=out_path,
            default_track=default_track, **kw)
        seen["argv"] = argv
        return [sys.executable, "-c", f"open({out_path!r}, 'wb').write(b'v')"]
    monkeypatch.setattr(pk, "build_package_argv", _argv)
    r = _post({"container": "mkv", "audio_lang": "en", "audio_label": "\ud800",
               "subtitles": [{"lang": "en", "label": "\ud800x",
                              "srt": "1\n00:00:00,000 --> 00:00:01,000\nHi\n"}]})
    assert r.status_code == 200, r.text
    for arg in seen["argv"]:
        arg.encode("utf-8")              # no lone surrogate left anywhere
    assert "title=x" in seen["argv"]


def test_package_srt_over_the_size_cap_422(client, package_enabled):
    mid = _upload(client).json()["media_id"]
    big = "1\n00:00:00,000 --> 00:00:01,000\n" + "x" * (pk.MAX_SRT_BYTES + 1)
    r = client.post(f"/v1/audio/media/{mid}/package",
                    json={"container": "mkv", "subtitles": [{"lang": "en", "srt": big}]})
    assert r.status_code == 422
    assert "larger" in r.json()["detail"]


def test_package_unknown_id_404_and_disabled_403_and_no_ffmpeg_503(
        client, package_enabled, monkeypatch):
    assert client.post(f"/v1/audio/media/{_ID}/package",
                       json={"container": "mkv"}).status_code == 404
    monkeypatch.setattr(pk, "ffmpeg_capabilities",
                        lambda: pk.FfmpegCaps(False, False, False, "no ffmpeg binary", None))
    r = client.post(f"/v1/audio/media/{_ID}/package", json={"container": "mkv"})
    assert r.status_code == 503 and "ffmpeg" in r.json()["detail"]
    body = client.get("/v1/me").json()
    assert body["media_package_enabled"] is False
    assert body["media_package"]["reason"] == "no ffmpeg binary"
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_PACKAGE_ENABLED", False, raising=False)
    assert client.post(f"/v1/audio/media/{_ID}/package",
                       json={"container": "mkv"}).status_code == 403
    body = client.get("/v1/me").json()
    assert body["media_package_enabled"] is False and "media_package" not in body


def test_package_foreign_owner_404(client, package_enabled, make_user_key):
    from tests.conftest import bearer
    _ua, key_a = make_user_key("alice", is_admin=True)
    _ub, key_b = make_user_key("bob", is_admin=True)
    mid = _upload(client, headers=bearer(key_a)).json()["media_id"]
    assert client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"},
                       headers=bearer(key_b)).status_code == 404
    assert client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"},
                       headers=bearer(key_a)).status_code == 200


def test_package_inflight_limit_and_release(client, package_enabled, tmp_path):
    mid = _upload(client).json()["media_id"]
    latch = str(tmp_path / "latch")
    open(latch, "w").close()
    package_enabled._package_latch["path"] = latch
    result: dict = {}

    def _first():
        result["r"] = client.post(f"/v1/audio/media/{mid}/package",
                                  json={"container": "mkv"})
    t = threading.Thread(target=_first, daemon=True)
    t.start()
    try:
        # Wait until the first job demonstrably holds the slot (its package
        # step started), not a fixed sleep a loaded runner can outrun.
        deadline = time.monotonic() + 10
        while not os.path.exists(latch + ".started"):
            assert time.monotonic() < deadline, "the first package never started"
            time.sleep(0.02)
        r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
        assert r.status_code == 429
        # The in-flight guard's wording — the rate limiter's 429 also says
        # "video export".
        assert "running" in r.json()["detail"]
    finally:
        if os.path.exists(latch):
            os.unlink(latch)
        t.join(timeout=30)
        package_enabled._package_latch["path"] = None
    assert result["r"].status_code == 200
    assert client.post(f"/v1/audio/media/{mid}/package",
                       json={"container": "mkv"}).status_code == 200


def test_package_timeout_504(client, package_enabled, monkeypatch, tmp_path):
    monkeypatch.setattr(package_enabled.cfg, "MEDIA_PACKAGE_TIMEOUT_S", 1, raising=False)
    mid = _upload(client).json()["media_id"]
    latch = str(tmp_path / "latch2")
    open(latch, "w").close()
    package_enabled._package_latch["path"] = latch
    try:
        r = client.post(f"/v1/audio/media/{mid}/package", json={"container": "mkv"})
        assert r.status_code == 504
    finally:
        os.unlink(latch)
        package_enabled._package_latch["path"] = None


def test_me_reports_media_package_caps(client, package_enabled):
    body = client.get("/v1/me").json()
    assert body["media_package_enabled"] is True
    assert body["media_package"]["containers"] == ["mkv", "mp4"]
    assert body["media_package"]["max_tracks"] == pk.MAX_TRACKS
    assert body["media_package"]["max_upload_bytes"] == package_enabled.cfg.MEDIA_MAX_BYTES
    assert body["media_package"]["reason"] is None


def test_sweep_reaps_stale_upload_parts(package_enabled):
    part = os.path.join(ums.staging_dir(), "upload-deadbeef.mp4.part")
    with open(part, "wb") as f:
        f.write(b"x")
    stale = time.time() - (ums._UPLOAD_IDLE_MAX_SEC + 100)
    os.utime(part, (stale, stale))
    ums.sweep()
    assert not os.path.exists(part)


def test_retain_media_keeps_the_upload_and_returns_its_id(client, package_enabled):
    _FILE = {"file": ("a.mp4", b"RIFFxxxxWAVE" + b"\0" * 64, "video/mp4")}
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "response_format": "verbose_json",
                          "retain_media": "true"})
    assert r.status_code == 200, r.text
    mid = r.json()["source_media_id"]
    assert len(mid) == 32 and r.json()["source_media_expires_at"] > 0
    e = ums.resolve_entry(mid, user_id=None)
    assert e["kind"] == "video" and e["ext"] == "mp4"
    assert client.get(f"/v1/audio/url-media/{mid}").content.startswith(b"RIFF")
    # Without the flag nothing is retained; the spool (whisperup-) and any
    # retained copy (urlmedia-) are gone either way.
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "response_format": "verbose_json"})
    assert "source_media_id" not in r.json()
    assert not [n for n in os.listdir(tempfile.gettempdir())
                if n.startswith(("urlmedia-", "whisperup-")) and time.time() - os.path.getmtime(
                    os.path.join(tempfile.gettempdir(), n)) < 5]


def test_retain_media_with_a_text_response_is_422(client, package_enabled):
    # A text body has no field for the media id: refused up front, before
    # the spool (and its hardlinked copy) is written.
    _FILE = {"file": ("a.mp4", b"RIFFxxxxWAVE" + b"\0" * 64, "video/mp4")}
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "response_format": "text",
                          "retain_media": "true"})
    assert r.status_code == 422
    assert "retain_media" in r.json()["detail"]
    assert not [n for n in os.listdir(tempfile.gettempdir())
                if n.startswith(("urlmedia-", "whisperup-")) and time.time() - os.path.getmtime(
                    os.path.join(tempfile.gettempdir(), n)) < 5]
    assert not [m for m, e in ums._REG.items() if e.get("kind") == "video"]


def test_retain_media_refused_by_the_store_leaves_no_copy(client, package_enabled,
                                                          monkeypatch):
    # register() answering None (store full / unwritable) leaves the copy on
    # disk: the handler must still unlink it.
    monkeypatch.setattr(ums, "register", lambda *a, **kw: None)
    _FILE = {"file": ("a.mp4", b"RIFFxxxxWAVE" + b"\0" * 64, "video/mp4")}
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1", "response_format": "verbose_json",
                          "retain_media": "true"})
    assert r.status_code == 200, r.text
    assert "source_media_id" not in r.json()
    assert not [n for n in os.listdir(tempfile.gettempdir())
                if n.startswith(("urlmedia-", "whisperup-")) and time.time() - os.path.getmtime(
                    os.path.join(tempfile.gettempdir(), n)) < 5]
