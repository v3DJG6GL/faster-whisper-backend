"""probe()'s derivation from the site's info dict (video ladder, subtitle
list, spoken language, segmented source) walks every format and subtitle
key the site sent. That is CPU work proportional to an attacker-shaped
payload, so it runs on _PROBE_POOL, never on the event loop that every
live dictation socket and SSE stream shares."""
from __future__ import annotations

from asyncio import run as _run
import threading

import pytest

from faster_whisper_backend.media import download as udl
from faster_whisper_backend.media import segmented as _seg
from faster_whisper_backend.media import subtitles as _subs
from tests.media.test_download import _stand_in_yt_dlp

_INFO = {
    "extractor_key": "Youtube", "title": "t", "duration": 100.0,
    "filesize": 900_000, "ext": "m4a", "abr": 129.5,
    "language": "en",
    "subtitles": {"en": [{"ext": "vtt", "url": "https://e.example/en.vtt"}]},
    "formats": [
        {"format_id": "137", "vcodec": "avc1", "acodec": "none",
         "height": 1080, "ext": "mp4", "filesize": 5_000_000},
    ],
}


def _recording(monkeypatch, module, name, seen):
    real = getattr(module, name)

    def wrapped(*a, **k):
        seen[name] = threading.current_thread().name
        return real(*a, **k)
    monkeypatch.setattr(module, name, wrapped)


@pytest.fixture
def probed(monkeypatch):
    _stand_in_yt_dlp(monkeypatch, "Youtube", dict(_INFO))
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 100_000_000, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_VIDEO_ENABLED", True, raising=False)
    seen: dict = {}
    _recording(monkeypatch, udl, "build_video_ladder", seen)
    _recording(monkeypatch, _subs, "list_tracks", seen)
    _recording(monkeypatch, _subs, "language_of", seen)
    _recording(monkeypatch, _seg, "source_of", seen)
    return seen


def test_probe_derives_ladder_and_subtitles_on_the_probe_pool(probed):
    info = _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))
    assert set(probed) == {"build_video_ladder", "list_tracks",
                           "language_of", "source_of"}
    assert all(t.startswith("url-probe") for t in probed.values()), probed
    # The result is the same one the loop-side derivation built.
    assert [t["id"] for t in info.subtitle_tracks] == ["m-en"]
    assert info.language == "en"
    assert info.video_ladder and info.video_ladder[-1]["kind"] == "audio"


def test_probe_derivation_error_still_reaches_the_caller(monkeypatch, probed):
    """An exception the derivation raises surfaces from probe() unchanged,
    as it did when the derivation ran inline."""
    def boom(info):
        raise udl.UrlDownloadError("odd subtitles")
    monkeypatch.setattr(_subs, "list_tracks", boom)
    with pytest.raises(udl.UrlDownloadError, match="odd subtitles"):
        _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))


def test_probe_derive_is_held_to_the_probe_deadline(monkeypatch, probed):
    """The derivation hop spends from the probe's one deadline like the
    policy and extraction hops: a derivation that cannot finish in time
    (a saturated pool, a huge payload) answers with the curated timeout
    instead of overrunning URL_PREVIEW_TIMEOUT_S."""
    import time

    release = threading.Event()

    def stuck(info):
        release.wait(10)
        return [], {}
    monkeypatch.setattr(_subs, "list_tracks", stuck)
    t0 = time.monotonic()
    try:
        with pytest.raises(udl.UrlTimeoutError, match="too long"):
            _run(udl.probe("https://example.com/watch?v=x", timeout=1.0))
    finally:
        release.set()
    assert time.monotonic() - t0 < 4.0
