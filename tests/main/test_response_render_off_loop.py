"""The end-of-run response render of POST /v1/audio/transcriptions runs in a
worker thread: jsonable_encoder + json.dumps of a long verbose_json (2 700
segments, 27 000 words for 3 h of audio) and the per-segment pipeline walk of
an srt/vtt document would otherwise freeze every live stream on the worker
for a few hundred ms. Bytes, status and headers stay what FastAPI's own
field-less serialisation gives."""

import asyncio
import json

import pytest
from starlette.responses import JSONResponse

from faster_whisper_backend.media import subtitles as _subtitles
from tests._transcription_helpers import post_transcription, serve_model
from tests.conftest import FakeInfo, FakeModel, FakeSegment, FakeWord

_FORMATS = ("json", "verbose_json", "srt", "vtt")


def _on_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def three_segments(monkeypatch):
    segs = [
        FakeSegment("Grüße aus Köln", 0.0, 1.5,
                    [FakeWord("Grüße", 0.0, 0.6), FakeWord("aus", 0.6, 0.9),
                     FakeWord("Köln", 0.9, 1.5)]),
        FakeSegment("zweiter Satz", 1.5, 3.0,
                    [FakeWord("zweiter", 1.5, 2.2), FakeWord("Satz", 2.2, 3.0)]),
        FakeSegment("日本語も", 3.0, 4.25, [FakeWord("日本語も", 3.0, 4.25)]),
    ]
    serve_model(monkeypatch, FakeModel(segments=segs,
                                       info=FakeInfo(duration=4.25)))


def _post(client, fmt):
    r = post_transcription(client, response_format=fmt,
                           timestamp_granularities="word")
    assert r.status_code == 200, r.text
    return r


@pytest.mark.parametrize("fmt", ("json", "verbose_json"))
def test_json_response_renders_off_the_loop(client, three_segments,
                                            monkeypatch, fmt):
    seen = []
    real = JSONResponse.render

    def _render(self, content):
        if isinstance(content, dict) and "text" in content:
            seen.append(_on_loop())
        return real(self, content)
    monkeypatch.setattr(JSONResponse, "render", _render)
    _post(client, fmt)
    assert seen == [False]


@pytest.mark.parametrize("fmt", ("srt", "vtt"))
def test_subtitle_document_renders_off_the_loop(client, three_segments,
                                                monkeypatch, fmt):
    seen = []
    real = _subtitles.render

    def _render(cues, fmt_):
        seen.append(_on_loop())
        return real(cues, fmt_)
    monkeypatch.setattr(_subtitles, "render", _render)
    _post(client, fmt)
    assert seen == [False]


@pytest.mark.parametrize("fmt", _FORMATS)
def test_rendered_bytes_match_fastapis_own_serialisation(
        client, three_segments, fmt):
    r = _post(client, fmt)
    assert r.headers["content-length"] == str(len(r.content))
    if fmt in ("json", "verbose_json"):
        assert r.headers["content-type"] == "application/json"
        assert r.content == json.dumps(
            r.json(), ensure_ascii=False, allow_nan=False, indent=None,
            separators=(",", ":")).encode("utf-8")
    else:
        assert r.headers["content-type"] == "text/plain; charset=utf-8"



def test_joined_postprocess_runs_off_the_loop(client, fake_model, monkeypatch,
                                              recording_captures):
    """The joined transcript's pipeline walk (and the capture-form second
    walk) is many rule subs over the whole text: on a 3 h run each walk
    costs ~0.1 s, so both run in a worker thread like the srt/vtt walk."""
    from faster_whisper_backend.pipeline import engine as pl_engine
    seen = []
    real = pl_engine._postprocess_text

    def _walk(text, *a, **kw):
        seen.append(("capture" if "extra_excludes" in kw else "user",
                     _on_loop()))
        return real(text, *a, **kw)
    monkeypatch.setattr(pl_engine, "_postprocess_text", _walk)
    r = post_transcription(client, response_format="json")
    assert r.status_code == 200, r.text
    assert len(recording_captures) == 1
    assert sorted(seen) == [("capture", False), ("user", False)]
