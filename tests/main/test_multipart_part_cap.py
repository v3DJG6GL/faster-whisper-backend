"""POST /v1/audio/transcriptions and /v1/audio/translations refuse a
multipart body with more parts than any legitimate upload carries, while the
body streams in. FastAPI runs `await request.form()` before the handler with
Starlette's defaults (1000 text fields of 1 MiB held in RAM, 1000 file parts
of 1 MiB spooled in RAM), and the upload routes keep the full
MAX_REQUEST_BYTES, so one authenticated request could otherwise pin ~2 GB of
RSS for its whole lifetime. _max_body_mw counts the boundary delimiters on
the receive channel and ends the body past _MAX_UPLOAD_PARTS."""

import pytest

from faster_whisper_backend.transcription import models as tx_models
from tests._transcription_helpers import post_json_ok

_BOUNDARY = b"pc-test-boundary"
_CTYPE = {"Content-Type": "multipart/form-data; boundary=pc-test-boundary"}
_UPLOAD_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")
_WAV = b"RIFFxxxxWAVE" + b"\0" * 4096

# Every Form()/File() param the transcription route declares, plus the
# repeatable timestamp_granularities in both spellings the handler reads.
_EVERY_FIELD = {
    "source_url": "", "model": "whisper-1", "response_format": "json",
    "language": "", "temperature": "0", "prompt": "", "decode_overrides": "",
    "override_profile": "", "task": "transcribe", "diarize": "",
    "num_speakers": "", "min_speakers": "", "max_speakers": "",
    "diarization_model": "", "separate_bgm": "", "separation_model": "",
    "translate_to": "", "translation_model": "", "translation_mode": "",
    "translation_glossary": "", "context_segments": "", "keep_video": "",
    "video_max_height": "", "video_format": "", "retain_media": "",
    "prefetched_media_id": "", "progress_id": "", "preload_plan": "",
}
_GRANULARITIES = [("timestamp_granularities[]", "word"),
                  ("timestamp_granularities[]", "segment"),
                  ("timestamp_granularities", "word"),
                  ("timestamp_granularities", "segment")]


def _part(name, value, filename=None):
    disp = f'form-data; name="{name}"'
    head = f"Content-Disposition: {disp}"
    if filename is not None:
        head += f'; filename="{filename}"\r\nContent-Type: audio/wav'
    body = value if isinstance(value, bytes) else value.encode()
    return b"--" + _BOUNDARY + b"\r\n" + head.encode() + b"\r\n\r\n" + body + b"\r\n"


def _multipart(fields):
    """A multipart body: the WAV file part, then each (name, value) field."""
    return (_part("file", _WAV, "a.wav")
            + b"".join(_part(n, v) for n, v in fields)
            + b"--" + _BOUNDARY + b"--\r\n")


def _padding(n):
    return [(f"pad{i}", "x") for i in range(n)]


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_a_body_with_too_many_parts_is_cut_off(client, monkeypatch, path):
    # The handler's first call: it must never run.
    calls = []
    real = tx_models._resolve_model_name
    monkeypatch.setattr(tx_models, "_resolve_model_name",
                        lambda name: calls.append(name) or real(name))
    r = client.post(path, content=_multipart(_padding(299)), headers=_CTYPE)
    # The receive channel reports a disconnect, which FastAPI answers as a
    # body parse error: the same answer the streaming byte cap gives.
    assert r.status_code == 400, r.text
    assert "parsing the body" in r.json()["detail"]
    assert calls == []


def test_the_cap_admits_exactly_max_upload_parts(client, app_module):
    # The file part, the model field, and padding up to the cap.
    cap = app_module._MAX_UPLOAD_PARTS
    ok = client.post("/v1/audio/transcriptions",
                     content=_multipart(_padding(cap - 2) + [("model", "whisper-1")]),
                     headers=_CTYPE)
    assert ok.status_code == 200, ok.text
    over = client.post("/v1/audio/transcriptions",
                       content=_multipart(_padding(cap - 1) + [("model", "whisper-1")]),
                       headers=_CTYPE)
    assert over.status_code == 400, over.text


@pytest.mark.parametrize("path", _UPLOAD_PATHS)
def test_an_upload_with_every_documented_field_still_succeeds(
        client, app_module, path):
    fields = list(_EVERY_FIELD.items()) + _GRANULARITIES
    if path.endswith("/translations"):
        fields = [(n, v) for n, v in fields if n != "task"]
    # Far below the cap: the largest legitimate request is ~35 parts.
    assert len(fields) + 1 < app_module._MAX_UPLOAD_PARTS // 4
    r = client.post(path, content=_multipart(fields), headers=_CTYPE)
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hallo welt"}


def test_a_regular_client_upload_is_untouched(client):
    assert post_json_ok(client) == {"text": "hallo welt"}


def _count_in_chunks(app_module, body, size):
    counter = app_module._MultipartDelimiterCounter(_BOUNDARY)
    for i in range(0, len(body), size):
        counter.feed(body[i:i + size])
    return counter.count


@pytest.mark.parametrize("size", [1, 2, 3, 7, len(_BOUNDARY) + 3,
                                  len(_BOUNDARY) + 4, 64, 1 << 20])
def test_delimiters_split_across_chunks_are_counted_once(app_module, size):
    body = _multipart(_padding(9))
    # 10 parts: their 10 opening delimiters plus the closing one.
    assert _count_in_chunks(app_module, body, size) == 11


def test_leading_crlf_before_the_first_delimiter_is_counted(app_module):
    # The parser skips leading CR/LF before the opening delimiter.
    body = b"\r\n\r\n" + _multipart(_padding(1))
    assert _count_in_chunks(app_module, body, 5) == 3
