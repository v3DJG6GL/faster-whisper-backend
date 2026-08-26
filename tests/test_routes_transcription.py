"""Integration tests for POST /v1/audio/transcriptions.

Drives the real handler with the FakeModel injected via the harness. The
fake model ignores the uploaded bytes, so a tiny dummy WAV payload is fine.
"""

from conftest import FakeModel

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def test_default_json_returns_text_object(client):
    r = _post(client, response_format="json")
    assert r.status_code == 200
    body = r.json()
    assert body == {"text": "hallo welt"}


def test_text_format_returns_plain_string(client):
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # response_format="text" returns the bare string (JSON-encoded string body).
    assert r.json() == "hallo welt"


def test_verbose_json_shape(client):
    r = _post(client, response_format="verbose_json")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "de"
    assert body["duration"] == 1.0
    assert body["text"] == "hallo welt"
    assert isinstance(body["segments"], list) and body["segments"]
    seg = body["segments"][0]
    assert seg["text"] == "hallo welt"
    # verbose_json with no explicit granularities still asks for words
    # (include_words = response_format == "verbose_json" and not granularities),
    # and WORD_TIMESTAMPS_ENABLED defaults True, so words are present.
    assert "words" in body
    assert [w["word"] for w in body["words"]] == ["hallo", "welt"]


def test_srt_falls_through_to_text_object(client):
    # Documented non-OpenAI behavior: srt/vtt aren't special-cased, so they
    # fall through to the default {"text": ...} JSON shape.
    r = _post(client, response_format="srt")
    assert r.status_code == 200
    assert r.json() == {"text": "hallo welt"}


def test_vtt_falls_through_to_text_object(client):
    r = _post(client, response_format="vtt")
    assert r.status_code == 200
    assert r.json() == {"text": "hallo welt"}


def test_words_gated_by_config_disabled(client, app_module):
    # WORD_TIMESTAMPS_ENABLED=False => want_word_ts is False even when the
    # request asks for word granularity, so the model gets word_timestamps=False
    # and the FakeModel returns no words; verbose_json["words"] is empty.
    app_module.cfg.WORD_TIMESTAMPS_ENABLED = False
    r = client.post(
        "/v1/audio/transcriptions",
        files=_FILE,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )
    assert r.status_code == 200
    assert r.json().get("words") == []


def test_words_included_with_granularity_field(client, app_module):
    # Default config has WORD_TIMESTAMPS_ENABLED=True. Explicitly request word
    # granularity on a json response: include_words drives the response, but
    # the default json shape ({"text":...}) does not surface words. So assert
    # the model was actually asked for word_timestamps=True via fake_model.
    r = client.post(
        "/v1/audio/transcriptions",
        files=_FILE,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )
    assert r.status_code == 200
    assert [w["word"] for w in r.json()["words"]] == ["hallo", "welt"]


def test_model_transcribe_raises_returns_500(client, app_module, monkeypatch):
    class BoomModel(FakeModel):
        def transcribe(self, path, **kwargs):
            raise RuntimeError("decode blew up")

    async def _loader(name):
        return BoomModel()

    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)
    r = _post(client, response_format="json")
    assert r.status_code == 500


def test_output_prefix_suffix_wrap(client, app_module):
    app_module.cfg.OUTPUT_PREFIX = "[de] "
    app_module.cfg.OUTPUT_SUFFIX = " (end)"
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # Wrappers applied, then a defensive outer trim (leading/trailing spaces of
    # the *whole* string are stripped). Inner content keeps the wrapper text.
    assert r.json() == "[de] hallo welt (end)"


def test_missing_file_is_422(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 422


def test_prompt_sentinel_inherit_clear_value(client, fake_model, app_module):
    """B4: `prompt` is a present-vs-absent sentinel read from the RAW form (FastAPI
    coerces an empty Form field to its default, so the handler reads request.form()
    directly). Absent → inherit DEFAULT_PROMPT; present-but-empty "" → CLEAR (no
    initial_prompt); value → verbatim."""
    app_module.cfg.DEFAULT_PROMPT = "SERVER PROMPT"
    # absent → inherit DEFAULT_PROMPT
    _post(client, response_format="json")
    assert fake_model.last_kwargs["initial_prompt"] == "SERVER PROMPT"
    # value → verbatim
    _post(client, response_format="json", prompt="my terms")
    assert fake_model.last_kwargs["initial_prompt"] == "my terms"
    # explicit empty → CLEAR. httpx drops empty `data`/`files` values and FastAPI
    # coerces an empty Form field to its default, so hand-build the multipart body
    # to deliver a genuine present-but-empty `prompt` part (what reqwest sends).
    b = "----p12boundary"
    body = (
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\nRIFFxxxxWAVE\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n\r\n'
        f'--{b}--\r\n'
    ).encode()
    r = client.post("/v1/audio/transcriptions", content=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["initial_prompt"] is None


def test_deeply_nested_decode_overrides_is_ignored_not_500(client, fake_model):
    """json.loads raises RecursionError (a RuntimeError, NOT a ValueError) on a
    deeply nested value. Without it in the guard's tuple the malformed value
    escaped to the handler's generic `except Exception` and became a 500 plus a
    permanent err_count bump, breaking the "malformed → ignored" contract."""
    import metrics

    before = metrics.err_count["/v1/audio/transcriptions"]
    r = _post(client, response_format="json", decode_overrides="[" * 200_000)
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hallo welt"}
    # Ignored, so nothing from it reached the model, and no error was recorded.
    assert metrics.err_count["/v1/audio/transcriptions"] == before


def test_absurdly_long_filename_extension_does_not_crash(client):
    # The client filename feeds tempfile.NamedTemporaryFile's suffix; a 250-char
    # extension used to raise OSError(36, 'File name too long') → 500.
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a." + "x" * 250, b"RIFFxxxxWAVE", "audio/wav")},
        data={"model": "whisper-1", "response_format": "json"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hallo welt"}


def test_null_byte_in_filename_does_not_crash(client):
    # httpx will not emit a NUL in a filename parameter, so hand-build the body.
    # It used to reach tempfile as a suffix → ValueError: embedded null character.
    b = "----nulboundary"
    body = (
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.w\x00av"\r\n'
        f'Content-Type: audio/wav\r\n\r\nRIFFxxxxWAVE\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
        f'--{b}--\r\n'
    ).encode()
    r = client.post("/v1/audio/transcriptions", content=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hallo welt"}


def test_safe_tmp_suffix_keeps_ordinary_extensions(app_module):
    f = app_module._safe_tmp_suffix
    # A well-formed upload keeps exactly the suffix it always had.
    for name in ("a.wav", "clip.MP3", "x.m4a", "recording.ogg", "a.b.flac"):
        assert f(name) == "." + name.rsplit(".", 1)[1]
    # Screened out → no suffix, never an exception.
    assert f("a." + "x" * 250) == ""
    assert f("a.w\x00av") == ""
    assert f("a.wav\n") == ""
    assert f(None) == ""
    assert f("noext") == ""


def test_model_id_regex_rejects_a_trailing_newline(app_module):
    # `$` also matched just BEFORE a final newline; \Z does not.
    assert app_module._MODEL_ID_RE.match("some-repo")
    assert app_module._MODEL_ID_RE.match("org/some-repo")
    assert not app_module._MODEL_ID_RE.match("some-repo\n")
    assert not app_module._MODEL_ID_RE.match("org/some-repo\n")


# --- task (translate) --------------------------------------------------------

def test_task_translate_reaches_model_and_response(client, fake_model):
    r = _post(client, response_format="verbose_json", task="translate")
    assert r.status_code == 200
    assert fake_model.last_kwargs["task"] == "translate"
    assert r.json()["task"] == "translate"


def test_task_default_is_not_forwarded(client, fake_model):
    # The common path must stay byte-identical to pre-feature kwargs: no
    # `task` key at all, and the response echoes "transcribe".
    r = _post(client, response_format="verbose_json")
    assert r.status_code == 200
    assert "task" not in fake_model.last_kwargs
    assert r.json()["task"] == "transcribe"


def test_task_invalid_is_422(client):
    r = _post(client, task="summarize")
    assert r.status_code == 422


def test_translations_endpoint_pins_task(client, fake_model):
    r = client.post(
        "/v1/audio/translations", files=_FILE,
        data={"model": "whisper-1", "response_format": "verbose_json"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["task"] == "translate"
    assert r.json()["task"] == "translate"


def test_task_config_default_applies_when_field_absent(client, app_module, fake_model):
    # TASK config (global layer) is the default when the request has no task
    # field; an explicit field still wins.
    app_module.cfg.TASK = "translate"
    try:
        r = _post(client, response_format="verbose_json")
        assert r.status_code == 200
        assert fake_model.last_kwargs["task"] == "translate"
        r = _post(client, response_format="verbose_json", task="transcribe")
        assert r.status_code == 200
        assert "task" not in fake_model.last_kwargs
    finally:
        app_module.cfg.TASK = "transcribe"


# --- progress reporting ------------------------------------------------------

def test_progress_endpoint_unknown_id(client):
    r = client.get("/v1/audio/transcriptions/progress/" + "a" * 32)
    assert r.status_code == 200
    assert r.json() == {"stage": "unknown"}


def test_progress_endpoint_malformed_id_422(client):
    r = client.get("/v1/audio/transcriptions/progress/NOT-HEX!")
    assert r.status_code == 422


def test_progress_live_entry_and_cleanup(client, app_module):
    pid = "b" * 32
    app_module._progress_set(pid, stage="transcribing", progress=0.5, duration=60.0)
    r = client.get(f"/v1/audio/transcriptions/progress/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "transcribing"
    assert body["progress"] == 0.5
    assert body["duration"] == 60.0
    # A request that carries the same progress_id pops the entry when done.
    r = _post(client, response_format="verbose_json", progress_id=pid)
    assert r.status_code == 200
    assert client.get(f"/v1/audio/transcriptions/progress/{pid}").json() == {
        "stage": "unknown"}


def test_progress_updates_during_decode(client, app_module, fake_model):
    # The FakeModel yields segments through the handler's _collect wrapper —
    # snapshot the entry from inside transcribe() to see the live stage.
    pid = "c" * 32
    seen = {}
    orig = fake_model.transcribe

    def spy(path, **kwargs):
        segs, info = orig(path, **kwargs)
        entry = app_module._BATCH_PROGRESS.get(pid)
        seen.update(entry or {})
        return segs, info

    fake_model.transcribe = spy
    r = _post(client, response_format="verbose_json", progress_id=pid)
    assert r.status_code == 200
    # transcribe() ran after the "waiting" stage was registered.
    assert seen.get("stage") == "waiting"
    # ...and the entry is gone once the response is built.
    assert pid not in app_module._BATCH_PROGRESS


def test_progress_malformed_id_is_ignored_on_post(client, app_module):
    r = _post(client, response_format="verbose_json", progress_id="Nope!")
    assert r.status_code == 200
    assert "Nope!" not in app_module._BATCH_PROGRESS


def test_progress_registry_is_bounded(app_module):
    for i in range(app_module._BATCH_PROGRESS_MAX + 20):
        app_module._progress_set(f"{i:032x}", stage="waiting")
    assert len(app_module._BATCH_PROGRESS) <= app_module._BATCH_PROGRESS_MAX
    app_module._BATCH_PROGRESS.clear()
