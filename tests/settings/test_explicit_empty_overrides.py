"""An EXPLICITLY empty override ("cleared — overrides inherited" in the
client, i.e. the key present with "" / []) must beat the inherited value;
only an ABSENT key inherits. These pin every site where an explicit empty
used to be swallowed as "not set".
"""

import json

from faster_whisper_backend.settings import effective_config as ec
from tests.conftest import bearer
from faster_whisper_backend.transcription import models as tx_models

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


# --- per-request decode_overrides -------------------------------------------

def test_explicit_empty_suppress_tokens_means_suppress_nothing():
    for cleared in ([], ""):
        kw = tx_models._apply_decode_overrides(
            {"suppress_tokens": [-1, 50256]}, "whisper-1",
            {"suppress_tokens": cleared})
        assert kw["suppress_tokens"] is None, cleared
    # a NON-empty value that merely filters away is not a clear: config stays
    too_big = tx_models._SUPPRESS_TOKEN_ID_MAX + 1
    for junk in (str(too_big), [too_big], "abc"):
        kw = tx_models._apply_decode_overrides(
            {"suppress_tokens": [-1, 50256]}, "whisper-1",
            {"suppress_tokens": junk})
        assert kw["suppress_tokens"] == [-1, 50256], junk


def test_null_bool_override_inherits_instead_of_forcing_false():
    base = {"condition_on_previous_text": True, "vad_filter": True,
            "vad_parameters": {"threshold": 0.5}}
    kw = tx_models._apply_decode_overrides(dict(base), "whisper-1",
                                      {"condition_on_previous_text": None,
                                       "vad_filter": None})
    assert kw["condition_on_previous_text"] is True
    assert kw["vad_filter"] is True and kw["vad_parameters"] == {"threshold": 0.5}
    kw = tx_models._apply_decode_overrides(dict(base), "whisper-1",
                                      {"condition_on_previous_text": False,
                                       "vad_filter": False})
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False and kw["vad_parameters"] is None


def test_multilingual_override_true_null_and_locked():
    kw = tx_models._apply_decode_overrides({}, "whisper-1", {"multilingual": True})
    assert kw["multilingual"] is True
    # null inherits the configured value
    kw = tx_models._apply_decode_overrides({"multilingual": True}, "whisper-1",
                                      {"multilingual": None})
    assert kw["multilingual"] is True
    # a locked key is dropped: the admin value stands
    locked = ec.Resolved(locked_client_keys=frozenset({"multilingual"}))
    kw = tx_models._apply_decode_overrides({}, "whisper-1", {"multilingual": True},
                                      ident=locked)
    assert "multilingual" not in kw


# --- profile / per-model layer ------------------------------------------------

def _assemble(values, language="", overrides=None):
    return tx_models.assemble_transcribe_kwargs(
        None, None, language=language, temperature=0.0, vad_filter=False,
        vad_parameters=None, want_word_ts=False, initial_prompt=None,
        overrides=overrides, ident=ec.Resolved(values=values))


def test_multilingual_applies_to_auto_detect_only():
    assert _assemble({}, overrides={"multilingual": True})["multilingual"] is True
    assert _assemble({"MULTILINGUAL": True})["multilingual"] is True
    # A chosen language wins: faster-whisper would re-detect per window.
    assert "multilingual" not in _assemble(
        {}, language="de", overrides={"multilingual": True})
    assert "multilingual" not in _assemble({"MULTILINGUAL": True}, language="de")


def test_profile_blank_punctuation_is_forwarded_not_dropped():
    kw = _assemble({"PREPEND_PUNCTUATIONS": "", "APPEND_PUNCTUATIONS": ""})
    assert kw["prepend_punctuations"] == ""
    assert kw["append_punctuations"] == ""


def test_profile_blank_suppress_tokens_keeps_chars_without_the_default_set(monkeypatch):
    monkeypatch.setattr(tx_models, "_resolve_suppress_chars", lambda *a: [7, 8])
    # cleared list + configured chars → only the chars, no -1 default set
    kw = _assemble({"SUPPRESS_TOKENS": "", "SUPPRESS_CHARS": "."})
    assert kw["suppress_tokens"] == [7, 8]
    # an unset list still gets the -1 default set merged in
    kw = _assemble({"SUPPRESS_CHARS": "."})
    assert -1 in kw["suppress_tokens"] and {7, 8} <= set(kw["suppress_tokens"])


def test_suppress_chars_survive_a_client_suppress_tokens_override(monkeypatch):
    # The chars merge runs AFTER the client override, which used to replace
    # the merged ids wholesale.
    monkeypatch.setattr(tx_models, "_resolve_suppress_chars", lambda *a: [7, 8])
    kw = _assemble({"SUPPRESS_CHARS": "."}, overrides={"suppress_tokens": "5,6"})
    assert kw["suppress_tokens"] == [5, 6, 7, 8]
    # A client clear means "no list": the chars only, no -1 default set.
    for cleared in ("", []):
        kw = _assemble({"SUPPRESS_CHARS": "."},
                       overrides={"suppress_tokens": cleared})
        assert kw["suppress_tokens"] == [7, 8], cleared
    # A client list over a cleared config list: the list plus the chars.
    kw = _assemble({"SUPPRESS_TOKENS": "", "SUPPRESS_CHARS": "."},
                   overrides={"suppress_tokens": [-1]})
    assert kw["suppress_tokens"] == [-1, 7, 8]


# --- batch form fields ---------------------------------------------------------

def _bind_profile(client, make_user_key, **fields):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {"p": fields}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {},
                                                   "profiles": ["p"], "locks": []}})
    assert r.status_code == 200, r.text
    return raw_alice


def test_empty_language_form_field_is_explicit_auto_detect(client, make_user_key, fake_model):
    raw = _bind_profile(client, make_user_key, DEFAULT_LANGUAGE="de")
    base = {"model": "whisper-1", "response_format": "verbose_json"}
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw), data=base)
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["language"] == "de"          # absent → inherit
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw),
                    data={**base, "language": ""})
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["language"] is None          # "" → auto-detect


def test_empty_translate_to_and_glossary_form_fields_override_the_profile(
        client, app_module, make_user_key, monkeypatch):
    from faster_whisper_backend.audio import translation
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True, raising=False)
    calls = []

    async def _fake(segments, targets, *, source_lang=None, model_ref=None,
                    mode="fluent", glossary="", context_segments=None,
                    progress_cb=None, cancel_check=None, download_cb=None):
        calls.append({"targets": list(targets), "glossary": glossary})
        return ([{t: "x" for t in targets} for _ in segments], [],
                {"model": "org/m-GGUF:Q4", "source": "", "mode": mode})
    monkeypatch.setattr(translation, "translate_segments", _fake)

    raw = _bind_profile(client, make_user_key, TRANSLATE_TO="en",
                        TRANSLATION_GLOSSARY="Messung = measurement")
    base = {"model": "whisper-1", "response_format": "verbose_json"}
    # absent → both inherited from the profile
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw), data=base)
    assert r.status_code == 200, r.text
    assert calls[-1] == {"targets": ["en"], "glossary": "Messung = measurement"}
    # "" glossary → explicitly none, targets still inherited
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw),
                    data={**base, "translation_glossary": ""})
    assert r.status_code == 200, r.text
    assert calls[-1] == {"targets": ["en"], "glossary": ""}
    # "" translate_to → explicitly no targets: the stage does not run at all
    n = len(calls)
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw),
                    data={**base, "translate_to": ""})
    assert r.status_code == 200, r.text
    assert len(calls) == n
    assert "translations" not in json.dumps(r.json().get("segments", [{}])[0])


# --- dictation handshake ----------------------------------------------------------

def test_stream_handshake_language_is_tri_state(monkeypatch):
    """Absent → inherit DEFAULT_LANGUAGE; present-but-empty → auto-detect."""
    from faster_whisper_backend.streaming import routes as streaming_routes
    src = __import__("inspect").getsource(streaming_routes)
    assert 'req_language = _req_language.strip() if isinstance(_req_language, str) else None' in src
    assert 'language if language is not None' in src


def test_client_suppress_chars_replace_config_and_empty_means_none(monkeypatch):
    seen = []

    def _resolve(model_id, model, chars, from_client=False):
        seen.append((chars, from_client))
        return [7] if chars == "." else [9]
    monkeypatch.setattr(tx_models, "_resolve_suppress_chars", _resolve)
    kw = _assemble({"SUPPRESS_CHARS": "."}, overrides={"suppress_chars": "?"})
    assert kw["suppress_tokens"] == [-1, 9] and seen[-1] == ("?", True)
    # "" is an explicit "no chars": nothing resolved, faster-whisper default.
    seen.clear()
    kw = _assemble({"SUPPRESS_CHARS": "."}, overrides={"suppress_chars": ""})
    assert kw.get("suppress_tokens") == _assemble({}).get("suppress_tokens")
    assert seen == []
    # Locked: the configured chars stand.
    locked = ec.Resolved(values={"SUPPRESS_CHARS": "."},
                         locked_client_keys=frozenset({"suppress_chars"}))
    kw = tx_models.assemble_transcribe_kwargs(
        None, None, language="", temperature=0.0, vad_filter=False,
        vad_parameters=None, want_word_ts=False, initial_prompt=None,
        overrides={"suppress_chars": "?"}, ident=locked)
    assert kw["suppress_tokens"] == [-1, 7]


def test_suppress_chars_cache_is_a_capped_lru(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tx_models, "_suppress_chars_cache", tx_models.OrderedDict())
    monkeypatch.setattr(tx_models, "_SUPPRESS_CHARS_CACHE_MAX", 3)
    tok = SimpleNamespace(encode=lambda v, add_special_tokens=False: [ord(v[-1])])
    model = SimpleNamespace(hf_tokenizer=tok)
    for ch in "abc":
        tx_models._resolve_suppress_chars("m", model, ch, True)
    tx_models._resolve_suppress_chars("m", model, "a", True)   # hit → most recent
    tx_models._resolve_suppress_chars("m", model, "d", True)   # evicts "b"
    assert list(tx_models._suppress_chars_cache) == [("m", "c"), ("m", "a"), ("m", "d")]
    tx_models._drop_suppress_chars_cache("m")
    assert not tx_models._suppress_chars_cache


def test_hallucination_silence_zero_is_off(monkeypatch):
    # faster-whisper treats only None as off; 0 from the config or the
    # client never reaches it.
    assert "hallucination_silence_threshold" not in _assemble(
        {"HALLUCINATION_SILENCE_THRESHOLD": 0.0})
    assert _assemble({"HALLUCINATION_SILENCE_THRESHOLD": 2.0})[
        "hallucination_silence_threshold"] == 2.0
    assert "hallucination_silence_threshold" not in _assemble(
        {"HALLUCINATION_SILENCE_THRESHOLD": 2.0},
        overrides={"hallucination_silence_threshold": 0})
    assert _assemble({}, overrides={"hallucination_silence_threshold": 99})[
        "hallucination_silence_threshold"] == 60.0     # clamped to the field max


def test_language_detection_keys_clamped_and_auto_detect_only():
    kw = _assemble({}, overrides={"language_detection_segments": 50,
                                  "language_detection_threshold": 0.8})
    assert kw["language_detection_segments"] == 10
    assert kw["language_detection_threshold"] == 0.8
    ignored = []
    tx_models._note_auto_detect_only({"language_detection_segments": 2,
                                 "language_detection_threshold": 0.8}, "de", ignored)
    assert ignored == ["language_detection_threshold", "language_detection_segments"]
    ignored = []
    tx_models._note_auto_detect_only({"language_detection_segments": 2}, "", ignored)
    assert ignored == []


def test_hallucination_threshold_needs_word_timestamps():
    ignored = []
    tx_models._note_word_ts_only({"hallucination_silence_threshold": 2.0}, False, ignored)
    assert ignored == ["hallucination_silence_threshold"]
    for ov, wts in (({"hallucination_silence_threshold": 2.0}, True),
                    ({"hallucination_silence_threshold": 0}, False), ({}, False)):
        ignored = []
        tx_models._note_word_ts_only(ov, wts, ignored)
        assert ignored == [], (ov, wts)
