"""An EXPLICITLY empty override ("cleared — overrides inherited" in the
client, i.e. the key present with "" / []) must beat the inherited value;
only an ABSENT key inherits. These pin every site where an explicit empty
used to be swallowed as "not set".
"""

import json

import effective_config as ec
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


# --- per-request decode_overrides -------------------------------------------

def test_explicit_empty_suppress_tokens_means_suppress_nothing():
    import main
    for cleared in ([], ""):
        kw = main._apply_decode_overrides(
            {"suppress_tokens": [-1, 50256]}, "whisper-1",
            {"suppress_tokens": cleared})
        assert kw["suppress_tokens"] is None, cleared
    # a NON-empty value that merely filters away is not a clear: config stays
    too_big = main._SUPPRESS_TOKEN_ID_MAX + 1
    for junk in (str(too_big), [too_big], "abc"):
        kw = main._apply_decode_overrides(
            {"suppress_tokens": [-1, 50256]}, "whisper-1",
            {"suppress_tokens": junk})
        assert kw["suppress_tokens"] == [-1, 50256], junk


def test_null_bool_override_inherits_instead_of_forcing_false():
    import main
    base = {"condition_on_previous_text": True, "vad_filter": True,
            "vad_parameters": {"threshold": 0.5}}
    kw = main._apply_decode_overrides(dict(base), "whisper-1",
                                      {"condition_on_previous_text": None,
                                       "vad_filter": None})
    assert kw["condition_on_previous_text"] is True
    assert kw["vad_filter"] is True and kw["vad_parameters"] == {"threshold": 0.5}
    kw = main._apply_decode_overrides(dict(base), "whisper-1",
                                      {"condition_on_previous_text": False,
                                       "vad_filter": False})
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False and kw["vad_parameters"] is None


# --- profile / per-model layer ------------------------------------------------

def _assemble(values):
    import main
    return main.assemble_transcribe_kwargs(
        None, None, language="", temperature=0.0, vad_filter=False,
        vad_parameters=None, want_word_ts=False, initial_prompt=None,
        ident=ec.Resolved(values=values))


def test_profile_blank_punctuation_is_forwarded_not_dropped():
    kw = _assemble({"PREPEND_PUNCTUATIONS": "", "APPEND_PUNCTUATIONS": ""})
    assert kw["prepend_punctuations"] == ""
    assert kw["append_punctuations"] == ""


def test_profile_blank_suppress_tokens_keeps_chars_without_the_default_set(monkeypatch):
    import main
    monkeypatch.setattr(main, "_resolve_suppress_chars", lambda *a: [7, 8])
    # cleared list + configured chars → only the chars, no -1 default set
    kw = _assemble({"SUPPRESS_TOKENS": "", "SUPPRESS_CHARS": "."})
    assert kw["suppress_tokens"] == [7, 8]
    # an unset list still gets the -1 default set merged in
    kw = _assemble({"SUPPRESS_CHARS": "."})
    assert -1 in kw["suppress_tokens"] and {7, 8} <= set(kw["suppress_tokens"])


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
    import translation
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
    import streaming_routes
    src = __import__("inspect").getsource(streaming_routes)
    assert 'req_language = _req_language.strip() if isinstance(_req_language, str) else None' in src
    assert 'language if language is not None' in src
