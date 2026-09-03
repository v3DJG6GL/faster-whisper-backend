"""Translation stage on POST /v1/audio/transcriptions (soft-fail contract).
llama_cpp is never imported — `translation.translate_segments` is
monkeypatched at the exact boundary the handler uses (the test_diarization
`_stub_turns` pattern)."""

import json

import translation
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
_PID = "beef" * 8  # 32 hex chars — passes _PROGRESS_ID_RE


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_translate(monkeypatch, calls=None, warnings=None):
    """Deterministic translate_segments stub: every segment translates to
    'XLATED-<target>'. Records the handler-side call shape in `calls`."""
    async def _fake(segments, targets, *, source_lang=None, model_ref=None,
                    mode="fluent", glossary="", context_segments=None,
                    progress_cb=None, cancel_check=None,
                    download_cb=None):
        if calls is not None:
            calls.append({"segments": segments, "targets": list(targets),
                          "source_lang": source_lang, "model_ref": model_ref,
                          "mode": mode, "glossary": glossary,
                          "context_segments": context_segments})
        per_seg = [{t: f"XLATED-{t}" for t in targets} for _ in segments]
        return per_seg, list(warnings or []), {
            "model": (model_ref or "").strip() or "org/default-GGUF:Q4",
            "source": source_lang or "", "mode": mode}
    monkeypatch.setattr(translation, "translate_segments", _fake)


def _progress_spy(app_module, seen, pid):
    orig = app_module._progress_set

    def spy(p, **fields):
        orig(p, **fields)
        if p == pid:
            seen.update(app_module._BATCH_PROGRESS.get(pid) or {})

    return orig, spy


# --- route behaviour ---------------------------------------------------------

def test_translation_stage_populates_segments_and_response(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    r = _post(client, translate_to="en,fr")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["segments"][0]["translations"] == {"en": "XLATED-en",
                                                   "fr": "XLATED-fr"}
    assert body["translations"] == {"en": "XLATED-en", "fr": "XLATED-fr"}
    assert body["translation"]["targets"] == ["en", "fr"]
    assert body["translation"]["mode"] == "fluent"
    assert "warnings" not in body
    # The stage saw the raw segment texts + the detected source language.
    assert calls[0]["segments"] == [{"text": "hallo welt", "speaker": None}]
    assert calls[0]["source_lang"] == "de"
    # ...and the main transcript stays untranslated.
    assert "XLATED" not in body["text"]


def test_translations_kept_marks_guard_fallback_segments(
        client, app_module, monkeypatch):
    """A kept-original segment carries translations_kept in verbose_json."""
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _fake(segments, targets, **kwargs):
        # Guard fallback: the SOURCE text under every target.
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, ["segment 1: kept original — translation failed "
                         "(length ratio)"], {
            "model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {0: list(targets)}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    seg = r.json()["segments"][0]
    assert seg["translations"] == {"en": "hallo welt"}
    assert seg["translations_kept"] == ["en"]


def test_untranslated_segment_gets_explicit_empty_translations(
        client, app_module, monkeypatch):
    """When targets were requested, every segment carries a translations map
    — an untranslated one an explicit empty dict, never a missing key (and
    no translations_kept when nothing was kept)."""
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _fake(segments, targets, **kwargs):
        return [{} for _ in segments], [], {
            "model": "org/d:Q4", "source": "de", "mode": "fluent", "kept": {}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    seg = r.json()["segments"][0]
    assert seg["translations"] == {}
    assert "translations_kept" not in seg


def test_translation_disabled_soft_fails_with_progress_skip(
        client, app_module, monkeypatch):
    # TRANSLATION_ENABLED defaults off: the request still succeeds, no
    # translation runs, the warning explains why, and the progress entry
    # names the skipped stage the moment it is declined.
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    seen = {}
    orig, spy = _progress_spy(app_module, seen, _PID)
    app_module._progress_set = spy
    try:
        r = _post(client, translate_to="en", progress_id=_PID)
    finally:
        app_module._progress_set = orig
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert "translations" not in body
    assert any("TRANSLATION_ENABLED is off" in w for w in body["warnings"])
    assert seen.get("skipped") == ["translating"]


def test_translation_allowlist_miss_soft_fails(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL",
                        "org/default-GGUF:Q4", raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/allowed-GGUF:Q4"}, raising=False)
    calls = []
    _stub_translate(monkeypatch, calls=calls)

    r = _post(client, translate_to="en", translation_model="org/other-GGUF:Q4")
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert "translations" not in body
    assert any("TRANSLATION_ALLOWED_MODELS" in w for w in body["warnings"])

    # The configured default passes even when it is not in the allowlist
    # (ALLOWED_MODELS semantics), and an allowlisted request passes too.
    r = _post(client, translate_to="en")
    assert r.status_code == 200 and "translations" in r.json()
    r = _post(client, translate_to="en",
              translation_model="org/allowed-GGUF:Q4")
    assert r.status_code == 200 and "translations" in r.json()
    assert [c["model_ref"] for c in calls] == \
        ["org/default-GGUF:Q4", "org/allowed-GGUF:Q4"]


def test_translate_to_clamped_to_max_targets_with_warning(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    # Default TRANSLATION_MAX_TARGETS is 3; ask for 5 (+ a malformed entry
    # and a duplicate, both dropped silently BEFORE the clamp).
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    r = _post(client, translate_to="en,NOT_A_CODE,fr,en,it,es,pt")
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls[0]["targets"] == ["en", "fr", "it"]
    assert body["translation"]["targets"] == ["en", "fr", "it"]
    assert any("TRANSLATION_MAX_TARGETS" in w and "es, pt" in w
               for w in body["warnings"])


def test_locked_translate_to_ignores_client_param(client, app_module,
                                                  make_user_key, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {
                        "de-only": {"TRANSLATE_TO": "de",
                                    "locks": ["TRANSLATE_TO"]}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": ["de-only"], "locks": []}})
    assert r.status_code == 200, r.text

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "translate_to": "en"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls[0]["targets"] == ["de"]           # locked value wins
    assert "translate_to" in body["overrides_ignored"]


def test_translation_cancel_is_not_a_soft_fail(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _cancelled(*args, **kwargs):
        raise translation.TranslationCancelled()
    monkeypatch.setattr(translation, "translate_segments", _cancelled)
    r = _post(client, translate_to="en", progress_id=_PID)
    assert r.status_code == 499, r.text


def test_translation_error_becomes_warning(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _boom(*args, **kwargs):
        raise translation.TranslationError(
            "translation dependencies are not installed on this server — "
            "pip install -r requirements-translate.txt")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]                           # transcript survives
    assert "translations" not in body
    assert any("requirements-translate" in w for w in body["warnings"])


def test_invalid_translation_mode_is_422(client):
    r = _post(client, translate_to="en", translation_mode="poetic")
    assert r.status_code == 422
    assert "translation_mode" in r.json()["detail"]


def test_captures_never_store_translated_text(client, app_module, monkeypatch):
    # Regression guard for the stage's CRITICAL invariant: the capture row
    # (raw/final/training text + segment diag) must carry only the
    # source-language transcript, never the stage's translations.
    import captures_store
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True,
                        raising=False)
    _stub_translate(monkeypatch)
    stored = {}

    def _spy_create(**kw):
        stored.update(kw)
        return "cap-test-id"
    monkeypatch.setattr(captures_store, "create_capture", _spy_create)

    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    assert r.json()["translations"]["en"] == "XLATED-en"  # the stage ran
    assert stored, "capture was not persisted"

    # The invariant this test was written for, and the one that still holds:
    # the TRANSCRIPT fields carry the source language and nothing else.
    # Whisper learns to emit `final` / `text_for_training` for this audio, so
    # a translation leaking into them would teach it to translate when it was
    # asked to transcribe.
    transcript_fields = {k: v for k, v in stored.items()
                         if k in ("raw", "final", "text_for_training",
                                  "segments", "words", "language")}
    assert "XLATED" not in json.dumps(transcript_fields, default=str)

    # Translations ARE kept now — but only in their own keyed column, tagged
    # with the model and with the fact that they are machine output. Whisper's
    # translate task targets English only, so the exporter has to pick one
    # language out of this map, which it could not do from a joined blob.
    assert stored["translations"] == {"en": "XLATED-en"}
    assert stored["translation_source"] == "cascade-mt"
    assert stored["translation_model"]
    assert stored["task"] == "transcribe"   # the Whisper task that ran


def test_plain_json_carries_translations_and_warnings(client, app_module,
                                                      monkeypatch):
    """The default `json` shape must deliver the translation output (and the
    soft-fail warnings) too — a caller paying for the stage should not need
    verbose_json to see either. Additive keys, like source_media_id."""
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    _stub_translate(monkeypatch)
    r = _post(client, translate_to="en", response_format="json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["translations"] == {"en": "XLATED-en"}
    assert body["translation"]["targets"] == ["en"]

    # Soft-fail (stage disabled): the warning reaches the plain-json caller.
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", False,
                        raising=False)
    r = _post(client, translate_to="en", response_format="json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "translations" not in body
    assert any("TRANSLATION_ENABLED" in w for w in body["warnings"])
