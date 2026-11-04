"""Translation stage on POST /v1/audio/transcriptions (soft-fail contract).
llama_cpp is never imported — `translation.translate_segments` is
monkeypatched at the exact boundary the handler uses (the test_diarization
`_stub_turns` pattern)."""

import json

from faster_whisper_backend.translation import engine as translation
from faster_whisper_backend.transcription import progress as tx_progress
from tests._transcription_helpers import WAV_UPLOAD as _FILE
from tests._transcription_helpers import bind_profile
from tests._transcription_helpers import post_verbose as _post
from tests._transcription_helpers import post_verbose_ok as _post_ok
from tests._transcription_helpers import watch_progress
from tests._translation_helpers import enable_translation
from tests.conftest import bearer

_PID = "beef" * 8  # 32 hex chars — passes _PROGRESS_ID_RE


def _stub_translate(monkeypatch):
    """Deterministic translate_segments stub: every segment translates to
    'XLATED-<target>'. Returns the list recording the handler-side call shape."""
    calls = []

    async def _fake(segments, targets, *, source_lang=None, model_ref=None,
                    mode="fluent", glossary="", context_segments=None,
                    progress_cb=None, cancel_check=None,
                    download_cb=None):
        calls.append({"segments": segments, "targets": list(targets),
                      "source_lang": source_lang, "model_ref": model_ref,
                      "mode": mode, "glossary": glossary,
                      "context_segments": context_segments})
        per_seg = [{t: f"XLATED-{t}" for t in targets} for _ in segments]
        return per_seg, [], {
            "model": (model_ref or "").strip() or "org/default-GGUF:Q4",
            "source": source_lang or "", "mode": mode}
    monkeypatch.setattr(translation, "translate_segments", _fake)
    return calls


def test_translation_stage_populates_segments_and_response(
        client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)
    calls = _stub_translate(monkeypatch)
    body = _post_ok(client, translate_to="en,fr")
    assert body["segments"][0]["translations"] == {"en": "XLATED-en",
                                                   "fr": "XLATED-fr"}
    assert body["translations"] == {"en": "XLATED-en", "fr": "XLATED-fr"}
    assert body["translation"]["targets"] == ["en", "fr"]
    assert body["translation"]["mode"] == "fluent"
    assert "warnings" not in body
    assert calls[0]["segments"] == [{"text": "hallo welt", "speaker": None}]
    assert calls[0]["source_lang"] == "de"
    assert "XLATED" not in body["text"]


def test_translations_kept_marks_guard_fallback_segments(
        client, app_module, monkeypatch):
    """A kept-original segment carries translations_kept in verbose_json."""
    enable_translation(app_module, monkeypatch)

    async def _fake(segments, targets, **kwargs):
        # Guard fallback: the SOURCE text under every target.
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, ["segment 1: kept original — translation failed "
                         "(length ratio)"], {
            "model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {0: list(targets)}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    seg = _post_ok(client, translate_to="en")["segments"][0]
    assert seg["translations"] == {"en": "hallo welt"}
    assert seg["translations_kept"] == ["en"]


def test_untranslated_segment_gets_explicit_empty_translations(
        client, app_module, monkeypatch):
    """When targets were requested, every segment carries a translations map
    — an untranslated one an explicit empty dict, never a missing key (and
    no translations_kept when nothing was kept)."""
    enable_translation(app_module, monkeypatch)

    async def _fake(segments, targets, **kwargs):
        return [{} for _ in segments], [], {
            "model": "org/d:Q4", "source": "de", "mode": "fluent", "kept": {}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    seg = _post_ok(client, translate_to="en")["segments"][0]
    assert seg["translations"] == {}
    assert "translations_kept" not in seg


def test_translation_disabled_soft_fails_with_progress_skip(
        client, app_module, monkeypatch):
    # TRANSLATION_ENABLED defaults off: the request still succeeds, no
    # translation runs, the warning explains why, and the progress entry
    # names the skipped stage the moment it is declined.
    calls = _stub_translate(monkeypatch)
    seen = watch_progress(monkeypatch, _PID)
    body = _post_ok(client, translate_to="en", progress_id=_PID)
    assert calls == []
    assert "translations" not in body
    assert any("TRANSLATION_ENABLED is off" in w for w in body["warnings"])
    assert seen.get("skipped") == ["translating"]


def test_translation_allowlist_miss_soft_fails(client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL",
                        "org/default-GGUF:Q4", raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/allowed-GGUF:Q4"}, raising=False)
    calls = _stub_translate(monkeypatch)

    body = _post_ok(client, translate_to="en", translation_model="org/other-GGUF:Q4")
    assert calls == []
    assert "translations" not in body
    assert any("TRANSLATION_ALLOWED_MODELS" in w for w in body["warnings"])

    # The configured default passes even when it is not in the allowlist
    # (ALLOWED_MODELS semantics), and an allowlisted request passes too.
    assert "translations" in _post_ok(client, translate_to="en")
    assert "translations" in _post_ok(client, translate_to="en",
                                      translation_model="org/allowed-GGUF:Q4")
    assert [c["model_ref"] for c in calls] == \
        ["org/default-GGUF:Q4", "org/allowed-GGUF:Q4"]


def test_translate_to_clamped_to_max_targets_with_warning(
        client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_MAX_TARGETS", 3)
    # Ask for 5 (+ a malformed entry and a duplicate, both dropped silently
    # BEFORE the clamp).
    calls = _stub_translate(monkeypatch)
    body = _post_ok(client, translate_to="en,NOT_A_CODE,fr,en,it,es,pt")
    assert calls[0]["targets"] == ["en", "fr", "it"]
    assert body["translation"]["targets"] == ["en", "fr", "it"]
    assert any("TRANSLATION_MAX_TARGETS" in w and "es, pt" in w
               for w in body["warnings"])


def test_locked_translate_to_ignores_client_param(client, app_module,
                                                  make_user_key, monkeypatch):
    enable_translation(app_module, monkeypatch)
    calls = _stub_translate(monkeypatch)
    _, raw_alice = bind_profile(
        client, make_user_key, "de-only",
        {"TRANSLATE_TO": "de", "locks": ["TRANSLATE_TO"]})

    body = _post_ok(client, bearer(raw_alice), translate_to="en")
    assert calls[0]["targets"] == ["de"]           # locked value wins
    assert "translate_to" in body["overrides_ignored"]


def test_translation_cancel_is_not_a_soft_fail(client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)

    async def _cancelled(*args, **kwargs):
        raise translation.TranslationCancelled()
    monkeypatch.setattr(translation, "translate_segments", _cancelled)
    r = _post(client, translate_to="en", progress_id=_PID)
    assert r.status_code == 499, r.text


def test_translation_error_becomes_warning(client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)

    async def _boom(*args, **kwargs):
        raise translation.TranslationError(
            "translation dependencies are not installed on this server — "
            "pip install -r requirements-translate.txt")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    body = _post_ok(client, translate_to="en")
    assert body["text"]                           # transcript survives
    assert "translations" not in body
    assert any("requirements-translate" in w for w in body["warnings"])


def test_invalid_translation_mode_is_422(client):
    r = _post(client, translate_to="en", translation_mode="poetic")
    assert r.status_code == 422
    assert "translation_mode" in r.json()["detail"]


def test_captures_never_store_translated_text(client, app_module, monkeypatch, created_captures):
    enable_translation(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True,
                        raising=False)
    _stub_translate(monkeypatch)

    assert _post_ok(client, translate_to="en")["translations"]["en"] == "XLATED-en"  # the stage ran
    assert created_captures, "capture was not persisted"
    stored = created_captures[-1]

    # The TRANSCRIPT fields carry the source language only: Whisper learns to
    # emit `final` / `text_for_training` for this audio, so a translation
    # leaking into them would teach it to translate when asked to transcribe.
    transcript_fields = {k: v for k, v in stored.items()
                         if k in ("raw", "final", "text_for_training",
                                  "segments", "words", "language")}
    assert "XLATED" not in json.dumps(transcript_fields, default=str)

    # Translations live only in their own keyed column, tagged with the model
    # and as machine output. Whisper's translate task targets English only, so
    # the exporter has to pick one language out of this map, which it could
    # not do from a joined blob.
    assert stored["translations"] == {"en": "XLATED-en"}
    assert stored["translation_source"] == "cascade-mt"
    assert stored["translation_model"]
    assert stored["task"] == "transcribe"   # the Whisper task that ran


def test_plain_json_carries_translations_and_warnings(client, app_module,
                                                      monkeypatch):
    """The default `json` shape must deliver the translation output (and the
    soft-fail warnings) too — a caller paying for the stage should not need
    verbose_json to see either. Additive keys, like source_media_id."""
    enable_translation(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    body = _post_ok(client, translate_to="en", response_format="json")
    assert body["translations"] == {"en": "XLATED-en"}
    assert body["translation"]["targets"] == ["en"]

    # Soft-fail (stage disabled): the warning reaches the plain-json caller.
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", False,
                        raising=False)
    body = _post_ok(client, translate_to="en", response_format="json")
    assert "translations" not in body
    assert any("TRANSLATION_ENABLED" in w for w in body["warnings"])


def test_translation_progress_forwards_target_fields_into_plan_units(
        client, app_module, monkeypatch):
    """The stage's progress callback carries `target` / `target_progress`;
    the handler mirrors them into the progress entry and the run plan's
    units follow along."""
    enable_translation(app_module, monkeypatch)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        progress_cb(0.5, "en 1/1", "Hallo", target="en", target_progress=1.0)
        progress_cb(0.6, "fr 1/2", "Salut", target="fr", target_progress=0.5)
        seen["entry"] = dict(tx_progress._BATCH_PROGRESS.get(_PID) or {})
        seen["snap"] = tx_progress._RUN_PLAN_BY_PID[_PID].snapshot()
        per_seg = [{t: f"XLATED-{t}" for t in targets} for _ in segments]
        return per_seg, [], {"model": "org/m:Q4", "source": "de",
                             "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _fake)
    body = _post_ok(client, translate_to="en,fr", progress_id=_PID)
    assert seen["entry"]["target"] == "fr"
    assert seen["entry"]["target_progress"] == 0.5
    tr = next(s for s in seen["snap"]["plan"] if s["stage"] == "translating")
    by = {u["target"]: u for u in tr["units"]}
    assert by["en"]["state"] == "done"
    assert by["fr"]["state"] == "running" and by["fr"]["progress"] == 0.5
    receipt = next(s for s in body["plan"] if s["stage"] == "translating")
    assert {u["state"] for u in receipt["units"]} == {"done"}
    assert receipt["compute"] == "fluent"


def test_context_segments_form_field_inherit_value_clamp(
        client, app_module, monkeypatch):
    enable_translation(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_CONTEXT_SEGMENTS", 3)
    calls = _stub_translate(monkeypatch)
    assert _post(client, translate_to="en").status_code == 200
    assert calls[-1]["context_segments"] == 3          # absent → inherit
    assert _post(client, translate_to="en", context_segments="0").status_code == 200
    assert calls[-1]["context_segments"] == 0          # 0 is a value, not "unset"
    assert _post(client, translate_to="en", context_segments="99").status_code == 200
    assert calls[-1]["context_segments"] == 10         # clamped to the field
    # /v1/audio/translations forwards it (an omitted argument would be the
    # Form FieldInfo, not None).
    r = client.post("/v1/audio/translations", files=_FILE,
                    data={"model": "whisper-1", "response_format": "verbose_json",
                          "translate_to": "fr", "context_segments": "5"})
    assert r.status_code == 200, r.text
    assert calls[-1]["context_segments"] == 5
    r = client.post("/v1/audio/translations", files=_FILE,
                    data={"model": "whisper-1", "translate_to": "fr"})
    assert r.status_code == 200, r.text
    assert calls[-1]["context_segments"] == 3


def test_locked_context_segments_ignores_client_value(
        client, app_module, make_user_key, monkeypatch):
    enable_translation(app_module, monkeypatch)
    calls = _stub_translate(monkeypatch)
    _, raw_alice = bind_profile(
        client, make_user_key, "ctx",
        {"TRANSLATION_CONTEXT_SEGMENTS": 1,
         "locks": ["TRANSLATION_CONTEXT_SEGMENTS"]})
    body = _post_ok(client, bearer(raw_alice), translate_to="en", context_segments="6")
    assert calls[-1]["context_segments"] == 1
    assert "context_segments" in body["overrides_ignored"]
