"""POST /v1/text/translations — standalone text→text translation of an
already-held transcript. llama_cpp is never imported: the tests stub
`translation.translate_segments` at the handler boundary (the
test_diarization pattern)."""

import logging

import translation
from tests.conftest import bearer

URL = "/v1/text/translations"
_PID = "feed" * 8  # 32 hex chars — passes _PROGRESS_ID_RE


def _stub_translate(monkeypatch, calls=None):
    """Per-index-verifiable stub: segment i translates to '<text>-<target>'."""
    async def _fake(segments, targets, *, source_lang=None, model_ref=None,
                    mode="fluent", glossary="", context_segments=None,
                    progress_cb=None, cancel_check=None, download_cb=None):
        if calls is not None:
            calls.append({"segments": segments, "targets": list(targets),
                          "source_lang": source_lang, "model_ref": model_ref,
                          "mode": mode, "glossary": glossary,
                          "context_segments": context_segments})
        per_seg = [{t: f"{seg['text']}-{t}" for t in targets}
                   for seg in segments]
        return per_seg, [], {"model": (model_ref or "").strip() or "org/d:Q4",
                             "source": source_lang or "", "mode": mode}
    monkeypatch.setattr(translation, "translate_segments", _fake)


def _enable(app_module, monkeypatch, **cfg_fields):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    for name, value in cfg_fields.items():
        monkeypatch.setattr(app_module.cfg, name, value, raising=False)


def _body(**overrides):
    body = {"segments": [{"id": 7, "text": "eins", "speaker": "SPEAKER_00"},
                         {"id": 3, "text": "zwei"}],
            "targets": ["en"]}
    body.update(overrides)
    return body


# --- happy path --------------------------------------------------------------

def test_translates_and_echoes_ids_in_input_order(client, app_module,
                                                  monkeypatch):
    _enable(app_module, monkeypatch)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    r = client.post(URL, json=_body(targets=["en", "fr"], source="de",
                                    translation_mode="faithful"))
    assert r.status_code == 200, r.text
    body = r.json()
    # id echo + per-index alignment (ids deliberately out of order).
    assert body["segments"] == [
        {"id": 7, "translations": {"en": "eins-en", "fr": "eins-fr"}},
        {"id": 3, "translations": {"en": "zwei-en", "fr": "zwei-fr"}},
    ]
    assert body["translation"]["targets"] == ["en", "fr"]
    assert body["translation"]["mode"] == "faithful"
    assert body["warnings"] == []
    # The stub saw texts + speakers and the request's source language.
    assert calls[0]["segments"][0] == {"text": "eins", "speaker": "SPEAKER_00"}
    assert calls[0]["segments"][1] == {"text": "zwei", "speaker": None}
    assert calls[0]["source_lang"] == "de"
    assert calls[0]["mode"] == "faithful"


def test_segment_without_id_echoes_its_index(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    r = client.post(URL, json={"segments": [{"text": "a"}, {"text": "b"}],
                               "targets": ["en"]})
    assert r.status_code == 200, r.text
    assert [s["id"] for s in r.json()["segments"]] == [0, 1]


# --- gates -------------------------------------------------------------------

def test_401_when_locked_down(client, app_module, make_user_key, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    _, raw_admin = make_user_key("admin", is_admin=True)   # flips to lockdown
    assert client.post(URL, json=_body()).status_code == 401
    # ...and a plain (non-admin) bearer passes: no host gate, no page perm.
    _, raw_user = make_user_key("worker", is_admin=False)
    r = client.post(URL, json=_body(), headers=bearer(raw_user))
    assert r.status_code == 200, r.text


def test_403_when_translation_disabled(client, app_module, monkeypatch):
    # TRANSLATION_ENABLED defaults off.
    _stub_translate(monkeypatch)
    r = client.post(URL, json=_body())
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]


def test_429_after_rate_limit(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    for _ in range(app_module._TEXT_TRANSLATE_RATE_MAX):
        assert client.post(URL, json=_body()).status_code == 200
    r = client.post(URL, json=_body())
    assert r.status_code == 429
    assert "slow down" in r.json()["detail"]


def test_400_on_allowlist_miss(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch,
            TRANSLATION_DEFAULT_MODEL="org/default-GGUF:Q4",
            TRANSLATION_ALLOWED_MODELS={"org/allowed-GGUF:Q4"})
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    r = client.post(URL, json=_body(translation_model="org/other-GGUF:Q4"))
    assert r.status_code == 400
    assert "TRANSLATION_ALLOWED_MODELS" in r.json()["detail"]
    assert calls == []
    # The configured default and an allowlisted model both pass.
    assert client.post(URL, json=_body()).status_code == 200
    r = client.post(URL, json=_body(translation_model="org/allowed-GGUF:Q4"))
    assert r.status_code == 200


# --- shape validation --------------------------------------------------------

def test_413_over_total_char_cap(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    r = client.post(URL, json=_body(
        segments=[{"id": 0, "text": "x" * 200_001}]))
    assert r.status_code == 413


def test_422_malformed_shapes(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    cases = [
        {"segments": "not-a-list", "targets": ["en"]},
        {"segments": [], "targets": ["en"]},
        {"segments": [{"id": 0}], "targets": ["en"]},        # no text
        {"segments": [{"text": 5}], "targets": ["en"]},      # non-str text
        _body(targets=[]),                                    # empty targets
        _body(targets=["NOT A CODE"]),
        _body(targets=["en", "fr", "it", "es"]),              # over MAX (3)
        _body(translation_mode="poetic"),
        _body(context_segments="three"),
        "just a string",
    ]
    for case in cases:
        r = client.post(URL, json=case)
        assert r.status_code == 422, (case, r.status_code, r.text)
    assert calls == []


def test_translation_error_maps_to_400(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)

    async def _boom(*args, **kwargs):
        raise translation.TranslationError(
            "no translation model configured (TRANSLATION_DEFAULT_MODEL)")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    r = client.post(URL, json=_body())
    assert r.status_code == 400
    assert "TRANSLATION_DEFAULT_MODEL" in r.json()["detail"]


def test_unexpected_error_maps_to_generic_500(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)

    async def _boom(*args, **kwargs):
        raise RuntimeError("/secret/path/model.gguf exploded")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    r = client.post(URL, json=_body())
    assert r.status_code == 500
    assert r.json()["detail"] == "translation failed"     # never the raw text


# --- progress / cancel plumbing ----------------------------------------------

def test_progress_visible_and_cancel_honored(client, app_module, monkeypatch):
    # The stub runs mid-stage: it observes the live progress entry (proving
    # GET progress / POST cancel can see the id) and then flags the id in
    # _BATCH_CANCELLED — exactly what POST cancel/{id} does — before honoring
    # cancel_check. The request must abort with 499 and clean both registries.
    _enable(app_module, monkeypatch)
    seen = {}

    async def _slow(segments, targets, *, progress_cb=None, cancel_check=None,
                    **kwargs):
        seen["entry"] = dict(app_module._BATCH_PROGRESS.get(_PID) or {})
        app_module._BATCH_CANCELLED.add(_PID)
        if cancel_check():
            raise translation.TranslationCancelled()
        raise AssertionError("cancel_check ignored the flagged id")
    monkeypatch.setattr(translation, "translate_segments", _slow)

    r = client.post(URL, json=_body(progress_id=_PID))
    assert r.status_code == 499, r.text
    assert seen["entry"].get("stage") == "translating"
    assert seen["entry"].get("progress") == 0.0
    # The finally cleaned both registries.
    assert _PID not in app_module._BATCH_PROGRESS
    assert _PID not in app_module._BATCH_CANCELLED


def test_progress_wrapper_forwards_last_text_and_logs_receipts(
        client, app_module, monkeypatch, caplog):
    """The handler's progress wrapper merges last_text into the progress
    entry, and the run brackets itself with start + ✓ done log lines."""
    _enable(app_module, monkeypatch)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        progress_cb(0.5, "en 1/2", "Hello there")
        seen["entry"] = dict(app_module._BATCH_PROGRESS.get(_PID) or {})
        per_seg = [{t: f"{seg['text']}-{t}" for t in targets}
                   for seg in segments]
        return per_seg, [], {"model": "org/d:Q4", "source": "", "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = client.post(URL, json=_body(progress_id=_PID))
    assert r.status_code == 200, r.text
    assert seen["entry"].get("last_text") == "Hello there"
    assert seen["entry"].get("progress") == 0.5
    assert seen["entry"].get("step") == "en 1/2"
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("[translate] req=" in m and "start: 2 segments × 1 targets"
               in m for m in msgs)
    assert any("✓ done" in m and "2 segs → en" in m for m in msgs)


def test_progress_endpoint_serves_last_text(client, app_module):
    """The GET progress endpoint already whitelists last_text — verify."""
    app_module._progress_set(_PID, stage="translating", last_text="the tail")
    try:
        r = client.get(f"/v1/audio/transcriptions/progress/{_PID}")
        assert r.status_code == 200
        assert r.json()["last_text"] == "the tail"
    finally:
        app_module._BATCH_PROGRESS.pop(_PID, None)


def test_failure_logs_terminal_line(client, app_module, monkeypatch, caplog):
    _enable(app_module, monkeypatch)

    async def _boom(*args, **kwargs):
        raise translation.TranslationError("no translation model configured")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = client.post(URL, json=_body())
    assert r.status_code == 400
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("✗ failed" in m for m in msgs)


def test_locked_translation_model_applies_to_the_text_endpoint(
        client, app_module, make_user_key, monkeypatch):
    """Per-identity locks bind HERE too — otherwise a key locked to a small
    model on the batch path just switches endpoints (review finding)."""
    from tests.conftest import bearer
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    # Open allowlist: this test is about the lock, not the allowlist gate
    # (the shipped config now allowlists two real models by default).
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {
                        "small-mt": {
                            "TRANSLATION_MODEL": "org/small:Q4_K_M",
                            "locks": ["TRANSLATION_MODEL"]}}})
    assert r.status_code == 200, r.text
    uid, raw_bob = make_user_key("bob", is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": ["small-mt"], "locks": []}})
    assert r.status_code == 200, r.text

    r = client.post(
        "/v1/text/translations", headers=bearer(raw_bob),
        json={"segments": [{"id": 0, "text": "Hallo"}], "targets": ["en"],
              "translation_model": "org/huge:Q8_0"})
    assert r.status_code == 200, r.text
    assert calls[0]["model_ref"] == "org/small:Q4_K_M"  # locked value wins
    assert any("locked" in w for w in r.json().get("warnings", []))
