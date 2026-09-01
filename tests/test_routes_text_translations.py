"""POST /v1/text/translations — standalone text→text translation of an
already-held transcript. llama_cpp is never imported: the tests stub
`translation.translate_segments` at the handler boundary (the
test_diarization pattern)."""

import logging

import pytest

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


def test_kept_original_surfaces_with_client_ids(client, app_module,
                                                monkeypatch):
    """Guard fallbacks are marked per segment via kept_original, and warning
    text references the CLIENT ids (deliberately non-sequential to prove the
    positional→id mapping), never the 1-based positions."""
    _enable(app_module, monkeypatch)

    async def _fake(segments, targets, **kwargs):
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, [
            "segment 2 (en): kept original — translation failed (length ratio)",
            "segments 2-3 (en): kept original — translation failed (empty output)",
        ], {"model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {1: ["en"], 2: ["en"]}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = client.post(URL, json={
        "segments": [{"id": 7, "text": "eins"}, {"id": 3, "text": "zwei"},
                     {"id": 42, "text": "drei"}],
        "targets": ["en"], "source": "de"})
    assert r.status_code == 200, r.text
    body = r.json()
    segs = body["segments"]
    assert "kept_original" not in segs[0]                # clean → absent
    assert "translations_kept" not in segs[0]
    # Both names carry the same fact: kept_original (this endpoint's original
    # key) and translations_kept (the batch endpoint's per-segment key).
    assert segs[1] == {"id": 3, "translations": {"en": "zwei"},
                       "kept_original": ["en"],
                       "translations_kept": ["en"]}
    assert segs[2]["kept_original"] == ["en"]
    # Positional "segment 2" → client id 3; group span → the member ids.
    assert any(w.startswith("segment 3 (en): kept original") for w
               in body["warnings"])
    assert any(w.startswith("segments 3, 42 (en): kept original") for w
               in body["warnings"])
    assert not any("segment 2" in w for w in body["warnings"])


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


def test_serial_translations_are_not_rate_limited(client, app_module,
                                                  monkeypatch):
    """One request at a time, over and over, is what the review UI does when a
    user retranslates line by line — below the per-minute ceiling it must
    never 429. The protection is the in-flight cap below; the per-minute
    counter is only a flood backstop (next test)."""
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    for _ in range(25):
        assert client.post(URL, json=_body()).status_code == 200


def test_per_minute_backstop_429s_and_releases_the_held_receipt(
        client, app_module, monkeypatch):
    """The per-minute counter is hit FIRST inside the release-on-reject
    bracket: over the ceiling the request 429s with the config field named,
    and a parked dictation receipt is handed back instead of left to the
    sweeper."""
    import receipt_hold

    _enable(app_module, monkeypatch, TRANSLATE_RATE_PER_MIN=2)
    _stub_translate(monkeypatch)
    window = app_module._text_translate_rate
    window._state.clear()
    try:
        for _ in range(2):
            assert client.post(URL, json=_body()).status_code == 200
        r = client.post(URL, json=_body())
        assert r.status_code == 429, r.text
        body = r.json()
        assert body["error"]["param"] == "TRANSLATE_RATE_PER_MIN"
        assert r.headers["Retry-After"] == str(body["error"]["retry_after"])

        receipt_hold.park("cap2", {"file_label": "utt#1", "model_name": "m",
                                   "raw": "r", "final": "f", "seg_diag": [],
                                   "kwargs": {}}, hold_s=90)
        r = client.post(URL, json=_body(captured_id="cap2"))
        assert r.status_code == 429
        assert receipt_hold.pending() == 0
    finally:
        window._state.clear()
        receipt_hold._reset_for_tests()


# --- in-flight cap -----------------------------------------------------------

def _open_key():
    """The identity a request from the open-mode `client` fixture resolves to
    — the synthetic admin's user_id."""
    import api_keys_store
    return api_keys_store.OPEN_MODE_USER["user_id"]


def test_inflight_cap_rejects_when_the_identity_is_full(client, app_module,
                                                        monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    gauge = app_module._translate_inflight
    limit = int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)
    for _ in range(limit):
        gauge.acquire(_open_key())

    r = client.post(URL, json=_body())
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["param"] == "TRANSLATE_MAX_INFLIGHT_PER_USER"
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["detail"] == body["error"]["message"]
    assert r.headers["Retry-After"] == str(body["error"]["retry_after"])
    # A refused request must not release a slot it never held.
    assert gauge.count(_open_key()) == limit

    gauge.clear()
    assert client.post(URL, json=_body()).status_code == 200


def test_inflight_slot_is_released_on_success(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    assert client.post(URL, json=_body()).status_code == 200
    assert app_module._translate_inflight._counts == {}


@pytest.mark.parametrize("make_exc,status", [
    (lambda: translation.TranslationError("nope"), 400),
    (lambda: translation.TranslationCancelled(), 499),
    (lambda: RuntimeError("boom"), 500),
])
def test_inflight_slot_is_released_on_every_error_path(
        client, app_module, monkeypatch, make_exc, status):
    _enable(app_module, monkeypatch)

    async def _boom(*args, **kwargs):
        raise make_exc()
    monkeypatch.setattr(translation, "translate_segments", _boom)
    assert client.post(URL, json=_body()).status_code == status
    assert app_module._translate_inflight._counts == {}


def test_inflight_slot_is_released_on_cancellation(client, app_module,
                                                   monkeypatch):
    """CancelledError is a BaseException, so every `except Exception` arm in
    the handler is skipped on a client disconnect — only the `finally` runs.
    That is why the release lives there and not in an except arm."""
    import asyncio

    _enable(app_module, monkeypatch)

    async def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(translation, "translate_segments", _cancelled)

    with pytest.raises(BaseException):
        client.post(URL, json=_body())
    assert app_module._translate_inflight._counts == {}


def test_inflight_zero_is_unlimited(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch, TRANSLATE_MAX_INFLIGHT_PER_USER=0)
    _stub_translate(monkeypatch)
    # A stale count from before the field was zeroed must not gate anything.
    app_module._translate_inflight._counts[_open_key()] = 99
    assert client.post(URL, json=_body()).status_code == 200


def test_inflight_cap_is_per_user(client, app_module, make_user_key,
                                  monkeypatch):
    """The loopback `client` fixture is OPEN MODE — one synthetic admin, one
    bucket — so real keys are what prove the gauge is keyed per identity."""
    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    uid_a, key_a = make_user_key("alice", is_admin=True)
    _uid_b, key_b = make_user_key("bob", is_admin=False)
    gauge = app_module._translate_inflight
    for _ in range(int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)):
        gauge.acquire(uid_a)

    assert client.post(URL, json=_body(),
                       headers=bearer(key_a)).status_code == 429
    assert client.post(URL, json=_body(),
                       headers=bearer(key_b)).status_code == 200


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
        {"segments": [{"id": i, "text": "x"} for i
                      in range(app_module._TEXT_TRANSLATE_MAX_SEGMENTS + 1)],
         "targets": ["en"]},                                  # over entry cap
    ]
    for case in cases:
        r = client.post(URL, json=case)
        assert r.status_code == 422, (case, r.status_code, r.text)
    assert "capped at" in r.json()["detail"]     # the last case: entry cap
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


def test_download_progress_is_published(client, app_module, monkeypatch):
    """A cold-model fetch reports bytes through download_cb: the progress
    entry flips to stage=downloading with a byte fraction + total, and the
    first real progress tick flips it back to translating."""
    _enable(app_module, monkeypatch)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, download_cb=None,
                    **kwargs):
        download_cb(0, 0)
        seen["unknown_total"] = dict(app_module._BATCH_PROGRESS.get(_PID)
                                     or {})
        download_cb(512, 2048)
        seen["download"] = dict(app_module._BATCH_PROGRESS.get(_PID) or {})
        progress_cb(0.0, "en 1/1", None)
        seen["after"] = dict(app_module._BATCH_PROGRESS.get(_PID) or {})
        per_seg = [{t: f"{seg['text']}-{t}" for t in targets}
                   for seg in segments]
        return per_seg, [], {"model": "org/d:Q4", "source": "", "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = client.post(URL, json=_body(progress_id=_PID))
    assert r.status_code == 200, r.text
    assert seen["unknown_total"].get("stage") == "downloading"
    assert seen["unknown_total"].get("progress") is None
    assert seen["download"].get("stage") == "downloading"
    assert seen["download"].get("progress") == 0.25
    assert seen["download"].get("total_bytes") == 2048
    assert seen["after"].get("stage") == "translating"


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


def test_inflight_refusal_releases_the_held_receipt(client, app_module,
                                                     monkeypatch):
    """The in-flight acquire sits OUTSIDE the handler's try, so its 429 never
    reached the `except HTTPException` release. The parked dictation receipt
    must still be released — not left for the sweeper to log 90 s later."""
    import receipt_hold

    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    gauge = app_module._translate_inflight
    for _ in range(int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)):
        gauge.acquire(_open_key())
    receipt_hold.park("cap1", {"file_label": "utt#1", "model_name": "m",
                               "raw": "r", "final": "f", "seg_diag": [],
                               "kwargs": {}}, hold_s=90)
    try:
        r = client.post(URL, json=_body(captured_id="cap1"))
        assert r.status_code == 429
        assert receipt_hold.pending() == 0
    finally:
        gauge.clear()
        receipt_hold._reset_for_tests()


def test_validation_reject_releases_the_held_receipt(client, app_module,
                                                     monkeypatch):
    """Every validation exit (422 shape, 413 size) runs inside the same
    release-on-reject bracket as the rate hit: a parked receipt must be
    handed back on a malformed request, not left for the 90 s sweeper."""
    import receipt_hold

    _enable(app_module, monkeypatch)
    _stub_translate(monkeypatch)
    payload = {"file_label": "utt#1", "model_name": "m", "raw": "r",
               "final": "f", "seg_diag": [], "kwargs": {}}
    try:
        receipt_hold.park("cap-v", payload, hold_s=90)
        r = client.post(URL, json={"segments": [], "targets": ["en"],
                                   "captured_id": "cap-v"})
        assert r.status_code == 422, r.text
        assert receipt_hold.pending() == 0

        receipt_hold.park("cap-s", payload, hold_s=90)
        r = client.post(URL, json=_body(
            segments=[{"id": 0, "text": "x" * 200_001}], captured_id="cap-s"))
        assert r.status_code == 413, r.text
        assert receipt_hold.pending() == 0
    finally:
        receipt_hold._reset_for_tests()


def test_admin_pinned_model_passes_the_allowlist_gate(
        client, app_module, make_user_key, monkeypatch):
    """The allowlist constrains only the CLIENT-requested value (the
    diarization/separation stance): an admin who pins TRANSLATION_MODEL in a
    profile need not also add it to the global allowlist — the pinned
    identity used to 400 on every request even with no translation_model
    sent at all."""
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/public-GGUF:Q4"}, raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL",
                        "org/public-GGUF:Q4", raising=False)
    calls = []
    _stub_translate(monkeypatch, calls=calls)
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {
                        "pinned-mt": {
                            "TRANSLATION_MODEL": "org/pinned-GGUF:Q4",
                            "locks": ["TRANSLATION_MODEL"]}}})
    assert r.status_code == 200, r.text
    uid, raw_bob = make_user_key("bob", is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": ["pinned-mt"], "locks": []}})
    assert r.status_code == 200, r.text

    # No translation_model in the request: the pinned (admin-policy) model
    # resolves and passes even though it is not on the allowlist.
    r = client.post(
        "/v1/text/translations", headers=bearer(raw_bob),
        json={"segments": [{"id": 0, "text": "Hallo"}], "targets": ["en"]})
    assert r.status_code == 200, r.text
    assert calls[0]["model_ref"] == "org/pinned-GGUF:Q4"

    # A CLIENT asking for a non-allowlisted model is still refused.
    r = client.post(
        "/v1/text/translations", headers=bearer(raw_admin),
        json={"segments": [{"id": 0, "text": "Hallo"}], "targets": ["en"],
              "translation_model": "org/evil-GGUF:Q8"})
    assert r.status_code == 400, r.text


def test_translation_error_records_status_error(client, app_module,
                                                monkeypatch):
    # The terminal-error status is "error" — the batch handler's spelling
    # and metrics.py's default — never "failed": both land in the same
    # recent_transcriptions.status column rendered verbatim by /stats.
    _enable(app_module, monkeypatch)

    async def _boom(*a, **kw):
        raise translation.TranslationError("nope")
    monkeypatch.setattr(translation, "translate_segments", _boom)
    recorded = []
    _orig = app_module.metrics.record_transcription

    def _spy(**kw):
        recorded.append(kw)
        return _orig(**kw)
    monkeypatch.setattr(app_module.metrics, "record_transcription", _spy)
    r = client.post(URL, json=_body())
    assert r.status_code == 400, r.text
    assert recorded and recorded[-1]["status"] == "error"
    assert recorded[-1]["kind"] == "translate"
    assert "key_label" in recorded[-1]
    import transcriptions_store
    rows = transcriptions_store.list_recent(limit=5)
    assert rows and rows[0]["status"] == "error"


def test_empty_translation_allowlist_admits_any_well_formed_ref(app_module,
                                                                monkeypatch):
    # Documented (config_store TRANSLATION_ALLOWED_MODELS help): an EMPTY
    # allowlist is permissive — unlike the diarization/separation gates.
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    assert app_module._translation_model_allowed(
        "someone/other:Q4", requested="someone/other:Q4") is True
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/a:Q4"}, raising=False)
    assert app_module._translation_model_allowed(
        "someone/other:Q4", requested="someone/other:Q4") is False
