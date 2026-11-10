"""POST /v1/text/translations — standalone text→text translation of an
already-held transcript. llama_cpp is never imported: the tests stub
`translation.translate_segments` at the handler boundary."""

import asyncio
import contextlib
import logging
import re

import pytest

from faster_whisper_backend.translation import engine as translation
from faster_whisper_backend.translation import gating as tr_gating
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.transcription import receipt_hold
from tests._transcription_helpers import bind_profile, record_runs
from tests._translation_helpers import (
    enable_translation as _enable,
    stub_translate as _stub_translate,
    text_translation_body as _body,
)
from tests.conftest import bearer, on_event_loop, record_calls
from faster_whisper_backend.translation import routes as tr_routes

URL = "/v1/text/translations"
_PID = "feed" * 8  # 32 hex chars — passes _PROGRESS_ID_RE


def _park(key, *, hold_s, **extra):
    """Park a dictation receipt the way the transcribe route does — one
    entry shape for every test, so a new required key lands in one place."""
    payload = {"file_label": "utt#1", "model_name": "m", "raw": "r",
               "final": "f", "seg_diag": [], "kwargs": {}, "info": None}
    payload.update(extra)
    receipt_hold.park(key, payload, hold_s=hold_s)


def _enable_stubbed(app_module, monkeypatch, **cfg_fields):
    """Translation on, translator stubbed; returns the stub's call record."""
    _enable(app_module, monkeypatch, **cfg_fields)
    return _stub_translate(monkeypatch)


def _post_status(client, status, headers=None, **body):
    """POST a text-translation body that must answer ``status``; returns the response."""
    r = client.post(URL, json=_body(**body), headers=headers)
    assert r.status_code == status, r.text
    return r


def _post_ok(client, headers=None, **body):
    """POST a text-translation body that must answer 200; returns the response."""
    return _post_status(client, 200, headers, **body)


def _assert_parked_receipt_released(client, key, status, **body):
    """A request naming parked receipt ``key`` is refused with ``status``; the
    receipt is handed back, not left for the sweeper."""
    _park(key, hold_s=90)
    _post_status(client, status, captured_id=key, **body)
    assert receipt_hold.pending() == 0


def _echo_result(segments, targets):
    """The '<text>-<target>' success tuple a hand-written fake returns."""
    per_seg = [{t: f"{seg['text']}-{t}" for t in targets} for seg in segments]
    return per_seg, [], {"model": "org/d:Q4", "source": "", "mode": "fluent"}


def _stub_raising(monkeypatch, exc):
    """translate_segments stub that raises ``exc`` on its call."""
    async def _boom(*args, **kwargs):
        raise exc
    monkeypatch.setattr(translation, "translate_segments", _boom)


def test_translates_and_echoes_ids_in_input_order(client, app_module,
                                                  monkeypatch):
    calls = _enable_stubbed(app_module, monkeypatch)
    r = _post_ok(client, targets=["en", "fr"], source="de",
                 translation_mode="faithful")
    body = r.json()
    # id echo + per-index alignment (ids deliberately out of order).
    assert body["segments"] == [
        {"id": 7, "translations": {"en": "eins-en", "fr": "eins-fr"}},
        {"id": 3, "translations": {"en": "zwei-en", "fr": "zwei-fr"}},
    ]
    assert body["translation"]["targets"] == ["en", "fr"]
    assert body["translation"]["mode"] == "faithful"
    assert body["warnings"] == []
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

    r = _post_ok(
        client,
        segments=[{"id": 7, "text": "eins"}, {"id": 3, "text": "zwei"},
                  {"id": 42, "text": "drei"}],
        source="de")
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


def test_warning_ids_are_clipped(client, app_module, monkeypatch):
    """Segment ids are not counted against the character cap, and every
    kept-original fallback names them once per target: warnings clip each
    long id, while segments[].id still echoes it byte-identical."""
    _enable(app_module, monkeypatch)

    async def _fake(segments, targets, **kwargs):
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, [
            "segment 1 (en): kept original — translation failed (length ratio)",
            "segments 1-2 (en): kept original — translation failed (empty output)",
        ], {"model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {0: ["en"], 1: ["en"]}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    long_x, long_y = "x" * 1000, "y" * 1000
    r = _post_ok(client, segments=[{"id": long_x, "text": "eins"},
                                   {"id": long_y, "text": "zwei"}],
                 source="de")
    body = r.json()
    assert all(len(w) < 250 for w in body["warnings"]), body["warnings"]
    clip_x, clip_y = "x" * 64 + "…", "y" * 64 + "…"
    assert any(w.startswith(f"segment {clip_x} (en): kept original")
               for w in body["warnings"])
    assert any(w.startswith(f"segments {clip_x}, {clip_y} (en): kept original")
               for w in body["warnings"])
    assert [s["id"] for s in body["segments"]] == [long_x, long_y]


def test_segment_without_id_echoes_its_index(client, app_module, monkeypatch):
    _enable_stubbed(app_module, monkeypatch)
    r = _post_ok(client, segments=[{"text": "a"}, {"text": "b"}])
    assert [s["id"] for s in r.json()["segments"]] == [0, 1]


def test_401_when_locked_down(client, app_module, make_user_key, monkeypatch):
    _enable_stubbed(app_module, monkeypatch)
    _, raw_admin = make_user_key("admin", is_admin=True)   # flips to lockdown
    _post_status(client, 401)
    # ...and a plain (non-admin) bearer passes: no host gate, no page perm.
    _, raw_user = make_user_key("worker", is_admin=False)
    _post_ok(client, headers=bearer(raw_user))


def test_403_when_translation_disabled(client, app_module, monkeypatch):
    # TRANSLATION_ENABLED defaults off.
    _stub_translate(monkeypatch)
    r = _post_status(client, 403)
    assert "disabled" in r.json()["detail"]


def test_serial_translations_are_not_rate_limited(client, app_module,
                                                  monkeypatch):
    """One request at a time, over and over, is what the review UI does when a
    user retranslates line by line — below the per-minute ceiling it must
    never 429. The protection is the in-flight cap below; the per-minute
    counter is only a flood backstop (next test)."""
    _enable_stubbed(app_module, monkeypatch)
    for _ in range(25):
        _post_ok(client)


def test_per_minute_backstop_429s_and_releases_the_held_receipt(
        client, app_module, monkeypatch):
    """The per-minute counter is hit FIRST inside the release-on-reject
    bracket — ahead of the body validation, so a loop of malformed bodies is
    counted too: over the ceiling the request 429s with the config field
    named, and a parked dictation receipt is handed back instead of left to
    the sweeper."""
    _enable_stubbed(app_module, monkeypatch, TRANSLATE_RATE_PER_MIN=2)
    window = tr_routes._text_translate_rate
    window._state.clear()
    # A body that fails validation still counts against the window.
    _post_status(client, 422, segments=[])
    _post_ok(client)
    # Over the ceiling a malformed body is refused by the window, not
    # by the validation it never reaches.
    _post_status(client, 429, segments=[])
    r = _post_status(client, 429)
    body = r.json()
    assert body["error"]["param"] == "TRANSLATE_RATE_PER_MIN"
    assert r.headers["Retry-After"] == str(body["error"]["retry_after"])

    _assert_parked_receipt_released(client, "cap2", 429)


def _open_key():
    """The identity a request from the open-mode `client` fixture resolves to
    — its loopback client host (conftest `client` binds 127.0.0.1); the
    "(open-mode)" sentinel is not charged, see rate_limit.identity_key."""
    return "127.0.0.1"


def test_inflight_cap_rejects_when_the_identity_is_full(client, app_module,
                                                        monkeypatch):
    _enable_stubbed(app_module, monkeypatch)
    gauge = tr_routes._translate_inflight
    limit = int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)
    for _ in range(limit):
        gauge.acquire(_open_key())

    r = _post_status(client, 429)
    body = r.json()
    assert body["error"]["param"] == "TRANSLATE_MAX_INFLIGHT_PER_USER"
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["detail"] == body["error"]["message"]
    assert r.headers["Retry-After"] == str(body["error"]["retry_after"])
    # A refused request must not release a slot it never held.
    assert gauge.count(_open_key()) == limit

    gauge.clear()
    _post_ok(client)


def test_inflight_slot_is_released_on_success(client, app_module, monkeypatch):
    _enable_stubbed(app_module, monkeypatch)
    _post_ok(client)
    assert tr_routes._translate_inflight._counts == {}


@pytest.mark.parametrize("make_exc,status", [
    (lambda: translation.TranslationError("nope"), 400),
    (lambda: translation.TranslationCancelled(), 499),
    (lambda: RuntimeError("boom"), 500),
])
def test_inflight_slot_is_released_on_every_error_path(
        client, app_module, monkeypatch, make_exc, status):
    _enable(app_module, monkeypatch)
    _stub_raising(monkeypatch, make_exc())
    _post_status(client, status)
    assert tr_routes._translate_inflight._counts == {}


def test_inflight_slot_is_released_on_cancellation(client, app_module,
                                                   monkeypatch):
    """CancelledError is a BaseException, so every `except Exception` arm in
    the handler is skipped on a client disconnect — only the `finally` runs.
    That is why the release lives there and not in an except arm. The run is
    still recorded (as cancelled), like the batch handler's aborted runs."""
    _enable(app_module, monkeypatch)
    seen = {}
    recorded = record_runs(app_module, monkeypatch)

    async def _cancelled(*args, **kwargs):
        # Proves the slot was really taken: a stub never reached would leave
        # `_counts == {}` true trivially.
        seen["n"] = tr_routes._translate_inflight.count(_open_key())
        raise asyncio.CancelledError()
    monkeypatch.setattr(translation, "translate_segments", _cancelled)

    # TestClient surfaces the cancellation as RuntimeError, not as
    # asyncio.CancelledError — do not narrow this.
    with pytest.raises(BaseException):
        client.post(URL, json=_body())
    assert seen == {"n": 1}
    assert tr_routes._translate_inflight._counts == {}
    assert [(kw["status"], kw["kind"]) for kw in recorded] == \
        [("cancelled", "translate")]


def test_cancellation_stamps_the_job_row_cancelled(client, app_module,
                                                   monkeypatch):
    """A disconnect unwinds past every arm: the `finally` stamps the jobs
    row, and it says "cancelled", not "error". The write runs on a worker,
    never on the event loop: it waits on jobs_store's lock, which a prune
    or a big finish can hold for a while."""
    import threading

    _enable(app_module, monkeypatch)
    finished = []
    loop_thread = []
    done = threading.Event()

    async def _cancelled(*args, **kwargs):
        loop_thread.append(threading.current_thread())
        raise asyncio.CancelledError()

    def _finish(pid, **kw):
        finished.append((pid, kw, threading.current_thread()))
        done.set()
    monkeypatch.setattr(translation, "translate_segments", _cancelled)
    monkeypatch.setattr(tx_progress, "_jobs_finish_sync", _finish)
    with pytest.raises(BaseException):
        client.post(URL, json=_body(progress_id=_PID))
    assert done.wait(5)
    assert len(finished) == 1
    assert finished[0][2] is not loop_thread[0], "ran on the event loop"
    assert finished[0][0] == _PID
    assert finished[0][1]["status"] == "cancelled"
    assert finished[0][1]["error"] == "request aborted"


def test_inflight_zero_is_unlimited(client, app_module, monkeypatch):
    _enable_stubbed(app_module, monkeypatch, TRANSLATE_MAX_INFLIGHT_PER_USER=0)
    # A stale count from before the field was zeroed must not gate anything.
    tr_routes._translate_inflight._counts[_open_key()] = 99
    _post_ok(client)


def test_inflight_cap_is_per_user(client, app_module, make_user_key,
                                  monkeypatch):
    """The loopback `client` fixture is OPEN MODE — one synthetic admin, one
    bucket — so real keys are what prove the gauge is keyed per identity."""
    _enable_stubbed(app_module, monkeypatch)
    uid_a, key_a = make_user_key("alice", is_admin=True)
    _uid_b, key_b = make_user_key("bob", is_admin=False)
    gauge = tr_routes._translate_inflight
    for _ in range(int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)):
        gauge.acquire(uid_a)

    _post_status(client, 429, headers=bearer(key_a))
    _post_ok(client, headers=bearer(key_b))


def test_400_on_allowlist_miss(client, app_module, monkeypatch):
    calls = _enable_stubbed(app_module, monkeypatch,
                            TRANSLATION_DEFAULT_MODEL="org/default-GGUF:Q4",
                            TRANSLATION_ALLOWED_MODELS={"org/allowed-GGUF:Q4"})
    r = _post_status(client, 400, translation_model="org/other-GGUF:Q4")
    assert "TRANSLATION_ALLOWED_MODELS" in r.json()["detail"]
    assert calls == []
    # The configured default and an allowlisted model both pass.
    _post_ok(client)
    _post_ok(client, translation_model="org/allowed-GGUF:Q4")


def test_413_over_total_char_cap(client, app_module, monkeypatch):
    _enable_stubbed(app_module, monkeypatch)
    _post_status(client, 413, segments=[{"id": 0, "text": "x" * 200_001}])


def test_422_malformed_shapes(client, app_module, monkeypatch):
    calls = _enable_stubbed(app_module, monkeypatch, TRANSLATION_MAX_TARGETS=3)
    cases = [
        _body(segments="not-a-list"),
        _body(segments=[]),
        _body(segments=[{"id": 0}]),                          # no text
        _body(segments=[{"text": 5}]),                        # non-str text
        _body(targets=[]),                                    # empty targets
        _body(targets=["NOT A CODE"]),
        _body(targets=["en", "fr", "it", "es"]),              # over MAX (3)
        _body(translation_mode="poetic"),
        _body(context_segments="three"),
        _body(context_segments=True),                         # bool is not an int here
        "just a string",
        _body(segments=[{"id": i, "text": "x"} for i
                        in range(tr_routes._TEXT_TRANSLATE_MAX_SEGMENTS + 1)]),  # over entry cap
    ]
    for case in cases:
        r = client.post(URL, json=case)
        assert r.status_code == 422, (case, r.status_code, r.text)
    assert "capped at" in r.json()["detail"]     # the last case: entry cap
    assert calls == []


def test_422_lone_surrogate_in_text_or_id(client, app_module, monkeypatch):
    """A JSON "\\ud800" escape parses into a str UTF-8 cannot encode. A text
    copied through unchanged or an echoed id would only fail when the
    response (or the job row) is encoded, after the run was recorded as ok:
    refused up front instead, and a parked receipt is handed back."""
    import json

    calls = _enable_stubbed(app_module, monkeypatch)
    for seg in ({"id": 1, "text": "a\ud800"},
                {"id": "\ud800", "text": "a"}):
        _park("cap-u", hold_s=90)
        r = client.post(URL, content=json.dumps(
            _body(segments=[seg], captured_id="cap-u")),
            headers={"Content-Type": "application/json"})
        assert r.status_code == 422, r.text
        assert "surrogate" in r.json()["detail"]
        assert receipt_hold.pending() == 0
    assert calls == []


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_422_non_finite_number_in_body(client, app_module, monkeypatch,
                                       literal):
    """json.loads accepts NaN/Infinity, the JSON response refuses them: an
    echoed id would fail the response only after the run was recorded as
    ok. Refused up front instead, without a translation call."""
    calls = _enable_stubbed(app_module, monkeypatch)
    r = client.post(
        URL,
        content='{"segments": [{"id": %s, "text": "eins"}], '
                '"targets": ["en"]}' % literal,
        headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert "non-finite" in r.json()["detail"]
    assert calls == []


def test_targets_cap_fires_before_the_whole_list_is_walked(client, app_module,
                                                          monkeypatch):
    """De-dup is a set and the cap is checked inside the loop: a body of
    10k distinct valid codes is one cheap 422, not a quadratic walk on the
    event loop. Duplicates under the cap still collapse."""
    calls = _enable_stubbed(app_module, monkeypatch, TRANSLATION_MAX_TARGETS=2)
    # The invalid entry sits AFTER the code that trips the cap: only an
    # in-loop cap answers "capped at" — a cap checked after the walk would
    # reach the int first and answer "invalid language code".
    r = _post_status(client, 422, targets=["en", "fr", "de", 5])
    assert "capped at" in r.json()["detail"]
    assert "invalid" not in r.json()["detail"]
    _post_ok(client, targets=["en", "fr", "en", "fr"])
    assert calls[-1]["targets"] == ["en", "fr"]
    # Normalised like the batch csv (core.languages.language_codes).
    _post_ok(client, targets=["DE", "fr-ca", "fr-CA"])
    assert calls[-1]["targets"] == ["de", "fr-CA"]


def test_duplicate_targets_are_bounded_before_the_walk(client, app_module,
                                                      monkeypatch):
    """The in-loop cap counts DISTINCT codes only: a raw list of one repeated
    code is bounded by its length before the walk, inside the
    release-on-reject bracket."""
    calls = _enable_stubbed(app_module, monkeypatch, TRANSLATION_MAX_TARGETS=2)
    r = _post_status(client, 422, targets=["de"] * 65)
    assert "capped at 64" in r.json()["detail"]
    assert calls == []
    _post_ok(client, targets=["de"] * 64)
    assert calls[-1]["targets"] == ["de"]
    _assert_parked_receipt_released(client, "dup-t", 422, targets=["de"] * 65)


def test_source_must_be_a_language_code(client, app_module, monkeypatch):
    """`source` reaches every prompt and the receipt: a code, "auto" or
    nothing — never a forged log line."""
    calls = _enable_stubbed(app_module, monkeypatch)
    _post_status(client, 422, source="de\n[translate] req=deadbeef forged")
    assert calls == []
    r = client.post(URL, json=_body(source="de"))
    assert r.status_code == 200 and calls[-1]["source_lang"] == "de"
    r = client.post(URL, json=_body(source="auto"))
    assert r.status_code == 200 and calls[-1]["source_lang"] is None
    r = client.post(URL, json=_body(source="DE"))
    assert r.status_code == 200 and calls[-1]["source_lang"] == "de"


def test_speaker_is_clipped_and_counted_against_the_char_cap(
        client, app_module, monkeypatch):
    """The speaker prefixes every context line of every prompt: clipped at
    parse time, control characters neutralised, and counted in the cap."""
    calls = _enable_stubbed(app_module, monkeypatch)
    _post_ok(client, segments=[
        {"id": 0, "text": "eins", "speaker": "S\nPEAKER" + "x" * 500}])
    spk = calls[-1]["segments"][0]["speaker"]
    assert len(spk) == tr_routes._TEXT_TRANSLATE_MAX_SPEAKER
    assert spk.startswith("S?PEAKER")
    n = tr_routes._TEXT_TRANSLATE_MAX_CHARS // 100
    _post_status(client, 413, segments=[
        {"id": i, "text": "x" * 90, "speaker": "y" * 64} for i in range(n)])


def test_ok_run_stamps_its_job_row_off_the_event_loop(client, app_module,
                                                      monkeypatch):
    """The success payload (up to 2000 segments x 10 targets) is encoded and
    written on a worker thread, never on the loop."""
    from faster_whisper_backend.transcription import jobs_store
    _enable_stubbed(app_module, monkeypatch)
    seen = record_calls(monkeypatch, jobs_store, "finish",
                        lambda **kw: (kw.get("state"), on_event_loop()))
    _post_ok(client, progress_id=_PID)
    assert seen == [("done", False)]


def test_result_is_rendered_off_the_event_loop(client, app_module, monkeypatch):
    """The answer echoes every client id (up to a 4 MiB body of them): its
    jsonable_encoder walk and JSON render run on a worker thread, and the
    bytes stay those FastAPI renders for the same dict."""
    import json

    from fastapi.encoders import jsonable_encoder
    from starlette.responses import JSONResponse
    _enable_stubbed(app_module, monkeypatch)
    seen = []
    real = JSONResponse.render

    def _spy(self, content):
        if isinstance(content, dict) and "segments" in content:
            seen.append(on_event_loop())
        return real(self, content)
    monkeypatch.setattr(JSONResponse, "render", _spy)
    r = _post_ok(client, segments=[{"id": "ä", "text": "eins"},
                                   {"id": [1, {"k": None}], "text": ""}])
    assert seen == [False]
    assert r.headers["content-type"] == "application/json"
    assert r.content == json.dumps(
        jsonable_encoder(r.json()), ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")
    assert [s["id"] for s in r.json()["segments"]] == ["ä", [1, {"k": None}]]


def _register_load(secs):
    """What translation.engine's load records in the model registry (minus
    the model-size ledger write): the stub model loaded NOW in `secs`."""
    import time

    from faster_whisper_backend.runtime import model_registry
    with model_registry._loaded_models_lock:
        model_registry._loaded_models["gguf:org/d:Q4"] = {
            "name": "gguf:org/d:Q4", "device": "cuda", "compute_type": "gguf",
            "vram_bytes": None, "load_secs": secs, "loaded_at": time.time(),
            "last_used": time.time(), "last_used_monotonic": time.monotonic()}


def _logged_load_s(caplog):
    done = [m for m in caplog.messages if "✓ done in" in m]
    return float(re.search(r"\(load ([\d.]+)s", done[0]).group(1))


def test_cold_load_excludes_the_gpu_queue_wait(client, app_module,
                                               monkeypatch, caplog):
    """load = the registry's record of a load that landed after the slot
    was held. The GpuGate queue before it is wait (metrics.take_wait
    reports it), never load — and a load another request finished while
    this one queued was not paid for here."""
    from faster_whisper_backend.transcription import models as tx_models
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(translation, "_resolve_device", lambda: "cuda")
    monkeypatch.setattr(translation, "is_resident", lambda ref: False)

    preloaded = {"during_queue": False}

    @contextlib.asynccontextmanager
    async def _busy_gate():
        await asyncio.sleep(0.6)            # queued behind batch runs
        if preloaded["during_queue"]:
            _register_load(5.0)             # someone else's load, meanwhile
            await asyncio.sleep(0.05)
        yield
    monkeypatch.setattr(tx_models, "get_inference_semaphore", _busy_gate)

    async def _cold(segments, targets, *, progress_cb=None, **kwargs):
        if not preloaded["during_queue"]:
            await asyncio.sleep(0.1)        # the model load
            _register_load(0.1)
        progress_cb(1.0)
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _cold)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _post_ok(client)
    assert _logged_load_s(caplog) == pytest.approx(0.1)
    caplog.clear()
    preloaded["during_queue"] = True
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _post_ok(client)
    assert _logged_load_s(caplog) == 0.0


def test_load_is_not_cut_short_by_an_early_callback(client, app_module,
                                                     monkeypatch, caplog):
    """A same-language first target (or a blank first group) calls back
    before any model is loaded: the cold load that follows is still load,
    not infer."""
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(translation, "is_resident", lambda ref: False)

    async def _verbatim_then_cold(segments, targets, *, progress_cb=None,
                                  **kwargs):
        progress_cb(0.5, "de 1/1", target="de", target_progress=1.0)
        await asyncio.sleep(0.3)            # the model load for "en"
        _register_load(0.3)
        progress_cb(1.0, "en 1/1", target="en", target_progress=1.0)
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, [], {"model": "org/d:Q4", "source": "de",
                             "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _verbatim_then_cold)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _post_ok(client, targets=["de", "en"], source="de")
    assert _logged_load_s(caplog) == pytest.approx(0.3)


def test_held_receipt_survives_a_gpu_queue_with_a_resident_model(
        client, app_module, monkeypatch):
    """The other half of the heartbeat: the model is resident, but the
    request is queued behind batch runs for the GPU slot and reports no
    progress meanwhile. The receipt must outlive that queue."""
    from faster_whisper_backend.transcription import models as tx_models

    _enable_stubbed(app_module, monkeypatch)
    monkeypatch.setattr(tr_routes, "_HOLD_HEARTBEAT_S", 0.05)
    monkeypatch.setattr(translation, "_resolve_device", lambda: "cuda")
    monkeypatch.setattr(translation, "is_resident", lambda ref: True)
    swept = []

    @contextlib.asynccontextmanager
    async def _busy_gate():
        # 0.5 s in the queue, a hold of 0.2 s, the sweeper running all along.
        for _ in range(10):
            await asyncio.sleep(0.05)
            swept.extend(receipt_hold.sweep())
        yield
    monkeypatch.setattr(tx_models, "get_inference_semaphore", _busy_gate)
    _park("cap-q", hold_s=0.2)
    _post_ok(client, captured_id="cap-q")
    assert swept == []
    assert receipt_hold.pending() == 0          # claimed by the request


def test_plan_is_clocked_without_a_progress_id(client, app_module,
                                               monkeypatch):
    """progress_id is optional: without one the registry has no entry to
    tick the plan through, so the handler ticks it itself — the response's
    plan reports what the run took, not 0 s."""
    _enable(app_module, monkeypatch)

    async def _slow(segments, targets, *, progress_cb=None, **kwargs):
        progress_cb(0.0, "de 0/1", target="de", target_progress=0.0)
        await asyncio.sleep(0.6)
        progress_cb(1.0, "de 1/1", target="de", target_progress=1.0)
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _slow)
    r = _post_ok(client, targets=["de"])
    stage = r.json()["plan"][0]
    assert stage["state"] == "done"
    assert stage["took_s"] > 0
    assert stage["units"][0]["took_s"] > 0


def test_kept_original_reaches_the_usage_stage(client, app_module,
                                               monkeypatch):
    """The usage ledger counts guard fallbacks from the stage dict's
    structured kept_original, as the batch stage reports it."""
    _enable(app_module, monkeypatch)

    async def _fake(segments, targets, **kwargs):
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, [
            "segment 1 (en): kept original — translation failed (empty output)",
        ], {"model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {0: ["en"], 1: []}}
    monkeypatch.setattr(translation, "translate_segments", _fake)
    recorded = record_runs(app_module, monkeypatch)
    _post_ok(client)
    assert [kw["stages"][0]["kept_original"] for kw in recorded] == [1]


def test_held_receipt_survives_a_queued_translation(client, app_module,
                                                    monkeypatch):
    """Queued for the GPU slot (or loading a model) the request reports no
    progress; the heartbeat keeps the held receipt's idle timer fresh, so
    the sweeper does not release it as "no result" meanwhile."""
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(tr_routes, "_HOLD_HEARTBEAT_S", 0.05)
    monkeypatch.setattr(translation, "is_resident", lambda ref: False)
    swept = []

    async def _slow(segments, targets, **kwargs):
        # A model that is still loading: no progress callback for 0.5 s,
        # a hold of 0.2 s, the sweeper running all along.
        for _ in range(10):
            await asyncio.sleep(0.05)
            swept.extend(receipt_hold.sweep())
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _slow)
    _park("cap-q", hold_s=0.2)
    _post_ok(client, captured_id="cap-q")
    assert swept == []
    assert receipt_hold.pending() == 0          # claimed by the request


def test_translation_error_maps_to_400(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_raising(monkeypatch, translation.TranslationError(
        "no translation model configured (TRANSLATION_DEFAULT_MODEL)"))
    r = _post_status(client, 400)
    assert "TRANSLATION_DEFAULT_MODEL" in r.json()["detail"]


def test_unexpected_error_maps_to_generic_500(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    _stub_raising(monkeypatch, RuntimeError("/secret/path/model.gguf exploded"))
    r = _post_status(client, 500)
    assert r.json()["detail"] == "translation failed"     # never the raw text


def test_progress_visible_and_cancel_honored(client, app_module, monkeypatch):
    # The stub runs mid-stage: it observes the live progress entry (proving
    # GET progress / POST cancel can see the id) and then flags the id in
    # _BATCH_CANCELLED — exactly what POST cancel/{id} does — before honoring
    # cancel_check. The request must abort with 499 and clean both registries.
    _enable(app_module, monkeypatch)
    seen = {}

    async def _slow(segments, targets, *, progress_cb=None, cancel_check=None,
                    **kwargs):
        seen["entry"] = dict(tx_progress._BATCH_PROGRESS.get(_PID) or {})
        tx_progress._BATCH_CANCELLED.add(_PID)
        if cancel_check():
            raise translation.TranslationCancelled()
        raise AssertionError("cancel_check ignored the flagged id")
    monkeypatch.setattr(translation, "translate_segments", _slow)

    _post_status(client, 499, progress_id=_PID)
    assert seen["entry"].get("stage") == "translating"
    assert seen["entry"].get("progress") == 0.0
    assert _PID not in tx_progress._BATCH_PROGRESS
    assert _PID not in tx_progress._BATCH_CANCELLED


def test_progress_wrapper_forwards_last_text_and_logs_receipts(
        client, app_module, monkeypatch, caplog):
    """The handler's progress wrapper merges last_text into the progress
    entry, and the run brackets itself with start + ✓ done log lines."""
    _enable(app_module, monkeypatch)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        progress_cb(0.5, "en 1/2", "Hello there")
        seen["entry"] = dict(tx_progress._BATCH_PROGRESS.get(_PID) or {})
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _fake)

    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _post_ok(client, progress_id=_PID)
    assert seen["entry"].get("last_text") == "Hello there"
    assert seen["entry"].get("progress") == 0.5
    assert seen["entry"].get("step") == "en 1/2"
    msgs = caplog.messages
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
        seen["unknown_total"] = dict(tx_progress._BATCH_PROGRESS.get(_PID)
                                     or {})
        download_cb(512, 2048)
        seen["download"] = dict(tx_progress._BATCH_PROGRESS.get(_PID) or {})
        progress_cb(0.0, "en 1/1", None)
        seen["after"] = dict(tx_progress._BATCH_PROGRESS.get(_PID) or {})
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _fake)

    _post_ok(client, progress_id=_PID)
    assert seen["unknown_total"].get("stage") == "downloading"
    assert seen["unknown_total"].get("progress") is None
    assert seen["download"].get("stage") == "downloading"
    assert seen["download"].get("progress") == 0.25
    assert seen["download"].get("total_bytes") == 2048
    assert seen["after"].get("stage") == "translating"


def test_progress_endpoint_serves_last_text(client, app_module):
    """last_text is on the GET progress endpoint's field whitelist."""
    tx_progress._progress_set(_PID, stage="translating", last_text="the tail")
    try:
        r = client.get(f"/v1/audio/transcriptions/progress/{_PID}")
        assert r.status_code == 200
        assert r.json()["last_text"] == "the tail"
    finally:
        tx_progress._BATCH_PROGRESS.pop(_PID, None)


def test_failure_logs_terminal_line(client, app_module, monkeypatch, caplog):
    _enable(app_module, monkeypatch)
    _stub_raising(monkeypatch, translation.TranslationError(
        "no translation model configured"))
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        r = client.post(URL, json=_body())
    assert r.status_code == 400
    assert any("✗ failed" in m for m in caplog.messages)


def test_locked_translation_model_applies_to_the_text_endpoint(
        client, app_module, make_user_key, monkeypatch):
    """Per-identity locks bind HERE too — otherwise a key locked to a small
    model on the batch path just switches endpoints."""
    # Open allowlist: this test is about the lock, not the allowlist gate
    # (the shipped config allowlists two real models).
    calls = _enable_stubbed(app_module, monkeypatch,
                            TRANSLATION_ALLOWED_MODELS=set())
    _, raw_bob = bind_profile(client, make_user_key, "small-mt", {
        "TRANSLATION_MODEL": "org/small:Q4_K_M",
        "locks": ["TRANSLATION_MODEL"]})

    r = _post_ok(client, headers=bearer(raw_bob),
                 translation_model="org/huge:Q8_0")
    assert calls[0]["model_ref"] == "org/small:Q4_K_M"  # locked value wins
    assert any("locked" in w for w in r.json().get("warnings", []))


def test_locked_translate_to_binds_the_text_route(
        client, app_module, make_user_key, monkeypatch):
    """A TRANSLATE_TO lock binds here as on the batch path: the locked targets
    win over the body's, and a locked empty value refuses the route."""
    calls = _enable_stubbed(app_module, monkeypatch)
    _, raw_bob = bind_profile(client, make_user_key, "de-only", {
        "TRANSLATE_TO": "de", "locks": ["TRANSLATE_TO"]})
    r = _post_ok(client, headers=bearer(raw_bob), targets=["en"])
    assert calls[-1]["targets"] == ["de"]
    assert r.json()["translation"]["targets"] == ["de"]
    assert any(w.startswith("targets is locked")
               for w in r.json()["warnings"])
    # Echoing the locked value is no override: no warning.
    r = _post_ok(client, headers=bearer(raw_bob), targets=["DE"])
    assert not any("locked" in w for w in r.json()["warnings"])


def test_locked_empty_translate_to_refuses_the_text_route(
        client, app_module, make_user_key, monkeypatch):
    calls = _enable_stubbed(app_module, monkeypatch)
    _, raw_bob = bind_profile(client, make_user_key, "no-mt", {
        "TRANSLATE_TO": "", "locks": ["TRANSLATE_TO"]})
    _park("off-t", hold_s=90)
    r = _post_status(client, 403, headers=bearer(raw_bob), targets=["en"],
                     captured_id="off-t")
    assert "TRANSLATE_TO" in r.json()["detail"]
    assert calls == []
    assert receipt_hold.pending() == 0


def test_inflight_refusal_releases_the_held_receipt(client, app_module,
                                                     monkeypatch):
    """The in-flight acquire sits OUTSIDE the handler's try, so its 429 never
    reaches the `except HTTPException` release. The parked dictation receipt
    must still be released — not left for the sweeper to log 90 s later."""
    from faster_whisper_backend.transcription import receipt as tx_receipt

    _enable_stubbed(app_module, monkeypatch)
    logged = []
    monkeypatch.setattr(tx_receipt, "_log_held_receipts", logged.extend)
    gauge = tr_routes._translate_inflight
    for _ in range(int(app_module.cfg.TRANSLATE_MAX_INFLIGHT_PER_USER)):
        gauge.acquire(_open_key())
    _assert_parked_receipt_released(client, "cap1", 429)
    assert len(logged) == 1               # handed back, not dropped
    assert any("too many in flight" in n for n in logged[0]["skipped"])


def test_validation_reject_releases_the_held_receipt(client, app_module,
                                                     monkeypatch):
    """Every validation exit (422 shape, 413 size) runs inside the same
    release-on-reject bracket as the rate hit: a parked receipt must be
    handed back on a malformed request, not left for the 90 s sweeper."""
    _enable_stubbed(app_module, monkeypatch)
    _assert_parked_receipt_released(client, "cap-v", 422, segments=[])
    _assert_parked_receipt_released(
        client, "cap-s", 413, segments=[{"id": 0, "text": "x" * 200_001}])


def test_admin_pinned_model_passes_the_allowlist_gate(
        client, app_module, make_user_key, monkeypatch):
    """The allowlist constrains only the CLIENT-requested value (the
    diarization/separation stance): an admin who pins TRANSLATION_MODEL in a
    profile need not also add it to the global allowlist."""
    calls = _enable_stubbed(app_module, monkeypatch,
                            TRANSLATION_ALLOWED_MODELS={"org/public-GGUF:Q4"},
                            TRANSLATION_DEFAULT_MODEL="org/public-GGUF:Q4")
    admin_h, raw_bob = bind_profile(client, make_user_key, "pinned-mt", {
        "TRANSLATION_MODEL": "org/pinned-GGUF:Q4",
        "locks": ["TRANSLATION_MODEL"]})

    # No translation_model in the request: the pinned (admin-policy) model
    # resolves and passes even though it is not on the allowlist.
    _post_ok(client, headers=bearer(raw_bob))
    assert calls[0]["model_ref"] == "org/pinned-GGUF:Q4"

    # A CLIENT asking for a non-allowlisted model is still refused.
    _post_status(client, 400, headers=admin_h,
                 translation_model="org/evil-GGUF:Q8")


def test_translation_error_records_status_error(client, app_module,
                                                monkeypatch, make_user_key):
    # The terminal-error status is "error" — the batch handler's spelling
    # and metrics.py's default — never "failed": both land in the same
    # recent_transcriptions.status column rendered verbatim by /stats.
    _enable(app_module, monkeypatch)
    _stub_raising(monkeypatch, translation.TranslationError("nope"))
    recorded = record_runs(app_module, monkeypatch)
    from faster_whisper_backend.auth import api_keys_store
    make_user_key("root", is_admin=True)            # lock the server down
    uid = api_keys_store.create_user("alice", is_admin=False)
    raw, _ = api_keys_store.create_key(uid, label="alice-laptop")
    _post_status(client, 400, headers=bearer(raw))
    assert recorded and recorded[-1]["status"] == "error"
    assert recorded[-1]["kind"] == "translate"
    assert recorded[-1]["key_label"] == "alice-laptop"
    from faster_whisper_backend.stats import recent_transcriptions_store
    rows = recent_transcriptions_store.list_recent(limit=5)
    assert rows and rows[0]["status"] == "error"


def test_empty_translation_allowlist_admits_any_well_formed_ref(app_module,
                                                                monkeypatch):
    # Documented (settings/descriptions.py TRANSLATION_ALLOWED_MODELS help):
    # an EMPTY allowlist is permissive — unlike the diarization/separation gates.
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    assert tr_gating._translation_model_allowed(
        "someone/other:Q4", requested="someone/other:Q4") is True
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/a:Q4"}, raising=False)
    assert tr_gating._translation_model_allowed(
        "someone/other:Q4", requested="someone/other:Q4") is False


def test_text_run_has_a_one_stage_plan_with_units(client, app_module,
                                                  monkeypatch):
    """A text run is a plan of exactly one stage whose units are the
    targets; the response carries its receipt."""
    _enable(app_module, monkeypatch)
    seen = {}

    async def _fake(segments, targets, *, progress_cb=None, **kwargs):
        progress_cb(0.5, "en 1/1", None, target="en", target_progress=0.5)
        # What the progress route would answer right now (no nested HTTP
        # call from inside the server: read the same sources it reads).
        seen["progress"] = {
            **dict(tx_progress._BATCH_PROGRESS.get(_PID) or {}),
            **tx_progress._plan_fields(_PID)}
        return _echo_result(segments, targets)
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = _post_ok(client, progress_id=_PID)
    live = seen["progress"]
    assert [s["stage"] for s in live["plan"]] == ["translating"]
    assert live["plan"][0]["units"][0] == {
        "target": "en", "state": "running", "progress": 0.5,
        "elapsed_s": live["plan"][0]["units"][0]["elapsed_s"],
        "est_s": live["plan"][0]["units"][0]["est_s"]}
    assert live["target"] == "en" and live["target_progress"] == 0.5
    assert 0.0 < live["overall"] < 1.0
    receipt = r.json()["plan"]
    assert receipt[0]["state"] == "done"
    assert receipt[0]["units"][0]["state"] == "done"
    assert _PID not in tx_progress._RUN_PLAN_BY_PID


def test_unheld_request_logs_a_standalone_receipt(client, app_module,
                                                  monkeypatch, caplog):
    """A translate that names no captured_id (a stop-timing one-shot, a
    plain API caller) still gets ONE receipt block with the model, targets
    and identity — not just the progress lines."""
    _enable_stubbed(app_module, monkeypatch)

    with caplog.at_level(logging.INFO, logger="whisper-api"):
        _post_ok(client, client_job="a" * 32)
    blocks = [rec.getMessage() for rec in caplog.records
              if "/v1/text/translations" in rec.getMessage()
              and "═══" in rec.getMessage()]
    assert len(blocks) == 1
    blk = blocks[0]
    assert "org/d:Q4" in blk
    assert re.search(r"^\s+targets\s+en\s*$", blk, re.M), blk
    assert "Identity" in blk
    # Links to the session's utterance receipts by the same `job=` token.
    assert "job=aaaaaaaa" in blk


def test_claimed_dictation_receipt_folds_into_the_utterance_row(
        client, app_module, monkeypatch):
    """With a held receipt, the translation becomes a second stage on the
    utterance's recent-jobs row instead of a separate translate row — the
    /stats table shows ONE dictate job with transcribing + translating."""
    from faster_whisper_backend.stats import recent_transcriptions_store as rts

    _enable_stubbed(app_module, monkeypatch)
    _fast = translation.translate_segments

    async def _slow(*a, **kw):
        # A measurable wall time: the fold must ADD it to the utterance row.
        await asyncio.sleep(0.1)
        return await _fast(*a, **kw)
    monkeypatch.setattr(translation, "translate_segments", _slow)
    rts.record_timing(request_id="utt-rid", model="w", audio_s=4.0,
                      processing_s=0.9, status="ok", words=7, kind="dictate",
                      stages=[{"name": "transcribing", "secs": 0.9}])
    _park("cap-fold", hold_s=90, file_label="utt#0", model_name="w",
          request_id="utt-rid")
    _post_ok(client, captured_id="cap-fold")
    assert receipt_hold.pending() == 0
    rows = rts.list_recent(limit=10)
    assert [x["kind"] for x in rows].count("translate") == 0
    utt = next(x for x in rows if x["request_id"] == "utt-rid")
    assert [s["name"] for s in utt["stages"]] == ["transcribing", "translating"]
    tr_secs = utt["stages"][1]["secs"]
    assert tr_secs >= 0.1
    assert utt["processing_s"] >= 0.95
    assert utt["processing_s"] == pytest.approx(0.9 + tr_secs, abs=0.01)


def test_dictation_translation_joins_its_session_usage_job(
        client, app_module, monkeypatch, make_user_key):
    """The desktop client mints a fresh progress_id per translate call and
    names the dictation session only as client_job. The translating stage
    joins that session's usage job (dictation.targets lists the target, the
    translating meter counts the dictation once), and a client_job naming
    another user's session never attaches."""
    import zoneinfo

    from faster_whisper_backend.stats import usage_store
    _enable_stubbed(app_module, monkeypatch)
    uid, raw = make_user_key("alice", is_admin=True)
    bob, raw_bob = make_user_key("bob")
    sess = "5e55" * 8
    usage_store.record_usage(key_id="k", user_id=uid, audio_s=5.0, words=20,
                             status="ok", kind="dictation", job_id=sess)
    _post_ok(client, client_job=sess, progress_id="a1" * 16,
             headers=bearer(raw))
    _post_ok(client, client_job=sess, progress_id="b2" * 16,
             headers=bearer(raw_bob))
    utc = zoneinfo.ZoneInfo("UTC")
    doc = usage_store.document(uid, days=1, tz=utc, tz_name="UTC")
    assert doc["dictation"]["targets"] == [
        {"code": "en", "runs": 1, "kept_original": 0}]
    meter = next(s for s in doc["stages"] if s["stage"] == "translating")
    assert (meter["runs"], meter["of_runs"]) == (1, 1)
    # Bob's run is a text job of his own.
    assert usage_store._require_conn().execute(
        "SELECT kind FROM usage_jobs WHERE job_id = ?",
        ("b2" * 16,)).fetchone()["kind"] == "text"


def test_a_foreign_captured_id_neither_claims_nor_releases_the_hold(
        client, app_module, monkeypatch, make_user_key):
    """The capture id is client-named: another identity naming it must not
    steal the receipt into its own translation, release it early as
    "request rejected", or keep it alive. The hold stays for its owner."""
    _enable_stubbed(app_module, monkeypatch)
    alice, raw_alice = make_user_key("alice", is_admin=True)
    _bob, raw_bob = make_user_key("bob")
    _park("cap-x", hold_s=90, user_id=alice, request_id="alice-rid")
    _post_status(client, 422, bearer(raw_bob), captured_id="cap-x", segments=[])
    assert receipt_hold.pending() == 1
    _post_ok(client, bearer(raw_bob), captured_id="cap-x")
    assert receipt_hold.pending() == 1
    _post_ok(client, bearer(raw_alice), captured_id="cap-x")
    assert receipt_hold.pending() == 0


def test_huge_bidi_speaker_is_clipped_before_sanitising(client, app_module,
                                                        monkeypatch):
    """The sanitiser runs over the clipped speaker only, not a body-sized
    one, and the label is the same as sanitising first then clipping."""
    calls = _enable_stubbed(app_module, monkeypatch)
    seen = record_calls(monkeypatch, tr_routes, "_log_safe",
                        lambda s, **kw: len(s))
    _post_ok(client, segments=[
        {"id": 0, "text": "eins", "speaker": "‮" * 200_000}])
    assert seen and max(seen) <= tr_routes._TEXT_TRANSLATE_MAX_SPEAKER
    assert calls[-1]["segments"][0]["speaker"] == "?" * 64
