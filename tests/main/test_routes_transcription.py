"""Integration tests for POST /v1/audio/transcriptions.

Drives the real handler with the FakeModel injected via the harness. The
fake model ignores the uploaded bytes, so a tiny dummy WAV payload is fine.
"""

import pytest
from fastapi import HTTPException

from tests.conftest import FakeModel, FakeSegment, bearer, record_calls, write_wav
from tests._translation_helpers import enable_translation, stub_translate
from tests.main._stage_helpers import capture_receipt
from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import transcode as audio_transcode
from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.media import media_store
from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.transcription import run_plan
from tests._transcription_helpers import (
    fail_model_load, get_progress, post_json_ok, post_transcription as _post,
    post_translation, post_verbose_ok, record_runs, serve_model, watch_progress)


def test_default_json_returns_text_object(client):
    assert post_json_ok(client) == {"text": "hallo welt"}


def test_text_format_returns_plain_string(client):
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # Like OpenAI: the raw transcript as text/plain, not a JSON string literal.
    assert r.text == "hallo welt"
    assert r.headers["content-type"] == "text/plain; charset=utf-8"


def test_translations_text_format_returns_plain_string(client):
    r = post_translation(client, response_format="text")
    assert r.status_code == 200, r.text
    assert r.text == "hallo welt"
    assert r.headers["content-type"].startswith("text/plain")


def test_verbose_json_shape(client):
    body = post_verbose_ok(client)
    assert body["task"] == "transcribe"
    assert body["language"] == "de"
    assert body["duration"] == 1.0
    assert body["text"] == "hallo welt"
    assert isinstance(body["segments"], list) and body["segments"]
    seg = body["segments"][0]
    assert seg["text"] == "hallo welt"
    # verbose_json with no explicit granularities still asks for words, and
    # WORD_TIMESTAMPS_ENABLED defaults True.
    assert "words" in body
    assert [w["word"] for w in body["words"]] == ["hallo", "welt"]


def _two_segments(fake_model, monkeypatch):
    from faster_whisper_backend.pipeline import engine as pl_engine
    fake_model._segments = [FakeSegment(" hallo welt", 0.0, 1.5),
                            FakeSegment(" zweiter satz", 1.5, 3661.25)]
    seen = []

    def _pp(text, **kw):
        seen.append(kw.get("language"))
        return text.upper()
    monkeypatch.setattr(pl_engine, "_postprocess_text", _pp)
    return seen


def test_srt_format_returns_a_subrip_document(client, fake_model, monkeypatch):
    # Like OpenAI: numbered cues at the segments' own times, as text/plain.
    # The cue text is the pipeline's output (upper-cased by the stub), not
    # the raw segment text verbose_json's segments[] carry.
    seen = _two_segments(fake_model, monkeypatch)
    r = _post(client, response_format="srt")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.text == ("1\n00:00:00,000 --> 00:00:01,500\nHALLO WELT\n\n"
                      "2\n00:00:01,500 --> 01:01:01,250\nZWEITER SATZ\n")
    assert "de" in seen


def test_vtt_format_returns_a_webvtt_document(client, fake_model, monkeypatch):
    _two_segments(fake_model, monkeypatch)
    r = _post(client, response_format="vtt")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.text == ("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHALLO WELT\n\n"
                      "00:00:01.500 --> 01:01:01.250\nZWEITER SATZ\n")


def test_translations_srt_format_returns_a_subrip_document(client):
    r = post_translation(client, response_format="srt")
    assert r.status_code == 200, r.text
    assert r.text == "1\n00:00:00,000 --> 00:00:01,000\nhallo welt\n"


def test_unknown_response_format_is_refused_before_the_model_loads(
        client, monkeypatch):
    # A typo must not land the json shape in the caller's file with a 200.
    loads = []

    async def _loader(name, *, lease=False):
        loads.append(name)
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)
    r = _post(client, response_format="xml")
    assert r.status_code == 422
    assert r.json()["detail"] == ("response_format must be one of: "
                                  "json, text, srt, verbose_json, vtt")
    assert loads == []


def test_words_gated_by_config_disabled(client, app_module):
    # WORD_TIMESTAMPS_ENABLED=False beats a request asking for word granularity.
    app_module.cfg.WORD_TIMESTAMPS_ENABLED = False
    body = post_verbose_ok(client, **{"timestamp_granularities[]": "word"})
    assert body.get("words") == []


def test_words_included_with_granularity_field(client, app_module):
    body = post_verbose_ok(client, **{"timestamp_granularities[]": "word"})
    assert [w["word"] for w in body["words"]] == ["hallo", "welt"]


def _serve_failing_decode(monkeypatch, msg):
    """Every load returns a model whose transcribe() raises RuntimeError(msg)."""
    class _Failing(FakeModel):
        def transcribe(self, path, **kwargs):
            raise RuntimeError(msg)
    serve_model(monkeypatch, _Failing())


def test_model_transcribe_raises_returns_500(client, app_module, monkeypatch):
    _serve_failing_decode(monkeypatch, "decode blew up")
    r = _post(client, response_format="json")
    assert r.status_code == 500


def test_output_prefix_suffix_wrap(client, app_module):
    app_module.cfg.OUTPUT_PREFIX = "[de] "
    app_module.cfg.OUTPUT_SUFFIX = " (end)"
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # The post-wrapper trim strips only the ends of the whole string.
    assert r.text == "[de] hallo welt (end)"


def test_missing_file_is_422(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 422


def test_prompt_sentinel_inherit_clear_value(client, fake_model, app_module):
    """`prompt` is a present-vs-absent sentinel read from the RAW form (FastAPI
    coerces an empty Form field to its default, so the handler reads request.form()
    directly). Absent → inherit DEFAULT_PROMPT; present-but-empty "" → CLEAR (no
    initial_prompt); value → verbatim."""
    app_module.cfg.DEFAULT_PROMPT = "SERVER PROMPT"
    _post(client, response_format="json")
    assert fake_model.last_kwargs["initial_prompt"] == "SERVER PROMPT"
    _post(client, response_format="json", prompt="my terms")
    assert fake_model.last_kwargs["initial_prompt"] == "my terms"
    # explicit empty → CLEAR. httpx drops empty `data`/`files` values, so
    # hand-build the multipart body for a present-but-empty `prompt` part
    # (what reqwest sends).
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
    from faster_whisper_backend.stats import metrics

    before = metrics.err_count["/v1/audio/transcriptions"]
    assert post_json_ok(client, decode_overrides="[" * 200_000) == {"text": "hallo welt"}
    # Ignored, so nothing from it reached the model, and no error was recorded.
    assert metrics.err_count["/v1/audio/transcriptions"] == before


def test_absurdly_long_filename_extension_does_not_crash(client):
    # The client filename feeds tempfile.NamedTemporaryFile's suffix; a 250-char
    # extension used to raise OSError(36, 'File name too long') → 500.
    assert post_json_ok(
        client, files={"file": ("a." + "x" * 250, b"RIFFxxxxWAVE", "audio/wav")}
    ) == {"text": "hallo welt"}


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
    assert tx_models._MODEL_ID_RE.match("some-repo")
    assert tx_models._MODEL_ID_RE.match("org/some-repo")
    assert not tx_models._MODEL_ID_RE.match("some-repo\n")
    assert not tx_models._MODEL_ID_RE.match("org/some-repo\n")


def test_task_translate_reaches_model_and_response(client, fake_model):
    body = post_verbose_ok(client, task="translate")
    assert fake_model.last_kwargs["task"] == "translate"
    assert body["task"] == "translate"


def test_task_default_is_not_forwarded(client, fake_model):
    body = post_verbose_ok(client)
    assert "task" not in fake_model.last_kwargs
    assert body["task"] == "transcribe"


def test_task_invalid_is_422(client):
    r = _post(client, task="summarize")
    assert r.status_code == 422


def test_translations_endpoint_pins_task(client, fake_model):
    r = post_translation(client, response_format="verbose_json")
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["task"] == "translate"
    assert r.json()["task"] == "translate"


def test_task_config_default_applies_when_field_absent(client, app_module, fake_model):
    app_module.cfg.TASK = "translate"
    try:
        post_verbose_ok(client)
        assert fake_model.last_kwargs["task"] == "translate"
        post_verbose_ok(client, task="transcribe")
        assert "task" not in fake_model.last_kwargs
    finally:
        app_module.cfg.TASK = "transcribe"


def test_progress_endpoint_unknown_id(client):
    r = get_progress(client, "a" * 32)
    assert r.status_code == 200
    assert r.json() == {"stage": "unknown"}


def test_progress_endpoint_malformed_id_422(client):
    r = client.get("/v1/audio/transcriptions/progress/NOT-HEX!")
    assert r.status_code == 422


def test_progress_live_entry_and_cleanup(client, app_module):
    pid = "b" * 32
    tx_progress._progress_set(pid, stage="transcribing", progress=0.5, duration=60.0)
    r = get_progress(client, pid)
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "transcribing"
    assert body["progress"] == 0.5
    assert body["duration"] == 60.0
    # Release the seeded entry first: the handler treats an id still in
    # flight as absent, so it would neither adopt nor pop it.
    tx_progress._BATCH_PROGRESS.pop(pid, None)
    post_verbose_ok(client, progress_id=pid)
    assert get_progress(client, pid).json() == {"stage": "unknown"}


def test_progress_updates_during_decode(client, app_module, fake_model):
    # The FakeModel yields segments through the handler's _collect wrapper —
    # snapshot the entry from inside transcribe() to see the live stage.
    pid = "c" * 32
    seen = {}
    orig = fake_model.transcribe

    def spy(path, **kwargs):
        segs, info = orig(path, **kwargs)
        entry = tx_progress._BATCH_PROGRESS.get(pid)
        seen.update(entry or {})
        return segs, info

    fake_model.transcribe = spy
    post_verbose_ok(client, progress_id=pid)
    # transcribe() runs inside the executor slot, after "waiting" gave way
    # to the "analyzing" stage (audio decode + VAD window).
    assert seen.get("stage") == "analyzing"
    assert pid not in tx_progress._BATCH_PROGRESS


def test_pid_bindings_of_a_same_id_repost_survive_the_first_run(
        client, app_module, monkeypatch):
    # The inner finally closes the progress entry, and (in the URL flow) the
    # rmtree is awaited before the outer finally pops the pid bindings; a
    # same-id re-post can claim the id in that gap and bind its own job /
    # plan / run plan. Simulated right at the close: the first run must
    # leave those bindings alone.
    pid = "b2" * 16
    foreign_plan = object()
    real_close = tx_progress._progress_close

    def _close_then_repost(p):
        real_close(p)
        if p == pid:
            tx_progress._JOB_BY_PID[pid] = "other-request"
            tx_progress._PLAN_BY_PID[pid] = "other-plan"
            tx_progress._RUN_PLAN_BY_PID[pid] = foreign_plan
    monkeypatch.setattr(tx_progress, "_progress_close", _close_then_repost)
    post_json_ok(client, progress_id=pid)
    assert tx_progress._JOB_BY_PID.get(pid) == "other-request"
    assert tx_progress._PLAN_BY_PID.get(pid) == "other-plan"
    assert tx_progress._RUN_PLAN_BY_PID.get(pid) is foreign_plan


def test_pid_bindings_are_popped_when_the_run_still_owns_them(client):
    pid = "b3" * 16
    post_json_ok(client, progress_id=pid)
    assert pid not in tx_progress._JOB_BY_PID
    assert pid not in tx_progress._PLAN_BY_PID
    assert pid not in tx_progress._RUN_PLAN_BY_PID


def test_job_finish_is_scoped_to_the_runs_request_id(
        client, app_module, monkeypatch):
    # The handler closes its progress entry before the job-row finish, so a
    # same-id re-post can own the row by then: the finish carries the run's
    # own request_id (the one the start stored) to leave that row alone.
    pid = "a1" * 16

    def _rid(p, **kw):
        return kw.get("request_id")
    starts = record_calls(monkeypatch, tx_progress, "_jobs_start_async", _rid)
    finishes = record_calls(monkeypatch, tx_progress, "_jobs_finish", _rid)
    post_json_ok(client, progress_id=pid)
    assert len(finishes) == 1
    assert starts[0] and finishes == starts


def test_progress_entry_is_closed_on_a_413_before_the_decode(
        client, app_module, monkeypatch):
    # The entry is seeded before the Content-Length gate; a failure there
    # never reaches the inner try, so the outer finally must close it.
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "MEDIA_MAX_BYTES", 10, raising=False)
    pid = "e" * 32
    r = _post(client, response_format="json", progress_id=pid)
    assert r.status_code == 413
    assert pid not in tx_progress._BATCH_PROGRESS
    assert pid not in tx_progress._PROGRESS_OWNER
    assert get_progress(client, pid).json() == {"stage": "unknown"}


def test_progress_entry_is_closed_when_the_model_load_fails(
        client, app_module, monkeypatch):
    fail_model_load(monkeypatch, HTTPException(status_code=400, detail="model not allowed"))
    pid = "f" * 32
    r = _post(client, response_format="json", progress_id=pid)
    assert r.status_code == 400
    assert pid not in tx_progress._BATCH_PROGRESS
    assert get_progress(client, pid).json() == {"stage": "unknown"}


def test_verbose_json_reports_duration_after_vad(client, fake_model):
    # VAD_FILTER defaults on → the response carries the VAD receipt, and it
    # is the VAD figure itself, not the audio duration.
    fake_model._info.duration_after_vad = 0.4
    body = post_verbose_ok(client)
    assert body["duration_after_vad"] == 0.4
    assert body["duration"] == 1.0


def test_verbose_json_omits_vad_receipt_when_filter_off(client):
    body = post_verbose_ok(client, decode_overrides='{"vad_filter": false}')
    assert "duration_after_vad" not in body


def test_progress_reports_skipped_separation(client, app_module, monkeypatch):
    # A requested-but-disabled optional stage is named in the progress entry
    # the moment it is declined, so a polling client marks the rail row
    # "skipped" live instead of inferring it; the final response still
    # carries the warning text.
    pid = "d" * 32
    seen = watch_progress(monkeypatch, pid)
    body = post_verbose_ok(client, separate_bgm="true", progress_id=pid)
    assert seen.get("skipped") == ["separating"]
    assert any("music separation requested" in w
               for w in (body.get("warnings") or []))


def test_progress_malformed_id_is_ignored_on_post(client, app_module):
    post_verbose_ok(client, progress_id="Nope!")
    assert "Nope!" not in tx_progress._BATCH_PROGRESS


def test_progress_registry_is_bounded(app_module):
    for i in range(tx_progress._BATCH_PROGRESS_MAX + 20):
        tx_progress._progress_set(f"{i:032x}", stage="waiting")
    assert len(tx_progress._BATCH_PROGRESS) == tx_progress._BATCH_PROGRESS_MAX
    tx_progress._BATCH_PROGRESS.clear()


def test_transcribing_row_bills_a_cold_whisper_load(client, app_module,
                                                    monkeypatch):
    """The transcribing row's `load` is measured from the perf origin of the
    model load itself, not from decode start (which is always later, so the
    row said `load 0.0s` for every stone-cold load). The load is folded into
    the row's `secs` so `run = secs - load` and the wall total both hold."""
    import time as _time
    # The harness default is "" (whisper-1 → no model), and a row without a
    # stats key gets no load column at all — give it a real one.
    monkeypatch.setattr(app_module.cfg, "DEFAULT_MODEL",
                        "Systran/faster-whisper-large-v3", raising=False)
    seen = {}

    async def _loader(name, *, lease=False):
        seen["loaded_wall"] = _time.time()
        return FakeModel()
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)

    def _load_secs_since(name, since_ts):
        # Behaves like the real registry: the load is visible only to a
        # stage whose origin precedes the moment the model was (re)loaded.
        return 3.0 if since_ts <= seen["loaded_wall"] else 0.0
    monkeypatch.setattr(model_registry, "load_secs_since", _load_secs_since)

    capture_receipt(monkeypatch, seen)

    assert _post(client).status_code == 200
    row = next(s for s in seen["stages"] if s["name"] == "transcribing")
    assert row["load_secs"] == 3.0
    assert row["secs"] >= 3.0


def test_record_transcription_carries_the_key_label(client, app_module,
                                                    monkeypatch, make_user_key):
    # The recent-jobs row snapshots the key's display label straight from
    # the auth record (labels are mutable; read-time resolution rewrites
    # history), so the handler passes it beside key_id.
    from faster_whisper_backend.auth import api_keys_store
    make_user_key("root", is_admin=True)            # lock the server down
    uid = api_keys_store.create_user("alice", is_admin=False)
    raw, _ = api_keys_store.create_key(uid, label="alice-laptop")
    recorded = record_runs(app_module, monkeypatch)
    post_json_ok(client, bearer(raw))
    assert recorded and recorded[-1]["key_label"] == "alice-laptop"


def test_upload_spool_carries_the_reclaim_prefix(client, app_module,
                                                 monkeypatch):
    # The upload temp file is named "whisperup-*" so the hard-restart sweep
    # (restart_service.reclaim_hard_restart_orphans) can tell it from every
    # other process's "tmp*" tempfile; tempfile's default prefix was
    # structurally unreclaimable.
    seen = record_calls(monkeypatch, app_module.tempfile, "NamedTemporaryFile",
                        lambda *a, **kw: kw.get("prefix"))
    post_json_ok(client)
    assert seen == ["whisperup-"]


def _ledger_row():
    from faster_whisper_backend.stats import recent_transcriptions_store
    rows = recent_transcriptions_store.list_recent(limit=5)
    return rows[0]


def test_cuda_oom_is_classified_on_the_ledger(client, app_module, monkeypatch):
    _serve_failing_decode(monkeypatch, "CUDA failed with error out of memory")
    r = _post(client, response_format="json")
    assert r.status_code == 500
    row = _ledger_row()
    assert row["status"] == "error"
    assert (row["error_class"], row["error_stage"]) == ("cuda_oom", "transcribing")
    from faster_whisper_backend.stats import usage_store
    job = usage_store._require_conn().execute(
        "SELECT error_class, error_stage, status FROM usage_jobs"
        " ORDER BY created_ts DESC LIMIT 1").fetchone()
    assert tuple(job) == ("cuda_oom", "transcribing", "error")


def test_unknown_failure_is_other_and_success_has_no_class(client, app_module,
                                                          monkeypatch):
    post_json_ok(client)
    ok = _ledger_row()
    assert ok["error_class"] is None and ok["error_stage"] is None

    _serve_failing_decode(monkeypatch, "decode blew up")
    assert _post(client, response_format="json").status_code == 500
    row = _ledger_row()
    assert (row["error_class"], row["error_stage"]) == ("other", "transcribing")


def test_curated_4xx_is_rejected_not_a_server_failure(client, app_module, monkeypatch):
    """A caller error that the handler turns into a 4xx must not count as a
    server failure class."""
    fail_model_load(monkeypatch, HTTPException(status_code=400, detail="model not allowed"))
    assert _post(client, response_format="json").status_code == 400
    row = _ledger_row()
    assert row["status"] == "error" and row["error_class"] == "rejected"


def test_progress_carries_plan_overall_and_eta(client, app_module):
    """A polled entry with a run plan behind it publishes the plan, the
    overall fraction and the ETA; an entry without one (the admin prompt
    lab's) answers None for all three, never a KeyError."""
    pid = "e" * 32
    plan = run_plan.RunPlan(kind="file")
    plan.set_stages(["transcribing", "translating"])
    plan.set_audio_seconds(600.0, src="decoder")
    plan.set_translation(["fr"], model="m", device="cuda", mode="fluent")
    plan.set_segments(160)
    tx_progress._RUN_PLAN_BY_PID[pid] = plan
    try:
        tx_progress._progress_set(pid, stage="transcribing", progress=0.5)
        body = get_progress(client, pid).json()
    finally:
        tx_progress._RUN_PLAN_BY_PID.pop(pid, None)
        tx_progress._BATCH_PROGRESS.pop(pid, None)
    assert [s["stage"] for s in body["plan"]] == ["transcribing", "translating"]
    assert body["plan"][0]["state"] == "active"
    assert body["plan"][1]["units"][0]["target"] == "fr"
    assert 0.0 < body["overall"] < 1.0
    assert body["eta_s"] > 0
    pid2 = "f" * 32
    tx_progress._progress_set(pid2, stage="translating", progress=0.1)
    try:
        body = get_progress(client, pid2).json()
    finally:
        tx_progress._BATCH_PROGRESS.pop(pid2, None)
    assert body["plan"] is None and body["overall"] is None
    assert body["eta_s"] is None


def test_verbose_json_carries_plan_receipt_and_entry_is_gone(client,
                                                             app_module):
    """The response carries the plan's receipt (every stage done with its
    took_s) and the plan registry is cleaned with the progress entry."""
    pid = "a1" * 16
    plan = post_verbose_ok(client, progress_id=pid)["plan"]
    assert [s["stage"] for s in plan] == ["transcribing"]
    assert plan[0]["state"] == "done"
    assert plan[0]["took_s"] >= 0.0
    # (The test config's DEFAULT_MODEL is empty, so no model key rides.)
    assert pid not in tx_progress._RUN_PLAN_BY_PID
    # The plain json shape stays the OpenAI-compatible object.
    assert "plan" not in post_json_ok(client)


def test_plan_receipt_is_clocked_without_a_progress_id(client, app_module,
                                                      fake_model,
                                                      monkeypatch):
    """A plain OpenAI-style POST sends no progress_id, so nothing publishes
    progress — but the run plan must still be ticked, or every stage closes
    with took_s 0.0 and the rates ledger never learns from API callers."""
    import time
    from faster_whisper_backend.runtime import stage_rates
    monkeypatch.setattr(run_plan, "_MIN_SAMPLE_S", 0.0)
    orig = fake_model.transcribe

    def slow(path, **kw):
        segs, info = orig(path, **kw)

        def _gen():
            time.sleep(0.05)
            yield from segs
        return _gen(), info
    monkeypatch.setattr(fake_model, "transcribe", slow)
    plan = {s["stage"]: s for s in post_verbose_ok(client)["plan"]}
    assert plan["transcribing"]["state"] == "done"
    assert plan["transcribing"]["took_s"] > 0.0
    assert any(k.startswith("transcribing|") for k in stage_rates._read())


def test_plan_reports_skipped_separation(client, app_module):
    """A requested-but-declined stage is in the plan as `skipped`, not
    dropped: the client's rail still gets a row to explain."""
    seen = {}
    pid = "b2" * 16
    orig = tx_progress._progress_set

    def spy(p, **fields):
        orig(p, **fields)
        if p == pid and p in tx_progress._RUN_PLAN_BY_PID:
            seen["snap"] = tx_progress._RUN_PLAN_BY_PID[p].snapshot()

    tx_progress._progress_set = spy
    try:
        body = post_verbose_ok(client, separate_bgm="true", progress_id=pid)
    finally:
        tx_progress._progress_set = orig
    states ={s["stage"]: s["state"] for s in seen["snap"]["plan"]}
    assert states["separating"] == "skipped"
    # The last tick lands mid-decode; the receipt in the response is final.
    assert states["transcribing"] in ("active", "done")
    receipt = {s["stage"]: s["state"] for s in body["plan"]}
    assert receipt == {"separating": "skipped", "transcribing": "done"}


def test_translate_task_cascades_from_english_text(client, app_module,
                                                   fake_model, monkeypatch):
    # task=translate decodes to ENGLISH while info.language stays the spoken
    # one: the T2T stage must be told "en", or translate_to=de on German
    # audio short-circuits as same-language and copies the English text.
    enable_translation(app_module, monkeypatch)
    calls = stub_translate(monkeypatch)
    assert fake_model._info.language == "de"
    body = post_json_ok(client, task="translate", translate_to="de")
    assert calls[-1]["source_lang"] == "en"
    assert body["translations"] == {"de": "hallo welt-de"}
    assert body["translation"]["source"] == "en"


def test_translate_task_scopes_the_pipeline_by_the_output_language(
        client, fake_model):
    # German audio, English output: the de-* pipeline rules (sentence-start
    # lowercasing of German wordlist hits like "Was") must not run on it.
    fake_model._segments = [
        FakeSegment("It is over. Was it good? In fact, yes.", 0.0, 1.0)]
    assert "Was it good" in post_json_ok(client, task="translate")["text"]
    # Same text, transcribe task: the de rules do apply (the guard above is
    # only meaningful while they would).
    assert "Was it good" not in post_json_ok(client)["text"]


def test_huge_translate_to_field_is_walked_and_echoed_bounded(client, app_module,
                                                              monkeypatch):
    # A multipart field can carry ~1 MiB of codes: the walk stops early and
    # the over-MAX_TARGETS warning never echoes the whole field back.
    enable_translation(app_module, monkeypatch)
    stub_translate(monkeypatch)
    codes =",".join(f"aa-{i:05d}" for i in range(50_000))
    body = post_verbose_ok(client, translate_to=codes)
    dropped = [w for w in body.get("warnings", []) if "were dropped" in w]
    assert len(dropped) == 1
    assert len(dropped[0]) < 1024


def test_ffconcat_upload_is_refused_before_any_decoder(client, app_module,
                                                       fake_model, tmp_path,
                                                       monkeypatch):
    # faster-whisper's decode_audio (the lead-pad pre-decode and
    # model.transcribe) opens the spool with no protocol whitelist or format
    # check: an extensionless ffconcat naming a sibling decoded that sibling.
    import tempfile

    import faster_whisper.audio
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    write_wav(str(tmp_path / "secret.wav"), b"\x00\x00" * 16000)
    decoded = []
    monkeypatch.setattr(faster_whisper.audio, "decode_audio",
                        lambda *a, **k: decoded.append(a))
    r = _post(client, files={"file": ("upload", b"ffconcat version 1.0\n"
                                      b"file 'secret.wav'\n",
                                      "application/octet-stream")})
    assert r.status_code == 400, r.text
    assert fake_model.last_audio is None
    assert decoded == []


def test_upload_with_no_audio_stream_is_a_400_not_a_server_fault(
        client, fake_model, tmp_path):
    # A video-only clip opens fine, but faster-whisper's decode_audio then
    # raises a bare IndexError: a 500 a client reads as a retryable fault.
    import av
    import numpy as np
    clip = tmp_path / "v.mp4"
    with av.open(str(clip), "w") as c:
        s = c.add_stream("mpeg4", rate=5)
        s.width = s.height = 16
        s.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), np.uint8), format="rgb24")
            for pk in s.encode(frame):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    r = _post(client, files={"file": ("v.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "source has no audio stream"
    assert fake_model.last_audio is None


def test_a_refused_bgm_transcode_never_hands_the_original_to_the_separator(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    calls = []

    async def _separate(path, **kw):
        calls.append(path)
        raise AssertionError("separator reached")
    monkeypatch.setattr(bgm_separation, "separate", _separate)

    def _refuse(src, dst, *, rate=44100, layout="stereo"):
        raise ValueError("unsupported container")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _refuse)
    body = post_verbose_ok(client, separate_bgm="true",
                           files={"file": ("a.mp3", b"ID3xxxx", "audio/mpeg")})
    assert calls == []
    assert any("music separation skipped" in w
               for w in body.get("warnings") or [])


def test_a_pyav_error_in_the_bgm_transcode_falls_back_to_the_original(
        client, app_module, monkeypatch):
    """PyAV's errors subclass ValueError too, but only _open_audio's own
    ValueError is a refusal: any other av error keeps the fallback (the
    separator decodes the original) instead of skipping separation."""
    import av
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    calls = []

    async def _separate(path, **kw):
        calls.append(path)
        raise bgm_separation.BgmSeparationError("stub separator")
    monkeypatch.setattr(bgm_separation, "separate", _separate)

    def _invalid(src, dst, *, rate=44100, layout="stereo"):
        raise av.error.InvalidDataError(1094995529, "Invalid data",
                                        "avcodec_send_packet()")
    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _invalid)
    body = post_verbose_ok(client, separate_bgm="true",
                           files={"file": ("a.mp3", b"ID3xxxx", "audio/mpeg")})
    assert len(calls) == 1 and not calls[0].endswith(".wav")
    assert not any("could not be prepared" in w
                   for w in body.get("warnings") or [])


def _record_captures_on(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_SAMPLE_RATE", 1.0,
                        raising=False)


def test_a_full_capture_store_still_records_a_batch_request(
        client, app_module, fake_model, monkeypatch, created_captures):
    """At CAPTURES_MAX the batch gate must still reach create_capture so its
    _evict_to_cap rotates oldest-first; refusing up front means no later
    request records anything (streaming twin: tests/streaming/test_finalize_paths.py)."""
    _record_captures_on(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_MAX", 10, raising=False)
    monkeypatch.setattr(captures_store, "count_evictable", lambda: 10)
    # Not a store full of "ready" rows (the real DB state must not decide).
    monkeypatch.setattr(captures_store, "ready_fills_cap", lambda: False)

    post_json_ok(client)
    assert len(created_captures) == 1
    assert fake_model.last_kwargs["word_timestamps"] is True   # forced DTW


def test_a_store_full_of_ready_captures_skips_the_batch_capture(
        client, app_module, fake_model, monkeypatch, created_captures):
    """Ready rows alone fill CAPTURES_MAX: _evict_to_cap would evict the new
    row itself, so the gate records nothing and does not force DTW."""
    _record_captures_on(app_module, monkeypatch)
    monkeypatch.setattr(captures_store, "ready_fills_cap", lambda: True)

    post_json_ok(client)
    assert created_captures == []
    # The costly half: a json response with no timestamp_granularities runs
    # without DTW once the gate declines.
    assert not fake_model.last_kwargs.get("word_timestamps")


def test_batch_capture_gate_reads_the_store_off_the_event_loop(
        client, app_module, fake_model, monkeypatch):
    """ready_fills_cap queries the shared captures connection, which clear_all
    holds across a full VACUUM: called on the loop, that VACUUM would freeze
    every request and stream. It must run in a worker thread."""
    import asyncio

    _record_captures_on(app_module, monkeypatch)
    on_loop = []

    def _probe():
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return True
    monkeypatch.setattr(captures_store, "ready_fills_cap", _probe)

    post_json_ok(client)
    assert on_loop == [False]


def _retain_on(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "MEDIA_PACKAGE_ENABLED", True,
                        raising=False)


def test_retain_media_over_the_store_cap_is_declined_with_a_warning(
        client, app_module, monkeypatch):
    """A store cap below MEDIA_MAX_BYTES: an upload the store would drop at
    once is never copied or registered, and the response says why instead
    of silently omitting source_media_id."""
    _retain_on(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "RETAINED_MEDIA_MAX_BYTES", 4,
                        raising=False)
    calls = []
    monkeypatch.setattr(media_store, "register",
                        lambda *a, **kw: calls.append("register"))
    monkeypatch.setattr(media_store, "make_pipeline_copy",
                        lambda *a, **kw: calls.append("copy"))
    body = post_json_ok(client, retain_media="true")
    assert "source_media_id" not in body
    assert any("RETAINED_MEDIA_MAX_BYTES" in w for w in body["warnings"])
    assert calls == []


@pytest.mark.parametrize("fmt", ["text", "srt", "vtt"])
def test_retain_media_with_a_body_that_cannot_name_it_is_refused(
        client, app_module, monkeypatch, fmt):
    """A plain-text or subtitle body has no field for source_media_id: the
    retained copy could never be addressed, so no copy is staged at all."""
    _retain_on(app_module, monkeypatch)
    calls = []
    monkeypatch.setattr(media_store, "make_pipeline_copy",
                        lambda *a, **kw: calls.append("copy"))
    r = _post(client, response_format=fmt, retain_media="true")
    assert r.status_code == 422
    assert r.json()["detail"] == "retain_media requires a json response format"
    assert calls == []


def test_retain_media_refused_by_register_says_so(client, app_module, monkeypatch):
    _retain_on(app_module, monkeypatch)
    monkeypatch.setattr(media_store, "register", lambda *a, **kw: None)
    body = post_json_ok(client, retain_media="true")
    assert "source_media_id" not in body
    assert any(w.startswith("retain_media:") for w in body["warnings"])


def test_retain_media_staging_failure_says_so(client, app_module, monkeypatch):
    """make_pipeline_copy returns None when both the hardlink and the copy
    fail: the register block never runs, so the response must say why
    instead of silently omitting source_media_id."""
    _retain_on(app_module, monkeypatch)
    calls = []
    monkeypatch.setattr(media_store, "register",
                        lambda *a, **kw: calls.append("register"))
    monkeypatch.setattr(media_store, "make_pipeline_copy",
                        lambda *a, **kw: None)
    body = post_json_ok(client, retain_media="true")
    assert "source_media_id" not in body
    assert any(w.startswith("retain_media:") and "could not stage" in w
               for w in body["warnings"])
    assert calls == []
