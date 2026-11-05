"""The batch /v1/audio/transcriptions path honours per-identity config.
Asserts the kwargs the FakeModel receives + the overrides_ignored
feedback. Driven through the real app; no faster-whisper needed."""

import json
from types import SimpleNamespace

from tests.conftest import bearer, bind_user, first_key_id, locked_user
from tests._translation_helpers import enable_translation, stub_translate
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import receipt as tx_receipt
from faster_whisper_backend.translation import gating as tr_gating
from tests._transcription_helpers import (
    bind_profile, post_json_ok, post_transcription, post_verbose_ok)
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _setup_profile(client, admin_h, name, **fields):
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def _bind_profile_user(client, make_user_key, profile, user="alice", **fields):
    """Admin saves override profile `profile` and binds it to a fresh
    non-admin `user`; returns that user's raw key."""
    return bind_profile(client, make_user_key, profile, fields, user=user)[1]


def test_identity_profile_applies_to_decode_kwargs(client, make_user_key, fake_model):
    raw_alice = _bind_profile_user(client, make_user_key, "p", BEAM_SIZE=8,
                                   BEST_OF=5, locks=["BEAM_SIZE"])

    body = post_verbose_ok(client, bearer(raw_alice),
                           decode_overrides=json.dumps({"beam_size": 20, "best_of": 3}))
    kw = fake_model.last_kwargs
    assert kw["beam_size"] == 8        # profile value; client 20 dropped (locked)
    assert kw["best_of"] == 3          # unlocked → client override wins
    assert body["overrides_ignored"] == ["beam_size"]


def test_locked_language_ignores_client_param(client, make_user_key, fake_model):
    raw_alice = _bind_profile_user(client, make_user_key, "de", DEFAULT_LANGUAGE="de",
                                   locks=["DEFAULT_LANGUAGE"])

    body = post_verbose_ok(client, bearer(raw_alice), language="fr")
    assert fake_model.last_kwargs["language"] == "de"      # locked → client 'fr' ignored
    assert "language" in body["overrides_ignored"]


def test_batch_language_reduces_to_a_whisper_code(client, make_user_key, fake_model):
    # /dictate sends one language box to both modes and the streaming handshake
    # folds it to a Whisper code, so batch must too: faster-whisper's tokenizer
    # raises (a 500) on a region tag, an underscore locale, a withdrawn
    # ISO 639-1 code ("iw", "in") or the standard "jv" (Whisper keeps "jw").
    # Blank is auto-detect; a value that reduces to no code is a 422 (below).
    _, raw = make_user_key("alice", is_admin=False)
    for sent, expected in (("DE", "de"), ("de-DE", "de"), (" pt-BR ", "pt"),
                           ("de_DE", "de"), ("pt_BR", "pt"),
                           ("iw_IL", "he"), ("in-ID", "id"), ("iw", "he"),
                           ("JI", "yi"), ("jv", "jw"),
                           ("haw", "haw"), ("  ", None)):
        r = post_transcription(client, bearer(raw), language=sent)
        assert r.status_code == 200, (sent, r.text)
        assert fake_model.last_kwargs["language"] == expected, sent


def test_unknown_language_is_a_422_before_the_decode(client, make_user_key,
                                                     fake_model):
    # A value that reduces to no Whisper code is the caller's input: a 422
    # naming the code on both routes, never a tokenizer 500 from the decode,
    # and the model is never called.
    _, raw = make_user_key("alice", is_admin=False)
    for route in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        for sent in ("xx", "auto", "english"):
            fake_model.last_kwargs = {}
            r = post_transcription(client, bearer(raw), route=route, language=sent)
            assert r.status_code == 422, (route, sent, r.text)
            assert repr(sent) in r.json()["detail"], (route, sent)
            assert fake_model.last_kwargs == {}, (route, sent)


def test_locked_language_still_ignores_an_unknown_client_code(
        client, make_user_key, fake_model):
    # A locked DEFAULT_LANGUAGE drops the client's field, so a bad code there
    # is ignored like any other value, not refused.
    raw_alice = _bind_profile_user(client, make_user_key, "de", DEFAULT_LANGUAGE="de",
                                   locks=["DEFAULT_LANGUAGE"])
    body = post_verbose_ok(client, bearer(raw_alice), language="xx")
    assert fake_model.last_kwargs["language"] == "de"
    assert "language" in body["overrides_ignored"]


def test_no_identity_config_is_unchanged(client, make_user_key, fake_model):
    # A user with no binding decodes with the global defaults + the client
    # override applied (no lock) and no overrides_ignored field.
    _, raw_alice = locked_user(make_user_key)
    body = post_verbose_ok(client, bearer(raw_alice), decode_overrides=json.dumps({"beam_size": 7}))
    assert fake_model.last_kwargs["beam_size"] == 7        # client override applies
    assert "overrides_ignored" not in body


def test_stream_only_keys_never_reach_batch(client, make_user_key, fake_model):
    # Live-dictation keys ride decode_overrides like every client key, but a
    # batch request neither applies them (no model.transcribe kwarg) nor
    # reports a locked one as ignored — it was never a batch knob.
    raw_alice = _bind_profile_user(client, make_user_key, "p",
                                   STREAMING_VAD_OUTER_SILENCE_MS=1500,
                                   locks=["STREAMING_VAD_OUTER_SILENCE_MS"])
    body = post_verbose_ok(client, bearer(raw_alice), decode_overrides=json.dumps({
        "streaming_vad_outer_silence_ms": 3000,
        "streaming_vad_threshold": 0.3,
        "streaming_hard_break_separator": "\n"}))
    assert not any(k.startswith("streaming_") for k in fake_model.last_kwargs)
    assert "overrides_ignored" not in body


def test_decode_overrides_drop_non_finite_floats():
    """JSON permits NaN/Infinity literals; a non-finite float override is dropped
    (ignored) rather than clamped to the field's bound, matching the integer path.
    Shared by the batch route and the streaming FINAL decode via
    _apply_decode_overrides."""
    assert tx_models._apply_decode_overrides({}, "whisper-1", {"temperature": 0.7})["temperature"] == 0.7
    for literal in ("NaN", "Infinity", "-Infinity"):
        ov = json.loads('{"temperature": %s, "no_speech_threshold": %s}' % (literal, literal))
        kw = tx_models._apply_decode_overrides({}, "whisper-1", ov)
        assert "temperature" not in kw and "no_speech_threshold" not in kw, (literal, kw)


def test_client_temperature_can_be_a_retry_ladder():
    """A client temperature is a number or a ladder (list / comma string, the
    TEMPERATURE format): rungs clamped to [0, 1], at most 16, one rung stays a
    float, several become faster-whisper's tuple; unparseable or empty input
    is dropped (the configured value stands)."""

    def t(v):
        return tx_models._apply_decode_overrides({"temperature": 0.3}, "whisper-1",
                                            {"temperature": v})["temperature"]
    assert t(0.7) == 0.7
    assert t("0.4") == 0.4
    assert t("0, 0.4, 0.8") == (0.0, 0.4, 0.8)
    assert t([0, 0.5, 1]) == (0.0, 0.5, 1.0)
    assert t("0,2,-1") == (0.0, 1.0, 0.0)                 # clamped rung by rung
    assert t([0.1] * 40) == tuple([0.1] * 16)              # at most 16 rungs
    assert t(",".join(["0.1"] * 40)) == tuple([0.1] * 16)  # ...of a string too
    for junk in ("", ",", "0,abc", [], [0, "x"], [0, None], "nan", {}, None):
        assert t(junk) == 0.3, junk


def test_decode_overrides_parse_bool_strings_instead_of_truthiness():
    """A bool decode key sent as a string ("false" from a shell/form client)
    means what it says; bool("false") is True. Unparseable values are dropped
    (the configured value stands), like the clamp paths."""
    base = {"vad_filter": True, "vad_parameters": {"threshold": 0.5},
            "condition_on_previous_text": True}
    kw = tx_models._apply_decode_overrides(dict(base), "whisper-1", {
        "vad_filter": "false", "condition_on_previous_text": "False",
        "multilingual": " off "})
    assert kw["vad_filter"] is False and kw["vad_parameters"] is None
    assert kw["condition_on_previous_text"] is False
    assert kw["multilingual"] is False
    kw = tx_models._apply_decode_overrides({"vad_filter": False}, "whisper-1", {
        "vad_filter": "true", "multilingual": 1})
    assert kw["vad_filter"] is True and kw["multilingual"] is True
    for junk in ("maybe", "", 2, [], {}):
        kw = tx_models._apply_decode_overrides(dict(base), "whisper-1", {
            "vad_filter": junk, "condition_on_previous_text": junk})
        assert kw["vad_filter"] is True and kw["condition_on_previous_text"] is True, junk
    # "false" multilingual with a language set is honoured, not reported ignored
    ignored = []
    tx_models._note_auto_detect_only({"multilingual": "false"}, "de", ignored)
    assert ignored == []


def test_decode_overrides_drop_overflowing_suppress_tokens():
    """A suppress_tokens override whose member overflows int() is dropped, not
    500'd: a JSON 1e999 / Infinity literal parses to float('inf') and int(inf)
    raises OverflowError. Covers the batch route and the streaming FINAL decode
    via the shared _apply_decode_overrides."""
    # a valid list still applies (float members truncate through int())
    assert tx_models._apply_decode_overrides({}, "whisper-1",
        {"suppress_tokens": [1, 2]})["suppress_tokens"] == [1, 2]
    assert tx_models._apply_decode_overrides({}, "whisper-1",
        {"suppress_tokens": [1.9, 2.0]})["suppress_tokens"] == [1, 2]
    for st in ([json.loads("1e999")], [float("inf")], [float("-inf")],
               [float("nan")], [None], [{}], "1,nan"):
        kw = tx_models._apply_decode_overrides({}, "whisper-1", {"suppress_tokens": st})
        assert "suppress_tokens" not in kw, (st, kw)


_SEG_DIAG = [{"id": 0, "start": 0.0, "end": 1.0, "alp": -0.1, "nsp": 0.01,
              "cr": 1.2, "temp": 0.0, "text": "hi"}]


def _info(language="de", prob=0.99):
    return SimpleNamespace(language=language, language_probability=prob,
                           duration=1.0, duration_after_vad=1.0)


def _block(raw="hi", final="hi", *, info=None, model_name="whisper-1", kwargs=None, **kw):
    return tx_receipt._format_request_block(
        file_label="x", model_name=model_name, info=info or _info(), kwargs=kwargs or {},
        seg_diag=_SEG_DIAG, raw=raw, final=final, **kw)


def test_request_block_identity_section():
    ident = SimpleNamespace(layers=["user.profile:clinic-de"], locked={"BEAM_SIZE"},
                            profiles_applied=["clinic-de"])
    block = _block(kwargs={"beam_size": 8}, ident=ident, overrides_ignored=["beam_size"])
    assert "Identity" in block
    assert "clinic-de" in block and "BEAM_SIZE" in block
    assert "overrides_ignored" in block and "beam_size" in block
    # no identity + nothing ignored → no Identity section (logs stay terse)
    empty = SimpleNamespace(layers=[], locked=set(), profiles_applied=[])
    block2 = _block(ident=empty)
    assert "Identity" not in block2


def test_request_block_pipeline_header_counts_only_changed_steps():
    """The PIPELINE header must count only steps that rewrote the text
    (before != after) as "changed" — skipped/EXCLUDED no-op rows are
    "unchanged", the same split the /quick-config + /reports viewers make of
    the same trace."""
    info = _info("en", 1.0)
    # 7 no-op (EXCLUDED) rows + 1 that actually trimmed → 1 changed, 7 unchanged.
    steps = [(f"#{i} excluded [EXCLUDED for m]", " hi", " hi") for i in range(7)]
    steps.append(("#18 Trim edges", " hi", "hi"))
    block = _block(" hi", info=info, model_name="m", steps=steps)
    assert "PIPELINE  (1 step changed text, 7 unchanged)" in block
    assert "8 step" not in block

    # All-changed → no "unchanged" tail, plural "steps".
    block2 = _block("a", "c", info=info, model_name="m",
                    steps=[("#1 x", "a", "b"), ("#2 y", "b", "c")])
    assert "PIPELINE  (2 steps changed text)" in block2
    assert "unchanged" not in block2

    # Nothing changed (every rule excluded) → "0 steps changed text, N unchanged".
    block3 = _block(info=info, model_name="m",
                    steps=[("#1 x", "hi", "hi"), ("#2 y", "hi", "hi")])
    assert "PIPELINE  (0 steps changed text, 2 unchanged)" in block3


def test_request_block_identity_always_shows_user_even_without_layers():
    """An authenticated caller with NO per-identity binding still gets an
    Identity block naming them + an explicit 'inherits' note — so a missing
    binding (the classic 'my override didn't apply') is visible in the log."""
    empty = SimpleNamespace(layers=[], locked=set(), profiles_applied=[])
    block = _block(ident=empty, user_id="abcd1234ef", key_id="ffee0011bb", username="Admin")
    assert "Identity" in block
    assert "Admin" in block and "abcd1234" in block      # username + short user id
    assert "ffee0011" in block                            # short key id
    assert "inherits" in block                            # the "(none — inherits…)" note


def test_config_version_bumps_on_binding_and_profile_changes(client, make_user_key, root_key):
    """Saving a profile or a per-user / per-key binding bumps settings.version's
    version counter — the signal a live streaming connection polls to know it
    must re-resolve its ident (so edits apply without a reconnect)."""
    from faster_whisper_backend.settings import version as settings_version
    admin_h = bearer(root_key)
    v0 = settings_version.config_version()
    _setup_profile(client, admin_h, "pv", BEAM_SIZE=6)
    v1 = settings_version.config_version()
    assert v1 > v0                                         # profile save bumps
    uid, _ = make_user_key("bob", is_admin=False)
    bind_user(client, admin_h, uid, profiles=["pv"])
    v2 = settings_version.config_version()
    assert v2 > v1                                         # per-user binding bumps
    kid = first_key_id(client, admin_h, uid)
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=admin_h,
                     json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})
    assert r.status_code == 200, r.text
    v3 = settings_version.config_version()
    assert v3 > v2                                         # per-key binding bumps


def test_per_key_override_beats_user(client, make_user_key, fake_model, root_key):
    admin_h = bearer(root_key)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    bind_user(client, admin_h, uid, overrides={"BEAM_SIZE": 8})
    kid = first_key_id(client, admin_h, uid)
    client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=admin_h,
                 json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})

    post_json_ok(client, bearer(raw_alice))
    assert fake_model.last_kwargs["beam_size"] == 4       # key.direct wins over user.direct


def test_locked_blank_temperature_ignores_client_form_field(client, make_user_key,
                                                            fake_model):
    """A LOCKED TEMPERATURE has to bind the OpenAI-compat `temperature` Form
    field the way a locked DEFAULT_LANGUAGE binds `language`. A non-empty
    resolved ladder masks the field anyway; blank the ladder at the winning
    layer (a supported shape — value-less locks are documented) and the Form
    field was the one way past the lock."""
    raw_alice = _bind_profile_user(client, make_user_key, "t0", TEMPERATURE="",
                                   locks=["TEMPERATURE"])

    body = post_verbose_ok(client, bearer(raw_alice), temperature="0.97")
    assert fake_model.last_kwargs["temperature"] == 0.0     # client 0.97 dropped
    assert "temperature" in body["overrides_ignored"]


def test_unlocked_blank_temperature_still_takes_the_client_form_field(
        client, make_user_key, fake_model, app_module):
    # Same blank-ladder shape, no lock: the client's Form value must still win.
    # (With the shipped non-empty ladder the field is masked either way.)
    app_module.cfg.TEMPERATURE = ""
    _, raw_alice = make_user_key("alice", is_admin=False)
    post_verbose_ok(client, bearer(raw_alice), temperature="0.5")
    assert fake_model.last_kwargs["temperature"] == 0.5


def test_locked_task_ignores_client_param(client, make_user_key, fake_model):
    # A profile that locks TASK (value-less lock pins the inherited value,
    # global default "transcribe") forbids the client's `task` form field —
    # the same shape as the DEFAULT_LANGUAGE lock above.
    raw_alice = _bind_profile_user(client, make_user_key, "notask", locks=["TASK"])

    body = post_verbose_ok(client, bearer(raw_alice), task="translate")
    assert "task" not in fake_model.last_kwargs      # pinned to transcribe
    assert "task" in body["overrides_ignored"]
    assert body["task"] == "transcribe"


def test_locked_diarize_ignores_client_param(client, app_module, make_user_key,
                                             fake_model, monkeypatch):
    # A profile locking DIARIZE (value-less lock pins the inherited global,
    # default false) forbids the client's `diarize` form field.
    calls = _record_diarize(monkeypatch, app_module)
    raw_alice = _bind_profile_user(client, make_user_key, "nodiar",
                                   locks=["DIARIZE"])

    body = post_verbose_ok(client, bearer(raw_alice), diarize="true")
    assert calls == []                           # pinned off
    assert "diarize" in body["overrides_ignored"]


def _record_diarize(monkeypatch, app_module) -> list:
    from faster_whisper_backend.audio import diarization

    calls = []

    async def _fake_diarize(path, *, num_speakers=None, min_speakers=None,
                            max_speakers=None, model_id=None, progress_cb=None,
                            cancel_check=None):
        calls.append({"num": num_speakers, "min": min_speakers,
                      "max": max_speakers})
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake_diarize)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    return calls


def test_explicit_speaker_range_beats_inherited_num_speakers(
        client, app_module, fake_model, monkeypatch):
    # An unlocked DIARIZATION_NUM_SPEAKERS default must not erase the bounds
    # the client sent: absent inherits, explicit wins.
    calls = _record_diarize(monkeypatch, app_module)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_NUM_SPEAKERS", 2)
    post_verbose_ok(client, diarize="true", min_speakers="3", max_speakers="6")
    assert calls == [{"num": None, "min": 3, "max": 6}]


def test_locked_num_speakers_reports_dropped_client_bounds(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = _record_diarize(monkeypatch, app_module)
    raw_alice = _bind_profile_user(client, make_user_key, "twospk",
                                   DIARIZATION_NUM_SPEAKERS=2,
                                   locks=["DIARIZATION_NUM_SPEAKERS"])

    body = post_verbose_ok(client, bearer(raw_alice), diarize="true", min_speakers="3",
                           max_speakers="6")
    assert calls == [{"num": 2, "min": None, "max": None}]
    ignored = body["overrides_ignored"]
    assert "min_speakers" in ignored and "max_speakers" in ignored


def test_echoing_a_pinned_translation_model_passes_the_allowlist_gate(
        client, app_module, make_user_key, monkeypatch):
    """A client that ECHOES (or is locked to) the identity-effective
    TRANSLATION_MODEL has not chosen anything: it must pass the allowlist
    gate exactly like a request that sent no model, not get a 400 on
    /v1/text/translations."""
    calls = stub_translate(monkeypatch)
    enable_translation(app_module, monkeypatch,
                       TRANSLATION_ALLOWED_MODELS={"org/public-GGUF:Q4"},
                       TRANSLATION_DEFAULT_MODEL="org/public-GGUF:Q4")
    raw_bob = _bind_profile_user(client, make_user_key, "pinned-mt", user="bob",
                                 TRANSLATION_MODEL="org/pinned-GGUF:Q4",
                                 locks=["TRANSLATION_MODEL"])
    body = {"segments": [{"id": 0, "text": "Hallo"}], "targets": ["en"],
            "translation_model": "org/pinned-GGUF:Q4"}
    r = client.post("/v1/text/translations", headers=bearer(raw_bob), json=body)
    assert r.status_code == 200, r.text
    assert [c["model_ref"] for c in calls] == ["org/pinned-GGUF:Q4"]
    # Unit-level: the helper admits the inherited value, still refuses a
    # genuinely different client choice.
    allowed = tr_gating._translation_model_allowed
    assert allowed("org/pinned-GGUF:Q4", requested="org/pinned-GGUF:Q4",
                   inherited="org/pinned-GGUF:Q4") is True
    assert allowed("org/evil-GGUF:Q8", requested="org/evil-GGUF:Q8",
                   inherited="org/pinned-GGUF:Q4") is False


def test_multilingual_with_a_set_language_is_reported_ignored(
        client, app_module, fake_model, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_LANGUAGE", "", raising=False)
    overrides = json.dumps({"multilingual": True})
    body = post_verbose_ok(client, decode_overrides=overrides)
    assert fake_model.last_kwargs["multilingual"] is True
    assert "overrides_ignored" not in body
    body = post_verbose_ok(client, decode_overrides=overrides, language="de")
    assert "multilingual" not in fake_model.last_kwargs
    assert body["overrides_ignored"] == ["multilingual"]


def test_client_output_wrappers_in_batch(client, app_module):
    app_module.cfg.OUTPUT_PREFIX = "[cfg] "
    r = post_transcription(client, response_format="text",
                           decode_overrides=json.dumps({"output_prefix": ">> ",
                                                        "output_suffix": " <<"}))
    assert r.status_code == 200, r.text
    assert r.text == ">> hallo welt <<"
    # "" is an explicit "no prefix" over the configured one.
    r = post_transcription(client, response_format="text",
                           decode_overrides=json.dumps({"output_prefix": ""}))
    assert r.text == "hallo welt"


def test_batch_decode_overrides_drop_lone_surrogates(client, fake_model):
    """decode_overrides can carry a lone UTF-16 surrogate (half an emoji a
    client cut with a UTF-16 slice). In an output wrapper it made the success
    response fail to encode — a 500 after the whole file was decoded — and in
    hotwords it reached the tokenizer. Dropped, as the streaming twin does."""
    # json.dumps escapes the surrogate as "\\ud83d", as a browser sends it.
    body = post_json_ok(client,
                        decode_overrides=json.dumps({"output_prefix": ">> \ud83d",
                                                     "hotwords": "Kardiologie \ud83d"}))
    assert body["text"] == ">> hallo welt"
    assert fake_model.last_kwargs["hotwords"] == "Kardiologie "


def test_client_output_wrappers_master_gate_off(client, app_module):
    app_module.cfg.OUTPUT_PREFIX = "[cfg] "
    app_module.cfg.ALLOW_REQUEST_DECODE_OVERRIDES = False
    body = post_verbose_ok(client,
                           decode_overrides=json.dumps({"output_prefix": ">> "}))
    assert body["text"] == "[cfg] hallo welt"
    assert body["overrides_ignored"] == ["output_prefix"]


def test_batch_reports_hallucination_threshold_without_word_timestamps(
        client, fake_model):
    body = post_json_ok(client,
                        decode_overrides=json.dumps({"hallucination_silence_threshold": 2}))
    assert fake_model.last_kwargs["word_timestamps"] is False
    assert body["overrides_ignored"] == ["hallucination_silence_threshold"]
