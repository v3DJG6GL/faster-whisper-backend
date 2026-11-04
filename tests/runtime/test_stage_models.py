"""Per-request stage models (diarization_model / separation_model), the
optional-stage startup preloads, and translation eviction-on-edit. All heavy
deps stay uninstalled — stage modules are stubbed at the same handler
boundaries as tests/audio/test_diarization.py / test_bgm_separation.py."""

import asyncio
import os
import tempfile
import time

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.translation import engine as translation
from faster_whisper_backend.translation import gating as tr_gating
from faster_whisper_backend.transcription import progress as tx_progress
from tests.conftest import bearer
from tests._transcription_helpers import bind_profile as _bind_profile
from tests._transcription_helpers import post_verbose as _post
from tests._transcription_helpers import watch_progress as _watch_progress
from tests.runtime.test_preload_registry import _FakeQueue
from tests.runtime.test_routes_preload import _enable as _routes_enable


def _stub_diarize(app_module, monkeypatch, **cfg):
    """Turn diarization on (plus any cfg pins) and stub diarize(); returns the
    call log."""
    for k, v in {"DIARIZATION_ENABLED": True, **cfg}.items():
        monkeypatch.setattr(app_module.cfg, k, v, raising=False)
    calls = []

    async def _fake(path, *, num_speakers=None, min_speakers=None,
                    max_speakers=None, model_id=None, progress_cb=None,
                    cancel_check=None):
        calls.append({"model_id": model_id})
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake)
    return calls


def _stub_separate(app_module, monkeypatch, **cfg):
    """Turn BGM separation on (plus any cfg pins) and stub separate(); returns
    the call log."""
    for k, v in {"BGM_SEPARATION_ENABLED": True, **cfg}.items():
        monkeypatch.setattr(app_module.cfg, k, v, raising=False)
    calls = []

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        calls.append({"model_filename": model_filename})
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        return out
    monkeypatch.setattr(bgm_separation, "separate", _fake)
    return calls


def test_per_request_diarization_model_honored(client, app_module, monkeypatch):
    calls = _stub_diarize(app_module, monkeypatch)
    # In the default DIARIZATION_ALLOWED_MODELS, differs from the config
    # default (community-1).
    r = _post(client, diarize="true",
              diarization_model="pyannote/speaker-diarization-3.1")
    assert r.status_code == 200, r.text
    assert calls[0]["model_id"] == "pyannote/speaker-diarization-3.1"
    assert "warnings" not in r.json()
    # Absent field inherits the config default.
    r = _post(client, diarize="true")
    assert r.status_code == 200
    assert calls[1]["model_id"] == "pyannote/speaker-diarization-community-1"


def test_locked_diarization_model_ignores_client_param(client, app_module,
                                                       make_user_key,
                                                       monkeypatch):
    calls = _stub_diarize(app_module, monkeypatch)
    _, raw_alice = _bind_profile(client, make_user_key, "pin-diar", {
        "DIARIZATION_MODEL": "pyannote/speaker-diarization-community-1",
        "locks": ["DIARIZATION_MODEL"]})

    r = _post(client, bearer(raw_alice), diarize="true",
              diarization_model="pyannote/speaker-diarization-3.1")
    assert r.status_code == 200, r.text
    assert calls[0]["model_id"] == "pyannote/speaker-diarization-community-1"
    assert "diarization_model" in r.json()["overrides_ignored"]


def test_diarization_model_allowlist_miss_skips_stage(client, app_module,
                                                      monkeypatch):
    calls = _stub_diarize(app_module, monkeypatch, DIARIZATION_ALLOWED_MODELS=[
        "pyannote/speaker-diarization-community-1"])
    pid = "abcd" * 8
    seen = _watch_progress(monkeypatch, pid)
    r = _post(client, diarize="true", progress_id=pid,
              diarization_model="pyannote/speaker-diarization-3.1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert "speakers" not in body
    assert any("DIARIZATION_ALLOWED_MODELS" in w for w in body["warnings"])
    assert seen.get("skipped") == ["diarizing"]


def test_per_request_separation_model_honored(client, app_module, monkeypatch):
    calls = _stub_separate(app_module, monkeypatch)
    r = _post(client, separate_bgm="true",
              separation_model="UVR-MDX-NET-Inst_HQ_5")
    assert r.status_code == 200, r.text
    assert calls[0]["model_filename"] == "UVR-MDX-NET-Inst_HQ_5"
    assert "warnings" not in r.json()
    # Absent field inherits the config default (the handler resolves it
    # through _resolve_request_knob before the call, so the module sees the
    # config string, not None).
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200
    assert calls[1]["model_filename"] == "UVR-MDX-NET-Inst_HQ_4"


def test_separation_model_allowlist_miss_skips_stage(client, app_module,
                                                     monkeypatch):
    calls = _stub_separate(app_module, monkeypatch)
    r = _post(client, separate_bgm="true", separation_model="Evil-Model")
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert body["text"]                       # original audio transcribed
    assert any("BGM_SEPARATION_ALLOWED_MODELS" in w for w in body["warnings"])


def test_bgm_model_filename_override_resolution():
    assert bgm_separation._model_filename("Foo") == "Foo.onnx"
    assert bgm_separation._model_filename("bar.ckpt") == "bar.ckpt"
    # Empty/None falls back to the config default.
    assert bgm_separation._model_filename(None) == "UVR-MDX-NET-Inst_HQ_4.onnx"
    assert bgm_separation._model_filename("  ") == "UVR-MDX-NET-Inst_HQ_4.onnx"


def test_preload_extras_loads_configured_models(app_module, monkeypatch):
    loaded = []

    async def _fake_get_model(ref):
        loaded.append(("translation", ref))
    monkeypatch.setattr(translation, "_get_model", _fake_get_model)

    async def _fake_get_pipeline(model_id=None):
        loaded.append(("diarization", model_id))
    monkeypatch.setattr(diarization, "_get_pipeline", _fake_get_pipeline)

    async def _fake_get_separator(model_filename=None):
        loaded.append(("bgm", model_filename))
    monkeypatch.setattr(bgm_separation, "_get_separator", _fake_get_separator)

    cfg = app_module.cfg
    monkeypatch.setattr(cfg, "TRANSLATION_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_PRELOAD_MODELS",
                        ["org/a-GGUF:Q4", "org/b-GGUF:Q4"], raising=False)
    # Non-empty allowlist filters the preload list (default stays exempt).
    # The cap is raised to the list length so the ALLOWLIST is what drops
    # org/b — the cap truncation runs first and would otherwise mask it.
    monkeypatch.setattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 2, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/a-GGUF:Q4"}, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_PRELOAD", True, raising=False)
    # BGM preload asked for but the feature is off → no load.
    monkeypatch.setattr(cfg, "BGM_SEPARATION_ENABLED", False, raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_PRELOAD", True, raising=False)

    asyncio.run(app_module._preload_extras())
    assert loaded == [("translation", "org/a-GGUF:Q4"), ("diarization", None)]


def test_preload_extras_survives_load_failures(app_module, monkeypatch):
    loaded = []

    async def _boom_get_model(ref):
        raise RuntimeError("download failed")
    monkeypatch.setattr(translation, "_get_model", _boom_get_model)

    async def _boom_get_pipeline(model_id=None):
        raise diarization.DiarizationError("gated model")
    monkeypatch.setattr(diarization, "_get_pipeline", _boom_get_pipeline)

    async def _fake_get_separator(model_filename=None):
        loaded.append(("bgm", model_filename))
    monkeypatch.setattr(bgm_separation, "_get_separator", _fake_get_separator)

    cfg = app_module.cfg
    monkeypatch.setattr(cfg, "TRANSLATION_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_PRELOAD_MODELS",
                        ["org/a-GGUF:Q4"], raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_PRELOAD", True, raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_PRELOAD", True, raising=False)

    # Both failures are logged-and-swallowed; the separator still preloads.
    asyncio.run(app_module._preload_extras())
    assert loaded == [("bgm", None)]


def test_translation_model_allowed_shape_checks_client_ref(app_module,
                                                          monkeypatch):
    """An EMPTY allowlist admits any WELL-FORMED ref (_translation_model_allowed's
    contract, and what the whisper path enforces): the client value reaches
    hf_hub_download as a repo id, so a traversal string or an unbounded blob
    must be refused — while an inherited ref is admin policy and passes."""
    cfg = app_module.cfg
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "", raising=False)
    allowed = tr_gating._translation_model_allowed
    assert allowed("../../etc/passwd", requested="../../etc/passwd") is False
    assert allowed("a" * 200, requested="a" * 200) is False
    assert allowed("no-slash", requested="no-slash") is False
    assert allowed("org/repo-GGUF:Q4_K_M",
                   requested="org/repo-GGUF:Q4_K_M") is True
    # Config-inherited refs are not the client's to be gated.
    assert allowed("../../etc/passwd", requested=None) is True
    assert allowed("../../etc/passwd", requested="org/other") is True


def test_translation_device_edit_dispatches_eviction(client, monkeypatch):
    """Editing a field in the 'translation' EXTRAS_EVICTION bucket awaits
    translation's evictor via the generic post_state loop, the same dispatch
    the diarization/bgm buckets use."""
    calls = []

    async def _spy():
        calls.append("translation")
    monkeypatch.setitem(pl_apply.EVICTORS, "translation", _spy)

    r = client.post("/settings/state", json={"TRANSLATION_DEVICE": "cpu"})
    assert r.status_code == 200, r.text
    assert calls == ["translation"]

    # An untouched bucket stays quiet.
    calls.clear()
    r = client.post("/settings/state", json={"TRANSLATION_MODE": "faithful"})
    assert r.status_code == 200, r.text
    assert calls == []


def test_empty_bgm_allowlist_admits_only_the_configured_model(
        client, app_module, monkeypatch):
    """Empty allowlist = the configured model ONLY — clearing the list is a
    lockdown, never an open gate."""
    calls = _stub_separate(app_module, monkeypatch,
                           BGM_SEPARATION_ALLOWED_MODELS=[])
    r = _post(client, separate_bgm="true",
              separation_model="Kim_Vocal_2.onnx")
    assert r.status_code == 200, r.text
    body = r.json()
    assert any("not allowed" in w for w in body.get("warnings", []))
    assert not calls
    # The configured default still runs untouched.
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200
    assert len(calls) == 1


def test_allowlist_never_blocks_the_config_inherited_default(
        client, app_module, monkeypatch):
    """A narrowed allowlist missing the configured default must not disable
    the stage for requests that named no model at all."""
    # Config default stays community-1 — NOT in the allowlist.
    calls = _stub_diarize(app_module, monkeypatch, DIARIZATION_ALLOWED_MODELS=[
        "pyannote/speaker-diarization-3.1"])
    r = _post(client, diarize="true")
    assert r.status_code == 200, r.text
    assert len(calls) == 1
    assert calls[0]["model_id"] == "pyannote/speaker-diarization-community-1"
    assert "warnings" not in r.json()


def test_allowlist_admits_the_identity_effective_model_when_echoed(
        client, app_module, make_user_key, monkeypatch):
    """An identity/per-model pinned model outside the allowlist runs when the
    client sends nothing, so echoing that exact effective value must not be
    refused — the allowlist constrains only what the CLIENT adds."""
    calls = _stub_diarize(app_module, monkeypatch, DIARIZATION_ALLOWED_MODELS=[
        "pyannote/speaker-diarization-community-1"])
    pinned = "pyannote/speaker-diarization-3.1"
    admin_h, raw_alice = _bind_profile(
        client, make_user_key, "pin-diar",
        {"DIARIZATION_MODEL": pinned, "locks": []})
    r = _post(client, bearer(raw_alice), diarize="true", diarization_model=pinned)
    assert r.status_code == 200, r.text
    assert calls[0]["model_id"] == pinned
    assert "warnings" not in r.json()
    # Locked knob, same echoed value: the stage still runs and nothing is
    # reported as ignored (the request agrees with the lock).
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {"pin-diar": {
                        "DIARIZATION_MODEL": pinned,
                        "locks": ["DIARIZATION_MODEL"]}}})
    assert r.status_code == 200, r.text
    r = _post(client, bearer(raw_alice), diarize="true", diarization_model=pinned)
    assert r.status_code == 200, r.text
    assert len(calls) == 2
    assert calls[1]["model_id"] == pinned
    assert "warnings" not in r.json()
    assert "diarization_model" not in r.json().get("overrides_ignored", [])


def test_allowlist_admits_the_identity_effective_separation_model_when_echoed(
        client, app_module, make_user_key, monkeypatch):
    calls = _stub_separate(app_module, monkeypatch,
                           BGM_SEPARATION_ALLOWED_MODELS=[])
    _, raw_alice = _bind_profile(
        client, make_user_key, "pin-sep",
        {"BGM_SEPARATION_UVR_MODEL": "UVR-Pinned", "locks": []})
    # Absent field: the identity-pinned model runs.
    r = _post(client, bearer(raw_alice), separate_bgm="true")
    assert r.status_code == 200, r.text
    assert len(calls) == 1
    assert calls[0]["model_filename"] == "UVR-Pinned"
    # Echoing that exact effective value must yield the same call.
    r = _post(client, bearer(raw_alice), separate_bgm="true",
              separation_model="UVR-Pinned")
    assert r.status_code == 200, r.text
    assert len(calls) == 2
    assert calls[1]["model_filename"] == "UVR-Pinned"
    assert not any("not allowed" in w for w in r.json().get("warnings", []))


def _plan_with_stub_queue(app_module, monkeypatch, entries, pid="ab" * 8):
    """Register a plan bound to `pid` with the enqueue path stubbed, so the
    test observes exactly what the cursor decided to warm."""
    # REGISTRATION must admit nothing, so what the CURSOR enqueues next is
    # unambiguous: on_stage_start does not consult the ladder (the worker
    # re-admits at dequeue), which is the split under test.
    # _routes_enable stubs a definite "no room", not size_unknown: the
    # ladder TRIES on size_unknown, so registration would enqueue everything
    # and leave the cursor with nothing to assert on.
    _routes_enable(app_module, monkeypatch)

    enqueued = []
    monkeypatch.setattr(preload, "_enqueue_threadsafe", enqueued.append)
    monkeypatch.setattr(preload, "_queue", _FakeQueue())

    plan = preload.register_plan("u", entries, plan_id=pid)
    enqueued.clear()          # registration's own admissions are not the test
    tx_progress._PLAN_BY_PID[pid] = plan["plan_id"]
    return preload, enqueued


def test_stage_ahead_cursor_is_monotone(app_module, monkeypatch):
    """separating → diarizing → separating enqueues the diarization model once
    and nothing at all on the replay."""
    pid = "ab" * 8
    preload, enqueued = _plan_with_stub_queue(
        app_module, monkeypatch,
        [("separation", "UVR-A"), ("diarization", "p/x")], pid=pid)

    tx_progress._progress_set(pid, stage="separating")
    assert [e[1:] for e in enqueued] == [("diarization", "p/x")]

    enqueued.clear()
    tx_progress._progress_set(pid, stage="diarizing")
    assert enqueued == []       # nothing left past the cursor

    tx_progress._progress_set(pid, stage="separating")
    assert enqueued == []       # replayed stage: the cursor never walks back
    assert preload._plans[pid].cursor == preload.STAGE_INDEX["diarizing"]


def test_waiting_and_analyzing_map_to_the_transcribing_index(app_module,
                                                             monkeypatch):
    """Both are sub-stages of the decode. Mapped anywhere else they would read
    as unknown stages and stall the cursor mid-pipeline."""
    pid = "cd" * 8
    preload, enqueued = _plan_with_stub_queue(
        app_module, monkeypatch,
        [("diarization", "p/x"), ("translation", "o/r:Q4")], pid=pid)

    tx_progress._progress_set(pid, stage="waiting")
    assert preload._plans[pid].cursor == preload.STAGE_INDEX["transcribing"]
    assert [e[1:] for e in enqueued] == [("diarization", "p/x")]

    enqueued.clear()
    tx_progress._progress_set(pid, stage="analyzing")
    # Same index — not an advance, so no second enqueue.
    assert preload._plans[pid].cursor == preload.STAGE_INDEX["transcribing"]
    assert enqueued == []


def test_stageless_progress_tick_restamps_the_plan_ttl(app_module, monkeypatch):
    """A per-segment decode tick carries no stage=; it must still restamp the
    TTL, or a decode longer than MODEL_PRELOAD_WARM_TTL_S loses its plan to
    the sweeper mid-stage."""
    pid = "ee" * 8
    preload, enqueued = _plan_with_stub_queue(
        app_module, monkeypatch,
        [("diarization", "p/x"), ("translation", "o/r:Q4")], pid=pid)
    tx_progress._progress_set(pid, stage="transcribing")
    enqueued.clear()
    plan = preload._plans[pid]
    plan.expires_mono = 0.0

    tx_progress._progress_set(pid, progress=0.5, position=1.0, last_text="x")

    assert plan.expires_mono > time.monotonic()
    assert plan.cursor == preload.STAGE_INDEX["transcribing"]
    assert enqueued == []  # a tick is not an advance
    # The warm lease follows the restamp.
    assert (preload._warm[preload.stats_key("translation", "o/r:Q4")]
            == plan.expires_mono)


def test_stage_ahead_is_a_no_op_without_a_bound_plan(app_module, monkeypatch):
    calls = []
    monkeypatch.setattr(preload, "on_stage_start",
                        lambda *a: calls.append(a))
    tx_progress._progress_set("ef" * 8, stage="transcribing")
    assert calls == []
