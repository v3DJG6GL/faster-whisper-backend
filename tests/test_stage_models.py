"""Per-request stage models (diarization_model / separation_model), the
optional-stage startup preloads, and translation eviction-on-edit. All heavy
deps stay uninstalled — stage modules are stubbed at the same handler
boundaries as test_diarization/test_bgm."""

import asyncio
import os
import tempfile

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.audio import translation

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_diarize(monkeypatch, calls):
    async def _fake(path, *, num_speakers=None, min_speakers=None,
                    max_speakers=None, model_id=None, progress_cb=None,
                    cancel_check=None):
        calls.append({"model_id": model_id})
        return [(0.0, 1.0, "SPEAKER_00")]
    monkeypatch.setattr(diarization, "diarize", _fake)


def _stub_separate(monkeypatch, calls):
    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        calls.append({"model_filename": model_filename})
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        return out
    monkeypatch.setattr(bgm_separation, "separate", _fake)


# --- per-request diarization model -------------------------------------------

def test_per_request_diarization_model_honored(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_diarize(monkeypatch, calls)
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
    from tests.conftest import bearer
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_diarize(monkeypatch, calls)
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {"pin-diar": {
                        "DIARIZATION_MODEL":
                            "pyannote/speaker-diarization-community-1",
                        "locks": ["DIARIZATION_MODEL"]}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": ["pin-diar"], "locks": []}})
    assert r.status_code == 200, r.text

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "diarize": "true",
              "diarization_model": "pyannote/speaker-diarization-3.1"})
    assert r.status_code == 200, r.text
    assert calls[0]["model_id"] == "pyannote/speaker-diarization-community-1"
    assert "diarization_model" in r.json()["overrides_ignored"]


def test_diarization_model_allowlist_miss_skips_stage(client, app_module,
                                                      monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ALLOWED_MODELS",
                        ["pyannote/speaker-diarization-community-1"],
                        raising=False)
    calls = []
    _stub_diarize(monkeypatch, calls)
    pid = "abcd" * 8
    seen = {}
    orig = app_module._progress_set

    def spy(p, **fields):
        orig(p, **fields)
        if p == pid:
            seen.update(app_module._BATCH_PROGRESS.get(pid) or {})
    app_module._progress_set = spy
    try:
        r = _post(client, diarize="true", progress_id=pid,
                  diarization_model="pyannote/speaker-diarization-3.1")
    finally:
        app_module._progress_set = orig
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert "speakers" not in body
    assert any("DIARIZATION_ALLOWED_MODELS" in w for w in body["warnings"])
    assert seen.get("skipped") == ["diarizing"]


# --- per-request separation model --------------------------------------------

def test_per_request_separation_model_honored(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_separate(monkeypatch, calls)
    r = _post(client, separate_bgm="true",
              separation_model="UVR-MDX-NET-Inst_HQ_5")
    assert r.status_code == 200, r.text
    assert calls[0]["model_filename"] == "UVR-MDX-NET-Inst_HQ_5"
    assert "warnings" not in r.json()
    # Absent field inherits the config default (passed as None → the module
    # falls back to cfg.BGM_SEPARATION_UVR_MODEL).
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200
    assert calls[1]["model_filename"] == "UVR-MDX-NET-Inst_HQ_4"


def test_separation_model_allowlist_miss_skips_stage(client, app_module,
                                                     monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    calls = []
    _stub_separate(monkeypatch, calls)
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


# --- startup preloads --------------------------------------------------------

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
    # The cap is raised above the list length so the ALLOWLIST is what drops
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
    """An EMPTY allowlist admits any WELL-FORMED ref (the docstring's contract,
    and what the whisper path enforces): the client value reaches
    hf_hub_download as a repo id, so a traversal string or an unbounded blob
    must be refused — while an inherited ref is admin policy and passes."""
    cfg = app_module.cfg
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS", set(),
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "", raising=False)
    allowed = app_module._translation_model_allowed
    assert allowed("../../etc/passwd", requested="../../etc/passwd") is False
    assert allowed("a" * 200, requested="a" * 200) is False
    assert allowed("no-slash", requested="no-slash") is False
    assert allowed("org/repo-GGUF:Q4_K_M",
                   requested="org/repo-GGUF:Q4_K_M") is True
    # Config-inherited refs are not the client's to be gated.
    assert allowed("../../etc/passwd", requested=None) is True
    assert allowed("../../etc/passwd", requested="org/other") is True


# --- eviction-on-edit --------------------------------------------------------

def test_translation_device_edit_dispatches_eviction(client, monkeypatch):
    """Editing a field in the 'translation' EXTRAS_EVICTION bucket awaits
    translation's evictor via the generic post_state loop (the e5167ba
    dispatch — same template as the diarization/bgm buckets)."""
    from faster_whisper_backend.admin import routes as admin_routes

    calls = []

    async def _spy():
        calls.append("translation")
    monkeypatch.setitem(admin_routes._EVICTORS, "translation", _spy)

    r = client.post("/settings/state", json={"TRANSLATION_DEVICE": "cpu"})
    assert r.status_code == 200, r.text
    assert calls == ["translation"]

    # An untouched bucket stays quiet.
    calls.clear()
    r = client.post("/settings/state", json={"TRANSLATION_MODE": "faithful"})
    assert r.status_code == 200, r.text
    assert calls == []


# ── review-fix regressions: allowlist semantics ─────────────────────────────

def test_empty_bgm_allowlist_admits_only_the_configured_model(
        client, app_module, monkeypatch):
    """Empty allowlist = the configured model ONLY — clearing the list is a
    lockdown, never an open gate."""
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ALLOWED_MODELS", [],
                        raising=False)
    calls = []
    _stub_separate(monkeypatch, calls)
    r = _post(client, separate_bgm="true",
              separation_model="Kim_Vocal_2.onnx")
    assert r.status_code == 200, r.text
    body = r.json()
    assert any("not allowed" in w for w in body.get("warnings", []))
    assert not calls  # the stage was skipped, nothing loaded
    # The configured default still runs untouched.
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200
    assert len(calls) == 1


def test_allowlist_never_blocks_the_config_inherited_default(
        client, app_module, monkeypatch):
    """A narrowed allowlist missing the configured default must not disable
    the stage for requests that named no model at all."""
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ALLOWED_MODELS",
                        ["pyannote/speaker-diarization-3.1"], raising=False)
    # Config default stays community-1 — NOT in the allowlist.
    calls = []
    _stub_diarize(monkeypatch, calls)
    r = _post(client, diarize="true")
    assert r.status_code == 200, r.text
    assert len(calls) == 1
    assert calls[0]["model_id"] == "pyannote/speaker-diarization-community-1"
    assert "warnings" not in r.json()


# --- stage-ahead (the _progress_set hook) ------------------------------------

def _plan_with_stub_queue(app_module, monkeypatch, entries, pid="ab" * 8):
    """Register a plan bound to `pid` with the enqueue path stubbed, so the
    test observes exactly what the cursor decided to warm."""
    from faster_whisper_backend.runtime import preload
    from faster_whisper_backend.runtime import model_sizes
    # REGISTRATION must admit nothing, so the plan starts with an empty queue
    # and what the CURSOR does next is unambiguous. on_stage_start
    # deliberately does not consult the ladder; the worker re-admits at
    # dequeue, which is exactly the split under test.
    #
    # A definite "no room" is the right refusal to stub. `size_unknown` used
    # to serve here, but only by accident: it means "cannot say", and the
    # ladder now TRIES on it, because refusing meant an unmeasured model was
    # never loaded and therefore never measured. Left as size_unknown this
    # helper would silently let registration enqueue everything, leaving the
    # cursor with nothing to do and the test asserting on an empty list.
    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: (False, "insufficient_vram"))
    monkeypatch.setattr(preload, "_idle_peer", lambda *a, **k: None)
    for k, v in (("MODEL_PRELOAD_ENABLED", True),
                 ("MODEL_PRELOAD_WARM_TTL_S", 180),
                 ("DIARIZATION_ENABLED", True),
                 ("BGM_SEPARATION_ENABLED", True),
                 ("TRANSLATION_ENABLED", True)):
        monkeypatch.setattr(app_module.cfg, k, v, raising=False)

    enqueued = []
    monkeypatch.setattr(preload, "_enqueue_threadsafe", enqueued.append)

    class _StubQueue:
        def qsize(self):
            return 0

        def empty(self):
            return True
    monkeypatch.setattr(preload, "_queue", _StubQueue())

    plan = preload.register_plan("u", entries, plan_id=pid)
    enqueued.clear()          # registration's own admissions are not the test
    app_module._PLAN_BY_PID[pid] = plan["plan_id"]
    return preload, enqueued


def test_stage_ahead_cursor_is_monotone(app_module, monkeypatch):
    """separating → diarizing → separating enqueues the diarization model once
    and nothing at all on the replay."""
    pid = "ab" * 8
    preload, enqueued = _plan_with_stub_queue(
        app_module, monkeypatch,
        [("separation", "UVR-A"), ("diarization", "p/x")], pid=pid)

    app_module._progress_set(pid, stage="separating")
    assert [e[1:] for e in enqueued] == [("diarization", "p/x")]

    enqueued.clear()
    app_module._progress_set(pid, stage="diarizing")
    assert enqueued == []       # nothing left past the cursor

    app_module._progress_set(pid, stage="separating")
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

    app_module._progress_set(pid, stage="waiting")
    assert preload._plans[pid].cursor == preload.STAGE_INDEX["transcribing"]
    assert [e[1:] for e in enqueued] == [("diarization", "p/x")]

    enqueued.clear()
    app_module._progress_set(pid, stage="analyzing")
    # Same index — not an advance, so no second enqueue.
    assert preload._plans[pid].cursor == preload.STAGE_INDEX["transcribing"]
    assert enqueued == []


def test_stage_ahead_is_a_no_op_without_a_bound_plan(app_module, monkeypatch):
    from faster_whisper_backend.runtime import preload
    calls = []
    monkeypatch.setattr(preload, "on_stage_start",
                        lambda *a: calls.append(a))
    app_module._progress_set("ef" * 8, stage="transcribing")
    assert calls == []
