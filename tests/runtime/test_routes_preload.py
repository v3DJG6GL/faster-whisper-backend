"""Integration tests for POST /v1/models/preload and the residency flags the
/v1 discovery endpoints publish.

The contract under test is mostly about what does NOT happen: no 4xx for a
model the server declines to warm, no duplicate plans on a repeat POST, no
broken plan when a loader raises. Stage modules are stubbed at the same
boundary as test_stage_models — nothing here imports pyannote or onnxruntime.
"""


from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import diarization
from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.translation import engine as translation

from tests.conftest import bearer
from faster_whisper_backend.transcription import models as tx_models

_URL = "/v1/models/preload"


def _enable(app_module, monkeypatch, **over):
    cfg = app_module.cfg
    defaults = {
        "MODEL_PRELOAD_ENABLED": True,
        "MODEL_PRELOAD_WARM_TTL_S": 180,
        "DIARIZATION_ENABLED": True,
        "BGM_SEPARATION_ENABLED": True,
        "TRANSLATION_ENABLED": True,
    }
    defaults.update(over)
    for k, v in defaults.items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    # No room and no evictable peer → every admitted entry defers, which
    # keeps these tests off the REAL loaders (pyannote / UVR / a hub whisper
    # fetch) unless they opt in. size_unknown would not: with nothing to
    # evict, admission tries the load. Same stubs as test_stage_models.
    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: (False, "insufficient_vram"))
    monkeypatch.setattr(preload, "_idle_peer", lambda *a, **k: None)
    return cfg


def _body(**over):
    b = {"models": [{"family": "diarization",
                     "id": "pyannote/speaker-diarization-community-1"}]}
    b.update(over)
    return b


def test_enable_never_reaches_a_loader(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    loads = []

    async def _rec(family, model_id):
        loads.append((family, model_id))
    monkeypatch.setattr(preload, "_load", _rec)
    for body in (_body(),
                 {"models": [{"family": "separation", "id": "UVR-Only"}]},
                 {"models": [{"family": "whisper", "id": "Systran/x-y"}]}):
        assert client.post(_URL, json=body).status_code == 202
    for _ in range(10):
        client.get("/v1/models")
    assert loads == []


def test_a_client_trigger_cannot_pass_for_the_job_binding(
        client, app_module, monkeypatch):
    # "job" is the server's own job binding (it rewinds the plan cursor);
    # non-printables are stripped so a client cannot forge log rows.
    _enable(app_module, monkeypatch)
    seen = []
    real = preload.register_plan

    def _spy(*a, **kw):
        seen.append(kw.get("trigger"))
        return real(*a, **kw)
    monkeypatch.setattr(preload, "register_plan", _spy)
    client.post(_URL, json=_body(trigger="job"))
    client.post(_URL, json=_body(trigger="dict\nation"))
    client.post(_URL, json=_body(trigger="\n"))
    assert seen == ["client:job", "dictation", None]


# --- auth --------------------------------------------------------------------

def test_401_unauthenticated(client, app_module, monkeypatch, make_user_key):
    _enable(app_module, monkeypatch)
    make_user_key("admin", is_admin=True)          # flips to locked-down
    r = client.post(_URL, json=_body())
    assert r.status_code == 401


def test_202_for_a_plain_user(client, app_module, monkeypatch, make_user_key):
    _enable(app_module, monkeypatch)
    make_user_key("admin", is_admin=True)
    _uid, raw = make_user_key("bob")
    r = client.post(_URL, json=_body(), headers=bearer(raw))
    assert r.status_code == 202, r.text


# --- response shape ----------------------------------------------------------

def test_response_shape(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    r = client.post(_URL, json=_body())
    assert r.status_code == 202
    j = r.json()
    assert set(j) == {"plan_id", "expires_in_s", "models"}
    assert isinstance(j["plan_id"], str) and j["plan_id"]
    assert j["expires_in_s"] == 180
    row = j["models"][0]
    assert row["family"] == "diarization"
    assert row["state"] in ("resident", "loading", "queued", "deferred")


def test_unknown_body_key_is_422(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    assert client.post(_URL, json=_body(bogus=1)).status_code == 422
    # Unknown key inside a model entry too (extra="forbid" on both models).
    assert client.post(_URL, json={
        "models": [{"family": "whisper", "id": "x", "nope": 1}]}).status_code == 422
    # Structurally invalid: empty list, unknown family, blank id.
    assert client.post(_URL, json={"models": []}).status_code == 422
    assert client.post(_URL, json={
        "models": [{"family": "vad", "id": "silero"}]}).status_code == 422
    assert client.post(_URL, json={
        "models": [{"family": "whisper", "id": ""}]}).status_code == 422
    assert client.post(_URL, json=_body(plan_id="NOT-HEX")).status_code == 422


# --- everything that is NOT a 4xx --------------------------------------------

def _one(client, body):
    r = client.post(_URL, json=body)
    # The whole point: a server that declines still answers 202.
    assert r.status_code == 202, r.text
    return r.json()["models"][0]


def test_disallowed_model_is_202_deferred_not_allowed(client, app_module,
                                                      monkeypatch):
    _enable(app_module, monkeypatch)
    row = _one(client, {"models": [{"family": "diarization",
                                    "id": "somebody/not-on-the-list"}]})
    assert row == {"family": "diarization", "id": "somebody/not-on-the-list",
                   "state": "deferred", "reason": "not_allowed"}


def test_empty_stage_allowlist_means_the_configured_model_only(
        client, app_module, monkeypatch):
    cfg = _enable(app_module, monkeypatch)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", [], raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "UVR-Only",
                        raising=False)
    # The configured model passes even though the allowlist is empty...
    ok = _one(client, {"models": [{"family": "separation", "id": "UVR-Only"}]})
    assert ok.get("reason") != "not_allowed"
    # ...and an empty allowlist admits nothing ELSE, rather than everything.
    row = _one(client, {"models": [{"family": "separation",
                                    "id": "UVR-Something-Else"}]})
    assert row["reason"] == "not_allowed"


def test_a_blank_stage_model_id_is_never_allowed(client, app_module,
                                                 monkeypatch):
    """An unset configured model used to put "" on the allowlist, so a
    whitespace-only id (it passes min_length, then strips to "") joined the
    plan, could never be warmed and held its /stats row open until the TTL."""
    from faster_whisper_backend.runtime import preload_routes
    cfg = _enable(app_module, monkeypatch)
    for k in ("DIARIZATION_MODEL", "BGM_SEPARATION_UVR_MODEL"):
        monkeypatch.setattr(cfg, k, "", raising=False)
    assert preload_routes._allowed("diarization", "", {}) is False
    assert preload_routes._allowed("separation", "", {}) is False
    for fam in ("diarization", "separation"):
        r = client.post(_URL, json={"models": [{"family": fam, "id": "   "}]})
        assert r.status_code == 202
        assert r.json()["models"][0]["reason"] == "not_allowed"
        plan = preload._plans.get(r.json()["plan_id"])
        assert plan is None or (plan.entries == [] and plan.job_id is None)


def test_identity_effective_stage_model_is_not_reported_not_allowed(
        client, app_module, monkeypatch):
    # The batch handler admits the caller's EFFECTIVE stage model (identity
    # profile / MODEL_OVERRIDES) as admin policy; the preload route must too,
    # or the up-front warm silently does nothing for exactly those users.
    from faster_whisper_backend.settings import effective_config
    cfg = _enable(app_module, monkeypatch)
    monkeypatch.setattr(cfg, "DIARIZATION_ALLOWED_MODELS", [], raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "p/global", raising=False)
    real = effective_config.cfg_for

    def _cfg_for(model, key, ident=None):
        if key == "DIARIZATION_MODEL" and ident is not None:
            return "p/profiled"
        return real(model, key, ident)
    monkeypatch.setattr(effective_config, "cfg_for", _cfg_for)
    row = _one(client, {"models": [{"family": "diarization",
                                    "id": "p/profiled"}]})
    assert row.get("reason") != "not_allowed"
    # Anything else stays refused by the empty allowlist.
    row = _one(client, {"models": [{"family": "diarization", "id": "p/other"}]})
    assert row["reason"] == "not_allowed"


def test_disabled_stage_is_202_deferred_stage_disabled(client, app_module,
                                                       monkeypatch):
    _enable(app_module, monkeypatch, DIARIZATION_ENABLED=False)
    row = _one(client, _body())
    assert row["state"] == "deferred"
    assert row["reason"] == "stage_disabled"


def test_feature_off_is_202_deferred_disabled(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch, MODEL_PRELOAD_ENABLED=False)
    row = _one(client, _body())
    assert row["state"] == "deferred"
    assert row["reason"] == "disabled"


def test_feature_off_registers_no_plan_and_no_warm_lease(client, app_module,
                                                          monkeypatch):
    """The master switch means load-on-first-use: a POST while disabled must
    leave nothing behind that could hold a model against the idle evictors."""
    _enable(app_module, monkeypatch, MODEL_PRELOAD_ENABLED=False)
    _one(client, _body())
    assert preload._plans == {}
    # is_warm short-circuits on the very switch under test, so _warm is the
    # assertion that checks the registry really left nothing behind.
    assert preload._warm == {}
    key = preload.stats_key("diarization", _body()["models"][0]["id"])
    assert preload.is_warm(key) is False


def test_path_shaped_whisper_id_is_deferred_with_an_empty_allowlist(
        client, app_module, monkeypatch):
    """Parity with transcription.models._get_or_load_model: an empty ALLOWED_MODELS admits
    well-formed ids, not filesystem paths — the loader would reject this
    anyway, so it must not cost a queue slot and a warm key first."""
    cfg = _enable(app_module, monkeypatch)
    monkeypatch.setattr(cfg, "ALLOWED_MODELS", [], raising=False)
    monkeypatch.setattr(cfg, "DEFAULT_MODEL", "large-v3", raising=False)
    row = _one(client, {"models": [{"family": "whisper",
                                    "id": "../etc/passwd"}]})
    assert row["state"] == "deferred"
    assert row["reason"] == "not_allowed"
    # A well-formed id still passes the same branch.
    ok = _one(client, {"models": [{"family": "whisper", "id": "Systran/x-y"}]})
    assert ok.get("reason") != "not_allowed"


# --- idempotency -------------------------------------------------------------

def test_repeat_post_reuses_the_plan_and_does_not_grow_the_queue(
        client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(model_sizes, "fits", lambda *a, **k: (True, None))
    # Freeze the worker so the queue is observable.
    if preload._worker is not None:
        preload._worker.cancel()

    r1 = client.post(_URL, json=_body()).json()
    depth = preload._queue.qsize()
    r2 = client.post(_URL, json=_body()).json()
    assert r2["plan_id"] == r1["plan_id"]
    assert len(preload._plans) == 1
    assert preload._queue.qsize() == depth


def test_client_supplied_plan_id_is_honoured(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    j = client.post(_URL, json=_body(plan_id="deadbeef")).json()
    assert j["plan_id"] == "deadbeef"


# --- worker robustness -------------------------------------------------------

def test_a_raising_loader_leaves_the_plan_intact(client, app_module,
                                                 monkeypatch):
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(model_sizes, "fits", lambda *a, **k: (True, None))

    seen = []

    async def _boom(model_id=None, **_kw):
        seen.append(model_id)
        raise RuntimeError("gated model")
    monkeypatch.setattr(diarization, "_get_pipeline", _boom)

    j = client.post(_URL, json=_body()).json()
    # Give the worker a moment to pick the item up and fail on it.
    for _ in range(50):
        if seen and preload._queue.qsize() == 0:
            break
        client.get("/v1/models")
    assert preload._queue.qsize() == 0
    assert seen, "worker never dequeued the item"
    plan = preload._plans[j["plan_id"]]
    # The plan survives the failure — the stage simply loads in-band later.
    assert plan.dead is False
    assert preload.stats_key("diarization",
                             "pyannote/speaker-diarization-community-1") \
        not in plan.warmed


# --- residency flags agree with preload.is_resident --------------------------

def test_v1_models_loaded_flag_agrees(client, app_module, monkeypatch):
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_MODEL", "small",
                        raising=False)
    tx_models._loaded_models["small"] = object()
    try:
        data = client.get("/v1/models").json()["data"]
        by_id = {d["id"]: d["loaded"] for d in data}
        assert by_id["small"] is True
        assert by_id["small"] == preload.is_resident("whisper", "small")
    finally:
        tx_models._loaded_models.pop("small", None)


def test_v1_me_loaded_flags_agree_for_all_four_families(client, app_module,
                                                        monkeypatch):
    cfg = _enable(app_module, monkeypatch)
    # /v1/me seeds each list with the CONFIGURED model before the allowlist,
    # so pin both to the allowlist's first entry to keep the dicts exact.
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "p/x", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_ALLOWED_MODELS", ["p/x", "p/y"],
                        raising=False)
    # A UVR FRIENDLY name (no ".onnx") — the case where the two open-coded
    # predicates this replaced would disagree.
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "UVR-Foo",
                        raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS",
                        ["UVR-Foo", "UVR-Bar"], raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS", {"o/r:Q4"},
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "", raising=False)

    monkeypatch.setattr(diarization, "_pipeline_key", ("p/x", "cpu", 4))
    monkeypatch.setattr(bgm_separation, "_separator_key",
                        ("UVR-Foo.onnx", "cpu"))
    translation._models["o/r:Q4"] = object()

    caps = client.get("/v1/me").json()
    for key, family in (("diarization_models", "diarization"),
                        ("separation_models", "separation"),
                        ("translation_models", "translation")):
        for row in caps[key]:
            assert row["loaded"] == preload.is_resident(family, row["id"]), row
    assert {r["id"]: r["loaded"] for r in caps["diarization_models"]} == {
        "p/x": True, "p/y": False}
    # The friendly name resolves through the shared .onnx mapping.
    assert {r["id"]: r["loaded"] for r in caps["separation_models"]} == {
        "UVR-Foo": True, "UVR-Bar": False}


# --- /stats surfaces the diagnostics ----------------------------------------

def test_stats_snapshot_carries_preload_diagnostics(client, app_module,
                                                    monkeypatch):
    _enable(app_module, monkeypatch)
    j = client.get("/stats/snapshot").json()
    assert set(j["preload"]) == {"enabled", "worker_alive", "plans", "warm",
                                 "queue_depth"}
    assert j["preload"]["worker_alive"] is True


# --- the batch handler's job plan honours the translation allowlist ----------

def test_job_plan_skips_non_allowlisted_translation_model(
        client, app_module, monkeypatch):
    """The stage-ahead plan a transcription registers must apply the same
    TRANSLATION_ALLOWED_MODELS verdict the stage itself renders later — a
    ref the stage will refuse must never be pre-downloaded/loaded."""
    _enable(app_module, monkeypatch)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL",
                        "org/default-GGUF:Q4", raising=False)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/default-GGUF:Q4"}, raising=False)

    async def _fake(segments, targets, **kw):
        return ([{t: "x" for t in targets} for _ in segments], [],
                {"model": "org/default-GGUF:Q4", "source": "de",
                 "mode": "fluent"})
    monkeypatch.setattr(translation, "translate_segments", _fake)

    pid = "abcd" * 8
    files = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
    r = client.post("/v1/audio/transcriptions", files=files, data={
        "model": "whisper-1", "translate_to": "en",
        "translation_model": "org/evil-GGUF:Q8", "progress_id": pid})
    assert r.status_code == 200, r.text
    assert not any(fam == "translation"
                   for plan in preload._plans.values()
                   for fam, _mid in plan.entries)

    # Control: an allowlisted request DOES stage the translation model.
    preload._reset_for_tests()
    r = client.post("/v1/audio/transcriptions", files=files, data={
        "model": "whisper-1", "translate_to": "en",
        "translation_model": "org/default-GGUF:Q4", "progress_id": pid})
    assert r.status_code == 200, r.text
    assert any((fam, mid) == ("translation", "org/default-GGUF:Q4")
               for plan in preload._plans.values()
               for fam, mid in plan.entries)
