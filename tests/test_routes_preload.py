"""Integration tests for POST /v1/models/preload and the residency flags the
/v1 discovery endpoints publish.

The contract under test is mostly about what does NOT happen: no 4xx for a
model the server declines to warm, no duplicate plans on a repeat POST, no
broken plan when a loader raises. Stage modules are stubbed at the same
boundary as test_stage_models — nothing here imports pyannote or onnxruntime.
"""

import asyncio

import bgm_separation
import diarization
import model_sizes
import preload
import translation

from conftest import bearer

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
    # Nothing measured → every admitted entry defers with size_unknown, which
    # keeps these tests off the loaders unless they opt in.
    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: (None, "size_unknown"))
    return cfg


def _body(**over):
    b = {"models": [{"family": "diarization",
                     "id": "pyannote/speaker-diarization-community-1"}]}
    b.update(over)
    return b


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

    async def _boom(model_id=None, **_kw):
        raise RuntimeError("gated model")
    monkeypatch.setattr(diarization, "_get_pipeline", _boom)

    j = client.post(_URL, json=_body()).json()
    # Give the worker a moment to pick the item up and fail on it.
    for _ in range(50):
        if preload._queue.qsize() == 0:
            break
        client.get("/v1/models")
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
    app_module._loaded_models["small"] = object()
    try:
        data = client.get("/v1/models").json()["data"]
        by_id = {d["id"]: d["loaded"] for d in data}
        assert by_id["small"] is True
        assert by_id["small"] == preload.is_resident("whisper", "small")
    finally:
        app_module._loaded_models.pop("small", None)


def test_v1_me_loaded_flags_agree_for_all_four_families(client, app_module,
                                                        monkeypatch):
    cfg = _enable(app_module, monkeypatch)
    monkeypatch.setattr(cfg, "DIARIZATION_ALLOWED_MODELS", ["p/x", "p/y"],
                        raising=False)
    # A UVR FRIENDLY name (no ".onnx") — the case where the two open-coded
    # predicates this replaced would disagree.
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
