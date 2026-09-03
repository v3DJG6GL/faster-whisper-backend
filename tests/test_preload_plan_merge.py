"""Unit tests for preload.py's plan MERGE path and its seams with main.py.

A repeat registration of the same plan_id merges into the existing plan; the
first registration is the well-trodden path, the merge is where the entries
list, the cursor, the warmed set and the stage_ahead flag all have to agree
with a plan that already exists. Same monkeypatch boundary as
test_preload_registry (registry dicts and loader coroutines, no heavy deps).
"""

import asyncio

from faster_whisper_backend.audio import diarization
from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.runtime import system_stats

from conftest import bearer


def _fits(monkeypatch, verdict=(True, None)):
    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: verdict, raising=True)


def _enable(monkeypatch, **over):
    from faster_whisper_backend import config as cfg
    defaults = {
        "MODEL_PRELOAD_ENABLED": True,
        "MODEL_PRELOAD_WARM_TTL_S": 180,
        "MODEL_PRELOAD_VRAM_RESERVE_MB": 1024,
        "MODEL_PRELOAD_RAM_RESERVE_MB": 2048,
        "MODEL_PRELOAD_EVICT_IDLE_MODELS": True,
        "DIARIZATION_ENABLED": True,
        "BGM_SEPARATION_ENABLED": True,
        "TRANSLATION_ENABLED": True,
        "MODEL_DEVICE": "cpu",
        "MODEL_COMPUTE_TYPE": "int8",
        "MAX_LOADED_MODELS": 1,
        "ALLOWED_MODELS": set(),
        "DEFAULT_MODEL": "large-v3",
    }
    defaults.update(over)
    for k, v in defaults.items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    return cfg


# --- PC1: a merged plan stays in stage order --------------------------------

def test_merge_restores_stage_order_so_stage_ahead_picks_the_nearer_stage(
        monkeypatch):
    """The documented main path: a client POSTs [whisper, translation], then
    the job merges [separation, diarization, translation] into the same
    plan. Appended unsorted, `transcribing` would warm translation and
    diarization would never be warmed at all."""
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))     # nothing enqueues yet
    preload.register_plan("u", [("whisper", "large-v3"),
                                ("translation", "o/r:Q4")], plan_id="2" * 8)
    preload.register_plan("u", [("separation", "UVR-A"),
                                ("diarization", "p/x"),
                                ("translation", "o/r:Q4")], plan_id="2" * 8)
    plan = preload._plans["2" * 8]
    assert plan.stages == sorted(plan.stages) == [0, 1, 2, 3]
    assert len(plan.stages) == len(plan.entries)

    _fits(monkeypatch, (True, None))

    async def _run():
        await preload.start()
        preload._worker.cancel()
        preload.on_stage_start("2" * 8, "transcribing")
        await asyncio.sleep(0)
        assert preload._queue.qsize() == 1
        item = preload._queue.get_nowait()
        assert item == ("2" * 8, "diarization", "p/x")
    asyncio.run(_run())


# --- PC2: one plan's entries are bounded ------------------------------------

def test_plan_entries_are_capped_on_repeat_posts(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    for i in range(4):
        preload.register_plan(
            "u", [("diarization", f"p/{2 * i}"),
                  ("diarization", f"p/{2 * i + 1}")], plan_id="3" * 8)
    plan = preload._plans["3" * 8]
    assert len(plan.entries) <= preload._MAX_PLAN_ENTRIES
    assert len(plan.stages) == len(plan.entries)


def test_an_entry_past_the_cap_is_deferred_not_enqueued(monkeypatch):
    """An entry that did not join the plan must not be handed to the worker:
    it would be loaded with no warm lease and no plan to settle."""
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))

    async def _run():
        await preload.start()
        preload._worker.cancel()
        preload.register_plan(
            "u", [("diarization", f"p/{i}")
                  for i in range(preload._MAX_PLAN_ENTRIES)],
            plan_id="3" * 8)
        await asyncio.sleep(0)
        depth = preload._queue.qsize()
        r = preload.register_plan("u", [("diarization", "p/extra")],
                                  plan_id="3" * 8)
        await asyncio.sleep(0)
        assert r["models"][0]["state"] == "deferred"
        assert preload._queue.qsize() == depth
        assert ("diarization", "p/extra") not in preload._plans["3" * 8].entries
    asyncio.run(_run())


# --- PC3: whisper placement honours MODEL_OVERRIDES --------------------------

def test_whisper_placement_uses_the_per_model_override(monkeypatch):
    _enable(monkeypatch, MODEL_DEVICE="cuda", MODEL_COMPUTE_TYPE="float16",
            MODEL_OVERRIDES={"tiny": {"MODEL_DEVICE": "cpu",
                                      "MODEL_COMPUTE_TYPE": "int8"}})
    assert preload._placement("whisper", "tiny") == ("cpu", "int8")
    assert preload._placement("whisper", "large-v3") == ("cuda", "float16")
    # The bare call (no id) is the global pair.
    assert preload._placement("whisper") == ("cuda", "float16")


# --- PC4: a second lifespan gets a queue bound to ITS loop -------------------

def test_second_start_on_a_new_loop_does_not_reuse_the_old_queue():
    async def _first():
        await preload.start()
        await preload.stop()
    asyncio.run(_first())
    old_queue = preload._queue

    async def _second():
        await preload.start()
        assert preload._queue is not old_queue
        preload._enqueue_threadsafe(("nosuch", "diarization", "p/x"))
        await asyncio.sleep(0.05)
        # Consumed by a live worker, not stuck behind a "bound to a different
        # event loop" RuntimeError that killed it.
        assert not preload._worker.done()
        assert preload._queue.qsize() == 0
        await preload.stop()
    asyncio.run(_second())


# --- PC5: the queue cap counts the batch being admitted ---------------------

def test_queue_cap_counts_the_in_flight_batch(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))

    async def _run():
        await preload.start()
        preload._worker.cancel()
        for _ in range(preload._MAX_QUEUE - 1):
            preload._queue.put_nowait(("x", "diarization", "filler"))
        r = preload.register_plan("u", [("diarization", "p/x"),
                                        ("translation", "o/r:Q4")],
                                  plan_id="d" * 8)
        states = [(m["state"], m.get("reason")) for m in r["models"]]
        assert sum(s in ("loading", "queued") for s, _ in states) == 1
        assert ("deferred", "queue_full") in states
        await asyncio.sleep(0)
        assert preload._queue.qsize() <= preload._MAX_QUEUE
    asyncio.run(_run())


# --- PC7: _reset_for_tests clears the warm predicate -------------------------

def test_reset_for_tests_clears_the_warm_predicate():
    async def _run():
        await preload.start()
        assert system_stats._warm_predicate is preload.is_warm
        preload._reset_for_tests()
        assert system_stats._warm_predicate is None
        assert system_stats.is_warm("anything") is False
    asyncio.run(_run())


# --- PC9: stage_ahead is upgrade-only on merge -------------------------------

def test_stage_ahead_upgrades_on_merge_and_never_downgrades(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    e = [("diarization", "p/x")]
    preload.register_plan("u", e, plan_id="4" * 8, stage_ahead=False)
    assert preload._plans["4" * 8].stage_ahead is False
    preload.register_plan("u", e, plan_id="4" * 8)
    assert preload._plans["4" * 8].stage_ahead is True
    preload.register_plan("u", e, plan_id="4" * 8, stage_ahead=False)
    assert preload._plans["4" * 8].stage_ahead is True


# --- PC10: a new job on the same plan starts fresh ---------------------------

def test_merge_prunes_warmed_keys_that_are_no_longer_resident(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    monkeypatch.setattr(preload, "is_resident", lambda f, m: False)
    preload.register_plan("u", [("diarization", "p/x")], plan_id="5" * 8)
    preload._plans["5" * 8].warmed.add("pyannote:p/x")   # then evicted
    r = preload.register_plan("u", [("diarization", "p/x")], plan_id="5" * 8)
    assert r["models"][0]["state"] != "resident"
    assert "pyannote:p/x" not in preload._plans["5" * 8].warmed


def test_merge_keeps_warmed_keys_that_are_still_resident(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/x", "cpu", 4))
    preload.register_plan("u", [("diarization", "p/x")], plan_id="5" * 8)
    assert "pyannote:p/x" in preload._plans["5" * 8].warmed
    r = preload.register_plan("u", [("diarization", "p/x")], plan_id="5" * 8)
    assert r["models"][0]["state"] == "resident"
    assert "pyannote:p/x" in preload._plans["5" * 8].warmed


def test_a_job_binding_rewinds_the_cursor_a_client_post_does_not(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    e = [("separation", "UVR-A"), ("diarization", "p/x")]
    preload.register_plan("u", e, plan_id="6" * 8, trigger="job")
    preload.on_stage_start("6" * 8, "diarizing")
    plan = preload._plans["6" * 8]
    assert plan.cursor == 2
    preload.register_plan("u", e, plan_id="6" * 8, trigger=None)
    assert plan.cursor == 2
    preload.register_plan("u", e, plan_id="6" * 8, trigger="job")
    assert plan.cursor == -1


# --- PC11: a whisper preload never makes main evict a warm peer --------------

def test_whisper_full_cache_with_only_a_warm_peer_is_family_busy(monkeypatch):
    _enable(monkeypatch, MAX_LOADED_MODELS=1)
    _fits(monkeypatch, (True, None))
    from faster_whisper_backend import main
    monkeypatch.setattr(main, "_loaded_models", {"peer": object()})
    monkeypatch.setattr(main, "_model_leases", {})
    system_stats.set_warm_predicate(lambda k: k == "peer")
    assert preload._admit("whisper", "large-v3") == ("deferred", "family_busy")

    # The same cache with a COLD peer: admitted (the worker drops the peer).
    system_stats.set_warm_predicate(None)
    assert preload._admit("whisper", "large-v3") in (("loading", None),
                                                     ("queued", None))

    # A cold peer, but eviction switched off: refused rather than letting
    # main's LRU loop pick the victim.
    _enable(monkeypatch, MAX_LOADED_MODELS=1,
            MODEL_PRELOAD_EVICT_IDLE_MODELS=False)
    assert preload._admit("whisper", "large-v3") == ("deferred", "family_busy")


def test_worker_evicts_the_cold_whisper_peer_it_chose_even_when_it_fits(
        monkeypatch):
    _enable(monkeypatch, MAX_LOADED_MODELS=1)
    _fits(monkeypatch, (True, None))
    from faster_whisper_backend import main
    monkeypatch.setattr(main, "_loaded_models", {"peer": object()})
    monkeypatch.setattr(main, "_model_leases", {})
    evicted = []
    loaded = []

    async def _evict(family, peer):
        evicted.append((family, peer))
    monkeypatch.setattr(preload, "_evict", _evict)

    async def _load(family, mid):
        loaded.append((family, mid))
    monkeypatch.setattr(preload, "_load", _load)

    preload.register_plan("u", [("whisper", "large-v3")], plan_id="7" * 8)

    asyncio.run(preload._handle(("7" * 8, "whisper", "large-v3")))
    assert evicted == [("whisper", "peer")]
    assert loaded == [("whisper", "large-v3")]


# --- PC12: the warm lease comes from the plan, not from _admit ---------------

def test_resident_entry_is_warm_through_its_plan_only(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/x", "cpu", 4))
    # A bare _admit touches the idle clock but takes no lease of its own...
    assert preload._admit("diarization", "p/x") == ("resident", None)
    assert preload.is_warm("pyannote:p/x") is False
    assert preload._warm == {}
    # ...the plan is what holds it, for exactly the plan's lifetime.
    preload.register_plan("u", [("diarization", "p/x")], plan_id="8" * 8)
    assert preload.is_warm("pyannote:p/x") is True
    assert preload._warm["pyannote:p/x"] == preload._plans["8" * 8].expires_mono


# --- GS1: preload speaks the transcribe route's whisper vocabulary ----------

def test_whisper_1_alias_resolves_to_the_default_model(monkeypatch):
    _enable(monkeypatch, DEFAULT_MODEL="large-v3")
    from faster_whisper_backend import main
    assert preload.normalize_id("whisper", "whisper-1") == "large-v3"
    assert preload.stats_key("whisper", "whisper-1") == "large-v3"
    assert preload.stats_key("whisper", "small") == "small"
    monkeypatch.setitem(main._loaded_models, "large-v3", object())
    assert preload.is_resident("whisper", "whisper-1") is True
    assert preload.is_resident("whisper", "large-v3") is True
    # The two spellings are one plan entry as far as the ids are concerned.
    assert preload.derive_plan_id("u", [("whisper", "whisper-1")]) == \
        preload.derive_plan_id("u", [("whisper", "large-v3")])


def test_whisper_1_with_no_default_stays_blank_and_is_not_admitted(monkeypatch):
    _enable(monkeypatch, DEFAULT_MODEL="")
    _fits(monkeypatch, (True, None))
    assert preload.normalize_id("whisper", "whisper-1") == ""
    assert preload._admit("whisper", "whisper-1") == ("deferred", "not_allowed")


def test_route_judges_the_resolved_whisper_id(client, app_module, monkeypatch):
    """`whisper-1` is admitted exactly when DEFAULT_MODEL is, and a non-empty
    ALLOWED_MODELS that omits the default rejects both — as the transcribe
    gate does, instead of queueing a load that 400s there."""
    from faster_whisper_backend.runtime import preload_routes
    cfg = _enable(monkeypatch, DEFAULT_MODEL="large-v3",
                  ALLOWED_MODELS={"small"})
    assert preload_routes._allowed("whisper", "whisper-1") is False
    assert preload_routes._allowed("whisper", "large-v3") is False
    assert preload_routes._allowed("whisper", "small") is True
    monkeypatch.setattr(cfg, "ALLOWED_MODELS", {"large-v3"}, raising=False)
    assert preload_routes._allowed("whisper", "whisper-1") is True
    monkeypatch.setattr(cfg, "ALLOWED_MODELS", set(), raising=False)
    assert preload_routes._allowed("whisper", "whisper-1") is True
    assert preload_routes._allowed("whisper", "../etc/passwd") is False

    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: (None, "size_unknown"))
    r = client.post("/v1/models/preload", json={
        "models": [{"family": "whisper", "id": "whisper-1"}]})
    assert r.status_code == 202, r.text
    row = r.json()["models"][0]
    assert row.get("reason") != "not_allowed"
    # The plan's warm lease and job row are keyed under the REAL model.
    assert preload.is_warm("large-v3") is True
    assert preload.is_warm("whisper-1") is False


# --- PC6: the translation allowlist rule is main's --------------------------

def test_translation_allowlist_is_mains_rule(monkeypatch):
    from faster_whisper_backend.runtime import preload_routes
    cfg = _enable(monkeypatch, TRANSLATION_ALLOWED_MODELS=set(),
                  TRANSLATION_DEFAULT_MODEL="")
    assert preload_routes._allowed("translation", "any/ref:Q4") is True
    monkeypatch.setattr(cfg, "TRANSLATION_ALLOWED_MODELS", {"o/r:Q4"},
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", " o/d:Q8 ",
                        raising=False)
    assert preload_routes._allowed("translation", "o/r:Q4") is True
    assert preload_routes._allowed("translation", "o/d:Q8") is True
    assert preload_routes._allowed("translation", "o/evil:Q8") is False


# --- GS2: the /stats page renders the preload diagnostics --------------------

def test_stats_page_renders_the_preload_diagnostics(client, app_module,
                                                    make_user_key):
    make_user_key("admin", is_admin=True)
    _uid, raw = make_user_key("admin2", is_admin=True)
    r = client.get("/stats", headers=bearer(raw))
    assert r.status_code == 200, r.text
    html = r.text
    assert 'id="preload-line"' in html
    assert "snap.preload" in html
    assert "worker_alive" in html and "queue_depth" in html
