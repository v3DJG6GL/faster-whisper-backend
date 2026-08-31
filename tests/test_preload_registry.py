"""Unit tests for preload.py — the admission ladder, warm-lease cascade,
idempotent registration and thread safety.

No FastAPI and no heavy deps: every family module is monkeypatched at the same
module boundary test_diarization / test_bgm / test_stage_models use (the
registry dicts and the loader coroutines), so nothing here can import pyannote,
onnxruntime, llama.cpp or CTranslate2.
"""

import asyncio
import concurrent.futures
import time

import bgm_separation
import diarization
import model_sizes
import preload
import system_stats
import translation

_GB = 1024 * 1024 * 1024


def _fits(monkeypatch, verdict=(True, None)):
    """Pin model_sizes.fits so the ladder's rung 2 is deterministic."""
    monkeypatch.setattr(model_sizes, "fits",
                        lambda *a, **k: verdict, raising=True)


def _enable(monkeypatch, **over):
    import config as cfg
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
    }
    defaults.update(over)
    for k, v in defaults.items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    return cfg


# --- keys and residency ------------------------------------------------------

def test_stats_keys_and_uvr_friendly_name_normalisation():
    assert preload.stats_key("whisper", "large-v3") == "large-v3"
    assert preload.stats_key("diarization", "a/b") == "pyannote:a/b"
    assert preload.stats_key("translation", "o/r:Q4") == "gguf:o/r:Q4"
    # The one id shape that differs from what a client types: the separator
    # caches by on-disk filename.
    assert preload.stats_key("separation", "UVR-Foo") == "uvr:UVR-Foo.onnx"
    assert preload.stats_key("separation", "UVR-Foo.ckpt") == "uvr:UVR-Foo.ckpt"


def test_is_resident_per_family(monkeypatch):
    assert preload.is_resident("diarization", "p/x") is False
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/x", "cpu", 4))
    assert preload.is_resident("diarization", "p/x") is True
    assert preload.is_resident("diarization", "p/y") is False

    monkeypatch.setattr(bgm_separation, "_separator_key", ("UVR-Foo.onnx", "cpu"))
    # The friendly name (no extension) must agree with the cached filename.
    assert preload.is_resident("separation", "UVR-Foo") is True
    assert preload.is_resident("separation", "UVR-Foo.onnx") is True
    assert preload.is_resident("separation", "Other") is False

    translation._models["o/r:Q4"] = object()
    assert preload.is_resident("translation", "o/r:Q4") is True
    assert preload.is_resident("translation", "o/r:Q8") is False


# --- the admission ladder ----------------------------------------------------

def test_rung1_resident_touches_and_warms(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/x", "cpu", 4))
    touched = []
    monkeypatch.setattr(system_stats, "touch_loaded_model", touched.append)

    state, reason = preload._admit("diarization", "p/x")
    assert (state, reason) == ("resident", None)
    assert touched == ["pyannote:p/x"]
    # The warm lease is taken immediately, not only once a plan sweeps.
    assert preload.is_warm("pyannote:p/x") is True


def test_rung2_fits_says_yes(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))
    assert preload._admit("diarization", "p/x") == ("loading", None)


def test_rung3_idle_peer_evictable_admits_despite_no_room(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (False, "insufficient_vram"))
    # A resident, UNLEASED, UNWARMED peer of the same family is the only thing
    # in the way → admitted (the worker drops the peer).
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/other", "cpu", 4))
    assert preload._admit("diarization", "p/x") == ("loading", None)

    # Same state with eviction switched off → refused, nothing disturbed.
    _enable(monkeypatch, MODEL_PRELOAD_EVICT_IDLE_MODELS=False)
    assert preload._admit("diarization", "p/x") == ("deferred",
                                                    "insufficient_vram")


def test_rung4_deferred_reasons(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    assert preload._admit("diarization", "p/x") == ("deferred", "size_unknown")
    _fits(monkeypatch, (False, "vram_unknown"))
    assert preload._admit("diarization", "p/x") == ("deferred", "vram_unknown")

    _enable(monkeypatch, DIARIZATION_ENABLED=False)
    _fits(monkeypatch, (True, None))
    assert preload._admit("diarization", "p/x") == ("deferred",
                                                    "stage_disabled")

    _enable(monkeypatch, MODEL_PRELOAD_ENABLED=False)
    assert preload._admit("diarization", "p/x") == ("deferred", "disabled")


def test_job_leased_singleton_is_family_busy_not_an_orphan(monkeypatch):
    """A leased singleton must never be displaced by a speculative warm-up —
    _get_pipeline's force-drop would ORPHAN it and hold both in VRAM."""
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/other", "cpu", 4))
    diarization._leases["p/other"] = 1
    assert preload._admit("diarization", "p/x") == ("deferred", "family_busy")
    # And it is not offered up as an evictable peer either.
    assert preload._idle_peer("diarization", "p/x") is None

    monkeypatch.setattr(bgm_separation, "_separator_key", ("o.onnx", "cpu"))
    bgm_separation._leases["o.onnx"] = 1
    assert preload._admit("separation", "x") == ("deferred", "family_busy")


def test_whisper_defers_while_the_model_load_lock_is_held(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))

    class _Locked:
        def locked(self):
            return True

    import main
    monkeypatch.setattr(main, "_model_load_lock", _Locked())
    assert preload._admit("whisper", "large-v3") == ("deferred", "family_busy")


def test_a_warm_peer_is_not_an_evictable_peer(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (False, "insufficient_vram"))
    monkeypatch.setattr(diarization, "_pipeline_key", ("p/other", "cpu", 4))
    system_stats.set_warm_predicate(lambda k: k == "pyannote:p/other")
    assert preload._idle_peer("diarization", "p/x") is None
    assert preload._admit("diarization", "p/x") == ("deferred",
                                                    "insufficient_vram")


# --- warm-lease cascade ------------------------------------------------------

def test_ttl_sweep_drops_a_plans_keys_together_overlap_survives(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))  # nothing enqueues

    preload.register_plan("u1", [("diarization", "p/x"),
                                 ("separation", "UVR-A")], plan_id="a" * 8)
    preload.register_plan("u2", [("separation", "UVR-A"),
                                 ("translation", "o/r:Q4")], plan_id="b" * 8)
    for k in ("pyannote:p/x", "uvr:UVR-A.onnx", "gguf:o/r:Q4"):
        assert preload.is_warm(k), k

    # Expire only the first plan.
    preload._plans["a" * 8].expires_mono = time.monotonic() - 1

    async def _one_sweep():
        task = asyncio.ensure_future(preload.sweeper_loop())
        monkeypatch.setattr(preload, "_SWEEP_S", 0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    monkeypatch.setattr(preload, "_SWEEP_S", 0.01)
    asyncio.run(_one_sweep())

    assert "a" * 8 not in preload._plans
    # Its exclusive key went with it; the key the second plan also holds did
    # not — that is the cascade, with no refcount to leak.
    assert preload.is_warm("pyannote:p/x") is False
    assert preload.is_warm("uvr:UVR-A.onnx") is True
    assert preload.is_warm("gguf:o/r:Q4") is True


def test_is_warm_never_raises_unset_or_throwing():
    # Unset predicate → False, no exception (a unit test importing only
    # system_stats must not blow up an evictor).
    system_stats.set_warm_predicate(None)
    assert system_stats.is_warm("anything") is False

    def _boom(_k):
        raise RuntimeError("nope")
    system_stats.set_warm_predicate(_boom)
    assert system_stats.is_warm("anything") is False


# --- registration ------------------------------------------------------------

def _drain(n=64):
    """Pop what the (loop-less) queue collected, if any."""
    out = []
    q = preload._queue
    if q is None:
        return out
    for _ in range(n):
        try:
            out.append(q.get_nowait())
        except Exception:
            break
    return out


def test_reregistration_restamps_and_does_not_double_enqueue(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))

    async def _run():
        await preload.start()
        # Cancel the consumer up front so the queue accounting is observable:
        # this test is about registration, not about loading.
        preload._worker.cancel()

        r1 = preload.register_plan("u1", [("diarization", "p/x")],
                                   plan_id="c" * 8)
        assert r1["plan_id"] == "c" * 8
        assert r1["models"][0]["state"] in ("loading", "queued")
        await asyncio.sleep(0)          # let the enqueue callback run
        assert preload._queue.qsize() == 1

        preload._plans["c" * 8].expires_mono = 0.0
        r2 = preload.register_plan("u1", [("diarization", "p/x")],
                                   plan_id="c" * 8)
        # Restamped...
        assert preload._plans["c" * 8].expires_mono > time.monotonic()
        # ...and reported as still queued rather than enqueued a second time.
        assert r2["models"][0]["state"] == "queued"
        await asyncio.sleep(0)
        assert preload._queue.qsize() == 1

    asyncio.run(_run())


def test_derived_plan_id_is_stable_and_user_scoped():
    e = [("diarization", "p/x"), ("separation", "UVR-A")]
    assert preload.derive_plan_id("u1", e) == preload.derive_plan_id(
        "u1", list(reversed(e)))
    assert preload.derive_plan_id("u1", e) != preload.derive_plan_id("u2", e)


def test_plan_cap_evicts_the_plan_closest_to_expiry(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    for i in range(preload._MAX_PLANS):
        preload.register_plan("u", [("diarization", f"p/{i}")],
                              plan_id=f"{i:08x}")
    assert len(preload._plans) == preload._MAX_PLANS
    preload._plans["00000003"].expires_mono = 0.0
    preload.register_plan("u", [("diarization", "p/new")], plan_id="ff" * 4)
    assert len(preload._plans) == preload._MAX_PLANS
    assert "00000003" not in preload._plans
    assert "ff" * 4 in preload._plans


def test_queue_cap_defers_rather_than_growing(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (True, None))

    async def _run():
        await preload.start()
        preload._worker.cancel()
        for _ in range(preload._MAX_QUEUE):
            preload._queue.put_nowait(("x", "diarization", "filler"))
        r = preload.register_plan("u", [("diarization", "p/x")],
                                  plan_id="d" * 8)
        assert r["models"][0] == {"family": "diarization", "id": "p/x",
                                  "state": "deferred", "reason": "queue_full"}
    asyncio.run(_run())


def test_denied_entries_are_reported_and_never_join_the_plan(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    r = preload.register_plan(
        "u", [("diarization", "p/x"), ("diarization", "p/bad")],
        plan_id="e" * 8,
        denied={("diarization", "p/bad"): "not_allowed"})
    rows = {m["id"]: m for m in r["models"]}
    assert rows["p/bad"]["state"] == "deferred"
    assert rows["p/bad"]["reason"] == "not_allowed"
    assert preload._plans["e" * 8].entries == [("diarization", "p/x")]
    # A denied model gets no warm lease either.
    assert preload.is_warm("pyannote:p/bad") is False


def test_register_plan_never_raises(monkeypatch):
    _enable(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("ledger on fire")
    monkeypatch.setattr(model_sizes, "fits", _boom)
    r = preload.register_plan("u", [("diarization", "p/x")], plan_id="f" * 8)
    # The ladder's own guard turns the blowup into a deferral, not a 500.
    assert r["models"][0]["state"] == "deferred"


# --- thread safety -----------------------------------------------------------

def test_sync_entry_points_are_thread_safe_against_the_sweeper(monkeypatch):
    """register_plan / on_stage_start / is_warm are called from executor
    threads while the sweeper mutates the same dicts on the loop."""
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    monkeypatch.setattr(preload, "_SWEEP_S", 0.001)

    async def _run():
        task = asyncio.ensure_future(preload.sweeper_loop())

        def _hammer(i):
            pid = f"{i % 4:08x}"
            for _ in range(50):
                preload.register_plan(
                    f"u{i}", [("diarization", f"p/{i}"),
                              ("translation", f"o/r{i}:Q4")], plan_id=pid)
                preload.on_stage_start(pid, "diarizing")
                preload.on_stage_start(pid, "separating")  # replayed: no-op
                preload.is_warm(f"pyannote:p/{i}")

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            await asyncio.gather(*[
                loop.run_in_executor(ex, _hammer, i) for i in range(8)])
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    # The invariant that matters: the registry stayed bounded and _warm never
    # outgrew the union of the surviving plans' entries.
    assert len(preload._plans) <= preload._MAX_PLANS
    keys = {preload.stats_key(f, m)
            for p in preload._plans.values() for f, m in p.entries}
    assert set(preload._warm) <= keys


# --- stage-ahead cursor ------------------------------------------------------

def test_cursor_is_monotone_and_only_looks_forward(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    preload.register_plan("u", [("separation", "UVR-A"),
                                ("diarization", "p/x")], plan_id="1" * 8)
    plan = preload._plans["1" * 8]
    assert plan.cursor == -1
    preload.on_stage_start("1" * 8, "diarizing")
    assert plan.cursor == 2
    preload.on_stage_start("1" * 8, "separating")
    assert plan.cursor == 2          # never walks backwards
    # waiting/analyzing are sub-stages of the decode, not unknown stages.
    assert preload.STAGE_INDEX["waiting"] == preload.STAGE_INDEX["transcribing"]
    assert preload.STAGE_INDEX["analyzing"] == preload.STAGE_INDEX["transcribing"]


def test_diagnostics_shape():
    d = preload.diagnostics()
    assert set(d) == {"enabled", "worker_alive", "plans", "warm",
                      "queue_depth"}
