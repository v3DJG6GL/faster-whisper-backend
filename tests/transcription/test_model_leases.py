"""Whisper job leases: a model a request is decoding on is never freed.

faster_whisper is never imported — the load path's only heavy dependency is
stubbed in sys.modules, exactly the boundary _get_or_load_model uses.
"""

import asyncio
import threading
import logging
import sys
import time

import pytest
from fastapi import HTTPException

from faster_whisper_backend.runtime import model_registry
from tests.conftest import FakeModel
from faster_whisper_backend.transcription import models as tx_models
from tests.transcription._load_helpers import register_fake_model as _register
from tests.transcription._load_helpers import stub_hub_snapshot as _stub_hub
from tests.transcription._load_helpers import stub_whisper_load as _stub_load
from tests._transcription_helpers import post_transcription as _post


def _hub_response(status, repo):
    """The Hub's HTTP reply to a model-info request for `repo`."""
    import httpx2
    return httpx2.Response(status, request=httpx2.Request(
        "GET", f"https://huggingface.co/api/models/{repo}"))


def test_drop_refuses_while_leased():
    _register("a", leases=1)
    assert tx_models._drop_loaded_model("a") is False
    assert "a" in tx_models._loaded_models
    assert model_registry._loaded_models.get("a") is not None


def test_drop_force_evicts_a_leased_model():
    _register("a", leases=1)
    assert tx_models._drop_loaded_model("a", force=True) is True
    assert "a" not in tx_models._loaded_models
    assert "a" not in model_registry._loaded_models


def test_drop_unleased_still_drops():
    _register("a")
    assert tx_models._drop_loaded_model("a") is True
    assert "a" not in tx_models._loaded_models


def test_lru_skips_a_leased_entry(monkeypatch):
    _stub_load(monkeypatch, max_loaded=2)
    _register("a")
    _register("b")
    tx_models._model_leases["a"] = 1  # oldest, but in use

    asyncio.run(tx_models._get_or_load_model("c"))

    # "b" was the first UNLEASED entry, so it paid instead of "a".
    assert set(tx_models._loaded_models) == {"a", "c"}


def test_all_leased_overflows_the_cap(monkeypatch, caplog):
    _stub_load(monkeypatch, max_loaded=1)
    _register("a", leases=1)

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        asyncio.run(tx_models._get_or_load_model("b"))

    assert set(tx_models._loaded_models) == {"a", "b"}
    assert any("temporarily exceeding MAX_LOADED_MODELS" in r.getMessage()
               for r in caplog.records)


def _doomed_constructor(built):
    """WhisperModel that raises faster-whisper's ValueError for an unknown
    short size and records every build, so a test can assert none ran."""
    def _build(path, **kw):
        built.append(path)
        if path not in sys.modules["faster_whisper"].available_models():
            raise ValueError(f"Invalid model size '{path}'")
        return FakeModel()
    return _build


def test_unknown_short_name_evicts_nothing(monkeypatch):
    """A typo'd size can only fail in the constructor, so it is rejected
    before the LRU loop and costs the hot model no cold reload."""
    built = []
    _stub_load(monkeypatch, whisper_model=_doomed_constructor(built),
               max_loaded=1)
    _register("a")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model("largev3", lease=True))

    assert exc.value.status_code == 400
    assert set(tx_models._loaded_models) == {"a"}
    assert built == [] and tx_models._model_leases == {}


def test_unknown_short_name_fails_the_shared_gate(monkeypatch):
    """_check_model_name is what the preload, the URL language check and
    GET /v1/request-default-settings call to refuse what the loader refuses,
    so the short-name rule lives there, not only inside the load."""
    _stub_load(monkeypatch)
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)

    with pytest.raises(HTTPException) as exc:
        tx_models._check_model_name("largev3")

    assert exc.value.status_code == 400
    tx_models._check_model_name("a")
    tx_models._check_model_name("org/repo")


@pytest.mark.parametrize("field, value", [
    ("DEFAULT_MODEL", "my-model"),
    ("ALLOWED_MODELS", {"my-model"}),
    ("PRELOAD_MODELS", ["my-model"]),
    ("STREAMING_PARTIAL_MODEL", "my-model"),
    ("MODEL_OVERRIDES", {"my-model": {"BEAM_SIZE": 1}}),
])
def test_short_name_of_a_local_dir_still_loads(monkeypatch, tmp_path,
                                               field, value):
    _stub_load(monkeypatch, max_loaded=4)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my-model").mkdir()
    monkeypatch.setattr(tx_models.cfg, field, value, raising=False)

    asyncio.run(tx_models._get_or_load_model("my-model"))

    assert "my-model" in tx_models._loaded_models


def test_an_operator_named_two_level_dir_still_loads(monkeypatch, tmp_path):
    _stub_load(monkeypatch, max_loaded=4)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "my-ft").mkdir(parents=True)
    monkeypatch.setattr(tx_models.cfg, "PRELOAD_MODELS", ["models/my-ft"],
                        raising=False)

    tx_models._check_model_name("models/my-ft")
    asyncio.run(tx_models._get_or_load_model("models/my-ft"))

    assert "models/my-ft" in tx_models._loaded_models


@pytest.mark.parametrize("name", ["static", "docs/brand"])
def test_an_unconfigured_dir_fails_the_shared_gate(monkeypatch, tmp_path,
                                                   name):
    """With no allowlist, a client's "static" or "docs/brand" is a directory
    under the server's cwd too: refused up front (preload included), not
    loaded after the LRU loop evicted the hot model for it. The two-level one
    has the org/repo shape and would skip the Hub pre-download."""
    built = []
    _stub_load(monkeypatch, whisper_model=_doomed_constructor(built),
               max_loaded=1)
    monkeypatch.chdir(tmp_path)
    (tmp_path / name).mkdir(parents=True)
    monkeypatch.setattr(tx_models.cfg, "DEFAULT_MODEL", "a", raising=False)
    monkeypatch.setattr(tx_models.cfg, "PRELOAD_MODELS", [], raising=False)
    _register("a")

    with pytest.raises(HTTPException) as exc:
        tx_models._check_model_name(name)
    assert exc.value.status_code == 400
    # The reason and the fix, not the short-name "not a ... repo id" wording
    # ("docs/brand" has the repo-id shape).
    assert "local directory" in exc.value.detail
    assert "ALLOWED_MODELS" in exc.value.detail
    assert "repo id" not in exc.value.detail
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model(name, lease=True))

    assert exc.value.status_code == 400
    assert set(tx_models._loaded_models) == {"a"}
    assert built == [] and tx_models._model_leases == {}


@pytest.mark.parametrize("absolute", [False, True])
def test_a_resident_dir_is_refused_once_default_model_moves_off_it(
        monkeypatch, tmp_path, absolute):
    """DEFAULT_MODEL hot-applies and evicts nothing: the cache-hit fast path
    must refuse the withdrawn dir as _check_model_name does, not serve it
    until an LRU eviction or a restart."""
    _stub_load(monkeypatch, max_loaded=4)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "my-ft").mkdir(parents=True)
    name = str(tmp_path / "models" / "my-ft") if absolute else "models/my-ft"
    monkeypatch.setattr(tx_models.cfg, "PRELOAD_MODELS", [], raising=False)
    monkeypatch.setattr(tx_models.cfg, "DEFAULT_MODEL", name, raising=False)
    asyncio.run(tx_models._get_or_load_model(name))
    monkeypatch.setattr(tx_models.cfg, "DEFAULT_MODEL", "a", raising=False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model(name, lease=True))

    assert exc.value.status_code == 400
    assert name in tx_models._loaded_models     # resident, still refused
    assert tx_models._model_leases == {}


@pytest.mark.parametrize("err", ["RepositoryNotFoundError",
                                 "RevisionNotFoundError"])
def test_repo_missing_on_the_hub_evicts_nothing(monkeypatch, err):
    """The constructor's own snapshot_download (same arguments) would raise
    these too, so the load fails before the LRU loop instead of swallowing
    them as a best-effort pre-download failure."""
    from huggingface_hub import errors as hf_errors
    built = []
    _stub_load(monkeypatch, whisper_model=_doomed_constructor(built),
               max_loaded=1)
    response = _hub_response(404, "org/nope")

    def _missing(**kw):
        raise getattr(hf_errors, err)("not found", response=response)
    _stub_hub(monkeypatch, _missing)
    _register("a")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model("org/nope", lease=True))

    assert exc.value.status_code == 400
    assert err in exc.value.detail
    assert set(tx_models._loaded_models) == {"a"}
    assert built == [] and tx_models._model_leases == {}


def _concurrent_cold_loads(monkeypatch, snapshot, n=4):
    """n concurrent misses for one cold Hub repo; their outcomes."""
    _stub_load(monkeypatch, max_loaded=4)
    monkeypatch.setattr(tx_models, "_model_load_lock", asyncio.Lock())
    _stub_hub(monkeypatch, snapshot)

    async def run():
        return await asyncio.gather(
            *(tx_models._get_or_load_model("org/cold") for _ in range(n)),
            return_exceptions=True)
    return asyncio.run(run())


def test_concurrent_misses_share_one_hub_download(monkeypatch):
    """Concurrent misses for one cold repo share one snapshot_download: one
    each would pin a default-executor worker per later caller on the blob
    lock."""
    calls = []

    def _slow(**kw):
        calls.append(kw["repo_id"])
        time.sleep(0.2)
    got = _concurrent_cold_loads(monkeypatch, _slow)

    assert calls == ["org/cold"]
    assert all(m is tx_models._loaded_models["org/cold"] for m in got)
    assert tx_models._hub_prefetches == {}


def test_a_shared_hub_refusal_reaches_every_caller_and_is_retried(
        monkeypatch):
    from huggingface_hub import errors as hf_errors
    calls = []
    response = _hub_response(404, "org/cold")

    def _missing(**kw):
        calls.append(kw["repo_id"])
        time.sleep(0.2)
        raise hf_errors.RepositoryNotFoundError("not found", response=response)
    got = _concurrent_cold_loads(monkeypatch, _missing)

    assert len(calls) == 1
    assert [getattr(e, "status_code", e) for e in got] == [400] * 4
    assert tx_models._hub_prefetches == {}
    # Not cached: the next miss asks the Hub again.
    _concurrent_cold_loads(monkeypatch, _missing, n=1)
    assert len(calls) == 2


@pytest.mark.parametrize("err, said", [
    ("GatedRepoError", "gated"),
    ("RepositoryNotFoundError", "private or gated"),
])
def test_hub_refusal_of_an_existing_repo_points_at_the_token(
        monkeypatch, caplog, err, said):
    """A gated repo (GatedRepoError, a RepositoryNotFoundError subclass) or a
    private one without the right token: the client is told to check
    HF_TOKEN / access, not only to hunt a typo, and the start of the Hub's
    own text (log_safe caps it at LOG_FIELD_MAX) reaches the log."""
    from huggingface_hub import errors as hf_errors
    _stub_load(monkeypatch)
    response = _hub_response(403, "org/gated")

    def _refused(**kw):
        raise getattr(hf_errors, err)("Access to model org/gated is "
                                      "restricted.\nVisit the page.",
                                      response=response)
    _stub_hub(monkeypatch, _refused)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(tx_models._get_or_load_model("org/gated"))

    assert exc.value.status_code == 400
    assert said in exc.value.detail and "HF_TOKEN" in exc.value.detail
    assert err in exc.value.detail
    assert any("is restricted.?Visit" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("status, refused", [(401, True), (503, False)])
def test_hub_rejecting_the_token_evicts_nothing(monkeypatch, status, refused):
    """A wrong or revoked HF_TOKEN is a plain HfHubHTTPError 401 (not
    RepositoryNotFoundError), on public repos too, and the constructor would
    send the same token: refused before the LRU loop with a pointer at the
    token. Any other HTTP error stays a best-effort pre-download failure."""
    from huggingface_hub import errors as hf_errors
    built = []
    _stub_load(monkeypatch,
               whisper_model=lambda path, **kw: built.append(path) or FakeModel(),
               max_loaded=1)
    response = _hub_response(status, "org/public")

    def _rejected(**kw):
        raise hf_errors.HfHubHTTPError("Invalid credentials in Authorization "
                                       "header", response=response)
    _stub_hub(monkeypatch, _rejected)
    _register("a")

    if not refused:
        asyncio.run(tx_models._get_or_load_model("org/public"))
        assert built == ["org/public"]
        return
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model("org/public", lease=True))

    assert exc.value.status_code == 400
    assert "rejected HF_TOKEN" in exc.value.detail
    assert set(tx_models._loaded_models) == {"a"}
    assert built == [] and tx_models._model_leases == {}


def test_lease_taken_on_cache_hit_and_on_load(monkeypatch):
    _stub_load(monkeypatch, max_loaded=4)
    _register("a")

    asyncio.run(tx_models._get_or_load_model("a", lease=True))   # lock-free hit
    asyncio.run(tx_models._get_or_load_model("b", lease=True))   # fresh load
    assert tx_models._model_leases == {"a": 1, "b": 1}

    asyncio.run(tx_models._get_or_load_model("a"))               # no lease asked
    assert tx_models._model_leases == {"a": 1, "b": 1}


def test_rejected_names_take_no_lease(monkeypatch):
    _stub_load(monkeypatch)
    # Allowlist gate: it sits before the cache and the traversal guard.
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", {"a"}, raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tx_models._get_or_load_model("b", lease=True))
    assert exc.value.status_code == 400
    assert tx_models._model_leases == {}

    # No allowlist: the id-shape guard is the only thing between the request
    # string and os.path.isdir() / the converter (DEFAULT_MODEL stays exempt).
    monkeypatch.setattr(tx_models.cfg, "ALLOWED_MODELS", set(), raising=False)
    monkeypatch.setattr(tx_models.cfg, "DEFAULT_MODEL", "base", raising=False)
    for bad in ("../etc/passwd", "http://x/y"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(tx_models._get_or_load_model(bad, lease=True))
        assert exc.value.status_code == 400
        assert tx_models._model_leases == {}


def test_release_restamps_last_used():
    _register("a", leases=1)
    info = model_registry._loaded_models["a"]
    info["last_used_monotonic"] = time.monotonic() - 500
    stale = info["last_used_monotonic"]

    tx_models._release_model_lease("a", tx_models._loaded_models["a"])

    assert tx_models._model_leases == {}
    assert model_registry._loaded_models["a"]["last_used_monotonic"] > stale


def test_release_decrements_before_zero():
    _register("a", leases=2)
    tx_models._release_model_lease("a", tx_models._loaded_models["a"])
    assert tx_models._model_leases == {"a": 1}
    assert tx_models._drop_loaded_model("a") is False


def test_release_tolerates_a_dropped_model():
    _register("a", leases=1)
    old = tx_models._loaded_models["a"]
    tx_models._drop_loaded_model("a", force=True)   # drain_then_evict shape
    tx_models._release_model_lease("a", old)        # must not KeyError
    assert tx_models._model_leases == {}


def test_a_reload_after_drain_does_not_inherit_the_old_lease(monkeypatch):
    # R1 still decodes on the instance drain_then_evict dropped; the reload
    # nobody leased must stay evictable, or a miss for "b" overflows the cap.
    _stub_load(monkeypatch, max_loaded=1)
    _register("a", leases=1)

    async def scenario():
        assert await tx_models.drain_then_evict("a") == ["a"]
        await tx_models._get_or_load_model("a")
        await tx_models._get_or_load_model("b")

    asyncio.run(scenario())
    assert list(tx_models._loaded_models) == ["b"]
    assert tx_models._model_leases == {}


def test_an_old_holder_release_leaves_the_reload_lease(monkeypatch):
    _stub_load(monkeypatch, max_loaded=1)
    _register("a", leases=1)
    old = tx_models._loaded_models["a"]

    async def scenario():
        await tx_models.drain_then_evict("a")
        await tx_models._get_or_load_model("a", lease=True)   # R2 on the reload
        tx_models._release_model_lease("a", old)              # R1, old instance
        await tx_models._get_or_load_model("b")

    asyncio.run(scenario())
    # R2 still decodes on the reload: it must survive the miss for "b".
    assert tx_models._model_leases == {"a": 1}
    assert "a" in tx_models._loaded_models
    tx_models._release_model_lease("a", tx_models._loaded_models["a"])
    assert tx_models._model_leases == {}


def test_a_reload_holder_release_leaves_the_old_holder_lease_orphaned(
        monkeypatch):
    # The reverse order: R1 (a dictation, leased until teardown) still decodes
    # on the drained instance when the short R2 on the reload ends. R2's
    # release must free the reload, or a miss for "b" overflows the cap for
    # the rest of R1's session.
    _stub_load(monkeypatch, max_loaded=1)
    _register("a", leases=1)
    old = tx_models._loaded_models["a"]

    async def scenario():
        await tx_models.drain_then_evict("a")
        reload = await tx_models._get_or_load_model("a", lease=True)
        tx_models._release_model_lease("a", reload)           # R2 ends first
        await tx_models._get_or_load_model("b")

    asyncio.run(scenario())
    assert list(tx_models._loaded_models) == ["b"]
    assert tx_models._model_leases == {}
    tx_models._release_model_lease("a", old)                  # R1 ends
    assert tx_models._orphan_leases == {} and tx_models._model_leases == {}


def test_a_failed_register_leaves_no_lease_a_cache_hit_took_meanwhile(
        monkeypatch):
    # The model is cached before the register write, so a lock-free cache hit
    # leases it while that write runs. When the write then raises, that
    # holder's release must not leave a count behind that pins the next load
    # of "a" (a miss for "b" overflowed the cap until restart).
    _stub_load(monkeypatch, max_loaded=1)
    entered, go = threading.Event(), threading.Event()
    real_register = model_registry.register_loaded_model

    def _failing_register(*a, **kw):
        real_register(*a, **kw)
        if not entered.is_set():   # only the first load's write fails
            entered.set()
            go.wait(5)
            # Raised past the insert: the registry entry is in already.
            raise RuntimeError("registry write failed")
    monkeypatch.setattr(model_registry, "register_loaded_model",
                        _failing_register)

    async def scenario():
        loader = asyncio.create_task(
            tx_models._get_or_load_model("a", lease=True))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        held = await tx_models._get_or_load_model("a", lease=True)
        go.set()
        with pytest.raises(RuntimeError):
            await loader
        tx_models._release_model_lease("a", held)
        assert "a" not in model_registry._loaded_models
        await tx_models._get_or_load_model("a")
        await tx_models._get_or_load_model("b")

    try:
        asyncio.run(scenario())
    finally:
        go.set()
    assert list(tx_models._loaded_models) == ["b"]
    assert tx_models._orphan_leases == {} and tx_models._model_leases == {}


def test_last_release_trims_a_cap_lowered_under_leases(monkeypatch):
    # The cap was lowered while every resident model was leased, so
    # trim_to_cap skipped them all; later requests only hit the cache, so no
    # miss ever reclaims the excess. The last release must.
    _register("a")
    _register("b")
    tx_models._model_leases.update({"a": 1, "b": 1})
    _stub_load(monkeypatch, max_loaded=1)

    async def scenario():
        assert await tx_models.trim_to_cap() == []
        tx_models._release_model_lease("a", tx_models._loaded_models["a"])
        tx_models._release_model_lease("b", tx_models._loaded_models["b"])
        await tx_models._trim_task
        await tx_models._get_or_load_model("b")        # a hit, not a miss

    asyncio.run(scenario())
    assert list(tx_models._loaded_models) == ["b"]


def test_release_within_the_cap_schedules_no_trim(monkeypatch):
    _register("a", leases=1)
    monkeypatch.setattr(tx_models.cfg, "MAX_LOADED_MODELS", 1, raising=False)

    async def scenario():
        tx_models._release_model_lease("a", tx_models._loaded_models["a"])
        return tx_models._trim_task

    assert asyncio.run(scenario()) is None
    assert list(tx_models._loaded_models) == ["a"]


def test_release_in_an_all_leased_overflow_keeps_the_released_model(
        monkeypatch):
    # A long streaming session pins "a"; each batch request for "b" overflows
    # the cap. Its release must not evict "b" (the only unleased entry), or
    # every later "b" request is a cold reload.
    builds: "list[str]" = []

    def _model(path, **kw):
        builds.append(path)
        return FakeModel()
    _stub_load(monkeypatch, whisper_model=_model, max_loaded=1)
    _register("a", leases=1)

    async def scenario():
        for _ in range(3):
            await tx_models._get_or_load_model("b", lease=True)
            tx_models._release_model_lease("b", tx_models._loaded_models["b"])
            if tx_models._trim_task is not None:
                await tx_models._trim_task

    asyncio.run(scenario())
    assert builds == ["b"]
    assert list(tx_models._loaded_models) == ["a", "b"]


def test_a_leased_load_keeps_a_release_scheduled_trim_from_evicting(
        monkeypatch):
    # "a" is pinned and "b" overflows the cap; "b"'s release schedules a trim
    # that spares it while a request for "c" is still building. Caching the
    # leased "c" must not take that protection away from "b".
    entered, gate = threading.Event(), threading.Event()

    def _model(path, **kw):
        if path == "c":
            entered.set()
            gate.wait(5)
        return FakeModel()
    _stub_load(monkeypatch, whisper_model=_model, max_loaded=1)
    _register("a", leases=1)

    async def scenario():
        await tx_models._get_or_load_model("b", lease=True)
        load = asyncio.ensure_future(
            tx_models._get_or_load_model("c", lease=True))
        await asyncio.to_thread(entered.wait, 5)
        tx_models._release_model_lease("b", tx_models._loaded_models["b"])
        gate.set()
        await load
        await tx_models._trim_task

    asyncio.run(scenario())
    assert sorted(tx_models._loaded_models) == ["a", "b", "c"]


def test_release_during_an_unleased_over_cap_load_trims_after_it(monkeypatch):
    # "a" is leased, so the preload of "b" overflows the cap; "a"'s lease goes
    # while "b" is still being built (nothing over the cap yet to trim), and
    # the unleased "b" never releases a lease. The load itself must reclaim.
    entered, gate = threading.Event(), threading.Event()

    def _slow_model(path, **kw):
        entered.set()
        gate.wait(5)
        return FakeModel()
    _stub_load(monkeypatch, whisper_model=_slow_model, max_loaded=1)
    _register("a", leases=1)

    async def scenario():
        load = asyncio.ensure_future(tx_models.load_unleased("b"))
        await asyncio.to_thread(entered.wait, 5)
        tx_models._release_model_lease("a", tx_models._loaded_models["a"])
        gate.set()
        await load
        if tx_models._trim_task is not None:
            await tx_models._trim_task
        await tx_models._get_or_load_model("b")        # a hit, not a miss

    asyncio.run(scenario())
    assert list(tx_models._loaded_models) == ["b"]


def test_idle_evictor_defers_then_evicts(monkeypatch):
    _register("a", leases=1)
    model_registry._loaded_models["a"]["last_used_monotonic"] = \
        time.monotonic() - 500
    monkeypatch.setattr(tx_models.cfg, "MODEL_IDLE_TIMEOUT_S", 1, raising=False)

    ticks = {"n": 0}
    seen: "list[bool]" = []

    async def _fake_sleep(_secs):
        ticks["n"] += 1
        if ticks["n"] == 2:
            # After the first (refused) sweep.
            seen.append("a" in tx_models._loaded_models)
            tx_models._release_model_lease("a", tx_models._loaded_models["a"])
            model_registry._loaded_models["a"]["last_used_monotonic"] = \
                time.monotonic() - 500
        if ticks["n"] > 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(tx_models.asyncio, "sleep", _fake_sleep)
    asyncio.run(tx_models._idle_evictor())

    assert seen == [True]                      # the lease deferred the evict
    assert "a" not in tx_models._loaded_models      # the release let it through


def _leasing_loader(app_module, model, held):
    """`held` records every name leased, so the tests can key the mid-decode
    snapshot off the RESOLVED name the handler asked for, not "whisper-1"."""
    async def _loader(name, *, lease=False):
        # Cached like a real load: the release tells it from a drained one.
        tx_models._loaded_models[name] = model
        if lease:
            held.append(name)
            tx_models._model_leases[name] = \
                tx_models._model_leases.get(name, 0) + 1
        return model
    return _loader


class _SnapshotModel(FakeModel):
    """Records the live lease map at the moment the decode runs — the
    property the lease exists for is "held DURING the decode"."""

    def __init__(self, app_module, boom=None):
        super().__init__()
        self._app = app_module
        self._boom = boom
        self.leases_during = None

    def transcribe(self, path, **kwargs):
        self.leases_during = dict(tx_models._model_leases)
        if self._boom is not None:
            raise self._boom
        return super().transcribe(path, **kwargs)


def test_transcribe_releases_the_lease_on_success(client, app_module,
                                                  monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module)
    monkeypatch.setattr(tx_models, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = _post(client)
    assert r.status_code == 200
    assert held and model.leases_during == {held[0]: 1}
    assert tx_models._model_leases == {}


def test_transcribe_releases_the_lease_on_error(client, app_module,
                                                monkeypatch):
    held: "list[str]" = []
    model = _SnapshotModel(app_module, boom=RuntimeError("decode blew up"))
    monkeypatch.setattr(tx_models, "_get_or_load_model",
                        _leasing_loader(app_module, model, held))
    r = _post(client)
    assert r.status_code == 500
    assert held and model.leases_during == {held[0]: 1}
    assert tx_models._model_leases == {}


def test_register_loaded_model_runs_off_the_loop(monkeypatch):
    """register_loaded_model persists the measurement (model_sizes.record ->
    atomic_json.save_lock + fsync) and can block for the lock timeout when
    a peer worker holds the ledger; the load path must hand it to a thread."""
    _stub_load(monkeypatch, max_loaded=4)
    seen = {}

    def fake_register(name, **kw):
        seen[name] = threading.current_thread()
    monkeypatch.setattr(model_registry, "register_loaded_model", fake_register)

    asyncio.run(tx_models._get_or_load_model("a"))

    assert "a" in tx_models._loaded_models
    assert seen["a"] is not threading.main_thread()
