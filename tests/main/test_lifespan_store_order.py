"""Startup ordering: the recent-transcriptions store must be open BEFORE the
model preload loop, because every preload download persists its 'download'
recent-jobs row through it (download_progress.capture -> metrics.record_download
-> recent_transcriptions_store.record_timing). With the store still closed those rows
were lost with a "persist failed" warning on exactly the boot that matters."""

from fastapi.testclient import TestClient
from faster_whisper_backend.transcription import models as tx_models


def test_recent_store_is_open_when_the_preload_loop_runs(app_module,
                                                         monkeypatch,
                                                         fake_model):
    from faster_whisper_backend.stats import recent_transcriptions_store
    from faster_whisper_backend.stats import usage_store

    monkeypatch.setattr(app_module.cfg, "PRELOAD_MODELS", ["tiny"],
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "ALLOWED_MODELS", set(), raising=False)
    seen = {}

    async def _loader(name, *, lease=False):
        # What the preload's download hook would do: touch the store.
        seen["recent_open"] = recent_transcriptions_store._conn is not None
        seen["usage_open"] = usage_store._conn is not None
        recent_transcriptions_store._require_conn()
        return fake_model
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)
    # An earlier test may have left the stores bound; unbind so a passing
    # assertion proves THIS lifespan opened them before the preload ran.
    monkeypatch.setattr(recent_transcriptions_store, "_conn", None)
    monkeypatch.setattr(usage_store, "_conn", None)

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        pass
    assert seen == {"recent_open": True, "usage_open": True}


def test_failing_startup_sweeps_still_start_the_hourly_loops(app_module,
                                                             monkeypatch):
    # A transient "database is locked" in a store's one-time startup sweep or
    # reconcile used to skip that store's hourly retention loop for the whole
    # process lifetime, while the open store kept taking writes.
    from faster_whisper_backend.captures import samples_store
    from faster_whisper_backend.captures import store as captures_store
    from faster_whisper_backend.reports import store as reports_store
    from faster_whisper_backend.stats import recent_transcriptions_store
    from faster_whisper_backend.transcription import jobs_store

    def _boom(*_a, **_k):
        raise RuntimeError("database is locked")
    for mod, name in ((recent_transcriptions_store, "sweep_retention"),
                      (reports_store, "sweep_retention"),
                      (captures_store, "reconcile_on_startup"),
                      (samples_store, "reconcile_on_startup"),
                      (captures_store, "sweep_retention"),
                      (jobs_store, "mark_running_as_failed"),
                      (jobs_store, "sweep_retention")):
        monkeypatch.setattr(mod, name, _boom)

    started = []
    for loop in ("_recent_tx_retention_loop", "_reports_retention_loop",
                 "_captures_retention_loop", "_jobs_retention_loop"):
        async def _stub(_loop=loop):
            started.append(_loop)
        monkeypatch.setattr(app_module, loop, _stub)

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        pass
    assert sorted(started) == ["_captures_retention_loop",
                               "_jobs_retention_loop",
                               "_recent_tx_retention_loop",
                               "_reports_retention_loop"]
