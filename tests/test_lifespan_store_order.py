"""Startup ordering: the recent-transcriptions store must be open BEFORE the
model preload loop, because every preload download persists its 'download'
recent-jobs row through it (download_progress.capture -> metrics.record_download
-> recent_transcriptions_store.record_timing). With the store still closed those rows
were lost with a "persist failed" warning on exactly the boot that matters."""

from fastapi.testclient import TestClient


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
    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)

    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        pass
    assert seen == {"recent_open": True, "usage_open": True}
