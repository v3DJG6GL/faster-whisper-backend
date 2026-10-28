"""A repeat registration while the plan's own load is running.

Once the worker has dequeued an entry, the family's load_in_progress() is
that very load. A re-POST of the same plan (or the job's binding into the
client's plan) must report the entry `queued` from plan.inflight, not run
the admission ladder and read its own load as `family_busy`.
"""

import asyncio

import pytest

from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.translation import engine as translation


def _enable(monkeypatch):
    from faster_whisper_backend.settings import config as cfg
    for k, v in {
        "MODEL_PRELOAD_ENABLED": True,
        "MODEL_PRELOAD_WARM_TTL_S": 180,
        "MODEL_PRELOAD_EVICT_IDLE_MODELS": True,
        "TRANSLATION_ENABLED": True,
        "MODEL_DEVICE": "cpu",
        "MODEL_COMPUTE_TYPE": "int8",
        "MAX_LOADED_MODELS": 2,
        "ALLOWED_MODELS": set(),
        "DEFAULT_MODEL": "small",
    }.items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    monkeypatch.setattr(model_sizes, "fits", lambda *a, **k: (True, None))


@pytest.mark.parametrize("family, model_id, busy_module", [
    ("whisper", "medium", tx_models),
    ("translation", "o/r:Q4", translation),
])
def test_repost_during_the_plans_own_load_reports_queued(
        monkeypatch, family, model_id, busy_module):
    _enable(monkeypatch)
    pid = "e" * 8

    async def _run():
        await preload.start()
        preload._worker.cancel()
        first = preload.register_plan("u", [(family, model_id)], plan_id=pid)
        assert first["models"][0]["state"] in ("loading", "queued")
        await asyncio.sleep(0)
        assert preload._queue.qsize() == 1
        # The worker has taken the item and is loading it.
        preload._queue.get_nowait()
        monkeypatch.setattr(busy_module, "load_in_progress", lambda: True)

        for trig in (None, "job"):
            again = preload.register_plan("u", [(family, model_id)],
                                          plan_id=pid, trigger=trig)
            assert again["models"] == [
                {"family": family, "id": model_id, "state": "queued"}]
        await asyncio.sleep(0)
        assert preload._queue.qsize() == 0      # not enqueued a second time
        assert preload._plans[pid].inflight == {
            preload.stats_key(family, model_id)}
    asyncio.run(_run())
