"""engine.rebuild_caches runs in asyncio.to_thread and not every hot-apply
caller holds rules_lock() (a SUPPRESS_CHARS-only admin save does not), so two
rebuild threads can overlap. Without engine._REBUILD_LOCK a slow stale compile
finished last and rebound _COMPILED_RULES to the OLD rules after the newer
save had already answered 200."""

import threading

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg


def _map_rule(m):
    return {"name": "m", "label": "m", "type": "callback:map", "enabled": True, "map": m}


def test_overlapping_rebuilds_end_on_the_latest_rules(monkeypatch):
    real_compile = pl_engine._dictation_map.compile_map
    started = threading.Event()
    calls = {"n": 0}

    def slow_first_compile(m):
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            # Long enough for the second thread to compile and finish first
            # if nothing serialises the two rebuilds.
            threading.Event().wait(0.3)
        return real_compile(m)

    monkeypatch.setattr(pl_engine._dictation_map, "compile_map", slow_first_compile)
    monkeypatch.setattr(cfg, "PIPELINE_RULES", [_map_rule({"alt": "OLD"})])
    gen0 = pl_engine._RULES_GEN
    try:
        t_a = threading.Thread(target=pl_engine.rebuild_caches)
        t_a.start()
        assert started.wait(5)
        cfg.PIPELINE_RULES = [_map_rule({"neu": "NEW"})]
        t_b = threading.Thread(target=pl_engine.rebuild_caches)
        t_b.start()
        t_a.join(5)
        t_b.join(5)
        lookups = [r.map_lookup for r in pl_engine._COMPILED_RULES]
        assert lookups == [{"neu": "NEW"}]
        assert pl_engine._RULES_GEN == gen0 + 2
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()
