"""engine.rebuild_caches runs in asyncio.to_thread. Every route that rebuilds
holds rules_lock() today, but engine._REBUILD_LOCK is the defensive guard for
a caller that does not (and for the import-time / test rebuilds): if two
rebuild threads overlap without it, a slow stale compile finishes last and
rebinds _COMPILED_RULES to the OLD rules after the newer save has answered."""

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


def test_rebuild_drops_the_old_rule_sets_holdback_specs(monkeypatch):
    """A spec from an older rule set is never served again (the gen keys the
    cache), but it stayed cached until 64 newer specs pushed it out — MBs
    each on a large dictionary map."""
    monkeypatch.setattr(cfg, "PIPELINE_RULES", [_map_rule({"komma eins": ","})])
    try:
        pl_engine.rebuild_caches()
        pl_engine.holdback_start("a komma", language="de")
        assert pl_engine._holdback_spec.cache_info().currsize == 1
        pl_engine.rebuild_caches()
        assert pl_engine._holdback_spec.cache_info().currsize == 0
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()
