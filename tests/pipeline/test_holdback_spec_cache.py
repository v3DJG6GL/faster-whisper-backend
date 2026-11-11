"""engine._holdback_spec caches one seam hold-back spec per SELECTION of
callback:map rows, not per (scoping, language) key: a spec on a large
dictionary map costs a long build and tens of MB, and every language that runs
the same rows needs the identical spec."""

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import config as cfg


def _map_rule(m, languages=None):
    r = {"name": "m", "label": "m", "type": "callback:map", "enabled": True, "map": m}
    if languages is not None:
        r["languages"] = languages
    return r


def test_holdback_spec_shared_across_languages_for_untagged_map(monkeypatch):
    monkeypatch.setattr(cfg, "PIPELINE_RULES", [_map_rule({"komma eins": ","})])
    try:
        pl_engine.rebuild_caches()
        starts = [pl_engine.holdback_start("a komma", language=lang)
                  for lang in ("de", "en", "fr", "und", None)]
        assert starts == [1] * 5
        # One warm spec serves every language; nothing falls through to the LRU.
        assert len(pl_engine._WARM_SPECS) == 1
        assert pl_engine._holdback_spec.cache_info().currsize == 0
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()


def test_language_tagged_map_still_holds_only_for_its_language(monkeypatch):
    monkeypatch.setattr(cfg, "PIPELINE_RULES",
                        [_map_rule({"komma eins": ","}, languages=["de"])])
    try:
        pl_engine.rebuild_caches()
        assert pl_engine.holdback_start("a komma", language="de") == 1
        assert pl_engine.holdback_start("a komma", language="en") == len("a komma")
        # Every language outside the tag selects no rows and shares one spec.
        assert pl_engine.holdback_start("a komma", language="fr") == len("a komma")
        assert len(pl_engine._WARM_SPECS) == 2
        assert pl_engine._holdback_spec.cache_info().currsize == 0
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()


def test_rebuild_warms_default_selections_off_loop(monkeypatch):
    """rebuild_caches (a worker thread) builds the spec for every selection a
    session without per-identity / per-model scoping can make, so the first
    streaming compose after a save — on the event loop — never builds one;
    on a large map that build stalls the loop for most of a second."""
    monkeypatch.setattr(cfg, "PIPELINE_RULES",
                        [_map_rule({"komma eins": ","}, languages=["de"])])
    try:
        pl_engine.rebuild_caches()

        def _no_build(_lookups):
            raise AssertionError("hold-back spec built outside rebuild_caches")

        with monkeypatch.context() as m:
            m.setattr(pl_engine._seam_holdback, "build_spec", _no_build)
            assert pl_engine.holdback_start("a komma", language="de") == 1
            assert pl_engine.holdback_start("a komma", language=None) == 1
            assert pl_engine.holdback_start("a komma", language="en") == len("a komma")
        # A warm spec is the one the lazy path builds for the same selection.
        for (gen, sel), spec in pl_engine._WARM_SPECS.items():
            assert spec == pl_engine._holdback_spec(gen, sel)
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()
