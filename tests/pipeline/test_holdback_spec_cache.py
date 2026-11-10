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
        assert pl_engine._holdback_spec.cache_info().currsize == 1
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
        assert pl_engine._holdback_spec.cache_info().currsize == 2
    finally:
        monkeypatch.undo()
        pl_engine.rebuild_caches()
