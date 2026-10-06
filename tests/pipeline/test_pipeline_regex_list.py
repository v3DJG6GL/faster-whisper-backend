"""Engine tests for the regex-list rule type: entries expand + apply in order,
the whole card toggles together, empty-pattern entries are no-ops, and per-model
EXCLUDE keys on the CARD slug. Exercises the real pipeline engine via the
app_module fixture (which reloads config and recompiles the rules per test, so
cfg/_COMPILED_RULES mutations are isolated)."""

from faster_whisper_backend.pipeline import engine as pl_engine


def _term():
    return {"name": "trim-edges", "label": "Trim", "type": "terminal"}


def _set(app_module, card):
    app_module.cfg.PIPELINE_RULES = [card, _term()]
    pl_engine.rebuild_caches()


def test_regex_list_entries_apply_in_order(app_module):
    # entry N's output feeds N+1 : a -> b -> c
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": "a", "replacement": "b"},
                                  {"pattern": "b", "replacement": "c"}]})
    assert pl_engine._postprocess_text("a", model_name="") == "c"


def test_regex_list_disabled_card_skips_all_entries(app_module):
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": False,
                      "entries": [{"pattern": "a", "replacement": "b"}]})
    assert pl_engine._postprocess_text("a", model_name="") == "a"


def test_regex_list_empty_pattern_entry_is_noop(app_module):
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": "", "replacement": "z"},
                                  {"pattern": "a", "replacement": "b"}]})
    assert pl_engine._postprocess_text("a", model_name="") == "b"


def test_regex_list_excluded_by_card_slug(app_module):
    # The whole card is the EXCLUDE unit — all expanded entries share the slug.
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": "a", "replacement": "b"},
                                  {"pattern": "b", "replacement": "c"}]})
    assert pl_engine._postprocess_text("a", model_name="", extra_excludes={"rl"}) == "a"
