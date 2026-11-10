"""Engine tests for the regex-list rule type: entries expand + apply in order,
the whole card toggles together, empty-pattern entries are no-ops, and per-model
EXCLUDE keys on the CARD slug. Exercises the real pipeline engine via the
app_module fixture (which reloads config and recompiles the rules per test, so
cfg/_COMPILED_RULES mutations are isolated)."""

from faster_whisper_backend.pipeline import engine as pl_engine
from tests.settings._rules_helpers import terminal_rule as _term


def _set(app_module, card):
    app_module.cfg.PIPELINE_RULES = [card, _term()]
    pl_engine.rebuild_caches()


def test_regex_list_entries_apply_in_order(app_module):
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
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": "a", "replacement": "b"},
                                  {"pattern": "b", "replacement": "c"}]})
    assert pl_engine._postprocess_text("a", model_name="", extra_excludes={"rl"}) == "a"


def test_seam_culprit_stops_at_the_output_bound(app_module, monkeypatch):
    # Expanding entries compound (~9x each). _postprocess_text stops once the
    # text passes _POSTPROCESS_MAX_CHARS; the streaming seam diagnosis walks
    # the same rules and must stop there too, not keep growing two strings
    # that are already past the bound.
    grow = {"pattern": "[a-z]", "replacement": "\\g<0>" + "x" * 8}
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [grow] * 8})
    bound = 10_000
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", bound)
    inputs: list[int] = []
    real = pl_engine._apply_rule

    def spy(rule, text):
        inputs.append(len(text))
        return real(rule, text)

    monkeypatch.setattr(pl_engine, "_apply_rule", spy)
    got = pl_engine.seam_culprit("ab", "abc", model_name="")
    assert got.startswith("#1.") and got.endswith("(output bound hit)")
    assert max(inputs) <= bound


def test_overflowing_repeat_count_skips_the_entry_not_the_rebuild(app_module):
    # re.compile raises OverflowError, not re.error, for a repeat count past
    # MAXREPEAT; it escaped the skip-bad-entry handler and aborted the rebuild.
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": "a{4294967296}", "replacement": "z"},
                                  {"pattern": "a", "replacement": "b"}]})
    assert pl_engine._postprocess_text("a", model_name="") == "b"
    _set(app_module, {"name": "dd", "label": "DD", "type": "callback:dedup",
                      "enabled": True, "pattern": "a{4294967296}"})
    assert pl_engine._postprocess_text("a", model_name="") == "a"


def test_flag_conflict_and_deep_nesting_skip_the_entry_not_the_rebuild(app_module):
    # re.compile raises ValueError for conflicting inline flags and
    # RecursionError for groups nested too deep; both escaped the
    # skip-bad-entry handler and aborted the rebuild (the dry run skips them).
    for bad in ("(?a)(?u)bar", "(" * 512):
        _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list",
                          "enabled": True,
                          "entries": [{"pattern": "foo", "replacement": "FOO"},
                                      {"pattern": bad, "replacement": "x"}]})
        assert pl_engine._postprocess_text("foo bar", model_name="") == "FOO bar"
        _set(app_module, {"name": "dd", "label": "DD", "type": "callback:dedup",
                          "enabled": True, "pattern": bad})
        assert pl_engine._postprocess_text("bar", model_name="") == "bar"


def test_postprocess_output_bound_holds_during_the_sub(app_module, monkeypatch):
    # The bound used to be checked only after a rule's sub, so the entry that
    # crossed it built its whole result first: one 60x entry on a text just
    # under 4 M chars allocated 240 M. The entry that would pass the bound is
    # now not applied at all and the text from before it is kept.
    dots = {"pattern": r"\.", "replacement": "." * 60}
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [dots] * 3})
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 10_000)
    trace: list = []
    out = pl_engine._postprocess_text("Satz. Satz.", model_name="", trace=trace)
    assert out == "Satz" + "." * 3600 + " Satz" + "." * 3600
    assert any("OUTPUT BOUND" in step[0] for step in trace)


def test_postprocess_output_bound_holds_during_a_map_sub(app_module, monkeypatch):
    # Same for a callback:map rule: a 1-char key may carry a 64-char value.
    from faster_whisper_backend.pipeline import dictation_map

    value = " ".join(["a"] * 32)
    rules = [{"name": f"m{i}", "label": f"M{i}", "type": "callback:map",
              "enabled": True, "map": {"a": value}} for i in range(2)]
    app_module.cfg.PIPELINE_RULES = [*rules, _term()]
    pl_engine.rebuild_caches()
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 10_000)
    text = "a " * 20
    cre, replacer, _lookup = dictation_map.compile_map({"a": value})
    once = cre.sub(replacer, text)
    assert len(once) <= 10_000 < len(cre.sub(replacer, once))
    out = pl_engine._postprocess_text(text, model_name="")
    assert out == once.strip(" \t\r")


def test_postprocess_output_bound_counts_a_lookbehind_capture(app_module, monkeypatch):
    # A group captured in a lookbehind lies outside the zero-width match, so
    # it copies characters the match never consumed: the growth estimate
    # that lets a small text skip the exact count must not apply to it.
    _set(app_module, {"name": "rl", "label": "RL", "type": "regex-list", "enabled": True,
                      "entries": [{"pattern": r"(?<=(.{5}))", "replacement": "\\1" * 10}]})
    monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", 10_000)
    text = "x" * 900
    assert pl_engine._postprocess_text(text, model_name="") == text


_SAMPLE = ("hallo welt Komma das ist ein Test Punkt und noch ein satz Fragezeichen "
           "neue Zeile die Zahl 2 Komma 5 und 1000 Euro, ja, , doch Punkt Punkt "
           "Ausrufezeichen ende")


def test_postprocess_below_bound_is_byte_identical(app_module, monkeypatch):
    # The shipped rule set plus an expanding entry: the result is the same
    # whether the bound is far off (cheap path) or just above the largest
    # intermediate text (every rule takes the exact count first).
    extra = {"name": "grow", "label": "Grow", "type": "regex-list", "enabled": True,
             "entries": [{"pattern": r"(\w)(\w*)", "replacement": "\\g<0>\\1\\2"},
                         {"pattern": r"(?<=(..))!", "replacement": "!\\1"}]}
    app_module.cfg.PIPELINE_RULES = [extra, *app_module.cfg.PIPELINE_RULES]
    pl_engine.rebuild_caches()
    sample = _SAMPLE * 40
    trace: list = []
    ref = pl_engine._postprocess_text(sample, model_name="", trace=trace)
    assert ref != sample
    biggest = max(max(len(b), len(a)) for _label, b, a in trace)
    for bound in (10_000_000, biggest):
        monkeypatch.setattr(pl_engine, "_POSTPROCESS_MAX_CHARS", bound)
        assert pl_engine._postprocess_text(sample, model_name="") == ref
