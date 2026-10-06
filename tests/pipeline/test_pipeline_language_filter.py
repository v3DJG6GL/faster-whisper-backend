"""Language-scoped pipeline rules: a rule with `languages: ["es"]` runs only
when the detected language is Spanish. Empty `languages` means 'apply to all'.
Language mismatch skips even force-INCLUDEd rules (language is a correctness
constraint). Unknown language (None) skips nothing (safe for streaming partials)."""


def _term():
    return {"name": "trim-edges", "label": "Trim", "type": "terminal"}


def _set(app_module, *cards):
    app_module.cfg.PIPELINE_RULES = [*cards, _term()]
    app_module.rebuild_caches()


def _rule(name="rl", pattern="a", replacement="b", enabled=True, languages=None):
    r = {"name": name, "label": name, "type": "regex-list", "enabled": enabled,
         "entries": [{"pattern": pattern, "replacement": replacement}]}
    if languages is not None:
        r["languages"] = languages
    return r


def test_language_match_applies(app_module):
    _set(app_module, _rule(languages=["es"]))
    assert app_module._postprocess_text("a", model_name="", language="es") == "b"


def test_language_mismatch_skips(app_module):
    _set(app_module, _rule(languages=["es"]))
    assert app_module._postprocess_text("a", model_name="", language="de") == "a"


def test_language_empty_applies_to_all(app_module):
    _set(app_module, _rule(languages=[]))
    assert app_module._postprocess_text("a", model_name="", language="de") == "b"


def test_language_none_applies_to_all(app_module):
    """When language is None (streaming partial before detection), no rule is
    skipped — safe default for ephemeral display text."""
    _set(app_module, _rule(languages=["es"]))
    assert app_module._postprocess_text("a", model_name="", language=None) == "b"


def test_language_multi_match(app_module):
    _set(app_module, _rule(languages=["es", "ca"]))
    assert app_module._postprocess_text("a", model_name="", language="ca") == "b"


def test_language_multi_mismatch(app_module):
    _set(app_module, _rule(languages=["es", "ca"]))
    assert app_module._postprocess_text("a", model_name="", language="de") == "a"


def test_language_include_mismatch_still_skips(app_module):
    """Language wins over INCLUDE — a force-INCLUDEd rule with languages:["es"]
    must NOT fire on German audio."""
    _set(app_module, _rule(enabled=False, languages=["es"]))
    app_module.cfg.MODEL_OVERRIDES = {"m": {"PIPELINE_RULES_INCLUDE": ["rl"]}}
    assert app_module._postprocess_text("a", model_name="m", language="de") == "a"


def test_language_include_match_applies(app_module):
    """INCLUDE + matching language: disabled rule force-enabled, language
    matches → runs."""
    _set(app_module, _rule(enabled=False, languages=["es"]))
    app_module.cfg.MODEL_OVERRIDES = {"m": {"PIPELINE_RULES_INCLUDE": ["rl"]}}
    assert app_module._postprocess_text("a", model_name="m", language="es") == "b"


def test_language_exclude_wins(app_module):
    """EXCLUDE beats everything — even a matching language."""
    _set(app_module, _rule(languages=["es"]))
    assert app_module._postprocess_text("a", model_name="", language="es",
                                         extra_excludes={"rl"}) == "a"


def test_language_mismatch_trace_entry(app_module):
    """Trace log records the language skip reason."""
    _set(app_module, _rule(languages=["es"]))
    trace = []
    app_module._postprocess_text("a", model_name="", language="de", trace=trace)
    skip_entries = [t for t in trace if "SKIPPED lang:" in t[0]]
    assert len(skip_entries) == 1
    assert "de" in skip_entries[0][0]
    assert "es" in skip_entries[0][0]


def test_language_default_empty_no_field(app_module):
    """A rule dict without a `languages` key behaves like languages=[] —
    backwards compatible with existing configs."""
    _set(app_module, {"name": "rl", "label": "rl", "type": "regex-list",
                      "enabled": True,
                      "entries": [{"pattern": "a", "replacement": "b"}]})
    assert app_module._postprocess_text("a", model_name="", language="de") == "b"
