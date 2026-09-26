"""Tests for the committed factory pipeline-rules layer (config.json).

Covers config_store.load_factory_rules / save_factory_rules: validation,
round-trip, the `note` field, the terminal-rule invariant, and the
fail-fast behaviour on a missing/corrupt file.

Runnable two ways:
    pytest test_factory_rules.py
    python  test_factory_rules.py        (no pytest needed)

Only depends on pydantic (same as config_store) — not the full app stack.
"""

import json
import os
import tempfile

from pydantic import ValidationError

from faster_whisper_backend import config_store as cs


def _regex_rule(name, pattern="x", replacement="y", **kw):
    # A one-entry regex-list == a former single `regex` rule.
    r = {"name": name, "label": name, "type": "regex-list",
         "entries": [{"pattern": pattern, "replacement": replacement}]}
    r.update(kw)
    return r


def _terminal():
    return {"name": "trim-edges", "label": "Trim edges", "type": "terminal"}


def _map_rule(name, mapping=None, map_meta=None, **kw):
    r = {"name": name, "label": name, "type": "callback:map",
         "map": mapping if mapping is not None else {"foo": "=>"}}
    if map_meta is not None:
        r["map_meta"] = map_meta
    r.update(kw)
    return r


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)          # save_factory_rules creates it
    return path


def test_load_real_config_json():
    """The committed config.json loads, validates, and ends with a terminal."""
    rules = cs.load_factory_rules()
    assert len(rules) >= 2
    assert rules[-1]["type"] == "terminal"
    assert sum(1 for r in rules if r["type"] == "terminal") == 1


def test_save_load_roundtrip():
    """save_factory_rules → load_factory_rules is a stable round-trip."""
    path = _tmp_path()
    try:
        rules = [_regex_rule("alpha"), _regex_rule("beta"), _terminal()]
        saved = cs.save_factory_rules(rules, path=path)
        loaded = cs.load_factory_rules(path=path)
        assert loaded == saved
        assert [r["name"] for r in loaded] == ["alpha", "beta", "trim-edges"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_load_preserves_given_order():
    """Rule order is persisted EXACTLY as given — not sorted or canonicalised.

    This is the backend invariant the WebUI's "Promote order" action relies on:
    the array we POST is the array config.json keeps.
    """
    path = _tmp_path()
    try:
        rules = [_regex_rule("gamma"), _regex_rule("alpha"),
                 _regex_rule("beta"), _terminal()]
        saved = cs.save_factory_rules(rules, path=path)
        loaded = cs.load_factory_rules(path=path)
        expected = ["gamma", "alpha", "beta", "trim-edges"]
        assert [r["name"] for r in saved] == expected
        assert [r["name"] for r in loaded] == expected
        # sort_keys=False must leave the PIPELINE_RULES array order untouched.
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        assert [r["name"] for r in raw["PIPELINE_RULES"]] == expected
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_wrapped_object_shape():
    """The on-disk file is {schema_version, PIPELINE_RULES}, not a bare array."""
    path = _tmp_path()
    try:
        cs.save_factory_rules([_regex_rule("only"), _terminal()], path=path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["schema_version"] == 1
        assert isinstance(raw["PIPELINE_RULES"], list)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_note_round_trips():
    """The `note` field survives a save/load cycle."""
    path = _tmp_path()
    try:
        why = "explains why this rule exists"
        cs.save_factory_rules([_regex_rule("noted", note=why), _terminal()], path=path)
        loaded = cs.load_factory_rules(path=path)
        assert loaded[0]["note"] == why
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_normalizes_seeded():
    """save_factory_rules forces seeded=True on every written rule — a rule in
    the committed factory file is a factory default by definition."""
    path = _tmp_path()
    try:
        rules = [_regex_rule("a", seeded=False), _regex_rule("b"), _terminal()]
        saved = cs.save_factory_rules(rules, path=path)
        assert all(r["seeded"] is True for r in saved), saved
        loaded = cs.load_factory_rules(path=path)
        assert all(r["seeded"] is True for r in loaded), loaded
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_map_meta_round_trips():
    """The server-owned `map_meta` timestamps survive a save/load cycle."""
    path = _tmp_path()
    try:
        meta = {"foo": 1700000000, "bar": 1700000123}
        rule = _map_rule("words", mapping={"foo": "=>", "bar": "->"}, map_meta=meta)
        cs.save_factory_rules([rule, _terminal()], path=path)
        loaded = cs.load_factory_rules(path=path)
        assert loaded[0]["map_meta"] == meta, loaded[0]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_map_meta_prunes_unknown_keys():
    """A map_meta key with no matching `map` entry is dropped on validation."""
    rule = cs.MapRule(**_map_rule(
        "words", mapping={"foo": "=>"}, map_meta={"foo": 5, "ghost": 9}))
    assert rule.map_meta == {"foo": 5}, rule.map_meta


def test_map_meta_defaults_empty():
    """A cb:map rule with no map_meta validates and defaults to {}."""
    rule = cs.MapRule(**_map_rule("words", mapping={"foo": "=>"}))
    assert rule.map_meta == {}


def test_bad_regex_rejected():
    """An uncompilable pattern is rejected before anything is written."""
    path = _tmp_path()
    try:
        bad = [_regex_rule("broken", pattern="("), _terminal()]
        try:
            cs.save_factory_rules(bad, path=path)
            assert False, "expected ValidationError for an invalid regex"
        except ValidationError:
            pass
        assert not os.path.exists(path), "nothing should be written on a bad save"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_terminal_must_be_last():
    """The terminal rule must be the final entry."""
    path = _tmp_path()
    try:
        try:
            cs.save_factory_rules([_terminal(), _regex_rule("after")], path=path)
            assert False, "expected ValidationError for a non-last terminal"
        except ValidationError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_duplicate_slug_rejected():
    """Two rules with the same name are rejected."""
    path = _tmp_path()
    try:
        try:
            cs.save_factory_rules(
                [_regex_rule("dup"), _regex_rule("dup"), _terminal()], path=path)
            assert False, "expected ValidationError for a duplicate slug"
        except ValidationError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_missing_file_raises():
    """load_factory_rules fails fast on a missing file (config.json is required)."""
    try:
        cs.load_factory_rules(path="/nonexistent/does-not-exist.json")
        assert False, "expected RuntimeError for a missing file"
    except RuntimeError as e:
        assert "git checkout config.json" in str(e)


def test_corrupt_json_raises():
    """load_factory_rules fails fast on malformed JSON."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, b"{ not valid json")
        os.close(fd)
        try:
            cs.load_factory_rules(path=path)
            assert False, "expected RuntimeError for malformed JSON"
        except RuntimeError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_missing_pipeline_rules_key_raises():
    """A JSON object without a PIPELINE_RULES key is rejected."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, b'{"schema_version": 1}')
        os.close(fd)
        try:
            cs.load_factory_rules(path=path)
            assert False, "expected RuntimeError for a missing PIPELINE_RULES key"
        except RuntimeError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_neuenzeile_fires_at_end_of_utterance():
    """The committed dictation map's newline keys fire in the common
    dictation positions (end of utterance, before punctuation).

    Regression: config.json once shipped "Neuenzeile " with a trailing
    space; compiled word-bounded, the escaped space + closing \\b demanded
    a following word character, so the key never fired at end of utterance.
    Compiles the real factory map with core.dictation_map.compile_map — the
    one compile main.rebuild_caches and the /settings/pipeline dry run use
    (a pure module, so this file stays pydantic-only).
    """
    sub = _factory_map_sub()
    assert "\n" in sub("Text Neuenzeile")
    assert "\n" in sub("Text Neuenzeile.")


def _factory_map():
    for rule in cs.load_factory_rules():
        if rule.get("type") == "callback:map" and "Neuenzeile" in rule.get("map", {}):
            return rule["map"]
    assert False, "no factory callback:map rule with a 'Neuenzeile' key"


def _factory_map_sub(m=None):
    from faster_whisper_backend.core.dictation_map import compile_map
    cre, replacer, _lookup = compile_map(m if m is not None else _factory_map())
    return lambda text: cre.sub(replacer, text)


def _factory_entry(label):
    for rule in cs.load_factory_rules():
        for e in rule.get("entries") or []:
            if e.get("label") == label:
                return e
    assert False, f"no factory entry {label!r}"


def test_quote_spacing_pairs_quotes_left_to_right():
    """tighten-quote-spacing pairs quotes in order: an opening quote drops the
    spaces after it, a closing one the spaces before it — also across a line
    break — and an unclosed quote at the end already counts as opening. The
    old pattern paired a quote left open at the end of one utterance
    differently once the next one closed it, so live dictation typed the
    quotation twice."""
    import re
    e = _factory_entry("tighten-quote-spacing")
    cre, rep = re.compile(e["pattern"]), e["replacement"]
    sub = lambda t: cre.sub(rep, t)
    assert sub('Sie sagt " mir ist schwindlig " Punkt') == 'Sie sagt "mir ist schwindlig" Punkt'
    # an open quote at the end is formatted as opening — the same as once closed
    assert sub('Sie sagt " mir ist') == 'Sie sagt "mir ist'
    assert sub('Sie sagt " mir ist schwindlig " .').startswith(sub('Sie sagt " mir ist'))
    # a quotation spanning a line break
    assert sub('" eins\nzwei "') == '"eins\nzwei"'
    # two quotations: each pair on its own
    assert sub('a " b " c " d "') == 'a "b" c "d"'


def test_eszett_keys_also_match_their_ss_spelling():
    """The Swiss cleanup turns ß into ss before the map runs, so a ß key must
    also match its ss spelling; an explicit ss key wins over the derived one."""
    sub = _factory_map_sub()
    assert sub("Der Wert ist grösser als 5") == "Der Wert ist > 5"
    assert sub("Der Wert ist größer als 5") == "Der Wert ist > 5"
    custom = _factory_map_sub({"Fußnote": "[1]", "Fussnote": "(fn)"})
    assert custom("eine Fussnote") == "eine (fn)"
    assert custom("eine Fußnote") == "eine [1]"
    assert _factory_map_sub({"Straße": "Str."})("Hauptstrasse Strasse") == "Hauptstrasse Str."


def test_dictated_punctuation_wins_over_whisper_punctuation():
    """Whisper's own punctuation directly before a dictated punctuation word
    is replaced together with it; a mark between two digits is untouched."""
    sub = _factory_map_sub()
    assert sub(" HB 12... Komma 5") == " HB 12, 5"
    assert sub(" HB 12. Punkt") == " HB 12."
    assert sub(" Dosis 1, Komma, 5 mg") == " Dosis 1,, 5 mg"
    assert sub("12.5 und 12,5") == "12.5 und 12,5"
    # a non-punctuation key keeps the mark before it
    assert sub("Ende. Gradzeichen") == "Ende. °"
    # a longer key starting with a punctuation word keeps Whisper's mark
    custom = _factory_map_sub({"Komma": ",", "Komma Strich": "x"})
    assert custom("a. Komma Strich") == "a. x"
    assert custom("a. Komma") == "a,"


if __name__ == "__main__":
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
