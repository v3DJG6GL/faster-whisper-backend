"""Tests for regex_guard's probe surface.

Focus: the two holes that let a MANUFACTURED-input attack through.

  1. `_nested_repetition` only refuses a repeat INSIDE a repeated group, so
     eight SIBLING quantified groups — `(-+)(-+)(-+)…#` — are structurally
     invisible while still being polynomially explosive.
  2. Every timed probe used a fixed alphabet. `FIXTURE` and all four
     `_ADVERSARIAL` inputs contain no `-` at all, so that pattern matched
     NOTHING in any probe and returned in microseconds.

Together they accepted the pair `("(Wetter)", "-"*60)` + `("(-+)…#", "")` in
0.05 s, after which every transcription pinned a CPU core uninterruptibly:
the first entry manufactures a 60-dash run and the second one explodes on it.

The counterpart tests here (ordinary rules, the shipped config.json) exist
because the fix must be purely ADDITIVE — it may not reject anything the guard
accepted before.
"""

import pytest

import regex_guard as g


# The exact reproduction: entry 0 manufactures the run, entry 1 detonates on it.
_MANUFACTURED = [
    ("e0", r"(Wetter)", "-" * 60),
    ("e1", r"(-+)(-+)(-+)(-+)(-+)(-+)(-+)(-+)#", ""),
]


def test_fixtures_still_lack_the_attack_character():
    """Pins the premise: no fixed probe input contains a `-`, so neither the
    fixture nor the adversarial inputs can ever exercise the attack pattern.
    If this ever fails, the two probes below are passing for the wrong reason.
    """
    assert "-" not in g.FIXTURE
    assert not any("-" in adv for adv in g._ADVERSARIAL)


def test_prefix_ambiguous_alternations_are_structurally_visible():
    """The prefix-ambiguous repeated-alternation family — one run of input
    that splits many ways, no branch repeated verbatim — must trip
    _nested_repetition. (Moved here from test_config_store.py: the private
    helper's unit coverage belongs next to the module.)"""
    for pat in ("(a|ab)+", "(x|xx)+y", "(ab|a|b)+c", "(n|d|nd)+#", "(|a)+"):
        assert g._nested_repetition(pat), pat


def test_sibling_quantified_groups_are_structurally_invisible():
    """The attack pattern is NOT nested repetition — that screen cannot see it,
    which is why the timed probes have to."""
    assert not g._nested_repetition(_MANUFACTURED[1][1])


def test_manufactured_input_chain_is_rejected():
    """End to end: the pair that used to be accepted in 0.05 s now fails the
    save. The verdict is the parent's timeout kill, so no CPU core is left
    pinned by the check itself."""
    with pytest.raises(ValueError) as ei:
        g.validate(_MANUFACTURED, timeout=1.5)
    assert "e1" in str(ei.value)


def test_chain_carries_an_earlier_entrys_output_downstream():
    """Half one of the fix, without the timing: entry 0's replacement really
    does reach the running fixture that entry 1 is probed against — including
    when entry 0's pattern matches nothing in the static fixture (the guard
    seeds a witness so the entry fires at all)."""
    import re
    pattern, replacement = _MANUFACTURED[0][1], _MANUFACTURED[0][2]
    assert not re.search(pattern, g.FIXTURE), "premise: fixture never says Wetter"
    chained = g._chain_advance(re.compile(pattern), pattern, replacement, g.FIXTURE)
    assert "-" * 60 in chained


def test_chain_is_length_capped():
    """The chaining must not become the blowup: an aggressively expanding
    entry is truncated instead of compounding across entries."""
    import re
    chained = g.FIXTURE
    for _ in range(6):
        chained = g._chain_advance(re.compile(r"\."), r"\.", "." * 40, chained)
    assert len(chained) <= g._CHAIN_CAP


def test_chain_never_raises_on_a_broken_entry():
    """Chaining is extra signal only — a bad replacement must not turn into a
    new failure mode here (the in-process template check already reports it)."""
    import re
    assert g._chain_advance(re.compile("(a)"), "(a)", r"\9", g.FIXTURE) == ""


def test_synthetic_probe_targets_the_patterns_own_alphabet():
    """Half two of the fix: a run of the character the first unbounded atom
    matches, terminated by something the pattern cannot match so the engine is
    forced to exhaust the split search."""
    synth = g._synthetic_fixture(_MANUFACTURED[1][1])
    assert synth is not None
    assert synth.startswith("-" * g._SYNTH_RUN)
    assert synth[-1] not in "-#"
    # A shorthand class resolves too; a pattern with no repetition does not.
    assert g._synthetic_fixture(r"(\d+),(\d+)").startswith("1" * 10)
    assert g._synthetic_fixture(r"\bfoo\b") is None


@pytest.mark.parametrize("pat,expected", [
    (r"(Wetter)", "Wetter"),
    (r"(\d+),(\d+)", "111,111"),
    (r"z\.B\.", "z.B."),
    (r"(Herr|Frau) ", "Herr "),
    (r"\bfoo\b", "foo"),
])
def test_witness_builds_a_matching_string(pat, expected):
    """The witness must actually match its pattern, or seeding the chain does
    nothing."""
    import re
    assert g._witness(pat) == expected
    assert re.search(pat, g._witness(pat)), pat


def test_ordinary_rules_are_still_accepted():
    """No rejection may be ADDED for the rules this feature exists to serve."""
    g.validate([
        ("plain", r"\bfoo\b", "bar"),
        ("decimal", r"(\d+),(\d+)", r"\1.\2"),
        ("anrede", r"(Herr|Frau) ", r"\1 "),
        ("expand", r"z\.B\.", "zum Beispiel"),
    ])


def test_shipped_factory_rules_still_validate():
    """The committed config.json must keep passing the SAVE-path guard — a
    regression here breaks the product on the first admin save."""
    import config_store as cs
    checks = []
    for idx, rule in enumerate(cs.load_factory_rules()):
        for eidx, entry in enumerate(rule.get("entries") or []):
            if entry.get("pattern"):
                checks.append((f"rule {idx} entry {eidx}", entry["pattern"],
                               entry.get("replacement") or ""))
        if rule.get("type") != "regex-list" and rule.get("pattern"):
            checks.append((f"rule {idx}", rule["pattern"], ""))
    assert checks, "config.json should ship at least one regex rule"
    g.validate(checks)


def test_shipped_factory_rules_validate_through_the_save_path():
    """Same, through the real AdminConfig save validator (guard_regex context),
    which is what the admin UI and /v1/pipeline-rules actually call."""
    import config_store as cs
    cs.AdminConfig.model_validate(
        {"PIPELINE_RULES": cs.load_factory_rules()},
        context={"guard_regex": True})


@pytest.mark.parametrize("pat", [
    r"(\d{1,3}(?:\.\d{3})+)",  # thousands separator
    r"(\d{4})+",
    r"(?:\.\d{3})+",
])
def test_fixed_count_repetition_is_not_screened(pat):
    """A comma-less `{n}` matches exactly one way, so nesting it inside a
    repeated group cannot backtrack ambiguously — ordinary formatting rules
    like a thousands separator must pass the structural screen."""
    assert not g._nested_repetition(pat)


@pytest.mark.parametrize("pat", [
    r"(x{2,})+",
    r"(\w+ ?)+",
])
def test_variable_nested_repetition_stays_rejected(pat):
    assert g._nested_repetition(pat)


def test_short_literal_expansions_are_accepted():
    """A bounded literal expansion of a short token is a normal dictation
    rule; the analytic growth bound must not refuse it."""
    g.validate([
        ("deg", "°", "Grad Celsius"),
        ("it", r"\bIT\b", "Informationstechnologie"),
    ])


@pytest.mark.parametrize("pattern,repl", [
    ("n", "n" * 512),
    ("(n+)", "\\1" * 256),
])
def test_large_unmeasurable_growth_stays_rejected(pattern, repl):
    with pytest.raises(ValueError):
        g.validate([("amp", pattern, repl)])


def test_negated_shorthand_class_gets_a_real_witness():
    """`[^\\w]` used to get '1' — a character it can never match — so the
    synthetic probe silently no-op'd on any negated shorthand class."""
    import re
    cand = g._class_char(r"\w", True)
    assert cand is not None and re.match(r"[^\w]", cand)
    synth = g._synthetic_fixture(r"[^\w]+#")
    assert synth is not None and re.match(r"[^\w]", synth)


def test_scaling_ratio_alone_cannot_reject_a_fast_pattern(monkeypatch):
    """The scaling verdict needs BOTH a confirmed ratio and _SCALE_MIN_REJECT
    of real CPU on the scaled fixture. Under CPU contention (the guard shares
    the box with live transcription) a de-schedule mid-probe faked 20-36x
    ratios on microsecond-scale factory rules and 422'd valid saves. With the
    allowance forced to zero, every pattern trips the ratio — the absolute
    floor must still wave a fast pattern through."""
    monkeypatch.setattr(g, "_SCALE_ALLOWANCE", 0)
    assert g._probe([["Komma", ","]]) is None
    # A representative slice of shipped-style rules, all microsecond-scale.
    assert g._probe([[r"(\d+),(\d+)", r"\1.\2"], [r"z\.B\.", "zum Beispiel"]]) is None
