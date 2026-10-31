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

The counterpart tests here (ordinary rules, the shipped config.json) pin
what the guard must keep ACCEPTING: every later tightening (the analytic
growth bound, the reference cap, the shape screens below) refuses a named
attack shape, never the rules this feature exists to serve.
"""

import pytest

from faster_whisper_backend.pipeline import regex_guard as g


# The exact reproduction: entry 0 manufactures the run, entry 1 detonates on it.
_MANUFACTURED = [
    ("e0", r"(Wetter)", "-" * 60),
    ("e1", r"(-+)(-+)(-+)(-+)(-+)(-+)(-+)(-+)#", ""),
]
# The same pair with an e1 whose OWN synthetic run misses (it is built from
# the leading `x*`), so e1 passes alone and the pair can only be refused
# through the chained fixture. _MANUFACTURED's e1 is refused on its own
# synthetic run, which proves nothing about the chain.
_CHAIN_ONLY = [
    _MANUFACTURED[0],
    ("e1", r"x*(-+)(-+)(-+)(-+)(-+)(-+)(-+)(-+)#", ""),
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


def test_chain_only_pair_is_rejected_through_the_chain():
    """The chain-only e1 is harmless alone (no nesting, its synthetic run is
    x's), so refusing the pair proves entry 0's output reached its probe."""
    assert not g._nested_repetition(_CHAIN_ONLY[1][1])
    g.validate([_CHAIN_ONLY[1]], timeout=1.5)
    with pytest.raises(ValueError) as ei:
        g.validate(_CHAIN_ONLY, timeout=1.5)
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
    synths = g._synthetic_fixtures(_MANUFACTURED[1][1])
    assert all(s.startswith("-" * g._SYNTH_RUN) for s in synths)
    assert "-" * g._SYNTH_RUN + "!" in synths
    # A shorthand class resolves too; a pattern with no repetition does not.
    assert "1" * g._SYNTH_RUN + "#" in g._synthetic_fixtures(r"(\d+),(\d+)")
    assert g._synthetic_fixtures(r"\bfoo\b") == []


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
    from faster_whisper_backend.settings import config_store as cs
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
    from faster_whisper_backend.settings import config_store as cs
    from faster_whisper_backend.settings import schema as settings_schema
    settings_schema.AdminConfig.model_validate(
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


def test_named_backref_is_not_a_group_frame():
    """`(?P=name)` and a leading `(?i)` close themselves; they used to push
    a phantom frame that mis-popped the enclosing group (false 422 on
    `(?P<n>a)((?P=n)|b)+`, and the exponential `(\\w+ ?(?P=n))+#` shape
    slipped past the structural screen)."""
    assert not g._nested_repetition(r"(?P<n>a)((?P=n)|b)+")
    assert not g._nested_repetition(r"(?i)(a|b)+c")
    assert g._nested_repetition(r"(?P<n>x)(a+(?P=n))+")
    assert g._nested_repetition(r"(?P<n>)(\w+ ?(?P=n))+#")
    assert g._nested_repetition(r"(?i)(a+)+")
    assert g._next_atom(r"(?P<n>x)(?P=n)(y)", 8) == (None, None, 14)
    assert g._first_repeated_char(r"(?P<x>a)(?P=x)b+") == "b"


def test_conditional_group_is_a_real_frame():
    """`(?(1)yes|no)` used to be read as self-closing like `(?P=name)`: its
    own `)` popped the ENCLOSING group, so the exponential `(\\w+\\s?)+#`
    shape hidden behind a condition passed the screen (and every timed probe,
    none of which contains the `x` that arms group 1). Its yes/no `|` is not
    an overlapping alternation."""
    assert g._nested_repetition(r"(x)?((?(1)\w+\s?|b))+#")
    assert not g._nested_repetition(r"(x)?((?(1)a|b))+")
    assert not g._nested_repetition(r"(?P<q>x)?((?(q)a|a))+c")
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", r"(x)?((?(1)\w+\s?|b))+#", "")])


def test_verbose_mode_is_scanned_without_its_whitespace_and_comments():
    """Under `(?x)` whitespace is ignored and `#` starts a comment, so the
    `+` in `(?x)(a+) +#` repeats the group — the screen used to count it as
    a root-level repeat and accept the pattern."""
    assert g._nested_repetition(r"(?x)(a+) +#")
    assert g._nested_repetition(r"(?x)(a|a) +#")
    assert g._nested_repetition(r"(?x)z*([nd]+\ ?) +\#")
    assert g._nested_repetition(r"(?x:(a+) +)b")
    # whitespace and `#` stay literal in a class, after a backslash and
    # outside verbose mode; a `)` inside a comment closes nothing
    assert not g._nested_repetition(r"(?x)[ #]+ \d")
    assert not g._nested_repetition(r"(?x:a b) (c+) +")
    assert not g._nested_repetition("(?x)(a+) # ) +\n b")
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", r"(?x)z*([nd]+\ ?) +\#", "")])


def test_verbose_mode_whitespace_does_not_steer_the_synthetic_probe():
    """The synthetic probe and the chain seed must read a `(?x)` pattern the
    way the engine does: `- * - * …` is `-*-*…`, whose run is dashes. Built
    from the raw spacing, the probe was 240 spaces the pattern never touches
    and validate() accepted what the unspaced spelling is refused for."""
    pat = r"(?x)- * - * - * - * - * [#]"
    assert "-" * g._SYNTH_RUN + "!" in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("t", pat, "x")])


@pytest.mark.parametrize("pat", [
    r"(?i)\b(?:äh|Äh)+\b",
    r"(?i)(?:ja |JA )+x",
    r"(?:(?i:x)|y)(?:ab|AB)+#",
])
def test_case_variant_branches_overlap_under_ignorecase(pat):
    """Under `(?i)` branches that differ only in case match the same input, so
    a repeated group of them is `(a|a)+`: the overlap screen compared them
    case-sensitively and validate() accepted an exponential pattern. A scoped
    `(?i:...)` anywhere counts too (over-approximation)."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("e", pat, "")])


def test_case_variant_branches_stay_distinct_without_ignorecase():
    """Without IGNORECASE `äh` and `Äh` are different inputs, and an escape's
    case is never folded: `\\d` and `\\D` are disjoint classes."""
    assert not g._nested_repetition(r"(?:äh|Äh)+\b")
    assert not g._nested_repetition(r"(?i)(?:\d|\D)+#")


def test_overlap_through_a_later_nested_alternative_is_seen():
    """A branch holding a nested alternation is compared through one witness
    per alternative: `(?:a|b)c` overlaps `bc` through its SECOND alternative,
    which the first-alternative witness ("ac") missed — the class spelling
    `(?:[ab]c|bc)+#` was refused all along."""
    assert g._witnesses("(?:a|b)c") == ["ac", "bc"]
    assert g._nested_repetition(r"(?:(?:a|b)c|bc)+#")
    assert g._nested_repetition(r"(?:bc|(?:a|b)c)+#")
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("e", r"(?:(?:a|b)c|bc)+#", "")])


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


def test_growth_through_a_branch_the_fixture_does_not_match_is_rejected():
    """`^Hallo|n` matches the fixture once (a tiny measured ratio), so the
    analytic bound used to be skipped — yet every "n" of a real transcript
    then became 512 characters, and three chained entries turned "Guten
    Morgen" into 268 M characters."""
    with pytest.raises(ValueError):
        g.validate([("e", "^Hallo|n", "x" * 512)])


def test_reference_growth_under_the_absolute_floor_is_rejected():
    """64 references to an unbounded capture total 64 "characters" — not past
    the literal floor — but copy the capture 64x per match and compound
    across entries. References are held to the ratio alone."""
    with pytest.raises(ValueError):
        g.validate([("e", "(n+)", "\\1" * 64)])
    g.validate([
        ("deg", "°", "Grad Celsius"),
        ("it", r"\bIT\b", "Informationstechnologie"),
        ("decimal", r"(\d+),(\d+)", r"\1.\2"),
    ])


def test_branches_overlapping_through_a_class_are_rejected():
    """No branch text is a prefix of another, but `[bc]x` matches `cx`, so a
    run of "cx" splits both ways and the match time doubles per unit."""
    for pat in (r"(cx|[bc]x)+#", r"([bc]x|cx)+#", r"(\wx|cx)+#"):
        assert g._nested_repetition(pat), pat
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("p", r"(cx|[bc]x)+#", "")])
    for pat in (r"(Herr|Frau)+ ", r"(?:\.|,)+", r"(a|b)+c", r"(\d|x)+"):
        assert not g._nested_repetition(pat), pat


def test_negated_shorthand_class_gets_a_real_witness():
    """`[^\\w]` used to get '1' — a character it can never match — so the
    synthetic probe silently no-op'd on any negated shorthand class."""
    import re
    cand = g._class_char(r"\w", True)
    assert cand is not None and re.match(r"[^\w]", cand)
    synths = g._synthetic_fixtures(r"[^\w]+#")
    assert synths and all(re.match(r"[^\w]", s) for s in synths)


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


def test_chain_seed_survives_a_saturated_chain():
    """`_chain_advance` appends the witness to the TAIL but `[:_CHAIN_CAP]`
    keeps the HEAD, so once enough non-matching entries had saturated the
    chain the seed — and the manufactured run it produces — fell off the end
    and the later `(-+)…#` entry never saw it. The seed must stay inside the
    cap."""
    import re
    chained = g.FIXTURE
    for i in range(400):
        pat = r"\bwort%d\b" % i
        chained = g._chain_advance(re.compile(pat), pat, "X", chained)
    # Saturated: the chain sits at the cap (minus the last entry's shrink).
    assert len(chained) > g._CHAIN_CAP - 64, "premise: the chain is saturated"
    pattern, replacement = _MANUFACTURED[0][1], _MANUFACTURED[0][2]
    out = g._chain_advance(re.compile(pattern), pattern, replacement, chained)
    assert len(out) <= g._CHAIN_CAP
    assert "-" * 60 in out


def test_manufactured_pair_is_still_rejected_deep_in_a_large_list():
    """End to end: placing the pair after enough harmless entries to saturate
    the chain must not evade the chained probe."""
    checks = [(f"r{i}", r"\bwort%d\b" % i, "X") for i in range(400)]
    checks += _CHAIN_ONLY
    with pytest.raises(ValueError) as ei:
        g.validate(checks, timeout=1.5)
    assert "e1" in str(ei.value)


def test_non_ascii_brace_run_is_a_literal_not_a_crash():
    """`re.compile('a{²}')` succeeds (a non-ASCII-digit brace run is a
    literal to CPython), but `str.isdigit()` is True for `²` and `int()` then
    raised a raw ValueError out of the structural screen naming no rule."""
    assert g._read_quantifier("a{²}", 1) == (False, False, 1, False)
    assert g._read_quantifier("a{١,²}", 1) == (False, False, 1, False)
    g.validate([("rule 0", "a{²}", "b")])
    # Ordinary ASCII quantifiers are unaffected.
    assert g._read_quantifier("a{2}", 1)[0] is True
    assert g._read_quantifier("a{2,}", 1) == (True, False, 5, True)


def test_budget_grows_with_the_list_and_a_large_rule_set_passes(monkeypatch, caplog):
    """A flat 2 s budget 422'd a large legitimate rule set as 'catastrophic
    backtracking' (~0.44 ms/entry measured, 40k entries permitted). The
    budget is now 2 s plus a per-entry allowance, capped. The budget the
    helper is actually run with is captured, and the acceptance must come
    from a helper that ran — validate fails OPEN (with a warning) on a crash.
    The helper gets what the in-process screen left of the budget, hence the
    one-second slack below each budget."""
    import logging
    import subprocess
    real_run = subprocess.run
    seen = []

    def _recording_run(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _recording_run)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        g.validate([(f"r{i}", r"\bwort%d\b" % i, "X") for i in range(3000)])
    budget = min(g._GUARD_TIMEOUT + g._PER_CHECK_BUDGET * 3000, g._GUARD_TIMEOUT_MAX)
    assert len(seen) == 1 and budget - 1 < seen[0] <= budget
    assert seen[0] > g._GUARD_TIMEOUT
    assert "regex guard skipped" not in caplog.text

    # Below the cap the budget is the per-entry formula; far past it, the cap.
    def _ok_run(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(subprocess, "run", _ok_run)
    seen.clear()
    g.validate([(f"r{i}", "x", "") for i in range(400)])
    g.validate([(f"r{i}", "x", "") for i in range(20000)])
    budget = g._GUARD_TIMEOUT + g._PER_CHECK_BUDGET * 400
    assert budget - 1 < seen[0] <= budget
    assert g._GUARD_TIMEOUT < seen[0] < g._GUARD_TIMEOUT_MAX
    assert g._GUARD_TIMEOUT_MAX - 1 < seen[1] <= g._GUARD_TIMEOUT_MAX


def test_an_entry_with_a_synthetic_run_gets_its_own_allowance(monkeypatch):
    """`(\\d+)\\s*(\\d*)\\s*Euro` spends ~0.1 s on its 240-digit synthetic
    run, so 20 such amount rules overran 2 s + 20 x 5 ms and were refused as
    catastrophic backtracking. Each entry the child gives a synthetic run now
    adds _SYNTH_BUDGET (capped); a literal entry, which gets none, does not."""
    import subprocess
    seen = []

    def _ok_run(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(subprocess, "run", _ok_run)
    units = ["Euro", "Dollar", "Franken", "Prozent", "Grad"] * 4
    g.validate([(u, r"(\d+)\s*(\d*)\s*" + u, r"\1,\2 " + u) for u in units])
    g.validate([(u, r"\b" + u + r"\b", u) for u in units])
    g.validate([(f"r{i}", r"(\d+)\s*Euro%d" % i, "") for i in range(100)])
    budget = g._GUARD_TIMEOUT + (g._PER_CHECK_BUDGET + g._SYNTH_BUDGET) * 20
    assert budget < g._GUARD_TIMEOUT_MAX
    assert budget - 1 < seen[0] <= budget
    budget = g._GUARD_TIMEOUT + g._PER_CHECK_BUDGET * 20
    assert budget - 1 < seen[1] <= budget
    assert g._GUARD_TIMEOUT_MAX - 1 < seen[2] <= g._GUARD_TIMEOUT_MAX


def test_a_slow_entry_cannot_borrow_the_allowance_of_cheap_ones(monkeypatch):
    """The allowances are pooled into ONE child timeout, so a cubic amount
    rule refused on its own was accepted behind 31 `Wort{i}\\s+` entries
    (~8 s). The child caps each entry's own CPU time at _GUARD_TIMEOUT, here
    lowered so a quadratic rule (~0.1 s) stands in for the cubic one."""
    monkeypatch.setattr(g, "_GUARD_TIMEOUT", 0.03)
    cheap = [[rf"Wort{i}\s+", ""] for i in range(31)]
    assert g._probe(cheap) is None
    idx, msg = g._probe(cheap + [[r"(\d+)\s*(\d*)\s*Euro", r"\1 EUR"]])
    assert idx == 31
    assert msg.startswith("this rule took ")
    assert "more than any single rule may (0.03 s)" in msg
    assert "test inputs" in msg and "catastrophic backtracking" in msg


def test_helper_is_not_charged_the_callers_compiles_twice(monkeypatch):
    """The child compiles every pattern again, so the seconds the caller
    already spent compiling are NOT taken off its timeout as well: that
    refused every save whose compiles cost over about half the budget. It gets
    the budget minus the in-process screen; nothing left of the budget still
    refuses the save before the helper starts."""
    import subprocess
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: seen.append(k["timeout"])
                        or subprocess.CompletedProcess(a, 0, '{"ok": true}', ""))
    g.validate([("r", "x", "")], timeout=1.0, spent=0.75)
    assert len(seen) == 1 and 0.9 < seen[0] <= 1.0
    with pytest.raises(ValueError, match="r: checking the submitted patterns took"):
        g.validate([("r", "x", "")], timeout=1.0, spent=1.0)
    assert len(seen) == 1


def test_whole_guard_hold_stays_under_the_save_lock_timeout(monkeypatch):
    """Caller's compiles + screen + child run under atomic_json.save_lock,
    whose waiters give up after SAVE_LOCK_TIMEOUT_S; two full budgets at the
    cap would pass it, so the child is trimmed to what _HOLD_MAX leaves."""
    import subprocess
    from faster_whisper_backend.core import atomic_json
    assert g._GUARD_TIMEOUT_MAX <= g._HOLD_MAX < atomic_json.SAVE_LOCK_TIMEOUT_S
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: seen.append(k["timeout"])
                        or subprocess.CompletedProcess(a, 0, '{"ok": true}', ""))
    monkeypatch.setattr(g, "_HOLD_MAX", 1.5)
    g.validate([("r", "x", "")], timeout=1.0, spent=0.75)
    assert len(seen) == 1 and 0.5 < seen[0] <= 0.75


def test_helper_timeout_after_slow_compiles_does_not_blame_backtracking(monkeypatch):
    """When the caller's compiles already took over half the child's time,
    the child (compiling them again) timing out is reported as slow
    compiling; only a timeout the compiles cannot explain blames
    backtracking, quoting the time the child really had."""
    import subprocess

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(a[0], k["timeout"], stderr=b"0\n")

    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(ValueError) as ei:
        g.validate([("r", "x", "")], timeout=1.0, spent=0.6)
    msg = str(ei.value)
    assert msg.startswith("r: checking the submitted patterns took > 1 s")
    assert "backtracking" not in msg
    with pytest.raises(ValueError, match=r"r: regex took > 1 s .* catastrophic backtracking"):
        g.validate([("r", "x", "")], timeout=1.0, spent=0.0)


def test_timeout_verdict_names_the_real_probe_inputs(monkeypatch):
    """A genuinely catastrophic entry still trips the (lowered) budget, and
    the message no longer claims the pattern was slow on 'a 1 KB fixture'
    when the child also runs scaled, adversarial, synthetic and chained
    inputs."""
    monkeypatch.setattr(g, "_GUARD_TIMEOUT", 0.5)
    with pytest.raises(ValueError) as ei:
        g.validate(_MANUFACTURED)
    msg = str(ei.value)
    assert "e1" in msg
    assert "1 KB fixture" not in msg
    assert "test inputs" in msg and "catastrophic backtracking" in msg


@pytest.mark.parametrize("pat", [
    r"[]](a+)+",      # a `]` first in the class is a literal
    r"[^]\\](a+)+",   # negated, leading literal `]`, escaped backslash
    r"[a\]b](a+)+",   # escaped `]` inside the class
])
def test_class_skipper_agrees_on_the_closing_bracket(pat):
    """The class scan is shared by the structural screen and the synthetic
    probe (`_skip_class`); the leading-`]` and escape paths must still find
    the real closing bracket so the nested repeat after it is seen."""
    assert g._nested_repetition(pat)
    # The same scan drives _next_atom: the class is one atom ending at `]`.
    _char, body, end = g._next_atom(pat, 0)
    assert body is None and pat[end - 1] == "]" and pat[end] == "("


@pytest.mark.parametrize("pat", [
    r"(?:a|\d|(?:13))+#",
    r"(?:a|(?:\d)|13)+#",
    r"(?P<x>a|\d|(?P<y>13))+#",
])
def test_group_wrapped_overlapping_branch_is_rejected(pat):
    """`(?:13)` used to strip to `?:13` — uncompilable, with a `:13` witness —
    so wrapping an overlapping branch in a non-capturing or named group got
    past the overlap check, and `re.sub` on a "1313…" run went exponential."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [r"(?:(a|\d|13))+#", r"((a|ab))+c", r"(?:(?:(a|ab)))+c"])
def test_alternation_wrapped_in_a_whole_body_group_is_rejected(pat):
    """The `|` of `(?:(a|\\d|13))+#` sits on the un-quantified child group, the
    quantified parent has none — so neither frame ran the overlap test."""
    assert g._nested_repetition(pat)


def test_benign_group_wrapped_alternations_still_pass():
    for pat in (r"(?:km|m)\b", r"(?:(km|m))\b", r"(?:(a|b))+c", r"(?:(?>a|ab))+c"):
        assert not g._nested_repetition(pat), pat
    g.validate([("km", r"(?:km|m)\b", "x"), ("km2", r"(?:(km|m))\b", "x")])


@pytest.mark.parametrize("pat", [r"((?:ab){2})+", r"(?:(?:\.\d{3}){2})+"])
def test_fixed_count_group_inside_a_repeat_is_not_screened(pat):
    """A group with a fixed `{n}` matches exactly one way, like `(?:a{2})+`
    — it used to mark its parent as repeating and earn a false 422."""
    assert not g._nested_repetition(pat)
    assert g._nested_repetition(r"((?:ab){2,3})+")


def test_overlap_check_never_runs_a_grouped_branch_in_process():
    """`_branches_overlap` matches one branch against another's witness IN the
    parent process; nested unquantified alternations in that branch backtrack
    exponentially (n=28 took 5 s, n=40 hours). Run in a child with a timeout
    so a regression fails here instead of hanging the suite."""
    import subprocess
    import sys
    from pathlib import Path
    # Loaded by path, like the guard's own child: stdlib-only, no package import.
    code = ("import importlib.util as u, sys; "
            "s = u.spec_from_file_location('rg', sys.argv[1]); g = u.module_from_spec(s); "
            "s.loader.exec_module(g); "
            "g._nested_repetition('(?:' + 'x' * 64 + '|' + '(?:x|xx)' * 40 + '!)+')")
    subprocess.run([sys.executable, "-I", "-c", code, str(Path(g.__file__).resolve())],
                   check=True, timeout=30)


def test_overlap_check_never_runs_an_optional_atom_branch_in_process():
    """`?` / `{0,1}` are not variable quantifiers, so a group-free branch of
    optional atoms never set the frame's "rep" — and `_branches_overlap`
    matched `a?…a?a…a` against an `a…a` witness in-process: the classic
    a?^n a^n blowup (n=26 took 3 s, doubling per +1). Same child + timeout
    as the grouped-branch case, so a regression fails instead of hanging."""
    import subprocess
    import sys
    from pathlib import Path
    code = ("import importlib.util as u, sys; "
            "s = u.spec_from_file_location('rg', sys.argv[1]); g = u.module_from_spec(s); "
            "s.loader.exec_module(g); "
            "g._nested_repetition('(?:' + 'a' * 40 + '|' + 'a?' * 40 + 'a' * 40 + ')+'); "
            "g._nested_repetition('(?:' + 'a' * 40 + '|' + 'a{0,1}' * 40 + 'a' * 40 + ')+')")
    subprocess.run([sys.executable, "-I", "-c", code, str(Path(g.__file__).resolve())],
                   check=True, timeout=30)


@pytest.mark.parametrize("pat", [r"(?:c|(n|nn))+#", r"(?:(n|nn)c?)+#"])
def test_overlapping_alternation_inside_a_repeated_group_is_rejected(pat):
    """The `(n|nn)` split is repeated by the enclosing `+` exactly as in
    `(n|nn)+#`, even when the alternation is not the group's whole body —
    and no probe input holds an `n` run, so the timed probes never saw it."""
    assert g._nested_repetition(pat)


def test_unrepeated_overlapping_alternation_still_passes():
    for pat in (r"(n|nn)#", r"(?:x(n|nn)y)", r"(?:x(n|nn)y)?#"):
        assert not g._nested_repetition(pat), pat


@pytest.mark.parametrize("pat", [r"((?:a|aa){2})+$", r"(?:(?:ha|haha){2})+$", r"((a|aa){3})+"])
def test_fixed_count_overlapping_alternation_inside_a_repeat_is_rejected(pat):
    """`X{2}` is `XX`: a fixed count only matches one way when its body does.
    An overlapping alternation under `{2}` still splits every run many ways,
    so repeating it is exponential (the run-7 `((a|aa){3})+` blowup)."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [r"((?:h)a|haha)+$", r"((?:a)b|[a]b[a]b)+$"])
def test_group_spelled_overlapping_branch_is_rejected(pat):
    """A branch that holds a group but no `|` matches linearly, so the
    in-process overlap match still runs on it — `(?:h)a` overlaps `haha`."""
    assert g._nested_repetition(pat)


@pytest.mark.parametrize("pat", [
    r"(?:\b(?:um|umm)\b[,.] )+",
    r"(?:(?:um|umm), )+",
    r"(?:(?:the|then) )+",
    r"((?:\d|\d\d)\.)+",
    r"(?:(?:Mr|Mrs)\. [A-Z]\w)+",
])
def test_overlap_pinned_by_a_mandatory_disjoint_follower_passes(pat):
    """`(?:(?:um|umm), )+`: the `,` no branch can match ends the group's match
    at one place, and only one branch has that width — one parse per
    repetition, so ordinary filler/abbreviation cleanups are not refused."""
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"((?:a|aa){2})+$",
    r"(?:c|(n|nn))+#",
    r"((?:a|aa)a?)+#",
    r"(?:(?:ax|[ab]x)c)+#",
    r"(?:(?:a|ab)b)+#",
    r"(?i)(?:(?:um|umm)M)+#",
])
def test_overlap_without_a_pinning_follower_is_still_rejected(pat):
    """A fixed count, a `)`, an optional or overlapping follower, same-width
    branches, or a follower a branch matches under (?i) — the split stays
    ambiguous."""
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


def _returns_within(fn, seconds=5.0):
    import threading
    out = {}

    def _run():
        try:
            out["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - asserted on by the caller
            out["error"] = exc
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "hung"
    return out


def test_nested_zero_width_fixed_counts_are_refused_without_matching():
    """`(?:(?:a{0}){N}){N}` loops N*N times on any input, so the in-process
    overlap match on that branch used to pin the request thread. Refused on
    the counts alone, before anything matches the pattern."""
    pat = "(x|(?:(?:a{0}){4000000000}){4000000000}x)+"
    out = _returns_within(lambda: g.validate([("r", pat, "")]))
    assert isinstance(out.get("error"), ValueError)
    assert "nested fixed counts" in str(out["error"])
    out = _returns_within(lambda: g._nested_repetition(pat))
    assert "error" not in out
    with pytest.raises(ValueError, match="nested fixed counts"):
        g.check_fixed_counts("r", "(?:(?:a{0}){100000}){100000}")


def test_fixed_count_product_counts_only_forced_iterations():
    assert g._fixed_count_product(r"(?:(?:a{0}){30}){40}") == 1200
    assert g._fixed_count_product(r"(?:\.\d{3}){2}") == 6
    # Past the minimum an empty iteration ends the loop, so these are cheap.
    assert g._fixed_count_product(r"(?:(?:a{0}){0,100000}){0,100000}") == 1
    assert g._fixed_count_product(r".{0,20000}") == 1
    # Escaped / class / comment braces are not quantifiers.
    assert g._fixed_count_product(r"\{99999\}[{](?#{99999})") == 1
    g.check_fixed_counts("r", r"\d{4}-\d{2}-\d{2}")


def test_leading_zeros_do_not_shrink_a_fixed_count():
    """The engine reads `{00000000003000}` as 3000; truncating the digit
    string to ten characters BEFORE int() read it as 0 and waved the 9e6
    loop — 2 s of GIL-holding `re.sub` on "" — through to the template check."""
    pat = "(?:(?:a{0}){00000000003000}){00000000003000}"
    assert g._fixed_count_product(pat) == 9_000_000
    with pytest.raises(ValueError, match="nested fixed counts"):
        g.check_fixed_counts("r", pat)


@pytest.mark.parametrize("pat", [
    "(?:(?:a{0})(?#x){3000}){3000}",
    "(?:(?:(?:a{0})(?#){3000})(?#){3000}){3000}",
])
def test_quantifier_after_a_comment_still_counts(pat):
    """The engine drops a `(?#...)` comment, so a count after it binds to the
    atom before it; the scanner used to skip the comment and read `{3000}`
    as plain text."""
    with pytest.raises(ValueError, match="nested fixed counts"):
        g.check_fixed_counts("r", pat)


def test_nested_repeat_behind_a_comment_is_seen():
    assert g._nested_repetition(r"(a+)(?#x)+#")
    assert g._nested_repetition(r"(a+)(?#a\)b)+#")  # escaped `)` in a comment
    assert not g._nested_repetition(r"(a+)(?#x)b+#")


def test_numeric_escapes_decode_to_their_character():
    """`\\x2d` is `-`, not `x` followed by the atoms `2` and `d`."""
    assert g._next_atom("\\x2d", 0) == ("-", None, 4)
    assert g._next_atom("\\u2013", 0) == ("\u2013", None, 6)
    assert g._next_atom("\\N{EN DASH}", 0) == ("\u2013", None, 11)
    assert g._next_atom("\\101", 0) == ("A", None, 4)
    assert g._next_atom("\\12", 0) == (None, None, 3)  # group reference
    assert g._class_char("\\x2d", False) == "-"


@pytest.mark.parametrize("pat", [
    "\\x2d*" * 5 + "#",
    "\\u2013*" * 5 + "#",
    "[\\x2d]*" * 5 + "#",
])
def test_escaped_spelling_of_a_polynomial_run_is_rejected(pat):
    """The synthetic probe ran on `d` / `3` (the escape's tail) and matched
    nothing, so these were accepted while `-*-*-*-*-*#` was refused."""
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [
    r"(\W+)(\W+)(\W+)(\W+)(\W+)(\W+)(\W+)#",
    r"\W*\W*\W*\W*\W*\W*\W*#",
])
def test_non_word_run_ending_in_hash_is_rejected(pat):
    """The `\\W` run was built of `#`, the pattern's own terminator, so the
    synthetic probe matched at once and these were accepted while the same
    pattern ending in `x` was refused."""
    assert "\x01" * g._SYNTH_RUN + "!" in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, probe", [
    (r"\d*\d*\d*\d*\d*\d*\d*1", "2"),
    (r"(\d+)(\d+)(\d+)(\d+)(\d+)(\d+)(\d+)1", "2"),
    (r"\s*\s*\s*\s*\s*\s*\s* ", "\t"),
    (r"(\s+)(\s+)(\s+)(\s+)(\s+)(\s+)(\s+) ", "\t"),
])
def test_shorthand_run_ending_in_its_own_representative_is_rejected(pat, probe):
    """The `\\d` / `\\s` run was built of `1` / a space, the pattern's own
    final literal, so the synthetic probe matched at once and these were
    accepted. A literal run keeps its own character."""
    assert probe * g._SYNTH_RUN + "#" in g._synthetic_fixtures(pat)
    assert g._synthetic_fixtures("1*" * 7 + "x")[0].startswith("1" * 10)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [
    r"\d{1,}" * 7 + "1",
    r"\d{1,9}" * 7 + "1",
    r"(\d{1,})" * 7 + "1",
])
def test_counted_shorthand_run_ending_in_its_own_representative_is_rejected(pat):
    """Telling the `\\d` run from a literal one blanked every `1` of the
    text, the count's too: `\\d{1,}` became `\\d{\\x00,}`, no repeat, so the
    run stayed `1` and these were accepted while `\\d*…1` was refused."""
    assert "2" * g._SYNTH_RUN + "#" in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, written", [
    (r"\s*" * 7 + r"\x20", " "),
    (r"\s*" * 7 + r"\040", " "),
    (r"\s*" * 7 + r"\N{SPACE}", " "),
    (r"\s*" * 7 + r"[ \t]", " \t"),
])
def test_shorthand_run_ending_in_an_escaped_representative_is_rejected(pat, written):
    """Whether the rule writes the run character (or the sibling replacing
    it) read the raw text, so an escaped space or tab went unseen: the run
    stayed spaces / became tabs and the escape matched it at once."""
    assert any(not set(s) & set(written) for s in g._synthetic_fixtures(pat))
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [r"\s+\x20", r"(\d+)\x20Uhr"])
def test_ordinary_rule_with_an_escaped_space_is_accepted(pat):
    g.validate([("r", pat, "")], timeout=1.5)


def test_leading_bracket_is_part_of_the_class_body():
    """`[]]` had an empty body, so no witness and no synthetic probe."""
    assert g._skip_class("[]]", 0)[0] == 1
    assert g._next_atom("[]]", 0) == ("]", None, 3)
    with pytest.raises(ValueError):
        g.validate([("r", "[]]*" * 5 + "#", "")], timeout=1.5)


def test_possessive_atom_inside_a_repeated_group_passes():
    """A possessive atom cannot give characters back, like the body of an
    atomic group — `(?:\\w++ ?)+#` is linear and is screened like
    `(?:(?>\\w+) ?)+#`, while the plain `\\w+` form stays refused."""
    assert not g._nested_repetition(r"(?:\w++ ?)+#")
    assert not g._nested_repetition(r"(?:(?>\w+) ?)+#")
    assert g._nested_repetition(r"(?:\w+ ?)+#")
    g.validate([("r", r"(?:\w++ ?)+#", "")])


@pytest.mark.parametrize("pat", [
    r"m(?:n?n)+#",
    r"m(?:nn?)+#",
    r"x*(n?n)+#",
    r"Komma(?: ? )+#",
    r"\b(?:\s?\s)+#",
    r"m(?:[mn]?n)+#",
    r"m(?:n{0,1}n)+#",
])
def test_optional_atom_beside_a_same_character_atom_is_rejected(pat):
    """`(?:n?n)+` is `(?:n{1,2})+` spelled with `?`, which is not a variable
    quantifier — so the screen never saw it, and the synthetic run (built
    from the leading `m` / `x`) never reached the group: `re.sub` on
    "m" + "n"*36 + "!" took over a second, growing ~1.6x per character."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


def test_optional_atom_with_a_disjoint_neighbour_passes():
    for pat in (r"(?:n?m)+#", r"(?:,\s?)+", r"(?:\d\.?)+", r"(?:Mr\.?\s)+",
                r"(?:n?+n)+#", r"(?>n?n)+#", r"(?:n?n)#"):
        assert not g._nested_repetition(pat), pat
    g.validate([("r", r"(?:Mr\.?\s)+", "")])


@pytest.mark.parametrize("pat", [
    r"m(?:[ab]?[cb])+#",
    r"(?:[-–]?[ –])+#",
    r"m(?:[ay]?[xy])+#",
    r"m(?:n?x?n)+#",
    r"m(?:(?:n)?n)+#",
    r"m(?:n?(?:n))+#",
    r"(?:\s?-?\s)+#",
])
def test_optional_overlap_on_a_shared_class_member_or_across_atoms_is_rejected(pat):
    """Each atom was tested against the OTHER atom's first member only, and
    only beside its direct neighbour: `[ab]?[cb]` (both take `b`), an en dash
    shared by `[-–]` and `[ –]`, `n?x?n` (x absent is `n?n`) and a wrapped
    `(?:n)?` all passed, and `re.sub` on "m" + 38 shared characters + "!"
    took seconds."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:\d?\d\.)+",
    r"(?:\d?\d\.)+\d{4}",
    r"\b(?:\d?\d[.:])+\d\d\b",
    r"(?:a?ab)+",
    r"(?:\d\d?\.)+",
])
def test_optional_overlap_pinned_by_a_disjoint_mandatory_atom_passes(pat):
    """A date or time rule like `(?:\\d?\\d\\.)+` must end every repetition on
    its `.`, so each one matches a single way and the pattern is linear —
    yet it was refused as a nested repetition."""
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:\d\d?\.\s?)+",
    r"\b(?:\d\d?\.\s?)+\d{4}\b",
    r"(?:[A-Z]\w?\.\s?)+",
    r"(?:\d\d?\.,?)+",
    r"(?:,? ?und )+",
    r"(?:[,.]? ?Punkt )+",
    r"(?:\.? ?neue Zeile )+",
])
def test_forced_optional_atom_does_not_count_as_a_second_split(pat):
    """`\\s?` after the `.` of `\\d\\d?\\.` (or `,?` before ` ?und`) can only
    take a character no other atom takes, so it never offers a second split —
    yet it counted as the second optional atom beside `\\d?` and dotted dates
    and repeated dictation cleanups were refused as nested repetitions."""
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [r"(?:a?b?ab)+#", r"(?:xa?[ab]b?)+#"])
def test_optional_atoms_overlapping_outside_their_stretch_stay_rejected(pat):
    """Each optional atom here shares a character with another atom of the
    branch, so both still count: `a?b?ab` reads "abab" as one repetition or
    two, and `xa?[ab]b?` matches "xab" as `a` + `[ab]` or `[ab]` + `b?`."""
    assert g._nested_repetition(pat)


@pytest.mark.parametrize("pat", [
    r"(?:,?\s?)+\.",
    r"(?:[,;]?\s?)*$",
    r"(?:h?a?)+!",
    r"(?:x?y?z?)+!",
    r"(?:\s?[,.]?)+(?=[A-Z])",
    r"((?:,?\s?){2})+#",
])
def test_repeated_body_of_only_optional_atoms_is_rejected(pat):
    """No atom shares a character with another, but nothing pins a
    repetition either: ", " is one repetition or two, so a run of k pairs
    reads 2**k ways — `(?:x?y?z?)+!` ran 12 s on "xyz" * 12 + "q"."""
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:(?:,?\s?){1})+#",
    r"(?:,?(?:\s?){1})+#",
    r"(?:(?:,?){1}\s?)+#",
    r"(?:(?:,?\s?){1,1}?)+#",
])
def test_optional_only_body_behind_a_single_count_is_rejected(pat):
    """`{1}` matches exactly as the bare group does, so wrapping the optional
    pair (or one atom of it) in `{1}` must not hide `(?:,?\\s?)+` from the
    screen — re.search ran 6-8 s on ", " * 22."""
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


def test_single_count_group_with_a_mandatory_atom_passes():
    """Seen through `{1}`, `,` still pins every repetition: linear."""
    assert not g._nested_repetition(r"(?:(?:,\s?){1})+#")


@pytest.mark.parametrize("pat", [
    r"(?:,?\s?){2}",
    r"(?:\w,?\s?)+",
    r"(?:\w(?:,?\s?))+",
    r"(?>,?\s?)+!",
    r"(?:,?\s?)++!",
    r"(?:[,;]?)+$",
])
def test_optional_only_body_that_never_repeats_or_splits_passes(pat):
    """A single run of optional atoms, one pinned by a mandatory atom, an
    atomic or possessive repeat and a lone optional atom match each input
    one way."""
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


def test_optional_overlap_between_pins_still_splits_ambiguously():
    """A pin only helps when the stretch it bounds holds one optional atom:
    `n?n?nx` and `n?nxn?n` (the tail and the next head) still split each
    stretch of n's two ways, 2**k on k repetitions."""
    assert g._nested_repetition(r"m(?:n?n?nx)+#")
    assert g._nested_repetition(r"m(?:n?nxn?n)+#")


@pytest.mark.parametrize("pat", [
    r"(?:[,.]?(?:\.|x))+#",
    r"(?:[ ,]?(?:,|;))+\.",
    r"\s(?:-?(?:-|=))+>",
    r"m(?:n?(?:n|m))+#",
    r"m(?:n?(?:n|m)n?x)+#",
])
def test_optional_atom_before_an_overlapping_alternation_is_rejected(pat):
    """`(?:\\.|x)` is the class `[.x]`, yet an alternation group stood as a
    pin, so `[,.]?(?:\\.|x)` passed where `[,.]?[.x]` is refused — and
    re.sub on a hallucinated run of 40 dots or dashes hung the worker."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [r"(?:a?(?:b|c))+", r"(?:\d?\d(?:\.|,))+x"])
def test_optional_atom_before_a_disjoint_alternation_passes(pat):
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:a?(?:a|aa)c)+#",
    r"(?:\w?(?:\w|\w\w)\.)+#",
    r"(?:\d?(?:\d|\d\d)\.)+x",
    r"(?:(?:a|aa)ca?)+#",
    r"(?:a?(?:(?:a|aa)c))+#",
])
def test_optional_atom_beside_a_multi_width_alternation_is_rejected(pat):
    """The `c` after `(?:a|aa)` pins where the group ends, so the follower
    exemption let it pass, while the group stood as a pin for the optional
    atom before it (or after the follower, for the next repetition): each
    repetition matched `aac` two ways and `"ab." * 40 + "!"` hung re.sub."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [r"(?:a?b(?:a|aa)c)+#", r"(?:(?:a|aa)c)+a?#"])
def test_multi_width_alternation_pinned_off_an_optional_atom_passes(pat):
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:(?:Komma|,) )+",
    r"(?:(?:Komma|,)\s)+",
    r"(?:\s?(?:Komma|,)\s)+",
    r"(?:(?:Fragezeichen|\?)\s)+",
    r"(?:\s(?:Ausrufezeichen|!))+",
    r"(?:[,;] (?:Klammer|\())+",
])
def test_letters_of_one_alternation_tail_do_not_split_against_each_other(pat):
    """`(?:Komma|,)` was read as `[K,]o?m?m?a?`: the two `m?` of the spoken
    word took each other's `m`, so every repeated rule holding a spoken
    punctuation word next to its sign was refused — though `Komma` comes
    whole or not at all."""
    assert not g._nested_repetition(pat)
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat", [
    r"(?:c(?:[bc]|[ab]cb)[bc][ab])+#",
    r"(?:(?:c|[ab]b[ab])a?)+#",
])
def test_alternation_tail_trading_with_the_atoms_around_it_stays_rejected(pat):
    """The tail positions still count one by one against the atoms outside
    the alternation: here the tail (or the next repetition's head) takes
    what the atoms after it take — the first runs for seconds on
    "cbcbcb" * 20 + "!"."""
    assert g._nested_repetition(pat)


@pytest.mark.parametrize("pat", [
    r"(?:[,.]?(?:\.(?=\.)|x))+#",
    r"\b(?:\.?(?:\.(?=\.)|;))+#",
    r"m(?:x?(?:x(?=x)|y))+#",
])
def test_lookaround_inside_an_overlapping_alternation_is_still_rejected(pat):
    """The class-like atom was the raw group text, lookahead included, and a
    trailing `(?=\\.)` never matches one character alone — the overlap
    vanished and a 40-dot run took about 15 s in re.sub."""
    assert g._nested_repetition(pat)
    with pytest.raises(ValueError, match="nested repetition"):
        g.validate([("r", pat, "")])


def test_negated_class_member_does_not_steer_the_run_off_its_character():
    """A negated class's members were read as literals the rule matches, so
    `[^1\\D]` swapped the `\\d` run to `2`, which the class DOES match: the
    probe matched at once and this was accepted."""
    pat = r"\d*\d*\d*\d*\d*\d*\d*[^1\D]"
    assert "1" * g._SYNTH_RUN + "#" in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [
    r"(?a)" + r"\s*" * 7 + r"[ \t]",
    r"(?a:" + r"\s*" * 7 + r"[ \t])",
    r"\s*" * 7 + r"[ \t ]",
    r"\s*" * 7 + r"[ \t\n ]",
])
def test_whitespace_run_has_a_sibling_every_s_mode_matches(pat):
    """The `\\s` run's only siblings were a tab and U+2003: under `(?a)` the
    U+2003 run matched nothing, and a rule writing all three left the space
    run in place, which the class matched at once. Both were accepted."""
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, run", [
    (r"\s*" * 7 + r"\S", " "),
    (r"\d*" * 7 + r"\D", "1"),
    (r"\d*" * 7 + r"[^1]", "1"),
])
def test_run_ending_in_a_set_that_matches_the_terminator_is_rejected(pat, run):
    """The terminator was only checked as raw text, so a trailing negated set
    matched the `#` and the probe ended on the first try: these were
    accepted, yet re.sub over a 30-space or 30-digit tail took over a
    second, growing ~n^7."""
    assert run * g._SYNTH_RUN in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, probe", [
    (r"\d*" * 7 + r"[1-8]", "9" * 240),
    (r"\s*" * 7 + r"[\t-\r ]", " " * 240),
    (r"\s*" * 7 + r"(?=\S)", " " * 240),
    (r"\d*" * 7 + r"(?=\D)", "1" * 240),
    (r"\s*" * 7 + r"(?:x|$)|[^\w\s]", " " * 240 + "#"),
    (r"\s*" * 7 + r"(?:x|\Z)|\S", " " * 240 + "#"),
    (r"\s*" * 7 + r".", "\n" * 240),
    (r"\s*" * 7 + r"[^\n]", "\n" * 240),
    (r"\W*" * 7 + r".", "\n" * 240),
])
def test_run_that_one_probe_choice_misses_is_rejected(pat, probe):
    """One probe string, picked by heuristics, always had a rule that matched
    it at once: a sibling inside the rule's class range, a `#` a lookahead or
    another branch consumes, a bare run a trailing `.` ends on. These were
    accepted, yet re.sub over a run of ~40 such characters takes seconds,
    growing ~n^7. The probes are a union now."""
    assert any(s.startswith(probe) for s in g._synthetic_fixtures(pat))
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [r"\s+$", r"\s+([.,!?])", r"(\w+)\W", r"(\d+)\s*Uhr"])
def test_ordinary_rule_keeps_its_terminated_probe(pat):
    assert g._synthetic_fixtures(pat)[0].endswith("#")
    g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [
    r"\s*" * 7 + "$",
    r"-*" * 7 + r"\Z",
    "(?m)" + r"[ \t]*" * 5 + "$",
    "(?:" + r"\s*" * 7 + "$)?",
])
def test_run_ending_in_an_anchor_keeps_its_terminator(pat):
    """The terminator skip took ANY match, and the empty one the `$` finds at
    the end of `#` dropped every terminator: the bare run satisfied the `$`
    at once, so these were accepted, yet re.sub over 100 spaces or dashes
    before a word took 18-25 s."""
    assert g._synthetic_fixtures(pat)[0].endswith("#")
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, probe", [
    (r"\s*" * 7 + r"(?:[^\w\s]|$)", " " * 240 + "z"),
    (r"\d*" * 7 + r"(?:\W|$)", "1" * 240 + "z"),
    (r"\d*" * 7 + r"(?:[!-/]|$)", "1" * 240 + "@"),
    (r"\s*" * 7 + r"(?:[!-@]|$)", " " * 240 + "~"),
])
def test_run_whose_tail_eats_every_union_terminator_is_rejected(pat, probe):
    """Every terminator of the union was punctuation, the bare run ends at
    `$`, and the pre-union pick (an unwritten terminator the rule cannot match
    non-empty) had dropped out of it: a tail eating `\\W` or `[!-/]` plus the
    end of the text matched every probe at once. These were accepted, yet
    re.sub over a ~40-character run followed by a letter or space takes
    10-20 s."""
    assert probe in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


_NUMBERS_RULES = [
    ("rule 'numbers' entry 1", r"(\d+)[.,]?(\d*)\s*(?:Prozent|%)", r"\1,\2 %"),
    ("rule 'numbers' entry 2", r"(\d+)\s*(\d*)\s*Euro", r"\1\2 €"),
]


def test_ordinary_polynomial_rules_are_probed_once_and_accepted():
    """The union probed every digit with every terminator as a full 240-char
    run, ~30 per entry against a budget sized for one, so these two ordinary
    rules (polynomial on a digit run no transcript holds) were refused as
    "likely catastrophic backtracking". Runs the rule treats alike at every
    position are probed once."""
    for _name, pat, _repl in _NUMBERS_RULES:
        assert len(g._synthetic_fixtures(pat)) == 1
    g.validate(_NUMBERS_RULES)


@pytest.mark.parametrize("pat", [
    r"(\d+)\s*(\d*)\s*(?i:euro)",
    r"(\d+)\s*(\d*)\s*" + "".join(chr(0xC0 + i) for i in range(70)),
])
def test_ordinary_rule_with_a_flag_group_or_many_atoms_is_probed_once(pat):
    """A scoped flag group (or more than 64 distinct atoms) made the guard
    give up on telling run characters apart, so every digit got its own
    240-char probe: the pair `(?i:euro)` / `(?i:prozent)` was refused as
    "likely catastrophic backtracking" while the `(?i)` spelling saved."""
    assert len(g._synthetic_fixtures(pat)) == 1
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat, probe", [
    (r"\n" + r"\s*" * 7 + "x", "\n" * 240 + "#"),
    (r"(?<=\n)" + r"\s*" * 7 + "x", "\n" * 240 + "#"),
    (r"[ \t]*" + r"\n*" * 6 + "x", "\n" * 240 + "#"),
    (r"(?=\t)" + r"\s*" * 6 + "x", "\t" * 240 + "#"),
])
def test_run_no_candidate_matches_keeps_the_run_the_rule_starts_on(pat, probe):
    """The screen kept one candidate per outcome, and "no match anywhere" is
    every candidate's outcome here: only the space run was probed, which the
    leading `\\n` or `\\t` refuses at once. These were accepted, yet re.sub
    over 30 newlines before a `#` takes 0.4 s, growing ~n^7."""
    assert probe in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, probe", [
    (r"\s*x|" + r"\n*" * 7 + r"y|\s", "\n" * 240),
    (r"2" + r"\d*" * 7 + r"x|\d+", "2" * 240),
    (r"\d*" * 7 + r"[^9#]|\d+", "9" * 240),
])
def test_run_a_cheap_branch_matches_like_its_sibling_keeps_its_probe(pat, probe):
    """The screen keyed a run the rule matches on its spans alone, and the
    cheap last branch gives every run character the same ones: only the
    space or `1` run was probed, which the slow branch refuses at once.
    These were accepted, yet re.sub over 40 newlines, `2`s or `9`s takes
    over a second, growing ~n^7."""
    assert any(s.startswith(probe) for s in g._synthetic_fixtures(pat))
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat, probe", [
    (r"(?=\s+[a-z])" + r"\s*" * 7 + "#", " " * 240 + "z"),
    (r"(?=\d+z)" + r"\d*" * 7 + "#", "1" * 240 + "z"),
])
def test_run_whose_lookahead_needs_a_letter_keeps_the_letter_probe(pat, probe):
    """A run the rule matches nowhere was keyed without its terminator, so
    every terminator collapsed onto the first one, `!`, which the lookahead
    refuses at once. These were accepted, yet re.sub over 30 spaces or
    digits before a word takes about a second, growing ~n^7."""
    assert probe in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


@pytest.mark.parametrize("pat", [r"(\d+)\s*%", r"\s+([,.!?])", r"(\d+)\s*Uhr\b"])
def test_ordinary_rule_probes_its_terminators_once(pat):
    """Keying on the terminator must not split terminators the rule cannot
    tell apart: `Prozent`'s `z` (in _NUMBERS_RULES) never meets a digit run's
    end, and neither does a `\\b` behind `Uhr`."""
    assert len(g._synthetic_fixtures(pat)) == 1
    g.validate([("r", pat, "")])


@pytest.mark.parametrize("pat, probe", [
    (r"\d+" * 20 + r"(?:\W|$)", "1" * 240 + "z"),
    (r"\s+" * 20 + r"(?:[^\w\s]|$)", " " * 240 + "z"),
])
def test_rule_wider_than_the_short_run_keeps_the_terminator_it_refuses(pat, probe):
    """The short run was capped at 16 characters, so a rule needing 20 matched
    no candidate and only the first terminator, which its tail eats at once,
    was probed. These were accepted, yet re.sub over 26 digits or spaces
    before a letter takes 3 s, and over 30 more than a minute."""
    assert probe in g._synthetic_fixtures(pat)
    with pytest.raises(ValueError):
        g.validate([("r", pat, "")], timeout=1.5)


def test_wide_ordinary_rule_is_probed_once_and_accepted():
    pat = r"(\d+)\s*(\d*)\s*Euro und Cent Betrag"
    assert len(g._synthetic_fixtures(pat)) == 1
    g.validate([("r", pat, "")])
