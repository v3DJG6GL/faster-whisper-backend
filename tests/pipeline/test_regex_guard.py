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
    synth = g._synthetic_fixture(pat)
    assert synth is not None and synth.startswith("-" * g._SYNTH_RUN)
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
    from a helper that ran — validate fails OPEN (with a warning) on a crash."""
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
    assert seen == [min(g._GUARD_TIMEOUT + g._PER_CHECK_BUDGET * 3000,
                        g._GUARD_TIMEOUT_MAX)]
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
    assert seen[0] == g._GUARD_TIMEOUT + g._PER_CHECK_BUDGET * 400
    assert g._GUARD_TIMEOUT < seen[0] < g._GUARD_TIMEOUT_MAX
    assert seen[1] == g._GUARD_TIMEOUT_MAX


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
