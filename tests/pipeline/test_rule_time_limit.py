"""The time limit on the pipeline rules when they run (pipeline/engine.py
"Time limit", pipeline/rule_helper.py).

With the helper pool running, a walk is one round trip to a helper process
with a deadline. A rule that would run for minutes is skipped, quarantined,
logged and traced, and the walk is redone without it; every walk that
finishes in time gives exactly what the in-process walk gives.

The suite runs with the pool off (tests/conftest.py); these tests start one
themselves. The slow rule is installed straight into the compiled rules: the
save-time screen (regex_guard) refuses it, and what is tested here is the
case where a pattern got past that screen.
"""

import asyncio
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.pipeline import rule_helper
from faster_whisper_backend.settings import config as cfg
from tests.conftest import read_repo_text

# Exponential on a run of "a" that does not end the text: 2**40 steps here.
_SLOW = r"(a|a)+$"
_SLOW_TEXT = "x " + "a" * 40 + "! b"
_SLOW_WANT = "y " + "a" * 40 + "! c"
# Cubic on a run of "a" with no "b" after it, and finite: it can be sized to
# a share of a deadline (_cubic_run).
_CUBIC = r"a*a*a*b"


@pytest.fixture
def pool():
    """A running pool of two helpers, ready before the test starts."""
    rule_helper.start(2)
    running = rule_helper._POOL
    assert running.wait_ready(30)
    yield running
    rule_helper.shutdown()


def _row(name, label, pattern, repl, card_no, sub_no=1, **kw):
    cre = re.compile(pattern)
    template, growth = pl_engine._template_growth(cre, repl)
    return pl_engine._CompiledRule(
        name, label, "regex-list", cre, repl, kw.pop("enabled", True), card_no, sub_no,
        max_growth=growth, template=template, **kw)


def _slow_rows():
    return [_row("card", "Card · x", "x", "y", 1, 1),
            _row("card", "Card · slow", _SLOW, "", 1, 2),
            _row("tail", "Tail · b", "b", "c", 2, 1)]


def _install_rows(monkeypatch, rows):
    """Bind `rows` as the compiled rule set (no save-time screen)."""
    monkeypatch.setattr(pl_engine, "_COMPILED_RULES", rows)
    monkeypatch.setattr(pl_engine, "_TERMINAL_CARD_NO", rows[-1].card_no + 1)
    return rows


def _install_cfg_rules(monkeypatch, rules):
    """Compile `rules` (cfg.PIPELINE_RULES shape) the way a save does; the
    engine's rule state is put back afterwards."""
    for name in ("_COMPILED_RULES", "_TERMINAL_NAME", "_TERMINAL_LABEL",
                 "_TERMINAL_CARD_NO", "_RULES_GEN", "_WARM_SPECS", "_QUARANTINE"):
        monkeypatch.setattr(pl_engine, name, getattr(pl_engine, name))
    monkeypatch.setattr(cfg, "PIPELINE_RULES", rules)
    pl_engine.rebuild_caches()
    return pl_engine._COMPILED_RULES


def _in_process(fn):
    """fn() with the pool out of sight: the walk this process always made."""
    running, rule_helper._POOL = rule_helper._POOL, None
    try:
        return fn()
    finally:
        rule_helper._POOL = running


def _count_calls(monkeypatch):
    calls = []
    real = rule_helper.call

    def counting(helper, payload, timeout):
        calls.append(timeout)
        return real(helper, payload, timeout)

    monkeypatch.setattr(rule_helper, "call", counting)
    return calls


def _cubic_run(seconds):
    """(text, took): a run of "a" that _CUBIC takes about `seconds` on in
    this process, and what it took when measured."""
    def took(n):
        text = "a" * n + "!"
        t0 = time.perf_counter()
        re.sub(_CUBIC, "", text)
        return time.perf_counter() - t0

    n = 150
    n = max(n, round(n * (seconds / min(took(n), took(n))) ** (1 / 3)))
    return "a" * n + "!", min(took(n), took(n))


def _limit_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "time limit" in r.getMessage()]


# --- the slow rule ----------------------------------------------------------

def test_a_rule_that_never_finishes_is_skipped_and_named_while_the_loop_stays_live(
        pool, monkeypatch, caplog):
    rules = _install_rows(monkeypatch, _slow_rows())
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.3)
    trace: list = []

    async def main():
        loop = asyncio.get_running_loop()
        gaps = []

        async def ticker():
            last = loop.time()
            while True:
                await asyncio.sleep(0.01)
                now = loop.time()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        out = await asyncio.to_thread(pl_engine._postprocess_text, _SLOW_TEXT, trace=trace)
        tick.cancel()
        return out, max(gaps)

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        t0 = time.perf_counter()
        out, worst_gap = asyncio.run(main())
        took = time.perf_counter() - t0

    # The rules around the slow one still ran, on a second helper.
    assert out == _SLOW_WANT
    assert took < 1.5
    # In-process the match holds the GIL: the loop would not tick at all.
    assert worst_gap < 0.2
    assert [r[0] for r in trace] == [
        "#1.1 Card · x", "#1.2 Card · slow [TIME LIMIT — not applied]", "#2.1 Tail · b"]
    warned = _limit_warnings(caplog)
    assert len(warned) == 1
    assert "#1.2" in warned[0] and "'card'" in warned[0] and "Card · slow" in warned[0]
    assert pl_engine._skip_for(rules) == {1}


def test_the_next_walk_does_not_wait_and_new_rules_end_the_quarantine(
        pool, monkeypatch, caplog):
    _install_rows(monkeypatch, _slow_rows())
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.3)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert pl_engine._postprocess_text(_SLOW_TEXT) == _SLOW_WANT
        trace: list = []
        t0 = time.perf_counter()
        assert pl_engine._postprocess_text(_SLOW_TEXT, trace=trace) == _SLOW_WANT
        assert time.perf_counter() - t0 < 0.15          # half a deadline
        assert "#1.2 Card · slow [TIME LIMIT — not applied]" in [r[0] for r in trace]
        assert len(_limit_warnings(caplog)) == 1        # logged once, not per walk

        # The quarantine names rows of one rule set: the same rows compiled
        # again are a new set, tried again and quarantined again.
        again = _install_rows(monkeypatch, _slow_rows())
        assert pl_engine._skip_for(again) == frozenset()
        assert pl_engine._postprocess_text(_SLOW_TEXT) == _SLOW_WANT
        assert pl_engine._skip_for(again) == {1}
        assert len(_limit_warnings(caplog)) == 2

    # And a rebuild (what any save of the rules runs) drops it outright.
    _install_cfg_rules(monkeypatch, [])
    assert pl_engine._QUARANTINE == (None, frozenset())


def test_a_walk_stops_after_a_few_slow_rules_and_returns_its_input(pool, monkeypatch, caplog):
    rows = [_row("card", f"Card · slow {i}", _SLOW, "", 1, i) for i in range(1, 5)]
    rules = _install_rows(monkeypatch, rows)
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.1)
    monkeypatch.setattr(pl_engine, "_WALK_ATTEMPTS", 2)
    trace: list = []
    budget = pl_engine.WalkBudget(len(_SLOW_TEXT))
    left = budget.left
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        out = pl_engine._postprocess_text(_SLOW_TEXT, trace=trace, budget=budget)
    assert out == _SLOW_TEXT
    assert trace == [("pipeline [TIME LIMIT — rules not applied]", _SLOW_TEXT, _SLOW_TEXT)]
    assert budget.left == left - len(_SLOW_TEXT)
    assert pl_engine._skip_for(rules) == {0, 1}
    assert "without the rules applied" in caplog.text


def test_seam_culprit_is_under_the_same_limit(pool, monkeypatch):
    rules = _install_rows(monkeypatch, _slow_rows())
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.3)
    t0 = time.perf_counter()
    got = pl_engine.seam_culprit("x " + "a" * 40 + "!", _SLOW_TEXT)
    assert time.perf_counter() - t0 < 1.5
    assert pl_engine._skip_for(rules) == {1}
    assert got == _in_process(
        lambda: pl_engine.seam_culprit("x " + "a" * 40 + "!", _SLOW_TEXT))


# --- which rule a passed deadline is put down to -------------------------------------

def test_slowish_rows_that_only_together_pass_the_deadline_quarantine_nobody(
        pool, monkeypatch, caplog):
    # Twelve finite rows of a fifth of the deadline each: whichever row is in
    # the slot when the deadline passes has only just started.
    text, each = _cubic_run(0.08)
    rows = _install_rows(monkeypatch, [
        _row("card", f"Card · {i}", _CUBIC, "", 1, i) for i in range(1, 13)]
        + [_row("tail", "Tail · !", "!", "?", 2)])
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S",
                        each / 0.2 - pl_engine._DEADLINE_PER_CHAR_S * len(text))
    calls = _count_calls(monkeypatch)
    trace: list = []
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert pl_engine._postprocess_text(text, trace=trace) == text
    assert pl_engine._skip_for(rows) == frozenset()
    assert trace == [("pipeline [TIME LIMIT — rules not applied]", text, text)]
    assert "no single rule" in caplog.text and "skipped from now on" not in caplog.text
    assert len(calls) == 2                    # a load and ONE walk: not redone
    # Every rule is still there for a text they are quick on.
    assert pl_engine._postprocess_text("aa!") == "aa?"


def test_a_rule_that_never_finishes_is_blamed_after_a_slowish_neighbour(pool, monkeypatch):
    text, took = _cubic_run(0.12)
    rows = _install_rows(monkeypatch, [_row("card", "Card · finite", _CUBIC, "", 1),
                                       _row("stuck", "Stuck", _SLOW, "", 2),
                                       _row("tail", "Tail · !", "!", "?", 3)])
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S",
                        took / 0.4 - pl_engine._DEADLINE_PER_CHAR_S * len(text))
    assert pl_engine._postprocess_text(text) == text[:-1] + "?"
    assert pl_engine._skip_for(rows) == {1}


def _past_deadline(monkeypatch, mark, on_wait):
    """A stand-in helper whose walk is past its deadline with `mark` in the
    progress slot; `on_wait(helper)` plays the wait that follows."""
    helper = SimpleNamespace(mark=mark)
    waits: list = []

    def timed_out(_helper, _payload, _timeout):
        raise rule_helper.HelperTimeout

    def wait(_helper, timeout):
        waits.append(timeout)
        return on_wait(helper)

    monkeypatch.setattr(rule_helper, "call", timed_out)
    monkeypatch.setattr(rule_helper, "wait", wait)
    return helper, waits


def _still_running(_helper):
    raise rule_helper.HelperTimeout


def test_a_row_is_blamed_only_after_its_own_share_of_the_deadline(monkeypatch):
    now = time.monotonic_ns
    share = 10.0 * pl_engine._BLAME_SHARE

    def blame(mark, on_wait):
        helper, waits = _past_deadline(monkeypatch, mark, on_wait)
        with pytest.raises(pl_engine._WalkTimeout) as err:
            pl_engine._call_walk(helper, b"", 10.0)
        return err.value.blame, waits

    # Outside any rule; and a row that has had its share already: at once.
    assert blame([-1, 0], _still_running) == (-1, [])
    assert blame([3, now() - int((share + 1) * 1e9)], _still_running) == (3, [])

    # A row that only just started gets the rest of its share first ...
    got, waits = blame([3, now() - int(1e9)], _still_running)
    assert got == 3                           # ... and is blamed if it is still there,
    assert len(waits) == 1 and share - 1.5 < waits[0] <= share - 1

    def moved_on(helper):
        helper.mark[:] = [4, now()]
        raise rule_helper.HelperTimeout

    got, waits = blame([3, now() - int(1e9)], moved_on)
    assert got is None and len(waits) == 1    # nobody if the walk went on to another row,

    helper, waits = _past_deadline(monkeypatch, [3, now() - int(1e9)], lambda _h: b"reply")
    assert pl_engine._call_walk(helper, b"", 10.0) == b"reply"     # nothing if it finished.

    # A start stamp this process cannot place (ahead of its own clock) never
    # buys more than one share.
    got, waits = blame([3, now() + int(3600e9)], _still_running)
    assert got == 3 and waits == [share]


def test_a_straggler_of_an_older_rule_set_leaves_the_live_quarantine_alone(pool, monkeypatch):
    old = _install_rows(monkeypatch, _slow_rows())
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.4)
    got: list = []
    walker = threading.Thread(
        target=lambda: got.append(pl_engine._postprocess_text(_SLOW_TEXT)), daemon=True)
    walker.start()
    end = time.monotonic() + 10
    while len(pool._idle) == 2 and time.monotonic() < end:
        time.sleep(0.01)                      # until the walk holds a helper
    # The rules are saved under the walk: a new set, with a quarantine of
    # its own.
    live = _install_rows(monkeypatch, _slow_rows() + [_row("extra", "Extra", "q", "q", 3)])
    pl_engine._quarantine(live, 3, "ran past the time limit")
    walker.join(30)
    assert got == [_SLOW_WANT]                # the old walk's retry skipped its slow rule
    assert pl_engine._skip_for(live) == {3}
    assert pl_engine._skip_for(old) == frozenset()


# --- same output as the in-process walk ----------------------------------------

_WORDS = {
    "de": ["Hallo", "Komma", "Punkt", "neue Zeile", "das ist", "größer als", "Straße",
           "Fragezeichen", "12... Komma 5", "Herz-kreislauf", "und", "Doppelpunkt",
           "10.23 Uhr", "ähm", "Klammer auf", "Klammer zu", "120 Schrägstrich 80"],
    "en": ["Hello", "the patient", "comma", "is fine.", "No, really!", "3 - 5", "e.g.",
           "what?", "new line", "a-b", "...", "  twice  "],
    "ja": ["こんにちは", "患者は", "、", "。", "元気です", "１２", "新しい行", "テスト", " "],
}


def _texts(seed, n):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        langs = rng.choice((["de"], ["en"], ["ja"], ["de", "en", "ja"]))
        words = [rng.choice(_WORDS[rng.choice(langs)]) for _ in range(rng.randint(0, 40))]
        out.append(rng.choice(("", " ", "\n")) + rng.choice((" ", ", ", ". ", "\n")).join(words)
                   + rng.choice(("", " ", ".", " \t")))
    return out


def _factory_rules():
    return json.loads(read_repo_text("config.json"))["PIPELINE_RULES"]


def _variant_rules():
    """The factory rules with two cards switched off (so a force-include has
    something to do), one repeating, and an expanding card that can reach
    the output bound."""
    rules = _factory_rules()
    for rule in rules:
        if rule["name"] in ("dedup-punct", "final-cleanup"):
            rule["enabled"] = False
        if rule["name"] == "capitalize-letter-after-hyphen":
            rule["repeat"] = True
    rules.insert(0, {"name": "grow", "label": "Grow", "type": "regex-list", "repeat": True,
                     "entries": [{"pattern": "e", "replacement": "ee"},
                                 {"pattern": "(Hallo)", "replacement": "\\1 \\1"}]})
    return rules


def _same_trace(a, b):
    assert a == b
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a[1:], row_b[1:]):
            assert type(x) is type(y)
            assert getattr(x, "full_len", None) == getattr(y, "full_len", None)
            assert getattr(x, "changed", None) == getattr(y, "changed", None)


@pytest.mark.parametrize("make_rules, seed, trace_max, rows_max", [
    (_factory_rules, 1, 50_000, 500),      # config.json as shipped
    (_variant_rules, 2, 50_000, 500),
    (_variant_rules, 3, 40, 6),            # cut trace texts, a short trace window
])
def test_a_walk_in_a_helper_gives_what_the_in_process_walk_gives(
        pool, monkeypatch, caplog, make_rules, seed, trace_max, rows_max):
    rules = _install_cfg_rules(monkeypatch, make_rules())
    slugs = sorted({r.name for r in rules} | {"trim-edges"})
    monkeypatch.setattr(pl_engine, "_TRACE_TEXT_MAX", trace_max)
    monkeypatch.setattr(pl_engine, "_TRACE_MAX_ROWS", rows_max)
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES", {}, raising=False)
    calls = _count_calls(monkeypatch)
    rng = random.Random(seed)
    texts = _texts(seed, 200)
    mismatches = 0
    for text in texts:
        pick = lambda: rng.sample(slugs, rng.randint(0, 3))  # noqa: E731
        cfg.MODEL_OVERRIDES = {"m1": {"PIPELINE_RULES_EXCLUDE": pick(),
                                      "PIPELINE_RULES_INCLUDE": pick()}}
        kw = {
            "model_name": rng.choice((None, "m1", "other")),
            "language": rng.choice((None, "de", "en", "es", "ja")),
            "extra_excludes": rng.choice((None, set(pick()))),
            "ident": rng.choice((None, None, SimpleNamespace(
                pipeline_exclude=pick(), pipeline_include=pick()))),
            "bound": rng.choice((None, None, len(text) + rng.randint(0, 30), 100)),
        }
        with_trace = rng.random() < 0.7
        prior = [("earlier", "a", "b")] * rng.randint(0, 8)
        budget_len = rng.choice((None, len(text), 3 * len(text)))

        def walk():
            trace = list(prior) if with_trace else None
            budget = None if budget_len is None else pl_engine.WalkBudget(budget_len)
            caplog.clear()
            out = pl_engine._postprocess_text(text, trace=trace, budget=budget, **kw)
            return out, trace, budget and budget.left, [r.getMessage() for r in caplog.records]

        with caplog.at_level(logging.WARNING, logger="whisper-api"):
            want = _in_process(walk)
            before = len(calls)
            got = walk()
        assert len(calls) > before, "the walk did not go through a helper"
        if got != want:
            mismatches += 1
            continue
        if with_trace:
            _same_trace(got[1], want[1])
        # The seam diagnosis walks the same rule selection.
        if len(text) > 4 and rng.random() < 0.2:
            cut = text[:len(text) // 2]
            seam_kw = {k: kw[k] for k in ("model_name", "language", "ident")}
            assert pl_engine.seam_culprit(cut, text, **seam_kw) == _in_process(
                lambda: pl_engine.seam_culprit(cut, text, **seam_kw))
    assert mismatches == 0
    assert pl_engine._skip_for(rules) == frozenset()


def test_an_error_a_rule_raises_reaches_the_caller_as_it_did(pool, monkeypatch):
    bad = pl_engine._CompiledRule(
        "bad", "Bad · template", "regex-list", re.compile("(b)"), "\\2", True, 2, 1)
    _install_rows(monkeypatch, [_row("card", "Card · x", "x", "yy", 1, 1), bad])

    def walk():
        trace: list = []
        budget = pl_engine.WalkBudget(4)
        with pytest.raises(re.error) as err:
            pl_engine._postprocess_text("x ab", trace=trace, budget=budget)
        return str(err.value), trace, budget.left

    calls = _count_calls(monkeypatch)
    assert walk() == _in_process(walk)
    assert calls


# --- the rule set a helper holds ---------------------------------------------------

def test_a_changed_rule_set_reaches_the_helper_before_the_next_walk(monkeypatch):
    rule_helper.start(1)                      # one helper: every walk meets it
    assert rule_helper._POOL.wait_ready(30)
    first = _install_rows(monkeypatch, [_row("card", "Card · x", "x", "1", 1)])
    assert pl_engine._postprocess_text("x") == "1"
    _install_rows(monkeypatch, [_row("card", "Card · x", "x", "2", 1)])
    assert pl_engine._postprocess_text("x") == "2"
    _install_rows(monkeypatch, first)
    assert pl_engine._postprocess_text("x") == "1"
    # Compiled from cfg, callbacks included (a map and a wordlist replacer
    # are closures: the helper rebuilds them from their source).
    _install_cfg_rules(monkeypatch, _factory_rules())
    text = "Hallo Komma das ist gut Punkt Neue Zeile ja"
    assert pl_engine._postprocess_text(text, language="de") == _in_process(
        lambda: pl_engine._postprocess_text(text, language="de"))
    assert len(rule_helper._POOL.pids()) == 1


def test_walks_from_many_threads_share_the_pool(pool, monkeypatch):
    _install_cfg_rules(monkeypatch, _factory_rules())
    texts = _texts(7, 40)
    want = _in_process(lambda: [pl_engine._postprocess_text(t, language="de") for t in texts])
    calls = _count_calls(monkeypatch)
    got: dict = {}

    def worker(k):
        got[k] = [pl_engine._postprocess_text(t, language="de") for t in texts]

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert all(got.get(k) == want for k in range(6))
    assert len(calls) >= 6 * len(texts)


# --- helpers that go away ------------------------------------------------------------

def test_helpers_killed_between_walks_are_replaced(pool, monkeypatch, caplog):
    rules = _install_rows(monkeypatch, [_row("card", "Card · x", "x", "y", 1)])
    assert pl_engine._postprocess_text("x") == "y"
    before = pool.pids()
    for proc in list(pool._procs):
        proc.kill()
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert pl_engine._postprocess_text("x") == "y"
    assert "helper process was lost" in caplog.text
    assert pl_engine._skip_for(rules) == frozenset()
    assert pool.wait_ready(30)
    assert len(pool.pids()) == 2 and not set(pool.pids()) & set(before)


def test_without_a_helper_the_rules_run_in_process_and_the_log_says_so_once(
        monkeypatch, caplog):
    def no_helper(register):
        raise OSError("no more processes")

    monkeypatch.setattr(rule_helper, "_spawn", no_helper)
    _install_rows(monkeypatch, [_row("card", "Card · x", "x", "y", 1)])
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        rule_helper.start(2)
        assert pl_engine._postprocess_text("x") == "y"
        assert pl_engine._postprocess_text("x x") == "y y"
    said = [r.getMessage() for r in caplog.records if "could not be started" in r.getMessage()]
    assert len(said) == 1 and "no more processes" in said[0]
    assert "without a time limit" in said[0]


def test_a_rule_set_a_helper_cannot_rebuild_runs_in_process(pool, monkeypatch, caplog):
    odd = pl_engine._CompiledRule(
        "odd", "Odd", "callback:upper", re.compile("x"), lambda m: "y", True, 1)
    _install_rows(monkeypatch, [odd])
    calls = _count_calls(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert pl_engine._postprocess_text("x") == "y"
        assert pl_engine._postprocess_text("xx") == "yy"
    assert not calls
    assert caplog.text.count("cannot be sent to a helper process") == 1


def test_shutdown_leaves_no_child_even_under_a_walk_in_flight(pool, monkeypatch):
    _install_rows(monkeypatch, _slow_rows())
    procs = list(pool._procs)
    assert len(procs) == 2 and all(p.poll() is None for p in procs)
    done: list = []
    walker = threading.Thread(
        target=lambda: done.append(pl_engine._postprocess_text(_SLOW_TEXT)), daemon=True)
    walker.start()                            # deadline: the default, seconds away
    end = time.monotonic() + 10
    while len(pool._idle) == 2 and time.monotonic() < end:
        time.sleep(0.01)                      # until the walk holds a helper
    time.sleep(0.05)

    rule_helper.shutdown()

    assert rule_helper._POOL is None
    assert all(p.poll() is not None for p in procs)      # killed and reaped
    assert pool.pids() == []
    # The walk lost its helper inside a rule: it ends without its rules
    # rather than run that rule in this process.
    walker.join(10)
    assert done == [_SLOW_TEXT]


# --- every helper held ---------------------------------------------------------------

def test_a_busy_pool_never_sends_a_walk_into_the_server_process(pool, monkeypatch, caplog):
    rows = [_row("card", f"Card · slow {i}", _SLOW, "", 1, i) for i in range(1, 9)]
    _install_rows(monkeypatch, rows)
    monkeypatch.setattr(pl_engine, "_DEADLINE_FLOOR_S", 0.5)
    monkeypatch.setattr(pl_engine, "_WALK_ATTEMPTS", 2)
    monkeypatch.setattr(rule_helper, "_ACQUIRE_TIMEOUT_S", 0.15)
    in_server: list = []

    def recorded(text, *args, **kwargs):
        # In place of the in-process walk, which would never return here.
        in_server.append(threading.current_thread().name)
        return text

    monkeypatch.setattr(pl_engine, "_walk", recorded)
    outs: list = []
    holders = [threading.Thread(
        target=lambda: outs.append(pl_engine._postprocess_text(_SLOW_TEXT)), daemon=True)
        for _ in range(2)]
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        for t in holders:
            t.start()
        end = time.monotonic() + 10
        while pool._idle and time.monotonic() < end:
            time.sleep(0.01)                  # until both helpers are held
        trace: list = []
        budget = pl_engine.WalkBudget(len(_SLOW_TEXT))
        left = budget.left
        t0 = time.perf_counter()
        out = pl_engine._postprocess_text(_SLOW_TEXT, trace=trace, budget=budget)
        took = time.perf_counter() - t0
        for t in holders:
            t.join(30)
    # Neither the walk that found every helper held nor the holders' own
    # retries (their replacements are still starting) ran in this process.
    assert in_server == []
    assert out == _SLOW_TEXT and took < 0.45
    assert trace == [("pipeline [TIME LIMIT — rules not applied]", _SLOW_TEXT, _SLOW_TEXT)]
    assert budget.left == left - len(_SLOW_TEXT)
    assert outs == [_SLOW_TEXT, _SLOW_TEXT]
    assert "came free" in caplog.text and "without the rules applied" in caplog.text
    assert "inside the server process" not in caplog.text


def test_a_pool_with_every_helper_held_answers_busy_and_says_so_once(pool, caplog):
    held = [pool.acquire(), pool.acquire()]
    try:
        with caplog.at_level(logging.WARNING, logger="whisper-api"):
            t0 = time.monotonic()
            assert pool.acquire(0.05) is rule_helper.BUSY
            assert pool.acquire(0) is rule_helper.BUSY
            assert time.monotonic() - t0 < 1
        assert caplog.text.count("came free") == 1
    finally:
        for helper in held:
            pool.release(helper)
    got = pool.acquire(0)                     # a free helper needs no wait
    assert got is not None and got is not rule_helper.BUSY
    pool.release(got)


def test_helpers_that_never_start_in_time_still_mean_in_process(monkeypatch, caplog):
    gate = threading.Event()

    def stuck(register):
        gate.wait(20)
        raise OSError("stuck start")

    monkeypatch.setattr(rule_helper, "_spawn", stuck)
    monkeypatch.setattr(rule_helper, "_ACQUIRE_TIMEOUT_S", 0.2)
    _install_rows(monkeypatch, [_row("card", "Card · x", "x", "y", 1)])
    try:
        rule_helper.start(2)
        with caplog.at_level(logging.WARNING, logger="whisper-api"):
            # No helper has ever come up: that is the platform's doing, not
            # the rules', and the text is still formatted.
            assert pl_engine._postprocess_text("x") == "y"
        assert "without a time limit" in caplog.text
    finally:
        gate.set()
        rule_helper.shutdown()


# --- a helper's start ----------------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/net/unix")
def test_another_local_process_finds_no_endpoint_of_a_helper():
    """The pair a helper talks over has no name: nothing for a process without
    it to find, connect to, or end a start with."""
    mine = f"@fwb-rules-{os.getpid()}-"
    seen: list = []
    stop = threading.Event()

    def names():
        with open("/proc/net/unix", encoding="utf-8", errors="replace") as fh:
            return [line.split()[-1] for line in fh if mine in line]

    def stranger():
        while not stop.is_set():
            for name in names():
                seen.append(name)
                with socket.socket(socket.AF_UNIX) as sock:
                    try:
                        sock.connect("\0" + name[1:])
                    except OSError:
                        pass

    watcher = threading.Thread(target=stranger, daemon=True)
    watcher.start()
    try:
        rule_helper.start(3)
        ready = rule_helper._POOL.wait_ready(30)
        listed = names()
    finally:
        stop.set()
        watcher.join(10)
        rule_helper.shutdown()
    assert seen == [] and listed == []
    assert ready


def test_a_failed_start_is_made_up_after_the_retry_time(monkeypatch):
    real = rule_helper._spawn
    failed: list = []

    def first_fails(register):
        if not failed:
            failed.append(1)
            raise OSError("no more processes")
        return real(register)

    monkeypatch.setattr(rule_helper, "_spawn", first_fails)
    monkeypatch.setattr(rule_helper, "_RETRY_AFTER_S", 0.2)
    rule_helper.start(2)
    pool = rule_helper._POOL
    for starter in list(pool._starters):
        starter.join(30)
    assert failed and len(pool.pids()) == 1
    time.sleep(0.3)
    pool.release(pool.acquire())              # the next walk tops the pool up
    assert pool.wait_ready(30)
    assert len(pool.pids()) == 2


def test_a_helper_that_never_reports_ready_fails_the_start_in_time(monkeypatch):
    monkeypatch.setattr(rule_helper, "_BOOT", "import time; time.sleep(60)")
    monkeypatch.setattr(rule_helper, "_START_TIMEOUT_S", 1.0)
    procs: list = []
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="did not answer within 1 s"):
        rule_helper._spawn(procs.append)
    assert time.monotonic() - t0 < 5
    assert len(procs) == 1 and procs[0].poll() is not None      # killed and reaped


def test_a_helper_that_exits_while_starting_fails_the_start_at_once(monkeypatch):
    # It reads what the parent writes first, so the parent's write never
    # meets a closed pipe.
    monkeypatch.setattr(rule_helper, "_BOOT",
                        "import sys; sys.stdin.buffer.read(); raise SystemExit(7)")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 7"):
        rule_helper._spawn(lambda proc: None)
    assert time.monotonic() - t0 < 5


def test_a_module_in_the_working_directory_does_not_load_in_a_helper(tmp_path, monkeypatch):
    # A module every helper imports, as a file in the directory the server
    # was started from.
    (tmp_path / "pickle.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rule_helper.start(1)
    try:
        assert rule_helper._POOL.wait_ready(30)
    finally:
        rule_helper.shutdown()                # before tmp_path stops being a cwd


def test_a_starter_thread_that_cannot_start_leaves_the_walks_in_process(monkeypatch, caplog):
    real = threading.Thread.start

    def no_thread(self):
        if self.name == "rule-helper-start":
            raise RuntimeError("can't start new thread")
        real(self)

    _install_rows(monkeypatch, [_row("card", "Card · x", "x", "y", 1)])
    rule_helper.start(1)
    pool = rule_helper._POOL
    assert pool.wait_ready(30)
    for proc in list(pool._procs):
        proc.kill()
    monkeypatch.setattr(threading.Thread, "start", no_thread)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        assert pl_engine._postprocess_text("x") == "y"        # the replacement's start fails
        assert pool._alive == 0
        t0 = time.monotonic()
        assert pl_engine._postprocess_text("x x") == "y y"    # and no walk waits for it
        assert time.monotonic() - t0 < 5
    assert caplog.text.count("could not be started") == 1
    rule_helper.shutdown()                                    # joins no unstarted thread

    rule_helper.start(2)                                      # at startup: returns
    assert rule_helper._POOL._alive == 0
    assert rule_helper._POOL.acquire() is None


# --- the switch --------------------------------------------------------------------

@pytest.mark.parametrize("on", [True, False])
def test_the_lifespan_starts_and_stops_the_pool_only_when_the_setting_is_on(
        app_module, monkeypatch, on):
    from starlette.testclient import TestClient

    events: list = []
    monkeypatch.setattr(rule_helper, "start", lambda *a, **k: events.append("start"))
    monkeypatch.setattr(rule_helper, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(app_module.cfg, "PIPELINE_RULE_ISOLATION", on, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        assert events == (["start"] if on else [])
        events.clear()
    assert events == ["shutdown"]


def test_the_setting_ships_on_and_needs_a_restart():
    from faster_whisper_backend.settings import schema as settings_schema

    assert json.loads(read_repo_text("config.json"))["PIPELINE_RULE_ISOLATION"] is True
    assert "PIPELINE_RULE_ISOLATION" in settings_schema.RESTART_REQUIRED_FIELDS
