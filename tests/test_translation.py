"""Unit tests for translation.py (llama.cpp T2T module).

Everything runs WITHOUT llama_cpp installed: the module lazy-imports it only
inside _load_blocking, and these tests stub either the module-level
``_complete`` seam (one function = the single llama call) or
``_load_blocking`` itself. Async entry points are driven with asyncio.run()
(no pytest-asyncio), matching the streaming tests.
"""

import asyncio
import os
import re
import sys
import threading
import time

import pytest

from faster_whisper_backend import config as cfg
from faster_whisper_backend.audio import translation


# ---------------------------------------------------------------------------
# Import purity — CI has no llama_cpp; importing the module must not pull it.
# ---------------------------------------------------------------------------

def test_module_imports_without_llama_cpp():
    assert "llama_cpp" not in sys.modules


# ---------------------------------------------------------------------------
# Model-ref parsing / family detection
# ---------------------------------------------------------------------------

def test_parse_model_ref_splits_on_last_colon():
    assert translation._parse_model_ref("org/repo") == ("org/repo", None)
    assert translation._parse_model_ref("org/repo:Q4_K_M") == \
        ("org/repo", "Q4_K_M")
    # LAST colon wins (defensive — the config pattern forbids extra colons).
    assert translation._parse_model_ref("a:b:c") == ("a:b", "c")


@pytest.mark.parametrize("ref,family", [
    ("tencent/HY-MT1.5-7B-GGUF:Q4_K_M", "hunyuan"),
    ("bartowski/HY-MT-1.5-GGUF", "hunyuan"),
    ("google/translategemma-9b-GGUF", "gemma-translate"),
    ("mradermacher/MiLMMT-8B-GGUF", "milmmt"),
    ("ByteDance-Seed/Seed-X-PPO-7B-GGUF", "seedx"),
    ("org/SeedX-quant", "seedx"),
    ("org/some-generic-model", "chatml"),
])
def test_detect_family(ref, family):
    assert translation.detect_family(ref) == family


def test_resolve_family_honors_config_override(monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_FAMILY", "milmmt",
                        raising=False)
    assert translation.resolve_family("x/hunyuan-mt") == "milmmt"
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_FAMILY", "auto",
                        raising=False)
    assert translation.resolve_family("x/hunyuan-mt") == "hunyuan"


def test_list_languages_en_first_sorted():
    langs = translation.list_languages()
    assert langs[0] == "en"
    assert langs[1:] == sorted(langs[1:])
    assert "de" in langs and len(langs) == len(set(langs))


# ---------------------------------------------------------------------------
# Prompt rendering per family
# ---------------------------------------------------------------------------

def _build(family, text, *, source="de", target="en", context="", glossary=""):
    fam = translation._FAMILIES[family]
    return fam.build(text, source, translation._lang_name(source), target,
                     translation._lang_name(target), context, glossary)


def test_milmmt_raw_prompt_uses_english_names():
    prompt = _build("milmmt", "Hallo Welt")
    assert isinstance(prompt, str)                      # raw completion
    assert prompt == ("Translate this from German to English:\n"
                      "German: Hallo Welt\nEnglish:")


def test_seedx_raw_prompt_shape():
    prompt = _build("seedx", "Hallo", target="fr")
    assert isinstance(prompt, str)
    assert prompt == ("Translate the following German sentence into "
                      "French:\nHallo <fr>")


def test_gemma_structured_turn_ignores_context_and_glossary():
    msgs = _build("gemma-translate", "Hallo", context="ctx",
                  glossary="Herz = heart")
    assert msgs == [{"role": "user", "content":
                     "type:text,source_lang_code:de,target_lang_code:en,"
                     "text:Hallo"}]


def test_hunyuan_prompt_plain_context_and_glossary():
    # No-context branch: the official English XX<=>XX prompt, unchanged.
    plain = _build("hunyuan", "Hallo")[0]["content"]
    assert plain == ("Translate the following segment into English, "
                     "without additional explanation.\n\nHallo")
    # Contextual branch: Tencent's official Chinese-instruction template —
    # context above the instruction, the explicit "do not translate the
    # preceding text" clause, payload on its own line after the fullwidth
    # colon.
    with_ctx = _build("hunyuan", "Hallo", context="A: Guten Tag")[0]["content"]
    assert with_ctx == (
        "A: Guten Tag\n"
        "参考上面的信息，把下面的文本翻译成英语，"
        "注意不需要翻译上文，也不要额外解释：\n"
        "Hallo")
    assert with_ctx.index("A: Guten Tag") < with_ctx.index("参考上面的信息")
    # Glossary (official terminology-intervention style) comes FIRST in the
    # contextual template; malformed lines are skipped.
    both = _build("hunyuan", "Hallo", context="A: Guten Tag",
                  glossary="Herz = heart\nbogus line")[0]["content"]
    assert both.startswith(
        "参考下面的翻译：\nHerz 翻译成 heart\n\nA: Guten Tag\n")
    assert "bogus" not in both
    # Unknown target code: falls back to the English name (mixed-language
    # instruction still works).
    fallback = _build("hunyuan", "Hallo", target="rm",
                      context="A: Guten Tag")[0]["content"]
    assert "把下面的文本翻译成Rm，" in fallback
    # No-context glossary stays in the English prompt style, as today.
    with_gl = _build("hunyuan", "Hallo",
                     glossary="Herz = heart\nbogus line")[0]["content"]
    assert with_gl.startswith("with reference to: Herz -> heart\n")
    assert "bogus" not in with_gl                       # malformed line skipped


def test_chatml_generic_prompt():
    content = _build("chatml", "Hallo")[0]["content"]
    assert content == ("Translate the following text from German into "
                       "English. Reply with ONLY the translation.\n\nHallo")


def test_custom_template_renders_all_slots(monkeypatch):
    monkeypatch.setattr(
        cfg, "TRANSLATION_PROMPT_TEMPLATE",
        "S={source_language} T={target_language} C={context} "
        "G={glossary}\n{text}", raising=False)
    content = _build("custom", "Hallo", context="ctx here",
                     glossary="a = b")[0]["content"]
    assert content == "S=German T=English C=ctx here G=a = b\nHallo"
    # Missing optional slots render as "" (never KeyError / literal braces).
    content = _build("custom", "Hallo", context="", glossary="")[0]["content"]
    assert content == "S=German T=English C= G=\nHallo"


def test_lang_name_fallbacks():
    assert translation._lang_name("de") == "German"
    assert translation._lang_name("fr-CA") == "French"   # region → base name
    assert translation._lang_name("rm") == "Rm"          # title-case fallback
    assert translation._lang_name(None) == ""


# ---------------------------------------------------------------------------
# Numbered-list build/parse
# ---------------------------------------------------------------------------

def test_build_and_parse_numbered_roundtrip():
    texts = ["eins", "zwei", "drei"]
    built = translation._build_numbered(texts)
    assert built == "1. eins\n2. zwei\n3. drei"
    assert translation._parse_numbered(built, 3) == texts


def test_parse_numbered_tolerates_formats_and_noise():
    reply = "Here you go:\n 1) one \n2: two\n3. three\nthanks!"
    assert translation._parse_numbered(reply, 3) == ["one", "two", "three"]


def test_parse_numbered_mismatch_returns_none():
    assert translation._parse_numbered("1. one\n2. two", 3) is None
    assert translation._parse_numbered("free text, no numbers", 2) is None
    assert translation._parse_numbered("", 1) is None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_guard_reasons():
    g = translation._guard_reason
    assert g("Hallo", "") == "empty output"
    # Ratio applies only past the 20-char floor.
    assert g("Hi", "Hello there friend") is None        # short: no ratio check
    long_src = "Dies ist ein deutlich laengerer Satz mit vielen Worten."
    assert "length ratio" in g(long_src, "ok")
    assert "length ratio" in g("Hi", "x" * 40)          # output side hits floor
    # Digit guard: a CHANGED number fails; number-word normalization in
    # either direction is a correct translation and must pass.
    assert g("Nimm 5 Tabletten", "Take 3 pills") == "digit mismatch"
    assert g("Nimm 5 Tabletten", "Take five pills") is None
    assert g("Nimm sechzehn Tabletten", "Take 16 pills") is None
    assert g("Nimm 5 von den 10", "Take 5 of the ten") == "digit mismatch"
    assert g("Hallo Welt", "Hallo Welt") == "output copies input"
    assert g("Ein normaler Satz hier mit Laenge",
             "abcdefghijkl" * 4) == "repetition loop"
    assert g("Hallo Welt", "Hello world") is None


def test_guard_repetition_scan_is_fast_on_large_clean_output():
    """The repetition pattern is superlinear on non-matching text; the guard
    scans only the leading window so a large CLEAN translation cannot stall
    the event loop for seconds."""
    import random
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima"]
    src = " ".join(random.Random(1).choices(words, k=9000))      # ~50 kB
    out = " ".join(random.Random(2).choices(words, k=9000))
    t0 = time.monotonic()
    assert translation._guard_reason(src, out) is None
    assert time.monotonic() - t0 < 5.0


# ---------------------------------------------------------------------------
# Fluent-mode helpers (pure)
# ---------------------------------------------------------------------------

def _segs(*texts):
    return [{"text": t, "speaker": None} for t in texts]


def test_merge_sentences_groups_on_final_punctuation():
    groups = translation._merge_sentences(
        _segs("Hallo Welt.", "Wie geht", "es dir?", "Na", "gut…"))
    assert groups == [[0], [1, 2], [3, 4]]


def test_merge_sentences_caps_group_size():
    groups = translation._merge_sentences(_segs(*(["kein Punkt"] * 8)))
    assert groups == [[0, 1, 2, 3, 4, 5], [6, 7]]


def test_merge_sentences_trailing_open_group():
    assert translation._merge_sentences(_segs("Fertig.", "offen")) == \
        [[0], [1]]


def test_redistribute_proportional_word_boundaries():
    # Equal source shares → cut at the space nearest the middle.
    assert translation._redistribute(["aaaa", "bbbb"],
                                     "one two three four") == \
        ["one two", "three four"]
    # Skewed shares → the cut moves with the source proportion.
    out = translation._redistribute(["aaaaaaaaaaaa", "bbb"],
                                    "first part is long tail")
    assert out == ["first part is long", "tail"]
    assert " ".join(out) == "first part is long tail"


def test_redistribute_edges():
    assert translation._redistribute(["a"], "whole thing") == ["whole thing"]
    assert translation._redistribute(["a", "b"], "") == ["", ""]
    # No spaces at all → raw proportional cut.
    joined = "".join(translation._redistribute(["aa", "bb"], "abcdef"))
    assert joined == "abcdef"


def test_redistribute_never_emits_empty_members():
    """3+ members whose ideal cuts snap to the same space must not yield an
    empty slice (the cue would silently render blank in SRT/VTT/UI)."""
    pieces = translation._redistribute(["Ich", "habe", "gegessen."], "I ate.")
    assert all(pieces)
    assert "".join(pieces) == "Iate."           # nothing lost but the space
    pieces = translation._redistribute(
        ["Hallo", "Welt", "wie", "geht", "es", "dir"], "Hi there")
    assert all(pieces)
    assert "".join(pieces) == "Hithere"
    # Leading cuts reserve one char per remaining member: a 6-char reply over
    # 6 members yields six single chars, never a fat head and blank tail.
    pieces = translation._redistribute(
        ["Hallo", "Welt", "wie", "geht", "es", "dir"], "abcdef")
    assert pieces == list("abcdef")
    # Shorter than the member count is unsplittable: the documented
    # degenerate contract is one piece per member, nothing lost, and the
    # CALLER (translate_segments) reverts the group on the empties.
    pieces = translation._redistribute(
        ["uh", "um", "ja", "ne", "so", "hm"], "ok")
    assert len(pieces) == 6
    assert "".join(pieces) == "ok"
    assert not all(pieces)


# ---------------------------------------------------------------------------
# translate_segments — stubbed through the _complete seam
# ---------------------------------------------------------------------------

def _payload_of(prompt_or_msgs):
    """Recover the {text} payload from a chatml prompt (tests use chatml)."""
    content = prompt_or_msgs[-1]["content"] if isinstance(prompt_or_msgs, list) \
        else prompt_or_msgs
    return content.split("\n\n", 1)[1]


def _stub_get_model(monkeypatch, llm="STUB-LLM"):
    """Install a _get_model stub tolerant of the real call shape
    (lease=..., download_cb=...)."""
    async def _fake(ref, **kwargs):
        return llm
    monkeypatch.setattr(translation, "_get_model", _fake)


def _install_fake(monkeypatch, transform, calls=None):
    """Stub translation._complete: applies `transform` per item, handling the
    numbered-list batch payloads transparently."""
    def fake(llm, family, prompt_or_msgs, max_tokens):
        payload = _payload_of(prompt_or_msgs)
        if calls is not None:
            calls.append(payload)
        if payload.startswith(translation._NUMBERED_INSTRUCTION):
            body = payload.split("\n\n", 1)[1]
            out = []
            for line in body.splitlines():
                m = re.match(r"(\d+)\.\s(.*)", line)
                out.append(f"{m.group(1)}. {transform(m.group(2))}")
            return "\n".join(out)
        return transform(payload)
    monkeypatch.setattr(translation, "_complete", fake)
    _stub_get_model(monkeypatch)


def _xlate(text):
    """A well-behaved pseudo-translation: differs from the input, keeps
    digits and roughly the length (guards all pass)."""
    return text.swapcase()


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def base_cfg(monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "org/model",
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_FAMILY", "auto",
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 8, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_CONTEXT_SEGMENTS", 0, raising=False)
    return monkeypatch


def test_no_model_configured_raises(base_cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "", raising=False)
    with pytest.raises(translation.TranslationError, match="no translation model"):
        _run(translation.translate_segments(_segs("hi"), ["de"]))


def test_same_language_short_circuit_no_model_calls(base_cfg, monkeypatch):
    calls = []
    _install_fake(monkeypatch, _xlate, calls)

    async def boom(ref, **kwargs):
        raise AssertionError("model must not load for a same-language target")
    monkeypatch.setattr(translation, "_get_model", boom)

    res, warns, meta = _run(translation.translate_segments(
        _segs("Hallo Welt.", "Noch was."), ["de", "de-CH"],
        source_lang="de", mode="faithful"))
    assert calls == []
    assert res == [{"de": "Hallo Welt.", "de-CH": "Hallo Welt."},
                   {"de": "Noch was.", "de-CH": "Noch was."}]
    assert warns == []
    # The verbatim copy is a legitimate translation — never flagged as kept.
    assert meta == {"model": "org/model", "source": "de", "mode": "faithful",
                    "kept": {}}


def test_faithful_batches_as_numbered_list(base_cfg, monkeypatch):
    calls = []
    _install_fake(monkeypatch, _xlate, calls)
    segs = _segs("Erstens 1.", "Zweitens.", "Drittens.")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="faithful"))
    assert warns == []
    assert [r["en"] for r in res] == \
        ["eRSTENS 1.", "zWEITENS.", "dRITTENS."]
    # One batched call, numbered payload.
    assert len(calls) == 1
    assert calls[0].startswith(translation._NUMBERED_INSTRUCTION)
    assert "1. Erstens 1." in calls[0] and "3. Drittens." in calls[0]


def test_faithful_mismatch_halves_batch_down_to_success(base_cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 4, raising=False)
    calls = []

    def fake(llm, family, prompt_or_msgs, max_tokens):
        payload = _payload_of(prompt_or_msgs)
        calls.append(payload)
        if payload.startswith(translation._NUMBERED_INSTRUCTION):
            body = payload.split("\n\n", 1)[1]
            lines = body.splitlines()
            if len(lines) > 2:
                return "sorry, here is a poem instead"   # count mismatch
            out = []
            for line in lines:
                m = re.match(r"(\d+)\.\s(.*)", line)
                out.append(f"{m.group(1)}. {_xlate(m.group(2))}")
            return "\n".join(out)
        return _xlate(payload)
    monkeypatch.setattr(translation, "_complete", fake)

    _stub_get_model(monkeypatch)

    segs = _segs("Eins zwei.", "Drei vier.", "Fuenf sechs.", "Sieben acht.")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="faithful"))
    assert warns == []
    assert [r["en"] for r in res] == [
        "eINS ZWEI.", "dREI VIER.", "fUENF SECHS.", "sIEBEN ACHT."]
    # First attempt with 4 items failed → halved to 2 → two good batches.
    assert len(calls) == 3
    assert "4. Sieben acht." in calls[0]


_GUARD_SRC = "Nimm 5 Tabletten und ruf mich morgen frueh wieder an"
_GUARD_SRC_NO_DIGITS = "Nimm die Tabletten und ruf mich morgen wieder an"


@pytest.mark.parametrize("src,bad_out,reason", [
    (_GUARD_SRC,
     lambda t: "Take 3 pills now and call me again in the morning",
     "digit mismatch"),
    (_GUARD_SRC, lambda t: t, "output copies input"),
    (_GUARD_SRC, lambda t: "ok", "length ratio"),
    (_GUARD_SRC_NO_DIGITS, lambda t: "abcdefghijkl" * 4, "repetition loop"),
])
def test_guards_keep_original_after_one_retry(base_cfg, monkeypatch,
                                              src, bad_out, reason):
    calls = []
    _install_fake(monkeypatch, bad_out, calls)
    res, warns, meta = _run(translation.translate_segments(
        _segs(src), ["en"], source_lang="de", mode="faithful"))
    assert res == [{"en": src}]                          # original kept
    # Greedy family (chatml) + no context on the first attempt: the retry
    # would re-send a bit-identical prompt at temperature 0 — skipped.
    assert len(calls) == 1
    assert len(warns) == 1
    assert "segment 1 (en): kept original" in warns[0]
    assert reason in warns[0]
    assert meta["kept"] == {0: ["en"]}


def _install_hunyuan_model(monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL",
                        "tencent/HY-MT1.5-7B-GGUF:Q4", raising=False)

    _stub_get_model(monkeypatch)


def test_retry_drops_context(base_cfg, monkeypatch):
    """A guard failure retries WITHOUT context: first attempt carries the
    context lines, the retry prompt does not (context echo is the dominant
    failure mode). max_tokens is sized for the context on the first attempt."""
    monkeypatch.setattr(cfg, "TRANSLATION_CONTEXT_SEGMENTS", 1, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 1, raising=False)
    _install_hunyuan_model(monkeypatch)
    prompts = []

    def fake(llm, family, msgs, max_tokens):
        content = msgs[0]["content"]
        prompts.append((content, max_tokens))
        if "参考上面的信息" in content:
            return "x" * 400          # simulated context echo → ratio guard
        return "A GOOD ENOUGH TRANSLATION HERE"
    monkeypatch.setattr(translation, "_complete", fake)

    segs = _segs("Hallo Welt du schoene.", "Wie geht es dir denn heute?")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="faithful"))
    assert warns == []
    assert res[1]["en"] == "A GOOD ENOUGH TRANSLATION HERE"
    # seg 1 (no context) + seg 2 first attempt (context) + seg 2 retry.
    assert len(prompts) == 3
    assert "Hallo Welt du schoene." in prompts[1][0]     # context present
    assert "参考上面的信息" in prompts[1][0]
    assert "Hallo Welt" not in prompts[2][0]             # retry: context-free
    assert "参考上面的信息" not in prompts[2][0]
    assert prompts[1][1] > prompts[2][1]                 # max_tokens saw ctx


def test_hunyuan_no_context_still_rerolls(base_cfg, monkeypatch):
    """Sampled family (hunyuan, temp 0.7): even a context-free first attempt
    gets ONE re-roll before keeping the original."""
    _install_hunyuan_model(monkeypatch)
    calls = []

    def fake(llm, family, msgs, max_tokens):
        calls.append(msgs[0]["content"])
        return ""                     # empty output → guard fails every time
    monkeypatch.setattr(translation, "_complete", fake)

    res, warns, _ = _run(translation.translate_segments(
        _segs("Hallo Welt"), ["en"], source_lang="de", mode="faithful"))
    assert res == [{"en": "Hallo Welt"}]
    assert len(calls) == 2                               # initial + ONE retry
    assert calls[0] == calls[1]      # same (context-free) prompt, re-rolled
    assert len(warns) == 1 and "kept original" in warns[0]


def test_batch_guard_failure_retries_that_segment_alone(base_cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 2, raising=False)
    calls = []

    def per_item(t):
        # Sabotage only the second segment inside the batch; the standalone
        # retry (payload without numbering) then succeeds.
        if t == "Zweitens kaputt.":
            return t                                     # copy → guard fails
        return _xlate(t)
    _install_fake(monkeypatch, per_item, calls)

    # The standalone retry sends the bare text, which per_item would sabotage
    # again — patch on top so the retry path is distinguishable and succeeds.
    inner = translation._complete

    def fake(llm, family, prompt_or_msgs, max_tokens):
        payload = _payload_of(prompt_or_msgs)
        if payload == "Zweitens kaputt.":
            calls.append(payload)
            return "sECOND FIXED."
        return inner(llm, family, prompt_or_msgs, max_tokens)
    monkeypatch.setattr(translation, "_complete", fake)

    res, warns, _ = _run(translation.translate_segments(
        _segs("Erstens gut.", "Zweitens kaputt."), ["en"],
        source_lang="de", mode="faithful"))
    assert warns == []
    assert res[0]["en"] == "eRSTENS GUT."
    assert res[1]["en"] == "sECOND FIXED."
    assert calls[-1] == "Zweitens kaputt."               # retried alone


def test_batch_kept_original_recorded_per_segment(base_cfg, monkeypatch):
    """A batch member whose standalone retry ALSO fails lands in meta['kept']
    under its segment index; the clean member does not."""
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 2, raising=False)

    def per_item(t):
        if t == "Zweitens kaputt.":
            return t                                     # copy → guard fails
        return _xlate(t)
    _install_fake(monkeypatch, per_item)

    res, warns, meta = _run(translation.translate_segments(
        _segs("Erstens gut.", "Zweitens kaputt."), ["en"],
        source_lang="de", mode="faithful"))
    assert res[0]["en"] == "eRSTENS GUT."
    assert res[1]["en"] == "Zweitens kaputt."            # source kept
    assert meta["kept"] == {1: ["en"]}
    assert len(warns) == 1 and "segment 2 (en): kept original" in warns[0]


def test_fluent_merges_translates_and_redistributes(base_cfg, monkeypatch):
    calls = []

    def fake_group(t):
        assert t == "Wie geht es dir?" or t == "Hallo Welt."
        return {"Hallo Welt.": "Hello world.",
                "Wie geht es dir?": "How are you doing?"}[t]
    _install_fake(monkeypatch, fake_group, calls)

    segs = _segs("Hallo Welt.", "Wie geht", "es dir?")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="fluent"))
    assert warns == []
    assert res[0]["en"] == "Hello world."
    # Group ["Wie geht", "es dir?"] redistributed across both members at a
    # word boundary, nothing lost.
    assert res[1]["en"] and res[2]["en"]
    assert f'{res[1]["en"]} {res[2]["en"]}' == "How are you doing?"
    assert len(calls) == 2                               # one call per group


def test_fluent_short_reply_reverts_group_instead_of_blank_cues(
        base_cfg, monkeypatch):
    """A fluent reply shorter than the group's member count cannot be
    redistributed without blank cues — the group reverts to its originals
    and every member lands in meta['kept'], exactly like a guard hit."""
    _install_fake(monkeypatch, lambda t: "ok")           # 2 chars, 6 members
    segs = _segs("uh", "um", "ja", "ne", "so", "hm")
    res, warns, meta = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="fluent"))
    assert all(r["en"] for r in res)
    assert [r["en"] for r in res] == [s["text"] for s in segs]
    assert len(warns) == 1 and "redistribution left empty cues" in warns[0]
    assert meta["kept"] == {i: ["en"] for i in range(6)}


def test_fluent_group_failure_keeps_member_originals(base_cfg, monkeypatch):
    _install_fake(monkeypatch, lambda t: t)              # always a copy → fail
    segs = _segs("Hallo Welt und mehr Text", "es dir heute wirklich?")
    res, warns, meta = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="fluent"))
    assert res[0]["en"] == segs[0]["text"]
    assert res[1]["en"] == segs[1]["text"]
    assert len(warns) == 1 and "kept original" in warns[0]
    assert "segments 1-2" in warns[0]
    # Whole-group revert flags EVERY member as kept-original.
    assert meta["kept"] == {0: ["en"], 1: ["en"]}


def test_progress_and_cancel(base_cfg, monkeypatch):
    _install_fake(monkeypatch, _xlate)
    fractions = []
    steps = []
    res, _, _ = _run(translation.translate_segments(
        _segs("Eins.", "Zwei."), ["en", "fr"], source_lang="de",
        mode="fluent",
        progress_cb=lambda f, s, t=None: (fractions.append(f),
                                          steps.append(s))))
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert steps[0].startswith("en ") and steps[-1].startswith("fr ")

    with pytest.raises(translation.TranslationCancelled):
        _run(translation.translate_segments(
            _segs("Eins."), ["en"], source_lang="de",
            cancel_check=lambda: True))


def test_progress_carries_last_text_tail(base_cfg, monkeypatch):
    """Three-arg callbacks get the last completed translation's tail."""
    _install_fake(monkeypatch, _xlate)
    tails = []
    _run(translation.translate_segments(
        _segs("Eins.", "Zwei."), ["en"], source_lang="de", mode="fluent",
        progress_cb=lambda f, s=None, last_text=None: tails.append(last_text)))
    assert any(t for t in tails)                       # a tail arrived
    assert all(t is None or len(t) <= 160 for t in tails)
    # The tail is the (pseudo-)translated text — swapcased, not the source.
    assert any(t and "WEI" in t for t in tails)


def test_progress_callback_raising_typeerror_fires_once(base_cfg, monkeypatch):
    """A TypeError raised INSIDE the callback must be swallowed, not
    misread as a wrong arity and answered with a ghost second invocation
    (which would re-run the callback's side effects)."""
    _install_fake(monkeypatch, _xlate)
    calls = []

    def cb(f, s, t=None):
        calls.append((f, s))
        raise TypeError("internal bug")

    res, warns, _ = _run(translation.translate_segments(
        _segs("Eins."), ["en"], source_lang="de", mode="fluent",
        progress_cb=cb))
    assert warns == []
    assert res[0]["en"]
    assert len(calls) == 1


def test_unknown_source_is_not_asserted_as_english(base_cfg, monkeypatch):
    """source_lang omitted: the prompt must not claim the input is English —
    the source clause is dropped instead."""
    prompts = []

    def fake(llm, family, msgs, max_tokens):
        prompts.append(msgs[0]["content"])
        return "Bonjour le monde ici"
    monkeypatch.setattr(translation, "_complete", fake)
    _stub_get_model(monkeypatch)

    res, warns, _ = _run(translation.translate_segments(
        _segs("Hallo Welt hier"), ["fr"], source_lang=None, mode="faithful"))
    assert warns == []
    assert res[0]["fr"] == "Bonjour le monde ici"
    assert "English" not in prompts[0]
    assert "French" in prompts[0]


def test_greedy_context_blind_family_skips_context_retry(base_cfg,
                                                         monkeypatch):
    """chatml's builder ignores `context`, so at temperature 0 the
    context-free retry would re-send a bit-identical prompt — skipped even
    when the first attempt carried context (hunyuan, sampled, still
    re-rolls: see test_retry_drops_context)."""
    monkeypatch.setattr(cfg, "TRANSLATION_CONTEXT_SEGMENTS", 1, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 1, raising=False)
    calls = []
    _install_fake(monkeypatch, lambda t: t, calls)       # copy → guard fails
    segs = _segs("Hallo Welt und mehr", "Wie geht es dir denn?")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="faithful"))
    assert [r["en"] for r in res] == [s["text"] for s in segs]
    assert len(calls) == 2       # one per segment — NO context-free retry
    assert len(warns) == 2


# ---------------------------------------------------------------------------
# GGUF pre-download (progress-visible fetch) + from_pretrained fallback
# ---------------------------------------------------------------------------

def _install_fake_llama(monkeypatch, record):
    import types

    mod = types.ModuleType("llama_cpp")

    class Llama:
        def __init__(self, *, model_path=None, **kw):
            record.append(("direct", model_path, kw))

        @classmethod
        def from_pretrained(cls, **kw):
            record.append(("from_pretrained", kw))
            return "LLM-FP"

    mod.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)


def test_load_uses_predownloaded_path(monkeypatch):
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(translation, "_predownload_gguf",
                        lambda repo, quant, cb=None: "/models/x.Q4.gguf")
    translation._load_blocking_inner("org/repo:Q4_K_M", "cpu", "chatml")
    assert record[0][0] == "direct"
    assert record[0][1] == "/models/x.Q4.gguf"
    assert record[0][2]["n_gpu_layers"] == 0
    assert record[0][2]["n_ctx"] == translation._ctx_for("chatml")


def test_load_falls_back_when_predownload_raises(monkeypatch, caplog):
    import logging as _logging
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)

    def boom(repo, quant, cb=None):
        raise RuntimeError("listing failed")
    monkeypatch.setattr(translation, "_predownload_gguf", boom)
    with caplog.at_level(_logging.WARNING, logger="whisper-server"):
        out = translation._load_blocking_inner("org/repo:Q4", "cuda", "chatml")
    assert out == "LLM-FP"
    assert record[0][0] == "from_pretrained"
    assert record[0][1]["repo_id"] == "org/repo"
    assert record[0][1]["filename"] == "*Q4.gguf"
    assert record[0][1]["n_gpu_layers"] == -1          # cuda: full offload
    assert record[0][1]["n_ctx"] == translation._ctx_for("chatml")
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_load_n_ctx_follows_family(monkeypatch):
    """n_ctx is the FAMILY's window, not one module constant."""
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(translation, "_predownload_gguf",
                        lambda repo, quant, cb=None: "/models/h.gguf")
    translation._load_blocking_inner("org/hy", "cpu", "hunyuan")
    assert record[0][2]["n_ctx"] == translation._ctx_for("hunyuan")
    assert translation._ctx_for("hunyuan") != translation._ctx_for("chatml")


def test_load_falls_back_when_no_unique_match(monkeypatch):
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(translation, "_predownload_gguf",
                        lambda repo, quant, cb=None: None)
    translation._load_blocking_inner("org/repo", "cpu", "chatml")
    assert record[0][0] == "from_pretrained"
    assert record[0][1]["filename"] == "*.gguf"


def test_local_files_only_skips_predownload(monkeypatch):
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)

    def never(repo, quant, cb=None):
        raise AssertionError("pre-download must not run offline")
    monkeypatch.setattr(translation, "_predownload_gguf", never)
    translation._load_blocking_inner("org/repo:Q4", "cpu", "chatml")
    assert record[0][0] == "from_pretrained"
    # The hub freezes HF_HUB_OFFLINE at import, so the env var alone cannot
    # keep the load off the network — the flag must reach the call itself.
    assert record[0][1]["local_files_only"] is True


def test_online_load_passes_local_files_only_false(monkeypatch):
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(translation, "_predownload_gguf",
                        lambda repo, quant, cb=None: None)
    translation._load_blocking_inner("org/repo:Q4", "cpu", "chatml")
    assert record[0][1]["local_files_only"] is False


def test_load_blocking_threads_offline_flag_explicitly(monkeypatch):
    """_load_blocking hands its ONE snapshot of LOCAL_FILES_ONLY down as the
    explicit ``offline`` argument (not just via the env var)."""
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen = {}

    def inner(ref, device, family, download_cb=None, offline=None):
        seen["offline"] = offline
        return "LLM"
    monkeypatch.setattr(translation, "_load_blocking_inner", inner)
    assert translation._load_blocking("o/r", "cpu", "chatml") == "LLM"
    assert seen["offline"] is True


# ---------------------------------------------------------------------------
# HF cache dir is passed EXPLICITLY (the hub freezes HF_HOME at import)
# ---------------------------------------------------------------------------

def _install_fake_hub(monkeypatch, record, files=("m.Q4.gguf",)):
    import types

    class _Sib:
        def __init__(self, name):
            self.rfilename = name

    class _Info:
        siblings = [_Sib(f) for f in files]

    class HfApi:
        def model_info(self, repo):
            return _Info()

    def hf_hub_download(**kw):
        record.append(kw)
        return "/cache/m.Q4.gguf"

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = HfApi
    mod.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_predownload_passes_download_root_cache_dir(monkeypatch, tmp_path):
    record = []
    _install_fake_hub(monkeypatch, record)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    out = translation._predownload_gguf("org/repo", "Q4")
    assert out == "/cache/m.Q4.gguf"
    assert record[0]["repo_id"] == "org/repo"
    assert record[0]["filename"] == "m.Q4.gguf"
    assert record[0]["cache_dir"] == str(tmp_path / "hf" / "hub")


def test_predownload_set_hf_home_wins_over_download_root(monkeypatch,
                                                         tmp_path):
    record = []
    _install_fake_hub(monkeypatch, record)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom"))
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path / "root"),
                        raising=False)
    translation._predownload_gguf("org/repo", "Q4")
    assert record[0]["cache_dir"] == str(tmp_path / "custom" / "hub")


def _install_fake_progress(monkeypatch):
    """Stub download_progress.capture + the jobs registry around the real
    _predownload_gguf; returns the recorder (the capture cb lands in
    rec["cb"])."""
    import contextlib

    from faster_whisper_backend.runtime import download_progress
    from faster_whisper_backend.core import jobs

    rec = {"cb": None, "jobs": []}

    @contextlib.contextmanager
    def capture(label, cb=None, **kw):
        rec["cb"] = cb
        rec["label"] = label
        yield type("Cap", (), {"tqdm_kwargs": {}})()
    monkeypatch.setattr(download_progress, "capture", capture)
    monkeypatch.setattr(jobs, "job_start",
                        lambda kind, **kw: rec["jobs"].append(
                            ("start", kind, kw)) or "JOB-1")
    monkeypatch.setattr(jobs, "job_update",
                        lambda job_id, **kw: rec["jobs"].append(
                            ("update", job_id, kw)))
    monkeypatch.setattr(jobs, "job_end",
                        lambda job_id: rec["jobs"].append(("end", job_id)))
    return rec


def test_predownload_matches_quant_case_insensitively(monkeypatch):
    record = []
    _install_fake_hub(monkeypatch, record,
                      files=("a.Q4_K_M.GGUF", "README.md", "b.Q8_0.gguf"))
    rec = _install_fake_progress(monkeypatch)
    out = translation._predownload_gguf("org/repo", "q4_k_m")
    assert out == "/cache/m.Q4.gguf"
    assert record[0]["filename"] == "a.Q4_K_M.GGUF"
    assert rec["jobs"][0] == ("start", "download", {
        "model": "gguf:org/repo:q4_k_m", "detail": "a.Q4_K_M.GGUF"})
    assert rec["jobs"][-1] == ("end", "JOB-1")


def test_predownload_ambiguous_listing_returns_none(monkeypatch):
    record = []
    _install_fake_hub(monkeypatch, record,
                      files=("m-00001.Q4.gguf", "m-00002.Q4.gguf"))
    rec = _install_fake_progress(monkeypatch)
    assert translation._predownload_gguf("org/repo", "Q4") is None
    assert record == [] and rec["jobs"] == []      # from_pretrained speaks


def test_predownload_no_quant_matches_any_gguf(monkeypatch):
    record = []
    _install_fake_hub(monkeypatch, record, files=("only.GGUF", "cfg.json"))
    rec = _install_fake_progress(monkeypatch)
    assert translation._predownload_gguf("org/repo", None) == \
        "/cache/m.Q4.gguf"
    assert record[0]["filename"] == "only.GGUF"
    assert rec["label"] == "gguf:org/repo"


def test_predownload_hook_updates_job_and_shields_download_cb(monkeypatch):
    record = []
    _install_fake_hub(monkeypatch, record)
    rec = _install_fake_progress(monkeypatch)
    seen = []
    translation._predownload_gguf("org/repo", "Q4",
                                  lambda d, t: seen.append((d, t)))
    hook = rec["cb"]
    hook(5, 10)
    assert ("update", "JOB-1", {"progress": 0.5, "total_bytes": 10}) \
        in rec["jobs"]
    assert seen == [(5, 10)]
    hook(5, 0)                                    # unknown total: no ratio
    assert rec["jobs"][-1] == ("update", "JOB-1",
                               {"progress": None, "total_bytes": None})

    def raising_cb(d, t):
        raise RuntimeError("client went away")
    rec2 = _install_fake_progress(monkeypatch)
    translation._predownload_gguf("org/repo", "Q4", raising_cb)
    rec2["cb"](1, 2)                              # swallowed, job still moves
    assert rec2["jobs"][-1] == ("update", "JOB-1",
                                {"progress": 0.5, "total_bytes": 2})


def test_predownload_ends_job_when_download_raises(monkeypatch):
    _install_fake_hub(monkeypatch, [])
    rec = _install_fake_progress(monkeypatch)

    def boom(**kw):
        raise OSError("disk full")
    sys.modules["huggingface_hub"].hf_hub_download = boom
    with pytest.raises(OSError, match="disk full"):
        translation._predownload_gguf("org/repo", "Q4")
    assert rec["jobs"] == [("start", "download",
                            {"model": "gguf:org/repo:Q4",
                             "detail": "m.Q4.gguf"}),
                           ("end", "JOB-1")]


def test_from_pretrained_fallback_passes_cache_dir(monkeypatch, tmp_path):
    record = []
    _install_fake_llama(monkeypatch, record)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setattr(translation, "_predownload_gguf",
                        lambda repo, quant, cb=None: None)
    translation._load_blocking_inner("org/repo", "cpu", "chatml")
    assert record[0][1]["cache_dir"] == str(tmp_path / "hf" / "hub")


def test_hf_cache_dir_none_without_root_or_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    assert translation._hf_cache_dir() is None


def test_offline_env_restored_when_flag_flips_mid_load(monkeypatch):
    """LOCAL_FILES_ONLY is snapshotted ONCE per load: an admin flipping the
    hot setting during the (minutes-long) load must not leak
    HF_HUB_OFFLINE=1 process-wide."""
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def inner(ref, device, family, download_cb=None, offline=None):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
        return "LLM"
    monkeypatch.setattr(translation, "_load_blocking_inner", inner)
    assert translation._load_blocking("o/r", "cpu", "chatml") == "LLM"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_env_preset_value_restored_after_load(monkeypatch):
    """An operator's process-wide HF_HUB_OFFLINE (here '0') survives one
    offline load — restored, not popped."""
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    def inner(ref, device, family, download_cb=None, offline=None):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return "LLM"
    monkeypatch.setattr(translation, "_load_blocking_inner", inner)
    assert translation._load_blocking("o/r", "cpu", "chatml") == "LLM"
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_online_load_leaves_preset_offline_env_untouched(monkeypatch):
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def inner(ref, device, family, download_cb=None, offline=None):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert offline is False
        return "LLM"
    monkeypatch.setattr(translation, "_load_blocking_inner", inner)
    translation._load_blocking("o/r", "cpu", "chatml")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_offline_env_restored_when_load_raises(monkeypatch):
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    def inner(ref, device, family, download_cb=None, offline=None):
        raise translation.TranslationError("load failed")
    monkeypatch.setattr(translation, "_load_blocking_inner", inner)
    with pytest.raises(translation.TranslationError):
        translation._load_blocking("o/r", "cpu", "chatml")
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_context_lines_prefix_speakers(base_cfg):
    segs = [{"text": "Guten Tag", "speaker": "SPEAKER_00"},
            {"text": "Hallo", "speaker": None},
            {"text": "Weiter", "speaker": "SPEAKER_01"}]
    assert translation._context_lines(segs, 2, 2) == \
        "SPEAKER_00: Guten Tag\nHallo"
    assert translation._context_lines(segs, 0, 2) == ""
    assert translation._context_lines(segs, 2, 0) == ""


# ---------------------------------------------------------------------------
# LRU bookkeeping (monkeypatched _load_blocking; no llama_cpp involved)
# ---------------------------------------------------------------------------

class _FakeLlama:
    def __init__(self, ref):
        self.ref = ref
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture()
def lru_env(monkeypatch):
    made = {}

    def fake_load(ref, device, family, download_cb=None):
        made[ref] = _FakeLlama(ref)
        return made[ref]
    monkeypatch.setattr(translation, "_load_blocking", fake_load)
    monkeypatch.setattr(translation.system_stats, "gpu_mem_used_bytes",
                        lambda: None)
    stats = {"registered": [], "unregistered": [], "touched": []}
    monkeypatch.setattr(translation.system_stats, "register_loaded_model",
                        lambda name, vram, device, kind, load_secs=None:
                        stats["registered"].append((name, device, kind)))
    monkeypatch.setattr(translation.system_stats, "unregister_loaded_model",
                        lambda name: stats["unregistered"].append(name))
    monkeypatch.setattr(translation.system_stats, "touch_loaded_model",
                        lambda name: stats["touched"].append(name))
    monkeypatch.setattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 2,
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEVICE", "cpu", raising=False)
    _clear_lru_state()
    yield made, stats
    _clear_lru_state()


def _clear_lru_state():
    translation._models.clear()
    translation._last_used.clear()
    translation._active.clear()
    translation._params.clear()
    translation._loading.clear()


def test_lru_eviction_bookkeeping(lru_env):
    made, stats = lru_env

    async def run():
        a = await translation._get_model("o/a")
        b = await translation._get_model("o/b")
        # Cache hit refreshes recency (a moves behind b in eviction order).
        assert await translation._get_model("o/a") is a
        c = await translation._get_model("o/c")
        return a, b, c
    a, b, c = asyncio.run(run())

    assert set(translation._models) == {"o/a", "o/c"}    # b LRU-evicted
    assert b.closed and not a.closed and not c.closed
    assert stats["unregistered"] == ["gguf:o/b"]
    assert stats["touched"] == ["gguf:o/a"]              # the one cache hit
    assert [n for n, _, _ in stats["registered"]] == \
        ["gguf:o/a", "gguf:o/b", "gguf:o/c"]
    assert all(kind == "gguf" for _, _, kind in stats["registered"])
    assert set(translation._last_used) == {"o/a", "o/c"}


def test_drop_models_clears_everything(lru_env):
    made, stats = lru_env

    async def run():
        await translation._get_model("o/a")
        await translation._get_model("o/b")
        await translation.drop_models()
    asyncio.run(run())

    assert translation._models == {} and translation._last_used == {}
    assert made["o/a"].closed and made["o/b"].closed
    assert sorted(stats["unregistered"]) == ["gguf:o/a", "gguf:o/b"]


def test_drop_models_keeps_leased_model_resident(lru_env):
    """'Unload all' over a busy model: the leased one stays (closing a
    llama.cpp context under a running decode is a use-after-free), the
    idle one still goes and is unregistered."""
    made, stats = lru_env

    async def run():
        await translation._get_model("o/a", lease=True)
        await translation._get_model("o/b")
        await translation.drop_models()
    asyncio.run(run())

    assert list(translation._models) == ["o/a"]
    assert not made["o/a"].closed and made["o/b"].closed
    assert stats["unregistered"] == ["gguf:o/b"]


_real_sleep = asyncio.sleep


async def _run_evictor_once(monkeypatch):
    """Drive idle_evictor_loop through exactly ONE wake: the first sleep
    returns immediately, the second cancels the loop."""
    calls = {"n": 0}

    async def fake_sleep(secs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError()
        await _real_sleep(0)
    monkeypatch.setattr(translation.asyncio, "sleep", fake_sleep)
    try:
        with pytest.raises(asyncio.CancelledError):
            await translation.idle_evictor_loop()
    finally:
        monkeypatch.setattr(translation.asyncio, "sleep", _real_sleep)
    assert calls["n"] == 2


def test_idle_evictor_honours_warm_lease_then_evicts(lru_env, monkeypatch):
    made, stats = lru_env
    monkeypatch.setattr(cfg, "TRANSLATION_IDLE_TIMEOUT_S", 1, raising=False)
    warm = {"gguf:o/a"}
    monkeypatch.setattr(translation.system_stats, "is_warm",
                        lambda name: name in warm)

    async def run():
        await translation._get_model("o/a")
        await translation._get_model("o/b")
        translation._last_used["o/a"] -= 100
        await _run_evictor_once(monkeypatch)
        # A warm (preload) lease suspends the idle clock.
        assert set(translation._models) == {"o/a", "o/b"}
        warm.clear()
        await _run_evictor_once(monkeypatch)
    asyncio.run(run())
    assert list(translation._models) == ["o/b"]         # o/b: recently used
    assert made["o/a"].closed and not made["o/b"].closed
    assert stats["unregistered"] == ["gguf:o/a"]


def test_idle_evictor_zero_timeout_never_evicts(lru_env, monkeypatch):
    made, stats = lru_env
    monkeypatch.setattr(cfg, "TRANSLATION_IDLE_TIMEOUT_S", 0, raising=False)
    monkeypatch.setattr(translation.system_stats, "is_warm",
                        lambda name: False)

    async def run():
        await translation._get_model("o/a")
        translation._last_used["o/a"] -= 100_000
        await _run_evictor_once(monkeypatch)
    asyncio.run(run())
    assert list(translation._models) == ["o/a"]
    assert not made["o/a"].closed and stats["unregistered"] == []


def test_warm_hit_does_not_block_on_cold_load(lru_env, monkeypatch):
    """A warm cache hit must return while ANOTHER ref's cold load is still
    running — taking the module _lock before the hit check (as this once
    did) parked every warm job behind any multi-GB load."""
    made, stats = lru_env

    async def run():
        warm = await translation._get_model("o/warm")
        started = threading.Event()
        release = threading.Event()

        def slow_load(ref, device, family, download_cb=None):
            started.set()
            assert release.wait(5)
            made[ref] = _FakeLlama(ref)
            return made[ref]
        monkeypatch.setattr(translation, "_load_blocking", slow_load)
        loop = asyncio.get_running_loop()
        cold = asyncio.ensure_future(translation._get_model("o/cold"))
        await loop.run_in_executor(None, started.wait, 5)
        # The warm hit returns while the cold load is still parked.
        assert await asyncio.wait_for(
            translation._get_model("o/warm"), 1) is warm
        release.set()
        assert await cold is made["o/cold"]
    asyncio.run(run())


def _two_overlapping_loads(monkeypatch, made, refs=("o/a", "o/b")):
    """Run _get_model on two refs whose _load_blocking calls overlap in the
    executor (both parked until both have started), then return the models."""
    started = threading.Barrier(2, timeout=5)

    def slow_load(ref, device, family, download_cb=None):
        started.wait()                 # neither finishes before both start
        made[ref] = _FakeLlama(ref)
        return made[ref]
    monkeypatch.setattr(translation, "_load_blocking", slow_load)

    async def run():
        return await asyncio.gather(*(translation._get_model(r)
                                      for r in refs))
    return asyncio.run(run())


def test_concurrent_misses_respect_max_loaded_models(lru_env, monkeypatch):
    """Two misses on DIFFERENT refs both pass the pre-load cap check (the
    load runs outside _lock) — the post-load insert must re-trim, or the
    default cap of 1 silently holds two multi-GB contexts."""
    made, stats = lru_env
    monkeypatch.setattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 1,
                        raising=False)
    _two_overlapping_loads(monkeypatch, made)
    assert len(translation._models) == 1
    kept = next(iter(translation._models))
    evicted = "o/b" if kept == "o/a" else "o/a"
    assert made[evicted].closed
    assert stats["unregistered"] == [translation._STATS_PREFIX + evicted]
    assert not made[kept].closed


def test_overlapping_loads_publish_no_vram_delta(lru_env, monkeypatch):
    """With the cap >= 2 two refs load concurrently; a before/after NVML
    delta straddling the other load belongs to neither, so both register
    vram=None instead of a mis-attributed number."""
    made, stats = lru_env
    reads = iter([1000, 1000, 5000, 5000])
    monkeypatch.setattr(translation.system_stats, "gpu_mem_used_bytes",
                        lambda: next(reads))
    seen = {}
    monkeypatch.setattr(translation.system_stats, "register_loaded_model",
                        lambda name, vram, device, kind, load_secs=None:
                        seen.__setitem__(name, vram))
    _two_overlapping_loads(monkeypatch, made)
    assert seen == {translation._STATS_PREFIX + "o/a": None,
                    translation._STATS_PREFIX + "o/b": None}
    assert translation._loads_in_flight == 0


def test_solo_load_still_publishes_vram_delta(lru_env, monkeypatch):
    made, stats = lru_env
    reads = iter([1000, 5000])
    monkeypatch.setattr(translation.system_stats, "gpu_mem_used_bytes",
                        lambda: next(reads))
    seen = {}
    monkeypatch.setattr(translation.system_stats, "register_loaded_model",
                        lambda name, vram, device, kind, load_secs=None:
                        seen.__setitem__(name, vram))
    asyncio.run(translation._get_model("o/a"))
    assert seen == {translation._STATS_PREFIX + "o/a": 4000}


def test_cancelled_translation_releases_lease(lru_env, monkeypatch):
    """A cancellation mid-inference still releases the model lease — a
    leaked _active entry would pin the model against eviction forever."""
    monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL", "o/a",
                        raising=False)

    async def boom(*a, **k):
        raise asyncio.CancelledError()
    monkeypatch.setattr(translation, "_run_completion", boom)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await translation.translate_segments(
                _segs("Hallo Welt"), ["en"], source_lang="de",
                mode="faithful")
    asyncio.run(run())
    assert translation._active == {}
    assert "o/a" in translation._models      # released, not torn down


def test_device_change_rekeys_model_after_lease_release(lru_env, monkeypatch):
    """An admin TRANSLATION_DEVICE edit landing during a job: the leased
    model keeps serving (never close a llama.cpp context under a running
    decode), and the first unleased _get_model afterwards reloads with the
    new parameters instead of serving the old-device model forever."""
    made, stats = lru_env

    async def run():
        a = await translation._get_model("o/a", lease=True)
        monkeypatch.setattr(cfg, "TRANSLATION_DEVICE", "cuda", raising=False)
        assert await translation._get_model("o/a") is a   # leased: stale ok
        translation._release_model("o/a")
        b = await translation._get_model("o/a")
        return a, b
    a, b = asyncio.run(run())
    assert b is not a
    assert a.closed and not b.closed


def test_resolve_device_auto_follows_model_device(monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_DEVICE", "auto", raising=False)
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cuda", raising=False)
    assert translation._resolve_device() == "cuda"
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cpu", raising=False)
    assert translation._resolve_device() == "cpu"
    monkeypatch.setattr(cfg, "TRANSLATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cuda", raising=False)
    assert translation._resolve_device() == "cpu"


# ---------------------------------------------------------------------------
# template_override (admin template-test path)
# ---------------------------------------------------------------------------

def test_template_override_forces_custom_family_and_renders_it(
        base_cfg, monkeypatch):
    """translate_segments(template_override=...) must render THAT template
    (not cfg.TRANSLATION_PROMPT_TEMPLATE) and force the custom family, even
    for a model whose name detects a different family."""
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_TEMPLATE",
                        "SAVED {text}", raising=False)
    prompts = []

    def fake(llm, family, prompt_or_msgs, max_tokens):
        prompts.append((family, prompt_or_msgs))
        return "Hello world."
    monkeypatch.setattr(translation, "_complete", fake)

    _stub_get_model(monkeypatch)

    res, warns, meta = _run(translation.translate_segments(
        _segs("Hallo Welt."), ["en"], source_lang="de", mode="faithful",
        model_ref="tencent/HY-MT1.5-7B-GGUF:Q4",   # would detect hunyuan
        template_override="OVERRIDE {text} -> {target_language}"))
    assert res == [{"en": "Hello world."}]
    family, msgs = prompts[0]
    assert family == "custom"
    assert msgs == [{"role": "user",
                     "content": "OVERRIDE Hallo Welt. -> English"}]


def test_without_override_custom_family_still_reads_cfg(base_cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_FAMILY", "custom",
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_PROMPT_TEMPLATE",
                        "SAVED {text}", raising=False)
    prompts = []

    def fake(llm, family, prompt_or_msgs, max_tokens):
        prompts.append(prompt_or_msgs)
        return "Hello world."
    monkeypatch.setattr(translation, "_complete", fake)

    _stub_get_model(monkeypatch)

    _run(translation.translate_segments(
        _segs("Hallo Welt."), ["en"], source_lang="de", mode="faithful"))
    assert prompts[0] == [{"role": "user", "content": "SAVED Hallo Welt."}]


# ── review-fix regressions ───────────────────────────────────────────────────

def test_busy_model_is_never_evicted():
    """A leased model declines eviction (closing llama.cpp's context under a
    running decode is a native use-after-free)."""
    class _Closable:
        closed = False
        def close(self):
            self.closed = True
    llm = _Closable()
    translation._models["org/busy:Q4"] = llm
    translation._last_used["org/busy:Q4"] = 0.0
    translation._active["org/busy:Q4"] = 1
    try:
        assert translation._drop_locked("org/busy:Q4") is False
        assert "org/busy:Q4" in translation._models
        assert not llm.closed
        translation._active.pop("org/busy:Q4", None)
        assert translation._drop_locked("org/busy:Q4") is True
        assert llm.closed
    finally:
        translation._models.pop("org/busy:Q4", None)
        translation._last_used.pop("org/busy:Q4", None)
        translation._active.pop("org/busy:Q4", None)


def test_cjk_target_ratio_bounds_admit_compressed_output():
    src = "Wir haben die Messung gestern wiederholt und dokumentiert."
    out = "我们昨天重复了测量。"  # ~0.17x the chars — valid Chinese
    assert translation._guard_reason(src, out, target="zh") is None
    # …and the default band still rejects it for a Latin target.
    assert translation._guard_reason(src, out, target="fr") is not None


def test_cjk_source_ratio_bounds_admit_expanded_output():
    """The mirror branch: a Chinese source renders ~4x longer in English —
    the default 3.0 cap would keep the untranslated Chinese line."""
    src = "我们昨天重复了测量并记录了结果，今天再检查。"      # 22 chars
    out = ("We repeated the measurement yesterday and documented the "
           "result, and we are checking it again today.")     # ~4.5x
    assert translation._guard_reason(src, out, target="en") is None
    too_long = out + " " + out + " " + out[:20]              # >9x
    assert translation._guard_reason(src, too_long, target="en") \
        is not None


def test_both_cjk_keeps_default_ratio_band():
    src = "我们昨天重复了测量并记录了结果，今天再检查一次。"
    out = "昨日測定。"                          # ~0.2x: rejected zh->ja
    assert translation._guard_reason(src, out, target="ja") is not None


def test_ratio_bounds_three_branches():
    latin = "Wir haben die Messung gestern wiederholt."
    cjk = "我们昨天重复了测量并记录了结果。"
    assert translation._ratio_bounds(latin, "zh") == (0.12, 3.0)
    assert translation._ratio_bounds(cjk, "en") == (0.4, 9.0)
    assert translation._ratio_bounds(cjk, "ja-JP") == (0.4, 3.0)
    assert translation._ratio_bounds(latin, "fr") == (0.4, 3.0)
    assert translation._ratio_bounds(latin, None) == (0.4, 3.0)


def test_custom_template_is_single_pass():
    """Placeholder tokens INSIDE the transcript text must not be
    re-substituted by later slots."""
    rendered = translation._render_custom_template(
        "{text} -> {target_language}",
        text="say {glossary} verbatim",
        source_name="German", target_name="English",
        context="CTX", glossary="Rechnung = invoice")
    assert rendered == "say {glossary} verbatim -> English"


# ---------------------------------------------------------------------------
# render_prompt (admin prompt-lab preview — no model, no llama import)
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_hunyuan_chat_messages(self):
        p = translation.render_prompt(
            "Hallo Welt", "en", source="de",
            model_ref="tencent/HY-MT1.5-7B-GGUF:Q4_K_M")
        assert p["family"] == "hunyuan" and p["chat"] is True
        assert p["messages"][0]["role"] == "user"
        assert "Hallo Welt" in p["messages"][0]["content"]
        assert "English" in p["messages"][0]["content"]
        assert p["model_loaded"] is False
        assert p["sampling"]["temperature"] == 0.7

    def test_raw_family_returns_text(self):
        p = translation.render_prompt(
            "Hallo", "fr", source="de", family="milmmt")
        assert p["chat"] is False and "messages" not in p
        assert p["text"].endswith("French:")

    def test_family_override_beats_model_detection(self):
        p = translation.render_prompt(
            "x", "en", model_ref="tencent/HY-MT1.5-7B-GGUF:Q4_K_M",
            family="seedx")
        assert p["family"] == "seedx"

    def test_template_forces_custom(self):
        p = translation.render_prompt(
            "Hallo", "en", template="T {text} -> {target_language}")
        assert p["family"] == "custom"
        assert p["messages"][0]["content"] == "T Hallo -> English"

    def test_glossary_reaches_hunyuan_prompt(self):
        p = translation.render_prompt(
            "Die Messung", "en", family="hunyuan",
            glossary="Messung = measurement")
        assert "Messung -> measurement" in p["messages"][0]["content"]

    def test_no_llama_import(self):
        translation.render_prompt("x", "en", family="chatml")
        assert "llama_cpp" not in sys.modules

    def test_unknown_source_omits_from_clause(self):
        # The preview must match what translate_segments really sends: an
        # unknown source is left out, never asserted to be English.
        p = translation.render_prompt("Hallo", "fr", family="chatml")
        content = p["messages"][0]["content"]
        assert "from English" not in content
        assert "into French" in content

    def test_unknown_source_milmmt_matches_builder(self):
        p = translation.render_prompt("Hallo", "fr", family="milmmt")
        assert p["text"] == translation._build_milmmt(
            "Hallo", "", "", "fr", "French", "", "")
        assert "English" not in p["text"]

    def test_unknown_source_custom_template_renders_empty(self):
        p = translation.render_prompt(
            "Hallo", "fr", template="[{source_language}] {text}")
        assert p["messages"][0]["content"] == "[] Hallo"


class TestFamilyOverride:
    def test_translate_segments_family_override(self, monkeypatch):
        monkeypatch.setattr(cfg, "TRANSLATION_DEFAULT_MODEL",
                            "org/m-GGUF:Q4", raising=False)
        seen = {}

        def fake_complete(llm, family, prompt, max_tokens):
            seen["family"] = family
            seen["prompt"] = prompt
            return "ok"

        _stub_get_model(monkeypatch)
        monkeypatch.setattr(translation, "_complete", fake_complete)
        results, warnings, meta = _run(translation.translate_segments(
            [{"text": "Hallo"}], ["en"], source_lang="de", mode="faithful",
            family_override="seedx",
            template_override="stale {text} {target_language}"))
        assert seen["family"] == "seedx"
        # seedx is a raw family: the prompt is the builder's string, and the
        # stale textarea template never leaks into it.
        assert isinstance(seen["prompt"], str)
        assert "stale" not in seen["prompt"]
        assert seen["prompt"].endswith("Hallo <en>")
        assert results[0]["en"] == "ok"


class TestComplete:
    """translation._complete is the one llama_cpp return-shape adapter —
    exercised with duck-typed fakes (no llama_cpp import)."""

    class _Llm:
        def __init__(self, chat_out=None, raw_out=None):
            self.chat_out, self.raw_out = chat_out, raw_out
            self.chat_calls, self.raw_calls = [], []

        def create_chat_completion(self, messages, max_tokens, **kw):
            self.chat_calls.append((messages, max_tokens, kw))
            return self.chat_out

        def __call__(self, prompt, max_tokens, **kw):
            self.raw_calls.append((prompt, max_tokens, kw))
            return self.raw_out

    def test_chat_family_strips_content_and_forwards_sampling(self):
        llm = self._Llm(chat_out={"choices": [
            {"message": {"content": "  Bonjour  "}}]})
        msgs = [{"role": "user", "content": "Hallo"}]
        assert translation._complete(llm, "hunyuan", msgs, 64) == "Bonjour"
        assert llm.chat_calls == [
            (msgs, 64, translation._FAMILIES["hunyuan"].sampling)]
        assert llm.chat_calls[0][2]["temperature"] == 0.7
        assert llm.raw_calls == []

    def test_raw_family_calls_llm_directly(self):
        llm = self._Llm(raw_out={"choices": [{"text": " Salut "}]})
        assert translation._complete(llm, "milmmt", "PROMPT", 32) == "Salut"
        assert llm.raw_calls == [("PROMPT", 32, translation._GREEDY)]
        assert llm.chat_calls == []

    def test_greedy_chat_family_pins_temperature_zero(self):
        llm = self._Llm(chat_out={"choices": [
            {"message": {"content": "x"}}]})
        translation._complete(llm, "chatml", [], 8)
        assert llm.chat_calls[0][2] == {"temperature": 0.0}

    def test_missing_content_coerces_to_empty(self):
        chat = self._Llm(chat_out={"choices": [{"message": {"content": None}}]})
        assert translation._complete(chat, "hunyuan", [], 8) == ""
        raw = self._Llm(raw_out={"choices": [{}]})
        assert translation._complete(raw, "seedx", "p", 8) == ""
