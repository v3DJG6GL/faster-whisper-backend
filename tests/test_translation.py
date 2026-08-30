"""Unit tests for translation.py (llama.cpp T2T module).

Everything runs WITHOUT llama_cpp installed: the module lazy-imports it only
inside _load_blocking, and these tests stub either the module-level
``_complete`` seam (one function = the single llama call) or
``_load_blocking`` itself. Async entry points are driven with asyncio.run()
(no pytest-asyncio), matching the streaming tests.
"""

import asyncio
import re
import sys

import pytest

import config as cfg
import translation


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
    ("tencent/Hunyuan-MT-7B-GGUF:Q4_K_M", "hunyuan"),
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
    langs = translation.list_languages("chatml")
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
    plain = _build("hunyuan", "Hallo")[0]["content"]
    assert plain == ("Translate the following segment into English, "
                     "without additional explanation.\n\nHallo")
    with_ctx = _build("hunyuan", "Hallo", context="A: Guten Tag")[0]["content"]
    assert "A: Guten Tag" in with_ctx
    assert ("referring to the information above, translate the following "
            "text into English: Hallo") in with_ctx
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
    assert g("Nimm 5 Tabletten", "Take five pills") == "digit mismatch"
    assert g("Hallo Welt", "Hallo Welt") == "output copies input"
    assert g("Hallo Welt", "Hallo Welt", check_copy=False) is None
    assert g("Ein normaler Satz hier mit Laenge",
             "abcdefghijkl" * 4) == "repetition loop"
    assert g("Hallo Welt", "Hello world") is None


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


# ---------------------------------------------------------------------------
# translate_segments — stubbed through the _complete seam
# ---------------------------------------------------------------------------

def _payload_of(prompt_or_msgs):
    """Recover the {text} payload from a chatml prompt (tests use chatml)."""
    content = prompt_or_msgs[-1]["content"] if isinstance(prompt_or_msgs, list) \
        else prompt_or_msgs
    return content.split("\n\n", 1)[1]


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

    async def fake_get_model(ref):
        return "STUB-LLM"
    monkeypatch.setattr(translation, "_get_model", fake_get_model)


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

    async def boom(ref):
        raise AssertionError("model must not load for a same-language target")
    monkeypatch.setattr(translation, "_get_model", boom)

    res, warns, meta = _run(translation.translate_segments(
        _segs("Hallo Welt.", "Noch was."), ["de", "de-CH"],
        source_lang="de", mode="faithful"))
    assert calls == []
    assert res == [{"de": "Hallo Welt.", "de-CH": "Hallo Welt."},
                   {"de": "Noch was.", "de-CH": "Noch was."}]
    assert warns == []
    assert meta == {"model": "org/model", "source": "de", "mode": "faithful"}


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

    async def fake_get_model(ref):
        return "STUB-LLM"
    monkeypatch.setattr(translation, "_get_model", fake_get_model)

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
     lambda t: "Take five pills now and call me again in the morning",
     "digit mismatch"),
    (_GUARD_SRC, lambda t: t, "output copies input"),
    (_GUARD_SRC, lambda t: "ok", "length ratio"),
    (_GUARD_SRC_NO_DIGITS, lambda t: "abcdefghijkl" * 4, "repetition loop"),
])
def test_guards_keep_original_after_one_retry(base_cfg, monkeypatch,
                                              src, bad_out, reason):
    calls = []
    _install_fake(monkeypatch, bad_out, calls)
    res, warns, _ = _run(translation.translate_segments(
        _segs(src), ["en"], source_lang="de", mode="faithful"))
    assert res == [{"en": src}]                          # original kept
    assert len(calls) == 2                               # initial + ONE retry
    assert len(warns) == 1
    assert "segment 1: kept original" in warns[0]
    assert reason in warns[0]


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


def test_fluent_group_failure_keeps_member_originals(base_cfg, monkeypatch):
    _install_fake(monkeypatch, lambda t: t)              # always a copy → fail
    segs = _segs("Hallo Welt und mehr Text", "es dir heute wirklich?")
    res, warns, _ = _run(translation.translate_segments(
        segs, ["en"], source_lang="de", mode="fluent"))
    assert res[0]["en"] == segs[0]["text"]
    assert res[1]["en"] == segs[1]["text"]
    assert len(warns) == 1 and "kept original" in warns[0]
    assert "segments 1-2" in warns[0]


def test_progress_and_cancel(base_cfg, monkeypatch):
    _install_fake(monkeypatch, _xlate)
    fractions = []
    steps = []
    res, _, _ = _run(translation.translate_segments(
        _segs("Eins.", "Zwei."), ["en", "fr"], source_lang="de",
        mode="fluent",
        progress_cb=lambda f, s: (fractions.append(f), steps.append(s))))
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert steps[0].startswith("en ") and steps[-1].startswith("fr ")

    with pytest.raises(translation.TranslationCancelled):
        _run(translation.translate_segments(
            _segs("Eins."), ["en"], source_lang="de",
            cancel_check=lambda: True))


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

    def fake_load(ref, device, family):
        made[ref] = _FakeLlama(ref)
        return made[ref]
    monkeypatch.setattr(translation, "_load_blocking", fake_load)
    monkeypatch.setattr(translation.system_stats, "gpu_mem_used_bytes",
                        lambda: None)
    stats = {"registered": [], "unregistered": [], "touched": []}
    monkeypatch.setattr(translation.system_stats, "register_loaded_model",
                        lambda name, vram, device, kind:
                        stats["registered"].append((name, device, kind)))
    monkeypatch.setattr(translation.system_stats, "unregister_loaded_model",
                        lambda name: stats["unregistered"].append(name))
    monkeypatch.setattr(translation.system_stats, "touch_loaded_model",
                        lambda name: stats["touched"].append(name))
    monkeypatch.setattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 2,
                        raising=False)
    monkeypatch.setattr(cfg, "TRANSLATION_DEVICE", "cpu", raising=False)
    translation._models.clear()
    translation._last_used.clear()
    return made, stats


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

    async def fake_get_model(ref):
        return "STUB-LLM"
    monkeypatch.setattr(translation, "_get_model", fake_get_model)

    res, warns, meta = _run(translation.translate_segments(
        _segs("Hallo Welt."), ["en"], source_lang="de", mode="faithful",
        model_ref="tencent/Hunyuan-MT-7B-GGUF:Q4",   # would detect hunyuan
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

    async def fake_get_model(ref):
        return "STUB-LLM"
    monkeypatch.setattr(translation, "_get_model", fake_get_model)

    _run(translation.translate_segments(
        _segs("Hallo Welt."), ["en"], source_lang="de", mode="faithful"))
    assert prompts[0] == [{"role": "user", "content": "SAVED Hallo Welt."}]
