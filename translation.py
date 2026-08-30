"""Text-to-text translation via llama.cpp GGUF models (optional install).

A third model kind next to the WhisperModel cache (main.py) and the pyannote
singleton (diarization.py), with the same lifecycle discipline scaled to a tiny
LRU dict of loaded models: lazy import (the dependency set is the optional
``requirements-translate.txt``), load on first use under an asyncio.Lock with
an NVML VRAM delta, registration in ``system_stats`` (as ``gguf:<ref>``) so
/stats shows each loaded model, an idle-eviction loop driven live by
``TRANSLATION_IDLE_TIMEOUT_S``, and admin-triggered eviction via
:func:`drop_models`.

Models are addressed by a GGUF ref ``org/repo[:quant]`` — the optional
``:quant`` suffix selects a quantization file inside the repo (resolved as the
``*<quant>.gguf`` filename glob), e.g. ``mradermacher/Hunyuan-MT-7B-GGUF:Q4_K_M``.

Prompting is family-based (``_FAMILIES``): the family is auto-detected from
the model name (``detect_family``) unless ``TRANSLATION_PROMPT_FAMILY`` pins
one; ``custom`` renders ``TRANSLATION_PROMPT_TEMPLATE``. Two run modes:
``faithful`` translates segment-by-segment (batched as a numbered list the
model must echo back — exact cue alignment), ``fluent`` merges consecutive
segments into sentence groups, translates each group, and redistributes the
translation across the member segments proportionally by source length.

Failure contract (soft-fail): every load/inference problem surfaces as a
``TranslationError`` whose message is safe to echo to the client — the caller
turns it into a response ``warnings`` entry and returns the untranslated
transcript. ``TranslationCancelled`` (cooperative, via ``cancel_check``) must
abort the whole request instead. Raw third-party exception text stays in the
server log only.

Concurrency: llama.cpp inference is NOT thread-safe — every completion runs
under the module ``_infer_mutex``. The GPU/CPU inference-semaphore policy
(whether a translation counts against INFERENCE_CONCURRENCY, and on which
device pool) is deliberately the CALLER's job; this module only serializes
its own llama calls.
"""

import asyncio
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

import config as cfg
import system_stats

logger = logging.getLogger("whisper-server")


class TranslationCancelled(Exception):
    """The caller cancelled mid-translation (cooperative: raised between
    batches when ``cancel_check`` answers True). Deliberately NOT a
    TranslationError — the handler must abort the whole request, not
    soft-fail into an untranslated transcript."""


class TranslationError(RuntimeError):
    """Translation could not run; str(exc) is CLIENT-SAFE (our own wording)."""


# Serializes actual llama.cpp inference across threads — llama.cpp contexts
# are NOT thread-safe (same hazard class as diarization._infer_mutex).
_infer_mutex = threading.Lock()
# ref → count of jobs currently holding a lease on the cached model — an
# in-use model is never evicted (see _drop_locked).
_active: "dict[str, int]" = {}

_lock = asyncio.Lock()
# ref → loaded llama_cpp.Llama, insertion-ordered oldest-first (LRU: a cache
# hit moves the ref to the end; eviction pops from the front).
_models: "OrderedDict[str, object]" = OrderedDict()
# ref → time.monotonic() of last use, for the idle evictor.
_last_used: "dict[str, float]" = {}
_STATS_PREFIX = "gguf:"

# The idle loop mirrors main._idle_evictor's cadence.
_EVICTOR_WAKE_S = 30


def _resolve_device() -> str:
    """TRANSLATION_DEVICE with "auto" following MODEL_DEVICE. No torch probe:
    llama.cpp checks CUDA availability itself at load and falls back to CPU
    layers, so auto simply means "cuda unless the server is pinned to cpu"."""
    want = (getattr(cfg, "TRANSLATION_DEVICE", "auto") or "auto").lower()
    if want == "auto":
        want = "cpu" if (getattr(cfg, "MODEL_DEVICE", "cpu") or "cpu").lower() == "cpu" else "cuda"
    return want if want in ("cuda", "cpu") else "cpu"


def _parse_model_ref(ref: str) -> "tuple[str, str | None]":
    """Split ``org/repo[:quant]`` on the LAST ":" → (repo_id, quant|None)."""
    repo, sep, quant = ref.rpartition(":")
    if not sep:
        return ref, None
    return repo, (quant or None)


# =============================================================================
# Prompt families
# =============================================================================

# English language names for the app languages (+ a title-case fallback for
# anything else — see _lang_name). Keys are lowercase base codes.
_LANG_NAMES: "dict[str, str]" = {
    "en": "English", "de": "German", "fr": "French", "it": "Italian",
    "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "cs": "Czech", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "tr": "Turkish",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "hu": "Hungarian",
}


def _lang_name(code: "str | None") -> str:
    """English name for a language code; region subtags fall back to the base
    code's name; unknown codes are title-cased ("rm" → "Rm")."""
    if not code:
        return ""
    low = code.strip().lower()
    base = low.split("-")[0]
    return _LANG_NAMES.get(low) or _LANG_NAMES.get(base) or base.title()


def list_languages(family: str) -> "list[str]":
    """Language codes offered for the given family (for /v1/me): the
    _LANG_NAMES keys, sorted, "en" first. Good enough for every family —
    the models themselves accept far more."""
    codes = sorted(_LANG_NAMES)
    codes.remove("en")
    return ["en"] + codes


def _glossary_pairs(glossary: str) -> "list[tuple[str, str]]":
    """Parse "source = target" lines (one pair per line; malformed lines are
    skipped)."""
    out: "list[tuple[str, str]]" = []
    for line in (glossary or "").splitlines():
        src, sep, tgt = line.partition("=")
        if sep and src.strip() and tgt.strip():
            out.append((src.strip(), tgt.strip()))
    return out


def _build_hunyuan(text, source_code, source_name, target_code, target_name,
                   context, glossary):
    parts: "list[str]" = []
    for src, tgt in _glossary_pairs(glossary):
        parts.append(f"with reference to: {src} -> {tgt}")
    if context:
        parts.append(context)
        parts.append(
            f"referring to the information above, translate the following "
            f"text into {target_name}: {text}")
    else:
        parts.append(
            f"Translate the following segment into {target_name}, "
            f"without additional explanation.\n\n{text}")
    return [{"role": "user", "content": "\n".join(parts)}]


def _build_gemma(text, source_code, source_name, target_code, target_name,
                 context, glossary):
    # TranslateGemma's structured single turn. The source code is REQUIRED by
    # the format — the caller defaults an unknown source to "en". Context and
    # glossary are silently ignored (the format has no slot for them).
    return [{"role": "user", "content": (
        f"type:text,source_lang_code:{source_code or 'en'},"
        f"target_lang_code:{target_code},text:{text}")}]


def _build_milmmt(text, source_code, source_name, target_code, target_name,
                  context, glossary):
    # Raw completion with ENGLISH language names (the MiLMMT training format).
    return (f"Translate this from {source_name} to {target_name}:\n"
            f"{source_name}: {text}\n{target_name}:")


def _build_seedx(text, source_code, source_name, target_code, target_name,
                 context, glossary):
    return (f"Translate the following {source_name} sentence into "
            f"{target_name}:\n{text} <{target_code}>")


def _build_chatml(text, source_code, source_name, target_code, target_name,
                  context, glossary):
    return [{"role": "user", "content": (
        f"Translate the following text from {source_name} into "
        f"{target_name}. Reply with ONLY the translation.\n\n{text}")}]


def _render_custom_template(tpl, text, source_name, target_name, context,
                            glossary):
    # SINGLE-PASS rendering (never str.format — transcript text may carry
    # braces, and sequential .replace would re-substitute placeholder tokens
    # occurring literally INSIDE the user-controlled transcript). Missing
    # optional slots render as "". The config validator guarantees {text} and
    # {target_language} are present in a saved template; the admin test
    # endpoint may pass an UNSAVED template through
    # translate_segments(template_override=...), rendered by these same rules.
    slots = {
        "{text}": text,
        "{target_language}": target_name,
        "{source_language}": source_name or "",
        "{context}": context or "",
        "{glossary}": glossary or "",
    }
    return re.sub(
        r"\{(?:text|target_language|source_language|context|glossary)\}",
        lambda m: slots[m.group(0)],
        tpl,
    )


def _build_custom(text, source_code, source_name, target_code, target_name,
                  context, glossary):
    tpl = getattr(cfg, "TRANSLATION_PROMPT_TEMPLATE", "") or ""
    return [{"role": "user", "content": _render_custom_template(
        tpl, text, source_name, target_name, context, glossary)}]


@dataclass(frozen=True)
class Family:
    """One prompt-family entry: chat template vs raw completion, the prompt
    builder, sampling params, context size and stop strings."""
    chat: bool
    build: "object"     # (text, source_code, source_name, target_code,
    #                      target_name, context, glossary) -> str | list[dict]
    sampling: dict = field(default_factory=dict)
    n_ctx: int = 8192
    stop: "list[str]" = field(default_factory=list)


_GREEDY = {"temperature": 0.0}

_FAMILIES: "dict[str, Family]" = {
    "hunyuan": Family(
        chat=True, build=_build_hunyuan, n_ctx=8192,
        sampling={"top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05,
                  "temperature": 0.7}),
    "gemma-translate": Family(
        chat=True, build=_build_gemma, n_ctx=2048, sampling=dict(_GREEDY)),
    "milmmt": Family(
        chat=False, build=_build_milmmt, n_ctx=8192, sampling=dict(_GREEDY)),
    "seedx": Family(
        chat=False, build=_build_seedx, n_ctx=8192, sampling=dict(_GREEDY)),
    "chatml": Family(
        chat=True, build=_build_chatml, n_ctx=4096, sampling=dict(_GREEDY)),
    "custom": Family(
        chat=True, build=_build_custom, n_ctx=8192, sampling=dict(_GREEDY)),
}


def detect_family(ref: str) -> str:
    """Prompt family from the model name (lowered substring match)."""
    low = (ref or "").lower()
    if "hunyuan" in low or "hy-mt" in low:
        return "hunyuan"
    if "translategemma" in low:
        return "gemma-translate"
    if "milmmt" in low:
        return "milmmt"
    if "seed-x" in low or "seedx" in low:
        return "seedx"
    return "chatml"


def resolve_family(ref: str) -> str:
    """detect_family(ref) unless TRANSLATION_PROMPT_FAMILY pins one."""
    configured = (getattr(cfg, "TRANSLATION_PROMPT_FAMILY", "auto")
                  or "auto").lower()
    if configured != "auto" and configured in _FAMILIES:
        return configured
    return detect_family(ref)


def _ctx_for(family: str) -> int:
    return _FAMILIES.get(family, _FAMILIES["chatml"]).n_ctx


# =============================================================================
# Model loading / LRU cache
# =============================================================================

def _load_blocking(ref: str, device: str, family: str):
    """Import llama_cpp and load the GGUF model. Runs in the default
    executor."""
    # Keep HF downloads on the models volume (whisper weights already live
    # there via download_root); a set HF_HOME always wins.
    download_root = getattr(cfg, "DOWNLOAD_ROOT", None)
    if download_root:
        os.environ.setdefault("HF_HOME", os.path.join(download_root, "hf"))
    # LOCAL_FILES_ONLY is a HOT setting — scope the offline env var to this
    # load and restore it after, or one offline load would poison every later
    # huggingface_hub download until a process restart.
    offline_prev = os.environ.get("HF_HUB_OFFLINE")
    if getattr(cfg, "LOCAL_FILES_ONLY", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return _load_blocking_inner(ref, device, family)
    finally:
        if getattr(cfg, "LOCAL_FILES_ONLY", False):
            if offline_prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = offline_prev


def _load_blocking_inner(ref: str, device: str, family: str):
    try:
        import llama_cpp
    except ImportError as e:
        # A broken install raises ImportError too — the log carries the real
        # cause (same diagnosability note as diarization._load_blocking).
        logger.error("[translate] llama_cpp import failed: %s", e)
        raise TranslationError(
            "translation dependencies are not installed on this server — "
            "pip install -r requirements-translate.txt"
        ) from e

    repo, quant = _parse_model_ref(ref)
    try:
        return llama_cpp.Llama.from_pretrained(
            repo_id=repo,
            filename=(f"*{quant}.gguf" if quant else "*.gguf"),
            n_gpu_layers=(-1 if device == "cuda" else 0),
            n_ctx=_ctx_for(family),
            verbose=False,
        )
    except TranslationError:
        raise
    except Exception as e:
        logger.error("[translate] model load failed for %s: %s", ref, e)
        raise TranslationError(
            f"could not load translation model {ref} — check the ref "
            f"('org/repo[:quant]'), the quantization filename, and that the "
            f"download is reachable"
        ) from e


def _drop_locked(ref: str) -> bool:
    """Drop one cached model. Caller holds _lock. Declines (False) while a
    job holds a lease on it — closing llama.cpp's native context under a
    running decode is a use-after-free that takes the whole process down."""
    if _active.get(ref, 0) > 0:
        logger.info("[translate] model %s is in use — eviction deferred", ref)
        return False
    llm = _models.pop(ref, None)
    _last_used.pop(ref, None)
    if llm is None:
        return True
    try:
        llm.close()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    system_stats.unregister_loaded_model(_STATS_PREFIX + ref)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("[translate] model %s unloaded", ref)
    return True


async def _get_model(ref: str, *, lease: bool = False):
    """Return the cached model for ``ref``, loading (and LRU-evicting past
    TRANSLATION_MAX_LOADED_MODELS) on a miss."""
    async with _lock:
        if ref in _models:
            _models.move_to_end(ref)
            _last_used[ref] = time.monotonic()
            system_stats.touch_loaded_model(_STATS_PREFIX + ref)
            if lease:
                _active[ref] = _active.get(ref, 0) + 1
            return _models[ref]
        cap = max(1, int(getattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 1) or 1))
        while len(_models) >= cap:
            victim = next(
                (r for r in _models if not _active.get(r, 0)), None)
            if victim is None:
                # Every cached model is mid-job — overflow the cap rather
                # than free a context under a running decode; the idle
                # evictor trims the excess once the jobs release.
                logger.warning(
                    "[translate] all %d cached models are in use — "
                    "temporarily exceeding TRANSLATION_MAX_LOADED_MODELS",
                    len(_models))
                break
            _drop_locked(victim)
        device = _resolve_device()
        family = resolve_family(ref)
        vram_before = system_stats.gpu_mem_used_bytes()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        llm = await loop.run_in_executor(None, _load_blocking, ref, device, family)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        system_stats.register_loaded_model(
            _STATS_PREFIX + ref, vram, device, "gguf")
        logger.info("[translate] model %s loaded on %s in %.1fs",
                    ref, device, time.perf_counter() - t0)
        _models[ref] = llm
        _last_used[ref] = time.monotonic()
        if lease:
            _active[ref] = _active.get(ref, 0) + 1
        return llm


async def _release_model(ref: str) -> None:
    """Release a lease taken by ``_get_model(..., lease=True)`` and restart
    the model's idle clock (a long job must not be evicted the moment it
    ends because its LOAD time stamp aged past the timeout)."""
    async with _lock:
        n = _active.get(ref, 0) - 1
        if n <= 0:
            _active.pop(ref, None)
        else:
            _active[ref] = n
        if ref in _models:
            _last_used[ref] = time.monotonic()


async def drop_models() -> None:
    """Evict every cached model (admin eviction / shutdown)."""
    async with _lock:
        for ref in list(_models):
            _drop_locked(ref)


async def idle_evictor_loop() -> None:
    """Unload models idle for TRANSLATION_IDLE_TIMEOUT_S seconds. Reads the
    timeout live (an admin edit applies without restart), like
    diarization.idle_evictor_loop. Started/cancelled by lifespan."""
    while True:
        await asyncio.sleep(_EVICTOR_WAKE_S)
        try:
            timeout = int(getattr(cfg, "TRANSLATION_IDLE_TIMEOUT_S", 0) or 0)
            if timeout <= 0 or not _models:
                continue
            now = time.monotonic()
            async with _lock:
                for ref, last in list(_last_used.items()):
                    if ref in _models and now - last >= timeout:
                        _drop_locked(ref)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            logger.error("[translate] idle evictor error: %s", e)


# =============================================================================
# Inference seam
# =============================================================================

def _complete(llm, family: str, prompt_or_msgs, max_tokens: int) -> str:
    """The single blocking completion call — chat families via
    create_chat_completion, raw families via a plain llm() call — holding
    ``_infer_mutex`` (llama.cpp is not thread-safe). Runs in the executor.
    Tests stub THIS function to exercise everything above it without
    llama_cpp installed."""
    fam = _FAMILIES[family]
    kwargs = dict(fam.sampling)
    if fam.stop:
        kwargs["stop"] = fam.stop
    with _infer_mutex:
        if fam.chat:
            out = llm.create_chat_completion(
                messages=prompt_or_msgs, max_tokens=max_tokens, **kwargs)
            return (out["choices"][0]["message"].get("content") or "").strip()
        out = llm(prompt_or_msgs, max_tokens=max_tokens, **kwargs)
        return (out["choices"][0].get("text") or "").strip()


# =============================================================================
# Fluent-mode helpers (pure)
# =============================================================================

# Sentence-final punctuation that closes a fluent-mode group.
_SENTENCE_FINAL = ".!?…。"
# Max segments merged into one fluent group.
_MAX_GROUP_SEGMENTS = 6


def _merge_sentences(segments: "list[dict]") -> "list[list[int]]":
    """Merge consecutive segments into sentence groups: accumulate until a
    segment's text ends with sentence-final punctuation or the group already
    spans _MAX_GROUP_SEGMENTS segments. Returns groups of segment indices."""
    groups: "list[list[int]]" = []
    current: "list[int]" = []
    for i, seg in enumerate(segments):
        current.append(i)
        text = (seg.get("text") or "").rstrip()
        if (text and text[-1] in _SENTENCE_FINAL) or \
                len(current) >= _MAX_GROUP_SEGMENTS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _redistribute(group_texts: "list[str]", translated: str) -> "list[str]":
    """Split one group's translated text back across its member segments,
    proportionally by each member's share of the group's source char length,
    cutting at the word boundary (space) nearest each proportional cut."""
    n = len(group_texts)
    if n <= 1:
        return [translated.strip()] if n == 1 else []
    translated = translated.strip()
    if not translated:
        return [""] * n
    spaces = [i for i, ch in enumerate(translated) if ch == " "]
    total = sum(len(t) for t in group_texts) or 1
    cuts: "list[int]" = []
    acc = 0
    prev = 0
    for t in group_texts[:-1]:
        acc += len(t)
        ideal = len(translated) * acc / total
        if spaces:
            cut = min(spaces, key=lambda s: abs(s - ideal))
        else:
            cut = int(round(ideal))
        cut = max(cut, prev)   # keep cuts monotonic
        cuts.append(cut)
        prev = cut
    pieces: "list[str]" = []
    start = 0
    for c in cuts:
        pieces.append(translated[start:c].strip())
        start = c
    pieces.append(translated[start:].strip())
    return pieces


# =============================================================================
# Faithful-mode numbered-list helpers (pure)
# =============================================================================

_NUMBERED_INSTRUCTION = (
    "Keep the numbered list format: reply with the same numbered list, one "
    "translated line per item.")


def _build_numbered(texts: "list[str]") -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))


def _parse_numbered(reply: str, n: int) -> "list[str] | None":
    """Parse a numbered-list reply back into n strings; None on any count
    mismatch (missing or unparseable lines) — the caller then halves the
    batch size and retries."""
    found: "dict[int, str]" = {}
    for line in (reply or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[.):]\s*(.*)", line)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n and idx not in found:
            found[idx] = m.group(2).strip()
    if len(found) != n:
        return None
    return [found[i + 1] for i in range(n)]


# =============================================================================
# Output guards (pure)
# =============================================================================

# Any 12+ char substring repeated 4+ times = a generation loop.
_REPETITION_RE = re.compile(r"(.{12,}?)(?:\1){3,}", re.DOTALL)
# Length-ratio guard bounds, applied only past the absolute floor below.
_RATIO_LO, _RATIO_HI = 0.4, 3.0
_RATIO_FLOOR_CHARS = 20


_CJK_TARGETS = {"zh", "ja", "ko"}
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _ratio_bounds(src: str, target: "str | None") -> "tuple[float, float]":
    """Char-count ratio bounds, script-aware: CJK text carries ~2-4x the
    information per character of Latin text, so the default [0.4, 3.0] band
    would reject nearly every valid Latin→CJK translation (and vice versa)."""
    tgt_cjk = bool(target) and target.strip().lower().split("-")[0] in _CJK_TARGETS
    cjk_chars = len(_CJK_CHAR_RE.findall(src))
    src_cjk = len(src) > 0 and cjk_chars / len(src) > 0.3
    if tgt_cjk and not src_cjk:
        return (0.12, _RATIO_HI)
    if src_cjk and not tgt_cjk:
        return (_RATIO_LO, 9.0)
    return (_RATIO_LO, _RATIO_HI)


def _guard_reason(src: str, out: str, *, target: "str | None" = None,
                  check_copy: bool = True) -> "str | None":
    """Reason string when a translated segment fails a sanity guard, else
    None. Guards: empty output; length ratio outside script-aware bounds
    (only once either side reaches the 20-char floor); digit multiset
    mismatch; verbatim input copy (when the target differs from the source);
    repetition loop."""
    s = (src or "").strip()
    o = (out or "").strip()
    if not o:
        return "empty output"
    if s and (len(s) >= _RATIO_FLOOR_CHARS or len(o) >= _RATIO_FLOOR_CHARS):
        lo, hi = _ratio_bounds(s, target)
        ratio = len(o) / len(s)
        if not (lo <= ratio <= hi):
            return f"length ratio {ratio:.2f} outside [{lo}, {hi}]"
    if Counter(re.findall(r"\d", s)) != Counter(re.findall(r"\d", o)):
        return "digit mismatch"
    if check_copy and o == s:
        return "output copies input"
    if _REPETITION_RE.search(o):
        return "repetition loop"
    return None


# =============================================================================
# Main entry
# =============================================================================

def _same_lang(a: "str | None", b: "str | None") -> bool:
    if not a or not b:
        return False
    return a.strip().lower().split("-")[0] == b.strip().lower().split("-")[0]


def _context_lines(segments: "list[dict]", upto: int, count: int) -> str:
    """The previous ``count`` SOURCE segment texts before index ``upto``,
    speaker labels prefixed (context lines only — never the payload)."""
    if count <= 0 or upto <= 0:
        return ""
    lines: "list[str]" = []
    for seg in segments[max(0, upto - count):upto]:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n".join(lines)


async def _run_completion(llm, family: str, text: str, source_code: str,
                          target_code: str, context: str, glossary: str,
                          template_override: "str | None" = None) -> str:
    """Build the family prompt for ``text`` and run one completion in the
    executor (via the _complete seam). ``template_override`` (custom family
    only) renders THAT template instead of cfg.TRANSLATION_PROMPT_TEMPLATE —
    the admin template-test path, never persisted."""
    fam = _FAMILIES[family]
    if template_override is not None and family == "custom":
        prompt = [{"role": "user", "content": _render_custom_template(
            template_override, text, _lang_name(source_code),
            _lang_name(target_code), context, glossary)}]
    else:
        prompt = fam.build(
            text, source_code, _lang_name(source_code), target_code,
            _lang_name(target_code), context, glossary)
    max_tokens = len(text) // 2 + 256
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _complete, llm, family, prompt, max_tokens)


async def translate_segments(
    segments: "list[dict]",
    targets: "list[str]",
    *,
    source_lang: "str | None" = None,
    model_ref: "str | None" = None,
    mode: str = "fluent",
    glossary: str = "",
    context_segments: "int | None" = None,
    progress_cb=None,
    cancel_check=None,
    template_override: "str | None" = None,
) -> "tuple[list[dict[str, str]], list[str], dict]":
    """Translate ``segments`` (``[{"text": str, "speaker": str|None}, …]``)
    into every language in ``targets``.

    Returns ``(per_segment_translations, warnings, meta)``:
    one ``{target_code: translated_text}`` dict per INPUT segment (cue count
    and timing stay the caller's — this function only returns per-segment
    strings), client-safe warning strings for segments kept untranslated,
    and ``meta = {"model", "source", "mode"}``.

    ``model_ref`` empty → ``TRANSLATION_DEFAULT_MODEL``; both empty →
    :class:`TranslationError`. A target equal to ``source_lang`` is copied
    verbatim without a model call. ``mode``: see the module docstring.
    ``context_segments`` overrides ``TRANSLATION_CONTEXT_SEGMENTS`` when not
    None. ``progress_cb(done_fraction, step_str)`` fires after each batch;
    ``cancel_check`` (no-arg, truthy = abort) is polled between batches and
    raises :class:`TranslationCancelled`. ``template_override`` (the admin
    template-test path) forces the ``custom`` family and renders THAT
    template for this call only — cfg stays untouched.
    """
    ref = (model_ref or "").strip() or \
        (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
    if not ref:
        raise TranslationError(
            "no translation model configured (TRANSLATION_DEFAULT_MODEL)")
    family = "custom" if template_override is not None else resolve_family(ref)
    source_code = (source_lang or "").strip() or "en"
    ctx_n = int(getattr(cfg, "TRANSLATION_CONTEXT_SEGMENTS", 3) or 0) \
        if context_segments is None else int(context_segments)

    results: "list[dict[str, str]]" = [{} for _ in segments]
    warnings: "list[str]" = []
    meta = {"model": ref, "source": source_lang or "", "mode": mode}
    if not segments or not targets:
        return results, warnings, meta

    total_units = len(segments) * len(targets)
    done_units = 0

    def _progress(step: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(done_units / total_units, step)
            except Exception:  # noqa: BLE001 — progress must never break us
                pass

    def _check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise TranslationCancelled()

    llm = None   # loaded lazily — a same-language-only request never loads

    async def _model():
        nonlocal llm
        if llm is None:
            llm = await _get_model(ref, lease=True)
        return llm

    async def _translate_one(text: str, target: str, context: str) -> str:
        return await _run_completion(
            await _model(), family, text, source_code, target, context,
            glossary, template_override=template_override)

    async def _guarded_single(text: str, target: str, context: str,
                              seg_no: int) -> str:
        """One translation + guard; on failure ONE retry alone, then keep the
        original text + a warning."""
        out = await _translate_one(text, target, context)
        reason = _guard_reason(text, out, target=target)
        if reason is None:
            return out
        out = await _translate_one(text, target, context)
        reason = _guard_reason(text, out, target=target)
        if reason is None:
            return out
        warnings.append(
            f"segment {seg_no}: kept original — translation failed ({reason})")
        return text

    try:
      for target in targets:
        _check_cancel()
        if _same_lang(target, source_lang):
            # Same-language short-circuit: copy each text verbatim.
            for i, seg in enumerate(segments):
                results[i][target] = (seg.get("text") or "")
            done_units += len(segments)
            _progress(f"{target} 1/1")
            continue

        if mode == "faithful":
            k = max(1, int(getattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 8) or 1))
            n_batches = max(1, -(-len(segments) // k))
            i = 0
            batch_no = 0
            while i < len(segments):
                _check_cancel()
                batch_idx = list(range(i, min(i + k, len(segments))))
                texts = [(segments[j].get("text") or "") for j in batch_idx]
                context = _context_lines(segments, i, ctx_n)
                if len(batch_idx) == 1:
                    results[batch_idx[0]][target] = await _guarded_single(
                        texts[0], target, context, batch_idx[0] + 1)
                else:
                    payload = (f"{_NUMBERED_INSTRUCTION}\n\n"
                               f"{_build_numbered(texts)}")
                    reply = await _translate_one(payload, target, context)
                    parsed = _parse_numbered(reply, len(batch_idx))
                    if parsed is None:
                        # Line-count mismatch → halve the batch and retry the
                        # same position (down to per-segment prompts).
                        k = max(1, k // 2)
                        n_batches = max(1, -(-(len(segments) - i) // k)) + batch_no
                        continue
                    for j, out in zip(batch_idx, parsed):
                        src = segments[j].get("text") or ""
                        reason = _guard_reason(src, out, target=target)
                        if reason is None:
                            results[j][target] = out
                        else:
                            # ONE retry of that segment alone, else keep
                            # original + warning.
                            retried = await _translate_one(
                                src, target,
                                _context_lines(segments, j, ctx_n))
                            reason = _guard_reason(src, retried, target=target)
                            if reason is None:
                                results[j][target] = retried
                            else:
                                results[j][target] = src
                                warnings.append(
                                    f"segment {j + 1}: kept original — "
                                    f"translation failed ({reason})")
                i += len(batch_idx)
                batch_no += 1
                done_units += len(batch_idx)
                _progress(f"{target} {min(batch_no, n_batches)}/{n_batches}")
        else:
            # FLUENT: sentence-group merge → translate → redistribute.
            groups = _merge_sentences(segments)
            for g_no, group in enumerate(groups, start=1):
                _check_cancel()
                src_texts = [(segments[j].get("text") or "") for j in group]
                joined = " ".join(t.strip() for t in src_texts).strip()
                context = _context_lines(segments, group[0], ctx_n)
                translated = await _translate_one(joined, target, context)
                reason = _guard_reason(joined, translated, target=target)
                if reason is not None:
                    translated = await _translate_one(joined, target, context)
                    reason = _guard_reason(joined, translated, target=target)
                if reason is not None:
                    for j in group:
                        results[j][target] = segments[j].get("text") or ""
                    span = (f"segment {group[0] + 1}" if len(group) == 1 else
                            f"segments {group[0] + 1}-{group[-1] + 1}")
                    warnings.append(
                        f"{span}: kept original — translation failed "
                        f"({reason})")
                else:
                    for j, piece in zip(group,
                                        _redistribute(src_texts, translated)):
                        results[j][target] = piece
                done_units += len(group)
                _progress(f"{target} {g_no}/{len(groups)}")
    finally:
        if llm is not None:
            await _release_model(ref)

    return results, warnings, meta
