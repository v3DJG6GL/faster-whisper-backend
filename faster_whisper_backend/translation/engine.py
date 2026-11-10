"""Text-to-text translation via llama.cpp GGUF models (optional install).

A third model kind next to the WhisperModel cache (transcription/models.py)
and the pyannote singleton (audio/diarization.py), with the same lifecycle
discipline scaled to a tiny LRU dict of loaded models: lazy import (the
dependency set is the optional
``requirements-translate.txt``), load on first use under an asyncio.Lock with
an NVML VRAM delta, registration in ``model_registry`` (as ``gguf:<ref>``) so
/stats shows each loaded model, an idle-eviction loop driven live by
``TRANSLATION_IDLE_TIMEOUT_S``, and admin-triggered eviction via
:func:`drop_models`.

Models are addressed by a GGUF ref ``org/repo[:quant]`` — the optional
``:quant`` suffix selects a quantization file inside the repo (resolved as the
``*<quant>.gguf`` filename glob), e.g. ``tencent/HY-MT1.5-7B-GGUF:Q4_K_M``.

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
import bisect
import importlib
import logging
import os
import re
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.runtime import hf_cache
from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.core import device_memory
from faster_whisper_backend.core.languages import (
    _NAMES, ZH_HANT_REGIONS, _normalise_code, canonical_code, language_codes,
    language_label, language_name, lookup, same_language)
from faster_whisper_backend.core.loop_lock import CancelLatch, LoopLock, release_count, run_every
from faster_whisper_backend.translation import gating as tr_gating

logger = logging.getLogger("whisper-server")


class TranslationCancelled(Exception):
    """The caller cancelled mid-translation (cooperative: raised between
    batches when ``cancel_check`` answers True). Deliberately NOT a
    TranslationError — the handler must abort the whole request, not
    soft-fail into an untranslated transcript."""


class TranslationError(RuntimeError):
    """Translation could not run; str(exc) is CLIENT-SAFE (our own wording)."""


class _ContextOverflow(TranslationError):
    """The prompt alone is longer than the model's n_ctx (llama.cpp's
    "Requested tokens … exceed context window" ValueError). translate_segments
    handles it per batch — halve, or keep the original — instead of failing
    the whole run."""


# Serializes actual llama.cpp inference across threads — llama.cpp contexts
# are NOT thread-safe (same hazard class as diarization._infer_mutex).
_infer_mutex = threading.RLock()   # re-entered: _run_completion → _complete
# Async front gate for the mutex above: waiters queue HERE, on the event
# loop, instead of each pinning a default-executor thread while _infer_mutex
# is held — N queued translations would otherwise occupy N pool threads
# shared with whisper decodes and model loads. _infer_mutex stays as the
# defensive thread guard inside _complete.
_infer_gate = LoopLock()   # as _lock below: a plain asyncio.Lock binds to one loop
# ref → count of jobs currently holding a lease on the cached model — an
# in-use model is never evicted (see _drop_locked).
_active: "dict[str, int]" = {}
# id(llm) → completions handed to the executor for that model and not yet
# finished. A cancelled job's lease is gone while its executor thread may
# still be queued for or inside the decode (see _run_completion); this, not
# the module-wide _infer_mutex (held by a decode on ANY model), is what
# defers that one model's eviction. Written from the loop and from executor
# threads, hence its own lock.
_decoding: "dict[int, int]" = {}
_decoding_mutex = threading.Lock()

_lock = LoopLock()   # see core.loop_lock: survives a test suite's loop-per-lifespan
# ref → loaded llama_cpp.Llama, insertion-ordered oldest-first (LRU: a cache
# hit moves the ref to the end; eviction pops from the front).
_models: "OrderedDict[str, object]" = OrderedDict()
# ref → time.monotonic() of last use, for the idle evictor.
_last_used: "dict[str, float]" = {}
# ref → LoopLock serializing that one ref's cold load, so a warm hit on
# any OTHER ref never waits behind a multi-GB download (see _get_model).
# Entries persist for the process — one tiny Lock per distinct ref ever
# requested (a failed load keeps its entry too); popping them on release
# would open a re-create race that lets two jobs load the same model
# concurrently.
_loading: "dict[str, LoopLock]" = {}
# Loads currently inside _load_blocking, and how many times a load has
# started while another was already running — a VRAM delta measured across
# such an overlap belongs to neither load (see _get_model).
_loads_in_flight = 0
_load_overlaps = 0
# ref → (device, n_ctx) the cached model was loaded with. A cache hit re-keys
# on these so an admin device/prompt-family edit reloads the model instead of
# serving one built for the old parameters forever.
_params: "dict[str, tuple]" = {}
_STATS_PREFIX = "gguf:"

# The idle loop mirrors transcription.models._idle_evictor's cadence.
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


def _gguf_glob(quant: "str | None") -> str:
    """The *.gguf filename glob Llama.from_pretrained applies for ``quant``."""
    return f"*{quant}.gguf" if quant else "*.gguf"


# Chinese language names for HY-MT's official contextual template (the
# instruction that carries them is Chinese — see _build_hunyuan): the 38
# HY-MT1.5 languages as its model card names them, plus a few more. Keys are
# lowercase codes; anything absent falls back to the English name.
_ZH_LANG_NAMES: "dict[str, str]" = {
    "en": "英语", "de": "德语", "fr": "法语", "es": "西班牙语",
    "it": "意大利语", "pt": "葡萄牙语", "ru": "俄语", "ja": "日语",
    "ko": "韩语", "zh": "中文", "nl": "荷兰语", "pl": "波兰语",
    "tr": "土耳其语", "ar": "阿拉伯语", "cs": "捷克语", "sv": "瑞典语",
    "da": "丹麦语", "fi": "芬兰语", "no": "挪威语", "uk": "乌克兰语",
    "ro": "罗马尼亚语", "hu": "匈牙利语", "el": "希腊语", "hi": "印地语",
    "th": "泰语", "vi": "越南语", "id": "印尼语",
    "ms": "马来语", "tl": "菲律宾语", "zh-hant": "繁体中文", "km": "高棉语",
    "my": "缅甸语", "fa": "波斯语", "gu": "古吉拉特语", "ur": "乌尔都语",
    "te": "泰卢固语", "mr": "马拉地语", "he": "希伯来语", "bn": "孟加拉语",
    "ta": "泰米尔语", "bo": "藏语", "kk": "哈萨克语", "mn": "蒙古语",
    "ug": "维吾尔语", "yue": "粤语",
}


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
    if context:
        # Tencent's official HY-MT1.5 contextual template — documented ONLY
        # with a Chinese instruction, used verbatim because that is what the
        # model was tuned on. The 注意不需要翻译上文 clause ("do not translate
        # the preceding text") is load-bearing: without it the model
        # translates context + payload together (context echo).
        parts: "list[str]" = []
        pairs = _glossary_pairs(glossary)
        if pairs:
            # Official terminology-intervention style, placed FIRST.
            parts.append("参考下面的翻译：")
            for src, tgt in pairs:
                parts.append(f"{src} 翻译成 {tgt}")
            parts.append("")
        parts.append(context)
        # Fallback: the English target name — the mixed-language instruction
        # still works for targets outside the mapped set. A subtag the table
        # has no entry for keeps its region after the base's Chinese name
        # ("pt-BR" → 葡萄牙语（BR）), as target_name keeps it.
        zh_target = _ZH_LANG_NAMES.get((target_code or "").lower())
        if zh_target is None:
            base, _, sub = (target_code or "").partition("-")
            zh_target = lookup(_ZH_LANG_NAMES, base)
            if zh_target and sub:
                sub = sub.title() if len(sub) == 4 and sub.isalpha() \
                    else sub.upper()
                zh_target = f"{zh_target}（{sub}）"
            zh_target = zh_target or target_name
        parts.append(
            f"参考上面的信息，把下面的文本翻译成{zh_target}，"
            f"注意不需要翻译上文，也不要额外解释：")
        parts.append(text)
        return [{"role": "user", "content": "\n".join(parts)}]
    # No context: the official English XX<=>XX prompt.
    parts = []
    for src, tgt in _glossary_pairs(glossary):
        parts.append(f"with reference to: {src} -> {tgt}")
    parts.append(
        f"Translate the following segment into {target_name}, "
        f"without additional explanation.\n\n{text}")
    return [{"role": "user", "content": "\n".join(parts)}]


def _build_gemma(text, source_code, source_name, target_code, target_name,
                 context, glossary):
    # TranslateGemma's structured single turn. The source code is REQUIRED by
    # the format — an unknown source falls back to "en" HERE, in the one
    # builder whose wire format needs a code. Context and glossary are
    # silently ignored (the format has no slot for them).
    return [{"role": "user", "content": (
        f"type:text,source_lang_code:{source_code or 'en'},"
        f"target_lang_code:{target_code},text:{text}")}]


def _build_milmmt(text, source_code, source_name, target_code, target_name,
                  context, glossary):
    # Raw completion with ENGLISH language names (the MiLMMT training format).
    # An unknown source is simply left out of the prompt, never asserted to
    # be some default language the input may not actually be in.
    if not source_name:
        return (f"Translate this to {target_name}:\n"
                f"{text}\n{target_name}:")
    return (f"Translate this from {source_name} to {target_name}:\n"
            f"{source_name}: {text}\n{target_name}:")


def _build_seedx(text, source_code, source_name, target_code, target_name,
                 context, glossary):
    if not source_name:
        return (f"Translate the following sentence into "
                f"{target_name}:\n{text} <{target_code}>")
    return (f"Translate the following {source_name} sentence into "
            f"{target_name}:\n{text} <{target_code}>")


def _build_chatml(text, source_code, source_name, target_code, target_name,
                  context, glossary):
    if not source_name:
        return [{"role": "user", "content": (
            f"Translate the following text into "
            f"{target_name}. Reply with ONLY the translation.\n\n{text}")}]
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
    builder, sampling params and context size. ``uses_context`` records
    whether the builder actually renders the ``context`` argument — the
    context-free guard retry is a no-op for a greedy family whose builder
    ignores it (identical prompt, temperature 0). ``languages`` are the codes
    the family's model card names as supported (() = unknown)."""
    chat: bool
    build: "object"     # (text, source_code, source_name, target_code,
    #                      target_name, context, glossary) -> str | list[dict]
    sampling: dict = field(default_factory=dict)
    n_ctx: int = 8192
    uses_context: bool = True
    languages: "tuple[str, ...]" = ()


_GREEDY = {"temperature": 0.0}

_FAMILIES: "dict[str, Family]" = {
    # HY-MT1.5 model card: 33 languages + 5 dialects/minority languages.
    "hunyuan": Family(
        chat=True, build=_build_hunyuan, n_ctx=8192,
        sampling={"top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05,
                  "temperature": 0.7},
        languages=(
            "zh", "en", "fr", "pt", "es", "ja", "tr", "ru", "ar", "ko", "th",
            "it", "de", "vi", "ms", "id", "tl", "hi", "zh-Hant", "pl", "cs",
            "nl", "km", "my", "fa", "gu", "ur", "te", "mr", "he", "bn", "ta",
            "uk", "bo", "kk", "mn", "ug", "yue")),
    # TranslateGemma: the WMT24++ languages it was evaluated on
    # (arXiv 2601.09012, Table 4); "tl" and "zh-Hant" are chat-template keys.
    "gemma-translate": Family(
        chat=True, build=_build_gemma, n_ctx=2048, sampling=dict(_GREEDY),
        uses_context=False,
        languages=(
            "en", "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "es", "et",
            "fa", "fi", "tl", "fr", "gu", "he", "hi", "hr", "hu", "id", "is",
            "it", "ja", "kn", "ko", "lt", "lv", "ml", "mr", "nl", "no", "pa",
            "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta", "te",
            "th", "tr", "uk", "ur", "vi", "zh", "zh-Hant", "zu")),
    "milmmt": Family(
        chat=False, build=_build_milmmt, n_ctx=8192, sampling=dict(_GREEDY),
        uses_context=False,
        languages=(
            "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es",
            "fa", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "kk",
            "km", "ko", "lo", "ms", "my", "no", "nl", "pl", "pt", "ro", "ru",
            "sk", "sl", "sv", "ta", "th", "tl", "tr", "ur", "uz", "vi", "yue",
            "zh", "zh-Hant")),
    "seedx": Family(
        chat=False, build=_build_seedx, n_ctx=8192, sampling=dict(_GREEDY),
        uses_context=False,
        languages=(
            "ar", "fr", "ms", "ru", "cs", "hr", "nb", "sv", "da", "hu", "nl",
            "th", "de", "id", "no", "tr", "en", "it", "pl", "uk", "es", "ja",
            "pt", "vi", "fi", "ko", "ro", "zh")),
    "chatml": Family(
        chat=True, build=_build_chatml, n_ctx=4096, sampling=dict(_GREEDY),
        uses_context=False),
    # custom: whether context reaches the prompt depends on the template
    # having a {context} slot — translate_segments checks that itself.
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


# repo → the language codes its model card lists, captured from the
# model_info call _predownload_gguf makes anyway (never a request of its own,
# never under LOCAL_FILES_ONLY). In memory only: known after the first load.
_card_languages: "dict[str, tuple[str, ...]]" = {}


def _card_codes(card_data) -> "tuple[tuple[str, ...], int, int]":
    """card_data.language (a list or one string) → (the codes the language
    table can name, spelled its way (core.languages.canonical_code): "eng"
    and "EN" are "en", "zh_hant" is "zh-Hant"; how many listed items were
    renamed; how many dropped). "multilingual" and codes no table names
    drop, so a client never labels a language "Eng"."""
    out: "list[str]" = []
    renamed = dropped = 0
    for item in _card_items(card_data):
        code = canonical_code(item)
        if code is None:
            dropped += 1
            continue
        renamed += code != item.strip()
        if code not in out:
            out.append(code)
    return tuple(out), renamed, dropped


def _card_items(card_data) -> "list[str]":
    """card_data.language as listed: a list or one string, strings only."""
    raw = getattr(card_data, "language", None)
    if isinstance(raw, str):
        raw = [raw]
    return [i for i in raw if isinstance(i, str)] if isinstance(raw, list) else []


def languages_for(ref: str) -> "list[str] | None":
    """The target codes a model supports: the TRANSLATION_LANGUAGES override,
    else its (resolved) family table ∪ its model card, else None (unknown)."""
    override = language_codes(getattr(cfg, "TRANSLATION_LANGUAGES", ""))
    if override:
        return override
    family = _FAMILIES.get(resolve_family(ref), _FAMILIES["chatml"])
    card = _card_languages.get(_parse_model_ref(ref or "")[0], ())
    return sorted({*family.languages, *card}) or None


def _ctx_for(family: str) -> int:
    return _FAMILIES.get(family, _FAMILIES["chatml"]).n_ctx


def render_prompt(text: str, target: str, *, source: "str | None" = None,
                  model_ref: "str | None" = None,
                  family: "str | None" = None, glossary: str = "",
                  template: "str | None" = None,
                  context: str = "") -> dict:
    """Build the prompt EXACTLY as :func:`_run_completion` would — without
    importing llama_cpp or loading a model. The admin WebUI prompt lab's
    preview call. ``family`` overrides the configured/auto family (the
    UNSAVED select value); ``template`` renders the custom family with THAT
    template. Returns a JSON-shaped dict: chat families carry ``messages``,
    raw-completion families carry ``text``."""
    ref = tr_gating._translation_model_or_default(model_ref)
    fam_name = (family or "").strip().lower()
    if fam_name not in _FAMILIES:
        fam_name = "custom" if template is not None else resolve_family(ref)
    # An unknown source stays EMPTY, exactly as translate_segments passes
    # it (each builder handles that itself). Both codes take the run's fold
    # (_prompt_target, inside _build_prompt).
    fam = _FAMILIES[fam_name]
    prompt = _build_prompt(fam_name, text, (source or "").strip(), target,
                           context, glossary, template)
    out: dict = {
        "family": fam_name,
        "chat": fam.chat,
        "sampling": dict(fam.sampling),
        "n_ctx": fam.n_ctx,
        "model": ref,
        "model_loaded": ref in _models,
    }
    if isinstance(prompt, list):
        out["messages"] = prompt
    else:
        out["text"] = prompt
    return out


def _load_blocking(ref: str, device: str, family: str, download_cb=None):
    """Import llama_cpp and load the GGUF model. Runs in the default
    executor. ``download_cb(done_bytes, total_bytes)`` (optional) receives
    byte progress while the weights download."""
    # Where the GGUF lands is hf_cache.hub_cache_dir(), passed explicitly to
    # every hub call below — an HF_HOME setdefault here could not redirect
    # a download (the hub froze its cache path at import).
    # LOCAL_FILES_ONLY is a HOT setting — scope the offline env var to this
    # load and restore it after, or one offline load would poison every later
    # huggingface_hub download until a process restart. Snapshot the flag
    # ONCE: an admin can flip it during this minutes-long load, and a second
    # live read in the finally would then skip the restore (leaking
    # HF_HUB_OFFLINE=1) or clobber a value this call never set. The env var
    # only reaches subprocesses / a fresh import — the hub freezes
    # HF_HUB_OFFLINE at import — so the flag is ALSO threaded down
    # explicitly as ``offline`` (which resolves the GGUF from the local
    # cache instead of calling from_pretrained).
    offline = bool(getattr(cfg, "LOCAL_FILES_ONLY", False))
    offline_prev = os.environ.get("HF_HUB_OFFLINE")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return _load_blocking_inner(ref, device, family, download_cb,
                                    offline=offline)
    finally:
        if offline:
            if offline_prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = offline_prev


def _load_blocking_inner(ref: str, device: str, family: str,
                         download_cb=None, offline: "bool | None" = None):
    if offline is None:
        offline = bool(getattr(cfg, "LOCAL_FILES_ONLY", False))
    try:
        import llama_cpp
    except ImportError as e:
        # A broken install raises ImportError too — the log carries the real
        # cause.
        logger.error("[translate] llama_cpp import failed: %s", e)
        raise TranslationError(
            "translation dependencies are not installed on this server — "
            "pip install -r requirements-translate.txt"
        ) from e

    repo, quant = _parse_model_ref(ref)
    # Pre-download under a progress capture so the server log / jobs
    # registry / a waiting client see the multi-GB fetch move —
    # Llama.from_pretrained downloads through hf_hub_download silently.
    local_path = None
    if offline:
        # from_pretrained ALWAYS lists the repo over the network
        # (HfFileSystem.ls) and has no offline switch — resolve the
        # cached file ourselves and never touch the hub.
        local_path = _cached_gguf(repo, quant)
        if not local_path:
            raise TranslationError(
                f"translation model {ref} is not in the local HF cache "
                f"and LOCAL_FILES_ONLY is set — download it once with "
                f"LOCAL_FILES_ONLY disabled")
    else:
        try:
            local_path = _predownload_gguf(repo, quant, download_cb)
        except Exception as e:  # noqa: BLE001 — pre-download is best-effort
            if _is_hub_timeout(e):
                # The fallback below lists the same repo on the same hub
                # with NO deadline (HfFileSystem.ls): it would pin the
                # loader thread and the load lock forever — fail instead.
                # model_info runs before hf_hub_download (which would serve
                # the cache on a HEAD timeout): load a copy already on disk
                # with the no-network resolver before giving up.
                local_path = _cached_gguf(repo, quant)
                if not local_path:
                    logger.error("[translate] pre-download of %s timed "
                                 "out: %s", ref, e)
                    raise TranslationError(
                        f"the Hugging Face hub did not answer while fetching "
                        f"translation model {ref} — try again later") from e
                logger.warning("[translate] pre-download of %s timed out "
                               "(%s) — loading the cached copy", ref, e)
            else:
                # An unreachable hub (DNS, refused, hub 5xx) or a hub HTTP
                # refusal (a revoked HF_TOKEN's 401, a gated or removed repo)
                # fails fast, so the fallback below cannot hang, but its
                # listing fails the same way: load a copy already on disk
                # first, as whisper's snapshot_download does.
                if _is_hub_unreachable(e) or _is_hub_refusal(e):
                    local_path = _cached_gguf(repo, quant)
                if local_path:
                    logger.warning("[translate] pre-download of %s failed "
                                   "(%s) — loading the cached copy", ref, e)
                elif _is_hub_token_rejected(e):
                    logger.error("[translate] pre-download of %s failed: %s",
                                 ref, e)
                    raise TranslationError(
                        f"could not fetch translation model {ref}: the "
                        f"Hugging Face hub rejected HF_TOKEN as invalid or "
                        f"expired (HTTP 401)") from e
                else:
                    logger.warning(
                        "[translate] pre-download of %s failed (%s) — "
                        "falling back to llama.cpp's own downloader", ref, e)
        else:
            if local_path is None:
                # 0 or several hub files match (the quant was removed or
                # renamed upstream, or a second file now matches the glob):
                # from_pretrained lists the same files and refuses too —
                # load the copy already on disk first.
                local_path = _cached_gguf(repo, quant)
                if local_path:
                    logger.warning("[translate] no unique %s file on the hub "
                                   "for %s — loading the cached copy",
                                   _gguf_glob(quant), ref)
    try:
        if local_path:
            return llama_cpp.Llama(
                model_path=local_path,
                n_gpu_layers=(-1 if device == "cuda" else 0),
                n_ctx=_ctx_for(family),
                verbose=False,
            )
        return llama_cpp.Llama.from_pretrained(
            repo_id=repo,
            filename=_gguf_glob(quant),
            cache_dir=hf_cache.hub_cache_dir(),
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


def _hub_http_errors(name: str) -> "tuple[type, ...]":
    """The hub HTTP client's exception class `name`: huggingface_hub 2.x
    speaks httpx2, 1.x httpx, and either one may be the only one installed.
    Runs inside error classification, so a missing module must never raise."""
    classes = []
    for module in ("httpx2", "httpx"):
        try:
            classes.append(getattr(importlib.import_module(module), name))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def _is_hub_timeout(exc: BaseException) -> bool:
    """True when ``exc`` or its cause chain is an httpx timeout —
    hf_hub_download re-raises a HEAD timeout as LocalEntryNotFoundError
    (``from`` it). Only ``__cause__`` counts: the hub resumes a timed-out
    download INSIDE its except-timeout block, so a disk-full or size-check
    error of the resumed call carries a timeout as mere ``__context__``."""
    timeouts = _hub_http_errors("TimeoutException")
    return _in_cause_chain(exc, lambda e: isinstance(e, timeouts))


def _is_hub_unreachable(exc: BaseException) -> bool:
    """True when ``exc`` or its cause chain is a connection-level httpx
    error (DNS failure, connection refused, no route, …), a hub 5xx, or the
    hub's own offline mode (HF_HUB_OFFLINE=1 refuses every request).
    Same ``__cause__``-only rule as _is_hub_timeout: a disk-full or
    size-check error is no network error."""
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
    transport = _hub_http_errors("TransportError")

    def network(e: BaseException) -> bool:
        if isinstance(e, (*transport, OfflineModeIsEnabled)):
            return True
        if isinstance(e, HfHubHTTPError):
            status = getattr(getattr(e, "response", None), "status_code", 0)
            return isinstance(status, int) and status >= 500
        return False
    return _in_cause_chain(exc, network)


def _is_hub_refusal(exc: BaseException) -> bool:
    """True when ``exc`` or its cause chain is any hub HTTP error (a 401,
    a gated or missing repo): from_pretrained's listing asks the same hub
    and gets the same answer. Same ``__cause__``-only rule as above."""
    from huggingface_hub.errors import HfHubHTTPError
    return _in_cause_chain(exc, lambda e: isinstance(e, HfHubHTTPError))


def _is_hub_token_rejected(exc: BaseException) -> bool:
    """True when the cause chain holds the hub's 401 for a token it rejects.
    That one comes as a plain HfHubHTTPError; a private or missing repo's
    401 comes as RepositoryNotFoundError and says nothing about the token."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    def rejected(e: BaseException) -> bool:
        status = getattr(getattr(e, "response", None), "status_code", 0)
        return (isinstance(e, HfHubHTTPError)
                and not isinstance(e, RepositoryNotFoundError)
                and status == 401)
    return _in_cause_chain(exc, rejected)


def _in_cause_chain(exc: "BaseException | None", match) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if match(exc):
            return True
        seen.add(id(exc))
        exc = exc.__cause__
    return False


def _predownload_gguf(repo: str, quant: "str | None",
                      download_cb=None) -> "str | None":
    """Resolve the concrete .gguf filename (the same glob
    Llama.from_pretrained applies) and fetch it under a download_progress
    capture + a "download" jobs entry. Returns the local path, or None when
    the listing is empty/ambiguous (the caller then loads a cached copy, else
    lets from_pretrained refuse). Raises on listing/download failures for the
    caller to classify. Runs in the executor (blocking network I/O)."""
    import fnmatch

    from huggingface_hub import HfApi, constants as hf_constants, hf_hub_download

    from faster_whisper_backend.runtime import download_progress
    from faster_whisper_backend.core import jobs

    pattern = _gguf_glob(quant).lower()
    # model_info has no deadline by default: a stalled hub would pin the
    # loader thread (and the model's load lock) forever. Bound it like the
    # hub bounds its own metadata calls, so a stall fails the load instead.
    info = HfApi().model_info(repo, timeout=hf_constants.HF_HUB_ETAG_TIMEOUT)
    _card = getattr(info, "card_data", None)
    _kept, _renamed, _dropped = _card_codes(_card)
    _card_languages[repo] = _kept
    _listed = len(_card_items(_card))
    if _listed:
        logger.info("[translate] model card of %s: %d language(s) kept of %d "
                    "listed (%d renamed, %d dropped)", repo, len(_kept),
                    _listed, _renamed, _dropped)
    files = [s.rfilename for s in (info.siblings or [])]
    matches = sorted(f for f in files
                     if fnmatch.fnmatch(f.lower(), pattern))
    if len(matches) != 1:
        return None
    label = _STATS_PREFIX + repo + (f":{quant}" if quant else "")
    job_id = jobs.job_start("download", model=label, detail=matches[0])

    def _hook(done: int, total: int) -> None:
        jobs.job_update(job_id, **jobs.download_fields(done, total))
        if download_cb is not None:
            try:
                download_cb(done, total)
            except Exception:  # noqa: BLE001 — progress must never break us
                pass

    try:
        with download_progress.capture(label, cb=_hook) as cap:
            return hf_hub_download(repo_id=repo, filename=matches[0],
                                   cache_dir=hf_cache.hub_cache_dir(),
                                   **cap.tqdm_kwargs)
    finally:
        jobs.job_end(job_id)


def _cached_gguf(repo: str, quant: "str | None") -> "str | None":
    """Resolve the .gguf for repo[:quant] from the local HF cache without
    any network. Returns the path when exactly one cached file matches the
    same glob from_pretrained applies, else None.
    Revisions are tried like the online path resolves them: the one "main"
    points at first, then newest first; the first revision holding a match
    decides, so a stale snapshot of a re-uploaded file never wins."""
    import fnmatch

    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    pattern = _gguf_glob(quant).lower()
    try:
        info = scan_cache_dir(cache_dir=hf_cache.hub_cache_dir())
    except CacheNotFound:
        return None
    revisions = [rev for r in info.repos
                 if r.repo_type == "model" and r.repo_id == repo
                 for rev in r.revisions]
    # scan_cache_dir hands revisions over as a frozenset: sort for a pick
    # that does not change with the hash seed of each process start.
    revisions.sort(key=lambda rev: ("main" in rev.refs, rev.last_modified,
                                    rev.commit_hash), reverse=True)
    for rev in revisions:
        matches = [str(f.file_path) for f in rev.files
                   if fnmatch.fnmatch(f.file_name.lower(), pattern)]
        if matches:
            return matches[0] if len(matches) == 1 else None
    return None


def _draining(ref: str) -> bool:
    """Whether a completion on ``ref``'s cached model is still in flight
    (see _decoding)."""
    llm = _models.get(ref)
    return llm is not None and _decoding.get(id(llm), 0) > 0


def _drop_locked(ref: str) -> bool:
    """Drop one cached model. Caller holds _lock. Declines (False) while a
    job holds a lease on it or a decode on it is still draining — closing
    llama.cpp's native context under a running decode is a use-after-free
    that takes the whole process down."""
    if _active.get(ref, 0) > 0:
        logger.info("[translate] model %s is in use — eviction deferred", ref)
        return False
    # A cancelled job has already released its lease while its executor
    # thread may still be inside llm.create_chat_completion (see
    # _run_completion): a decode in flight on THIS model defers eviction
    # exactly like a held lease. A decode on another model does not.
    if _draining(ref):
        logger.info("[translate] a decode is still running — eviction of %s "
                    "deferred", ref)
        return False
    llm = _models.pop(ref, None)
    _last_used.pop(ref, None)
    _params.pop(ref, None)
    if llm is None:
        return True
    try:
        llm.close()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    model_registry.unregister_loaded_model(_STATS_PREFIX + ref)
    # llama.cpp never touches torch: the reclaim skips a never-imported one.
    device_memory.reclaim_after_unload()
    logger.info("[translate] model %s unloaded", ref)
    return True


def _trim_locked(cap: int) -> None:
    """Evict unleased models until the cache is under ``cap``. Caller holds
    _lock. Runs BEFORE a load (free VRAM before allocating) and AGAIN right
    before the post-load insert: the load itself runs outside _lock, so two
    misses on different refs both see room and would both insert."""
    while len(_models) >= cap:
        victim = next(
            (r for r in _models
             if not _active.get(r, 0) and not _draining(r)), None)
        if victim is None:
            # Every cached model is leased or still decoding — overflow
            # the cap rather than free a context under a running decode;
            # the next load re-trims (the idle evictor only drops a model
            # once it has idled past TRANSLATION_IDLE_TIMEOUT_S).
            logger.warning(
                "[translate] all %d cached models are in use — "
                "temporarily exceeding TRANSLATION_MAX_LOADED_MODELS",
                len(_models))
            break
        if not _drop_locked(victim):
            # Defensive: the victim filter mirrors _drop_locked's refusals;
            # stop rather than spin if the two ever diverge.
            break


def _load_cap() -> int:
    return max(1, int(getattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 1) or 1))


async def _get_model(ref: str, *, lease: bool = False, download_cb=None):
    """Return the cached model for ``ref``, loading (and LRU-evicting past
    TRANSLATION_MAX_LOADED_MODELS) on a miss. ``download_cb(done, total)``
    receives byte progress when the miss also has to download weights."""
    device = _resolve_device()
    family = resolve_family(ref)
    params = (device, _ctx_for(family))

    def _hit(llm):
        _models.move_to_end(ref)
        _last_used[ref] = time.monotonic()
        model_registry.touch_loaded_model(_STATS_PREFIX + ref)
        if lease:
            _active[ref] = _active.get(ref, 0) + 1
        return llm

    # Lock-free cache hit, read into a LOCAL so a concurrent _drop_locked
    # cannot pop it between the check and the return. No _lock here: it
    # would block every warm hit for the whole of another job's cold load.
    llm = _models.get(ref)
    if llm is not None and (_params.get(ref) == params
                            or _active.get(ref, 0) > 0):
        # A LEASED model with stale load params still serves — closing
        # llama.cpp's context under a running decode is a use-after-free;
        # the first unleased call after the admin edit re-keys it below.
        return _hit(llm)

    # Miss (or stale load params): serialize per REF so two jobs never load
    # the same model twice, while warm hits on other refs stay unblocked.
    load_lock = _loading.setdefault(ref, LoopLock())
    async with load_lock:
        llm = _models.get(ref)
        if llm is not None:
            if _params.get(ref) == params or _active.get(ref, 0) > 0:
                return _hit(llm)   # a waiter arrived after another's load
            # Load params changed (TRANSLATION_DEVICE, or the prompt family's
            # n_ctx) and no lease pins the model: drop it and reload.
            async with _lock:
                if not _drop_locked(ref):
                    return _hit(llm)   # decode draining — keep serving, re-key next time
        async with _lock:
            _trim_locked(_load_cap())
        # The VRAM delta is only attributable to THIS load when no other
        # translation load overlapped the before/after window (only the
        # per-ref lock is held here); otherwise publish None, as the
        # CPU / no-NVML path already does, rather than a fabricated number.
        global _loads_in_flight, _load_overlaps
        overlaps_before = _load_overlaps
        _loads_in_flight += 1
        if _loads_in_flight > 1:
            _load_overlaps += 1
        vram_before = system_stats.gpu_mem_used_bytes()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        # A cancellation that lands on the load await (a dropped /v1/translate
        # connection, a batch run's disconnect) cannot stop llama.cpp on the
        # executor thread. Released early, the per-ref lock would let the next
        # request start a second load of the same model next to it (twice the
        # peak VRAM, a polluted VRAM delta) — so the build is finished under
        # the lock, cached unleased, and the cancellation re-raised once it is
        # registered (CancelLatch, as in transcription.models' _construct).
        latch = CancelLatch()
        fut = loop.run_in_executor(
            None, _load_blocking, ref, device, family, download_cb)
        try:
            llm = await latch.wait(fut)
        except Exception as e:
            if latch.cancelled is not None:
                raise latch.cancelled from e
            raise
        finally:
            _loads_in_flight -= 1
        vram_after = system_stats.gpu_mem_used_bytes()
        vram = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None
            and _load_overlaps == overlaps_before) else None
        load_secs = time.perf_counter() - t0
        # Cached BEFORE the register await: a cancellation landing on it
        # cannot stop the registry write already on the thread, and a
        # registry entry with no cached model is a phantom that only
        # _drop_locked would ever unregister.
        async with _lock:
            _trim_locked(_load_cap())
            _models[ref] = llm
            _params[ref] = params
            _last_used[ref] = time.monotonic()
            # A cancelled caller never receives the model: it stays unleased.
            leased = lease and latch.cancelled is None
            if leased:
                _active[ref] = _active.get(ref, 0) + 1
        # Shielded and awaited to the end under the per-ref lock even when
        # cancelled: a cancelled to_thread future drops a write still queued
        # on the default executor (a cached model /stats never lists), and
        # the phantom check below must run after the write has landed.
        reg_fut = asyncio.ensure_future(asyncio.to_thread(
            model_registry.register_loaded_model,
            _STATS_PREFIX + ref, vram, device, "gguf", load_secs))
        try:
            await latch.wait(reg_fut)
        except BaseException as e:
            if leased:
                _release_model(ref)   # the caller never receives the model
            if latch.cancelled is not None and \
                    not isinstance(e, asyncio.CancelledError):
                raise latch.cancelled from e
            raise
        if latch.cancelled is not None and leased:
            _release_model(ref)   # cancelled during the register
        if _models.get(ref) is not llm:
            # An unleased load (preload / load_unleased) was trimmed or
            # evicted while its register ran on the thread: _drop_locked's
            # unregister was a no-op then, so this entry would be a phantom
            # nothing ever removes. The held per-ref lock rules out a newer
            # instance of the same ref having registered meanwhile.
            model_registry.unregister_loaded_model(_STATS_PREFIX + ref)
        logger.info("[translate] model %s loaded on %s in %.1fs",
                    ref, device, load_secs)
        model_registry.record_load_stat(_STATS_PREFIX + ref, load_secs)
        if latch.cancelled is not None:
            raise latch.cancelled
        return llm


def _release_model(ref: str) -> None:
    """Release a lease taken by ``_get_model(..., lease=True)`` and restart
    the model's idle clock (a long job must not be evicted the moment it
    ends because its LOAD time stamp aged past the timeout). Synchronous and
    lock-free — plain dict ops are atomic on the single event loop, and an
    awaiting release could be abandoned by a cancellation delivered while
    suspended on the lock, leaking the lease and pinning the model against
    eviction forever (same shape as transcription.models._release_model_lease)."""
    release_count(_active, ref)
    if ref in _models:
        _last_used[ref] = time.monotonic()


async def drop_models() -> None:
    """Evict every cached model (admin eviction / shutdown)."""
    async with _lock:
        for ref in list(_models):
            _drop_locked(ref)


# =============================================================================
# Residency API (runtime.preload's view of this cache)
# =============================================================================
# The same questions transcription.models answers for the whisper cache. Each
# reads the module globals at call time, so a test that rebinds _models /
# _active / _loads_in_flight still steers it.

def is_resident(ref: str) -> bool:
    """Is ``ref`` loaded right now?"""
    return ref in _models


def load_in_progress() -> bool:
    """True while any cold load is inside _load_blocking. A cold load runs
    OUTSIDE _lock (only the per-ref _loading lock is held) and is not yet in
    _models, so a cap check alone would admit a second multi-GB load beside
    it: both VRAM measurements void, and with cap 1 the later insert trims
    the other."""
    return _loads_in_flight > 0


def cache_full() -> bool:
    """Is the cache at TRANSLATION_MAX_LOADED_MODELS? A load past that point
    goes through _trim_locked, which drops the first UNLEASED ref without
    consulting the warm predicate."""
    return len(_models) >= _load_cap()


def idle_peer(exclude: str,
              is_warm: "Callable[[str], bool]") -> "str | None":
    """The first cached ref, in LRU order, other than ``exclude`` that no job
    holds a lease on, that no cancelled job still decodes on (_draining —
    _drop_locked declines it, and _trim_locked would drop the NEXT, maybe
    warm, peer instead) and for which ``is_warm(ref)`` is false — a model
    that could be dropped to make room — or None."""
    for ref in _models:
        if ref == exclude or _active.get(ref, 0) or _draining(ref):
            continue
        if not is_warm(ref):
            return ref
    return None


def placement() -> "tuple[str, str]":
    """(device, compute_type) a load registers under in the loaded-model
    registry and the size ledger."""
    return (_resolve_device(), "gguf")


async def load_unleased(ref: str) -> None:
    """Load ``ref`` WITHOUT a lease: the model stays evictable by _trim_locked
    and the idle evictor the moment a real job needs the memory."""
    await _get_model(ref)


async def evict(ref: str) -> bool:
    """Drop one cached model under _lock. Declines (False) while a job holds
    a lease on it or a decode is still draining (see _drop_locked)."""
    async with _lock:
        return _drop_locked(ref)


async def idle_evictor_loop() -> None:
    """Unload models idle for TRANSLATION_IDLE_TIMEOUT_S seconds. Reads the
    timeout live (an admin edit applies without restart), like
    diarization.idle_evictor_loop. Started/cancelled by lifespan."""
    async def _tick() -> None:
        timeout = int(getattr(cfg, "TRANSLATION_IDLE_TIMEOUT_S", 0) or 0)
        if timeout <= 0 or not _models:
            return
        now = time.monotonic()
        async with _lock:
            for ref, last in list(_last_used.items()):
                # A warm lease (a live preload plan still expects this
                # model) suspends the idle clock — never a job lease,
                # which _drop_locked already honours. Inverted through
                # model_registry so preload can reach us without either
                # module importing the other.
                if model_registry.is_warm(_STATS_PREFIX + ref):
                    continue
                # A leased model mid-job reads as stale every tick; skip
                # it here so the tick does not log a deferred eviction
                # for it (_release_model restarts its clock), as
                # transcription.models._idle_evictor does.
                if _active.get(ref, 0) > 0:
                    continue
                if ref in _models and now - last >= timeout:
                    _drop_locked(ref)

    await run_every(lambda: _EVICTOR_WAKE_S, _tick, logger,
                    "[translate] idle evictor error: %s")


# =============================================================================
# Inference seam
# =============================================================================

def _complete(llm, family: str, prompt_or_msgs, max_tokens: int) -> str:
    """The single blocking completion call — chat families via
    create_chat_completion, raw families via a plain llm() call — holding
    ``_infer_mutex`` (llama.cpp is not thread-safe). Runs in the executor.
    Tests stub THIS function to exercise everything above it without
    llama_cpp installed.

    A reply cut off at ``max_tokens`` (finish_reason "length") comes back
    as "": the guards then see an empty output and the retry /
    keep-original path runs, instead of a shortened translation passing
    every guard and being spread over the group's cues."""
    fam = _FAMILIES[family]
    kwargs = dict(fam.sampling)
    with _infer_mutex:
        if fam.chat:
            out = llm.create_chat_completion(
                messages=prompt_or_msgs, max_tokens=max_tokens, **kwargs)
            choice = out["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        else:
            out = llm(prompt_or_msgs, max_tokens=max_tokens, **kwargs)
            choice = out["choices"][0]
            text = (choice.get("text") or "").strip()
    if choice.get("finish_reason") == "length":
        logger.warning("[translate] output truncated at max_tokens=%d "
                       "(%s, %d chars) — discarded", max_tokens, family,
                       len(text))
        return ""
    return text


# =============================================================================
# Fluent-mode helpers (pure)
# =============================================================================

# Sentence-final punctuation that closes a fluent-mode group (the full-width
# forms are what whisper writes for Chinese / Japanese; the danda for
# Hindi / Marathi / Nepali, ။ for Burmese, ។ ៕ for Khmer, ؟ for Arabic,
# ۔ for Urdu, ։ for Armenian, ። for Amharic).
# No Greek ";": a Latin semicolon is no sentence end.
_SENTENCE_FINAL = ".!?…。！？।॥။។៕؟۔։።‼⁇⁈⁉"
# Closing quotes / brackets that may follow the sentence end ('."', '?»', '」',
# German '.“' / '.‘', Swiss '.›'). Source side only: in Chinese “ opens.
_CLOSERS = "\"'”’»«“‘›‹」』）)]】}］｝》〉"
_MAX_GROUP_SEGMENTS = 6


def _merge_sentences(segments: "list[dict]") -> "list[list[int]]":
    """Merge consecutive segments into sentence groups: accumulate until a
    segment's text ends with sentence-final punctuation (closing quotes and
    brackets after it allowed) or the group already spans
    _MAX_GROUP_SEGMENTS segments. A speaker change closes the group too, so
    the proportional redistribution never moves one speaker's words into the
    other's cue. Returns groups of segment indices."""
    groups: "list[list[int]]" = []
    current: "list[int]" = []
    # The group's last KNOWN speaker: an unlabelled cue between A and B must
    # not hide the change.
    prev = None
    for i, seg in enumerate(segments):
        spk = seg.get("speaker")
        if current and prev and spk and prev != spk:
            groups.append(current)
            current = []
        current.append(i)
        prev = spk or prev
        # Spaces too: French typography puts one before the » ('Oui. »').
        # One index scan: re-slicing per closer was O(n²) on a long run.
        text = seg.get("text") or ""
        end = len(text)
        while end and (text[end - 1] in _CLOSERS or text[end - 1].isspace()):
            end -= 1
        if (end and text[end - 1] in _SENTENCE_FINAL) or \
                len(current) >= _MAX_GROUP_SEGMENTS:
            groups.append(current)
            current = []
            prev = None
    if current:
        groups.append(current)
    return groups


# Clause punctuation an unspaced (Han/kana, Burmese ၊ ။, Khmer ។ ៕)
# translation can be cut after — no ASCII "." or ",", which also sit inside
# numbers ("3.5", "1,000"). Thai and Lao mark clauses with a space.
_UNSPACED_CUT_PUNCT = frozenset("，。、！？；：!?;၊။។៕")
# The closers a cut after such a mark always takes along: no straight quote
# (which may open the next clause) and no « (which opens one in French).
_CUT_CLOSERS = frozenset("”’»」』）)]】")
_NON_BLANK_RE = re.compile(r"\S")
_UNSPACED_RE = re.compile(
    r"[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u9fff]")
# Thai/Lao vowels written before their consonant, and the Myanmar/Khmer
# stacking marks: each belongs to the character AFTER it, so no raw cut
# may fall right behind one.
_JOINS_NEXT = frozenset("\u0e40\u0e41\u0e42\u0e43\u0e44"
                        "\u0ec0\u0ec1\u0ec2\u0ec3\u0ec4\u1039\u17d2")


def _redistribute(group_texts: "list[str]", translated: str) -> "list[str]":
    """Split one group's translated text back across its member segments,
    proportionally by each member's share of the group's source char length,
    cutting at the word boundary (space) nearest each proportional cut.
    Unspaced (Chinese/Japanese/Thai/Lao/Khmer/Burmese) text has no word
    boundaries — its spaces sit around a Latin name or between phrases —
    so it is cut after the nearest clause punctuation, else the nearest
    space, and only when that boundary lies within a quarter of the member's
    share of the ideal cut; else at the raw proportional position (stepped
    past combining marks and any _JOINS_NEXT character). Members whose
    source is blank get an empty piece; a reply with fewer character
    clusters than members is unsplittable and comes back with blank
    pieces, which the caller reverts."""
    n = len(group_texts)
    if n <= 1:
        return [translated.strip()] if n == 1 else []
    live = [i for i, t in enumerate(group_texts) if t.strip()]
    if len(live) < n:
        if not live:
            return [""] * n
        sub = _redistribute([group_texts[i] for i in live], translated)
        out = [""] * n
        for i, piece in zip(live, sub):
            out[i] = piece
        return out
    translated = translated.strip()
    if not translated:
        return [""] * n
    spaces = [i for i, ch in enumerate(translated) if ch == " "]
    unspaced = len(_UNSPACED_RE.findall(translated)) / len(translated) > 0.3
    # Unspaced: a cut lands AFTER the punctuation, which stays on its clause
    # together with any closing quotes / brackets after it ('。”', '。」'),
    # so the next cue never opens on a stray closer. A straight quote is
    # taken along only when it closes an open one: after '：' or '。' it as
    # often opens the next clause ('他说："你好"').
    punct: "list[int]" = []
    if unspaced:
        quotes = {'"': 0, "'": 0}
        for i, ch in enumerate(translated):
            if ch in quotes:
                quotes[ch] += 1
            if ch not in _UNSPACED_CUT_PUNCT:
                continue
            seen = dict(quotes)
            i += 1
            while i < len(translated):
                c = translated[i]
                if c in seen and seen[c] % 2:
                    seen[c] += 1
                elif c not in _CUT_CLOSERS:
                    break
                i += 1
            punct.append(i)
    tiers = [punct, spaces] if unspaced else [spaces]
    total = sum(len(t) for t in group_texts) or 1

    def _rooms(tier: "list[int]") -> "dict[int, int]":
        # How many more cuts this tier can still take after a cut at each of
        # its boundaries, each leaving a non-blank slice. One backward pass:
        # the next cut after s is the first boundary past the first non-blank
        # char at or after s. (Rescanning the tier per candidate was O(B²)
        # per cut, which froze the event loop on a long reply.)
        rooms = [0] * len(tier)
        text_at, end = len(translated), len(translated)
        for k in range(len(tier) - 1, -1, -1):
            m = _NON_BLANK_RE.search(translated, tier[k], end)
            if m:
                text_at = m.start()
            end = tier[k]
            j = bisect.bisect_right(tier, text_at)
            if j < len(tier) and tier[j] < len(translated):
                rooms[k] = rooms[j] + 1
        return dict(zip(tier, rooms))

    rooms = [_rooms(tier) for tier in tiers]
    # Positions of the non-blank chars: the tail is reserved in THESE, since
    # a reserved char that is a space ("beaucoup !") still left a blank cue.
    # Only where a character cluster starts: a cut clamped onto a tone mark
    # ('ได|้') made the last cue that lone mark.
    non_blank = [k for k, ch in enumerate(translated)
                 if not ch.isspace()
                 and unicodedata.category(ch) not in ("Mn", "Mc", "Me")
                 and not (k and translated[k - 1] in _JOINS_NEXT)]
    if len(non_blank) < n:
        # Fewer clusters than members: any split cuts inside one ('ดี' over
        # two cues gave 'ด' + a lone 'ี'). A blank piece makes the caller
        # revert the group.
        return [translated] + [""] * (n - 1)

    cuts: "list[int]" = []
    acc = 0
    prev = 0
    for i, t in enumerate(group_texts[:-1]):
        acc += len(t)
        ideal = len(translated) * acc / total
        reach = (len(translated) * len(t) / total / 4 if unspaced
                 else float("inf"))
        # Only boundaries strictly PAST the previous cut are candidates — two
        # ideals snapping to the same space would otherwise clamp into an
        # empty member slice (a cue with source text rendering blank). The
        # slice must hold text too: in a double space ("know.  Yes") the
        # second space is past a cut on the first, yet slices to " ".
        # A boundary that leaves too few for the members still to come
        # would force those to a raw cut through a word, so it is dropped
        # whenever the tier holds enough boundaries for every member — but
        # only among those within reach: dropping the one in-reach clause
        # mark for an out-of-reach one would itself force a raw cut.
        cut = None
        for tier, room in zip(tiers, rooms):
            cands = [s for s in tier
                     if s > prev and abs(s - ideal) <= reach
                     and translated[prev:s].strip()]
            cands = [s for s in cands if room[s] >= n - 2 - i] or cands
            if cands:
                cut = min(cands, key=lambda s: abs(s - ideal))
                break
        snapped = cut is not None
        if cut is None:
            cut = int(round(ideal))
        # Forced forward, and reserving one cluster start per REMAINING
        # member so the leading cuts can never eat the tail.
        limit = non_blank[-(n - 1 - i)]
        cut = max(prev + 1, min(cut, limit))
        if not snapped:
            # No word boundary left: raw proportional cut, pushed past any
            # whitespace-only slice (which .strip() would empty).
            while cut < len(translated) and not translated[prev:cut].strip():
                cut += 1
            # Nor inside a character cluster: a cue that opens on a vowel or
            # tone mark shows it on a dotted circle, and the word before it
            # loses that mark (Thai, Khmer, Burmese, decomposed accents).
            while cut < limit and (
                    unicodedata.category(translated[cut]) in ("Mn", "Mc", "Me")
                    or translated[cut - 1] in _JOINS_NEXT):
                cut += 1
        cuts.append(cut)
        prev = cut
    pieces: "list[str]" = []
    start = 0
    for c in cuts:
        pieces.append(translated[start:c].strip())
        start = c
    pieces.append(translated[start:].strip())
    return pieces


# Han and kana plus CJK / full-width punctuation: where two of these meet,
# the text has no space. Han/kana ranges as in captures/samples.py's
# _CJK_SCRIPT, widened to the whole CJK-symbol and full-width blocks.
# Thai, Lao, Khmer and Burmese stay out: they put a space between
# sentences, so gluing would eat it (a raw cut inside a word there still
# rejoins with a space).
_GLUED_SCRIPT_RE = re.compile(
    "[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\uf900-\ufaff\uff01-\uff9f\U00020000-\U0003134f]")
# Western quotes and brackets a zh / ja piece can open or close on (「」（）
# are already in _GLUED_SCRIPT_RE's ranges).
_JOIN_MARKS = "“”‘’«»()[]\"'"
# A digit or Latin letter: in a zh / ja target it glues to a Han / kana
# neighbour ('我买了3' | '个苹果').
_DIGIT_LATIN_RE = re.compile("[0-9A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f]")


def join_translations(parts: "list[str]", target: "str | None" = None) -> str:
    """Join one target's per-segment translations into its full text: " "
    between pieces, but nothing where a Han / kana piece (or one ending in
    "。" and its closing quote) meets one opening in Han / kana — a fluent
    group cut back across its cues ('我今天去商' | '店买一些面包。') must not
    come back with a space inside a word. Quotes and brackets on either side
    of the cut are looked past: '他说：' | '“走吧。”' is one sentence too.
    When ``target`` is zh / ja / yue, a digit or Latin letter meeting Han /
    kana glues as well ('我买了3' | '个苹果'); any other target keeps that
    space — an English text quoting a Chinese original reads 'in 北京 today'."""
    han_target = _target_base(target) in _HAN_SHARING_TARGETS

    def _kind(c: str) -> str:
        if _GLUED_SCRIPT_RE.match(c):
            return "glued"
        return "latin" if han_target and _DIGIT_LATIN_RE.match(c) else ""

    out = ""
    for p in (s.strip() for s in parts):
        if not p:
            continue
        if out:
            # A side made of marks only (a lone '“' cue) lets the other
            # side decide, or '他说：' | '“' | '走开。”' gets two spaces.
            # Latin | Latin never glues: 'Mü' | 'ller' is a cut word no
            # script rule can tell from two words.
            kinds = {_kind(c) for c in (out.rstrip(_JOIN_MARKS)[-1:],
                                        p.lstrip(_JOIN_MARKS)[:1]) if c}
            out += "" if "glued" in kinds and "" not in kinds else " "
        out += p
    return out


# =============================================================================
# Faithful-mode numbered-list helpers (pure)
# =============================================================================

_NUMBERED_INSTRUCTION = (
    "Keep the numbered list format: reply with the same numbered list, one "
    "translated line per item.")


def _build_numbered(texts: "list[str]") -> str:
    # One line per item is the contract the model echoes back: an embedded
    # newline in a segment would split it into two numbered candidates.
    return "\n".join(f"{i + 1}. {' '.join(t.split())}"
                     for i, t in enumerate(texts))


def _parse_numbered(reply: str, n: int) -> "list[str] | None":
    """Parse a numbered-list reply back into n strings; None on any count
    mismatch (missing or unparseable lines, an index past n or a repeated
    one: an item split in two and renumbered would shift every later cue)
    — the caller then halves the batch size and retries. Chatter before
    the first item, and after the n-th once a blank line sets it apart,
    is skipped, and so is a Markdown code fence anywhere; any other
    unnumbered line is an item wrapped over two lines, whose cut first
    half would pass as the whole translation, so it is a mismatch too."""
    found: "dict[int, str]" = {}
    after_item = False
    for line in (reply or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[.):]\s*(.*)", line)
        if not m:
            # A fence (a chat model wrapping its list in ```) is markup,
            # never the second half of an item: it sets items apart like a
            # blank line.
            if (line.strip() and not line.lstrip().startswith(("```", "~~~"))
                    and found and (len(found) < n or after_item)):
                return None
            after_item = False
            continue
        # No batch holds a million items; a longer digit run is garbage,
        # and past 4300 digits int() raises instead of reporting a mismatch.
        if len(m.group(1)) > 6:
            return None
        idx = int(m.group(1))
        if not 1 <= idx <= n or idx in found:
            return None
        found[idx] = m.group(2).strip()
        after_item = True
    if len(found) != n:
        return None
    return [found[i + 1] for i in range(n)]


# =============================================================================
# Output guards (pure)
# =============================================================================

# Any 12+ char substring repeated 4+ times = a generation loop. The pattern
# is superlinear on NON-matching text (the common, clean case: ~228 ms for
# 4 kB, quadratic beyond), so the guard scans only the leading window — a
# real generation loop is always visible in the first couple of kB.
_REPETITION_RE = re.compile(r"(.{12,}?)(?:\1){3,}", re.DOTALL)
_REPETITION_SCAN_CHARS = 2000
# Length-ratio guard bounds, applied only past the absolute floor below.
_RATIO_LO, _RATIO_HI = 0.4, 3.0
_RATIO_FLOOR_CHARS = 20


_CJK_TARGETS = {"zh", "ja", "ko", "yue"}
# The CJK targets written with Han characters: Korean output is Hangul, so a
# Han clause copied into a "ko" target is untranslated.
_HAN_SHARING_TARGETS = {"zh", "ja", "yue"}
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
# Kana and Hangul only: no two different CJK targets share these (unlike Han).
_KANA_HANGUL_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")


def _target_base(target: "str | None") -> str:
    """``target``'s base subtag, folded the way same_language folds: an API
    client may send the ISO 639-3 spelling ("jpn", "cmn")."""
    if not target:
        return ""
    return (canonical_code(target) or target).strip().lower().split("-")[0]


def _target_is_cjk(target: "str | None") -> bool:
    """Whether ``target`` names a CJK language (see _target_base)."""
    return _target_base(target) in _CJK_TARGETS


def _digits(text: str) -> "Counter[int]":
    """Digit VALUES in ``text``: a model writing the target script's native
    digits ("۱۶", "１６") keeps the number, not the characters."""
    return Counter(unicodedata.decimal(c) for c in re.findall(r"\d", text))


def _ratio_bounds(src: str, target: "str | None") -> "tuple[float, float]":
    """Char-count ratio bounds, script-aware: CJK text carries ~2-4x the
    information per character of Latin text, so the default [0.4, 3.0] band
    would reject nearly every valid Latin→CJK translation (and vice versa)."""
    tgt_cjk = _target_is_cjk(target)
    cjk_chars = len(_CJK_CHAR_RE.findall(src))
    src_cjk = len(src) > 0 and cjk_chars / len(src) > 0.3
    if tgt_cjk and not src_cjk:
        return (0.12, _RATIO_HI)
    if src_cjk and not tgt_cjk:
        return (_RATIO_LO, 9.0)
    return (_RATIO_LO, _RATIO_HI)


def _guard_reason(src: str, out: str, *,
                  target: "str | None" = None,
                  source_known: bool = True) -> "str | None":
    """Reason string when a translated segment fails a sanity guard, else
    None. Guards: empty output; length ratio outside script-aware bounds
    (only once either side reaches the 20-char floor); digit mismatch
    (a source digit vanished while the output still carries digits — a
    changed number; number-word normalization in EITHER direction is legal,
    spoken-transcript MT constantly writes "sechzehn" as "16" and an exact
    multiset check rejected correct translations en masse; digits compare
    by value, so native-script digits are the same number); verbatim input
    copy of two words or more (each CJK character counts as one, as an
    unspaced clause is a single \\w+ run; for a Han-writing target — zh,
    ja, yue — Han text is shared and does not count, only kana and Hangul
    do) when the target differs from the source — skipped when the source
    language is unknown (``source_known=False``: a target in the text's own
    language is not short-circuited then, and the unchanged text is its
    correct translation); repetition loop not present in the source (a
    repeated chant translates to a repeated chant — but only as many looped
    runs as the source has: a chant does not excuse a loop elsewhere in the
    same output)."""
    s = (src or "").strip()
    o = (out or "").strip()
    if not o:
        return "empty output"
    if s and (len(s) >= _RATIO_FLOOR_CHARS or len(o) >= _RATIO_FLOOR_CHARS):
        lo, hi = _ratio_bounds(s, target)
        ratio = len(o) / len(s)
        if not (lo <= ratio <= hi):
            return f"length ratio {ratio:.2f} outside [{lo}, {hi}]"
    out_digits = _digits(o)
    if out_digits and (_digits(s) - out_digits):
        return "digit mismatch"
    # A one-word line ("OK.", "Netflix.", "Hm.") often translates to
    # itself; only a copied phrase is evidence of an untranslated output.
    # Whitespace is compared collapsed: a numbered batch prompt carries each
    # segment with its newlines and runs of spaces folded. For a Han-writing
    # target (zh, ja, yue) Han text counts neither as words nor toward the
    # CJK-character floor, which counts kana and Hangul only: zh→zh-Hant
    # ("大家好，我是小明。") and ja→zh ("日本、東京。") legitimately share
    # Han text — punctuation included — but an unchanged kana or Hangul
    # clause is never another CJK language's translation. A Korean target
    # counts Han too: Korean output is Hangul.
    shares_han = _target_base(target) in _HAN_SHARING_TARGETS
    if (source_known and " ".join(o.split()) == " ".join(s.split())
            and (len(re.findall(r"\w+", _HAN_RE.sub(" ", s) if shares_han
                                else s)) >= 2
                 or len((_KANA_HANGUL_RE if shares_han
                         else _CJK_CHAR_RE).findall(s)) >= 2)):
        return "output copies input"
    if _REPETITION_RE.search(o[:_REPETITION_SCAN_CHARS]) and (
            len(_REPETITION_RE.findall(o[:_REPETITION_SCAN_CHARS]))
            > len(_REPETITION_RE.findall(s[:_REPETITION_SCAN_CHARS]))):
        return "repetition loop"
    return None


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


def _prompt_target(code: str) -> str:
    """A language code as the prompt spells it — ONE fold for the run
    (_run_completion) and the prompt lab's preview (render_prompt), for the
    target and the source alike. The lab passes its free-form codes raw
    ("jpn", "zh_hant"): folded to the table's spelling ("ja", "zh-Hant") the
    code-using builders (Gemma, Seed-X) and the language name match what the
    run sends. A script entry the table names wins over any region after it
    ("zh_Hant_TW" → "zh-Hant", Traditional Chinese, as canonical_code folds
    it), and a Traditional-writing Chinese region is that script too ("zh-TW"
    → "zh-Hant": the bare "zh" name gets Simplified); otherwise
    _normalise_code: a script subtag the table does not list ("sr-Latn") is
    still the target's script and must reach the prompt."""
    canon = canonical_code(code) if code else None
    if canon and "-" in canon and canon.lower() in _NAMES:
        return canon
    base, _, region = (canon or "").lower().partition("-")
    if base == "zh" and region in ZH_HANT_REGIONS:
        return "zh-Hant"
    return _normalise_code(code) or code


def _build_prompt(family: str, text: str, source_code: str, target_code: str,
                  context: str, glossary: str,
                  template: "str | None" = None):
    """The one prompt build for the run (_run_completion) and the prompt
    lab's preview (render_prompt). ``template`` (custom family only)
    renders THAT template instead of cfg.TRANSLATION_PROMPT_TEMPLATE."""
    source_code = _prompt_target(source_code or "")
    target_code = _prompt_target(target_code)
    # The target's name keeps its region ("pt-BR" → "Portuguese (BR)"): a
    # region is a target of its own (language_codes), so "pt-BR" and "pt-PT"
    # must not get byte-identical prompts.
    if template is not None and family == "custom":
        return [{"role": "user", "content": _render_custom_template(
            template, text, language_name(source_code),
            language_label(target_code), context, glossary)}]
    return _FAMILIES[family].build(
        text, source_code, language_name(source_code), target_code,
        language_label(target_code), context, glossary)


_CONTEXT_OVERFLOW = "input too long for the model's context window"


async def _run_completion(llm, family: str, text: str, source_code: str,
                          target_code: str, context: str, glossary: str,
                          template_override: "str | None" = None) -> str:
    """Build the family prompt for ``text`` and run one completion in the
    executor (via the _complete seam). ``template_override`` (custom family
    only) renders THAT template instead of cfg.TRANSLATION_PROMPT_TEMPLATE —
    the admin template-test path, never persisted."""
    prompt = _build_prompt(family, text, source_code, target_code, context,
                           glossary, template_override)
    # Context counts toward the budget: a context-echoing model (the guard
    # catches it) must not ALSO be truncated mid-sentence into a secondary
    # guard failure.
    max_tokens = (len(text) + len(context or "")) // 2 + 256
    loop = asyncio.get_running_loop()
    # Count this completion against the model until its thread is done (see
    # _decoding). Whoever takes ``claimed`` first settles the count: the
    # thread when its decode starts, or this coroutine when the job was
    # cancelled (or the submit failed) before that — that thread then skips
    # the decode instead of running it on a model already evicted.
    key, claimed = id(llm), threading.Lock()

    def _settle() -> None:
        with _decoding_mutex:
            release_count(_decoding, key)

    def _decode() -> str:
        # Claimed only once _infer_mutex is ours: a job cancelled while its
        # thread still waits behind another decode skips its own, instead
        # of queueing a full doomed decode ahead of every later request.
        with _infer_mutex:
            if not claimed.acquire(blocking=False):
                return ""
            try:
                return _complete(llm, family, prompt, max_tokens)
            finally:
                _settle()

    async with _infer_gate:
        with _decoding_mutex:
            _decoding[key] = _decoding.get(key, 0) + 1
        try:
            return await loop.run_in_executor(None, _decode)
        except asyncio.CancelledError:
            logger.warning("translation cancelled — executor thread will "
                           "complete on its own (%s)", family)
            raise
        except TranslationError:
            raise
        except Exception as e:  # noqa: BLE001 — the failure contract
            # llama.cpp raises ValueError for a prompt past n_ctx; a CUDA
            # OOM or any other runtime error lands here too. Raw text stays
            # in the log (module docstring: failure contract).
            if isinstance(e, ValueError) and \
                    "exceed context window" in str(e):
                logger.warning("[translate] prompt too long (%s): %s",
                               family, e)
                raise _ContextOverflow(_CONTEXT_OVERFLOW) from None
            logger.error("[translate] inference failed (%s): %s", family, e)
            raise TranslationError(
                "translation inference failed (input too long for the "
                "model's context window, or a runtime error) — see the "
                "server log") from None
        finally:
            if claimed.acquire(blocking=False):
                _settle()


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
    family_override: "str | None" = None,
    download_cb=None,
) -> "tuple[list[dict[str, str]], list[str], dict]":
    """Translate ``segments`` (``[{"text": str, "speaker": str|None}, …]``)
    into every language in ``targets``.

    Returns ``(per_segment_translations, warnings, meta)``:
    one ``{target_code: translated_text}`` dict per INPUT segment (cue count
    and timing stay the caller's — this function only returns per-segment
    strings), client-safe warning strings for segments kept untranslated,
    and ``meta = {"model", "source", "mode", "kept"}`` where ``kept`` maps
    each segment INDEX whose guard fallback kept the SOURCE text to the list
    of target codes it was kept for (``dict[int, list[str]]`` — empty when
    every segment translated cleanly; the same-language verbatim copy is a
    legitimate translation, never flagged).

    ``model_ref`` empty → ``TRANSLATION_DEFAULT_MODEL``; both empty →
    :class:`TranslationError`, even for a same-language-only request. A
    target equal to ``source_lang`` is copied verbatim without a model call.
    ``mode``: see the module docstring.
    ``context_segments`` overrides ``TRANSLATION_CONTEXT_SEGMENTS`` when not
    None. ``progress_cb(done_fraction, step_str, last_text, target=...,
    target_progress=...)`` fires after each batch — ``last_text`` is the tail
    (≤160 chars) of the last completed translation, for live run panels;
    ``target`` names the language being translated and ``target_progress``
    (0..1) how far that language is, so a panel can draw one bar per target
    on top of the global fraction;
    ``cancel_check`` (no-arg, truthy = abort) is polled between batches and
    raises :class:`TranslationCancelled`. ``template_override`` (the admin
    template-test path) forces the ``custom`` family and renders THAT
    template for this call only — cfg stays untouched. ``family_override``
    (the admin prompt lab) forces a specific family for this call.
    ``download_cb(done_bytes, total_bytes)`` receives byte progress when a
    cold model also has to download its weights.
    """
    ref = tr_gating._translation_model_or_default(model_ref)
    if not ref:
        raise TranslationError(
            "no translation model configured (TRANSLATION_DEFAULT_MODEL)")
    # family_override is the admin prompt lab's UNSAVED family select.
    if family_override and family_override in _FAMILIES:
        family = family_override
    elif template_override is not None:
        family = "custom"
    else:
        family = resolve_family(ref)
    if family != "custom":
        template_override = None   # a stale textarea must not leak in
    # An unknown source stays EMPTY: the builders that name the source
    # language in prose (milmmt/seedx/chatml) drop the clause rather than
    # falsely asserting the input is English; only _build_gemma, whose wire
    # format requires a code, defaults to "en" itself.
    source_code = (source_lang or "").strip()
    ctx_n = int(getattr(cfg, "TRANSLATION_CONTEXT_SEGMENTS", 3) or 0) \
        if context_segments is None else int(context_segments)

    results: "list[dict[str, str]]" = [{} for _ in segments]
    warnings: "list[str]" = []
    kept: "dict[int, list[str]]" = {}
    meta = {"model": ref, "source": source_lang or "", "mode": mode,
            "kept": kept}
    if not segments or not targets:
        return results, warnings, meta

    total_units = len(segments) * len(targets)
    done_units = 0
    # Tail of the last GUARD-PASSING translation — the live-preview line.
    # Guard-kept originals must never land here: the card would show the
    # source language and read as "not translating".
    last_ok: "str | None" = None

    # Progress within the CURRENT target: reset at every target boundary so
    # the run panel can show one bar per language on top of the global
    # fraction. A plain dict is enough: the loop is single-threaded.
    unit_state = {"target": None, "done": 0, "bucket": 0, "t0": 0.0,
                  "instant": False, "started": False, "steps": 1}

    def _progress(step: str, last_text: "str | None" = None) -> None:
        tp = (unit_state["done"] / len(segments)) if segments else 1.0
        # The 5 % bucket line every sibling stage prints, per target, plus
        # one line when a target finishes — so a run is diagnosable from
        # the server log alone and the client's receipts can be checked.
        b = int(min(1.0, tp) * 20)
        if b > unit_state["bucket"] and not unit_state["instant"]:
            unit_state["bucket"] = b
            if b < 20:
                logger.info("[translate] %s %d%% (%d/%d)", unit_state["target"],
                            b * 5, unit_state["done"], len(segments))
        if tp >= 1.0 and unit_state["bucket"] < 21:
            unit_state["bucket"] = 21
            if unit_state["instant"]:
                logger.info("[translate] %s copied verbatim (same language)",
                            unit_state["target"])
            else:
                logger.info("[translate] %s done in %.1fs (%d segments)",
                            unit_state["target"],
                            time.perf_counter() - unit_state["t0"], len(segments))
        if progress_cb is None:
            return
        frac = done_units / total_units
        tail = (last_text or "").strip()[:160] or None
        try:
            progress_cb(frac, step, tail,
                        target=unit_state["target"], target_progress=tp)
        except Exception:  # noqa: BLE001 — progress must never break us
            pass

    def _blank_progress(step: str) -> None:
        """The tick for blank members no model call was made for. Before the
        target's start tick (_translate_one) it would start the run plan's
        unit ahead of the model load and bill a cold load as translation
        time, so it is held back then, unless it finishes the target (an
        all-blank target loads nothing)."""
        if unit_state["started"] or unit_state["done"] >= len(segments):
            _progress(step, last_ok)

    def _check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise TranslationCancelled()

    llm = None   # loaded lazily — a same-language-only request never loads

    async def _model():
        nonlocal llm
        if llm is None:
            llm = await _get_model(ref, lease=True, download_cb=download_cb)
        return llm

    async def _translate_one(text: str, target: str, context: str) -> str:
        m = await _model()
        if not unit_state["started"]:
            # Start tick: the run plan clocks a target's unit from the first
            # tick naming it. Without this one that is the END of the first
            # batch (a one-batch target closed with took_s 0 and taught the
            # rate ledger nothing). Sent after the model load, so a cold
            # load never counts as translation time.
            unit_state["started"] = True
            _progress(f"{target} 0/{unit_state['steps']}", last_ok)
        return await _run_completion(
            m, family, text, source_code, target, context,
            glossary, template_override=template_override)

    # Greedy (temperature-0) families are deterministic: re-sending an
    # identical prompt is a guaranteed no-op, so their retry only makes sense
    # when dropping the context actually changes the prompt. Sampled families
    # (hunyuan, temp 0.7) get a meaningful re-roll either way.
    greedy = not _FAMILIES[family].sampling.get("temperature")
    # ...and dropping the context only changes the prompt when the builder
    # renders it (Family.uses_context; custom: the template's {context} slot).
    ctx_changes = _FAMILIES[family].uses_context and (
        family != "custom" or "{context}" in (
            template_override if template_override is not None
            else (getattr(cfg, "TRANSLATION_PROMPT_TEMPLATE", "") or "")))

    def _guard(src: str, out: str, target: str) -> "str | None":
        # No source: same_language cannot short-circuit a target in the
        # text's own language, so an unchanged output is no evidence of a
        # failed translation (_guard_reason's source_known).
        return _guard_reason(src, out, target=target,
                             source_known=bool(source_code))

    def _mark_kept(idx: int, target: str) -> None:
        kept.setdefault(idx, []).append(target)

    async def _translate_fit(text: str, target: str,
                             context: str) -> "tuple[str, str | None]":
        """_translate_one, with a prompt past the model's context window
        turned into a failure reason for this batch alone: ("", reason)."""
        try:
            return await _translate_one(text, target, context), None
        except _ContextOverflow:
            return "", _CONTEXT_OVERFLOW

    async def _translate_guarded(text: str, target: str,
                                 context: str) -> "tuple[str, str | None]":
        """_translate_fit plus the output guard: (output, guard_reason).
        The guard runs in a worker thread: on a long member it costs tens
        of ms of regex work, which would stall every other client's I/O."""
        out, reason = await _translate_fit(text, target, context)
        if not reason:
            reason = await asyncio.to_thread(_guard, text, out, target)
        return out, reason

    def _keep_original(span: str, indices: "list[int]", target: str,
                       reason: str) -> None:
        """Guard fallback: warn once for ``span`` and mark every member
        segment as a kept-original for ``target``."""
        warnings.append(
            f"{span} ({target}): kept original — translation failed "
            f"({reason})")
        for j in indices:
            _mark_kept(j, target)

    async def _guarded_single(text: str, target: str, context: str,
                              seg_idx: int) -> str:
        """One translation + guard; on failure ONE retry WITHOUT context
        (context echo is the dominant failure mode — a context-free prompt
        is the highest-value second attempt), then keep the original text +
        a warning. Greedy families skip the retry when it would re-send an
        identical prompt (see ``greedy`` / ``ctx_changes``)."""
        nonlocal last_ok
        out, reason = await _translate_guarded(text, target, context)
        if reason is None:
            last_ok = out
            return out
        if (context and ctx_changes) or not greedy:
            out, reason = await _translate_guarded(text, target, "")
            if reason is None:
                last_ok = out
                return out
        _keep_original(f"segment {seg_idx + 1}", [seg_idx], target, reason)
        return text

    try:
      for target in targets:
        _check_cancel()
        unit_state["target"] = target
        unit_state["done"] = 0
        unit_state["bucket"] = 0
        unit_state["started"] = False
        unit_state["t0"] = time.perf_counter()
        unit_state["instant"] = same_language(target, source_lang)
        if unit_state["instant"]:
            for i, seg in enumerate(segments):
                results[i][target] = (seg.get("text") or "")
            done_units += len(segments)
            unit_state["done"] = len(segments)
            _progress(f"{target} 1/1",
                      (segments[-1].get("text") or "") if segments else None)
            continue

        if mode == "faithful":
            k_cfg = max(1, int(getattr(cfg, "TRANSLATION_BATCH_SEGMENTS", 8)
                               or 1))
            k = k_cfg
            # Growing back after a halving re-tries a size that just failed.
            # A grown size that fails a SECOND time with no batch of that
            # size or larger numbered in between marks it as one the model
            # cannot number (not one bad cue): k never grows to it again in
            # this target, or every few cues would pay a failed decode.
            k_ceil = k_cfg
            grown = False
            failed_grown: set = set()
            n_batches = max(1, -(-len(segments) // k))
            unit_state["steps"] = n_batches
            i = 0
            batch_no = 0
            while i < len(segments):
                _check_cancel()
                batch_idx = list(range(i, min(i + k, len(segments))))
                # Blank-source members never reach the model: they would
                # cost a call and a bogus "empty output" warning.
                live_idx = [j for j in batch_idx
                            if (segments[j].get("text") or "").strip()]
                for j in batch_idx:
                    if j not in live_idx:
                        results[j][target] = ""
                if not live_idx:
                    i += len(batch_idx)
                    batch_no += 1
                    done_units += len(batch_idx)
                    unit_state["done"] += len(batch_idx)
                    _blank_progress(
                        f"{target} {min(batch_no, n_batches)}/{n_batches}")
                    continue
                texts = [(segments[j].get("text") or "") for j in live_idx]
                context = _context_lines(segments, i, ctx_n)
                if len(live_idx) == 1:
                    results[live_idx[0]][target] = await _guarded_single(
                        texts[0], target, context, live_idx[0])
                else:
                    payload = (f"{_NUMBERED_INSTRUCTION}\n\n"
                               f"{_build_numbered(texts)}")
                    reply, too_long = await _translate_fit(
                        payload, target, context)
                    parsed = (None if too_long else
                              _parse_numbered(reply, len(live_idx)))
                    if parsed is None:
                        # Line-count mismatch (or a prompt past the model's
                        # context window) → halve the batch and retry the
                        # same position (down to per-segment prompts). Halve
                        # the span from the first to the last live member
                        # actually sent, not k: a short tail or leading
                        # blanks would otherwise resend the identical prompt.
                        # The leading blanks stay on top of the halved span —
                        # the next batch starts at them again, and would
                        # otherwise spend the halved size on blanks alone.
                        cap = grown and k in failed_grown
                        if grown:
                            failed_grown.add(k)
                        k = max(1, (live_idx[0] - i)
                                + (live_idx[-1] - live_idx[0] + 1) // 2)
                        if cap:
                            k_ceil = k
                        grown = False
                        n_batches = max(1, -(-(len(segments) - i) // k)) + batch_no
                        continue
                    for j, out in zip(live_idx, parsed):
                        src = segments[j].get("text") or ""
                        reason = await asyncio.to_thread(_guard, src, out, target)
                        if reason is None:
                            results[j][target] = out
                            last_ok = out
                        else:
                            # ONE retry of that segment alone WITHOUT context
                            # (the standalone prompt already differs from the
                            # batched one, so the retry is never a no-op even
                            # for a greedy context-blind family), else keep
                            # original + warning.
                            retried, reason = await _translate_guarded(
                                src, target, "")
                            if reason is None:
                                results[j][target] = retried
                                last_ok = retried
                            else:
                                results[j][target] = src
                                _keep_original(f"segment {j + 1}", [j],
                                               target, reason)
                    # A size that numbered fine is no size the model cannot
                    # number: only a failure IN A ROW at it caps k_ceil.
                    failed_grown = {s for s in failed_grown if s > k}
                i += len(batch_idx)
                batch_no += 1
                done_units += len(batch_idx)
                unit_state["done"] += len(batch_idx)
                grown = k < k_ceil and i < len(segments)
                if grown:
                    # A halving gets past ONE failing position: grow k back
                    # once a batch went through, or one early mis-numbering
                    # would leave every later cue of the target tiny batches.
                    k = min(k_ceil, 2 * k)
                    n_batches = max(1, -(-(len(segments) - i) // k)) + batch_no
                _progress(f"{target} {min(batch_no, n_batches)}/{n_batches}",
                          last_ok)
        else:
            # FLUENT: sentence-group merge → translate → redistribute.
            groups = _merge_sentences(segments)
            unit_state["steps"] = len(groups)
            for g_no, group in enumerate(groups, start=1):
                _check_cancel()
                src_texts = [(segments[j].get("text") or "") for j in group]
                # No space where zh / ja cues meet: whisper cuts them mid-word.
                joined = join_translations(src_texts)
                if not joined:
                    # Every member is blank — nothing to translate, and a
                    # model call would only invent text for the cues.
                    for j in group:
                        results[j][target] = ""
                    done_units += len(group)
                    unit_state["done"] += len(group)
                    _blank_progress(f"{target} {g_no}/{len(groups)}")
                    continue
                context = _context_lines(segments, group[0], ctx_n)
                translated, reason = await _translate_guarded(
                    joined, target, context)
                if reason is not None and \
                        ((context and ctx_changes) or not greedy):
                    # ONE retry WITHOUT context (see _guarded_single).
                    translated, reason = await _translate_guarded(
                        joined, target, "")
                pieces: "list[str]" = []
                if reason is None:
                    pieces = await asyncio.to_thread(
                        _redistribute, src_texts, translated)
                    if not all(p.strip() for p, s in zip(pieces, src_texts)
                               if s.strip()):
                        # A reply shorter than the member count cannot be
                        # split without blank cues for segments that have
                        # source text — revert the group like a guard hit.
                        reason = "redistribution left empty cues"
                if reason is not None:
                    for j in group:
                        results[j][target] = segments[j].get("text") or ""
                    span = (f"segment {group[0] + 1}" if len(group) == 1 else
                            f"segments {group[0] + 1}-{group[-1] + 1}")
                    # A blank member stays blank — never a "kept original".
                    _keep_original(span, [j for j, t in zip(group, src_texts)
                                          if t.strip()], target, reason)
                else:
                    for j, piece in zip(group, pieces):
                        results[j][target] = piece
                    last_ok = results[group[-1]].get(target) or last_ok
                done_units += len(group)
                unit_state["done"] += len(group)
                _progress(f"{target} {g_no}/{len(groups)}", last_ok)
    finally:
        # Synchronous release: an await here could itself be interrupted by a
        # second cancellation, abandoning the lease and pinning the model
        # against eviction forever.
        if llm is not None:
            _release_model(ref)

    return results, warnings, meta


def _reset_for_tests() -> None:
    """Test-only: empty the loaded-model LRU (tests only ever put stubs in
    it) with its load params and per-ref load locks — a stale params tuple
    turns a later test's stub into a hit or a reload by test order — the
    job leases and in-flight decode counts — a leaked lease or count makes
    every later eviction test see a refusal — and the model-card language
    lists a predownload test records."""
    _models.clear()
    _last_used.clear()
    _params.clear()
    _loading.clear()
    _active.clear()
    with _decoding_mutex:
        _decoding.clear()
    _card_languages.clear()
