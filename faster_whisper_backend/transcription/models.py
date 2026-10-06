"""The whisper model cache and everything a decode needs from it: the LRU of
loaded WhisperModels with per-request leases (_get_or_load_model,
_release_model_lease, _drop_loaded_model, drain_then_evict, _idle_evictor)
and the residency API runtime.preload reads it through (is_resident,
idle_peer, load_unleased, evict, ...), the model-id gates, the opt-in HF→CT2 conversion, the shared GPU inference
gate and URL-download limiter, the SUPPRESS_CHARS token cache, and the
transcribe-kwargs assembly with the clamped client decode overrides
(assemble_transcribe_kwargs, _apply_decode_overrides).

Callers go through the module attribute (``tx_models._get_or_load_model(...)``)
— never a ``from ... import`` of a name here — so one patch (the test suite's
fake loader) reaches every caller. Must not import main, runtime.preload or
the request-progress registry (preload imports this module eagerly).
"""
import asyncio
import logging
import math
import os
import re
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable

from fastapi import HTTPException

from faster_whisper_backend.transcription import decode_trace as _decode_trace
from faster_whisper_backend.core import jobs
from faster_whisper_backend.runtime import model_registry
from faster_whisper_backend.runtime import system_stats
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.stats import metrics

# faster_whisper pulls the heavy native stack (ctranslate2/onnxruntime/av). It is
# imported lazily at first model load (see _get_or_load_model) so this module
# stays importable for tests/tooling on a box without the CUDA stack installed.
# TYPE_CHECKING keeps the WhisperModel annotation resolvable for type checkers
# without importing it at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger("whisper-api")


# =============================================================================
# Per-request model selection with LRU cache
# =============================================================================
# Clients can ask for any faster-whisper-compatible model via the OpenAI
# `model` form param. We resolve the OpenAI default `whisper-1` (and empty)
# to WHISPER_DEFAULT_MODEL, lazy-load on first use, and keep up to
# WHISPER_MAX_LOADED_MODELS hot in VRAM (LRU eviction).
#
# Examples a client can pass:
#   "whisper-1"                                        OpenAI default -> our default
#   "large-v2"                                         faster-whisper short name
#   "large-v3" / "large-v3-turbo" / "distil-large-v3"
#   "Systran/faster-whisper-large-v3"                  full HF repo id
#   "primeline/whisper-large-v3-turbo-german"          German-finetuned
#
# Set WHISPER_ALLOWED_MODELS to restrict which model names are accepted (a
# comma-separated allowlist; empty = any well-formed model id goes, useful on
# a private LAN).
# Source: cfg.DEFAULT_MODEL / cfg.ALLOWED_MODELS / cfg.MAX_LOADED_MODELS.

# Insertion order = LRU order (oldest at front). move_to_end on hit.
_loaded_models: "OrderedDict[str, WhisperModel]" = OrderedDict()
_model_load_lock = asyncio.Lock()
# name → count of requests currently decoding on the cached model. A leased
# model is never freed (see _drop_loaded_model): closing CTranslate2's native
# translator under a running decode is a use-after-free that takes the whole
# process down, and the caller's local Python reference alone does not stop
# the LRU/idle paths from dropping the entry.
#
# Deliberately NOT guarded by _model_load_lock: every mutation happens on the
# event loop with no await between the check and the write, and every eviction
# path (_drop_loaded_model's callers: the LRU loop, _idle_evictor,
# drain_then_evict, shutdown) is a synchronous block on that same loop. Loop
# semantics therefore make check-then-mutate atomic — the same reasoning that
# already lets the cache-hit fast path in _get_or_load_model run lock-free.
_model_leases: "dict[str, int]" = {}


# =============================================================================
# SUPPRESS_CHARS resolution cache
# =============================================================================
# Resolve the user's SUPPRESS_CHARS string to vocabulary token IDs via the
# loaded model's hf_tokenizer. The encoding depends on the model's BPE
# table, so the cache key is (model_id, chars_str). Invalidated on model
# unload (LRU/idle/evict-on-edit) and naturally rekeyed when SUPPRESS_CHARS
# changes. An LRU capped at _SUPPRESS_CHARS_CACHE_MAX entries: a client's
# suppress_chars decode key makes the strings caller-chosen.
_suppress_chars_cache: "OrderedDict[tuple[str, str], tuple[int, ...]]" = OrderedDict()
_SUPPRESS_CHARS_CACHE_MAX = 256


def _resolve_suppress_chars(model_id: str,
                            model: "WhisperModel",
                            chars: "str | None",
                            from_client: bool = False) -> "tuple[int, ...]":
    """Return the sorted tuple of vocab IDs to suppress for the given chars.
    Each char is encoded both bare and with a leading space — Whisper's BPE
    often tokenizes a punct char differently in those positions (mirrors
    faster-whisper's own non_speech_tokens approach). Multi-piece results
    are skipped with a warning (suppressing only the first piece would
    block every word that starts with that piece). ``from_client``: the
    chars came from a request, so their resolution logs at DEBUG only."""
    if not chars:
        return ()
    key = (model_id, chars)
    cached = _suppress_chars_cache.get(key)
    if cached is not None:
        try:
            _suppress_chars_cache.move_to_end(key)
        except KeyError:   # dropped by a concurrent unload; still a valid answer
            pass
        return cached
    tok = getattr(model, "hf_tokenizer", None)
    ids: set[int] = set()
    if tok is not None:
        for ch in chars:
            if ch.isspace():
                continue
            for variant in (ch, " " + ch):
                try:
                    enc = tok.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                raw_ids = getattr(enc, "ids", None)
                if raw_ids is None and isinstance(enc, list):
                    raw_ids = enc
                if raw_ids is None:
                    continue
                if len(raw_ids) == 1:
                    ids.add(int(raw_ids[0]))
                else:
                    logger.warning(
                        "SUPPRESS_CHARS %r tokenises to %d pieces; skipping",
                        variant, len(raw_ids),
                    )
    out = tuple(sorted(ids))
    _suppress_chars_cache[key] = out
    while len(_suppress_chars_cache) > _SUPPRESS_CHARS_CACHE_MAX:
        _suppress_chars_cache.popitem(last=False)
    if out:
        (logger.debug if from_client else logger.info)(
            "SUPPRESS_CHARS resolved for %s (%r): %r", model_id, chars, out)
    return out


# Per-request decode-param overrides (the client's "decode overrides"). Optional;
# absent leaves behavior identical to before (config-only). Every value is clamped
# to the SAME bounds the admin config enforces (settings/schema.py), so an untrusted
# client cannot request unbounded compute on the shared server. Applied AFTER config
# resolution, so the order is: request > per-model override > global default.
# Only model.transcribe kwargs belong in these tables: every key in them is
# forwarded as one (live-dictation keys are applied by the streaming route).
def _client_bounds(key: str) -> "tuple":
    """(min, max) of a client decode key, read from its config field's own
    bounds (settings_schema.client_key_bounds) rather than copied by hand."""
    b = settings_schema.client_key_bounds()[key]
    return b["min"], b["max"]


_DECODE_INT_BOUNDS = {
    "beam_size": (1, 20),
    "best_of": (1, 20),
    "no_repeat_ngram_size": (0, 10),
    "language_detection_segments": _client_bounds("language_detection_segments"),
}
_DECODE_FLOAT_BOUNDS = {
    "no_speech_threshold": (0.0, 1.0),
    "log_prob_threshold": (-10.0, 0.0),
    "compression_ratio_threshold": (0.0, 10.0),
    "patience": (0.5, 5.0),
    "length_penalty": (0.1, 5.0),
    "repetition_penalty": (0.5, 5.0),
    "hallucination_silence_threshold": _client_bounds("hallucination_silence_threshold"),
    "language_detection_threshold": _client_bounds("language_detection_threshold"),
}
_DECODE_STR_CAPS = {
    "hotwords": 2048,
    "prepend_punctuations": 64,
    "append_punctuations": 64,
}
# suppress_tokens is a list, so it gets a length cap plus a per-id range instead
# of a scalar clamp. 256 ids is far more than any real suppression set; the range
# is "any token id the tokenizer could hold", with -1 kept as faster-whisper's
# "also suppress the non-speech set" sentinel.
_SUPPRESS_TOKENS_MAX = 256
_SUPPRESS_TOKEN_ID_MAX = 2 ** 31
# A client temperature is a number or a retry ladder (list / comma string,
# like TEMPERATURE): every rung clamped to this range, at most this many rungs.
_TEMPERATURE_BOUNDS = (0.0, 1.0)
_TEMPERATURE_RUNGS_MAX = 16


def _clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float('inf')) from a JSON number like 1e999.
        return None


def _clamp_float(v, lo, hi):
    try:
        r = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(r):
        # JSON permits NaN/Infinity literals; reject them so a non-finite value
        # is ignored (override dropped) like the int path, not silently clamped
        # to a bound (NaN/+inf → hi, -inf → lo).
        return None
    return max(lo, min(hi, r))


def _clamp_context_segments(v: "int | None") -> "int | None":
    """A request's TRANSLATION_CONTEXT_SEGMENTS, clamped to the field's own
    bounds (0–10; None = absent, inherit). Shared by the audio routes (main) and
    the text route (translation/routes.py)."""
    if v is None:
        return None
    b = settings_schema.field_bounds()["TRANSLATION_CONTEXT_SEGMENTS"]
    return _clamp_int(v, b["min"], b["max"])


def _apply_decode_overrides(kwargs, resolved_model, overrides, ident=None):
    """Merge clamped per-request decode overrides into transcribe_kwargs (request
    wins). Unknown keys and unparseable values are ignored. Keys LOCKED by an
    identity layer (``ident.locked_client_keys``) are dropped before clamping —
    the admin-set value stands and the client override is ignored."""
    if not isinstance(overrides, dict) or not overrides:
        return kwargs
    # Lock gate: an identity layer can forbid the client from overriding a field.
    locked_client_keys = ident.locked_client_keys if ident is not None else frozenset()
    if locked_client_keys:
        overrides = {k: v for k, v in overrides.items() if k not in locked_client_keys}
        if not overrides:
            return kwargs
    for key, (lo, hi) in _DECODE_INT_BOUNDS.items():
        if key in overrides:
            cv = _clamp_int(overrides[key], lo, hi)
            if cv is not None:
                kwargs[key] = cv
    for key, (lo, hi) in _DECODE_FLOAT_BOUNDS.items():
        if key in overrides:
            cv = _clamp_float(overrides[key], lo, hi)
            if cv is not None:
                kwargs[key] = cv
    if "temperature" in overrides:
        tv = _client_temperature(overrides["temperature"])
        if tv is not None:
            kwargs["temperature"] = tv
    # A JSON null on a bool override means "inherit", not False.
    for key in ("condition_on_previous_text", "multilingual"):
        if overrides.get(key) is not None:
            kwargs[key] = bool(overrides[key])
    for key, cap in _DECODE_STR_CAPS.items():
        if key in overrides and isinstance(overrides[key], str):
            kwargs[key] = overrides[key][:cap]
    # A blank hotwords override means CLEAR the admin DEFAULT_HOTWORDS — remove
    # the kwarg entirely. Forwarding a whitespace-only string is NOT a clear:
    # any truthy hotwords value makes faster-whisper emit <|startofprev|> (the
    # fake previous-transcript slot), which alone biases the decoder to treat a
    # recording that starts mid-speech as a window continuation and drop its
    # opening words.
    if isinstance(kwargs.get("hotwords"), str) and not kwargs["hotwords"].strip():
        kwargs.pop("hotwords")
    if "suppress_tokens" in overrides:
        st = overrides["suppress_tokens"]
        ids = None
        try:
            if isinstance(st, list):
                ids = [int(x) for x in st]
            elif isinstance(st, str):
                ids = [int(t.strip()) for t in st.split(",") if t.strip()]
        except (TypeError, ValueError, OverflowError):
            # OverflowError: int(float('inf')) from a JSON Infinity / 1e999
            # literal — drop the malformed override like the clamp paths,
            # never let it 500 the request.
            ids = None
        if ids is not None:
            ids = [i for i in ids[:_SUPPRESS_TOKENS_MAX]
                   if -1 <= i < _SUPPRESS_TOKEN_ID_MAX]
            if ids:
                kwargs["suppress_tokens"] = ids
            elif not (st.strip() if isinstance(st, str) else st):
                # An EXPLICITLY empty list / blank string is the client's
                # "cleared — overrides inherited" state (distinct from the
                # key being absent), and faster-whisper's own spelling of
                # "suppress nothing" is None. A non-empty value that merely
                # filtered away (out-of-range ids) is NOT a clear: leave the
                # config value in place rather than forward an override the
                # caller didn't really make.
                kwargs["suppress_tokens"] = None
    # VAD: toggle + sub-params (sub-params rebuilt from config defaults when on).
    if overrides.get("vad_filter") is not None:
        vf = bool(overrides["vad_filter"])
        kwargs["vad_filter"] = vf
        if not vf:
            kwargs["vad_parameters"] = None
    if kwargs.get("vad_filter"):
        vp = dict(kwargs.get("vad_parameters") or dict(
            min_silence_duration_ms=effective_config.cfg_for(resolved_model, "VAD_MIN_SILENCE_MS", ident),
            speech_pad_ms=effective_config.cfg_for(resolved_model, "VAD_SPEECH_PAD_MS", ident),
            threshold=effective_config.cfg_for(resolved_model, "VAD_THRESHOLD", ident),
        ))
        if "vad_min_silence_duration_ms" in overrides:
            cv = _clamp_int(overrides["vad_min_silence_duration_ms"], 0, 10000)
            if cv is not None:
                vp["min_silence_duration_ms"] = cv
        if "vad_speech_pad_ms" in overrides:
            cv = _clamp_int(overrides["vad_speech_pad_ms"], 0, 2000)
            if cv is not None:
                vp["speech_pad_ms"] = cv
        if "vad_threshold" in overrides:
            cv = _clamp_float(overrides["vad_threshold"], 0.0, 1.0)
            if cv is not None:
                vp["threshold"] = cv
        kwargs["vad_parameters"] = vp
    return kwargs


def _temperature_ladder(s: "str | None") -> "tuple[float, ...]":
    """Parse a comma-separated TEMPERATURE ladder; () for blank, token-less
    (",") or unparseable text — i.e. for every value that yields no ladder."""
    def _finite(tok: str) -> float:
        v = float(tok)
        if not math.isfinite(v):
            raise ValueError(f"non-finite temperature: {tok!r}")
        return v

    try:
        return tuple(_finite(t.strip()) for t in (s or "").split(",") if t.strip())
    except ValueError:
        return ()


def _client_temperature(v) -> "float | tuple[float, ...] | None":
    """A client `temperature` decode key: a number, or a retry ladder as a
    list or a comma string (TEMPERATURE's own format, parsed by
    _temperature_ladder). Rungs are clamped to _TEMPERATURE_BOUNDS and capped
    at _TEMPERATURE_RUNGS_MAX; one rung is a float (as before), several a
    tuple (faster-whisper's ladder). None for anything unparseable or empty —
    the override is then dropped, like the other clamp paths."""
    if isinstance(v, str):
        raw = _temperature_ladder(v)
    elif isinstance(v, (list, tuple)):
        raw = tuple(v)
    else:
        raw = (v,)
    rungs = [_clamp_float(r, *_TEMPERATURE_BOUNDS)
             for r in raw[:_TEMPERATURE_RUNGS_MAX]]
    if not rungs or None in rungs:
        return None
    return rungs[0] if len(rungs) == 1 else tuple(rungs)


def assemble_transcribe_kwargs(resolved_model, model, *, language, temperature,
                               vad_filter, vad_parameters, want_word_ts,
                               initial_prompt, overrides=None, ident=None,
                               task="transcribe"):
    """Assemble the full ``model.transcribe`` kwargs from per-model config.

    Single source of truth shared by the batch endpoint and the streaming FINAL
    decode, so the two produce identical results for the same audio (there must
    be no difference between streaming and batch). The per-request values
    (``language``, ``temperature``, ``vad_filter``, ``vad_parameters``,
    ``want_word_ts``, ``initial_prompt``) are passed in already resolved; every
    other knob is read here via ``cf`` which layers per-identity (``ident``) >
    per-model > global. ``ident=None`` is byte-identical to the pre-feature path.
    """
    def cf(field):
        return effective_config.cfg_for(resolved_model, field, ident)

    transcribe_kwargs = dict(
        language=language if language else None,
        beam_size=cf("BEAM_SIZE"),
        best_of=cf("BEST_OF"),
        temperature=temperature,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=want_word_ts,
        condition_on_previous_text=cf("CONDITION_ON_PREVIOUS_TEXT"),
        initial_prompt=initial_prompt,
        no_speech_threshold=cf("NO_SPEECH_THRESHOLD"),
        log_prob_threshold=cf("LOG_PROB_THRESHOLD"),
        compression_ratio_threshold=cf("COMPRESSION_RATIO_THRESHOLD"),
    )
    # Whisper task — only forwarded off-default, so the kwargs dict (and the
    # streaming FINAL decode, which never passes `task`) stay byte-identical
    # to the pre-feature path for plain transcription.
    if task and task != "transcribe":
        transcribe_kwargs["task"] = task
    # Optional advanced kwargs — only forwarded when set, so the
    # transcribe_kwargs dict stays clean for the common path.
    _hotwords = cf("DEFAULT_HOTWORDS")
    if _hotwords and _hotwords.strip():
        transcribe_kwargs["hotwords"] = _hotwords
    _temp_str = cf("TEMPERATURE")
    if _temp_str:
        # Per-model/identity override of the temperature ladder. Comma-
        # separated floats; falls back to the per-request `temperature`
        # (default 0.0) when unset.
        ladder = _temperature_ladder(_temp_str)
        if ladder:
            transcribe_kwargs["temperature"] = ladder
    _patience = cf("PATIENCE")
    if _patience and _patience != 1.0:
        transcribe_kwargs["patience"] = _patience
    _length_penalty = cf("LENGTH_PENALTY")
    if _length_penalty and _length_penalty != 1.0:
        transcribe_kwargs["length_penalty"] = _length_penalty
    _repetition_penalty = cf("REPETITION_PENALTY")
    if _repetition_penalty and _repetition_penalty != 1.0:
        transcribe_kwargs["repetition_penalty"] = _repetition_penalty
    _no_repeat_ngram = cf("NO_REPEAT_NGRAM_SIZE")
    if _no_repeat_ngram:
        transcribe_kwargs["no_repeat_ngram_size"] = _no_repeat_ngram
    _prompt_reset_t = cf("PROMPT_RESET_ON_TEMPERATURE")
    if _prompt_reset_t is not None and _prompt_reset_t != 0.5:
        transcribe_kwargs["prompt_reset_on_temperature"] = _prompt_reset_t
    if cf("MULTILINGUAL"):
        transcribe_kwargs["multilingual"] = True
    _lang_thresh = cf("LANGUAGE_DETECTION_THRESHOLD")
    if _lang_thresh is not None and _lang_thresh != 0.5:
        transcribe_kwargs["language_detection_threshold"] = _lang_thresh
    _lang_segs = cf("LANGUAGE_DETECTION_SEGMENTS")
    if _lang_segs and _lang_segs != 1:
        transcribe_kwargs["language_detection_segments"] = _lang_segs
    _hallu_silence = cf("HALLUCINATION_SILENCE_THRESHOLD")
    if _hallu_silence:
        # 0 is "off" here, but an active threshold to faster-whisper (it only
        # treats None as off) — so only a positive value is forwarded.
        transcribe_kwargs["hallucination_silence_threshold"] = _hallu_silence
    _suppress_blank = cf("SUPPRESS_BLANK")
    if _suppress_blank is False:
        transcribe_kwargs["suppress_blank"] = False
    _suppress_tokens_str = cf("SUPPRESS_TOKENS")
    # An explicitly blank SUPPRESS_TOKENS (profile / per-model / global) is
    # "suppress nothing" — faster-whisper's spelling is None, which the
    # SUPPRESS_CHARS merge below reads as "cleared".
    if _suppress_tokens_str is not None:
        if _suppress_tokens_str.strip():
            try:
                transcribe_kwargs["suppress_tokens"] = [
                    int(t.strip()) for t in _suppress_tokens_str.split(",") if t.strip()
                ]
            except ValueError:
                pass
        else:
            transcribe_kwargs["suppress_tokens"] = None
    # "" is an explicit "no punctuation splitting" (a cleared profile /
    # per-model field), so it is forwarded like the per-request override
    # path does; only an ABSENT value leaves faster-whisper's default.
    _prepend_p = cf("PREPEND_PUNCTUATIONS")
    if _prepend_p is not None:
        transcribe_kwargs["prepend_punctuations"] = _prepend_p
    _append_p = cf("APPEND_PUNCTUATIONS")
    if _append_p is not None:
        transcribe_kwargs["append_punctuations"] = _append_p
    # Per-request overrides win (clamped), EXCEPT fields locked by an identity
    # layer (skipped). No-op when None/empty.
    _apply_decode_overrides(transcribe_kwargs, resolved_model, overrides, ident=ident)
    # A client hallucination_silence_threshold of 0 is "off" too (see above).
    if not transcribe_kwargs.get("hallucination_silence_threshold"):
        transcribe_kwargs.pop("hallucination_silence_threshold", None)
    # SUPPRESS_CHARS — chars resolved to vocab IDs via the loaded model's
    # tokenizer, then merged into the EFFECTIVE suppress_tokens list, i.e.
    # after a client suppress_tokens override (which used to replace the
    # merged ids). Genuinely additive: key absent = faster-whisper's default
    # (-1, the non-speech set) plus the chars; a cleared list (None, from the
    # config or the client) = the chars only; a list = the list plus the chars.
    # A client suppress_chars (unless locked) replaces the configured string;
    # "" is an explicit "no chars".
    _suppress_chars = cf("SUPPRESS_CHARS")
    _client_chars = (overrides or {}).get("suppress_chars")
    _chars_from_client = (
        isinstance(_client_chars, str)
        and "suppress_chars" not in (ident.locked_client_keys
                                     if ident is not None else frozenset()))
    if _chars_from_client:
        _suppress_chars = _client_chars[
            :settings_schema.client_key_bounds()["suppress_chars"]["maxlen"]]
    if _suppress_chars:
        extra_ids = _resolve_suppress_chars(resolved_model, model, _suppress_chars,
                                            _chars_from_client)
        if extra_ids:
            if "suppress_tokens" not in transcribe_kwargs:
                merged_ids = sorted({-1, *extra_ids})
            elif transcribe_kwargs["suppress_tokens"] is None:
                merged_ids = sorted(set(extra_ids))
            else:
                merged_ids = sorted(set(transcribe_kwargs["suppress_tokens"])
                                    | set(extra_ids))
            transcribe_kwargs["suppress_tokens"] = merged_ids
    # multilingual re-detects the language on every 30 s window and IGNORES a
    # given language — a chosen language must win, so it applies to
    # auto-detect only.
    if transcribe_kwargs.get("language"):
        transcribe_kwargs.pop("multilingual", None)
    return transcribe_kwargs


# Client keys that only act while the language is auto-detected.
_AUTO_DETECT_ONLY_KEYS = ("multilingual", "language_detection_threshold",
                          "language_detection_segments")


def _note_auto_detect_only(overrides: dict, language: "str | None",
                           ignored: list) -> None:
    """multilingual and the language-detection knobs apply to auto-detect
    only (assemble_transcribe_kwargs drops multilingual when a language is
    set; faster-whisper skips detection): a client override for one of them
    then lands in `overrides_ignored`, like a locked key, instead of
    vanishing."""
    if not language:
        return
    for key in _AUTO_DETECT_ONLY_KEYS:
        if overrides.get(key) is not None and key not in ignored:
            ignored.append(key)


def _note_word_ts_only(overrides: dict, word_timestamps: bool,
                       ignored: list) -> None:
    """faster-whisper reads hallucination_silence_threshold only from a decode
    with word timestamps: without them a client value (0 = off aside) lands in
    `overrides_ignored` instead of vanishing."""
    if (not word_timestamps and overrides.get("hallucination_silence_threshold")
            and "hallucination_silence_threshold" not in ignored):
        ignored.append("hallucination_silence_threshold")


def _output_wrappers(resolved_model, ident, overrides: "dict | None") -> "tuple[str, str]":
    """(OUTPUT_PREFIX, OUTPUT_SUFFIX) for this request: the client's
    output_prefix / output_suffix decode keys win unless locked, else the
    resolved config. Gated on ident.locked_client_keys, which carries the
    decode-override master gate too (_resolve_request_knob's ident.locked
    would miss it). "" is an explicit "none"; capped at the field length.
    Shared by the batch route and the streaming wrappers (handshake and
    _refresh_ident)."""
    locked = ident.locked_client_keys if ident is not None else frozenset()
    bounds = settings_schema.client_key_bounds()
    out = []
    for field, key in (("OUTPUT_PREFIX", "output_prefix"),
                       ("OUTPUT_SUFFIX", "output_suffix")):
        v = (overrides or {}).get(key)
        if isinstance(v, str) and key not in locked:
            out.append(v[:bounds[key]["maxlen"]])
        else:
            out.append(effective_config.cfg_for(resolved_model, field, ident) or "")
    return out[0], out[1]


def _drop_suppress_chars_cache(model_id: str) -> None:
    """Drop all cache entries for a given model. Called from unload paths."""
    for k in list(_suppress_chars_cache):
        if k[0] == model_id:
            _suppress_chars_cache.pop(k, None)


def _drop_loaded_model(name: str, *, force: bool = False) -> bool:
    """Single unload entry point: pop the cached WhisperModel, drop its
    suppress-chars entries, and unregister from the loaded-model registry.
    Caller is responsible for holding _model_load_lock when the unload is
    racy with loads (LRU eviction and idle eviction paths).

    Declines (False) while a request holds a lease on the model, unless
    ``force``. ``force`` is for the drain-then-evict / shutdown paths, whose
    documented contract (see drain_then_evict) is that in-flight requests keep
    running on the local reference they already captured."""
    if not force and _model_leases.get(name, 0) > 0:
        logger.info("Model %s is in use — eviction deferred", name)
        return False
    _loaded_models.pop(name, None)
    _drop_suppress_chars_cache(name)
    model_registry.unregister_loaded_model(name)
    return True


def _release_model_lease(name: str) -> None:
    """Release a lease taken by ``_get_or_load_model(..., lease=True)`` and
    restart the model's idle clock — a long transcription must not be evicted
    the instant it ends because the LOAD timestamp aged past the idle timeout.

    Synchronous and lock-free (see the _model_leases comment). Tolerates a name
    that is no longer cached: drain_then_evict/shutdown force-drop entries out
    from under their lease holders by design."""
    n = _model_leases.get(name, 0) - 1
    if n <= 0:
        _model_leases.pop(name, None)
    else:
        _model_leases[name] = n
    # No-op for a name the registry no longer knows (force-dropped mid-job).
    model_registry.touch_loaded_model(name)


def _resolve_model_name(requested: str) -> str:
    """Map OpenAI-compatible 'whisper-1' (or empty) to our configured default;
    pass anything else through as a faster-whisper / HF model identifier."""
    if not requested or requested == "whisper-1":
        return cfg.DEFAULT_MODEL
    return requested


# =============================================================================
# Auto HF→CT2 conversion (opt-in via AUTO_CONVERT_HF_MODELS)
# =============================================================================
# Cache structure: <root>/<sanitised_id>/<quantization>/{model.bin, ...}
# - root: cfg.CONVERTED_MODELS_DIR or ~/.cache/whisper-ct2
# - sanitised_id: model id with "/" replaced by "__"
# - quantization: e.g. "float16" — encoded in the path so changing the cfg
#                 doesn't collide with the previously-saved version.
#
# Locking strategy:
# - Per-model asyncio.Lock (held during conversion, NOT held during the
#   subsequent WhisperModel load — so cached-model fast paths for OTHER
#   models stay snappy).
# - filelock.FileLock for cross-process safety (uvicorn --workers > 1).
# - Atomic publish: write to <output_dir>.tmp, then os.rename to final.
#   Crash mid-conversion leaves no false-positive "model.bin exists" state.

_CT2_QUANTIZATIONS = {
    "float32", "float16", "bfloat16", "int16",
    "int8", "int8_float32", "int8_float16", "int8_bfloat16",
}

# Per-model asyncio locks for conversion. Lazy-populated.
_convert_locks: "dict[str, asyncio.Lock]" = {}
_convert_locks_meta = asyncio.Lock()


def _converted_root() -> str:
    """Resolve the output root for converted models. Honours
    cfg.CONVERTED_MODELS_DIR when set, else ~/.cache/whisper-ct2."""
    return getattr(cfg, "CONVERTED_MODELS_DIR", None) or os.path.join(
        os.path.expanduser("~"), ".cache", "whisper-ct2"
    )


def _converted_dir_for(model_id: str, quantization: str) -> str:
    """Compute the deterministic output directory for `model_id` at the given
    quantisation. Sanitisation: HF repo IDs only contain `[A-Za-z0-9_.-]` plus
    one `/`, so a single replace is enough. ":" is folded too: a local
    HF-format dir like `C:\\models\\x` would otherwise keep its drive
    letter and make os.path.join() discard the converted root on Windows."""
    sanitised = model_id
    for ch in ("/", os.sep, ":"):
        sanitised = sanitised.replace(ch, "__")
    return os.path.join(_converted_root(), sanitised, quantization)


def _model_needs_conversion(model_id: str) -> bool:
    """Return True if `model_id` is an HF transformers Whisper checkpoint
    (has model.safetensors / pytorch_model.bin but no model.bin in the repo).
    False for already-CT2 repos and for local paths.

    Implementation: probe the HF Hub file list. Network call (~1 s) but only
    runs when AUTO_CONVERT_HF_MODELS is on AND the converted-output cache
    misses, so it's at worst once per model per process lifetime."""
    # Local path that exists → never convert.
    if os.path.isdir(model_id):
        return not os.path.isfile(os.path.join(model_id, "model.bin"))
    # Heuristic: HF repo id always contains a single "/".
    if "/" not in model_id or model_id.count("/") != 1:
        return False
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(model_id))
    except Exception as e:
        logger.warning("auto-convert: could not probe %s file list (%s); "
                       "assuming no conversion needed", model_id, e)
        return False
    if "model.bin" in files:
        return False  # already CT2
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    # Unknown layout — let WhisperModel try and fail naturally.
    return False


def _convert_blocking(model_id: str, output_dir: str, quantization: str) -> None:
    """Synchronous CT2 conversion. Runs in a thread executor so the event
    loop stays responsive. Lazy-imports torch / transformers / ctranslate2
    converter machinery; missing extras → RuntimeError with pip command.

    Atomic publish: writes to `<output_dir>.tmp` then renames to `output_dir`
    so a crash mid-write leaves no false-positive (next start re-detects the
    missing model.bin and retries cleanly)."""
    try:
        from ctranslate2.converters import TransformersConverter
        import transformers  # noqa: F401  ensure dep present
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"AUTO_CONVERT_HF_MODELS=true but the conversion extras are not "
            f"installed (missing {e.name!r}). Run: "
            f"pip install -r requirements-convert.txt"
        ) from e

    tmp_dir = output_dir + ".tmp"
    # Clean any stale tmp from a prior crashed run.
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    logger.info("auto-convert: %s → %s (quantisation=%s)",
                model_id, output_dir, quantization)
    t0 = time.perf_counter()
    converter = TransformersConverter(
        model_name_or_path=model_id,
        # tokenizer.json + preprocessor_config.json are required by faster-
        # whisper at runtime (transcribe.py:700, :732). vocabulary.json is
        # generated by CT2 itself; copying the HF vocab.json is harmless but
        # not necessary.
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        # Loading the source as fp16 keeps RAM ~halved during conversion;
        # HF Whisper checkpoints typically ship as fp16 anyway, no precision
        # loss. low_cpu_mem_usage avoids HF's duplicate-on-CPU intermediate.
        load_as_float16=(quantization in ("float16", "int8_float16")),
        low_cpu_mem_usage=True,
    )
    converter.convert(tmp_dir, quantization=quantization, force=True)
    # Atomic publish. os.replace can't swap onto a non-empty dir on any OS (and
    # os.rename also fails onto an existing dir on Windows), so clear any stale
    # publish first, then replace.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    logger.info("auto-convert: %s completed in %.1fs",
                model_id, time.perf_counter() - t0)


async def _ensure_ct2_model(name: str) -> str:
    """If `name` is an HF transformers Whisper repo and AUTO_CONVERT_HF_MODELS
    is on, ensure a CT2 conversion exists locally and return its path.
    Otherwise return `name` unchanged.

    Locking: per-name asyncio.Lock + filelock.FileLock (cross-process).
    Conversion runs in a thread executor (blocking torch / numpy work)."""
    if not getattr(cfg, "AUTO_CONVERT_HF_MODELS", False):
        return name
    quantization = getattr(cfg, "CONVERT_QUANTIZATION", None) or "float16"
    if quantization not in _CT2_QUANTIZATIONS:
        logger.warning("auto-convert: invalid CONVERT_QUANTIZATION %r; "
                       "falling back to float16", quantization)
        quantization = "float16"
    output_dir = _converted_dir_for(name, quantization)
    # Fast path: already converted (idempotent across restarts).
    if os.path.isfile(os.path.join(output_dir, "model.bin")):
        return output_dir
    # Skip the file-list probe + conversion for already-CT2 repos and
    # local paths. OFF the loop: the probe makes a synchronous
    # huggingface_hub.list_repo_files() HTTPS call (its own docstring says
    # "~1 s"), and this is an async def. Pure predicate, evaluated before the
    # per-model lock below, so nothing can reorder against it.
    if not await asyncio.to_thread(_model_needs_conversion, name):
        return name

    # Per-model asyncio lock (lazy create). Ensures only one conversion of
    # a given model proceeds within this worker, without serialising loads
    # of OTHER models behind a global lock.
    async with _convert_locks_meta:
        lk = _convert_locks.setdefault(name, asyncio.Lock())
    async with lk:
        # Re-check inside the lock — another coroutine may have just finished.
        if os.path.isfile(os.path.join(output_dir, "model.bin")):
            return output_dir
        # Cross-process file-lock so multi-worker uvicorn doesn't double-convert.
        # Both the lock acquisition (polling, up to 600 s) and the conversion
        # itself are blocking, so the whole section runs off the event loop.
        from filelock import FileLock, Timeout as FileLockTimeout
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        lock_path = output_dir + ".lock"

        def _locked_convert() -> None:
            with FileLock(lock_path, timeout=600):
                if os.path.isfile(os.path.join(output_dir, "model.bin")):
                    return
                _convert_blocking(name, output_dir, quantization)

        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _locked_convert,
            )
        except FileLockTimeout:
            logger.warning(
                "[convert] auto-convert of %r timed out waiting for a peer "
                "worker (>10 min); lock file: %s", name, lock_path,
            )
            raise HTTPException(
                status_code=503,
                detail=f"Auto-convert of {name!r} timed out waiting for "
                       f"a peer worker (>10 min).",
            )
    return output_dir


# Shared GPU inference limiter — caps concurrent model.transcribe() calls across
# BOTH the streaming WebSocket and the batch /transcribe route so they don't
# oversubscribe the GPU under the ~10-concurrent target. Built lazily on first
# use (binds to the running loop); width = cfg.INFERENCE_CONCURRENCY (restart-
# required). streaming/routes.py acquires the same object via this getter.
_inference_semaphore: "asyncio.Semaphore | None" = None


def get_inference_semaphore() -> "asyncio.Semaphore":
    global _inference_semaphore
    if _inference_semaphore is None:
        n = max(1, int(getattr(cfg, "INFERENCE_CONCURRENCY", 2)))
        # A timed semaphore: same `async with` contract, plus queue depth /
        # oldest wait for /stats and the per-request wait_s on the ledger.
        _inference_semaphore = metrics.GpuGate(n)
        metrics.gpu_gate = _inference_semaphore
    return _inference_semaphore


# Separate limiter for transcribe-from-URL downloads: network-bound work that
# must NOT occupy a GPU slot (a slow site would starve inference otherwise).
# Same lazy-build/restart-required contract as the inference semaphore.
_url_download_semaphore: "asyncio.Semaphore | None" = None


def _get_url_download_semaphore() -> "asyncio.Semaphore":
    global _url_download_semaphore
    if _url_download_semaphore is None:
        n = max(1, int(getattr(cfg, "URL_DOWNLOAD_CONCURRENCY", 2)))
        _url_download_semaphore = asyncio.Semaphore(n)
    return _url_download_semaphore


# faster-whisper short name OR HuggingFace repo id (org/name) — the same shape
# settings_schema._MODEL_ID_PATTERN validates configured model ids against. Used
# below to bound what an EMPTY ALLOWED_MODELS accepts from a request.
# \Z, not $: `$` also matches just BEFORE a trailing newline, so "some-repo\n"
# passed the gate. \Z anchors at the true end of the string and is a pure
# tightening here (no legitimate model id ends in a newline); fixing it in the
# pattern also covers every other caller of this regex, which stripping inside
# _resolve_model_name would not.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?\Z")


def _check_model_allowed(name: str) -> None:
    """400 when an ALLOWED_MODELS allowlist is set and ``name`` is not on it."""
    if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' is not in the allowed list. "
                   f"Allowed: {sorted(cfg.ALLOWED_MODELS)}",
        )


def _check_model_shape(name: str) -> None:
    """400 for a malformed model id when no allowlist is set.

    No allowlist configured: the name arrives verbatim from the request, and
    in the loader it reaches os.path.isdir() / the HF hub / the CT2 converter.
    Accept DEFAULT_MODEL (which may legitimately be a local directory) and
    otherwise only well-formed model ids — no filesystem paths, no "..", no URLs."""
    if (
        not cfg.ALLOWED_MODELS
        and name != cfg.DEFAULT_MODEL
        and (".." in name or not _MODEL_ID_RE.match(name))
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' is not a valid model id. Use a "
                   f"faster-whisper name or a HuggingFace repo id, or add it "
                   f"to ALLOWED_MODELS.",
        )


def _check_model_name(name: str) -> None:
    """Both model gates the loader applies, without loading anything — for
    endpoints that only read a model's config (GET /v1/request-default-settings)."""
    _check_model_allowed(name)
    _check_model_shape(name)


async def _get_or_load_model(name: str, *, lease: bool = False) -> "WhisperModel":
    """Return the cached WhisperModel for ``name``, loading it on a miss.

    ``lease=True`` marks the model as in-use until the caller passes the same
    name to :func:`_release_model_lease` (a `finally:` — see the transcribe
    handler). A leased model survives LRU and idle eviction. The startup
    preload deliberately does NOT lease: it wants plain LRU/idle semantics."""
    # Lazy import (see the TYPE_CHECKING note up top): only when a model is
    # actually loaded do we need the native faster_whisper stack.
    from faster_whisper import WhisperModel  # noqa: F401  (used in executor lambdas below)

    # The allowlist is read live per request and is in neither
    # settings_schema.RESTART_REQUIRED_FIELDS nor LOAD_TIME_FIELDS, so narrowing it
    # is reported to the admin as hot-applied and evicts nothing. Gate BEFORE the
    # cache fast path, or a model that is still resident keeps being served to
    # clients after the admin withdrew it (MODEL_IDLE_TIMEOUT_S defaults to 0,
    # so the entry only leaves on LRU pressure or restart).
    _check_model_allowed(name)

    cached = _loaded_models.get(name)
    if cached is not None:
        # Tolerate the race against _drop_loaded_model from _idle_evictor
        # or drain_then_evict, both of which hold _model_load_lock; this
        # cache-hit fast path runs lock-free so move_to_end can KeyError
        # if the entry was popped between .get() and here.
        try:
            _loaded_models.move_to_end(name)
        except KeyError:
            pass
        model_registry.touch_loaded_model(name)
        if lease:
            _model_leases[name] = _model_leases.get(name, 0) + 1
        return cached

    _check_model_shape(name)

    # Auto-convert HF transformers Whisper repos to CT2 format if enabled.
    # Runs OUTSIDE _model_load_lock so loads of OTHER cached models stay
    # snappy during the (rare, slow) conversion step. Returns `name`
    # unchanged if conversion is off, repo is already CT2, or it's a
    # local path with model.bin.
    load_path = await _ensure_ct2_model(name)

    loop = asyncio.get_running_loop()
    # Per-model override > global default. Each loaded model can pin its
    # own device/compute_type/etc. independently.
    primary_device = effective_config.cfg_for(name, "MODEL_DEVICE")
    primary_compute = effective_config.cfg_for(name, "MODEL_COMPUTE_TYPE")
    fallback_device = effective_config.cfg_for(name, "MODEL_DEVICE_FALLBACK")
    fallback_compute = effective_config.cfg_for(name, "MODEL_COMPUTE_TYPE_FALLBACK")
    # Load-time hardware kwargs (also per-model overrideable).
    load_kwargs = {
        "device": primary_device,
        "compute_type": primary_compute,
        "device_index": effective_config.cfg_for(name, "DEVICE_INDEX"),
        "cpu_threads": effective_config.cfg_for(name, "CPU_THREADS"),
        "num_workers": effective_config.cfg_for(name, "NUM_WORKERS"),
    }
    # Optional load-time fields — only forwarded if non-default to keep
    # WhisperModel(...) clean for the common path.
    _download_root = effective_config.cfg_for(name, "DOWNLOAD_ROOT")
    if _download_root:
        load_kwargs["download_root"] = _download_root
    if effective_config.cfg_for(name, "LOCAL_FILES_ONLY"):
        load_kwargs["local_files_only"] = True
    _auth_token = effective_config.cfg_for(name, "HF_TOKEN")
    if _auth_token:
        load_kwargs["use_auth_token"] = _auth_token
    # PM-only field (no global counterpart): read directly from override.
    _overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
    _m_over = _overrides.get(name) if isinstance(_overrides, dict) else None
    _revision = _m_over.get("REVISION") if isinstance(_m_over, dict) else None
    if _revision:
        load_kwargs["revision"] = _revision

    # Pre-download the repo under a progress capture when the weights
    # will come from the Hub — faster-whisper hardcodes a disabled tqdm,
    # so the constructor's own multi-GB fetch is otherwise invisible in
    # the log / jobs registry. Same snapshot args as faster_whisper's
    # download_model (allow_patterns, cache_dir=download_root, revision,
    # token), so the constructor then finds a warm cache. Best-effort:
    # ANY failure falls through to the constructor's stock download.
    #
    # Runs OUTSIDE _model_load_lock, like _ensure_ct2_model above and for the
    # same reason: held across the fetch, one cold multi-GB download stalled
    # EVERY other whisper load on the server for its whole duration. The
    # re-check under the lock below is what keeps a concurrent loader that
    # won the race honoured.
    if not load_kwargs.get("local_files_only") and not os.path.isdir(load_path):
        try:
            from faster_whisper_backend.runtime import download_progress
            from huggingface_hub import snapshot_download
            _dl_repo = load_path
            if "/" not in _dl_repo:
                from faster_whisper.utils import _MODELS as _FW_MODELS
                _dl_repo = _FW_MODELS.get(_dl_repo) or ""
            if _dl_repo:
                _dl_label = f"whisper:{name}"
                _dl_job = jobs.job_start("download", model=_dl_label)

                def _dl_hook(done, total, _job=_dl_job):
                    jobs.job_update(
                        _job,
                        progress=(done / total) if total else None,
                        total_bytes=total or None)

                _snap_kwargs = {
                    "repo_id": _dl_repo,
                    "allow_patterns": [
                        "config.json", "preprocessor_config.json",
                        "model.bin", "tokenizer.json", "vocabulary.*",
                    ],
                }
                if _download_root:
                    _snap_kwargs["cache_dir"] = _download_root
                if _revision:
                    _snap_kwargs["revision"] = _revision
                if _auth_token:
                    _snap_kwargs["token"] = _auth_token
                try:
                    with download_progress.capture(
                            _dl_label, cb=_dl_hook) as _cap:
                        _snap_kwargs.update(_cap.tqdm_kwargs)
                        await loop.run_in_executor(
                            None,
                            lambda: snapshot_download(**_snap_kwargs))
                finally:
                    jobs.job_end(_dl_job)
        except Exception as _dl_err:  # noqa: BLE001 — best-effort
            logger.warning(
                "Pre-download of %s failed (%s); the model constructor "
                "will download instead", name, _dl_err)

    # load_secs = constructor time only: the best-effort Hub pre-download
    # above is excluded the same way the CT2 conversion and the lock wait are.
    load_t0 = time.perf_counter()
    _lock_wait_t0 = time.perf_counter()
    async with _model_load_lock:
        # Time spent queueing behind the lock is another model's load cost,
        # not this one's — keep it out of load_secs.
        _lock_wait = time.perf_counter() - _lock_wait_t0
        # Re-check under the lock — another request may have loaded it.
        cached = _loaded_models.get(name)
        if cached is not None:
            _loaded_models.move_to_end(name)
            model_registry.touch_loaded_model(name)
            if lease:
                _model_leases[name] = _model_leases.get(name, 0) + 1
            return cached

        # Evict the least-recently-used UNLEASED model(s) until we have room.
        while len(_loaded_models) >= cfg.MAX_LOADED_MODELS:
            evicted_name = next(
                (n for n in _loaded_models if not _model_leases.get(n, 0)), None)
            if evicted_name is None:
                # Every cached model is mid-request — overflow the cap rather
                # than free a model under a running decode. The excess is
                # reclaimed by the next load that finds an unleased entry, or
                # by the idle evictor when MODEL_IDLE_TIMEOUT_S > 0 (it is 0
                # by default, so do not rely on it).
                logger.warning(
                    "All %d cached models are in use — temporarily exceeding "
                    "MAX_LOADED_MODELS", len(_loaded_models))
                break
            logger.info("Evicting model from VRAM (LRU, max=%d): %s",
                        cfg.MAX_LOADED_MODELS, evicted_name)
            _drop_loaded_model(evicted_name)

        logger.info("Loading model: %s", name)
        # NVML delta sampling: compare GPU memory before/after construction
        # to estimate this model's VRAM footprint. Done under
        # _model_load_lock so concurrent loads can't pollute the delta.
        # Subsequent loads of the same size may under-report due to
        # CTranslate2's caching allocator (cached freed memory gets reused).
        vram_before = system_stats.gpu_mem_used_bytes()
        loaded_device = primary_device
        loaded_compute = primary_compute
        try:
            # `load_path` is `name` for already-CT2 / local repos; for
            # auto-converted HF repos it's the local converted directory.
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(load_path, **load_kwargs),
            )
            _decode_trace.install(new_model)
            logger.info("Model loaded on %s: %s", primary_device, name)
        except Exception as e:
            logger.error("%s load failed for %s, falling back to %s: %s",
                         primary_device, name, fallback_device, e)
            fallback_kwargs = {
                **load_kwargs,
                "device": fallback_device,
                "compute_type": fallback_compute,
            }
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(load_path, **fallback_kwargs),
            )
            _decode_trace.install(new_model)
            loaded_device = fallback_device
            loaded_compute = fallback_compute
            logger.info("Model loaded on %s: %s", fallback_device, name)

        load_secs = time.perf_counter() - load_t0 - _lock_wait
        metrics.record_model_load(name, load_secs)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram_delta = (vram_after - vram_before
                      if vram_before is not None and vram_after is not None
                      else None)
        # Negative deltas can happen if another process freed VRAM during load
        # (or the CT2 allocator did). Clamp to 0 rather than store nonsense.
        if vram_delta is not None and vram_delta < 0:
            vram_delta = 0
        # Off the loop: register_loaded_model persists the measurement
        # (model_sizes.record -> atomic_json.save_lock + fsync), which can
        # block for the lock timeout when a peer worker holds the file.
        await asyncio.to_thread(
            model_registry.register_loaded_model,
            name,
            vram_bytes=vram_delta,
            device=loaded_device,
            compute_type=loaded_compute,
            load_secs=load_secs,
        )

        _loaded_models[name] = new_model
        if lease:
            _model_leases[name] = _model_leases.get(name, 0) + 1
        return new_model


# =============================================================================
# Residency API (runtime.preload's view of this cache)
# =============================================================================
# The preloader asks the same handful of questions of all four model caches.
# These answer them for the whisper cache so the invariants behind each answer
# live next to the state they describe instead of being hand-copied into
# preload. Every function reads the module globals at call time, so a test
# that rebinds _loaded_models / _model_leases / _model_load_lock still steers
# them.

def resolve_model_name(requested: str) -> str:
    """The id this cache keys a request for ``requested`` by (``whisper-1``
    or empty → cfg.DEFAULT_MODEL). Public face of _resolve_model_name."""
    return _resolve_model_name(requested)


def is_resident(name: str) -> bool:
    """Is ``name`` (an already-resolved id) loaded right now?"""
    return name in _loaded_models


def load_in_progress() -> bool:
    """True while a whisper load holds _model_load_lock. The lock is held
    across the load itself, so a speculative warm-up that waited on it would
    be the reason a real job's load queues behind it."""
    return _model_load_lock.locked()


def cache_full() -> bool:
    """Is the cache at MAX_LOADED_MODELS? A load past that point goes through
    _get_or_load_model's cap loop, which evicts the LRU UNLEASED model without
    consulting the warm predicate."""
    cap = max(1, int(getattr(cfg, "MAX_LOADED_MODELS", 1) or 1))
    return len(_loaded_models) >= cap


def idle_peer(exclude: str,
              is_warm: "Callable[[str], bool]") -> "str | None":
    """The first loaded model, in LRU order, other than ``exclude`` that no
    request holds a lease on and for which ``is_warm(name)`` is false — a
    model that could be dropped to make room — or None."""
    for name in _loaded_models:
        if name == exclude or _model_leases.get(name, 0):
            continue
        if not is_warm(name):
            return name
    return None


def placement(name: "str | None") -> "tuple[str, str]":
    """(device, compute_type) a load of ``name`` registers under: the same
    per-model ``cfg_for`` ladder (MODEL_OVERRIDES > global) _get_or_load_model
    resolves it through, not the global fields alone. Never raises — falls
    back to the global fields."""
    try:
        return ((effective_config.cfg_for(name, "MODEL_DEVICE") or "cpu"),
                (effective_config.cfg_for(name, "MODEL_COMPUTE_TYPE") or ""))
    except Exception:  # noqa: BLE001 — a placement probe must never raise
        return ((getattr(cfg, "MODEL_DEVICE", "cpu") or "cpu"),
                (getattr(cfg, "MODEL_COMPUTE_TYPE", "") or ""))


async def load_unleased(name: str) -> None:
    """Load ``name`` WITHOUT a lease: the model stays evictable by the LRU and
    idle paths the moment a real request needs the memory."""
    await _get_or_load_model(name)


async def evict(name: str) -> bool:
    """Drop one cached model under _model_load_lock (racy with loads
    otherwise). Declines (False) while a request holds a lease on it."""
    async with _model_load_lock:
        return _drop_loaded_model(name)


async def drain_then_evict(model_id: "str | None" = None) -> list[str]:
    """Drain-then-evict pattern. Drops the cached entry for `model_id` (or all
    entries when None) so the next request for that id reloads the model with
    current cfg / per-model settings.

    "Drain" comes for free from Python reference counting: in-flight transcribe
    requests already hold their own `model` reference (captured via `_get_or_
    load_model` before the executor call), so they continue running on the
    old WhisperModel instance until they finish. Only NEW requests for the
    evicted id pay the reload cost. Returns the list of evicted ids.

    Called from pipeline.apply.apply_hot_changes when a load-time field (MODEL_DEVICE,
    MODEL_COMPUTE_TYPE, NUM_WORKERS, DEVICE_INDEX, …) changes either globally
    or in a per-model override. Either case can require reload to take
    effect; this helper makes that reload lazy and non-disruptive.
    """
    evicted: list[str] = []
    async with _model_load_lock:
        if model_id is None:
            names = list(_loaded_models.keys())
        else:
            names = [model_id] if model_id in _loaded_models else []
        for name in names:
            logger.info("[evict-on-edit] dropping %s from cache; "
                        "reload on next request", name)
            # force: the drain contract above IS the lease's guarantee — an
            # in-flight request keeps its own reference and finishes on it.
            _drop_loaded_model(name, force=True)
            evicted.append(name)
    return evicted


async def _idle_evictor() -> None:
    """Periodically unload models that haven't been touched for
    cfg.MODEL_IDLE_TIMEOUT_S seconds. Wakes every 30 s; cheap when
    timeout is 0 (early return) or no models are loaded. Acquires the
    same _model_load_lock used by _get_or_load_model so concurrent loads
    can't race with eviction.

    VRAM reclamation: pop the WhisperModel reference from _loaded_models
    so its CT2 destructor can run, then gc.collect() to break any
    remaining cycles. If torch is importable and CUDA is active, also
    call torch.cuda.empty_cache() to release pool-cached blocks.
    """
    import gc
    while True:
        try:
            await asyncio.sleep(30)
            timeout = getattr(cfg, "MODEL_IDLE_TIMEOUT_S", 0) or 0
            if timeout <= 0 or not _loaded_models:
                continue
            now = time.monotonic()
            stale: list[str] = []
            for name, info in list(model_registry._loaded_models.items()):
                if name not in _loaded_models:
                    continue
                if model_registry.is_warm(name):
                    continue
                last = info.get("last_used_monotonic", now)
                if now - last >= timeout:
                    stale.append(name)
            if not stale:
                continue
            async with _model_load_lock:
                now = time.monotonic()
                for name in stale:
                    if name not in _loaded_models:
                        continue
                    if model_registry.is_warm(name):
                        continue
                    info = model_registry._loaded_models.get(name)
                    if info and now - info.get("last_used_monotonic", now) < timeout:
                        continue
                    if _drop_loaded_model(name):
                        logger.info("[idle-evict] unloaded %s after %ds idle",
                                    name, timeout)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("[idle-evict] evictor loop error")


def _reset_for_tests() -> None:
    """Drop every cached model, lease and suppress-chars entry, and rebind the
    locks and lazily-built semaphores so none stays bound to a previous test's
    event loop (metrics' own hook clears the published metrics.gpu_gate)."""
    global _model_load_lock, _convert_locks_meta
    global _inference_semaphore, _url_download_semaphore
    _loaded_models.clear()
    _model_leases.clear()
    _suppress_chars_cache.clear()
    _convert_locks.clear()
    _model_load_lock = asyncio.Lock()
    _convert_locks_meta = asyncio.Lock()
    _inference_semaphore = None
    _url_download_semaphore = None
