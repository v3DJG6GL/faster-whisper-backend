import os
import json
import random
import sys
import ctypes
import logging
import logging.handlers
import math
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

# BOOT_ID (the per-process restart marker surfaced via /v1/models) lives in
# build_info with the rest of the server identity; imported this early —
# before config — so it exists exactly as soon as it used to.
from build_info import APP_VERSION, BOOT_ID, SERVER_NAME

import config as cfg
# system_stats imports psutil + pynvml at module load and primes psutil's
# non-blocking counters. Imported here (early) so the priming happens before
# any request handler runs.
import system_stats
# Imported for the log-file mode hardening below; module-level cost is nil (os).
import store_common
# Text-to-text translation stage (llama.cpp GGUF). Module-level import is
# deliberate and cheap — like diarization/bgm_separation the module is
# import-safe without its optional deps (llama_cpp loads lazily inside the
# model-load path), and the stage + lifespan both need it.
import translation as _tr
# Shared per-identity limiters. Imports only stdlib + fastapi + config, so it
# is safe this early and cannot close an import cycle back through main.
import rate_limit as _rl

# =============================================================================
# Logging setup: stderr (with colors when TTY) + rotating file (no colors)
# =============================================================================
# Log path and rotation policy come from config.py / WHISPER_LOG_FILE.
# The file copy strips ANSI escape codes so it stays grep-friendly and the
# /logs web viewer can re-color via CSS based on content.
# An uncreatable log dir (e.g. the container-first /data default on a
# bare-metal box without WHISPER_DATA_DIR) must not kill the import — the
# server degrades to stderr-only logging, the standard container posture.
_log_dir_ok = True
try:
    os.makedirs(os.path.dirname(cfg.LOG_FILE), exist_ok=True)
except OSError as _log_exc:
    _log_dir_ok = False
    print(
        f"WARNING: cannot create log directory for {cfg.LOG_FILE!r} ({_log_exc}) "
        "— file logging disabled, logging to stderr only. Set WHISPER_LOG_FILE "
        "or WHISPER_DATA_DIR to a writable location.",
        file=sys.stderr,
    )

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    # Emit file timestamps in UTC (ISO-8601 with a trailing 'Z'). The log file
    # is then unambiguous regardless of the server's timezone; the /logs web
    # viewer converts each line to the reader's local time (like every other
    # timestamp surface). gmtime is a class attribute so it applies to asctime.
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE_RE.sub("", super().format(record))


_root = logging.getLogger()
_root.setLevel(logging.INFO)
# Remove any handlers a previous import (or basicConfig) added so we don't
# double-log on auto-reload.
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
_root.addHandler(_console_handler)

class _SecureRotatingFileHandler(logging.handlers.RotatingFileHandler):
    # Every request block written here carries RAW WHISPER / FINAL transcript
    # text, the upload filename, the username and the key label — the same
    # plaintext dictation store_common.secure_db_file() keeps at 0600 in the
    # SQLite stores. The handler would otherwise create the file at the process
    # umask (0644 typical), so tighten it on open and after every rollover.
    def _open(self):  # type: ignore[override]
        stream = super()._open()
        store_common.secure_file(self.baseFilename)
        return stream


if _log_dir_ok:
    store_common.secure_dir(os.path.dirname(cfg.LOG_FILE))
    _file_handler = _SecureRotatingFileHandler(
        cfg.LOG_FILE, maxBytes=cfg.LOG_MAX_BYTES, backupCount=cfg.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _file_handler.setFormatter(_StripAnsiFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",   # UTC (converter=gmtime); viewer localizes
    ))
    _root.addHandler(_file_handler)

# Tail WARNING+ records into an in-memory ring used by the nav-row severity
# pills and the /stats page. Does no I/O — append to a deque and return.
from web_common import SeverityCounter
_root.addHandler(SeverityCounter())

logger = logging.getLogger("whisper-api")

# Surface any env-var coercion problems collected while config.py was imported
# (it runs before logging is configured, so it just stashes messages).
for _msg in getattr(cfg, "_ENV_WARNINGS", ()):
    logger.warning("config env override ignored: %s", _msg)


# =============================================================================
# Hugging Face token propagation
# =============================================================================
# faster-whisper accepts `use_auth_token=` per-WhisperModel-call and forwards
# it to huggingface_hub.snapshot_download(token=...). That covers the model
# weights download. But OTHER HF calls in the process — Silero VAD model
# load, tokenizer fetches, metadata pings — don't see that kwarg and would
# log "unauthenticated requests" warnings + hit the lower anonymous rate
# limit. Promoting cfg.HF_TOKEN to os.environ["HF_TOKEN"] silences
# those calls AND lifts the ceiling. Per-model HF_TOKEN overrides
# still win at the per-WhisperModel-call kwarg level, so a model that
# needs a different token (rare) still works.
#
# Live edits: admin_routes.post_state re-syncs the env var whenever
# cfg.HF_TOKEN changes via the admin UI, so a save takes effect without
# a service restart. Clearing the config field unsets the env var.
if cfg.HF_TOKEN:
    os.environ["HF_TOKEN"] = cfg.HF_TOKEN
    logger.info("HF_TOKEN set from cfg.HF_TOKEN (silences HF rate-limit "
                "warnings for non-WhisperModel calls)")


def _preload_windows_cuda_dlls() -> None:
    base_path = os.path.dirname(sys.executable)
    if os.path.basename(base_path).lower() == "scripts":
        base_path = os.path.dirname(base_path)

    nvidia_base = os.path.join(base_path, "Lib", "site-packages", "nvidia")
    cudnn_bin = os.path.join(nvidia_base, "cudnn", "bin")
    cublas_bin = os.path.join(nvidia_base, "cublas", "bin")

    # -Full installs bring torch, whose Windows cu126 wheel BUNDLES its own
    # cuDNN/cuBLAS in torch\lib (it does not use the nvidia-* wheels above).
    # Two cudnn64_9.dll versions then live in one venv, Windows resolves DLL
    # dependencies by module NAME with first-loaded-wins — so preloading the
    # newer wheel here made the other family's cudnn_cnn64_9.dll fail with
    # WinError 127 (procedure not found) at model load. Prefer torch's copy so
    # the process holds ONE consistent stack — the same outcome the Linux
    # -full image reaches by letting pip downgrade the nvidia wheels to
    # torch's pins (see Dockerfile.gpu). cuDNN 9.x / cuBLAS 12.x is all
    # ctranslate2 requires. Bonus: torch\lib also carries cudart/cufft/curand,
    # which onnxruntime-gpu's CUDA provider needs and the lean dirs lack.
    torch_lib = os.path.join(base_path, "Lib", "site-packages", "torch", "lib")
    if os.path.isfile(os.path.join(torch_lib, "cudnn64_9.dll")):
        cudnn_bin = cublas_bin = torch_lib

    logger.info("Base path: %s", base_path)
    logger.info("cuDNN path: %s", cudnn_bin)

    # Idempotent prepend: this runs on every `import main` — including the
    # importlib.reload(main) the test suite does once per app_module test — so a
    # naive unconditional prepend grows PATH without bound until it trips
    # Windows' 32767-char per-variable limit (and bloats the env block enough to
    # fail subprocess spawns with WinError 8). Only add dirs not already present.
    parts = os.environ.get("PATH", "").split(os.pathsep)
    missing = [d for d in (cudnn_bin, cublas_bin) if d not in parts]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + parts)

    if hasattr(os, "add_dll_directory"):
        if os.path.exists(cudnn_bin):
            os.add_dll_directory(cudnn_bin)
        if os.path.exists(cublas_bin):
            os.add_dll_directory(cublas_bin)

    dlls = [
        (cublas_bin, "cublas64_12.dll"),
        (cublas_bin, "cublasLt64_12.dll"),
        (cudnn_bin, "cudnn_graph64_9.dll"),
        (cudnn_bin, "cudnn_ops64_9.dll"),
        (cudnn_bin, "cudnn_cnn64_9.dll"),
        (cudnn_bin, "cudnn_adv64_9.dll"),
        (cudnn_bin, "cudnn64_9.dll"),
    ]
    try:
        for directory, name in dlls:
            ctypes.CDLL(os.path.join(directory, name))
        logger.info("NVIDIA DLLs pre-loaded successfully.")
    except OSError as e:
        logger.warning("Failed to pre-load DLLs: %s", e)


def _add_local_ffmpeg_to_path() -> None:
    """install-service.ps1 -Full drops a pinned shared-build ffmpeg into
    <repo>\\ffmpeg\\bin when no shared ffmpeg is on PATH (torchcodec and
    audio-separator load the avutil/avcodec DLLs, which the bundled
    imageio-ffmpeg executable does not ship). Make it visible to this process
    and its subprocesses. Prepended so its DLLs also win over a static build
    elsewhere on PATH. Idempotent — same reload concern as the CUDA preloader."""
    ff_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")
    if not os.path.isfile(os.path.join(ff_bin, "ffmpeg.exe")):
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if ff_bin not in parts:
        os.environ["PATH"] = os.pathsep.join([ff_bin] + parts)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(ff_bin)
    logger.info("Repo-local ffmpeg on PATH: %s", ff_bin)


if sys.platform == "win32":
    _preload_windows_cuda_dlls()
    _add_local_ffmpeg_to_path()


from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, Depends

# Auth dep used by /v1/audio/transcriptions and /auth/whoami. In open mode
# (no admin key in DB) it returns the synthetic admin — but only to callers on
# ADMIN_WEBUI_ALLOWED_HOSTS — so the operator can bootstrap; otherwise it 401s
# on missing/invalid bearer.
from auth import Permissions, get_current_user as _get_current_user_dep
from auth import open_mode_host_ok as _open_mode_host_ok
from auth import user_from_session_cookie as _user_from_session_cookie

# faster_whisper pulls the heavy native stack (ctranslate2/onnxruntime/av). It is
# imported lazily at first model load (see _get_or_load_model) so this module
# stays importable for tests/tooling/template rendering on a box without the CUDA
# stack installed. TYPE_CHECKING keeps the WhisperModel annotation resolvable for
# type checkers without importing it at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from faster_whisper import WhisperModel


# =============================================================================
# Text post-processing pipeline — unified rules list (cfg.PIPELINE_RULES)
# =============================================================================
# A single ordered list of rules is applied to the joined transcript. Each
# rule is one of: "regex-list" (ordered batch of pattern→replacement entries), "callback:lowercase-wordlist"
# (smart German non-noun lowercaser), "callback:map" (dictation word→symbol
# map), "callback:dedup" (collapse adjacent punctuation runs), "callback:upper"
# (capitalize after sentence terminator), or "terminal" (final lstrip+rstrip;
# always last). See config.py:PIPELINE_RULES for the canonical seeded list and
# config_store.py for the Pydantic schema.
#
# rebuild_caches() compiles each rule's regex pattern once at module load and
# again on admin WebUI save (CACHE_REBUILD_FIELDS = {"PIPELINE_RULES"}).
# Disabled rules and skipped types (terminal, empty patterns) are filtered
# out of the compiled list — the runtime walker is just a tight for-loop.

from dataclasses import dataclass


@dataclass(frozen=True)
class _CompiledRule:
    """One row of the compiled pipeline. `payload` carries type-specific data:
      regex-list (per entry)      → replacement string
      callback:lowercase-wordlist → frozenset[str] of lowercase words
      callback:map                → dict[str_lower, str] lookup
      callback:dedup              → None (callback hardcoded)
      callback:upper              → None (callback hardcoded)
    `name` is the rule slug used for per-model EXCLUDE / INCLUDE matching.
    `enabled` mirrors the global `rule.enabled` flag — checked at runtime
    rather than at compile time so per-model PIPELINE_RULES_INCLUDE can
    force-enable a globally-disabled rule.
    """
    name: str
    label: str
    type: str
    pattern: "re.Pattern[str]"
    payload: object
    enabled: bool
    # 1-based position of this rule's CARD in cfg.PIPELINE_RULES (terminal card
    # included) — identical to the /settings/pipeline card ordinal. `sub_no` is
    # the 1-based entry row within a regex-list card (None for single-compile
    # rules). Together they drive the `#P` / `#P.S` trace step number.
    card_no: int
    sub_no: "int | None" = None


_COMPILED_RULES: list[_CompiledRule] = []
# Captured from the terminal rule's name/label in cfg.PIPELINE_RULES at cache
# build time. Falls back to the constants below if the user removed the
# terminal row. The slug is used to honor exclude-set membership (a captures
# pipeline can drop the trim by listing `trim-edges` in
# CAPTURES_PIPELINE_RULES_EXCLUDE).
_TERMINAL_NAME: str = "trim-edges"
_TERMINAL_LABEL: str = "Trim edges (always-last)"
# 1-based card position of the terminal trim, captured at rebuild time so its
# trace step number matches the /settings/pipeline card ordinal. Falls back to
# len(PIPELINE_RULES)+1 when the user deleted the terminal row.
_TERMINAL_CARD_NO: int = 0
# Ceiling on the pipeline's output, checked after every rule. A 30-minute
# transcript is a few hundred KB, so 4 M chars only ever trips on a rule set
# that expands its own input (the pipeline runs on admin-editable regexes).
_POSTPROCESS_MAX_CHARS = 4_000_000


def _rule_ordinal(card_no: int, sub_no: "int | None" = None) -> str:
    """Format a pipeline step number to match the /settings/pipeline card
    position: '#P' for a single-compile rule, '#P.S' for a regex-list entry
    (S = the entry's 1-based row within that card)."""
    return f"#{card_no}.{sub_no}" if sub_no else f"#{card_no}"


def _dedup_callback(match: "re.Match[str]") -> str:
    """Pick the user-intended punct from a run of 2+ adjacent marks. Whisper
    emits its own commas as soft pauses around dictation keywords; after
    substitution we get ",." (Punkt) or ",;" (Semikolon). Prefer any non-
    comma; within non-commas prefer the LAST (dictation came after the
    Whisper pause). Pure commas → single comma."""
    run = match.group(0)
    non_comma = [c for c in run if c != ","]
    return non_comma[-1] if non_comma else ","


def _upper_callback(match: "re.Match[str]") -> str:
    """Uppercase group(2) if the pattern produced two groups; else uppercase
    the entire match. Default seeded pattern produces ([.?!]\\s+|\\n+\\s*)
    + ([a-zäöüß])."""
    try:
        g1, g2 = match.group(1), match.group(2)
    except IndexError:
        return match.group(0).upper()
    return g1 + g2.upper()


def _make_lowercase_wordlist_replacer(wordlist: frozenset):
    """Returns a regex-sub callback that strips the matched terminator and
    lowercases group(2) IFF (group(2)+group(3)).lower() is in `wordlist`.
    The default seeded pattern produces three groups:
      group(1) = whitespace between terminator and next word
      group(2) = first letter of the next word
      group(3) = rest of the next word
    If the user's pattern has fewer than 3 groups we degrade to plain strip.
    """
    def replace(m: "re.Match[str]") -> str:
        try:
            ws, first, rest = m.group(1), m.group(2), m.group(3)
        except IndexError:
            return ""
        if (first + rest).lower() in wordlist:
            return ws + first.lower() + rest
        return ws + first + rest
    return replace


def _make_map_replacer(lookup: dict):
    """Returns a regex-sub callback that does a case-insensitive dict lookup
    on the entire match. Used by callback:map rules."""
    def replace(m: "re.Match[str]") -> str:
        return lookup.get(m.group(0).lower(), m.group(0))
    return replace


def rebuild_caches() -> None:
    """(Re)compile every rule in cfg.PIPELINE_RULES into _COMPILED_RULES.

    Called once at module load (just below) and again by the admin WebUI
    after a config change to PIPELINE_RULES (CACHE_REBUILD_FIELDS).

    The terminal row is filtered out (it runs as the implicit final trim,
    not via the walker). Globally-DISABLED rules are still compiled — the
    runtime filter consults `rule.enabled` per-call so per-model
    PIPELINE_RULES_INCLUDE can force-enable a globally-disabled rule. Rules
    with invalid regex are logged + skipped (the save-time validator
    usually catches these, but a hand-edited config.py or a runtime
    catastrophic-backtracking case might surface here).
    """
    global _COMPILED_RULES, _TERMINAL_NAME, _TERMINAL_LABEL, _TERMINAL_CARD_NO
    compiled: list[_CompiledRule] = []
    terminal_name = _TERMINAL_NAME
    terminal_label = _TERMINAL_LABEL
    terminal_card_no = len(cfg.PIPELINE_RULES) + 1
    for card_no, rule in enumerate(cfg.PIPELINE_RULES, start=1):
        rtype = rule.get("type")
        if rtype == "terminal":
            terminal_name = rule.get("name", terminal_name)
            terminal_label = rule.get("label", terminal_label)
            terminal_card_no = card_no
            continue
        rule_enabled = bool(rule.get("enabled", True))

        # regex-list: expand each entry into its own _CompiledRule row, all
        # sharing the card's name + enabled (so per-model EXCLUDE/INCLUDE and the
        # global toggle flip the whole card together). Entries run in list order
        # — NO longest-first sort. A bad entry is skipped (not the whole card);
        # an empty-pattern entry is a no-op. Each row is a plain string-replacement
        # sub, exactly like the retired single `regex` type.
        if rtype == "regex-list":
            rname = rule.get("name", "?")
            rlabel = rule.get("label", rname)
            for sub_no, entry in enumerate(rule.get("entries", []) or [], start=1):
                epat = entry.get("pattern", "")
                if not epat:
                    continue
                try:
                    ecre = re.compile(epat)
                except re.error as e:
                    logger.warning("[pipeline] rule %r entry has invalid regex "
                                   "(%s) — skipping entry", rname, e)
                    continue
                # sub_no is the entry's row in the card editor (the enumerate index
                # advances across skipped empty/bad entries above), so the trace
                # number `#card_no.sub_no` lines up with what the admin sees.
                compiled.append(_CompiledRule(
                    rname, f"{rlabel} · {entry.get('label') or epat}",
                    "regex-list", ecre, entry.get("replacement", "") or "",
                    rule_enabled, card_no, sub_no))
            continue

        try:
            if rtype == "callback:map":
                # Auto-build alternation regex from map keys, longest-first,
                # word-bounded, case-insensitive — matches the legacy
                # _DICTATION_REGEX behaviour exactly.
                m = rule.get("map", {}) or {}
                if not m:
                    continue
                alternation = "|".join(re.escape(k) for k in sorted(m, key=len, reverse=True))
                cre = re.compile(r"\b(" + alternation + r")\b", re.IGNORECASE)
                # Pre-bind the per-rule replacer once at compile time. _apply_rule
                # then becomes a uniform pattern.sub(payload, text) for every rule
                # type — no closure allocation on the hot path (twice per request).
                payload: object = _make_map_replacer({k.lower(): v for k, v in m.items()})
            else:
                pattern = rule.get("pattern", "")
                if not pattern:
                    # Empty pattern on a callback rule → skip (no-op).
                    continue
                cre = re.compile(pattern)
                if rtype == "callback:lowercase-wordlist":
                    wordlist = frozenset(w.lower() for w in (rule.get("wordlist", []) or []))
                    payload = _make_lowercase_wordlist_replacer(wordlist)
                elif rtype == "callback:dedup":
                    payload = _dedup_callback
                elif rtype == "callback:upper":
                    payload = _upper_callback
                else:
                    logger.warning("[pipeline] unknown rule type %r — skipping", rtype)
                    continue
        except re.error as e:
            logger.warning("[pipeline] rule %r has invalid regex (%s) — skipping",
                           rule.get("name"), e)
            continue
        compiled.append(_CompiledRule(rule.get("name", "?"),
                                       rule.get("label", rule.get("name", "?")),
                                       rtype, cre, payload, rule_enabled, card_no))
    _COMPILED_RULES = compiled
    _TERMINAL_NAME = terminal_name
    _TERMINAL_LABEL = terminal_label
    _TERMINAL_CARD_NO = terminal_card_no


def _apply_rule(rule: _CompiledRule, text: str) -> str:
    """Dispatch on rule type. Hot path — payload is pre-bound at
    rebuild_caches() time (a replacement string for `regex-list`, a pre-built
    callable for every callback:* type), so every type collapses to a
    single pattern.sub call with no per-request closure allocation."""
    return rule.pattern.sub(rule.payload, text)  # type: ignore[arg-type]


rebuild_caches()


def _postprocess_text(text: str, model_name: "str | None" = None,
                       trace: "list | None" = None,
                       extra_excludes: "set[str] | None" = None,
                       ident=None) -> str:
    """Run the unified pipeline rule list on `text`. If `trace` is a list,
    each rule that changes the text appends `(label_with_ordinal, before, after)`
    so the per-request log block can render a diff view.

    Per-model scoping (precedence top-down):
      1. PIPELINE_RULES_EXCLUDE — force-DISABLE for this model (highest priority).
      2. PIPELINE_RULES_INCLUDE — force-ENABLE for this model, even if globally
         disabled.
      3. Otherwise inherit `rule.enabled` from the global PIPELINE_RULES list.

    Effective:  (rule.enabled AND slug NOT in EXCLUDE) OR (slug IN INCLUDE).
    A rule cannot appear in both lists — pydantic validator rejects that.

    `extra_excludes` is an additional set of rule slugs to skip on top of
    the per-model EXCLUDE. Used by the /captures storage path to produce
    a training-form transcript (cfg.CAPTURES_PIPELINE_RULES_EXCLUDE) while
    leaving the runtime /transcribe response untouched. Rules in
    `extra_excludes` are skipped even when they appear in INCLUDE — the
    captures-specific intent overrides the per-model force-on.

    The terminal "trim-edges" step (filtered out of _COMPILED_RULES at
    rebuild time) runs as the always-last step here, gated by the same
    exclude set so a trainer can preserve trailing whitespace by adding
    the slug to CAPTURES_PIPELINE_RULES_EXCLUDE. The live /transcribe
    path applies an additional unconditional trim after the output
    wrappers, so per-model exclusion of trim-edges has no effect there.
    """
    exclude: "set[str]" = set()
    include: "set[str]" = set()
    if ident is not None:
        # The resolver already folded the per-model layer into these sets
        # (identity rules win first-mention; per-model is the fallback layer).
        # Don't re-read MODEL_OVERRIDES here or it would double-apply.
        exclude = set(ident.pipeline_exclude)
        include = set(ident.pipeline_include)
    elif model_name:
        overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
        m_over = overrides.get(model_name) if isinstance(overrides, dict) else None
        if isinstance(m_over, dict):
            ex = m_over.get("PIPELINE_RULES_EXCLUDE") or []
            inc = m_over.get("PIPELINE_RULES_INCLUDE") or []
            if isinstance(ex, list):
                exclude = set(ex)
            if isinstance(inc, list):
                include = set(inc)
    if extra_excludes:
        exclude = exclude | extra_excludes
    for rule in _COMPILED_RULES:
        # Step number mirrors the /settings/pipeline card position (`#P` / `#P.S`),
        # NOT the flat index over the expanded compiled list.
        ordinal = _rule_ordinal(rule.card_no, rule.sub_no)
        # Force-EXCLUDE wins outright — admin explicitly turned this off.
        if rule.name in exclude:
            if trace is not None:
                trace.append((f"{ordinal} {rule.label} [EXCLUDED for {model_name}]",
                              text, text))
            continue
        forced_in = rule.name in include
        # Globally disabled and not force-included → skip silently.
        # When tracing, surface the skip so the log explains why a rule
        # didn't run.
        if not rule.enabled and not forced_in:
            if trace is not None:
                trace.append((f"{ordinal} {rule.label} [SKIPPED globally disabled]",
                              text, text))
            continue
        before = text
        text = _apply_rule(rule, before)
        if trace is not None:
            # Force-included rule: tag the trace line so the admin sees the
            # rule ran *because of* the per-model override, not the global
            # state. Always emit even when before == after, to make the
            # override path visible.
            if forced_in and not rule.enabled:
                trace.append(
                    (f"{ordinal} {rule.label} [FORCED on for {model_name}]",
                     before, text)
                )
            elif before != text:
                trace.append((f"{ordinal} {rule.label}", before, text))
        # Absolute output bound. Rules whose replacement template expands what
        # it matches compose: each one feeds the next, so a handful of them can
        # multiply a short transcript into gigabytes. Stop the pipeline instead
        # and keep what we have — no real transcript comes near this size.
        if len(text) > _POSTPROCESS_MAX_CHARS:
            logger.warning(
                "[pipeline] output exceeded %d chars at rule %r — remaining "
                "rules skipped", _POSTPROCESS_MAX_CHARS, rule.name)
            break
    term_ordinal = _rule_ordinal(_TERMINAL_CARD_NO)
    if _TERMINAL_NAME in exclude:
        if trace is not None:
            trace.append(
                (f"{term_ordinal} {_TERMINAL_LABEL} [EXCLUDED for {model_name}]",
                 text, text)
            )
    else:
        before_trim = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if trace is not None and before_trim != text:
            trace.append((f"{term_ordinal} {_TERMINAL_LABEL}", before_trim, text))
    return text


# =============================================================================
# Per-request log block
# =============================================================================
# Always emitted (regardless of cfg.TRACE_ENABLED) — surfaces the decode
# params actually applied + per-segment metadata so empty-output failures
# can be diagnosed from the log alone. The per-pipeline transformation
# trace is folded in only when TRACE_ENABLED.
#
# ANSI color is intentionally dropped: the service runs under WinSW (no TTY)
# and the SSE log viewer reads raw bytes — escape codes hurt both consumers.
_LOG_WIDTH = 78
_NAME_COL = 32        # value column starts at this character
_SEG_TEXT_MAX = 80    # truncate per-segment text in the table (full text in FINAL)
_SEG_ROWS_MAX = 30    # truncate the segment table itself
_LOG_FIELD_MAX = store_common.LOG_FIELD_MAX  # cap on a client-supplied label

# Single implementation lives in store_common so the stores can sanitise their
# own audit lines without importing main (which imports them).
_log_safe = store_common.log_safe


# Maps decode-kwarg name → cfg-default key in cfg._BASELINE. Used by the
# `*` non-default marker. Only scalar fields are listed; lists/dicts skipped.
# `temperature` and `suppress_tokens` are intentionally absent — their cfg
# baselines are strings ("0.0,0.2,…", "-1") while the kwargs are tuples/lists,
# so equality comparison is meaningless without parsing both sides.
_KWARG_TO_CFG = {
    # Search / sampling
    "beam_size": "BEAM_SIZE",
    "best_of": "BEST_OF",
    "patience": "PATIENCE",
    "length_penalty": "LENGTH_PENALTY",
    "repetition_penalty": "REPETITION_PENALTY",
    "no_repeat_ngram_size": "NO_REPEAT_NGRAM_SIZE",
    "prompt_reset_on_temperature": "PROMPT_RESET_ON_TEMPERATURE",
    # VAD
    "vad_filter": "VAD_FILTER",
    "min_silence_duration_ms": "VAD_MIN_SILENCE_MS",
    "speech_pad_ms": "VAD_SPEECH_PAD_MS",
    "threshold": "VAD_THRESHOLD",
    # Output shape
    "word_timestamps": "WORD_TIMESTAMPS_ENABLED",
    # Prompt context
    "condition_on_previous_text": "CONDITION_ON_PREVIOUS_TEXT",
    "initial_prompt": "DEFAULT_PROMPT",
    "hotwords": "DEFAULT_HOTWORDS",
    # Safety / thresholds
    "no_speech_threshold": "NO_SPEECH_THRESHOLD",
    "log_prob_threshold": "LOG_PROB_THRESHOLD",
    "compression_ratio_threshold": "COMPRESSION_RATIO_THRESHOLD",
    "hallucination_silence_threshold": "HALLUCINATION_SILENCE_THRESHOLD",
    # Language detection
    "multilingual": "MULTILINGUAL",
    "language_detection_threshold": "LANGUAGE_DETECTION_THRESHOLD",
    "language_detection_segments": "LANGUAGE_DETECTION_SEGMENTS",
    # Token suppression / punctuation
    "suppress_blank": "SUPPRESS_BLANK",
    "prepend_punctuations": "PREPEND_PUNCTUATIONS",
    "append_punctuations": "APPEND_PUNCTUATIONS",
    # Post-decode guards (pseudo-kwargs: rendered in the log block's guards
    # section, never passed to model.transcribe)
    "segment_max_words_per_sec": "SEGMENT_MAX_WORDS_PER_SEC",
    "tail_trim_pad_ms": "STREAMING_TAIL_TRIM_PAD_MS",
    "final_drop_min_avg_logprob": "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB",
    "final_drop_temperature": "STREAMING_FINAL_DROP_TEMPERATURE",
}


def _pretty_value(v) -> str:
    """Compact display form for a config value: `true`/`false`, `(none)` for
    None, `(empty)` for "", trimmed-zero floats, repr'd strings."""
    if v is None:
        return "(none)"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        # Preserve at least one decimal so 0.0 / -1.0 / 0.5 still read as
        # floats (not as ints). Strip extra trailing zeros only.
        s = f"{v:.2f}"
        if "." in s and s.endswith("0"):
            s = s.rstrip("0")
            if s.endswith("."):
                s += "0"
        return s
    if isinstance(v, list):
        return "[" + ", ".join(_pretty_value(x) for x in v) + "]"
    if isinstance(v, str):
        if not v:
            return "(empty)"
        if len(v) > 60:
            return repr(v[:57] + "...")
        return repr(v)
    return str(v)


def _is_non_default(key: str, value) -> bool:
    """`*` marker test. True iff a known cfg-default exists and the current
    scalar value differs from it. Skips non-scalars to avoid surprises."""
    cfg_key = _KWARG_TO_CFG.get(key)
    if not cfg_key:
        return False
    baseline_dict = getattr(cfg, "_BASELINE", None)
    if baseline_dict is None:
        return False
    baseline = baseline_dict.get(cfg_key)
    scalar = (bool, int, float, str, type(None))
    if not isinstance(value, scalar) or not isinstance(baseline, scalar):
        return False
    return value != baseline


def _param_row(indent: str, key: str, value) -> str:
    """`indent + key + spaces + value [*]` row. Value column lands at _NAME_COL
    regardless of indent depth so top-level and nested rows align."""
    star = " *" if _is_non_default(key, value) else ""
    pretty = _pretty_value(value)
    pad = max(1, _NAME_COL - len(indent) - len(key))
    return f"{indent}{key}{' ' * pad}{pretty}{star}"


def _section_rule(label: str) -> str:
    """`  ─── label ──────…` inner rule, padded to _LOG_WIDTH."""
    head = f"  ─── {label} "
    fill = max(0, _LOG_WIDTH - len(head))
    return head + ("─" * fill)


def _format_decode_params(kwargs: dict) -> list[str]:
    """Render decode params as aligned rows, with VAD parameters indented
    under vad_filter to show the relationship visually. Fields are only
    printed when present in `kwargs` — most non-default knobs (patience,
    repetition_penalty, etc.) are conditionally added at request build
    time, so absence here means "at faster-whisper / config default".
    Order is grouped by intent (search → sampling → VAD → output → context
    → thresholds → language detection → suppression)."""
    out: list[str] = []
    order = (
        # Search / sampling
        "beam_size", "best_of", "patience", "length_penalty",
        "repetition_penalty", "no_repeat_ngram_size",
        "temperature", "prompt_reset_on_temperature",
        # VAD
        "vad_filter",
        # Output shape
        "word_timestamps",
        # Prompt context
        "condition_on_previous_text", "initial_prompt", "hotwords",
        # Safety / thresholds
        "no_speech_threshold", "log_prob_threshold",
        "compression_ratio_threshold", "hallucination_silence_threshold",
        # Language detection
        "multilingual", "language_detection_threshold",
        "language_detection_segments",
        # Token suppression / punctuation
        "suppress_blank", "suppress_tokens",
        "prepend_punctuations", "append_punctuations",
    )
    for k in order:
        if k not in kwargs:
            continue
        out.append(_param_row("    ", k, kwargs[k]))
        if k == "vad_filter" and kwargs[k] and kwargs.get("vad_parameters"):
            for vk, vv in kwargs["vad_parameters"].items():
                out.append(_param_row("      ", vk, vv))
    return out


def _format_segments_section(seg_diag: list[dict], info, kwargs: dict) -> list[str]:
    """Either a fixed-width segments table OR an empty-output diagnostic
    banner whose hint depends on `info.duration_after_vad` and `kwargs`."""
    n = len(seg_diag)
    if n == 0:
        out = [_section_rule("Segments  (n=0)  [!] no output produced")]
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        dav = getattr(info, "duration_after_vad", None)
        ip = kwargs.get("initial_prompt")
        if duration > 0 and dav is not None and float(dav) < 0.3 * duration:
            out.append(f"    likely cause: VAD ate audio  "
                       f"(duration_after_vad={float(dav):.2f}s vs {duration:.2f}s)")
            out.append("    next step:    set VAD_FILTER=false or "
                       "VAD_MIN_SILENCE_MS=250 in /settings")
        elif ip:
            out.append("    likely cause: initial_prompt may be poisoning decode")
            out.append("                  (tnfru/primeline finetunes); "
                       "clear DEFAULT_PROMPT in /settings")
        else:
            out.append("    likely cause: thresholds suppressed all segments")
            out.append("                  try disabling NO_SPEECH / LOG_PROB / "
                       "COMPRESSION_RATIO thresholds in /settings")
        return out

    dropped_n = sum(1 for s in seg_diag if s.get("dropped"))
    label = f"Segments  (n={n})"
    if dropped_n:
        label += f"  [✗ = {dropped_n} dropped by post-decode guard]"
    out = [_section_rule(label)]
    out.append(
        f"    {'#':>3}  {'start':>7}  {'end':>7}  "
        f"{'alp':>6}  {'nsp':>5}  {'cr':>5}  {'T':>4}    text"
    )
    rows = min(n, _SEG_ROWS_MAX)
    for i in range(rows):
        s = seg_diag[i]
        text = s["text"]
        if len(text) > _SEG_TEXT_MAX:
            text = text[:_SEG_TEXT_MAX - 3] + "..."
        mark = "✗" if s.get("dropped") else " "
        out.append(
            f"    {s['id']:>3d}  "
            f"{s['start']:>6.2f}s  {s['end']:>6.2f}s  "
            f"{s['alp']:>+6.2f}  {s['nsp']:>5.2f}  {s['cr']:>5.2f}  "
            f"{s['temp']:>4.1f}  {mark} {text}"
        )
    if n > rows:
        out.append(f"    … (+{n - rows} more)")
    return out


def _model_compute_device(name: str) -> "tuple[str | None, str | None]":
    """Look up the actual device + compute_type a model was loaded with —
    these may differ from cfg.MODEL_* if the fallback path was taken."""
    for entry in system_stats.loaded_models_snapshot():
        if entry.get("name") == name:
            return entry.get("compute_type"), entry.get("device")
    return None, None


def _format_request_block(
    *,
    file_label: str,
    model_name: str,
    info,
    kwargs: dict,
    seg_diag: list[dict],
    raw: str,
    final: str,
    steps: "list | None" = None,
    request_id: str | None = None,
    captured_id: str | None = None,
    endpoint: str = "/v1/audio/transcriptions",
    audio_source: str | None = None,
    ident=None,
    overrides_ignored: "list | None" = None,
    user_id: str | None = None,
    key_id: str | None = None,
    username: str | None = None,
    key_label: str | None = None,
    guards: "dict | None" = None,
    translate_to: "list | None" = None,
    translation_model: str | None = None,
) -> str:
    """Full per-request log block. `steps` is the per-pipeline trace; passed
    in only when cfg.TRACE_ENABLED so the block stays a single message.

    `request_id` (uuid4 hex) is the cross-reference key between this
    durable log block and a report submitted via /quick-config. When
    present, the title line carries `req=<id[:8]>` so an admin reading
    a /reports row can grep the log for the matching block.

    `captured_id` is the capture row id when the capture pipeline fired
    for this request — admins can grep for `captured=<id[:8]>` to find
    the audio+timestamps row on /captures.

    `endpoint` is the route that produced the block — `/v1/audio/transcriptions`
    for the batch (file-upload) route, `…/stream` for live dictation — so the two
    sources are distinguishable in the log. `audio_source` (when given) describes
    the input transport/codec + rate, shown as an `input` line in the Audio
    section (the model itself always decodes at 16 kHz mono)."""
    title_rule = "═" * _LOG_WIDTH
    rule = "─" * _LOG_WIDTH

    status = "[!] empty output" if len(seg_diag) == 0 else "✓ ok"
    if request_id:
        status = f"req={request_id[:8]}  {status}"
    if captured_id:
        status = f"captured={captured_id[:8]}  {status}"
    title = "  " + endpoint
    pad = max(1, _LOG_WIDTH - len(title) - len(status))
    title_line = f"{title}{' ' * pad}{status}"

    lines: list[str] = ["", title_rule, title_line, title_rule]

    lines.append(f"  file   {file_label}")
    model_line = f"  model  {model_name}"
    compute, device = _model_compute_device(model_name)
    extras = []
    if compute:
        extras.append(f"compute={compute}")
    if device:
        extras.append(f"device={device}")
    if extras:
        model_line += "   " + "  ".join(extras)
    lines.append(model_line)
    # Translation-stage receipt — only when the stage actually ran.
    if translate_to:
        trans_line = f"  trans  → {', '.join(translate_to)}"
        if translation_model:
            trans_line += f"   model={translation_model}"
        lines.append(trans_line)

    lines.append(_section_rule("Audio"))
    if audio_source:
        lines.append(f"    {'input':<{_NAME_COL - 4}}{audio_source}")
    lang = getattr(info, "language", "?")
    lang_prob = getattr(info, "language_probability", None)
    lang_str = f"{lang}  (prob={lang_prob:.2f})" if lang_prob is not None else str(lang)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    lines.append(f"    {'language':<{_NAME_COL - 4}}{lang_str}")
    lines.append(f"    {'duration':<{_NAME_COL - 4}}{duration:.2f}s")
    dav = getattr(info, "duration_after_vad", None)
    if dav is not None:
        retained = (float(dav) / duration * 100) if duration > 0 else 0.0
        lines.append(
            f"    {'duration_after_vad':<{_NAME_COL - 4}}"
            f"{float(dav):.2f}s   ({retained:.0f} % retained)"
        )

    lines.append(_section_rule("Decode params  (* = non-default)"))
    lines.extend(_format_decode_params(kwargs))

    # Post-decode guards — applied AFTER model.transcribe (word-rate drop,
    # tail trim, streaming final-drop thresholds), so they are not kwargs and
    # would otherwise be invisible in the block. Rows marked ✗ in the segments
    # table were removed by one of these.
    if guards:
        lines.append(_section_rule("Post-decode guards  (* = non-default)"))
        for gk, gv in guards.items():
            lines.append(_param_row("    ", gk, gv))

    lines.extend(_format_segments_section(seg_diag, info, kwargs))

    # Identity section — always shown when the caller is known, so the resolved
    # user/key (and any applied per-identity overrides, or their ABSENCE) is
    # visible at a glance. This was previously suppressed for no-config
    # requests, which made per-identity mismatches invisible in the log.
    def _short_id(v):
        v = v or ""
        return v if v.startswith("(") else (v[:8] if v else "—")

    _ident_detail = ident is not None and (getattr(ident, "layers", None)
                                           or getattr(ident, "locked", None)
                                           or overrides_ignored)
    if user_id or key_id or _ident_detail:
        lines.append(_section_rule("Identity"))
        who = f"{username} ({_short_id(user_id)})" if username else _short_id(user_id)
        lines.append(f"    {'user':<{_NAME_COL - 4}}{who}")
        if key_id:
            which = (f"{key_label} ({_short_id(key_id)})"
                     if key_label else _short_id(key_id))
            lines.append(f"    {'key':<{_NAME_COL - 4}}{which}")
        if ident is not None and ident.profiles_applied:
            lines.append(f"    {'profiles':<{_NAME_COL - 4}}{' → '.join(ident.profiles_applied)}")
        if ident is not None and ident.layers:
            lines.append(f"    {'layers':<{_NAME_COL - 4}}{', '.join(ident.layers)}")
        elif user_id or key_id:
            # No identity layer resolved — call it out explicitly so a missing
            # binding (the classic "my override didn't apply") is obvious.
            lines.append(f"    {'overrides':<{_NAME_COL - 4}}(none — inherits per-model / global)")
        if ident is not None and ident.locked:
            lines.append(f"    {'locked':<{_NAME_COL - 4}}{', '.join(sorted(ident.locked))}")
        if overrides_ignored:
            lines.append(f"    {'overrides_ignored':<{_NAME_COL - 4}}{', '.join(overrides_ignored)}")

    lines.append(rule)
    lines.append(f"  RAW WHISPER  {raw!r}")
    lines.append(rule)
    if steps:
        # Count only steps that actually rewrote the text (before != after) as
        # "changed"; the rest (EXCLUDED for this model, globally disabled, no-op)
        # are "unchanged". This matches the /quick-config and /reports viewers,
        # which render the same trace and split it the same way — the header used
        # to print len(steps) and label them all "changed", overcounting skips.
        changed = sum(1 for _, before, after in steps if before != after)
        unchanged = len(steps) - changed
        plural = "s" if changed != 1 else ""
        header = f"  PIPELINE  ({changed} step{plural} changed text"
        if unchanged:
            header += f", {unchanged} unchanged"
        lines.append(header + ")")
        for name, before, after in steps:
            lines.append(f"    ▸ {name}")
            lines.append(f"        {before!r}")
            lines.append(f"     →  {after!r}")
        lines.append(rule)
    lines.append(f"  FINAL        {final!r}")
    lines.append(title_rule)

    return "\n".join(lines)


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
import asyncio
import functools
from collections import OrderedDict

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
# changes.
_suppress_chars_cache: "dict[tuple[str, str], tuple[int, ...]]" = {}


def _resolve_suppress_chars(model_id: str,
                            model: "WhisperModel",
                            chars: "str | None") -> "tuple[int, ...]":
    """Return the sorted tuple of vocab IDs to suppress for the given chars.
    Each char is encoded both bare and with a leading space — Whisper's BPE
    often tokenizes a punct char differently in those positions (mirrors
    faster-whisper's own non_speech_tokens approach). Multi-piece results
    are skipped with a warning (suppressing only the first piece would
    block every word that starts with that piece)."""
    if not chars:
        return ()
    key = (model_id, chars)
    cached = _suppress_chars_cache.get(key)
    if cached is not None:
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
    if out:
        logger.info("SUPPRESS_CHARS resolved for %s (%r): %r",
                    model_id, chars, out)
    return out


# Per-request decode-param overrides (the client's "decode overrides"). Optional;
# absent leaves behavior identical to before (config-only). Every value is clamped
# to the SAME bounds the admin config enforces (config_store.py), so an untrusted
# client cannot request unbounded compute on the shared server. Applied AFTER config
# resolution, so the order is: request > per-model override > global default.
_DECODE_INT_BOUNDS = {
    "beam_size": (1, 20),
    "best_of": (1, 20),
    "no_repeat_ngram_size": (0, 10),
}
_DECODE_FLOAT_BOUNDS = {
    "temperature": (0.0, 1.0),
    "no_speech_threshold": (0.0, 1.0),
    "log_prob_threshold": (-10.0, 0.0),
    "compression_ratio_threshold": (0.0, 10.0),
    "patience": (0.5, 5.0),
    "length_penalty": (0.1, 5.0),
    "repetition_penalty": (0.5, 5.0),
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
    if "condition_on_previous_text" in overrides:
        kwargs["condition_on_previous_text"] = bool(overrides["condition_on_previous_text"])
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
            # Nothing left after bounding → leave the config value in place
            # rather than forward an override the caller didn't really make.
            if ids:
                kwargs["suppress_tokens"] = ids
    # VAD: toggle + sub-params (sub-params rebuilt from config defaults when on).
    if "vad_filter" in overrides:
        vf = bool(overrides["vad_filter"])
        kwargs["vad_filter"] = vf
        if not vf:
            kwargs["vad_parameters"] = None
    if kwargs.get("vad_filter"):
        vp = dict(kwargs.get("vad_parameters") or dict(
            min_silence_duration_ms=cfg_for(resolved_model, "VAD_MIN_SILENCE_MS", ident),
            speech_pad_ms=cfg_for(resolved_model, "VAD_SPEECH_PAD_MS", ident),
            threshold=cfg_for(resolved_model, "VAD_THRESHOLD", ident),
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
        return cfg_for(resolved_model, field, ident)

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
        try:
            ladder = tuple(float(t.strip()) for t in _temp_str.split(",") if t.strip())
            if ladder:
                transcribe_kwargs["temperature"] = ladder
        except ValueError:
            pass
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
    if _hallu_silence is not None:
        transcribe_kwargs["hallucination_silence_threshold"] = _hallu_silence
    _suppress_blank = cf("SUPPRESS_BLANK")
    if _suppress_blank is False:
        transcribe_kwargs["suppress_blank"] = False
    _suppress_tokens_str = cf("SUPPRESS_TOKENS")
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
    # SUPPRESS_CHARS — chars resolved to vocab IDs via the loaded
    # model's tokenizer, then merged into the effective suppress_tokens
    # list. Genuinely additive: existing IDs from SUPPRESS_TOKENS are
    # preserved.
    _suppress_chars = cf("SUPPRESS_CHARS")
    if _suppress_chars:
        extra_ids = _resolve_suppress_chars(resolved_model, model, _suppress_chars)
        if extra_ids:
            existing = transcribe_kwargs.get("suppress_tokens")
            if existing is None:
                merged_ids = sorted({-1, *extra_ids})
            else:
                merged_ids = sorted(set(existing) | set(extra_ids))
            transcribe_kwargs["suppress_tokens"] = merged_ids
    _prepend_p = cf("PREPEND_PUNCTUATIONS")
    if _prepend_p:
        transcribe_kwargs["prepend_punctuations"] = _prepend_p
    _append_p = cf("APPEND_PUNCTUATIONS")
    if _append_p:
        transcribe_kwargs["append_punctuations"] = _append_p
    # Per-request overrides win (clamped), EXCEPT fields locked by an identity
    # layer (skipped). No-op when None/empty.
    _apply_decode_overrides(transcribe_kwargs, resolved_model, overrides, ident=ident)
    return transcribe_kwargs


# Below this many words a segment's rate is statistically meaningless (a single
# short interjection in a tight VAD chunk can legitimately look "fast").
_WORD_RATE_MIN_WORDS = 3


def segment_exceeds_word_rate(seg, max_wps: float) -> bool:
    """Post-decode anti-hallucination guard (SEGMENT_MAX_WORDS_PER_SEC), shared
    by the batch route and the streaming FINAL decode.

    When trailing non-speech audio survives the VAD into a decode, Whisper
    re-decodes the sub-second leftover after the last aligned word as its own
    zero-padded window and confidently replays its text context — segments of
    20+ words crammed into half a second. Those pass every confidence gate
    (high avg_logprob, temperature 0.0, no_speech_prob possibly below the
    threshold); the impossible word density is their one reliable signature.
    Real speech peaks around ~6 words/s, so the default limit of 10 has wide
    margin on both sides."""
    if not max_wps or max_wps <= 0:
        return False
    words = getattr(seg, "words", None)
    n = len(words) if words else len((getattr(seg, "text", "") or "").split())
    if n < _WORD_RATE_MIN_WORDS:
        return False
    duration = float(getattr(seg, "end", 0.0) or 0.0) - float(getattr(seg, "start", 0.0) or 0.0)
    if duration <= 0:
        return True
    return (n / duration) > float(max_wps)


def _drop_suppress_chars_cache(model_id: str) -> None:
    """Drop all cache entries for a given model. Called from unload paths."""
    for k in list(_suppress_chars_cache):
        if k[0] == model_id:
            _suppress_chars_cache.pop(k, None)


def _drop_loaded_model(name: str, *, force: bool = False) -> bool:
    """Single unload entry point: pop the cached WhisperModel, drop its
    suppress-chars entries, and unregister from the system_stats registry.
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
    system_stats.unregister_loaded_model(name)
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
    system_stats.touch_loaded_model(name)


def _resolve_model_name(requested: str) -> str:
    """Map OpenAI-compatible 'whisper-1' (or empty) to our configured default;
    pass anything else through as a faster-whisper / HF model identifier."""
    if not requested or requested == "whisper-1":
        return cfg.DEFAULT_MODEL
    return requested


# =============================================================================
# Per-model config resolution (per-model override > global default)
# =============================================================================
# cfg_for(model_id, field) is the canonical reader for any G/PM-scoped setting.
# It walks: cfg.MODEL_OVERRIDES[model_id][field] (if set and not None) → cfg.X
# (global default). Pure-G fields (DEFAULT_MODEL, ALLOWED_MODELS, server, log)
# are read with plain cfg.X — they have no per-model meaning.
#
# Precedence (highest to lowest):
#   request-arg  >  per-model override  >  global default  >  faster-whisper
# The first three are this function's business; the last is whatever
# faster-whisper itself defaults to when we omit a kwarg.

def cfg_for(model_id: "str | None", field: str, ident=None):
    """Resolve a G/PM config field for the given model_id (and optional caller
    identity).

    Precedence: per-identity override (``ident``) > per-model override >
    global cfg.X. ``ident`` is an effective_config.Resolved whose ``values``
    already merged the key/user/profile layers; passing ``ident=None`` (the
    default everywhere except the request paths) is byte-identical to the
    pre-feature behaviour. Pass model_id=None to skip the per-model layer.
    """
    if ident is not None and field in ident.values:
        return ident.values[field]
    overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
    if model_id and isinstance(overrides, dict):
        m_over = overrides.get(model_id)
        if isinstance(m_over, dict):
            v = m_over.get(field)
            if v is not None:
                return v
    return getattr(cfg, field)

def _resolve_request_knob(resolved_model, ident, ignored: "list[str]",
                          cfg_name: str, client_name: str, req_val,
                          default=""):
    """The locked-wins / request-wins / config-inherits ladder every
    per-request string knob shares (the speaker counts use the numeric tuple
    loop next to their form parsing). Locked: the resolved server value wins
    and a differing request lands in ``ignored`` under its client name."""
    if cfg_name in ident.locked:
        val = cfg_for(resolved_model, cfg_name, ident) or default
        if req_val is not None and req_val != val:
            ignored.append(client_name)
        return val
    if req_val is not None:
        return req_val
    return cfg_for(resolved_model, cfg_name, ident) or default



def build_ident(user: "dict | None", model_id: "str | None",
                request_overrides: "dict | None" = None,
                request_profile: "str | None" = None):
    """Resolve the per-identity effective config ONCE for a request / streaming
    handshake / capture row, to thread through cfg_for / assemble_transcribe_
    kwargs / _postprocess_text. Open mode and callers with no per-identity
    config yield a Resolved with no identity layers (per-model rules still
    folded) — equivalent to threading ident=None."""
    import effective_config
    user = user or {}
    return effective_config.resolve(
        model_id,
        user_id=user.get("user_id"),
        key_id=user.get("key_id"),
        request_overrides=request_overrides or {},
        request_profile=request_profile,
    )


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
    one `/`, so a single replace is enough."""
    sanitised = model_id.replace("/", "__").replace(os.sep, "__")
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
        from filelock import FileLock, Timeout as FileLockTimeout
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        lock_path = output_dir + ".lock"
        try:
            with FileLock(lock_path, timeout=600):
                # Re-check after winning the cross-process race.
                if os.path.isfile(os.path.join(output_dir, "model.bin")):
                    return output_dir
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, _convert_blocking, name, output_dir, quantization,
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
# required). streaming_routes.py acquires the same object via this getter.
_inference_semaphore: "asyncio.Semaphore | None" = None


def get_inference_semaphore() -> "asyncio.Semaphore":
    global _inference_semaphore
    if _inference_semaphore is None:
        n = max(1, int(getattr(cfg, "INFERENCE_CONCURRENCY", 2)))
        _inference_semaphore = asyncio.Semaphore(n)
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
# config_store._MODEL_ID_PATTERN validates configured model ids against. Used
# below to bound what an EMPTY ALLOWED_MODELS accepts from a request.
# \Z, not $: `$` also matches just BEFORE a trailing newline, so "some-repo\n"
# passed the gate. \Z anchors at the true end of the string and is a pure
# tightening here (no legitimate model id ends in a newline); fixing it in the
# pattern also covers every other caller of this regex, which stripping inside
# _resolve_model_name would not.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?\Z")


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
    # config_store.RESTART_REQUIRED_FIELDS nor LOAD_TIME_FIELDS, so narrowing it
    # is reported to the admin as hot-applied and evicts nothing. Gate BEFORE the
    # cache fast path, or a model that is still resident keeps being served to
    # clients after the admin withdrew it (MODEL_IDLE_TIMEOUT_S defaults to 0,
    # so the entry only leaves on LRU pressure or restart).
    if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' is not in the allowed list. "
                   f"Allowed: {sorted(cfg.ALLOWED_MODELS)}",
        )

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
        system_stats.touch_loaded_model(name)
        if lease:
            _model_leases[name] = _model_leases.get(name, 0) + 1
        return cached

    # No allowlist configured: the name arrives verbatim from the request, and
    # below it reaches os.path.isdir() / the HF hub / the CT2 converter. Accept
    # DEFAULT_MODEL (which may legitimately be a local directory) and otherwise
    # only well-formed model ids — no filesystem paths, no "..", no URLs.
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

    # Auto-convert HF transformers Whisper repos to CT2 format if enabled.
    # Runs OUTSIDE _model_load_lock so loads of OTHER cached models stay
    # snappy during the (rare, slow) conversion step. Returns `name`
    # unchanged if conversion is off, repo is already CT2, or it's a
    # local path with model.bin.
    load_path = await _ensure_ct2_model(name)

    loop = asyncio.get_running_loop()
    load_t0 = time.perf_counter()
    # Per-model override > global default. Each loaded model can pin its
    # own device/compute_type/etc. independently.
    primary_device = cfg_for(name, "MODEL_DEVICE")
    primary_compute = cfg_for(name, "MODEL_COMPUTE_TYPE")
    fallback_device = cfg_for(name, "MODEL_DEVICE_FALLBACK")
    fallback_compute = cfg_for(name, "MODEL_COMPUTE_TYPE_FALLBACK")
    # Load-time hardware kwargs (also per-model overrideable).
    load_kwargs = {
        "device": primary_device,
        "compute_type": primary_compute,
        "device_index": cfg_for(name, "DEVICE_INDEX"),
        "cpu_threads": cfg_for(name, "CPU_THREADS"),
        "num_workers": cfg_for(name, "NUM_WORKERS"),
    }
    # Optional load-time fields — only forwarded if non-default to keep
    # WhisperModel(...) clean for the common path.
    _download_root = cfg_for(name, "DOWNLOAD_ROOT")
    if _download_root:
        load_kwargs["download_root"] = _download_root
    if cfg_for(name, "LOCAL_FILES_ONLY"):
        load_kwargs["local_files_only"] = True
    _auth_token = cfg_for(name, "HF_TOKEN")
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
            import download_progress
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

    async with _model_load_lock:
        # Re-check under the lock — another request may have loaded it.
        cached = _loaded_models.get(name)
        if cached is not None:
            _loaded_models.move_to_end(name)
            system_stats.touch_loaded_model(name)
            if lease:
                _model_leases[name] = _model_leases.get(name, 0) + 1
            return cached

        # Evict the least-recently-used UNLEASED model(s) until we have room.
        while len(_loaded_models) >= cfg.MAX_LOADED_MODELS:
            evicted_name = next(
                (n for n in _loaded_models if not _model_leases.get(n, 0)), None)
            if evicted_name is None:
                # Every cached model is mid-request — overflow the cap rather
                # than free a translator under a running decode; the idle
                # evictor trims the excess once the requests release.
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
            loaded_device = fallback_device
            loaded_compute = fallback_compute
            logger.info("Model loaded on %s: %s", fallback_device, name)

        load_secs = time.perf_counter() - load_t0
        metrics.record_model_load(name, load_secs)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram_delta = (vram_after - vram_before
                      if vram_before is not None and vram_after is not None
                      else None)
        # Negative deltas can happen if another process freed VRAM during load
        # (or the CT2 allocator did). Clamp to 0 rather than store nonsense.
        if vram_delta is not None and vram_delta < 0:
            vram_delta = 0
        system_stats.register_loaded_model(
            name,
            vram_bytes=vram_delta,
            device=loaded_device,
            compute_type=loaded_compute,
        )

        _loaded_models[name] = new_model
        if lease:
            _model_leases[name] = _model_leases.get(name, 0) + 1
        return new_model


async def drain_then_evict(model_id: "str | None" = None) -> list[str]:
    """Drain-then-evict pattern. Drops the cached entry for `model_id` (or all
    entries when None) so the next request for that id reloads the model with
    current cfg / per-model settings.

    "Drain" comes for free from Python reference counting: in-flight transcribe
    requests already hold their own `model` reference (captured via `_get_or_
    load_model` before the executor call), so they continue running on the
    old WhisperModel instance until they finish. Only NEW requests for the
    evicted id pay the reload cost. Returns the list of evicted ids.

    Called from admin_routes.post_state when a load-time field (MODEL_DEVICE,
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
    try:
        while True:
            await asyncio.sleep(30)
            timeout = getattr(cfg, "MODEL_IDLE_TIMEOUT_S", 0) or 0
            if timeout <= 0 or not _loaded_models:
                continue
            now = time.monotonic()
            stale: list[str] = []
            for name, info in list(system_stats._loaded_models.items()):
                if name not in _loaded_models:
                    continue
                last = info.get("last_used_monotonic", now)
                if now - last >= timeout:
                    stale.append(name)
            if not stale:
                continue
            async with _model_load_lock:
                for name in stale:
                    if name not in _loaded_models:
                        continue   # raced with another path
                    logger.info("[idle-evict] unloading %s after %ds idle",
                                name, timeout)
                    # Not forced: a refusal costs nothing, the next 30 s tick
                    # picks the model up again once the request releases.
                    _drop_loaded_model(name)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    except asyncio.CancelledError:
        return


async def _reports_retention_loop() -> None:
    """Hourly retention sweep for the reports store. Lazy-imports
    cfg.REPORTS_RETENTION_DAYS each tick so admin /settings edits take
    effect on the next cycle without a service restart. Cancellation
    on shutdown is the normal exit path."""
    import reports_store
    while True:
        try:
            await asyncio.sleep(3600)
            reports_store.sweep_retention()
        except asyncio.CancelledError:
            raise
        except Exception as _re:
            logger.error("[reports] retention loop error: %s", _re)


async def _captures_retention_loop() -> None:
    """Hourly retention sweep for the captures store. Same shape as the
    reports loop; lazy reads cfg.CAPTURES_RETENTION_DAYS each tick."""
    import captures_store
    while True:
        try:
            await asyncio.sleep(3600)
            captures_store.sweep_retention()
        except asyncio.CancelledError:
            raise
        except Exception as _ce:
            logger.error("[captures] retention loop error: %s", _ce)


async def _sessions_purge_loop() -> None:
    """Hourly reap of revoked/expired session rows.

    Nothing else purges them at runtime: the lazy eviction in lookup_session
    only fires for a token that is presented again, so with the 30-day sliding
    TTL a login loop grew both the sessions table and the in-memory index
    without bound. /auth/login takes any valid key and has no rate limit.
    Same shape as the retention loops above."""
    import sessions_store
    while True:
        try:
            await asyncio.sleep(3600)
            # Off the loop: purge_expired rebuilds the whole session index,
            # which is O(live sessions) and measured ~33 ms at 20 000 rows.
            await asyncio.to_thread(sessions_store.purge_expired)
        except asyncio.CancelledError:
            raise
        except Exception as _se:
            logger.error("[sessions] purge loop error: %s", _se)


# Strength floor for the operator-supplied bootstrap admin key. Well below
# generate_raw_key()'s output (wk_ + 43 chars) so nothing machine-generated
# trips it; the distinct-character test rejects "aaaaaaaa..."-shaped values
# that clear the length bar without carrying the entropy it implies.
_BOOTSTRAP_KEY_MIN_LEN = 20
_BOOTSTRAP_KEY_MIN_DISTINCT = 8


def _bootstrap_key_is_strong(raw_key: str) -> bool:
    key = (raw_key or "").strip()
    return (
        len(key) >= _BOOTSTRAP_KEY_MIN_LEN
        and len(set(key)) >= _BOOTSTRAP_KEY_MIN_DISTINCT
    )


def _bootstrap_admin_from_env(raw_key: str) -> None:
    """If WHISPER_BOOTSTRAP_ADMIN_KEY is set, ensure a `bootstrap-admin`
    user holds that exact key. Idempotent — if the key hash is already in
    the DB we no-op. The raw key never gets persisted in plaintext;
    only the SHA-256 hash hits disk.

    A minimum strength is enforced before the key is ever created. Keys are
    stored as an unsalted single-round SHA-256, which api_keys_store justifies
    with "high-entropy random keys (256-bit) make slow password hashes
    pointless" — true for generate_raw_key()'s output, but this value is
    human-chosen, and /auth/login has no rate limit or lockout, so a short one
    is brute-forceable online straight to full admin.

    Rejection is safe for an existing install: the hash check below no-ops when
    the key is already in the DB, so a key created before this floor existed
    keeps working. Only first-time creation of a weak value is refused, and the
    server then stays in its usual no-admin-key state rather than failing to
    boot.
    """
    import api_keys_store
    if not _bootstrap_key_is_strong(raw_key):
        logger.error(
            "[auth] WHISPER_BOOTSTRAP_ADMIN_KEY is too weak — refusing to "
            "create an admin key from it. It must be at least %d characters "
            "with at least %d distinct ones. Generate one with: "
            "python -c \"import secrets; print('wk_' + secrets.token_urlsafe(32))\"",
            _BOOTSTRAP_KEY_MIN_LEN, _BOOTSTRAP_KEY_MIN_DISTINCT,
        )
        return
    h = api_keys_store.hash_key(raw_key)
    # If this hash already maps to an active key, nothing to do.
    if api_keys_store._KEY_INDEX.get(h) is not None:
        return
    # _KEY_INDEX is built from live rows only (revoked_ts IS NULL), so a hash
    # belonging to a REVOKED key is invisible above and used to fall through to
    # the INSERT, hit the UNIQUE on key_hash, and get swallowed silently — the
    # server then booted into OPEN mode with the operator believing the env key
    # had locked it down, which is exactly the failure the lifespan below calls
    # fatal. Fail loudly instead. Un-revoking here would resurrect a key the
    # operator deliberately killed, so that is not the answer either.
    _revoked = api_keys_store._require_conn().execute(
        "SELECT id FROM api_keys WHERE key_hash = ? AND revoked_ts IS NOT NULL",
        (h,),
    ).fetchone()
    if _revoked is not None:
        raise RuntimeError(
            "WHISPER_BOOTSTRAP_ADMIN_KEY matches an API key that has been "
            "REVOKED. Refusing to start: silently ignoring it would leave the "
            "server with no admin key while you believe it is locked down. "
            "Set the variable to a different key, or clear it and use the "
            "existing admin credentials."
        )
    # Reuse or create the bootstrap-admin user.
    existing = [
        u for u in api_keys_store.list_users()
        if u["username"] == "bootstrap-admin"
    ]
    if existing:
        uid = existing[0]["id"]
        if not existing[0]["is_admin"]:
            logger.warning(
                "[auth] bootstrap-admin user exists but is_admin=False; "
                "leaving as-is. Recreate manually to escalate."
            )
            return
    else:
        uid = api_keys_store.create_user("bootstrap-admin", is_admin=True)
    # Insert the raw key (bypass generate path so we honour the env value).
    import sqlite3 as _sql
    kp, k4 = api_keys_store._split_display_parts(raw_key)
    try:
        with api_keys_store._lock:
            api_keys_store._require_conn().execute(
                "INSERT INTO api_keys"
                " (id, user_id, key_hash, key_prefix, key_last4, label,"
                "  created_ts, revoked_ts, last_used_ts)"
                " VALUES (?,?,?,?,?,?,?,NULL,NULL)",
                (
                    uuid.uuid4().hex, uid, h, kp, k4,
                    "bootstrap (env)", time.time(),
                ),
            )
            api_keys_store._rebuild_index_locked()
        # Deliberately NOT logging kp here: unlike a generated key, this value
        # is human-chosen and only has to clear _BOOTSTRAP_KEY_MIN_LEN, so the
        # 8-char display prefix is a real fraction of the secret — and the
        # server log is readable by every non-admin, who get the /logs page by
        # default. The hash prefix identifies the key without revealing it.
        logger.info(
            "[auth] bootstrap admin key registered from "
            "WHISPER_BOOTSTRAP_ADMIN_KEY (user=bootstrap-admin, sha256=%s)",
            h[:8],
        )
    except _sql.IntegrityError:
        # The live-hash check and the revoked-hash check above both passed, so
        # a UNIQUE violation here means the row appeared underneath us. Never
        # silent: the same open-mode-without-noticing outcome applies.
        raise RuntimeError(
            "WHISPER_BOOTSTRAP_ADMIN_KEY could not be registered (the key hash "
            "already exists). Refusing to start rather than leaving the server "
            "without the admin key you configured."
        )


async def _preload_extras() -> None:
    """Best-effort startup preloads for the optional stages (translation
    GGUFs, the diarization pipeline, the BGM separator). Every failure logs
    and continues — the model then loads on first use; the server always
    starts. Called from lifespan after the whisper preload loop."""
    if getattr(cfg, "TRANSLATION_ENABLED", False):
        _allowed = getattr(cfg, "TRANSLATION_ALLOWED_MODELS", set()) or set()
        _default = (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
        _preload = list(dict.fromkeys(
            getattr(cfg, "TRANSLATION_PRELOAD_MODELS", []) or []))
        _cap = max(1, int(getattr(cfg, "TRANSLATION_MAX_LOADED_MODELS", 1) or 1))
        if len(_preload) > _cap:
            # Mirror the whisper preload's cap warning — loading past the LRU
            # cap would silently close each earlier preload as the next loads.
            logger.warning(
                "TRANSLATION_PRELOAD_MODELS lists %d models but "
                "TRANSLATION_MAX_LOADED_MODELS is %d — preloading only the "
                "first %d (dropped: %s)",
                len(_preload), _cap, _cap, ", ".join(_preload[_cap:]))
            _preload = _preload[:_cap]
        for ref in _preload:
            # Same allowlist semantics as the request path: a non-empty
            # allowlist admits its members plus the configured default.
            if _allowed and ref not in _allowed and ref != _default:
                logger.error(
                    "Cannot preload translation model '%s' - it is not in "
                    "TRANSLATION_ALLOWED_MODELS.", ref)
                continue
            try:
                logger.info("Preloading translation model: %s", ref)
                await _tr._get_model(ref)
            except Exception as e:  # noqa: BLE001 — best-effort
                logger.error("Failed to preload translation model '%s': %s",
                             ref, e)
    if getattr(cfg, "DIARIZATION_PRELOAD", False) and \
            getattr(cfg, "DIARIZATION_ENABLED", False):
        import diarization as _diar
        try:
            logger.info("Preloading the diarization pipeline")
            await _diar._get_pipeline()
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.error("Failed to preload the diarization pipeline: %s", e)
    if getattr(cfg, "BGM_SEPARATION_PRELOAD", False) and \
            getattr(cfg, "BGM_SEPARATION_ENABLED", False):
        import bgm_separation as _bgm
        try:
            logger.info("Preloading the BGM separation model")
            await _bgm._get_separator()
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.error("Failed to preload the separation model: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s %s starting (boot %s)", SERVER_NAME, APP_VERSION, BOOT_ID[:8])

    # If PRELOAD_MODELS is empty, fall back to preloading just DEFAULT_MODEL
    # so a fresh start always has at least one ready-to-serve model.
    to_preload = list(dict.fromkeys(cfg.PRELOAD_MODELS or [cfg.DEFAULT_MODEL]))

    if len(to_preload) > cfg.MAX_LOADED_MODELS:
        logger.warning(
            "PRELOAD_MODELS has %d entries but MAX_LOADED_MODELS=%d; "
            "LRU eviction will discard the earliest preloaded models. "
            "Bump MAX_LOADED_MODELS to at least %d to keep them all hot.",
            len(to_preload), cfg.MAX_LOADED_MODELS, len(to_preload),
        )

    for name in to_preload:
        if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
            logger.error(
                "Cannot preload '%s' - it is not in ALLOWED_MODELS. "
                "Add it to the allowlist or remove from PRELOAD_MODELS.", name,
            )
            continue
        try:
            logger.info("Preloading model: %s", name)
            await _get_or_load_model(name)
        except Exception as e:
            logger.error("Failed to preload model '%s': %s", name, e)

    evictor_task = asyncio.create_task(_idle_evictor())
    # The diarization pipeline gets its own idle unloader (module-local
    # singleton, DIARIZATION_IDLE_TIMEOUT_S read live). Import is cheap and
    # dependency-free — pyannote itself loads lazily on first use.
    import diarization as _diarization
    diarization_evictor_task = asyncio.create_task(
        _diarization.idle_evictor_loop())
    import bgm_separation as _bgm_separation
    bgm_evictor_task = asyncio.create_task(
        _bgm_separation.idle_evictor_loop())
    # The translation LRU gets its own idle unloader too (module-level
    # import at the top of the file; TRANSLATION_IDLE_TIMEOUT_S read live).
    translation_evictor_task = asyncio.create_task(
        _tr.idle_evictor_loop())

    # Best-effort preloads for the optional stages (translation GGUFs,
    # diarization pipeline, BGM separator).
    await _preload_extras()

    # Transcribe-from-URL retention: wipe the dir (ids die with the process,
    # so every surviving file is an orphan) and start the TTL/LRU janitor.
    url_media_janitor_task = None
    if getattr(cfg, "URL_DOWNLOAD_ENABLED", False):
        import url_media_store as _url_media_store
        _url_media_store.startup_reset()
        url_media_janitor_task = asyncio.create_task(
            _url_media_store.janitor_loop())

    # Open the API-keys SQLite store and start the open-mode warning loop.
    # In OPEN mode (no admin key exists yet) the loop nags every 60 s; this
    # is the operator's prompt to bootstrap an admin via /settings/api-keys.
    # Optional WHISPER_BOOTSTRAP_ADMIN_KEY env var creates the very first
    # admin in one shot without any UI.
    #
    # FATAL, unlike every other store below: without it auth can resolve
    # nobody, so the service answers 401 to everything while still looking
    # healthy. Raising here aborts startup — uvicorn logs the failure and
    # exits non-zero, so the container/unit restarts instead of running as a
    # black hole. A failed bootstrap-admin ingest is fatal for the mirror
    # image of that reason: it would leave the server in OPEN mode with the
    # operator believing the env key locked it down.
    open_mode_task = None
    try:
        import api_keys_store
        import auth as _auth
        api_keys_store.init_db(cfg.API_KEYS_DB)
        bootstrap_key = getattr(cfg, "BOOTSTRAP_ADMIN_KEY", None)
        if bootstrap_key:
            # Only inserts if hash isn't already in api_keys. Idempotent.
            _bootstrap_admin_from_env(bootstrap_key)
        logger.info(
            "API keys store initialized at %s (locked_down=%s)",
            cfg.API_KEYS_DB, api_keys_store.is_locked_down(),
        )
        open_mode_task = asyncio.create_task(
            _auth.open_mode_warning_loop()
        )
    except Exception as _ae:
        logger.critical(
            "Failed to initialize the API keys store at %s: %s — refusing to "
            "start (check WHISPER_API_KEYS_DB / WHISPER_DB_DIR / "
            "WHISPER_DATA_DIR and the directory's permissions)",
            cfg.API_KEYS_DB, _ae,
        )
        raise RuntimeError(
            f"API keys store unavailable at {cfg.API_KEYS_DB}: {_ae}"
        ) from _ae

    # Open the browser-session store (HttpOnly cookie auth for the WebUI).
    # Non-fatal: if this fails, cookie login is unavailable but bearer auth
    # (API clients) and open mode keep working.
    sessions_purge_task = None
    try:
        import sessions_store
        sessions_store.init_db(cfg.SESSIONS_DB)
        logger.info("Session store initialized at %s", cfg.SESSIONS_DB)
        sessions_purge_task = asyncio.create_task(_sessions_purge_loop())
    except Exception as _se:
        logger.error("Failed to initialize session store: %s", _se)

    # Open the reports SQLite store (durable, plaintext dictation content
    # on disk) and run an immediate retention sweep before serving traffic.
    # Failure here is non-fatal: the rest of the app must keep working even if
    # the reports surface is broken, but the /reports page will error.
    reports_sweep_task = None
    try:
        import reports_store
        reports_store.init_db(cfg.REPORTS_DB)
        reports_store.sweep_retention()
        logger.info("Reports store initialized at %s", cfg.REPORTS_DB)
        reports_sweep_task = asyncio.create_task(
            _reports_retention_loop()
        )
    except Exception as _re:
        logger.error("Failed to initialize reports store: %s", _re)

    # Open the durable recent-transcriptions store. Replaces the legacy
    # in-memory ring buffers (quick_config_state.recent_traces +
    # metrics.recent_tx) so the /quick-config trace panel + /stats
    # dashboard widget survive service restart and scale beyond 20 rows.
    try:
        import transcriptions_store
        transcriptions_store.init_db(cfg.RECENT_TRANSCRIPTIONS_DB)
        logger.info(
            "Recent-transcriptions store initialized at %s",
            cfg.RECENT_TRANSCRIPTIONS_DB,
        )
    except Exception as _te:
        logger.error("Failed to initialize recent-transcriptions store: %s", _te)

    # Open the durable usage-rollup store. Backs the per-key/per-user usage
    # numbers on /api-keys and the usage-over-time section on /stats. Non-fatal.
    try:
        import usage_store
        usage_store.init_db(cfg.USAGE_DB)
        logger.info("Usage rollup store initialized at %s", cfg.USAGE_DB)
    except Exception as _ue:
        logger.error("Failed to initialize usage store: %s", _ue)

    # Open the desktop-client settings-sync store (one opaque blob per
    # account, served at /v1/client-settings). Non-fatal: sync degrades to
    # 503s but transcription keeps working. No retention loop — bounded at
    # one row per account.
    try:
        import client_settings_store
        client_settings_store.init_db(cfg.CLIENT_SETTINGS_DB)
        logger.info(
            "Client-settings store initialized at %s", cfg.CLIENT_SETTINGS_DB
        )
    except Exception as _cse:
        logger.error(
            "Failed to initialize client-settings store at %s: %s — "
            "/v1/client-settings will answer 503 until this is fixed "
            "(WHISPER_CLIENT_SETTINGS_DB / WHISPER_DB_DIR / WHISPER_DATA_DIR)",
            cfg.CLIENT_SETTINGS_DB, _cse,
        )

    # Open the captures store. Audio + word-timestamps for Whisper
    # fine-tuning, gated by CAPTURE_RECORDINGS_ENABLED. Reconcile drift
    # before serving (row says audio exists / disk says it doesn't, or
    # vice versa).
    captures_sweep_task = None
    try:
        import captures_store
        captures_store.init(cfg.CAPTURES_DB, cfg.CAPTURES_DIR)
        # capture_samples_store reuses the captures DB connection — single
        # SQLite file holds both tables. Init it before the first
        # sweep_retention(): the sweep's sample-expiry pass needs it.
        import capture_samples_store
        capture_samples_store.init(captures_store._require_conn(), cfg.CAPTURES_DIR)
        captures_store.reconcile_on_startup()
        capture_samples_store.reconcile_on_startup()
        captures_store.sweep_retention()
        logger.info(
            "Captures store initialized at %s (audio dir: %s, enabled=%s)",
            cfg.CAPTURES_DB, cfg.CAPTURES_DIR,
            getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False),
        )
        captures_sweep_task = asyncio.create_task(
            _captures_retention_loop()
        )
    except Exception as _ce:
        logger.error("Failed to initialize captures store: %s", _ce)

    yield

    async def _cancel(task) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    await _cancel(evictor_task)
    await _cancel(diarization_evictor_task)
    await _diarization.drop_pipeline()
    await _cancel(bgm_evictor_task)
    await _bgm_separation.drop_separator()
    await _cancel(translation_evictor_task)
    await _tr.drop_models()
    if url_media_janitor_task is not None:
        await _cancel(url_media_janitor_task)
    await _cancel(reports_sweep_task)
    await _cancel(captures_sweep_task)
    await _cancel(sessions_purge_task)
    await _cancel(open_mode_task)

    # force: the process is going away, so leases buy nothing — same contract
    # as drain_then_evict, where an in-flight request finishes on its own ref.
    for _name in list(_loaded_models):
        _drop_loaded_model(_name, force=True)
    _model_leases.clear()
    # Best-effort NVML shutdown so the service exit doesn't leak driver
    # handles. Safe to call when NVML didn't init.
    system_stats.shutdown()


# docs_url/redoc_url/openapi_url=None disables FastAPI's built-in (unauthenticated)
# docs; they're re-added below behind the admin-tier host gate (+ admin key on
# /openapi.json) so the API surface isn't exposed to arbitrary hosts.
app = FastAPI(
    title="Faster Whisper API", version=APP_VERSION, lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,
)

# CORS — opt-in, off by default (empty allowlist → no middleware, no
# Access-Control-* headers, unchanged behavior). Enable by listing browser
# origins in CORS_ALLOW_ORIGINS so cross-origin JSON-API calls work (e.g. the
# /dictate demo's batch-mode fetch from a different origin than the backend).
# '*' allows any origin, in which case credentials must be disabled per the CORS
# spec. The WebSocket streaming path is not subject to CORS.
_cors_origins = list(getattr(cfg, "CORS_ALLOW_ORIGINS", []) or [])
# Decided outside the `if` because the origin guard below reads it too (an
# empty allowlist is never "allow all").
_cors_allow_all = "*" in _cors_origins
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if _cors_allow_all else _cors_origins,
        allow_credentials=not _cors_allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s",
                "* (any)" if _cors_allow_all else ", ".join(_cors_origins))

# Extra origins the unsafe-method origin guard accepts besides the request's
# own Host — for a reverse proxy that rewrites Host to the upstream. Read ONLY
# by _origin_is_allowed: it adds no CORS headers and no cross-origin access.
_trusted_origins = list(getattr(cfg, "TRUSTED_ORIGINS", []) or [])
if _trusted_origins:
    logger.info("Trusted origins (same-origin check): %s",
                ", ".join(_trusted_origins))

# Static assets for the /stats dashboard (vendored uPlot, etc). Local-only —
# do not put anything sensitive under static/.
from fastapi.staticfiles import StaticFiles
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# API docs are admin-tier — treated like /settings. The HTML shells (/docs,
# /redoc) are gated by the admin host allowlist only (a keyless browser must be
# able to load the swagger UI), while /openapi.json — the actual API surface —
# additionally requires an admin key. In OPEN mode (no admin key yet) the
# synthetic admin passes, so loopback docs work out of the box; once locked
# down, /openapi.json needs an admin session/key (the swagger UI fetches it
# same-origin with the session cookie).
from web_common import require_user_webui_host, require_admin_webui_host
from auth import require_admin as _require_admin
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import JSONResponse as _JSONResponse


@app.get(
    "/openapi.json",
    include_in_schema=False,
    dependencies=[Depends(require_admin_webui_host), Depends(_require_admin)],
)
async def _openapi_json():
    return _JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False, dependencies=[Depends(require_admin_webui_host)])
async def _swagger_ui():
    # Vendored, not FastAPI's cdn.jsdelivr.net defaults. These pages run in the
    # app's own origin with the admin's session cookie, so anyone able to alter
    # the CDN response would be executing code with admin rights here — the
    # same reasoning that had uPlot and GridStack vendored (static/VENDOR.md).
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " — docs",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon-32.png",
        # Suppresses swagger-ui's OnlineValidatorBadge, which otherwise
        # defaults to https://validator.swagger.io/validator and emits an
        # <img>/<a> pointing at it. Its only guard is a "localhost"/"127.0.0.1"
        # substring test on the definition URL, so the moment an operator adds
        # their subnet to ADMIN_WEBUI_ALLOWED_HOSTS this page starts telling a
        # third party the backend's internal host and port — and asks that
        # third party to fetch it. Vendoring the bundle did not stop this;
        # it is a runtime config default, not a script URL.
        swagger_ui_parameters={"validatorUrl": None},
    )


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(require_admin_webui_host)])
async def _redoc_ui():
    # Vendored — see the note on /docs above.
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " — redoc",
        redoc_js_url="/static/redoc.standalone.js",
        redoc_favicon_url="/static/favicon-32.png",
    )

# Per-request metrics middleware. Records (path, status, duration) for every
# HTTP request — bumps in_flight tracked separately by the transcribe handler.
import metrics

# Central running-jobs registry (transcribe/dictate/translate/download) —
# feeds /stats and the WebUI header activity cluster.
import jobs


_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# Paths that issue a session and therefore can't carry a CSRF token yet.
# Exempt from the TOKEN check only — the origin check still applies (these
# paths hand out a cookie, so a cross-site caller must not reach them).
_CSRF_EXEMPT_PATHS = frozenset({"/auth/login"})


# A rejected Origin is logged at most this often: the check runs before any
# credential, so an unauthenticated caller could otherwise drive the log at
# request rate (same reasoning as auth.py's open-mode nag interval).
_ORIGIN_REJECT_LOG_INTERVAL_S = 60.0
_origin_reject_logged_at = 0.0


def _origin_is_allowed(request) -> bool:
    """True when an unsafe-method request may proceed based on its Origin.

    Takes any Starlette HTTPConnection — only `.headers` is read — so the
    WebSocket handshake can reuse it. `@app.middleware("http")` wraps in
    BaseHTTPMiddleware, which passes non-http scopes straight through, so a
    websocket scope never reaches _csrf_mw and has to call this itself.

    Browsers attach Origin to every cross-site unsafe-method request (a plain
    auto-submitting <form> included), so this catches the case the token check
    cannot: OPEN mode, where a request carries no cookie and no bearer yet
    still resolves to the synthetic admin. Non-browser clients (curl, SDKs)
    send no Origin at all — absent means allow.

    Only host:port is compared against Host: a TLS-terminating proxy rewrites
    neither header, but the scheme in front of it is unknowable from here. A
    proxy that DOES rewrite Host to the upstream never matches, which is what
    TRUSTED_ORIGINS is for (CORS_ALLOW_ORIGINS also counts, for compatibility
    — but it additionally switches CORS on, so it is the wrong knob here).

    CORS_ALLOW_ORIGINS="*" does NOT satisfy this check. The two answer
    different questions: CORS decides whether a cross-origin page may READ a
    response, this decides whether it may PERFORM a state change — and the
    wildcard used to short-circuit here, silently disabling the guard for every
    unsafe method app-wide (and for the WebSocket handshake, which calls this
    directly). A deployment that genuinely needs cross-origin writes lists its
    real origins in TRUSTED_ORIGINS, which exists for exactly that.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin in _cors_origins or origin in _trusted_origins:
        return True
    host = request.headers.get("host", "")
    from urllib.parse import urlsplit
    return bool(host) and urlsplit(origin).netloc == host


def _log_origin_rejected(request) -> None:
    """Throttled WARNING naming the two headers that decided it — the only way
    an operator can tell a genuine cross-site POST from a reverse proxy that
    rewrote Host. Both values are attacker-controlled, hence _log_safe."""
    global _origin_reject_logged_at
    now = time.monotonic()
    if now - _origin_reject_logged_at < _ORIGIN_REJECT_LOG_INTERVAL_S:
        return
    _origin_reject_logged_at = now
    logger.warning(
        "Rejected %s %s: Origin %r does not match Host %r. If this is your own "
        "reverse proxy rewriting Host, add the public origin to "
        "TRUSTED_ORIGINS (WHISPER_TRUSTED_ORIGINS); otherwise it was a "
        "cross-site request. Further rejections are logged at most every %.0f s.",
        getattr(request, "method", "WEBSOCKET"), _log_safe(request.url.path),
        _log_safe(request.headers.get("origin")),
        _log_safe(request.headers.get("host")),
        _ORIGIN_REJECT_LOG_INTERVAL_S,
    )


@app.middleware("http")
async def _csrf_mw(request: Request, call_next):
    """Double-submit CSRF guard for COOKIE-authenticated mutations, plus a
    same-origin check on every unsafe method.

    Cookies are auto-sent by the browser, so a cross-site POST would ride
    the session cookie — hence we require an X-CSRF-Token header matching
    the session's stored token on unsafe methods. Requests WITHOUT a
    session cookie (Authorization: Bearer API clients — curl, SDKs) are
    untouched by the token half: they can't be CSRF'd and must keep working
    without a token. The origin half runs regardless of which credential the
    request carries (or none, in open mode).
    """
    if request.method.upper() not in _CSRF_SAFE_METHODS:
        from fastapi.responses import JSONResponse
        if not _origin_is_allowed(request):
            _log_origin_rejected(request)
            return JSONResponse(
                {"detail": "Origin not allowed for this host"},
                status_code=403,
            )
        if request.url.path not in _CSRF_EXEMPT_PATHS:
            cookie = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
            if cookie:
                import hmac
                import sessions_store
                sess = sessions_store.lookup_session(cookie)
                header_tok = request.headers.get("x-csrf-token", "")
                if (
                    sess is None
                    or not header_tok
                    or not hmac.compare_digest(header_tok, sess["csrf_token"])
                ):
                    return JSONResponse(
                        {"detail": "CSRF token missing or invalid"},
                        status_code=403,
                    )
    return await call_next(request)


@app.middleware("http")
async def _max_body_mw(request: Request, call_next):
    """Service-wide ceiling on a declared request body, rejected before the
    body is read. Route-level caps stay authoritative for their own endpoint
    (MAX_REQUEST_BYTES sits well above MAX_UPLOAD_BYTES, so the transcription
    413 still fires first); this one exists for the JSON routes, where
    Starlette buffers the whole body and json.loads expands it several-fold
    before any handler-side size check can run. Content-Length is advisory
    (absent on chunked bodies), so the header check only buys an early exit —
    the receive-side counter below is what actually enforces the cap: a
    `Transfer-Encoding: chunked` body declares no length, so without it an
    unauthenticated POST /auth/login could buffer unbounded bytes before any
    credential was checked.

    Registered between _csrf_mw and _metrics_mw: outside the CSRF guard and
    the router, inside _metrics_mw so these rejections still get recorded.
    """
    max_body = int(getattr(cfg, "MAX_REQUEST_BYTES", 268_435_456))
    # A JSON body gets a much tighter ceiling than the service-wide one. FastAPI
    # calls `await request.json()` BEFORE solve_dependencies (fastapi/routing.py
    # 0.141.x), so the payload is buffered AND json.loads-expanded ahead of the
    # host gate, get_current_user and every in-handler rate limiter — measured
    # ~24x RSS amplification on nested empty lists, i.e. ~6 GB from one
    # unauthenticated request at the 256 MB ceiling. Route-level or pydantic
    # max_length cannot help: pydantic never sees the payload until the parse
    # has already built it. 4 MiB is ~2.5x the largest legitimate JSON body (a
    # full 10 000-entry callback:map patch at ~1.5 MiB worst case — 64-char
    # keys + values; next largest is the 512 KB client_settings cap).
    # getattr default, so no config-schema change is required. multipart audio
    # uploads keep the full MAX_REQUEST_BYTES — the prefix test only matches
    # application/json.
    if request.headers.get("content-type", "").startswith("application/json"):
        max_body = min(max_body, int(getattr(cfg, "MAX_JSON_BODY_BYTES", 4_194_304)))
    _clen = request.headers.get("content-length")
    if _clen and _clen.isdigit() and int(_clen) > max_body:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "request body too large"}, status_code=413)

    # Count bytes as they stream in. Past the cap the receive channel reports
    # the client as disconnected, which unwinds whatever is consuming the body
    # (Starlette raises ClientDisconnect out of Request.body()/form()) instead
    # of letting it accumulate. Legitimate requests are unaffected: the cap is
    # MAX_REQUEST_BYTES, well above every route-level limit.
    _received = 0
    _orig_receive = request.receive

    async def _counting_receive():
        nonlocal _received
        message = await _orig_receive()
        if message.get("type") == "http.request":
            _received += len(message.get("body", b"") or b"")
            if _received > max_body:
                logger.warning(
                    "Aborting %s %s: request body exceeded the effective cap "
                    "(%d bytes) with no declared Content-Length",
                    request.method, _log_safe(request.url.path), max_body,
                )
                return {"type": "http.disconnect"}
        return message

    request._receive = _counting_receive  # type: ignore[attr-defined]
    return await call_next(request)


@app.middleware("http")
async def _metrics_mw(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    response = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Prefer the route's templated path (e.g. /captures/api/{cid}) so
        # per-ID URLs collapse to a single counter entry; unbounded raw-path
        # keys would otherwise grow the dict forever and turn the /stats
        # endpoint-counters panel into noise. Starlette stores the matched
        # route in the scope after routing — fall back to the raw URL path
        # for 404s and pre-routing failures.
        route = request.scope.get("route") if response is not None else None
        path = route.path if route is not None else request.url.path
        metrics.record_request(path, status,
                               (time.perf_counter() - start) * 1000.0,
                               unmatched=route is None)


# Responses that legitimately want to be cached: the vendored bundles, the
# fonts and the icons under /static are versioned assets with no per-identity
# content, and making the browser re-fetch ~2.5 MB of swagger/redoc on every
# page load would be a real regression.
_CACHEABLE_PREFIXES = ("/static/",)

# Deliberately NO default-src / script-src / style-src / img-src / media-src /
# worker-src. Every page in this product is a single document with inline
# <script> and <style> blocks, the shared header emits an inline onclick=, the
# dictate page builds its AudioWorklet from a blob: URL, and several pages set
# CSS backgrounds from data: SVGs and play audio through createObjectURL. A
# nonce-less script-src 'self' would break all of it, starting with microphone
# capture. These four directives are the ones that cost nothing here: no HTML
# in the tree contains an <iframe>, there is no <base> tag, both <form>s post
# to self, and there is no <object>/<embed>.
_CSP = (
    "frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'self'; object-src 'none'"
)


@app.middleware("http")
async def _security_headers_mw(request: Request, call_next):
    """Outermost layer: response headers every route should carry.

    Registered last so it wraps _metrics_mw/_max_body_mw/_csrf_mw and therefore
    also stamps their early 403/413 returns.

    Cache-Control is set as a DEFAULT, not an override — a handler that already
    chose its own value keeps it. Before this, the tree set no-store on seven
    responses by hand and left the rest bare, including /reports/api/list, whose
    body is scope-filtered per caller: a shared cache keyed on URL alone could
    hand an admin's full-corpus response to a scope="own" user. A 200 GET with
    no Cache-Control, no Expires and no Vary is heuristically cacheable under
    RFC 9111, and this deployment expects a reverse proxy in front
    (TRUSTED_ORIGINS exists for exactly that). Defaulting to no-store everywhere
    outside /static is the version of that rule nobody can forget to apply to a
    new endpoint.

    Framing: session cookies are SameSite=lax, so a cross-site frame of an admin
    page loads without the cookie and shows the login gate. That is not true in
    OPEN mode, where an allowlisted-host victim resolves to the synthetic admin
    with no cookie at all and a framed /settings is fully privileged — and the
    CSRF guard does not help, because the victim clicks the real page, so the
    page's own JS attaches a valid token and a same-origin Origin.
    """
    response = await call_next(request)
    path = request.url.path
    if not path.startswith(_CACHEABLE_PREFIXES):
        response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", _CSP)
    return response


@app.exception_handler(_rl.RateLimited)
async def _rate_limited_handler(request: Request, exc: _rl.RateLimited):
    """Render every rate-limit refusal as the typed envelope from
    RateLimited.body() — which config field refused, and for how long.

    Registered for the SUBCLASS: Starlette walks type(exc).__mro__ looking for
    a handler, so this wins over the built-in HTTPException one even though
    RateLimited is an HTTPException. Without it the default handler would emit
    {"detail": …} and drop error.param/error.retry_after.
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(exc.body(), status_code=429,
                        headers={"Retry-After": str(exc.retry_after)})


def _shift_to_original_timeline(segments, info, pad_s: float):
    """Map decode results from the LEADING_SILENCE_PAD_MS-padded timeline back
    to the uploaded audio's timeline: segment/word times shift by -pad_s
    (clamped at 0 — VAD speech-padding can place an onset inside the injected
    silence) and ``duration`` drops the pad. Times must keep fitting the
    UN-padded audio: the API response, /captures rows, and the sample-merge
    tooling all interpret them against the original WAV. duration_after_vad is
    clamped to the original duration — how much of the injected pad the VAD
    swallowed is unknowable, so the pad/real-silence split is approximate
    there (diagnostic-only field). In-place mutation is safe: the objects come
    fresh from the decoder with this request as their only consumer."""
    for seg in segments:
        seg.start = max(0.0, seg.start - pad_s)
        seg.end = max(0.0, seg.end - pad_s)
        for w in (getattr(seg, "words", None) or []):
            w.start = max(0.0, w.start - pad_s)
            w.end = max(0.0, w.end - pad_s)
    orig_dur = max(0.0, float(getattr(info, "duration", 0.0) or 0.0) - pad_s)
    info.duration = orig_dur
    dav = getattr(info, "duration_after_vad", None)
    if dav is not None:
        info.duration_after_vad = min(orig_dur, float(dav))
    return segments, info


# Read granularity for the uploaded part. 1 MiB matches Starlette's own
# spool threshold, so a typical clip is copied in a handful of steps.
_UPLOAD_CHUNK_BYTES = 1024 * 1024

# The temp-file suffix is only a convenience for ffmpeg/av format sniffing, but
# it comes straight off the client-supplied filename and is handed to
# tempfile.NamedTemporaryFile, which builds a real path out of it: a 250-char
# extension raised OSError(36, 'File name too long') and an embedded NUL raised
# ValueError, both surfacing as a 500 plus an err_count bump. Traversal is NOT
# the concern (splitext splits after the last separator, so the extension can
# never contain one) — length and exotic bytes are. Screen it, and fall back to
# no suffix rather than rejecting the upload.
_TMP_SUFFIX_RE = re.compile(r"\A\.[A-Za-z0-9_-]{1,15}\Z")


def _safe_tmp_suffix(filename: "str | None") -> str:
    """Extension of `filename` if it is short and plainly safe, else ""."""
    ext = os.path.splitext(filename or "")[1]
    return ext if _TMP_SUFFIX_RE.match(ext) else ""


# ── Batch progress registry ──────────────────────────────────────────────────
# Optional per-request progress for the file-upload path: a client that sends a
# `progress_id` form field can poll GET /v1/audio/transcriptions/progress/<id>
# while its POST is in flight. Entries live only for the request (popped in the
# handler's finally); the cap + stale sweep below bound a client that invents
# ids and never posts. Plain dict + GIL: every writer does a single dict-entry
# update, and the poller only reads.
_BATCH_PROGRESS: "dict[str, dict]" = {}
_BATCH_PROGRESS_MAX = 200
_BATCH_PROGRESS_STALE_S = 2 * 3600
_PROGRESS_ID_RE = re.compile(r"\A[0-9a-f]{8,64}\Z")

# One target-language code inside the `translate_to` csv: a 2-3 letter base
# ("en", "de", "gsw") plus an optional BCP-47-ish subtag ("fr-CA", "zh-Hant").
_TRANSLATE_CODE_RE = re.compile(r"\A[a-z]{2,3}(-[A-Za-z0-9]{2,8})?\Z")


# progress_id → job id: handlers that registered a job in jobs.py bind their
# progress_id here so every _progress_set call (all stages already flow
# through it) mirrors stage/progress into the central registry for free.
# last_text is deliberately NOT mirrored — job rows never carry transcript
# text (see jobs.jobs_snapshot's scrubbing contract).
_JOB_BY_PID: "dict[str, str]" = {}
_JOB_MIRROR_FIELDS = ("stage", "progress", "step", "model", "total_bytes")


def _progress_set(pid: "str | None", **fields) -> None:
    """Merge `fields` into the progress entry for `pid` (no-op without one)."""
    if not pid:
        return
    _job_id = _JOB_BY_PID.get(pid)
    if _job_id:
        jobs.job_update(_job_id,
                        **{k: fields[k] for k in _JOB_MIRROR_FIELDS
                           if fields.get(k) is not None})
    entry = _BATCH_PROGRESS.get(pid)
    if entry is None:
        now = time.monotonic()
        for k in [k for k, v in _BATCH_PROGRESS.items()
                  if now - v.get("updated", 0) > _BATCH_PROGRESS_STALE_S]:
            _BATCH_PROGRESS.pop(k, None)
        if len(_BATCH_PROGRESS) >= _BATCH_PROGRESS_MAX:
            oldest = min(_BATCH_PROGRESS,
                         key=lambda k: _BATCH_PROGRESS[k].get("updated", 0))
            _BATCH_PROGRESS.pop(oldest, None)
        entry = _BATCH_PROGRESS[pid] = {}
    entry.update(fields)
    entry["updated"] = time.monotonic()


# Cooperative cancellation for in-flight batch requests: POST
# /v1/audio/transcriptions/cancel/<id> flags the id here, and the handler's
# stage callbacks (which already fire every demix chunk / decoded segment /
# pyannote step) poll the flag and abort. Closing the HTTP connection alone
# does NOT stop the work — the stages run in executor threads that outlive a
# cancelled handler task. Only ids with a live _BATCH_PROGRESS entry can be
# flagged, and the handler's finally discards, so the set stays bounded by
# the number of in-flight requests.
_BATCH_CANCELLED: "set[str]" = set()


class _ClientCancelled(Exception):
    """The client cancelled this request via the cancel endpoint."""


def _cancel_requested(pid: "str | None") -> bool:
    return bool(pid) and pid in _BATCH_CANCELLED


def _check_cancelled(pid: "str | None") -> None:
    if _cancel_requested(pid):
        raise _ClientCancelled()


def _form_bool(value: "str | None") -> "bool | None":
    """Tri-state multipart boolean: multipart values arrive as strings, and
    FastAPI's bool coercion can't keep "absent" (inherit the config default)
    distinct from "false" (explicitly off). Unrecognised spellings read as
    absent — the sloppy-caller-keeps-working stance of the clamped knobs."""
    if value is None:
        return None
    s = value.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: "UploadFile | None" = File(None),
    source_url: "str | None" = Form(None),
    model_name: str = Form("whisper-1", alias="model"),
    response_format: str = Form("json"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    prompt: str | None = Form(None),
    decode_overrides: str = Form(None),
    override_profile: str = Form(None),
    task: str | None = Form(None),
    diarize: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    diarization_model: str | None = Form(None),
    separate_bgm: str | None = Form(None),
    separation_model: str | None = Form(None),
    translate_to: str | None = Form(None),
    translation_model: str | None = Form(None),
    translation_mode: str | None = Form(None),
    translation_glossary: str | None = Form(None),
    progress_id: str | None = Form(None),
    user: dict = Depends(_get_current_user_dep),
):
    resolved_model = _resolve_model_name(model_name)
    # Opt-in progress reporting (see _BATCH_PROGRESS). A malformed id is
    # treated as absent — progress is a convenience, never a 422.
    _pid = progress_id if (progress_id and _PROGRESS_ID_RE.match(progress_id)) else None
    # Seed the registry entry NOW so the cancel endpoint (which only accepts
    # ids it can see in-flight) has a target from the first moment — the
    # first stage-driven _progress_set can otherwise be seconds away (model
    # load, semaphore queue). URL runs seed as "resolving": their pipeline
    # starts at the link, and "waiting" maps onto the transcribe row in the
    # client's rail — which would paint the download as already done.
    _progress_set(_pid,
                  stage=("resolving" if source_url is not None else "waiting"),
                  progress=None)
    # Whisper's only two tasks; anything else is a caller error, not something
    # to silently coerce (unlike the clamped numeric knobs below, a wrong task
    # would return output in the wrong language with no other signal).
    task = (task or "").strip() or None
    if task is not None and task not in ("transcribe", "translate"):
        raise HTTPException(status_code=422,
                            detail="task must be 'transcribe' or 'translate'")
    # Translation run mode: same caller-error stance as `task` — a wrong mode
    # would return differently-aligned output with no other signal, so 422
    # rather than a silent coerce.
    translation_mode = (translation_mode or "").strip() or None
    if translation_mode is not None and translation_mode not in ("fluent", "faithful"):
        raise HTTPException(
            status_code=422,
            detail="translation_mode must be 'fluent' or 'faithful'")
    # Speaker-count hints: clamp like the other schemaless numeric knobs.
    _spk_clamp = lambda v: min(32, max(1, v)) if v is not None else None  # noqa: E731
    num_speakers = _spk_clamp(num_speakers)
    min_speakers = _spk_clamp(min_speakers)
    max_speakers = _spk_clamp(max_speakers)
    # Bound the two OpenAI-compatible knobs that carry no schema of their own.
    # Both are already bounded on every sibling path — DEFAULT_PROMPT is
    # Field(max_length=2048) and the `hotwords` client override is capped by
    # _DECODE_STR_CAPS, while `temperature` inside decode_overrides is clamped
    # by _DECODE_FLOAT_BOUNDS. Clamp rather than 422 so a caller that is merely
    # sloppy keeps working; NaN fails every comparison, hence the self-test.
    if prompt is not None:
        prompt = prompt[:_DECODE_STR_CAPS.get("prompt", 2048)]
    _t_lo, _t_hi = _DECODE_FLOAT_BOUNDS.get("temperature", (0.0, 1.0))
    temperature = (
        min(_t_hi, max(_t_lo, temperature)) if temperature == temperature else _t_lo
    )
    # Normalise the optional override-profile name the same way the streaming
    # handshake does (trim; blank → None) so both endpoints honor an identical
    # set of names instead of batch silently rejecting whitespace-padded ones.
    override_profile = (override_profile or "").strip() or None

    # Transcribe-from-URL: exactly one source. Both/neither is a caller error
    # (422, like a bad `task`); a URL on a server with the feature off is a
    # curated 403 so the client can say "not enabled here" instead of
    # guessing. Gate BEFORE the model load — no GPU work for a rejected URL.
    source_url = (source_url or "").strip() or None
    if source_url is not None and file is not None:
        raise HTTPException(status_code=422,
                            detail="provide either a file or a source_url, "
                                   "not both")
    if source_url is None and file is None:
        raise HTTPException(status_code=422,
                            detail="provide a file or a source_url")
    if source_url is not None and not getattr(cfg, "URL_DOWNLOAD_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="URL download is not enabled on this server")

    # Bracket the entire request with metrics.in_flight + record_transcription
    # so failed loads / failed transcriptions still surface in the dashboard.
    # request_id is generated up-front (was deferred to post-transcribe) so
    # the outer finally can correlate timing-only writes to the SQLite store
    # on the error path too.
    metrics.in_flight_transcriptions += 1
    _t0 = time.perf_counter()
    _status = "ok"
    _audio_dur: float = 0.0
    _words: int = 0
    # Per-stage wall-clock receipts ({name, secs, model?, detail?}) — the
    # durations were previously computed for log lines and discarded; now
    # they also persist as the recent-jobs row's stages_json.
    _stage_timings: "list[dict]" = []
    tmp_path = None
    # Transcribe-from-URL state: the private download dir (rmtree'd in the
    # inner finally on every path) and the retention id echoed to the client.
    _url_job_dir: "str | None" = None
    _source_media_id: "str | None" = None
    # Set only AFTER the load returns, so the outer finally never releases a
    # lease that was never taken (a rejected/failed load takes none).
    _leased_model: "str | None" = None
    request_id = uuid.uuid4().hex
    _user_id = user.get("user_id")
    _key_id = user.get("key_id")
    # Central running-jobs registry: one entry per in-flight request; stage/
    # progress mirror in via _progress_set (bound through _JOB_BY_PID) when
    # the client opted into progress reporting.
    jobs.job_start("transcribe", id=request_id, model=resolved_model,
                   user=_user_id, key=_key_id)
    if _pid:
        _JOB_BY_PID[_pid] = request_id
        jobs.job_update(request_id, progress_id=_pid)
    try:
        # Upload ceiling. Content-Length is advisory (absent on chunked
        # bodies), so it only buys us an early exit before the model load —
        # the chunked read below is what actually enforces the bound.
        max_upload = int(getattr(cfg, "MAX_UPLOAD_BYTES", 200_000_000))
        _clen = request.headers.get("content-length")
        if _clen and _clen.isdigit() and int(_clen) > max_upload:
            raise HTTPException(status_code=413, detail="upload too large")

        model = await _get_or_load_model(resolved_model, lease=True)
        _leased_model = resolved_model

        # Resolve the caller's effective per-identity config ONCE for this
        # request: layered decode params, pipeline include/exclude, output
        # wrappers, and which fields are locked against client overrides.
        # Open mode / no per-identity config → no identity layers (≡ today).
        # `override_profile` (if sent + allowed) joins as the least-specific layer.
        ident = build_ident(user, resolved_model, request_profile=override_profile)

        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = form_data.getlist("timestamp_granularities")

        # prompt sentinel: FastAPI coerces an empty `prompt` Form field to the
        # parameter default (None), erasing the present-but-empty signal. Read the
        # RAW form value so an explicit "" (CLEAR the inherited prompt) stays
        # distinct from an absent field (INHERIT DEFAULT_PROMPT). A non-str (e.g. an
        # accidental file part) is treated as absent.
        _prompt_field = form_data.get("prompt")
        prompt = _prompt_field if isinstance(_prompt_field, str) else None
        # Re-apply the clamp from above: this re-read overwrote the bounded
        # Form value with the raw one, so the cap was dead code and the field
        # reached the tokenizer at whatever size the multipart parser allowed.
        if prompt is not None:
            prompt = prompt[:_DECODE_STR_CAPS.get("prompt", 2048)]

        include_words = "word" in timestamp_granularities or (
            response_format == "verbose_json" and not timestamp_granularities
        )

        try:
            if source_url is not None:
                # Transcribe-from-URL: policy-gated metadata probe, then a
                # yt-dlp subprocess download into a private job dir. The
                # result is moved into the retention store (so the client can
                # fetch it once for playback) and a pipeline-owned copy takes
                # the tmp_path slot — everything downstream, including the
                # BGM tmp_path swap and the finally's unlink, is unchanged.
                # The download deliberately does NOT hold the inference
                # semaphore (network-bound); it has its own, narrower one.
                import url_download as _udl
                import url_media_store as _ums
                _url_max = int(getattr(cfg, "URL_MAX_BYTES", 0) or 0) or max_upload
                _progress_set(_pid, stage="resolving", progress=None)
                logger.info("[url-dl] transcribe-from-url requested (host %s)",
                            _url_host_for_log(source_url))
                try:
                    _check_cancelled(_pid)
                    _url = _udl.validate_url(source_url)
                    _uinfo = await _udl.probe(
                        _url,
                        timeout=float(getattr(cfg, "URL_PREVIEW_TIMEOUT_SEC", 20)))
                    logger.info(
                        "[url-dl] resolved (host %s): extractor=%s duration=%s"
                        " — starting download",
                        _url_host_for_log(_url), _uinfo.extractor_key,
                        f"{_uinfo.duration:.0f}s"
                        if _uinfo.duration is not None else "?")
                    _progress_set(_pid, stage="downloading", progress=None,
                                  total_bytes=None,
                                  step=(_uinfo.extractor_key or None))
                    _url_job_dir = tempfile.mkdtemp(prefix="urldl-")
                    async with _get_url_download_semaphore():
                        _check_cancelled(_pid)
                        _dl_path = await _udl.download(
                            _url,
                            dest_dir=_url_job_dir,
                            max_bytes=_url_max,
                            timeout=float(getattr(
                                cfg, "URL_DOWNLOAD_TIMEOUT_SEC", 900)),
                            progress_cb=lambda f, tot: _progress_set(
                                _pid, stage="downloading", progress=f,
                                total_bytes=tot),
                            cancel_check=lambda: _cancel_requested(_pid))
                except _udl.UrlCancelled:
                    logger.info("[url-dl] download cancelled by client "
                                "(host %s)", _url_host_for_log(source_url))
                    raise _ClientCancelled() from None
                except _udl.UrlDownloadError as _ue:
                    # str() is client-safe by the module's contract.
                    logger.info("[url-dl] rejected (host %s): %s",
                                _url_host_for_log(source_url),
                                _log_safe(str(_ue)))
                    raise HTTPException(status_code=400, detail=str(_ue))
                audio_bytes = os.path.getsize(_dl_path)
                # Pipeline copy FIRST (hardlink where possible), THEN move
                # the original into the retention store — afterwards each
                # side owns its file outright: tmp_path follows the normal
                # unlink-in-finally lifecycle (including the BGM swap), and
                # the retained file serves GET /v1/audio/url-media/{id}.
                tmp_path = _ums.make_pipeline_copy(_dl_path)
                if tmp_path is None:
                    # Disk trouble (logged by the store) — generic 500 path.
                    raise RuntimeError("url pipeline copy failed")
                # Retention is a playback nicety: None just means the client
                # gets no audio copy, never a failed transcription.
                _source_media_id = _ums.register(_dl_path, user_id=_user_id)
            else:
                # Stream the part to the temp file in chunks, counting bytes as
                # we go: the upload is never fully resident, and an oversized
                # body is cut off mid-read instead of after it has been
                # materialised. Only the SIZE is needed downstream (capture
                # size guard + log block).
                audio_bytes = 0
                with tempfile.NamedTemporaryFile(delete=False, suffix=_safe_tmp_suffix(file.filename)) as tmp_file:
                    tmp_path = tmp_file.name
                    while True:
                        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        audio_bytes += len(chunk)
                        if audio_bytes > max_upload:
                            raise HTTPException(status_code=413, detail="upload too large")
                        tmp_file.write(chunk)

            # word_timestamps: AND of the (per-model-overrideable) global
            # config knob and the per-request ask. Disabled (False) bypasses
            # the DTW alignment path entirely — required for primeline-style
            # finetunes that hit faster-whisper#1212.
            gate_word_ts = cfg_for(resolved_model, "WORD_TIMESTAMPS_ENABLED", ident)
            want_word_ts = gate_word_ts and include_words

            # Capture-for-fine-tuning decision. We gate via gate_word_ts
            # (NOT override): per-model WORD_TIMESTAMPS_ENABLED=False is
            # used on primeline/tnfru-family fine-tunes where DTW is
            # broken — forcing word_timestamps=True there produces empty
            # transcripts. Skip capture instead.
            #
            # Sampling roll + cap check + size guard happen at handler
            # entry so we don't waste DTW CPU on requests that won't
            # land. Duration filter is post-transcribe (we don't know
            # the duration yet).
            will_capture = False
            captured_id: str | None = None
            if (getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)
                    and gate_word_ts):
                try:
                    import captures_store as _cap_store
                    cap_max = int(getattr(cfg, "CAPTURES_MAX", 5000))
                    hard_lim = int(getattr(
                        cfg, "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT",
                        100_000_000,
                    ))
                    sample_rate = float(getattr(
                        cfg, "CAPTURE_RECORDINGS_SAMPLE_RATE", 1.0,
                    ))
                    if (_cap_store.count() < cap_max
                            and audio_bytes < hard_lim
                            and random.random() < sample_rate):
                        will_capture = True
                        want_word_ts = True  # force DTW for capture
                except Exception as _ce:
                    logger.warning("[capture] eligibility check failed: %s", _ce)

            # Empty string is NOT equivalent to None for tnfru / primeline
            # finetunes — passing "" to model.transcribe(initial_prompt=...)
            # triggers the failure mode their model card warns about. Coerce.
            # Prompt: a LOCKED DEFAULT_PROMPT forbids the client's `prompt`
            # param — the admin value stands. `ignored` collects what we drop
            # so the verbose_json response can surface it (never silent).
            # prompt sentinel: None (field absent) = inherit DEFAULT_PROMPT; an
            # explicit "" (field present-but-empty) = CLEAR (no initial_prompt); a
            # value is used verbatim. The `if _prompt else None` coerce below turns
            # an explicit "" into None for model.transcribe (passing "" trips the
            # tnfru/primeline finetunes' documented failure mode).
            ignored: "list[str]" = []
            if "DEFAULT_PROMPT" in ident.locked:
                _prompt = cfg_for(resolved_model, "DEFAULT_PROMPT", ident)
                if prompt is not None and prompt != _prompt:
                    ignored.append("prompt")
            elif prompt is not None:
                _prompt = prompt
            else:
                _prompt = cfg_for(resolved_model, "DEFAULT_PROMPT", ident)
            initial_prompt_arg = _prompt if _prompt else None

            _vad_filter = cfg_for(resolved_model, "VAD_FILTER", ident)
            vad_parameters = dict(
                min_silence_duration_ms=cfg_for(resolved_model, "VAD_MIN_SILENCE_MS", ident),
                speech_pad_ms=cfg_for(resolved_model, "VAD_SPEECH_PAD_MS", ident),
                threshold=cfg_for(resolved_model, "VAD_THRESHOLD", ident),
            ) if _vad_filter else None

            _lead_pad_ms = int(cfg_for(resolved_model, "LEADING_SILENCE_PAD_MS", ident) or 0)

            # Coerce empty to None — faster-whisper validates the value against
            # its accepted-codes list, so "" raises ValueError; None triggers
            # the first-30s auto-detect path, which is what an empty
            # DEFAULT_LANGUAGE is documented to mean. A LOCKED DEFAULT_LANGUAGE
            # likewise forbids the client's `language` param.
            if "DEFAULT_LANGUAGE" in ident.locked:
                _language = cfg_for(resolved_model, "DEFAULT_LANGUAGE", ident)
                if language and language != _language:
                    ignored.append("language")
            else:
                _language = language or cfg_for(resolved_model, "DEFAULT_LANGUAGE", ident)
            # Task: absent field inherits the resolved TASK config (per-identity
            # > per-model > global, default "transcribe"); a LOCKED TASK forbids
            # the client's `task` param the way a locked DEFAULT_LANGUAGE binds
            # `language` above.
            if "TASK" in ident.locked:
                _task = cfg_for(resolved_model, "TASK", ident) or "transcribe"
                if task is not None and task != _task:
                    ignored.append("task")
            else:
                _task = task or cfg_for(resolved_model, "TASK", ident) or "transcribe"
            # Diarization request knobs: same absent-inherits / locked-wins
            # shape as task above. The capacity gate (DIARIZATION_ENABLED)
            # is checked at the stage itself and soft-fails into `warnings`.
            _diarize_req = _form_bool(diarize)
            if "DIARIZE" in ident.locked:
                _diarize = bool(cfg_for(resolved_model, "DIARIZE", ident))
                if _diarize_req is not None and _diarize_req != _diarize:
                    ignored.append("diarize")
            elif _diarize_req is not None:
                _diarize = _diarize_req
            else:
                _diarize = bool(cfg_for(resolved_model, "DIARIZE", ident))
            _spk = {}
            for _cfg_name, _client_name, _client_val in (
                ("DIARIZATION_NUM_SPEAKERS", "num_speakers", num_speakers),
                ("DIARIZATION_MIN_SPEAKERS", "min_speakers", min_speakers),
                ("DIARIZATION_MAX_SPEAKERS", "max_speakers", max_speakers),
            ):
                if _cfg_name in ident.locked:
                    _spk[_client_name] = cfg_for(resolved_model, _cfg_name, ident)
                    if _client_val is not None and _client_val != _spk[_client_name]:
                        ignored.append(_client_name)
                elif _client_val is not None:
                    _spk[_client_name] = _client_val
                else:
                    _spk[_client_name] = cfg_for(resolved_model, _cfg_name, ident)
            # pyannote treats num alongside min/max as an error — num wins.
            if _spk.get("num_speakers"):
                _spk["min_speakers"] = _spk["max_speakers"] = None
            # Music separation: same shape again. Soft-failed optional stages
            # (this and diarization) collect their explanations in _warnings.
            _warnings: "list[str]" = []
            # Requested stages this server declines to run (feature disabled).
            # Mirrored into the progress entry the moment each skip is known,
            # so a polling client can mark the stage "skipped" live instead of
            # inferring it — the warning text alone arrives only with the
            # final response.
            _skipped: "list[str]" = []
            _sep_req = _form_bool(separate_bgm)
            if "SEPARATE_BGM" in ident.locked:
                _separate = bool(cfg_for(resolved_model, "SEPARATE_BGM", ident))
                if _sep_req is not None and _sep_req != _separate:
                    ignored.append("separate_bgm")
            elif _sep_req is not None:
                _separate = _sep_req
            else:
                _separate = bool(cfg_for(resolved_model, "SEPARATE_BGM", ident))
            # Per-request stage models (pyannote pipeline id / UVR model):
            # same ladder again. A non-empty allowlist that misses the
            # resolved value soft-fails by skipping THAT stage — before its
            # enabled gate, so the warning names the actual reason.
            _dm_req = (diarization_model or "").strip() or None
            _diarization_model = _resolve_request_knob(
                resolved_model, ident, ignored,
                "DIARIZATION_MODEL", "diarization_model", _dm_req)
            # The allowlist constrains only the CLIENT-requested value (a
            # config/identity-inherited model is admin policy and always
            # passes) and always admits the configured default — so an EMPTY
            # allowlist means "the configured model only", never "anything".
            _diar_allowed = set(
                getattr(cfg, "DIARIZATION_ALLOWED_MODELS", []) or [])
            _diar_allowed.add(getattr(cfg, "DIARIZATION_MODEL", "") or "")
            if (_diarize and _dm_req is not None
                    and _diarization_model == _dm_req
                    and _diarization_model not in _diar_allowed):
                _warnings.append(
                    "requested diarization model is not allowed on this "
                    "server (DIARIZATION_ALLOWED_MODELS)")
                _skipped.append("diarizing")
                _progress_set(_pid, skipped=list(_skipped))
                _diarize = False
            _sm_req = (separation_model or "").strip() or None
            _separation_model = _resolve_request_knob(
                resolved_model, ident, ignored,
                "BGM_SEPARATION_UVR_MODEL", "separation_model", _sm_req)
            _sep_allowed = set(
                getattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", []) or [])
            _sep_allowed.add(getattr(cfg, "BGM_SEPARATION_UVR_MODEL", "") or "")
            if (_separate and _sm_req is not None
                    and _separation_model == _sm_req
                    and _separation_model not in _sep_allowed):
                _warnings.append(
                    "requested separation model is not allowed on this "
                    "server (BGM_SEPARATION_ALLOWED_MODELS)")
                _skipped.append("separating")
                _progress_set(_pid, skipped=list(_skipped))
                _separate = False
            # Translation (T2T) request knobs: the same locked-wins /
            # request-wins / config-inherits ladder as diarize/separate_bgm
            # above. The capacity gates (TRANSLATION_ENABLED, model allowlist)
            # live at the stage itself and soft-fail into `_warnings`.
            _tt_req = (translate_to or "").strip() or None
            _tt_raw = _resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATE_TO", "translate_to", _tt_req)
            # csv → deduped ordered list of well-formed codes. Malformed
            # entries drop silently (the sloppy-caller stance of the clamped
            # knobs); the MAX_TARGETS clamp warns, naming what it dropped.
            _translate_to: "list[str]" = []
            for _code in (_tt_raw or "").split(","):
                _code = _code.strip()
                if (_code and _code not in _translate_to
                        and _TRANSLATE_CODE_RE.match(_code)):
                    _translate_to.append(_code)
            _translation_max_targets = int(cfg_for(
                resolved_model, "TRANSLATION_MAX_TARGETS", ident) or 1)
            if len(_translate_to) > _translation_max_targets:
                _warnings.append(
                    "translation targets over TRANSLATION_MAX_TARGETS "
                    f"({_translation_max_targets}) were dropped: "
                    + ", ".join(_translate_to[_translation_max_targets:]))
                _translate_to = _translate_to[:_translation_max_targets]
            _tm_req = (translation_model or "").strip() or None
            _translation_model = _resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_MODEL", "translation_model", _tm_req)
            _translation_mode = _resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_MODE", "translation_mode", translation_mode,
                default="fluent")
            _tg_req = translation_glossary if (translation_glossary or "").strip() else None
            _translation_glossary = _resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_GLOSSARY", "translation_glossary", _tg_req)
            # The config field is Field(max_length=4000); cap the raw client
            # value to the same bound rather than 422ing a sloppy caller.
            _translation_glossary = (_translation_glossary or "")[:4000]
            _translation_context = int(cfg_for(
                resolved_model, "TRANSLATION_CONTEXT_SEGMENTS", ident) or 0)
            # Optional per-request decode overrides (JSON object). Malformed → ignored.
            _overrides = {}
            if decode_overrides:
                try:
                    _parsed = json.loads(decode_overrides)
                    if isinstance(_parsed, dict):
                        _overrides = _parsed
                except (ValueError, TypeError, RecursionError):
                    # RecursionError (a RuntimeError, NOT a ValueError) is what
                    # json.loads raises on a deeply nested array/object, so
                    # without it here a malformed value escapes to the handler's
                    # generic `except Exception` and becomes a logged 500 plus a
                    # permanent err_count bump — breaking the documented
                    # "malformed → ignored" contract. Matches the streaming twin.
                    _overrides = {}
            # Per-request decode keys dropped by a lock (assemble_transcribe_
            # kwargs enforces the drop; we record it here for the response).
            ignored.extend(sorted(k for k in _overrides if k in ident.locked_client_keys))
            # A locked TEMPERATURE has to bind the OpenAI-compat `temperature`
            # Form field too, the way a locked DEFAULT_PROMPT/DEFAULT_LANGUAGE
            # binds `prompt`/`language` above. assemble_transcribe_kwargs only
            # drops the LOCKED CLIENT KEY, i.e. `temperature` inside
            # decode_overrides; the Form field is threaded straight through and
            # is normally masked only because a non-empty resolved ladder
            # overwrites it. Blank the ladder at the winning layer (a supported
            # shape — effective_config documents value-less locks) and the Form
            # field became the one way past the lock.
            _temperature = temperature
            if "TEMPERATURE" in ident.locked:
                _locked_ladder = cfg_for(resolved_model, "TEMPERATURE", ident)
                if not (_locked_ladder or "").strip():
                    if temperature != _t_lo and "temperature" not in ignored:
                        ignored.append("temperature")
                    _temperature = _t_lo
            # Single source of truth — the streaming FINAL decode builds its kwargs
            # from this exact assembler too, so streaming and batch never diverge.
            transcribe_kwargs = assemble_transcribe_kwargs(
                resolved_model, model,
                language=_language, temperature=_temperature,
                vad_filter=_vad_filter, vad_parameters=vad_parameters,
                want_word_ts=want_word_ts, initial_prompt=initial_prompt_arg,
                overrides=_overrides, ident=ident, task=_task,
            )

            # Pre-decode music-separation stage (soft-fail): replaces the
            # uploaded tmp file with a vocals-only WAV, so the decode AND the
            # capture path below both see the separated audio (deliberate —
            # captures should match what was transcribed). The original upload
            # is unlinked here; the vocals file takes over tmp_path and the
            # finally unlinks it. Serialized on the shared semaphore like
            # every GPU stage.
            if _separate:
                if not getattr(cfg, "BGM_SEPARATION_ENABLED", False):
                    _warnings.append(
                        "music separation requested but not enabled on this "
                        "server (BGM_SEPARATION_ENABLED is off)")
                    _skipped.append("separating")
                    _progress_set(_pid, skipped=list(_skipped))
                else:
                    import bgm_separation as _bgm
                    try:
                        _sep_t0 = time.perf_counter()
                        _progress_set(
                            _pid, stage="separating", progress=None,
                            position=None, last_text=None,
                            model=(_separation_model or None),
                            # The ONNX session's real placement once a model
                            # is loaded (a CUDA provider that fails to load
                            # falls back to CPU silently); the config-resolved
                            # device only before the first load.
                            device=(_bgm.actual_device()
                                    or _bgm._resolve_device()))
                        _check_cancelled(_pid)
                        # libsndfile can't open AAC/MP4-family containers
                        # (m4a/mp4/webm…): the separator would fall back to a
                        # slow audioread/ffmpeg-subprocess decode — a silent
                        # minute for a 20-minute source — and log a
                        # "Format not recognised" warning. Hand it what it
                        # natively consumes instead: 44.1 kHz stereo s16 WAV
                        # via PyAV. (NOT 16 kHz mono — MDX separates on the
                        # full band and wants the Separator's 44100 default.)
                        # wav/flac are safe in every libsndfile; leave those.
                        _sep_src = tmp_path
                        _sep_wav = None
                        _sep_ext = (os.path.splitext(tmp_path)[1]
                                    .lstrip(".").lower())
                        if _sep_ext not in ("wav", "flac"):
                            try:
                                import audio_transcode as _atc
                                _tfd, _sep_wav = tempfile.mkstemp(
                                    prefix="sepsrc-", suffix=".wav")
                                os.close(_tfd)
                                _tc0 = time.perf_counter()
                                _progress_set(_pid, step="preparing")
                                await asyncio.to_thread(
                                    _atc.transcode_to_wav, tmp_path,
                                    _sep_wav, rate=44100, layout="stereo")
                                logger.info(
                                    "[bgm] input .%s → 44.1 kHz WAV for "
                                    "separation in %.1fs (%.1f MB)",
                                    _sep_ext or "?",
                                    time.perf_counter() - _tc0,
                                    os.path.getsize(_sep_wav) / 1e6)
                                _sep_src = _sep_wav
                            except Exception as _te:  # noqa: BLE001
                                logger.warning(
                                    "[bgm] input transcode failed (%s); "
                                    "separator will decode the original",
                                    _log_safe(str(_te)))
                                if _sep_wav is not None:
                                    try:
                                        os.unlink(_sep_wav)
                                    except OSError:
                                        pass
                                    _sep_wav = None
                                _sep_src = tmp_path
                        try:
                            _check_cancelled(_pid)
                            async with get_inference_semaphore():
                                _check_cancelled(_pid)
                                # "preparing" stays up through model load and
                                # the separator's own audio load/normalize
                                # (~40 s on long inputs); the first demix
                                # chunk clears it via the progress callback.
                                _progress_set(_pid, step="preparing")
                                _vocals_path = await _bgm.separate(
                                    _sep_src,
                                    model_filename=(_separation_model or None),
                                    progress_cb=lambda f: _progress_set(
                                        _pid, progress=f, step=None),
                                    cancel_check=lambda: _cancel_requested(
                                        _pid))
                        finally:
                            _progress_set(_pid, step=None)
                            # The intermediate WAV is ours alone — unlink it
                            # even on cancel/failure (it's ~10× the source;
                            # leaking one per request adds up fast).
                            if _sep_wav is not None:
                                try:
                                    os.unlink(_sep_wav)
                                except OSError:
                                    pass
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        tmp_path = _vocals_path
                        logger.info("[bgm] separated in %.1fs",
                                    time.perf_counter() - _sep_t0)
                        _stage_timings.append({
                            "name": "separating",
                            "secs": round(time.perf_counter() - _sep_t0, 2),
                            "model": _separation_model or None,
                            "detail": "incl. transcode",
                        })
                    except _bgm.BgmCancelled:
                        raise _ClientCancelled() from None
                    except _bgm.BgmSeparationError as _se:
                        # str(_se) is client-safe by the module's contract.
                        _warnings.append(str(_se))
                    except Exception as _se:  # noqa: BLE001 — soft-fail
                        logger.error("[bgm] unexpected failure: %s",
                                     _log_safe(str(_se)))
                        _warnings.append(
                            "music separation failed; transcribing the "
                            "original audio")

            # Run the synchronous CTranslate2 inference in a thread executor
            # so the event loop stays responsive. CT2 releases the GIL
            # internally, so two concurrent requests on different models can
            # decode in parallel (subject to GPU compute scheduling). The
            # generator returned by transcribe() does its work lazily on
            # iteration, so we materialize it inside the executor too.
            #
            # LEADING_SILENCE_PAD_MS: decode to 16 kHz mono ourselves (the
            # exact call transcribe() makes internally for a path input),
            # prepend silence, and shift the results back to the original
            # timeline. A recording that starts mid-speech at t=0 combined
            # with a hotwords prompt (injected as fake previous-transcript
            # context) makes the decoder drop the opening clause as "already
            # transcribed"; leading silence defuses that. If the pre-decode
            # fails (undecodable bytes, native stack absent), fall back to
            # passing the path so the request fails — or a stubbed model
            # succeeds — exactly as without the feature.
            def _do_transcribe(_model=model, _path=tmp_path,
                               _kw=transcribe_kwargs, _pad_ms=_lead_pad_ms):
                # Materialize the lazy segment generator WITH live progress:
                # each yielded segment carries its end time, and info.duration
                # is known up front — that ratio is genuine decode progress
                # (the executor thread's dict writes are GIL-atomic).
                def _collect(_gen, _info):
                    _dur = float(getattr(_info, "duration", 0.0) or 0.0)
                    _compute, _dev = _model_compute_device(resolved_model)
                    # VAD receipt: transcribe() ran Silero eagerly before
                    # returning, so duration_after_vad is already known here.
                    # Only meaningful when the filter actually ran.
                    _dav = getattr(_info, "duration_after_vad", None)
                    _retained = (
                        max(0.0, min(1.0, float(_dav) / _dur))
                        if _kw.get("vad_filter") and _dur > 0 and _dav is not None
                        else None)
                    _progress_set(_pid, stage="transcribing", progress=0.0,
                                  duration=_dur or None, position=None,
                                  last_text=None, model=resolved_model,
                                  device=_dev, compute=_compute,
                                  vad_retained=_retained)
                    _out = []
                    _log_bucket = 0  # 5%-step INFO trail, like the other stages
                    for _s in _gen:
                        # Cooperative cancel between decoded segments — this
                        # executor thread is the only thing that can stop a
                        # cancelled request's decode.
                        if _cancel_requested(_pid):
                            raise _ClientCancelled()
                        _out.append(_s)
                        if _dur > 0:
                            _frac = min(1.0, float(_s.end) / _dur)
                            _progress_set(
                                _pid,
                                progress=_frac,
                                position=float(_s.end),
                                # Live tail for the client's run panel.
                                last_text=(_s.text or "").strip()[:300] or None)
                            _b = int(_frac * 20)
                            if _b > _log_bucket:
                                _log_bucket = _b
                                logger.info(
                                    "[transcribe] %d%% (%.1fs / %.1fs)",
                                    _b * 5, float(_s.end), _dur)
                    return _out
                # This executor thread has the semaphore slot now: everything
                # until _collect's first entry (lead-pad decode, transcribe()'s
                # eager audio decode + Silero VAD pass) used to be misreported
                # as "waiting". Own stage so the client can label it honestly.
                _progress_set(_pid, stage="analyzing", progress=None,
                              position=None, last_text=None, step=None,
                              model=None, device=None, compute=None)
                _audio = None
                if _pad_ms > 0:
                    try:
                        import numpy as _np
                        from faster_whisper.audio import decode_audio as _fw_decode
                        _audio = _np.concatenate([
                            _np.zeros(_pad_ms * 16, dtype="float32"),  # 16 samples/ms @ 16 kHz
                            _fw_decode(_path, sampling_rate=16000),
                        ])
                    except Exception as _pad_err:
                        logger.warning(
                            "[lead-pad] pre-decode failed, transcribing unpadded: %s",
                            _pad_err)
                        _audio = None
                if _audio is not None:
                    _segs, _info = _model.transcribe(_audio, **_kw)
                    return (*_shift_to_original_timeline(
                        _collect(_segs, _info), _info, _pad_ms / 1000.0), True)
                _segs, _info = _model.transcribe(_path, **_kw)
                return _collect(_segs, _info), _info, False
            loop = asyncio.get_running_loop()
            _progress_set(_pid, stage="waiting", progress=None,
                          position=None, last_text=None, step=None,
                          model=None, device=None, compute=None)
            _check_cancelled(_pid)
            async with get_inference_semaphore():
                _check_cancelled(_pid)
                _dec_t0 = time.perf_counter()
                segments_iter, info, _pad_applied = await loop.run_in_executor(
                    None, _do_transcribe)
                _stage_timings.append({
                    "name": "transcribing",
                    "secs": round(time.perf_counter() - _dec_t0, 2),
                    "model": resolved_model,
                })

            all_words = []
            segments_list = []
            # Compact per-segment metadata for the log block. Separate from
            # segments_list (the API response shape) so we can include it in
            # the diagnostic output without mutating the wire format.
            seg_diag: list[dict] = []
            # We collect raw segment text so the full transcription can be
            # post-processed in ONE pass — multi-word dictation phrases like
            # "neue Zeile" / "neuer Absatz" frequently get split across Whisper's
            # VAD-based segments, and a per-segment pass would never see them
            # together.
            raw_full_text_parts = []

            # Post-decode word-rate guard (SEGMENT_MAX_WORDS_PER_SEC): drops
            # hallucinated echo segments — see segment_exceeds_word_rate.
            _max_wps = float(cfg_for(resolved_model, "SEGMENT_MAX_WORDS_PER_SEC", ident) or 0)

            for i, segment in enumerate(segments_iter):
                # segment.temperature reflects CT2's actual after-fallback
                # value (may differ from the request `temperature` if fallback
                # kicked in). segment.compression_ratio is the real gzip ratio
                # used by the suppression check — was previously hardcoded 1.0.
                seg_temp = getattr(segment, "temperature", temperature)
                seg_cr = getattr(segment, "compression_ratio", 1.0)

                dropped = segment_exceeds_word_rate(segment, _max_wps)
                seg_diag.append({
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "alp": segment.avg_logprob,
                    "nsp": segment.no_speech_prob,
                    "cr": seg_cr,
                    "temp": seg_temp,
                    "text": segment.text,
                    "dropped": dropped,
                })
                if dropped:
                    _dur = float(segment.end) - float(segment.start)
                    logger.info(
                        "[transcribe] dropped word-rate-anomalous segment "
                        "(%.2f-%.2fs, %.1f w/s > %.1f): %r",
                        segment.start, segment.end,
                        (len(getattr(segment, "words", None) or [])
                         or len((segment.text or "").split())) / max(_dur, 1e-6),
                        _max_wps, segment.text)
                    continue

                raw_full_text_parts.append(segment.text)

                # NOTE: segments[].text and words[].word carry RAW Whisper
                # output. Only the joined `text` field below is post-processed.
                # Multi-word dictation phrases ("neue Zeile") frequently get
                # split across VAD segment boundaries, so per-segment post-
                # processing would produce inconsistent results — the joined
                # pass is the authoritative one. Clients that need cleaned
                # per-segment text should read `text` (joined) and split it.
                segments_list.append({
                    "id": len(segments_list),
                    "seek": 0,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": [],
                    "temperature": seg_temp,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": seg_cr,
                    "no_speech_prob": segment.no_speech_prob,
                })

                if getattr(segment, "words", None):
                    for word in segment.words:
                        all_words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        })

            # Post-decode diarization stage (soft-fail): a failure or a
            # disabled server never costs the caller the transcript — it
            # arrives without speaker labels plus a `warnings` entry. The tmp
            # file is still on disk here (the capture path below reads it too;
            # the finally unlinks it after the response is built). Runs under
            # the shared inference semaphore so GPU stages serialize.
            speakers_list: "list[str]" = []
            if _diarize and segments_list:
                if not getattr(cfg, "DIARIZATION_ENABLED", False):
                    _warnings.append(
                        "diarization requested but not enabled on this "
                        "server (DIARIZATION_ENABLED is off)")
                    _skipped.append("diarizing")
                    _progress_set(_pid, skipped=list(_skipped))
                else:
                    # The module itself is import-safe without the optional
                    # deps (pyannote is imported inside the load path).
                    import diarization as _diar
                    try:
                        _diar_t0 = time.perf_counter()
                        _progress_set(
                            _pid, stage="diarizing", progress=None,
                            position=None, last_text=None, step=None,
                            model=(_diarization_model or None),
                            device=_diar._resolve_device())
                        _check_cancelled(_pid)
                        async with get_inference_semaphore():
                            _check_cancelled(_pid)
                            _turns = await _diar.diarize(
                                tmp_path,
                                num_speakers=_spk.get("num_speakers"),
                                min_speakers=_spk.get("min_speakers"),
                                max_speakers=_spk.get("max_speakers"),
                                model_id=(_diarization_model or None),
                                progress_cb=lambda f, step=None: _progress_set(
                                    _pid, progress=f, step=step),
                                cancel_check=lambda: _cancel_requested(_pid),
                            )
                        speakers_list = _diar.assign_speakers(segments_list, _turns)
                        logger.info(
                            "[diarize] %d turns → %d speakers across %d "
                            "segments in %.1fs",
                            len(_turns), len(speakers_list), len(segments_list),
                            time.perf_counter() - _diar_t0)
                        _stage_timings.append({
                            "name": "diarizing",
                            "secs": round(
                                time.perf_counter() - _diar_t0, 2),
                            "model": _diarization_model or None,
                            "detail": f"{len(speakers_list)} speakers",
                        })
                    except _diar.DiarizeCancelled:
                        raise _ClientCancelled() from None
                    except _diar.DiarizationError as _de:
                        # str(_de) is client-safe by the module's contract.
                        _warnings.append(str(_de))
                    except Exception as _de:  # noqa: BLE001 — soft-fail
                        logger.error("[diarize] unexpected failure: %s",
                                     _log_safe(str(_de)))
                        _warnings.append(
                            "diarization failed; the transcript has no "
                            "speaker labels")
            elif _diarize:
                _warnings.append("diarization skipped: no speech segments")

            # Post-decode translation stage (soft-fail). CRITICAL invariant:
            # translated text lives ONLY in seg["translations"] and the
            # top-level `translations`/`translation` response blocks — it must
            # never reach captures, raw_full_text, full_text_str or the
            # quick-config trace, which all carry source-language dictation.
            _translation_meta: "dict | None" = None
            if _translate_to and segments_list:
                if not getattr(cfg, "TRANSLATION_ENABLED", False):
                    _warnings.append(
                        "translation requested but TRANSLATION_ENABLED is "
                        "off on this server")
                    _skipped.append("translating")
                    _progress_set(_pid, skipped=list(_skipped))
                else:
                    # Empty request/config model resolves to the server
                    # default at stage time (a live admin edit applies).
                    _tr_default = (getattr(
                        cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
                    _tr_model = (_translation_model or "").strip() or _tr_default
                    _tr_allowed = getattr(
                        cfg, "TRANSLATION_ALLOWED_MODELS", set()) or set()
                    if (_tr_allowed and _tr_model not in _tr_allowed
                            and _tr_model != _tr_default):
                        # Soft-fail like the enabled gate — never a 4xx after
                        # the transcript already exists.
                        _warnings.append(
                            "requested translation model is not in "
                            "TRANSLATION_ALLOWED_MODELS on this server")
                        _skipped.append("translating")
                        _progress_set(_pid, skipped=list(_skipped))
                    else:
                        try:
                            _tr_t0 = time.perf_counter()
                            _check_cancelled(_pid)
                            _progress_set(
                                _pid, stage="translating", progress=0.0,
                                position=None, last_text=None, step=None,
                                model=(_tr_model or None),
                                device=_tr._resolve_device(),
                                compute="gguf")

                            async def _run_translation():
                                return await _tr.translate_segments(
                                    [{"text": seg["text"],
                                      "speaker": seg.get("speaker")}
                                     for seg in segments_list],
                                    _translate_to,
                                    source_lang=info.language,
                                    model_ref=_tr_model,
                                    mode=_translation_mode,
                                    glossary=_translation_glossary,
                                    context_segments=_translation_context,
                                    # last_text: live tail of the last
                                    # translated line for the run panel —
                                    # only merged when present so a tick
                                    # without one doesn't blank the field.
                                    # stage is re-asserted per tick so the
                                    # first batch flips a cold-download's
                                    # "downloading" back to "translating".
                                    progress_cb=lambda f, step=None,
                                        last_text=None:
                                        _progress_set(
                                            _pid, stage="translating",
                                            progress=f, step=step,
                                            **({"last_text": last_text}
                                               if last_text else {})),
                                    cancel_check=lambda:
                                        _cancel_requested(_pid),
                                    download_cb=lambda done, total:
                                        _progress_set(
                                            _pid, stage="downloading",
                                            progress=((done / total)
                                                      if total else None),
                                            total_bytes=total or None),
                                )
                            # The inference semaphore is held ONLY for GPU
                            # translation: a llama.cpp CPU run can take
                            # minutes, and parking it in a GPU slot would
                            # starve decode/diarization for that long. CPU
                            # translation is serialized by the module's own
                            # _infer_mutex instead.
                            if _tr._resolve_device() == "cuda":
                                async with get_inference_semaphore():
                                    _check_cancelled(_pid)
                                    _per_seg, _tr_warn, _tr_meta = \
                                        await _run_translation()
                            else:
                                _per_seg, _tr_warn, _tr_meta = \
                                    await _run_translation()
                            _warnings.extend(_tr_warn)
                            _tr_kept = _tr_meta.get("kept") or {}
                            for _i, _seg_tr in enumerate(_per_seg):
                                # Unconditional: an untranslated segment
                                # carries an explicit empty map, not a
                                # missing key. translations_kept names the
                                # targets whose guard fallback kept the
                                # SOURCE text (absent when clean).
                                segments_list[_i]["translations"] = _seg_tr
                                if _tr_kept.get(_i):
                                    segments_list[_i]["translations_kept"] = \
                                        list(_tr_kept[_i])
                            _translation_meta = {
                                "model": _tr_meta.get("model"),
                                "targets": list(_translate_to),
                                "source": _tr_meta.get("source"),
                                "mode": _tr_meta.get("mode"),
                            }
                            logger.info(
                                "[translate] %d segments → %s in %.1fs",
                                len(segments_list), ",".join(_translate_to),
                                time.perf_counter() - _tr_t0)
                            _stage_timings.append({
                                "name": "translating",
                                "secs": round(
                                    time.perf_counter() - _tr_t0, 2),
                                "model": _tr_meta.get("model"),
                                "detail": (f"{len(segments_list)} segs → "
                                           f"{','.join(_translate_to)}"),
                            })
                        except _tr.TranslationCancelled:
                            raise _ClientCancelled() from None
                        except _tr.TranslationError as _te:
                            # str(_te) is client-safe by the module's contract.
                            _warnings.append(str(_te))
                        except Exception as _te:  # noqa: BLE001 — soft-fail
                            logger.error("[translate] unexpected failure: %s",
                                         _log_safe(str(_te)))
                            _warnings.append(
                                "translation failed; the transcript is "
                                "untranslated")

            raw_full_text = "".join(raw_full_text_parts)
            trace: "list | None" = [] if cfg.TRACE_ENABLED else None
            full_text_str = _postprocess_text(raw_full_text, model_name=resolved_model, trace=trace, ident=ident)
            # Captures-form text — same pipeline minus the captures-specific
            # exclude set (default-skips `dictation-map` + `capitalize-after-
            # terminator` so the stored text matches Whisper's raw output
            # under SUPPRESS_CHARS for fine-tune training). Only computed when
            # the capture eligibility gate has already passed at handler
            # entry — sampling missed / captures disabled / count cap full
            # are the common case and shouldn't pay for a second pipeline
            # walk per request. No trace participation: the runtime trace
            # describes the user-facing pipeline, not the training-form
            # variant.
            if will_capture:
                training_text_str = _postprocess_text(
                    raw_full_text,
                    model_name=resolved_model,
                    trace=None,
                    extra_excludes=cfg.CAPTURES_PIPELINE_RULES_EXCLUDE,
                    ident=ident,
                )
            # Output wrappers (G/PM): plain prefix/suffix concatenated to
            # the final transcript text after the pipeline runs (including
            # the in-pipeline terminal trim) and BEFORE a defensive
            # post-wrapper trim. Per-model overrides win.
            _output_prefix = cfg_for(resolved_model, "OUTPUT_PREFIX", ident) or ""
            _output_suffix = cfg_for(resolved_model, "OUTPUT_SUFFIX", ident) or ""
            if _output_prefix or _output_suffix:
                _wrap_before = full_text_str
                full_text_str = _output_prefix + full_text_str + _output_suffix
                if trace is not None and _wrap_before != full_text_str:
                    # Trailer step — not a rule card, so no `#N` prefix.
                    trace.append(("output-wrapper",
                                  _wrap_before, full_text_str))
            # Post-wrapper trim — strips whitespace that the wrapper config
            # itself may carry. Runs unconditionally (the per-model exclude
            # only governs the in-pipeline trim). Preserves a leading or
            # trailing "\n" emitted by "neue Zeile" / "neuer Absatz" at the
            # edges of the utterance, since the user explicitly asked for
            # the line break.
            before_trim = full_text_str
            full_text_str = full_text_str.lstrip(" \t\r").rstrip(" \t\r")
            if trace is not None and before_trim != full_text_str:
                # Defensive trim AFTER the output wrappers — distinct from the
                # in-pipeline terminal trim (which already carried `#{card}`),
                # so use a distinct unnumbered label to avoid a duplicate line.
                trace.append(("Trim edges (post-wrapper)",
                              before_trim, full_text_str))

            # request_id was generated at handler entry (so the outer
            # finally can record_timing() on the error path too); it is
            # stamped on the log block (req=<id[:8]> in the title line),
            # on each /reports submission, and on the recent-transcriptions
            # store row for the /quick-config trace panel.

            # Persist the capture if eligibility passed at handler entry
            # AND duration falls in the configured window AND we have
            # enough disk free. Done BEFORE the log block so the block
            # can record `captured=<id_prefix>` for traceability. The
            # tmp_path is still on disk — the finally block unlinks it
            # AFTER this. We copy (not move) so the existing cleanup
            # path is unchanged.
            if will_capture:
                try:
                    import captures_store as _cap_store
                    audio_dur_s = float(getattr(info, "duration", 0.0) or 0.0)
                    min_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MIN_DURATION_SEC", 0.5))
                    max_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MAX_DURATION_SEC", 600.0))
                    if not raw_full_text.strip():
                        # Pure-silence clip: Whisper returned no speech, so the
                        # capture would store as "(empty)" with zero training
                        # value. Skip it. The tmp audio is unlinked by the outer
                        # finally, so nothing is orphaned. raw_full_text is the
                        # exact text that would be passed as raw= below.
                        logger.info(
                            "[capture] skipped empty transcription (no speech) req=%s",
                            request_id[:8],
                        )
                    elif min_s <= audio_dur_s <= max_s:
                        # Disk-free guard. Skip on <1 GB free; don't fail
                        # the transcription. Best-effort: a failure to
                        # query free space (e.g. inaccessible dir) is
                        # treated as "OK to try" and the create_capture
                        # path itself surfaces the real error.
                        try:
                            _free = (await asyncio.to_thread(
                                shutil.disk_usage, cfg.CAPTURES_DIR)).free
                        except OSError:
                            _free = 1 << 40  # large enough to proceed
                        if _free > 1_000_000_000:
                            # OFF the loop: create_capture runs a full PyAV
                            # demux/decode/resample/encode of the uploaded
                            # clip, a multi-MB write and an os.replace retry
                            # loop that time.sleep()s. Measured 1.2-1.4 s of
                            # frozen event loop for a 10-minute upload — the
                            # decode above is already offloaded, this was the
                            # last blocking step left inline. captures_store
                            # opens with check_same_thread=False and guards
                            # writes with its own lock; the outer finally still
                            # unlinks tmp_path after this returns.
                            captured_id = await asyncio.to_thread(
                                functools.partial(
                                    _cap_store.create_capture,
                                    audio_src_path=tmp_path,
                                    request_id=request_id,
                                    model=resolved_model,
                                    language=info.language,
                                    duration_seconds=audio_dur_s,
                                    raw=raw_full_text,
                                    final=full_text_str,
                                    text_for_training=training_text_str,
                                    words=all_words,
                                    segments=seg_diag,
                                    user_id=user.get("user_id"),
                                ))
                        else:
                            logger.warning(
                                "[capture] skipped due to low disk free "
                                "(%.1f MB free, need >1 GB)",
                                _free / (1024 * 1024),
                            )
                    else:
                        logger.info(
                            "[capture] skipped duration filter: %.1fs "
                            "(window %.1f-%.1f)",
                            audio_dur_s, min_s, max_s,
                        )
                except Exception as _ce:
                    logger.warning("[capture] persistence failed: %s", _ce)

            # Always emit the rich diagnostic block — it's how empty-output
            # failures are debugged. The per-pipeline transformation trace
            # is only included when cfg.TRACE_ENABLED is on.
            if source_url is not None:
                # Never the full URL in the log block (query strings carry
                # tokens); the host is enough to correlate, and it still goes
                # through _log_safe like every caller-supplied string.
                import urllib.parse as _uparse
                _url_host = _log_safe(
                    _uparse.urlsplit(source_url).hostname or "?")
                _src_fmt = _log_safe(
                    os.path.splitext(tmp_path or "")[1].lstrip(".") or "audio")
                _file_label = (f"url:{_url_host}  ({audio_bytes/1024:.1f} KB, "
                               f"{_log_safe(response_format)})")
                _audio_src_label = f"{_src_fmt} → 16 kHz mono (url download via yt-dlp"
            else:
                _src_fmt = _log_safe(file.content_type
                                     or os.path.splitext(file.filename or "")[1].lstrip(".")
                                     or "audio")
                _file_label = (f"{_log_safe(file.filename)}  ({audio_bytes/1024:.1f} KB, "
                               f"{_log_safe(response_format)})")
                _audio_src_label = f"{_src_fmt} → 16 kHz mono (file upload"
            logger.info(_format_request_block(
                file_label=_file_label,
                model_name=resolved_model,
                info=info,
                kwargs=transcribe_kwargs,
                seg_diag=seg_diag,
                raw=raw_full_text,
                final=full_text_str,
                steps=trace,
                request_id=request_id,
                captured_id=captured_id,
                endpoint="/v1/audio/transcriptions",
                audio_source=(_audio_src_label
                              + (f"; +{_lead_pad_ms} ms lead pad)"
                                 if _pad_applied else ")")),
                ident=ident,
                overrides_ignored=ignored,
                user_id=user.get("user_id"),
                key_id=user.get("key_id"),
                username=user.get("username"),
                key_label=user.get("key_label"),
                guards={"segment_max_words_per_sec": _max_wps},
                translate_to=(_translation_meta["targets"]
                              if _translation_meta else None),
                translation_model=(_translation_meta["model"]
                                   if _translation_meta else None),
            ))

            # Persist the trace to the durable recent-transcriptions store
            # (SQLite, WAL) and broadcast it to /quick-config SSE
            # subscribers in one step. Lazy import keeps main.py decoupled
            # at module-load time. metrics.record_transcription() in the
            # outer finally adds the timing half via UPSERT on the same
            # request_id.
            try:
                import quick_config_state
                quick_config_state.record_trace(
                    request_id=request_id,
                    model=resolved_model,
                    raw=raw_full_text,
                    steps=trace if trace is not None else [],
                    final=full_text_str,
                    language=info.language,
                    user_id=_user_id,
                )
            except Exception as _qc_err:
                logger.error("[quick-config] record_trace failed: %s", _qc_err)

            _audio_dur = float(info.duration)
            # Word count from the final post-processed text — matches what the
            # client actually receives. Counting len(all_words) instead would
            # yield 0 whenever WORD_TIMESTAMPS_ENABLED is off or the request
            # didn't ask for word-level granularity (the common case).
            _words = len(full_text_str.split())

            if response_format == "text":
                return full_text_str

            if response_format == "verbose_json":
                response = {
                    "task": _task,
                    "language": info.language,
                    "duration": info.duration,
                    "text": full_text_str,
                    "segments": segments_list,
                }
                # VAD receipt (additive): how much audio survived the silence
                # filter — lets the client warn when the filter ate the file.
                # Only when the filter actually ran (absent ⇒ off/unknown).
                _dav = getattr(info, "duration_after_vad", None)
                if transcribe_kwargs.get("vad_filter") and _dav is not None:
                    response["duration_after_vad"] = float(_dav)
                if include_words:
                    response["words"] = all_words
                if speakers_list:
                    response["speakers"] = speakers_list
                if _translation_meta is not None:
                    # Joined per-language transcripts. Deliberately NOT run
                    # through _postprocess_text: the pipeline's rules are
                    # German-dictation-shaped (dictation-map, punctuation
                    # words) and would mangle translated text.
                    response["translations"] = {
                        _lang: " ".join(
                            _s for _s in (
                                (seg.get("translations") or {})
                                .get(_lang, "").strip()
                                for seg in segments_list)
                            if _s).strip()
                        for _lang in _translation_meta["targets"]
                    }
                    response["translation"] = _translation_meta
                # Soft-failed optional stages (diarization) explain themselves
                # here instead of failing the request.
                if _warnings:
                    response["warnings"] = _warnings
                # Surface (never silently drop) any client override the admin
                # config locked out, so the caller can see why it had no effect.
                if ignored:
                    response["overrides_ignored"] = ignored
                # Echo which server profile actually applied (None if the name
                # was unknown or the feature is gated off) — only when asked.
                if override_profile:
                    response["profile_applied"] = ident.request_profile_applied
                # URL flow: where the client can fetch the downloaded audio
                # for local playback, and how long that offer stands.
                # Additive keys — OpenAI-compat callers ignore them.
                if _source_media_id is not None:
                    import url_media_store as _ums
                    response["source_media_id"] = _source_media_id
                    response["source_media_expires_at"] = (
                        _ums.expires_at_unix(_source_media_id))
                return response

            if _source_media_id is not None:
                import url_media_store as _ums
                return {"text": full_text_str,
                        "source_media_id": _source_media_id,
                        "source_media_expires_at":
                            _ums.expires_at_unix(_source_media_id)}
            return {"text": full_text_str}

        except _ClientCancelled:
            # The client asked (via the cancel endpoint) to abort. Not an
            # error — the stages stopped cooperatively; the response status
            # is moot (the caller usually dropped the connection already).
            _status = "cancelled"
            logger.info("[batch] transcription cancelled by client")
            raise HTTPException(status_code=499,
                                detail="cancelled by the client")
        except HTTPException:
            # Preserve curated HTTP errors (e.g. an allowed-models 400) with
            # their status + message intact — only unexpected errors below are
            # genericised.
            _status = "error"
            raise
        except Exception as e:
            _status = "error"
            # Log the raw exception server-side, but return a GENERIC detail to
            # the client: str(e) here can carry model-dir / filesystem (temp)
            # paths (av/ffmpeg decode + model-load errors). Mirrors the
            # streaming WS hardening — never forward raw str(exc) to a caller.
            # _log_safe: str(e) can echo caller-supplied text verbatim (the
            # `language` Form field is an unvalidated str and faster-whisper's
            # tokenizer quotes an unknown code back into its ValueError), and
            # multipart values are binary-safe, so raw CR/LF would otherwise
            # forge extra lines in the /logs viewer.
            logger.error("Transcription error: %s", _log_safe(str(e)))
            raise HTTPException(status_code=500, detail="transcription failed")

        finally:
            if _pid:
                _BATCH_PROGRESS.pop(_pid, None)
                _BATCH_CANCELLED.discard(_pid)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            # URL flow: the private download dir (partials, fragments) goes
            # on every path — cancel, 4xx, 500 included. The retained copy
            # (url_media_store) has its own TTL lifecycle.
            if _url_job_dir:
                shutil.rmtree(_url_job_dir, ignore_errors=True)

    except Exception:
        # Catches failures BEFORE the inner try (e.g. _get_or_load_model
        # raising HTTPException, await request.form() blowing up), which
        # previously bypassed `_status = "error"` and inflated the success
        # counter with failed requests.
        _status = "error"
        raise
    finally:
        if _leased_model is not None:
            _release_model_lease(_leased_model)
        metrics.in_flight_transcriptions -= 1
        if _pid:
            _JOB_BY_PID.pop(_pid, None)
        jobs.job_end(request_id)
        metrics.record_transcription(
            model=resolved_model,
            audio_dur=_audio_dur,
            proc_dur=time.perf_counter() - _t0,
            status=_status,
            words=_words,
            request_id=request_id,
            user_id=_user_id,
            key_id=_key_id,
            stages=_stage_timings or None,
        )


@app.post("/v1/audio/translations")
async def translate_audio(
    request: Request,
    file: "UploadFile | None" = File(None),
    source_url: "str | None" = Form(None),
    model_name: str = Form("whisper-1", alias="model"),
    response_format: str = Form("json"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    prompt: str | None = Form(None),
    decode_overrides: str = Form(None),
    override_profile: str = Form(None),
    diarize: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    diarization_model: str | None = Form(None),
    separate_bgm: str | None = Form(None),
    separation_model: str | None = Form(None),
    translate_to: str | None = Form(None),
    translation_model: str | None = Form(None),
    translation_mode: str | None = Form(None),
    translation_glossary: str | None = Form(None),
    progress_id: str | None = Form(None),
    user: dict = Depends(_get_current_user_dep),
):
    """OpenAI-compatible translation endpoint: the transcription handler with
    `task` pinned to "translate" (into English — Whisper's only target).
    `language` still means the SOURCE language, exactly as on the sibling
    endpoint. A locked TASK still wins inside the handler and reports
    `overrides_ignored: ["task"]`. Distinct from the text-to-text translation
    stage (`translate_to`) and POST /v1/text/translations, which translate a
    finished transcript into arbitrary target languages via GGUF models."""
    return await transcribe(
        request=request,
        file=file,
        source_url=source_url,
        model_name=model_name,
        response_format=response_format,
        language=language,
        temperature=temperature,
        prompt=prompt,
        decode_overrides=decode_overrides,
        override_profile=override_profile,
        task="translate",
        diarize=diarize,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_model=diarization_model,
        separate_bgm=separate_bgm,
        separation_model=separation_model,
        translate_to=translate_to,
        translation_model=translation_model,
        translation_mode=translation_mode,
        translation_glossary=translation_glossary,
        progress_id=progress_id,
        user=user,
    )


# ── Text-to-text translation (translate an existing transcript) ─────────────

# Every accepted request is real llama.cpp inference, possibly minutes of it,
# so the meaningful limit is CONCURRENCY, not rate: what hurts is one client
# holding the model while everyone else waits, and a per-minute counter cannot
# express that (a single request can outlive its own window). The in-flight
# gauge is the protection; the window below is only a backstop against a
# runaway loop that never reaches the gauge because each attempt fails
# validation first.
_translate_inflight = _rl.InFlight(
    config_field="TRANSLATE_MAX_INFLIGHT_PER_USER",
    default_max=2,
    message="you already have {limit} translations running — "
            "wait for one to finish",
)
_text_translate_rate = _rl.FixedWindow(
    config_field="TRANSLATE_RATE_PER_MIN",
    window_s=60.0,
    default_max=120,
    message="too many translation requests — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


# Request-shape ceilings: entry count and total characters. The 4 MiB JSON
# body cap (_max_body_mw) bounds the wire size before either check runs.
_TEXT_TRANSLATE_MAX_SEGMENTS = 2000
_TEXT_TRANSLATE_MAX_CHARS = 200_000

# translate_segments' warning strings name segments by 1-based POSITION
# ("segment 2", "segments 1-3"). On this endpoint clients address segments by
# their own ids — rewrite the references before returning.
_TR_SEG_WARN_RE = re.compile(r"segments (\d+)-(\d+)|segment (\d+)")


def _client_id_warnings(warnings: "list[str]", ids: "list") -> "list[str]":
    """Rewrite positional 1-based segment references in translation warnings
    to the CLIENT-supplied segment ids (a group span expands to the member
    ids, which need not be sequential)."""
    def _sub(m: "re.Match") -> str:
        if m.group(3) is not None:
            i = int(m.group(3)) - 1
            return f"segment {ids[i]}" if 0 <= i < len(ids) else m.group(0)
        a, b = int(m.group(1)) - 1, int(m.group(2)) - 1
        if 0 <= a <= b < len(ids):
            return "segments " + ", ".join(str(ids[j])
                                           for j in range(a, b + 1))
        return m.group(0)
    return [_TR_SEG_WARN_RE.sub(_sub, w) for w in warnings]


@app.post("/v1/text/translations")
async def translate_text(request: Request,
                         user: dict = Depends(_get_current_user_dep)):
    """Translate already-transcribed segments (text→text via GGUF models) —
    the standalone twin of the batch handler's `translate_to` stage, for
    translating a transcript the client already holds without re-uploading
    the audio. Body: {"segments": [{"id", "text", "speaker"?}], "targets":
    [codes], "source"?, "translation_model"?, "translation_mode"?,
    "translation_glossary"?, "context_segments"?, "progress_id"?}. The
    optional progress_id plugs into the same GET progress / POST cancel
    endpoints as a batch transcription. Answers each input segment's id in
    input order with its {target: text} translations."""
    if not getattr(cfg, "TRANSLATION_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="translation is disabled on this server")
    # Cheap and early: the per-minute backstop costs one dict lookup and needs
    # nothing parsed. The in-flight slot is taken much later, right before the
    # work starts — see below.
    _inflight_key = _rl.identity_key(user, request)
    _text_translate_rate.hit(_inflight_key)
    # Canonical job id — stamped on every log line of this run (req=<id8>)
    # so a multi-minute translation can be followed through the log.
    request_id = uuid.uuid4().hex
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body is a caller error
        raise HTTPException(status_code=422, detail="expected a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="expected a JSON object")

    segments = body.get("segments")
    if not isinstance(segments, list) or not segments:
        raise HTTPException(status_code=422,
                            detail="segments must be a non-empty list")
    if len(segments) > _TEXT_TRANSLATE_MAX_SEGMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"segments is capped at {_TEXT_TRANSLATE_MAX_SEGMENTS} "
                   "entries")
    seg_in: "list[dict]" = []
    ids: "list" = []
    total_chars = 0
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
            raise HTTPException(
                status_code=422,
                detail=f"segments[{i}] must be an object with a string "
                       "'text'")
        total_chars += len(seg["text"])
        speaker = seg.get("speaker")
        seg_in.append({"text": seg["text"],
                       "speaker": speaker if isinstance(speaker, str) else None})
        ids.append(seg.get("id", i))
    if total_chars > _TEXT_TRANSLATE_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"segments exceed {_TEXT_TRANSLATE_MAX_CHARS} total "
                   "characters")

    # Per-identity policy (locks + overrides) applies here exactly as on the
    # batch path — reading bare cfg would let a locked-down key bypass its
    # profile by using this endpoint instead of the transcription form.
    ident = build_ident(user, None)
    _ignored: "list[str]" = []

    def _knob(cfg_name: str, client_name: str, body_val):
        if cfg_name in ident.locked:
            inherited = cfg_for(None, cfg_name, ident)
            if body_val is not None and body_val != inherited:
                _ignored.append(client_name)
            return inherited
        if body_val is not None:
            return body_val
        return cfg_for(None, cfg_name, ident)

    raw_targets = body.get("targets")
    max_targets = int(_knob("TRANSLATION_MAX_TARGETS", "", None) or 1)
    if not isinstance(raw_targets, list) or not raw_targets:
        raise HTTPException(status_code=422,
                            detail="targets must be a non-empty list of "
                                   "language codes")
    targets: "list[str]" = []
    for t in raw_targets:
        code = t.strip() if isinstance(t, str) else ""
        if not _TRANSLATE_CODE_RE.match(code):
            raise HTTPException(
                status_code=422,
                detail=f"targets contains an invalid language code: {t!r}")
        if code not in targets:
            targets.append(code)
    if len(targets) > max_targets:
        raise HTTPException(
            status_code=422,
            detail=f"targets is capped at TRANSLATION_MAX_TARGETS "
                   f"({max_targets})")

    source = body.get("source")
    if source is not None and not isinstance(source, str):
        raise HTTPException(status_code=422, detail="source must be a string")
    mode = body.get("translation_mode")
    if mode is not None and mode not in ("fluent", "faithful"):
        raise HTTPException(
            status_code=422,
            detail="translation_mode must be 'fluent' or 'faithful'")
    mode = _knob("TRANSLATION_MODE", "translation_mode", mode) or "fluent"
    glossary = body.get("translation_glossary")
    if glossary is not None and not isinstance(glossary, str):
        raise HTTPException(status_code=422,
                            detail="translation_glossary must be a string")
    glossary = (_knob("TRANSLATION_GLOSSARY", "translation_glossary",
                      glossary) or "")[:4000]
    context_segments = body.get("context_segments")
    if context_segments is not None and not isinstance(context_segments, int):
        raise HTTPException(status_code=422,
                            detail="context_segments must be an integer")
    if context_segments is not None:
        context_segments = min(10, max(0, context_segments))
    _ctx_resolved = _knob("TRANSLATION_CONTEXT_SEGMENTS", "context_segments",
                          context_segments)
    context_segments = int(_ctx_resolved) if _ctx_resolved is not None else None

    model_ref = body.get("translation_model")
    if model_ref is not None and not isinstance(model_ref, str):
        raise HTTPException(status_code=422,
                            detail="translation_model must be a string")
    _tr_default = (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
    _tr_model = (_knob("TRANSLATION_MODEL", "translation_model",
                       (model_ref or "").strip() or None) or "").strip() \
        or _tr_default
    _tr_allowed = getattr(cfg, "TRANSLATION_ALLOWED_MODELS", set()) or set()
    if (_tr_allowed and _tr_model not in _tr_allowed
            and _tr_model != _tr_default):
        raise HTTPException(
            status_code=400,
            detail="requested translation model is not in "
                   "TRANSLATION_ALLOWED_MODELS on this server")

    # Optional progress/cancel plumbing: a valid id joins _BATCH_PROGRESS so
    # the existing GET progress and POST cancel endpoints work unchanged
    # (cancel only accepts ids it can see in flight). Malformed → absent,
    # matching the batch handler's stance.
    progress_id = body.get("progress_id")
    _pid = progress_id if (isinstance(progress_id, str)
                           and _PROGRESS_ID_RE.match(progress_id)) else None

    # ── Canonical job logging ────────────────────────────────────────────
    # Start receipt now, throttled heartbeats from the progress wrapper,
    # and a mirrored terminal line (✓ done / ✗ failed / ✗ cancelled) on
    # every exit path — this endpoint used to log only on cancel/failure.
    _uid = (user.get("user_id") or "")
    logger.info(
        "[translate] req=%s start: %d segments × %d targets (%s) model=%s "
        "mode=%s user=%s",
        request_id[:8], len(seg_in), len(targets), ",".join(targets),
        _tr_model or "?", mode, _uid[:8] or "-")
    _t0 = time.perf_counter()
    # Heartbeat + load-time bookkeeping shared with the progress wrapper.
    # first_cb approximates "model ready" — good enough to split load from
    # infer on the completion line when the model was cold.
    _hb = {"last_log": _t0, "last_pct": 0, "first_cb": None}
    _was_loaded = _tr_model in _tr._models
    # Take the in-flight slot HERE rather than next to the rate check at
    # the top: a dozen `raise HTTPException` validation exits sit between
    # the two, and each one would have to remember to release a slot it
    # never actually used. From this line to the finally there is exactly
    # one path out.
    _translate_inflight.acquire(_inflight_key)
    # Set only AFTER a successful acquire — the acquire itself sits
    # outside the try, so a refused request never releases a slot it does
    # not hold.
    _inflight_held: "str | None" = _inflight_key
    try:
        # Central running-jobs registry entry. Progress feeds in directly from
        # _on_progress below (works whether or not the client sent a progress_id).
        jobs.job_start("translate", id=request_id, model=(_tr_model or None),
                       user=(_uid or None), key=user.get("key_id"),
                       detail=f"{len(seg_in)} segs → {','.join(targets)}",
                       )
        jobs.job_update(request_id, stage="translating", progress_id=_pid)

        def _on_progress(f, step=None, last_text=None):
            now = time.perf_counter()
            if _hb["first_cb"] is None:
                _hb["first_cb"] = now
            pct = int(max(0.0, min(1.0, f or 0.0)) * 100)
            # Log on every crossed 10% boundary, and at least every 30 s.
            if (now - _hb["last_log"] >= 30.0
                    or pct // 10 > _hb["last_pct"] // 10):
                _hb["last_log"] = now
                _hb["last_pct"] = pct
                logger.info("[translate] req=%s %s %d%%",
                            request_id[:8], step or "translating", pct)
            # stage is re-asserted on every tick so the first real batch flips
            # a "downloading" entry (cold model fetch) back to "translating".
            jobs.job_update(request_id, stage="translating", progress=f,
                            step=step)
            fields = {"stage": "translating", "progress": f, "step": step}
            if last_text:
                fields["last_text"] = last_text
            _progress_set(_pid, **fields)

        def _on_download(done, total):
            frac = (done / total) if total else None
            jobs.job_update(request_id, stage="downloading", progress=frac,
                            total_bytes=total or None)
            _progress_set(_pid, stage="downloading", progress=frac,
                          total_bytes=total or None)

        def _record_run(status: str) -> None:
            """Persist this run as a recent-jobs row (kind='translate') on every
            terminal path. No audio duration; segment count lives in the stage
            detail (words_count=0 — a segment count is not a word count)."""
            secs = round(time.perf_counter() - _t0, 3)
            metrics.record_transcription(
                model=(_tr_model or ""),
                audio_dur=0.0,
                proc_dur=secs,
                status=status,
                words=0,
                request_id=request_id,
                user_id=(_uid or None),
                key_id=user.get("key_id"),
                kind="translate",
                stages=[{"name": "translate", "secs": secs,
                         "model": (_tr_model or None),
                         "detail": f"{len(seg_in)} segs → {','.join(targets)}"}],
            )

        _progress_set(_pid, stage="translating", progress=0.0,
                      model=(_tr_model or None),
                      device=_tr._resolve_device(), compute="gguf")
        try:
            _check_cancelled(_pid)

            async def _run_translation():
                return await _tr.translate_segments(
                    seg_in, targets,
                    source_lang=(source or None),
                    model_ref=_tr_model,
                    mode=mode,
                    glossary=glossary,
                    context_segments=context_segments,
                    progress_cb=_on_progress,
                    cancel_check=lambda: _cancel_requested(_pid),
                    download_cb=_on_download,
                )
            # Same policy as the batch stage: the GPU inference semaphore is
            # held only when translation actually runs on cuda — a llama.cpp
            # CPU run must not occupy a GPU slot for its duration.
            if _tr._resolve_device() == "cuda":
                async with get_inference_semaphore():
                    _check_cancelled(_pid)
                    per_seg, warnings, meta = await _run_translation()
            else:
                per_seg, warnings, meta = await _run_translation()
        except _tr.TranslationCancelled:
            raise _ClientCancelled() from None
    except _ClientCancelled:
        logger.info("[translate] req=%s ✗ cancelled after %.1fs",
                    request_id[:8], time.perf_counter() - _t0)
        _record_run("cancelled")
        raise HTTPException(status_code=499, detail="cancelled by the client")
    except _tr.TranslationError as e:
        # str(e) is client-safe by the module's contract.
        logger.info("[translate] req=%s ✗ failed after %.1fs (%s)",
                    request_id[:8], time.perf_counter() - _t0,
                    _log_safe(str(e)))
        _record_run("failed")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — never forward raw errors
        logger.error("[translate] req=%s ✗ failed after %.1fs: %s",
                     request_id[:8], time.perf_counter() - _t0,
                     _log_safe(str(e)))
        _record_run("failed")
        raise HTTPException(status_code=500, detail="translation failed")
    finally:
        # Release FIRST — before any await and before job_end. A client
        # disconnect raises CancelledError, which is a BaseException, so every
        # `except Exception` arm above is skipped and only `finally` runs; a
        # release sitting after an await in here would in turn be skipped if
        # the cancellation landed on that await. That is exactly the bug class
        # documented at streaming_routes.py:1200-1207, where it permanently
        # burned one of STREAMING_MAX_SESSIONS. Nullable local: only a
        # successful acquire sets it, so a refused request releases nothing.
        if _inflight_held is not None:
            _translate_inflight.release(_inflight_held)
            _inflight_held = None
        jobs.job_end(request_id)
        if _pid:
            _BATCH_PROGRESS.pop(_pid, None)
            _BATCH_CANCELLED.discard(_pid)

    _elapsed = time.perf_counter() - _t0
    # Cold model: everything up to the first progress callback is load (the
    # cache layer logs the exact load line too); warm model: all infer.
    _load_s = (max(0.0, _hb["first_cb"] - _t0)
               if (not _was_loaded and _hb["first_cb"] is not None) else 0.0)
    _chars_out = sum(len(t) for d in per_seg for t in d.values())
    logger.info(
        "[translate] req=%s ✓ done in %.1fs (load %.1fs · infer %.1fs) · "
        "%d segs → %s · %d chars in / %d out · %d guard fallbacks",
        request_id[:8], _elapsed, _load_s, max(0.0, _elapsed - _load_s),
        len(seg_in), ",".join(targets), total_chars, _chars_out,
        len(warnings))
    _record_run("ok")

    # kept_original: targets for which the guard fallback returned the SOURCE
    # text — without it a kept German line under translations["en"] is
    # indistinguishable from a real translation. Absent when clean.
    _kept = meta.get("kept") or {}
    return {
        "segments": [{"id": ids[i], "translations": per_seg[i],
                      **({"kept_original": list(_kept[i])}
                         if _kept.get(i) else {})}
                     for i in range(len(ids))],
        "translation": {"model": meta.get("model"), "targets": targets,
                        "source": meta.get("source"), "mode": meta.get("mode")},
        "warnings": _client_id_warnings(warnings, ids) + [
            f"{name} is locked on this server — your value was ignored"
            for name in _ignored if name
        ],
    }


@app.get("/v1/audio/transcriptions/progress/{progress_id}",
         dependencies=[Depends(_get_current_user_dep)])
async def transcription_progress(progress_id: str):
    """Live progress of an in-flight file transcription that was posted with a
    matching `progress_id` form field. Stages: waiting (semaphore queue) →
    [resolving → downloading (URL flow: `progress` 0..1 when the size is
    known, with `total_bytes`)] → separating → analyzing (audio decode +
    VAD, inside transcribe()) → transcribing (with `progress` 0..1 and the
    audio `duration`) → diarizing → translating. A requested-but-declined
    stage lands in `skipped` instead ("separating" / "diarizing" /
    "translating"). An unknown/finished id answers stage "unknown" — the
    POST's own response is the completion signal, so the poller just
    stops."""
    if not _PROGRESS_ID_RE.match(progress_id):
        raise HTTPException(status_code=422, detail="malformed progress_id")
    entry = _BATCH_PROGRESS.get(progress_id)
    if entry is None:
        return {"stage": "unknown"}
    return {
        "stage": entry.get("stage"),
        "progress": entry.get("progress"),
        "duration": entry.get("duration"),
        # Rich run-panel fields (all optional, stage-scoped): seconds of
        # audio decoded, the diarization pipeline's current step, the last
        # decoded segment's text, and the active stage's model/device.
        "position": entry.get("position"),
        "step": entry.get("step"),
        "last_text": entry.get("last_text"),
        "model": entry.get("model"),
        "device": entry.get("device"),
        "compute": entry.get("compute"),
        # Fraction of the audio the VAD kept (0..1), set once decoding starts;
        # null when the filter was off. Persists for the rest of the run.
        "vad_retained": entry.get("vad_retained"),
        # URL flow, downloading stage: bytes expected (progress is the
        # downloaded fraction when this is known; null on fragmented streams).
        "total_bytes": entry.get("total_bytes"),
        # Requested stages this server declined to run (feature disabled) —
        # "separating" / "diarizing" / "translating". Set the moment the skip
        # is known, so the client's rail can say "skipped" instead of guessing.
        "skipped": entry.get("skipped"),
    }


@app.post("/v1/audio/transcriptions/cancel/{progress_id}",
          dependencies=[Depends(_get_current_user_dep)])
async def transcription_cancel(progress_id: str):
    """Abort the in-flight transcription posted with this `progress_id`.

    Closing the upload connection does NOT stop the server-side work (the
    stages run in executor threads that outlive the handler task), so a
    client's Cancel button calls this too. The flag is checked cooperatively
    between demix chunks / decoded segments / pyannote steps, so the abort
    lands within a chunk, not instantly. Only ids currently in flight are
    accepted; an unknown/finished id answers cancelled=false."""
    if not _PROGRESS_ID_RE.match(progress_id):
        raise HTTPException(status_code=422, detail="malformed progress_id")
    if progress_id not in _BATCH_PROGRESS:
        return {"cancelled": False}
    _BATCH_CANCELLED.add(progress_id)
    logger.info("[batch] cancel requested for an in-flight transcription")
    return {"cancelled": True}


# ── Transcribe-from-URL: preview + retained-media endpoints ─────────────────

# Per-identity window for the metadata probe: each preview is a real outbound
# fetch to the linked site, so a keystroke-happy client must not turn the
# server into a probe cannon. This is the only endpoint in the tree whose
# limit protects a THIRD PARTY — the server is the abuse vector and the victim
# is somebody else's infrastructure, which is why the default is far tighter
# than anything else here.
_url_preview_rate = _rl.FixedWindow(
    config_field="URL_PREVIEW_RATE_PER_MIN",
    window_s=60.0,
    default_max=10,
    message="too many link previews — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


def _url_host_for_log(url: str) -> str:
    """Best-effort hostname for log lines — never the full URL (it can carry
    tokens/identifiers we don't want in logs)."""
    try:
        import urllib.parse as _p
        return _log_safe(_p.urlsplit(url.strip()).hostname or "?")
    except Exception:  # noqa: BLE001 — logging must never raise
        return "?"


@app.post("/v1/audio/url-preview")
async def url_preview(request: Request,
                      user: dict = Depends(_get_current_user_dep)):
    """Metadata for a pasted media link, WITHOUT downloading: title,
    duration, uploader, and a server-proxied thumbnail (data: URI — the
    client never talks to the media site itself). Advisory: the client may
    still POST a URL whose preview failed; the download re-checks the same
    policy authoritatively. Client-safe 400s from the policy taxonomy."""
    if not getattr(cfg, "URL_DOWNLOAD_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="URL download is not enabled on this server")
    _url_preview_rate.hit(_rl.identity_key(user, request))
    import url_download as _udl
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body is a caller error
        raise HTTPException(status_code=422, detail="expected a JSON body")
    url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(status_code=422, detail="expected {\"url\": …}")
    _uhost = _url_host_for_log(url)
    logger.info("[url-dl] preview requested (host %s)", _uhost)
    try:
        info = await _udl.probe(
            url, timeout=float(getattr(cfg, "URL_PREVIEW_TIMEOUT_SEC", 20)))
    except _udl.UrlDownloadError as e:
        # str() is client-safe by the module's contract.
        logger.info("[url-dl] preview rejected (host %s): %s",
                    _uhost, _log_safe(str(e)))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — never forward raw errors
        logger.error("[url-dl] preview failed (host %s): %s",
                     _uhost, _log_safe(str(e)))
        raise HTTPException(status_code=500, detail="link preview failed")
    logger.info(
        "[url-dl] preview ok (host %s): extractor=%s duration=%s est_bytes=%s",
        _uhost, info.extractor_key,
        f"{info.duration:.0f}s" if info.duration is not None else "?",
        info.filesize_approx if info.filesize_approx is not None else "?")
    thumb = await _udl.fetch_thumbnail_data_uri(info.thumbnail_url)
    return {
        "title": info.title,
        "duration": info.duration,
        "uploader": info.uploader,
        "extractor": info.extractor_key,
        "estimated_bytes": info.filesize_approx,
        "thumbnail": thumb,
        # The audio format the download would actually fetch (DOWNLOAD_FORMAT
        # selection): container ext + bitrate, for the client's format chip.
        "ext": info.ext,
        "abr": info.abr,
    }


_URL_MEDIA_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


@app.get("/v1/audio/url-media/{media_id}")
async def url_media(media_id: str,
                    user: dict = Depends(_get_current_user_dep)):
    """The retained audio of a finished transcribe-from-URL run, so the
    client can pull ONE local copy for playback. Short-lived (URL_MEDIA_TTL
    _SEC, wiped on restart); unknown, expired and foreign-owner ids all
    answer the same 404 — no oracle. FileResponse handles Range, so the
    client player can seek without re-downloading."""
    from fastapi.responses import FileResponse
    if not getattr(cfg, "URL_DOWNLOAD_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="URL download is not enabled on this server")
    if not _URL_MEDIA_ID_RE.match(media_id):
        raise HTTPException(status_code=422, detail="malformed media id")
    import url_media_store as _ums
    resolved = _ums.resolve(media_id, user_id=user.get("user_id"))
    if resolved is None:
        raise HTTPException(status_code=404, detail="media not found")
    path, ext = resolved
    mime = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
        "oga": "audio/ogg", "opus": "audio/ogg", "flac": "audio/flac",
        "m4a": "audio/mp4", "mp4": "audio/mp4", "aac": "audio/aac",
        "webm": "audio/webm", "mka": "audio/x-matroska",
        "mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")
    return FileResponse(
        path=path,
        media_type=mime,
        filename=f"{media_id}.{ext}",
        # Owner-gated audio: a shared cache must never answer the next,
        # differently-authenticated caller (same stance as captures audio).
        headers={"Cache-Control": "no-store"},
    )


@app.get("/v1/models", dependencies=[Depends(_get_current_user_dep)])
async def list_models():
    """OpenAI-style model listing — currently-loaded models plus the configured
    default. Useful for clients to discover what's available without trial.

    User-tier auth like its /v1 siblings: the payload carries the build version,
    the per-process boot_id and the whole ALLOWED_MODELS list, so it is a
    fingerprint, not public metadata."""
    now = int(time.time())
    names: list[str] = list(_loaded_models.keys())
    if cfg.DEFAULT_MODEL not in names:
        names.append(cfg.DEFAULT_MODEL)
    if cfg.ALLOWED_MODELS:
        for n in sorted(cfg.ALLOWED_MODELS):
            if n not in names:
                names.append(n)
    return {
        "object": "list",
        "boot_id": BOOT_ID,
        # Build identity (non-standard, like boot_id): lets clients show
        # "faster-whisper-backend · v0.1.0" instead of a generic detection tag.
        "server_name": SERVER_NAME,
        "server_version": APP_VERSION,
        "data": [
            {
                "id": n,
                "object": "model",
                "created": now,
                "owned_by": "local",
                "loaded": n in _loaded_models,
            }
            for n in names
        ],
    }


@app.get("/v1/me")
async def whoami_capabilities(user: dict = Depends(_get_current_user_dep)):
    """The caller's effective request-override capabilities — drives the client
    UI (hide the decode editor / override-profile picker it isn't permitted to
    use). The client UI is convenience only; the server enforces everything in
    effective_config.resolve regardless. User-tier auth: any valid key; 401
    without one when the server is locked down.

    Returns: {can_request_override_profile, can_request_decode_overrides,
    allowed_override_profiles: ["*"] | [names…] | [], vad_filter_default}."""
    import effective_config
    caps = effective_config.resolve_capabilities(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    # Additive: the server-wide VAD default, so the client's "Default" segment
    # on its Skip-silence control can say which way inherit points. Server-wide
    # (not per-model/identity) — it labels a ghost, it doesn't gate anything.
    caps["vad_filter_default"] = bool(getattr(cfg, "VAD_FILTER", True))
    # Additive: whether the optional pipeline stages exist on this server at
    # all, so the client can disable its "Separate music" / "Speaker
    # diarization" toggles pre-flight instead of letting a request soft-fail
    # into a warning. Server-wide feature switches, not per-identity grants.
    caps["bgm_separation_enabled"] = bool(
        getattr(cfg, "BGM_SEPARATION_ENABLED", False))
    caps["diarization_enabled"] = bool(
        getattr(cfg, "DIARIZATION_ENABLED", False))
    # Additive: transcribe-from-URL capacity switch + the installed yt-dlp
    # version (or null). The version is deliberately visible to user-tier
    # callers: "is the downloader stale?" is the first question when a site
    # stops working, and clients surface it in their error guidance.
    caps["url_download_enabled"] = bool(
        getattr(cfg, "URL_DOWNLOAD_ENABLED", False))
    if caps["url_download_enabled"]:
        import url_download as _udl
        caps["yt_dlp_version"] = _udl.yt_dlp_version()
    # Additive: text-to-text translation capability surface. The flag is
    # always present (pre-flight for the client's Translate control); the
    # detail keys ride only when the stage exists — same shape discipline as
    # yt_dlp_version above.
    caps["translation_enabled"] = bool(
        getattr(cfg, "TRANSLATION_ENABLED", False))
    if caps["translation_enabled"]:
        import translation as _tr
        _t_default = (getattr(cfg, "TRANSLATION_DEFAULT_MODEL", "") or "").strip()
        _t_refs: "list[str]" = [_t_default] if _t_default else []
        for _ref in sorted(set(getattr(cfg, "TRANSLATION_ALLOWED_MODELS", None)
                               or set()) | set(_tr._models)):
            if _ref not in _t_refs:
                _t_refs.append(_ref)
        caps["translation_models"] = [
            {"id": _ref, "loaded": _ref in _tr._models} for _ref in _t_refs]
        # Language menu for the client's target picker. resolve_family("")
        # is exception-free: no configured pin → detect_family("") → the
        # generic chatml family, whose list is the shared code set anyway.
        caps["translation_languages"] = _tr.list_languages(
            _tr.resolve_family(_t_default))
        # The CALLER's effective TRANSLATE_TO default (per-identity overrides
        # respected — unlike vad_filter_default above, which is a server-wide
        # ghost label), parsed csv → list like the transcribe handler does.
        _ident = build_ident(user, None)
        _tt_raw = cfg_for(None, "TRANSLATE_TO", _ident) or ""
        _tt_list: "list[str]" = []
        for _code in _tt_raw.split(","):
            _code = _code.strip()
            if (_code and _code not in _tt_list
                    and _TRANSLATE_CODE_RE.match(_code)):
                _tt_list.append(_code)
        caps["translate_to_default"] = _tt_list
        # Engine version, yt_dlp_version-style best-effort (null when the
        # optional dependency set isn't installed).
        try:
            import importlib.metadata
            caps["llama_cpp_version"] = importlib.metadata.version(
                "llama-cpp-python")
        except Exception:  # noqa: BLE001 — absence is a supported state
            caps["llama_cpp_version"] = None
    # Additive: the stage-model allowlists with a loaded flag, mirroring
    # translation_models — the client's model pickers pre-flight on these.
    # "Loaded" = the module's single cached instance is exactly this model
    # (both stages cache one pipeline/separator at a time).
    import bgm_separation as _bgm
    import diarization as _diar
    _diar_loaded = (_diar._pipeline_key[0]
                    if _diar._pipeline_key else None)
    caps["diarization_models"] = [
        {"id": _m, "loaded": _m == _diar_loaded}
        for _m in (getattr(cfg, "DIARIZATION_ALLOWED_MODELS", None) or [])]
    # The separator caches by on-disk FILENAME (".onnx" implied), the
    # allowlist holds friendly names — compare through the same mapping.
    _sep_loaded = (_bgm._separator_key[0]
                   if _bgm._separator_key else None)
    caps["separation_models"] = [
        {"id": _m,
         "loaded": (_m if "." in _m else f"{_m}.onnx") == _sep_loaded}
        for _m in (getattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", None) or [])]
    return caps


@app.get("/v1/override-profiles")
async def list_override_profiles(user: dict = Depends(_get_current_user_dep)):
    """Names of the server-side OVERRIDE_PROFILES THIS caller may reference via the
    per-request `override_profile` field — filtered by the global gate, the
    caller's per-identity gate + allowlist, and each profile's `requestable` flag.
    Names only — never the profile contents; empty list when the caller may not
    request any. User-tier auth: any valid key (admin not required); 401 without
    one when the server is locked down."""
    import effective_config
    names = effective_config.allowed_profile_names(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    return {"profiles": names}


@app.get("/v1/override-profiles/{name}")
async def get_override_profile(name: str,
                               user: dict = Depends(_get_current_user_dep)):
    """The decode-relevant values + locked client keys of a single override-
    profile THIS caller may request — for the client to preview as inherited
    defaults. 404 when the profile doesn't exist OR the caller may not request it
    (don't leak internal / disallowed profiles). The returned `values` are the
    profile's OWN contribution projected to the client decode keys; admin locks
    elsewhere can still win at request time (reported then via overrides_ignored).
    User-tier auth."""
    import effective_config
    allowed = effective_config.allowed_profile_names(
        user_id=user.get("user_id"), key_id=user.get("key_id"))
    if name not in allowed:
        raise HTTPException(status_code=404, detail="override-profile not found")
    profiles = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
    blob = profiles.get(name)
    values, locked = effective_config.project_profile_to_client(blob)
    # `prompt` is exposed SEPARATELY (not in `values`, which is exactly the 19
    # client decode keys): the client's "Vocabulary / prompt" maps to the server's
    # DEFAULT_PROMPT, which has no client decode key, so the editor needs it here to
    # ghost the profile's prompt as an inherited default.
    prompt = None
    prompt_locked = False
    if isinstance(blob, dict):
        _p = blob.get("DEFAULT_PROMPT")
        if isinstance(_p, str) and _p:
            prompt = _p
        prompt_locked = "DEFAULT_PROMPT" in (blob.get("locks") or [])
    return {"name": name, "values": values, "locked": locked,
            "prompt": prompt, "prompt_locked": prompt_locked}


# =============================================================================
# /logs - live log viewer
# =============================================================================
# A self-contained dark-theme log tailer. Loads recent context from the log
# file, then streams new lines via Server-Sent Events. Color is reapplied
# client-side based on content (since we strip ANSI before writing the file).
import io
from fastapi.responses import HTMLResponse, StreamingResponse

def _read_tail(path: str, n: int) -> list[str]:
    """Return the last n lines of `path` (or fewer if the file is shorter)."""
    if not os.path.exists(path):
        return []
    # Read from the end in chunks to avoid loading huge files into memory.
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read = min(block, size)
            size -= read
            f.seek(size)
            data = f.read(read) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


def _rotated_chain(active_path: str) -> list[str]:
    """Newest→oldest list of paths in the rotation chain: the active log
    followed by .1, .2, … up to LOG_BACKUP_COUNT. Files that don't exist
    are silently skipped — rotation may have produced fewer backups than
    the configured count, and a freshly-deployed service starts with
    just the active file."""
    out = [active_path]
    for i in range(1, int(getattr(cfg, "LOG_BACKUP_COUNT", 10)) + 1):
        p = f"{active_path}.{i}"
        if os.path.exists(p):
            out.append(p)
    return out


# How deep the "Load older" cursor may go, in pages of LOG_VIEWER_INITIAL_LINES.
# Past this the reader walks the whole chain for a window no browser is still
# holding, so the cursor is clamped rather than served.
_LOG_OLDER_MAX_PAGES = 500


def _read_chain_window(active_path: str, skip: int, want: int) -> "tuple[list[str], int | None]":
    """Read up to `want` lines from the rotation chain (newest→oldest),
    starting `skip` lines back from the chain head. Returns
    (lines_oldest_first, next_skip).

    next_skip is None when the chain has no more older content — either
    we returned a partial page (fewer than `want` lines) or we exactly
    reached the chain tail. Otherwise next_skip == skip + len(lines)
    and the caller may re-call with that value to fetch the next
    older window.

    Walks the chain newest-file first (active log → .1 → .2 → …),
    reading each file backward in 8 KB blocks until we've accumulated
    `skip + want` lines across the chain. One file is held in memory
    at a time; ~10 MB worst case for the default LOG_MAX_BYTES."""
    target = skip + want
    # `collected` is built in oldest→newest order: each older file's
    # tail is prepended to the running list as we walk the chain
    # newest→oldest. By construction the OLDEST line in the chain
    # window we've seen so far sits at collected[0].
    collected: list[str] = []
    chain = _rotated_chain(active_path)
    # When we break out of the file loop because `target` is satisfied
    # without opening every file, older rotated files still on disk
    # remain unread — `exhausted` must account for that or the caller
    # ("Load older" UI) loses access to anything beyond the first file
    # whenever its line count meets `target` exactly.
    more_files_after_break = False
    for i, path in enumerate(chain):
        try:
            with open(path, "rb") as f:
                f.seek(0, io.SEEK_END)
                size = f.tell()
                block = 8192
                # Blocks are appended newest-last and joined once at the end:
                # prepending to a bytes object re-copies (and re-counts) the
                # whole buffer on every 8 KB step, which is quadratic in the
                # file size once `need` is large.
                blocks: list[bytes] = []
                newlines = 0
                need = target - len(collected)
                while size > 0 and newlines <= need:
                    read = min(block, size)
                    size -= read
                    f.seek(size)
                    buf = f.read(read)
                    newlines += buf.count(b"\n")
                    blocks.append(buf)
                data = b"".join(reversed(blocks))
        except OSError:
            continue
        collected = data.decode("utf-8", errors="replace").splitlines() + collected
        if len(collected) >= target:
            more_files_after_break = (i + 1) < len(chain)
            break
    # Slice in newest-first frame so `skip` is unambiguous.
    newest_first = list(reversed(collected))
    window = newest_first[skip:skip + want]
    exhausted = ((skip + len(window)) >= len(newest_first)
                 and not more_files_after_break)
    next_skip = None if exhausted else skip + len(window)
    return list(reversed(window)), next_skip


async def _stream_log_lines():
    """Yield SSE events: one for each existing tail line, then live tail."""
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    # Off the loop, same as /logs/older: this walks the rotation chain
    # backwards in 8 KB blocks and an async generator inside a
    # StreamingResponse runs on the event loop, so doing it inline let one
    # subscriber stall every other request while it read.
    backlog, _ = await asyncio.to_thread(
        _read_chain_window, cfg.LOG_FILE, 0, initial,
    )
    for line in backlog:
        yield f"data: {line}\n\n"

    # Sentinel — marks the boundary between backlog and the live poll
    # loop. The client's append() early-returns on this line; pill counts
    # are driven entirely by SEV_POLLER_JS against severity_counts().
    yield "data: __LIVE_TAIL__\n\n"

    # Live tail: open at end-of-file, poll for new lines. Reopen on rotation
    # (when the file shrinks below our last position).
    pos = os.path.getsize(cfg.LOG_FILE) if os.path.exists(cfg.LOG_FILE) else 0
    while True:
        await asyncio.sleep(0.5)
        try:
            size = os.path.getsize(cfg.LOG_FILE)
        except OSError:
            yield ": waiting-for-file\n\n"
            continue
        if size < pos:
            pos = 0  # rotated
        if size == pos:
            yield ": keepalive\n\n"
            continue
        # Also off the loop — this runs every 0.5 s for the lifetime of every
        # subscriber, and the delta after a burst can be megabytes.
        def _read_delta(start: int) -> "tuple[str, int]":
            with open(cfg.LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(start)
                return f.read(), f.tell()

        chunk, pos = await asyncio.to_thread(_read_delta, pos)
        for line in chunk.splitlines():
            yield f"data: {line}\n\n"


_LOG_VIEWER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<script>(function(){var v=localStorage.getItem('whisper-log-zoom');
  if(v)document.documentElement.style.setProperty('--log-zoom',v);})();</script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d;
  }
  /* Font tokens, --font-sans, --font-mono and html font-size live in
     NAV_CSS (injected further down). Important: never embed the NAV_CSS
     template placeholder inside another comment block — render_page() does
     a naive string replace and would inject NAV_CSS into this comment,
     prematurely closing it (NAV_CSS contains its own internal comments)
     and silently dropping every CSS rule that follows. Header chrome
     (title, pills, buttons) gets --font-sans by default; the log lines
     themselves opt into --font-mono via .line so timestamps and tabular
     fields stay aligned. */
  html { height: 100%; }
  body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills, the #filter input) are all centralized in NAV_CSS. */
  /* width:100% + box-sizing:border-box are the fix for the "tiny centered
     column with text clipped at the start" rendering — without them the
     container sits on a content-sized box that overflows the viewport.
     No max-width: log content uses the full viewport so long lines (HF
     URLs, model paths) sit on a single line on wide monitors instead of
     wrapping into the empty side-bands. The header bar stays centered at
     68.75rem (its own .header-inner cap) so controls remain in a predictable
     spot. pre-wrap still wraps lines that genuinely exceed the viewport.
     font-size = global rem * --log-zoom is the multiplicative log-only
     zoom; bumping the global picker grows logs and chrome together, and
     the [-]/[+] buttons in the header then scale logs only on top. */
  #log { padding: 0.5rem 0.875rem;
    width: 100%;
    box-sizing: border-box;
    font-family: var(--font-mono);
    font-size: calc(1rem * var(--log-zoom, 1));
    white-space: pre-wrap; overflow-wrap: anywhere;
    overflow-anchor: none; }
  .line { display: block; word-break: break-word; }
  /* Log-zoom control — independent from the global UI scale picker. */
  .log-zoom { display: inline-flex; align-items: center; gap: 0.25rem;
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.125rem 0.25rem; flex-shrink: 0;
    font-size: var(--fs-xs); }
  .log-zoom button { background: transparent; border: none; color: var(--fg);
    cursor: pointer; padding: 0 0.375rem; line-height: 1;
    font-size: var(--fs-md); font-family: var(--font-mono); }
  .log-zoom button:hover:not(:disabled) { color: var(--cyan); }
  .log-zoom button:disabled { opacity: 0.35; cursor: not-allowed; }
  .log-zoom #log-zoom-pct { color: var(--dim); font-variant-numeric: tabular-nums;
    min-width: 2.75rem; text-align: center; }
  #load-older-row { display: flex; justify-content: center;
    padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
  #loadOlderBtn { background: transparent; color: var(--cyan);
    border: 1px solid var(--border); padding: 0.375rem 1rem;
    border-radius: 4px; font: inherit; cursor: pointer; }
  #loadOlderBtn:hover:not([disabled]) { background: var(--panel); }
  #loadOlderBtn[disabled] { opacity: 0.5; cursor: default; }
  .tz-hint { color: var(--dim); font-size: var(--fs-xs); cursor: help; }
  .line.hidden { display: none; }
  .line.rule    { color: var(--dim); }
  .line.title   { color: var(--bold); font-weight: 600; }
  .line.meta    { color: var(--cyan); }
  .line.raw     { color: var(--bold); }
  .line.step    { color: var(--cyan); }
  .line.before  { color: var(--dim); }
  .line.after   { color: var(--green); }
  .line.final   { color: var(--green); font-weight: 600; }
  .line.warning { color: var(--yellow); }
  .line.error   { color: var(--red); }
  .line.info    { color: var(--fg); }
  /* Dimmed pipeline no-ops — a rule force-EXCLUDED for this model or globally
     disabled (SKIPPED). Mirrors the /quick-config skipped-step treatment;
     opacity (not a color) so the label/before/after keep their hue, just faded. */
  .line.dim     { opacity: 0.55; }
  {{NAV_CSS}}
</style></head>
<body>
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>{{HEADER_VTAG}}
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">Logs</span>
    <div class="subbar-left">
      <input id="filter" type="text" placeholder="filter (case-insensitive substring)…">
    </div>
    <div class="subbar-right">
      <span class="log-zoom" title="zoom log content only">
        <button id="log-zoom-out" type="button" aria-label="decrease log size">−</button>
        <span id="log-zoom-pct">100%</span>
        <button id="log-zoom-in" type="button" aria-label="increase log size">+</button>
      </span>
      <span class="tz-hint" title="log timestamps are stored in UTC and shown here in your browser's local timezone">local time</span>
      <button id="pauseBtn">pause</button>
      <button id="clearBtn">clear</button>
      <span id="status" class="pill live">live</span>
    </div>
  </div>
</header>
<div id="load-older-row">
  <button id="loadOlderBtn" type="button" style="display:none;">Load older</button>
</div>
<div id="log"></div>
<script>
  const log = document.getElementById('log');
  const statusEl = document.getElementById('status');
  const filterEl = document.getElementById('filter');
  const pauseBtn = document.getElementById('pauseBtn');
  const clearBtn = document.getElementById('clearBtn');
  let paused = false;
  let filterText = '';

  // Honor ?filter=... so the severity pills in the nav can deep-link.
  const initialFilter = new URLSearchParams(location.search).get('filter');
  if (initialFilter) {
    filterEl.value = initialFilter;
    filterText = initialFilter.toLowerCase();
  }

  function classify(line) {
    if (/^═+$/.test(line.trim()) || /^─+$/.test(line.trim())) return 'rule';
    if (/\\/v1\\/audio\\/transcriptions/.test(line)) return 'title';
    if (/RAW WHISPER/.test(line)) return 'raw';
    if (/FINAL\\s+'/.test(line)) return 'final';
    // Step label: "▸ #1.1 …" / "▸ #18 …" (the leading # arrived with the
    // card-position numbering) — and the older bare "▸ 8 …" rotated format.
    if (/▸\\s+#?\\d/.test(line)) return 'step';
    if (/^\\s*→\\s/.test(line)) return 'after';
    if (/file=|lang=|duration=|segments=|words=|format=/.test(line)) return 'meta';
    if (/(WARNING|WARN)/.test(line)) return 'warning';
    if (/(ERROR|CRITICAL)/.test(line)) return 'error';
    if (/^\\s+'.*'$/.test(line)) return 'before';
    return 'info';
  }
  // Pipeline steps that didn't run — force-EXCLUDED for this model, or globally
  // disabled (SKIPPED) — are logged as a 3-line group: a ▸ label carrying the
  // marker, then an identical before/after pair. Dim the whole group (same as
  // the /quick-config + /reports trace viewers) so the eye skips the no-ops.
  // `st.dimLeft` is threaded per render pass (live stream vs "Load older" batch
  // each get their own state object) so groups never dim across a boundary.
  const _SKIP_MARK = /\\[(EXCLUDED|SKIPPED)/;
  function decorate(line, st) {
    const cls = classify(line);
    if (cls === 'step') {
      if (_SKIP_MARK.test(line)) { st.dimLeft = 2; return cls + ' dim'; }
      st.dimLeft = 0; return cls;            // a step that ran resets the group
    }
    if (cls === 'rule' || cls === 'title' || cls === 'raw' || cls === 'final') {
      st.dimLeft = 0; return cls;            // section boundary — stop dimming
    }
    if (st.dimLeft > 0) { st.dimLeft--; return cls + ' dim'; }
    return cls;
  }
  function applyFilter(el) {
    if (filterText && !el.textContent.toLowerCase().includes(filterText)) {
      el.classList.add('hidden');
    } else {
      el.classList.remove('hidden');
    }
  }
  function _p2(n) { return (n < 10 ? '0' : '') + n; }
  // Log lines start with a UTC ISO-8601 'Z' timestamp; show it in the reader's
  // local time as 'YYYY-MM-DD HH:MM:SS'. Lines that don't start with that token
  // (continuation/traceback lines, or pre-UTC rotated lines) pass through as-is.
  function localizeLogTs(line) {
    const m = line.match(/^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z)\\s/);
    if (!m) return line;
    const d = new Date(m[1]);
    if (isNaN(d.getTime())) return line;
    const local = d.getFullYear() + '-' + _p2(d.getMonth() + 1) + '-' + _p2(d.getDate())
      + ' ' + _p2(d.getHours()) + ':' + _p2(d.getMinutes()) + ':' + _p2(d.getSeconds());
    return local + line.slice(m[1].length);
  }
  // DOM cap for live-tail appends only. The "Load older" path bypasses
  // this so the user can scroll back through arbitrarily many rotated
  // lines without their click silently dropping the freshest content.
  const _LOG_DOM_MAX = {{LOG_VIEWER_DOM_MAX}};
  // Persists across append() calls so a skipped step's 3 lines (arriving as 3
  // separate SSE events) dim as one group.
  const _liveDim = { dimLeft: 0 };
  function append(line) {
    // __LIVE_TAIL__ sentinel marks the boundary between backlog and the
    // live poll loop. After it fires we know the freshest page is in the
    // DOM and the "Load older" button can become active.
    if (line === '__LIVE_TAIL__') {
      const lo = document.getElementById('loadOlderBtn');
      if (lo) lo.style.display = '';
      return;
    }
    const el = document.createElement('span');
    el.className = 'line ' + decorate(line, _liveDim);
    el.textContent = localizeLogTs(line) + '\\n';
    applyFilter(el);
    log.appendChild(el);
    while (log.childElementCount > _LOG_DOM_MAX) log.firstChild.remove();
    if (!paused) window.scrollTo(0, document.body.scrollHeight);
  }

  filterEl.addEventListener('input', () => {
    filterText = filterEl.value.toLowerCase();
    for (const el of log.children) applyFilter(el);
  });
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'resume' : 'pause';
    statusEl.textContent = paused ? 'paused' : 'live';
    statusEl.className = 'pill ' + (paused ? 'paused' : 'live');
    if (!paused) window.scrollTo(0, document.body.scrollHeight);
  });
  clearBtn.addEventListener('click', () => {
    log.innerHTML = '';
    // Resetting the DOM after Clear: the next live append re-fills
    // from the bottom, but the older-chain pointer is unchanged so
    // the user can still walk back if they want.
  });

  // "Load older" cursor: how many lines from the chain head are already
  // in the DOM. Seeded with the initial backlog size; each successful
  // /logs/older response bumps it by the returned-batch length.
  let _logsSkip = {{LOG_VIEWER_INITIAL_LINES}};
  let _logsOlderBusy = false;
  const loadOlderBtn = document.getElementById('loadOlderBtn');
  if (loadOlderBtn) {
    loadOlderBtn.addEventListener('click', async () => {
      if (_logsOlderBusy) return;
      _logsOlderBusy = true;
      loadOlderBtn.disabled = true;
      const prevLabel = loadOlderBtn.textContent;
      loadOlderBtn.textContent = 'Loading…';
      try {
        // Session cookie is sent automatically with the fetch.
        const url = '/logs/older?skip=' + encodeURIComponent(_logsSkip)
                  + '&limit={{LOG_VIEWER_INITIAL_LINES}}';
        const r = await fetch(url);
        if (!r.ok) {
          console.warn('load-older failed', r.status);
          return;
        }
        const j = await r.json();
        const lines = (j && j.lines) || [];
        // Prepend to top of #log, preserving line order (oldest-first
        // batch → first inserted ends up at the very top, last inserted
        // sits just above the existing content). Filter is reapplied per
        // line so the new batch honors any active substring search.
        const frag = document.createDocumentFragment();
        const olderDim = { dimLeft: 0 };   // batch-local; no leak to live tail
        for (const line of lines) {
          const el = document.createElement('span');
          el.className = 'line ' + decorate(line, olderDim);
          el.textContent = localizeLogTs(line) + '\\n';
          applyFilter(el);
          frag.appendChild(el);
        }
        log.insertBefore(frag, log.firstChild);
        _logsSkip += lines.length;
        if (j.next_skip == null) loadOlderBtn.style.display = 'none';
      } catch (e) {
        console.warn('load-older error', e);
      } finally {
        _logsOlderBusy = false;
        loadOlderBtn.disabled = false;
        loadOlderBtn.textContent = prevLabel;
      }
    });
  }

  // EventSource sends the HttpOnly session cookie automatically (same-origin),
  // so the server's _require_logs_page_sse dependency resolves the user
  // without the legacy ?key= fallback.
  let es = null;
  let _logRecoveryTimer = null;
  function openLogStream() {
    if (es) { try { es.close(); } catch (_) {} es = null; }
    es = new EventSource('/logs/stream');
    es.onmessage = (e) => append(e.data);
    es.onerror = () => {
      statusEl.textContent = 'reconnecting…';
      statusEl.className = 'pill paused';
      // EventSource does NOT auto-reconnect after an HTTP error (e.g. an
      // intermittent 401 where the cookie wasn't attached to the SSE
      // handshake). Mirror /stats: poll a cheap endpoint until it 200s,
      // then reopen the stream.
      // Back off on repeated failures (3s → ×1.7 → cap 30s).
      if (_logRecoveryTimer) return;
      let delay = 3000;
      const probe = async () => {
        try {
          const r = await fetch('/v1/models', { cache: 'no-store' });
          if (r.ok) {
            clearTimeout(_logRecoveryTimer);
            _logRecoveryTimer = null;
            openLogStream();
            return;
          }
        } catch (_) { /* keep polling */ }
        delay = Math.min(delay * 1.7, 30000);
        _logRecoveryTimer = setTimeout(probe, delay);
      };
      _logRecoveryTimer = setTimeout(probe, delay);
    };
    es.onopen = () => {
      // role-admin used to be added here unconditionally — that leaked
      // admin chrome to non-admins. OPEN_MODE_BANNER_JS is now the single
      // source of truth (sets role-admin iff whoami.is_admin=true).
      if (_logRecoveryTimer) { clearTimeout(_logRecoveryTimer); _logRecoveryTimer = null; }
      if (!paused) {
        statusEl.textContent = 'live';
        statusEl.className = 'pill live';
      }
    };
  }
  openLogStream();

  // --- Log-only zoom (independent of the global UI scale picker) ---------
  // Multiplies on top of --fs-base via #log { font-size: calc(1rem * --log-zoom) }.
  // Discrete steps so clicks "snap" to recognizable sizes like browser zoom.
  (function(){
    const KEY='whisper-log-zoom';
    const STEPS=[0.7, 0.85, 1, 1.2, 1.4, 1.6, 1.8, 2.0];
    const minus=document.getElementById('log-zoom-out');
    const plus =document.getElementById('log-zoom-in');
    const pct  =document.getElementById('log-zoom-pct');
    if(!minus||!plus||!pct) return;
    function nearestIdx(v){
      let best=2, dist=Infinity;
      STEPS.forEach((s,i)=>{ const d=Math.abs(s-v); if(d<dist){dist=d;best=i;} });
      return best;
    }
    let idx = nearestIdx(parseFloat(localStorage.getItem(KEY)) || 1);
    function apply(){
      const v = STEPS[idx];
      document.documentElement.style.setProperty('--log-zoom', v);
      pct.textContent = Math.round(v*100) + '%';
      minus.disabled = idx === 0;
      plus.disabled  = idx === STEPS.length - 1;
      localStorage.setItem(KEY, v);
    }
    minus.addEventListener('click', () => { if(idx>0){idx--; apply();} });
    plus .addEventListener('click', () => { if(idx<STEPS.length-1){idx++; apply();} });
    apply();
  })();
</script>
{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
</body></html>"""


def _require_logs_page_sse(request: Request) -> dict:
    """SSE-aware variant of require_page("logs"). Two credential carriers:
    the `Authorization: Bearer` header and the HttpOnly session cookie,
    which EventSource sends automatically on a same-origin stream —
    EventSource cannot set a header. In OPEN mode (no admin key yet) the
    synthetic admin sails through from the admin host allowlist only
    (auth.open_mode_host_ok); in locked-down mode the credential must
    resolve to a user with scope("logs") == "all" — the log file isn't
    user-partitionable (a single request block carries every user's
    transcripts, filenames, and final text via _format_request_block), so
    "own" can't be enforced line-by-line and is rejected as access-only at
    the schema layer."""
    import api_keys_store
    if not api_keys_store.is_locked_down() and _open_mode_host_ok(request):
        return dict(api_keys_store.OPEN_MODE_USER)
    auth_header = request.headers.get("authorization") or ""
    raw = ""
    if auth_header.lower().startswith("bearer "):
        raw = auth_header.split(" ", 1)[1].strip()
    rec = api_keys_store.lookup_by_raw_key(raw) if raw else None
    if rec is None:
        rec = _user_from_session_cookie(request)
    if rec is None:
        raise HTTPException(
            401, "invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    perms = Permissions(
        rec.get("permissions_raw") or {}, bool(rec.get("is_admin")),
    )
    # scope("logs") (not can("logs")) — defends against legacy DB rows
    # still storing "own" from before logs joined ACCESS_ONLY_PAGES.
    if perms.scope("logs") != "all":
        raise HTTPException(403, "no access to /logs")
    return rec


@app.get("/logs", response_class=HTMLResponse, dependencies=[Depends(require_user_webui_host)])
async def logs_viewer():
    # User-tier page. Shell gated by USER_WEBUI_ALLOWED_HOSTS (loopback always);
    # a keyless browser navigation loads the shell + login popup. The SSE
    # /logs/stream + /logs/older endpoints stack the host gate with their own
    # require_page("logs") check (bearer header or session cookie — EventSource
    # sends the cookie), so the data layer requires a "logs" API key.
    import web_common
    return HTMLResponse(
        web_common.render_page(_LOG_VIEWER_HTML, current="logs"),
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/logs/stream",
    dependencies=[Depends(require_user_webui_host), Depends(_require_logs_page_sse)],
)
async def logs_stream():
    import web_common
    return web_common.sse_response(_stream_log_lines())


@app.get(
    "/logs/older",
    dependencies=[Depends(require_user_webui_host), Depends(_require_logs_page_sse)],
)
async def logs_older(skip: int = 0, limit: int = 0):
    """Fetch the next older page from the rotation chain. `skip` is the
    number of lines from the chain head that have already been loaded
    into the browser DOM; `limit` defaults to LOG_VIEWER_INITIAL_LINES
    and is server-clamped to the same value (per-click max page size).
    `skip` is clamped to _LOG_OLDER_MAX_PAGES pages of that size.

    Response: `{lines: [...], next_skip: <int|null>}`. lines are
    oldest-first so the client can prepend them to the top of the
    log container as a contiguous older window. next_skip=null means
    the rotation chain is exhausted — the client hides the button."""
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    want = max(1, min(limit or initial, initial))
    skip = max(0, min(int(skip), initial * _LOG_OLDER_MAX_PAGES))
    # _read_chain_window does blocking disk I/O over the whole rotation
    # chain — run it off the event loop so one deep page can't stall the
    # live /logs/stream tail or any concurrent request.
    lines, next_skip = await asyncio.to_thread(
        _read_chain_window, cfg.LOG_FILE, skip=skip, want=want)
    return {"lines": lines, "next_skip": next_skip}


@app.get("/auth/whoami")
async def whoami(
    request: Request,
    user: dict = Depends(_get_current_user_dep),
):
    """Resolve the caller to a user payload the WebUI uses to render the
    login modal + user-aware chrome.

    Returns `{open_mode, user_id, username, is_admin, permissions, build,
    csrf_token?}`. `build` carries the header chip's facts (version, boot,
    start) — the shared header ships as an empty shell because its pages are
    only host-gated, so the facts ride this authenticated route instead. The
    `permissions` object is `{pages: {logs:
    'own'|'all'|'none', ...}}` — used by each page's JS to hide nav links
    the user can't reach and to render scope hints. `csrf_token` is
    present only for cookie-authenticated callers (set by
    user_from_session_cookie on request.state) so the client can attach
    X-CSRF-Token without parsing the cookie. A 401 means no valid
    credential AND the server is locked down — the WebUI re-prompts."""
    import api_keys_store as _ak
    import build_info
    perms = user.get("permissions")
    out = {
        "open_mode": not _ak.is_locked_down(),
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "is_admin": bool(user.get("is_admin")),
        "permissions": perms.to_dict() if perms is not None else {"pages": {}},
        "build": {
            "server": build_info.SERVER_NAME,
            "version": build_info.APP_VERSION,
            "version_short": build_info.VERSION_SHORT,
            "boot": build_info.BOOT_ID[:8],
            "started": build_info.STARTED_UTC,
        },
    }
    csrf = getattr(request.state, "session_csrf", None)
    if csrf:
        out["csrf_token"] = csrf
    # Per-identity, and it carries the session's CSRF token. A 200 GET with no
    # Cache-Control and no Vary is heuristically cacheable under RFC 9111, and
    # this deployment expects a reverse proxy in front (TRUSTED_ORIGINS exists
    # for exactly that) — a shared cache could otherwise hand one user's
    # identity payload, admin flag and CSRF token to the next caller. The page
    # JS already sends cache:'no-store', but that binds only the browser's own
    # cache, not an intermediary's.
    return _JSONResponse(out, headers={"Cache-Control": "no-store"})


# Keyed by client HOST, not identity: a login attempt has no identity yet, and
# the key it presents is exactly what must not be trusted. Only FAILURES are
# counted and a success clears the window, so an operator fat-fingering one
# paste never walks toward a lockout.
_login_failures = _rl.FixedWindow(
    config_field="LOGIN_FAILURE_RATE",
    window_s=60.0,
    default_max=10,
    message="too many failed sign-ins ({limit}/min) — "
            "retry in {retry_after}s",
)


@app.post("/auth/login")
async def login(request: Request, response: Response):
    """Exchange a pasted API key for an HttpOnly session cookie.

    Open mode → no-op (everyone is already the synthetic admin). Locked
    down → validate the key via api_keys_store, create a server-side
    session, and set two cookies: the HttpOnly session token and a
    JS-readable CSRF token (double-submit). Returns the same shape as
    /auth/whoami so the client can populate chrome without a second
    round-trip. CSRF-exempt (no session exists yet)."""
    import api_keys_store as _ak
    import sessions_store
    if not _ak.is_locked_down():
        return {"open_mode": True}
    # Below the open-mode short-circuit on purpose: open mode checks no
    # credential, so there is nothing to throttle, and locking an operator out
    # of an already-unlocked box would be absurd.
    host = request.client.host if request.client else ""
    _login_failures.guard(host)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed/empty body → treat as no key
        body = {}
    key = body.get("key") if isinstance(body, dict) else None
    rec = _ak.lookup_by_raw_key(key or "")
    if rec is None:
        # NEVER log the attempted key — it is a credential, right or wrong.
        if _login_failures.penalize(host):
            logger.info(
                "[auth] login failure limit (LOGIN_FAILURE_RATE=%d) reached "
                "for host %s", _login_failures.limit(), _log_safe(host))
        raise HTTPException(
            401, "invalid API key", headers={"WWW-Authenticate": "Bearer"},
        )
    # Clear BEFORE create_session: a session-store failure below is the
    # server's problem, and must not leave the host carrying penalties for a
    # credential that was in fact correct.
    _login_failures.reset(host)
    raw_token, csrf_token = sessions_store.create_session(
        rec["user_id"], cfg.SESSION_TTL_SECONDS, key_id=rec.get("key_id"),
    )
    ttl = int(cfg.SESSION_TTL_SECONDS)
    secure = bool(cfg.SESSION_COOKIE_SECURE)
    response.set_cookie(
        cfg.SESSION_COOKIE_NAME, raw_token, max_age=ttl,
        httponly=True, samesite="lax", secure=secure, path="/",
    )
    response.set_cookie(
        cfg.SESSION_CSRF_COOKIE_NAME, csrf_token, max_age=ttl,
        httponly=False, samesite="lax", secure=secure, path="/",
    )
    perms = Permissions(rec.get("permissions_raw") or {}, bool(rec.get("is_admin")))
    return {
        "open_mode": False,
        "csrf_token": csrf_token,
        "user_id": rec.get("user_id"),
        "username": rec.get("username"),
        "is_admin": bool(rec.get("is_admin")),
        "permissions": perms.to_dict(),
    }


@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Revoke the current session and clear its cookies. CSRF-protected
    like any other cookie-authenticated mutation (the WebUI sends the
    X-CSRF-Token header)."""
    import sessions_store
    raw = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
    if raw:
        sessions_store.revoke_session(raw)
    response.delete_cookie(cfg.SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(cfg.SESSION_CSRF_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get(
    "/sev",
    dependencies=[Depends(require_user_webui_host), Depends(_get_current_user_dep)],
)
async def severity_snapshot():
    """Tiny JSON endpoint polled by every page's nav-row pill poller.

    Returns the same `severity_counts()` the server uses everywhere else
    (nav HTML render, /stats payload) — WARNING+ records since process
    start, bounded by the 2000-entry ring. Three integers, no PII.

    User-tier: USER_WEBUI_ALLOWED_HOSTS (loopback always) AND any
    authenticated user (`_get_current_user_dep` — in OPEN mode the synthetic
    admin passes, so the pill works before lockdown). The default-open user
    allowlist covers every page that embeds SEV_POLLER_JS — /stats, /logs,
    /settings, … — so the pill keeps live-updating wherever it's shown. A
    403/401 here fails the poller silently; the nav still shows the
    server-rendered count."""
    import web_common
    return web_common.severity_counts()


# =============================================================================
# /stats - system overview dashboard (always on, user-tier allowlist-gated)
# =============================================================================
# Always registered. The route's host gate reads cfg.USER_WEBUI_ALLOWED_HOSTS
# at request time, so the admin UI can broaden/narrow access without a service
# restart. Loopback is always allowed; the data endpoints require a "stats" key.
try:
    from stats_routes import router as _stats_router
    app.include_router(_stats_router)
    logger.info(
        "Stats dashboard at /stats (allowlist=%s; loopback always permitted)",
        cfg.USER_WEBUI_ALLOWED_HOSTS,
    )
except Exception as _e:
    logger.error("Failed to load stats router: %s", _e)


# =============================================================================
# / - landing hub (always on, user-tier allowlist-gated)
# =============================================================================
# The WebUI's front door: signed-out visitors get the shared login gate,
# signed-in ones a launcher filtered to the pages their key can reach (plus
# the admin section for admins). Same host tier as the other user page
# shells; nothing sensitive is rendered server-side. See home_routes.py.
try:
    from home_routes import router as _home_router
    app.include_router(_home_router)
    logger.info(
        "Landing hub at / (allowlist=%s; loopback always permitted)",
        cfg.USER_WEBUI_ALLOWED_HOSTS,
    )
except Exception as _e:
    logger.error("Failed to load home router: %s", _e)


# =============================================================================
# /v1/audio/transcriptions/stream - live (streaming) dictation WebSocket
# =============================================================================
# Always registered; the handler self-gates on cfg.STREAMING_ENABLED (toggleable
# at runtime) and resolves auth per connection (same user records as the batch
# route). Reuses the model cache + _postprocess_text; see streaming_routes.py.
try:
    from streaming_routes import router as _streaming_router
    app.include_router(_streaming_router)
    logger.info(
        "Streaming transcription at /v1/audio/transcriptions/stream "
        "(enabled=%s, max_sessions=%s)",
        getattr(cfg, "STREAMING_ENABLED", True),
        getattr(cfg, "STREAMING_MAX_SESSIONS", 10),
    )
except Exception as _e:
    logger.error("Failed to load streaming router: %s", _e)


# =============================================================================
# /v1/pipeline-rules - client API for the desktop "Dictionary" editor
# =============================================================================
# Always registered (unlike the /quick-config WebUI, which rides
# ADMIN_UI_ENABLED). Same tag/exposed gating + per-type field allow-list +
# validation as /quick-config (shared build_visible_rules / apply_rules_patch in
# quick_config_routes), but in the /v1 namespace with NO host allowlist — auth is
# the per-user API key (bearer) plus the quick_config page permission. Lets the
# desktop client view + edit the post-processing rules the caller is permitted to.
try:
    from quick_config_routes import v1_router as _pipeline_v1_router
    app.include_router(_pipeline_v1_router)
    logger.info("Pipeline-rules client API at GET/PATCH /v1/pipeline-rules")
except Exception as _e:
    logger.error("Failed to load pipeline-rules v1 router: %s", _e)


# =============================================================================
# /v1/client-settings - desktop-client settings sync
# =============================================================================
# Always registered (a route-level 404 must keep meaning "backend build too
# old for sync"). User-tier bearer auth only — deliberately NO page gate and
# NO host allowlist: settings sync is account infrastructure for remote
# desktop clients (same rationale as /v1/usage). One opaque blob per account
# with optimistic versioning; see client_settings_routes.py.
try:
    from client_settings_routes import router as _client_settings_router
    app.include_router(_client_settings_router)
    logger.info(
        "Client-settings sync at GET/PUT/DELETE /v1/client-settings"
    )
except Exception as _e:
    logger.error("Failed to load client-settings router: %s", _e)


# =============================================================================
# /settings - admin WebUI (opt-in)
# =============================================================================
# Off by default: registered only when cfg.ADMIN_UI_ENABLED is True (set in
# config.py or via WHISPER_ADMIN_UI=1). Auth on the endpoints themselves is
# per-user API keys (require_admin) layered on top of cfg.ADMIN_WEBUI_ALLOWED_HOSTS.
# In OPEN mode (no admin key in DB) every caller is the synthetic admin so the
# operator can bootstrap.
if cfg.ADMIN_UI_ENABLED:
    try:
        from admin_routes import router as _admin_router
        app.include_router(_admin_router)
        # /settings/api-keys — admin UI for per-user key management. Same
        # auth shape (admin host + admin key) as /settings.
        from api_keys_routes import router as _api_keys_router
        app.include_router(_api_keys_router)
        # /settings/overrides — admin UI for layered per-identity config
        # profiles + the effective-config Explorer. Same auth shape as /settings.
        from overrides_routes import router as _overrides_router
        app.include_router(_overrides_router)
        logger.info(
            "Admin UI enabled at /settings (allowlist=%s; auth=API key)",
            cfg.ADMIN_WEBUI_ALLOWED_HOSTS,
        )
        # /quick-config is a user-tier page (USER_WEBUI_ALLOWED_HOSTS) with
        # per-user API key auth; it just rides the same ADMIN_UI_ENABLED switch.
        from quick_config_routes import router as _quick_router
        app.include_router(_quick_router)
        logger.info("Quick-config UI enabled at /quick-config")
        # /reports: admin-only triage page for user-submitted transcription
        # error reports. The submission endpoint /quick-config/reports/api/submit
        # lives on the same router and accepts any active API key.
        from reports_routes import router as _reports_router
        app.include_router(_reports_router)
        logger.info(
            "Reports UI enabled at /reports (admin key required for triage; "
            "user submissions %s)",
            "enabled" if getattr(cfg, "REPORTS_ALLOW_USER_SUBMIT", True)
            else "disabled",
        )
        # /captures: admin-only Whisper fine-tuning data capture + review.
        # Master switch is cfg.CAPTURE_RECORDINGS_ENABLED — the page is
        # always registered so the admin can browse existing rows even
        # after disabling new capture.
        from captures_routes import router as _captures_router
        app.include_router(_captures_router)
        logger.info(
            "Captures UI enabled at /captures (admin token required; "
            "new capture %s)",
            "enabled" if getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)
            else "disabled",
        )
    except Exception as _e:
        logger.error("Failed to load admin router: %s", _e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=cfg.SERVER_HOST,
                port=cfg.SERVER_PORT,
                workers=cfg.SERVER_WORKERS,
                log_level=cfg.SERVER_LOG_LEVEL,
                # WS keepalive: a live decode no longer blocks the receive loop, so
                # pings stay answered; a generous timeout tolerates a momentary
                # stall. `or None` lets an admin disable either knob with 0.
                ws_ping_interval=getattr(cfg, "STREAMING_WS_PING_INTERVAL_SEC", 20.0) or None,
                ws_ping_timeout=getattr(cfg, "STREAMING_WS_PING_TIMEOUT_SEC", 60.0) or None,
                # Per-message ceiling on the streaming socket. MAX_REQUEST_BYTES
                # covers HTTP only — its middleware is registered http-only and
                # never sees a websocket scope — so without this the effective
                # limit is the websocket library's 16 MiB default. Real clients
                # send ~32 KB audio frames and a JSON handshake far under 1 MiB.
                ws_max_size=1024 * 1024)
