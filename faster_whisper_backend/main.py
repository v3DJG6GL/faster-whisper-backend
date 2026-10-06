"""The FastAPI application: process setup (logging, the Windows CUDA-DLL and
ffmpeg preload), the lifespan (store init, preload, retention and sampler
loops), middleware (CORS, CSRF, body cap, metrics, security headers), the
OpenAI-compatible transcription / translation routes, and the router
includes for every sub-package. Launched via the root main.py shim or
``python -m faster_whisper_backend``.
"""
import asyncio
import os
import json
import random
import sys
import ctypes
import functools
import importlib
import logging
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

# BOOT_ID (the per-process restart marker surfaced via /v1/models) lives in
# build_info with the rest of the server identity; imported this early —
# before config — so it exists exactly as soon as it used to.
from faster_whisper_backend.build_info import APP_VERSION, BOOT_ID, SERVER_NAME
from faster_whisper_backend.transcription import decode_trace as _decode_trace
from faster_whisper_backend.transcription import segment_guards
from faster_whisper_backend.core.languages import language_codes

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.settings import schema as settings_schema
# system_stats imports psutil + pynvml at module load and primes psutil's
# non-blocking counters. Imported here (early) so the priming happens before
# any request handler runs.
from faster_whisper_backend.runtime import system_stats
# log_safe (below); module-level cost is nil (os).
from faster_whisper_backend.core import store_common
# Text-to-text translation stage (llama.cpp GGUF). Module-level import is
# deliberate and cheap — like diarization/bgm_separation the module is
# import-safe without its optional deps (llama_cpp loads lazily inside the
# model-load path), and the stage + lifespan both need it.
from faster_whisper_backend.translation import engine as _tr
from faster_whisper_backend.translation import gating as tr_gating
# Shared per-identity limiters. Imports only stdlib + fastapi + config, so it
# is safe this early and cannot close an import cycle back through main.
from faster_whisper_backend.auth import rate_limit as _rl
# The media-id regex (MEDIA_ID_RE / MEDIA_ID_PATTERN) is defined once there.
# Imports only stdlib + config + store_common.
from faster_whisper_backend.media import media_store as url_media_store

# Logging: root handlers (stderr console, rotating owner-only file, severity
# ring) — see core/log_setup.py. Installed here, before the heavy imports below,
# so their import-time log lines already reach the file.
from faster_whisper_backend.core import log_setup
from faster_whisper_backend.paths import REPO_ROOT

log_setup.install()

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
# Live edits: pipeline.apply.apply_hot_changes re-syncs the env var whenever
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
    ff_bin = os.path.join(REPO_ROOT, "ffmpeg", "bin")
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


from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends

# Auth dep used by /v1/audio/transcriptions. In open mode (no admin key in DB)
# it returns the synthetic admin — but only to callers on
# ADMIN_WEBUI_ALLOWED_HOSTS — so the operator can bootstrap; otherwise it 401s
# on missing/invalid bearer.
from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
# The API-keys store: opened (fatally) by the lifespan, which also ingests
# WHISPER_BOOTSTRAP_ADMIN_KEY through it. Already imported by auth.dependencies.
from faster_whisper_backend.auth import api_keys_store
# Request-origin gates: the host allowlist tiers (the /docs shells) and the
# same-origin guard _csrf_mw runs on every unsafe method — see auth/hosts.py.
from faster_whisper_backend.auth import hosts as auth_hosts
# The startup TMPDIR sweep for what a hard restart orphaned (stdlib-only).
from faster_whisper_backend.admin import restart_service

# Text post-processing rules engine (cfg.PIPELINE_RULES). Imported here, after
# log_setup.install(), because its import compiles the rules and logs any bad
# regex — see pipeline/engine.py.
from faster_whisper_backend.pipeline import engine as pl_engine


# The per-request receipt, the post-decode guard helpers and the whisper model
# cache / decode-kwargs assembly — see transcription/. Called through the module
# attribute (tx_models._get_or_load_model(...)) so a test patch reaches main too.
from faster_whisper_backend.transcription import guards as tx_guards
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import receipt as tx_receipt

# The stores the lifespan opens and sweeps, the optional stages (import-safe
# without pyannote / audio-separator, which load inside their own load paths),
# the link-download + media-retention helpers and the 1 Hz sampler. None of
# them imports main, so they load at module level like the rest.
from faster_whisper_backend.audio import bgm_separation as _bgm_separation
from faster_whisper_backend.audio import diarization as _diarization
from faster_whisper_backend.audio import transcode as _transcode
from faster_whisper_backend.auth import dependencies as _auth
from faster_whisper_backend.auth import sessions_store
from faster_whisper_backend.captures import samples_store as capture_samples_store
from faster_whisper_backend.captures import store as captures_store
from faster_whisper_backend.client_settings import store as client_settings_store
from faster_whisper_backend.media import download as url_download
from faster_whisper_backend.media import subtitle_mux as _subtitle_mux
from faster_whisper_backend.quick_config import recent_feed as qc_recent_feed
from faster_whisper_backend.reports import store as reports_store
from faster_whisper_backend.stats import recent_transcriptions_store
from faster_whisper_backend.stats import sampler as _stats_sampler
from faster_whisper_backend.stats import system_metrics_store
from faster_whisper_backend.stats import usage_store

# Single implementation lives in store_common so the stores can sanitise their
# own audit lines without importing main (which imports them).
_log_safe = store_common.log_safe


async def _reports_retention_loop() -> None:
    """Hourly retention sweep for the reports store. The sweep reads
    cfg.REPORTS_RETENTION_DAYS each tick so admin /settings edits take
    effect on the next cycle without a service restart. Cancellation
    on shutdown is the normal exit path."""
    while True:
        try:
            await asyncio.sleep(3600)
            # Blocking SQLite + unlinks; off the loop like _sessions_purge_loop
            # (the store serialises writers behind its own lock).
            await asyncio.to_thread(reports_store.sweep_retention)
        except asyncio.CancelledError:
            raise
        except Exception as _re:
            logger.error("[reports] retention loop error: %s", _re)


async def _captures_retention_loop() -> None:
    """Hourly retention sweep for the captures store. Same shape as the
    reports loop; the sweep reads cfg.CAPTURES_RETENTION_DAYS each tick."""
    while True:
        try:
            await asyncio.sleep(3600)
            # A backlog sweep (retention day boundary, lowered
            # CAPTURES_RETENTION_DAYS) dissolves samples and unlinks up to
            # two files per row — off the loop, never blocking in-flight
            # transcriptions, dictation frames or SSE ticks.
            await asyncio.to_thread(captures_store.sweep_retention)
        except asyncio.CancelledError:
            raise
        except Exception as _ce:
            logger.error("[captures] retention loop error: %s", _ce)


async def _jobs_retention_loop() -> None:
    """Hourly retention sweep for the server-jobs store (TTL, row cap, byte
    cap). Same shape as the captures loop; the store reads cfg.JOBS_* live
    each tick."""
    while True:
        try:
            await asyncio.sleep(3600)
            await asyncio.to_thread(_jobs_store.sweep_retention)
        except asyncio.CancelledError:
            raise
        except Exception as _je:
            logger.error("[jobs] retention loop error: %s", _je)


async def _usage_retention_loop() -> None:
    """Hourly sweep for the usage-statistics store: closes out dictation
    sessions whose outcome never arrived and prunes the job/app rows past
    their retention. Same shape as the reports loop; the four knobs are
    read from cfg on every tick so a /settings edit applies without a
    restart."""
    while True:
        try:
            await asyncio.sleep(3600)
            counts = await asyncio.to_thread(
                usage_store.sweep,
                unreported_after_h=int(getattr(cfg, "USAGE_UNREPORTED_AFTER_H", 24)),
                jobs_retention_days=int(getattr(cfg, "USAGE_JOBS_RETENTION_DAYS", 365)),
                app_retention_days=int(getattr(cfg, "USAGE_APP_RETENTION_DAYS", 90)),
                hourly_retention_days=int(getattr(cfg, "USAGE_RETENTION_DAYS", 0)),
            )
            if any(counts.values()):
                logger.info("[usage] sweep: %s", counts)
        except asyncio.CancelledError:
            raise
        except Exception as _ue:
            logger.error("[usage] retention loop error: %s", _ue)


async def _sessions_purge_loop() -> None:
    """Hourly reap of revoked/expired session rows.

    Nothing else purges them at runtime: the lazy eviction in lookup_session
    only fires for a token that is presented again, so with the 30-day sliding
    TTL a login loop grew both the sessions table and the in-memory index
    without bound. /auth/login takes any valid key and has no rate limit.
    Same shape as the retention loops above."""
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


async def _preload_extras() -> None:
    """Best-effort startup preloads for the optional stages (translation
    GGUFs, the diarization pipeline, the BGM separator). Every failure logs
    and continues — the model then loads on first use; the server always
    starts. Called from lifespan after the whisper preload loop."""
    if getattr(cfg, "TRANSLATION_ENABLED", False):
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
            # Same allowlist semantics as the request path (the shared
            # helper); requested=ref because a preload entry is an explicit
            # ask for exactly that ref, never an inherited fallback.
            if not tr_gating._translation_model_allowed(ref, requested=ref):
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
        try:
            logger.info("Preloading the diarization pipeline")
            await _diarization._get_pipeline()
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.error("Failed to preload the diarization pipeline: %s", e)
    if getattr(cfg, "BGM_SEPARATION_PRELOAD", False) and \
            getattr(cfg, "BGM_SEPARATION_ENABLED", False):
        try:
            logger.info("Preloading the BGM separation model")
            await _bgm_separation._get_separator()
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.error("Failed to preload the separation model: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s %s starting (boot %s)", SERVER_NAME, APP_VERSION, BOOT_ID[:8])

    # Open the durable recent-transcriptions store. Replaces the legacy
    # in-memory ring buffers (quick_config_state.recent_traces +
    # metrics.recent_tx) so the /quick-config trace panel + /stats
    # dashboard widget survive service restart and scale beyond 20 rows.
    # Opened BEFORE the preload loop: every startup download (whisper,
    # translation GGUF, pyannote) persists its 'download' recent-jobs row
    # through this store, and with it still closed those rows were lost
    # with a "persist failed" warning. Needs only cfg + store_common.
    try:
        recent_transcriptions_store.init_db(cfg.RECENT_TRANSCRIPTIONS_DB)
        logger.info(
            "Recent-transcriptions store initialized at %s",
            cfg.RECENT_TRANSCRIPTIONS_DB,
        )
    except Exception as _te:
        logger.error("Failed to initialize recent-transcriptions store: %s", _te)

    # Open the system-metrics store (the /stats history charts and the
    # GPU-busy share). Rolling telemetry, its own file. Adopts the
    # pre-split sys_samples rows out of the recent-transcriptions DB once.
    try:
        system_metrics_store.init_db(cfg.STATS_SYSTEM_METRICS_DB)
        # The one-shot legacy copy is its own best-effort step: the store is
        # open and serving by now, so a missing recent-transcriptions
        # connection must not be reported as a store-init failure.
        try:
            _moved = system_metrics_store.adopt_legacy(
                recent_transcriptions_store._require_conn())
            if _moved:
                logger.info("Moved %d legacy sys_samples rows into %s",
                            _moved, cfg.STATS_SYSTEM_METRICS_DB)
        except Exception as _ae:  # noqa: BLE001 — best-effort
            logger.warning("Legacy sys_samples adoption skipped: %s", _ae)
        logger.info("System-metrics store initialized at %s",
                    cfg.STATS_SYSTEM_METRICS_DB)
    except Exception as _se:
        logger.error("Failed to initialize system-metrics store: %s", _se)

    # Open the durable usage-rollup store. Backs the per-key/per-user usage
    # numbers on /api-keys, the usage-over-time section on /stats and the
    # desktop app's statistics (/v1/usage). Non-fatal. Its sweep task is
    # started further down, after the fatal api-keys init (no long-lived
    # task may exist before that point) and only when the store opened —
    # without a connection it would log the same error every hour.
    usage_store_ready = False
    try:
        usage_store.init_db(cfg.USAGE_DB)
        logger.info("Usage rollup store initialized at %s", cfg.USAGE_DB)
        usage_store_ready = True
    except Exception as _ue:
        logger.error("Failed to initialize usage store: %s", _ue)

    # Open the API-keys SQLite store and start the open-mode warning loop.
    # Placed BEFORE the preload loop and before any long-lived task is
    # created: its failure is fatal, and @asynccontextmanager never reaches
    # the post-yield teardown when startup raises, so anything already
    # running would be orphaned — and burning a cold-start multi-GB model
    # download before refusing to start would turn one misconfiguration into
    # a multi-minute crash loop under restart=always.
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
        api_keys_store.init_db(cfg.API_KEYS_DB)
        bootstrap_key = getattr(cfg, "BOOTSTRAP_ADMIN_KEY", None)
        if bootstrap_key:
            # Only inserts if hash isn't already in api_keys. Idempotent.
            api_keys_store.bootstrap_admin_from_env(bootstrap_key)
        logger.info(
            "API keys store initialized at %s (locked_down=%s)",
            cfg.API_KEYS_DB, api_keys_store.is_locked_down(),
        )
        open_mode_task = asyncio.create_task(
            _auth.open_mode_warning_loop()
        )
    except api_keys_store.BootstrapAdminError as _be:
        # Its own wording names the real cause (a revoked/raced env key);
        # the store-unavailable text below would send the operator to check
        # filesystem permissions for a problem that has nothing to do with
        # them.
        logger.critical("[auth] %s", _be)
        raise
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
            await tx_models._get_or_load_model(name)
        except Exception as e:
            logger.error("Failed to preload model '%s': %s", name, e)

    evictor_task = asyncio.create_task(tx_models._idle_evictor())
    # The diarization pipeline gets its own idle unloader (module-local
    # singleton, DIARIZATION_IDLE_TIMEOUT_S read live). pyannote itself loads
    # lazily on first use.
    diarization_evictor_task = asyncio.create_task(
        _diarization.idle_evictor_loop())
    bgm_evictor_task = asyncio.create_task(
        _bgm_separation.idle_evictor_loop())
    # The translation LRU gets its own idle unloader too (module-level
    # import at the top of the file; TRANSLATION_IDLE_TIMEOUT_S read live).
    translation_evictor_task = asyncio.create_task(
        _tr.idle_evictor_loop())
    # Model preloading: bind the worker to THIS loop (on_stage_start hands
    # work to it from executor threads) and run the plan sweeper on the same
    # cadence shape as the four evictors above.
    await preload.start()
    preload_sweeper_task = asyncio.create_task(preload.sweeper_loop())
    receipt_sweeper_task = asyncio.create_task(tx_receipt._receipt_sweeper())

    # Best-effort preloads for the optional stages (translation GGUFs,
    # diarization pipeline, BGM separator).
    await _preload_extras()

    # Transcribe-from-URL retention: wipe the dir (ids die with the process,
    # so every surviving file is an orphan) and start the TTL/byte-cap
    # janitor. Unconditional — URL_DOWNLOAD_ENABLED is a LIVE toggle (no
    # restart badge), so the janitor must already be running when an admin
    # flips it on; with the feature off both are no-ops. The TMPDIR sweep
    # reclaims what a hard restart (restart_service) orphaned.
    url_media_store.startup_reset()
    restart_service.reclaim_hard_restart_orphans()
    url_media_janitor_task = asyncio.create_task(
        url_media_store.janitor_loop())
    # Subtitle packaging: probe ffmpeg's muxers/encoders ONCE, off the loop,
    # so /v1/me never pays for the two subprocesses on a request.
    try:
        _pk_caps = await asyncio.to_thread(_subtitle_mux.ffmpeg_capabilities)
        if _pk_caps.available:
            logger.info("[package] ffmpeg %s: containers %s",
                        _pk_caps.version or "?",
                        "mkv+mp4" if _pk_caps.mp4 else "mkv")
        else:
            logger.warning("[package] subtitle packaging unavailable: %s",
                           _pk_caps.reason)
    except Exception as _pk_err:  # noqa: BLE001 — a probe never blocks startup
        logger.warning("[package] ffmpeg probe failed: %s", _pk_err)

    # Install + verify the SSRF guard that every yt-dlp fetch rides on
    # (ytdlp_plugins/, see url_download.guard_self_check). Doing it here makes
    # a broken guard an operator-visible startup line instead of a surprise on
    # the first pasted link; probe()/download() re-check and refuse anyway, so
    # a failure here is logged, never fatal. Skipped when yt-dlp isn't
    # installed at all — there is then nothing to guard.
    if url_download.yt_dlp_version():
        try:
            url_download.guard_self_check(force=True)
        except Exception as _ge:  # noqa: BLE001 — guard_self_check already logged
            logger.error("[url-dl] link downloads will be refused: %s", _ge)

    # Open the browser-session store (HttpOnly cookie auth for the WebUI).
    # Non-fatal: if this fails, cookie login is unavailable but bearer auth
    # (API clients) and open mode keep working.
    sessions_purge_task = None
    try:
        sessions_store.init_db(cfg.SESSIONS_DB)
        logger.info("Session store initialized at %s", cfg.SESSIONS_DB)
        sessions_purge_task = asyncio.create_task(_sessions_purge_loop())
    except Exception as _se:
        logger.error("Failed to initialize session store: %s", _se)

    usage_sweep_task = (asyncio.create_task(_usage_retention_loop())
                        if usage_store_ready else None)
    # 1 Hz machine sampler: the GPU-busy share and the /stats history charts.
    # Needs the system-metrics store.
    stats_sampler_task = asyncio.create_task(_stats_sampler.loop())

    # Open the reports SQLite store (durable, plaintext dictation content
    # on disk) and run an immediate retention sweep before serving traffic.
    # Failure here is non-fatal: the rest of the app must keep working even if
    # the reports surface is broken, but the /reports page will error.
    reports_sweep_task = None
    try:
        reports_store.init_db(cfg.REPORTS_DB)
        reports_store.sweep_retention()
        logger.info("Reports store initialized at %s", cfg.REPORTS_DB)
        reports_sweep_task = asyncio.create_task(
            _reports_retention_loop()
        )
    except Exception as _re:
        logger.error("Failed to initialize reports store: %s", _re)

    # Open the desktop-client settings-sync store (one opaque blob per
    # account, served at /v1/synced-client-settings). Non-fatal: sync
    # degrades to 503s but transcription keeps working. No retention loop —
    # bounded at one row per account.
    try:
        client_settings_store.init_db(cfg.CLIENT_SETTINGS_DB)
        logger.info(
            "Client-settings store initialized at %s", cfg.CLIENT_SETTINGS_DB
        )
    except Exception as _cse:
        logger.error(
            "Failed to initialize client-settings store at %s: %s — "
            "/v1/synced-client-settings will answer 503 until this is fixed "
            "(WHISPER_CLIENT_SETTINGS_DB / WHISPER_DB_DIR / WHISPER_DATA_DIR)",
            cfg.CLIENT_SETTINGS_DB, _cse,
        )

    # Open the captures store. Audio + word-timestamps for Whisper
    # fine-tuning, gated by CAPTURES_RECORDING_ENABLED. Reconcile drift
    # before serving (row says audio exists / disk says it doesn't, or
    # vice versa).
    captures_sweep_task = None
    try:
        captures_store.init_db(cfg.CAPTURES_DB, cfg.CAPTURES_DIR)
        # capture_samples_store reuses the captures DB connection — single
        # SQLite file holds both tables. Init it before the first
        # sweep_retention(): the sweep's sample-expiry pass needs it.
        capture_samples_store.init_db(captures_store._require_conn(), cfg.CAPTURES_DIR)
        captures_store.reconcile_on_startup()
        capture_samples_store.reconcile_on_startup()
        captures_store.sweep_retention()
        logger.info(
            "Captures store initialized at %s (audio dir: %s, enabled=%s)",
            cfg.CAPTURES_DB, cfg.CAPTURES_DIR,
            getattr(cfg, "CAPTURES_RECORDING_ENABLED", False),
        )
        captures_sweep_task = asyncio.create_task(
            _captures_retention_loop()
        )
    except Exception as _ce:
        logger.error("Failed to initialize captures store: %s", _ce)

    # Server jobs: the durable job resource. A row still `running` now was
    # interrupted by the previous process's death — flip it, or it reads as
    # in flight forever (the poller would wait out the whole TTL).
    jobs_sweep_task = None
    try:
        _jobs_store.init_db(cfg.JOBS_DB)
        _n_interrupted = _jobs_store.mark_running_as_failed("server restarted")
        _jobs_store.sweep_retention()
        logger.info("Jobs store initialized at %s (%d interrupted run(s) marked failed)",
                    cfg.JOBS_DB, _n_interrupted)
        jobs_sweep_task = asyncio.create_task(_jobs_retention_loop())
    except Exception as _je:
        logger.error("Failed to initialize jobs store: %s", _je)

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
    # Before the model drops below: stop() clears the warm predicate, so the
    # shutdown teardown cannot be second-guessed by a lease nobody can renew.
    await _cancel(preload_sweeper_task)
    await _cancel(receipt_sweeper_task)
    # Anything still waiting on a translation that will now never come. A
    # restart must not eat a receipt that was merely being patient.
    tx_receipt._log_held_receipts(receipt_hold.flush_all())
    await preload.stop()
    await _cancel(url_media_janitor_task)
    await _cancel(reports_sweep_task)
    await _cancel(captures_sweep_task)
    await _cancel(jobs_sweep_task)
    await _cancel(usage_sweep_task)
    await _cancel(stats_sampler_task)
    # Whatever the sampler queued in its last minute.
    try:
        await asyncio.to_thread(_stats_sampler.flush)
    except Exception:  # noqa: BLE001 — shutdown must not care
        pass
    await _cancel(sessions_purge_task)
    await _cancel(open_mode_task)

    # force: the process is going away, so leases buy nothing — same contract
    # as drain_then_evict, where an in-flight request finishes on its own ref.
    for _name in list(tx_models._loaded_models):
        tx_models._drop_loaded_model(_name, force=True)
    tx_models._model_leases.clear()
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
# origins in CORS_ALLOW_ORIGINS so cross-origin JSON-API calls work (e.g. a
# third-party browser app on another origin calling this backend; /dictate
# itself always talks to its own origin).
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
# by auth_hosts._origin_is_allowed: it adds no CORS headers and no
# cross-origin access.
_trusted_origins = list(getattr(cfg, "TRUSTED_ORIGINS", []) or [])
if _trusted_origins:
    logger.info("Trusted origins (same-origin check): %s",
                ", ".join(_trusted_origins))
# Hand both lists to the same-origin guard (auth/hosts.py). Done here, at app
# build, so a reload of this module re-arms the guard with the reloaded config.
auth_hosts.configure_origins(_cors_origins, _trusted_origins)

# Static assets for the /stats dashboard (vendored uPlot, etc). Local-only —
# do not put anything sensitive under static/.
from fastapi.staticfiles import StaticFiles
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(REPO_ROOT, "static")),
    name="static",
)

# API docs are admin-tier — treated like /settings. The HTML shells (/docs,
# /redoc) are gated by the admin host allowlist only (a keyless browser must be
# able to load the swagger UI), while /openapi.json — the actual API surface —
# additionally requires an admin key. In OPEN mode (no admin key yet) the
# synthetic admin passes, so loopback docs work out of the box; once locked
# down, /openapi.json needs an admin session/key (the swagger UI fetches it
# same-origin with the session cookie).
from faster_whisper_backend.auth.dependencies import require_admin as _require_admin
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import JSONResponse as _JSONResponse


@app.get(
    "/openapi.json",
    include_in_schema=False,
    dependencies=[Depends(auth_hosts.require_admin_webui_host), Depends(_require_admin)],
)
async def _openapi_json():
    return _JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False, dependencies=[Depends(auth_hosts.require_admin_webui_host)])
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


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(auth_hosts.require_admin_webui_host)])
async def _redoc_ui():
    # Vendored — see the note on /docs above.
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " — redoc",
        redoc_js_url="/static/redoc.standalone.js",
        redoc_favicon_url="/static/favicon-32.png",
        # FastAPI defaults this to True and injects a fonts.googleapis.com
        # stylesheet <link> — a live CDN dependency on an admin-origin page,
        # same class as the validatorUrl residue on /docs. Redoc falls back
        # to system fonts without it.
        with_google_fonts=False,
    )

# Per-request metrics middleware. Records (path, status, duration) for every
# HTTP request — bumps in_flight tracked separately by the transcribe handler.
from faster_whisper_backend.stats import metrics

# Central running-jobs registry (transcribe/dictate/translate/download/preload)
# — feeds /stats and the WebUI header activity cluster.
from faster_whisper_backend.core import jobs
from faster_whisper_backend.transcription import jobs_store as _jobs_store

# Model preloading. Imported here rather than lazily because call sites below
# (the two `loaded` flag endpoints) and tx_progress._progress_set reach it on
# hot paths. preload never imports main (it reaches the whisper cache through
# transcription.models), so no cycle.
from faster_whisper_backend.runtime import preload
from faster_whisper_backend.transcription import run_plan as _run_plan

# Dictation receipts held open until their translation arrives on a separate
# request. Imports nothing from the app, so no cycle.
from faster_whisper_backend.transcription import receipt_hold

# Batch progress / cancel / plan registries and the job-ledger writes for runs
# posted with a progress_id — see transcription/progress.py. Called through the
# module attribute (tx_progress._progress_set(...)) so a test patch reaches main.
from faster_whisper_backend.transcription import progress as tx_progress

# Link-run helpers the transcription handler shares with the url/media routes
# (keep_video download task, guarded audio download, prefetched audio) — see
# media/video.py. Called through the module attribute so a test patch reaches
# both callers.
from faster_whisper_backend.media import video as media_video


_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# Paths that issue a session and therefore can't carry a CSRF token yet.
# Exempt from the TOKEN check only — the origin check still applies (these
# paths hand out a cookie, so a cross-site caller must not reach them).
_CSRF_EXEMPT_PATHS = frozenset({"/auth/login"})


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
        if not auth_hosts._origin_is_allowed(request):
            auth_hosts._log_origin_rejected(request)
            return JSONResponse(
                {"detail": "Origin not allowed for this host"},
                status_code=403,
            )
        if request.url.path not in _CSRF_EXEMPT_PATHS:
            cookie = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
            if cookie:
                import hmac
                sess = sessions_store.lookup_session(cookie)
                # Hand the resolved row to auth.user_from_session_cookie via
                # the shared scope state so the dependency does not look the
                # same cookie up a second time (each lookup takes the store
                # lock twice on the event loop).
                request.state.session_record = sess
                header_tok = request.headers.get("x-csrf-token", "")
                _auth_header = request.headers.get("authorization", "")
                if sess is None and _auth_header.lower().startswith("bearer "):
                    # A cookie that no longer resolves (row expired, sessions
                    # DB wiped or moved, cookie outliving its 30 d row) riding
                    # along with a bearer credential: the bearer wins in
                    # auth._resolve_user and is not cookie-driven, so it cannot
                    # be CSRF'd — refusing it would only break every API client
                    # that also holds a stale browser cookie. Deliberately
                    # NARROW: a dead cookie with no bearer still fails closed,
                    # because in open mode the request would otherwise resolve
                    # to the synthetic admin behind nothing but the Origin
                    # check above.
                    pass
                elif (
                    sess is None
                    or not header_tok
                    # Starlette decodes header bytes as latin-1, and
                    # compare_digest(str, str) raises TypeError on any
                    # non-ASCII character — a one-byte `\xe9` header would
                    # turn this 403 into an unhandled 500. Tokens are hex,
                    # so a non-ASCII value can never match.
                    or not header_tok.isascii()
                    or not hmac.compare_digest(header_tok, sess["csrf_token"])
                ):
                    return JSONResponse(
                        {"detail": "CSRF token missing or invalid"},
                        status_code=403,
                    )
    return await call_next(request)


# Non-JSON, non-multipart bodies never legitimately approach the media cap;
# they keep the pre-media-cap service ceiling.
_NON_UPLOAD_BODY_BACKSTOP = 268_435_456
_MEDIA_PACKAGE_PATH_RE = re.compile(
    rf"\A/v1/audio/media/{url_media_store.MEDIA_ID_PATTERN}/package\Z")


def _media_package_max_body_bytes() -> int:
    """The packaging request's body cap, derived from the route's own bounds
    (MAX_TRACKS SRT files of MAX_SRT_BYTES each = 12 x 2 MiB today) with 2x
    headroom for JSON escaping — so a body that respects every per-track
    limit reaches the route and its 422s instead of a bare 413 here."""
    return _subtitle_mux.MAX_TRACKS * _subtitle_mux.MAX_SRT_BYTES * 2


@app.middleware("http")
async def _max_body_mw(request: Request, call_next):
    """Service-wide ceiling on a declared request body, rejected before the
    body is read. Route-level caps stay authoritative for their own endpoint
    (MAX_REQUEST_BYTES sits well above MEDIA_MAX_BYTES, so the transcription
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
    # getattr default, so no config-schema change is required. The media type
    # is parsed, not prefix-matched: FastAPI treats `application/*+json`
    # (merge-patch+json, ld+json, ...) AND a request with NO Content-Type at
    # all as JSON and calls request.json() on it, so both must share the
    # ceiling or they bypass it. multipart media uploads always declare their
    # own media type and keep the full MAX_REQUEST_BYTES — sized for a
    # MEDIA_MAX_BYTES video, i.e. gigabytes. Every OTHER non-JSON body keeps
    # the old 256 MiB backstop: nothing but an upload has business being
    # larger, and the media cap must not silently widen text/plain PUTs.
    _ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    _main, _, _sub = _ctype.partition("/")
    if not _ctype or (_main == "application"
                      and (_sub == "json" or _sub.endswith("+json"))):
        max_body = min(max_body, int(getattr(cfg, "MAX_JSON_BODY_BYTES", 4_194_304)))
    elif _ctype != "multipart/form-data":
        max_body = min(max_body, _NON_UPLOAD_BODY_BACKSTOP)
    # Two path-exact exceptions for the media export: the raw-body video
    # upload (the route counts its own bytes against MEDIA_MAX_BYTES, so the
    # ceiling here just matches it) and the packaging request, whose JSON
    # carries up to MAX_TRACKS subtitle files (bigger than any other JSON
    # body, still tightly bounded).
    if request.method == "POST":
        _path = request.url.path
        if _path == "/v1/audio/media":
            max_body = int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000))
        elif _MEDIA_PACKAGE_PATH_RE.match(_path):
            max_body = _media_package_max_body_bytes()
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
                # Shared with the handler through the ASGI scope (the
                # handler's Request is a different object over the same
                # scope), so a route that streams its body can answer 413
                # instead of a bare disconnect.
                request.scope.setdefault("state", {})["body_cap_hit"] = True
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
        # route in the scope after routing (before the handler runs, so it
        # is present even when the handler raised) — fall back to the raw
        # URL path for 404s and pre-routing failures.
        route = request.scope.get("route")
        path = route.path if route is not None else request.url.path
        metrics.record_request(path, status,
                               (time.perf_counter() - start) * 1000.0,
                               unmatched=route is None)


# Renamed client endpoints: old path → its successor. Both paths are served by
# the same handler (stacked route decorators, the old one deprecated=True), so
# a client built before the rename keeps working; no redirect, because a 307/
# 308 makes the client replay a PUT body. _legacy_path_mw stamps the RFC 9745 /
# RFC 8594 headers on every answer from an old path.
_LEGACY_PATHS = {
    "/v1/decode-defaults": "/v1/request-default-settings",
    "/v1/client-settings": "/v1/synced-client-settings",
}
# When the old paths were deprecated (2026-10-05 00:00 UTC), as the
# `Deprecation: @<epoch>` structured-field date.
_LEGACY_SINCE_EPOCH = 1791158400
# Old paths already logged — once per path per process is enough to see which
# deployments still run old clients.
_legacy_paths_logged: set[str] = set()


@app.middleware("http")
async def _legacy_path_mw(request: Request, call_next):
    """Deprecation headers for the renamed endpoints in _LEGACY_PATHS.

    A middleware, not an injected Response: the PUT handler returns its own
    JSONResponse and errors are HTTPExceptions, both of which drop headers set
    on an injected Response. Registered before _security_headers_mw, so it
    wraps the inner middlewares and also stamps their early 403/413."""
    response = await call_next(request)
    path = request.url.path
    successor = _LEGACY_PATHS.get(path)
    if successor is not None:
        response.headers["Deprecation"] = f"@{_LEGACY_SINCE_EPOCH}"
        response.headers["Link"] = f'<{successor}>; rel="successor-version"'
        if path not in _legacy_paths_logged:
            _legacy_paths_logged.add(path)
            logger.info(
                "Deprecated path %s called (successor %s, user-agent %s); "
                "logged once per path",
                path, successor,
                _log_safe(request.headers.get("user-agent") or "-"))
    return response


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
    context_segments: int | None = Form(None),
    keep_video: str | None = Form(None),
    video_max_height: int | None = Form(None),
    video_format: str | None = Form(None),
    retain_media: str | None = Form(None),
    prefetched_media_id: str | None = Form(None),
    progress_id: str | None = Form(None),
    preload_plan: str | None = Form(None),
    user: dict = Depends(_get_current_user_dep),
):
    resolved_model = tx_models._resolve_model_name(model_name)
    _pid = tx_progress._claim_progress_id(progress_id)
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
    # rung by rung to _TEMPERATURE_BOUNDS. Clamp rather than 422 so a caller that is merely
    # sloppy keeps working; NaN fails every comparison, hence the self-test.
    if prompt is not None:
        prompt = prompt[:tx_models._DECODE_STR_CAPS.get("prompt", 2048)]
    _t_lo, _t_hi = tx_models._TEMPERATURE_BOUNDS
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

    # Bracket the entire request with metrics.in_flight_transcriptions +
    # record_transcription so failed loads / failed transcriptions still
    # surface in the dashboard.
    # request_id is generated up-front (was deferred to post-transcribe) so
    # the outer finally can correlate timing-only writes to the SQLite store
    # on the error path too.
    metrics.in_flight_transcriptions += 1
    # GPU-gate wait for this request accumulates here (every acquire in
    # this task, and in tasks it spawns, charges it); read in the finally.
    metrics.seed_wait()
    _t0 = time.perf_counter()
    _status = "ok"
    # Failure classification for the ledger (metrics.classify_error in the
    # finally): the stage the request was in when it failed and the
    # exception the arms saw. A stage that pre-classifies (the URL policy)
    # sets _error_class directly and the finally keeps it.
    _cur_stage: "str | None" = None
    _exc: "BaseException | None" = None
    _error_class: "str | None" = None
    _error_stage: "str | None" = None
    # Server jobs: whether this run has a row, and the response object as
    # returned (captured right before each `return` so the outer finally can
    # persist exactly what the client received — or would have, had it
    # still been connected).
    _job_row = False
    _response_payload: "str | dict | None" = None
    _task_now: "str | None" = task or "transcribe"
    _audio_dur: float = 0.0
    _words: int = 0
    # Detected (or requested) language for the usage job row; None until the
    # decode has run, and stays None on the error path.
    _language: "str | None" = None
    # Per-stage wall-clock receipts ({name, secs, model?, detail?}) — the
    # durations were previously computed for log lines and discarded; now
    # they also persist as the recent-jobs row's stages.
    _stage_timings: "list[dict]" = []
    # The server-owned plan behind the progress route (transcription/run_plan.py):
    # seeded with the stages the Form args imply, refined as facts land,
    # ticked by every _progress_set, read by every poll.
    _rplan = _run_plan.RunPlan(kind="url" if source_url is not None else "file")
    tmp_path = None
    # Transcribe-from-URL state: the private download dir (rmtree'd in the
    # inner finally on every path) and the retention id echoed to the client.
    _url_job_dir: "str | None" = None
    _source_media_id: "str | None" = None
    # The pipeline's copy of a language check's retained audio, when the run
    # reuses it (prefetched_media_id) instead of downloading.
    _reused_copy: "str | None" = None
    # Optional VIDEO of a link (keep_video): fetched by a task that runs
    # alongside the pipeline and may outlive this handler — see
    # media_video._download_video_for_run for the progress-entry hand-over.
    _video_task: "asyncio.Task | None" = None
    _video_result: "dict | None" = None
    _run_finished = [False]
    # retain_media: an uploaded VIDEO the client wants packaged with its
    # subtitles right after — a hardlinked copy of the spool, registered in
    # the media store once the run succeeds (never for a failed one).
    _retain_media = (_form_bool(retain_media) is True and source_url is None
                     and bool(getattr(cfg, "MEDIA_PACKAGE_ENABLED", True)))
    _retained_upload: "str | None" = None
    # Set only AFTER the load returns, so the outer finally never releases a
    # lease that was never taken (a rejected/failed load takes none).
    _leased_model: "str | None" = None
    request_id = uuid.uuid4().hex
    _user_id = user.get("user_id")
    _key_id = user.get("key_id")
    # Central running-jobs registry: one entry per in-flight request; stage/
    # progress mirror in via _progress_set (bound through _JOB_BY_PID) when
    # the client opted into progress reporting.
    # `user` is the display name so the running row in /stats reads the same
    # as the finished rows beneath it ("alice", not an opaque row id).
    jobs.job_start("transcribe", id=request_id, model=resolved_model,
                   user=user.get("username") or _user_id, key=_key_id,
                   user_id=_user_id)
    tx_progress._bind_job_pid(_pid, request_id)
    try:
        # Seed the registry entry EARLY so the cancel endpoint (which only
        # accepts ids it can see in-flight) has a target well before the
        # first stage-driven _progress_set (model load, semaphore queue can
        # be seconds away) — but only INSIDE this try, whose finally pops the
        # entry: seeding before the validation gates above leaked an orphan
        # "waiting" entry (and a cancellable id) on every early 4xx. URL runs
        # seed as "resolving": their pipeline starts at the link, and
        # "waiting" maps onto the transcribe row in the client's rail — which
        # would paint the download as already done. `owner` binds the entry
        # to this caller: the progress/cancel endpoints treat a mismatched
        # caller exactly like an unknown id.
        _keep_video = _form_bool(keep_video) is True
        _video_max_height = media_video._clamp_video_height(video_max_height)
        _video_format = media_video._clean_video_format(video_format)
        if _keep_video:
            if source_url is None:
                raise HTTPException(
                    status_code=422,
                    detail="keep_video applies to a link, not an uploaded file")
            if not getattr(cfg, "URL_VIDEO_ENABLED", False):
                raise HTTPException(
                    status_code=403,
                    detail="video download is not enabled on this server")
            if not _pid:
                # The fetch outlives the response and reports through the
                # progress entry alone: without an id nobody could poll,
                # cancel or ever learn the media id it registers.
                raise HTTPException(
                    status_code=422,
                    detail="keep_video needs a progress_id to report the "
                           "video through")
            media_video._url_video_rate.hit(_rl.identity_key(user, request))
        if _retain_media and response_format == "text":
            # A text body has no field for the media id, so the retained
            # copy could never be addressed — refuse before the spool (and
            # its hardlink) is written, like keep_video above.
            raise HTTPException(
                status_code=422,
                detail="retain_media requires a json response format")
        _rplan.set_stages(tx_progress._provisional_stages(
            is_url=source_url is not None,
            separate=_form_bool(separate_bgm), diarize=_form_bool(diarize),
            translate_to=translate_to))
        if _pid:
            tx_progress._RUN_PLAN_BY_PID[_pid] = _rplan
        tx_progress._progress_set(_pid,
                      stage=("resolving" if source_url is not None
                             else "waiting"),
                      progress=None,
                      owner=(_user_id or _key_id))
        _job_row = tx_progress._jobs_start(
            _pid, request_id=request_id, kind="transcribe",
            user_id=_user_id, key_id=_key_id, model=resolved_model,
            source_kind=("url" if source_url is not None else "file"),
            # A basename or a host — never the full URL (tokens) or path.
            source_name=(media_video._url_host_for_log(source_url)
                         if source_url is not None
                         else os.path.basename(
                             getattr(file, "filename", None) or "")),
            task=_task_now, response_format=response_format)
        # Upload ceiling. Content-Length is advisory (absent on chunked
        # bodies), so it only buys us an early exit before the model load —
        # the chunked read below is what actually enforces the bound.
        max_upload = int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000))
        _clen = request.headers.get("content-length")
        if _clen and _clen.isdigit() and int(_clen) > max_upload:
            raise HTTPException(status_code=413, detail="upload too large")

        # Perf origin for the whisper load: the transcribing row's decode t0
        # is taken long after this, so `load_secs_since` from there could
        # never see a cold load — the receipt reported `load 0.0s` always.
        _model_t0 = time.perf_counter()
        model = await tx_models._get_or_load_model(resolved_model, lease=True)
        _leased_model = resolved_model
        _plan_compute, _plan_device = tx_receipt._model_compute_device(resolved_model)
        _rplan.set_stage_model("transcribing", model=resolved_model,
                               device=_plan_device, compute=_plan_compute)

        # Resolve the caller's effective per-identity config ONCE for this
        # request: layered decode params, pipeline include/exclude, output
        # wrappers, and which fields are locked against client overrides.
        # Open mode / no per-identity config → no identity layers (≡ today).
        # `override_profile` (if sent + allowed) joins as the least-specific layer.
        ident = effective_config.build_ident(user, resolved_model, request_profile=override_profile)

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
        # Same sentinel for the other tri-state text fields: present-but-empty
        # is the client's "cleared — overrides inherited" state (auto-detect
        # language / no translation targets / no glossary), absent inherits.
        _lang_field = form_data.get("language")
        language = _lang_field if isinstance(_lang_field, str) else None
        _tt_field = form_data.get("translate_to")
        translate_to = _tt_field if isinstance(_tt_field, str) else None
        _tg_field = form_data.get("translation_glossary")
        translation_glossary = _tg_field if isinstance(_tg_field, str) else None
        # Re-apply the clamp from above: this re-read overwrote the bounded
        # Form value with the raw one, so the cap was dead code and the field
        # reached the tokenizer at whatever size the multipart parser allowed.
        if prompt is not None:
            prompt = prompt[:tx_models._DECODE_STR_CAPS.get("prompt", 2048)]

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
                logger.info("[url-dl] transcribe-from-url requested (host %s)",
                            media_video._url_host_for_log(source_url))
                _dl_t0 = time.perf_counter()
                _cur_stage = "downloading"
                try:
                    tx_progress._check_cancelled(_pid)
                    # probe() validates the URL; info.url is the normalised one.
                    _uinfo = await url_download.probe(
                        source_url,
                        timeout=float(getattr(cfg, "URL_PREVIEW_TIMEOUT_S", 20)))
                    _url = _uinfo.url
                    logger.info(
                        "[url-dl] resolved (host %s): extractor=%s duration=%s"
                        " — starting download",
                        media_video._url_host_for_log(_url), _uinfo.extractor_key,
                        f"{_uinfo.duration:.0f}s"
                        if _uinfo.duration is not None else "?")
                    _rplan.set_audio_seconds(_uinfo.duration, src="probe")
                    _rplan.set_download_bytes(
                        _uinfo.filesize_approx
                        or (int(_uinfo.duration * _uinfo.abr * 125)
                            if _uinfo.duration and _uinfo.abr else None),
                        extractor=(_uinfo.extractor_key or None))
                    tx_progress._progress_set(_pid, stage="downloading", progress=None,
                                  total_bytes=None,
                                  step=(_uinfo.extractor_key or None))
                    # The audio a language check of this link already
                    # downloaded (POST /v1/audio/url-language): reused when
                    # the caller may, else the run downloads as usual.
                    _reuse = media_video._prefetched_audio(prefetched_media_id, _url,
                                               _user_id)
                    if _reuse:
                        _reused_copy = await asyncio.to_thread(
                            url_media_store.make_pipeline_copy, _reuse["path"])
                    if _reused_copy:
                        _dl_path = _reuse["path"]
                        logger.info("[url-dl] reusing the audio the language"
                                    " check downloaded (host %s)",
                                    media_video._url_host_for_log(_url))
                    else:
                        if prefetched_media_id:
                            logger.info("[url-dl] prefetched audio not "
                                        "reusable (host %s) — downloading",
                                        media_video._url_host_for_log(_url))
                        # Not a staging job: the download outlives this
                        # block (pipeline copy, retention) and the handler's
                        # finally removes the dir.
                        _url_job_dir = tempfile.mkdtemp(prefix="urldl-")
                        _dl_path = await media_video._guarded_audio_download(
                            _pid, _url, _url_job_dir, max_bytes=max_upload)
                    # Resolve + download as one receipt row: on a long link
                    # it can dominate wall time. Cancelled/error paths leave
                    # no entry.
                    _stage_timings.append({
                        "name": "downloading",
                        "secs": round(time.perf_counter() - _dl_t0, 2),
                        "detail": _uinfo.extractor_key or None,
                    })
                except url_download.UrlCancelled:
                    logger.info("[url-dl] download cancelled by client "
                                "(host %s)", media_video._url_host_for_log(source_url))
                    raise tx_progress._ClientCancelled() from None
                except url_download.UrlDownloadError as _ue:
                    # str() is client-safe by the module's contract.
                    logger.info("[url-dl] rejected (host %s): %s",
                                media_video._url_host_for_log(source_url),
                                _log_safe(str(_ue)))
                    # Classify HERE: below, the 400 is all the arms see.
                    _error_class, _error_stage = metrics.classify_error(
                        _ue, status="error", stage="downloading")
                    raise HTTPException(status_code=400, detail=str(_ue))
                audio_bytes = os.path.getsize(_dl_path)
                _rplan.set_download_bytes(audio_bytes)
                _rplan.stage_done("downloading")
                # Pipeline copy FIRST (hardlink where possible), THEN move
                # the original into the retention store — afterwards each
                # side owns its file outright: tmp_path follows the normal
                # unlink-in-finally lifecycle (including the BGM swap), and
                # the retained file serves GET /v1/audio/url-media/{id}.
                # to_thread: the copy usually hardlinks but can degrade to a
                # full copy, and register()'s move crosses filesystems
                # (TMPDIR → URL_MEDIA_DIR) — up to MEDIA_MAX_BYTES of blocking
                # I/O that must not pin the event loop.
                if _reused_copy:
                    # Already retained under the check's id: hand that out.
                    tmp_path, _source_media_id = _reused_copy, prefetched_media_id
                else:
                    tmp_path = await asyncio.to_thread(
                        url_media_store.make_pipeline_copy, _dl_path)
                    if tmp_path is None:
                        # Disk trouble (logged by the store) — generic 500 path.
                        raise RuntimeError("url pipeline copy failed")
                    # Retention is a playback nicety: None just means the
                    # client gets no audio copy, never a failed transcription.
                    _source_media_id = await asyncio.to_thread(
                        url_media_store.register, _dl_path, user_id=_user_id,
                        source_url=_url)
                if _keep_video:
                    _rung = url_download.pick_rung(_uinfo.video_ladder, _video_max_height,
                                           _video_format)
                    if _rung is None:
                        _video_result = media_video._video_state(
                            state="failed", error="this link has no video track")
                        tx_progress._progress_set(_pid, video=dict(_video_result))
                    elif _rung.get("over_cap"):
                        _video_result = media_video._video_state(
                            state="failed",
                            error="the video exceeds the server's size limit",
                            height=_rung.get("height"),
                            container=_rung.get("container"))
                        tx_progress._progress_set(_pid, video=dict(_video_result))
                    else:
                        # Network-bound and off the GPU path: it runs beside
                        # the pipeline, never delays the transcript, and is
                        # reported through the entry's `video` sub-object.
                        _video_task = asyncio.create_task(
                            media_video._download_video_for_run(
                                _pid, _url, _rung,
                                capped=_video_max_height is not None,
                                user_id=_user_id, protect=_source_media_id,
                                run_finished=_run_finished,
                                job_row=_job_row))
                        # keep_video is refused without a progress_id.
                        media_video._VIDEO_TASKS[_pid] = _video_task
            else:
                # Stream the part to the temp file in chunks, counting bytes as
                # we go: the upload is never fully resident, and an oversized
                # body is cut off mid-read instead of after it has been
                # materialised. Only the SIZE is needed downstream (capture
                # size guard + log block).
                audio_bytes = 0
                # "whisperup-" marks the file for restart_service.reclaim_hard_restart_orphans
                # — tempfile's default "tmp" prefix is every process's.
                with tempfile.NamedTemporaryFile(delete=False, prefix="whisperup-", suffix=_safe_tmp_suffix(file.filename)) as tmp_file:
                    tmp_path = tmp_file.name
                    while True:
                        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        audio_bytes += len(chunk)
                        if audio_bytes > max_upload:
                            raise HTTPException(status_code=413, detail="upload too large")
                        tmp_file.write(chunk)
                # Until the decoder measures the audio, size it as 128 kbps.
                _rplan.set_audio_seconds(
                    audio_bytes / _run_plan.BYTES_PER_AUDIO_SECOND,
                    src="bytes-prior")
                if _retain_media:
                    # A hardlink of the spool (same TMPDIR): the pipeline may
                    # replace tmp_path with a vocals stem and unlinks it in
                    # the finally; this copy survives for the media store.
                    _retained_upload = await asyncio.to_thread(
                        url_media_store.make_pipeline_copy, tmp_path)

            # word_timestamps: AND of the (per-model-overrideable) global
            # config knob and the per-request ask. Disabled (False) bypasses
            # the DTW alignment path entirely — required for primeline-style
            # finetunes that hit faster-whisper#1212.
            gate_word_ts = effective_config.cfg_for(resolved_model, "WORD_TIMESTAMPS_ENABLED", ident)
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
            if (getattr(cfg, "CAPTURES_RECORDING_ENABLED", False)
                    and gate_word_ts):
                try:
                    cap_max = int(getattr(cfg, "CAPTURES_MAX", 5000))
                    hard_lim = int(getattr(
                        cfg, "CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT",
                        100_000_000,
                    ))
                    sample_rate = float(getattr(
                        cfg, "CAPTURES_RECORDING_SAMPLE_RATE", 1.0,
                    ))
                    if (captures_store.count_evictable() < cap_max
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
                _prompt = effective_config.cfg_for(resolved_model, "DEFAULT_PROMPT", ident)
                if prompt is not None and prompt != _prompt:
                    ignored.append("prompt")
            elif prompt is not None:
                _prompt = prompt
            else:
                _prompt = effective_config.cfg_for(resolved_model, "DEFAULT_PROMPT", ident)
            initial_prompt_arg = _prompt if _prompt else None

            _vad_filter = effective_config.cfg_for(resolved_model, "VAD_FILTER", ident)
            vad_parameters = dict(
                min_silence_duration_ms=effective_config.cfg_for(resolved_model, "VAD_MIN_SILENCE_MS", ident),
                speech_pad_ms=effective_config.cfg_for(resolved_model, "VAD_SPEECH_PAD_MS", ident),
                threshold=effective_config.cfg_for(resolved_model, "VAD_THRESHOLD", ident),
            ) if _vad_filter else None

            _lead_pad_ms = int(effective_config.cfg_for(resolved_model, "LEADING_SILENCE_PAD_MS", ident) or 0)

            # Coerce empty to None — faster-whisper validates the value against
            # its accepted-codes list, so "" raises ValueError; None triggers
            # the first-30s auto-detect path, which is what an empty
            # DEFAULT_LANGUAGE is documented to mean. A LOCKED DEFAULT_LANGUAGE
            # likewise forbids the client's `language` param.
            # (`_decode_language`, not the `_language` ledger field above:
            # that one stays None until the decode has actually run.)
            if "DEFAULT_LANGUAGE" in ident.locked:
                _decode_language = effective_config.cfg_for(resolved_model, "DEFAULT_LANGUAGE", ident)
                if language is not None and language != _decode_language:
                    ignored.append("language")
            else:
                # Present-but-empty is an explicit "auto-detect" (the client's
                # cleared state); only an ABSENT field inherits the config.
                _decode_language = (language if language is not None
                                    else effective_config.cfg_for(resolved_model, "DEFAULT_LANGUAGE", ident))
            # Task: absent field inherits the resolved TASK config (per-identity
            # > per-model > global, default "transcribe"); a LOCKED TASK forbids
            # the client's `task` param the way a locked DEFAULT_LANGUAGE binds
            # `language` above.
            _task = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TASK", "task", task, default="transcribe")
            _task_now = _task
            # Diarization request knobs: same absent-inherits / locked-wins
            # shape as task above. The capacity gate (DIARIZATION_ENABLED)
            # is checked at the stage itself and soft-fails into `warnings`.
            _diarize_req = _form_bool(diarize)
            if "DIARIZE" in ident.locked:
                _diarize = bool(effective_config.cfg_for(resolved_model, "DIARIZE", ident))
                if _diarize_req is not None and _diarize_req != _diarize:
                    ignored.append("diarize")
            elif _diarize_req is not None:
                _diarize = _diarize_req
            else:
                _diarize = bool(effective_config.cfg_for(resolved_model, "DIARIZE", ident))
            _spk = {}
            for _cfg_name, _client_name, _client_val in (
                ("DIARIZATION_NUM_SPEAKERS", "num_speakers", num_speakers),
                ("DIARIZATION_MIN_SPEAKERS", "min_speakers", min_speakers),
                ("DIARIZATION_MAX_SPEAKERS", "max_speakers", max_speakers),
            ):
                if _cfg_name in ident.locked:
                    _spk[_client_name] = effective_config.cfg_for(resolved_model, _cfg_name, ident)
                    if _client_val is not None and _client_val != _spk[_client_name]:
                        ignored.append(_client_name)
                elif _client_val is not None:
                    _spk[_client_name] = _client_val
                else:
                    _spk[_client_name] = effective_config.cfg_for(resolved_model, _cfg_name, ident)
            # pyannote treats num alongside min/max as an error — num wins.
            if _spk.get("num_speakers"):
                _spk["min_speakers"] = _spk["max_speakers"] = None
            # pyannote also rejects an inverted range — clamp min down to max.
            elif (_spk.get("min_speakers") and _spk.get("max_speakers")
                  and _spk["min_speakers"] > _spk["max_speakers"]):
                _spk["min_speakers"] = _spk["max_speakers"]
            # Music separation: same shape again. Soft-failed optional stages
            # (this and diarization) collect their explanations in _warnings.
            _warnings: "list[str]" = []
            # Requested stages this server declines to run (feature disabled).
            # Mirrored into the progress entry the moment each skip is known,
            # so a polling client can mark the stage "skipped" live instead of
            # inferring it — the warning text alone arrives only with the
            # final response.
            _skipped: "list[str]" = []

            def _skip(stage: str) -> None:
                """Record a declined stage + mirror it into the progress
                entry — the second half is pure bookkeeping whose omission
                would silently stop the client's rail from showing it."""
                _skipped.append(stage)
                _rplan.skip(stage)
                tx_progress._progress_set(_pid, skipped=list(_skipped))

            _sep_req = _form_bool(separate_bgm)
            if "SEPARATE_BGM" in ident.locked:
                _separate = bool(effective_config.cfg_for(resolved_model, "SEPARATE_BGM", ident))
                if _sep_req is not None and _sep_req != _separate:
                    ignored.append("separate_bgm")
            elif _sep_req is not None:
                _separate = _sep_req
            else:
                _separate = bool(effective_config.cfg_for(resolved_model, "SEPARATE_BGM", ident))
            # Per-request stage models (pyannote pipeline id / UVR model):
            # same ladder again. A non-empty allowlist that misses the
            # resolved value soft-fails by skipping THAT stage — before its
            # enabled gate, so the warning names the actual reason.
            _dm_req = (diarization_model or "").strip() or None
            _diarization_model = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "DIARIZATION_MODEL", "diarization_model", _dm_req)
            # The allowlist constrains only the CLIENT-requested value (a
            # config/identity-inherited model is admin policy and always
            # passes) and always admits the configured default (global AND
            # the identity/per-model effective value) — so an EMPTY
            # allowlist means "the configured model only", never "anything".
            _diar_allowed = set(
                getattr(cfg, "DIARIZATION_ALLOWED_MODELS", []) or [])
            _diar_allowed.add(getattr(cfg, "DIARIZATION_MODEL", "") or "")
            _diar_allowed.add(effective_config.cfg_for(resolved_model, "DIARIZATION_MODEL", ident) or "")
            if (_diarize and _dm_req is not None
                    and _diarization_model == _dm_req
                    and _diarization_model not in _diar_allowed):
                _warnings.append(
                    "requested diarization model is not allowed on this "
                    "server (DIARIZATION_ALLOWED_MODELS)")
                _skip("diarizing")
                _diarize = False
            _sm_req = (separation_model or "").strip() or None
            _separation_model = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "BGM_SEPARATION_UVR_MODEL", "separation_model", _sm_req)
            _sep_allowed = set(
                getattr(cfg, "BGM_SEPARATION_ALLOWED_MODELS", []) or [])
            _sep_allowed.add(getattr(cfg, "BGM_SEPARATION_UVR_MODEL", "") or "")
            _sep_allowed.add(effective_config.cfg_for(resolved_model, "BGM_SEPARATION_UVR_MODEL", ident) or "")
            if (_separate and _sm_req is not None
                    and _separation_model == _sm_req
                    and _separation_model not in _sep_allowed):
                _warnings.append(
                    "requested separation model is not allowed on this "
                    "server (BGM_SEPARATION_ALLOWED_MODELS)")
                _skip("separating")
                _separate = False
            # Translation (T2T) request knobs: the same locked-wins /
            # request-wins / config-inherits ladder as diarize/separate_bgm
            # above. The capacity gates (TRANSLATION_ENABLED, model allowlist)
            # live at the stage itself and soft-fail into `_warnings`.
            # Present-but-empty is an explicit "no targets" (overrides an
            # inherited TRANSLATE_TO); only an ABSENT field inherits.
            _tt_req = translate_to.strip() if translate_to is not None else None
            _tt_raw = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATE_TO", "translate_to", _tt_req)
            # csv → deduped ordered list of well-formed codes. Malformed
            # entries drop silently (the sloppy-caller stance of the clamped
            # knobs); the MAX_TARGETS clamp warns, naming what it dropped.
            _translate_to = language_codes(_tt_raw)
            _translation_max_targets = int(effective_config.cfg_for(
                resolved_model, "TRANSLATION_MAX_TARGETS", ident) or 1)
            if len(_translate_to) > _translation_max_targets:
                _warnings.append(
                    "translation targets over TRANSLATION_MAX_TARGETS "
                    f"({_translation_max_targets}) were dropped: "
                    + ", ".join(_translate_to[_translation_max_targets:]))
                _translate_to = _translate_to[:_translation_max_targets]
            _tm_req = (translation_model or "").strip() or None
            _translation_model = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_MODEL", "translation_model", _tm_req)
            # The identity/per-model effective model: echoing (or being
            # locked to) it is not a client choice the allowlist gates.
            _tm_inherited = (effective_config.cfg_for(resolved_model, "TRANSLATION_MODEL",
                                     ident) or "").strip() or None
            _translation_mode = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_MODE", "translation_mode", translation_mode,
                default="fluent")
            # Present-but-empty is an explicit "no glossary" (overrides an
            # inherited TRANSLATION_GLOSSARY); only an ABSENT field inherits.
            _translation_glossary = effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_GLOSSARY", "translation_glossary",
                translation_glossary)
            # The config field is Field(max_length=4000); cap the raw client
            # value to the same bound rather than 422ing a sloppy caller.
            _translation_glossary = (_translation_glossary or "")[:4000]
            # Context segments: the same ladder; a request value is clamped
            # to the field's range like the text route does.
            _translation_context = int(effective_config._resolve_request_knob(
                resolved_model, ident, ignored,
                "TRANSLATION_CONTEXT_SEGMENTS", "context_segments",
                tx_models._clamp_context_segments(context_segments),
                default=effective_config._NO_DEFAULT) or 0)

            # The run plan's FINAL stage list and per-stage models, now that
            # every enable/allowlist/soft-skip verdict has landed (a stage
            # _skip()ped above keeps its skipped state).
            _rplan.set_stages([
                *(["downloading"] if source_url is not None else []),
                *(["separating"] if _separate else []),
                "transcribing",
                *(["diarizing"] if _diarize else []),
                *(["translating"] if _translate_to else [])])
            if _separate:
                try:
                    _sep_dev = _bgm_separation.actual_device() or _bgm_separation._resolve_device()
                except Exception:  # noqa: BLE001 — optional dep absent
                    _sep_dev = None
                _rplan.set_stage_model("separating",
                                       model=(_separation_model or None),
                                       device=_sep_dev)
            if _diarize:
                try:
                    _diar_dev = _diarization._resolve_device()
                except Exception:  # noqa: BLE001
                    _diar_dev = None
                _rplan.set_stage_model("diarizing",
                                       model=(_diarization_model or None),
                                       device=_diar_dev)
            if _translate_to:
                _rplan.set_translation(
                    list(_translate_to),
                    model=((_translation_model or "").strip()
                           or tr_gating._translation_default_model() or None),
                    device=_tr._resolve_device(),
                    mode=_translation_mode,
                    # A pinned decode language already tells which targets
                    # are verbatim copies; auto-detect learns it post-decode.
                    source_lang=(_decode_language or None))

            # Stage-ahead: the stage plan is fully resolved here (every
            # enable/allowlist/soft-skip verdict above has landed), so this is
            # the first point at which the server knows which models this job
            # will actually need. Registered through the SAME register_plan the
            # endpoint calls — the client-driven and server-driven paths must
            # be one mechanism, or they will diverge.
            #
            # Whisper is deliberately NOT a stage-ahead target of its own job:
            # it is loaded above, before this plan can exist. Only a CLIENT
            # plan ever warms whisper. Do not "fix" that.
            #
            # A client that already POSTed a plan hands back its id in the
            # `preload_plan` form field instead of the server duplicating it.
            _preload_entries: "list[tuple[str, str]]" = []
            if _separate and _separation_model:
                _preload_entries.append(("separation", _separation_model))
            if _diarize and _diarization_model:
                _preload_entries.append(("diarization", _diarization_model))
            if _translate_to:
                _tr_ref = ((_translation_model or "").strip()
                           or tr_gating._translation_default_model())
                # Same allowlist verdict the stage itself renders later: a
                # ref the stage will refuse must not be pre-warmed (the plan
                # would download/load an arbitrary GGUF the request cannot
                # use). The stage's soft-fail warning still fires there.
                if _tr_ref and tr_gating._translation_model_allowed(
                        _tr_ref, requested=_tm_req,
                        inherited=_tm_inherited):
                    _preload_entries.append(("translation", _tr_ref))
            if _pid and _preload_entries:
                _plan_hint = (preload_plan or "").strip() or None
                if _plan_hint is not None and not tx_progress._PROGRESS_ID_RE.match(_plan_hint):
                    _plan_hint = None   # malformed → derive one, never a 422
                _plan = preload.register_plan(
                    _user_id, _preload_entries, plan_id=_plan_hint,
                    trigger="job")
                tx_progress._PLAN_BY_PID[_pid] = _plan["plan_id"]

            # Now that the stage plan is resolved, tell the running-jobs
            # registry what this job is going to DO. job_start fires before
            # the plan exists, so /stats showed a bare "—" for the whole life
            # of a job that was about to separate, diarize and translate.
            _plan_names = [n for n, on in (("separate", _separate),
                                           ("diarize", _diarize),
                                           ("translate", bool(_translate_to)))
                           if on]
            if _plan_names:
                _job_detail = " + ".join(_plan_names)
                if _translate_to:
                    _job_detail += f" → {','.join(_translate_to)}"
                jobs.job_update(request_id, detail=_job_detail)

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
            # Live-dictation keys are not batch knobs at all: a locked one was
            # never going to apply here, so it is not reported either.
            ignored.extend(sorted(
                k for k in _overrides if k in ident.locked_client_keys
                and k not in settings_schema.STREAM_ONLY_CLIENT_KEYS))
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
                _locked_ladder = effective_config.cfg_for(resolved_model, "TEMPERATURE", ident)
                # Same parse as the assembler: a blank, token-less (",") or
                # unparseable ladder all leave the Form field in force.
                if not tx_models._temperature_ladder(_locked_ladder):
                    if temperature != _t_lo and "temperature" not in ignored:
                        ignored.append("temperature")
                    _temperature = _t_lo
            # Single source of truth — the streaming FINAL decode builds its kwargs
            # from this exact assembler too, so streaming and batch never diverge.
            transcribe_kwargs = tx_models.assemble_transcribe_kwargs(
                resolved_model, model,
                language=_decode_language, temperature=_temperature,
                vad_filter=_vad_filter, vad_parameters=vad_parameters,
                want_word_ts=want_word_ts, initial_prompt=initial_prompt_arg,
                overrides=_overrides, ident=ident, task=_task,
            )
            tx_models._note_auto_detect_only(_overrides, transcribe_kwargs.get("language"),
                                   ignored)
            tx_models._note_word_ts_only(_overrides, want_word_ts, ignored)

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
                    _skip("separating")
                else:
                    try:
                        _sep_t0 = time.perf_counter()
                        _cur_stage = "separating"
                        tx_progress._progress_set(
                            _pid, stage="separating", progress=None,
                            position=None, last_text=None,
                            model=(_separation_model or None),
                            # The ONNX session's real placement once a model
                            # is loaded (a CUDA provider that fails to load
                            # falls back to CPU silently); the config-resolved
                            # device only before the first load.
                            device=(_bgm_separation.actual_device()
                                    or _bgm_separation._resolve_device()))
                        tx_progress._check_cancelled(_pid)
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
                                _tfd, _sep_wav = tempfile.mkstemp(
                                    prefix="sepsrc-", suffix=".wav")
                                os.close(_tfd)
                                _tc0 = time.perf_counter()
                                tx_progress._progress_set(_pid, step="preparing")
                                await asyncio.to_thread(
                                    _transcode.transcode_to_wav, tmp_path,
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
                            tx_progress._check_cancelled(_pid)
                            async with tx_models.get_inference_semaphore():
                                tx_progress._check_cancelled(_pid)
                                # "preparing" stays up through model load and
                                # the separator's own audio load/normalize
                                # (~40 s on long inputs); the first demix
                                # chunk clears it via the progress callback.
                                tx_progress._progress_set(_pid, step="preparing")
                                _vocals_path = await _bgm_separation.separate(
                                    _sep_src,
                                    model_filename=(_separation_model or None),
                                    progress_cb=lambda f: tx_progress._progress_set(
                                        _pid, progress=f, step=None),
                                    cancel_check=lambda: tx_progress._cancel_requested(
                                        _pid))
                        finally:
                            tx_progress._progress_set(_pid, step=None)
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
                            "detail": ("incl. transcode"
                                       if _sep_wav is not None else None),
                            **tx_receipt._stage_extras(
                                preload.stats_key("separation",
                                                  _separation_model or ""),
                                _sep_t0),
                        })
                        _rplan.stage_done("separating")
                    except tx_progress._ClientCancelled:
                        raise
                    except _bgm_separation.BgmCancelled:
                        raise tx_progress._ClientCancelled() from None
                    except _bgm_separation.BgmSeparationError as _se:
                        # str(_se) is client-safe by the module's contract.
                        _warnings.append(str(_se))
                        _stage_timings.append(tx_receipt._failed_stage(
                            "separating", _sep_t0, _separation_model, _se))
                        _rplan.stage_failed("separating")
                    except Exception as _se:  # noqa: BLE001 — soft-fail
                        logger.error("[bgm] unexpected failure: %s",
                                     _log_safe(str(_se)))
                        _warnings.append(
                            "music separation failed; transcribing the "
                            "original audio")
                        _stage_timings.append(tx_receipt._failed_stage(
                            "separating", _sep_t0, _separation_model, _se))
                        _rplan.stage_failed("separating")

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
            # transcribe() decodes the audio and runs Silero EAGERLY, before
            # it hands back the segment generator (see _collect's note). That
            # makes the pre-roll separately measurable from the decode itself,
            # which is the only reason VAD can be a real stage row rather than
            # a footnote. Written once from the executor thread; a plain dict
            # write is GIL-atomic, same as the _progress_set traffic below.
            _decode_timing: dict = {}
            # Residual-window stop (DECODE_SKIP_RESIDUAL_WINDOWS): refuse the
            # sub-second leftover faster-whisper re-decodes after the last
            # word of a window that already reached the end of the audio —
            # the temperature ladder loops there for tens of seconds and the
            # result is dropped anyway. Resolved here (event loop) like every
            # other cfg_for read; the executor thread only carries the bool.
            _skip_residual = bool(effective_config.cfg_for(
                resolved_model, "DECODE_SKIP_RESIDUAL_WINDOWS", ident))
            # Per-rung token limit scaled to the window's length: a decode
            # that loops otherwise runs to the model's hard limit at ~83 ms a
            # token (transcription/decode_trace.py, "Token cap"). 0 = off.
            _token_cap = float(effective_config.cfg_for(
                resolved_model, "DECODE_TOKEN_CAP_PER_SECOND", ident) or 0.0)

            def _do_transcribe(_model=model, _path=tmp_path,
                               _kw=transcribe_kwargs, _pad_ms=_lead_pad_ms,
                               _t=_decode_timing, _skip=_skip_residual,
                               _cap=_token_cap):
                # Materialize the lazy segment generator WITH live progress:
                # each yielded segment carries its end time, and info.duration
                # is known up front — that ratio is genuine decode progress
                # (the executor thread's dict writes are GIL-atomic).
                def _collect(_gen, _info):
                    _dur = float(getattr(_info, "duration", 0.0) or 0.0)
                    _compute, _dev = tx_receipt._model_compute_device(resolved_model)
                    # VAD receipt: transcribe() ran Silero eagerly before
                    # returning, so duration_after_vad is already known here.
                    # Only meaningful when the filter actually ran.
                    _dav = getattr(_info, "duration_after_vad", None)
                    _retained = (
                        max(0.0, min(1.0, float(_dav) / _dur))
                        if _kw.get("vad_filter") and _dur > 0 and _dav is not None
                        else None)
                    tx_progress._progress_set(_pid, stage="transcribing", progress=0.0,
                                  duration=_dur or None, position=None,
                                  last_text=None, model=resolved_model,
                                  device=_dev, compute=_compute,
                                  vad_retained=_retained)
                    _rplan.set_audio_seconds(_dur, src="decoder")
                    _rplan.set_vad_retained(_retained)
                    _out = []
                    _log_bucket = 0  # 5%-step INFO trail, like the other stages
                    try:
                        for _s in _gen:
                            # Cooperative cancel between decoded segments —
                            # this executor thread is the only thing that can
                            # stop a cancelled request's decode.
                            if tx_progress._cancel_requested(_pid):
                                raise tx_progress._ClientCancelled()
                            _out.append(_s)
                            if _dur > 0:
                                _frac = min(1.0, float(_s.end) / _dur)
                                tx_progress._progress_set(
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
                    except _decode_trace.ResidualWindowSkipped:
                        # The stop rule ended the stream after the window that
                        # reached the end of the audio; everything yielded so
                        # far is the complete output.
                        pass
                    return _out
                # This executor thread has the semaphore slot now: everything
                # until _collect's first entry (lead-pad decode, transcribe()'s
                # eager audio decode + Silero VAD pass) used to be misreported
                # as "waiting". Own stage so the client can label it honestly.
                tx_progress._progress_set(_pid, stage="analyzing", progress=None,
                              position=None, last_text=None, step=None,
                              model=None, device=None, compute=None)
                # One origin for both branches, taken BEFORE the lead-pad
                # pre-decode, so the "vad" row bills decode + pad + Silero
                # as its own detail claims (a padded ndarray input skips
                # transcribe()'s decode; without this the decode cost slid
                # into the "transcribing" row).
                _pre = time.perf_counter()
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
                # Decode trace (windows / rungs / tokens) for the receipt:
                # the lazy generator is consumed inside the capture, that is
                # where every window after the first is decoded.
                with _decode_trace.capture(_kw, skip_residual=_skip,
                                           token_cap_per_s=_cap) as _tr:
                    if _audio is not None:
                        _segs, _info = _model.transcribe(_audio, **_kw)
                        _t["pre_secs"] = time.perf_counter() - _pre
                        _out = _collect(_segs, _info)
                        _t["trace"] = _decode_trace.finish(_tr, _out, _info)
                        return (*_shift_to_original_timeline(
                            _out, _info, _pad_ms / 1000.0), True)
                    _segs, _info = _model.transcribe(_path, **_kw)
                    _t["pre_secs"] = time.perf_counter() - _pre
                    _out = _collect(_segs, _info)
                    _t["trace"] = _decode_trace.finish(_tr, _out, _info)
                    return _out, _info, False
            loop = asyncio.get_running_loop()
            tx_progress._progress_set(_pid, stage="waiting", progress=None,
                          position=None, last_text=None, step=None,
                          model=None, device=None, compute=None)
            tx_progress._check_cancelled(_pid)
            async with tx_models.get_inference_semaphore():
                tx_progress._check_cancelled(_pid)
                _dec_t0 = time.perf_counter()
                # The executor decodes (av) then transcribes; an av error is
                # told apart by its module, not by this marker.
                _cur_stage = "transcribing"
                segments_iter, info, _pad_applied = await loop.run_in_executor(
                    None, _do_transcribe)
                _total_secs = time.perf_counter() - _dec_t0
                # VAD is the stage that most often eats the user's audio and
                # it was the only one with no row anywhere — /stats, the
                # receipt and the trace viewer were all blind to it. Its cost
                # is the eager pre-roll, so bill it separately and leave the
                # decode row showing decode only, rather than double-counting.
                _pre_secs = float(_decode_timing.get("pre_secs") or 0.0)
                if transcribe_kwargs.get("vad_filter") and _pre_secs > 0:
                    _dav_v = getattr(info, "duration_after_vad", None)
                    _dur_v = float(getattr(info, "duration", 0.0) or 0.0)
                    _detail = "audio decode + Silero"
                    # Kept-audio fraction, structured for the usage
                    # statistics (the detail string is for the receipt).
                    _retained = None
                    if _dav_v is not None and _dur_v > 0:
                        _retained = float(_dav_v) / _dur_v
                        _detail += (f" · {float(_dav_v):.2f}s kept of "
                                    f"{_dur_v:.2f}s "
                                    f"({_retained * 100:.0f} %)")
                    _stage_timings.append({
                        "name": "vad",
                        "secs": round(_pre_secs, 2),
                        "model": "silero",
                        "device": "cpu",
                        "detail": _detail,
                        "retained": _retained,
                    })
                    _total_secs = max(0.0, _total_secs - _pre_secs)
                # Measured from the load's own origin (not _dec_t0) so a cold
                # whisper load shows up as this row's `load`; it is folded into
                # `secs` too so `run = secs - load` and the wall total both hold.
                _rplan.stage_done("transcribing")
                _tr_extras = tx_receipt._stage_extras(resolved_model, _model_t0)
                _stage_timings.append({
                    "name": "transcribing",
                    "secs": round(
                        _total_secs + float(_tr_extras.get("load_secs") or 0.0),
                        2),
                    "model": resolved_model,
                    **_tr_extras,
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

            # Post-decode word-rate guard (SEGMENT_MAX_WORDS_PER_S): drops
            # hallucinated echo segments — see segment_exceeds_word_rate.
            _max_wps = float(effective_config.cfg_for(resolved_model, "SEGMENT_MAX_WORDS_PER_S", ident) or 0)
            # Tail cuts inside a segment (transcription/segment_guards.py). They run
            # AFTER the whole-segment verdict, on the survivors: a segment made
            # up from start to end is still dropped whole rather than trimmed to
            # two garbage words.
            _tail_limits = tx_guards.tail_guard_limits(resolved_model, ident)
            _tail_cuts: list[dict] = []
            # Head cut (SEGMENT_HEAD_ECHO_MIN_WORDS): an echo of the prompt's
            # last words at the start of the decode. Only the first surviving
            # segment can carry it — the prompt precedes the first window — and
            # it runs before the tail cuts, on the same survivor.
            _head_min = tx_guards.head_echo_min_words(resolved_model, ident)
            _head_prompt = segment_guards.prompt_tail_text(transcribe_kwargs)
            _head_pending = bool(_head_min and _head_prompt)
            _head_cut: "dict | None" = None

            for i, segment in enumerate(segments_iter):
                # segment.temperature reflects CT2's actual after-fallback
                # value (may differ from the request `temperature` if fallback
                # kicked in); the fallback is `_temperature`, the value that
                # was actually handed to the decoder (a locked TEMPERATURE may
                # have reset it). segment.compression_ratio is the real gzip
                # ratio used by the suppression check — was previously
                # hardcoded 1.0.
                seg_temp = getattr(segment, "temperature", _temperature)
                seg_cr = getattr(segment, "compression_ratio", 1.0)

                dropped = tx_guards.segment_exceeds_word_rate(segment, _max_wps)
                _cut = None
                _hcut = None
                if not dropped and _head_pending:
                    _head_pending = False
                    _hcut = segment_guards.apply_head_echo_guard(
                        segment, _head_prompt, _head_min)
                    if _hcut is None:
                        _hw = segment_guards.head_words_diag(
                            segment, _head_prompt, _head_min)
                        if _hw:
                            logger.info("[transcribe] head_words (prompt repeated, "
                                        "looks spoken, nothing cut): %s", _hw)
                    _head_cut = _hcut
                if not dropped and (_hcut is None or (segment.text or "").strip()):
                    # Cuts words / text / end IN PLACE, so every consumer below
                    # (diag row, segments, words, joined text, capture) carries
                    # the cut version.
                    _cut = segment_guards.apply_tail_guards(segment, **_tail_limits)
                _emptied = ((_cut is not None or _hcut is not None)
                            and not (segment.text or "").strip())
                seg_diag.append({
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "alp": segment.avg_logprob,
                    "nsp": segment.no_speech_prob,
                    "cr": seg_cr,
                    "temp": seg_temp,
                    # An emptied row shows what was removed, not "".
                    "text": ((_hcut or {}).get("text", "") + (_cut or {}).get("text", "")
                             if _emptied else segment.text),
                    # An emptied segment counts as dropped (speaker alignment
                    # hands one label to every kept row).
                    "dropped": dropped or _emptied,
                    **({"cut": _cut} if _cut else {}),
                    **({"head_cut": _hcut} if _hcut else {}),
                })
                if _hcut:
                    # "emptied" is counted once per segment: here only when the
                    # head cut alone left nothing (the tail cut then never ran).
                    tx_guards.record_tail_cut(_hcut, emptied=_emptied and _cut is None)
                    logger.info(
                        "[transcribe] cut prompt echo from the start of the first "
                        "segment (%d words %.2f-%.2fs): %r",
                        _hcut["n"], _hcut["from"], _hcut["to"], _hcut["text"])
                    if _emptied and _cut is None:
                        continue
                if _cut:
                    _tail_cuts.append(_cut)
                    tx_guards.record_tail_cut(_cut, emptied=_emptied)
                    logger.info(
                        "[transcribe] cut made-up tail (%s, %d words%s): %r",
                        "+".join(_cut["rules"]), _cut["n"],
                        "" if _cut.get("from") is None else f" from {_cut['from']:.2f}s",
                        _cut["text"])
                    if _emptied:
                        continue
                elif not dropped:
                    _tw = segment_guards.tail_words_diag(segment)
                    if _tw:
                        logger.info("[transcribe] tail_words (zero-length word, "
                                    "nothing cut): %s", _tw)
                if dropped:
                    metrics.record_guard_hit("word_rate")
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
            # The translation estimate scales with the segment count, and
            # the detected language decides which targets are verbatim.
            _rplan.set_segments(len(segments_list))
            _rplan.mark_instant(getattr(info, "language", None) or None)
            speakers_list: "list[str]" = []
            if _diarize and segments_list:
                if not getattr(cfg, "DIARIZATION_ENABLED", False):
                    _warnings.append(
                        "diarization requested but not enabled on this "
                        "server (DIARIZATION_ENABLED is off)")
                    _skip("diarizing")
                else:
                    try:
                        _diar_t0 = time.perf_counter()
                        _cur_stage = "diarizing"
                        tx_progress._progress_set(
                            _pid, stage="diarizing", progress=None,
                            position=None, last_text=None, step=None,
                            model=(_diarization_model or None),
                            device=_diarization._resolve_device())
                        tx_progress._check_cancelled(_pid)
                        async with tx_models.get_inference_semaphore():
                            tx_progress._check_cancelled(_pid)
                            _turns = await _diarization.diarize(
                                tmp_path,
                                num_speakers=_spk.get("num_speakers"),
                                min_speakers=_spk.get("min_speakers"),
                                max_speakers=_spk.get("max_speakers"),
                                model_id=(_diarization_model or None),
                                progress_cb=lambda f, step=None, **kw: tx_progress._progress_set(
                                    _pid, progress=f, step=step,
                                    target=kw.get("target"),
                                    target_progress=kw.get("target_progress")),
                                cancel_check=lambda: tx_progress._cancel_requested(_pid),
                            )
                        # Pure-Python O(segments × turns) — off the loop so a
                        # long file doesn't stall every other request.
                        speakers_list = await asyncio.to_thread(
                            _diarization.assign_speakers, segments_list, _turns)
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
                            "speakers": len(speakers_list),
                            **tx_receipt._stage_extras(
                                preload.stats_key("diarization",
                                                  _diarization_model or ""),
                                _diar_t0),
                        })
                        _rplan.stage_done("diarizing")
                    except tx_progress._ClientCancelled:
                        raise
                    except _diarization.DiarizeCancelled:
                        raise tx_progress._ClientCancelled() from None
                    except _diarization.DiarizationError as _de:
                        # str(_de) is client-safe by the module's contract.
                        _warnings.append(str(_de))
                        _stage_timings.append(tx_receipt._failed_stage(
                            "diarizing", _diar_t0, _diarization_model, _de))
                        _rplan.stage_failed("diarizing")
                    except Exception as _de:  # noqa: BLE001 — soft-fail
                        logger.error("[diarize] unexpected failure: %s",
                                     _log_safe(str(_de)))
                        _warnings.append(
                            "diarization failed; the transcript has no "
                            "speaker labels")
                        _stage_timings.append(tx_receipt._failed_stage(
                            "diarizing", _diar_t0, _diarization_model, _de))
                        _rplan.stage_failed("diarizing")
            elif _diarize:
                _warnings.append("diarization skipped: no speech segments")
                _skip("diarizing")

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
                    _skip("translating")
                else:
                    # Empty request/config model resolves to the server
                    # default at stage time (a live admin edit applies).
                    # The allowlist gate (shared helper) constrains only the
                    # CLIENT-requested value — an admin-pinned per-model/
                    # per-identity TRANSLATION_MODEL is policy and passes,
                    # exactly like the diarization/separation gates above.
                    _tr_model = ((_translation_model or "").strip()
                                 or tr_gating._translation_default_model())
                    if not tr_gating._translation_model_allowed(
                            _tr_model, requested=_tm_req,
                            inherited=_tm_inherited):
                        # Soft-fail like the enabled gate — never a 4xx after
                        # the transcript already exists.
                        _warnings.append(
                            "requested translation model is not in "
                            "TRANSLATION_ALLOWED_MODELS on this server")
                        _skip("translating")
                    else:
                        try:
                            _tr_t0 = time.perf_counter()
                            _cur_stage = "translating"
                            tx_progress._check_cancelled(_pid)
                            tx_progress._progress_set(
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
                                        last_text=None, target=None,
                                        target_progress=None:
                                        tx_progress._progress_set(
                                            _pid, stage="translating",
                                            model=(_tr_model or None),
                                            compute="gguf",
                                            progress=f, step=step,
                                            target=target,
                                            target_progress=target_progress,
                                            **({"last_text": last_text}
                                               if last_text else {})),
                                    cancel_check=lambda:
                                        tx_progress._cancel_requested(_pid),
                                    download_cb=lambda done, total:
                                        tx_progress._progress_set(
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
                                async with tx_models.get_inference_semaphore():
                                    tx_progress._check_cancelled(_pid)
                                    _per_seg, _tr_warn, _tr_meta = \
                                        await _run_translation()
                            else:
                                _per_seg, _tr_warn, _tr_meta = \
                                    await _run_translation()
                            _warnings.extend(_tr_warn)
                            _tr_kept = _tr_meta.get("kept") or {}
                            # target -> sorted segment indices whose guard
                            # fallback kept the SOURCE text. Rides in the
                            # shared meta so the default `json` shape (no
                            # segment rows) can tell a kept original from a
                            # real translation; {} when clean.
                            _kept_by_lang: "dict[str, list[int]]" = {}
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
                                    for _lang in _tr_kept[_i]:
                                        _kept_by_lang.setdefault(
                                            _lang, []).append(_i)
                            _translation_meta = {
                                "model": _tr_meta.get("model"),
                                "targets": list(_translate_to),
                                "source": _tr_meta.get("source"),
                                "mode": _tr_meta.get("mode"),
                                "kept": {_l: sorted(_ix) for _l, _ix
                                         in _kept_by_lang.items()},
                            }
                            _rplan.stage_done("translating")
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
                                "targets": list(_translate_to),
                                # Segments whose guard fallback kept the
                                # source text in at least one target.
                                "kept_original": sum(
                                    1 for _k in _tr_kept.values() if _k),
                                **tx_receipt._stage_extras(
                                    preload.stats_key(
                                        "translation",
                                        _tr_meta.get("model") or ""),
                                    _tr_t0),
                            })
                        except tx_progress._ClientCancelled:
                            raise
                        except _tr.TranslationCancelled:
                            raise tx_progress._ClientCancelled() from None
                        except _tr.TranslationError as _te:
                            # str(_te) is client-safe by the module's contract.
                            _warnings.append(str(_te))
                            _stage_timings.append(tx_receipt._failed_stage(
                                "translating", _tr_t0, _tr_model, _te))
                            _rplan.stage_failed("translating")
                        except Exception as _te:  # noqa: BLE001 — soft-fail
                            logger.error("[translate] unexpected failure: %s",
                                         _log_safe(str(_te)))
                            _warnings.append(
                                "translation failed; the transcript is "
                                "untranslated")
                            _stage_timings.append(tx_receipt._failed_stage(
                                "translating", _tr_t0, _tr_model, _te))
                            _rplan.stage_failed("translating")
            elif _translate_to:
                _warnings.append("translation skipped: no speech segments")
                _skip("translating")

            raw_full_text = "".join(raw_full_text_parts)
            trace: "list | None" = [] if cfg.TRACE_ENABLED else None
            _detected_lang = getattr(info, "language", None)
            full_text_str = pl_engine._postprocess_text(raw_full_text, model_name=resolved_model, trace=trace, ident=ident, language=_detected_lang)
            # Captures-form text — same pipeline minus the captures-specific
            # exclude set (default-skips `de-dictation-map` + `capitalize-after-
            # terminator` so the stored text matches Whisper's raw output
            # under SUPPRESS_CHARS for fine-tune training). Only computed when
            # the capture eligibility gate has already passed at handler
            # entry — sampling missed / captures disabled / count cap full
            # are the common case and shouldn't pay for a second pipeline
            # walk per request. No trace participation: the runtime trace
            # describes the user-facing pipeline, not the training-form
            # variant.
            if will_capture:
                training_text_str = pl_engine._postprocess_text(
                    raw_full_text,
                    model_name=resolved_model,
                    trace=None,
                    extra_excludes=cfg.CAPTURES_PIPELINE_RULES_EXCLUDE,
                    ident=ident,
                    language=_detected_lang,
                )
            # Per-language transcripts for the capture row, joined the same
            # way verbose_json joins them. Kept as a KEYED map, not a blob:
            # only the English track is ever eligible as Whisper training
            # data (its translate task targets English and nothing else), so
            # the exporter has to be able to pick one language out.
            # Segments whose guard fallback KEPT the source text for a target
            # (translations_kept) are skipped for that target: the response
            # flags them per segment, but the stored track has no such
            # marker, so a kept original would be exported as translated
            # text — source-language audio labelled task=translate.
            _capture_translations: "dict[str, str] | None" = None
            if will_capture and _translation_meta is not None:
                _capture_translations = {
                    _lang: " ".join(
                        _s for _s in (
                            (seg.get("translations") or {}).get(_lang, "").strip()
                            for seg in segments_list
                            if _lang not in (seg.get("translations_kept") or ()))
                        if _s).strip()
                    for _lang in _translation_meta["targets"]
                }
                _capture_translations = {
                    k: v for k, v in _capture_translations.items() if v}
            # Output wrappers (G/PM): plain prefix/suffix concatenated to
            # the final transcript text after the pipeline runs (including
            # the in-pipeline terminal trim) and BEFORE a defensive
            # post-wrapper trim. Per-model overrides win; the client's
            # output_prefix / output_suffix win over both unless locked.
            _output_prefix, _output_suffix = tx_models._output_wrappers(
                resolved_model, ident, _overrides)
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
                    audio_dur_s = float(getattr(info, "duration", 0.0) or 0.0)
                    min_s = float(getattr(cfg, "CAPTURES_RECORDING_MIN_DURATION_S", 0.5))
                    max_s = float(getattr(cfg, "CAPTURES_RECORDING_MAX_DURATION_S", 600.0))
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
                                    captures_store.create_capture,
                                    audio_src_path=tmp_path,
                                    request_id=request_id,
                                    model=resolved_model,
                                    language=info.language,
                                    audio_s=audio_dur_s,
                                    raw=raw_full_text,
                                    final=full_text_str,
                                    text_for_training=training_text_str,
                                    words=all_words,
                                    segments=seg_diag,
                                    user_id=user.get("user_id"),
                                    # _task has existed since the translate
                                    # endpoint landed and is echoed in the
                                    # response, but never reached the store —
                                    # so the finetuning manifest could not say
                                    # whether a row trains transcribe or
                                    # translate.
                                    task=_task,
                                    translations=_capture_translations,
                                    translation_model=(
                                        _translation_meta.get("model")
                                        if _translation_meta else None),
                                    translation_source=(
                                        "cascade-mt" if _translation_meta
                                        else None),
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
                _url_host = media_video._url_host_for_log(source_url)
                # The downloaded container, not tmp_path — BGM separation
                # swaps tmp_path for the vocals-only WAV.
                _src_fmt = _log_safe(
                    os.path.splitext(_dl_path or "")[1].lstrip(".") or "audio")
                _file_label = (f"url:{_url_host}  ({audio_bytes/1024:.1f} KB, "
                               f"{_log_safe(response_format)})")
                _audio_src_label = (
                    f"{_src_fmt} → 16 kHz mono (url download via yt-dlp"
                    + (", reused from the language check" if _reused_copy else ""))
            else:
                _src_fmt = _log_safe(file.content_type
                                     or os.path.splitext(file.filename or "")[1].lstrip(".")
                                     or "audio")
                _file_label = (f"{_log_safe(file.filename)}  ({audio_bytes/1024:.1f} KB, "
                               f"{_log_safe(response_format)})")
                _audio_src_label = f"{_src_fmt} → 16 kHz mono (file upload"
            logger.info(tx_receipt._format_request_block(
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
                guards={"segment_max_words_per_sec": _max_wps,
                        **tx_guards.tail_guard_rows(_tail_limits),
                        **tx_guards.tail_cut_rows(_tail_cuts),
                        **tx_guards.head_echo_rows(_head_min, _head_cut),
                        "skip_residual_windows": _skip_residual,
                        "token_cap_per_second": _token_cap},
                decode_trace=_decode_timing.get("trace"),
                # Post-decode pipeline. Reconstructed from the locals in
                # scope rather than from preload._plans: the plan omits
                # whisper (loaded before the plan exists) and is absent
                # entirely when there is no progress id.
                stages=_stage_timings or None,
                separation=({
                    "model": _separation_model or None,
                    "device": tx_receipt._stage_field(_stage_timings, "separating", "device"),
                    "resample": "44100 Hz stereo",
                    "stem": "vocals",
                } if tx_receipt._stage_ran(_stage_timings, "separating") else None),
                diarization=({
                    "model": _diarization_model or None,
                    "device": tx_receipt._stage_field(_stage_timings, "diarizing", "device"),
                    "num_speakers": _spk.get("num_speakers"),
                    "min_speakers": _spk.get("min_speakers"),
                    "max_speakers": _spk.get("max_speakers"),
                    "embedding_batch_size": getattr(
                        cfg, "DIARIZATION_EMBEDDING_BATCH_SIZE", tx_receipt._OMIT),
                    "result": (f"{len(speakers_list)} speakers across "
                               f"{len(segments_list)} segments"),
                } if tx_receipt._stage_ran(_stage_timings, "diarizing") else None),
                translation=({
                    "model": _translation_meta.get("model"),
                    "device": tx_receipt._stage_field(_stage_timings, "translating", "device"),
                    "targets": list(_translation_meta.get("targets") or []),
                    "source": _translation_meta.get("source") or tx_receipt._OMIT,
                    "mode": _translation_meta.get("mode"),
                    "context_segments": _translation_context,
                    "glossary": (f"{len(_translation_glossary)} chars"
                                 if _translation_glossary else tx_receipt._OMIT),
                    "result": f"{len(segments_list)} segs",
                } if _translation_meta else None),
                # Per-SEGMENT labels (assign_speakers stamped them on the
                # dicts), not the distinct `speakers_list` — the align helper
                # consumes one label per kept seg_diag row.
                speakers=tx_receipt._align_speakers_to_diag(
                    seg_diag,
                    [str(s.get("speaker") or "") for s in segments_list]
                    if speakers_list else None),
                warnings=_warnings or None,
                skipped=_skipped or None,
            ))

            # Persist the trace to the durable recent-transcriptions store
            # (SQLite, WAL) and broadcast it to /quick-config SSE
            # subscribers in one step. metrics.record_transcription() in the
            # outer finally adds the timing half via UPSERT on the same
            # request_id.
            try:
                qc_recent_feed.record_trace(
                    request_id=request_id,
                    model=resolved_model,
                    raw=raw_full_text,
                    steps=trace if trace is not None else [],
                    final=full_text_str,
                    language=info.language,
                    # A URL fetch and an uploaded file are different jobs, and
                    # this branch already knows which one ran (see the
                    # url:<host> label built for the request block above). The
                    # distinction used to reach the log and stop there, so
                    # /quick-config chipped every URL download as "file".
                    source="url" if source_url is not None else "file",
                    user_id=_user_id,
                )
            except Exception as _qc_err:
                logger.error("[quick-config] record_trace failed: %s", _qc_err)

            _audio_dur = float(info.duration)
            _language = getattr(info, "language", None) or None
            # Word count from the final post-processed text — matches what the
            # client actually receives. Counting len(all_words) instead would
            # yield 0 whenever WORD_TIMESTAMPS_ENABLED is off or the request
            # didn't ask for word-level granularity (the common case).
            _words = len(full_text_str.split())

            if response_format == "text":
                _response_payload = full_text_str
                return full_text_str

            # Joined per-language transcripts, shared by verbose_json AND the
            # default json shape below. Deliberately NOT run through
            # _postprocess_text: the pipeline's rules are German-dictation-
            # shaped (de-dictation-map, punctuation words) and would mangle
            # translated text.
            _translations_joined = ({
                _lang: " ".join(
                    _s for _s in (
                        (seg.get("translations") or {})
                        .get(_lang, "").strip()
                        for seg in segments_list)
                    if _s).strip()
                for _lang in _translation_meta["targets"]
            } if _translation_meta is not None else None)

            if _retained_upload is not None:
                # Only a finished run retains its upload; the id rides on the
                # same keys a link run uses (the export panel reads one).
                _source_media_id = await asyncio.to_thread(
                    url_media_store.register, _retained_upload, user_id=_user_id,
                    kind="video")
                if _source_media_id is not None:
                    # register() moved the file; on a refusal (None) it may
                    # still be on disk, and the inner finally unlinks it.
                    _retained_upload = None

            if response_format == "verbose_json":
                response = {
                    "task": _task,
                    "language": info.language,
                    "duration": info.duration,
                    "text": full_text_str,
                    "segments": segments_list,
                }
                # The run plan's receipt: every stage's took_s and the
                # per-language units. The progress entry is popped before
                # this response leaves, so the last poll never sees the
                # final unit finish — the receipt is the only complete copy.
                response["plan"] = _rplan.snapshot()["plan"]
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
                    # `speakers` gives labels but never said WHICH pipeline
                    # produced them — unlike translation, which has always
                    # reported its model. Same shape, so a client can render
                    # both provenances identically.
                    response["diarization"] = {
                        "model": _diarization_model or None,
                        "speakers": len(set(speakers_list)),
                    }
                # Separation left no trace in the response at all, so a client
                # could not tell a vocals-only transcript from a raw one.
                if tx_receipt._stage_ran(_stage_timings, "separating"):
                    response["separation"] = {
                        "model": _separation_model or None,
                        "stem": "vocals",
                    }
                # Per-stage timings were built for the log and the client's
                # progress rail and then dropped from the response, so a
                # caller could see WHAT ran but never what it cost.
                if _stage_timings:
                    response["stages"] = _stage_timings
                if _translations_joined is not None:
                    response["translations"] = _translations_joined
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
                response.update(media_video._video_response_keys(_video_task, _video_result))
                if _source_media_id is not None:
                    response["source_media_id"] = _source_media_id
                    response["source_media_expires_at"] = (
                        url_media_store.expires_at_unix(_source_media_id))
                _response_payload = response
                return response

            # Default `json` shape. Additive keys only (OpenAI-compat callers
            # ignore them, exactly like source_media_id below): a caller that
            # paid for translation/diarization must not need verbose_json to
            # see the output — or the soft-fail warnings explaining why an
            # optional stage silently didn't run.
            response = {"text": full_text_str}
            if _translations_joined is not None:
                response["translations"] = _translations_joined
                response["translation"] = _translation_meta
            if speakers_list:
                response["speakers"] = speakers_list
            if _warnings:
                response["warnings"] = _warnings
            # Same conditions as the verbose_json branch: an ignored override
            # is never silently dropped, and profile_applied echoes only when
            # a profile was asked for.
            if ignored:
                response["overrides_ignored"] = ignored
            if override_profile:
                response["profile_applied"] = ident.request_profile_applied
            response.update(media_video._video_response_keys(_video_task, _video_result))
            if _source_media_id is not None:
                response["source_media_id"] = _source_media_id
                response["source_media_expires_at"] = (
                    url_media_store.expires_at_unix(_source_media_id))
            _response_payload = response
            return response

        except tx_progress._ClientCancelled:
            # The client asked (via the cancel endpoint) to abort. Not an
            # error — the stages stopped cooperatively; the response status
            # is moot (the caller usually dropped the connection already).
            _status = "cancelled"
            logger.info("[batch] transcription cancelled by client")
            raise HTTPException(status_code=499,
                                detail="cancelled by the client")
        except HTTPException as _he:
            # Preserve curated HTTP errors (e.g. an allowed-models 400) with
            # their status + message intact — only unexpected errors below are
            # genericised.
            _status = "error"
            _exc = _he
            raise
        except Exception as e:
            _status = "error"
            _exc = e
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
                if _video_task is not None and not _video_task.done():
                    # The client keeps polling this id for `video.state`
                    # (and may still cancel it); the task's finally pops
                    # the entry when it ends.
                    _run_finished[0] = True
                else:
                    tx_progress._progress_close(_pid)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if _retained_upload:
                try:
                    os.unlink(_retained_upload)
                except OSError:
                    pass
            # URL flow: the private download dir (partials, fragments) goes
            # on every path — cancel, 4xx, 500 included. The retained copy
            # (url_media_store) has its own TTL lifecycle.
            if _url_job_dir:
                shutil.rmtree(_url_job_dir, ignore_errors=True)

    except asyncio.CancelledError:
        # A dropped connection (and lifespan shutdown) unwinds this handler
        # as a BaseException, past every arm below; without this the finally
        # would persist the run as status="ok" with zero words. Same
        # reasoning as translation/routes.py translate_text's finally.
        if _status != "cancelled":
            _status = "cancelled"
        raise
    except Exception as _oe:
        # Catches failures BEFORE the inner try (e.g. _get_or_load_model
        # raising HTTPException, await request.form() blowing up), which
        # previously bypassed `_status = "error"` and inflated the success
        # counter with failed requests. The inner _ClientCancelled arm's 499
        # also lands here (its own sibling arms can't see it) — that one is
        # NOT an error, so "cancelled" must survive into the recorded row.
        if _status != "cancelled":
            _status = "error"
        if _exc is None:
            _exc = _oe
        raise
    finally:
        if _video_task is not None and _status != "ok" and not _video_task.done():
            _video_task.cancel()
        if _leased_model is not None:
            tx_models._release_model_lease(_leased_model)
        metrics.in_flight_transcriptions -= 1
        if _pid:
            tx_progress._JOB_BY_PID.pop(_pid, None)
            # The plan itself is NOT cancelled here: its warm leases are what
            # keep the models this job just used alive for the next one, and
            # the TTL retires them on its own.
            tx_progress._PLAN_BY_PID.pop(_pid, None)
            tx_progress._RUN_PLAN_BY_PID.pop(_pid, None)
        jobs.job_end(request_id)
        if _status != "ok" and _error_class is None:
            _error_class, _error_stage = metrics.classify_error(
                _exc, status=_status, stage=_cur_stage)
        metrics.record_transcription(
            model=resolved_model,
            audio_dur=_audio_dur,
            proc_dur=time.perf_counter() - _t0,
            status=_status,
            words=_words,
            error_class=_error_class,
            error_stage=_error_stage,
            request_id=request_id,
            user_id=_user_id,
            key_id=_key_id,
            username=user.get("username"),
            key_label=user.get("key_label"),
            stages=_stage_timings or None,
            # The client's progress id names this run on its side too, so
            # the usage job row is addressable by both ends.
            job_id=_pid or request_id,
            usage_kind="url" if source_url is not None else "file",
            language=_language,
            wait_s=metrics.take_wait(),
        )
        # The awaits come LAST: a cancellation landing on one of them (a
        # second cancel at shutdown) escapes this finally, and nothing that
        # must never be skipped may sit behind it. The job row first — its
        # write is shielded and outlives the cancel.
        if _job_row:
            await tx_progress._jobs_finish(
                _pid, status=_status,
                payload=(_response_payload if _status == "ok" else None),
                error=tx_progress._job_error_text(_status, _exc),
                stages=(_stage_timings or None),
                plan=_rplan.snapshot()["plan"],
                model=resolved_model, task=_task_now)
        # Teach the rates ledger from the stages that ran clean. Off the
        # loop: one locked, fsync'd rewrite of the ledger file.
        try:
            await asyncio.to_thread(_rplan.finish_run, _status)
        except Exception:  # noqa: BLE001 — a ledger write never fails a run
            pass


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
    context_segments: int | None = Form(None),
    keep_video: str | None = Form(None),
    video_max_height: int | None = Form(None),
    video_format: str | None = Form(None),
    retain_media: str | None = Form(None),
    prefetched_media_id: str | None = Form(None),
    progress_id: str | None = Form(None),
    preload_plan: str | None = Form(None),
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
        keep_video=keep_video,
        video_max_height=video_max_height,
        video_format=video_format,
        retain_media=retain_media,
        prefetched_media_id=prefetched_media_id,
        translate_to=translate_to,
        translation_model=translation_model,
        translation_mode=translation_mode,
        translation_glossary=translation_glossary,
        # Explicit like every argument here: an omitted one would receive
        # its Form(None) FieldInfo, not None.
        context_segments=context_segments,
        progress_id=progress_id,
        preload_plan=preload_plan,
        user=user,
    )


def _include_router(label: str, module: str, attr: str = "router") -> bool:
    """Import `module` and mount its `attr` router. Fail-soft per router: an
    import error drops only this router, logged with its traceback, so the
    rest of the app (and every other router) still comes up."""
    try:
        app.include_router(getattr(importlib.import_module(module), attr))
    except Exception:
        logger.exception("Failed to load %s router", label)
        return False
    return True


# =============================================================================
# The always-on API routers that used to live in this file
# =============================================================================
# POST /v1/text/translations (text→text; /v1/audio/translations stays above
# next to the transcription handler) — translation/routes.py.
_include_router("text translations", "faster_whisper_backend.translation.routes")
# Batch progress / cancel + the durable job resource /v1/jobs* —
# transcription/jobs_routes.py.
_include_router("jobs", "faster_whisper_backend.transcription.jobs_routes")
# Transcribe-from-URL (url-preview / -subtitles / -language / url-media*) and
# the video export (/v1/audio/media*) — media/routes.py.
_include_router("url/media", "faster_whisper_backend.media.routes")
# /v1/models, /v1/me, /v1/override-profiles*, /v1/request-default-settings —
# transcription/catalog_routes.py.
_include_router("catalog", "faster_whisper_backend.transcription.catalog_routes")
# /logs (live log viewer) and /sev (nav severity pills) — admin/logs_routes.py.
_include_router("logs", "faster_whisper_backend.admin.logs_routes")

# /auth/login, /auth/logout, /auth/whoami — auth/routes.py. Deliberately NOT
# through the fail-soft _include_router: an import error there must fail
# startup loudly rather than come up with no way to sign in.
from faster_whisper_backend.auth import routes as auth_routes
app.include_router(auth_routes.router)


# =============================================================================
# /stats - system overview dashboard (always on, user-tier allowlist-gated)
# =============================================================================
# Always registered. The route's host gate reads cfg.USER_WEBUI_ALLOWED_HOSTS
# at request time, so the admin UI can broaden/narrow access without a service
# restart. Loopback is always allowed; the data endpoints require a "stats" key.
if _include_router("stats", "faster_whisper_backend.stats.routes"):
    logger.info(
        "Stats dashboard at /stats (allowlist=%s; loopback always permitted)",
        cfg.USER_WEBUI_ALLOWED_HOSTS,
    )


# =============================================================================
# / - landing hub (always on, user-tier allowlist-gated)
# =============================================================================
# The WebUI's front door: signed-out visitors get the shared login gate,
# signed-in ones a launcher filtered to the pages their key can reach (plus
# the admin section for admins). Same host tier as the other user page
# shells; nothing sensitive is rendered server-side. See core/home_routes.py.
if _include_router("home", "faster_whisper_backend.core.home_routes"):
    logger.info(
        "Landing hub at / (allowlist=%s; loopback always permitted)",
        cfg.USER_WEBUI_ALLOWED_HOSTS,
    )


# =============================================================================
# /v1/audio/transcriptions/stream - live (streaming) dictation WebSocket
# =============================================================================
# Always registered; the handler self-gates on cfg.STREAMING_ENABLED (toggleable
# at runtime) and resolves auth per connection (same user records as the batch
# route). Reuses the model cache + pl_engine._postprocess_text; see streaming/routes.py.
if _include_router("streaming", "faster_whisper_backend.streaming.routes"):
    logger.info(
        "Streaming transcription at /v1/audio/transcriptions/stream "
        "(enabled=%s, max_sessions=%s)",
        getattr(cfg, "STREAMING_ENABLED", True),
        getattr(cfg, "STREAMING_MAX_SESSIONS", 10),
    )


# =============================================================================
# /v1/pipeline-rules - client API for the desktop "Dictionary" editor
# =============================================================================
# Always registered (unlike the /quick-config WebUI, which rides
# ADMIN_UI_ENABLED). Same tag/exposed gating + per-type field allow-list +
# validation as /quick-config (shared build_visible_rules / apply_rules_patch in
# quick_config_routes), but in the /v1 namespace with NO host allowlist — auth is
# the per-user API key (bearer) plus the quick_config page permission. Lets the
# desktop client view + edit the post-processing rules the caller is permitted to.
if _include_router("pipeline-rules v1", "faster_whisper_backend.quick_config.routes", "v1_router"):
    logger.info("Pipeline-rules client API at GET/PATCH /v1/pipeline-rules")


# =============================================================================
# /v1/synced-client-settings - desktop-client settings sync
# =============================================================================
# (the old /v1/client-settings path is a deprecated alias, see _LEGACY_PATHS)
# Always registered (a route-level 404 must keep meaning "backend build too
# old for sync"). User-tier bearer auth only — deliberately NO page gate and
# NO host allowlist: settings sync is account infrastructure for remote
# desktop clients (/v1/usage skips the host allowlist for the same reason, but
# keeps its quick_config page gate). One opaque blob per account
# with optimistic versioning; see client_settings/routes.py.
if _include_router("client-settings", "faster_whisper_backend.client_settings.routes"):
    logger.info(
        "Client-settings sync at GET/PUT/DELETE /v1/synced-client-settings"
    )

# Dictation outcomes from the desktop app (activation / delivery / app /
# translation per session). Same always-on, user-tier, no-host-gate stance
# as the settings sync above; see usage_routes.py.
if _include_router("usage", "faster_whisper_backend.stats.usage_routes"):
    logger.info("Usage outcomes at POST /v1/usage/outcome")


# =============================================================================
# /v1/models/preload - ask the server to warm the models a job will need
# =============================================================================
# Always registered, same rationale as /v1/synced-client-settings above: a
# route-level 404 must keep meaning "backend build too old for preloading", never
# "preloading is off here" — the latter is a 202 with every entry `deferred`.
# User-tier bearer auth only (the tier /v1/models and /v1/me already use to
# publish `loaded` flags for these exact models); no page gate, no host
# allowlist. See preload_routes.py.
if _include_router("preload", "faster_whisper_backend.runtime.preload_routes"):
    logger.info("Model preloading at POST /v1/models/preload (enabled=%s)",
                bool(getattr(cfg, "MODEL_PRELOAD_ENABLED", True)))


# =============================================================================
# /settings - admin WebUI (opt-in)
# =============================================================================
# Off by default: registered only when cfg.ADMIN_UI_ENABLED is True (set in
# config.py or via WHISPER_ADMIN_UI=1). Auth on the endpoints themselves is
# per-user API keys (require_admin) layered on top of cfg.ADMIN_WEBUI_ALLOWED_HOSTS.
# In OPEN mode (no admin key in DB) every caller is the synthetic admin so the
# operator can bootstrap.
if cfg.ADMIN_UI_ENABLED:
    if _include_router("admin settings", "faster_whisper_backend.admin.routes"):
        logger.info(
            "Admin UI enabled at /settings (allowlist=%s; auth=API key)",
            cfg.ADMIN_WEBUI_ALLOWED_HOSTS,
        )
    # /settings/api-keys — admin UI for per-user key management. Same
    # auth shape (admin host + admin key) as /settings.
    _include_router("api-keys", "faster_whisper_backend.admin.api_keys_routes")
    # /settings/overrides — admin UI for layered per-identity config
    # profiles + the effective-config Explorer. Same auth shape as /settings.
    _include_router("overrides", "faster_whisper_backend.admin.overrides_routes")
    # /quick-config is a user-tier page (USER_WEBUI_ALLOWED_HOSTS) with
    # per-user API key auth; it just rides the same ADMIN_UI_ENABLED switch.
    if _include_router("quick-config", "faster_whisper_backend.quick_config.routes"):
        logger.info("Quick-config UI enabled at /quick-config")
    # /reports: admin-only triage page for user-submitted transcription
    # error reports. The submission endpoint /quick-config/reports/api/submit
    # lives on the same router and accepts any active API key.
    if _include_router("reports", "faster_whisper_backend.reports.routes"):
        logger.info(
            "Reports UI enabled at /reports (admin key required for triage; "
            "user submissions %s)",
            "enabled" if getattr(cfg, "REPORTS_ALLOW_USER_SUBMIT", True)
            else "disabled",
        )
    # /captures: admin-only Whisper fine-tuning data capture + review.
    # Master switch is cfg.CAPTURES_RECORDING_ENABLED — the page is
    # always registered so the admin can browse existing rows even
    # after disabling new capture.
    if _include_router("captures", "faster_whisper_backend.captures.routes"):
        logger.info(
            "Captures UI enabled at /captures (admin token required; "
            "new capture %s)",
            "enabled" if getattr(cfg, "CAPTURES_RECORDING_ENABLED", False)
            else "disabled",
        )

def run() -> None:
    """Serve the app. Called by the root main.py shim, by `python -m
    faster_whisper_backend` and by nothing else; the import string names
    the package module so uvicorn (and its worker processes) import the
    same object this module built instead of re-running it as `main`."""
    import uvicorn
    uvicorn.run("faster_whisper_backend.main:app",
                host=cfg.SERVER_HOST,
                port=cfg.SERVER_PORT,
                workers=cfg.SERVER_WORKERS,
                log_level=cfg.SERVER_LOG_LEVEL,
                # WS keepalive: a live decode no longer blocks the receive loop, so
                # pings stay answered; a generous timeout tolerates a momentary
                # stall. `or None` lets an admin disable either knob with 0.
                ws_ping_interval=getattr(cfg, "STREAMING_WS_PING_INTERVAL_S", 20.0) or None,
                ws_ping_timeout=getattr(cfg, "STREAMING_WS_PING_TIMEOUT_S", 60.0) or None,
                # Per-message ceiling on the streaming socket. MAX_REQUEST_BYTES
                # covers HTTP only — its middleware is registered http-only and
                # never sees a websocket scope — so without this the effective
                # limit is the websocket library's 16 MiB default. Real clients
                # send ~32 KB audio frames and a JSON handshake far under 1 MiB.
                ws_max_size=1024 * 1024)


if __name__ == "__main__":
    run()
