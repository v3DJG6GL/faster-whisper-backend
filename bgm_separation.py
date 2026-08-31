"""Background-music separation via audio-separator / UVR MDX-Net (optional
install: ``requirements-bgm.txt``).

Second optional model kind next to diarization.py, same lifecycle shape:
lazy import, singleton Separator cached under an asyncio.Lock with an NVML
delta registered in ``system_stats`` (as ``uvr:<model>``), an idle unloader
driven live by ``BGM_SEPARATION_IDLE_TIMEOUT_S``, drop-on-edit, and the same
soft-fail contract: every problem surfaces as ``BgmSeparationError`` whose
message is CLIENT-SAFE; raw third-party text stays in the server log.

The stage runs PRE-decode: ``separate(path)`` writes a vocals-only WAV next to
the system temp files and returns its path — the caller swaps it in for the
original upload (decode AND captures then see the separated audio) and owns
unlinking both files.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid

import config as cfg
import system_stats

logger = logging.getLogger("whisper-server")

# use_soundfile=True below is a deliberate choice, but audio_separator
# announces it with a WARNING ("Using soundfile for writing.") on every stem
# write — drop that one message so real warnings stay visible. Registered
# once at import; logging filters accumulate if added per load.
logging.getLogger("audio_separator.separator.separator").addFilter(
    lambda record: "Using soundfile for writing" not in record.getMessage()
)

# CPU-pool cap for the separation stage. ONNX Runtime sizes its intra-op pool
# to EVERY logical core by default and its worker threads spin-wait between
# tasks — on a many-core host that reads as the separation stage "maxing the
# CPU" even while the actual math runs on the CUDA session. torch's CPU pool
# (used for the STFT round-trips when the device is cpu) defaults the same
# way. Eight threads is plenty for the numpy/copy work these stages do on the
# CPU side.
_CPU_THREADS_CAP = 8


def _cpu_threads() -> int:
    return max(1, min(_CPU_THREADS_CAP, os.cpu_count() or _CPU_THREADS_CAP))


class BgmSeparationError(RuntimeError):
    """Separation could not run; str(exc) is CLIENT-SAFE (our own wording)."""


class BgmCancelled(Exception):
    """The caller cancelled mid-separation (cooperative: raised from the tqdm
    shim between demix chunks when ``cancel_check`` answers True).
    Deliberately NOT a BgmSeparationError — the handler must abort the whole
    request, not soft-fail into transcribing the original audio."""


# Per-thread progress plumbing for the tqdm shim below. audio-separator has
# no progress API; its MDX demix loop iterates chunks under a module-level
# `tqdm`. separate() runs the whole separation inside one executor thread, so
# a thread-local callback set around that call reaches exactly the right
# tqdm instances and nothing else (other threads see cb=None → stock tqdm).
_progress_tls = threading.local()

# MDX separates in two demix passes: the model pass over every chunk, then a
# cheap STFT-only "match mix" pass for the secondary stem. Weight the first
# heavier — it carries the ONNX inference.
_PASS1_WEIGHT = 0.85

# True once the loaded model's match-mix pass is skipped (see _load_blocking:
# with invert_using_spec off its result is never read) — the model pass then
# IS the whole separation and owns the full 0..1 span.
_single_pass = False


def _pass_fraction(pass_no: int, frac: float) -> float:
    """Map a within-pass fraction to the overall 0..1 separation progress."""
    frac = min(1.0, max(0.0, frac))
    w = 1.0 if _single_pass else _PASS1_WEIGHT
    if pass_no <= 1:
        return w * frac
    return w + (1.0 - w) * frac


_shims_installed = False

# Ground truth from the ORT session the shim created: "cuda"/"cpu" once a
# model is loaded, None before. The requested provider list means nothing —
# a CUDA provider that fails to dlopen (e.g. a CUDA 13 wheel on a CUDA 12
# stack) makes ORT fall back to CPU silently at session creation.
_session_device: "str | None" = None


def actual_device() -> "str | None":
    """The device the loaded ONNX session ACTUALLY runs on (None = unknown)."""
    return _session_device


def _install_shims() -> None:
    """Wrap audio-separator's MDX internals (once): a tqdm subclass that
    reports chunk progress to the thread-local callback, and an onnxruntime
    shim that tames the session's CPU thread pool (see _CPU_THREADS_CAP).
    Both are audio-separator internals — every step is defensive and a
    failure just means stock behavior."""
    global _shims_installed
    if _shims_installed:
        return
    try:
        from audio_separator.separator.architectures import mdx_separator
    except Exception as e:  # noqa: BLE001 — shims are best-effort
        logger.debug("[bgm] shims not installed: %s", e)
        return

    try:
        _base_tqdm = mdx_separator.tqdm

        class _ReportingTqdm(_base_tqdm):
            def __init__(self, *args, **kwargs):
                _progress_tls.pass_no = getattr(_progress_tls, "pass_no", 0) + 1
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                # Cooperative cancel between demix chunks: raised OUTSIDE the
                # swallow-everything progress guard below, so it unwinds
                # through separate() to the request handler.
                cancel = getattr(_progress_tls, "cancel", None)
                if cancel is not None and cancel():
                    raise BgmCancelled()
                out = super().update(n)
                cb = getattr(_progress_tls, "cb", None)
                total = getattr(self, "total", None)
                if cb is not None and total:
                    try:
                        # The first chunk carries the CUDA/ORT warmup cost —
                        # one INFO separates warmup from steady-state.
                        if not getattr(_progress_tls, "ticked", False):
                            _progress_tls.ticked = True
                            logger.info(
                                "[bgm] first chunk done — device warmed up")
                        overall = _pass_fraction(
                            getattr(_progress_tls, "pass_no", 1),
                            float(self.n) / float(total))
                        cb(overall)
                        # 5%-step INFO trail so the server log shows the
                        # stage moving too, without tqdm's \r noise.
                        b = int(overall * 20)
                        if b > getattr(_progress_tls, "log_bucket", 0):
                            _progress_tls.log_bucket = b
                            logger.info("[bgm] separating %d%%", b * 5)
                    except Exception:  # noqa: BLE001 — never break the loop
                        pass
                return out

        mdx_separator.tqdm = _ReportingTqdm
    except Exception as e:  # noqa: BLE001
        logger.debug("[bgm] tqdm shim not installed: %s", e)

    try:
        class _OrtShim:
            """Proxy for the onnxruntime module as mdx_separator sees it:
            InferenceSession gains capped, non-spinning CPU pools; everything
            else passes through untouched. Scoped to this one module — the
            global onnxruntime (Silero VAD etc.) is unaffected."""

            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def InferenceSession(self, *args, **kwargs):  # noqa: N802 — ORT API
                so = kwargs.get("sess_options")
                if so is None:
                    so = self._real.SessionOptions()
                    kwargs["sess_options"] = so
                try:
                    so.intra_op_num_threads = _cpu_threads()
                    so.inter_op_num_threads = 1
                    so.add_session_config_entry(
                        "session.intra_op.allow_spinning", "0")
                    so.add_session_config_entry(
                        "session.inter_op.allow_spinning", "0")
                except Exception:  # noqa: BLE001 — tuning only
                    logger.debug("[bgm] could not tune ORT session options")
                session = self._real.InferenceSession(*args, **kwargs)
                # mdx_separator keeps the session only inside a closure, so
                # THIS is the one place the real placement is visible.
                global _session_device
                try:
                    active = list(session.get_providers())
                    _session_device = (
                        "cuda" if "CUDAExecutionProvider" in active else "cpu")
                    logger.info(
                        "[bgm] onnx session providers (actual): %s", active)
                    wanted = kwargs.get("providers") or []
                    if ("CUDAExecutionProvider" in wanted
                            and "CUDAExecutionProvider" not in active):
                        logger.warning(
                            "[bgm] the CUDA provider failed to initialize — "
                            "separation runs on the CPU (onnxruntime printed "
                            "the dlopen error to stderr; usual cause: an "
                            "onnxruntime-gpu build for a different CUDA "
                            "major than this image ships)")
                except Exception:  # noqa: BLE001 — diagnostics only
                    _session_device = None
                return session

        if not isinstance(mdx_separator.ort, _OrtShim):
            mdx_separator.ort = _OrtShim(mdx_separator.ort)
    except Exception as e:  # noqa: BLE001
        logger.debug("[bgm] ort shim not installed: %s", e)

    _shims_installed = True


# Serializes ACTUAL separator use across threads. The request path already
# holds the shared inference semaphore, but a CANCELLED request's separation
# keeps running in its executor thread (threads can't be aborted) — and
# Separator is a singleton whose per-file state (audio_file_path, sources)
# is cleared at the end of each run. A zombie finishing mid-way through a
# live run wiped that state under it (observed 2026-08-26: write_audio saw
# audio_file_path=None → "separator returned no output files"). A plain
# threading.Lock in the executor thread covers zombies too.
_separate_mutex = threading.Lock()

_lock = asyncio.Lock()
_separator = None
_separator_key: "tuple[str, str] | None" = None  # (model_filename, device)
_last_used_monotonic: float = 0.0
_STATS_PREFIX = "uvr:"
# model filename → count of jobs currently running inference on the CACHED
# separator, and → count still running on one already dropped from the cache
# (see _drop_locked's orphan branch). Tearing an ONNX session down while
# another job's executor thread is inside it is a use-after-free in native
# code; until this existed, the only thing preventing it was the caller's
# local Python reference.
#
# Mutated without _lock on the cache-hit fast path only — see the same note on
# diarization._leases for why loop semantics make that safe.
_leases: "dict[str, int]" = {}
_orphans: "dict[str, int]" = {}
_EVICTOR_WAKE_S = 30


def _model_filename(name: "str | None" = None) -> str:
    """Resolve the friendly model name (a per-request override, else
    cfg.BGM_SEPARATION_UVR_MODEL) to the on-disk filename audio-separator
    wants (MDX models are .onnx)."""
    name = (name or "").strip() or \
        (getattr(cfg, "BGM_SEPARATION_UVR_MODEL", "") or "").strip()
    if not name:
        raise BgmSeparationError("no BGM_SEPARATION_UVR_MODEL configured")
    return name if "." in name else f"{name}.onnx"


def _resolve_device() -> str:
    want = (getattr(cfg, "BGM_SEPARATION_DEVICE", "auto") or "auto").lower()
    if want == "auto":
        want = (getattr(cfg, "MODEL_DEVICE", "cpu") or "cpu").lower()
    return "cuda" if want == "cuda" else "cpu"


def _load_blocking(model_filename: str, device: str):
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        # The client-safe message says "not installed", but a BROKEN install
        # (a transitive import blowing up) raises ImportError too — log the
        # real cause so the operator can tell the two apart.
        logger.error("[bgm] audio_separator import failed: %s", e)
        raise BgmSeparationError(
            "music-separation dependencies are not installed on this server "
            "(pip install -r requirements-bgm.txt)"
        ) from e
    _install_shims()
    try:
        # Decisive GPU evidence for the log: a session can be CREATED with the
        # CUDA provider yet the installed onnxruntime build (CPU-only wheel
        # shadowing onnxruntime-gpu) or missing libs make it a dead letter.
        import onnxruntime as _ort
        logger.info("[bgm] onnxruntime build=%s available_providers=%s",
                    _ort.get_device(), _ort.get_available_providers())
    except Exception:  # noqa: BLE001 — diagnostics only
        pass
    try:
        # torch's CPU intra-op pool also defaults to every core (ctranslate2 /
        # whisper has its own pool and is unaffected by this knob).
        import torch
        torch.set_num_threads(_cpu_threads())
    except Exception:  # noqa: BLE001 — tuning only
        pass
    models_dir = getattr(cfg, "DOWNLOAD_ROOT", None) or tempfile.gettempdir()
    sep = Separator(
        log_level=logging.WARNING,
        model_file_dir=os.path.join(models_dir, "audio-separator"),
        output_dir=tempfile.gettempdir(),
        output_format="WAV",
        output_single_stem="Vocals",
        # Write stems with soundfile directly instead of the pydub default,
        # which quantizes/interleaves in Python and pipes the whole WAV
        # through an ffmpeg subprocess — ~20 s of dead time on long audio.
        use_soundfile=True,
    )
    if device == "cpu":
        # Separator autodetects CUDA in __init__ (setup_torch_device); there is
        # no constructor knob, but the chosen device/provider is only consumed
        # at load_model — overriding the two attributes here pins it to CPU.
        try:
            import torch
            sep.torch_device = torch.device("cpu")
            sep.onnx_execution_provider = ["CPUExecutionProvider"]
        except Exception:
            logger.debug("[bgm] could not pin separator to cpu")
    try:
        sep.load_model(model_filename=model_filename)
    except Exception as e:
        logger.error("[bgm] model load failed: %s", e)
        raise BgmSeparationError(
            f"could not load separation model {model_filename} — check the "
            "model name and that the server can download it"
        ) from e
    # MDX's separate() always runs a SECOND full demix pass ("match mix") to
    # build the secondary stem — but with invert_using_spec off (the default,
    # and ours) its result is never read: the secondary comes from
    # `mix - primary` instead. Skip the dead pass; on an hour of audio that
    # is minutes of work for nothing. Version-pinned internals (>=0.44.5),
    # so every step is defensive; the timing wrapper doubles as per-pass
    # evidence in the log.
    global _single_pass
    _single_pass = False
    inst = getattr(sep, "model_instance", None)
    if (inst is not None and hasattr(inst, "demix")
            and getattr(inst, "invert_using_spec", None) is False):
        _orig_demix = inst.demix

        def _demix(mix, is_match_mix=False):
            if is_match_mix:
                # Only ever read under invert_using_spec — see above.
                return mix
            # Everything between _run's load_t0 and here is prepare_mix:
            # reading + validating + normalizing the whole input (tens of
            # seconds on long audio) — bracket it so the log has no dead air.
            load_t0 = getattr(_progress_tls, "load_t0", None)
            if load_t0 is not None:
                _progress_tls.load_t0 = None
                logger.info(
                    "[bgm] audio loaded and normalized in %.1fs — "
                    "model pass starting", time.perf_counter() - load_t0)
            t0 = time.perf_counter()
            out = _orig_demix(mix, is_match_mix=is_match_mix)
            logger.info("[bgm] model pass done in %.1fs",
                        time.perf_counter() - t0)
            # What follows in the library is stem synthesis + the WAV write.
            logger.info("[bgm] assembling and writing the vocals stem")
            return out

        inst.demix = _demix
        _single_pass = True
        logger.info("[bgm] match-mix pass skipped (invert_using_spec off)")
    # Placement truth comes from the _OrtShim above (the session lives only
    # in a closure inside mdx_separator; sep.onnx_execution_provider is the
    # REQUESTED list and proves nothing).
    if device == "cuda" and _session_device == "cpu":
        logger.warning(
            "[bgm] cuda was requested but the session runs on the CPU — "
            "separation will be slow; see the shim warning above")
    return sep


def _free_locked(model: str) -> None:
    """Give the separator's memory back. Caller holds _lock and has already
    removed it from the cache."""
    # A same-model RELOAD can have happened while an orphan was still
    # draining; the stats entry then describes the live separator.
    if not (_separator_key and _separator_key[0] == model):
        system_stats.unregister_loaded_model(_STATS_PREFIX + model)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("[bgm] separation model %s unloaded", model)


def _drop_locked(*, force: bool = False) -> bool:
    """Drop the cached separator. Caller holds _lock. Returns False when it
    declined.

    Leased and not ``force`` → declines, so the idle evictor just retries on
    its next tick. Leased and ``force`` → ORPHANS the separator: the cache is
    cleared (new callers reload) but nothing is unregistered or collected,
    because the job inside the executor is still calling into it on its own
    reference. The last ``_release_separator`` finishes the teardown — the
    "drain comes for free from refcounting" contract main.drain_then_evict
    documents, made explicit for a singleton."""
    global _separator, _separator_key
    if _separator is None:
        return True
    model = _separator_key[0] if _separator_key else "?"
    leased = _leases.get(model, 0)
    if leased and not force:
        logger.info("[bgm] separation model %s is in use — eviction deferred",
                    model)
        return False
    _separator = None
    _separator_key = None
    if leased:
        _leases.pop(model, None)
        _orphans[model] = _orphans.get(model, 0) + leased
        logger.info("[bgm] separation model %s evicted with %d job(s) still "
                    "running — freed when the last one finishes", model, leased)
        return True
    _free_locked(model)
    return True


def _release_locked(model: str) -> None:
    """Drop one lease. Caller holds _lock. Orphans are decremented first: a
    holder that outlived its separator is by definition one of them."""
    global _last_used_monotonic
    n = _orphans.get(model, 0)
    if n:
        n -= 1
        if n:
            _orphans[model] = n
        else:
            _orphans.pop(model, None)
            _free_locked(model)
        return
    n = _leases.get(model, 0) - 1
    if n <= 0:
        _leases.pop(model, None)
    else:
        _leases[model] = n
    if _separator is not None and _separator_key and _separator_key[0] == model:
        # Restart the idle clock: a long separation must not be evicted the
        # instant it ends because the LOAD timestamp aged past the timeout.
        _last_used_monotonic = time.monotonic()
        system_stats.touch_loaded_model(_STATS_PREFIX + model)


async def _release_separator(model: str) -> None:
    """Release a lease taken by ``_get_separator(..., lease=True)``."""
    async with _lock:
        _release_locked(model)


async def _get_separator(model_filename: "str | None" = None, *,
                         lease: bool = False):
    """Return the cached separator, (re)loading when config (or a per-request
    ``model_filename`` override) changed it — the (model, device) key below
    re-keys the singleton per call."""
    global _separator, _separator_key, _last_used_monotonic
    model = _model_filename(model_filename)
    device = _resolve_device()
    key = (model, device)
    # Lock-free cache hit. Read the singleton into a LOCAL first so a
    # concurrent _drop_locked cannot null it between the check and the return.
    # Taking _lock before the key comparison (as this did) made every job on a
    # warm separator block for the full duration of any other job's load.
    sep = _separator
    if sep is not None and _separator_key == key:
        _last_used_monotonic = time.monotonic()
        system_stats.touch_loaded_model(_STATS_PREFIX + model)
        if lease:
            _leases[model] = _leases.get(model, 0) + 1
        return sep
    async with _lock:
        if _separator is not None and _separator_key == key:
            _last_used_monotonic = time.monotonic()
            system_stats.touch_loaded_model(_STATS_PREFIX + model)
            if lease:
                _leases[model] = _leases.get(model, 0) + 1
            return _separator
        # force: a request for a DIFFERENT model must never be blocked by a
        # running job — orphaning lets both coexist until the old one drains.
        _drop_locked(force=True)
        vram_before = system_stats.gpu_mem_used_bytes()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        # Coarse download-or-load job entry (indeterminate progress):
        # audio-separator downloads through its own requests+tqdm stack, so
        # the hub shim can't see the bytes — the entry still tells /stats
        # and the header cluster that a model fetch/load is in flight.
        import jobs
        _dl_job = jobs.job_start("download", model=_STATS_PREFIX + model,
                                 detail="download-or-load")
        try:
            sep = await loop.run_in_executor(
                None, _load_blocking, model, device)
        finally:
            jobs.job_end(_dl_job)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        load_secs = time.perf_counter() - t0
        system_stats.register_loaded_model(_STATS_PREFIX + model, vram, device,
                                           "onnx", load_secs)
        logger.info("[bgm] separation model %s loaded on %s in %.1fs",
                    model, device, load_secs)
        try:
            import metrics
            metrics.record_model_load(_STATS_PREFIX + model, load_secs)
        except Exception:  # noqa: BLE001 — stats only
            pass
        _separator = sep
        _separator_key = key
        _last_used_monotonic = time.monotonic()
        if lease:
            _leases[model] = _leases.get(model, 0) + 1
        return sep


async def separate(path: str, *, model_filename: "str | None" = None,
                   progress_cb=None, cancel_check=None) -> str:
    """Separate the file → absolute path of the vocals-only WAV.

    The output lands in the system temp dir under a unique name; the caller
    owns unlinking it (and the original it replaces). ``model_filename``
    overrides cfg.BGM_SEPARATION_UVR_MODEL for this call (per-request stage
    models — the caller has already applied its allowlist). ``progress_cb``
    (called from the executor thread with a 0..1 float) reports demix chunk
    progress via the tqdm shim — best-effort, and monotone across the two
    passes. ``cancel_check`` (no-arg, truthy = abort) is polled at the same
    cadence; a positive answer raises :class:`BgmCancelled` out of this
    coroutine.

    The job lease is taken HERE, not in the caller — deliberately asymmetric
    with translation.py, whose lease spans many completions and therefore has
    to live in its handler. Separation has exactly one entry point that both
    loads and runs, so no handler code has to know about leases at all.
    """
    sep = await _get_separator(model_filename, lease=True)
    leased = _model_filename(model_filename)
    try:
        return await _separate_with(sep, path, progress_cb=progress_cb,
                                    cancel_check=cancel_check)
    finally:
        await _release_separator(leased)


async def _separate_with(sep, path: str, *, progress_cb, cancel_check) -> str:
    """Run one separation on an already-leased separator (see ``separate``)."""
    global _last_used_monotonic
    out_name = f"vocals-{uuid.uuid4().hex}"

    def _run() -> str:
        if cancel_check is not None and cancel_check():
            raise BgmCancelled()
        _progress_tls.cb = progress_cb
        _progress_tls.cancel = cancel_check
        _progress_tls.pass_no = 0
        _progress_tls.log_bucket = 0
        _progress_tls.ticked = False
        try:
            if _separate_mutex.locked():
                # A cancelled request's zombie separation is still running
                # (see the mutex comment above) — say so instead of stalling
                # silently.
                logger.info(
                    "[bgm] waiting for a previous separation to finish")
            with _separate_mutex:
                _progress_tls.load_t0 = time.perf_counter()
                logger.info("[bgm] loading audio for separation")
                outputs = sep.separate(
                    path, custom_output_names={"Vocals": out_name})
        finally:
            _progress_tls.cb = None
            _progress_tls.cancel = None
            _progress_tls.load_t0 = None
        if not outputs:
            raise RuntimeError("separator returned no output files")
        out = outputs[0]
        if not os.path.isabs(out):
            out = os.path.join(tempfile.gettempdir(), out)
        if not os.path.exists(out):
            raise RuntimeError(f"separator output missing: {out}")
        return out

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run)
    except BgmCancelled:
        logger.info("[bgm] separation cancelled by client")
        raise
    except Exception as e:
        logger.error("[bgm] separation failed: %s", e)
        raise BgmSeparationError(
            "music separation failed on this file; transcribing the "
            "original audio"
        ) from e
    _last_used_monotonic = time.monotonic()
    if _separator_key:
        system_stats.touch_loaded_model(_STATS_PREFIX + _separator_key[0])
    return result


async def drop_separator(*, force: bool = True) -> bool:
    """Evict the cached separator (admin edited model/device, or shutdown).

    Forced by default: an admin edit must take effect on the next request, and
    a running job simply orphans the old separator and frees it when it ends.
    The idle evictor passes force=False — there a refusal costs nothing."""
    async with _lock:
        return _drop_locked(force=force)


async def idle_evictor_loop() -> None:
    """Unload after BGM_SEPARATION_IDLE_TIMEOUT_S idle seconds (read live)."""
    while True:
        await asyncio.sleep(_EVICTOR_WAKE_S)
        try:
            timeout = int(getattr(cfg, "BGM_SEPARATION_IDLE_TIMEOUT_S", 0) or 0)
            if timeout <= 0 or _separator is None:
                continue
            # A warm lease (a live preload plan still expects this separator)
            # suspends the idle clock — never a job lease, which _drop_locked
            # already honours. Inverted through system_stats so preload can
            # reach us without either module importing the other.
            if _separator_key and system_stats.is_warm(
                    _STATS_PREFIX + _separator_key[0]):
                continue
            if time.monotonic() - _last_used_monotonic >= timeout:
                await drop_separator(force=False)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            logger.error("[bgm] idle evictor error: %s", e)
