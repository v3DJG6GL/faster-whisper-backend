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


def _pass_fraction(pass_no: int, frac: float) -> float:
    """Map a within-pass fraction to the overall 0..1 separation progress."""
    frac = min(1.0, max(0.0, frac))
    if pass_no <= 1:
        return _PASS1_WEIGHT * frac
    return _PASS1_WEIGHT + (1.0 - _PASS1_WEIGHT) * frac


_shims_installed = False


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
                out = super().update(n)
                cb = getattr(_progress_tls, "cb", None)
                total = getattr(self, "total", None)
                if cb is not None and total:
                    try:
                        cb(_pass_fraction(
                            getattr(_progress_tls, "pass_no", 1),
                            float(self.n) / float(total)))
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
                return self._real.InferenceSession(*args, **kwargs)

        if not isinstance(mdx_separator.ort, _OrtShim):
            mdx_separator.ort = _OrtShim(mdx_separator.ort)
    except Exception as e:  # noqa: BLE001
        logger.debug("[bgm] ort shim not installed: %s", e)

    _shims_installed = True


_lock = asyncio.Lock()
_separator = None
_separator_key: "tuple[str, str] | None" = None  # (model_filename, device)
_last_used_monotonic: float = 0.0
_STATS_PREFIX = "uvr:"
_EVICTOR_WAKE_S = 30


def _model_filename() -> str:
    name = (getattr(cfg, "BGM_SEPARATION_UVR_MODEL", "") or "").strip()
    if not name:
        raise BgmSeparationError("no BGM_SEPARATION_UVR_MODEL configured")
    # The config carries the friendly model name; audio-separator wants the
    # on-disk filename (MDX models are .onnx).
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
    # "Loaded on cuda" only means cuda was REQUESTED — onnxruntime's CUDA
    # provider silently falls back to CPU when its runtime libraries don't
    # resolve (the classic symptom: separation maxes the CPU). Surface the
    # provider the CREATED session actually runs on; every getattr is
    # defensive because these are audio-separator internals.
    providers = None
    session = getattr(getattr(sep, "model_instance", None), "model_run", None)
    get_providers = getattr(session, "get_providers", None)
    if callable(get_providers):
        try:
            providers = list(get_providers())
        except Exception:  # noqa: BLE001 — diagnostics only
            providers = None
    if providers is None:
        providers = getattr(sep, "onnx_execution_provider", None)
    logger.info("[bgm] onnx execution providers: %s", providers)
    if (device == "cuda" and isinstance(providers, list)
            and "CUDAExecutionProvider" not in providers):
        logger.warning(
            "[bgm] cuda was requested but the ONNX session runs on %s — the "
            "CUDA provider could not initialize (usually cudart/cufft/curand "
            "missing from LD_LIBRARY_PATH); separation will hammer the CPU",
            providers)
    return sep


def _drop_locked() -> None:
    """Drop the cached separator. Caller holds _lock."""
    global _separator, _separator_key
    if _separator is None:
        return
    model = _separator_key[0] if _separator_key else "?"
    _separator = None
    _separator_key = None
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


async def _get_separator():
    global _separator, _separator_key, _last_used_monotonic
    model = _model_filename()
    device = _resolve_device()
    key = (model, device)
    async with _lock:
        if _separator is not None and _separator_key == key:
            _last_used_monotonic = time.monotonic()
            system_stats.touch_loaded_model(_STATS_PREFIX + model)
            return _separator
        _drop_locked()
        vram_before = system_stats.gpu_mem_used_bytes()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        sep = await loop.run_in_executor(None, _load_blocking, model, device)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        system_stats.register_loaded_model(_STATS_PREFIX + model, vram, device, "onnx")
        logger.info("[bgm] separation model %s loaded on %s in %.1fs",
                    model, device, time.perf_counter() - t0)
        _separator = sep
        _separator_key = key
        _last_used_monotonic = time.monotonic()
        return sep


async def separate(path: str, *, progress_cb=None) -> str:
    """Separate the file → absolute path of the vocals-only WAV.

    The output lands in the system temp dir under a unique name; the caller
    owns unlinking it (and the original it replaces). ``progress_cb`` (called
    from the executor thread with a 0..1 float) reports demix chunk progress
    via the tqdm shim — best-effort, and monotone across the two passes.
    """
    global _last_used_monotonic
    sep = await _get_separator()
    out_name = f"vocals-{uuid.uuid4().hex}"

    def _run() -> str:
        _progress_tls.cb = progress_cb
        _progress_tls.pass_no = 0
        try:
            outputs = sep.separate(
                path, custom_output_names={"Vocals": out_name})
        finally:
            _progress_tls.cb = None
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


async def drop_separator() -> None:
    """Evict the cached separator (admin edited model/device, or shutdown)."""
    async with _lock:
        _drop_locked()


async def idle_evictor_loop() -> None:
    """Unload after BGM_SEPARATION_IDLE_TIMEOUT_S idle seconds (read live)."""
    while True:
        await asyncio.sleep(_EVICTOR_WAKE_S)
        try:
            timeout = int(getattr(cfg, "BGM_SEPARATION_IDLE_TIMEOUT_S", 0) or 0)
            if timeout <= 0 or _separator is None:
                continue
            if time.monotonic() - _last_used_monotonic >= timeout:
                await drop_separator()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            logger.error("[bgm] idle evictor error: %s", e)
