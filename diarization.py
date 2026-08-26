"""Speaker diarization via pyannote.audio (optional install).

The pyannote pipeline is a second model kind next to the WhisperModel cache in
main.py, with the same lifecycle discipline scaled down to a singleton: lazy
import (the dependency set is the optional ``requirements-diarize.txt``), load
on first use under an asyncio.Lock with an NVML VRAM delta, registration in
``system_stats`` (as ``pyannote:<model>``) so /stats shows it, an idle-eviction
loop driven live by ``DIARIZATION_IDLE_TIMEOUT_S``, and drop-on-edit when the
admin changes the model/device fields.

Failure contract (soft-fail): every load/inference problem surfaces as a
``DiarizationError`` whose message is safe to echo to the client — the batch
handler turns it into a response ``warnings`` entry and returns the transcript
without speaker labels. Raw third-party exception text (which can carry
filesystem paths) stays in the server log only.
"""

import asyncio
import logging
import os
import time

import config as cfg
import system_stats

logger = logging.getLogger("whisper-server")


class DiarizationError(RuntimeError):
    """Diarization could not run; str(exc) is CLIENT-SAFE (our own wording)."""


_lock = asyncio.Lock()
_pipeline = None
# (model_id, device, embedding_batch_size) the loaded pipeline was built with —
# a config edit that changes any of these makes the cached pipeline stale.
_pipeline_key: "tuple[str, str, int] | None" = None
_last_used_monotonic: float = 0.0
_STATS_PREFIX = "pyannote:"

# The idle loop mirrors main._idle_evictor's cadence.
_EVICTOR_WAKE_S = 30


def _resolve_device() -> str:
    """DIARIZATION_DEVICE with "auto" following MODEL_DEVICE, downgraded to
    cpu when torch reports no CUDA (mirrors the whisper load fallback)."""
    want = (getattr(cfg, "DIARIZATION_DEVICE", "auto") or "auto").lower()
    if want == "auto":
        want = (getattr(cfg, "MODEL_DEVICE", "cpu") or "cpu").lower()
    if want != "cuda":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("[diarize] cuda requested but not available — using cpu")
    except ImportError:
        pass
    return "cpu"


def _load_blocking(model_id: str, device: str, batch_size: int):
    """Import pyannote and build the pipeline. Runs in the default executor."""
    # Keep HF downloads on the models volume (whisper weights already live
    # there via download_root); a set HF_HOME always wins.
    download_root = getattr(cfg, "DOWNLOAD_ROOT", None)
    if download_root:
        os.environ.setdefault("HF_HOME", os.path.join(download_root, "hf"))
    if getattr(cfg, "LOCAL_FILES_ONLY", False):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(
            "diarization dependencies are not installed on this server "
            "(pip install -r requirements-diarize.txt)"
        ) from e

    token = getattr(cfg, "HF_TOKEN", None) or None
    try:
        try:
            pipe = Pipeline.from_pretrained(model_id, token=token)
        except TypeError:
            # pyannote 3.x spells the kwarg use_auth_token.
            pipe = Pipeline.from_pretrained(model_id, use_auth_token=token)
    except DiarizationError:
        raise
    except Exception as e:
        logger.error("[diarize] pipeline load failed: %s", e)
        raise DiarizationError(
            f"could not load {model_id} — the model is gated on huggingface.co "
            "(accept its terms and set HF_TOKEN), or the download failed"
        ) from e
    if pipe is None:
        # pyannote 3.x returns None (with its own warning) on a gated repo.
        raise DiarizationError(
            f"could not load {model_id} — accept the model terms on "
            "huggingface.co and set HF_TOKEN"
        )

    pipe.to(torch.device(device))
    try:
        # pyannote-audio#1963: the default embedding batch spikes several GB
        # of VRAM on hour-long audio; a small batch flattens the peak.
        pipe.embedding_batch_size = int(batch_size)
    except Exception:
        logger.debug("[diarize] pipeline has no embedding_batch_size knob")
    return pipe


def _drop_locked() -> None:
    """Drop the cached pipeline. Caller holds _lock."""
    global _pipeline, _pipeline_key
    if _pipeline is None:
        return
    model_id = _pipeline_key[0] if _pipeline_key else "?"
    _pipeline = None
    _pipeline_key = None
    system_stats.unregister_loaded_model(_STATS_PREFIX + model_id)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("[diarize] pipeline %s unloaded", model_id)


async def _get_pipeline():
    """Return the cached pipeline, (re)loading when config changed it."""
    global _pipeline, _pipeline_key, _last_used_monotonic
    model_id = getattr(cfg, "DIARIZATION_MODEL", "") or ""
    if not model_id:
        raise DiarizationError("no DIARIZATION_MODEL configured")
    device = _resolve_device()
    batch = int(getattr(cfg, "DIARIZATION_EMBEDDING_BATCH_SIZE", 4) or 4)
    key = (model_id, device, batch)
    async with _lock:
        if _pipeline is not None and _pipeline_key == key:
            _last_used_monotonic = time.monotonic()
            system_stats.touch_loaded_model(_STATS_PREFIX + model_id)
            return _pipeline
        _drop_locked()
        vram_before = system_stats.gpu_mem_used_bytes()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        pipe = await loop.run_in_executor(
            None, _load_blocking, model_id, device, batch)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram = (vram_after - vram_before) if (
            vram_before is not None and vram_after is not None) else None
        system_stats.register_loaded_model(
            _STATS_PREFIX + model_id, vram, device, "torch")
        logger.info("[diarize] pipeline %s loaded on %s in %.1fs",
                    model_id, device, time.perf_counter() - t0)
        _pipeline = pipe
        _pipeline_key = key
        _last_used_monotonic = time.monotonic()
        return pipe


async def diarize(path: str, *, num_speakers: "int | None" = None,
                  min_speakers: "int | None" = None,
                  max_speakers: "int | None" = None,
                  ) -> "list[tuple[float, float, str]]":
    """Diarize the audio file → [(start_s, end_s, label), ...] sorted by start.

    ``num_speakers`` wins over the min/max bounds (the caller enforces that
    already; this just doesn't forward the bounds alongside it — pyannote
    treats the combination as an error).
    """
    global _last_used_monotonic
    pipe = await _get_pipeline()
    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    def _run():
        result = pipe(path, **kwargs)
        # pyannote 4.x returns a result object; the exclusive (non-overlapping)
        # view is purpose-built for aligning with STT segments. 3.x returns the
        # Annotation itself.
        ann = result
        for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
            candidate = getattr(result, attr, None)
            if candidate is not None:
                ann = candidate
                break
        turns = [
            (float(turn.start), float(turn.end), str(label))
            for turn, _, label in ann.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t[0])
        return turns

    loop = asyncio.get_running_loop()
    try:
        turns = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.error("[diarize] inference failed: %s", e)
        raise DiarizationError(
            "diarization failed on this file; the transcript has no speaker "
            "labels"
        ) from e
    _last_used_monotonic = time.monotonic()
    model_id = _pipeline_key[0] if _pipeline_key else ""
    if model_id:
        system_stats.touch_loaded_model(_STATS_PREFIX + model_id)
    return turns


def assign_speakers(segments_list: "list[dict]",
                    turns: "list[tuple[float, float, str]]") -> "list[str]":
    """Attach a ``"speaker"`` key to each segment dict (largest time overlap;
    nearest turn as fallback for segments inside diarization gaps). Returns
    the distinct labels in order of first appearance."""
    labels: "list[str]" = []
    if not turns:
        return labels
    for seg in segments_list:
        s, e = float(seg["start"]), float(seg["end"])
        best_label = None
        best_overlap = 0.0
        for ts, te, label in turns:
            overlap = min(e, te) - max(s, ts)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label is None:
            mid = (s + e) / 2.0
            best_label = min(
                turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        seg["speaker"] = best_label
        if best_label not in labels:
            labels.append(best_label)
    return labels


async def drop_pipeline() -> None:
    """Evict the cached pipeline (admin edited model/device, or shutdown)."""
    async with _lock:
        _drop_locked()


async def idle_evictor_loop() -> None:
    """Unload the pipeline after DIARIZATION_IDLE_TIMEOUT_S idle seconds.
    Reads the timeout live (an admin edit applies without restart), like
    main._idle_evictor for whisper models. Started/cancelled by lifespan."""
    while True:
        await asyncio.sleep(_EVICTOR_WAKE_S)
        try:
            timeout = int(getattr(cfg, "DIARIZATION_IDLE_TIMEOUT_S", 0) or 0)
            if timeout <= 0 or _pipeline is None:
                continue
            if time.monotonic() - _last_used_monotonic >= timeout:
                await drop_pipeline()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the loop must survive
            logger.error("[diarize] idle evictor error: %s", e)
