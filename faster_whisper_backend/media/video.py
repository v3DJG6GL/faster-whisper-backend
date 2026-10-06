"""Video and link-audio fetch helpers shared by the transcription handler
(main.transcribe: a link run that keeps the video, the language check's
prefetched audio) and the url/media routes (media/routes.py): the keep_video
download task and its progress sub-object, the guarded whole-audio download,
the per-identity video-download window and the staging job dir. Callers go
through the module attribute (``media_video._download_video_for_run(...)``)
so a test patch reaches both. Never imports media/routes or main.
"""
import asyncio
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.transcription import jobs_store as _jobs_store
from faster_whisper_backend.core import store_common
from faster_whisper_backend.media import media_store as url_media_store
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.media import download as _udl

logger = logging.getLogger("whisper-api")
_log_safe = store_common.log_safe


def _prefetched_audio(media_id: "str | None", url: str,
                      user_id: "str | None") -> "dict | None":
    """The retained audio a link's language check downloaded, when this
    caller may reuse it for `url`: a well-formed id the media store resolves
    for them (its owner rule), kind audio, downloaded from the same
    validated URL. Any miss is None and the run simply downloads — no error
    a caller could use to probe other users' ids."""
    if not isinstance(media_id, str) or not url_media_store.MEDIA_ID_RE.match(media_id):
        return None
    entry = url_media_store.resolve_entry(media_id, user_id=user_id)
    if not entry or entry["kind"] != "audio" or entry.get("source_url") != url:
        return None
    return entry


# Per-identity window for VIDEO downloads (a link run that keeps the video, or
# the on-demand route): each one pulls up to MEDIA_MAX_BYTES from a third
# party — the preview limiter's reasoning, with a fatter payload.
_url_video_rate = _rl.FixedWindow(
    config_field="URL_VIDEO_RATE_PER_MIN",
    window_s=60.0,
    default_max=6,
    message="too many video downloads — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


# progress_id → the keep_video task still running past its transcription
# handler. Nothing reads it back: it is the strong reference that keeps
# asyncio from collecting a detached task mid-flight (the loop only holds
# tasks weakly). Bounded by the number of in-flight video fetches; each task
# pops its own key in its finally.
_VIDEO_TASKS: "dict[str, asyncio.Task]" = {}


_VIDEO_MIN_HEIGHT, _VIDEO_MAX_HEIGHT = 144, 4320


def _clamp_video_height(value) -> "int | None":
    """A client's height cap as an int in 144..4320, else None (= best)."""
    if value is None:
        return None
    try:
        h = int(value)
    except (TypeError, ValueError, OverflowError):   # JSON 1e999 / Infinity
        return None
    if h <= 0:
        return None
    return max(_VIDEO_MIN_HEIGHT, min(_VIDEO_MAX_HEIGHT, h))


def _clean_video_format(value) -> "str | None":
    """A client's rung choice: a yt-dlp format id the ladder listed, or
    None (= best). Regex-gated here; pick_rung ignores ids no longer on
    the ladder, so a stale choice degrades to the height rule."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if (v and _udl.FORMAT_ID_RE.match(v)) else None


def _video_state(**fields) -> dict:
    """The progress entry's `video` sub-object: one dict the client renders
    as the Video row beside the Audio stage. `state` in {done, failed,
    cancelled} is terminal — the poller stops on it. `total_approx` says
    whether `total_bytes` is an estimate (the client then prints "≈" and
    never lets the bar run past it)."""
    base = {"state": "queued", "progress": None, "downloaded_bytes": None,
            "total_bytes": None, "total_approx": False, "height": None,
            "container": None, "label": None, "vcodec": None, "acodec": None,
            "format_id": None, "media_id": None, "expires_at": None,
            "bytes": None, "error": None}
    base.update(fields)
    return base


async def _download_video_for_run(pid: "str | None", url: str, rung: dict, *,
                                  capped: bool, user_id: "str | None",
                                  protect: "str | None",
                                  run_finished: "list[bool]",
                                  mirror_stage: bool = False,
                                  job_row: bool = False) -> dict:
    """Fetch the VIDEO of `url` at `rung` into the media store and report it
    through the progress entry's `video` sub-object (and, for the on-demand
    route, the entry's own stage/progress). Returns the terminal state dict;
    never raises except for task cancellation.

    The transcription handler may return before this finishes: its finally
    then leaves the progress entry to us (`run_finished`), so the client can
    keep polling for `video.state` and still cancel the fetch. `job_row`:
    this run owns the durable job row under `pid`, whose stored result is
    patched with the outcome when it was written while we were pending."""
    state = _video_state(height=rung.get("height"),
                         container=rung.get("container") or "mkv",
                         total_bytes=rung.get("approx_bytes"),
                         total_approx=bool(rung.get("bytes_approx")),
                         label=rung.get("label"), vcodec=rung.get("vcodec"),
                         acodec=rung.get("acodec"),
                         format_id=rung.get("format_id"))
    _fid = rung.get("format_id")
    _format_ids = ((_fid, rung.get("audio_format_id"))
                   if isinstance(_fid, str) and _fid else None)
    _legs: dict = {}
    if _format_ids:
        if rung.get("video_bytes"):
            _legs[_format_ids[0]] = int(rung["video_bytes"])
        if _format_ids[1] and rung.get("audio_bytes"):
            _legs[_format_ids[1]] = int(rung["audio_bytes"])

    def _pub(**fields) -> None:
        state.update(fields)
        extra: dict = {}
        if mirror_stage and state["state"] in ("queued", "downloading"):
            extra = {"stage": "downloading", "progress": state["progress"],
                     "total_bytes": state["total_bytes"]}
        tx_progress._progress_set(pid, video=dict(state), **extra)

    cancelled = False
    try:
        _pub(state="queued")
        async with _url_staging_job() as job:
            async with tx_models._get_url_download_semaphore():
                if tx_progress._cancel_requested(pid):
                    raise _udl.UrlCancelled()
                _pub(state="downloading")
                path = await _udl.download_video(
                    url, dest_dir=job,
                    max_bytes=int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000)),
                    max_height=(rung.get("height") if capped else None),
                    container=state["container"],
                    expected_total=rung.get("approx_bytes"),
                    format_ids=_format_ids, leg_estimates=(_legs or None),
                    timeout=float(getattr(cfg, "URL_VIDEO_DOWNLOAD_TIMEOUT_S", 3600)),
                    progress_cb=lambda f, tot, done: _pub(
                        state="downloading", progress=f, total_bytes=tot,
                        downloaded_bytes=done),
                    cancel_check=lambda: tx_progress._cancel_requested(pid))
            # Report the container that LANDED: an un-merged single download
            # keeps the site's own extension (a webm behind an "mkv" rung), and
            # the client names its export after this field.
            _landed = os.path.splitext(path)[1].lstrip(".").lower()
            _pub(state="registering", progress=1.0,
                 **({"container": _landed} if _landed else {}))
            size = os.path.getsize(path)
            # Teach the ledger what this site's estimate was worth: the next
            # preview of a fragmented rung from the same extractor scales its
            # peak-bitrate numbers by actual/estimated.
            _est = rung.get("approx_bytes")
            # Measured against the UNSCALED estimate: against the already-scaled
            # one the EWMA settles on sqrt(true ratio) instead of the ratio.
            _raw = rung.get("raw_approx_bytes") or _est
            if rung.get("bytes_approx") and _raw and size > 0:
                from faster_whisper_backend.runtime import stage_rates as _rates
                _rates.record(_udl.RATIO_STAGE, rung.get("extractor"),
                              rung.get("protocol"), None, size / float(_raw))
                logger.info("[url-dl] video estimate %.1f MB → actual %.1f MB "
                            "(ratio %.2f, %s/%s)", (_est or _raw) / 1e6, size / 1e6,
                            size / float(_raw), rung.get("extractor"),
                            rung.get("protocol"))
            mid = await asyncio.to_thread(
                url_media_store.register, path, user_id=user_id, kind="video",
                protect=({protect} if protect else None))
            if mid is None:
                _pub(state="failed", error="the server could not retain the video")
            else:
                _pub(state="done", media_id=mid, expires_at=url_media_store.expires_at_unix(mid),
                     bytes=size)
                logger.info("[url-dl] video retained (%s, %.1f MB, host %s)",
                            state["container"], size / 1e6, _url_host_for_log(url))
    except _udl.UrlCancelled:
        _pub(state="cancelled")
    except asyncio.CancelledError:
        cancelled = True
        _pub(state="cancelled")
        raise
    except _udl.UrlDownloadError as e:
        # str() is client-safe by the module's contract.
        logger.info("[url-dl] video download failed (host %s): %s",
                    _url_host_for_log(url), _log_safe(str(e)))
        _pub(state="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 — never a raw error to the client
        logger.error("[url-dl] video download error (host %s): %s",
                     _url_host_for_log(url), _log_safe(str(e)))
        _pub(state="failed", error="video download failed")
    finally:
        if pid:
            _VIDEO_TASKS.pop(pid, None)
            if cancelled and run_finished[0]:
                tx_progress._progress_close(pid)
    try:
        if pid and job_row and run_finished[0]:
            # The handler stored `source_video_pending`; a client re-attaching
            # via /v1/jobs/{id}/result needs the outcome. (A cancelled task
            # never gets here: the scrub then simply drops the flag.)
            await asyncio.to_thread(_jobs_attach_video_sync, pid, dict(state))
    finally:
        # Closed only AFTER the attach: /result keeps the pending flag while
        # the progress entry is open, so closing first left a window where
        # the row carried neither the flag nor the video keys.
        if pid and run_finished[0]:
            tx_progress._progress_close(pid)
    return dict(state)


def _jobs_attach_video_sync(pid: str, state: dict) -> None:
    """Swap a finished job row's `source_video_pending` for the video keys
    the response would have carried had the fetch ended in time. Only a
    `done` row whose payload still holds the flag is touched; swallows and
    logs like every ledger helper."""
    try:
        row = _jobs_store.get(pid)
        for _ in range(10):
            # The fetch can end while the handler's own finish is still on
            # its thread; blocking is fine here (worker thread).
            if not row or row.get("state") != "running":
                break
            time.sleep(0.1)
            row = _jobs_store.get(pid)
        if not row or row.get("state") != "done":
            return

        def _swap(payload: dict) -> bool:
            if not payload.pop("source_video_pending", None):
                return False
            payload.update(_video_response_keys(None, state))
            return True
        # patch_result keeps finished_at / the TTL where the run's own
        # finish put them: the video landing is not a second finish.
        _jobs_store.patch_result(pid, _swap)
    except Exception as e:  # noqa: BLE001 — never fail the fetch on the ledger
        logger.warning("[jobs] could not attach the video to the job: %s", e)


def _video_response_keys(task: "asyncio.Task | None",
                         result: "dict | None") -> dict:
    """The transcription response's video keys: the id when the fetch already
    finished, `source_video_pending` while it still runs (the client keeps
    polling the progress id), or the client-safe error."""
    if task is not None:
        if not task.done():
            return {"source_video_pending": True}
        if task.cancelled():
            st = _video_state(state="cancelled")
        else:
            try:
                st = task.result()
            except Exception:  # noqa: BLE001 — the task already logged it
                st = _video_state(state="failed", error="video download failed")
    elif result is not None:
        st = result
    else:
        return {}
    if st.get("state") == "done":
        return {
            "source_video_media_id": st.get("media_id"),
            "source_video_expires_at": st.get("expires_at"),
            "source_video_height": st.get("height"),
            "source_video_container": st.get("container"),
            "source_video_bytes": st.get("bytes"),
        }
    return {"source_video_error": (
        st.get("error") or ("cancelled" if st.get("state") == "cancelled"
                            else "video download failed"))}


def _url_host_for_log(url: str) -> str:
    """Best-effort hostname for log lines — never the full URL (it can carry
    tokens/identifiers we don't want in logs). Delegates to the module that
    owns that contract."""
    return _udl.host_for_log(url)


@asynccontextmanager
async def _url_staging_job():
    """A private job dir in the media store's staging area for one link
    fetch, removed on every exit path (sweep() catches a crashed process).
    The removal runs off the loop — a cancelled or failed video fetch
    leaves a multi-GB partial or thousands of fragments behind — and is
    shielded, so a cancel landing on it cannot skip it."""
    job = url_media_store.new_staging_job()
    try:
        yield job
    finally:
        await asyncio.shield(asyncio.to_thread(shutil.rmtree, job, True))


async def _guarded_audio_download(pid: "str | None", url: str, dest_dir: str,
                                  *, max_bytes: "int | None" = None) -> str:
    """A link's whole AUDIO into `dest_dir` through the guarded download()
    under the URL download semaphore, reporting under `pid` and honouring
    its cancel. Raises UrlDownloadError / UrlCancelled / _ClientCancelled."""
    async with tx_models._get_url_download_semaphore():
        tx_progress._check_cancelled(pid)
        tx_progress._progress_set(pid, stage="downloading", progress=None)
        return await _udl.download(
            url, dest_dir=dest_dir, max_bytes=max_bytes,
            timeout=float(getattr(cfg, "URL_DOWNLOAD_TIMEOUT_S", 900)),
            progress_cb=lambda f, tot: tx_progress._progress_set(
                pid, stage="downloading", progress=f, total_bytes=tot),
            cancel_check=lambda: tx_progress._cancel_requested(pid))


def _reset_for_tests() -> None:
    """Drop the detached keep_video task references (tests/conftest.py
    _RESET_HOOKS); cleared in place."""
    _VIDEO_TASKS.clear()
