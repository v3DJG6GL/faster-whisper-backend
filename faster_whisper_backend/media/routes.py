"""Transcribe-from-URL and retained-media routes: /v1/audio/url-preview,
url-subtitles, url-language, url-media/{id}, url-media/video, url-media/audio,
and the video export (/v1/audio/media upload, /streams, /package), with
their per-identity limiters. The fetch helpers the transcription handler
shares live in media/video.py.
"""
import asyncio
import contextlib
import errno
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from faster_whisper_backend.auth import rate_limit as _rl
from faster_whisper_backend.auth.dependencies import get_current_user as _get_current_user_dep
from faster_whisper_backend.core import jobs
from faster_whisper_backend.core import store_common
from faster_whisper_backend.core.languages import (
    TRANSLATE_CODE_RE as _TRANSLATE_CODE_RE, language_label)
from faster_whisper_backend.media import media_store as url_media_store
from faster_whisper_backend.media import video as media_video
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.transcription import models as tx_models
from faster_whisper_backend.transcription import progress as tx_progress
from faster_whisper_backend.audio import transcode as _transcode
from faster_whisper_backend.media import download as _udl
from faster_whisper_backend.media import language_check as _lc
from faster_whisper_backend.media import segmented as _seg
from faster_whisper_backend.media import subtitle_mux as _pk
from faster_whisper_backend.media import subtitles as _subs

logger = logging.getLogger("whisper-api")
_log_safe = store_common.log_safe

router = APIRouter()


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


# Per-identity window for subtitle fetches: each one re-probes the link and
# GETs up to 8 small tracks from the site — the preview limiter's reasoning.
_url_subtitles_rate = _rl.FixedWindow(
    config_field="URL_SUBTITLES_RATE_PER_MIN",
    window_s=60.0,
    default_max=6,
    message="too many subtitle downloads — slow down "
            "({limit}/min; retry in {retry_after}s)",
)

# Per-identity window for link language checks — the costliest URL route: a
# whole audio download from a third party AND three detections on the GPU.
_url_language_rate = _rl.FixedWindow(
    config_field="URL_LANGUAGE_RATE_PER_MIN",
    window_s=60.0,
    default_max=4,
    message="too many language checks — slow down "
            "({limit}/min; retry in {retry_after}s)",
)


_VIDEO_MIME = {
    "mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska",
    "mov": "video/quicktime",
}


async def _url_request(request: Request, user: dict,
                       rate: "_rl.FixedWindow | None", *,
                       switch: "str | None" = None,
                       what: str = "") -> "tuple[dict, str]":
    """The prelude every POST /v1/audio/url-* route shares: the URL feature
    (then the route's own `switch`) gates with a curated 403, the route's
    per-identity window counts the call, and the JSON body must carry a
    non-empty `url`. Returns (body, url)."""
    if not getattr(cfg, "URL_DOWNLOAD_ENABLED", False):
        raise HTTPException(status_code=403,
                            detail="URL download is not enabled on this server")
    if switch and not getattr(cfg, switch, False):
        raise HTTPException(status_code=403,
                            detail=f"{what} is not enabled on this server")
    if rate is not None:
        rate.hit(_rl.identity_key(user, request))
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body is a caller error
        raise HTTPException(status_code=422, detail="expected a JSON body")
    url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(status_code=422, detail="expected {\"url\": …}")
    return body, url


async def _probe_link(url: str, what: str, failed: "str | None" = None):
    """The policy-gated probe a url-* route answers from: a rejected link is
    a client-safe 400, anything else a logged generic 500 (`failed`, else
    "<what> failed") — never a raw error."""
    _uhost = media_video._url_host_for_log(url)
    try:
        return await _udl.probe(
            url, timeout=float(getattr(cfg, "URL_PREVIEW_TIMEOUT_S", 20)))
    except _udl.UrlDownloadError as e:
        # str() is client-safe by the module's contract.
        logger.info("[url-dl] %s rejected (host %s): %s", what, _uhost,
                    _log_safe(str(e)))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — never forward raw errors
        logger.error("[url-dl] %s failed (host %s): %s", what, _uhost,
                     _log_safe(str(e)))
        raise HTTPException(status_code=500, detail=failed or f"{what} failed")


@router.post("/v1/audio/url-preview")
async def url_preview(request: Request,
                      user: dict = Depends(_get_current_user_dep)):
    """Metadata for a pasted media link, WITHOUT downloading: title,
    duration, uploader, and a server-proxied thumbnail (data: URI — the
    client never talks to the media site itself). Advisory: the client may
    still POST a URL whose preview failed; the download re-checks the same
    policy authoritatively. Client-safe 400s from the policy taxonomy."""
    _body, url = await _url_request(request, user, _url_preview_rate)
    _uhost = media_video._url_host_for_log(url)
    logger.info("[url-dl] preview requested (host %s)", _uhost)
    info = await _probe_link(url, "preview", "link preview failed")
    logger.info(
        "[url-dl] preview ok (host %s): extractor=%s duration=%s est_bytes=%s"
        " language=%s subtitle_tracks=%d",
        _uhost, info.extractor_key,
        f"{info.duration:.0f}s" if info.duration is not None else "?",
        info.filesize_approx if info.filesize_approx is not None else "?",
        info.language or "?", len(info.subtitle_tracks))
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
        # The video rungs the site offers (highest first, one trailing
        # "audio only" entry), each with the container a merge would give
        # and an over-cap flag: a video rung against retained_max_bytes (the
        # largest file the media store keeps), the "audio only" rung against
        # media_max_bytes (the transcription ceiling — a run fetches it
        # without keeping it). [] when video is off or the link has none.
        "video_ladder": info.video_ladder,
        "media_max_bytes": int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000)),
        "retained_max_bytes": url_media_store.max_retainable_bytes(),
        # The longest link this server downloads (the client flags a longer
        # one before "Add link").
        "url_max_duration_s": int(getattr(cfg, "URL_MAX_DURATION_S", 0) or 0),
        # The spoken language the site names (null when it names none) and
        # its own subtitle tracks, ids only — the source URLs stay here.
        "language": info.language,
        "subtitle_tracks": (info.subtitle_tracks
                            if getattr(cfg, "URL_SUBTITLES_ENABLED", False)
                            else []),
    }


@router.post("/v1/audio/url-subtitles")
async def url_subtitles(request: Request,
                        user: dict = Depends(_get_current_user_dep)):
    """A link's own subtitle tracks as text: {url, tracks: [ids ≤ 8 from
    the preview]} → {tracks: [{id, lang, kind, ext, text}], failed: [{id,
    error}]}. The client names tracks by id only; a fresh probe re-applies
    the URL policy and mints fresh signed source URLs, which never leave
    the server (nor its log — they carry tokens)."""
    body, url = await _url_request(request, user, _url_subtitles_rate,
                                   switch="URL_SUBTITLES_ENABLED",
                                   what="subtitle download")
    ids = body.get("tracks")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= _subs.MAX_FETCH
            or not all(isinstance(i, str) and _subs.TRACK_ID_RE.match(i)
                       for i in ids)):
        raise HTTPException(
            status_code=422,
            detail=f"expected 1–{_subs.MAX_FETCH} subtitle track ids")
    ids = list(dict.fromkeys(ids))
    _uhost = media_video._url_host_for_log(url)
    logger.info("[url-dl] subtitles requested (host %s): %s", _uhost,
                ", ".join(ids))
    _t0 = time.perf_counter()
    info = await _probe_link(url, "subtitles", "subtitle download failed")
    tracks, failed = await _subs.fetch_tracks(info, ids)
    # Ids, sizes and the client-safe reasons only: the source URLs are
    # signed (YouTube's carry a pot token) and never reach the log.
    (logger.warning if failed else logger.info)(
        "[url-dl] subtitles (host %s): %d fetched%s, %d failed%s in %.1fs",
        _uhost, len(tracks),
        "".join(f" · {t['id']} {len(t['text']) / 1024:.0f} KB" for t in tracks),
        len(failed),
        "".join(f" · {f['id']}: {_log_safe(f['error'])}" for f in failed),
        time.perf_counter() - _t0)
    return {"tracks": tracks, "failed": failed}


@router.get("/v1/audio/url-media/{media_id}")
async def url_media(media_id: str,
                    user: dict = Depends(_get_current_user_dep)):
    """The retained audio of a finished transcribe-from-URL run, so the
    client can pull ONE local copy for playback. Short-lived (URL_MEDIA_TTL_S,
    wiped on restart); unknown, expired and foreign-owner ids all
    answer the same 404 — no oracle. FileResponse handles Range, so the
    client player can seek without re-downloading."""
    # Either producer hands out these ids: a link run (URL download) or an
    # uploaded / retained video (media packaging, on by default). Gating on
    # the URL feature alone left the packaging ids live but unreadable.
    if not (getattr(cfg, "URL_DOWNLOAD_ENABLED", False)
            or getattr(cfg, "MEDIA_PACKAGE_ENABLED", True)):
        raise HTTPException(status_code=403,
                            detail="media retention is not enabled on this server")
    if not url_media_store.MEDIA_ID_RE.match(media_id):
        raise HTTPException(status_code=422, detail="malformed media id")
    entry = url_media_store.resolve_entry(media_id, user_id=user.get("user_id"))
    if entry is None:
        raise HTTPException(status_code=404, detail="media not found")
    path, ext = entry["path"], entry["ext"]
    if entry.get("kind") == "video":
        mime = _VIDEO_MIME.get(ext, "application/octet-stream")
    else:
        mime = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
            "oga": "audio/ogg", "opus": "audio/ogg", "flac": "audio/flac",
            "m4a": "audio/mp4", "mp4": "audio/mp4", "aac": "audio/aac",
            "webm": "audio/webm", "mka": "audio/x-matroska",
            "mkv": "audio/x-matroska",
        }.get(ext, "application/octet-stream")
    return FileResponse(
        path=path,
        media_type=mime,
        filename=f"{media_id}.{ext}",
        # Owner-gated audio: a shared cache must never answer the next,
        # differently-authenticated caller (same stance as captures audio).
        headers={"Cache-Control": "no-store"},
    )


async def _url_media_on_demand(user: dict, body: dict, url: str, what: str,
                               fetch) -> dict:
    """The on-demand link fetch both url-media routes (and the language
    check) share: a job row, the progress entry under the body's optional
    `progress_id` (the shared cancel route aborts it), a fresh policy probe,
    then `await fetch(pid, validated_url, info)` for the route's own work
    and answer. Client-safe errors only: policy/download 400, cancel 499."""
    _pid = tx_progress._claim_progress_id(body.get("progress_id"))
    _user_id = user.get("user_id")
    _uhost = media_video._url_host_for_log(url)
    request_id = uuid.uuid4().hex
    jobs.job_start("download", id=request_id,
                   user=user.get("username") or _user_id, key=user.get("key_id"),
                   user_id=_user_id, detail=f"{what} · {_uhost}")
    tx_progress._bind_job_pid(_pid, request_id)
    logger.info("[url-dl] %s requested on demand (host %s)", what, _uhost)
    try:
        tx_progress._progress_set(_pid, stage="resolving", progress=None,
                      owner=(_user_id or user.get("key_id")))
        tx_progress._check_cancelled(_pid)
        info = await _probe_link(url, what)
        return await fetch(_pid, info.url, info)
    except (tx_progress._ClientCancelled, _udl.UrlCancelled):
        raise HTTPException(status_code=499, detail="cancelled by the client")
    except _udl.UrlDownloadError as e:
        # str() is client-safe by the module's contract.
        logger.info("[url-dl] %s rejected (host %s): %s", what, _uhost,
                    _log_safe(str(e)))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — never forward raw errors
        logger.error("[url-dl] %s failed (host %s): %s", what, _uhost,
                     _log_safe(str(e)))
        raise HTTPException(status_code=500, detail=f"{what} failed")
    finally:
        if _pid:
            tx_progress._progress_close(_pid)
            tx_progress._JOB_BY_PID.pop(_pid, None)
        jobs.job_end(request_id)


async def _download_link_audio(pid: "str | None", url: str,
                               user_id: "str | None") -> str:
    """A link's AUDIO into the media store through the guarded download()
    (the whole file — never yt-dlp's ranged ffmpeg fetch, which bypasses
    the SSRF guard) under the URL download semaphore; returns the media id,
    registered with `source_url` so a run of the same link can reuse it.
    Raises UrlDownloadError / UrlCancelled / _ClientCancelled."""
    async with contextlib.AsyncExitStack() as job_stack:
        # The staging job is made once the download slot is ours (see
        # _guarded_audio_download) and outlives it until register().
        path = await media_video._guarded_audio_download(
            pid, url, lambda: job_stack.enter_async_context(
                media_video._url_staging_job()),
            max_bytes=url_media_store.max_retainable_bytes())
        size = os.path.getsize(path)
        mid = await asyncio.to_thread(url_media_store.register, path, user_id=user_id,
                                      source_url=url)
        if mid is None:
            raise _udl.UrlDownloadError("the server could not retain the audio")
        logger.info("[url-dl] audio retained (%s, %.1f MB, host %s)",
                    os.path.splitext(path)[1].lstrip(".") or "?", size / 1e6,
                    media_video._url_host_for_log(url))
        return mid


async def _segmented_pieces(pid: "str | None", url: str, source: dict,
                            starts: "list[float]",
                            seconds: float) -> "tuple[list | None, str]":
    """The language check's pieces from only the stream segments under them
    (media/segmented.py), for a link whose audio is HLS/DASH: (pieces, how).
    (None, why) when that path cannot serve this stream — the caller then
    downloads the whole file, so nothing but a cancel escapes from here."""
    import urllib.error
    t0 = time.perf_counter()
    try:
        async with contextlib.AsyncExitStack() as job_stack:
            async with tx_models._get_url_download_semaphore():
                tx_progress._check_cancelled(pid)
                # Made once the slot is ours: a dir made while queued could be
                # reaped as stale (media_store._reap_stale_staging) mid-wait.
                job = await job_stack.enter_async_context(
                    media_video._url_staging_job())
                tx_progress._progress_set(pid, stage="downloading", progress=None)
                files, got = await _seg.fetch_pieces(
                    source, starts, seconds, job,
                    cancel_check=lambda: tx_progress._cancel_requested(pid),
                    progress_cb=lambda f: tx_progress._progress_set(
                        pid, stage="downloading", progress=f))
            audio = await asyncio.to_thread(lambda: [
                _transcode.decode_span_16k(path, skip, seconds)
                for path, skip in files])
        if not all(len(a) for a in audio):
            raise _seg.Unsupported("a piece decoded empty")
    except (tx_progress._ClientCancelled, _udl.UrlCancelled):
        raise
    except Exception as e:  # noqa: BLE001 — any failure: the full download
        why = (str(e) if isinstance(e, (_seg.Unsupported, _udl.UrlDownloadError))
               else f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError)
               else type(e).__name__)  # transport text can name the URL
        logger.info("[url-dl] language check samples no segments (host %s): "
                    "%s — downloading the whole audio",
                    media_video._url_host_for_log(url), _log_safe(why))
        return None, why
    return audio, (f"chunks {got['segments']} seg / {got['bytes'] / 1e6:.1f} MB"
                   f" in {time.perf_counter() - t0:.1f}s")


@router.post("/v1/audio/url-media/video")
async def url_media_video(request: Request,
                          user: dict = Depends(_get_current_user_dep)):
    """Fetch a link's VIDEO on demand (the export panel, for a run that did
    not keep it or whose copy expired) into the media store: {media_id,
    expires_at, height, container, bytes}. Same policy, guard and caps as
    the transcription's download; progress/cancel through the shared
    registry when the body carries a `progress_id` (the entry reports
    stage "downloading" plus the `video` sub-object)."""
    body, url = await _url_request(request, user, media_video._url_video_rate,
                                   switch="URL_VIDEO_ENABLED",
                                   what="video download")
    max_height = media_video._clamp_video_height(body.get("max_height"))
    format_id = media_video._clean_video_format(body.get("format_id"))

    async def _fetch(pid, url, info) -> dict:
        rung = _udl.pick_rung(info.video_ladder, max_height, format_id)
        if rung is None:
            raise HTTPException(status_code=400,
                                detail="this link has no video track")
        if rung.get("over_cap"):
            raise HTTPException(status_code=400,
                                detail="the video exceeds the server's size limit")
        tx_progress._progress_set(pid, stage="downloading", progress=None,
                      total_bytes=rung.get("approx_bytes"),
                      step=(info.extractor_key or None))
        state = await media_video._download_video_for_run(
            pid, url, rung, capped=max_height is not None,
            user_id=user.get("user_id"), protect=None, run_finished=[False],
            mirror_stage=True)
        if state.get("state") == "done":
            return {"media_id": state["media_id"],
                    "expires_at": state["expires_at"],
                    "height": state.get("height"),
                    "container": state.get("container"),
                    "bytes": state.get("bytes")}
        if state.get("state") == "cancelled":
            raise HTTPException(status_code=499, detail="cancelled by the client")
        raise HTTPException(status_code=400,
                            detail=state.get("error") or "video download failed")
    return await _url_media_on_demand(user, body, url, "video download", _fetch)


@router.post("/v1/audio/url-media/audio")
async def url_media_audio(request: Request,
                          user: dict = Depends(_get_current_user_dep)):
    """Fetch a link's AUDIO on demand into the media store: {media_id,
    expires_at, ext, bytes}. For a run whose transcript comes from the
    site's own subtitles — it still keeps the audio for playback. The same
    download (policy, guard, caps, semaphore) a source_url transcription
    makes, so no limiter of its own; progress/cancel as the video route."""
    body, url = await _url_request(request, user, None)

    async def _fetch(pid, url, info) -> dict:
        mid = await _download_link_audio(pid, url, user.get("user_id"))
        entry = url_media_store.resolve_entry(mid, user_id=user.get("user_id")) or {}
        return {"media_id": mid, "expires_at": url_media_store.expires_at_unix(mid),
                "ext": entry.get("ext"), "bytes": entry.get("size")}
    return await _url_media_on_demand(user, body, url, "audio download", _fetch)


def _has_speech(audio) -> bool:
    """Whether VAD hears speech in a language-check piece (silence, music
    and an empty piece past the end do not). faster-whisper 1.2 no longer
    raises for a piece without speech: detect_language then runs on padded
    silence and answers ("en", 0.34), so a silent or music-only link would
    claim English. Same VAD defaults detect_language's vad_filter uses."""
    if audio is None or len(audio) == 0:
        return False
    from faster_whisper.vad import get_speech_timestamps
    return bool(get_speech_timestamps(audio))


@router.post("/v1/audio/url-language")
async def url_language(request: Request,
                       user: dict = Depends(_get_current_user_dep)):
    """Which language does a link speak? {url, model?, progress_id?} →
    {language, probability, verdict: detected|mixed|unknown, also, pieces:
    [{at, language, probability}], media_id, media_expires_at}. A link
    whose audio is a segmented stream (HLS/DASH) has only the segments under
    three 20 s pieces fetched (media/segmented.py; media_id null — the run
    downloads normally); any other link, or a stream that path cannot
    serve, has the WHOLE audio downloaded through the guarded download (a
    ranged ffmpeg fetch would bypass the SSRF guard) and kept in the media
    store, so the run that follows reuses it (`prefetched_media_id`). Then
    Whisper's language detection listens to the pieces
    (media/language_check.py) under the inference semaphore and a model
    lease. Cancel through the shared cancel route on `progress_id`."""
    body, url = await _url_request(request, user, _url_language_rate,
                                   switch="URL_LANGUAGE_CHECK_ENABLED",
                                   what="the language check")
    _model = body.get("model")
    model_name = tx_models._resolve_model_name(_model.strip() if isinstance(_model, str) else "")
    # Gate the model before the link's audio is fetched, not after.
    tx_models._check_model_name(model_name)

    def _english_only() -> HTTPException:
        return HTTPException(
            status_code=400,
            detail=f"the language check needs a multilingual model; "
                   f"'{model_name}' is English-only")
    # A ".en" name is English-only by Whisper's own naming: refuse it before
    # the download (the loaded model's flag below stays the backstop for a
    # name that does not say so).
    if model_name.rstrip("/").rsplit("/", 1)[-1].lower().endswith(".en"):
        raise _english_only()
    _user_id = user.get("user_id")

    def _detect(model, audio) -> "tuple[str | None, float]":
        try:
            lang, prob, _all = model.detect_language(audio=audio, vad_filter=True)
        except ValueError:
            # Older faster-whisper: no speech left after VAD, or an empty piece.
            return None, 0.0
        return lang, float(prob)

    async def _check(pid, url, info) -> dict:
        t0 = time.perf_counter()
        starts = _lc.piece_starts(info.duration)
        audio, mid = None, None
        if info.segmented is not None:
            audio, how = await _segmented_pieces(
                pid, url, info.segmented, starts, _lc.PIECE_SECONDS)
        if audio is None:
            _d0 = time.perf_counter()
            mid = await _download_link_audio(pid, url, _user_id)
            how = f"full download in {time.perf_counter() - _d0:.1f}s"
            entry = url_media_store.resolve_entry(mid, user_id=_user_id)
            if entry is None:
                raise _udl.UrlDownloadError("the server could not retain the audio")
            try:
                audio = await asyncio.to_thread(
                    _transcode.decode_pieces_16k, entry["path"], starts,
                    _lc.PIECE_SECONDS)
            except Exception as e:  # noqa: BLE001 — PyAV's text names the path
                logger.info("[url-dl] language check could not decode the audio "
                            "(host %s): %s", media_video._url_host_for_log(url),
                            _log_safe(type(e).__name__))
                raise _udl.UrlDownloadError(
                    "the link's audio could not be decoded") from None
        # The VAD gate is CPU-only: run it before the model load and the GPU
        # gate, so a silent or music-only link never loads a model or waits
        # for a GPU slot only to answer "unknown".
        speech = await asyncio.to_thread(lambda: [_has_speech(p) for p in audio])
        heard: "list[tuple[str | None, float]]" = [(None, 0.0)] * len(audio)
        wait_s = 0.0
        if any(speech):
            tx_progress._progress_set(pid, stage="waiting", progress=None)
            model = await tx_models._get_or_load_model(model_name, lease=True)
            try:
                # CTranslate2 raises RuntimeError for an English-only model.
                if not getattr(getattr(model, "model", None), "is_multilingual", True):
                    raise _english_only()
                tx_progress._check_cancelled(pid)
                _w0 = time.perf_counter()
                async with tx_models.get_inference_semaphore():
                    wait_s = time.perf_counter() - _w0
                    for i, piece in enumerate(audio):
                        if not speech[i]:
                            continue
                        tx_progress._check_cancelled(pid)
                        tx_progress._progress_set(pid, stage="transcribing", step="language",
                                      progress=i / len(audio), model=model_name)
                        heard[i] = await asyncio.to_thread(_detect, model, piece)
            finally:
                tx_models._release_model_lease(model_name)
        result = _lc.vote(heard)
        pieces = [{"at": at, "language": lang, "probability": round(p, 3)}
                  for at, (lang, p) in zip(starts, heard)]
        logger.info(
            "[url-dl] language check (host %s): %s %s p=%.2f%s · pieces %s"
            " · %s · model %s · gpu wait %.1fs · %.1fs total",
            media_video._url_host_for_log(url), result["verdict"], result["language"] or "?",
            result["probability"],
            f" also {','.join(result['also'])}" if result["also"] else "",
            ", ".join(f"{p['at']:.0f}s {p['language'] or '-'} {p['probability']:.2f}"
                      for p in pieces),
            how, model_name, wait_s, time.perf_counter() - t0)
        return {**result, "pieces": pieces, "media_id": mid,
                "media_expires_at": url_media_store.expires_at_unix(mid) if mid else None}
    return await _url_media_on_demand(user, body, url, "language check", _check)


# ── Media export: upload, stream facts, subtitle packaging ──────────────────
# The client exports a video WITH its subtitle tracks: it generates the SRTs
# itself (edits, renames and speaker colours included), the server muxes them
# into the retained video — a link's, or one the client uploads here for the
# purpose — as soft subtitle streams, and streams the result back.
_media_upload_rate = _rl.FixedWindow(
    config_field="MEDIA_UPLOAD_RATE_PER_MIN",
    window_s=60.0,
    default_max=6,
    message="too many media uploads — slow down "
            "({limit}/min; retry in {retry_after}s)",
)
_media_package_rate = _rl.FixedWindow(
    config_field="MEDIA_PACKAGE_RATE_PER_MIN",
    window_s=60.0,
    default_max=12,
    message="too many video exports — slow down "
            "({limit}/min; retry in {retry_after}s)",
)
_media_package_inflight = _rl.InFlight(
    config_field="MEDIA_PACKAGE_MAX_INFLIGHT_PER_USER",
    default_max=1,
    message="you already have {limit} video export running — "
            "wait for it to finish",
)
_MEDIA_EXT_RE = re.compile(r"\A[a-z0-9]{1,5}\Z")
# Free space an upload must leave on the staging volume (the package route's
# margin): a near-full disk is refused before the transfer, not after it.
_UPLOAD_DISK_MARGIN = 64 * 1024 * 1024
_NO_DISK = "not enough disk space on the server"
# Letters and digits in any script (\w on a str pattern): FileResponse sends
# a non-ASCII name as RFC 5987 filename*=utf-8''…, so only controls, bidi
# marks, separators and punctuation need to go. NFC first, or a decomposed
# "Ü" would lose its combining mark (category Mn is not \w).
_MEDIA_FILENAME_RE = re.compile(r"[^\w .()\-]+")


def _package_gate() -> None:
    """403 when packaging is off, 503 when the server's ffmpeg cannot do it."""
    if not getattr(cfg, "MEDIA_PACKAGE_ENABLED", True):
        raise HTTPException(status_code=403,
                            detail="media packaging is not enabled on this server")
    caps = _pk.ffmpeg_capabilities()
    if not caps.available:
        raise HTTPException(status_code=503, detail=caps.reason or
                            "subtitle packaging is unavailable on this server")


def _media_streams_for(entry: dict, media_id: str) -> "dict | None":
    """Cached codec facts for a retained file (probed once per id)."""
    cached = url_media_store.probe_cache_get(media_id)
    if cached is not None:
        return cached
    try:
        facts = _pk.probe_streams(entry["path"]).as_dict()
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001 — a hostile file must not 500 the route
        logger.info("[package] stream probe failed for %s: %s", media_id,
                    _log_safe(str(e)))
        facts = _pk.unreadable_streams().as_dict()
    url_media_store.probe_cache_set(media_id, facts)
    return facts


@router.post("/v1/audio/media")
async def upload_media(request: Request,
                       user: dict = Depends(_get_current_user_dep)):
    """Raw-body upload of a local VIDEO into the media store (the client's
    own file, for packaging with its subtitles): `?ext=mp4`, the bytes as the
    body. Streamed to disk chunk by chunk against MEDIA_MAX_BYTES — never
    resident, never spooled twice (this is deliberately NOT multipart: the
    form parser spools the whole part before a handler sees a byte). Returns
    {media_id, expires_at, bytes}; the file lives URL_MEDIA_TTL_S."""
    _package_gate()
    _media_upload_rate.hit(_rl.identity_key(user, request))
    ext = (request.query_params.get("ext") or "").strip().lower()
    if not _MEDIA_EXT_RE.match(ext):
        raise HTTPException(status_code=422, detail="expected ?ext=<container>")
    # register() drops a file over RETAINED_MEDIA_MAX_BYTES on arrival:
    # refuse it here, before the transfer, naming the cap that binds.
    cap = url_media_store.max_retainable_bytes()
    too_large = ("upload too large"
                 if cap >= int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000))
                 else "upload too large for the server's media store "
                      "(RETAINED_MEDIA_MAX_BYTES)")
    _clen = request.headers.get("content-length")
    if _clen and _clen.isdigit() and int(_clen) > cap:
        raise HTTPException(status_code=413, detail=too_large)
    staging = url_media_store.staging_dir()
    if _clen and _clen.isdigit() and (
            shutil.disk_usage(staging).free < int(_clen) + _UPLOAD_DISK_MARGIN):
        raise HTTPException(status_code=507, detail=_NO_DISK)
    part = os.path.join(staging, f"upload-{uuid.uuid4().hex}.{ext}.part")
    received = 0
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), 0o600)
    from starlette.requests import ClientDisconnect
    try:
        try:
            with os.fdopen(fd, "wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > cap:
                        raise HTTPException(status_code=413, detail=too_large)
                    await asyncio.to_thread(f.write, chunk)
        except OSError as e:
            # A full disk (or quota) mid-stream — from a write or from the
            # close's flush: the curated answer, not a bare 500. The finally
            # drops the .part.
            if e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                raise HTTPException(status_code=507, detail=_NO_DISK) from None
            raise
        except ClientDisconnect:
            # main._max_body_mw cuts the receive channel at the same cap (a chunked
            # body declares no length) and flags it in the scope; anything
            # else is a client that went away mid-upload.
            if received >= cap or request.scope.get("state", {}).get("body_cap_hit"):
                raise HTTPException(status_code=413, detail=too_large)
            raise HTTPException(status_code=400, detail="upload interrupted")
        if received == 0:
            raise HTTPException(status_code=422, detail="empty upload")
        final = part[:-len(".part")]
        try:
            os.replace(part, final)
        except OSError:
            # The spool went away under us (a staging sweep, a wiped tmp
            # dir): a deliberate error instead of a bare traceback.
            raise HTTPException(status_code=500,
                                detail="upload staging file vanished")
        part = None
        media_id = await asyncio.to_thread(
            url_media_store.register, final, user_id=user.get("user_id"), kind="video")
        if media_id is None:
            try:
                os.unlink(final)
            except OSError:
                pass
            raise HTTPException(status_code=507,
                                detail="the server's media store is full")
    finally:
        if part:
            try:
                os.unlink(part)
            except OSError:
                pass
    logger.info("[package] upload retained (%.1f MB, %s)", received / 1e6, ext)
    return {"media_id": media_id, "expires_at": url_media_store.expires_at_unix(media_id),
            "bytes": received}


@router.get("/v1/audio/media/{media_id}/streams")
async def media_streams(media_id: str,
                        user: dict = Depends(_get_current_user_dep)):
    """Codec facts for a retained file — the export panel greys out MP4 with
    the reason before anyone waits for a mux. Same 404 for every miss."""
    _package_gate()
    if not url_media_store.MEDIA_ID_RE.match(media_id):
        raise HTTPException(status_code=422, detail="malformed media id")
    entry = url_media_store.resolve_entry(media_id, user_id=user.get("user_id"))
    if entry is None:
        raise HTTPException(status_code=404, detail="media not found")
    facts = await asyncio.to_thread(_media_streams_for, entry, media_id)
    if facts is None:
        raise HTTPException(status_code=503,
                            detail="stream probing is unavailable on this server (PyAV)")
    return facts


def _require_lang_code(value, field: str) -> str:
    """A body field that must hold one language code: stripped, else 422."""
    if not isinstance(value, str) or not _TRANSLATE_CODE_RE.match(value.strip()):
        raise HTTPException(status_code=422,
                            detail=f"{field} must be a language code")
    return value.strip()


# A lone UTF-16 surrogate survives json parsing ("\ud800") but not UTF-8
# encoding: in the SRT size check, or os.fsencode of ffmpeg's argv (a
# title=/handler_name= label), it is a bare 500 instead of a 422.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _clean_label(value) -> str:
    """A client track title: control characters and lone surrogates out, at
    most 64 chars; "" when absent or not a string."""
    return (re.sub(r"[\x00-\x1f\x7f\ud800-\udfff]", "", value).strip()[:64]
            if isinstance(value, str) else "")


def _track_index(body: dict, key: str, count: int) -> "int | None":
    """An optional subtitle-track index in the body: None, or 0..count-1
    (a bool is not an index), else 422."""
    value = body.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                              or not 0 <= value < count):
        raise HTTPException(status_code=422, detail=f"{key} is out of range")
    return value


class _PackageResponse(FileResponse):
    """FileResponse that removes the package's fwb-pkg- workdir however the
    response ends. Starlette runs `background` only after a normal body
    send: its Range branches (400 malformed, 416 unsatisfiable — honoured on
    POST too) return before it, and a client disconnect mid-stream raises
    past it, either way leaving a full copy of the video in TMPDIR until the
    next restart's sweep."""

    def __init__(self, *args, workdir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._workdir = workdir

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(asyncio.to_thread(shutil.rmtree, self._workdir, True))


@router.post("/v1/audio/media/{media_id}/package")
async def package_media(media_id: str, request: Request,
                        user: dict = Depends(_get_current_user_dep)):
    """Mux the client's SRT tracks into a retained video as soft subtitle
    streams and stream the file back. Body: {container: "mkv"|"mp4",
    subtitles: [{lang, label?, srt, default?, original?, hearing_impaired?}],
    default_track?, original_track?, audio_lang?, audio_label?, filename?}.
    The per-track booleans set that disposition on the track (several tracks
    may carry each one); the older `default_track` / `original_track`
    indices still work and combine with them (index match OR own flag).
    `original` is Matroska-only (MP4 has no such flag). MP4 only
    when the streams fit it (422 with code "mp4_incompatible" otherwise, and
    the reason). One packaging run per identity at a time."""
    _package_gate()
    if not url_media_store.MEDIA_ID_RE.match(media_id):
        raise HTTPException(status_code=422, detail="malformed media id")
    _key = _rl.identity_key(user, request)
    _media_package_rate.hit(_key)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body is a caller error
        raise HTTPException(status_code=422, detail="expected a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="expected a JSON object")
    container = body.get("container") or "mkv"
    if container not in _pk.CONTAINERS:
        raise HTTPException(status_code=422, detail="container must be mkv or mp4")
    caps = _pk.ffmpeg_capabilities()
    if container == "mp4" and not caps.mp4:
        raise HTTPException(
            status_code=422,
            detail="this server's ffmpeg has no mp4 muxer or mov_text encoder")
    raw_subs = body.get("subtitles")
    if raw_subs is None:
        raw_subs = []
    if not isinstance(raw_subs, list) or len(raw_subs) > _pk.MAX_TRACKS:
        raise HTTPException(status_code=422,
                            detail=f"subtitles must be a list of at most {_pk.MAX_TRACKS} tracks")
    tracks: "list[_pk.SubtitleTrack]" = []
    for i, t in enumerate(raw_subs):
        if not isinstance(t, dict):
            raise HTTPException(status_code=422, detail=f"subtitles[{i}] must be an object")
        lang = _require_lang_code(t.get("lang"), f"subtitles[{i}].lang")
        srt = t.get("srt")
        if (not isinstance(srt, str) or "-->" not in srt or "\x00" in srt
                or _SURROGATE_RE.search(srt)):
            raise HTTPException(status_code=422,
                                detail=f"subtitle track {i + 1} is not SRT")
        if len(srt.encode("utf-8")) > _pk.MAX_SRT_BYTES:
            raise HTTPException(status_code=422,
                                detail=f"subtitle track {i + 1} is larger than "
                                       f"{_pk.MAX_SRT_BYTES // (1024 * 1024)} MiB")
        label = _clean_label(t.get("label")) or language_label(lang)
        flags = {}
        for k in ("default", "original", "hearing_impaired"):
            v = t.get(k, False)
            if not isinstance(v, bool):
                raise HTTPException(status_code=422,
                                    detail=f"subtitles[{i}].{k} must be a boolean")
            flags[k] = v
        tracks.append(_pk.SubtitleTrack(lang=lang, label=label, srt=srt, **flags))
    default_track = _track_index(body, "default_track", len(tracks))
    original_track = _track_index(body, "original_track", len(tracks))
    audio_lang = body.get("audio_lang")
    if audio_lang is not None:
        audio_lang = _require_lang_code(audio_lang, "audio_lang")
    audio_label = _clean_label(body.get("audio_label")) or None
    filename = body.get("filename")
    stem = (_MEDIA_FILENAME_RE.sub(
        "", unicodedata.normalize("NFC", filename)).strip()[:80]
            if isinstance(filename, str) else "") or media_id
    entry = url_media_store.resolve_entry(media_id, user_id=user.get("user_id"))
    if entry is None:
        raise HTTPException(status_code=404, detail="media not found")
    facts = await asyncio.to_thread(_media_streams_for, entry, media_id)
    if facts is None:
        raise HTTPException(status_code=503,
                            detail="stream probing is unavailable on this server (PyAV)")
    if facts.get("unreadable"):
        # The probe could not open the file at all: that is not "no video".
        raise HTTPException(status_code=422,
                            detail={"code": "unreadable",
                                    "message": _pk.UNREADABLE_REASON})
    if not facts.get("video_codec"):
        raise HTTPException(status_code=422,
                            detail={"code": "no_video",
                                    "message": (
                                        "this media has only cover art, no video stream"
                                        if facts.get("cover_art_only")
                                        else "this media has no video stream")})
    if container == "mp4" and not facts.get("mp4_ok"):
        raise HTTPException(status_code=422,
                            detail={"code": "mp4_incompatible",
                                    "message": facts.get("mp4_reason")
                                    or "MP4 can't carry these streams — choose MKV"})
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if free < int(entry.get("size") or 0) + 64 * 1024 * 1024:
        raise HTTPException(status_code=507,
                            detail="not enough temporary disk space on the server")
    _took_slot = _media_package_inflight.acquire(_key)
    try:
        out = await _pk.package(
            entry["path"], tracks, container=container,
            default_track=default_track, original_track=original_track,
            audio_lang=audio_lang, audio_label=audio_label,
            timeout=float(getattr(cfg, "MEDIA_PACKAGE_TIMEOUT_S", 900)),
            # The real video can sit behind a cover-art stream.
            video_index=int(facts.get("video_index") or 0),
            video_codec=facts.get("video_codec"))
    except _pk.SubtitleParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except _pk.PackageTimeout as e:
        raise HTTPException(status_code=504, detail=str(e))
    except _pk.PackageError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if _took_slot:
            _media_package_inflight.release(_key)
    return _PackageResponse(
        path=out,
        media_type=_VIDEO_MIME[container],
        filename=f"{stem}.{container}",
        headers={"Cache-Control": "no-store"},
        workdir=os.path.dirname(out),
    )
