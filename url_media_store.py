"""Short-term retention for URL-downloaded audio (transcribe-from-URL).

The transcription request downloads a link's audio into a private job dir,
then hands the file to this store, which keeps it fetchable for a short
window via GET /v1/audio/url-media/{media_id} — the client pulls it ONCE
into its own local media store for playback and never needs it again.

Deliberately tiny and in-process:
  - The registry is a plain dict (GIL-atomic single-key updates, same stance
    as main._BATCH_PROGRESS). Ids die with the process, so startup_reset()
    wipes the directory — every file on disk without a registry entry is an
    orphan by definition. Multi-worker deployments are already documented as
    unsupported (SERVER_WORKERS: "keep at 1").
  - Bounded twice: per-entry TTL (URL_MEDIA_TTL_SEC) and a byte-capped LRU
    over the whole dir (URL_MEDIA_MAX_BYTES). sweep() runs from a lifespan
    task; register() also evicts inline so a burst can't overshoot until the
    next tick.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid

import config as cfg
from store_common import secure_dir, secure_file

logger = logging.getLogger("whisper-api")

# media_id -> {path, ext, user_id, created (time.monotonic), size}
_REG: "dict[str, dict]" = {}

# Extensions we expect yt-dlp to produce. Anything else keeps a neutral
# suffix — the pipeline sniffs content, and the client maps by Content-Type.
_KNOWN_EXTS = frozenset({
    "m4a", "mp4", "webm", "opus", "ogg", "oga", "mp3", "wav", "flac", "aac",
    "mka", "mkv",
})


def _dir() -> str:
    return getattr(cfg, "URL_MEDIA_DIR", "/data/url_media")


def startup_reset() -> None:
    """Wipe + recreate the retention dir. Called once from lifespan."""
    d = _dir()
    _REG.clear()
    try:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        secure_dir(d)
    except OSError as e:
        logger.error("[url-dl] retention dir unusable (%s): %s", d, e)


def register(src_path: str, *, user_id: "str | None") -> "str | None":
    """Move `src_path` into the retention dir under a fresh opaque id and
    return the media_id — or None when retention is unavailable (the
    transcription itself must not fail over a playback nicety)."""
    ext = os.path.splitext(src_path)[1].lstrip(".").lower()
    if ext not in _KNOWN_EXTS:
        ext = "bin"
    media_id = uuid.uuid4().hex
    dest = os.path.join(_dir(), f"{media_id}.{ext}")
    try:
        os.makedirs(_dir(), exist_ok=True)
        shutil.move(src_path, dest)
        secure_file(dest)
        size = os.path.getsize(dest)
    except OSError as e:
        logger.warning("[url-dl] could not retain downloaded audio: %s", e)
        return None
    _REG[media_id] = {
        "path": dest, "ext": ext, "user_id": user_id,
        "created": time.monotonic(), "size": size,
    }
    _evict_over_cap()
    return media_id


def make_pipeline_copy(src: str) -> "str | None":
    """A tempdir copy of `src` for the transcription pipeline to own and
    unlink (hardlink when the tempdir shares a filesystem, else a real
    copy). None when the copy fails. Called BEFORE register() moves the
    original away, so the two files' lifecycles stay independent."""
    ext = os.path.splitext(src)[1].lstrip(".").lower() or "bin"
    fd, tmp = tempfile.mkstemp(suffix=f".{ext}" if ext.isalnum() and len(ext) <= 8 else "")
    os.close(fd)
    try:
        os.unlink(tmp)  # replaced by link/copy below
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
    except OSError as e:
        logger.error("[url-dl] pipeline copy failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return tmp


def resolve(media_id: str, *, user_id: "str | None") -> "tuple[str, str] | None":
    """(abs_path, ext) when `media_id` exists, is still fresh, and belongs to
    `user_id` (None owner or None caller ⇒ open-mode, allow). None otherwise
    — the route maps every miss to one 404, no oracle."""
    entry = _REG.get(media_id)
    if entry is None:
        return None
    ttl = float(getattr(cfg, "URL_MEDIA_TTL_SEC", 3600))
    if time.monotonic() - entry["created"] > ttl:
        _drop(media_id)
        return None
    owner = entry.get("user_id")
    if owner is not None and user_id is not None and owner != user_id:
        return None
    path = entry["path"]
    if not os.path.isfile(path):
        _REG.pop(media_id, None)
        return None
    return path, entry["ext"]


def expires_at_unix(media_id: str) -> "int | None":
    """Wall-clock expiry hint for the response payload (advisory only — the
    registry works on the monotonic clock)."""
    entry = _REG.get(media_id)
    if entry is None:
        return None
    ttl = float(getattr(cfg, "URL_MEDIA_TTL_SEC", 3600))
    remaining = ttl - (time.monotonic() - entry["created"])
    return int(time.time() + max(0.0, remaining))


def sweep() -> None:
    """TTL expiry + LRU byte cap. Called by the lifespan janitor task."""
    ttl = float(getattr(cfg, "URL_MEDIA_TTL_SEC", 3600))
    now = time.monotonic()
    for mid in [m for m, e in list(_REG.items()) if now - e["created"] > ttl]:
        _drop(mid)
    _evict_over_cap()


def _evict_over_cap() -> None:
    cap = int(getattr(cfg, "URL_MEDIA_MAX_BYTES", 2_000_000_000) or 0)
    if cap <= 0:
        return
    total = sum(e["size"] for e in _REG.values())
    if total <= cap:
        return
    for mid, _e in sorted(_REG.items(), key=lambda kv: kv[1]["created"]):
        if total <= cap:
            break
        total -= _e["size"]
        _drop(mid)


def _drop(media_id: str) -> None:
    entry = _REG.pop(media_id, None)
    if entry is None:
        return
    try:
        os.unlink(entry["path"])
    except OSError:
        pass


async def janitor_loop(interval_s: float = 60.0) -> None:
    """Lifespan task: periodic sweep, same shape as the model idle evictors."""
    import asyncio

    while True:
        await asyncio.sleep(interval_s)
        try:
            sweep()
        except Exception as e:  # noqa: BLE001 — the janitor must never die
            logger.error("[url-dl] retention sweep failed: %s", e)
