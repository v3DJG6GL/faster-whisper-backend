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
  - Bounded twice: per-entry TTL (URL_MEDIA_TTL_SEC) and a byte cap over the
    whole dir (URL_MEDIA_MAX_BYTES, oldest download evicted first). sweep()
    runs from a lifespan task; register() also evicts inline so a burst
    can't overshoot until the next tick.
"""
from __future__ import annotations

import logging
import os
import re
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
        # The dir can be first created HERE (runtime enable of the URL
        # feature, no lifespan startup_reset) — keep it 0700 either way.
        secure_dir(_dir())
        shutil.move(src_path, dest)
        # Both move paths preserve the SOURCE mtime (rename, or copy2's
        # copystat), but sweep()'s orphan-age guard measures placement
        # time — refresh it so a just-placed file can never look old.
        os.utime(dest, None)
        secure_file(dest)
        size = os.path.getsize(dest)
    except OSError as e:
        logger.warning("[url-dl] could not retain downloaded audio: %s", e)
        return None
    _REG[media_id] = {
        "path": dest, "ext": ext, "user_id": user_id,
        "created": time.monotonic(), "size": size,
    }
    _evict_over_cap(protect=media_id)
    # A file that alone busts the byte cap is dropped straight away — the
    # caller must not advertise a media_id that would 404 immediately.
    return media_id if media_id in _REG else None


def make_pipeline_copy(src: str) -> "str | None":
    """A tempdir copy of `src` for the transcription pipeline to own and
    unlink (hardlink when the tempdir shares a filesystem, else a real
    copy). None when the copy fails. Called BEFORE register() moves the
    original away, so the two files' lifecycles stay independent."""
    ext = os.path.splitext(src)[1].lstrip(".").lower() or "bin"
    suffix = f".{ext}" if ext.isalnum() and len(ext) <= 8 else ""
    # A fresh unpredictable name that is NEVER pre-created then re-created
    # (an unlink/re-create window on a shared TMPDIR invites a planted
    # symlink): os.link refuses an existing path, and the copy fallback
    # opens with O_EXCL|O_NOFOLLOW at mode 0600.
    tmp = os.path.join(tempfile.gettempdir(), f"urlmedia-{uuid.uuid4().hex}{suffix}")
    try:
        try:
            os.link(src, tmp)
        except OSError:
            fd = os.open(
                tmp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                0o600)
            with open(src, "rb") as sf, os.fdopen(fd, "wb") as df:
                shutil.copyfileobj(sf, df)
    except OSError as e:
        logger.error("[url-dl] pipeline copy failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    # The os.link fast path inherits the download's umask mode (typically
    # 0644) in the shared tempdir — tighten it like the fallback already is.
    secure_file(tmp)
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


# Name shape register() creates: {32-hex uuid}.{ext}. Anything else in the
# dir was not put there by this store and is left alone.
_RETAINED_NAME_RE = re.compile(r"\A[0-9a-f]{32}\Z")

# An on-disk file must be at least this old before the orphan scan trusts
# "no registry row" — register() moves the file into place from a worker
# thread a moment before it inserts the row.
_ORPHAN_MIN_AGE_SEC = 60.0


def sweep() -> None:
    """TTL expiry + oldest-first (FIFO) byte cap + orphan scan. Called by
    the lifespan janitor task."""
    ttl = float(getattr(cfg, "URL_MEDIA_TTL_SEC", 3600))
    now = time.monotonic()
    for mid in [m for m, e in list(_REG.items()) if now - e["created"] > ttl]:
        _drop(mid)
    _evict_over_cap()
    # Orphan scan: a failed unlink in _drop (Windows keeps a file locked
    # while a FileResponse streams it) leaves a registry-less file that the
    # byte cap can't see. Retry those here until they go.
    live = {os.path.basename(e["path"]) for e in list(_REG.values())}
    d = _dir()
    try:
        names = os.listdir(d)
    except OSError:
        return
    wall = time.time()
    for name in names:
        stem, _dot, _ext = name.partition(".")
        if not _RETAINED_NAME_RE.match(stem) or name in live:
            continue
        path = os.path.join(d, name)
        try:
            if wall - os.path.getmtime(path) < _ORPHAN_MIN_AGE_SEC:
                continue
            os.unlink(path)
        except OSError:
            pass


def _evict_over_cap(protect: "str | None" = None) -> None:
    cap = int(getattr(cfg, "URL_MEDIA_MAX_BYTES", 2_000_000_000) or 0)
    if cap <= 0:
        return
    if protect is not None:
        entry = _REG.get(protect)
        if entry is not None and entry["size"] > cap:
            # The new file alone busts the cap: drop IT rather than first
            # wiping every older retained file to no avail.
            _drop(protect)
            protect = None
    # Snapshot: register() runs on worker threads and the janitor on the
    # loop thread, so a concurrent insert must not trip "dict changed size
    # during iteration" here.
    items = list(_REG.items())
    total = sum(e["size"] for _m, e in items)
    if total <= cap:
        return
    for mid, _e in sorted(items, key=lambda kv: kv[1]["created"]):
        if total <= cap:
            break
        if mid == protect or mid not in _REG:
            continue
        total -= _e["size"]
        _drop(mid)


def _drop(media_id: str) -> None:
    entry = _REG.pop(media_id, None)
    if entry is None:
        return
    try:
        os.unlink(entry["path"])
    except OSError as e:
        # Not lost for good: the file matches the retained-name shape, so
        # sweep()'s orphan scan retries it once it is no longer in use.
        logger.debug("[url-dl] retained-file unlink failed (%s): %s",
                     entry["path"], e)


async def janitor_loop(interval_s: float = 60.0) -> None:
    """Lifespan task: periodic sweep, same shape as the model idle evictors."""
    import asyncio

    while True:
        await asyncio.sleep(interval_s)
        try:
            sweep()
        except Exception as e:  # noqa: BLE001 — the janitor must never die
            logger.error("[url-dl] retention sweep failed: %s", e)
