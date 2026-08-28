"""Transcribe-from-URL support: fetch a client-supplied media link with yt-dlp.

Contract (mirrors bgm_separation.py):
  - str(UrlDownloadError) is CLIENT-SAFE — our own wording, never raw yt-dlp
    stderr (which can carry filesystem paths and full URLs with tokens).
  - UrlCancelled means the caller's cancel_check tripped; the request must
    abort (HTTP 499 via main's _ClientCancelled), not soft-fail.

Design notes:
  - Policy runs BEFORE any network I/O: the extractor that matches a URL is
    determined offline (regex match over yt-dlp's extractor registry), so a
    disallowed URL is rejected without the server ever touching it. The only
    pre-download fetches are the metadata probe (after policy) and the capped
    direct-media probe, which resolves the host first and refuses private /
    loopback / link-local ranges (SSRF).
  - The metadata probe uses the yt-dlp *Python API* (short-lived, we want the
    info dict); the actual download runs the yt-dlp *CLI in a subprocess* —
    crash isolation from ~2000 third-party extractors, trivial cancellation
    (terminate), and a real wall-clock timeout.
  - No user-controlled value ever becomes a flag: the URL is the only
    client-supplied argv element and always follows a literal "--".
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import ipaddress
import logging
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request

import config as cfg
from store_common import log_safe

logger = logging.getLogger("whisper-api")

_URL_MAX_LEN = 2048
# Anything a shell/log/terminal could misread. URLs have no business carrying
# raw whitespace or C0/C1 bytes (they'd be %-encoded in a real URL).
_URL_UNSAFE_RE = re.compile(r"[\s\x00-\x1f\x7f-\x9f]")

# Content types accepted by the direct-media probe. application/ogg is the
# registered type for .ogg/.opus; everything else must declare audio/* or
# video/*.
_DIRECT_MEDIA_TYPES = ("audio/", "video/", "application/ogg")


class UrlDownloadError(RuntimeError):
    """str() is CLIENT-SAFE by contract (our wording, never tool output)."""


class UrlCancelled(Exception):
    """Cooperative cancel: the caller's cancel_check returned True."""


@dataclasses.dataclass
class UrlMediaInfo:
    url: str
    extractor_key: str
    title: "str | None" = None
    duration: "float | None" = None  # seconds
    uploader: "str | None" = None
    filesize_approx: "int | None" = None
    is_live: bool = False
    thumbnail_url: "str | None" = None


def yt_dlp_version() -> "str | None":
    """Installed yt-dlp version, or None when the package is absent."""
    global _YTDLP_VERSION
    if _YTDLP_VERSION is _UNSET:
        try:
            import importlib.metadata
            _YTDLP_VERSION = importlib.metadata.version("yt-dlp")
        except Exception:  # noqa: BLE001 — absence is a supported state
            _YTDLP_VERSION = None
    return _YTDLP_VERSION


_UNSET = object()
_YTDLP_VERSION: "str | None | object" = _UNSET


def validate_url(url: str) -> str:
    """Normalise + gate a client-supplied URL. Raises UrlDownloadError."""
    u = (url or "").strip()
    if not u:
        raise UrlDownloadError("no URL was provided")
    if len(u) > _URL_MAX_LEN:
        raise UrlDownloadError("the URL is too long")
    if _URL_UNSAFE_RE.search(u):
        raise UrlDownloadError("the URL contains invalid characters")
    try:
        parts = urllib.parse.urlsplit(u)
    except ValueError:
        raise UrlDownloadError("the URL could not be parsed") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise UrlDownloadError("only http(s) URLs are supported")
    if not parts.hostname:
        raise UrlDownloadError("the URL has no host")
    return u


def _effective_max_bytes() -> int:
    n = int(getattr(cfg, "URL_MAX_BYTES", 0) or 0)
    if n <= 0:
        n = int(getattr(cfg, "MAX_UPLOAD_BYTES", 200_000_000))
    return n


def match_extractor(url: str) -> str:
    """The yt-dlp extractor key that would handle `url` — decided OFFLINE
    (pure regex match, no network), so policy can run before any fetch.
    "Generic" means no dedicated extractor claims the URL."""
    import yt_dlp.extractor  # lazy: optional dependency

    for ie in yt_dlp.extractor.gen_extractor_classes():
        key = ie.ie_key()
        if key == "Generic":
            continue
        try:
            if ie.suitable(url) and ie.working():
                return key
        except Exception:  # noqa: BLE001 — one broken pattern must not veto
            continue
    return "Generic"


def _host_is_forbidden(host: str) -> bool:
    """True when `host` resolves ONLY to addresses we refuse to fetch from:
    loopback, RFC1918, link-local (cloud metadata), CGNAT, ULA, reserved.
    Resolution failure counts as forbidden (we can't vouch for it)."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return True
    if not infos:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            return True
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return True
        if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NET:
            return True
    return False


_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


class _NoPrivateRedirects(urllib.request.HTTPRedirectHandler):
    """Re-run the scheme + private-address gate on every redirect hop, so a
    public URL can't 302 into the LAN or the cloud metadata service."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme.lower() not in ("http", "https"):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to a non-http URL", headers, fp)
        if not parts.hostname or _host_is_forbidden(parts.hostname):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to a forbidden address", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _direct_media_probe_sync(url: str, *, timeout: float) -> bool:
    """Capped GET (first byte only) that answers: does this URL serve
    audio/video directly? Host gate + redirect gate keep it off internal
    ranges. Never raises for 'no' — only returns False."""
    parts = urllib.parse.urlsplit(url)
    if not parts.hostname or _host_is_forbidden(parts.hostname):
        return False
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "faster-whisper-backend"},
        method="GET")
    opener = urllib.request.build_opener(_NoPrivateRedirects())
    try:
        with opener.open(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            resp.read(1)  # some servers ignore Range; never read more
    except Exception:  # noqa: BLE001 — unreachable/odd server ⇒ not direct media
        return False
    return ctype.startswith(_DIRECT_MEDIA_TYPES)


async def check_url_policy(url: str) -> str:
    """Enforce the operator's site policy for `url` BEFORE any yt-dlp fetch.
    Returns the matched extractor key. Raises UrlDownloadError on reject."""
    key = await asyncio.to_thread(match_extractor, url)
    if key == "Generic":
        if getattr(cfg, "URL_ALLOW_GENERIC", False):
            return key
        if getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", True):
            ok = await asyncio.to_thread(
                _direct_media_probe_sync, url,
                timeout=float(getattr(cfg, "URL_SOCKET_TIMEOUT_SEC", 15)))
            if ok:
                return key
            raise UrlDownloadError(
                "this link is neither a supported site nor a direct "
                "audio/video file")
        raise UrlDownloadError(
            "this site isn't allowed by the server's URL policy")
    allowed = [a.strip().lower()
               for a in (getattr(cfg, "URL_ALLOWED_EXTRACTORS", []) or [])
               if a and a.strip()]
    if allowed and key.lower() not in allowed:
        raise UrlDownloadError("this site isn't on the server's allowed list")
    return key


def _policy_check_info(info: dict) -> None:
    """Post-metadata policy: things only the info dict can tell us."""
    if info.get("_type") in ("playlist", "multi_video"):
        raise UrlDownloadError(
            "playlists aren't supported — link a single video or track")
    if info.get("is_live") or info.get("live_status") == "is_live":
        raise UrlDownloadError(
            "live streams aren't supported — try again after the stream ends")
    max_dur = int(getattr(cfg, "URL_MAX_DURATION_SEC", 14400))
    dur = info.get("duration")
    if dur is not None and float(dur) > max_dur:
        raise UrlDownloadError(
            f"this media runs {float(dur) / 3600:.1f} h — over the server's "
            f"{max_dur / 3600:.1f} h limit for link downloads")
    approx = info.get("filesize_approx") or info.get("filesize")
    if approx is not None and int(approx) > _effective_max_bytes():
        raise UrlDownloadError("this media exceeds the server's size limit")


async def probe(url: str, *, timeout: float) -> UrlMediaInfo:
    """Policy-gated metadata probe (no download). Client-safe errors only."""
    url = validate_url(url)
    key = await check_url_policy(url)

    def _extract() -> dict:
        import yt_dlp  # lazy: optional dependency

        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": float(getattr(cfg, "URL_SOCKET_TIMEOUT_SEC", 15)),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.sanitize_info(ydl.extract_info(url, download=False))

    try:
        info = await asyncio.wait_for(asyncio.to_thread(_extract), timeout)
    except asyncio.TimeoutError:
        raise UrlDownloadError("the site took too long to answer") from None
    except UrlDownloadError:
        raise
    except Exception as e:  # noqa: BLE001 — classify, never forward raw
        _log_probe_failure(url, e)
        raise UrlDownloadError(classify_error(str(e))) from None
    if not isinstance(info, dict):
        raise UrlDownloadError("the site returned no usable media info")
    _policy_check_info(info)
    return UrlMediaInfo(
        url=url,
        extractor_key=str(info.get("extractor_key") or key),
        title=info.get("title"),
        duration=(float(info["duration"]) if info.get("duration") is not None
                  else None),
        uploader=info.get("uploader") or info.get("channel"),
        filesize_approx=(int(info["filesize_approx"])
                         if info.get("filesize_approx") else None),
        is_live=bool(info.get("is_live")),
        thumbnail_url=info.get("thumbnail"),
    )


def _log_probe_failure(url: str, e: Exception) -> None:
    logger.warning("[url-dl] probe failed for host %s: %s",
                   log_safe(urllib.parse.urlsplit(url).hostname or "?"),
                   log_safe(str(e)))


async def fetch_thumbnail_data_uri(
    url: "str | None", *, max_bytes: int = 512_000, timeout: float = 5.0,
) -> "str | None":
    """Fetch a thumbnail server-side and return it as a data: URI (the client
    CSP forbids remote images, and fetching client-side would leak the
    client's IP to the media site). Soft-fails to None — a preview without a
    thumbnail is still a preview."""
    if not url:
        return None
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return None

    def _fetch() -> "str | None":
        if _host_is_forbidden(parts.hostname or ""):
            return None
        req = urllib.request.Request(
            url, headers={"User-Agent": "faster-whisper-backend"})
        opener = urllib.request.build_opener(_NoPrivateRedirects())
        try:
            with opener.open(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if not ctype.startswith("image/") or "svg" in ctype:
                    return None
                data = resp.read(max_bytes + 1)
        except Exception:  # noqa: BLE001 — soft-fail by contract
            return None
        if not data or len(data) > max_bytes:
            return None
        return f"data:{ctype};base64,{base64.b64encode(data).decode('ascii')}"

    try:
        return await asyncio.wait_for(asyncio.to_thread(_fetch), timeout + 2.0)
    except Exception:  # noqa: BLE001
        return None


# ── error taxonomy ──────────────────────────────────────────────────────────
# (substring-of-tool-output, client-safe message). Order matters: first hit
# wins, and the more specific conditions sit above the catch-alls.
_ERROR_TAXONOMY: "tuple[tuple[tuple[str, ...], str], ...]" = (
    # Age before bot: both messages start "Sign in to confirm …", so the
    # broader sign-in needle must not shadow the age variant.
    (("confirm your age", "age-restricted", "age restricted"),
     "this media is age-restricted and needs a signed-in account, which "
     "this server doesn't have"),
    (("confirm you're not a bot", "confirm you’re not a bot",
      "sign in to confirm"),
     "the site is asking this server to verify it isn't a bot — this mostly "
     "hits data-center IPs; running the backend on a residential connection "
     "usually avoids it"),
    (("private video", "this video is private"),
     "this video is private"),
    (("members-only", "join this channel", "channel members"),
     "this is members-only content"),
    (("not available in your country", "geo restricted", "geo-restricted",
      "not made this video available in your country"),
     "this media isn't available in the server's region"),
    (("video unavailable", "has been removed", "no longer available",
      "account associated with this video has been terminated"),
     "this media is unavailable or has been removed"),
    (("unsupported url",),
     "this site isn't supported by the downloader"),
    (("live event", "premieres in", "this live event"),
     "this stream hasn't finished yet — try again after it ends"),
    (("max-filesize", "file is larger than max-filesize"),
     "this media exceeds the server's size limit"),
    (("unable to download webpage", "failed to resolve", "getaddrinfo",
      "timed out", "connection refused"),
     "the site could not be reached from the server"),
)


def classify_error(stderr_tail: str) -> str:
    """Collapse yt-dlp output into a client-safe message. The raw text is the
    caller's to log (via log_safe); it never reaches a client."""
    low = (stderr_tail or "").lower()
    for needles, message in _ERROR_TAXONOMY:
        if any(n in low for n in needles):
            return message
    ver = yt_dlp_version() or "not installed"
    return (f"could not download media from this link (yt-dlp {ver} — "
            f"if the site recently changed, updating the server's yt-dlp "
            f"usually fixes this)")


# ── download ────────────────────────────────────────────────────────────────

# NOTE: in --progress-template, "download:" is the PHASE SELECTOR (consumed
# by yt-dlp, never printed) — the emitted line starts with our own "dl:"
# marker so progress lines are unambiguous against yt-dlp's [info] chatter.
_PROGRESS_PREFIX = "dl:"
_STDERR_TAIL_MAX = 4096


def _parse_progress_line(line: str) -> "tuple[int | None, int | None] | None":
    """Parse one --progress-template line into (downloaded, total). Total
    falls back to the estimate; unknown fields arrive as 'NA' (yt-dlp quirk:
    never empty strings)."""
    if not line.startswith(_PROGRESS_PREFIX):
        return None
    fields = line[len(_PROGRESS_PREFIX):].split()
    if len(fields) < 3:
        return None

    def _num(s: str) -> "int | None":
        try:
            return int(float(s))
        except ValueError:
            return None

    downloaded = _num(fields[0])
    total = _num(fields[1])
    if total is None:
        total = _num(fields[2])
    if downloaded is None:
        return None
    return downloaded, total


def build_download_argv(url: str, *, dest_dir: str, max_bytes: int) -> "list[str]":
    """The exact yt-dlp CLI invocation (separate function so tests can pin
    it). The URL is the only client-supplied element and follows '--'."""
    from streaming_transport import ffmpeg_exe

    return [
        sys.executable, "-m", "yt_dlp",
        # Whisper resamples to 16 kHz mono regardless, so "best audio" is
        # about container sanity, not fidelity: prefer m4a (PyAV-friendly),
        # fall back to any bestaudio, then best (video container with audio).
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "--no-playlist",
        "--playlist-items", "1",  # belt+braces: never more than one item
        "--restrict-filenames",
        "--max-filesize", str(max_bytes),
        "--socket-timeout", str(int(getattr(cfg, "URL_SOCKET_TIMEOUT_SEC", 15))),
        "--retries", "3",
        "--no-mtime",
        "--ffmpeg-location", ffmpeg_exe(),
        "-P", dest_dir,
        # NEVER %(title)s — titles are attacker-controlled and path-adjacent.
        "-o", "media.%(ext)s",
        "--newline", "--no-colors",
        "--progress-template",
        ("download:dl:%(progress.downloaded_bytes)s "
         "%(progress.total_bytes)s %(progress.total_bytes_estimate)s"),
        "--", url,
    ]


async def download(
    url: str,
    *,
    dest_dir: str,
    max_bytes: "int | None" = None,
    timeout: "float | None" = None,
    progress_cb=None,
    cancel_check=None,
) -> str:
    """Download the audio for `url` into `dest_dir` (a private, per-job
    directory owned by the caller) and return the resulting file path.

    progress_cb(fraction_or_None, total_bytes_or_None) is throttled to one
    call per 0.3 s; cancel_check() is polled continuously and a True answer
    terminates the subprocess and raises UrlCancelled. The whole download is
    bounded by `timeout` wall-clock seconds."""
    url = validate_url(url)
    max_bytes = int(max_bytes or _effective_max_bytes())
    timeout = float(timeout or getattr(cfg, "URL_DOWNLOAD_TIMEOUT_SEC", 900))
    argv = build_download_argv(url, dest_dir=dest_dir, max_bytes=max_bytes)

    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_tail = bytearray()

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        while True:
            chunk = await proc.stderr.read(1024)
            if not chunk:
                return
            stderr_tail.extend(chunk)
            if len(stderr_tail) > _STDERR_TAIL_MAX:
                del stderr_tail[:len(stderr_tail) - _STDERR_TAIL_MAX]

    async def _kill(grace: float = 5.0) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), grace)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    stderr_task = asyncio.create_task(_drain_stderr())
    last_cb = 0.0
    try:
        assert proc.stdout is not None
        while True:
            if time.monotonic() - t0 > timeout:
                await _kill()
                raise UrlDownloadError("the download timed out")
            if cancel_check is not None and cancel_check():
                await _kill()
                raise UrlCancelled()
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), 0.5)
            except asyncio.TimeoutError:
                continue
            if not raw:
                break
            parsed = _parse_progress_line(raw.decode("utf-8", "replace").strip())
            if parsed and progress_cb is not None:
                now = time.monotonic()
                if now - last_cb >= 0.3:
                    last_cb = now
                    downloaded, total = parsed
                    frac = (max(0.0, min(1.0, downloaded / total))
                            if total else None)
                    try:
                        progress_cb(frac, total)
                    except Exception:  # noqa: BLE001 — progress must not kill the run
                        pass
        await asyncio.wait_for(proc.wait(), max(5.0, timeout - (time.monotonic() - t0)))
    except asyncio.TimeoutError:
        await _kill()
        raise UrlDownloadError("the download timed out") from None
    finally:
        try:
            await asyncio.wait_for(stderr_task, 5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            stderr_task.cancel()

    if proc.returncode != 0:
        tail = stderr_tail.decode("utf-8", "replace")
        logger.warning("[url-dl] yt-dlp exited %s for host %s: %s",
                       proc.returncode,
                       log_safe(urllib.parse.urlsplit(url).hostname or "?"),
                       log_safe(tail[-300:]))
        raise UrlDownloadError(classify_error(tail))

    result = _find_result_file(dest_dir)
    if result is None:
        # --max-filesize skips (exit 0, no file) on some formats instead of
        # failing — a missing output after a clean exit means the cap bit.
        raise UrlDownloadError("this media exceeds the server's size limit")
    size = os.path.getsize(result)
    if size > max_bytes:
        raise UrlDownloadError("this media exceeds the server's size limit")
    if size == 0:
        raise UrlDownloadError("the downloaded file was empty")
    logger.info("[url-dl] downloaded %.1f MB in %.1fs (host %s)",
                size / 1e6, time.monotonic() - t0,
                log_safe(urllib.parse.urlsplit(url).hostname or "?"))
    return result


def _find_result_file(dest_dir: str) -> "str | None":
    """The completed download inside `dest_dir`, or None. Refuses partials
    and anything that escapes the directory (symlink games)."""
    root = os.path.realpath(dest_dir)
    best: "tuple[float, str] | None" = None
    try:
        names = os.listdir(dest_dir)
    except OSError:
        return None
    for name in names:
        if name.endswith((".part", ".ytdl", ".tmp")):
            continue
        if not name.startswith("media."):
            continue
        path = os.path.join(dest_dir, name)
        real = os.path.realpath(path)
        if not (real == root or real.startswith(root + os.sep)):
            continue
        if not os.path.isfile(real) or os.path.islink(path):
            continue
        mtime = os.path.getmtime(real)
        if best is None or mtime > best[0]:
            best = (mtime, real)
    return best[1] if best else None
