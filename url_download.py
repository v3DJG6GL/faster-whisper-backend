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
  - yt-dlp fetches with its OWN opener, which used to follow redirects and
    re-resolve DNS with no policy: a public link could 302 the downloader
    into the LAN or the cloud metadata service behind the probe's back. Both
    paths now install the guard in ytdlp_plugins/ (see guard_self_check) —
    same address policy (net_policy), applied to every hop, with the resolved
    IP pinned — and REFUSE to run if it cannot be installed.
  - No user-controlled value ever becomes a flag: the URL is the only
    client-supplied argv element and always follows a literal "--".
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import importlib.util
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import config as cfg
import net_policy
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


class UrlPolicyError(UrlDownloadError):
    """The operator's URL / size policy refused the input (not a fetch
    failure). Pre-classified for the failures card."""
    error_class = "policy_blocked"


class UrlTimeoutError(UrlDownloadError):
    """A probe or download ran past its timeout."""
    error_class = "timeout"


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
    # Of the format DOWNLOAD_FORMAT actually selects (audio), not the page's
    # default merged video: container ext and audio bitrate in kbps.
    ext: "str | None" = None
    abr: "float | None" = None


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

# The one format selector, shared by probe and download. Whisper resamples to
# 16 kHz mono regardless, so "best audio" is about container sanity, not
# fidelity: prefer m4a (PyAV-friendly), fall back to any bestaudio, then best
# (video container with audio). The probe MUST use the same selector —
# otherwise extract_info resolves yt-dlp's default (merged video+audio) and
# filesize_approx reflects the full VIDEO, tripping the size policy for media
# whose audio track is well within the cap.
DOWNLOAD_FORMAT = "bestaudio[ext=m4a]/bestaudio/best"


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


# "Which addresses do we refuse to fetch from" has exactly ONE definition,
# in net_policy — because the yt-dlp guard (ytdlp_plugins/) enforces the same
# rule from a separate process and must not carry a second copy of the list.
# Bound as a module global on purpose: the redirect handler and both probes
# look it up here, and tests monkeypatch it here.
_host_is_forbidden = net_policy.host_is_forbidden
_CGNAT_NET = net_policy.CGNAT_NET


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
    # `timeout` is a wall-clock budget: the opener's timeout is per-socket-
    # op, so a host dribbling one header byte per op could otherwise hold
    # this worker thread far past it (the outer wait_for abandons the
    # await, never the thread). Short per-op timeout + a monotonic deadline.
    deadline = time.monotonic() + timeout
    op_timeout = max(1.0, min(timeout, 5.0))
    parts = urllib.parse.urlsplit(url)
    if not parts.hostname or _host_is_forbidden(parts.hostname):
        return False
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "faster-whisper-backend"},
        method="GET")
    opener = urllib.request.build_opener(_NoPrivateRedirects())
    try:
        with opener.open(req, timeout=op_timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if time.monotonic() > deadline:
                return False
            resp.read(1)  # some servers ignore Range; never read more
    except Exception:  # noqa: BLE001 — unreachable/odd server ⇒ not direct media
        return False
    return ctype.startswith(_DIRECT_MEDIA_TYPES)


# The policy probes (extractor match, DNS in _host_is_forbidden, the capped
# direct-media GET) run on their own small pool: a wedged probe thread must
# only cost probe capacity, never the app-wide default executor that every
# other asyncio.to_thread in the process shares.
_PROBE_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="url-probe")


# ── the yt-dlp SSRF guard ───────────────────────────────────────────────────
# The probes above gate every hop THEY make, but the two fetches that actually
# move bytes — probe()'s extract_info and download()'s subprocess — go through
# yt-dlp's own opener, which follows redirects and re-resolves DNS with no
# policy at all. ytdlp_plugins/ ships a RequestHandler that re-applies
# net_policy to hop 0 and to every redirect hop, pins the resolved IP, and
# speaks only http(s); it also unregisters yt-dlp's built-in handlers so
# nothing can fall through to an unguarded opener. It is installed here for
# the in-process probe and by ytdlp_plugins/run_guarded_yt_dlp.py for the
# download subprocess.
GUARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ytdlp_plugins")
GUARD_LAUNCHER = os.path.join(GUARD_DIR, "run_guarded_yt_dlp.py")
GUARD_MODULE = os.path.join(GUARD_DIR, "fwb_ssrf_guard", "yt_dlp_plugins",
                            "extractor", "fwb_ssrf_guard.py")
# The needle every guard refusal carries; classify_error maps it to a
# client-safe message. Must equal fwb_ssrf_guard.MARKER (pinned by a test).
GUARD_MARKER = "fwb-ssrf-guard"

_guard_ok = False
_guard_announced = False


def guard_self_check(*, force: bool = False) -> None:
    """Install the yt-dlp SSRF guard in THIS process and prove that it took.

    Fail CLOSED: every caller that is about to let yt-dlp touch the network
    goes through here first, so a yt-dlp refactor that breaks the handler
    (it subclasses yt_dlp.networking._urllib.UrllibRH — private by name)
    stops link downloads instead of silently running them unguarded.

    Called once from the app lifespan so a broken guard is an operator-visible
    startup line rather than a surprise on the first pasted link, and again —
    cached — from probe() and download(). Raises UrlDownloadError, which is
    CLIENT-SAFE by contract: the yt-dlp version and the real cause go to the
    log, never to the caller."""
    global _guard_ok, _guard_announced
    if _guard_ok and not force:
        return
    try:
        for path in (GUARD_LAUNCHER, GUARD_MODULE):
            if not os.path.isfile(path):
                raise RuntimeError(f"guard file missing: {path}")
        spec = importlib.util.spec_from_file_location(
            "fwb_ssrf_guard_inproc", GUARD_MODULE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {GUARD_MODULE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["fwb_ssrf_guard_inproc"] = module
        spec.loader.exec_module(module)  # registers the handler on import
        if module.MARKER != GUARD_MARKER:
            raise RuntimeError("guard marker mismatch — classify_error would "
                               "no longer recognise a refusal")
        if not module.is_installed():
            raise RuntimeError("the guard handler did not register")
    except Exception as e:  # noqa: BLE001 — ANY failure must fail closed
        _guard_ok = False
        logger.error(
            "[url-dl] REFUSING link downloads: the yt-dlp SSRF guard could "
            "not be installed for yt-dlp %s (%s: %s). Without it yt-dlp would "
            "follow redirects into private addresses unchecked. Check %s.",
            yt_dlp_version() or "not installed", type(e).__name__,
            log_safe(str(e)), GUARD_MODULE)
        raise UrlDownloadError(
            "link downloads are unavailable on this server") from None
    _guard_ok = True
    if not _guard_announced:
        _guard_announced = True
        logger.info("[url-dl] SSRF guard active for yt-dlp %s (%s)",
                    yt_dlp_version() or "?", module.RH_NAME)


async def check_url_policy(url: str) -> str:
    """Enforce the operator's site policy for `url` BEFORE any yt-dlp fetch.
    Returns the matched extractor key. Raises UrlDownloadError on reject."""
    loop = asyncio.get_running_loop()
    key = await loop.run_in_executor(_PROBE_POOL, match_extractor, url)
    if key == "Generic":
        if getattr(cfg, "URL_ALLOW_GENERIC", False):
            return key
        if getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", True):
            probe_timeout = float(getattr(cfg, "URL_SOCKET_TIMEOUT_SEC", 15))
            ok = await loop.run_in_executor(
                _PROBE_POOL,
                lambda: _direct_media_probe_sync(url, timeout=probe_timeout))
            if ok:
                return key
            raise UrlDownloadError(
                "this link is neither a supported site nor a direct "
                "audio/video file")
        raise UrlPolicyError(
            "this site isn't allowed by the server's URL policy")
    allowed = [a.strip().lower()
               for a in (getattr(cfg, "URL_ALLOWED_EXTRACTORS", []) or [])
               if a and a.strip()]
    if allowed and key.lower() not in allowed:
        raise UrlPolicyError("this site isn't on the server's allowed list")
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
    """Policy-gated metadata probe (no download). Client-safe errors only.
    `timeout` is one wall-clock budget for the WHOLE probe — the policy
    check (which can do DNS + a capped direct-media GET) spends from the
    same deadline as the metadata extraction."""
    url = validate_url(url)
    deadline = time.monotonic() + timeout
    try:
        key = await asyncio.wait_for(check_url_policy(url), timeout)
    except asyncio.TimeoutError:
        raise UrlDownloadError("the site took too long to answer") from None

    # Fail closed BEFORE extract_info: the guard registers the RequestHandler
    # this process's YoutubeDL will pick, so it has to be in place (and
    # verified) before the instance is built. After the policy check, so a
    # rejected URL still gets its own message.
    guard_self_check()

    def _extract() -> dict:
        import yt_dlp  # lazy: optional dependency

        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            # Same selector as the download: filesize_approx must describe
            # what we'd actually fetch (audio), not the default merged video.
            "format": DOWNLOAD_FORMAT,
            # A channel page / playlist URL is a playlist to yt-dlp, and
            # without this it fully resolves EVERY entry (one round-trip
            # each) — a channel's /videos tab then times the probe out
            # before _policy_check_info can say "playlists aren't
            # supported". Flat entries keep the top-level _type intact and
            # resolve in one fetch; single videos are unaffected.
            "extract_flat": "in_playlist",
            "socket_timeout": float(getattr(cfg, "URL_SOCKET_TIMEOUT_SEC", 15)),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.sanitize_info(ydl.extract_info(url, download=False))

    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(_extract),
            max(1.0, deadline - time.monotonic()))
    except asyncio.TimeoutError:
        raise UrlDownloadError("the site took too long to answer") from None
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
        filesize_approx=(int(info.get("filesize_approx")
                             or info.get("filesize"))
                         if (info.get("filesize_approx")
                             or info.get("filesize")) else None),
        is_live=bool(info.get("is_live")),
        thumbnail_url=info.get("thumbnail"),
        ext=(str(info["ext"]) if info.get("ext") else None),
        abr=(float(info["abr"]) if info.get("abr") else None),
    )


def host_for_log(url: "str | None") -> str:
    """Hostname for log lines — never the full URL (it can carry tokens)."""
    try:
        return log_safe(urllib.parse.urlsplit((url or "").strip()).hostname or "?")
    except Exception:  # noqa: BLE001 — logging must never raise
        return "?"


def _log_probe_failure(url: str, e: Exception) -> None:
    logger.warning("[url-dl] probe failed for host %s: %s",
                   host_for_log(url), log_safe(str(e)))


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
                # Chunked against a monotonic deadline: `timeout` on the
                # opener is per-socket-op, so a host dribbling bytes under
                # it could otherwise hold this worker thread forever (the
                # outer wait_for abandons the await, never the thread).
                t0 = time.monotonic()
                buf = bytearray()
                while True:
                    chunk = resp.read(32768)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        return None
                    if time.monotonic() - t0 > timeout:
                        return None
        except Exception:  # noqa: BLE001 — soft-fail by contract
            return None
        if not buf:
            return None
        return f"data:{ctype};base64,{base64.b64encode(bytes(buf)).decode('ascii')}"

    try:
        return await asyncio.wait_for(asyncio.to_thread(_fetch), timeout + 2.0)
    except Exception:  # noqa: BLE001
        return None


# ── error taxonomy ──────────────────────────────────────────────────────────
# (substring-of-tool-output, client-safe message). Order matters: first hit
# wins, and the more specific conditions sit above the catch-alls.
_ERROR_TAXONOMY: "tuple[tuple[tuple[str, ...], str], ...]" = (
    # First, and by an exact marker: the SSRF guard refused a hop. Its text
    # names the host and the internal address it resolved to — server-log
    # material only, so it must never fall through to a message that quotes
    # tool output.
    ((GUARD_MARKER,),
     "the site could not be reached from the server"),
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
        except (ValueError, OverflowError):  # 'NA', noise, 'inf'/1e400
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
        # NOT `-m yt_dlp`: the launcher installs the SSRF guard first and
        # exits non-zero if it cannot (yt-dlp's plugin loader would only
        # print the import traceback and carry on unguarded), and running a
        # script puts ytdlp_plugins/ on sys.path instead of the repo root.
        sys.executable, GUARD_LAUNCHER,
        # Order matters: --no-plugin-dirs clears the defaults AND anything an
        # earlier --plugin-dirs added, so the guard's directory must follow
        # it. An operator's stray ~/.config/yt-dlp/plugins therefore cannot
        # pre-empt the guard.
        "--no-plugin-dirs", "--plugin-dirs", GUARD_DIR,
        "-f", DOWNLOAD_FORMAT,
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
    guard_self_check()  # fail closed: never spawn an unguarded downloader
    max_bytes = int(max_bytes or _effective_max_bytes())
    timeout = float(timeout or getattr(cfg, "URL_DOWNLOAD_TIMEOUT_SEC", 900))
    argv = build_download_argv(url, dest_dir=dest_dir, max_bytes=max_bytes)

    # YTDLP_NO_PLUGINS makes yt-dlp skip plugin loading entirely; the launcher
    # installs the guard directly and so is immune, but --plugin-dirs is the
    # belt to that braces and must not be silently disabled by the ambient
    # environment.
    env = dict(os.environ)
    env.pop("YTDLP_NO_PLUGINS", None)

    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
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

    def _emit(parsed: "tuple[int, int | None]") -> None:
        downloaded, total = parsed
        frac = (max(0.0, min(1.0, downloaded / total)) if total else None)
        try:
            progress_cb(frac, total)
        except Exception:  # noqa: BLE001 — progress must not kill the run
            pass

    stderr_task = asyncio.create_task(_drain_stderr())
    last_cb = 0.0
    last_parsed: "tuple[int, int | None] | None" = None
    last_emitted: "tuple[int, int | None] | None" = None
    try:
        assert proc.stdout is not None
        while True:
            if time.monotonic() - t0 > timeout:
                await _kill()
                raise UrlTimeoutError("the download timed out")
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
            # Belt and braces over --max-filesize, which only fires when the
            # size is known up front: a chunked / fragmented response with no
            # Content-Length would otherwise be written in full (until the
            # wall-clock timeout) before the post-hoc size check below.
            if parsed is not None and parsed[0] > max_bytes:
                await _kill()
                _discard_partials(dest_dir)
                raise UrlDownloadError("this media exceeds the server's size limit")
            if parsed and progress_cb is not None:
                last_parsed = parsed
                now = time.monotonic()
                if now - last_cb >= 0.3:
                    last_cb = now
                    last_emitted = parsed
                    _emit(parsed)
        # Flush the terminal line the 0.3 s throttle swallowed (yt-dlp emits
        # downloaded==total right on the heels of the previous line), so the
        # UI's download fraction reaches 100 %.
        if (progress_cb is not None and last_parsed is not None
                and last_parsed != last_emitted):
            _emit(last_parsed)
        await asyncio.wait_for(proc.wait(), max(5.0, timeout - (time.monotonic() - t0)))
    except asyncio.TimeoutError:
        await _kill()
        raise UrlDownloadError("the download timed out") from None
    finally:
        # Reached with the child still alive only when the TASK was
        # cancelled (e.g. uvicorn shutdown) — the deliberate abort paths
        # already reaped via _kill(). Kill synchronously: any await here
        # would just re-raise the pending CancelledError.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(stderr_task, 5.0)
        except asyncio.TimeoutError:
            stderr_task.cancel()
        except asyncio.CancelledError:
            stderr_task.cancel()
            raise
        except Exception:  # noqa: BLE001 — draining stderr must never mask the real error
            stderr_task.cancel()

    if proc.returncode != 0:
        tail = stderr_tail.decode("utf-8", "replace")
        logger.warning("[url-dl] yt-dlp exited %s for host %s: %s",
                       proc.returncode, host_for_log(url),
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
                size / 1e6, time.monotonic() - t0, host_for_log(url))
    return result


def _discard_partials(dest_dir: str) -> None:
    """Unlink whatever the killed child left in `dest_dir` (media.* and its
    .part), so an over-cap abort never leaves the bytes it refused."""
    try:
        names = os.listdir(dest_dir)
    except OSError:
        return
    for name in names:
        path = os.path.join(dest_dir, name)
        try:
            if not os.path.islink(path) and os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


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
