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
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request

from faster_whisper_backend import config as cfg
from faster_whisper_backend.core import net_policy
from faster_whisper_backend.core.store_common import log_safe
from faster_whisper_backend.paths import REPO_ROOT

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
    # The video rungs the site offers (build_video_ladder), highest first,
    # plus one trailing "audio only" entry; [] when the feature is off or
    # the link carries no video. Advisory for the client's picker — the
    # download's format selector is what actually decides.
    video_ladder: "list[dict]" = dataclasses.field(default_factory=list)


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

# The VIDEO selectors: best video + best audio merged by ffmpeg, capped at
# a height when the client picked a rung, with a pre-muxed fallback. yt-dlp's
# default sort already prefers fps → HDR → codec → bitrate inside one height.
VIDEO_FORMAT_BEST = "bv*+ba/b"
VIDEO_FORMAT_CAPPED = "bv*[height<={h}]+ba/b[height<={h}]"
VIDEO_CONTAINERS = ("mkv", "mp4")
_VIDEO_MIN_HEIGHT, _VIDEO_MAX_HEIGHT = 144, 4320
# The ladder never grows past this many rungs — a client select, not a table.
_LADDER_MAX_RUNGS = 12

# yt-dlp codec ids, ranked the way its default sort ranks them within a
# height (av01 > vp9 > hevc > avc1 > vp8).
_VCODEC_RANK = (
    ("av01", 5), ("vp09", 4), ("vp9", 4), ("hev1", 3), ("hvc1", 3),
    ("h265", 3), ("avc1", 2), ("h264", 2), ("vp8", 1),
)
# Stream families an MP4 carries with universal player support. VP9/AV1/Opus
# CAN be muxed into MP4 but play unevenly; Matroska holds all of them
# losslessly, so those rungs get "mkv".
_MP4_VIDEO = ("avc1", "h264", "hev1", "hvc1", "h265")
_MP4_AUDIO = ("mp4a", "aac")


def _vcodec_rank(vcodec: str) -> int:
    v = vcodec.lower()
    for prefix, rank in _VCODEC_RANK:
        if v.startswith(prefix):
            return rank
    return 0


def mp4_carries(vcodec: "str | None", acodec: "str | None") -> bool:
    """Whether an H.264/HEVC + AAC pair fits MP4 without re-encoding."""
    v = (vcodec or "").lower()
    a = (acodec or "").lower()
    if not v.startswith(_MP4_VIDEO):
        return False
    return (not a or a == "none") or a.startswith(_MP4_AUDIO)


def build_video_ladder(info: dict, *, max_bytes: int) -> "list[dict]":
    """The distinct video heights a site offers, from yt-dlp's `formats`
    list, highest first: one rung per height (the format yt-dlp's own sort
    would pick inside that height), with the container the merge would
    produce, an approximate merged size and an over-cap flag. Pure — no
    network; the info dict came from the probe's extract_info."""
    formats = info.get("formats")
    if not isinstance(formats, list):
        return []
    duration = info.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    def _bytes(f: dict) -> "int | None":
        fs = f.get("filesize") or f.get("filesize_approx")
        if fs:
            try:
                return int(fs)
            except (TypeError, ValueError):
                pass
        tbr = f.get("tbr")
        if tbr and duration:
            try:
                return int(float(tbr) * duration * 125)
            except (TypeError, ValueError):
                return None
        return None

    best_audio: "tuple[float, dict] | None" = None
    by_height: "dict[int, tuple[tuple, dict]]" = {}
    for f in formats:
        if not isinstance(f, dict) or f.get("has_drm"):
            continue
        proto = str(f.get("protocol") or "")
        if proto and not proto.startswith(("http", "m3u8")):
            continue
        if str(f.get("ext") or "") == "mhtml" or \
                "storyboard" in str(f.get("format_note") or "").lower():
            continue
        vcodec = f.get("vcodec")
        acodec = f.get("acodec")
        if vcodec in (None, "none"):
            if acodec and acodec != "none":
                score = float(f.get("abr") or f.get("tbr") or 0)
                if best_audio is None or score > best_audio[0]:
                    best_audio = (score, f)
            continue
        h = f.get("height")
        if not isinstance(h, (int, float)) or h <= 0:
            continue
        h = int(h)
        fps = float(f.get("fps") or 0)
        hdr = str(f.get("dynamic_range") or "SDR").upper() != "SDR"
        key = (fps, hdr, _vcodec_rank(str(vcodec)), float(f.get("tbr") or 0))
        cur = by_height.get(h)
        if cur is None or key > cur[0]:
            by_height[h] = (key, f)

    audio_f = best_audio[1] if best_audio else None
    audio_bytes = _bytes(audio_f) if audio_f else None
    audio_codec = str(audio_f.get("acodec")) if audio_f else None
    out: "list[dict]" = []
    for h in sorted(by_height, reverse=True)[:_LADDER_MAX_RUNGS]:
        _key, f = by_height[h]
        vbytes = _bytes(f)
        has_audio = f.get("acodec") not in (None, "none")
        approx = (vbytes + (audio_bytes or 0)) if (vbytes is not None and not has_audio) else vbytes
        vcodec = str(f.get("vcodec"))
        acodec = str(f.get("acodec")) if has_audio else audio_codec
        fps = float(f.get("fps") or 0)
        fps_i = int(round(fps)) if fps else None
        hdr = str(f.get("dynamic_range") or "SDR").upper() != "SDR"
        width = f.get("width")
        out.append({
            "kind": "video",
            "height": h,
            "width": int(width) if isinstance(width, (int, float)) and width > 0 else None,
            "fps": fps_i,
            "hdr": hdr,
            "vcodec": vcodec,
            "acodec": acodec,
            "container": "mp4" if mp4_carries(vcodec, acodec) else "mkv",
            "video_bytes": vbytes,
            "audio_bytes": None if has_audio else audio_bytes,
            "approx_bytes": approx,
            "over_cap": approx is not None and approx > max_bytes,
            "label": f"{h}p{fps_i if fps_i and fps_i > 30 else ''}{' HDR' if hdr else ''}",
        })
    return out


def pick_rung(ladder: "list[dict]", max_height: "int | None") -> "dict | None":
    """The rung a request gets: the highest video rung at or under the
    cap (None = best available). None when the link carries no video."""
    rungs = [r for r in ladder if r.get("kind") == "video"
             and isinstance(r.get("height"), int)]
    if not rungs:
        return None
    if max_height is not None:
        fitting = [r for r in rungs if r["height"] <= max_height]
        if fitting:
            return max(fitting, key=lambda r: r["height"])
        return min(rungs, key=lambda r: r["height"])
    return max(rungs, key=lambda r: r["height"])


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
    """The one media ceiling: a link can never admit more than an upload."""
    return int(getattr(cfg, "MEDIA_MAX_BYTES", 10_000_000_000))


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
            # No working() filter: yt-dlp still USES a _WORKING=False
            # extractor (it only warns), so the offline decision must match.
            if ie.suitable(url):
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
    public URL can't 302 into the LAN or the cloud metadata service. The
    gate is a pre-check only; the pinned handlers below are what stop a
    rebinding name from answering differently at connect time."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme.lower() not in ("http", "https"):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to a non-http URL", headers, fp)
        if not parts.hostname or _host_is_forbidden(parts.hostname):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to a forbidden address", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# build_opener() drops its default HTTPHandler/HTTPSHandler when a SUBCLASS
# is passed, so with these in the chain no unpinned handler is left: every
# connection resolves once through net_policy and dials that answer (DNS
# rebinding between the _host_is_forbidden gate and connect can't move it).
class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, conn_class=net_policy.PinnedHTTPConnection):
        super().__init__()
        self._conn_class = conn_class

    def http_open(self, req):
        # Proxied: conn.host is the operator's proxy, which the address policy
        # must trust (the target was gated by name in _host_is_forbidden).
        return self.do_open(net_policy.proxied_conn_factory(
            self._conn_class, req.has_proxy()), req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, conn_class=net_policy.PinnedHTTPSConnection):
        super().__init__()
        self._conn_class = conn_class

    def https_open(self, req):
        return self.do_open(net_policy.proxied_conn_factory(
            self._conn_class, req.has_proxy()), req, context=self._context)


class _WallClockCutoff:
    """A hard wall-clock limit on everything an opener does on the wire.

    The opener's `timeout` is per socket op and http.client reads headers
    line by line with a fresh timeout per recv, so a host dribbling one
    header byte per op can hold a probe thread for as long as it likes —
    and the outer wait_for only abandons the await, never the thread.
    Every socket the opener dials is recorded here; when the timer fires
    they are shut down, the blocked recv returns EOF at once and the
    caller's except turns that into its 'no' answer."""

    def __init__(self, seconds: float):
        self._socks: list = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(max(0.0, seconds), self.cut)
        self._timer.daemon = True

    def add(self, sock) -> None:
        with self._lock:
            self._socks.append(sock)

    def cut(self) -> None:
        with self._lock:
            socks, self._socks = self._socks, []
        for sock in socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __enter__(self):
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self._timer.cancel()
        return False


def _guarded_opener(cutoff: "_WallClockCutoff | None" = None,
                    ) -> urllib.request.OpenerDirector:
    # Keeps going through urllib.request.build_opener so tests can swap the
    # whole opener there.
    http_cls = net_policy.PinnedHTTPConnection
    https_cls = net_policy.PinnedHTTPSConnection
    if cutoff is not None:
        class http_cls(net_policy.PinnedHTTPConnection):  # type: ignore[no-redef]
            def connect(self):
                super().connect()
                cutoff.add(self.sock)

        class https_cls(net_policy.PinnedHTTPSConnection):  # type: ignore[no-redef]
            def connect(self):
                super().connect()
                cutoff.add(self.sock)
    return urllib.request.build_opener(
        _PinnedHTTPHandler(http_cls), _PinnedHTTPSHandler(https_cls),
        _NoPrivateRedirects())


def _direct_media_probe_sync(url: str, *, timeout: float) -> bool:
    """Capped GET (first byte only) that answers: does this URL serve
    audio/video directly? Host gate + redirect gate + pinned DNS keep it
    off internal ranges. Never raises for 'no' — only returns False."""
    # `timeout` is a wall-clock budget: the opener's timeout is per-socket-
    # op, so a host dribbling one header byte per op could otherwise hold
    # this worker thread far past it (the outer wait_for abandons the
    # await, never the thread). Short per-op timeout, a monotonic deadline
    # for the slow-but-answering case, and _WallClockCutoff to cut the
    # socket under a dribbler that never trips the per-op timeout.
    deadline = time.monotonic() + timeout
    op_timeout = max(1.0, min(timeout, 5.0))
    parts = urllib.parse.urlsplit(url)
    if not parts.hostname or _host_is_forbidden(parts.hostname):
        return False
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "faster-whisper-backend"},
        method="GET")
    try:
        with _WallClockCutoff(timeout) as cutoff, \
                _guarded_opener(cutoff).open(req, timeout=op_timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if time.monotonic() > deadline:
                return False
            resp.read(1)  # some servers ignore Range; never read more
    except Exception:  # noqa: BLE001 — unreachable/odd server ⇒ not direct media
        return False
    return ctype.startswith(_DIRECT_MEDIA_TYPES)


# The policy probes (extractor match, DNS in _host_is_forbidden, the capped
# direct-media GET, yt-dlp extract_info and the thumbnail GET) run on their
# own small pool: a wedged probe thread must only cost probe capacity, never
# the app-wide default executor that every other asyncio.to_thread in the
# process shares (main.py runs transcription and model loads there).
_PROBE_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="url-probe")


# ── the yt-dlp SSRF guard ───────────────────────────────────────────────────
# The probes above gate every hop THEY make (and pin DNS via net_policy's
# connection classes), but the two fetches that actually move bytes —
# probe()'s extract_info and download()'s subprocess — go through
# yt-dlp's own opener, which follows redirects and re-resolves DNS with no
# policy at all. ytdlp_plugins/ ships a RequestHandler that re-applies
# net_policy to hop 0 and to every redirect hop, pins the resolved IP, and
# speaks only http(s); it also unregisters yt-dlp's built-in handlers so
# nothing can fall through to an unguarded opener. It is installed here for
# the in-process probe and by ytdlp_plugins/run_guarded_yt_dlp.py for the
# download subprocess.
GUARD_DIR = os.path.join(REPO_ROOT, "ytdlp_plugins")
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
            probe_timeout = float(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15))
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
    max_dur = int(getattr(cfg, "URL_MAX_DURATION_S", 14400))
    dur = info.get("duration")
    if dur is not None and float(dur) > max_dur:
        raise UrlPolicyError(
            f"this media runs {float(dur) / 3600:.1f} h — over the server's "
            f"{max_dur / 3600:.1f} h limit for link downloads")
    approx = info.get("filesize_approx") or info.get("filesize")
    if approx is not None and int(approx) > _effective_max_bytes():
        raise UrlPolicyError("this media exceeds the server's size limit")


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
        raise UrlTimeoutError("the site took too long to answer") from None

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
            "socket_timeout": float(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15)),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.sanitize_info(ydl.extract_info(url, download=False))

    try:
        info = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _extract),
            max(1.0, deadline - time.monotonic()))
    except asyncio.TimeoutError:
        raise UrlTimeoutError("the site took too long to answer") from None
    except Exception as e:  # noqa: BLE001 — classify, never forward raw
        _log_probe_failure(url, e)
        raise UrlDownloadError(classify_error(str(e))) from None
    if not isinstance(info, dict):
        raise UrlDownloadError("the site returned no usable media info")
    _policy_check_info(info)
    ladder: "list[dict]" = []
    if getattr(cfg, "URL_VIDEO_ENABLED", False):
        ladder = build_video_ladder(info, max_bytes=_effective_max_bytes())
        if ladder:
            _fs = info.get("filesize_approx") or info.get("filesize")
            _ext = str(info["ext"]) if info.get("ext") else None
            _abr = float(info["abr"]) if info.get("abr") else None
            ladder.append({
                "kind": "audio", "height": None, "ext": _ext, "abr": _abr,
                "approx_bytes": int(_fs) if _fs else None,
                "over_cap": bool(_fs) and int(_fs) > _effective_max_bytes(),
                "label": "audio only" + (f" · {_ext}" if _ext else "")
                         + (f" · {int(_abr)} kbps" if _abr else ""),
            })
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
        video_ladder=ladder,
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
        try:
            # The header phase gets the same hard cutoff as the body loop
            # below: a dribbled status line never trips the per-op timeout.
            with _WallClockCutoff(timeout) as cutoff, \
                    _guarded_opener(cutoff).open(req, timeout=timeout) as resp:
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
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _fetch),
            timeout + 2.0)
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
    from faster_whisper_backend.streaming.transport import ffmpeg_exe

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
        "--socket-timeout", str(int(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15))),
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


def build_video_download_argv(url: str, *, dest_dir: str, max_bytes: int,
                              max_height: "int | None" = None,
                              container: str = "mkv") -> "list[str]":
    """The yt-dlp invocation for the VIDEO of a link: best video + best
    audio, merged by ffmpeg into `container`. Same launcher, guard, caps
    and output rules as build_download_argv; `max_height` is clamped to an
    int here so no client value ever reaches the selector as text."""
    from faster_whisper_backend.streaming.transport import ffmpeg_exe

    if container not in VIDEO_CONTAINERS:
        container = "mkv"
    if max_height is not None:
        h = max(_VIDEO_MIN_HEIGHT, min(_VIDEO_MAX_HEIGHT, int(max_height)))
        fmt = VIDEO_FORMAT_CAPPED.format(h=h)
    else:
        fmt = VIDEO_FORMAT_BEST
    return [
        sys.executable, GUARD_LAUNCHER,
        "--no-plugin-dirs", "--plugin-dirs", GUARD_DIR,
        "-f", fmt,
        "--merge-output-format", container,
        "--no-playlist",
        "--playlist-items", "1",
        "--restrict-filenames",
        "--max-filesize", str(max_bytes),
        "--socket-timeout", str(int(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15))),
        "--retries", "3",
        "--no-mtime",
        "--ffmpeg-location", ffmpeg_exe(),
        "-P", dest_dir,
        "-o", "media.%(ext)s",   # NEVER %(title)s — see build_download_argv
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
    timeout = float(timeout or getattr(cfg, "URL_DOWNLOAD_TIMEOUT_S", 900))
    argv = build_download_argv(url, dest_dir=dest_dir, max_bytes=max_bytes)

    def _emit(downloaded: int, total: "int | None") -> None:
        if progress_cb is None:
            return
        frac = (max(0.0, min(1.0, downloaded / total)) if total else None)
        progress_cb(frac, total)

    return await _run_yt_dlp(
        argv, url=url, dest_dir=dest_dir, max_bytes=max_bytes,
        timeout=timeout, emit=_emit, cancel_check=cancel_check,
        find_result=_find_result_file)


async def download_video(
    url: str,
    *,
    dest_dir: str,
    max_bytes: "int | None" = None,
    max_height: "int | None" = None,
    container: str = "mkv",
    expected_total: "int | None" = None,
    timeout: "float | None" = None,
    progress_cb=None,
    cancel_check=None,
) -> str:
    """Download the VIDEO of `url` (best video + best audio merged into
    `container`) into `dest_dir` and return the file path.

    progress_cb(fraction_or_None, total_bytes_or_None, downloaded_bytes) is
    throttled like download()'s, but counts CUMULATIVELY across the two
    streams yt-dlp fetches (video, then audio — each restarts its own
    counter at 0), against `expected_total` (the probe's merged estimate)
    when known. Bounded by URL_VIDEO_DOWNLOAD_TIMEOUT_S by default."""
    url = validate_url(url)
    guard_self_check()
    max_bytes = int(max_bytes or _effective_max_bytes())
    timeout = float(timeout or getattr(cfg, "URL_VIDEO_DOWNLOAD_TIMEOUT_S", 3600))
    if container not in VIDEO_CONTAINERS:
        container = "mkv"
    argv = build_video_download_argv(
        url, dest_dir=dest_dir, max_bytes=max_bytes, max_height=max_height,
        container=container)

    def _emit(downloaded: int, total: "int | None") -> None:
        if progress_cb is None:
            return
        frac = (max(0.0, min(1.0, downloaded / total)) if total else None)
        progress_cb(frac, total, downloaded)

    return await _run_yt_dlp(
        argv, url=url, dest_dir=dest_dir, max_bytes=max_bytes,
        timeout=timeout, emit=_emit, cancel_check=cancel_check,
        expected_total=expected_total,
        find_result=lambda d: _find_video_result(d, container))


async def _run_yt_dlp(
    argv: "list[str]",
    *,
    url: str,
    dest_dir: str,
    max_bytes: int,
    timeout: float,
    emit,
    cancel_check,
    find_result,
    expected_total: "int | None" = None,
) -> str:
    """The subprocess half shared by download() and download_video(): run
    yt-dlp, stream its progress lines into `emit(downloaded, total)` (one
    call per 0.3 s, cumulative across the files one invocation fetches),
    enforce the byte cap, the wall clock and cancellation, then hand the
    finished directory to `find_result`."""
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

    # yt-dlp prints one `dl:` series per file it fetches (a merge = video
    # then audio, each restarting at 0). `completed` carries the finished
    # files' bytes so the fraction, the cap and the log all see the sum.
    completed = 0
    last_downloaded = 0

    def _cumulative(parsed: "tuple[int, int | None]") -> "tuple[int, int | None]":
        nonlocal completed, last_downloaded
        downloaded, total = parsed
        if downloaded < last_downloaded:
            completed += last_downloaded
        last_downloaded = downloaded
        cum = completed + downloaded
        if expected_total:
            total_all: "int | None" = max(int(expected_total), cum)
        elif total:
            total_all = completed + total
        else:
            total_all = None
        return cum, total_all

    def _emit(parsed: "tuple[int, int | None]") -> None:
        try:
            emit(*parsed)
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
            if parsed is not None:
                parsed = _cumulative(parsed)
            # Belt and braces over --max-filesize, which only fires when the
            # size is known up front (and per FILE — a merge's two streams
            # can pass it separately): a chunked / fragmented response with
            # no Content-Length would otherwise be written in full (until
            # the wall-clock timeout) before the post-hoc size check below.
            if parsed is not None and parsed[0] > max_bytes:
                await _kill()
                _discard_partials(dest_dir)
                raise UrlPolicyError("this media exceeds the server's size limit")
            if parsed:
                last_parsed = parsed
                now = time.monotonic()
                if now - last_cb >= 0.3:
                    last_cb = now
                    last_emitted = parsed
                    _emit(parsed)
        # Flush the terminal line the 0.3 s throttle swallowed (yt-dlp emits
        # downloaded==total right on the heels of the previous line), so the
        # UI's download fraction reaches 100 %.
        if last_parsed is not None and last_parsed != last_emitted:
            _emit(last_parsed)
        await asyncio.wait_for(proc.wait(), max(5.0, timeout - (time.monotonic() - t0)))
    except asyncio.TimeoutError:
        await _kill()
        raise UrlTimeoutError("the download timed out") from None
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

    result = find_result(dest_dir)
    if result is None:
        # --max-filesize skips (exit 0, no file) on some formats instead of
        # failing — a missing output after a clean exit means the cap bit.
        raise UrlPolicyError("this media exceeds the server's size limit")
    size = os.path.getsize(result)
    if size > max_bytes:
        raise UrlPolicyError("this media exceeds the server's size limit")
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


# The finished output `-o media.%(ext)s` produces: exactly one dot. A merge's
# intermediates (`media.f251.webm`) and partials (`media.mkv.part`) both
# carry a second one and never qualify.
_RESULT_NAME_RE = re.compile(r"\Amedia\.[a-z0-9]{2,5}\Z")
_INTERMEDIATE_NAME_RE = re.compile(r"\Amedia\.f[^.]+\.[a-z0-9]+\Z")


def _find_result_file(dest_dir: str) -> "str | None":
    """The completed download inside `dest_dir`, or None. Refuses partials,
    merge intermediates and anything that escapes the directory (symlink
    games)."""
    root = os.path.realpath(dest_dir)
    best: "tuple[float, str] | None" = None
    try:
        names = os.listdir(dest_dir)
    except OSError:
        return None
    for name in names:
        # yt-dlp's control files (`media.mkv.part`, `.ytdl`) carry two dots
        # and fail the shape below; the bare-suffix spellings are refused
        # by name so no future naming change can smuggle one through.
        if name.endswith((".part", ".ytdl", ".tmp")):
            continue
        if not _RESULT_NAME_RE.match(name):
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


def _find_video_result(dest_dir: str, container: str) -> "str | None":
    """The merged video inside `dest_dir`: `media.<container>` when ffmpeg
    merged the streams; the single muxed download when the site served one
    file and no merge happened. Intermediates left behind (one stream of the
    pair skipped by --max-filesize, or a failed merge) are an error, never a
    result — the caller must not retain a video-only or audio-only stream
    as "the video"."""
    try:
        names = os.listdir(dest_dir)
    except OSError:
        return None
    root = os.path.realpath(dest_dir)
    final = os.path.join(dest_dir, f"media.{container}")
    if os.path.isfile(final) and not os.path.islink(final):
        real = os.path.realpath(final)
        if real == root or real.startswith(root + os.sep):
            return real
    intermediates = [n for n in names
                     if _INTERMEDIATE_NAME_RE.match(n) or n.endswith((".part", ".ytdl"))]
    single = _find_result_file(dest_dir)
    if single is not None and not intermediates:
        return single
    if intermediates:
        raise UrlDownloadError("the video could not be merged on the server")
    return None
