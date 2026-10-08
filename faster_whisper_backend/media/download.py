"""Transcribe-from-URL support: fetch a client-supplied media link with yt-dlp.

Contract (mirrors bgm_separation.py):
  - str(UrlDownloadError) is CLIENT-SAFE — our own wording, never raw yt-dlp
    stderr (which can carry filesystem paths and full URLs with tokens).
  - UrlCancelled means the caller's cancel_check tripped; the request must
    abort (HTTP 499 via transcription/progress's _ClientCancelled), not soft-fail.

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
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request

from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.core import net_policy
from faster_whisper_backend.core import proc as core_proc
from faster_whisper_backend.core.store_common import log_safe
from faster_whisper_backend.paths import REPO_ROOT

logger = logging.getLogger("whisper-api")

_URL_MAX_LEN = 2048
# Anything a shell/log/terminal could misread. URLs have no business carrying
# raw whitespace or C0/C1 bytes (they'd be %-encoded in a real URL). A lone
# surrogate (a JSON body can carry "\ud800") cannot be encoded into the
# download child's argv: refused here, not a 500 after the probe.
_URL_UNSAFE_RE = re.compile(r"[\s\x00-\x1f\x7f-\x9f\ud800-\udfff]")

# Content types accepted by the direct-media probe. application/ogg is the
# registered type for .ogg/.opus; everything else must declare audio/* or
# video/*.
_DIRECT_MEDIA_TYPES = ("audio/", "video/", "application/ogg")
# HLS manifest types, admitted only for a site → GenericIE hand-off (see
# _generic_handoff_target): the download child's match filter takes the same
# manifest ("protocol^=m3u8"), but a pasted link to a bare manifest stays
# held to _DIRECT_MEDIA_TYPES. Most CDNs answer the two application/* types
# (RFC 8216), which the audio/ prefix never matched.
_HLS_MANIFEST_TYPES = ("application/vnd.apple.mpegurl", "application/x-mpegurl",
                       "audio/mpegurl", "audio/x-mpegurl")


class UrlDownloadError(RuntimeError):
    """str() is CLIENT-SAFE by contract (our wording, never tool output)."""


class UrlPolicyError(UrlDownloadError):
    """The operator's URL / size policy refused the input (not a fetch
    failure). Pre-classified for the failures card."""
    error_class = "policy_blocked"


class UrlTimeoutError(UrlDownloadError):
    """A probe or download ran past its timeout."""
    error_class = "timeout"


class UrlTooLargeError(UrlDownloadError):
    """A capped GET's body ran past its max_bytes. Typed so a caller that
    passed a cap smaller than its own limit (subtitles' remaining budget)
    can tell which cap tripped without matching on the wording."""


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
    # The spoken language the site names, and its own subtitle tracks
    # (media/subtitles.py): the public list for the preview, and the
    # server-only {id: {url, ext}} the url-subtitles route fetches from —
    # signed URLs that must never reach a client or a log line.
    language: "str | None" = None
    subtitle_tracks: "list[dict]" = dataclasses.field(default_factory=list)
    subtitle_sources: "dict[str, dict]" = dataclasses.field(
        default_factory=dict, repr=False)
    # Server-only, signed URLs like subtitle_sources: the selected audio
    # format when it is a segmented stream (HLS/DASH), which the language
    # check samples without the whole file (media/segmented.py); None for a
    # progressive file.
    segmented: "dict | None" = dataclasses.field(default=None, repr=False)


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
# yt-dlp format ids as the selector accepts them (itags, "hls-1080p",
# "DASH_720", "http-2500k"…). Anything else never reaches `-f`.
FORMAT_ID_RE = re.compile(r"\A[A-Za-z0-9._+-]{1,40}\Z")
# A format_note that NAMES a rung rather than restating its height:
# YouTube's "Premium", Twitch's "Source", Vimeo's "Original".
_NAMED_NOTE_RE = re.compile(r"\A(premium|source|original)\Z", re.I)
# Two rungs inside one height whose bitrates sit this close are one rung.
_TIER_DEDUPE = 0.10
# A direct file the generic extractor could not inspect (no formats, no
# codecs) still IS a video when its extension says so.
_VIDEO_FILE_EXTS = ("mp4", "mkv", "webm", "mov", "m4v", "avi", "ts", "flv", "mpg", "mpeg")
# The ledger key under which the download's actual-over-estimated byte ratio
# is learned per extractor and protocol family (stage_rates has no seed for
# it: an unmeasured site shows the site's own numbers, marked approximate).
RATIO_STAGE = "dlratio"

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


def _fmt_bytes(f: dict, duration: "float | None") -> "tuple[int | None, bool]":
    """(bytes, approximate): the site's exact size when it lists one, else
    its own estimate, else bitrate × duration — the last two flagged."""
    fs = f.get("filesize")
    if fs:
        try:
            return int(fs), False
        except (TypeError, ValueError):
            pass
    fsa = f.get("filesize_approx")
    if fsa:
        try:
            return int(fsa), True
        except (TypeError, ValueError):
            pass
    tbr = f.get("tbr")
    if tbr and duration:
        try:
            return int(float(tbr) * duration * 125), True
        except (TypeError, ValueError):
            return None, True
    return None, True


def _fmt_rank(f: dict) -> tuple:
    """yt-dlp's own order within a site: quality, resolution, fps, HDR,
    source preference (YouTube's Premium boost lives here), codec, bitrate.
    Ranking the way the downloader ranks is what keeps the card and the
    fetched file in agreement."""
    try:
        q = float(f.get("quality") or 0)
    except (TypeError, ValueError):
        q = 0.0
    h = f.get("height")
    w = f.get("width")
    try:
        src = float(f.get("source_preference")
                    if f.get("source_preference") is not None else -1)
    except (TypeError, ValueError):
        src = -1.0
    return (
        q,
        int(h) if isinstance(h, (int, float)) and h > 0 else 0,
        int(w) if isinstance(w, (int, float)) and w > 0 else 0,
        float(f.get("fps") or 0),
        str(f.get("dynamic_range") or "SDR").upper() != "SDR",
        src,
        _vcodec_rank(str(f.get("vcodec") or "")),
        float(f.get("tbr") or 0),
    )


def protocol_family(protocol: "str | None") -> str:
    """The three families whose size honesty differs: fragmented HLS and
    DASH manifests list peak bitrates and no sizes; a plain https file is
    exact."""
    p = str(protocol or "").lower()
    if p.startswith("m3u8"):
        return "m3u8"
    if "dash" in p:
        return "dash"
    return "https"


def _rung_spec(f: dict) -> str:
    """The resolution half of a label: "1080p", "1080p60 HDR", the site's
    own "1280x720" when only that is known, else ""."""
    h = f.get("height")
    if isinstance(h, (int, float)) and h > 0:
        fps = float(f.get("fps") or 0)
        fps_i = int(round(fps)) if fps else 0
        hdr = str(f.get("dynamic_range") or "SDR").upper() != "SDR"
        return f"{int(h)}p{fps_i if fps_i > 30 else ''}{' HDR' if hdr else ''}"
    res = f.get("resolution")
    if isinstance(res, str) and res.strip() and res.strip().lower() != "audio only":
        return res.strip()[:16]
    return ""


def _rung_note(f: dict) -> "str | None":
    note = f.get("format_note")
    if isinstance(note, str) and _NAMED_NOTE_RE.match(note.strip()):
        return note.strip().title()
    return None


def _video_candidates(formats: list) -> "tuple[list[dict], dict | None]":
    """The rankable video formats and the best separate audio track."""
    best_audio: "tuple[tuple, dict] | None" = None
    vids: "list[dict]" = []
    for f in formats:
        if not isinstance(f, dict) or f.get("has_drm"):
            continue
        proto = str(f.get("protocol") or "")
        if proto and not proto.startswith(("http", "m3u8")):
            continue
        if str(f.get("ext") or "") == "mhtml" or \
                "storyboard" in str(f.get("format_note") or "").lower():
            continue
        fid = f.get("format_id")
        if not isinstance(fid, str) or not FORMAT_ID_RE.match(fid):
            continue
        vcodec = f.get("vcodec")
        acodec = f.get("acodec")
        if vcodec in (None, "none"):
            # YouTube's "DRC" twins are loudness-normalised duplicates of
            # the same track — never the leg a merge should carry.
            if acodec and acodec != "none" and \
                    "drc" not in str(f.get("format_note") or "").lower():
                # yt-dlp's own `ba` ranks the track's language ahead of its
                # bitrate (the extractor marks the original track 10, a dub
                # -1, audio description -10) — by bitrate alone a louder
                # dub became the kept video's soundtrack while the
                # transcript followed the original.
                lp = f.get("language_preference")
                q = f.get("quality")
                score = (lp if isinstance(lp, (int, float)) else -1,
                         q if isinstance(q, (int, float)) else 0,
                         float(f.get("abr") or f.get("tbr") or 0))
                if best_audio is None or score > best_audio[0]:
                    best_audio = (score, f)
            continue
        vids.append(f)
    return vids, (best_audio[1] if best_audio else None)


def build_video_ladder(info: dict, *, max_bytes: int,
                       extractor: "str | None" = None,
                       approx_ratio=None) -> "list[dict]":
    """The video rungs a site offers, best first, in yt-dlp's own order —
    so the top rung IS what `bv*+ba` would fetch (YouTube's Premium 1080p
    included). One rung per height, plus a second rung inside a height when
    the top one has no exact size and an exactly-sized alternative differs
    in bitrate by more than 10 % (the exact AV1 1080p beside the estimated
    Premium one). Sites that name nothing rankable but clearly serve a
    video file get one "Best available" rung.

    `approx_ratio(protocol_family) -> float | None` scales estimated bytes
    and bitrates by what this site's fragmented streams actually delivered
    last time (the ledger's learned actual/estimated ratio). Pure — no
    network; the info dict came from the probe's extract_info."""
    formats = info.get("formats")
    if not isinstance(formats, list):
        formats = []
    duration = info.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    vids, audio_f = _video_candidates(formats)
    groups: "dict[object, list[dict]]" = {}
    for f in vids:
        h = f.get("height")
        if isinstance(h, (int, float)) and h > 0:
            key: object = int(h)
        else:
            res = f.get("resolution")
            key = f"res:{res}" if isinstance(res, str) and res else f"id:{f['format_id']}"
        groups.setdefault(key, []).append(f)

    chosen: "list[dict]" = []
    for members in groups.values():
        members.sort(key=_fmt_rank, reverse=True)
        top = members[0]
        chosen.append(top)
        _tb, top_approx = _fmt_bytes(top, duration)
        if not top_approx:
            continue
        top_tbr = float(top.get("tbr") or 0)
        for m in members[1:]:
            _b, approx = _fmt_bytes(m, duration)
            if approx:
                continue
            mt = float(m.get("tbr") or 0)
            if top_tbr and mt and abs(mt - top_tbr) / top_tbr <= _TIER_DEDUPE:
                continue
            chosen.append(m)
            break
    chosen.sort(key=_fmt_rank, reverse=True)

    out = [_rung(f, audio_f=audio_f, duration=duration, max_bytes=max_bytes,
                 extractor=extractor, approx_ratio=approx_ratio)
           for f in chosen[:_LADDER_MAX_RUNGS]]
    if out:
        return out
    # Nothing rankable (no format survived as a video candidate — every one
    # would have made a rung). A direct file the generic extractor could
    # only name by extension is still a video; a podcast mp3 is not.
    ext = str(info.get("ext") or "").lower()
    vcodec = info.get("vcodec")
    if ext in _VIDEO_FILE_EXTS or vcodec not in (None, "none"):
        fs = info.get("filesize") or None
        b = int(fs) if fs else None
        return [{
            "kind": "video", "height": None, "width": None, "fps": None,
            "hdr": False, "vcodec": str(vcodec) if vcodec else None,
            "acodec": None, "container": "mkv", "protocol": "https",
            "format_id": None, "audio_format_id": None,
            "video_bytes": b, "audio_bytes": None, "approx_bytes": b,
            "bytes_approx": b is None, "tbr_kbps": None, "bitrate_approx": True,
            "note": None, "extractor": extractor,
            "over_cap": b is not None and b > max_bytes,
            "label": "Best available",
        }]
    return []


def _rung(f: dict, *, audio_f: "dict | None", duration: "float | None",
          max_bytes: int, extractor: "str | None", approx_ratio) -> dict:
    has_audio = f.get("acodec") not in (None, "none")
    proto = str(f.get("protocol") or "")
    fam = protocol_family(proto)
    vb, v_approx = _fmt_bytes(f, duration)
    ratio = None
    if v_approx and approx_ratio is not None:
        try:
            r = approx_ratio(fam)
            ratio = float(r) if r and r > 0 else None
        except Exception:  # noqa: BLE001 — a ledger hiccup never breaks a preview
            ratio = None
    # The estimate BEFORE the learned ratio: the recorder measures the real
    # size against this one, so each sample is actual/raw and the EWMA
    # converges on the true ratio (against the scaled value it settles on
    # its square root).
    vb_raw = vb
    if vb is not None and ratio:
        vb = int(vb * ratio)
    ab: "int | None" = None
    ab_raw: "int | None" = None
    a_approx = False
    a_tbr = 0.0
    if audio_f is not None and not has_audio:
        ab, a_approx = _fmt_bytes(audio_f, duration)
        ab_raw = ab
        a_tbr = float(audio_f.get("abr") or audio_f.get("tbr") or 0)
        if ab is not None and a_approx and approx_ratio is not None:
            try:
                r = approx_ratio(protocol_family(audio_f.get("protocol")))
                if r and r > 0:
                    ab = int(ab * float(r))
            except Exception:  # noqa: BLE001
                pass
    approx = (vb + (ab or 0)) if vb is not None else None
    v_tbr = float(f.get("tbr") or f.get("vbr") or 0)
    if v_tbr and v_approx and ratio:
        v_tbr *= ratio
    tbr = v_tbr + a_tbr
    vcodec = str(f.get("vcodec"))
    acodec = str(f.get("acodec")) if has_audio else (
        str(audio_f.get("acodec")) if audio_f is not None else None)
    fps = float(f.get("fps") or 0)
    fps_i = int(round(fps)) if fps else None
    hdr = str(f.get("dynamic_range") or "SDR").upper() != "SDR"
    h = f.get("height")
    width = f.get("width")
    note = _rung_note(f)
    spec = _rung_spec(f)
    label = " ".join(x for x in (spec, note) if x) or "Best available"
    # An explicitly video-only format with no usable audio candidate (an HLS
    # audio rendition carries no acodec and is dropped) still needs a merge:
    # "ba" keeps the exact video id and lets yt-dlp pick the audio, instead
    # of retaining a soundless file. acodec None (unknown, possibly muxed)
    # stays a single-file rung.
    if audio_f is not None and not has_audio:
        audio_id: "str | None" = str(audio_f["format_id"])
    elif audio_f is None and f.get("acodec") == "none":
        audio_id = "ba"
    else:
        audio_id = None
    return {
        "kind": "video",
        "height": int(h) if isinstance(h, (int, float)) and h > 0 else None,
        "width": int(width) if isinstance(width, (int, float)) and width > 0 else None,
        "fps": fps_i,
        "hdr": hdr,
        "vcodec": vcodec,
        "acodec": acodec,
        "container": "mp4" if mp4_carries(vcodec, acodec) else "mkv",
        "protocol": fam,
        "format_id": str(f["format_id"]),
        "audio_format_id": audio_id,
        "video_bytes": vb,
        "audio_bytes": None if has_audio else ab,
        "approx_bytes": approx,
        # Unscaled twin of approx_bytes (both legs, no learned ratio). The
        # sample the recorder files mixes both legs' bytes under the VIDEO
        # rung's protocol family.
        "raw_approx_bytes": (vb_raw + (ab_raw or 0)) if vb_raw is not None else None,
        "bytes_approx": bool(v_approx or (not has_audio and a_approx)),
        "tbr_kbps": int(round(tbr)) if tbr else None,
        # A fragmented stream's bitrate is the manifest's peak, not an
        # average — the learned ratio narrows it, the flag keeps the "≈".
        "bitrate_approx": fam != "https",
        "note": note,
        "extractor": extractor,
        "over_cap": approx is not None and approx > max_bytes,
        "label": label,
    }


def pick_rung(ladder: "list[dict]", max_height: "int | None",
              format_id: "str | None" = None) -> "dict | None":
    """The rung a request gets: the one whose format id the client named
    when it is still on the ladder, else the best-ranked rung at or under
    the height cap (None = best available), else the smallest. None when
    the link carries no video. The ladder is rank-ordered, so "best" is
    its first entry — YouTube's Premium rung outranks plain 1080p."""
    rungs = [r for r in ladder if r.get("kind") == "video"]
    if not rungs:
        return None
    if format_id:
        for r in rungs:
            if r.get("format_id") == format_id:
                return r
    if max_height is not None:
        fitting = [r for r in rungs if isinstance(r.get("height"), int)
                   and r["height"] <= max_height]
        if fitting:
            return fitting[0]
        with_h = [r for r in rungs if isinstance(r.get("height"), int)]
        if with_h:
            return min(with_h, key=lambda r: r["height"])
    return rungs[0]


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
    # An empty or >63-char label (`a..com`) or a lone surrogate makes
    # getaddrinfo raise UnicodeError downstream: answer a client 400 here.
    try:
        parts.hostname.encode("idna")
    except UnicodeError:
        raise UrlDownloadError("the URL's host name is invalid") from None
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
        # Set once the timer has shut the sockets: a body read cut here
        # ends in a short b"" (http.client never raises IncompleteRead for
        # read(amt)), which a caller must not take for a complete body.
        self.fired = False
        self._timer = threading.Timer(max(0.0, seconds), self.cut)
        self._timer.daemon = True

    def add(self, sock) -> None:
        # A socket whose connect() returns after the timer fired (a slow TLS
        # handshake, a redirect hop dialled past the cutoff) would otherwise
        # land in the list cut() already emptied and never be shut down.
        with self._lock:
            late = self.fired
            if not late:
                self._socks.append(sock)
        if late:
            self._shut(sock)

    def cut(self) -> None:
        with self._lock:
            self.fired = True
            socks, self._socks = self._socks, []
        for sock in socks:
            self._shut(sock)

    @staticmethod
    def _shut(sock) -> None:
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


def _direct_media_probe_sync(url: str, *, timeout: float,
                             accept_hls: bool = False) -> bool:
    """Capped GET (first byte only) that answers: does this URL serve
    audio/video directly? Host gate + redirect gate + pinned DNS keep it
    off internal ranges. Never raises for 'no' — only returns False.
    `accept_hls` also counts an HLS manifest (_HLS_MANIFEST_TYPES)."""
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
    return ctype.startswith(_DIRECT_MEDIA_TYPES) or (
        accept_hls and ctype in _HLS_MANIFEST_TYPES)


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
# Two refusals that are NOT "the site could not be reached" and carry their
# own, longer needle (still containing GUARD_MARKER, so the taxonomy matches
# them first): the guard refusing an external downloader (ffmpeg, aria2c…)
# for a format, and the launcher failing closed because it could not install
# the guard in the child. Must equal fwb_ssrf_guard.EXTERNAL_FD_MARKER and
# run_guarded_yt_dlp._UNAVAILABLE_MARKER (pinned by tests).
GUARD_EXTERNAL_FD_MARKER = GUARD_MARKER + "-external-fd"
GUARD_UNAVAILABLE_MARKER = GUARD_MARKER + "-unavailable"

_guard_ok = False
_guard_announced = False
# The guard module guard_self_check loaded (its observe_dials hook bounds the
# in-process probe by wall clock — see probe()). None until the first check.
_guard_module = None


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
    global _guard_ok, _guard_announced, _guard_module
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
        _guard_module = module
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
    allowed = _allowed_extractor_keys()
    if allowed and key.lower() not in allowed:
        raise UrlPolicyError("this site isn't on the server's allowed list")
    return key


def _allowed_extractor_keys() -> "list[str]":
    return [a.strip().lower()
            for a in (getattr(cfg, "URL_ALLOWED_EXTRACTORS", []) or [])
            if a and a.strip()]


def pinned_extractors(key: str) -> "list[str] | None":
    """yt-dlp's `allowed_extractors` (regexes over IE names) for a URL the
    offline match called `key`, so extraction can only hand off — GenericIE
    following an embed, a url_transparent delegation — to an extractor the
    site policy admits. None: no pin (every default extractor).

    Generic without URL_ALLOW_GENERIC was admitted as a direct media file,
    which GenericIE serves itself: nothing else may run. Under an allowlist,
    the allowlisted extractors (plus GenericIE when that is how the URL got
    in). Blocking IO-free but walks every extractor class: call it off the
    loop."""
    if key == "Generic" and not getattr(cfg, "URL_ALLOW_GENERIC", False):
        return ["generic"]
    allowed = _allowed_extractor_keys()
    if not allowed:
        return None
    import yt_dlp.extractor  # lazy: optional dependency

    want = set(allowed) | ({"generic"} if key == "Generic" else set())
    names = [re.escape(ie.IE_NAME.lower())
             for ie in yt_dlp.extractor.gen_extractor_classes()
             if ie.ie_key().lower() in want]
    return names or None


def handoff_match_filters(key: str) -> "list[str] | None":
    """yt-dlp `--match-filters` (OR'd) for the download subprocess of a URL
    the offline match called `key`. The subprocess re-extracts from scratch,
    so a site extractor can hand off to GenericIE there even though the
    probe's _policy_check_extractor refused nothing: with URL_ALLOW_GENERIC
    off, a GenericIE result is downloaded only when it is the plain media
    file or HLS manifest GenericIE served itself — what
    URL_ALLOW_DIRECT_MEDIA admits — never a scraped page. A filtered-out
    run exits 0 with no file; _run_yt_dlp reports it as the policy refusal.
    (The probe's verdict is stricter: it re-checks the Content-Type of the
    URL GenericIE was handed, so a scraped page embedding HLS is refused
    there while this filter's protocol^=m3u8 would take it.)
    None: no filter (Generic is the matched key — pinned_extractors already
    holds it to GenericIE — or GenericIE is allowed outright)."""
    if key == "Generic" or getattr(cfg, "URL_ALLOW_GENERIC", False):
        return None
    filters = ["extractor_key!=Generic"]
    if getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", True):
        filters += ["direct", "protocol^=m3u8"]
    return filters


def _is_outer_site_page(key: str, url: str, page: str) -> bool:
    """Does `page` (a GenericIE result's webpage_url) name the site page the
    probe started from — the probed `url` itself, or a page the matched
    `key` extractor claims (its canonical form)?"""
    if page == url:
        return True
    try:
        import yt_dlp.extractor  # lazy: optional dependency
        return bool(yt_dlp.extractor.get_info_extractor(key).suitable(page))
    except Exception:  # noqa: BLE001 — unknown key / broken pattern ⇒ not it
        return False


def _generic_handoff_target(key: str, info: dict, url: str) -> "str | None":
    """The URL to hold to check_url_policy's direct-media verdict when a site
    extractor (`key`, matched on the probed `url`) handed off to GenericIE
    with URL_ALLOW_GENERIC off and URL_ALLOW_DIRECT_MEDIA on; None when no
    such re-check applies. A "direct" result is the URL GenericIE served
    as-is ("url"). On a plain hand-off "webpage_url" is the URL GenericIE
    was handed: a manifest it served itself passes, a page it scraped (even
    one embedding HLS) answers text/html and does not. On a url_transparent
    hand-off (the common site → embed delegation) yt-dlp overlays the OUTER
    result's fields, so "webpage_url" names the site's own page; only then
    is the hand-off judged by the manifest GenericIE fetched (the formats'
    "manifest_url"), with webpage_url as the fallback when no format names
    one; formats naming two refuse (None)."""
    if (key == "Generic" or str(info.get("extractor_key") or key) != "Generic"
            or getattr(cfg, "URL_ALLOW_GENERIC", False)
            or not getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", True)):
        return None
    page = info.get("webpage_url")
    if info.get("direct"):
        target = info.get("url")
    elif isinstance(page, str) and page and not _is_outer_site_page(
            key, url, page):
        target = page
    else:
        fmts = [info, *(info.get("requested_formats") or []),
                *(info.get("formats") or [])]
        manifests = {f["manifest_url"] for f in fmts if isinstance(f, dict)
                     and isinstance(f.get("manifest_url"), str)
                     and f["manifest_url"]}
        if len(manifests) > 1:
            return None
        target = manifests.pop() if manifests else page
    return target if isinstance(target, str) and target else None


async def _handoff_serves_media(target: str, deadline: float) -> bool:
    """check_url_policy's direct-media verdict (the same guarded capped
    GET) for a GenericIE hand-off target, within the probe's deadline."""
    try:
        target = validate_url(target)
    except UrlDownloadError:
        return False
    budget = max(1.0, min(float(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15)),
                          deadline - time.monotonic()))
    return await asyncio.get_running_loop().run_in_executor(
        _PROBE_POOL,
        lambda: _direct_media_probe_sync(target, timeout=budget, accept_hls=True))


def _policy_check_extractor(key: str, info: dict, *,
                            handoff_is_media: bool = False) -> None:
    """The extractor that actually produced `info` must pass the policy the
    offline match was held to: a Generic URL admitted as direct media must
    not have been handed to a site extractor, and under an allowlist a
    delegation must land on an allowlisted extractor. A site extractor
    handing a scraped embed URL on to GenericIE is held to URL_ALLOW_GENERIC
    too (GenericIE scrapes any page), unless the hand-off target passed the
    same Content-Type check check_url_policy gives a pasted Generic URL
    (`handoff_is_media`, see _generic_handoff_target) — what
    URL_ALLOW_DIRECT_MEDIA admits. yt-dlp's own "direct" flag is not that
    verdict: it marks any non-HTML body, and never an HLS manifest."""
    xk = str(info.get("extractor_key") or key)
    if xk == key:
        return
    if key == "Generic" and not getattr(cfg, "URL_ALLOW_GENERIC", False):
        raise UrlPolicyError(
            "this link leads to a site the server's URL policy doesn't allow")
    if (xk == "Generic" and not getattr(cfg, "URL_ALLOW_GENERIC", False)
            and not (handoff_is_media
                     and getattr(cfg, "URL_ALLOW_DIRECT_MEDIA", True))):
        raise UrlPolicyError(
            "this link leads to a site the server's URL policy doesn't allow")
    allowed = _allowed_extractor_keys()
    if allowed and xk.lower() not in allowed:
        raise UrlPolicyError(
            "this link leads to a site that isn't on the server's allowed list")


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

        pin = pinned_extractors(key)
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
        if pin:
            opts["allowed_extractors"] = pin
        # socket_timeout is per socket op: a host dribbling one header byte
        # per op would hold this pool thread for as long as it likes (the
        # wait_for below abandons only the await). Every socket the guard
        # dials for this thread is shut at the deadline instead — the same
        # cutoff the direct-media probe and capped_get use.
        observe = getattr(_guard_module, "observe_dials", None)
        with _WallClockCutoff(max(1.0, deadline - time.monotonic())) as cutoff:
            if observe is not None:
                observe(cutoff.add)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.sanitize_info(
                        ydl.extract_info(url, download=False))
            finally:
                if observe is not None:
                    observe(None)

    try:
        info = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _extract),
            max(1.0, deadline - time.monotonic()))
    except asyncio.TimeoutError:
        raise UrlTimeoutError("the site took too long to answer") from None
    except Exception as e:  # noqa: BLE001 — classify, never forward raw
        _log_probe_failure(url, e)
        raise classified_error(str(e)) from None
    if not isinstance(info, dict):
        raise UrlDownloadError("the site returned no usable media info")
    _policy_check_info(info)
    handoff_is_media = False
    _target = _generic_handoff_target(key, info, url)
    if _target is not None:
        # Under the same deadline as the two steps above: the hand-off
        # host's DNS (picked by the site) and a busy _PROBE_POOL both sit
        # outside the probe's own socket cutoff.
        try:
            handoff_is_media = await asyncio.wait_for(
                _handoff_serves_media(_target, deadline),
                max(0.0, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            raise UrlTimeoutError("the site took too long to answer") from None
    _policy_check_extractor(key, info, handoff_is_media=handoff_is_media)
    ladder: "list[dict]" = []
    if getattr(cfg, "URL_VIDEO_ENABLED", False):
        from faster_whisper_backend.runtime import stage_rates as _rates
        _xk = str(info.get("extractor_key") or key)
        # A video rung is only ever fetched to be retained: price it
        # against what the media store can keep.
        from faster_whisper_backend.media import media_store as _ms
        ladder = build_video_ladder(
            info, max_bytes=_ms.max_retainable_bytes(), extractor=_xk,
            approx_ratio=lambda fam: _rates.lookup(RATIO_STAGE, _xk, fam).get("rate"))
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
    from faster_whisper_backend.media import segmented as _seg
    from faster_whisper_backend.media import subtitles as _subs
    tracks, sources = _subs.list_tracks(info)
    return UrlMediaInfo(
        url=url,
        extractor_key=str(info.get("extractor_key") or key),
        title=info.get("title"),
        duration=_finite_duration(info.get("duration")),
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
        language=_subs.language_of(info),
        subtitle_tracks=tracks,
        subtitle_sources=sources,
        segmented=_seg.source_of(info),
    )


def _finite_duration(value) -> "float | None":
    """A usable media length or None: a broken site payload can carry NaN
    (float_or_none("NaN")), which would seek the language-check pieces to
    nan and land "at": nan in a JSON answer."""
    if value is None:
        return None
    d = float(value)
    return d if math.isfinite(d) and d > 0 else None


def host_for_log(url: "str | None") -> str:
    """Hostname for log lines — never the full URL (it can carry tokens)."""
    try:
        return log_safe(urllib.parse.urlsplit((url or "").strip()).hostname or "?")
    except Exception:  # noqa: BLE001 — logging must never raise
        return "?"


def _log_probe_failure(url: str, e: Exception) -> None:
    logger.warning("[url-dl] probe failed for host %s: %s",
                   host_for_log(url), log_safe(str(e)))


def _capped_get(url: str, *, max_bytes: int, timeout: float,
                accept=lambda ctype: True,
                headers: "dict[str, str] | None" = None,
                want_url: bool = False) -> "tuple[str, bytes]":
    """One small GET under the address policy: http(s) only, a forbidden
    host refused before any I/O, the guarded opener (pinned DNS, every
    redirect hop re-checked), the body read in chunks under `max_bytes` and
    a wall-clock deadline (the opener's `timeout` is per socket op, and the
    header phase gets a hard cutoff too: a dribbled status line never trips
    the per-op timeout). Returns (content type, body). Sync — callers await
    capped_get, which runs it on _PROBE_POOL. `headers` (the extractor's,
    e.g. a site's Referer) go on top of our User-Agent. `want_url` adds the
    URL that answered (after redirects, each hop already policy-checked) as
    a third item — what an HLS playlist's relative URIs resolve against.
    Raises
    UrlDownloadError (client-safe) for a refused host, an unaccepted content
    type, an over-cap or over-time body; transport errors
    (urllib.error.HTTPError included) propagate for the caller."""
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme.lower() not in ("http", "https") or not parts.hostname
            or _host_is_forbidden(parts.hostname)):
        raise UrlDownloadError("the site could not be reached from the server")
    req = urllib.request.Request(
        url, headers={"User-Agent": "faster-whisper-backend", **(headers or {})})
    with _WallClockCutoff(timeout) as cutoff, \
            _guarded_opener(cutoff).open(req, timeout=timeout) as resp:
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not accept(ctype):
            raise UrlDownloadError("the site answered with an unexpected file type")
        # Chunked against a monotonic deadline: a host dribbling bytes under
        # the per-op timeout could otherwise hold this worker thread forever
        # (the outer wait_for abandons the await, never the thread).
        t0 = time.monotonic()
        buf = bytearray()
        while True:
            chunk = resp.read(32768)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise UrlTooLargeError("the file is over the server's size limit")
            if time.monotonic() - t0 > timeout:
                raise UrlTimeoutError("the site took too long to answer")
        if cutoff.fired:
            raise UrlTimeoutError("the site took too long to answer")
        length = resp.headers.get("Content-Length")
        if (length and length.isdigit() and not getattr(resp, "chunked", False)
                and len(buf) < int(length)):
            raise UrlDownloadError("the site sent an incomplete file")
        final = (resp.geturl() or url) if want_url else url
    return (ctype, bytes(buf), final) if want_url else (ctype, bytes(buf))


async def capped_get(url: str, *, max_bytes: int, deadline: float,
                     accept=lambda ctype: True,
                     headers: "dict[str, str] | None" = None,
                     want_url: bool = False) -> "tuple[str, bytes]":
    """_capped_get on _PROBE_POOL under a monotonic `deadline` shared by a
    whole request: this GET gets the socket timeout or what is left of the
    deadline, whichever is shorter (UrlTimeoutError when nothing is left),
    and the await is abandoned 2 s past it."""
    timeout = min(float(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15)),
                  deadline - time.monotonic())
    if timeout <= 0:
        raise UrlTimeoutError("the site took too long to answer")
    return await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(
        _PROBE_POOL, lambda: _capped_get(
            url, max_bytes=max_bytes, timeout=timeout, accept=accept,
            headers=headers, want_url=want_url)), timeout + 2.0)


async def fetch_thumbnail_data_uri(
    url: "str | None", *, max_bytes: int = 512_000, timeout: float = 5.0,
) -> "str | None":
    """Fetch a thumbnail server-side and return it as a data: URI (the client
    CSP forbids remote images, and fetching client-side would leak the
    client's IP to the media site). Soft-fails to None — a preview without a
    thumbnail is still a preview."""
    if not url:
        return None
    try:
        ctype, body = await capped_get(
            url, max_bytes=max_bytes, deadline=time.monotonic() + timeout,
            accept=lambda c: c.startswith("image/") and "svg" not in c)
    except Exception:  # noqa: BLE001 — soft-fail by contract
        return None
    if not body:
        return None
    return f"data:{ctype};base64,{base64.b64encode(body).decode('ascii')}"


# ── error taxonomy ──────────────────────────────────────────────────────────
# (substring-of-tool-output, client-safe message). Order matters: first hit
# wins, and the more specific conditions sit above the catch-alls.
_HANDOFF_REFUSED = "this link leads to a site the server's URL policy doesn't allow"
_ERROR_TAXONOMY: "tuple[tuple[tuple[str, ...], str], ...]" = (
    # The two guard refusals with their own needle, ABOVE the bare marker
    # they contain: neither is the site being unreachable.
    ((GUARD_UNAVAILABLE_MARKER,),
     "link downloads are unavailable on this server"),
    ((GUARD_EXTERNAL_FD_MARKER,),
     "this media needs a downloader the server does not allow"),
    # Next (first of the non-guard rows, and only after the two longer guard
    # needles above, which contain it), by an exact marker: the SSRF guard
    # refused a hop. Its text
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
    # The extractor pin (pinned_extractors) at work: the URL's extractor
    # handed off to one the policy does not load. Without a pin GenericIE
    # matches every URL, so this text only ever means a policy refusal.
    (("no suitable extractor",), _HANDOFF_REFUSED),
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


def classified_error(stderr_tail: str) -> UrlDownloadError:
    """classify_error's message as the exception to raise: a UrlPolicyError
    (the failures card's policy_blocked class) when the text is a policy
    refusal, a plain UrlDownloadError otherwise."""
    message = classify_error(stderr_tail)
    if message == _HANDOFF_REFUSED:
        return UrlPolicyError(message)
    return UrlDownloadError(message)


# ── download ────────────────────────────────────────────────────────────────

# NOTE: in --progress-template, "download:" is the PHASE SELECTOR (consumed
# by yt-dlp, never printed) — the emitted line starts with our own "dl:"
# marker so progress lines are unambiguous against yt-dlp's [info] chatter.
_PROGRESS_PREFIX = "dl:"
_STDERR_TAIL_MAX = 4096
# The stdout line yt-dlp prints for a result --match-filters rejected.
_FILTERED_NEEDLE = b"does not pass filter"


def _parse_progress_fields(line: str) -> "tuple[int, int | None, str | None] | None":
    """Parse one --progress-template line into (downloaded, total,
    format_id). Total falls back to the estimate; unknown fields arrive as
    'NA' (yt-dlp quirk: never empty strings). The id is the fourth field
    the video template adds, so a merge's two legs can be told apart; None
    on the audio template."""
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
    fid = fields[3] if len(fields) > 3 and fields[3] != "NA" else None
    return downloaded, total, fid


def build_download_argv(url: str, *, dest_dir: str, max_bytes: int,
                        extractors: "list[str] | None" = None,
                        match_filters: "list[str] | None" = None,
                        ) -> "list[str]":
    """The exact yt-dlp CLI invocation (separate function so tests can pin
    it). The URL is the only client-supplied element and follows '--'.
    `extractors` is pinned_extractors()' answer for the URL,
    `match_filters` handoff_match_filters()'."""
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
        # No yt-dlp config file (system, user, portable) may add options —
        # a `--downloader` there would hand fetches to an external program.
        "--ignore-config",
        # A format of protocol "m3u8" (not m3u8_native — Twitter Spaces,
        # twitcasting…) defaults to FFmpegFD, which the guard refuses: send
        # it to the native HlsFD, which fetches through the guarded handler.
        "--downloader", "m3u8:native",
        *_use_extractors(extractors),
        *_match_filter_args(match_filters),
        "-f", DOWNLOAD_FORMAT,
        "--no-playlist",
        "--playlist-items", "1",  # belt+braces: never more than one item
        "--restrict-filenames",
        "--max-filesize", str(max_bytes),
        "--socket-timeout", str(int(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15))),
        "--retries", "3",
        "--no-mtime",
        "--ffmpeg-location", audio_ffmpeg.ffmpeg_exe(),
        "-P", dest_dir,
        # NEVER %(title)s — titles are attacker-controlled and path-adjacent.
        "-o", "media.%(ext)s",
        "--newline", "--no-colors",
        "--progress-template",
        ("download:dl:%(progress.downloaded_bytes)s "
         "%(progress.total_bytes)s %(progress.total_bytes_estimate)s"),
        "--", url,
    ]


def _use_extractors(extractors: "list[str] | None") -> "list[str]":
    return ["--use-extractors", ",".join(extractors)] if extractors else []


def _match_filter_args(filters: "list[str] | None") -> "list[str]":
    # Repeated --match-filters are OR'd by yt-dlp.
    return [a for f in (filters or []) for a in ("--match-filters", f)]


async def _extractor_policy_for(url: str) -> "tuple[list[str] | None, list[str] | None]":
    """(pinned_extractors(), handoff_match_filters()) for `url`, matched
    offline again (the subprocess re-extracts from the URL), on the probe
    pool."""
    def _both():
        key = match_extractor(url)
        return pinned_extractors(key), handoff_match_filters(key)
    return await asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _both)


def video_format_selector(max_height: "int | None" = None,
                          format_ids: "tuple[str | None, str | None] | None" = None) -> str:
    """The `-f` string: the rung's exact ids first (so the file IS the rung
    the card priced), then the generic best-under-cap selector as the
    fallback for a site whose format list moved between probe and run."""
    if max_height is not None:
        h = max(_VIDEO_MIN_HEIGHT, min(_VIDEO_MAX_HEIGHT, int(max_height)))
        generic = VIDEO_FORMAT_CAPPED.format(h=h)
    else:
        generic = VIDEO_FORMAT_BEST
    if not format_ids:
        return generic
    vid, aud = format_ids
    if not (isinstance(vid, str) and FORMAT_ID_RE.match(vid)):
        return generic
    if aud is not None and not (isinstance(aud, str) and FORMAT_ID_RE.match(aud)):
        return generic
    exact = f"{vid}+{aud}" if aud else vid
    return f"{exact}/{generic}"


def build_video_download_argv(url: str, *, dest_dir: str, max_bytes: int,
                              max_height: "int | None" = None,
                              container: str = "mkv",
                              format_ids: "tuple[str | None, str | None] | None" = None,
                              extractors: "list[str] | None" = None,
                              match_filters: "list[str] | None" = None,
                              ) -> "list[str]":
    """The yt-dlp invocation for the VIDEO of a link: the rung's exact
    formats (best video + best audio as the fallback) merged by ffmpeg into
    `container`. Same launcher, guard, caps and output rules as
    build_download_argv; `max_height` is clamped to an int and the ids are
    regex-checked here so no client value ever reaches the selector as
    text."""
    if container not in VIDEO_CONTAINERS:
        container = "mkv"
    fmt = video_format_selector(max_height, format_ids)
    return [
        sys.executable, GUARD_LAUNCHER,
        "--no-plugin-dirs", "--plugin-dirs", GUARD_DIR,
        "--ignore-config",
        "--downloader", "m3u8:native",   # see build_download_argv
        *_use_extractors(extractors),
        *_match_filter_args(match_filters),
        "-f", fmt,
        "--merge-output-format", container,
        "--no-playlist",
        "--playlist-items", "1",
        "--restrict-filenames",
        "--max-filesize", str(max_bytes),
        "--socket-timeout", str(int(getattr(cfg, "URL_SOCKET_TIMEOUT_S", 15))),
        "--retries", "3",
        "--no-mtime",
        "--ffmpeg-location", audio_ffmpeg.ffmpeg_exe(),
        "-P", dest_dir,
        "-o", "media.%(ext)s",   # NEVER %(title)s — see build_download_argv
        "--newline", "--no-colors",
        "--progress-template",
        ("download:dl:%(progress.downloaded_bytes)s "
         "%(progress.total_bytes)s %(progress.total_bytes_estimate)s "
         "%(info.format_id)s"),
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
    extractors, match_filters = await _extractor_policy_for(url)
    argv = build_download_argv(url, dest_dir=dest_dir, max_bytes=max_bytes,
                               extractors=extractors,
                               match_filters=match_filters)

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
    format_ids: "tuple[str | None, str | None] | None" = None,
    leg_estimates: "dict[str, int] | None" = None,
    timeout: "float | None" = None,
    progress_cb=None,
    cancel_check=None,
) -> str:
    """Download the VIDEO of `url` (the rung's exact formats, best video +
    best audio as the fallback, merged into `container`) into `dest_dir`
    and return the file path.

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
    extractors, match_filters = await _extractor_policy_for(url)
    argv = build_video_download_argv(
        url, dest_dir=dest_dir, max_bytes=max_bytes, max_height=max_height,
        container=container, format_ids=format_ids,
        extractors=extractors, match_filters=match_filters)
    logger.info("[url-dl] video download starting (host %s): -f %s, expected %s",
                host_for_log(url),
                video_format_selector(max_height, format_ids).split("/")[0],
                f"{expected_total / 1e6:.1f} MB" if expected_total else "?")

    def _emit(downloaded: int, total: "int | None") -> None:
        if progress_cb is None:
            return
        frac = (max(0.0, min(1.0, downloaded / total)) if total else None)
        progress_cb(frac, total, downloaded)

    return await _run_yt_dlp(
        argv, url=url, dest_dir=dest_dir, max_bytes=max_bytes,
        timeout=timeout, emit=_emit, cancel_check=cancel_check,
        expected_total=expected_total, leg_estimates=leg_estimates,
        cumulative=True,
        expected_legs=len([i for i in (format_ids or ()) if i]) or None,
        find_result=lambda d: _find_video_result(d, container))


# At most one progress callback per this many seconds (the terminal line is
# flushed after EOF regardless). A name so tests can drop it to 0.
_PROGRESS_EMIT_MIN_S = 0.3


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
    leg_estimates: "dict[str, int] | None" = None,
    cumulative: bool = False,
    expected_legs: "int | None" = None,
) -> str:
    """The subprocess half shared by download() and download_video(): run
    yt-dlp, stream its progress lines into `emit(downloaded, total)` (one
    call per 0.3 s; with `cumulative`, summed across the files one
    invocation fetches — the audio selector never merges, so it is off there),
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

    async def _kill() -> None:
        await core_proc.terminate_then_kill(proc)

    # yt-dlp prints one `dl:` series per file it fetches (a merge = video
    # then audio, each restarting at 0). `completed` carries the finished
    # files' bytes so the fraction, the cap and the log all see the sum.
    # The denominator is per leg: a leg's own total once yt-dlp reports
    # it, the probe's estimate for that leg before that, and the merged
    # `expected_total` when no leg is known by id.
    completed = 0
    last_downloaded = 0
    cur_leg: "str | None" = None
    leg_total: "dict[str, int]" = {}       # id → best-known total
    leg_done: "dict[str, int]" = {}        # id → bytes when its series ended
    estimates = dict(leg_estimates or {})

    def _denominator(cum: int, cur_total: "int | None") -> "int | None":
        if not estimates and not leg_total:
            if expected_total:
                return max(int(expected_total), cum)
            return (completed + cur_total) if cur_total else None
        ids = set(estimates) | set(leg_total) | set(leg_done)
        total = 0
        for fid in ids:
            if fid in leg_done:
                total += leg_done[fid]
            elif fid in leg_total:
                total += leg_total[fid]
            elif fid in estimates:
                total += int(estimates[fid])
        # A leg yt-dlp named that the probe never priced: fall back to the
        # merged estimate rather than a denominator that is too small.
        if cur_leg is not None and cur_leg not in ids and expected_total:
            den = max(int(expected_total), cum)
        else:
            den = max(total, cum) if total else (
                max(int(expected_total), cum) if expected_total else None)
        # A declared leg not seen yet, or the current one still unpriced:
        # more bytes are coming, so the fraction must not read 100 % yet.
        seen = set(leg_done) | ({cur_leg} if cur_leg is not None else set())
        pending = (expected_legs is not None and len(seen) < expected_legs) or (
            cur_leg is not None and cur_leg not in leg_total
            and cur_leg not in estimates)
        if pending and den is not None and den <= cum:
            den = cum + 1
        return den

    def _cumulative(parsed: "tuple[int, int | None, str | None]") -> "tuple[int, int | None]":
        nonlocal completed, last_downloaded, cur_leg
        downloaded, total, fid = parsed
        # A named leg changes series when its id does — a counter that
        # drops INSIDE one leg is a retry that did not resume, not a new
        # file. Without ids the drop is the only signal, and only a merge
        # (cumulative) fetches more than one file.
        if fid is not None:
            new_series = cur_leg is not None and fid != cur_leg
        else:
            new_series = cumulative and downloaded < last_downloaded
        if new_series:
            completed += last_downloaded
            if cur_leg is not None:
                leg_done[cur_leg] = last_downloaded
        if fid is not None:
            cur_leg = fid
            if total:
                leg_total[fid] = int(total)
        last_downloaded = downloaded
        cum = completed + downloaded
        return cum, _denominator(cum, total)

    def _emit(parsed: "tuple[int, int | None]") -> None:
        try:
            emit(*parsed)
        except Exception:  # noqa: BLE001 — progress must not kill the run
            pass

    stderr_task = asyncio.create_task(_drain_stderr())
    last_cb = 0.0
    last_parsed: "tuple[int, int | None] | None" = None
    filtered_out = False
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
            if _FILTERED_NEEDLE in raw:
                filtered_out = True
            fields = _parse_progress_fields(raw.decode("utf-8", "replace").strip())
            parsed = _cumulative(fields) if fields is not None else None
            # Belt and braces over --max-filesize, which only fires when the
            # size is known up front (and per FILE — a merge's two streams
            # can pass it separately): a chunked / fragmented response with
            # no Content-Length would otherwise be written in full (until
            # the wall-clock timeout) before the post-hoc size check below.
            if parsed is not None and parsed[0] > max_bytes:
                await _kill()
                # Off the loop: the refused partial can be gigabytes.
                await asyncio.to_thread(_discard_partials, dest_dir)
                raise UrlPolicyError("this media exceeds the server's size limit")
            if parsed:
                last_parsed = parsed
                now = time.monotonic()
                if now - last_cb >= _PROGRESS_EMIT_MIN_S:
                    last_cb = now
                    last_emitted = parsed
                    _emit(parsed)
        # Flush the terminal line the _PROGRESS_EMIT_MIN_S throttle swallowed (yt-dlp emits
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
                       log_safe(tail, keep_end=True))
        raise classified_error(tail)

    result = find_result(dest_dir)
    if result is None and filtered_out:
        # handoff_match_filters at work: the run re-extracted to a result
        # the site policy does not admit (exit 0, nothing fetched).
        raise UrlPolicyError(_HANDOFF_REFUSED)
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
# carry a second one and never qualify. The extension is whatever yt-dlp
# took from the URL — its own case (`media.MP3`), or the literal
# `unknown_video` when it could not name one — so the class is the whole
# alphabet yt-dlp can produce, not "lowercase, five chars".
_RESULT_NAME_RE = re.compile(r"\Amedia\.[A-Za-z0-9_]{1,16}\Z")
_INTERMEDIATE_NAME_RE = re.compile(r"\Amedia\.f[^.]+\.[A-Za-z0-9_]+\Z")


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
