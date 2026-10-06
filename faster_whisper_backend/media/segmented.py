"""A link's language-check pieces without the whole file.

The language check (POST /v1/audio/url-language) listens to three 20 s
pieces. When the audio format yt-dlp picked (DOWNLOAD_FORMAT) is ONE file
(progressive http/https), the route downloads it whole — fast, and the run
that follows reuses it. When the site offers the audio only as a segmented
stream (HLS, DASH), the whole file means hundreds of small requests (RTVE:
203 s for a 46-minute programme), so this module fetches just the segments
under the pieces instead.

Every request — playlist and segment, on whatever host (CDNs) — goes through
download.capped_get: http(s) only, forbidden hosts refused before any I/O,
the guarded opener (pinned DNS, every redirect hop re-checked), a byte cap
and a wall-clock deadline. Never yt-dlp's download_ranges/--download-sections:
those hand the fetch to ffmpeg, which bypasses the SSRF guard.

Whatever this cannot handle faithfully raises Unsupported (or any other
error) and the route falls back to the full download — the new path never
fails a check on its own. The parsers (`parse_hls`, `dash_media`,
`source_of`) are pure.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import time
import urllib.parse

from faster_whisper_backend.media import download as _udl
from faster_whisper_backend.media.language_check import select_segments

# Caps for one check. 2 s DASH fragments (RTVE) need 12 to cover a 20 s
# piece plus margin; shorter segments than that fall back rather than fan
# out into dozens of requests per piece.
MAX_SEGMENTS_PER_PIECE = 16
SEGMENT_MAX_BYTES = 2 * 1024 * 1024
TOTAL_MAX_BYTES = 24 * 1024 * 1024     # all segments (+ init) of one check
PLAYLIST_MAX_BYTES = 2 * 1024 * 1024   # a 4 h playlist of 2 s segments ≈ 1 MB
BUDGET_S = 60.0                        # every request of one check, wall clock
MARGIN_S = 1.0                         # around each piece
# The extractor's request headers we pass on (some CDNs want the page as
# Referer); never its cookies.
_PASS_HEADERS = ("User-Agent", "Referer", "Origin")
_PIECE_EXTS = ("ts", "aac", "mp3", "m4a", "mp4")


class Unsupported(ValueError):
    """A stream the chunked fetch does not handle; str() is our own wording
    (log-safe: never a URL)."""


# Where a segment's bytes are: an absolute URL, and (offset, length) when
# the segment is a byte range of it (CMAF: one file, ranged requests).
Part = "tuple[str, tuple[int, int] | None]"


@dataclasses.dataclass
class Media:
    """A segmented stream: (part, seconds) per segment, in order, and the
    init segment (fMP4) every run of segments must start with."""
    segments: "list[tuple[Part, float]]"
    init: "Part | None" = None


def _duration(value) -> float:
    try:
        d = float(value)
    except (TypeError, ValueError):
        raise Unsupported("a segment without a usable duration") from None
    if not (d > 0 and math.isfinite(d)):
        raise Unsupported("a segment without a usable duration")
    return d


def source_of(info: dict) -> "dict | None":
    """The selected audio format as {protocol: hls|dash, url, fragments,
    base, headers} when it is a segmented stream, else None (a progressive
    file, or nothing usable: the full download handles those). Looks at the
    audio entry of `requested_formats` when the selection merged formats."""
    merged = info.get("requested_formats")
    if merged:
        audio = [f for f in merged if isinstance(f, dict)
                 and f.get("vcodec") == "none" and f.get("acodec") != "none"]
        if not audio:
            return None
        info = audio[0]
    protocol = str(info.get("protocol") or "")
    if protocol in ("m3u8", "m3u8_native"):
        kind = "hls"
    elif protocol == "http_dash_segments":
        kind = "dash"
    else:
        return None
    raw = info.get("http_headers") or {}
    headers = {k: v for k in _PASS_HEADERS
               if isinstance(v := raw.get(k), str) and v and "\r" not in v
               and "\n" not in v}
    return {"protocol": kind, "url": info.get("url"),
            "fragments": info.get("fragments"),
            "base": info.get("fragment_base_url"), "headers": headers}


_ATTR_RE = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')


def _attrs(line: str) -> "dict[str, str]":
    return {k: v.strip('"') for k, v in _ATTR_RE.findall(line.split(":", 1)[1])}


def parse_hls(text: str, base_url: str) -> "Media | str":
    """An HLS media playlist → Media; a master playlist → the URL to read
    next (its DEFAULT audio rendition, else its first audio rendition, else
    the leanest variant — audio muxed with the least video). Segment times
    are the running sum of #EXTINF (EXT-X-MEDIA-SEQUENCE only numbers them).
    Byte ranges (EXT-X-BYTERANGE, EXT-X-MAP BYTERANGE) become ranged parts.
    Unsupported: encryption, a playlist without EXT-X-ENDLIST (live or
    still growing), an init segment that changes mid-stream."""
    lines = [ln.strip() for ln in text.lstrip("\ufeff").splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("#EXTM3U"):
        raise Unsupported("not an HLS playlist")
    renditions: "list[tuple[bool, str]]" = []
    variants: "list[tuple[int, str]]" = []
    segments: "list[tuple[Part, float]]" = []
    init: "Part | None" = None
    extinf: "float | None" = None
    byterange: "tuple[int, int | None] | None" = None  # (length, offset)
    bandwidth: "int | None" = None
    ended = False
    for line in lines[1:]:
        tag = line.split(":", 1)[0]
        if tag in ("#EXT-X-KEY", "#EXT-X-SESSION-KEY"):
            if _attrs(line).get("METHOD", "NONE") != "NONE":
                raise Unsupported("the stream is encrypted")
        elif tag == "#EXT-X-BYTERANGE":
            byterange = _byterange(line.split(":", 1)[1])
        elif tag == "#EXT-X-MAP":
            a = _attrs(line)
            url = urllib.parse.urljoin(base_url, a.get("URI", ""))
            n, o = _byterange(a["BYTERANGE"]) if "BYTERANGE" in a else (0, None)
            part = (url, (o or 0, n) if n else None)
            if (init is not None and init != part) or (init is None and segments):
                raise Unsupported("the stream changes its init segment")
            init = part
        elif tag == "#EXTINF":
            extinf = _duration(line.split(":", 1)[1].split(",", 1)[0])
        elif tag == "#EXT-X-STREAM-INF":
            bw = _attrs(line).get("BANDWIDTH", "")
            bandwidth = int(bw) if bw.isdigit() else 0
        elif tag == "#EXT-X-MEDIA":
            a = _attrs(line)
            if a.get("TYPE") == "AUDIO" and a.get("URI"):
                renditions.append((a.get("DEFAULT") == "YES",
                                   urllib.parse.urljoin(base_url, a["URI"])))
        elif tag == "#EXT-X-ENDLIST":
            ended = True
        elif not line.startswith("#"):
            url = urllib.parse.urljoin(base_url, line)
            if bandwidth is not None:
                variants.append((bandwidth, url))
                bandwidth = None
            elif extinf is not None:
                rng = None
                if byterange is not None:
                    n, o = byterange
                    if o is None:  # continues the previous range of this file
                        prev = segments[-1][0] if segments else (None, None)
                        if prev[0] != url or prev[1] is None:
                            raise Unsupported("a byte range without an offset")
                        o = sum(prev[1])
                    rng = (o, n)
                segments.append(((url, rng), extinf))
                extinf, byterange = None, None
            else:
                raise Unsupported("a segment without a duration")
    if renditions:
        return next((u for default, u in renditions if default), renditions[0][1])
    if variants:
        return min(variants)[1]
    if not ended:
        raise Unsupported("the stream is live or still growing")
    if not segments:
        raise Unsupported("the playlist lists no segments")
    return Media(segments, init)


def _byterange(value: str) -> "tuple[int, int | None]":
    """'<length>[@<offset>]' → (length, offset or None)."""
    n, _, o = value.strip().strip('"').partition("@")
    if not n.isdigit() or int(n) == 0 or (o and not o.isdigit()):
        raise Unsupported("a malformed byte range")
    return int(n), (int(o) if o else None)


def dash_media(fragments, base_url: "str | None") -> Media:
    """yt-dlp's DASH fragment list → Media: a leading fragment without a
    duration is the init segment; every other one needs a duration."""
    init: "Part | None" = None
    segments: "list[tuple[Part, float]]" = []
    for i, frag in enumerate(fragments or ()):
        if not isinstance(frag, dict) or "range" in frag or "byte_range" in frag:
            raise Unsupported("a fragment this server does not fetch")
        url = frag.get("url") or (
            urllib.parse.urljoin(base_url, frag["path"])
            if base_url and frag.get("path") else None)
        if not isinstance(url, str) or not url:
            raise Unsupported("a fragment without an address")
        if i == 0 and frag.get("duration") is None:
            init = (url, None)
            continue
        segments.append(((url, None), _duration(frag.get("duration"))))
    if not segments:
        raise Unsupported("the stream lists no fragments")
    return Media(segments, init)


async def fetch_pieces(source: dict, starts: "list[float]", seconds: float,
                       dest_dir: str, *, cancel_check,
                       progress_cb=None) -> "tuple[list[tuple[str, float]], dict]":
    """Fetch the segments under each piece into one file per piece in
    `dest_dir` (init + segments, concatenated): ([(path, skip seconds)],
    {segments, bytes}). Raises Unsupported / UrlDownloadError (→ the caller
    falls back), UrlCancelled when `cancel_check()` trips."""
    deadline = time.monotonic() + BUDGET_S
    headers = source.get("headers") or {}

    async def get(part: Part, cap: int) -> bytes:
        url, rng = part
        if cancel_check():
            raise _udl.UrlCancelled()
        hdrs = headers if rng is None else {
            **headers, "Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"}
        _ctype, body = await _udl.capped_get(
            url, max_bytes=cap, deadline=deadline, headers=hdrs)
        if rng is not None and len(body) != rng[1]:
            # A server ignoring Range sends the whole file (a short one fits
            # the cap): its bytes would be the wrong part of the stream.
            raise Unsupported("the site ignored a byte range")
        return body

    if source.get("protocol") == "dash":
        media = dash_media(source.get("fragments"), source.get("base"))
    else:
        media, url = None, source.get("url")
        for _hop in range(2):  # a master playlist, then its media playlist
            if not isinstance(url, str) or not url:
                raise Unsupported("the stream has no playlist address")
            parsed = parse_hls((await get((url, None), PLAYLIST_MAX_BYTES)).decode(
                "utf-8", errors="replace"), url)
            if isinstance(parsed, Media):
                media = parsed
                break
            url = parsed
        if media is None:
            raise Unsupported("a master playlist that points at another")

    runs = select_segments([d for _u, d in media.segments], starts, seconds,
                           MARGIN_S)
    for first, end, _skip in runs:
        if end == first:
            raise Unsupported("a piece lies past the stream's end")
        if end - first > MAX_SEGMENTS_PER_PIECE:
            raise Unsupported("the stream's segments are too short to sample")
    head = [media.init] if media.init else []
    todo = list(dict.fromkeys(head + [
        part for first, end, _s in runs for part, _d in media.segments[first:end]]))
    got: "dict[Part, bytes]" = {}
    spent = 0
    for i, part in enumerate(todo):
        if spent >= TOTAL_MAX_BYTES:
            raise _udl.UrlDownloadError("the file is over the server's size limit")
        got[part] = await get(part, min(SEGMENT_MAX_BYTES, TOTAL_MAX_BYTES - spent))
        spent += len(got[part])
        if progress_cb is not None:
            progress_cb((i + 1) / len(todo))

    ext = os.path.splitext(urllib.parse.urlsplit(media.segments[0][0][0]).path)[1]
    ext = "mp4" if media.init else (ext.lstrip(".").lower() or "ts")
    ext = ext if ext in _PIECE_EXTS else "ts"
    pieces = []
    for n, (first, end, skip) in enumerate(runs):
        path = os.path.join(dest_dir, f"piece{n}.{ext}")
        with open(path, "wb") as f:
            for part in head + [p for p, _d in media.segments[first:end]]:
                f.write(got[part])
        pieces.append((path, skip))
    return pieces, {"segments": len(got) - len(head), "bytes": spent}
