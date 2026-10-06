"""A link's own subtitle tracks, from the yt-dlp info dict.

`list_tracks(info)` (pure) turns `subtitles` / `automatic_captions` into the short
list a preview shows plus the server-side sources the url-subtitles route
fetches from. Only ids leave the server: the source URLs are signed,
short-lived and on YouTube carry a `pot` token, so the client names a track
by id and the route re-probes for fresh URLs; `fetch_tracks` then GETs
them through the guarded, capped download.capped_get.

Rules:
  - every manual track (a person uploaded it);
  - of the site's speech-recognition tracks only the original language —
    YouTube lists `<lang>-orig` beside ~158 machine translations of it, and
    a machine translation of a machine transcript is never what anyone
    wants; without an `-orig` key, the auto track in `info.language`;
  - one format per track, vtt over srt; never an HLS playlist (SRF lists a
    248-byte m3u8 stub beside the real vtt — fetching its segments would
    be a fan-out of server requests the caps never saw);
  - at most MAX_TRACKS, ids `^[A-Za-z0-9_-]{1,32}$` minted from kind + the
    site's language key so the same link yields the same ids on every probe;
  - `lang` (and `language_of`) in the server's own spelling (`_code`), so
    the code a track carries is one the packaging route accepts.
"""
from __future__ import annotations

import asyncio
import re
import time
import urllib.error
import urllib.parse

from faster_whisper_backend.core.languages import canonical_code
from faster_whisper_backend.url import download as _udl

MAX_TRACKS = 24
_FORMATS = ("vtt", "srt")  # preference order
_LANG_RE = re.compile(r"\A[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,2}\Z")
TRACK_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,32}\Z")
_HOH_RE = re.compile(r"sdh|hoh|hearing|hörgeschädigt", re.I)
_NAME_MAX = 64


def _ext_of(f: dict) -> "str | None":
    """A format entry's subtitle type: its `ext`, else the URL's suffix."""
    ext = f.get("ext")
    if not ext:
        path = urllib.parse.urlsplit(str(f.get("url") or "")).path
        ext = path.rsplit(".", 1)[-1] if "." in path else None
    return str(ext).lower() if ext else None


def _is_playlist(f: dict) -> bool:
    url_path = urllib.parse.urlsplit(str(f.get("url") or "")).path.lower()
    return "m3u8" in str(f.get("protocol") or "") or url_path.endswith(".m3u8")


def _best_format(formats) -> "dict | None":
    """The vtt (else srt) entry of a track, skipping HLS playlists."""
    usable = [f for f in formats or () if isinstance(f, dict)
              and str(f.get("url") or "").startswith(("https://", "http://"))
              and not _is_playlist(f) and _ext_of(f) in _FORMATS]
    usable.sort(key=lambda f: _FORMATS.index(_ext_of(f)))
    return usable[0] if usable else None


def _code(key: str) -> str:
    """A site's language key (already _LANG_RE-shaped) as the server spells
    codes: canonical_code where the language table names it ("EN" → "en",
    "deu" → "de", "pt-br" → "pt-BR", "zh-Hans" → "zh"), else the lowercase
    base ("rm", "fil", "sr-Latn-RS" → "sr")."""
    return canonical_code(key) or key.split("-")[0].lower()


def language_of(info: dict) -> "str | None":
    """The spoken language the site names (YouTube's `language`), else the
    language of its one original-language auto track; None when unknown."""
    lang = info.get("language")
    if isinstance(lang, str) and _LANG_RE.match(lang):
        return _code(lang)
    orig = [k[:-5] for k in (info.get("automatic_captions") or {})
            if isinstance(k, str) and k.endswith("-orig")]
    return _code(orig[0]) if len(orig) == 1 and _LANG_RE.match(orig[0]) else None


def list_tracks(info: dict) -> "tuple[list[dict], dict[str, dict]]":
    """(public tracks, {id: {url, ext}}) — see the module docstring."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    picked = [(lang, "manual", fmts) for lang, fmts in manual.items()]
    orig = [(key[:-5], "auto", fmts) for key, fmts in auto.items()
            if isinstance(key, str) and key.endswith("-orig")]
    if not orig and isinstance(info.get("language"), str):
        orig = [(key, "auto", fmts) for key, fmts in auto.items()
                if key == info["language"]]
    public: "list[dict]" = []
    sources: "dict[str, dict]" = {}
    for lang, kind, fmts in picked + orig:
        if len(public) >= MAX_TRACKS:
            break
        if not isinstance(lang, str) or not _LANG_RE.match(lang):
            continue  # "live_chat", odd keys
        f = _best_format(fmts)
        tid = f"{kind[0]}-{lang}"
        if f is None or tid in sources or not TRACK_ID_RE.match(tid):
            continue
        name = str(f.get("name") or "")[:_NAME_MAX] or None
        public.append({
            "id": tid, "lang": _code(lang), "name": name, "kind": kind,
            "ext": _ext_of(f),
            "hoh": bool(_HOH_RE.search(f"{lang} {name or ''}")),
        })
        sources[tid] = {"url": str(f["url"]), "ext": _ext_of(f)}
    return public, sources


# ── fetching ────────────────────────────────────────────────────────────────
MAX_FETCH = 8                      # tracks per request
TRACK_MAX_BYTES = 2 * 1024 * 1024  # one track
TOTAL_MAX_BYTES = 8 * 1024 * 1024  # one request
_FETCH_BUDGET_S = 60.0             # all tracks of one request, wall clock


def sniff(body: bytes, ext: str) -> str:
    """The body as subtitle text, or UrlDownloadError: VTT must open with
    WEBVTT, SRT must carry a cue arrow, and an HLS playlist is refused
    whatever the site called it (fetching its segments would be a fan-out
    of requests no cap ever saw)."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = body.decode("cp1252", errors="replace")  # old SRT files
    if text.lstrip().startswith("#EXTM3U"):
        raise _udl.UrlDownloadError("the site sent a playlist, not subtitles")
    if (ext == "vtt" and not text.startswith("WEBVTT")) or (
            ext == "srt" and "-->" not in text):
        raise _udl.UrlDownloadError("the site sent no readable subtitles")
    return text


async def fetch_tracks(info, ids: "list[str]") -> "tuple[list[dict], list[dict]]":
    """Fetch the tracks `ids` names from a FRESH probe's `info` (its signed
    URLs are minutes old at most): ({id, lang, kind, ext, text} …,
    {id, error} …). One at a time through download.capped_get — guarded
    opener, TRACK_MAX_BYTES each, TOTAL_MAX_BYTES and _FETCH_BUDGET_S over
    the request. Errors are client-safe; a source URL never reaches one."""
    by_id = {t["id"]: t for t in info.subtitle_tracks}
    deadline = time.monotonic() + _FETCH_BUDGET_S
    budget = TOTAL_MAX_BYTES
    tracks: "list[dict]" = []
    failed: "list[dict]" = []
    for tid in ids:
        track, src = by_id.get(tid), info.subtitle_sources.get(tid)
        try:
            if track is None or src is None:
                raise _udl.UrlDownloadError("the link no longer offers this track")
            _ctype, body = await _udl.capped_get(
                src["url"], max_bytes=min(TRACK_MAX_BYTES, budget),
                deadline=deadline)
            budget -= len(body)
            tracks.append({"id": tid, "lang": track["lang"], "kind": track["kind"],
                           "ext": track["ext"], "text": sniff(body, track["ext"])})
        except urllib.error.HTTPError as e:
            failed.append({"id": tid, "error": (
                "the site is rate-limiting subtitle downloads" if e.code == 429
                else f"the site refused the subtitle download (HTTP {e.code})")})
        except _udl.UrlDownloadError as e:
            failed.append({"id": tid, "error": str(e)})
        except asyncio.TimeoutError:
            failed.append({"id": tid, "error": "the site took too long to answer"})
        except Exception:  # noqa: BLE001 — transport noise; never echo it
            failed.append({"id": tid, "error": "the subtitles could not be fetched"})
    return tracks, failed
