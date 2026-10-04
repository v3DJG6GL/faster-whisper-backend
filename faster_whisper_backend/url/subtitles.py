"""A link's own subtitle tracks, from the yt-dlp info dict (pure).

`list_tracks(info)` turns `subtitles` / `automatic_captions` into the short
list a preview shows plus the server-side sources the url-subtitles route
fetches from. Only ids leave the server: the source URLs are signed,
short-lived and on YouTube carry a `pot` token, so the client names a track
by id and the route re-probes for fresh URLs.

Rules:
  - every manual track (a person uploaded it);
  - of the site's speech-recognition tracks only the original language —
    YouTube lists `<lang>-orig` beside ~158 machine translations of it, and
    a machine translation of a machine transcript is never what anyone
    wants; without an `-orig` key, the auto track in `info.language`;
  - one format per track, vtt over srt; never an HLS playlist (SRF lists a
    248-byte m3u8 stub beside the real vtt — fetching its segments would
    be a fan-out of server requests the caps never saw);
  - at most MAX_TRACKS, ids `^[A-Za-z0-9_-]{1,32}$` minted from kind + lang
    so the same link yields the same ids on every probe.
"""
from __future__ import annotations

import re
import urllib.parse

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


def language_of(info: dict) -> "str | None":
    """The spoken language the site names (YouTube's `language`), else the
    language of its one original-language auto track; None when unknown."""
    lang = info.get("language")
    if isinstance(lang, str) and _LANG_RE.match(lang):
        return lang
    orig = [k[:-5] for k in (info.get("automatic_captions") or {})
            if isinstance(k, str) and k.endswith("-orig")]
    return orig[0] if len(orig) == 1 and _LANG_RE.match(orig[0]) else None


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
            "id": tid, "lang": lang, "name": name, "kind": kind,
            "ext": _ext_of(f),
            "hoh": bool(_HOH_RE.search(f"{lang} {name or ''}")),
        })
        sources[tid] = {"url": str(f["url"]), "ext": _ext_of(f)}
    return public, sources

