"""url/subtitles.py — track listing from realistic yt-dlp info dicts."""

from __future__ import annotations

from faster_whisper_backend.url import subtitles as subs


def _yt_formats(lang, name):
    base = f"https://www.youtube.com/api/timedtext?v=Dz_3b8WAWw4&lang={lang}&pot=TOKEN"
    return [{"ext": e, "url": f"{base}&fmt={e}", "name": name}
            for e in ("json3", "srv1", "srv2", "srv3", "ttml", "srt", "vtt")]


# YouTube Dz_3b8WAWw4: one manual de-CH track; the ASR original de-orig
# among ~158 machine translations; info.language "de".
_YOUTUBE = {
    "language": "de",
    "subtitles": {
        "de-CH": _yt_formats("de-CH", "German (Switzerland)"),
        "live_chat": [{"ext": "json", "url": "https://www.youtube.com/live_chat"}],
    },
    "automatic_captions": {
        "de-orig": _yt_formats("de", "German (Original)"),
        "de": _yt_formats("de", "German"),
        **{lang: _yt_formats(lang, lang) for lang in (
            "en", "fr", "it", "ja", "zh-Hans", "pt", "es")},
    },
}

# SRF Play: one language, a 248-byte HLS stub listed beside the full vtt
# and a TTML-ish xml; no language field.
_SRF = {
    "subtitles": {"de": [
        {"url": "https://subtitles.srf.ch/x/playlist.m3u8", "ext": "vtt",
         "protocol": "m3u8_native"},
        {"url": "https://subtitles.srf.ch/x/full.vtt"},
        {"url": "https://subtitles.srf.ch/x/full.xml", "ext": "xml"},
    ]},
}


def test_youtube_collapses_to_manual_plus_orig():
    public, sources = subs.list_tracks(_YOUTUBE)
    assert public == [
        {"id": "m-de-CH", "lang": "de-CH", "name": "German (Switzerland)",
         "kind": "manual", "ext": "vtt", "hoh": False},
        {"id": "a-de", "lang": "de", "name": "German (Original)",
         "kind": "auto", "ext": "vtt", "hoh": False},
    ]
    assert sources["a-de"]["url"].endswith("&fmt=vtt")
    assert subs.language_of(_YOUTUBE) == "de"


def test_srf_skips_the_hls_stub_and_infers_the_ext():
    public, sources = subs.list_tracks(_SRF)
    assert [(t["id"], t["ext"]) for t in public] == [("m-de", "vtt")]
    assert sources["m-de"]["url"] == "https://subtitles.srf.ch/x/full.vtt"
    assert subs.language_of(_SRF) is None


def test_auto_without_orig_falls_back_to_the_named_language():
    info = {"language": "en", "automatic_captions": {
        "en": _yt_formats("en", "English"), "fr": _yt_formats("fr", "French")}}
    assert [t["id"] for t in subs.list_tracks(info)[0]] == ["a-en"]
    # No language, no -orig: no auto track at all.
    assert subs.list_tracks({"automatic_captions": info["automatic_captions"]}) == ([], {})


def test_language_of_falls_back_to_the_single_orig_track():
    assert subs.language_of({"automatic_captions": {"fr-orig": [], "fr": []}}) == "fr"
    assert subs.language_of({"language": "not a code!"}) is None


def test_srt_when_no_vtt_and_nothing_else():
    info = {"subtitles": {
        "en": [{"url": "https://e.test/a.srt", "ext": "srt"}],
        "fr": [{"url": "https://e.test/a.ttml", "ext": "ttml"}],
        "it": [{"url": "ftp://e.test/a.vtt", "ext": "vtt"}],
        "../x": [{"url": "https://e.test/a.vtt", "ext": "vtt"}],
    }}
    assert [(t["id"], t["ext"]) for t in subs.list_tracks(info)[0]] == [("m-en", "srt")]


def test_hearing_impaired_from_names():
    info = {"subtitles": {
        "de": [{"url": "https://e.test/a.vtt", "name": "Untertitel für Hörgeschädigte"}],
        "en": [{"url": "https://e.test/b.vtt", "name": "English SDH"}],
        "fr": [{"url": "https://e.test/c.vtt", "name": "Français"}],
    }}
    assert [t["hoh"] for t in subs.list_tracks(info)[0]] == [True, True, False]


def test_caps_and_id_shape():
    info = {"subtitles": {f"l{chr(97 + i // 26)}{chr(97 + i % 26)}":
                          [{"url": f"https://e.test/{i}.vtt"}] for i in range(40)}}
    public, sources = subs.list_tracks(info)
    assert len(public) == subs.MAX_TRACKS == len(sources)
    assert all(subs.TRACK_ID_RE.match(t["id"]) for t in public)
    long_lang = "de-" + "-".join(["abcdefgh"] * 2)
    public, _ = subs.list_tracks({"subtitles": {long_lang: [{"url": "https://e.test/a.vtt"}]}})
    assert public and len(public[0]["id"]) <= 32
    assert subs.list_tracks({"subtitles": None, "automatic_captions": None}) == ([], {})


# ── fetching ────────────────────────────────────────────────────────────────

import asyncio  # noqa: E402

import pytest  # noqa: E402

from faster_whisper_backend.url import download as udl  # noqa: E402

_VTT = b"\xef\xbb\xbfWEBVTT\n\n00:00.000 --> 00:01.000\nHallo\n"


@pytest.fixture
def served(fake_capped_get):
    """(info_with, asked): info_with(lang → (ext, body | HTTP status int)) is a
    probe result whose track URLs the fake GET serves; asked records each
    max_bytes."""
    fake_capped_get.ctype = "text/vtt"

    def info_with(bodies: dict):
        fake_capped_get.table.update({
            f"https://subs.test/{lang}.{ext}": body
            for lang, (ext, body) in bodies.items()})
        tracks, sources = subs.list_tracks({"subtitles": {
            lang: [{"url": f"https://subs.test/{lang}.{ext}", "ext": ext}]
            for lang, (ext, _b) in bodies.items()}})
        return udl.UrlMediaInfo(url="https://e.test/v", extractor_key="X",
                                subtitle_tracks=tracks, subtitle_sources=sources)
    return info_with, fake_capped_get.caps


def _fetch(info, ids):
    return asyncio.run(subs.fetch_tracks(info, ids))


def test_sniff():
    assert subs.sniff(_VTT, "vtt").startswith("WEBVTT")
    assert "-->" in subs.sniff("1\n00:00:00,000 --> 00:00:01,000\nGrüezi\n".encode("cp1252"), "srt")
    for body, ext in ((b"#EXTM3U\n#EXT-X-TARGETDURATION:6\n", "vtt"),
                      (b"<tt xmlns='x'/>", "vtt"), (b"just words", "srt")):
        with pytest.raises(udl.UrlDownloadError):
            subs.sniff(body, ext)


def test_fetch_returns_text_and_client_safe_failures(served):
    info_with, _asked = served
    info = info_with({"de": ("vtt", _VTT), "en": ("vtt", 429), "fr": ("vtt", 403),
                      "it": ("vtt", b"#EXTM3U\n")})
    tracks, failed = _fetch(info, ["m-de", "m-en", "m-fr", "m-it", "m-xx"])
    assert tracks == [{"id": "m-de", "lang": "de", "kind": "manual", "ext": "vtt",
                       "text": _VTT.decode("utf-8-sig")}]
    errors = {f["id"]: f["error"] for f in failed}
    assert errors["m-en"] == "the site is rate-limiting subtitle downloads"
    assert "HTTP 403" in errors["m-fr"]
    assert "playlist" in errors["m-it"]
    assert "no longer offers" in errors["m-xx"]
    assert not any("subs.test" in e for e in errors.values())


def test_fetch_caps_each_track_and_the_request(served):
    info_with, asked = served
    big = b"WEBVTT\n" + b"x" * (subs.TRACK_MAX_BYTES - 7)   # exactly at the cap
    info = info_with({lang: ("vtt", big) for lang in ("aa", "ab", "ac", "ad", "ae")})
    tracks, failed = _fetch(info, [f"m-{l}" for l in ("aa", "ab", "ac", "ad", "ae")])
    assert len(tracks) == 4 and [f["id"] for f in failed] == ["m-ae"]
    assert asked == [subs.TRACK_MAX_BYTES] * 4 + [0]
    assert sum(len(t["text"]) for t in tracks) <= subs.TOTAL_MAX_BYTES
