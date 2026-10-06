"""media/segmented.py — the language check's chunked path: the pure HLS / DASH
parsers, the source picker, and fetch_pieces against a fake guarded GET
serving real (PyAV-encoded) TS and fMP4 segments."""

import asyncio
import io
import struct
import urllib.error

import numpy as np
import pytest

from faster_whisper_backend.audio import transcode
from faster_whisper_backend.media import download as udl
from faster_whisper_backend.media import segmented as seg

_BASE = "https://cdn.test/v/audio/index.m3u8"


# ── parse_hls ─────────────────────────────────────────────────────────────

_TS_PLAYLIST = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:120
#EXT-X-PLAYLIST-TYPE:VOD
#EXTINF:6.000,
seg-120.ts
#EXTINF:6.000,
https://other-cdn.test/seg-121.ts?token=abc
#EXT-X-DISCONTINUITY
#EXTINF:4.5,
/abs/seg-122.ts
#EXT-X-ENDLIST
"""


def test_hls_ts_media_playlist():
    media = seg.parse_hls(_TS_PLAYLIST, _BASE)
    assert media.init is None
    assert media.segments == [
        (("https://cdn.test/v/audio/seg-120.ts", None), 6.0),
        (("https://other-cdn.test/seg-121.ts?token=abc", None), 6.0),
        (("https://cdn.test/abs/seg-122.ts", None), 4.5)]


def test_hls_fmp4_with_map_and_bom():
    text = ("﻿#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-MAP:URI=\"init.mp4\"\n"
            "#EXTINF:2.002,\nseg1.m4s\n#EXTINF:2.002,title\nseg2.m4s\n"
            "#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:1.5,\nseg3.m4s\n#EXT-X-ENDLIST\n")
    media = seg.parse_hls(text, _BASE)
    assert media.init == ("https://cdn.test/v/audio/init.mp4", None)
    assert [d for _p, d in media.segments] == [2.002, 2.002, 1.5]


def test_hls_byte_ranges_one_file():
    # CMAF as arte.tv serves it: one file, init and segments as ranges; a
    # range without @offset continues the previous one.
    text = ("#EXTM3U\n#EXT-X-MAP:URI=\"a.mp4\",BYTERANGE=\"100@0\"\n"
            "#EXTINF:6.0,\n#EXT-X-BYTERANGE:500@100\na.mp4\n"
            "#EXTINF:6.0,\n#EXT-X-BYTERANGE:400\na.mp4\n#EXT-X-ENDLIST\n")
    media = seg.parse_hls(text, _BASE)
    url = "https://cdn.test/v/audio/a.mp4"
    assert media.init == (url, (0, 100))
    assert media.segments == [((url, (100, 500)), 6.0), ((url, (600, 400)), 6.0)]


def test_hls_master_picks_the_default_audio_rendition():
    text = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="English",LANGUAGE="en",URI="en/index.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="Deutsch",DEFAULT=YES,LANGUAGE="de",URI="de/index.m3u8"
#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="s",NAME="de",DEFAULT=YES,URI="subs.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=800000,CODECS="avc1.4d401f,mp4a.40.2",AUDIO="a"
video/480.m3u8
"""
    assert seg.parse_hls(text, "https://cdn.test/master.m3u8") == \
        "https://cdn.test/de/index.m3u8"


def test_hls_master_without_audio_renditions_picks_the_leanest_variant():
    text = ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080\n"
            "1080.m3u8\n#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH=90000,URI=\"if.m3u8\"\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=600000\n360.m3u8\n")
    assert seg.parse_hls(text, _BASE) == "https://cdn.test/v/audio/360.m3u8"


@pytest.mark.parametrize("text,why", [
    ("<html>nope</html>", "not an HLS"),
    ("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"k\"\n#EXTINF:6,\na.ts\n#EXT-X-ENDLIST\n",
     "encrypted"),
    ("#EXTM3U\n#EXT-X-SESSION-KEY:METHOD=SAMPLE-AES,URI=\"k\"\n", "encrypted"),
    ("#EXTM3U\n#EXTINF:6,\na.ts\n#EXTINF:6,\nb.ts\n", "live or still growing"),
    ("#EXTM3U\n#EXT-X-MAP:URI=\"i1.mp4\"\n#EXTINF:6,\na.m4s\n"
     "#EXT-X-MAP:URI=\"i2.mp4\"\n#EXTINF:6,\nb.m4s\n#EXT-X-ENDLIST\n", "init segment"),
    ("#EXTM3U\n#EXTINF:6,\n#EXT-X-BYTERANGE:400\na.mp4\n#EXT-X-ENDLIST\n",
     "without an offset"),
    ("#EXTM3U\na.ts\n#EXT-X-ENDLIST\n", "without a duration"),
    ("#EXTM3U\n#EXTINF:nan,\na.ts\n#EXT-X-ENDLIST\n", "usable duration"),
    ("#EXTM3U\n#EXT-X-ENDLIST\n", "no segments"),
])
def test_hls_unsupported(text, why):
    with pytest.raises(seg.Unsupported, match=why):
        seg.parse_hls(text, _BASE)


# ── dash_media / source_of ────────────────────────────────────────────────

def test_dash_fragments_with_init_and_absolute_urls():
    media = seg.dash_media([
        {"path": "init.dash?x=1"},
        {"path": "a-0.dash?x=1", "duration": 2.0},
        {"url": "https://cdn2.test/a-1.dash", "duration": 2.5}],
        "https://cdn.test/r/dash/")
    assert media.init == ("https://cdn.test/r/dash/init.dash?x=1", None)
    assert media.segments == [(("https://cdn.test/r/dash/a-0.dash?x=1", None), 2.0),
                              (("https://cdn2.test/a-1.dash", None), 2.5)]


@pytest.mark.parametrize("fragments", [
    None, [], [{"path": "init"}],
    [{"path": "a", "duration": 2.0}, {"path": "b"}],          # no duration
    [{"path": "a", "duration": 2.0, "range": "0-99"}],
    [{"duration": 2.0}],                                       # no address
])
def test_dash_unsupported(fragments):
    with pytest.raises(seg.Unsupported):
        seg.dash_media(fragments, "https://cdn.test/")


def test_source_of_picks_segmented_formats_only():
    headers = {"User-Agent": "UA", "Referer": "https://site.test/",
               "Cookie": "secret=1", "Origin": "bad\r\nX: y"}
    assert seg.source_of({"protocol": "https", "url": "https://x.test/a.m4a"}) is None
    assert seg.source_of({"protocol": "m3u8_native", "url": "https://x.test/a.m3u8",
                          "http_headers": headers}) == {
        "protocol": "hls", "url": "https://x.test/a.m3u8", "fragments": None,
        "base": None, "headers": {"User-Agent": "UA", "Referer": "https://site.test/"}}
    dash = seg.source_of({"protocol": "http_dash_segments", "fragments": [{"path": "i"}],
                          "fragment_base_url": "https://x.test/d/"})
    assert (dash["protocol"], dash["base"]) == ("dash", "https://x.test/d/")
    # A merged selection: its audio entry decides.
    merged = {"protocol": "https+m3u8_native", "requested_formats": [
        {"protocol": "https", "vcodec": "avc1", "acodec": "none"},
        {"protocol": "m3u8_native", "vcodec": "none", "acodec": "mp4a.40.2",
         "url": "https://x.test/audio.m3u8"}]}
    assert seg.source_of(merged)["url"] == "https://x.test/audio.m3u8"
    assert seg.source_of({"requested_formats": [
        {"protocol": "m3u8_native", "vcodec": "avc1", "acodec": "mp4a"}]}) is None


# ── fetch_pieces with real segments ───────────────────────────────────────

_RATE = 24000
SEG_S = 4.0


def _encode(fmt: str, levels, options=None) -> bytes:
    """One container of consecutive SEG_S-second 440 Hz tones, the i-th at
    amplitude levels[i] — so a decoded span tells which segment it is."""
    import av
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format=fmt, options=options or {})
    stream = out.add_stream("aac", rate=_RATE)
    stream.layout = "mono"
    n = int(SEG_S * _RATE)
    x = np.concatenate([lvl * np.sin(2 * np.pi * 440 * np.arange(i * n, (i + 1) * n)
                                     / _RATE) for i, lvl in enumerate(levels)])
    x = x.astype(np.float32)
    for k in range(0, x.size, 1024):
        frame = av.AudioFrame.from_ndarray(x[None, k:k + 1024], format="flt",
                                           layout="mono")
        frame.sample_rate = _RATE
        for p in stream.encode(frame):
            out.mux(p)
    for p in stream.encode(None):
        out.mux(p)
    out.close()
    return buf.getvalue()


def _level(i: int) -> float:
    return 0.05 * (i + 1)


def ts_segments(count: int) -> "list[bytes]":
    """HLS TS segments, each its own file (PAT/PMT, timestamps of its own)."""
    return [_encode("mpegts", [_level(i)]) for i in range(count)]


def fmp4_init_and_fragments(count: int) -> "tuple[bytes, list[bytes]]":
    """A fragmented MP4 split into its init (ftyp+moov) and moof+mdat pairs."""
    data = _encode("mp4", [_level(i) for i in range(count)], {
        "movflags": "frag_keyframe+empty_moov+default_base_moof",
        "frag_duration": str(int(SEG_S * 1e6))})
    boxes, o = [], 0
    while o < len(data):
        size, kind = struct.unpack(">I4s", data[o:o + 8])
        boxes.append((kind, data[o:o + size]))
        o += size
    init = b"".join(b for k, b in boxes if k in (b"ftyp", b"moov"))
    frags = [boxes[i][1] + boxes[i + 1][1] for i in range(len(boxes) - 1)
             if boxes[i][0] == b"moof"]
    return init, frags


def hls_playlist(names, *, extra="") -> str:
    return "#EXTM3U\n#EXT-X-TARGETDURATION:4\n" + extra + "".join(
        f"#EXTINF:{SEG_S:.3f},\n{n}\n" for n in names) + "#EXT-X-ENDLIST\n"


@pytest.fixture
def served(fake_capped_get):
    """url → body (bytes, or an int HTTP status); (url, headers) per request."""
    return fake_capped_get.table, fake_capped_get.calls


def _fetch(source, starts, seconds, dest, cancel=lambda: False):
    return asyncio.run(seg.fetch_pieces(source, starts, seconds, str(dest),
                                        cancel_check=cancel))


def _amp(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))) * np.sqrt(2))


def _assert_pieces_hit(pieces, starts, seconds):
    """Each piece is `seconds` long and plays the right segments at the right
    offsets: every quarter second clear of a segment boundary carries the
    level of the segment the piece's start + offset falls in."""
    for (path, skip), start in zip(pieces, starts):
        audio = transcode.decode_span_16k(path, skip, seconds)
        assert audio.size == int(seconds * 16000)
        for k in range(int(seconds * 4) - 1):
            t0, t1 = start + k / 4, start + (k + 1) / 4
            if int((t0 - 0.15) // SEG_S) != int((t1 + 0.15) // SEG_S):
                continue  # straddles a boundary (AAC smear, priming drift)
            got = _amp(audio[k * 4000:(k + 1) * 4000])
            assert got == pytest.approx(_level(int(t0 // SEG_S)), abs=0.012), (start, k)


def test_fetch_hls_ts_pieces(served, tmp_path):
    table, calls = served
    names = [f"s{i}.ts" for i in range(12)]
    for name, body in zip(names, ts_segments(12)):
        table[f"https://cdn.test/v/audio/{name}"] = body
    table[_BASE] = hls_playlist(names).encode()
    source = {"protocol": "hls", "url": _BASE, "headers": {"Referer": "https://site.test/"}}
    starts = [5.0, 21.0, 37.0]
    pieces, got = _fetch(source, starts, 6.0, tmp_path)
    # [start − 1 s, start + 6 s + 1 s] covers two 4 s segments; nothing else
    # is fetched.
    assert got["segments"] == 6 and got["bytes"] == sum(
        len(table[u]) for u, _h in calls[1:])
    assert all(h == {"Referer": "https://site.test/"} for _u, h in calls)
    assert [p.rsplit(".", 1)[1] for p, _s in pieces] == ["ts"] * 3
    _assert_pieces_hit(pieces, starts, 6.0)


def test_fetch_dash_fmp4_pieces(served, tmp_path):
    table, calls = served
    init, frags = fmp4_init_and_fragments(12)
    assert len(frags) >= 12
    base = "https://cdn.test/d/"
    table[base + "init.dash"] = init
    for i, body in enumerate(frags):
        table[f"{base}f{i}.dash"] = body
    source = {"protocol": "dash", "base": base, "fragments":
              [{"path": "init.dash"}] + [{"path": f"f{i}.dash", "duration": SEG_S}
                                          for i in range(len(frags))]}
    starts = [5.0, 21.0, 37.0]
    pieces, got = _fetch(source, starts, 6.0, tmp_path)
    assert [u for u, _h in calls].count(base + "init.dash") == 1     # fetched once
    assert got["segments"] == 6
    _assert_pieces_hit(pieces, starts, 6.0)


def test_fetch_byte_ranged_hls_through_a_master(served, tmp_path):
    table, calls = served
    init, frags = fmp4_init_and_fragments(6)
    blob = init + b"".join(frags)
    table["https://cdn.test/master.m3u8"] = (
        "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",DEFAULT=YES,URI=\"a/au.m3u8\"\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=1,AUDIO=\"a\"\nv.m3u8\n").encode()
    lines, o = [f"#EXTM3U\n#EXT-X-MAP:URI=\"one.mp4\",BYTERANGE=\"{len(init)}@0\"\n"], len(init)
    for f in frags[:6]:
        lines.append(f"#EXTINF:{SEG_S},\n#EXT-X-BYTERANGE:{len(f)}@{o}\none.mp4\n")
        o += len(f)
    table["https://cdn.test/a/au.m3u8"] = ("".join(lines) + "#EXT-X-ENDLIST\n").encode()
    table["https://cdn.test/a/one.mp4"] = blob
    pieces, got = _fetch({"protocol": "hls", "url": "https://cdn.test/master.m3u8"},
                         [9.0], 6.0, tmp_path)
    assert got["segments"] == 2
    assert calls[2][1]["Range"] == f"bytes=0-{len(init) - 1}"
    _assert_pieces_hit(pieces, [9.0], 6.0)


def test_fetch_refuses_a_server_that_ignores_the_range(served, tmp_path, monkeypatch):
    table, _calls = served
    table["https://cdn.test/a.m3u8"] = (
        "#EXTM3U\n#EXTINF:4,\n#EXT-X-BYTERANGE:10@0\nx.ts\n#EXT-X-ENDLIST\n").encode()
    table["https://cdn.test/x.ts"] = b"y" * 50
    real = udl._capped_get

    def _no_range(url, **kw):
        kw["headers"] = {}
        return real(url, **kw)
    monkeypatch.setattr(udl, "_capped_get", _no_range)
    with pytest.raises(seg.Unsupported, match="ignored a byte range"):
        _fetch({"protocol": "hls", "url": "https://cdn.test/a.m3u8"}, [0.0], 2.0, tmp_path)


def test_fetch_caps_and_failures(served, tmp_path, monkeypatch):
    table, calls = served
    names = [f"s{i}.ts" for i in range(40)]
    for n in names:
        table[f"https://cdn.test/v/audio/{n}"] = b"z" * 1000
    src = {"protocol": "hls", "url": _BASE}
    # Segments too short for a piece: refuse rather than fan out.
    table[_BASE] = ("#EXTM3U\n" + "".join(f"#EXTINF:1.0,\n{n}\n" for n in names)
                    + "#EXT-X-ENDLIST\n").encode()
    with pytest.raises(seg.Unsupported, match="too short"):
        _fetch(src, [0.0], 20.0, tmp_path)
    assert len(calls) == 1                                   # only the playlist
    # A piece past the end of the stream.
    table[_BASE] = hls_playlist(names[:3]).encode()
    with pytest.raises(seg.Unsupported, match="past the stream"):
        _fetch(src, [30.0], 6.0, tmp_path)
    # The byte budget over all segments.
    monkeypatch.setattr(seg, "TOTAL_MAX_BYTES", 2500)
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        _fetch(src, [0.0], 11.0, tmp_path)
    # A segment the site refuses: the transport error reaches the caller.
    monkeypatch.setattr(seg, "TOTAL_MAX_BYTES", 10**6)
    table["https://cdn.test/v/audio/s1.ts"] = 403
    with pytest.raises(urllib.error.HTTPError):
        _fetch(src, [0.0], 6.0, tmp_path)
    # Cancel: checked before every request.
    with pytest.raises(udl.UrlCancelled):
        _fetch(src, [0.0], 6.0, tmp_path, cancel=lambda: True)
