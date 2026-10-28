"""Video ladder over an HLS master playlist with EXT-X-MEDIA audio renditions.

yt-dlp never names a rendition's codec: it comes out with vcodec "none" and
acodec None. The STREAM-INF variants come out video-only (acodec "none")
when the master lists CODECS, and with both codecs None when it does not.
These tests parse a real master with yt-dlp's own m3u8 parser and run each
rung's `-f` selector through yt-dlp's own format resolution (no network),
so they pin the formats yt-dlp will actually fetch, not our selector text.
"""

from __future__ import annotations

import copy

import yt_dlp
from yt_dlp.extractor.common import InfoExtractor

from faster_whisper_backend.media import download as udl


def _master(codecs: bool) -> str:
    """A Twitter/X-style master: audio groups listed lowest bitrate first."""
    c = ',CODECS="avc1.640028,mp4a.40.2"' if codecs else ""
    lines = ["#EXTM3U"]
    for kbps in (32000, 64000, 128000):
        lines.append(f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio-{kbps}",NAME="Audio",'
                     f'DEFAULT=YES,AUTOSELECT=YES,URI="a{kbps}.m3u8"')
    for bw, res, grp in ((300000, "480x270", 32000), (900000, "1280x720", 64000),
                         (5000000, "1920x1080", 128000)):
        lines += [f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={res}{c},'
                  f'AUDIO="audio-{grp}"', f"v{bw}.m3u8"]
    return "\n".join(lines)


def _parse(codecs: bool) -> list:
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        formats, _subs = InfoExtractor(ydl)._parse_m3u8_formats_and_subtitles(
            _master(codecs), "https://media.example/m.m3u8", ext="mp4", m3u8_id="hls")
    return formats


def _resolve(selector: str, formats: list) -> list:
    """The format ids yt-dlp downloads for `selector` (the merge legs, or
    the single format)."""
    with yt_dlp.YoutubeDL({"format": selector, "quiet": True, "simulate": True}) as ydl:
        res = ydl.process_ie_result({
            "id": "x", "title": "t", "extractor": "generic", "extractor_key": "Generic",
            "webpage_url": "https://media.example/", "formats": copy.deepcopy(formats),
        }, download=False)
    legs = res.get("requested_formats") or [res]
    return [f["format_id"] for f in legs]


def _selector(rung: dict) -> str:
    return udl.video_format_selector(
        rung["height"], (rung["format_id"], rung["audio_format_id"]))


def test_video_only_hls_variants_merge_the_best_audio_group():
    """The renditions all tie on language/quality/bitrate: naming the first
    listed one kept a 1080p video with 32 kbps audio. "ba" lets yt-dlp rank
    them (source_preference) and pick the 128k group."""
    formats = _parse(codecs=True)
    ladder = udl.build_video_ladder({"duration": 60.0, "formats": formats},
                                    max_bytes=10**12)
    assert [r["format_id"] for r in ladder] == ["hls-5000", "hls-900", "hls-300"]
    assert _resolve(_selector(ladder[0]), formats) == ["hls-5000", "hls-audio-128000-Audio"]
    for rung in ladder:
        assert rung["audio_format_id"] == "ba"


def test_codecless_hls_rung_claims_only_the_leg_yt_dlp_fetches():
    """With acodec None on the variant yt-dlp drops the rendition from any
    merge: the rung must not price or count an audio leg it never gets."""
    formats = _parse(codecs=False)
    ladder = udl.build_video_ladder({"duration": 60.0, "formats": formats},
                                    max_bytes=10**12)
    top = ladder[0]
    assert top["format_id"] == "hls-5000"
    assert top["audio_format_id"] is None
    assert top["audio_bytes"] is None
    assert top["approx_bytes"] == top["video_bytes"]
    # An unknown codec stays unknown, never the string "None".
    assert top["acodec"] is None
    assert _resolve(_selector(top), formats) == ["hls-5000"]


def test_codecless_variant_takes_no_audio_leg_even_with_a_known_codec_track():
    """The drop is yt-dlp's, not the rendition's: a separate track with a
    known codec beside an acodec-None variant is dropped from the merge too."""
    formats = _parse(codecs=False) + [
        {"format_id": "aud-orig", "protocol": "m3u8_native", "ext": "mp4",
         "url": "https://media.example/orig.m3u8", "tbr": 128,
         "vcodec": "none", "acodec": "mp4a.40.2", "language_preference": 10}]
    assert _resolve("hls-5000+aud-orig", formats) == ["hls-5000"]
    ladder = udl.build_video_ladder({"duration": 60.0, "formats": formats},
                                    max_bytes=10**12)
    top = ladder[0]
    assert top["audio_format_id"] is None
    assert top["approx_bytes"] == top["video_bytes"]


def test_video_only_variant_merges_a_known_codec_track():
    formats = _parse(codecs=True) + [
        {"format_id": "aud-orig", "protocol": "m3u8_native", "ext": "mp4",
         "url": "https://media.example/orig.m3u8", "tbr": 128,
         "vcodec": "none", "acodec": "mp4a.40.2", "language_preference": 10}]
    ladder = udl.build_video_ladder({"duration": 60.0, "formats": formats},
                                    max_bytes=10**12)
    assert ladder[0]["audio_format_id"] == "aud-orig"
    assert _resolve(_selector(ladder[0]), formats) == ["hls-5000", "aud-orig"]
