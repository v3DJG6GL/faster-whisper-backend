"""Video ladder over an HLS master playlist that lists no CODECS.

yt-dlp leaves both codecs unknown then: the STREAM-INF variants come out
with vcodec/acodec None (and a height), the EXT-X-MEDIA TYPE=AUDIO rendition
with vcodec "none" and acodec None. The audio lives only in the rendition's
own playlist, so each rung must merge it in, or the kept video is silent.
"""

from __future__ import annotations

from faster_whisper_backend.media import download as udl

_HLS_NO_CODECS = {
    "duration": 600.0,
    "formats": [
        {"format_id": "hls-aud-en", "protocol": "m3u8_native", "ext": "mp4",
         "vcodec": "none", "language": "en"},
        {"format_id": "hls-1200", "protocol": "m3u8_native", "ext": "mp4",
         "height": 480, "width": 854, "tbr": 1200},
        {"format_id": "hls-5000", "protocol": "m3u8_native", "ext": "mp4",
         "height": 1080, "width": 1920, "tbr": 5000},
    ],
}


def test_codecless_hls_rungs_merge_the_audio_rendition():
    ladder = udl.build_video_ladder(_HLS_NO_CODECS, max_bytes=10**10)
    assert [r["format_id"] for r in ladder] == ["hls-5000", "hls-1200"]
    for rung in ladder:
        assert rung["audio_format_id"] == "hls-aud-en"
        # An unknown codec stays unknown, never the string "None".
        assert rung["acodec"] is None
    top = ladder[0]
    assert udl.video_format_selector(
        1080, (top["format_id"], top["audio_format_id"])
    ).startswith("hls-5000+hls-aud-en/")


def test_codecless_hls_audio_rendition_loses_to_a_known_original_track():
    """A rendition with unknown codec is a candidate, not a trump card: the
    original-language track still wins on language_preference."""
    info = {**_HLS_NO_CODECS, "formats": [
        {**_HLS_NO_CODECS["formats"][0], "language_preference": -1},
        {"format_id": "hls-aud-orig", "protocol": "m3u8_native", "ext": "mp4",
         "vcodec": "none", "acodec": "mp4a.40.2", "language_preference": 10},
        *_HLS_NO_CODECS["formats"][1:],
    ]}
    ladder = udl.build_video_ladder(info, max_bytes=10**10)
    assert ladder[0]["audio_format_id"] == "hls-aud-orig"
