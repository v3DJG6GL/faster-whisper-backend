"""media/download.py against site output yt-dlp passes through as-is: a
stdout line longer than the stream limit, and HLS variants that list no
codecs."""

from __future__ import annotations

import pytest

from faster_whisper_backend.media import download as udl
from tests.media.test_download import _patch_argv, _run


def test_download_over_long_filtered_line_is_still_a_policy_refusal(tmp_path, monkeypatch):
    # The "<title> does not pass filter" line carries the page's untrusted
    # title, well past asyncio's 64 KiB readline limit.
    _patch_argv(monkeypatch, """
print("[download] " + "t" * 200_000
      + " does not pass filter (extractor_key!=Generic), skipping ..",
      flush=True)
""")
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=30))


def test_download_keeps_reading_progress_after_an_over_long_line(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import os
print("[generic] Extracting URL: https://x/" + "a" * 100_000, flush=True)
print("dl:1000 1000 NA", flush=True)
open(os.path.join(r"__DEST__", "media.m4a"), "wb").write(b"x" * 64)
""")
    seen = []
    out = _run(udl.download(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=10_000,
        timeout=30, progress_cb=lambda f, tot: seen.append((f, tot))))
    assert out.endswith("media.m4a")
    assert seen and seen[-1] == (1.0, 1000)


def test_video_ladder_offers_heights_for_hls_variants_without_codecs():
    # yt-dlp leaves vcodec/acodec unset for an EXT-X-STREAM-INF with
    # RESOLUTION but no CODECS.
    formats = [
        {"format_id": "hls-1080", "protocol": "m3u8_native", "ext": "mp4",
         "height": 1080, "width": 1920, "tbr": 5000},
        {"format_id": "hls-480", "protocol": "m3u8_native", "ext": "mp4",
         "height": 480, "width": 854, "tbr": 1200},
        # An explicitly audio-only rendition stays out of the video rungs.
        {"format_id": "hls-audio", "protocol": "m3u8_native", "ext": "mp4",
         "vcodec": "none", "acodec": "mp4a.40.2", "abr": 128},
    ]
    ladder = udl.build_video_ladder({"duration": 100.0, "formats": formats},
                                    max_bytes=10**10)
    assert [r["format_id"] for r in ladder] == ["hls-1080", "hls-480"]
    assert [r["label"] for r in ladder] == ["1080p", "480p"]
    assert all(r["protocol"] == udl.protocol_family("m3u8_native") for r in ladder)
    assert ladder[0]["vcodec"] is None and ladder[0]["container"] == "mkv"
    assert ladder[1]["approx_bytes"] is not None


def test_video_ladder_keeps_unknown_codec_formats_without_height_out():
    # vcodec None with no height is not evidence of video: a podcast mp3
    # still gets no ladder.
    assert udl.build_video_ladder(
        {"ext": "mp3", "formats": [{"format_id": "mp3", "ext": "mp3",
                                    "protocol": "https"}]},
        max_bytes=1) == []
