"""media/download.py against site output yt-dlp passes through as-is: a
stdout line longer than the stream limit, and HLS variants that list no
codecs."""

from __future__ import annotations

import pytest

from faster_whisper_backend.media import download as udl
from tests.media.test_download import (
    _fetch, _fetch_progress, _patch_argv, _patch_video_argv)


def _forged_legs_line(n: int, done: int) -> str:
    # A site title with HTML-escaped newlines in yt-dlp's "does not pass
    # filter" line: every forged row parses as a progress line naming a
    # new format id.
    rows = "".join(f"\ndl:{done} {done} NA f{i}" for i in range(n))
    return f"[download] t{rows} does not pass filter (direct), skipping .."


def test_download_video_tracks_a_bounded_number_of_forged_legs(tmp_path, monkeypatch):
    # Each tracked leg costs work on every later progress line, so a flood
    # of forged ids must not grow the bookkeeping (nor the byte count)
    # past the cap.
    _patch_video_argv(monkeypatch, f"""
import os
print({_forged_legs_line(50, 10)!r}, flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
""")
    out, seen = _fetch_progress(udl.download_video, tmp_path)
    assert out.endswith("media.mkv")
    assert len(seen) == 50
    assert max(d for _f, _t, d in seen) <= 10 * (udl._MAX_PROGRESS_LEGS + 1)


def test_download_video_size_cap_still_sees_a_real_leg_after_forged_legs(
        tmp_path, monkeypatch):
    # Past the cap a new id is not tracked as a leg, but its bytes still
    # count: forged rows first must not blind the in-flight size check.
    _patch_video_argv(monkeypatch, f"""
import os
print({_forged_legs_line(50, 0)!r}, flush=True)
print("dl:5000 NA NA 137", flush=True)
print("dl:20000 NA NA 137", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
""")
    with pytest.raises(udl.UrlPolicyError, match="size limit"):
        _fetch(udl.download_video, tmp_path, max_bytes=10_000)


def test_download_video_size_cap_sees_a_real_leg_after_a_negative_forged_count(
        tmp_path, monkeypatch):
    # A site-controlled id echoed on stdout can carry a negative count;
    # counted as a finished leg it would drive the total below zero so the
    # in-flight size check never fires on the real legs after it.
    _patch_video_argv(monkeypatch, """
import os
print("dl:-9000000000000 NA NA zz", flush=True)
print("dl:5000 NA NA 137", flush=True)
print("dl:20000 NA NA 137", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
""")
    with pytest.raises(udl.UrlPolicyError, match="size limit"):
        _fetch(udl.download_video, tmp_path, max_bytes=10_000)


def test_progress_line_with_a_negative_count_is_not_progress():
    # yt-dlp never prints a negative count: the whole line is noise, and a
    # negative total falls back to the estimate.
    assert udl._parse_progress_fields("dl:-5 NA NA 137") is None
    assert udl._parse_progress_fields("dl:5 -9 NA")[:2] == (5, None)
    assert udl._parse_progress_fields("dl:5 -9 70")[:2] == (5, 70)
    assert udl._parse_progress_fields("dl:0 0 NA 137") == (0, 0, "137")


def test_download_video_two_named_legs_after_the_cap_count_in_full(tmp_path, monkeypatch):
    _patch_video_argv(monkeypatch, f"""
import os
print({_forged_legs_line(20, 0)!r}, flush=True)
for ln in ("dl:500 1000 NA 137", "dl:1000 1000 NA 137",
           "dl:100 300 NA 140", "dl:300 300 NA 140"):
    print(ln, flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
""")
    _, seen = _fetch_progress(udl.download_video, tmp_path)
    assert [d for _f, _t, d in seen][-4:] == [500, 1000, 1100, 1300], seen


def test_download_over_long_filtered_line_is_still_a_policy_refusal(tmp_path, monkeypatch):
    # The "<title> does not pass filter" line carries the page's untrusted
    # title, well past asyncio's 64 KiB readline limit.
    _patch_argv(monkeypatch, """
print("[download] " + "t" * 200_000
      + " does not pass filter (extractor_key!=Generic), skipping ..",
      flush=True)
""")
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        _fetch(udl.download, tmp_path)


def test_download_keeps_reading_progress_after_an_over_long_line(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import os
print("[generic] Extracting URL: https://x/" + "a" * 100_000, flush=True)
print("dl:1000 1000 NA", flush=True)
open(os.path.join(r"__DEST__", "media.m4a"), "wb").write(b"x" * 64)
""")
    out, seen = _fetch_progress(udl.download, tmp_path)
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
