"""The learned actual/estimated size ratio (stage_rates under RATIO_STAGE) is
shared by every user of an extractor + protocol family and persists across
restarts. A site's advertised bitrate or approximate size is its own claim:
a sample far outside what real sites land at (0.3-1.5) is not learned, and a
stored ratio outside the band is ignored, so one manifest cannot price that
key's every rung over the size cap."""

import asyncio
import math

import pytest

from faster_whisper_backend.media import download as udl
from faster_whisper_backend.media import video as media_video
from faster_whisper_backend.runtime import stage_rates
from faster_whisper_backend.transcription import models as tx_models
from tests.media._server_helpers import _stand_in_result


def _fmt(**kw):
    base = {"format_id": "x", "ext": "webm", "protocol": "https"}
    base.update(kw)
    return base


@pytest.mark.parametrize("value, expected", [
    (0.1, 0.1), (0.5, 0.5), (1, 1.0), (10.0, 10.0), ("2", 2.0),
    (0.09, None), (10.01, None), (4000.0, None), (0, None), (-1.0, None),
    (math.nan, None), (math.inf, None), (None, None), ("x", None), pytest.param(10**400, None, id="huge-int"),
])
def test_plausible_ratio_keeps_only_the_band(value, expected):
    assert udl.plausible_ratio(value) == expected


def _hls_info(audio: dict) -> dict:
    return {"duration": 100.0, "formats": [
        audio,
        _fmt(format_id="hls-1080", vcodec="avc1", acodec="none", height=1080,
             tbr=1000, protocol="m3u8_native", ext="mp4"),
    ]}


def test_ladder_ignores_an_implausible_learned_video_ratio():
    info = _hls_info(_fmt(format_id="a", vcodec="none", acodec="mp4a.40.2",
                          abr=128, filesize=1_000_000, ext="m4a"))
    raw_v = 1000 * 100 * 125
    raw = raw_v + 1_000_000
    ladder = udl.build_video_ladder(info, max_bytes=raw + 1, extractor="Generic",
                                    approx_ratio=lambda fam: 4000.0)
    rung = ladder[0]
    assert rung["over_cap"] is False
    assert rung["video_bytes"] == raw_v and rung["approx_bytes"] == raw
    assert rung["tbr_kbps"] == 1000 + 128


def test_ladder_ignores_an_implausible_learned_audio_ratio():
    info = {"duration": 100.0, "formats": [
        _fmt(format_id="a", vcodec="none", acodec="mp4a.40.2", abr=128,
             filesize_approx=2_000_000, protocol="m3u8_native", ext="m4a"),
        _fmt(format_id="v", vcodec="avc1", acodec="none", height=720,
             filesize=10_000_000, tbr=800, ext="mp4"),
    ]}
    raw = 10_000_000 + 2_000_000
    ladder = udl.build_video_ladder(info, max_bytes=raw + 1, extractor="Generic",
                                    approx_ratio=lambda fam: 4000.0)
    rung = ladder[0]
    assert rung["over_cap"] is False
    assert rung["audio_bytes"] == 2_000_000 and rung["approx_bytes"] == raw


def test_ladder_still_scales_by_an_in_band_ratio():
    info = _hls_info(_fmt(format_id="a", vcodec="none", acodec="mp4a.40.2",
                          abr=128, filesize_approx=1_000_000,
                          protocol="m3u8_native", ext="m4a"))
    rung = udl.build_video_ladder(info, max_bytes=10**10,
                                  approx_ratio=lambda fam: 0.5)[0]
    assert rung["video_bytes"] == int(1000 * 100 * 125 * 0.5)
    assert rung["audio_bytes"] == 500_000


def _run_video(monkeypatch, rung: dict) -> "tuple[dict, list]":
    async def _download_video(url, *, dest_dir, **kw):
        return _stand_in_result(dest_dir, "media.mkv", b"v" * 88)
    monkeypatch.setattr(udl, "download_video", _download_video)
    seen: list = []
    monkeypatch.setattr(stage_rates, "record", lambda *a: seen.append(a))

    async def _main():
        monkeypatch.setattr(tx_models, "_url_download_semaphore",
                            asyncio.Semaphore(1))
        return await media_video._download_video_for_run(
            None, "https://e.test/v", rung, capped=False, user_id=None,
            protect=None, run_finished=[False])
    return asyncio.run(_main()), seen


def _approx_rung(raw: int) -> dict:
    return {"height": 720, "container": "mkv", "approx_bytes": raw,
            "raw_approx_bytes": raw, "bytes_approx": True,
            "extractor": "Generic", "protocol": "m3u8"}


def test_a_ratio_outside_the_band_is_not_learned(monkeypatch):
    state, seen = _run_video(monkeypatch, _approx_rung(1))   # 88 / 1
    assert state["state"] == "done", state
    assert seen == []


def test_an_in_band_ratio_is_still_learned(monkeypatch):
    state, seen = _run_video(monkeypatch, _approx_rung(176))  # 88 / 176
    assert state["state"] == "done", state
    assert len(seen) == 1 and seen[0][:3] == (udl.RATIO_STAGE, "Generic", "m3u8")
    assert seen[0][-1] == pytest.approx(0.5)
