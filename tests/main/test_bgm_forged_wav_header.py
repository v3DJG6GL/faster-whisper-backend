"""A PCM WAV whose header lies about its length never reaches the separator
unmeasured: FFmpeg times a WAV whose data size overruns the file by its fmt
byte rate, while the separator's libsndfile reads every sample there is, so
a forged byte rate would let hours of PCM pass the RAM cap as a fraction of
a second. Fixtures are imported from the BGM budget tests, not copied."""

import av

from tests.audio.test_transcode import (
    _forge_wav_length,
    _write_src_wav,
)
from tests.main.test_bgm_transcode_budget import (  # noqa: F401 — fixtures
    _S16_BYTES_PER_S,
    _available_ram,
    _disk_usage_reporting,
    _skipped,
    separate_calls,
)
from tests._transcription_helpers import post_verbose_ok as _post_ok


def _declared_secs(path) -> float:
    with av.open(str(path)) as c:
        return c.duration / 1_000_000


def test_a_pcm_wav_with_a_forged_byte_rate_is_measured_by_the_transcode(
        client, monkeypatch, fake_model, separate_calls,  # noqa: F811
        tmp_path):
    """10 s of PCM declaring 10 ms, past a 2 s cap: the bounded transcode
    measures it and trips the cap."""
    honest = _write_src_wav(str(tmp_path / "src.wav"), rate=16000,
                            nchannels=1, dur_s=10.0)
    forged = _forge_wav_length(honest, str(tmp_path / "lie.wav"))
    assert _declared_secs(forged) < 1
    _available_ram(monkeypatch, available=4 * 2 * _S16_BYTES_PER_S)  # 2 s
    _disk_usage_reporting(monkeypatch, free=5_000_000_000)
    with open(forged, "rb") as f:
        body = _post_ok(client, separate_bgm="true",
                        files={"file": ("a.wav", f.read(), "audio/wav")})
    assert separate_calls == []
    assert _skipped(body)

