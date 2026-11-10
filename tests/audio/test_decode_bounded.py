"""transcode.decode_16k_bounded: faster-whisper's decode_audio sample for
sample, refusing once the decoded input passes a duration ceiling.

The identity tests are the tripwire for a faster-whisper upgrade that changes
its decode steps; the encoded fixtures come from the ffmpeg CLI and skip when
it (or one of its encoders) is unavailable."""

import subprocess

import numpy as np
import pytest
from faster_whisper.audio import decode_audio

from faster_whisper_backend.audio import transcode
from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
from tests.conftest import write_wav


def _tone(seconds, rate, channels=1):
    t = np.arange(int(seconds * rate)) / rate
    mono = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2")
    return np.repeat(mono, channels).tobytes()


def _encode(src, dst, *args):
    try:
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                            "-y", "-i", src, *args, dst],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if r.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {dst}")
    return dst


@pytest.fixture
def stereo_wav(tmp_path):
    return write_wav(str(tmp_path / "s.wav"), _tone(3.0, 44100, 2),
                     nchannels=2, rate=44100)


@pytest.mark.parametrize("name,args", [
    ("a.wav", ["-ar", "16000", "-ac", "1"]),
    ("a8k.wav", ["-ar", "8000", "-ac", "1"]),
    ("a.flac", ["-ar", "48000", "-c:a", "flac", "-frame_size", "65535"]),
    ("a.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    ("a.ogg", ["-c:a", "libopus", "-b:a", "24k"]),
    ("a.m4a", ["-c:a", "aac", "-b:a", "64k"]),
])
def test_output_matches_faster_whispers_decode(tmp_path, stereo_wav, name, args):
    src = _encode(stereo_wav, str(tmp_path / name), *args)
    want = decode_audio(src, sampling_rate=16000)
    got = transcode.decode_16k_bounded(src, max_seconds=None)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)


def test_the_original_wav_matches_too(stereo_wav):
    assert np.array_equal(transcode.decode_16k_bounded(stereo_wav, max_seconds=None),
                          decode_audio(stereo_wav, sampling_rate=16000))


def test_input_past_the_ceiling_is_refused(tmp_path):
    src = write_wav(str(tmp_path / "two.wav"), _tone(2.0, 16000))
    with pytest.raises(transcode.DecodeTooLong):
        transcode.decode_16k_bounded(src, max_seconds=1.0)
    assert transcode.decode_16k_bounded(src, max_seconds=3.0).shape == (32000,)


def test_a_one_hertz_block_is_refused_before_it_is_resampled(tmp_path):
    """One 65535-sample block declaring 1 Hz is 18 h of audio; decode_audio
    would resample its 500 000-sample group to ~10^10 samples in one call
    (2+ GB). The ceiling charges each block as it leaves the decoder."""
    raw = tmp_path / "z.raw"
    raw.write_bytes(bytes(range(7, 256, 7)) * 7282)
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "s16le", "-ar", "1", "-ac", "1", "-i", str(raw),
             "-c:a", "flac", "-frame_size", "65535", str(tmp_path / "low.flac")],
            capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if r.returncode != 0:
        pytest.skip("ffmpeg cannot write a 1 Hz flac")
    with pytest.raises(transcode.DecodeTooLong):
        transcode.decode_16k_bounded(str(tmp_path / "low.flac"), max_seconds=36000)
