"""An upload whose decode fails in the s16 mono resampler is answered with
the resampler's own error, not handed to transcribe() as a path:
faster-whisper's decode_audio runs the same resampler, fails on the same
group, and first buffers a whole 500 000-sample group of every channel
(~0.5 GB for 254 channels of a 1 MB Opus file)."""

import av
import pytest

from tests._transcription_helpers import post_transcription, wav_bytes


def _failing_decode(app_module, monkeypatch, cause):
    def _decode(path, *, max_seconds):
        raise app_module._transcode.ResampleFailed(str(cause)) from cause
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _decode)


def test_a_resample_failure_is_not_decoded_twice(
        client, app_module, fake_model, monkeypatch):
    """The answer is what transcribe(path) gives today for that error: a
    generic 500, the model never called."""
    _failing_decode(app_module, monkeypatch,
                    av.error.ArgumentError(22, "Invalid argument"))
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 500, r.text
    assert r.json()["detail"] == "transcription failed"
    assert fake_model.last_audio is None


def test_a_resample_failure_keeps_the_causes_status(
        client, app_module, fake_model, monkeypatch):
    """The av error itself reaches the handler's arms, so one the route maps
    (undecodable input: 400) keeps that mapping."""
    _failing_decode(app_module, monkeypatch,
                    av.error.InvalidDataError(1094995529, "Invalid data"))
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 400, r.text
    assert fake_model.last_audio is None


@pytest.mark.parametrize("exc", [RuntimeError("no decoder"),
                                 av.error.InvalidDataError(1094995529, "x")])
def test_other_decode_failures_still_hand_transcribe_the_path(
        client, app_module, fake_model, monkeypatch, exc):
    def _decode(path, *, max_seconds):
        raise exc
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _decode)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert isinstance(fake_model.last_audio, str)


def _g726_wav_refusing_its_decoder() -> bytes:
    """A WAV the whitelisted open reads (fmt tag 0x0045, g726) whose 9-bit
    code size the decoder refuses when the decode opens it: an av error
    raised after the open, as faster-whisper's decode_audio raises it too."""
    import struct
    data = bytes(range(256)) * 16
    fmt = struct.pack("<HHIIHH", 0x0045, 1, 8000, 16000, 1, 9) + b"\x00\x00"
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_an_av_error_after_the_open_never_reopens_the_path(
        client, app_module, fake_model, monkeypatch):
    """Past a successful whitelisted open, the decoder's own av error is the
    answer: handing transcribe() the path would only re-open it through
    faster-whisper's decode_audio, which probes every stream with every
    decoder, and meet the same error."""
    def _decode(path, *, max_seconds):
        raise app_module._transcode.DecodeFailed("x") from (
            av.error.ArgumentError(22, "Invalid argument"))
    monkeypatch.setattr(app_module._transcode, "decode_16k_bounded", _decode)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=1.0), "audio/wav")})
    assert r.status_code == 500, r.text
    assert r.json()["detail"] == "transcription failed"
    assert fake_model.last_audio is None


def test_a_wav_whose_decoder_refuses_to_open_is_not_handed_over(
        client, fake_model):
    r = post_transcription(client, files={
        "file": ("a.wav", _g726_wav_refusing_its_decoder(), "audio/wav")})
    assert r.status_code == 500, r.text
    assert r.json()["detail"] == "transcription failed"
    assert fake_model.last_audio is None
