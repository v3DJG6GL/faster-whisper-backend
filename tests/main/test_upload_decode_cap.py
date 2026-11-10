"""The transcription handler decodes an upload itself, under a ceiling
derived from free RAM, before Whisper sees it: only the byte size was capped
before, and a small file of compressed silence decodes to gigabytes."""

import numpy as np
import psutil

from tests._transcription_helpers import post_transcription, wav_bytes


def test_audio_decoding_past_the_ceiling_is_a_413(client, app_module,
                                                  fake_model, monkeypatch):
    monkeypatch.setattr(app_module, "_upload_decode_cap_s", lambda: 1.0)
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 413, r.text
    assert "too long to decode" in r.json()["detail"]
    assert fake_model.last_audio is None


def test_audio_within_the_ceiling_reaches_the_model_decoded(
        client, app_module, fake_model, monkeypatch):
    monkeypatch.setattr(app_module, "_upload_decode_cap_s", lambda: 3.0)
    app_module.cfg.LEADING_SILENCE_PAD_MS = 0
    r = post_transcription(client, files={
        "file": ("a.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert r.status_code == 200, r.text
    assert isinstance(fake_model.last_audio, np.ndarray)
    assert fake_model.last_audio.shape == (32000,)


def test_the_ceiling_is_half_the_free_ram_at_the_decode_peak(app_module,
                                                              monkeypatch):
    real = psutil.virtual_memory
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: real()._replace(available=16_000_000_000))
    # 8 GB at 16 kHz x 12 B per sample: ~11.6 h of audio.
    assert app_module._upload_decode_cap_s() == 16_000_000_000 / 2 / (16000 * 12)


def test_the_ceiling_fails_open_when_ram_cannot_be_read(app_module, monkeypatch):
    def _broken():
        raise OSError("no /proc/meminfo")
    monkeypatch.setattr(psutil, "virtual_memory", _broken)
    assert app_module._upload_decode_cap_s() is None
