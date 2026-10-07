"""Tests for the streaming endpointer: energy gate, the make_endpointer fallback
logic, and frame iteration. The bundled-Silero path needs faster-whisper (not
installed in the deps-free test env), so its availability is monkeypatched to keep
these deterministic on every platform.
"""

import numpy as np
import pytest

from faster_whisper_backend.streaming import vad as streaming_vad
from faster_whisper_backend.streaming.vad import (
    FRAME_SAMPLES,
    EnergyEndpointer,
    iter_frames,
    make_endpointer,
    rms_dbfs,
)


def _frame(level):
    return np.full(FRAME_SAMPLES, level, dtype=np.float32)


def test_rms_dbfs_silence_and_full_scale():
    assert rms_dbfs(np.zeros(FRAME_SAMPLES, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.ones(FRAME_SAMPLES, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)


def test_energy_endpointer_hysteresis():
    ep = EnergyEndpointer(threshold_dbfs=-42.0, hysteresis_db=6.0)
    assert ep.is_speech(_frame(0.0)) is False           # silence
    assert ep.is_speech(_frame(0.3)) is True            # loud → speech (~ -10 dBFS)
    # A small dip stays "speaking" until below the -48 dBFS off-threshold.
    assert ep.is_speech(_frame(0.01)) is True           # ~ -40 dBFS, above off
    assert ep.is_speech(_frame(0.0)) is False           # silence → off
    ep.reset()
    assert ep._speaking is False


def test_make_endpointer_energy_backend():
    ep = make_endpointer("energy")
    assert isinstance(ep, EnergyEndpointer)


def test_silero_off_threshold_floored_at_low_threshold():
    # The off-threshold (which releases the speech latch) must stay positive even
    # for a configured threshold < 0.15 — otherwise it goes negative and the
    # latch can never release (prob is always >= 0), so silence never ends the
    # utterance. Needs the bundled Silero model; skip where unavailable.
    pytest.importorskip("faster_whisper")
    assert streaming_vad.SileroEndpointer(threshold=0.1)._off >= 0.01
    # default keeps the standard Silero hysteresis (threshold - 0.15)
    assert streaming_vad.SileroEndpointer(threshold=0.5)._off == pytest.approx(0.35)


def test_silero_mid_stream_degradation_honours_the_configured_energy_gate(monkeypatch):
    """A Silero failure mid-session degrades to the energy gate at the
    configured STREAMING_GATE_RMS_DBFS (make_endpointer's energy_dbfs), not a
    hard-coded -42 dBFS that would gate quiet speech for the rest of it."""
    fw_vad = pytest.importorskip("faster_whisper.vad")

    class _FailsAfterProbe:
        calls = 0

        def __call__(self, audio):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("onnx session died")
            return np.zeros(audio.shape[0] // 512, dtype=np.float32)

    monkeypatch.setattr(fw_vad, "get_vad_model", _FailsAfterProbe)
    ep = make_endpointer("silero", energy_dbfs=-60.0)
    assert isinstance(ep, streaming_vad.SileroEndpointer)
    assert ep.is_speech(_frame(10 ** (-50 / 20))) is True     # -50 dBFS > -60


def test_make_endpointer_auto_falls_back_when_silero_unavailable(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no bundled silero here")

    monkeypatch.setattr(streaming_vad, "SileroEndpointer", _Boom)
    ep = make_endpointer("auto")
    assert isinstance(ep, EnergyEndpointer)


def test_make_endpointer_silero_backend_raises_when_unavailable(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no bundled silero here")

    monkeypatch.setattr(streaming_vad, "SileroEndpointer", _Boom)
    with pytest.raises(RuntimeError):
        make_endpointer("silero")


def test_make_endpointer_auto_uses_silero_when_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(streaming_vad, "SileroEndpointer", lambda **k: sentinel)
    assert make_endpointer("auto") is sentinel


def test_iter_frames_drops_partial_tail():
    samples = np.arange(FRAME_SAMPLES * 2 + 100, dtype=np.float32)
    frames = list(iter_frames(samples))
    assert len(frames) == 2
    assert all(f.shape[0] == FRAME_SAMPLES for f in frames)
