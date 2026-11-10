"""The streamed-capture WAV writer (streaming/routes._write_pcm16_wav)."""

import errno
import tempfile
import wave

import numpy as np
import pytest

from faster_whisper_backend.streaming import routes as streaming_routes


def test_a_failed_capture_wav_write_leaves_no_temp_file(tmp_path, monkeypatch):
    """The caller's unlink only covers a path the writer returned: a write that
    raises (full TMPDIR) must remove its own partial file."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def _full(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(wave.Wave_write, "writeframes", _full)
    with pytest.raises(OSError):
        streaming_routes._write_pcm16_wav(np.zeros(1600, np.float32))
    assert not list(tmp_path.glob("whisperup-*.wav"))


def test_a_capture_wav_write_returns_a_readable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = streaming_routes._write_pcm16_wav(np.zeros(1600, np.float32))
    with wave.open(path, "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getnframes()) == (1, 2, 1600)
