"""The Silero endpointer's rolling lookback reaches the model intact: the
bundled SileroVADModel zeroes the tail of its input array in place, and that
write must not land in the endpointer's own buffer."""

import numpy as np
import pytest

from faster_whisper_backend.streaming.vad import FRAME_SAMPLES, make_endpointer


def test_model_writes_do_not_corrupt_the_lookback_buffer(monkeypatch):
    fw_vad = pytest.importorskip("faster_whisper.vad")
    seen = []

    class _ZeroesItsInput:
        # faster-whisper 1.2.1 SileroVADModel.__call__ does
        # `context = batched_audio[..., -64:]; context[-1] = 0` on a view.
        def __call__(self, audio):
            seen.append(audio.copy())
            audio.reshape(-1, FRAME_SAMPLES)[..., -64:][-1] = 0
            return np.zeros(audio.shape[0] // FRAME_SAMPLES, dtype=np.float32)

    monkeypatch.setattr(fw_vad, "get_vad_model", _ZeroesItsInput)
    ep = make_endpointer("silero")
    frame = np.full(FRAME_SAMPLES, 0.25, dtype=np.float32)
    ep.is_speech(frame)
    ep.is_speech(frame)
    # The second window's previous frame is the first frame, whole.
    np.testing.assert_array_equal(seen[-1][-2 * FRAME_SAMPLES:-FRAME_SAMPLES], frame)
