"""Streaming endpointing (VAD) for the live transcription session.

One question, asked per fixed 512-sample (32 ms @ 16 kHz) float32 frame: *is this
speech?* The two-tier silence timing (inner partial-boundary gate, outer commit
gate) and the forced-commit cap live in :class:`streaming_session.StreamSession`,
not here — this module only classifies frames.

Two backends behind one interface:

  * :class:`SileroEndpointer` — Silero VAD (robust to room noise; the research's
    recommended production gate). Lazy-imported from the standalone ``silero_vad``
    package; constructing it raises if unavailable so the factory falls back.
  * :class:`EnergyEndpointer` — pure-numpy RMS gate with hysteresis. No extra
    dependencies; the default on hosts without Silero and in tests. Adequate for
    close-mic dictation in a quiet room; see the note in ``make_endpointer``.

Silero is not installed on every host (and is not importable on the Linux dev box
at all), so Energy is the safe default and Silero is an opt-in upgrade validated on
the production host.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero v5 requires exactly this at 16 kHz (= 32 ms)
FRAME_MS = 1000 * FRAME_SAMPLES // SAMPLE_RATE  # 32


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of float32 [-1, 1] samples in dBFS (full-scale = 0 dB)."""
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    if rms <= 1e-9:
        return float("-inf")
    return 20.0 * np.log10(rms)


class EnergyEndpointer:
    """RMS gate with hysteresis. Speech turns on above ``threshold_dbfs`` and off
    only after dropping ``hysteresis_db`` below it — so a momentary dip mid-word
    doesn't flip the state."""

    def __init__(self, threshold_dbfs: float = -42.0, hysteresis_db: float = 6.0):
        self._on = threshold_dbfs
        self._off = threshold_dbfs - hysteresis_db
        self._speaking = False

    def is_speech(self, frame: np.ndarray) -> bool:
        level = rms_dbfs(frame)
        if self._speaking:
            if level < self._off:
                self._speaking = False
        elif level > self._on:
            self._speaking = True
        return self._speaking

    def reset(self) -> None:
        self._speaking = False


class SileroEndpointer:
    """Silero VAD run frame-by-frame via the standalone ``silero_vad`` package.

    Raises ImportError/Exception on construction if the package (and its torch
    backend) isn't available; :func:`make_endpointer` catches that and falls back.
    """

    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad  # noqa: PLC0415  (optional dep)
        import torch  # noqa: PLC0415

        self._torch = torch
        self._model = load_silero_vad()
        self.threshold = threshold
        # Silero ends speech with built-in hysteresis at (threshold - 0.15).
        self._off = threshold - 0.15
        self._speaking = False

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != FRAME_SAMPLES:
            # Pad/truncate to the strict window — Silero v5 crashes otherwise.
            buf = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            n = min(FRAME_SAMPLES, frame.shape[0])
            buf[:n] = frame[:n]
            frame = buf
        prob = float(self._model(self._torch.from_numpy(frame), SAMPLE_RATE).item())
        if self._speaking:
            if prob < self._off:
                self._speaking = False
        elif prob >= self.threshold:
            self._speaking = True
        return self._speaking

    def reset(self) -> None:
        self._speaking = False
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()


def make_endpointer(backend: str = "auto", *, threshold: float = 0.5,
                    energy_dbfs: float = -42.0):
    """Build an endpointer. ``backend`` is ``"silero"``, ``"energy"`` or ``"auto"``
    (try Silero, fall back to Energy with a logged warning)."""
    if backend in ("silero", "auto"):
        try:
            return SileroEndpointer(threshold=threshold)
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            if backend == "silero":
                raise
            logger.warning(
                "[streaming-vad] Silero unavailable (%s); using energy gate "
                "(threshold %.0f dBFS). Install 'silero-vad' for noise-robust "
                "endpointing.", exc, energy_dbfs,
            )
    return EnergyEndpointer(threshold_dbfs=energy_dbfs)


def iter_frames(samples: np.ndarray, frame_samples: int = FRAME_SAMPLES):
    """Yield consecutive full ``frame_samples`` slices; drops a trailing partial
    frame (the caller carries it forward)."""
    n_full = samples.shape[0] // frame_samples
    for i in range(n_full):
        yield samples[i * frame_samples:(i + 1) * frame_samples]
