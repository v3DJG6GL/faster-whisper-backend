"""Spies shared by the tests/main/ files that drive a transcription through
its stages. Import as ``from tests.main._stage_helpers import ...``."""

from faster_whisper_backend.transcription import receipt as tx_receipt


def capture_receipt(monkeypatch, seen=None):
    """Record every _format_request_block kwarg (and the rendered block, as
    "rendered") into `seen`, a new dict by default; returns it."""
    seen = {} if seen is None else seen
    real_block = tx_receipt._format_request_block

    def _capture(**kw):
        seen.update(kw)
        seen["rendered"] = real_block(**kw)
        return seen["rendered"]
    monkeypatch.setattr(tx_receipt, "_format_request_block", _capture)
    return seen
