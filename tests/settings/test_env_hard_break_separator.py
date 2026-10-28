"""WHISPER_STREAMING_HARD_BREAK_SEPARATOR keeps whitespace values.

Every documented non-empty value (' ' = space, '\\n' = newline) is whitespace,
so the default stripping "str" reader dropped it without a warning and a hard
break in live dictation typed nothing between documents.
"""

import importlib

import pytest

from faster_whisper_backend.settings import config
from faster_whisper_backend.settings import config_store


@pytest.mark.parametrize("raw", [" ", "\n", ""])
def test_env_separator_kept_verbatim_and_pinned(monkeypatch, raw):
    try:
        monkeypatch.setenv("WHISPER_STREAMING_HARD_BREAK_SEPARATOR", raw)
        importlib.reload(config)
        assert config.STREAMING_HARD_BREAK_SEPARATOR == raw
        assert "STREAMING_HARD_BREAK_SEPARATOR" in config_store.env_pinned_fields()
    finally:
        monkeypatch.undo()
        importlib.reload(config)
