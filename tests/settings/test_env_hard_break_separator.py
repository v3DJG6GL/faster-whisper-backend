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


# .env.example documents the newline value as '\n'. python-dotenv keeps single
# quotes literal, and so do a compose env_file and a Windows service <env>, so
# the operator's newline arrives as the 2 characters backslash + n; read as is,
# live dictation typed "\n" into the document at every hard break.
@pytest.mark.parametrize("raw, want", [
    ("\\n", "\n"), ("\\r\\n", "\r\n"), (" \\n", " \n"), ("\\\\n", "\\n")])
def test_env_separator_decodes_spelled_escapes(monkeypatch, raw, want):
    try:
        monkeypatch.setenv("WHISPER_STREAMING_HARD_BREAK_SEPARATOR", raw)
        importlib.reload(config)
        assert config.STREAMING_HARD_BREAK_SEPARATOR == want
        assert "STREAMING_HARD_BREAK_SEPARATOR" in config_store.env_pinned_fields()
    finally:
        monkeypatch.undo()
        importlib.reload(config)
