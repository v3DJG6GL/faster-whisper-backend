"""Settings-domain test fixtures."""

import importlib

import pytest

from faster_whisper_backend.settings import config


@pytest.fixture
def restore_config(monkeypatch):
    """For a test that reloads config under patched env vars: undo the
    patches, then reload config so the module matches the clean env again."""
    yield
    monkeypatch.undo()
    importlib.reload(config)
