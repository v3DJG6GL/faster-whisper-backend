"""pl_apply.apply_hot_changes re-syncs os.environ["HF_TOKEN"] on an
HF_TOKEN save. Clearing the config field must hand back the token the service
environment set (the field is pinned by WHISPER_HF_TOKEN, not HF_TOKEN), not
delete it: Silero VAD, tokenizer fetches and gated pyannote loads would run
unauthenticated until the next restart brought it back."""

import asyncio
import os

import pytest

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.transcription import models as tx_models


def _save_hf_token(monkeypatch, value):
    async def no_evict(model_id=None):
        return []

    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    monkeypatch.setattr(config_store, "load_overrides", lambda: {"HF_TOKEN": value})
    monkeypatch.setattr(tx_models, "drain_then_evict", no_evict)
    asyncio.run(pl_apply.apply_hot_changes({"HF_TOKEN": value}))


@pytest.mark.parametrize("startup_env, expected", [
    ("hf_service", "hf_service"),
    (None, None),
])
def test_clearing_the_field_restores_the_service_token(monkeypatch, startup_env, expected):
    monkeypatch.setattr(pl_apply, "_ENV_HF_TOKEN", startup_env)
    monkeypatch.setattr(cfg, "HF_TOKEN", None, raising=False)
    monkeypatch.setenv("HF_TOKEN", startup_env or "unset")
    _save_hf_token(monkeypatch, "hf_admin")
    assert os.environ["HF_TOKEN"] == "hf_admin"
    _save_hf_token(monkeypatch, None)
    assert os.environ.get("HF_TOKEN") == expected
