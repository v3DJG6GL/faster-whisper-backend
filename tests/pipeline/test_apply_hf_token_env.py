"""pl_apply.apply_hot_changes re-syncs os.environ["HF_TOKEN"] on an
HF_TOKEN save. Clearing the config field must hand back the token the service
environment set (the field is pinned by WHISPER_HF_TOKEN, not HF_TOKEN), not
delete it: Silero VAD, tokenizer fetches and gated pyannote loads would run
unauthenticated until the next restart brought it back."""

import asyncio
import os
import subprocess
import sys

import pytest

from faster_whisper_backend.paths import REPO_ROOT
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


# Boots the app module the way the service does, with a token in the service
# environment AND one in config: main.py promotes the config one over the bare
# HF_TOKEN before pipeline.apply is imported. Prints the promoted value, what a
# uvicorn worker (which inherits that promoted environment) takes for the
# service token, and HF_TOKEN after the admin clears the config field.
_BOOT = """
import asyncio, os, subprocess, sys
import faster_whisper_backend.main  # noqa: F401
from faster_whisper_backend.pipeline import apply as pl_apply
if sys.argv[1] == "worker":
    print(pl_apply._ENV_HF_TOKEN)
    sys.exit()
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.transcription import models as tx_models
print(os.environ["HF_TOKEN"])
sys.stdout.flush()
subprocess.run([sys.executable, "-c", sys.argv[2], "worker"], check=True)
async def no_evict(model_id=None):
    return []
config_store.env_pinned_fields = lambda: frozenset()
config_store.load_overrides = lambda: {"HF_TOKEN": None}
tx_models.drain_then_evict = no_evict
asyncio.run(pl_apply.apply_hot_changes({"HF_TOKEN": None}))
print(os.environ.get("HF_TOKEN"))
"""


def test_boot_with_a_config_token_keeps_the_service_token():
    env = dict(os.environ, HF_TOKEN="hf_service", WHISPER_HF_TOKEN="hf_config")
    env.pop("WHISPER_HF_TOKEN_FILE", None)
    env.pop(pl_apply.SERVICE_HF_TOKEN_ENV, None)
    out = subprocess.run([sys.executable, "-c", _BOOT, "boot", _BOOT],
                         cwd=REPO_ROOT, env=env, capture_output=True,
                         text=True, timeout=120, check=True).stdout
    assert out.split() == ["hf_config", "hf_service", "hf_service"]
