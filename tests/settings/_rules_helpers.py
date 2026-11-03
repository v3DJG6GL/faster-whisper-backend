"""Shared helpers for the tests/settings/ rule and config.local.json tests."""
import json
import os

from faster_whisper_backend import paths


def factory_rules():
    """config.json's PIPELINE_RULES as shipped: a fresh, unvalidated parse on
    every call, so a caller may mutate the result."""
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]


def load_saved(tmp_path, raw):
    """`raw` written as tmp_path/config.local.json, read back through
    config_store.load_overrides."""
    from faster_whisper_backend.settings import config_store as cs
    p = tmp_path / "config.local.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return cs.load_overrides(str(p))
