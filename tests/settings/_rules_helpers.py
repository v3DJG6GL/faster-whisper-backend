"""Shared helpers for the tests/settings/ rule tests."""
import json
import os

from faster_whisper_backend import paths


def factory_rules():
    """config.json's PIPELINE_RULES as shipped: a fresh, unvalidated parse on
    every call, so a caller may mutate the result."""
    with open(os.path.join(paths.REPO_ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["PIPELINE_RULES"]
