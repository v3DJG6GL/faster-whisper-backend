"""pl_apply.apply_hot_changes must evict a model on a MODEL_OVERRIDES
save only when that model's LOAD-TIME subset (added, changed or removed
key) differs between the pre-save snapshot and the new bundle.

The editor sends the whole dict on every change, so "clear MODEL_DEVICE,
keep BEAM_SIZE" and "edit B's BEAM_SIZE while A holds an unchanged
MODEL_DEVICE" are ordinary UI flows — the first must evict, the second
must not."""

import asyncio

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.settings import config_store
from faster_whisper_backend.pipeline import apply as pl_apply
from faster_whisper_backend.transcription import models as tx_models


def _run(monkeypatch, old, new):
    calls: list = []

    async def spy(model_id=None):
        calls.append(model_id)
        return [model_id]

    monkeypatch.setattr(cfg, "MODEL_OVERRIDES", old, raising=False)
    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    monkeypatch.setattr(config_store, "load_overrides",
                        lambda: {"MODEL_OVERRIDES": new})
    monkeypatch.setattr(tx_models, "drain_then_evict", spy)
    asyncio.run(pl_apply.apply_hot_changes({"MODEL_OVERRIDES": new}, old))
    return calls


def test_removing_load_time_key_evicts_model(monkeypatch):
    old = {"A": {"MODEL_DEVICE": "cpu", "BEAM_SIZE": 5}}
    new = {"A": {"BEAM_SIZE": 5}}
    assert _run(monkeypatch, old, new) == ["A"]


def test_unrelated_decode_edit_does_not_evict(monkeypatch):
    old = {"A": {"MODEL_DEVICE": "cpu"}, "B": {"BEAM_SIZE": 5}}
    new = {"A": {"MODEL_DEVICE": "cpu"}, "B": {"BEAM_SIZE": 7}}
    assert _run(monkeypatch, old, new) == []


def test_removed_id_evicts_only_when_it_held_load_time_key(monkeypatch):
    old = {"C": {"BEAM_SIZE": 5}, "D": {"REVISION": "abc"}}
    new = {}
    assert _run(monkeypatch, old, new) == ["D"]


def test_env_per_model_values_stay_live_after_a_save(monkeypatch):
    # save_overrides keeps WHISPER_MODEL_OVERRIDE__ values out of the file, so
    # the reload must lay them back on — and an env load-time value the file
    # never held is no reason to evict.
    monkeypatch.setattr(cfg, "_ENV_OVERRIDE_VALUES",
                        {"A": {"MODEL_DEVICE": "cpu"}}, raising=False)
    old = {"A": {"MODEL_DEVICE": "cpu", "BEAM_SIZE": 5}}
    new = {"A": {"BEAM_SIZE": 7}}
    assert _run(monkeypatch, old, new) == []
    assert cfg.MODEL_OVERRIDES == {"A": {"MODEL_DEVICE": "cpu", "BEAM_SIZE": 7}}
