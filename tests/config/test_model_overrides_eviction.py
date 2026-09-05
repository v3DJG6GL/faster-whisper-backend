"""admin_routes._apply_hot_changes must evict a model on a MODEL_OVERRIDES
save only when that model's LOAD-TIME subset (added, changed or removed
key) differs between the pre-save snapshot and the new bundle.

The editor sends the whole dict on every change, so "clear MODEL_DEVICE,
keep BEAM_SIZE" and "edit B's BEAM_SIZE while A holds an unchanged
MODEL_DEVICE" are ordinary UI flows — the first must evict, the second
must not."""

import asyncio

from faster_whisper_backend import config as cfg
from faster_whisper_backend import config_store
from faster_whisper_backend import main
from faster_whisper_backend.admin import routes as admin_routes


def _run(monkeypatch, old, new):
    calls: list = []

    async def spy(model_id=None):
        calls.append(model_id)
        return [model_id]

    monkeypatch.setattr(cfg, "MODEL_OVERRIDES", old, raising=False)
    monkeypatch.setattr(config_store, "env_pinned_fields", lambda: frozenset())
    monkeypatch.setattr(config_store, "load_overrides",
                        lambda: {"MODEL_OVERRIDES": new})
    monkeypatch.setattr(main, "drain_then_evict", spy)
    asyncio.run(admin_routes._apply_hot_changes({"MODEL_OVERRIDES": new}, old))
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
