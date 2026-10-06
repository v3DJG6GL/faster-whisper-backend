"""Tests for model_registry: the loaded-model registry round-trip, the role
split by family prefix, and the negative-delta guard.

The module keeps a process-global registry (_loaded_models); a local autouse
fixture clears it between tests so cases don't observe each other's writes.
"""

import time

import pytest

from faster_whisper_backend.runtime import model_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the module-global loaded-model registry."""
    with model_registry._loaded_models_lock:
        model_registry._loaded_models.clear()
    yield
    with model_registry._loaded_models_lock:
        model_registry._loaded_models.clear()


# ---------------------------------------------------------------------------
# Loaded-model registry round-trip
# ---------------------------------------------------------------------------

def test_register_and_snapshot():
    model_registry.register_loaded_model("m1", 512 * 1024 * 1024, "cuda", "float16")
    snap = model_registry.loaded_models_snapshot()
    assert len(snap) == 1
    e = snap[0]
    assert e["name"] == "m1"
    assert e["device"] == "cuda"
    assert e["compute_type"] == "float16"
    assert e["vram_mb"] == 512.0
    assert e["age_sec"] >= 0
    assert e["idle_sec"] >= 0


def test_snapshot_tells_the_model_role_apart_by_prefix():
    """The stats page's loaded-models table shows what each model is for:
    the registration prefix of each family decides, and the label drops it."""
    model_registry.register_loaded_model("large-v3", None, "cuda", "float16")
    model_registry.register_loaded_model("pyannote:speaker-diarization-3.1", None, "cuda", "float32")
    model_registry.register_loaded_model("uvr:UVR-MDX-NET-Inst_HQ_3", None, "cuda", "float32")
    model_registry.register_loaded_model("gguf:org/gemma", None, "cpu", "q4")
    roles = {e["name"]: (e["role"], e["label"]) for e in model_registry.loaded_models_snapshot()}
    assert roles == {
        "large-v3": ("transcribing", "large-v3"),
        "pyannote:speaker-diarization-3.1": ("diarizing", "speaker-diarization-3.1"),
        "uvr:UVR-MDX-NET-Inst_HQ_3": ("separating", "UVR-MDX-NET-Inst_HQ_3"),
        "gguf:org/gemma": ("translating", "org/gemma"),
    }


def test_register_none_vram():
    model_registry.register_loaded_model("m2", None, "cpu", "int8")
    e = model_registry.loaded_models_snapshot()[0]
    assert e["vram_mb"] is None


def test_touch_updates_last_used():
    model_registry.register_loaded_model("m3", None, "cpu", "int8")
    with model_registry._loaded_models_lock:
        # Backdate last_used so the touch is observable.
        model_registry._loaded_models["m3"]["last_used"] = time.time() - 100
    before = model_registry.loaded_models_snapshot()[0]["idle_sec"]
    assert before >= 99
    model_registry.touch_loaded_model("m3")
    after = model_registry.loaded_models_snapshot()[0]["idle_sec"]
    assert after < before


def test_touch_unknown_model_noop():
    # Touching a name that was never registered must not raise or create it.
    model_registry.touch_loaded_model("nope")
    assert model_registry.loaded_models_snapshot() == []


def test_unregister():
    model_registry.register_loaded_model("m4", None, "cpu", "int8")
    assert len(model_registry.loaded_models_snapshot()) == 1
    model_registry.unregister_loaded_model("m4")
    assert model_registry.loaded_models_snapshot() == []
    # Unregistering an absent model is a no-op.
    model_registry.unregister_loaded_model("m4")
    assert model_registry.loaded_models_snapshot() == []


def test_snapshot_ordered_by_insertion():
    model_registry.register_loaded_model("a", None, "cpu", "int8")
    model_registry.register_loaded_model("b", None, "cpu", "int8")
    names = [m["name"] for m in model_registry.loaded_models_snapshot()]
    assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# register_loaded_model regressions
# ---------------------------------------------------------------------------

def test_negative_vram_delta_is_not_displayed(monkeypatch):
    """A concurrent free elsewhere on the GPU makes after < before; that is
    not a measurement and must not surface as a negative vram_mb."""
    from faster_whisper_backend.runtime import model_sizes
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: None)
    model_registry.register_loaded_model("neg", -512 * 1024 * 1024, "cuda",
                                         "int8")
    snap = model_registry.loaded_models_snapshot()
    assert snap[0]["name"] == "neg" and snap[0]["vram_mb"] is None
    model_registry.register_loaded_model("zero", 0, "cpu", "int8")
    assert model_registry.loaded_models_snapshot()[1]["vram_mb"] == 0
