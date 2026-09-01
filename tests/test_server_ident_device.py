"""Identity card device word: only DECODE (whisper) models may drive it.

system_stats' loaded-model registry is shared by every model family
(translation "gguf:", diarization "pyannote:", BGM separation "uvr:" —
whisper registers bare names), so the card must not read the device off
whichever family happened to load first."""

import config as cfg


def test_identity_device_ignores_cuda_non_decode_model(client, monkeypatch):
    """A cuda pyannote pipeline loaded before any whisper model must not make
    a MODEL_DEVICE=cpu box claim "gpu — …" in the card / copy-report."""
    import admin_routes
    import system_stats

    monkeypatch.setattr(system_stats, "gpu_name",
                        lambda: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(
        system_stats, "loaded_models_snapshot",
        lambda: [{"name": "pyannote:pyannote/segmentation-3.0",
                  "device": "cuda"},
                 {"name": "large-v3", "device": "cpu"}])
    ident = admin_routes._server_ident_fields()
    assert "gpu —" not in ident["runs_as"]
    assert "cpu" in ident["runs_as"]
    assert "gpu —" not in ident["report"]


def test_identity_device_prefers_decode_model_over_earlier_family(
        client, monkeypatch):
    """The reverse misreport: a cpu gguf translator loaded first must not hide
    that the decode model actually runs on cuda."""
    import admin_routes
    import system_stats

    monkeypatch.setattr(system_stats, "gpu_name",
                        lambda: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(
        system_stats, "loaded_models_snapshot",
        lambda: [{"name": "gguf:org/model-GGUF:Q4", "device": "cpu"},
                 {"name": "large-v3", "device": "cuda"}])
    ident = admin_routes._server_ident_fields()
    assert "gpu — NVIDIA GeForce RTX 3080" in ident["runs_as"]
