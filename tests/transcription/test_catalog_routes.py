"""transcription.catalog_routes: /v1/me's stage-model lists follow the
caller's effective model (an identity override), not only the global one.

The identity layer is stood in for by wrapping effective_config.cfg_for
(model_id=None is the /v1/me resolution); the HTTP wiring of real key/user
overrides is covered in tests/settings/test_per_identity_overrides.py."""

from faster_whisper_backend.settings import effective_config


def _identity_value(monkeypatch, field, value):
    orig = effective_config.cfg_for

    def cfg_for(model_id, f, ident=None):
        if f == field and model_id is None:
            return value
        return orig(model_id, f, ident)
    monkeypatch.setattr(effective_config, "cfg_for", cfg_for)


def test_me_stage_models_lead_with_the_callers_own_default(client, app_module,
                                                           monkeypatch):
    """Batch admission admits allowlist ∪ {global, the caller's effective
    value}; /v1/me must offer the caller's own default first."""
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_MODEL", "pyannote/x")
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ALLOWED_MODELS", [])
    _identity_value(monkeypatch, "DIARIZATION_MODEL", "pyannote/y")
    j = client.get("/v1/me").json()
    assert [m["id"] for m in j["diarization_models"]] == ["pyannote/y", "pyannote/x"]


def test_me_translation_models_lead_with_the_callers_effective_model(
        client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True)
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_DEFAULT_MODEL", "org/d:Q4")
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ALLOWED_MODELS",
                        {"org/a:Q4"})
    _identity_value(monkeypatch, "TRANSLATION_MODEL", "org/x:Q4")
    j = client.get("/v1/me").json()
    assert [m["id"] for m in j["translation_models"]] == [
        "org/x:Q4", "org/d:Q4", "org/a:Q4"]
