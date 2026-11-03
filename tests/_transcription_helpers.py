"""Helpers shared by the stage tests that drive POST /v1/audio/transcriptions
(the stub upload, a plain / verbose_json post, a profile-bound non-admin key).
Import as ``from tests._transcription_helpers import ...``."""

from tests.conftest import bearer

WAV_UPLOAD = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def post_transcription(client, headers=None, **data):
    """POST the stub upload with form fields `data` (model defaults to whisper-1)."""
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=WAV_UPLOAD, data=data,
                       headers=headers)


def post_verbose(client, headers=None, **data):
    data.setdefault("response_format", "verbose_json")
    return post_transcription(client, headers, **data)


def bind_profile(client, make_user_key, name, fields, user="alice"):
    """Create admin + non-admin `user` bound to the one OVERRIDE_PROFILES
    entry `name` = `fields`; return (admin_headers, user_raw_key)."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text
    uid, raw = make_user_key(user, is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": [name], "locks": []}})
    assert r.status_code == 200, r.text
    return admin_h, raw


def record_runs(app_module, monkeypatch):
    """Spy on metrics.record_transcription (passing through to the real one);
    return the list the spy appends each call's kwargs to."""
    recorded = []
    real = app_module.metrics.record_transcription

    def _spy(**kw):
        recorded.append(kw)
        return real(**kw)
    monkeypatch.setattr(app_module.metrics, "record_transcription", _spy)
    return recorded
